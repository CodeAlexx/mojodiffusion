# ops/rope_tables.mojo — multi-axis (3D) RoPE cos/sin table builder.
#
# RANK-1 gap from ops/parity/OPS_GAP_AUDIT_2026-06-03.md: every Phase 2-4 video/
# image DiT (wan22, wan_vace, hunyuan15, kandinsky5, cosmos, magihuman, nava
# video) builds a "3-axis / complex 3D RoPE" frequency table that, once laid out,
# is consumed by ONE existing apply kernel — `ops/rope.rope_interleaved` (complex
# / pair-interleaved layout, e.g. wan22 `view_as_complex`) or
# `ops/rope.rope_halfsplit` (GPT-NeoX half-split, e.g. cosmos). The apply kernel
# is already present and proven (zimage). What was missing — and re-implemented
# by hand (host loop) in `models/dit/zimage_dit.mojo::_build_rope` — is the
# *reusable* builder that turns per-token per-axis positions into the
# `[rows, head_dim/2]` cos/sin tables those kernels eat.
#
# Layout produced (matches `rope_interleaved`/`rope_halfsplit` "[rows, D/2]"):
#   For token `t`, the angle vector is the CONCATENATION over axes `a` of
#       angle[t, off_a + i] = pos[t, a] * theta^(-i / half_a),   i in [0, half_a)
#   where `half_a = axes_dims[a] / 2`, `off_a = sum_{b<a} half_b`, and the total
#   width is `half = sum_a half_a == head_dim/2`. cos/sin tables are
#   `[rows, half]` in the caller-requested storage dtype; trig math stays F32
#   inside the kernel.
#
#   inv_freq convention: `theta^(-i/half_a)`. This equals zimage's
#   `theta^(-2i/axis_dim)` (since half_a = axis_dim/2) and wan22/cosmos's
#   `theta^(-i/axis_half)` — verified identical in the gap audit §2.
#
# INPUT positions are passed as a FLAT F32 tensor `[rows * num_axes]`, row-major
# (token-major): index `t * num_axes + a` holds token t's position along axis a.
# (Integer grid positions cast to F32 by the caller; matches the Rust refs that
# decompose `si -> (frame,height,width)` then index the per-axis table.)
#
# One thread per (row, out-column i in [0, half)). The kernel walks the axis
# blocks to find which axis owns column i and the local index within it. num_axes
# is bounded (<= _MAX_AXES); axis dims are passed as a small device buffer.
#
# Mojo 1.0.0b1, NVIDIA GPU. Compute F32; output storage is explicit.

from std.math import exp, log, cos, sin
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _MAX_AXES = 4  # frame/height/width (+ optional 4th); bounded.


def _multiaxis_rope_kernel[out_dtype: DType](
    positions: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [rows*num_axes]
    axes_half: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],    # [num_axes]
    cos_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],          # [rows, half]
    sin_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],          # [rows, half]
    rows: Int,
    half: Int,         # sum of axes_half == head_dim/2
    num_axes: Int,
    theta: Float32,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx >= total:
        return
    var row = idx // half
    var col = idx % half

    # Walk axis blocks to find the owning axis `a` and local index `i`.
    var off = 0
    var a = 0
    var local_i = col
    var ha = 0
    while a < num_axes:
        ha = Int(rebind[Scalar[DType.int32]](axes_half[a]))
        if col < off + ha:
            local_i = col - off
            break
        off += ha
        a += 1

    # pos for token `row` along axis `a`: positions[row*num_axes + a].
    var pos = rebind[Scalar[DType.float32]](positions[row * num_axes + a])
    # inv_freq = theta^(-local_i / ha)
    var inv_freq = exp(-log(theta) * (Float32(local_i) / Float32(ha)))
    var angle = pos * inv_freq
    cos_t[row, col] = rebind[cos_t.element_type](cos(angle).cast[out_dtype]())
    sin_t[row, col] = rebind[sin_t.element_type](sin(angle).cast[out_dtype]())


def build_multiaxis_rope_tables(
    positions: Tensor,
    axes_dims: List[Int],
    theta: Float32,
    ctx: DeviceContext,
    out_dtype: STDtype,
) raises -> Tuple[Tensor, Tensor]:
    """Multi-axis (3D) RoPE cos/sin tables for `rope_interleaved`/`rope_halfsplit`.

    positions: [rows * num_axes] F32, token-major (index `t*num_axes + a` =
               token t's integer grid position on axis a, cast to F32).
    axes_dims: per-axis FULL rotary dim; each must be even. `sum(axes_dims)`
               must equal head_dim, and `sum(axes_dims)/2` (== head_dim/2) is the
               produced table width `half`. For head_dim=128 with a 3-axis (f,h,w)
               split, wan22 and cosmos both use [44,42,42] (sums to 128; do NOT
               pass the doubled [88,84,84] — that builds a [rows,128] table the
               apply kernels reject, since they validate cos.numel()==rows*64).
    theta:     rope_theta (wan22/cosmos = 10000.0; zimage = 256.0). NOTE: this is
               a single scalar theta — sufficient for default NTK ratios (1.0),
               but cannot express cosmos per-axis NTK extrapolation variants
               (e.g. _2b_image); those need a per-axis theta extension.
    returns (cos, sin), each [rows, half] out_dtype. Feed straight into
            ops/rope.rope_interleaved (complex/pair layout) or rope_halfsplit
            (GPT-NeoX half-split) with q/k of shape [..., head_dim].
    CALLER RESPONSIBILITY: build `positions` by decomposing each flat token index
            si into its (f,h,w) grid coords in the SAME axis order as `axes_dims`,
            laid out token-major (index t*num_axes + a). This op does not infer it.
    """
    var num_axes = len(axes_dims)
    if num_axes < 1 or num_axes > _MAX_AXES:
        raise Error("build_multiaxis_rope_tables: num_axes must be 1.._MAX_AXES")
    if positions.dtype() != STDtype.F32:
        raise Error("build_multiaxis_rope_tables: positions must be F32")
    var pn = positions.numel()
    if pn % num_axes != 0:
        raise Error(
            "build_multiaxis_rope_tables: positions numel must be rows*num_axes"
        )
    var rows = pn // num_axes

    var half = 0
    var axes_half_host = List[Int32]()
    for a in range(num_axes):
        var da = axes_dims[a]
        if da % 2 != 0:
            raise Error("build_multiaxis_rope_tables: each axis dim must be even")
        var ha = da // 2
        axes_half_host.append(Int32(ha))
        half += ha

    # Upload axes_half as a true-I32 device buffer (mirrors ops/moe pattern).
    var axes_host = ctx.enqueue_create_host_buffer[DType.uint8](num_axes * 4)
    var axes_hp = axes_host.unsafe_ptr().bitcast[Int32]()
    for a in range(num_axes):
        axes_hp[a] = axes_half_host[a]
    var axes_buf = ctx.enqueue_create_buffer[DType.uint8](num_axes * 4)
    ctx.enqueue_copy(dst_buf=axes_buf, src_buf=axes_host)

    var cos_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * half * out_dtype.byte_size()
    )
    var sin_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * half * out_dtype.byte_size()
    )

    var p_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](pn))
    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](num_axes))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))

    var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        positions.buf.unsafe_ptr().bitcast[Float32](), p_rl
    )
    var A = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        axes_buf.unsafe_ptr().bitcast[Int32](), a_rl
    )
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK
    var odt = out_dtype.to_mojo_dtype()
    if odt == DType.float32:
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            cos_buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            sin_buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        ctx.enqueue_function[
            _multiaxis_rope_kernel[DType.float32],
            _multiaxis_rope_kernel[DType.float32],
        ](P, A, C, S, rows, half, num_axes, theta, grid_dim=grid, block_dim=_BLOCK)
    elif odt == DType.bfloat16:
        var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            cos_buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            sin_buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        ctx.enqueue_function[
            _multiaxis_rope_kernel[DType.bfloat16],
            _multiaxis_rope_kernel[DType.bfloat16],
        ](P, A, C, S, rows, half, num_axes, theta, grid_dim=grid, block_dim=_BLOCK)
    else:
        var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            cos_buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            sin_buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        ctx.enqueue_function[
            _multiaxis_rope_kernel[DType.float16],
            _multiaxis_rope_kernel[DType.float16],
        ](P, A, C, S, rows, half, num_axes, theta, grid_dim=grid, block_dim=_BLOCK)
    # sync removed (single-stream ordering; was kernel-trailing host stall)

    var cos_shape = List[Int]()
    cos_shape.append(rows)
    cos_shape.append(half)
    var sin_shape = List[Int]()
    sin_shape.append(rows)
    sin_shape.append(half)
    var cos_out = Tensor(cos_buf^, cos_shape^, out_dtype)
    var sin_out = Tensor(sin_buf^, sin_shape^, out_dtype)
    return (cos_out^, sin_out^)


# ── PER-AXIS THETA extension (cosmos-predict2.5 NTK extrapolation) ────────────
# The scalar-theta builder above cannot express cosmos's per-axis NTK theta.
# Cosmos V2_2B production uses rope_{h,w,t}_extrapolation_ratio = {3,3,1}, where
# per-axis theta = 10000 * ratio^(dim_axis/(dim_axis-2)). For head_dim=128 the
# split is dim_t=44, dim_h=42, dim_w=42, giving t_theta=10000 but
# h_theta=w_theta=10000*3^(42/40) ≠ 10000. The angle for axis a, local index i is
#   angle = pos * theta_a^(-(2i)/dim_a) = pos * theta_a^(-i/half_a)
# i.e. SAME exponent form as the scalar kernel, but theta is per-axis. We upload
# the per-axis thetas as a small F32 device buffer and look up theta_a per column.
# Reference: build_cosmos_rope_freqs (cosmos_predict25_dit.rs:721-892), Python
# VideoRopePosition3DEmb.generate_embeddings (minimal_v4_dit.py:730-795).


def _multiaxis_rope_kernel_per_axis[out_dtype: DType](
    positions: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [rows*num_axes]
    axes_half: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],    # [num_axes]
    axes_theta: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], # [num_axes]
    cos_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],          # [rows, half]
    sin_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],          # [rows, half]
    rows: Int,
    half: Int,
    num_axes: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx >= total:
        return
    var row = idx // half
    var col = idx % half

    # Walk axis blocks to find the owning axis `a` and local index `i`.
    var off = 0
    var a = 0
    var local_i = col
    var ha = 0
    while a < num_axes:
        ha = Int(rebind[Scalar[DType.int32]](axes_half[a]))
        if col < off + ha:
            local_i = col - off
            break
        off += ha
        a += 1

    var pos = rebind[Scalar[DType.float32]](positions[row * num_axes + a])
    var theta_a = rebind[Scalar[DType.float32]](axes_theta[a])
    # inv_freq = theta_a^(-local_i / ha)
    var inv_freq = exp(-log(theta_a) * (Float32(local_i) / Float32(ha)))
    var angle = pos * inv_freq
    cos_t[row, col] = rebind[cos_t.element_type](cos(angle).cast[out_dtype]())
    sin_t[row, col] = rebind[sin_t.element_type](sin(angle).cast[out_dtype]())


def build_multiaxis_rope_tables_per_axis(
    positions: Tensor,
    axes_dims: List[Int],
    thetas: List[Float32],
    ctx: DeviceContext,
    out_dtype: STDtype,
) raises -> Tuple[Tensor, Tensor]:
    """Per-axis-theta variant of build_multiaxis_rope_tables (cosmos NTK).

    Identical layout/contract to build_multiaxis_rope_tables, but each axis `a`
    carries its OWN theta `thetas[a]` (the NTK-scaled base, already computed by
    the caller as 10000 * ratio^(dim_a/(dim_a-2))). len(thetas) must equal
    len(axes_dims). When all thetas are equal this reduces exactly to the scalar
    builder. Feeds rope_halfsplit (cosmos GPT-NeoX) or rope_interleaved. Output
    storage is out_dtype; trig math remains F32 inside the kernel.
    """
    var num_axes = len(axes_dims)
    if num_axes < 1 or num_axes > _MAX_AXES:
        raise Error("build_multiaxis_rope_tables_per_axis: bad num_axes")
    if len(thetas) != num_axes:
        raise Error("build_multiaxis_rope_tables_per_axis: thetas len != axes")
    if positions.dtype() != STDtype.F32:
        raise Error("build_multiaxis_rope_tables_per_axis: positions must be F32")
    var pn = positions.numel()
    if pn % num_axes != 0:
        raise Error("build_multiaxis_rope_tables_per_axis: positions numel bad")
    var rows = pn // num_axes

    var half = 0
    var axes_half_host = List[Int32]()
    for a in range(num_axes):
        var da = axes_dims[a]
        if da % 2 != 0:
            raise Error("build_multiaxis_rope_tables_per_axis: axis dim odd")
        var ha = da // 2
        axes_half_host.append(Int32(ha))
        half += ha

    # axes_half I32 buffer.
    var axes_host = ctx.enqueue_create_host_buffer[DType.uint8](num_axes * 4)
    var axes_hp = axes_host.unsafe_ptr().bitcast[Int32]()
    for a in range(num_axes):
        axes_hp[a] = axes_half_host[a]
    var axes_buf = ctx.enqueue_create_buffer[DType.uint8](num_axes * 4)
    ctx.enqueue_copy(dst_buf=axes_buf, src_buf=axes_host)

    # axes_theta F32 buffer.
    var theta_host = ctx.enqueue_create_host_buffer[DType.uint8](num_axes * 4)
    var theta_hp = theta_host.unsafe_ptr().bitcast[Float32]()
    for a in range(num_axes):
        theta_hp[a] = thetas[a]
    var theta_buf = ctx.enqueue_create_buffer[DType.uint8](num_axes * 4)
    ctx.enqueue_copy(dst_buf=theta_buf, src_buf=theta_host)

    var cos_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * half * out_dtype.byte_size()
    )
    var sin_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * half * out_dtype.byte_size()
    )

    var p_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](pn))
    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](num_axes))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))

    var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        positions.buf.unsafe_ptr().bitcast[Float32](), p_rl
    )
    var A = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        axes_buf.unsafe_ptr().bitcast[Int32](), a_rl
    )
    var TH = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        theta_buf.unsafe_ptr().bitcast[Float32](), a_rl
    )
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK
    var odt = out_dtype.to_mojo_dtype()
    if odt == DType.float32:
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            cos_buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            sin_buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        ctx.enqueue_function[
            _multiaxis_rope_kernel_per_axis[DType.float32],
            _multiaxis_rope_kernel_per_axis[DType.float32],
        ](P, A, TH, C, S, rows, half, num_axes, grid_dim=grid, block_dim=_BLOCK)
    elif odt == DType.bfloat16:
        var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            cos_buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            sin_buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        ctx.enqueue_function[
            _multiaxis_rope_kernel_per_axis[DType.bfloat16],
            _multiaxis_rope_kernel_per_axis[DType.bfloat16],
        ](P, A, TH, C, S, rows, half, num_axes, grid_dim=grid, block_dim=_BLOCK)
    else:
        var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            cos_buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            sin_buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        ctx.enqueue_function[
            _multiaxis_rope_kernel_per_axis[DType.float16],
            _multiaxis_rope_kernel_per_axis[DType.float16],
        ](P, A, TH, C, S, rows, half, num_axes, grid_dim=grid, block_dim=_BLOCK)
    # sync removed (single-stream ordering; was kernel-trailing host stall)

    var cos_shape = List[Int]()
    cos_shape.append(rows)
    cos_shape.append(half)
    var sin_shape = List[Int]()
    sin_shape.append(rows)
    sin_shape.append(half)
    var cos_out = Tensor(cos_buf^, cos_shape^, out_dtype)
    var sin_out = Tensor(sin_buf^, sin_shape^, out_dtype)
    return (cos_out^, sin_out^)
