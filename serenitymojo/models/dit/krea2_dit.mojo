# models/dit/krea2_dit.mojo — Krea-2-Raw (krea2) single-stream MMDiT.
#
# Reference: ai-toolkit krea2 src/mmdit.py (`SingleStreamDiT`). This file holds
# the inference port, chunk by chunk: the `Krea2Config` struct + 3-axis
# interleaved RoPE (chunk 1), RMSNorm/SwiGLU/Modulations (chunk 2), Attention
# (chunk 3), SingleStreamBlock (chunk 4), and the embedders + input/output heads
# (chunk 5: temb/tmlp/tproj/txtmlp/first/LastLayer). The top-level forward
# (chunk 6) wires these together.
#
# ── RoPE math (1:1 with mmdit.py rope()/ropeapply()/PositionalEncoding) ───────
#   rope(pos, dim, theta):
#     scale = arange(0, dim, 2, f64) / dim        # dim/2 entries
#     omega = 1 / theta^scale                     # = theta^(-i/(dim/2)) = theta^(-i/half_a)
#     out[n, d] = pos[n] * omega[d]               # d in [0, dim/2)
#     2x2 rotation per (n,d): [[cos, -sin], [sin, cos]]
#   PositionalEncoding.forward(pos):
#     cat over the 3 axes along the freq dim, each axis i using dim=axdims[i].
#     axis order = [global, h, w]; axdims = [32, 48, 48] for headdim=128.
#     -> table covering half = sum(axdims)/2 = 64 = headdim/2 freqs per token.
#   ropeapply(xq, xk, freqs):  xq.reshape(*shape, -1, 1, 2) -> INTERLEAVED pairs
#     (x[2i], x[2i+1]) with angle index i:
#       out0 = freqs[..,0,0]*x0 + freqs[..,0,1]*x1 = cos*x0 - sin*x1
#       out1 = freqs[..,1,0]*x0 + freqs[..,1,1]*x1 = sin*x0 + cos*x1
#     This is EXACTLY ops/rope.rope_interleaved's convention.
#
# inv_freq is theta^(-i/half_a), identical to ops/rope_tables's exponent. We do
# NOT reuse ops/rope_tables.build_multiaxis_rope_tables here, because it runs
# plain F32 trig with no range reduction: krea2's global axis reaches positions
# ~ seq-len (thousands) and theta=1e3 keeps omega ~ 1.0 for the low freqs, so the
# angle hits thousands of radians where F32 sin/cos is inaccurate. We mirror the
# F64-range-reduction idiom from models/dit/ideogram4_mrope.mojo (omega in F64,
# reduce the angle mod 2pi in F64, then F32 trig on the small remainder). The
# torch reference computes omega in F64 and trig in F32 with proper reduction, so
# this matches it. The apply step reuses ops/rope.rope_interleaved unchanged.
#
# Mojo 1.0.0b1, NVIDIA GPU. Inference-only.

from std.gpu.host import DeviceContext
from std.gpu import global_idx, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation, ArcPointer
from std.math import cos as fcos, sin as fsin, exp, log, floor, sqrt
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu as swiglu_op, sigmoid, gelu
from serenitymojo.ops.tensor_algebra import (
    add, slice, reshape, mul, mul_scalar, transpose, concat, zeros_device,
)
from serenitymojo.ops.attention import sdpa_nomask, sdpa
from serenitymojo.ops.gqa_backward import repeat_kv_f32
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.cast import cast_tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _MAX_AXES = 4  # global/h/w (+ optional 4th); bounded.


# ── Krea2Config ──────────────────────────────────────────────────────────────
@fieldwise_init
struct Krea2Config(Copyable, Movable):
    """Krea-2-Raw SingleStreamDiT config (KREA2_MMDIT_CONFIG, krea2.py:55-68).

    Defaults are the reference "single_mmdit_large_wide" architecture
    (oss_raw / oss_turbo share it). `headdim = features // heads = 128` and the
    3-axis RoPE split `axes = [headdim - 12*(headdim//16), 6*(headdim//16),
    6*(headdim//16)] = [32, 48, 48]` are derived (see `head_dim()`/`rope_axes()`).
    """

    var features: Int      # 6144  — model (token) width.
    var tdim: Int          # 256   — timestep-embedding sinusoid width.
    var txtdim: Int        # 2560  — Qwen3-VL text-feature width.
    var heads: Int         # 48    — image-stream attention heads.
    var kvheads: Int       # 12    — image-stream KV heads (GQA).
    var multiplier: Int    # 4     — SwiGLU hidden multiplier.
    var layers: Int        # 28    — SingleStreamBlock depth.
    var patch: Int         # 2     — latent patch size.
    var channels: Int      # 16    — latent channels (Qwen-Image VAE z_dim).
    var txtheads: Int      # 20    — TextFusion attention heads.
    var txtkvheads: Int    # 20    — TextFusion KV heads.
    var txtlayers: Int     # 12    — selected encoder hidden-state layers fed in.
    var theta: Float32     # 1e3   — RoPE base (config.theta; overrides rope()'s 1e4 default).
    var bias: Bool         # False — Linear bias.

    @staticmethod
    def default() -> Krea2Config:
        """KREA2_MMDIT_CONFIG defaults (krea2.py:55-68 + SingleMMDiTConfig)."""
        return Krea2Config(
            features=6144,
            tdim=256,
            txtdim=2560,
            heads=48,
            kvheads=12,
            multiplier=4,
            layers=28,
            patch=2,
            channels=16,
            txtheads=20,
            txtkvheads=20,
            txtlayers=12,
            theta=Float32(1.0e3),
            bias=False,
        )

    def head_dim(self) -> Int:
        """Head dim = features // heads = 6144 // 48 = 128 (mmdit.py:202/346)."""
        return self.features // self.heads

    def rope_axes(self) -> List[Int]:
        """Per-axis FULL rotary dims (mmdit.py:347-353).

            axes = [headdim - 12*(headdim//16), 6*(headdim//16), 6*(headdim//16)]

        For headdim=128 -> [32, 48, 48] (sums to headdim, all even). Axis order
        is [global, h, w]; pos[..., a] must follow this order.
        """
        var hd = self.head_dim()
        var unit = hd // 16
        var ax = List[Int]()
        ax.append(hd - 12 * unit)
        ax.append(6 * unit)
        ax.append(6 * unit)
        return ax^


# ── 3-axis interleaved RoPE table builder ────────────────────────────────────
# One GPU thread per (row, col) of the [rows, half] table. The axis-block walk
# finds which of the 3 axes owns column `col` and its local index within that
# axis, exactly like ops/rope_tables but with F64 range reduction.
def _krea2_rope_kernel[out_dtype: DType](
    positions: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [rows*num_axes]
    axes_half: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],    # [num_axes]
    cos_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],          # [rows, half]
    sin_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],          # [rows, half]
    rows: Int,
    half: Int,         # sum(axes_half) == head_dim/2
    num_axes: Int,
    log_theta: Float64,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx >= total:
        return
    var row = idx // half
    var col = idx % half

    # Walk axis blocks to find the owning axis `a` and local index `local_i`.
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
    # omega = 1/theta^(local_i/half_a) = theta^(-local_i/half_a). Compute in F64
    # for the small-magnitude exponent precision (mirrors torch rope() f64 arange).
    # log_theta is already F64 (computed F64 at the call site) so the whole
    # exponent path stays F64 — no F32 log widening.
    var inv = exp((-Float64(local_i) / Float64(ha)) * log_theta)
    var angle = Float64(pos) * inv
    # GPU has no F64 trig, but F64 arithmetic is fine: reduce the angle mod 2pi in
    # F64 (krea2's global axis hits thousands of radians where F32 trig is wrong),
    # then F32 trig on the small reduced remainder.
    comptime TWO_PI = Float64(6.283185307179586476925286766559)
    var k = floor(angle / TWO_PI + 0.5)
    var reduced = Float32(angle - k * TWO_PI)
    cos_t[row, col] = rebind[cos_t.element_type](fcos(reduced).cast[out_dtype]())
    sin_t[row, col] = rebind[sin_t.element_type](fsin(reduced).cast[out_dtype]())


def build_krea2_rope(
    positions: Tensor,
    axes_dims: List[Int],
    theta: Float32,
    ctx: DeviceContext,
    out_dtype: STDtype,
) raises -> Tuple[Tensor, Tensor]:
    """Krea2 3-axis interleaved RoPE cos/sin tables (mmdit.py PositionalEncoding).

    positions: [rows * num_axes] F32, token-major (index `t*num_axes + a` holds
               token t's grid position along axis a; axis order [global, h, w]).
    axes_dims: per-axis FULL rotary dim (each even); `sum(axes_dims)` must equal
               head_dim, and `sum(axes_dims)/2` (head_dim/2) is the produced
               table width `half`. For headdim=128: [32, 48, 48].
    theta:     RoPE base (krea2 config.theta = 1e3).
    returns (cos, sin), each [rows, half] in out_dtype. Concatenated over the 3
            axes with per-axis omega_i = theta^(-i/half_a). Feed straight into
            ops/rope.rope_interleaved with q/k of shape [..., head_dim].
    Trig is computed with F64 range reduction; storage casts to out_dtype.
    """
    var num_axes = len(axes_dims)
    if num_axes < 1 or num_axes > _MAX_AXES:
        raise Error("build_krea2_rope: num_axes must be 1.._MAX_AXES")
    if positions.dtype() != STDtype.F32:
        raise Error("build_krea2_rope: positions must be F32")
    var pn = positions.numel()
    if pn % num_axes != 0:
        raise Error("build_krea2_rope: positions numel must be rows*num_axes")
    var rows = pn // num_axes

    var half = 0
    var axes_half_host = List[Int32]()
    for a in range(num_axes):
        var da = axes_dims[a]
        if da % 2 != 0:
            raise Error("build_krea2_rope: each axis dim must be even")
        var ha = da // 2
        axes_half_host.append(Int32(ha))
        half += ha

    # Upload axes_half as a true-I32 device buffer (mirrors ops/rope_tables).
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
    var lt = log(Float64(theta))
    var odt = out_dtype.to_mojo_dtype()
    if odt == DType.float32:
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            cos_buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            sin_buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        ctx.enqueue_function[
            _krea2_rope_kernel[DType.float32],
            _krea2_rope_kernel[DType.float32],
        ](P, A, C, S, rows, half, num_axes, lt, grid_dim=grid, block_dim=_BLOCK)
    elif odt == DType.bfloat16:
        var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            cos_buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            sin_buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        ctx.enqueue_function[
            _krea2_rope_kernel[DType.bfloat16],
            _krea2_rope_kernel[DType.bfloat16],
        ](P, A, C, S, rows, half, num_axes, lt, grid_dim=grid, block_dim=_BLOCK)
    else:
        var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            cos_buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            sin_buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        ctx.enqueue_function[
            _krea2_rope_kernel[DType.float16],
            _krea2_rope_kernel[DType.float16],
        ](P, A, C, S, rows, half, num_axes, lt, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()

    var cos_shape = List[Int]()
    cos_shape.append(rows)
    cos_shape.append(half)
    var sin_shape = List[Int]()
    sin_shape.append(rows)
    sin_shape.append(half)
    var cos_out = Tensor(cos_buf^, cos_shape^, out_dtype)
    var sin_out = Tensor(sin_buf^, sin_shape^, out_dtype)
    return (cos_out^, sin_out^)


def apply_krea2_rope(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """Apply krea2 interleaved RoPE to q and k (mmdit.py ropeapply()).

    q, k:    [..., head_dim]  (rows = product of leading dims; here rows = the
             same L the cos/sin table was built for, so each row's q/k share its
             token's freqs).
    cos/sin: [rows, head_dim/2] from build_krea2_rope.
    returns (q_rot, k_rot), same shapes/dtype as q/k. Math is F32 inside the
    interleaved kernel (matches ropeapply's xq.float()). q and k are rotated
    with the SAME freqs table (ropeapply passes one `freqs` to both).
    """
    var q_rot = rope_interleaved(q, cos, sin, ctx)
    var k_rot = rope_interleaved(k, cos, sin, ctx)
    return (q_rot^, k_rot^)


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 2 — SingleStreamBlock leaf ops (RMSNorm / SwiGLU / Modulations).
# Reference: mmdit.py RMSNorm(163-177), SwiGLU(180-194), SimpleModulation(109-119),
# DoubleSharedModulation(122-133).
# ══════════════════════════════════════════════════════════════════════════════


# ── RMSNorm (mmdit.py:163-177) — F32-INTERNAL with weight = scale + 1.0 ───────
# The reference is precision-critical (the Rust noise-saga root cause):
#   t = x.float()                                            # bf16 -> F32
#   t = F.rms_norm(t, (features,), eps=1e-5, weight=scale.float() + 1.0)
#   return t.to(dtype)                                       # F32 -> bf16
# i.e. the rms reduction AND the weight multiply are F32, the weight is the
# F32 reparam (scale + 1.0), and bf16 is touched ONLY at the x-read upcast and
# the final store. ops/norm.rms_norm is NOT reused: its bf16 path reads the
# WEIGHT as bf16 (bf16-rounds scale+1 before the multiply) and applies the raw
# weight (no +1 reparam). We hand-roll an F32-internal kernel that keeps the
# scale F32 and adds 1.0 in F32 inside the multiply. x is read as its storage
# dtype and upcast to F32 (lossless for bf16: bf16 is a truncated F32, == .float()).
comptime _RMS_TPB = 256  # threads per block (one block per row)


def _krea2_rmsnorm_kernel[x_dtype: DType](
    x: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # F32 scale (NOT scale+1)
    o: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _RMS_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    # Sum of squares in F32 (matches F.rms_norm on x.float()).
    var local: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[x_dtype]](x[row, c]).cast[DType.float32]()
        local += v * v
        c += _RMS_TPB
    shared[tid] = local
    barrier()
    var active = _RMS_TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(shared[0] / Float32(cols) + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[x_dtype]](x[row, c]).cast[DType.float32]()
        # weight = scale + 1.0, kept F32 (the reference's scale.float() + 1.0).
        var w = rebind[Scalar[DType.float32]](scale[c]) + Float32(1.0)
        o[row, c] = rebind[o.element_type]((v * inv * w).cast[x_dtype]())
        c += _RMS_TPB


def krea2_rmsnorm(
    x: Tensor, scale: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Krea2 RMSNorm (mmdit.py:163-177). F32-internal; weight = scale + 1.0.

    x:     [..., features]  (storage dtype; read upcast to F32 == x.float()).
    scale: [features]       F32 (the zeros-init Parameter; weight is scale+1.0,
           added in F32 inside the kernel — pass the RAW scale, not scale+1).
    eps:   1e-5.
    returns [..., features] in x's dtype (F32 math, cast only at store).
    """
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("krea2_rmsnorm: x must have rank >= 1")
    var cols = xshape[len(xshape) - 1]
    if scale.dtype() != STDtype.F32:
        raise Error("krea2_rmsnorm: scale must be F32 (F32 weight reparam)")
    if scale.numel() != cols:
        raise Error("krea2_rmsnorm: scale numel must equal features")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](cols))
    var SC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), g_rl
    )
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[
            _krea2_rmsnorm_kernel[DType.float32],
            _krea2_rmsnorm_kernel[DType.float32],
        ](X, SC, O, cols, eps, grid_dim=rows, block_dim=_RMS_TPB)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[
            _krea2_rmsnorm_kernel[DType.bfloat16],
            _krea2_rmsnorm_kernel[DType.bfloat16],
        ](X, SC, O, cols, eps, grid_dim=rows, block_dim=_RMS_TPB)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[
            _krea2_rmsnorm_kernel[DType.float16],
            _krea2_rmsnorm_kernel[DType.float16],
        ](X, SC, O, cols, eps, grid_dim=rows, block_dim=_RMS_TPB)
    # No trailing sync (single-stream ordering; downstream .to_host() syncs).
    return Tensor(out_buf^, x.shape(), x.dtype())


def krea2_swiglu_mlpdim(features: Int, multiplier: Int) -> Int:
    """SwiGLU hidden dim (mmdit.py:186-187): int(2*features/3)*multiplier rounded
    UP to a multiple of 128. For features=6144, multiplier=4 -> 4096*4=16384."""
    var mlpdim = (Int(2 * features // 3)) * multiplier
    var multiple = 128
    mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
    return mlpdim


def krea2_swiglu(
    x: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Krea2 SwiGLU (mmdit.py:180-194): down(silu(gate(x)) * up(x)), no bias.

    REUSES ops/linear (x @ Wᵀ, F32 accum, bf16 storage) for the three projections
    and ops/activations.swiglu (= silu(gate)*up elementwise) for the gated core.
    gate_w/up_w: [mlpdim, features]; down_w: [features, mlpdim] (torch Linear
    weight layout). mlpdim is taken from the weight shapes, not recomputed.
    """
    var gate = linear(x, gate_w, None, ctx)      # [..., mlpdim]
    var up = linear(x, up_w, None, ctx)          # [..., mlpdim]
    var gated = swiglu_op(gate, up, ctx)         # silu(gate) * up
    return linear(gated, down_w, None, ctx)      # [..., features]


# ── SimpleModulation (mmdit.py:109-119) ──────────────────────────────────────
# param `lin` is [2, dim] zeros; out = vec + lin[None]; chunk(2, dim=1) ->
# (scale, shift). At inference (b=1) vec [1, dim] broadcasts against lin[1,2,dim]
# -> [1, 2, dim]; scale/shift are each [1, 1, dim]. We add then slice dim=1.
def krea2_simple_modulation(
    vec: Tensor, lin: Tensor, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """SimpleModulation.forward (mmdit.py:116-119). Returns (scale, shift).

    vec: [b, dim]   the (time) conditioning vector.
    lin: [2, dim]   the zeros-init modulation parameter.
    out = vec[:, None, :] + lin[None]  -> [b, 2, dim]; chunk along dim=1.
    Returns scale, shift each [b, 1, dim] (matching torch chunk(2, dim=1)).
    """
    var vshape = vec.shape()
    var b = vshape[0]
    var dim = vshape[len(vshape) - 1]
    # Reshape vec [b, dim] -> [b, 1, dim] so it broadcasts against lin [1, 2, dim].
    var vec3_shape = List[Int]()
    vec3_shape.append(b)
    vec3_shape.append(1)
    vec3_shape.append(dim)
    var vec3 = reshape(vec, vec3_shape^, ctx)
    # lin [2, dim] -> [1, 2, dim] for broadcast add.
    var lin3_shape = List[Int]()
    lin3_shape.append(1)
    lin3_shape.append(2)
    lin3_shape.append(dim)
    var lin3 = reshape(lin, lin3_shape^, ctx)
    var out = add(vec3, lin3, ctx)               # [b, 2, dim]
    var scale = slice(out, 1, 0, 1, ctx)         # [b, 1, dim]
    var shift = slice(out, 1, 1, 1, ctx)         # [b, 1, dim]
    return (scale^, shift^)


# ── DoubleSharedModulation (mmdit.py:122-133) ────────────────────────────────
# param `lin` is [6*dim] zeros; out = vec + lin; chunk(6, dim=-1) ->
# (prescale, preshift, pregate, postscale, postshift, postgate).
def krea2_double_shared_modulation(
    vec: Tensor, lin: Tensor, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """DoubleSharedModulation.forward (mmdit.py:128-133). Returns the 6 chunks
    (prescale, preshift, pregate, postscale, postshift, postgate), each [b, dim].

    vec: [b, 6*dim]   conditioning vector.
    lin: [6*dim]      zeros-init parameter (broadcasts over the batch).
    out = vec + lin; chunk into 6 along the last dim.
    """
    var vshape = vec.shape()
    var last = len(vshape) - 1
    var sixdim = vshape[last]
    if sixdim % 6 != 0:
        raise Error("krea2_double_shared_modulation: last dim must be 6*dim")
    var dim = sixdim // 6
    var out = add(vec, lin, ctx)                 # [b, 6*dim] (lin [6*dim] broadcasts)
    var c0 = slice(out, last, 0 * dim, dim, ctx)
    var c1 = slice(out, last, 1 * dim, dim, ctx)
    var c2 = slice(out, last, 2 * dim, dim, ctx)
    var c3 = slice(out, last, 3 * dim, dim, ctx)
    var c4 = slice(out, last, 4 * dim, dim, ctx)
    var c5 = slice(out, last, 5 * dim, dim, ctx)
    return (c0^, c1^, c2^, c3^, c4^, c5^)


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 3 — krea2 Attention (GQA + QKNorm + RoPE + sigmoid-gate).
# Reference: mmdit.py Attention(197-228), QKNorm(153-160), attention()(51-63).
# ══════════════════════════════════════════════════════════════════════════════
#
# q/k/v are kept in BSHD layout [1, L, H, Dh] throughout (the serenity sdpa /
# rope_interleaved convention, matching ideogram4_dit). This is numerically
# identical to the reference's [B, H, L, D] rearrange + torch SDPA: SDPA is
# per-head, so the head axis position is immaterial to the math.


# ── Per-head RoPE table tiling for BSHD ──────────────────────────────────────
# build_krea2_rope produces a per-token table [L, half]. In BSHD [1, L, H, Dh],
# rope_interleaved flattens to rows (l*H + h), so every head h of token l must
# read table[l]. Tile [L, half] -> [L*H, half] with row (l*H + h) = table[l].
def _tile_rope_table_bshd[t_dtype: DType](
    table: LayoutTensor[t_dtype, _DYN2, MutAnyOrigin],   # [L, half]
    out_t: LayoutTensor[t_dtype, _DYN2, MutAnyOrigin],   # [L*H, half]
    L: Int,
    H: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = L * H * half
    if idx >= total:
        return
    var col = idx % half
    var rest = idx // half
    var h = rest % H
    var l = rest // H
    out_t[l * H + h, col] = rebind[out_t.element_type](table[l, col])


def _tile_rope_table(
    table: Tensor, L: Int, H: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    """Tile a per-token RoPE table [L, half] -> [L*H, half] for BSHD apply
    (row (l*H + h) = table[l]). table dtype is preserved."""
    var dt = table.dtype().to_mojo_dtype()
    var out_n = L * H * half
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * table.dtype().byte_size()
    )
    var in_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](L, half))
    var out_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](L * H, half))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var T = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            table.buf.unsafe_ptr().bitcast[Float32](), in_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[
            _tile_rope_table_bshd[DType.float32],
            _tile_rope_table_bshd[DType.float32],
        ](T, O, L, H, half, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var T = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            table.buf.unsafe_ptr().bitcast[BFloat16](), in_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[
            _tile_rope_table_bshd[DType.bfloat16],
            _tile_rope_table_bshd[DType.bfloat16],
        ](T, O, L, H, half, grid_dim=grid, block_dim=_BLOCK)
    else:
        var T = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            table.buf.unsafe_ptr().bitcast[Float16](), in_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[
            _tile_rope_table_bshd[DType.float16],
            _tile_rope_table_bshd[DType.float16],
        ](T, O, L, H, half, grid_dim=grid, block_dim=_BLOCK)
    var out_shape = List[Int]()
    out_shape.append(L * H)
    out_shape.append(half)
    return Tensor(out_buf^, out_shape^, table.dtype())


def krea2_attention[L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int](
    x: Tensor,             # [1, L, features]
    wq: Tensor,            # [HEADS*HEADDIM, features]   = [6144, 6144]
    wk: Tensor,            # [KVHEADS*HEADDIM, features] = [1536, 6144]
    wv: Tensor,            # [KVHEADS*HEADDIM, features] = [1536, 6144]
    gate_w: Tensor,        # [features, features]        = [6144, 6144]
    wo: Tensor,            # [features, features]        = [6144, 6144]
    qnorm_scale: Tensor,   # [HEADDIM] F32  (QKNorm.qnorm.scale)
    knorm_scale: Tensor,   # [HEADDIM] F32  (QKNorm.knorm.scale)
    cos: Tensor,           # [L, HEADDIM/2]  per-token RoPE table (build_krea2_rope)
    sin: Tensor,           # [L, HEADDIM/2]
    mask: Optional[Tensor],  # None, or additive [1, HEADS, L, L] (pad-to-256 mask)
    ctx: DeviceContext,
) raises -> Tensor:
    """Krea2 Attention forward (mmdit.py:212-228), b==1 inference.

    L/HEADS/KVHEADS/HEADDIM are comptime (sdpa needs a comptime H/S/Dh). For the
    single_mmdit_large_wide arch: HEADS=48, KVHEADS=12, HEADDIM=128, features=6144.

    GQA: q has HEADS heads, k/v have KVHEADS; k/v are repeat_kv'd to HEADS
    (PyTorch repeat_interleave: dst head h reads kv head h//n_rep) before SDPA —
    numerically exact for torch's enable_gqa. QKNorm = krea2_rmsnorm over HEADDIM
    on q,k only (v untouched). RoPE applied to q,k (shared per-token table, tiled
    per head for BSHD). mask: None (single-stream call without padding) or the
    additive [1,HEADS,L,L] pad-to-256 mask (the main-block forward path) — when
    present it routes through the masked sdpa path (must be the q/k/v dtype).
    Output = wo(SDPA-out * sigmoid(gate)). Returns [1, L, features].
    """
    comptime heads = HEADS           # 48
    comptime kvheads = KVHEADS       # 12
    comptime headdim = HEADDIM       # 128
    comptime features = HEADS * HEADDIM  # 6144
    comptime half = HEADDIM // 2     # 64
    comptime n_rep = HEADS // KVHEADS  # 4

    # 1) Projections (no bias).
    var q = linear(x, wq, None, ctx)        # [1, L, heads*headdim]
    var k = linear(x, wk, None, ctx)        # [1, L, kvheads*headdim]
    var v = linear(x, wv, None, ctx)        # [1, L, kvheads*headdim]
    var gate = linear(x, gate_w, None, ctx) # [1, L, features]

    # 2) Reshape to BSHD [1, L, H, Dh].
    var q_shape = List[Int]()
    q_shape.append(1); q_shape.append(L); q_shape.append(heads); q_shape.append(headdim)
    q = reshape(q, q_shape^, ctx)
    var k_shape = List[Int]()
    k_shape.append(1); k_shape.append(L); k_shape.append(kvheads); k_shape.append(headdim)
    k = reshape(k, k_shape^, ctx)
    var v_shape = List[Int]()
    v_shape.append(1); v_shape.append(L); v_shape.append(kvheads); v_shape.append(headdim)
    v = reshape(v, v_shape^, ctx)

    # 3) QKNorm over headdim (krea2_rmsnorm, F32-internal, weight=scale+1). v untouched.
    # KEEP q/k/v IN F32 from here through SDPA. The reference's QKNorm scale can be
    # large (block-0 reaches scale+1 = 52.9×), driving q/k to magnitude ~600 and
    # attention scores to ~3e4 (near-one-hot softmax). bf16-storing q/k there loses
    # ~2-5 absolute per element, which flips the near-one-hot winner and diverges
    # (block-0 attn cos 0.73). torch is robust because F.sdpa upcasts; keeping q/k/v
    # F32 (matches "F32 between intra-block ops") removes the bf16-rounding source
    # — VERIFIED: F32 q/k/v -> F32 sdpa matches torch's bf16 sdpa_out at cos 0.99999.
    var qf = cast_tensor(q, STDtype.F32, ctx)
    var kf = cast_tensor(k, STDtype.F32, ctx)
    var vf = cast_tensor(v, STDtype.F32, ctx)
    qf = krea2_rmsnorm(qf, qnorm_scale, Float32(1.0e-5), ctx)
    kf = krea2_rmsnorm(kf, knorm_scale, Float32(1.0e-5), ctx)

    # 4) RoPE on q,k (F32). Both share the per-token table, but q has `heads` heads
    # and k has `kvheads`, so each gets its own per-head BSHD tiling. rope_interleaved
    # applies the exact ropeapply 2x2 form (chunk-1 verified).
    var cos_q = _tile_rope_table(cos, L, heads, half, ctx)
    var sin_q = _tile_rope_table(sin, L, heads, half, ctx)
    var cos_k = _tile_rope_table(cos, L, kvheads, half, ctx)
    var sin_k = _tile_rope_table(sin, L, kvheads, half, ctx)
    var q_rot = rope_interleaved(qf, cos_q, sin_q, ctx)   # F32
    var k_rot = rope_interleaved(kf, cos_k, sin_k, ctx)   # F32

    # 5) GQA: repeat_kv k,v from kvheads -> heads (BSHD [1,L,kvheads,Dh]), F32.
    var k_full = repeat_kv_f32(k_rot, L, kvheads, n_rep, headdim, ctx)  # F32 [1,L,heads,Dh]
    var v_full = repeat_kv_f32(vf, L, kvheads, n_rep, headdim, ctx)     # F32 [1,L,heads,Dh]

    # 6) SDPA in F32 (Dh=128 -> math-mode). mask present -> masked path (additive
    # [1,HEADS,L,L]); else sdpa_nomask. The mask must match q's dtype (F32 here).
    var scale = Float32(1.0) / sqrt(Float32(headdim))
    var attn_f32: Tensor
    if mask:
        var mask_f32 = cast_tensor(mask.value(), STDtype.F32, ctx)
        attn_f32 = sdpa[1, L, HEADS, HEADDIM](q_rot, k_full, v_full, mask_f32, scale, ctx)
    else:
        attn_f32 = sdpa_nomask[1, L, HEADS, HEADDIM](q_rot, k_full, v_full, scale, ctx)
    # 7) Merge heads, sigmoid-gate (on wo's INPUT), then wo. Keep the attention
    # output in F32 through the gate-mul into wo's input: block-0's attn output
    # reaches magnitude ~190 on the outlier channels (ch 2569/3389) where bf16
    # storage resolution (~1.5) loses precision; staying F32 here matches the
    # reference's internal precision more closely. wo's matmul accumulates F32
    # regardless; the F32 input avoids the lossy bf16 round of the 190-mag value.
    var merge_shape = List[Int]()
    merge_shape.append(1); merge_shape.append(L); merge_shape.append(features)
    var merged_f32 = reshape(attn_f32, merge_shape^, ctx)   # [1, L, features] F32
    var g_f32 = cast_tensor(sigmoid(gate, ctx), STDtype.F32, ctx)  # sigmoid(gate) F32
    var gated_f32 = mul(merged_f32, g_f32, ctx)            # F32 SDPA-out * sigmoid(gate)
    var gated = cast_tensor(gated_f32, x.dtype(), ctx)     # back to activation dtype for wo
    return linear(gated, wo, None, ctx)                    # [1, L, features]


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 4 — krea2 SingleStreamBlock (composes chunks 2+3).
# Reference: mmdit.py SingleStreamBlock (312-337).
# ══════════════════════════════════════════════════════════════════════════════
#
# forward(x, vec, freqs, mask=None)  (mmdit.py:328-337):
#   prescale,preshift,pregate,postscale,postshift,postgate = self.mod(vec)
#   x = x + pregate  * self.attn((1+prescale )*self.prenorm (x) + preshift,  freqs, mask)
#   x = x + postgate * self.mlp ((1+postscale)*self.postnorm(x) + postshift)
# self.mod=DoubleSharedModulation (chunk 2); prenorm/postnorm=krea2_rmsnorm(features);
# self.attn=krea2_attention (chunk 3); self.mlp=krea2_swiglu (chunk 2).
#
# AdaLN broadcast: prescale/preshift/pregate (etc.) are per-channel [features]
# vectors broadcast over the L token axis. ops/elementwise.modulate and
# residual_gate apply a [D] param per-channel over ALL leading rows — exactly the
# AdaLN-over-tokens form — so they are REUSED directly (no broadcast variant
# needed; verified the kernels index param[c] for every (row,c)).
# The +1 reparam lives in modulate's (1+scale); the chunks are RAW (no +1) — so
# we pass the raw modulation chunks straight to modulate (no double-add).


def _reshape_chunk_to_vec(
    chunk: Tensor, features: Int, ctx: DeviceContext
) raises -> Tensor:
    """Reshape a modulation chunk [1, features] -> [features] (a clean [D] param
    for modulate/residual_gate). (b==1 inference.)"""
    var s = List[Int]()
    s.append(features)
    return reshape(chunk, s^, ctx)


def krea2_single_stream_block[L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int](
    x: Tensor,             # [1, L, features]
    vec: Tensor,           # [1, 6*features]  (tproj(t); chunk 5)
    mod_lin: Tensor,       # [6*features]     (DoubleSharedModulation.lin)
    prenorm_scale: Tensor, # [features] F32   (prenorm.scale)
    postnorm_scale: Tensor,# [features] F32   (postnorm.scale)
    wq: Tensor, wk: Tensor, wv: Tensor, gate_w: Tensor, wo: Tensor,  # attn proj
    qnorm_scale: Tensor, knorm_scale: Tensor,                        # attn QKNorm [128] F32
    mlp_gate_w: Tensor, mlp_up_w: Tensor, mlp_down_w: Tensor,        # SwiGLU
    cos: Tensor, sin: Tensor,                                        # rope table [L, headdim/2]
    mask: Optional[Tensor],                                          # None or additive [1,HEADS,L,L]
    ctx: DeviceContext,
) raises -> Tensor:
    """Krea2 SingleStreamBlock forward (mmdit.py:328-337), b==1.

    Composes chunk-2 (DoubleSharedModulation, RMSNorm, SwiGLU) + chunk-3
    (Attention). vec is the timestep-derived [1, 6*features] modulation vector;
    its 6 raw chunks gate the two AdaLN-Zero residual branches. mask: None, or the
    additive [1,HEADS,L,L] pad-to-256 mask (the main-block forward path) routed
    through krea2_attention's masked sdpa path. Returns [1, L, features].
    """
    comptime features = HEADS * HEADDIM   # 6144

    # mod(vec) -> 6 raw chunks, each [1, features].
    var mods = krea2_double_shared_modulation(vec, mod_lin, ctx)
    var prescale = _reshape_chunk_to_vec(mods[0], features, ctx)
    var preshift = _reshape_chunk_to_vec(mods[1], features, ctx)
    var pregate = _reshape_chunk_to_vec(mods[2], features, ctx)
    var postscale = _reshape_chunk_to_vec(mods[3], features, ctx)
    var postshift = _reshape_chunk_to_vec(mods[4], features, ctx)
    var postgate = _reshape_chunk_to_vec(mods[5], features, ctx)

    # Attention branch: x = x + pregate * attn((1+prescale)*prenorm(x) + preshift).
    var xn = krea2_rmsnorm(x, prenorm_scale, Float32(1.0e-5), ctx)        # [1,L,features]
    var xm = modulate(xn, prescale, preshift, ctx)                       # (1+prescale)*xn + preshift
    var a = krea2_attention[L, HEADS, KVHEADS, HEADDIM](
        xm, wq, wk, wv, gate_w, wo, qnorm_scale, knorm_scale, cos, sin, mask, ctx
    )
    var x1 = residual_gate(x, pregate, a, ctx)                          # x + pregate*a

    # MLP branch: x = x + postgate * mlp((1+postscale)*postnorm(x) + postshift).
    var xn2 = krea2_rmsnorm(x1, postnorm_scale, Float32(1.0e-5), ctx)
    var xm2 = modulate(xn2, postscale, postshift, ctx)
    var m = krea2_swiglu(xm2, mlp_gate_w, mlp_up_w, mlp_down_w, ctx)
    var x2 = residual_gate(x1, postgate, m, ctx)
    return x2^


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 5 — embedders + input/output heads.
# Reference: mmdit.py temb(71-88), tmlp(374-378), tproj(395-397),
# txtmlp(387-392), first(358-360), LastLayer(231-242).
# ══════════════════════════════════════════════════════════════════════════════


def krea2_temb(
    t: Tensor, dim: Int, ctx: DeviceContext, out_dtype: STDtype
) raises -> Tensor:
    """Sinusoidal timestep embedding (mmdit.py:71-88). dim=tdim=256.

        half   = dim/2 = 128
        freqs  = exp(-log(1e4) * arange(half)/half)
        args   = (t * 1e3) * freqs          # tfactor=1e3 PRE-SCALE on t
        return cat(cos(args), sin(args), -1) # cos-FIRST, then sin

    REUSES ops/embeddings.timestep_embedding (cos-first, max_period arg) — its
    math is `angle = t_in * freq` with freq = exp(-log(max_period)*i/half), so we
    pass t_in = t * tfactor (=1e3) to fold in the pre-scale. period=1e4 -> the
    max_period arg. The cos-then-sin concat order matches exactly.
    t: [B] (any dtype). Returns [B, dim] (out_dtype). (Reference's extra unit dims
    are layout-only; the caller reshapes to [B,1,dim] as needed.)
    """
    var t_scaled = mul_scalar(t, Float32(1.0e3), ctx)   # tfactor pre-scale
    return timestep_embedding(t_scaled, dim, ctx, Float32(1.0e4), out_dtype)


def krea2_tmlp(
    temb: Tensor,
    w1: Tensor, b1: Tensor,    # Linear(tdim -> features)  (bias=True)
    w2: Tensor, b2: Tensor,    # Linear(features -> features)
    ctx: DeviceContext,
) raises -> Tensor:
    """Tmlp (mmdit.py:374-378): Linear(256->6144) -> GELU(tanh) -> Linear(6144->6144).
    temb [B,1,256] (or [B,256]) -> t [..., features]. Both Linears have bias."""
    var h = linear(temb, w1, Optional[Tensor](b1.clone(ctx)), ctx)
    var hg = gelu(h, ctx)
    return linear(hg, w2, Optional[Tensor](b2.clone(ctx)), ctx)


def krea2_tproj(
    t: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Tproj (mmdit.py:395-397): GELU(tanh) -> Linear(features -> 6*features).
    t [..., features] -> vec [..., 6*features]. Linear has bias."""
    var tg = gelu(t, ctx)
    return linear(tg, w, Optional[Tensor](b.clone(ctx)), ctx)


def krea2_txtmlp(
    context: Tensor,
    rms_scale: Tensor,         # [txtdim] F32  (RMSNorm.scale)
    w1: Tensor, b1: Tensor,    # Linear(txtdim -> features)
    w2: Tensor, b2: Tensor,    # Linear(features -> features)
    ctx: DeviceContext,
) raises -> Tensor:
    """Txtmlp (mmdit.py:387-392): RMSNorm(2560) -> Linear(2560->6144) ->
    GELU(tanh) -> Linear(6144->6144). context [1,L,txtdim] -> [1,L,features].
    RMSNorm = krea2_rmsnorm (F32-internal, scale+1)."""
    var cn = krea2_rmsnorm(context, rms_scale, Float32(1.0e-5), ctx)
    var h = linear(cn, w1, Optional[Tensor](b1.clone(ctx)), ctx)
    var hg = gelu(h, ctx)
    return linear(hg, w2, Optional[Tensor](b2.clone(ctx)), ctx)


def krea2_first(
    x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """First (mmdit.py:358-360): Linear(channels*patch^2 -> features), bias=True.
    Patchified latent [1, N, channels*patch^2 = 64] -> [1, N, features]."""
    return linear(x, w, Optional[Tensor](b.clone(ctx)), ctx)


def krea2_last_layer(
    x: Tensor,             # [1, L, features]
    tvec: Tensor,          # [1, 1, features]  (= t, the tmlp output — NOT tproj's vec)
    norm_scale: Tensor,    # [features] F32   (LastLayer.norm.scale)
    mod_lin: Tensor,       # [2, features]    (SimpleModulation.lin)
    lin_w: Tensor,         # [patch^2*channels, features]  = [64, 6144]
    lin_b: Tensor,         # [patch^2*channels]  = [64]
    features: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """LastLayer (mmdit.py:231-242). forward(x, tvec):
        scale, shift = SimpleModulation(tvec)
        x = (1 + scale) * RMSNorm(x) + shift
        x = Linear(x)                            # bias=True
    tvec = t (tmlp output [1,1,features]), NOT tproj's vec. Returns [1, L, 64].
    SimpleModulation generalizes b>1, but LastLayer (like the whole inference
    path) is b==1; scale/shift come out [1,1,features] -> reshaped to [features]
    for modulate's per-channel broadcast over the L tokens.
    """
    var mods = krea2_simple_modulation(tvec, mod_lin, ctx)  # (scale, shift) each [1,1,features]
    var scale = _reshape_chunk_to_vec(mods[0], features, ctx)  # [features]
    var shift = _reshape_chunk_to_vec(mods[1], features, ctx)
    var xn = krea2_rmsnorm(x, norm_scale, Float32(1.0e-5), ctx)  # [1,L,features]
    var xm = modulate(xn, scale, shift, ctx)                     # (1+scale)*xn + shift
    return linear(xm, lin_w, Optional[Tensor](lin_b.clone(ctx)), ctx)      # [1, L, 64]


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 6 — TextFusionTransformer (processes Qwen3-VL context before the blocks).
# Reference: mmdit.py TextFusionBlock (245-264), TextFusionTransformer (267-309),
# _mask (66-68). Attention here is NO-rope, NO-GQA (heads==kvheads).
# ══════════════════════════════════════════════════════════════════════════════
#
# SHAPE CONTRACT (derived by RUNNING the reference — the forward's local names are
# MISLEADING): context fed in is [B, Lt, n=txtlayers=12, d=2560] (pipeline.py
# predict_velocity:111-115). In TextFusionTransformer.forward the locals are
# `b, l, n, d = x.shape` so l=Lt (TOKENS), n=12 (LAYERS).
#   reshape(b*l, n, d)        -> [B*Lt, 12, d]     layerwise attends over 12 LAYERS
#   2x layerwise blocks (mask=None)
#   rearrange (b l) n d -> b l d n  -> [B, Lt, d, 12]
#   reshape(b*l, d, n)        -> [B*Lt, d, 12]
#   projector Linear(12->1)   -> [B*Lt, d, 1]      collapses the 12-LAYER axis
#   reshape(b, l, d)          -> [B, Lt, d]
#   2x refiner blocks (mask=txtmask)  attends over Lt TOKENS, masked
#
# _mask (66-68): keep-vector [B,Lt] (BOOL) -> [B,1,Lt,Lt] = keep[i] & keep[j]
# (bool outer product). The reference passes this BOOL mask to F.sdpa, which treats
# a BOOLEAN attn_mask as KEEP/MASK-OUT: True (both real) -> attend, False (either
# padded) -> score set to -inf (the position is masked OUT, not softly biased).
# MEASURED (2026-06-24): the reference's main-block mask is bool (mmdit.py:441
# `mask = _mask(mask)` with a bool padded keep), and additive `-1e4` on pad keys
# reproduces the bool reference EXACTLY (cos 1.0 on real block-0 attn). The earlier
# "+1 additive" reading was WRONG — it only matched chunk-6's gen which fed a FLOAT
# keep (float outer product -> additive), not the bool production path. We build
# the additive equivalent: 0.0 where both keep, -1e4 (bf16-safe, not -inf -> no NaN)
# where either is padded. Feed it as ops/attention.sdpa's additive [B,H,Lt,Lt] mask.
comptime _MASK_NEG = Float32(-1.0e4)  # additive pad penalty (bf16-safe stand-in for -inf)


# ── text key-padding mask -> additive [H, Lt, Lt] (B=1): 0 if both keep else -1e4 ─
def _krea2_text_mask_kernel[out_dtype: DType](
    keep: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [Lt]  (1.0 keep / 0.0 pad)
    out_m: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],      # [H*Lt, Lt]  (additive)
    H: Int,
    Lt: Int,
):
    var idx = Int(global_idx.x)
    var total = H * Lt * Lt
    if idx >= total:
        return
    var j = idx % Lt
    var rest = idx // Lt
    var i = rest % Lt
    var ki = rebind[Scalar[DType.float32]](keep[i])
    var kj = rebind[Scalar[DType.float32]](keep[j])
    # bool keep[i] AND keep[j] -> 0.0 (attend); else -1e4 (mask out).
    var v = Float32(0.0) if (ki * kj) > Float32(0.0) else _MASK_NEG
    out_m[rest, j] = rebind[out_m.element_type](v.cast[out_dtype]())


def build_krea2_text_mask(
    keep: Tensor, H: Int, Lt: Int, ctx: DeviceContext, out_dtype: STDtype
) raises -> Tensor:
    """Build the additive attention mask (reference _mask 66-68, BOOL semantics).

    keep: [Lt] F32 (1.0 = real token, 0.0 = padded). Returns [1, H, Lt, Lt] in
    out_dtype, additive mask m[i,j] = 0.0 if (keep[i] AND keep[j]) else -1e4. The
    reference _mask builds a BOOL keep[i]&keep[j] mask that F.sdpa renders as
    -inf-masking on padded positions; -1e4 is the bf16-safe additive equivalent
    (reproduces the bool reference cos 1.0). Broadcast over H for ops/attention.sdpa.

    out_dtype MUST match the q/k/v dtype: ops/attention.sdpa enforces
    q.dtype()==mask.dtype(). 0.0 and -1e4 are bf16-representable. The masked sdpa
    softmax accumulates F32 regardless of mask storage. Pass STDtype.BF16 for the
    bf16 inference/training path (chunk 7's pad-to-256 mask), STDtype.F32 only for
    an F32 model.
    """
    if keep.dtype() != STDtype.F32:
        raise Error("build_krea2_text_mask: keep must be F32")
    var out_n = H * Lt * Lt
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * out_dtype.byte_size()
    )
    var k_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](Lt))
    var m_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](H * Lt, Lt))
    var K = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        keep.buf.unsafe_ptr().bitcast[Float32](), k_rl
    )
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var odt = out_dtype.to_mojo_dtype()
    if odt == DType.float32:
        var M = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), m_rl
        )
        ctx.enqueue_function[
            _krea2_text_mask_kernel[DType.float32],
            _krea2_text_mask_kernel[DType.float32],
        ](K, M, H, Lt, grid_dim=grid, block_dim=_BLOCK)
    elif odt == DType.bfloat16:
        var M = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), m_rl
        )
        ctx.enqueue_function[
            _krea2_text_mask_kernel[DType.bfloat16],
            _krea2_text_mask_kernel[DType.bfloat16],
        ](K, M, H, Lt, grid_dim=grid, block_dim=_BLOCK)
    else:
        var M = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), m_rl
        )
        ctx.enqueue_function[
            _krea2_text_mask_kernel[DType.float16],
            _krea2_text_mask_kernel[DType.float16],
        ](K, M, H, Lt, grid_dim=grid, block_dim=_BLOCK)
    var shape = List[Int]()
    shape.append(1); shape.append(H); shape.append(Lt); shape.append(Lt)
    return Tensor(out_buf^, shape^, out_dtype)


# ── krea2_mha — no-rope, no-GQA multi-head attention (text path) ──────────────
# Same structure as krea2_attention MINUS rope and MINUS repeat_kv (heads==kvheads).
# QKNorm (over headdim), sigmoid-gate, and wo are STILL present. Optional additive
# mask (None for layerwise, the refiner txtmask for refiner blocks).
def krea2_mha[B: Int, S: Int, HEADS: Int, HEADDIM: Int](
    x: Tensor,             # [B, S, features]
    wq: Tensor, wk: Tensor, wv: Tensor, gate_w: Tensor, wo: Tensor,
    qnorm_scale: Tensor, knorm_scale: Tensor,   # [HEADDIM] F32
    mask: Optional[Tensor],                     # None, or additive [B,HEADS,S,S] F32
    ctx: DeviceContext,
) raises -> Tensor:
    """Krea2 text-path Attention (mmdit.py Attention with freqs=None, gqa=False).

    No RoPE (freqs is None in the TextFusionBlock call), no GQA (heads==kvheads),
    so q/k/v all have HEADS heads. QKNorm over HEADDIM on q,k; sigmoid-gate; wo.
    B is comptime (layerwise batches over the LT tokens with B=LT, S=NLAYERS;
    refiner runs B=1, S=LT). mask: additive [B,HEADS,S,S] (refiner) or None.
    Returns [B, S, features].
    """
    comptime features = HEADS * HEADDIM

    var q = linear(x, wq, None, ctx)         # [B,S,features]
    var k = linear(x, wk, None, ctx)
    var v = linear(x, wv, None, ctx)
    var gate = linear(x, gate_w, None, ctx)

    var q_shape = List[Int]()
    q_shape.append(B); q_shape.append(S); q_shape.append(HEADS); q_shape.append(HEADDIM)
    q = reshape(q, q_shape^, ctx)
    var k_shape = List[Int]()
    k_shape.append(B); k_shape.append(S); k_shape.append(HEADS); k_shape.append(HEADDIM)
    k = reshape(k, k_shape^, ctx)
    var v_shape = List[Int]()
    v_shape.append(B); v_shape.append(S); v_shape.append(HEADS); v_shape.append(HEADDIM)
    v = reshape(v, v_shape^, ctx)

    # QKNorm over headdim (q,k only); v untouched. No rope, no repeat_kv.
    q = krea2_rmsnorm(q, qnorm_scale, Float32(1.0e-5), ctx)
    k = krea2_rmsnorm(k, knorm_scale, Float32(1.0e-5), ctx)

    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))
    var attn: Tensor
    if mask:
        attn = sdpa[B, S, HEADS, HEADDIM](q, k, v, mask.value(), scale, ctx)  # [B,S,H,Dh]
    else:
        attn = sdpa_nomask[B, S, HEADS, HEADDIM](q, k, v, scale, ctx)

    var merge_shape = List[Int]()
    merge_shape.append(B); merge_shape.append(S); merge_shape.append(features)
    var merged = reshape(attn, merge_shape^, ctx)
    var g = sigmoid(gate, ctx)
    var gated = mul(merged, g, ctx)
    return linear(gated, wo, None, ctx)


# ── krea2_text_fusion_block (mmdit.py TextFusionBlock 245-264) ────────────────
def krea2_text_fusion_block[B: Int, S: Int, HEADS: Int, HEADDIM: Int](
    x: Tensor,             # [B, S, txtdim]
    prenorm_scale: Tensor, postnorm_scale: Tensor,   # [txtdim] F32
    wq: Tensor, wk: Tensor, wv: Tensor, gate_w: Tensor, wo: Tensor,
    qnorm_scale: Tensor, knorm_scale: Tensor,        # [HEADDIM] F32
    mlp_gate_w: Tensor, mlp_up_w: Tensor, mlp_down_w: Tensor,
    mask: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Tensor:
    """TextFusionBlock forward (mmdit.py:260-264):
        x = x + attn(prenorm(x), mask=mask)     # attn = krea2_mha (no rope/GQA)
        x = x + mlp(postnorm(x))                # mlp = krea2_swiglu
    NOTE: plain residual ADD (no AdaLN gate — TextFusionBlock has no modulation).
    """
    var xn = krea2_rmsnorm(x, prenorm_scale, Float32(1.0e-5), ctx)
    var a = krea2_mha[B, S, HEADS, HEADDIM](
        xn, wq, wk, wv, gate_w, wo, qnorm_scale, knorm_scale, mask, ctx
    )
    var x1 = add(x, a, ctx)
    var xn2 = krea2_rmsnorm(x1, postnorm_scale, Float32(1.0e-5), ctx)
    var m = krea2_swiglu(xn2, mlp_gate_w, mlp_up_w, mlp_down_w, ctx)
    return add(x1, m, ctx)


# ── krea2_text_fusion (mmdit.py TextFusionTransformer 267-309) ────────────────
@fieldwise_init
struct Krea2TextFusionWeights(Copyable, Movable):
    """The weight bundle for ONE TextFusionBlock (layerwise or refiner).

    All Tensors are ArcPointer-shared so the struct is Copyable/Movable (Tensor
    itself is move-only). prenorm/postnorm/qnorm/knorm scales are F32; the
    projections are storage-dtype."""

    var prenorm: ArcPointer[Tensor]
    var postnorm: ArcPointer[Tensor]
    var wq: ArcPointer[Tensor]
    var wk: ArcPointer[Tensor]
    var wv: ArcPointer[Tensor]
    var gate_w: ArcPointer[Tensor]
    var wo: ArcPointer[Tensor]
    var qnorm: ArcPointer[Tensor]
    var knorm: ArcPointer[Tensor]
    var mlp_gate: ArcPointer[Tensor]
    var mlp_up: ArcPointer[Tensor]
    var mlp_down: ArcPointer[Tensor]


def _run_text_fusion_block[B: Int, S: Int, HEADS: Int, HEADDIM: Int](
    x: Tensor, w: Krea2TextFusionWeights, mask: Optional[Tensor], ctx: DeviceContext
) raises -> Tensor:
    """Run one TextFusionBlock from a weight bundle (thin wrapper over
    krea2_text_fusion_block)."""
    return krea2_text_fusion_block[B, S, HEADS, HEADDIM](
        x,
        w.prenorm[], w.postnorm[],
        w.wq[], w.wk[], w.wv[], w.gate_w[], w.wo[],
        w.qnorm[], w.knorm[],
        w.mlp_gate[], w.mlp_up[], w.mlp_down[],
        mask, ctx,
    )


def krea2_text_fusion[LT: Int, NLAYERS: Int, HEADS: Int, HEADDIM: Int](
    context: Tensor,             # [1, LT, NLAYERS, txtdim]  (B=1; NLAYERS=12)
    layerwise0: Krea2TextFusionWeights,
    layerwise1: Krea2TextFusionWeights,
    projector_w: Tensor,         # [1, NLAYERS]  (Linear(NLAYERS -> 1), no bias)
    refiner0: Krea2TextFusionWeights,
    refiner1: Krea2TextFusionWeights,
    refiner_mask: Optional[Tensor],   # additive [1,HEADS,LT,LT] (refiner txtmask) or None
    ctx: DeviceContext,
) raises -> Tensor:
    """TextFusionTransformer forward (mmdit.py:294-309), B==1.

    context [1, LT, NLAYERS=12, txtdim]: LT = caption tokens, NLAYERS = 12 stacked
    Qwen3-VL layers, txtdim=2560. The 2 layerwise blocks attend over the 12 LAYERS
    (seq=NLAYERS, batched over the LT tokens), the projector Linear(12->1) collapses
    the layer axis, and the 2 refiner blocks attend over the LT TOKENS (seq=LT) with
    the txtmask. Returns [1, LT, txtdim].
    """
    var cshape = context.shape()
    var txtdim = cshape[len(cshape) - 1]

    # reshape [1, LT, NLAYERS, d] -> [LT, NLAYERS, d]  (reference reshape(b*l, n, d);
    # b*l = 1*LT = LT). Layerwise blocks attend over NLAYERS, batched over LT.
    var lw_shape = List[Int]()
    lw_shape.append(LT); lw_shape.append(NLAYERS); lw_shape.append(txtdim)
    var x = reshape(context, lw_shape^, ctx)   # [LT, NLAYERS, d]

    # 2 layerwise blocks: B=LT, S=NLAYERS, mask=None (block-diagonal over LT — the
    # SDPA's batch axis keeps each token's 12-layer attention independent, exactly
    # the reference's batched [b*l, n, d] call).
    x = _run_text_fusion_block[LT, NLAYERS, HEADS, HEADDIM](x, layerwise0, None, ctx)
    x = _run_text_fusion_block[LT, NLAYERS, HEADS, HEADDIM](x, layerwise1, None, ctx)

    # projector: collapse the NLAYERS axis (mmdit.py:299-304).
    #   rearrange (b l) n d -> b l d n ; reshape(b*l, d, n) ; Linear(n=NLAYERS -> 1)
    # Equivalent batched: x is [LT, NLAYERS, d]; transpose last two -> [LT, d, NLAYERS];
    # Linear acts on the last dim (NLAYERS) -> [LT, d, 1]; reshape -> [1, LT, d].
    var xt = transpose(x, 1, 2, ctx)                 # [LT, d, NLAYERS]
    var proj = linear(xt, projector_w, None, ctx)    # [LT, d, 1]
    var seq_shape = List[Int]()
    seq_shape.append(1); seq_shape.append(LT); seq_shape.append(txtdim)
    var xr = reshape(proj, seq_shape^, ctx)          # [1, LT, d]

    # 2 refiner blocks: B=1, S=LT, with the refiner txtmask.
    xr = _run_text_fusion_block[1, LT, HEADS, HEADDIM](xr, refiner0, refiner_mask, ctx)
    xr = _run_text_fusion_block[1, LT, HEADS, HEADDIM](xr, refiner1, refiner_mask, ctx)
    return xr^


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK 7a — krea2_forward = SingleStreamDiT.forward (WIRING, resident).
# Reference: mmdit.py SingleStreamDiT.forward (413-461).
# ══════════════════════════════════════════════════════════════════════════════


def _txtf_bundle(
    st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
) raises -> Krea2TextFusionWeights:
    """Load one TextFusionBlock bundle from the checkpoint (scales F32, proj bf16)."""
    return Krea2TextFusionWeights(
        ArcPointer(Tensor.from_view_as_f32(st.tensor_view(prefix + ".prenorm.scale"), ctx)),
        ArcPointer(Tensor.from_view_as_f32(st.tensor_view(prefix + ".postnorm.scale"), ctx)),
        ArcPointer(Tensor.from_view(st.tensor_view(prefix + ".attn.wq.weight"), ctx)),
        ArcPointer(Tensor.from_view(st.tensor_view(prefix + ".attn.wk.weight"), ctx)),
        ArcPointer(Tensor.from_view(st.tensor_view(prefix + ".attn.wv.weight"), ctx)),
        ArcPointer(Tensor.from_view(st.tensor_view(prefix + ".attn.gate.weight"), ctx)),
        ArcPointer(Tensor.from_view(st.tensor_view(prefix + ".attn.wo.weight"), ctx)),
        ArcPointer(Tensor.from_view_as_f32(st.tensor_view(prefix + ".attn.qknorm.qnorm.scale"), ctx)),
        ArcPointer(Tensor.from_view_as_f32(st.tensor_view(prefix + ".attn.qknorm.knorm.scale"), ctx)),
        ArcPointer(Tensor.from_view(st.tensor_view(prefix + ".mlp.gate.weight"), ctx)),
        ArcPointer(Tensor.from_view(st.tensor_view(prefix + ".mlp.up.weight"), ctx)),
        ArcPointer(Tensor.from_view(st.tensor_view(prefix + ".mlp.down.weight"), ctx)),
    )


def _pad_seq_zeros(
    x: Tensor, L: Int, LPAD: Int, F: Int, ctx: DeviceContext
) raises -> Tensor:
    """Pad x [1, L, F] -> [1, LPAD, F] with zeros on the seq axis (LPAD >= L)."""
    if LPAD == L:
        return x.clone(ctx)
    var pshape = List[Int]()
    pshape.append(1); pshape.append(LPAD - L); pshape.append(F)
    var pad = zeros_device(pshape^, x.dtype(), ctx)
    return concat(1, ctx, x, pad)


def krea2_forward[
    LFULL: Int,   # real combined seq length (txtlen + imglen), before pad-to-256
    LPAD: Int,    # padded seq length = ceil(LFULL/256)*256 (the main-block SDPA S)
    LT: Int,      # text/caption token length (txtfusion seq + the txtlen slice point)
    NBLOCKS: Int, # SingleStreamBlock depth (reduced=4 for 7a; 28 for production)
](
    st: ShardedSafeTensors,
    img: Tensor,        # [1, imglen, channels*patch^2 = 64] bf16
    context: Tensor,    # [1, LT, txtlayers=12, txtdim=2560] bf16
    t: Tensor,          # [1] f32 timestep
    pos: Tensor,        # [1, LFULL, 3] f32 (txt zeros + img grid ids)
    ctx: DeviceContext,
) raises -> Tensor:
    """Krea2 SingleStreamDiT.forward (mmdit.py:413-461), b==1 inference.

    LFULL/LPAD/LT/NBLOCKS are comptime. The arch is the single_mmdit_large_wide
    config: features=6144, heads=48, kvheads=12, headdim=128, txtheads=20,
    txtlayers=12, txtdim=2560, patch=2, channels=16, tdim=256, theta=1e3.
    mask is all-ones at b==1 inference (no text pad); the ONLY masked positions are
    the pad-to-LPAD region, which the main blocks mask out via the additive
    [1,heads,LPAD,LPAD] mask. Returns the velocity on the image tokens [1, imglen, 64].
    """
    comptime FEATURES = 6144
    comptime HEADS = 48
    comptime KVHEADS = 12
    comptime HEADDIM = 128
    comptime TXTHEADS = 20
    comptime TXTHD = 128          # txtdim/txtheads = 2560/20
    comptime NLAYERS_TXT = 12
    comptime TDIM = 256
    var imglen = img.shape()[1]

    # 1) img = first(img)  -> [1, imglen, FEATURES].
    var img_e = krea2_first(
        img,
        Tensor.from_view(st.tensor_view("w.first.weight"), ctx),
        Tensor.from_view(st.tensor_view("w.first.bias"), ctx),
        ctx,
    )

    # 2) t = tmlp(temb(t, tdim))  -> [1, 1, FEATURES].
    var te = krea2_temb(t, TDIM, ctx, STDtype.BF16)   # [1, 256]
    var t_vec = krea2_tmlp(
        te,
        Tensor.from_view(st.tensor_view("w.tmlp.0.weight"), ctx),
        Tensor.from_view(st.tensor_view("w.tmlp.0.bias"), ctx),
        Tensor.from_view(st.tensor_view("w.tmlp.2.weight"), ctx),
        Tensor.from_view(st.tensor_view("w.tmlp.2.bias"), ctx),
        ctx,
    )
    var tshape = List[Int]()
    tshape.append(1); tshape.append(1); tshape.append(FEATURES)
    var t3 = reshape(t_vec, tshape^, ctx)             # [1, 1, FEATURES]  (= LastLayer tvec)

    # 3) tvec = tproj(t)  -> [1, 1, 6*FEATURES]  (the block modulation vector).
    var blk_vec = krea2_tproj(
        t3,
        Tensor.from_view(st.tensor_view("w.tproj.1.weight"), ctx),
        Tensor.from_view(st.tensor_view("w.tproj.1.bias"), ctx),
        ctx,
    )
    var bvshape = List[Int]()
    bvshape.append(1); bvshape.append(6 * FEATURES)
    var blk_vec2 = reshape(blk_vec, bvshape^, ctx)    # [1, 6*FEATURES] for the block

    # 4-5) context = txtfusion(context, txtmask). At b==1 the txtmask is all-ones
    # (no caption padding) => refiner runs the no-op path (chunk-6: refiner mask=None).
    var lw0 = _txtf_bundle(st, "w.txtfusion.layerwise_blocks.0", ctx)
    var lw1 = _txtf_bundle(st, "w.txtfusion.layerwise_blocks.1", ctx)
    var rf0 = _txtf_bundle(st, "w.txtfusion.refiner_blocks.0", ctx)
    var rf1 = _txtf_bundle(st, "w.txtfusion.refiner_blocks.1", ctx)
    var ctx_fused = krea2_text_fusion[LT, NLAYERS_TXT, TXTHEADS, TXTHD](
        context, lw0, lw1,
        Tensor.from_view(st.tensor_view("w.txtfusion.projector.weight"), ctx),
        rf0, rf1, Optional[Tensor](None), ctx,
    )                                                  # [1, LT, txtdim]

    # 6) context = txtmlp(context)  -> [1, LT, FEATURES].
    var ctx_proj = krea2_txtmlp(
        ctx_fused,
        Tensor.from_view_as_f32(st.tensor_view("w.txtmlp.0.scale"), ctx),
        Tensor.from_view(st.tensor_view("w.txtmlp.1.weight"), ctx),
        Tensor.from_view(st.tensor_view("w.txtmlp.1.bias"), ctx),
        Tensor.from_view(st.tensor_view("w.txtmlp.3.weight"), ctx),
        Tensor.from_view(st.tensor_view("w.txtmlp.3.bias"), ctx),
        ctx,
    )

    # 7-8) combined = cat(context, img, dim=1)  -> [1, LFULL, FEATURES]  (context THEN img).
    var combined = concat(1, ctx, ctx_proj, img_e)     # [1, LFULL, FEATURES]

    # 9) pad-to-LPAD: combined (zeros), pos (zeros), mask (False) on the seq axis.
    var combined_p = _pad_seq_zeros(combined, LFULL, LPAD, FEATURES, ctx)  # [1, LPAD, F]

    # 10) main-block mask = _mask(padded keep). keep = ones[0:LFULL], zeros[LFULL:LPAD].
    # build_krea2_text_mask wants keep [LPAD] F32. Build it host-side.
    var keep_host = List[Float32]()
    for i in range(LPAD):
        keep_host.append(Float32(1.0) if i < LFULL else Float32(0.0))
    var keep_shape = List[Int]()
    keep_shape.append(LPAD)
    var keep = Tensor.from_host(keep_host^, keep_shape^, STDtype.F32, ctx)
    var blk_mask = build_krea2_text_mask(keep, HEADS, LPAD, ctx, STDtype.BF16)  # [1,HEADS,LPAD,LPAD]
    # Build the Optional ONCE (Tensor is move-only); pass it borrowed to every block.
    var blk_mask_opt = Optional[Tensor](blk_mask^)

    # 11) freqs = posemb(pos): pad pos to LPAD (zeros), build the rope table [LPAD,64].
    var pos_flat_shape = List[Int]()
    pos_flat_shape.append(LFULL * 3)
    var pos_flat = reshape(pos, pos_flat_shape^, ctx)         # [LFULL*3]
    var pos_pad_host = List[Float32]()
    var pos_host = pos_flat.to_host(ctx)
    for i in range(LFULL * 3):
        pos_pad_host.append(pos_host[i])
    for _i in range((LPAD - LFULL) * 3):
        pos_pad_host.append(Float32(0.0))
    var pos_pad_shape = List[Int]()
    pos_pad_shape.append(LPAD * 3)
    var pos_pad = Tensor.from_host(pos_pad_host^, pos_pad_shape^, STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_pad, axes, Float32(1.0e3), ctx, STDtype.F32)  # ([LPAD,64], [LPAD,64])

    # 12) N x SingleStreamBlock WITH the pad-to-LPAD mask. S = LPAD (comptime).
    var x = combined_p^
    for li in range(NBLOCKS):
        var p = String("w.blocks.") + String(li)
        x = krea2_single_stream_block[LPAD, HEADS, KVHEADS, HEADDIM](
            x,
            blk_vec2,
            Tensor.from_view(st.tensor_view(p + ".mod.lin"), ctx),
            Tensor.from_view_as_f32(st.tensor_view(p + ".prenorm.scale"), ctx),
            Tensor.from_view_as_f32(st.tensor_view(p + ".postnorm.scale"), ctx),
            Tensor.from_view(st.tensor_view(p + ".attn.wq.weight"), ctx),
            Tensor.from_view(st.tensor_view(p + ".attn.wk.weight"), ctx),
            Tensor.from_view(st.tensor_view(p + ".attn.wv.weight"), ctx),
            Tensor.from_view(st.tensor_view(p + ".attn.gate.weight"), ctx),
            Tensor.from_view(st.tensor_view(p + ".attn.wo.weight"), ctx),
            Tensor.from_view_as_f32(st.tensor_view(p + ".attn.qknorm.qnorm.scale"), ctx),
            Tensor.from_view_as_f32(st.tensor_view(p + ".attn.qknorm.knorm.scale"), ctx),
            Tensor.from_view(st.tensor_view(p + ".mlp.gate.weight"), ctx),
            Tensor.from_view(st.tensor_view(p + ".mlp.up.weight"), ctx),
            Tensor.from_view(st.tensor_view(p + ".mlp.down.weight"), ctx),
            rope[0], rope[1],
            blk_mask_opt,
            ctx,
        )

    # 13) final = last_layer(combined, t)  (tvec = t3, the tmlp output).
    var final = krea2_last_layer(
        x,
        t3,
        Tensor.from_view_as_f32(st.tensor_view("w.last.norm.scale"), ctx),
        Tensor.from_view(st.tensor_view("w.last.modulation.lin"), ctx),
        Tensor.from_view(st.tensor_view("w.last.linear.weight"), ctx),
        Tensor.from_view(st.tensor_view("w.last.linear.bias"), ctx),
        FEATURES,
        ctx,
    )                                                  # [1, LPAD, 64]

    # 14) output = final[:, txtlen : txtlen+imglen, :]  (the image tokens).
    return slice(final, 1, LT, imglen, ctx)            # [1, imglen, 64]
