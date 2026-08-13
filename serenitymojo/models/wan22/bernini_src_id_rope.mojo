# serenitymojo/models/wan22/bernini_src_id_rope.mojo
#
# BERNINI-R source-id RoPE — the ONE architectural delta of Bernini-R vs stock
# Wan2.2. Everything else in Bernini's WanRotaryPosEmbed is the standard Wan 3D
# rope we already build in `models/dit/wan22_dit.mojo::wan22_build_rope`
# (-> ops/rope_tables.build_multiaxis_rope_tables). This module adds the
# `use_src_id_rotary_emb` phase.
#
# ── The exact math (Bernini/bernini/models/transformer_wan.py, :274-289) ──
#   freqs_visual_id = get_1d_rotary_pos_embed(
#       attention_head_dim, pos=[source_id], theta,
#       use_real=False, repeat_interleave_real=False, freqs_dtype=float64)
#   freqs = freqs * freqs_visual_id            # complex multiply, broadcast over S
#
#   get_1d_rotary_pos_embed(dim, pos) with use_real=False returns
#       freqs_cis[k] = exp(i * pos * theta^(-2k/dim)),  k in [0, dim/2)
#   Here dim = attention_head_dim (the FULL head dim, e.g. 128), so
#       freqs_visual_id[k] = exp(i * source_id * theta^(-2k/head_dim)),  k in [0, half)
#   with half = head_dim/2. Note the exponent uses the FULL head_dim as the
#   denominator and the CONCAT column index k — it is NOT the per-band inv_freq.
#
# ── Band / column ordering matched (verify vs wan22_build_rope) ──
#   wan22_build_rope -> wan22_rope_axes(head_dim) = [head_dim-4*d6, 2*d6, 2*d6]
#   (d6 = head_dim//6). For head_dim=128 -> full dims [44,42,42], complex-pair
#   halves [22,21,21] laid out CONCAT [T | H | W] as columns 0..half-1
#   (ops/rope_tables.build_multiaxis_rope_tables: column `col` belongs to the
#   axis block it falls in; table is [rows, half]). Bernini's freqs_visual_id
#   multiplies the concatenated 64-vector component-wise by concat index k, so
#   the source-id angle for table COLUMN `k` is exactly
#       phi_k = source_id * theta^(-2k/head_dim) = source_id * theta^(-k/half).
#   This is independent of which T/H/W band column k sits in — it is a single
#   continuous 1-D rope over the full head dim, indexed by concat column.
#
# ── Applied to the standard cos/sin table ──
#   Standard column k stores (cos_k, sin_k) = (cos(a_k), sin(a_k)) for the
#   stock 3-axis angle a_k. The complex multiply by exp(i*phi_k) rotates that
#   pair by phi_k:
#       cos'_k = cos_k*cos(phi_k) - sin_k*sin(phi_k) = cos(a_k + phi_k)
#       sin'_k = cos_k*sin(phi_k) + sin_k*cos(phi_k) = sin(a_k + phi_k)
#   source_id = 0 => phi_k = 0 => cos(phi)=1, sin(phi)=0 => the table is
#   returned BIT-IDENTICAL to the stock wan rope (identity property).
#   source_id may be FRACTIONAL (Bernini interpolates reference ids).
#
# Pure table transform: no new attention kernel — the block's existing rope
# apply (ops/rope.rope_interleaved) consumes the modified cos/sin unchanged.
#
# Mojo 1.0.0b1, NVIDIA GPU. Trig math in F32; storage dtype preserved.

from std.math import exp, log, cos, sin
from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import concat
from serenitymojo.models.dit.wan22_dit import wan22_build_rope


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256


# One thread per (row, col). Reads stock (cos_k,sin_k), rotates by the per-column
# source-id phase phi_k = source_id * theta^(-col/half), writes rotated pair.
def _src_id_rotate_kernel[out_dtype: DType](
    cos_in: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],
    sin_in: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],
    cos_out: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],
    sin_out: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],
    rows_w: Int32,
    half_w: Int32,
    source_id: Float32,
    theta: Float32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx >= total:
        return
    var row = idx // half
    var col = idx % half

    # phi_k = source_id * theta^(-col/half) == source_id * theta^(-2*col/head_dim)
    # (head_dim == 2*half). Matches get_1d_rotary_pos_embed(dim=head_dim)[col].
    var omega = exp(-log(theta) * (Float32(col) / Float32(half)))
    var phi = source_id * omega
    var cp = cos(phi)
    var sp = sin(phi)

    var c = rebind[Scalar[out_dtype]](cos_in[row, col]).cast[DType.float32]()
    var s = rebind[Scalar[out_dtype]](sin_in[row, col]).cast[DType.float32]()

    # complex multiply (c + i s) * (cp + i sp)
    var nc = c * cp - s * sp
    var ns = c * sp + s * cp

    cos_out[row, col] = rebind[cos_out.element_type](nc.cast[out_dtype]())
    sin_out[row, col] = rebind[sin_out.element_type](ns.cast[out_dtype]())


# Apply the Bernini source-id rotation to a STANDARD wan rope table pair.
# cos_in/sin_in: [rows, half] (from wan22_build_rope), any of F32/BF16/F16.
# Returns (cos_out, sin_out) same shape/dtype. source_id=0 => bit-identical copy.
def apply_src_id_rope_rotation(
    cos_in: Tensor, sin_in: Tensor, source_id: Float32, theta: Float32,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var cshape = cos_in.shape()
    if len(cshape) != 2:
        raise Error("apply_src_id_rope_rotation: cos table must be rank-2 [rows,half]")
    var sshape = sin_in.shape()
    if len(sshape) != 2 or sshape[0] != cshape[0] or sshape[1] != cshape[1]:
        raise Error("apply_src_id_rope_rotation: cos/sin shape mismatch")
    if cos_in.dtype() != sin_in.dtype():
        raise Error("apply_src_id_rope_rotation: cos/sin dtype mismatch")
    var rows = cshape[0]
    var half = cshape[1]
    var out_dtype = cos_in.dtype()

    var cos_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * half * out_dtype.byte_size()
    )
    var sin_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * half * out_dtype.byte_size()
    )
    var rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK
    var odt = out_dtype.to_mojo_dtype()

    if odt == DType.float32:
        var CI = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos_in.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        var SI = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin_in.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        var CO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        var SO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_src_id_rotate_kernel[DType.float32]](CI, SI, CO, SO, Int32(rows), Int32(half), source_id, theta,
          grid_dim=grid, block_dim=_BLOCK)
    elif odt == DType.bfloat16:
        var CI = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos_in.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        var SI = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin_in.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        var CO = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        var SO = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_src_id_rotate_kernel[DType.bfloat16]](CI, SI, CO, SO, Int32(rows), Int32(half), source_id, theta,
          grid_dim=grid, block_dim=_BLOCK)
    elif odt == DType.float16:
        var CI = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos_in.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        var SI = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin_in.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        var CO = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        var SO = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_src_id_rotate_kernel[DType.float16]](CI, SI, CO, SO, Int32(rows), Int32(half), source_id, theta,
          grid_dim=grid, block_dim=_BLOCK)
    else:
        raise Error("apply_src_id_rope_rotation: dtype must be F32, BF16, or F16")

    var cos_shape = List[Int]()
    cos_shape.append(rows)
    cos_shape.append(half)
    var sin_shape = List[Int]()
    sin_shape.append(rows)
    sin_shape.append(half)
    return (
        Tensor(cos_buf^, cos_shape^, out_dtype),
        Tensor(sin_buf^, sin_shape^, out_dtype),
    )


# One packed conditioning segment: an (f,h,w) patch grid tagged with a source_id.
# target stream uses source_id=0 (identity); each conditioning reference uses
# 1, 2, ... (fractional ids allowed for interpolated references).
@fieldwise_init
struct BerniniRopeSegment(Copyable, Movable, ImplicitlyCopyable):
    var f: Int
    var h: Int
    var w: Int
    var source_id: Float32


# Build a full-sequence rope for a packed sequence of segments, each rotated by
# its own source_id, concatenated along the token (row) axis in list order.
# Returns (cos, sin) each [sum_rows, head_dim/2]. This is what Tier 2b's
# seq-concat conditioning calls; built now so it is gated.
def build_bernini_src_id_rope(
    segments: List[BerniniRopeSegment],
    head_dim: Int, theta: Float32, dtype: STDtype, ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    if len(segments) == 0:
        raise Error("build_bernini_src_id_rope: need at least one segment")

    var seg0 = segments[0]
    var std0 = wan22_build_rope(
        seg0.f, seg0.h, seg0.w, head_dim, theta, dtype, ctx
    )
    var rot0 = apply_src_id_rope_rotation(
        std0[0], std0[1], seg0.source_id, theta, ctx
    )
    # Seed the accumulators with an owned copy (single-tensor concat) so we never
    # have to move a tensor out of the returned tuple.
    var cos_acc = concat(0, ctx, rot0[0])
    var sin_acc = concat(0, ctx, rot0[1])

    for i in range(1, len(segments)):
        var seg = segments[i]
        var std = wan22_build_rope(
            seg.f, seg.h, seg.w, head_dim, theta, dtype, ctx
        )
        var rot = apply_src_id_rope_rotation(
            std[0], std[1], seg.source_id, theta, ctx
        )
        cos_acc = concat(0, ctx, cos_acc, rot[0])
        sin_acc = concat(0, ctx, sin_acc, rot[1])

    return (cos_acc^, sin_acc^)
