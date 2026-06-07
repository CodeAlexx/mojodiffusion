# models/dit/ideogram4_mrope.mojo — Ideogram-4 interleaved MRoPE cos/sin builder.
# 1:1 port of Ideogram4MRoPE.forward (ideogram4-ref modeling_ideogram4.py:65-104).
#
#   inv_freq = 1/base^(arange(0,head_dim,2)/head_dim)        # [half], exponent=2d/hd=d/half
#   freqs[axis] = inv_freq * position_ids[...,axis]
#   freqs_t = freqs[0]; H at idx d%3==1 & d<section[1]*3; W at d%3==2 & d<section[2]*3
#   emb = cat(freqs_t, freqs_t, -1); cos=emb.cos(), sin=emb.sin()  -> [1,L,head_dim]
# head_dim=256, base=5e6, section=(24,20,20) -> sec_h=sec_w=60. axis order (t,h,w).
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import cos as fcos, sin as fsin, exp, log, floor
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _ideogram4_mrope_kernel[out_dtype: DType](
    pos: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cosx: LayoutTensor[out_dtype, _DYN1, MutAnyOrigin],
    sinx: LayoutTensor[out_dtype, _DYN1, MutAnyOrigin],
    head_dim: Int,
    half: Int,
    sec_h: Int,
    sec_w: Int,
    log_theta: Float32,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx >= n:
        return
    var l = idx // head_dim
    var dd = idx % head_dim
    var d = dd if dd < half else dd - half
    var axis = 0
    var m = d % 3
    if m == 1 and d < sec_h:
        axis = 1
    elif m == 2 and d < sec_w:
        axis = 2
    var pos_val = rebind[Scalar[DType.float32]](pos[l * 3 + axis])
    # inv_freq is stored BF16 in the real bf16 model (m.to(bf16) casts the buffer,
    # forward upcasts to f32) — round to bf16 to match. At pos~65536 the bf16
    # rounding of inv dominates (f32 inv gives cos-sim 0.71, bf16 gives 1.0).
    var inv = exp((-Float32(d) / Float32(half)) * log_theta).cast[DType.bfloat16]().cast[DType.float32]()
    var angle_f32 = pos_val * inv
    # GPU has no F64 trig, but F64 arithmetic is fine: do an accurate range
    # reduction in F64 (Mojo F32 cos/sin is wrong for pos ~ 65536), then F32 trig
    # on the small reduced angle.
    comptime TWO_PI = Float64(6.283185307179586476925286766559)
    var a = Float64(angle_f32)
    var k = floor(a / TWO_PI + 0.5)
    var reduced = Float32(a - k * TWO_PI)
    cosx[idx] = rebind[cosx.element_type](fcos(reduced).cast[out_dtype]())
    sinx[idx] = rebind[sinx.element_type](fsin(reduced).cast[out_dtype]())


def build_ideogram4_mrope(
    position_ids: Tensor,
    head_dim: Int,
    mrope_section: List[Int],
    theta: Float32,
    ctx: DeviceContext,
    out_dtype: STDtype,
) raises -> Tuple[Tensor, Tensor]:
    """Ideogram-4 interleaved MRoPE. position_ids: F32 [1,L,3] (t,h,w). Returns
    (cos, sin) each [1, L, head_dim] in out_dtype."""
    if position_ids.dtype() != STDtype.F32:
        raise Error("build_ideogram4_mrope: position_ids must be F32 (use from_view_as_f32)")
    if head_dim % 2 != 0:
        raise Error("build_ideogram4_mrope: head_dim must be even")
    if len(mrope_section) != 3:
        raise Error("build_ideogram4_mrope: mrope_section needs 3 entries")
    var total = position_ids.numel()
    if total % 3 != 0:
        raise Error("build_ideogram4_mrope: position_ids numel must be L*3")
    var L = total // 3
    var half = head_dim // 2
    var sec_h = mrope_section[1] * 3
    var sec_w = mrope_section[2] * 3
    var n = L * head_dim

    var cos_bytes = n * out_dtype.byte_size()
    var cos_buf = ctx.enqueue_create_buffer[DType.uint8](cos_bytes)
    var sin_buf = ctx.enqueue_create_buffer[DType.uint8](cos_bytes)

    var p_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        position_ids.buf.unsafe_ptr().bitcast[Float32](), p_rl
    )
    var lt = log(theta)
    var grid = (n + _BLOCK - 1) // _BLOCK

    var out_shape = [1, L, head_dim]
    if out_dtype == STDtype.F32:
        var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](cos_buf.unsafe_ptr().bitcast[Float32](), o_rl)
        var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](sin_buf.unsafe_ptr().bitcast[Float32](), o_rl)
        ctx.enqueue_function[_ideogram4_mrope_kernel[DType.float32], _ideogram4_mrope_kernel[DType.float32]](
            P, C, S, head_dim, half, sec_h, sec_w, lt, n, grid_dim=grid, block_dim=_BLOCK)
    elif out_dtype == STDtype.BF16:
        var C = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](cos_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
        var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](sin_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
        ctx.enqueue_function[_ideogram4_mrope_kernel[DType.bfloat16], _ideogram4_mrope_kernel[DType.bfloat16]](
            P, C, S, head_dim, half, sec_h, sec_w, lt, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        raise Error("build_ideogram4_mrope: out_dtype must be F32 or BF16")
    ctx.synchronize()
    return (Tensor(cos_buf^, out_shape.copy(), out_dtype), Tensor(sin_buf^, out_shape^, out_dtype))
