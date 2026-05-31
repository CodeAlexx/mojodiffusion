# serenitymojo/models/klein/lora_block.mojo
#
# LoRA-ON-PROJECTION helpers shared by the Klein double/single block LoRA
# variants (double_block.mojo `double_block_lora_*`, single_block.mojo
# `single_block_lora_*`). This is the SAME LoRA math as
# serenitymojo/training/train_step.mojo (_lora_fwd:164, _lora_bwd:194,
# LoraAdapter:120, LoraGrads:185) — we REUSE the imported LoraAdapter / LoraGrads
# structs and replicate the two tiny linear-path helpers locally, with ONE
# addition the trainer's _lora_bwd discards: the LoRA branch's contribution to
# the projection INPUT grad (d_x).
#
# WHY THE TRAINER MATH IS THE AUTHORITY
#   For a projection y = linear(x, W) (W [out,in]), the LoRA-adapted output is
#       y' = linear(x, W) + scale·((x @ Aᵀ) @ Bᵀ)
#   with A=[rank,in], B=[out,rank], scale = alpha/rank. This MATCHES exactly the
#   inference merge semantics in serenitymojo/lora.mojo::_apply_slot (FULL slot
#   adds scale·B@A into W → merged W' gives y' = x @ (W + scale·B@A)ᵀ ≡ the same).
#
# BACKWARD (given d_y' at the projection output, x the saved projection input):
#       d_dy   = scale · d_y'                          [M,out]
#       d_B    = d_dyᵀ @ t           (t = x @ Aᵀ)       [out,rank]   (linear_backward d_w)
#       d_t    = d_dy  @ B                              [M,rank]     (linear_backward d_x)
#       d_A    = d_tᵀ  @ x                              [rank,in]    (linear_backward d_w)
#       d_x_lo = d_t   @ A                              [M,in]       (linear_backward d_x)
#   The base path (frozen W, base d_W discarded for LoRA) ALSO yields d_x_base =
#   d_y' @ W from the block's existing linear_backward; the caller SUMS d_x_lo
#   into that. d_A / d_B are returned to the optimizer.
#
# Mojo 1.0.0b1: def not fn; Tensor move-only crosses API as host List[Float32];
# no-bias linear = linear(x, w, Optional[Tensor](None), ctx).

from std.gpu.host import DeviceContext, HostBuffer
from std.collections import List, Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx, linear_backward_dw,
)
from serenitymojo.ops.tensor_algebra import add, concat, mul_scalar, slice

# REUSE the trainer's LoRA structs (the target authority).
from serenitymojo.training.train_step import LoraAdapter, LoraGrads


struct _HostGradPair(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]

    def __init__(out self, var d_a: List[Float32], var d_b: List[Float32]):
        self.d_a = d_a^
        self.d_b = d_b^


def _host_from_f32_buffer(
    host: HostBuffer[DType.uint8], n: Int
) -> List[Float32]:
    var out = List[Float32]()
    var fp = host.unsafe_ptr().bitcast[Float32]()
    for i in range(n):
        out.append(fp[i])
    return out^


def _to_host_pair_f32(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> _HostGradPair:
    if a.dtype() != STDtype.F32 or b.dtype() != STDtype.F32:
        raise Error("_to_host_pair_f32: expected F32 tensors")
    var ahost = ctx.enqueue_create_host_buffer[DType.uint8](a.nbytes())
    var bhost = ctx.enqueue_create_host_buffer[DType.uint8](b.nbytes())
    ctx.enqueue_copy(dst_buf=ahost, src_buf=a.buf)
    ctx.enqueue_copy(dst_buf=bhost, src_buf=b.buf)
    ctx.synchronize()
    var ah = _host_from_f32_buffer(ahost, a.numel())
    var bh = _host_from_f32_buffer(bhost, b.numel())
    return _HostGradPair(ah^, bh^)


# Adapter forward contribution on x [M,in] → [M,out] (host list in/out).
# Byte-identical to train_step._lora_fwd.
def klein_lora_fwd(
    x_h: List[Float32], lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var nb1 = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        nb1^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], STDtype.F32, ctx),
        nb2^, ctx,
    ).to_host(ctx)                                   # [M,out]
    var out = List[Float32]()
    for i in range(len(dy)):
        out.append(lo.scale * dy[i])
    return out^


# Device-resident sibling of klein_lora_fwd. A/B still come from the host-owned
# adapter, but the activation input and delta output stay on device.
def klein_lora_fwd_device(
    x: Tensor, lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> Tensor:
    var nb1 = Optional[Tensor](None)
    var t = linear(
        x,
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        nb1^, ctx,
    )
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        t,
        Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], STDtype.F32, ctx),
        nb2^, ctx,
    )
    return mul_scalar(dy, lo.scale, ctx)


# LoRA backward that ALSO returns the LoRA branch's contribution to d_x.
# d_a/d_b match train_step._lora_bwd exactly; d_x_lo is the term that file drops.
struct KleinLoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]   # LoRA contribution to the projection INPUT grad

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def klein_lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> KleinLoraGrads:
    # t = x @ Aᵀ  (recompute; cheap)
    var nb_t = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        nb_t^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var d_dy = List[Float32]()
    for i in range(len(d_contrib_h)):
        d_dy.append(lo.scale * d_contrib_h[i])       # [M,out]
    # dy = t @ Bᵀ  → d_B (d_w) and d_t (d_x)
    var lbB = linear_backward(
        Tensor.from_host(d_dy^, [M, lo.out_f], STDtype.F32, ctx),
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], STDtype.F32, ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.to_host(ctx)                   # [M,rank]
    var d_b = lbB.d_w.to_host(ctx)                   # [out_f,rank]
    # t = x @ Aᵀ  → d_A (d_w) and d_x_lo (d_x)
    var lbA = linear_backward(
        Tensor.from_host(d_t^, [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_x_lo = lbA.d_x.to_host(ctx)                # [M,in_f]
    var d_a = lbA.d_w.to_host(ctx)                   # [rank,in_f]
    return KleinLoraGrads(d_a^, d_b^, d_x_lo^)


struct KleinLoraDeviceGrads(Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: Tensor

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: Tensor
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


# Device-resident sibling of klein_lora_bwd. d_A/d_B still leave as host lists
# for the existing optimizer, while the large d_x_lo tensor stays on device.
def klein_lora_bwd_device(
    d_contrib: Tensor, x: Tensor, lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> KleinLoraDeviceGrads:
    var a_t = Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx)
    var b_t = Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], STDtype.F32, ctx)

    # t = x @ Aᵀ  (recompute; cheap)
    var nb_t = Optional[Tensor](None)
    var t = linear(x, a_t, nb_t^, ctx)                # [M,rank]
    var d_dy = mul_scalar(d_contrib, lo.scale, ctx)   # [M,out]

    # dy = t @ Bᵀ.
    var d_t = linear_backward_dx(d_dy, b_t, M, lo.rank, lo.out_f, ctx)
    var d_b_t = linear_backward_dw(d_dy, t, M, lo.rank, lo.out_f, ctx)

    # t = x @ Aᵀ.
    var d_x_lo = linear_backward_dx(d_t, a_t, M, lo.in_f, lo.rank, ctx)
    var d_a_t = linear_backward_dw(d_t, x, M, lo.in_f, lo.rank, ctx)

    var pair = _to_host_pair_f32(d_a_t, d_b_t, ctx)
    return KleinLoraDeviceGrads(pair.d_a.copy(), pair.d_b.copy(), d_x_lo^)


# ── host slice/scatter helpers for partial-projection LoRA (single block) ────
# Take the first `c0` columns of each row of a [rows, total] host buffer.
def klein_take_cols(
    src: List[Float32], rows: Int, total: Int, c0: Int
) -> List[Float32]:
    var o = List[Float32]()
    for r in range(rows):
        var base = r * total
        for c in range(c0):
            o.append(src[base + c])
    return o^


# Add a [rows, c0] delta into the first c0 columns of a [rows, total] buffer,
# returning the merged [rows, total] buffer (columns >= c0 unchanged).
def klein_add_cols(
    dst: List[Float32], delta: List[Float32], rows: Int, total: Int, c0: Int
) -> List[Float32]:
    var o = List[Float32]()
    for r in range(rows):
        var base = r * total
        for c in range(total):
            if c < c0:
                o.append(dst[base + c] + delta[r * c0 + c])
            else:
                o.append(dst[base + c])
    return o^


def klein_take_cols_device(
    src: Tensor, rows: Int, total: Int, c0: Int, ctx: DeviceContext
) raises -> Tensor:
    return slice(src, 1, 0, c0, ctx)


def klein_add_cols_device(
    dst: Tensor, delta: Tensor, rows: Int, total: Int, c0: Int, ctx: DeviceContext
) raises -> Tensor:
    var first = slice(dst, 1, 0, c0, ctx)
    var merged = add(first, delta, ctx)
    if c0 == total:
        return merged^
    var rest = slice(dst, 1, c0, total - c0, ctx)
    return concat(1, ctx, merged, rest)
