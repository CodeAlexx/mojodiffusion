# serenitymojo/models/sdxl/lora_block.mojo
#
# LoRA-ON-PROJECTION for the SDXL SpatialTransformer's BasicTransformerBlock.
# Mirrors the JUST-VERIFIED-GREEN ERNIE LoRA template (models/ernie/lora_block.mojo
# + the Klein original models/klein/lora_block.mojo), specialized to SDXL's TEN
# un-fused target projections per BasicTransformerBlock:
#   attn1.{to_q, to_k, to_v, to_out.0}   (self-attn: Q/K/V from the SAME LN1 input)
#   attn2.{to_q, to_k, to_v, to_out.0}   (cross-attn: Q from LN2 input, K/V from context)
#   ff.net.0.proj  (GEGLU in-projection, in=C  out=2*Cff)
#   ff.net.2       (FF out-projection,   in=Cff out=C)
# (SDXL has separate to_q/to_k/to_v — unlike Klein's fused qkv — so 10 adapters.)
#
# WHY THE TRAINER MATH IS THE AUTHORITY (identical to Ernie/Klein lora_block.mojo)
#   For a projection y = linear(x, W) (W [out,in]), the LoRA-adapted output is
#       y' = linear(x, W) + scale·((x @ Aᵀ) @ Bᵀ)
#   A=[rank,in], B=[out,rank], scale = alpha/rank. This MATCHES the inference merge
#   in inference-flame lora.rs (W' applied as base_out + scale·up(down(x))).
#
#   BACKWARD (given d_y' at the projection output, x the saved projection input):
#       d_dy = scale·d_y'                    [M,out]
#       d_B  = d_dyᵀ @ t   (t = x @ Aᵀ)      [out,rank]
#       d_t  = d_dy  @ B                      [M,rank]
#       d_A  = d_tᵀ  @ x                      [rank,in]
#       d_x  = d_t   @ A                      [M,in]   (LoRA branch's contribution
#                                                       to the projection INPUT grad)
#   The base path (frozen W) ALSO yields a d_x_base; the caller SUMS the LoRA d_x
#   into that. d_A/d_B go to the optimizer; the base W grad is discarded for LoRA.
#
# CARRIER NOTE: the SDXL ST works in device Tensors (TArc), unlike Ernie's host
#   List[Float32] block. The LoRA contribution helpers here take/return DEVICE
#   Tensors (M·* flattened internally) so they drop cleanly into the ST's
#   reshape/linear chain; internally they reuse train_step's host-list convention
#   (the parity-verified _lora_fwd/_lora_bwd math) via to_host/from_host, exactly
#   as Ernie does. This keeps the LoRA MATH byte-identical to the proven path.
#
# NO NEW ops/ PRIMITIVE: forward = two linear()s; backward = two linear_backward()s.
#
# Mojo 0.26.x: def not fn; comptime not alias; Tensor move-only -> TArc carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward

# REUSE the trainer's LoRA structs + the proven host-list LoRA math (the authority).
from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_fwd, _lora_bwd


comptime TArc = ArcPointer[Tensor]


# ── 10 canonical slots per BasicTransformerBlock (slot order is canonical) ──
comptime SDXL_SLOTS = 10
comptime SLOT_A1_Q = 0
comptime SLOT_A1_K = 1
comptime SLOT_A1_V = 2
comptime SLOT_A1_O = 3
comptime SLOT_A2_Q = 4
comptime SLOT_A2_K = 5
comptime SLOT_A2_V = 6
comptime SLOT_A2_O = 7
comptime SLOT_FF_PROJ = 8     # ff.net.0.proj  (GEGLU in-proj)  in=C  out=2*Cff
comptime SLOT_FF_OUT = 9      # ff.net.2       (FF out-proj)     in=Cff out=C


# ── host-list helpers (mirror ernie/lora_block.mojo) ──────────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


# Optionally-applied adapter forward on a DEVICE tensor base_y [M,out] with the
# DEVICE projection-input x [M,in]: if `lo` present, return base_y + LoRA contrib
# (device tensor); else return base_y unchanged (base-path no-regression). The
# LoRA contribution is computed through the parity-verified host-list _lora_fwd.
def sdxl_lora_apply(
    base_y: Tensor, x_in: Tensor, lo: Optional[LoraAdapter],
    M: Int, out_f: Int, ctx: DeviceContext,
) raises -> Tensor:
    if not lo:
        return base_y.clone(ctx)
    var x_h = x_in.to_host(ctx)
    var contrib = _lora_fwd(x_h, lo.value(), M, ctx)   # host [M,out]
    var base_h = base_y.to_host(ctx)                   # host [M,out]
    var summed = _add_lists(base_h, contrib)
    var sh = List[Int](); sh.append(M); sh.append(out_f)
    return Tensor.from_host(summed^, sh^, STDtype.F32, ctx)


# LoRA backward that ALSO returns the LoRA branch's contribution to d_x (as a host
# list, since the caller threads it into the base host/device d_x via _add_lists or
# a device add). d_a/d_b match train_step._lora_bwd exactly; d_x is the term that
# file drops (recomputed here from t @ A, identical to ernie_lora_bwd).
struct SdxlLoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]   # LoRA contribution to the projection INPUT grad [M,in]

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def sdxl_lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> SdxlLoraGrads:
    # d_a/d_b via the proven _lora_bwd; recompute d_x = d_t @ A (the dropped term).
    var lg = _lora_bwd(d_contrib_h, x_h, lo, M, ctx)   # LoraGrads(d_a [rank,in], d_b [out,rank])
    # t = x @ Aᵀ  (recompute; cheap)
    var nb_t = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        nb_t^, ctx,
    ).to_host(ctx)                                     # [M,rank]
    var d_dy = List[Float32]()
    for i in range(len(d_contrib_h)):
        d_dy.append(lo.scale * d_contrib_h[i])         # [M,out]
    # dy = t @ Bᵀ  -> d_t (the linear's d_x w.r.t. t)
    var lbB = linear_backward(
        Tensor.from_host(d_dy^, [M, lo.out_f], STDtype.F32, ctx),
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], STDtype.F32, ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.to_host(ctx)                     # [M,rank]
    # t = x @ Aᵀ  -> d_x_lo = (linear's d_x w.r.t. x)
    var lbA = linear_backward(
        Tensor.from_host(d_t^, [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_x_lo = lbA.d_x.to_host(ctx)                  # [M,in_f]
    return SdxlLoraGrads(lg.d_a.copy(), lg.d_b.copy(), d_x_lo^)


# Run base linear_backward d_x then add the LoRA branch's d_x (if present) on a
# DEVICE d_y / x_in. Returns the SUMMED d_x DEVICE tensor [M,in] and collects the
# LoRA d_a/d_b into the slot lists. The base d_w is computed by the caller's own
# linear_backward (the ST already does this) — here we only need the LoRA grads +
# the summed d_x. `d_x_base` is the base d_x the ST already produced [M,in].
def sdxl_proj_lora_into_dx(
    d_y: Tensor, x_in: Tensor, d_x_base: Tensor,
    lo: Optional[LoraAdapter], slot: Int,
    M: Int, in_f: Int,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> Tensor:
    if not lo:
        return d_x_base.clone(ctx)
    var d_y_h = d_y.to_host(ctx)
    var x_in_h = x_in.to_host(ctx)
    var lg = sdxl_lora_bwd(d_y_h, x_in_h, lo.value(), M, ctx)
    d_a_slots[slot] = lg.d_a.copy()
    d_b_slots[slot] = lg.d_b.copy()
    var base_dx = d_x_base.to_host(ctx)
    var summed = _add_lists(base_dx, lg.d_x)
    var sh = List[Int](); sh.append(M); sh.append(in_f)
    return Tensor.from_host(summed^, sh^, STDtype.F32, ctx)


# ── per-block LoRA carrier: the 10 optional adapters (slot order is canonical) ──
struct SdxlBlockLora(Copyable, Movable):
    var a1_q: Optional[LoraAdapter]
    var a1_k: Optional[LoraAdapter]
    var a1_v: Optional[LoraAdapter]
    var a1_o: Optional[LoraAdapter]
    var a2_q: Optional[LoraAdapter]
    var a2_k: Optional[LoraAdapter]
    var a2_v: Optional[LoraAdapter]
    var a2_o: Optional[LoraAdapter]
    var ff_proj: Optional[LoraAdapter]
    var ff_out: Optional[LoraAdapter]

    def __init__(
        out self,
        var a1_q: Optional[LoraAdapter], var a1_k: Optional[LoraAdapter],
        var a1_v: Optional[LoraAdapter], var a1_o: Optional[LoraAdapter],
        var a2_q: Optional[LoraAdapter], var a2_k: Optional[LoraAdapter],
        var a2_v: Optional[LoraAdapter], var a2_o: Optional[LoraAdapter],
        var ff_proj: Optional[LoraAdapter], var ff_out: Optional[LoraAdapter],
    ):
        self.a1_q = a1_q^; self.a1_k = a1_k^; self.a1_v = a1_v^; self.a1_o = a1_o^
        self.a2_q = a2_q^; self.a2_k = a2_k^; self.a2_v = a2_v^; self.a2_o = a2_o^
        self.ff_proj = ff_proj^; self.ff_out = ff_out^


# ── per-block LoRA grads (parallel to the 10 slots) ───────────────────────────
# d_a/d_b per present adapter; empty lists for absent slots. The block backward
# fills only the present slots (the others stay empty and AdamW skips them).
struct SdxlBlockLoraGrads(Copyable, Movable):
    var d_a: List[List[Float32]]   # SDXL_SLOTS entries (empty if slot absent)
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^
