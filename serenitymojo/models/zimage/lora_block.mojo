# serenitymojo/models/zimage/lora_block.mojo
#
# LoRA-ON-PROJECTION for the Z-Image (NextDiT) blocks. Mirrors the PROVEN Ernie
# LoRA template (models/ernie/lora_block.mojo), specialized to Z-Image's SEVEN
# un-fused target projections per block:
#   attention.{to_q, to_k, to_v, to_out.0}  and  feed_forward.{w1, w3, w2}
# (Z-Image has separate q/k/v — like Ernie — so 7 separate adapters.) The MLP is
# SwiGLU so the three MLP linears are w1 (gate), w3 (up), w2 (down), NOT
# gate_proj/up_proj/linear_fc2 — see the OT/diffusers key map in zimage_stack_lora.
#
# Z-Image has TWO block flavors, BOTH with these 7 Linear projections (so BOTH get
# LoRA per OneTrainer's default — LoRAModuleWrapper adapts EVERY nn.Linear child of
# the transformer; ZImageLoRASetup.py:57 passes no restrictive default filter):
#   * MODULATED block (noise refiners + main layers) — adaLN-tanh + SwiGLU.
#   * UNMODULATED block (context refiners) — plain sandwich-norm residual + SwiGLU.
# So this file provides a LoRA-aware forward/backward for EACH flavor, each reducing
# bit-for-bit to its base when adapters are absent.
#
# WHY THE TRAINER MATH IS THE AUTHORITY (identical to Ernie/Klein lora_block.mojo)
#   For a projection y = linear(x, W) (W [out,in]), the LoRA-adapted output is
#       y' = linear(x, W) + scale·((x @ Aᵀ) @ Bᵀ)
#   A=[rank,in], B=[out,rank], scale = alpha/rank. This MATCHES the inference merge
#   in inference-flame zimage_nextdit.rs `lora.apply(...)` (W' = W + scale·B@A) and
#   the OneTrainer forward (LoRAModule.py:328-329: orig_forward(x) + up(down(x)) *
#   alpha/rank).
#
#   BACKWARD (given d_y' at the projection output, x the saved projection input):
#       d_dy = scale·d_y'                    [M,out]
#       d_B  = d_dyᵀ @ t   (t = x @ Aᵀ)      [out,rank]
#       d_t  = d_dy  @ B                      [M,rank]
#       d_A  = d_tᵀ  @ x                      [rank,in]
#       d_x  = d_t   @ A                      [M,in]   (LoRA branch's contribution
#                                                       to the projection INPUT grad)
#   The base path (frozen W) ALSO yields d_x_base = d_y' @ W; the caller SUMS d_x
#   into that. d_A/d_B go to the optimizer; the base W grad is discarded for LoRA.
#
# Base weights are frozen during LoRA training, so the base projection backward
# computes d_x only. LoRA A/B grads still use full low-rank linear_backward.
#
# Mojo 1.0.0b1: def not fn; Tensor move-only crosses API as host List[Float32].

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx

# REUSE the trainer's LoRA structs (the target authority).
from serenitymojo.training.train_step import LoraAdapter, LoraGrads

# Forward + backward ops shared with the base block (Tenet 1: nothing new here).
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.unary import tanh_op
from serenitymojo.ops.tensor_algebra import reshape_owned, reshape_in_place, add
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import gate_residual_backward, rope_backward
from serenitymojo.ops.activation_backward import tanh_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import (
    ZImageModVecs,
    ZImageBlockSaved, ZImageBlockGrads,
    ZImageRefinerSaved, ZImageRefinerGrads,
)


comptime TArc = ArcPointer[Tensor]


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


# Adapter forward contribution on x [M,in] -> [M,out] (host list in/out).
# Byte-identical to train_step._lora_fwd / ernie_lora_fwd.
def zimage_lora_fwd(
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


# Optionally-applied adapter forward: if `lo` is present, return base_y + LoRA;
# else return base_y unchanged (base-path no-regression when an adapter is absent).
def zimage_lora_apply(
    base_y: List[Float32], x_h: List[Float32], lo: Optional[LoraAdapter],
    M: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    if not lo:
        return base_y.copy()
    var contrib = zimage_lora_fwd(x_h, lo.value(), M, ctx)
    var out = List[Float32]()
    for i in range(len(base_y)):
        out.append(base_y[i] + contrib[i])
    return out^


# LoRA backward that ALSO returns the LoRA branch's contribution to d_x.
struct ZImageLoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]   # LoRA contribution to the projection INPUT grad [M,in]

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def zimage_lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> ZImageLoraGrads:
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
    # dy = t @ Bᵀ  -> d_B (d_w) and d_t (d_x)
    var lbB = linear_backward(
        Tensor.from_host(d_dy^, [M, lo.out_f], STDtype.F32, ctx),
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], STDtype.F32, ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.to_host(ctx)                   # [M,rank]
    var d_b = lbB.d_w.to_host(ctx)                   # [out_f,rank]
    # t = x @ Aᵀ  -> d_A (d_w) and d_x_lo (d_x)
    var lbA = linear_backward(
        Tensor.from_host(d_t^, [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_x_lo = lbA.d_x.to_host(ctx)                # [M,in_f]
    var d_a = lbA.d_w.to_host(ctx)                   # [rank,in_f]
    return ZImageLoraGrads(d_a^, d_b^, d_x_lo^)


# ── per-block LoRA carrier: the 7 optional adapters (slot order is canonical) ──
# slot 0 to_q, 1 to_k, 2 to_v, 3 to_out.0, 4 feed_forward.w1, 5 .w3, 6 .w2.
comptime ZIMAGE_SLOTS = 7
comptime SLOT_Q = 0
comptime SLOT_K = 1
comptime SLOT_V = 2
comptime SLOT_O = 3
comptime SLOT_W1 = 4    # feed_forward.w1 (SwiGLU gate)
comptime SLOT_W3 = 5    # feed_forward.w3 (SwiGLU up)
comptime SLOT_W2 = 6    # feed_forward.w2 (SwiGLU down)


struct ZImageBlockLora(Copyable, Movable):
    var to_q: Optional[LoraAdapter]
    var to_k: Optional[LoraAdapter]
    var to_v: Optional[LoraAdapter]
    var to_out: Optional[LoraAdapter]
    var w1: Optional[LoraAdapter]
    var w3: Optional[LoraAdapter]
    var w2: Optional[LoraAdapter]

    def __init__(
        out self,
        var to_q: Optional[LoraAdapter], var to_k: Optional[LoraAdapter],
        var to_v: Optional[LoraAdapter], var to_out: Optional[LoraAdapter],
        var w1: Optional[LoraAdapter], var w3: Optional[LoraAdapter],
        var w2: Optional[LoraAdapter],
    ):
        self.to_q = to_q^
        self.to_k = to_k^
        self.to_v = to_v^
        self.to_out = to_out^
        self.w1 = w1^
        self.w3 = w3^
        self.w2 = w2^


# ── per-block LoRA grads (parallel to the 7 slots) ───────────────────────────
struct ZImageBlockLoraGrads(Copyable, Movable):
    var d_a: List[List[Float32]]   # ZIMAGE_SLOTS entries (empty if slot absent)
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


# proj-backward result: d_x [M,in] (base + LoRA summed). Base d_w is discarded
# for LoRA training and must not be materialized for Z-Image full depth.
struct _ProjGrads(Movable):
    var d_x: Tensor

    def __init__(out self, var d_x: Tensor):
        self.d_x = d_x^


# helper: run frozen-base d_x then add the LoRA branch's d_x (if present),
# collecting the LoRA d_a/d_b into the slot lists. Returns the SUMMED d_x [M,in].
def _proj_bwd_with_lora(
    d_y: Tensor, x_in: Tensor, w: Tensor, x_in_h: List[Float32],
    lo: Optional[LoraAdapter], slot: Int,
    M: Int, in_f: Int, out_f: Int,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> _ProjGrads:
    var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    if lo:
        var d_y_h = d_y.to_host(ctx)
        var lg = zimage_lora_bwd(d_y_h, x_in_h, lo.value(), M, ctx)
        d_a_slots[slot] = lg.d_a.copy()
        d_b_slots[slot] = lg.d_b.copy()
        var base_dx_h = base_dx.to_host(ctx)
        var summed = _add_lists(base_dx_h, lg.d_x)
        return _ProjGrads(_t(summed, [M, in_f], ctx))
    return _ProjGrads(base_dx^)


# ══════════════════════════════════════════════════════════════════════════════
# MODULATED block (noise refiners + main layers) — LoRA-aware fwd + bwd.
# Mirrors zimage_block_forward/_backward EXACTLY (models/zimage/block.mojo), adding
# the LoRA contribution to each of the 7 trained projection outputs BEFORE the
# downstream op consumes it. When all 7 adapters are absent this reduces bit-for-bit
# to the base forward. The `saved` activations are the LoRA-MODIFIED ones, so
# backward recompute regenerates them identically (same checkpoint contract).
# ══════════════════════════════════════════════════════════════════════════════
struct ZImageBlockForwardLora(Movable):
    var out: List[Float32]
    var saved: ZImageBlockSaved

    def __init__(out self, var out: List[Float32], var saved: ZImageBlockSaved):
        self.out = out^
        self.saved = saved^


def zimage_block_lora_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: ZImageBlockWeights, mv: ZImageModVecs, lora: ZImageBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockForwardLora:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var zeros = _zeros(D)
    var x_t = _t(x, [S, D], ctx)

    # --- attention sub-block (sandwich norm) ---
    var xn1 = rms_norm(x_t, w.n1[], eps, ctx)
    var xn1s = modulate(
        xn1, _t(mv.scale_msa.copy(), [D], ctx), _t(zeros.copy(), [D], ctx), ctx
    )
    var xn1s_h = xn1s.to_host(ctx)                              # [S,D] (LoRA input for q/k/v)

    var no_bias = Optional[Tensor](None)
    var q_base = linear(xn1s, w.wq[], no_bias^, ctx).to_host(ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear(xn1s, w.wk[], no_bias_k^, ctx).to_host(ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear(xn1s, w.wv[], no_bias_v^, ctx).to_host(ctx)

    var q_h = zimage_lora_apply(q_base, xn1s_h, lora.to_q, S, ctx)
    var k_h = zimage_lora_apply(k_base, xn1s_h, lora.to_k, S, ctx)
    var v_h = zimage_lora_apply(v_base, xn1s_h, lora.to_v, S, ctx)

    var q_pre = reshape_owned(_t(q_h, [S, D], ctx)^, [1, S, H, Dh])
    var k_pre = reshape_owned(_t(k_h, [S, D], ctx)^, [1, S, H, Dh])
    var v = reshape_owned(_t(v_h, [S, D], ctx)^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])
    var att_flat_h = att_flat.to_host(ctx)                      # [S,D] (LoRA input for wo)

    var no_bias_o = Optional[Tensor](None)
    var att_o_base = linear(att_flat, w.wo[], no_bias_o^, ctx).to_host(ctx)
    var att_o_h = zimage_lora_apply(att_o_base, att_flat_h, lora.to_out, S, ctx)
    var att_o = _t(att_o_h, [S, D], ctx)

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var gate_msa_raw = _t(mv.gate_msa.copy(), [D], ctx)
    var gate_msa_t = tanh_op(gate_msa_raw, ctx)
    var h = residual_gate(x_t, gate_msa_t, attn_n2, ctx)

    # --- MLP sub-block (SwiGLU, sandwich norm) ---
    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)
    var xfn1s = modulate(
        xfn1, _t(mv.scale_mlp.copy(), [D], ctx), _t(zeros.copy(), [D], ctx), ctx
    )
    var xfn1s_h = xfn1s.to_host(ctx)                            # [S,D] (LoRA input for w1/w3)

    var no_bias_g = Optional[Tensor](None)
    var g_base = linear(xfn1s, w.w1[], no_bias_g^, ctx).to_host(ctx)
    var no_bias_u = Optional[Tensor](None)
    var u_base = linear(xfn1s, w.w3[], no_bias_u^, ctx).to_host(ctx)
    var g_pre_h = zimage_lora_apply(g_base, xfn1s_h, lora.w1, S, ctx)
    var u_h = zimage_lora_apply(u_base, xfn1s_h, lora.w3, S, ctx)
    var g_pre = _t(g_pre_h, [S, F], ctx)
    var u = _t(u_h, [S, F], ctx)

    var act = swiglu(g_pre, u, ctx)
    var act_h = act.to_host(ctx)                                # [S,F] (LoRA input for w2)

    var no_bias_d = Optional[Tensor](None)
    var ff_base = linear(act, w.w2[], no_bias_d^, ctx).to_host(ctx)
    var ff_h = zimage_lora_apply(ff_base, act_h, lora.w2, S, ctx)
    var ff = _t(ff_h, [S, D], ctx)

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var gate_mlp_raw = _t(mv.gate_mlp.copy(), [D], ctx)
    var gate_mlp_t = tanh_op(gate_mlp_raw, ctx)
    var result = residual_gate(h, gate_mlp_t, ff_n2, ctx).to_host(ctx)

    var saved = ZImageBlockSaved(
        TArc(x_t^), TArc(xn1^), TArc(xn1s^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(att_o^),
        TArc(gate_msa_t^), TArc(gate_msa_raw^), TArc(h^),
        TArc(xfn1^), TArc(xfn1s^),
        TArc(g_pre^), TArc(u^), TArc(act^), TArc(ff^),
        TArc(gate_mlp_t^), TArc(gate_mlp_raw^),
    )
    return ZImageBlockForwardLora(result^, saved^)


struct ZImageBlockLoraBackward(Movable):
    var base: ZImageBlockGrads
    var lora: ZImageBlockLoraGrads

    def __init__(out self, var base: ZImageBlockGrads, var lora: ZImageBlockLoraGrads):
        self.base = base^
        self.lora = lora^


def zimage_block_lora_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: ZImageBlockWeights, mv: ZImageModVecs, lora: ZImageBlockLora,
    saved: ZImageBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockLoraBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var d_out_t = _t(d_out, [S, D], ctx)

    var d_a_slots = List[List[Float32]]()
    var d_b_slots = List[List[Float32]]()
    for _ in range(ZIMAGE_SLOTS):
        d_a_slots.append(List[Float32]())
        d_b_slots.append(List[Float32]())

    # out = residual_gate(h, gate_mlp_t, ff_n2); recompute ff_n2 = rms_norm(ff, fn2)
    var ff_n2_y = rms_norm(saved.ff[], w.fn2[], eps, ctx)
    var grg2 = gate_residual_backward(
        d_out_t, saved.h[], saved.gate_mlp_t[], ff_n2_y, ctx
    )
    var d_gate_mlp = tanh_backward(grg2.d_g, saved.gate_mlp_raw[], ctx).to_host(ctx)

    # ff_n2 = rms_norm(ff, fn2)
    var rb_fn2 = rms_norm_backward(grg2.d_y, saved.ff[], w.fn2[], eps, ctx)
    var d_fn2 = rb_fn2.d_g.to_host(ctx)

    # ff = linear(act, w2)[+LoRA(w2)]  W [D, F]
    var act_h = saved.act[].to_host(ctx)
    var lb_w2 = _proj_bwd_with_lora(
        rb_fn2.d_x, saved.act[], w.w2[], act_h,
        lora.w2, SLOT_W2, S, F, D, d_a_slots, d_b_slots, ctx,
    )
    var d_w2 = List[Float32]()

    # act = swiglu(g_pre, u) -> d_g_pre, d_u
    var sg = swiglu_backward(lb_w2.d_x, saved.g_pre[], saved.u[], ctx)

    # g_pre = linear(xfn1s, w1)[+LoRA] ; u = linear(xfn1s, w3)[+LoRA]  W [F, D]
    var xfn1s_h = saved.xfn1s[].to_host(ctx)
    var lb_w1 = _proj_bwd_with_lora(
        sg.d_gate, saved.xfn1s[], w.w1[], xfn1s_h,
        lora.w1, SLOT_W1, S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_w1 = List[Float32]()
    var lb_w3 = _proj_bwd_with_lora(
        sg.d_up, saved.xfn1s[], w.w3[], xfn1s_h,
        lora.w3, SLOT_W3, S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_w3 = List[Float32]()
    var d_xfn1s = add(lb_w1.d_x, lb_w3.d_x, ctx)

    # xfn1s = modulate(xfn1, scale_mlp, 0)
    var mb_mlp = modulate_backward(d_xfn1s, saved.xfn1[], _t(mv.scale_mlp.copy(), [D], ctx), ctx)
    var d_scale_mlp = mb_mlp.d_scale.to_host(ctx)
    var rb_fn1 = rms_norm_backward(mb_mlp.d_x, saved.h[], w.fn1[], eps, ctx)
    var d_fn1 = rb_fn1.d_g.to_host(ctx)
    var d_h = add(grg2.d_x, rb_fn1.d_x, ctx)

    # --- attention sub-block backward ---
    # h = residual_gate(x, gate_msa_t, attn_n2); recompute attn_n2 = rms_norm(att_o, n2)
    var attn_n2_y = rms_norm(saved.att_o[], w.n2[], eps, ctx)
    var grg1 = gate_residual_backward(
        d_h, saved.x[], saved.gate_msa_t[], attn_n2_y, ctx
    )
    var d_gate_msa = tanh_backward(grg1.d_g, saved.gate_msa_raw[], ctx).to_host(ctx)

    var rb_n2 = rms_norm_backward(grg1.d_y, saved.att_o[], w.n2[], eps, ctx)
    var d_n2 = rb_n2.d_g.to_host(ctx)

    # att_o = linear(att_flat, wo)[+LoRA(to_out)]  W [D, D]
    var att_flat_h = saved.att_flat[].to_host(ctx)
    var lb_o = _proj_bwd_with_lora(
        rb_n2.d_x, saved.att_flat[], w.wo[], att_flat_h,
        lora.to_out, SLOT_O, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wo = List[Float32]()

    reshape_in_place(lb_o.d_x, [1, S, H, Dh])
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )
    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # q/k/v = linear(xn1s, w{q,k,v})[+LoRA]  W [D, D]; xn1s feeds all three.
    var xn1s_h = saved.xn1s[].to_host(ctx)
    var lb_q = _proj_bwd_with_lora(
        rb_q.d_x, saved.xn1s[], w.wq[], xn1s_h,
        lora.to_q, SLOT_Q, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wq = List[Float32]()
    var lb_k = _proj_bwd_with_lora(
        rb_k.d_x, saved.xn1s[], w.wk[], xn1s_h,
        lora.to_k, SLOT_K, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wk = List[Float32]()
    var lb_v = _proj_bwd_with_lora(
        sb.d_v, saved.xn1s[], w.wv[], xn1s_h,
        lora.to_v, SLOT_V, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wv = List[Float32]()
    var d_xn1s = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)

    # xn1s = modulate(xn1, scale_msa, 0)
    var mb_sa = modulate_backward(d_xn1s, saved.xn1[], _t(mv.scale_msa.copy(), [D], ctx), ctx)
    var d_scale_msa = mb_sa.d_scale.to_host(ctx)
    var rb_n1 = rms_norm_backward(mb_sa.d_x, saved.x[], w.n1[], eps, ctx)
    var d_n1 = rb_n1.d_g.to_host(ctx)

    var d_x_res = grg1.d_x.to_host(ctx)
    var d_x_norm = rb_n1.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    var base = ZImageBlockGrads(
        d_x^, d_n1^,
        d_wq^, d_wk^, d_wv^, d_wo^,
        d_q_norm^, d_k_norm^,
        d_n2^, d_fn1^,
        d_w1^, d_w3^, d_w2^, d_fn2^,
        d_scale_msa^, d_gate_msa^,
        d_scale_mlp^, d_gate_mlp^,
    )
    return ZImageBlockLoraBackward(base^, ZImageBlockLoraGrads(d_a_slots^, d_b_slots^))


# ══════════════════════════════════════════════════════════════════════════════
# UNMODULATED block (context refiners) — LoRA-aware fwd + bwd.
# Mirrors zimage_refiner_forward/_backward EXACTLY (models/zimage/block.mojo). Same
# 7 LoRA target projections (attention + feed_forward Linears), no modulation/gate.
# ══════════════════════════════════════════════════════════════════════════════
struct ZImageRefinerForwardLora(Movable):
    var out: List[Float32]
    var saved: ZImageRefinerSaved

    def __init__(out self, var out: List[Float32], var saved: ZImageRefinerSaved):
        self.out = out^
        self.saved = saved^


def zimage_refiner_lora_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: ZImageBlockWeights, lora: ZImageBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageRefinerForwardLora:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var x_t = _t(x, [S, D], ctx)

    # --- attention sub-block (sandwich norm, NO modulation) ---
    var xn1 = rms_norm(x_t, w.n1[], eps, ctx)
    var xn1_h = xn1.to_host(ctx)                                # [S,D] (LoRA input for q/k/v)

    var no_bias = Optional[Tensor](None)
    var q_base = linear(xn1, w.wq[], no_bias^, ctx).to_host(ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear(xn1, w.wk[], no_bias_k^, ctx).to_host(ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear(xn1, w.wv[], no_bias_v^, ctx).to_host(ctx)

    var q_h = zimage_lora_apply(q_base, xn1_h, lora.to_q, S, ctx)
    var k_h = zimage_lora_apply(k_base, xn1_h, lora.to_k, S, ctx)
    var v_h = zimage_lora_apply(v_base, xn1_h, lora.to_v, S, ctx)

    var q_pre = reshape_owned(_t(q_h, [S, D], ctx)^, [1, S, H, Dh])
    var k_pre = reshape_owned(_t(k_h, [S, D], ctx)^, [1, S, H, Dh])
    var v = reshape_owned(_t(v_h, [S, D], ctx)^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])
    var att_flat_h = att_flat.to_host(ctx)                      # [S,D] (LoRA input for wo)

    var no_bias_o = Optional[Tensor](None)
    var att_o_base = linear(att_flat, w.wo[], no_bias_o^, ctx).to_host(ctx)
    var att_o_h = zimage_lora_apply(att_o_base, att_flat_h, lora.to_out, S, ctx)
    var att_o = _t(att_o_h, [S, D], ctx)

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var h = add(x_t, attn_n2, ctx)                  # PLAIN residual (no gate)

    # --- MLP sub-block (SwiGLU, sandwich norm, NO modulation) ---
    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)
    var xfn1_h = xfn1.to_host(ctx)                              # [S,D] (LoRA input for w1/w3)

    var no_bias_g = Optional[Tensor](None)
    var g_base = linear(xfn1, w.w1[], no_bias_g^, ctx).to_host(ctx)
    var no_bias_u = Optional[Tensor](None)
    var u_base = linear(xfn1, w.w3[], no_bias_u^, ctx).to_host(ctx)
    var g_pre_h = zimage_lora_apply(g_base, xfn1_h, lora.w1, S, ctx)
    var u_h = zimage_lora_apply(u_base, xfn1_h, lora.w3, S, ctx)
    var g_pre = _t(g_pre_h, [S, F], ctx)
    var u = _t(u_h, [S, F], ctx)

    var act = swiglu(g_pre, u, ctx)
    var act_h = act.to_host(ctx)                                # [S,F] (LoRA input for w2)

    var no_bias_d = Optional[Tensor](None)
    var ff_base = linear(act, w.w2[], no_bias_d^, ctx).to_host(ctx)
    var ff_h = zimage_lora_apply(ff_base, act_h, lora.w2, S, ctx)
    var ff = _t(ff_h, [S, D], ctx)

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var result = add(h, ff_n2, ctx).to_host(ctx)    # PLAIN residual (no gate)

    var saved = ZImageRefinerSaved(
        TArc(x_t^), TArc(xn1^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(att_o^), TArc(h^),
        TArc(xfn1^), TArc(g_pre^), TArc(u^), TArc(act^), TArc(ff^),
    )
    return ZImageRefinerForwardLora(result^, saved^)


struct ZImageRefinerLoraBackward(Movable):
    var base: ZImageRefinerGrads
    var lora: ZImageBlockLoraGrads

    def __init__(out self, var base: ZImageRefinerGrads, var lora: ZImageBlockLoraGrads):
        self.base = base^
        self.lora = lora^


def zimage_refiner_lora_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: ZImageBlockWeights, lora: ZImageBlockLora, saved: ZImageRefinerSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageRefinerLoraBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var d_out_t = _t(d_out, [S, D], ctx)

    var d_a_slots = List[List[Float32]]()
    var d_b_slots = List[List[Float32]]()
    for _ in range(ZIMAGE_SLOTS):
        d_a_slots.append(List[Float32]())
        d_b_slots.append(List[Float32]())

    # out = h + ff_n2 (plain residual); ff_n2 = rms_norm(ff, fn2)
    var rb_fn2 = rms_norm_backward(d_out_t, saved.ff[], w.fn2[], eps, ctx)
    var d_fn2 = rb_fn2.d_g.to_host(ctx)

    # ff = linear(act, w2)[+LoRA(w2)]  W [D, F]
    var act_h = saved.act[].to_host(ctx)
    var lb_w2 = _proj_bwd_with_lora(
        rb_fn2.d_x, saved.act[], w.w2[], act_h,
        lora.w2, SLOT_W2, S, F, D, d_a_slots, d_b_slots, ctx,
    )
    var d_w2 = List[Float32]()

    var sg = swiglu_backward(lb_w2.d_x, saved.g_pre[], saved.u[], ctx)

    # g_pre = linear(xfn1, w1)[+LoRA] ; u = linear(xfn1, w3)[+LoRA]  W [F, D]
    var xfn1_h = saved.xfn1[].to_host(ctx)
    var lb_w1 = _proj_bwd_with_lora(
        sg.d_gate, saved.xfn1[], w.w1[], xfn1_h,
        lora.w1, SLOT_W1, S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_w1 = List[Float32]()
    var lb_w3 = _proj_bwd_with_lora(
        sg.d_up, saved.xfn1[], w.w3[], xfn1_h,
        lora.w3, SLOT_W3, S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_w3 = List[Float32]()
    var d_xfn1 = add(lb_w1.d_x, lb_w3.d_x, ctx)

    # xfn1 = rms_norm(h, fn1)  (no modulation between)
    var rb_fn1 = rms_norm_backward(d_xfn1, saved.h[], w.fn1[], eps, ctx)
    var d_fn1 = rb_fn1.d_g.to_host(ctx)
    var d_h = add(d_out_t, rb_fn1.d_x, ctx)

    # --- attention sub-block backward ---
    # h = x + attn_n2 (plain residual); attn_n2 = rms_norm(att_o, n2)
    var rb_n2 = rms_norm_backward(d_h, saved.att_o[], w.n2[], eps, ctx)
    var d_n2 = rb_n2.d_g.to_host(ctx)

    # att_o = linear(att_flat, wo)[+LoRA(to_out)]  W [D, D]
    var att_flat_h = saved.att_flat[].to_host(ctx)
    var lb_o = _proj_bwd_with_lora(
        rb_n2.d_x, saved.att_flat[], w.wo[], att_flat_h,
        lora.to_out, SLOT_O, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wo = List[Float32]()

    reshape_in_place(lb_o.d_x, [1, S, H, Dh])
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )
    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # q/k/v = linear(xn1, w{q,k,v})[+LoRA]  W [D, D]; xn1 feeds all three.
    var xn1_h = saved.xn1[].to_host(ctx)
    var lb_q = _proj_bwd_with_lora(
        rb_q.d_x, saved.xn1[], w.wq[], xn1_h,
        lora.to_q, SLOT_Q, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wq = List[Float32]()
    var lb_k = _proj_bwd_with_lora(
        rb_k.d_x, saved.xn1[], w.wk[], xn1_h,
        lora.to_k, SLOT_K, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wk = List[Float32]()
    var lb_v = _proj_bwd_with_lora(
        sb.d_v, saved.xn1[], w.wv[], xn1_h,
        lora.to_v, SLOT_V, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wv = List[Float32]()
    var d_xn1 = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)

    # xn1 = rms_norm(x, n1)  (no modulation between)
    var rb_n1 = rms_norm_backward(d_xn1, saved.x[], w.n1[], eps, ctx)
    var d_n1 = rb_n1.d_g.to_host(ctx)

    var d_x_res = d_h.to_host(ctx)
    var d_x_norm = rb_n1.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    var base = ZImageRefinerGrads(
        d_x^, d_n1^,
        d_wq^, d_wk^, d_wv^, d_wo^,
        d_q_norm^, d_k_norm^,
        d_n2^, d_fn1^,
        d_w1^, d_w3^, d_w2^, d_fn2^,
    )
    return ZImageRefinerLoraBackward(base^, ZImageBlockLoraGrads(d_a_slots^, d_b_slots^))
