# models/dit/hidream_o1_train_block.mojo — HiDream-O1 per-block LoRA training
# forward+backward (campaign P1, HIDREAM_O1_TRAINING_CAMPAIGN.md).
#
# The block = one Qwen3-VL decoder layer, op-for-op the inference `_layer`
# (hidream_o1.mojo:817-903): rms_norm -> q/k/v proj (+LoRA) -> per-head q/k
# rms_norm -> halfsplit mrope -> GQA repeat_kv -> prefix-causal masked SDPA ->
# o_proj (+LoRA) -> residual -> rms_norm -> SwiGLU MLP (+LoRA gate/up/down)
# -> residual.
#
# LoRA slots (DiffSynth lora_target_modules, campaign doc): q,k,v,o,gate,up,
# down — 7 per block. Adapter struct reused from zimage
# (ZImageLoraAdapterDevice — model-agnostic delta math y += scale*(x@Aᵀ)@Bᵀ).
#
# Backward primitives (ALL parity-proven, no new math here):
#   rms_norm_backward_dx, linear_backward_dx +
#   zimage_lora_bwd_device_resident_tensors, rope_backward(interleaved=False)
#   == the rope_halfsplit inverse, repeat_kv_backward (ops/gqa_backward),
#   sdpa_backward_masked (ops/attention_backward — torch-parity PASS
#   2026-06-11), swiglu_backward.
#
# GATE: tests/hidream_o1_block_parity.mojo — synthetic-but-real-shaped
# weights/inputs vs a torch oracle implementing the SAME decoder-layer math
# (sourced from DiffSynth hidream_o1_image_dit.py, NOT re-derived).
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.collections import Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.rope_struct_backward import rope_backward
from serenitymojo.ops.gqa_backward import repeat_kv_backward
from serenitymojo.ops.attention_backward import sdpa_backward_masked
from serenitymojo.ops.tensor_algebra import reshape, add
from serenitymojo.models.zimage.lora_block import (
    ZImageLoraAdapterDevice,
    zimage_lora_apply_device,
    zimage_lora_bwd_device_resident_tensors,
)
from serenitymojo.ops.attention import sdpa
from serenitymojo.models.dit.hidream_o1 import _repeat_kv

comptime TArc = ArcPointer[Tensor]


struct HiDreamO1BlockWeights(Copyable, Movable):
    """One decoder layer's frozen base weights (device TArcs; the trainer
    feeds resident or block-swapped tensors — same struct either way)."""

    var in_ln: TArc      # [D]
    var qw: TArc         # [H*Dh, D]
    var kw: TArc         # [HKV*Dh, D]
    var vw: TArc         # [HKV*Dh, D]
    var q_norm: TArc     # [Dh]
    var k_norm: TArc     # [Dh]
    var ow: TArc         # [D, H*Dh]
    var post_ln: TArc    # [D]
    var gw: TArc         # [F, D]
    var uw: TArc         # [F, D]
    var dw: TArc         # [D, F]

    def __init__(
        out self,
        var in_ln: TArc, var qw: TArc, var kw: TArc, var vw: TArc,
        var q_norm: TArc, var k_norm: TArc, var ow: TArc,
        var post_ln: TArc, var gw: TArc, var uw: TArc, var dw: TArc,
    ):
        self.in_ln = in_ln^
        self.qw = qw^
        self.kw = kw^
        self.vw = vw^
        self.q_norm = q_norm^
        self.k_norm = k_norm^
        self.ow = ow^
        self.post_ln = post_ln^
        self.gw = gw^
        self.uw = uw^
        self.dw = dw^


struct HiDreamO1BlockLora(Copyable, Movable):
    """7 LoRA slots (DiffSynth target list). None = slot untrained."""

    var q: Optional[ZImageLoraAdapterDevice]
    var k: Optional[ZImageLoraAdapterDevice]
    var v: Optional[ZImageLoraAdapterDevice]
    var o: Optional[ZImageLoraAdapterDevice]
    var gate: Optional[ZImageLoraAdapterDevice]
    var up: Optional[ZImageLoraAdapterDevice]
    var down: Optional[ZImageLoraAdapterDevice]

    def __init__(
        out self,
        var q: Optional[ZImageLoraAdapterDevice] = None,
        var k: Optional[ZImageLoraAdapterDevice] = None,
        var v: Optional[ZImageLoraAdapterDevice] = None,
        var o: Optional[ZImageLoraAdapterDevice] = None,
        var gate: Optional[ZImageLoraAdapterDevice] = None,
        var up: Optional[ZImageLoraAdapterDevice] = None,
        var down: Optional[ZImageLoraAdapterDevice] = None,
    ):
        self.q = q^
        self.k = k^
        self.v = v^
        self.o = o^
        self.gate = gate^
        self.up = up^
        self.down = down^


struct HiDreamO1BlockSaved(Copyable, Movable):
    """Saved tape for one block's backward (recompute-free P1; the stack
    trainer may switch to recompute-checkpoint later — Klein precedent)."""

    var hidden: TArc     # [1,S,D] block input
    var normed: TArc     # [1,S,D] rms_norm(hidden, in_ln)
    var q_pre: TArc      # [1,S,H,Dh]  post proj+LoRA, pre q_norm
    var k_pre: TArc      # [1,S,HKV,Dh]
    var v: TArc          # [1,S,HKV,Dh]
    var q_rms: TArc      # [1,S,H,Dh]
    var k_rms: TArc      # [1,S,HKV,Dh]
    var q_rope: TArc     # [1,S,H,Dh]
    var k_rope: TArc     # [1,S,HKV,Dh]
    var k_rep: TArc      # [1,S,H,Dh]
    var v_rep: TArc      # [1,S,H,Dh]
    var attn_flat: TArc  # [1,S,H*Dh]
    var hidden2: TArc    # [1,S,D] post-attn residual
    var normed2: TArc    # [1,S,D]
    var gate_pre: TArc   # [1,S,F]
    var up_pre: TArc     # [1,S,F]
    var act: TArc        # [1,S,F]

    def __init__(
        out self,
        var hidden: TArc, var normed: TArc,
        var q_pre: TArc, var k_pre: TArc, var v: TArc,
        var q_rms: TArc, var k_rms: TArc,
        var q_rope: TArc, var k_rope: TArc,
        var k_rep: TArc, var v_rep: TArc,
        var attn_flat: TArc, var hidden2: TArc, var normed2: TArc,
        var gate_pre: TArc, var up_pre: TArc, var act: TArc,
    ):
        self.hidden = hidden^
        self.normed = normed^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.k_rep = k_rep^
        self.v_rep = v_rep^
        self.attn_flat = attn_flat^
        self.hidden2 = hidden2^
        self.normed2 = normed2^
        self.gate_pre = gate_pre^
        self.up_pre = up_pre^
        self.act = act^


struct HiDreamO1BlockForward(Movable):
    var out: TArc        # [1,S,D]
    var saved: HiDreamO1BlockSaved

    def __init__(out self, var out: TArc, var saved: HiDreamO1BlockSaved):
        self.out = out^
        self.saved = saved^


def _maybe_lora_apply(
    var base: Tensor, x: Tensor, lo: Optional[ZImageLoraAdapterDevice],
    rows: Int, ctx: DeviceContext,
) raises -> Tensor:
    """y = base [+ scale*(x@Aᵀ)@Bᵀ] — the zimage in-place-free apply."""
    if lo:
        return zimage_lora_apply_device(base^, x, lo.value(), rows, ctx)
    return base^


def hidream_o1_block_lora_forward[
    S: Int, H: Int, HKV: Int, Dh: Int
](
    hidden_in: TArc,
    w: HiDreamO1BlockWeights,
    lora: HiDreamO1BlockLora,
    cos_q: Tensor, sin_q: Tensor,   # [S*H*half] half-width per-head tables
    cos_k: Tensor, sin_k: Tensor,   # [S*HKV*half]
    mask: Tensor,                    # [1,H,S,S] additive (block dtype)
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> HiDreamO1BlockForward:
    """Op-for-op the inference `_layer` with LoRA adds + saved tape."""
    comptime n_rep = H // HKV
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var normed_t = rms_norm(hidden_in[], w.in_ln[], eps, ctx)
    var normed = TArc(normed_t^)

    var q_flat = _maybe_lora_apply(
        linear(normed[], w.qw[], None, ctx), normed[], lora.q, S, ctx)
    var k_flat = _maybe_lora_apply(
        linear(normed[], w.kw[], None, ctx), normed[], lora.k, S, ctx)
    var v_flat = _maybe_lora_apply(
        linear(normed[], w.vw[], None, ctx), normed[], lora.v, S, ctx)

    var q4 = reshape(q_flat, [1, S, H, Dh], ctx)
    var k4 = reshape(k_flat, [1, S, HKV, Dh], ctx)
    var v4 = reshape(v_flat, [1, S, HKV, Dh], ctx)
    var q_pre = TArc(q4^)
    var k_pre = TArc(k4^)
    var v = TArc(v4^)

    var q_rms = TArc(rms_norm(q_pre[], w.q_norm[], eps, ctx))
    var k_rms = TArc(rms_norm(k_pre[], w.k_norm[], eps, ctx))

    var q_rope = TArc(rope_halfsplit(q_rms[], cos_q, sin_q, ctx))
    var k_rope = TArc(rope_halfsplit(k_rms[], cos_k, sin_k, ctx))

    var k_rep = TArc(_repeat_kv(k_rope[], S, HKV, n_rep, Dh, ctx))
    var v_rep = TArc(_repeat_kv(v[], S, HKV, n_rep, Dh, ctx))

    var attn = sdpa[1, S, H, Dh](q_rope[], k_rep[], v_rep[], mask, scale, ctx)
    var attn_flat_t = reshape(attn, [1, S, H * Dh], ctx)
    var attn_flat = TArc(attn_flat_t^)

    var attn_out = _maybe_lora_apply(
        linear(attn_flat[], w.ow[], None, ctx), attn_flat[], lora.o, S, ctx)
    var hidden2 = TArc(add(hidden_in[], attn_out, ctx))

    var normed2 = TArc(rms_norm(hidden2[], w.post_ln[], eps, ctx))
    var gate_pre = TArc(_maybe_lora_apply(
        linear(normed2[], w.gw[], None, ctx), normed2[], lora.gate, S, ctx))
    var up_pre = TArc(_maybe_lora_apply(
        linear(normed2[], w.uw[], None, ctx), normed2[], lora.up, S, ctx))
    var act = TArc(swiglu(gate_pre[], up_pre[], ctx))
    var mlp_out = _maybe_lora_apply(
        linear(act[], w.dw[], None, ctx), act[], lora.down, S, ctx)
    var out = TArc(add(hidden2[], mlp_out, ctx))

    var saved = HiDreamO1BlockSaved(
        hidden_in.copy(), normed^, q_pre^, k_pre^, v^, q_rms^, k_rms^,
        q_rope^, k_rope^, k_rep^, v_rep^, attn_flat^, hidden2^, normed2^,
        gate_pre^, up_pre^, act^,
    )
    return HiDreamO1BlockForward(out^, saved^)


struct HiDreamO1BlockGrads(Movable):
    """d_hidden + per-slot adapter grads (slot order q,k,v,o,gate,up,down;
    entries None for untrained slots — mirrors HiDreamO1BlockLora)."""

    var d_hidden: TArc
    var d_a: List[Optional[TArc]]
    var d_b: List[Optional[TArc]]

    def __init__(
        out self, var d_hidden: TArc,
        var d_a: List[Optional[TArc]], var d_b: List[Optional[TArc]],
    ):
        self.d_hidden = d_hidden^
        self.d_a = d_a^
        self.d_b = d_b^


struct _ProjLoraBwd(Movable):
    var d_x: Tensor
    var d_a: Optional[TArc]
    var d_b: Optional[TArc]

    def __init__(
        out self, var d_x: Tensor,
        var d_a: Optional[TArc], var d_b: Optional[TArc],
    ):
        self.d_x = d_x^
        self.d_a = d_a^
        self.d_b = d_b^


def _proj_lora_bwd(
    d_y: Tensor, x_in: Tensor, w: Tensor,
    lo: Optional[ZImageLoraAdapterDevice],
    M: Int, in_f: Int, out_f: Int,
    ctx: DeviceContext,
) raises -> _ProjLoraBwd:
    """d_x for y = x@Wᵀ [+ lora]; adapter grads when the slot is live.
    The ops_record.proj_lora_backward call sequence (same callees, same
    fold: base d_x + lora d_x)."""
    var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    if lo:
        var lg = zimage_lora_bwd_device_resident_tensors(d_y, x_in, lo.value(), M, ctx)
        var summed = add(base_dx^, lg.d_x[], ctx)
        return _ProjLoraBwd(
            summed^, Optional[TArc](lg.d_a.copy()), Optional[TArc](lg.d_b.copy())
        )
    return _ProjLoraBwd(base_dx^, None, None)


def hidream_o1_block_lora_backward[
    S: Int, H: Int, HKV: Int, Dh: Int
](
    d_out: Tensor,
    w: HiDreamO1BlockWeights,
    lora: HiDreamO1BlockLora,
    saved: HiDreamO1BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    mask_f32: Tensor,                # [H*S, S] F32 (sdpa_backward_masked contract)
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> HiDreamO1BlockGrads:
    """Reverse chain of hidream_o1_block_lora_forward. Every arm is an
    existing parity-proven backward; slot order q,k,v,o,gate,up,down."""
    comptime n_rep = H // HKV
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var d_a = List[Optional[TArc]]()
    var d_b = List[Optional[TArc]]()
    for _ in range(7):
        d_a.append(Optional[TArc](None))
        d_b.append(Optional[TArc](None))

    # out = hidden2 + mlp_out
    # ── MLP branch ────────────────────────────────────────────────────────────
    var lb_down = _proj_lora_bwd(d_out, saved.act[], w.dw[], lora.down, S, F, D, ctx)
    if lb_down.d_a:
        d_a[6] = lb_down.d_a.value().copy()
        d_b[6] = lb_down.d_b.value().copy()
    var sg = swiglu_backward(lb_down.d_x, saved.gate_pre[], saved.up_pre[], ctx)
    var lb_gate = _proj_lora_bwd(sg.d_gate, saved.normed2[], w.gw[], lora.gate, S, D, F, ctx)
    if lb_gate.d_a:
        d_a[4] = lb_gate.d_a.value().copy()
        d_b[4] = lb_gate.d_b.value().copy()
    var lb_up = _proj_lora_bwd(sg.d_up, saved.normed2[], w.uw[], lora.up, S, D, F, ctx)
    if lb_up.d_a:
        d_a[5] = lb_up.d_a.value().copy()
        d_b[5] = lb_up.d_b.value().copy()
    var d_normed2 = add(lb_gate.d_x, lb_up.d_x, ctx)
    var d_h2_norm = rms_norm_backward_dx(d_normed2, saved.hidden2[], w.post_ln[], eps, ctx)
    # residual: d_hidden2 = d_out + d(through mlp norm)
    var d_hidden2 = add(d_out, d_h2_norm, ctx)

    # hidden2 = hidden + attn_out
    # ── attention branch ─────────────────────────────────────────────────────
    var lb_o = _proj_lora_bwd(
        d_hidden2, saved.attn_flat[], w.ow[], lora.o, S, H * Dh, D, ctx)
    if lb_o.d_a:
        d_a[3] = lb_o.d_a.value().copy()
        d_b[3] = lb_o.d_b.value().copy()
    var d_attn4 = reshape(lb_o.d_x, [1, S, H, Dh], ctx)

    var sb = sdpa_backward_masked[1, S, H, Dh](
        saved.q_rope[], saved.k_rep[], saved.v_rep[], mask_f32, d_attn4, scale, ctx)

    var d_k_rope = repeat_kv_backward(sb.d_k, S, HKV, n_rep, Dh, ctx)
    var d_v4 = repeat_kv_backward(sb.d_v, S, HKV, n_rep, Dh, ctx)

    var d_q_rms = rope_backward(sb.d_q, cos_q, sin_q, False, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, False, ctx)

    var d_q_pre = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)

    var d_q_flat = reshape(d_q_pre, [1, S, H * Dh], ctx)
    var d_k_flat = reshape(d_k_pre, [1, S, HKV * Dh], ctx)
    var d_v_flat = reshape(d_v4, [1, S, HKV * Dh], ctx)

    var lb_q = _proj_lora_bwd(d_q_flat, saved.normed[], w.qw[], lora.q, S, D, H * Dh, ctx)
    if lb_q.d_a:
        d_a[0] = lb_q.d_a.value().copy()
        d_b[0] = lb_q.d_b.value().copy()
    var lb_k = _proj_lora_bwd(d_k_flat, saved.normed[], w.kw[], lora.k, S, D, HKV * Dh, ctx)
    if lb_k.d_a:
        d_a[1] = lb_k.d_a.value().copy()
        d_b[1] = lb_k.d_b.value().copy()
    var lb_v = _proj_lora_bwd(d_v_flat, saved.normed[], w.vw[], lora.v, S, D, HKV * Dh, ctx)
    if lb_v.d_a:
        d_a[2] = lb_v.d_a.value().copy()
        d_b[2] = lb_v.d_b.value().copy()

    var d_normed = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)
    var d_h_norm = rms_norm_backward_dx(d_normed, saved.hidden[], w.in_ln[], eps, ctx)
    # residual: d_hidden = d_hidden2 + d(through input norm)
    var d_hidden = add(d_hidden2, d_h_norm, ctx)

    return HiDreamO1BlockGrads(TArc(d_hidden^), d_a^, d_b^)
