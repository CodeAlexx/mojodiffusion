# models/dit/hidream_o1_block_ft.mojo
#
# FULL-FINETUNE device block backward for the HiDream-O1 decoder layer —
# full-FT rollout P1 for hidream-o1 (docs/FULL_FINETUNE_ROLLOUT_PLAN_2026-07-07.md,
# blueprint: .claude/skills/reference-mojo-full-finetune — krea2 worked example;
# freshest precedents models/ideogram4/ideogram4_block_ft.mojo and
# models/flux/flux_block_ft.mojo).
#
# Same hand-chain as hidream_o1_block_lora_backward
# (models/dit/hidream_o1_train_block.mojo:397) but the base matmul WEIGHTS are
# TRAINABLE: each projection site emits d_W = d_yᵀ @ x_saved via
# ops/linalg_backward.linear_backward_dw with EXPLICIT STDtype.F32 output (the
# BOOL default sentinel means "match input dtype" -> bf16 grads the F32
# optimizer rejects — krea2 trap), alongside the plain d_x carry
# (linear_backward_dx). NO LoRA anywhere, NO quantized linears (reference trainer full-FT
# forbids them — quantized forwards detach the weight).
#
# v2 trainable surface (FULL_SURFACE_PLAN_2026-07-08 Phase B row 2) = ALL 11
# per-layer params, in HD_FT_SLOT_* order (0-6 the v1 matmuls — the LoRA slot
# order q,k,v,o,gate,up,down):
#   [0] qw [H*Dh, D]     (self_attn.q_proj.weight)
#   [1] kw [HKV*Dh, D]   (self_attn.k_proj.weight)
#   [2] vw [HKV*Dh, D]   (self_attn.v_proj.weight)
#   [3] ow [D, H*Dh]     (self_attn.o_proj.weight)
#   [4] gw [F, D]        (mlp.gate_proj.weight)
#   [5] uw [F, D]        (mlp.up_proj.weight)
#   [6] dw [D, F]        (mlp.down_proj.weight)
#   [7] in_ln [D]        (input_layernorm.weight — rms scale, full
#   [8] q_norm [Dh]       rms_norm_backward d_g; q_norm/k_norm are per-head
#   [9] k_norm [Dh]       over [1,S,H(kv),Dh] — d_g [Dh], rows = S*H(kv))
#   [10] post_ln [D]     (post_attention_layernorm.weight)
# The 1D d_g grads arrive in the block's storage dtype from rms_norm_backward
# (F32-accumulated d_g kernel) — CAST to F32 explicitly for the optimizer (the
# krea2 dW-dtype trap, same reason as the STDtype.F32 on the matmul dW).
# v2 REPLACES the v1 (matmul-only) surface on the FT arm — reference trainer/DiffSynth full-FT
# trains all layer params, there is no matmul-only reference trainer mode to keep.
#
# dX fold orders (C15) match hidream_o1_block_lora_backward EXACTLY (minus the
# adapter add, which is the only dX term LoRA contributes):
#   d_normed2 = add(gate_dx, up_dx);      d_hidden2 = add(d_out, d_h2_norm);
#   d_normed  = add(add(q_dx, k_dx), v_dx); d_hidden = add(d_hidden2, d_h_norm).
#
# dW = d_yᵀ·x is INVARIANT to LoRA presence, but the dX CHAIN is not (LoRA adds
# an adapter path) — so the gate uses the BASE-ONLY oracle mode
# (HIDREAM_FT_ORACLE=1, models/dit/parity/hidream_o1_block_oracle.py), never
# the LoRA-mode refs. Gate: models/dit/tests/hidream_o1_block_ft_parity.mojo,
# cos >= 0.999 at the REAL head config (H=32, HKV=8, Dh=128, D=4096, F=12288).
#
# Mojo 1.0.0b1, NVIDIA.

from std.math import sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linalg_backward import linear_backward_dx, linear_backward_dw
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.rope_struct_backward import rope_backward
from serenitymojo.ops.gqa_backward import repeat_kv_backward
from serenitymojo.ops.attention_backward import sdpa_backward_masked
from serenitymojo.ops.tensor_algebra import add, reshape

from serenitymojo.models.dit.hidream_o1_train_block import (
    HiDreamO1BlockWeights,
    HiDreamO1BlockSaved,
)

comptime TArc = ArcPointer[Tensor]

# Trainable-matmul slot order (== the LoRA slot order q,k,v,o,gate,up,down).
comptime HD_FT_SLOTS_PER_BLOCK = 7
comptime HD_FT_SLOT_Q = 0
comptime HD_FT_SLOT_K = 1
comptime HD_FT_SLOT_V = 2
comptime HD_FT_SLOT_O = 3
comptime HD_FT_SLOT_GATE = 4
comptime HD_FT_SLOT_UP = 5
comptime HD_FT_SLOT_DOWN = 6
# v2 full-surface slot indices 7-10 (the 4 per-layer 1D rms-norm scales).
# FIXED order — the host store tails, af_states flat index, sidecar and the
# parity gate all key off it.
comptime HD_FT_SLOT_IN_LN = 7
comptime HD_FT_SLOT_Q_NORM = 8
comptime HD_FT_SLOT_K_NORM = 9
comptime HD_FT_SLOT_POST_LN = 10
comptime HD_FT_SLOTS_V2 = 11


# FT grad carrier: the plain d_hidden carry + the 11 device-F32 param grads.
struct HiDreamO1BlockFTGrads(Movable):
    var d_hidden: TArc           # [1,S,D] input grad (inter-block carry)
    # len 11, F32 device, FIXED HD_FT_SLOT_* order:
    #   [0] d_qw [H*Dh,D] [1] d_kw [HKV*Dh,D] [2] d_vw [HKV*Dh,D]
    #   [3] d_ow [D,H*Dh] [4] d_gw [F,D] [5] d_uw [F,D] [6] d_dw [D,F]
    #   [7] d_in_ln [D] [8] d_q_norm [Dh] [9] d_k_norm [Dh] [10] d_post_ln [D]
    var dw: List[TArc]

    def __init__(out self, var d_hidden: TArc, var dw: List[TArc]):
        self.d_hidden = d_hidden^
        self.dw = dw^


def hidream_o1_block_ft_backward_dev[
    S: Int, H: Int, HKV: Int, Dh: Int
](
    d_out: Tensor,
    w: HiDreamO1BlockWeights,
    saved: HiDreamO1BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    mask_f32: Tensor,                # [H*S, S] F32 (sdpa_backward_masked contract)
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> HiDreamO1BlockFTGrads:
    """Reverse chain of the BASE block forward (hidream_o1_block_lora_forward
    with all-None adapters IS the base forward — same tape). Every arm is an
    existing parity-proven backward; the 7 matmul sites add a device-F32 dW
    (linear_backward_dw, EXPLICIT STDtype.F32); the 4 rms-norm sites add a
    device-F32 d_g (full rms_norm_backward, d_g cast F32 — v2 full surface)."""
    comptime n_rep = H // HKV
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var dw = List[TArc]()

    # out = hidden2 + mlp_out
    # ── MLP branch ────────────────────────────────────────────────────────────
    # mlp_out = linear(act, dw): TRAINABLE weight -> d_W (F32) + d_x carry.
    var d_dw = TArc(
        linear_backward_dw(d_out, saved.act[], S, F, D, ctx, STDtype.F32)
    )
    var down_dx = linear_backward_dx(d_out, w.dw[], S, F, D, ctx)

    var sg = swiglu_backward(down_dx, saved.gate_pre[], saved.up_pre[], ctx)

    # gate_pre = linear(normed2, gw): TRAINABLE weight.
    var d_gw = TArc(
        linear_backward_dw(sg.d_gate, saved.normed2[], S, D, F, ctx, STDtype.F32)
    )
    var gate_dx = linear_backward_dx(sg.d_gate, w.gw[], S, D, F, ctx)

    # up_pre = linear(normed2, uw): TRAINABLE weight.
    var d_uw = TArc(
        linear_backward_dw(sg.d_up, saved.normed2[], S, D, F, ctx, STDtype.F32)
    )
    var up_dx = linear_backward_dx(sg.d_up, w.uw[], S, D, F, ctx)

    # C15: gate branch first, up branch second (the LoRA fold order).
    var d_normed2 = add(gate_dx, up_dx, ctx)
    # post_ln rms scale TRAINABLE (v2) -> full rms backward (d_x + d_g);
    # d_g arrives in the block dtype (F32-accumulated) -> cast F32 for the opt.
    var rb_h2 = rms_norm_backward(d_normed2, saved.hidden2[], w.post_ln[], eps, ctx)
    var d_post_ln = TArc(cast_tensor(rb_h2.d_g, STDtype.F32, ctx))
    # residual: d_hidden2 = d_out + d(through mlp norm).
    var d_hidden2 = add(d_out, rb_h2.d_x, ctx)

    # hidden2 = hidden + attn_out
    # ── attention branch ──────────────────────────────────────────────────────
    # attn_out = linear(attn_flat, ow): TRAINABLE weight.
    var d_ow = TArc(
        linear_backward_dw(d_hidden2, saved.attn_flat[], S, H * Dh, D, ctx, STDtype.F32)
    )
    var o_dx = linear_backward_dx(d_hidden2, w.ow[], S, H * Dh, D, ctx)
    var d_attn4 = reshape(o_dx, [1, S, H, Dh], ctx)

    var sb = sdpa_backward_masked[1, S, H, Dh](
        saved.q_rope[], saved.k_rep[], saved.v_rep[], mask_f32, d_attn4, scale, ctx
    )

    var d_k_rope = repeat_kv_backward(sb.d_k, S, HKV, n_rep, Dh, ctx)
    var d_v4 = repeat_kv_backward(sb.d_v, S, HKV, n_rep, Dh, ctx)

    var d_q_rms = rope_backward(sb.d_q, cos_q, sin_q, False, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, False, ctx)

    # q_norm/k_norm rms scales TRAINABLE (v2) — q_pre [1,S,H,Dh] /
    # k_pre [1,S,HKV,Dh], d_g [Dh] (rows = S*H / S*HKV).
    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm_g = TArc(cast_tensor(rb_q.d_g, STDtype.F32, ctx))
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm_g = TArc(cast_tensor(rb_k.d_g, STDtype.F32, ctx))

    var d_q_flat = reshape(rb_q.d_x, [1, S, H * Dh], ctx)
    var d_k_flat = reshape(rb_k.d_x, [1, S, HKV * Dh], ctx)
    var d_v_flat = reshape(d_v4, [1, S, HKV * Dh], ctx)

    # q/k/v = linear(normed, qw/kw/vw): TRAINABLE weights.
    var d_qw = TArc(
        linear_backward_dw(d_q_flat, saved.normed[], S, D, H * Dh, ctx, STDtype.F32)
    )
    var q_dx = linear_backward_dx(d_q_flat, w.qw[], S, D, H * Dh, ctx)
    var d_kw = TArc(
        linear_backward_dw(d_k_flat, saved.normed[], S, D, HKV * Dh, ctx, STDtype.F32)
    )
    var k_dx = linear_backward_dx(d_k_flat, w.kw[], S, D, HKV * Dh, ctx)
    var d_vw = TArc(
        linear_backward_dw(d_v_flat, saved.normed[], S, D, HKV * Dh, ctx, STDtype.F32)
    )
    var v_dx = linear_backward_dx(d_v_flat, w.vw[], S, D, HKV * Dh, ctx)

    # C15: add(add(q, k), v) — the LoRA fold tree.
    var d_normed = add(add(q_dx, k_dx, ctx), v_dx, ctx)
    # in_ln rms scale TRAINABLE (v2).
    var rb_h = rms_norm_backward(d_normed, saved.hidden[], w.in_ln[], eps, ctx)
    var d_in_ln = TArc(cast_tensor(rb_h.d_g, STDtype.F32, ctx))
    # residual: d_hidden = d_hidden2 + d(through input norm).
    var d_hidden = add(d_hidden2, rb_h.d_x, ctx)

    dw.append(d_qw^)         # HD_FT_SLOT_Q
    dw.append(d_kw^)         # HD_FT_SLOT_K
    dw.append(d_vw^)         # HD_FT_SLOT_V
    dw.append(d_ow^)         # HD_FT_SLOT_O
    dw.append(d_gw^)         # HD_FT_SLOT_GATE
    dw.append(d_uw^)         # HD_FT_SLOT_UP
    dw.append(d_dw^)         # HD_FT_SLOT_DOWN
    dw.append(d_in_ln^)      # HD_FT_SLOT_IN_LN
    dw.append(d_q_norm_g^)   # HD_FT_SLOT_Q_NORM
    dw.append(d_k_norm_g^)   # HD_FT_SLOT_K_NORM
    dw.append(d_post_ln^)    # HD_FT_SLOT_POST_LN

    return HiDreamO1BlockFTGrads(TArc(d_hidden^), dw^)
