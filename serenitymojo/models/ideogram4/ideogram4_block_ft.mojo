# serenitymojo/models/ideogram4/ideogram4_block_ft.mojo
#
# FULL-FINETUNE device block backward for the Ideogram-4 transformer block —
# full-FT rollout P1 for ideogram4 (docs/FULL_FINETUNE_ROLLOUT_PLAN_2026-07-07.md,
# blueprint: .claude/skills/reference-mojo-full-finetune — krea2 worked example; flux/
# chroma precedent models/flux/flux_block_ft.mojo).
#
# Same hand-chain as ideogram4_block_lora_backward (models/ideogram4/block.mojo)
# but the base matmul WEIGHTS are TRAINABLE: each projection site emits
# d_W = d_yᵀ @ x_saved via ops/linalg_backward.linear_backward_dw with EXPLICIT
# STDtype.F32 output (the BOOL default sentinel means "match input dtype" ->
# bf16 grads the F32 optimizer rejects — krea2 trap), alongside the plain bf16
# d_x carry (linear_backward_dx). NO LoRA anywhere, NO int8 (reference trainer full-FT forbids
# quantized linears — quantized forwards detach the weight).
#
# v2 trainable surface (FULL_SURFACE_PLAN_2026-07-08 Phase B row 1) = ALL 13
# per-block params, in I4_FT_SLOT_* order (0-5 are the v1 I4_SLOT_* matmuls):
#   [0] qkv_w  [3*Hidden, Hidden]   (attention.qkv.weight)
#   [1] o_w    [Hidden, Hidden]     (attention.o.weight)
#   [2] w1     [FF, Hidden]         (feed_forward.w1.weight)
#   [3] w2     [Hidden, FF]         (feed_forward.w2.weight)
#   [4] w3     [FF, Hidden]         (feed_forward.w3.weight)
#   [5] adaln_w [4*Hidden, Adaln]   (adaln_modulation.weight)
#   [6] adaln_b [4*Hidden]          (adaln_modulation.bias — d_b = d_mod row-sum;
#                                    the mod linear has ONE row [1,Adaln] so the
#                                    row-sum is a reshape)
#   [7] attn_norm1 [Hidden]  [8] attn_norm2 [Hidden]   (rms scales, full
#   [9] ffn_norm1  [Hidden]  [10] ffn_norm2 [Hidden]    rms_norm_backward d_g)
#   [11] norm_q [Dh]         [12] norm_k [Dh]
# The 1D grads arrive bf16 from rms_norm_backward (F32-accumulated d_g kernel,
# bf16 store dtype) / bf16 d_mod — CAST to F32 explicitly for the optimizer
# (the krea2 dW-dtype trap, same reason as the STDtype.F32 on the matmul dW).
# v2 REPLACES the v1 (matmul-only) surface on the FT arm — reference trainer trains everything,
# there is no matmul-only reference trainer mode to keep.
#
# dX fold orders (C15) match ideogram4_block_lora_backward EXACTLY (minus the
# adapter add, which is the only dX term LoRA contributes):
#   d_mlp_in = add(w1_dx, w3_dx); d_x_mid = add(d_x_mid, ffn_norm1-dx);
#   d_x = add(d_x, attn_norm1-dx); d_qkv = concat(d_q, d_k, d_v);
#   d_mod = concat(d_scale_msa, d_gate_msa_raw, d_scale_mlp, d_gate_mlp_raw).
#
# SDPA arm: same comptime IDEOGRAM4_SDPA_FLASH dispatch as the LoRA backward
# (flag is False — cuDNN flash bwd has no Dh=256 plan; math sdpa_backward runs,
# which is also the parity-gate mode).
#
# Forward for this backward: the base block. ideogram4_block_lora_forward with
# zero-B adapters IS the base forward bit-for-bit (up = down @ 0 = 0; add(base,
# 0*scale) = base), so its Ideogram4BlockActs tape is the base tape — the gate
# (models/ideogram4/parity/ideogram4_block_ft_parity.mojo) drives it that way.
#
# Oracle: ideogram4_torchref_oracle.py IDEOGRAM4_FT_ORACLE=1 (base-only mode —
# per MJ-1041 the ideogram4 oracle is torchref; the LoRA-mode refs are NOT
# valid for FT dX, and the base weights are frozen there so they carry no
# W.grad at all). Gate bar cos >= 0.999 at the REAL dims (S=260, Hidden=4608,
# H=18, Dh=256, FF=12288, Adaln=512 — real checkpoint block-0 weights).
#
# Mojo current beta: `def`, `comptime`, `var`, explicit `raises`.

from std.math import sqrt
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.ops.linalg_backward import linear_backward_dx, linear_backward_dw
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.attention_flash import sdpa_flash_backward_f32
from serenitymojo.ops.activation_backward import tanh_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.shape_backward import broadcast_backward
from serenitymojo.ops.rope_struct_backward import rope_halfsplit_full_backward
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    add, concat, mul, reshape, zeros_device,
)

from serenitymojo.models.ideogram4.block import (
    IDEOGRAM4_SDPA_FLASH,
    I4_EPS,
    I4_SLOTS_PER_BLOCK,
    I4_SLOT_QKV,
    I4_SLOT_O,
    I4_SLOT_W1,
    I4_SLOT_W2,
    I4_SLOT_W3,
    I4_SLOT_ADALN,
    Ideogram4BlockActs,
    Ideogram4BlockWeights,
    _clone,
    _expand_rope_table,
    _shape1,
    _shape2,
    _shape4,
)

comptime TArc = ArcPointer[Tensor]

# v2 full-surface slot indices 6-12 (0-5 are block.mojo's I4_SLOT_* matmuls).
# FIXED order — the host store tails, af_states flat index, sidecar and the
# parity gate all key off it.
comptime I4_FT_SLOT_ADALN_B = 6
comptime I4_FT_SLOT_ATTN_NORM1 = 7
comptime I4_FT_SLOT_ATTN_NORM2 = 8
comptime I4_FT_SLOT_FFN_NORM1 = 9
comptime I4_FT_SLOT_FFN_NORM2 = 10
comptime I4_FT_SLOT_NORM_Q = 11
comptime I4_FT_SLOT_NORM_K = 12
comptime I4_FT_SLOTS_V2 = 13


# FT grad carrier: plain dX/d_adaln carries + the 13 device-F32 param grads.
struct Ideogram4BlockFTGrads(Movable):
    var d_x: Tensor              # [S, Hidden] bf16 input grad (inter-block carry)
    var d_adaln_input: Tensor    # [1, Adaln] bf16 adaln-input grad carry
    # len 13, F32 device, FIXED I4_FT_SLOT_* order:
    #   [0] d_qkv_w [3H,H]  [1] d_o_w [H,H]  [2] d_w1 [FF,H]
    #   [3] d_w2 [H,FF]     [4] d_w3 [FF,H]  [5] d_adaln_w [4H,Adaln]
    #   [6] d_adaln_b [4H]  [7-10] d_{attn,ffn}_norm{1,2} [H]
    #   [11] d_norm_q [Dh]  [12] d_norm_k [Dh]
    var dw: List[TArc]

    def __init__(
        out self, var d_x: Tensor, var d_adaln_input: Tensor, var dw: List[TArc]
    ):
        self.d_x = d_x^
        self.d_adaln_input = d_adaln_input^
        self.dw = dw^


def ideogram4_block_ft_backward_dev[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    d_out: Tensor,
    acts: Ideogram4BlockActs,
    cosf: Tensor,
    sinf: Tensor,
    w: Ideogram4BlockWeights,
    ctx: DeviceContext,
) raises -> Ideogram4BlockFTGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var dw = List[TArc]()
    for _ in range(I4_FT_SLOTS_V2):
        dw.append(TArc(zeros_device(_shape1(1), STDtype.F32, ctx)))

    # out = x_mid + gate_mlp * ff_n2
    var d_x_mid = _clone(d_out, ctx)
    var d_ff_n2 = mul(d_out, acts.gate_mlp, ctx)
    var d_gate_mlp = broadcast_backward(
        mul(d_out, acts.ff_n2, ctx), acts.gate_mlp.shape(), ctx
    )
    # ffn_norm2 rms scale TRAINABLE (v2) -> full rms backward (d_x + d_g);
    # d_g arrives bf16 (store dtype, F32-accumulated) -> cast F32 for the opt.
    var rb_fn2 = rms_norm_backward(d_ff_n2, acts.ff_out, w.ffn_norm2, I4_EPS, ctx)
    dw[I4_FT_SLOT_FFN_NORM2] = TArc(cast_tensor(rb_fn2.d_g, STDtype.F32, ctx))

    # ff_out = linear(ff_act, w2): TRAINABLE weight -> d_W (F32) + d_x carry.
    dw[I4_SLOT_W2] = TArc(
        linear_backward_dw(rb_fn2.d_x, acts.ff_act, S, FF, Hidden, ctx, STDtype.F32)
    )
    var w2_dx = linear_backward_dx(rb_fn2.d_x, w.w2, S, FF, Hidden, ctx)

    var sg = swiglu_backward(w2_dx, acts.ff_g, acts.ff_u, ctx)

    # ff_g = linear(mlp_in, w1): TRAINABLE weight.
    dw[I4_SLOT_W1] = TArc(
        linear_backward_dw(sg.d_gate, acts.mlp_in, S, Hidden, FF, ctx, STDtype.F32)
    )
    var w1_dx = linear_backward_dx(sg.d_gate, w.w1, S, Hidden, FF, ctx)

    # ff_u = linear(mlp_in, w3): TRAINABLE weight.
    dw[I4_SLOT_W3] = TArc(
        linear_backward_dw(sg.d_up, acts.mlp_in, S, Hidden, FF, ctx, STDtype.F32)
    )
    var w3_dx = linear_backward_dx(sg.d_up, w.w3, S, Hidden, FF, ctx)

    # C15: w1 branch first, w3 branch second (the LoRA fold order).
    var d_mlp_in = add(w1_dx, w3_dx, ctx)
    var d_scale_mlp = broadcast_backward(
        mul(d_mlp_in, acts.fn1, ctx), acts.mod_scale_mlp.shape(), ctx
    )
    var d_fn1 = mul(d_mlp_in, acts.mod_scale_mlp, ctx)
    # ffn_norm1 rms scale TRAINABLE (v2).
    var rb_fn1 = rms_norm_backward(d_fn1, acts.x_mid, w.ffn_norm1, I4_EPS, ctx)
    dw[I4_FT_SLOT_FFN_NORM1] = TArc(cast_tensor(rb_fn1.d_g, STDtype.F32, ctx))
    d_x_mid = add(d_x_mid, rb_fn1.d_x, ctx)

    # x_mid = x + gate_msa * attn_n2
    var d_x = _clone(d_x_mid, ctx)
    var d_attn_n2 = mul(d_x_mid, acts.gate_msa, ctx)
    var d_gate_msa = broadcast_backward(
        mul(d_x_mid, acts.attn_n2, ctx), acts.gate_msa.shape(), ctx
    )
    # attn_norm2 rms scale TRAINABLE (v2).
    var rb_an2 = rms_norm_backward(d_attn_n2, acts.attn_out, w.attn_norm2, I4_EPS, ctx)
    dw[I4_FT_SLOT_ATTN_NORM2] = TArc(cast_tensor(rb_an2.d_g, STDtype.F32, ctx))

    # attn_out = linear(attn_flat, o_w): TRAINABLE weight.
    dw[I4_SLOT_O] = TArc(
        linear_backward_dw(rb_an2.d_x, acts.attn_flat, S, Hidden, Hidden, ctx, STDtype.F32)
    )
    var o_dx = linear_backward_dx(rb_an2.d_x, w.o_w, S, Hidden, Hidden, ctx)

    var d_attn4 = reshape(o_dx, _shape4(1, S, Heads, Dh), ctx)
    var sd_d_q: Tensor
    var sd_d_k: Tensor
    var sd_d_v: Tensor
    comptime if IDEOGRAM4_SDPA_FLASH:
        if not acts.flash_stats:
            raise Error(
                "ideogram4 block ft bwd: IDEOGRAM4_SDPA_FLASH on but the tape"
                " has no flash set (fwd/bwd flag mismatch)"
            )
        var fb = sdpa_flash_backward_f32[1, S, Heads, Dh](
            acts.flash_q.value(), acts.flash_k.value(), acts.flash_v.value(),
            acts.flash_o.value(), acts.flash_stats.value(), d_attn4, scale, ctx,
        )
        sd_d_q = cast_tensor(fb.d_q, STDtype.BF16, ctx)
        sd_d_k = cast_tensor(fb.d_k, STDtype.BF16, ctx)
        sd_d_v = cast_tensor(fb.d_v, STDtype.BF16, ctx)
    else:
        var sd = sdpa_backward[1, S, Heads, Dh](
            acts.q_rope, acts.k_rope, acts.v_bshd, d_attn4, scale, ctx
        )
        sd_d_q = Tensor(sd.d_q.buf.copy(), sd.d_q.shape(), sd.d_q.dtype())
        sd_d_k = Tensor(sd.d_k.buf.copy(), sd.d_k.shape(), sd.d_k.dtype())
        sd_d_v = Tensor(sd.d_v.buf.copy(), sd.d_v.shape(), sd.d_v.dtype())
    var cos_full = _expand_rope_table[Heads, Dh](cosf, S, ctx)
    var sin_full = _expand_rope_table[Heads, Dh](sinf, S, ctx)
    var d_q_norm = rope_halfsplit_full_backward(sd_d_q, cos_full, sin_full, ctx)
    var d_k_norm = rope_halfsplit_full_backward(sd_d_k, cos_full, sin_full, ctx)

    # norm_q/norm_k rms scales TRAINABLE (v2) — q_raw/k_raw [1,S,Heads,Dh],
    # d_g [Dh] (rows = S*Heads).
    var rb_q = rms_norm_backward(d_q_norm, acts.q_raw, w.norm_q, I4_EPS, ctx)
    dw[I4_FT_SLOT_NORM_Q] = TArc(cast_tensor(rb_q.d_g, STDtype.F32, ctx))
    var rb_k = rms_norm_backward(d_k_norm, acts.k_raw, w.norm_k, I4_EPS, ctx)
    dw[I4_FT_SLOT_NORM_K] = TArc(cast_tensor(rb_k.d_g, STDtype.F32, ctx))
    var d_q = reshape(rb_q.d_x, _shape2(S, Hidden), ctx)
    var d_k = reshape(rb_k.d_x, _shape2(S, Hidden), ctx)
    var d_v = reshape(sd_d_v, _shape2(S, Hidden), ctx)
    var d_qkv = concat(1, ctx, d_q, d_k, d_v)

    # qkv = linear(attn_in, qkv_w): TRAINABLE weight.
    dw[I4_SLOT_QKV] = TArc(
        linear_backward_dw(d_qkv, acts.attn_in, S, Hidden, 3 * Hidden, ctx, STDtype.F32)
    )
    var qkv_dx = linear_backward_dx(d_qkv, w.qkv_w, S, Hidden, 3 * Hidden, ctx)

    var d_scale_msa = broadcast_backward(
        mul(qkv_dx, acts.an1, ctx), acts.mod_scale_msa.shape(), ctx
    )
    var d_an1 = mul(qkv_dx, acts.mod_scale_msa, ctx)
    # attn_norm1 rms scale TRAINABLE (v2).
    var rb_an1 = rms_norm_backward(d_an1, acts.x_in, w.attn_norm1, I4_EPS, ctx)
    dw[I4_FT_SLOT_ATTN_NORM1] = TArc(cast_tensor(rb_an1.d_g, STDtype.F32, ctx))
    d_x = add(d_x, rb_an1.d_x, ctx)

    # d_mod chunks: scale chunks are identity; gate chunks pass tanh backward.
    var d_gate_msa_raw = tanh_backward(d_gate_msa, acts.mod_gate_msa_raw, ctx)
    var d_gate_mlp_raw = tanh_backward(d_gate_mlp, acts.mod_gate_mlp_raw, ctx)
    var d_mod_axis = len(d_scale_msa.shape()) - 1
    var d_mod = concat(
        d_mod_axis, ctx, d_scale_msa, d_gate_msa_raw, d_scale_mlp, d_gate_mlp_raw
    )

    # mod = linear(adaln_input, adaln_w) + adaln_b: BOTH TRAINABLE (v2).
    # d_b = d_mod summed over the linear's rows — here S=1 ([1, 4H]), so the
    # row-sum is the reshape; cast bf16 -> F32 for the optimizer.
    dw[I4_FT_SLOT_ADALN_B] = TArc(
        cast_tensor(reshape(d_mod, _shape1(4 * Hidden), ctx), STDtype.F32, ctx)
    )
    dw[I4_SLOT_ADALN] = TArc(
        linear_backward_dw(d_mod, acts.adaln_input, 1, Adaln, 4 * Hidden, ctx, STDtype.F32)
    )
    var adaln_dx = linear_backward_dx(d_mod, w.adaln_w, 1, Adaln, 4 * Hidden, ctx)

    return Ideogram4BlockFTGrads(d_x^, _clone(adaln_dx, ctx), dw^)
