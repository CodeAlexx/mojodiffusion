# serenitymojo/models/ltx2/ltx2_video_backward.mojo
#
# LTX-2.3 22B VIDEO-ONLY (audio=None) BLOCK — activation-saving TRAIN forward +
# hand-chained BACKWARD for the video-mode T2V LoRA surface.
#
# This is the video-only training arm musubi runs when the audio modality is
# absent (BasicAVTransformerBlock._forward, audio=None):
#   /home/alex/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/transformer.py
#     run_vx  :589  -> True    (video present + enabled + numel>0)
#     run_ax  :590  -> False   (audio is None)
#     run_a2v :592  -> False   (run_vx AND audio-not-None -> False)
#     run_v2a :593  -> False   (run_ax -> False)
#   so ONLY the video stream runs: self-attn (attn1, video rope) :595-611,
#   text cross-attn (attn2) :613-628, video FFN :763-776. The audio self /
#   audio text-cross / cross-modal a2v/v2a / audio FFN blocks are all guarded
#   off (`if run_ax:` / `if run_a2v or run_v2a:`) and never execute.
#
# TENET (train/infer same math): this reuses the EXACT video-stream ops of the
# gated joint-AV train forward `ltx2_av_backward.ltx2_block_forward_av_train`
# (same weights struct, same `_av_attention_train`/`_av_attention_bwd`, same
# helper math from ltx2_dit). The only structural difference vs the AV path is
# that the FFN pre-norm consumes `hs2` (post text cross-attn) directly — there
# is no a2v cross-modal residual (`hs3 == hs2` when audio is absent).
#
# The gated AV file (ltx2_av_backward.mojo) is imported, NOT modified.
#
# BACKWARD SCOPE (LoRA training; base FROZEN):
#   outputs d_hidden (video stream input grad) + d_A/d_B for the video-mode
#   surface = 8 pairs/block: {to_q,to_k,to_v,to_out.0} x {attn1, attn2}
#   (musubi LTX2_INCLUDE_PATTERNS_T2V restricted to the video-active modules).
#
# Parity gate: serenitymojo/models/ltx2/parity/ltx2_video_bwd_parity.mojo vs the
# torch.autograd oracle scripts/ltx2_video_block_bwd_oracle.py.
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor is move-only (Movable result structs);
# struct fields moved with `^`.

from max.gpu.host import DeviceContext
from std.collections import List

from serenitymojo.tensor import Tensor

# forward ops (the spine's ops)
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import gelu, gelu_slab
from serenitymojo.ops.tensor_algebra import (
    reshape, add, mul, reshape_slab, add_slab, mul_slab, full_device_slab,
)
from serenitymojo.models.dit.ltx2_dit import (
    LTX2AVBlockWeights,
    _ada_row_pertok,
    _modulate_bc,
    _kv_modulate,
    _rms_norm_opt,
    _shape3,
    _ada_row_pertok_slab,
    _modulate_bc_slab,
    _kv_modulate_slab,
    _rms_norm_opt_slab,
)

# backward arms (all pre-built + gated)
from serenitymojo.ops.linalg_backward import linear_backward_dx, linear_backward_dx_slab
from serenitymojo.ops.activation_backward import gelu_backward, gelu_backward_slab

# shared AV helpers + per-attention train/bwd (imported, NOT modified)
from serenitymojo.models.ltx2.ltx2_av_backward import (
    LoraPairGrad,
    AVAttnActs,
    _av_attention_train,
    _av_attention_bwd,
    _lora_pair_bwd,
    _rms_bwd_opt,
    _one_plus,
    _sh2,
    _dummy_t,
    _av_attention_train_slab,
    _av_attention_bwd_slab,
    _lora_pair_bwd_slab,
    _rms_bwd_opt_slab,
    _one_plus_slab,
    LoraPairGradDev,
    _av_attention_bwd_slab_dev,
    _lora_pair_bwd_slab_dev,
    ltx2_lora_dev_readback,
    Ltx2LoraGradStore,
)
from serenitymojo.autograd_v2.step_slab import StepSlab


# v2v: cheap probe for an attached LoRA factor on `w_key` (the pair backward's
# own scan is the activeness check; this gates the FFN recompute so t2v — no FFN
# factors — does ZERO extra work).
def _has_lora_factor(weights: LTX2AVBlockWeights, w_key: String) -> Bool:
    for i in range(len(weights.lora_names)):
        if weights.lora_names[i] == w_key:
            return True
    return False


# ── block-level saved activations (video stream only) ────────────────────────
struct LTX2VideoBlockActs(Movable):
    var hidden: Tensor     # [1,S_V,4096] block video input
    var at1: AVAttnActs    # attn1 (video self)
    var hs1: Tensor        # video post self-attn
    var at2: AVAttnActs    # attn2 (video<-text)
    var hs2: Tensor        # video post text cross-attn (FFN pre-norm input)
    var h1_v: Tensor       # [1,S_V,16384] video FFN pre-gelu

    def __init__(
        out self,
        var hidden: Tensor,
        var at1: AVAttnActs,
        var hs1: Tensor,
        var at2: AVAttnActs,
        var hs2: Tensor,
        var h1_v: Tensor,
    ):
        self.hidden = hidden^
        self.at1 = at1^
        self.hs1 = hs1^
        self.at2 = at2^
        self.hs2 = hs2^
        self.h1_v = h1_v^


struct LTX2VideoTrainForward(Movable):
    var video_out: Tensor  # [1,S_V,4096]
    var acts: LTX2VideoBlockActs

    def __init__(
        out self, var video_out: Tensor, var acts: LTX2VideoBlockActs,
    ):
        self.video_out = video_out^
        self.acts = acts^


# ── TRAIN FORWARD (video-stream mirror of ltx2_block_forward_av_train) ───────
def ltx2_video_block_train_forward[S_V: Int, N_TXT: Int](
    weights: LTX2AVBlockWeights,
    hidden: Tensor,
    enc: Tensor,
    v_temb: Tensor,
    v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> LTX2VideoTrainForward:
    var VD = 4096
    var dummy = _dummy_t(hidden.dtype(), ctx)

    # ---- 1. video self-attn (AdaLN rows 0..2, rope, gated residual) ----
    ref vtab = weights._w("scale_shift_table")
    var v_shift_msa = _ada_row_pertok(vtab, v_temb, 0, VD, S_V, ctx)
    var v_scale_msa = _ada_row_pertok(vtab, v_temb, 1, VD, S_V, ctx)
    var v_gate_msa = _ada_row_pertok(vtab, v_temb, 2, VD, S_V, ctx)
    var v_shift_mlp = _ada_row_pertok(vtab, v_temb, 3, VD, S_V, ctx)
    var v_scale_mlp = _ada_row_pertok(vtab, v_temb, 4, VD, S_V, ctx)
    var v_gate_mlp = _ada_row_pertok(vtab, v_temb, 5, VD, S_V, ctx)

    var mod_h = _modulate_bc(
        _rms_norm_opt(hidden, weights, "norm1.weight", eps, ctx),
        v_scale_msa, v_shift_msa, ctx,
    )
    var r_at1 = _av_attention_train[S_V, S_V, 32, 128](
        weights, "attn1", mod_h, mod_h,
        True, v_cos, v_sin, False, dummy, dummy, eps, ctx,
    )
    var hs1 = add(hidden, mul(v_gate_msa, r_at1.out, ctx), ctx)

    # ---- 2. video text cross-attn (AdaLN rows 6..8, KV-modulated context) ----
    var v_shift_ca = _ada_row_pertok(vtab, v_temb, 6, VD, S_V, ctx)
    var v_scale_ca = _ada_row_pertok(vtab, v_temb, 7, VD, S_V, ctx)
    var v_gate_ca = _ada_row_pertok(vtab, v_temb, 8, VD, S_V, ctx)
    var mod_h2 = _modulate_bc(
        _rms_norm_opt(hs1, weights, "norm2.weight", eps, ctx),
        v_scale_ca, v_shift_ca, ctx,
    )
    var mv_ctx: Tensor
    if weights._has("prompt_scale_shift_table"):
        mv_ctx = _kv_modulate(
            enc, weights._w("prompt_scale_shift_table"), v_prompt_ts,
            N_TXT, VD, ctx,
        )
    else:
        mv_ctx = enc.clone(ctx)
    var r_at2 = _av_attention_train[S_V, N_TXT, 32, 128](
        weights, "attn2", mod_h2, mv_ctx,
        False, dummy, dummy, False, dummy, dummy, eps, ctx,
    )
    var hs2 = add(hs1, mul(v_gate_ca, r_at2.out, ctx), ctx)

    # ---- 3. video FFN (gated residual off hs2; no a2v when audio absent) ----
    var mod_ff = _modulate_bc(
        _rms_norm_opt(hs2, weights, "norm3.weight", eps, ctx),
        v_scale_mlp, v_shift_mlp, ctx,
    )
    var h1_v = weights._linear_b(
        mod_ff, "ff.net.0.proj.weight", "ff.net.0.proj.bias", ctx)
    var h1g_v = gelu(h1_v, ctx)
    var ff_v = weights._linear_b(h1g_v, "ff.net.2.weight", "ff.net.2.bias", ctx)
    var video_out = add(hs2, mul(v_gate_mlp, ff_v, ctx), ctx)

    var acts = LTX2VideoBlockActs(
        hidden.clone(ctx), r_at1^, hs1^, r_at2^, hs2^, h1_v^,
    )
    return LTX2VideoTrainForward(video_out^, acts^)


# ── BACKWARD result ───────────────────────────────────────────────────────────
struct LTX2VideoBlockGrads(Movable):
    var d_hidden: Tensor              # [1,S_V,4096]
    var lora: List[LoraPairGrad]      # d_A/d_B per attached adapter (8)

    def __init__(
        out self, var d_hidden: Tensor, var lora: List[LoraPairGrad],
    ):
        self.d_hidden = d_hidden^
        self.lora = lora^


# ── BLOCK BACKWARD (hand-chained reverse of ltx2_video_block_train_forward) ──
def ltx2_video_block_backward[S_V: Int, N_TXT: Int](
    weights: LTX2AVBlockWeights,
    acts: LTX2VideoBlockActs,
    d_video: Tensor,                    # [1,S_V,4096]
    v_temb: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> LTX2VideoBlockGrads:
    var VD = 4096
    var FFV = 16384
    var dummy = _dummy_t(d_video.dtype(), ctx)
    var lora_grads = List[LoraPairGrad]()

    # Recompute the AdaLN rows (cheap slices/adds; the same helper math).
    ref vtab = weights._w("scale_shift_table")
    var v_scale_msa = _ada_row_pertok(vtab, v_temb, 1, VD, S_V, ctx)
    var v_gate_msa = _ada_row_pertok(vtab, v_temb, 2, VD, S_V, ctx)
    var v_scale_mlp = _ada_row_pertok(vtab, v_temb, 4, VD, S_V, ctx)
    var v_gate_mlp = _ada_row_pertok(vtab, v_temb, 5, VD, S_V, ctx)
    var v_scale_ca = _ada_row_pertok(vtab, v_temb, 7, VD, S_V, ctx)
    var v_gate_ca = _ada_row_pertok(vtab, v_temb, 8, VD, S_V, ctx)

    # ---- 3r. video FFN: video_out = hs2 + v_gate_mlp * ff(mod(norm3(hs2))) ----
    # v2v LoRA on ff.net.0.proj (up, 4096->16384) + ff.net.2 (down, 16384->4096),
    # CONDITIONAL on the factors being attached so the t2v path does ZERO extra
    # work. `x` for each pair backward is recomputed from saved acts (no new saved
    # tensors): h1g_v = gelu(h1_v); mod_ff = modulate(norm3(hs2)) with v_shift_mlp
    # (ada row 3) + v_scale_mlp (row 4). Each pair's d_x is added into the base
    # chain BEFORE the next stage (ff.net.2 -> d_h1g_v pre-gelu_backward;
    # ff.net.0.proj -> d_mod_ff pre-_one_plus) so d_hidden stays exact under v2v.
    var ffn_lora = (
        _has_lora_factor(weights, "ff.net.2.weight")
        or _has_lora_factor(weights, "ff.net.0.proj.weight"))
    var d_hs2 = d_video.clone(ctx)
    var d_ff_v = reshape(mul(d_video, v_gate_mlp, ctx), _sh2(S_V, VD), ctx)
    var d_h1g_v = linear_backward_dx(
        d_ff_v, weights._w("ff.net.2.weight"), S_V, FFV, VD, ctx)
    if ffn_lora:
        # ff.net.2 down: y=ff_v[S_V,VD], x=h1g_v[S_V,FFV]=gelu(h1_v); d_y=d_ff_v.
        var h1g_v2 = gelu(reshape(acts.h1_v, _sh2(S_V, FFV), ctx), ctx)
        var l2 = _lora_pair_bwd(
            weights, String("ff.net.2.weight"), d_ff_v, h1g_v2, S_V, ctx, lora_grads)
        if l2:
            d_h1g_v = add(d_h1g_v, l2.value(), ctx)
    var d_h1_v = gelu_backward(
        d_h1g_v, reshape(acts.h1_v, _sh2(S_V, FFV), ctx), ctx)
    var d_mod_ff = linear_backward_dx(
        d_h1_v, weights._w("ff.net.0.proj.weight"), S_V, VD, FFV, ctx)
    if ffn_lora:
        # ff.net.0.proj up: y=h1_v[S_V,FFV], x=mod_ff[S_V,VD]; d_y=d_h1_v.
        var v_shift_mlp = _ada_row_pertok(vtab, v_temb, 3, VD, S_V, ctx)
        var mod_ff2 = reshape(
            _modulate_bc(
                _rms_norm_opt(acts.hs2, weights, "norm3.weight", eps, ctx),
                v_scale_mlp, v_shift_mlp, ctx),
            _sh2(S_V, VD), ctx)
        var l0 = _lora_pair_bwd(
            weights, String("ff.net.0.proj.weight"), d_h1_v, mod_ff2, S_V, ctx, lora_grads)
        if l0:
            d_mod_ff = add(d_mod_ff, l0.value(), ctx)
    var d_norm3 = mul(
        reshape(d_mod_ff, _shape3(1, S_V, VD), ctx),
        _one_plus(v_scale_mlp, ctx), ctx)
    d_hs2 = add(
        d_hs2, _rms_bwd_opt(d_norm3, acts.hs2, weights, "norm3.weight", eps, ctx),
        ctx)

    # ---- 2r. video text cross-attn: hs2 = hs1 + v_gate_ca * attn2(...) ----
    # KV grads (text context) are dropped — untrained leaves outside the block.
    var d_hs1 = d_hs2.clone(ctx)
    var d_vca = mul(d_hs2, v_gate_ca, ctx)
    var g_at2 = _av_attention_bwd[S_V, N_TXT, 32, 128](
        weights, "attn2", acts.at2, d_vca,
        False, dummy, dummy, False, dummy, dummy, eps, ctx, lora_grads)
    var d_norm2 = mul(g_at2.d_q_src, _one_plus(v_scale_ca, ctx), ctx)
    d_hs1 = add(
        d_hs1, _rms_bwd_opt(d_norm2, acts.hs1, weights, "norm2.weight", eps, ctx),
        ctx)

    # ---- 1r. video self-attn: hs1 = hidden + v_gate_msa * attn1(mod_h) ----
    # Q-source and KV-source are the SAME tensor (mod_h): sum both grads.
    var d_hidden = d_hs1.clone(ctx)
    var d_vsa = mul(d_hs1, v_gate_msa, ctx)
    var g_at1 = _av_attention_bwd[S_V, S_V, 32, 128](
        weights, "attn1", acts.at1, d_vsa,
        True, v_cos, v_sin, False, dummy, dummy, eps, ctx, lora_grads)
    var d_mod_h = add(g_at1.d_q_src, g_at1.d_kv_src, ctx)
    var d_norm1 = mul(d_mod_h, _one_plus(v_scale_msa, ctx), ctx)
    d_hidden = add(
        d_hidden,
        _rms_bwd_opt(d_norm1, acts.hidden, weights, "norm1.weight", eps, ctx),
        ctx)

    return LTX2VideoBlockGrads(d_hidden^, lora_grads^)


# ═════════════════════════════════════════════════════════════════════════════
# StepSlab TWINS of the video block train-forward + backward (autograd_v2 C8).
# Op-for-op identical to the pair above; every op swapped for its *_slab twin
# and every .clone → reshape_slab (byte-identical D2D copy). Saved acts are
# slab-backed and live across fwd→bwd within ONE apply_ltx2v mark/rewind window
# (the block twin's caller copies d_hidden + LoRA grads OUT before rewind).
# The absent-rope placeholder and the no-affine ones weight (_rms_norm_no_affine
# / _rms_bwd_opt) are built with full_device_slab — slab-allocated + GPU-filled,
# NO host upload / off-slab alloc — so the captured region stays alloc-clean;
# byte-identical to the from_host constants since 1.0 is exact in bf16/f32, so
# the bit gate holds.
# ═════════════════════════════════════════════════════════════════════════════
def ltx2_video_block_train_forward_slab[S_V: Int, N_TXT: Int](
    weights: LTX2AVBlockWeights,
    hidden: Tensor,
    enc: Tensor,
    v_temb: Tensor,
    v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> LTX2VideoTrainForward:
    """StepSlab twin of `ltx2_video_block_train_forward` (this file :122)."""
    var VD = 4096
    # Absent-rope placeholder, slab-routed + GPU-filled (never READ — attn2 has
    # no rope; capture-clean, no off-slab alloc).
    var dd = List[Int](); dd.append(1); dd.append(1)
    var dummy = full_device_slab(dd^, Float32(1.0), hidden.dtype(), ctx, slab)

    # ---- 1. video self-attn ----
    ref vtab = weights._w("scale_shift_table")
    var v_shift_msa = _ada_row_pertok_slab(vtab, v_temb, 0, VD, S_V, ctx, slab)
    var v_scale_msa = _ada_row_pertok_slab(vtab, v_temb, 1, VD, S_V, ctx, slab)
    var v_gate_msa = _ada_row_pertok_slab(vtab, v_temb, 2, VD, S_V, ctx, slab)
    var v_shift_mlp = _ada_row_pertok_slab(vtab, v_temb, 3, VD, S_V, ctx, slab)
    var v_scale_mlp = _ada_row_pertok_slab(vtab, v_temb, 4, VD, S_V, ctx, slab)
    var v_gate_mlp = _ada_row_pertok_slab(vtab, v_temb, 5, VD, S_V, ctx, slab)

    var mod_h = _modulate_bc_slab(
        _rms_norm_opt_slab(hidden, weights, "norm1.weight", eps, ctx, slab),
        v_scale_msa, v_shift_msa, ctx, slab,
    )
    var r_at1 = _av_attention_train_slab[S_V, S_V, 32, 128](
        weights, "attn1", mod_h, mod_h,
        True, v_cos, v_sin, False, dummy, dummy, eps, ctx, slab,
    )
    var hs1 = add_slab(hidden, mul_slab(v_gate_msa, r_at1.out, ctx, slab), ctx, slab)

    # ---- 2. video text cross-attn ----
    var v_shift_ca = _ada_row_pertok_slab(vtab, v_temb, 6, VD, S_V, ctx, slab)
    var v_scale_ca = _ada_row_pertok_slab(vtab, v_temb, 7, VD, S_V, ctx, slab)
    var v_gate_ca = _ada_row_pertok_slab(vtab, v_temb, 8, VD, S_V, ctx, slab)
    var mod_h2 = _modulate_bc_slab(
        _rms_norm_opt_slab(hs1, weights, "norm2.weight", eps, ctx, slab),
        v_scale_ca, v_shift_ca, ctx, slab,
    )
    var mv_ctx: Tensor
    if weights._has("prompt_scale_shift_table"):
        mv_ctx = _kv_modulate_slab(
            enc, weights._w("prompt_scale_shift_table"), v_prompt_ts,
            N_TXT, VD, ctx, slab,
        )
    else:
        mv_ctx = reshape_slab(enc, enc.shape(), ctx, slab)
    var r_at2 = _av_attention_train_slab[S_V, N_TXT, 32, 128](
        weights, "attn2", mod_h2, mv_ctx,
        False, dummy, dummy, False, dummy, dummy, eps, ctx, slab,
    )
    var hs2 = add_slab(hs1, mul_slab(v_gate_ca, r_at2.out, ctx, slab), ctx, slab)

    # ---- 3. video FFN ----
    var mod_ff = _modulate_bc_slab(
        _rms_norm_opt_slab(hs2, weights, "norm3.weight", eps, ctx, slab),
        v_scale_mlp, v_shift_mlp, ctx, slab,
    )
    var h1_v = weights._linear_b_slab(
        mod_ff, "ff.net.0.proj.weight", "ff.net.0.proj.bias", ctx, slab)
    var h1g_v = gelu_slab(h1_v, ctx, slab)
    var ff_v = weights._linear_b_slab(h1g_v, "ff.net.2.weight", "ff.net.2.bias", ctx, slab)
    var video_out = add_slab(hs2, mul_slab(v_gate_mlp, ff_v, ctx, slab), ctx, slab)

    var acts = LTX2VideoBlockActs(
        reshape_slab(hidden, hidden.shape(), ctx, slab), r_at1^, hs1^, r_at2^, hs2^, h1_v^,
    )
    return LTX2VideoTrainForward(video_out^, acts^)


def ltx2_video_block_backward_slab[S_V: Int, N_TXT: Int](
    weights: LTX2AVBlockWeights,
    acts: LTX2VideoBlockActs,
    d_video: Tensor,
    v_temb: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> LTX2VideoBlockGrads:
    """StepSlab twin of `ltx2_video_block_backward` (this file :205)."""
    var VD = 4096
    var FFV = 16384
    # Absent-rope placeholder, slab-routed + GPU-filled (capture-clean).
    var dd = List[Int](); dd.append(1); dd.append(1)
    var dummy = full_device_slab(dd^, Float32(1.0), d_video.dtype(), ctx, slab)
    var lora_grads = List[LoraPairGrad]()

    ref vtab = weights._w("scale_shift_table")
    var v_scale_msa = _ada_row_pertok_slab(vtab, v_temb, 1, VD, S_V, ctx, slab)
    var v_gate_msa = _ada_row_pertok_slab(vtab, v_temb, 2, VD, S_V, ctx, slab)
    var v_scale_mlp = _ada_row_pertok_slab(vtab, v_temb, 4, VD, S_V, ctx, slab)
    var v_gate_mlp = _ada_row_pertok_slab(vtab, v_temb, 5, VD, S_V, ctx, slab)
    var v_scale_ca = _ada_row_pertok_slab(vtab, v_temb, 7, VD, S_V, ctx, slab)
    var v_gate_ca = _ada_row_pertok_slab(vtab, v_temb, 8, VD, S_V, ctx, slab)

    # ---- 3r. video FFN ----
    var ffn_lora = (
        _has_lora_factor(weights, "ff.net.2.weight")
        or _has_lora_factor(weights, "ff.net.0.proj.weight"))
    var d_hs2 = reshape_slab(d_video, d_video.shape(), ctx, slab)
    var d_ff_v = reshape_slab(mul_slab(d_video, v_gate_mlp, ctx, slab), _sh2(S_V, VD), ctx, slab)
    var d_h1g_v = linear_backward_dx_slab(
        d_ff_v, weights._w("ff.net.2.weight"), S_V, FFV, VD, ctx, slab)
    if ffn_lora:
        var h1g_v2 = gelu_slab(reshape_slab(acts.h1_v, _sh2(S_V, FFV), ctx, slab), ctx, slab)
        var l2 = _lora_pair_bwd_slab(
            weights, String("ff.net.2.weight"), d_ff_v, h1g_v2, S_V, ctx, lora_grads, slab)
        if l2:
            d_h1g_v = add_slab(d_h1g_v, l2.value(), ctx, slab)
    var d_h1_v = gelu_backward_slab(
        d_h1g_v, reshape_slab(acts.h1_v, _sh2(S_V, FFV), ctx, slab), ctx, slab)
    var d_mod_ff = linear_backward_dx_slab(
        d_h1_v, weights._w("ff.net.0.proj.weight"), S_V, VD, FFV, ctx, slab)
    if ffn_lora:
        var v_shift_mlp = _ada_row_pertok_slab(vtab, v_temb, 3, VD, S_V, ctx, slab)
        var mod_ff2 = reshape_slab(
            _modulate_bc_slab(
                _rms_norm_opt_slab(acts.hs2, weights, "norm3.weight", eps, ctx, slab),
                v_scale_mlp, v_shift_mlp, ctx, slab),
            _sh2(S_V, VD), ctx, slab)
        var l0 = _lora_pair_bwd_slab(
            weights, String("ff.net.0.proj.weight"), d_h1_v, mod_ff2, S_V, ctx, lora_grads, slab)
        if l0:
            d_mod_ff = add_slab(d_mod_ff, l0.value(), ctx, slab)
    var d_norm3 = mul_slab(
        reshape_slab(d_mod_ff, _shape3(1, S_V, VD), ctx, slab),
        _one_plus_slab(v_scale_mlp, ctx, slab), ctx, slab)
    d_hs2 = add_slab(
        d_hs2, _rms_bwd_opt_slab(d_norm3, acts.hs2, weights, "norm3.weight", eps, ctx, slab),
        ctx, slab)

    # ---- 2r. video text cross-attn ----
    var d_hs1 = reshape_slab(d_hs2, d_hs2.shape(), ctx, slab)
    var d_vca = mul_slab(d_hs2, v_gate_ca, ctx, slab)
    var g_at2 = _av_attention_bwd_slab[S_V, N_TXT, 32, 128](
        weights, "attn2", acts.at2, d_vca,
        False, dummy, dummy, False, dummy, dummy, eps, ctx, lora_grads, slab)
    var d_norm2 = mul_slab(g_at2.d_q_src, _one_plus_slab(v_scale_ca, ctx, slab), ctx, slab)
    d_hs1 = add_slab(
        d_hs1, _rms_bwd_opt_slab(d_norm2, acts.hs1, weights, "norm2.weight", eps, ctx, slab),
        ctx, slab)

    # ---- 1r. video self-attn ----
    var d_hidden = reshape_slab(d_hs1, d_hs1.shape(), ctx, slab)
    var d_vsa = mul_slab(d_hs1, v_gate_msa, ctx, slab)
    var g_at1 = _av_attention_bwd_slab[S_V, S_V, 32, 128](
        weights, "attn1", acts.at1, d_vsa,
        True, v_cos, v_sin, False, dummy, dummy, eps, ctx, lora_grads, slab)
    var d_mod_h = add_slab(g_at1.d_q_src, g_at1.d_kv_src, ctx, slab)
    var d_norm1 = mul_slab(d_mod_h, _one_plus_slab(v_scale_msa, ctx, slab), ctx, slab)
    d_hidden = add_slab(
        d_hidden,
        _rms_bwd_opt_slab(d_norm1, acts.hidden, weights, "norm1.weight", eps, ctx, slab),
        ctx, slab)

    return LTX2VideoBlockGrads(d_hidden^, lora_grads^)


# ═════════════════════════════════════════════════════════════════════════════
# RUNG 1 — device-grad block backward. Byte-for-byte identical to
# ltx2_video_block_backward_slab except the LoRA grads are collected DEVICE-side
# (List[LoraPairGradDev] via _av_attention_bwd_slab_dev / _lora_pair_bwd_slab_dev,
# no per-adapter .to_host). d_hidden is bit-identical (same d_x math). The caller
# does ONE ltx2_lora_dev_readback at the boundary (outside the per-block region).
# ═════════════════════════════════════════════════════════════════════════════
struct LTX2VideoBlockGradsDev(Movable):
    var d_hidden: Tensor                  # [1,S_V,4096]
    var lora: List[LoraPairGradDev]       # device d_A/d_B per attached adapter

    def __init__(
        out self, var d_hidden: Tensor, var lora: List[LoraPairGradDev],
    ):
        self.d_hidden = d_hidden^
        self.lora = lora^


def ltx2_video_block_backward_slab_dev[S_V: Int, N_TXT: Int](
    weights: LTX2AVBlockWeights,
    acts: LTX2VideoBlockActs,
    d_video: Tensor,
    v_temb: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
    store: Ltx2LoraGradStore = Ltx2LoraGradStore(),
    block_idx: Int = -1,
) raises -> LTX2VideoBlockGradsDev:
    """Device-grad variant of `ltx2_video_block_backward_slab` (this file) —
    identical d_x math + fold order; LoRA d_A/d_B stay on device."""
    var VD = 4096
    var FFV = 16384
    var dd = List[Int](); dd.append(1); dd.append(1)
    var dummy = full_device_slab(dd^, Float32(1.0), d_video.dtype(), ctx, slab)
    var lora_grads = List[LoraPairGradDev]()

    ref vtab = weights._w("scale_shift_table")
    var v_scale_msa = _ada_row_pertok_slab(vtab, v_temb, 1, VD, S_V, ctx, slab)
    var v_gate_msa = _ada_row_pertok_slab(vtab, v_temb, 2, VD, S_V, ctx, slab)
    var v_scale_mlp = _ada_row_pertok_slab(vtab, v_temb, 4, VD, S_V, ctx, slab)
    var v_gate_mlp = _ada_row_pertok_slab(vtab, v_temb, 5, VD, S_V, ctx, slab)
    var v_scale_ca = _ada_row_pertok_slab(vtab, v_temb, 7, VD, S_V, ctx, slab)
    var v_gate_ca = _ada_row_pertok_slab(vtab, v_temb, 8, VD, S_V, ctx, slab)

    # ---- 3r. video FFN ----
    var ffn_lora = (
        _has_lora_factor(weights, "ff.net.2.weight")
        or _has_lora_factor(weights, "ff.net.0.proj.weight"))
    var d_hs2 = reshape_slab(d_video, d_video.shape(), ctx, slab)
    var d_ff_v = reshape_slab(mul_slab(d_video, v_gate_mlp, ctx, slab), _sh2(S_V, VD), ctx, slab)
    var d_h1g_v = linear_backward_dx_slab(
        d_ff_v, weights._w("ff.net.2.weight"), S_V, FFV, VD, ctx, slab)
    if ffn_lora:
        var h1g_v2 = gelu_slab(reshape_slab(acts.h1_v, _sh2(S_V, FFV), ctx, slab), ctx, slab)
        var l2 = _lora_pair_bwd_slab_dev(
            weights, String("ff.net.2.weight"), d_ff_v, h1g_v2, S_V, ctx, lora_grads, slab,
            store, block_idx)
        if l2:
            d_h1g_v = add_slab(d_h1g_v, l2.value(), ctx, slab)
    var d_h1_v = gelu_backward_slab(
        d_h1g_v, reshape_slab(acts.h1_v, _sh2(S_V, FFV), ctx, slab), ctx, slab)
    var d_mod_ff = linear_backward_dx_slab(
        d_h1_v, weights._w("ff.net.0.proj.weight"), S_V, VD, FFV, ctx, slab)
    if ffn_lora:
        var v_shift_mlp = _ada_row_pertok_slab(vtab, v_temb, 3, VD, S_V, ctx, slab)
        var mod_ff2 = reshape_slab(
            _modulate_bc_slab(
                _rms_norm_opt_slab(acts.hs2, weights, "norm3.weight", eps, ctx, slab),
                v_scale_mlp, v_shift_mlp, ctx, slab),
            _sh2(S_V, VD), ctx, slab)
        var l0 = _lora_pair_bwd_slab_dev(
            weights, String("ff.net.0.proj.weight"), d_h1_v, mod_ff2, S_V, ctx, lora_grads, slab,
            store, block_idx)
        if l0:
            d_mod_ff = add_slab(d_mod_ff, l0.value(), ctx, slab)
    var d_norm3 = mul_slab(
        reshape_slab(d_mod_ff, _shape3(1, S_V, VD), ctx, slab),
        _one_plus_slab(v_scale_mlp, ctx, slab), ctx, slab)
    d_hs2 = add_slab(
        d_hs2, _rms_bwd_opt_slab(d_norm3, acts.hs2, weights, "norm3.weight", eps, ctx, slab),
        ctx, slab)

    # ---- 2r. video text cross-attn ----
    var d_hs1 = reshape_slab(d_hs2, d_hs2.shape(), ctx, slab)
    var d_vca = mul_slab(d_hs2, v_gate_ca, ctx, slab)
    var g_at2 = _av_attention_bwd_slab_dev[S_V, N_TXT, 32, 128](
        weights, "attn2", acts.at2, d_vca,
        False, dummy, dummy, False, dummy, dummy, eps, ctx, lora_grads, slab,
        store, block_idx)
    var d_norm2 = mul_slab(g_at2.d_q_src, _one_plus_slab(v_scale_ca, ctx, slab), ctx, slab)
    d_hs1 = add_slab(
        d_hs1, _rms_bwd_opt_slab(d_norm2, acts.hs1, weights, "norm2.weight", eps, ctx, slab),
        ctx, slab)

    # ---- 1r. video self-attn ----
    var d_hidden = reshape_slab(d_hs1, d_hs1.shape(), ctx, slab)
    var d_vsa = mul_slab(d_hs1, v_gate_msa, ctx, slab)
    var g_at1 = _av_attention_bwd_slab_dev[S_V, S_V, 32, 128](
        weights, "attn1", acts.at1, d_vsa,
        True, v_cos, v_sin, False, dummy, dummy, eps, ctx, lora_grads, slab,
        store, block_idx)
    var d_mod_h = add_slab(g_at1.d_q_src, g_at1.d_kv_src, ctx, slab)
    var d_norm1 = mul_slab(d_mod_h, _one_plus_slab(v_scale_msa, ctx, slab), ctx, slab)
    d_hidden = add_slab(
        d_hidden,
        _rms_bwd_opt_slab(d_norm1, acts.hidden, weights, "norm1.weight", eps, ctx, slab),
        ctx, slab)

    return LTX2VideoBlockGradsDev(d_hidden^, lora_grads^)
