# KleinModel.mojo — the FULL Klein (FLUX.2 Klein DiT) transformer wrapper
# (hand-chained training fwd+bwd AND a separate no-grad inference forward) that
# Serenity's modules/model/Flux2Model.py drives via model.transformer(...).
#
# KLEIN == Serenity's FLUX_2 family (modules/model/Flux2Model.py: is_klein() =
# not is_dev() = transformer.config.num_attention_heads != 48; Klein 9B = 32
# heads). Confirmed by BaseFlux2Setup.py + Flux2ModelLoader.py (Qwen3 text encoder
# branch) + Flux2Sampler.py (Flux2KleinPipeline).
#
# BORROWED (structure + math, copied INTO this namespace, NOT a serenitymojo
# import) FROM:
#   serenitymojo/models/klein/{double_block,single_block,klein_stack,
#   klein_stack_lora,lora_block,weights}.mojo  -> copied to
#   serenity_trainer/model/klein/* (cross-imports rewritten; foundation
#   serenitymojo/{tensor,io,ops,scratch_ring} imported unchanged).
#   serenitymojo/models/dit/klein_dit.mojo::build_klein_rope_tables -> vendored
#   below (build_klein_rope_tables_port) with the reference trainer/diffusers position scheme.
#
# Klein 9B (model/klein/config: klein9b.json):
#   inner_dim D=4096, in_channels=128, out_channels=128, joint_attention_dim=12288
#   num_double=8, num_single=24, num_heads H=32, head_dim Dh=128,
#   mlp_hidden F=12288, timestep_dim=256, rope_theta=2000.
#   -> 8*12 + 24*2 = 144 LoRA adapters, 1:1 with Serenity's SEPARATE nn.Linear
#      wrapping (Flux2LoRASetup.py:57). Double: per img/txt stream q,k,v,out,ff_in,
#      ff_out (transformer_flux2.py:526-544,314-316). Single: qkv/out.
#
# Serenity CONVENTION (modules/modelSetup/BaseFlux2Setup.py::predict, the SOLE
# spec for input/output/timestep/sigma):
#   - patchify+scale latent (Flux2Model.patchify_latents/scale_latents), add noise
#     via _add_noise_discrete -> scaled_noisy_latent_image, sigma;
#   - timestep INTO the transformer is `timestep / 1000` (BaseFlux2Setup.py:144);
#     so the value feeding the timestep MLP (diffusers Timesteps num_channels=256,
#     downscale_freq_shift=0) is timestep/1000 ∈ [0,1] ≈ sigma. We build the
#     shared modulation vectors from THIS value (NOT serenitymojo's sigma*1000).
#   - guidance: only if transformer.config.guidance_embeds (BaseFlux2Setup.py:132),
#     a runtime checkpoint value. The flagship FLUX.2-klein-base-9B is
#     guidance_embeds=FALSE (verified vs transformer/config.json) ⇒ guidance=None
#     and this branch is a no-op. Only guidance-distilled variants set it; there the
#     guidance branch is supplied by the caller as a separate add into vec (the
#     time_guidance_embed; transformer_flux2.py:1004-1014). Modeled by the caller
#     passing the already-summed modvecs; KleinModel consumes the modvecs.
#   - model OUTPUT = packed predicted FLOW; predict() unpacks/unpatchifies it and
#     sets target = latent_noise - scaled_latent_image (BaseFlux2Setup.py:142-166).
#     KleinModel returns the raw [N_IMG, out_ch] flow; the reference trainer predict() port does
#     the unpack/unpatchify/target arithmetic.
#
# TRAINING vs INFERENCE (the Z-Image lesson — DISTINCT code paths):
#   - klein_training_forward: device-resident, BUILDS a KleinStackForward tape
#     (per-block inputs + tail saved acts) for the hand-chained backward.
#   - klein_backward: consumes that tape, returns per-adapter d_A/d_B (KleinLoraGrads).
#   - klein_inference_forward: NO tape, NO recompute checkpoints, NO per-step host
#     syncs — uses the predict_* resident-scratch block variants. The sampler
#     (Flux2Sampler.py under @torch.no_grad()) uses THIS, never the training fwd.
#
# DTYPE: BF16 storage / F32 compute (foundation kernels accumulate F32). Comptime
# H/Dh/N_IMG/N_TXT/S so the unified sequence length is compile-time for sdpa.

from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog
from std.collections import List, Optional
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator

from serenity_trainer.model.klein.double_block import (
    DoubleBlockWeights, ModVecs, ModVecsDevice, modvecs_to_device,
)
from serenity_trainer.model.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleModVecsDevice, single_modvecs_to_device,
)
from serenity_trainer.model.klein.klein_stack import KleinStackBase, KleinStackForward
from serenity_trainer.model.klein.lora_adapter import LoraAdapter, LoraGrads, _lora_adamw
from serenity_trainer.model.klein.klein_stack_lora import (
    KleinLoraSet, KleinLoraDeviceSet, KleinLoraGrads,
    DBL_SLOTS, SGL_SLOTS, BK_DOUBLE, BK_SINGLE,
    make_lora_adapter, build_klein_lora_set, klein_lora_set_to_device, klein_lora_get,
    klein_lora_adamw_step,
    klein_stack_lora_forward_device_inputs_resident_moddev_rope,
    klein_stack_lora_backward_resident_moddev_rope,
    klein_stack_lora_predict_resident_moddev_rope_scratch,
)


comptime TArc = ArcPointer[Tensor]


# ── Klein 9B comptime dims (mirror model/klein/config klein9b.json) ───────────
comptime KH = 32                       # num_heads
comptime KDh = 128                     # head_dim
comptime KDIM = 4096                   # inner_dim = KH * KDh
comptime KF = 12288                    # mlp_hidden
comptime KIN_CH = 128                  # in_channels (patchified latent ch)
comptime KOUT_CH = 128                 # out_channels
comptime KTXT_CH = 12288               # joint_attention_dim (txt feature dim)
comptime KNUM_DOUBLE = 8               # num_double
comptime KNUM_SINGLE = 24              # num_single
comptime KTIMESTEP_DIM = 256           # timestep_dim
comptime KROPE_THETA = Float32(2000.0) # rope_theta
comptime KEPS = Float32(1e-6)          # rms/ln eps (Klein head-norm eps)
comptime KN_ADAPTERS = KNUM_DOUBLE * DBL_SLOTS + KNUM_SINGLE * SGL_SLOTS  # 8*12 + 24*2 = 144


# ── RoPE tables (BORROWED from serenitymojo/models/dit/klein_dit.mojo:555-605) ─
# Position scheme = Serenity Flux2Model.prepare_latent_image_ids /
# prepare_text_ids (modules/model/Flux2Model.py:240-294): 4 axes (t,h,w,l).
# img token (idx = tok - N_TXT): axis1 = idx // img_w (row), axis2 = idx % img_w
# (col); text token: axis3 = tok (the L-axis). Each axis contributes 16 of the
# 64 rotary freq pairs (Dh/2 = 64 = 4 axes * 16); inv_freq = theta^(-2i/32).
# Real cached Flux2/Klein training batches are not guaranteed square, so the
# Serenity-faithful builder takes patchified image height and width.
def build_klein_rope_tables_hw_port[
    IMG_H: Int, IMG_W: Int, N_TXT: Int, N_HEADS: Int, Dh: Int
](ctx: DeviceContext, dtype: STDtype) raises -> Tuple[Tensor, Tensor]:
    comptime assert Dh == 128, "Klein head dim must be 128"
    comptime assert IMG_H > 0 and IMG_W > 0, "image grid must be positive"
    comptime N_IMG = IMG_H * IMG_W
    comptime S = N_IMG + N_TXT
    comptime HALF = Dh // 2
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(KROPE_THETA)
    for tok in range(S):
        var p0 = 0
        var p1 = 0
        var p2 = 0
        var p3 = 0
        if tok >= N_TXT:
            var idx = tok - N_TXT
            p1 = idx // IMG_W
            p2 = idx % IMG_W
        else:
            p3 = tok
        for _h in range(N_HEADS):
            for axis in range(4):
                var pos = p0
                if axis == 1:
                    pos = p1
                elif axis == 2:
                    pos = p2
                elif axis == 3:
                    pos = p3
                for i in range(16):
                    var inv_freq = fexp(-log_theta * Float32(2 * i) / Float32(32))
                    var angle = Float32(pos) * inv_freq
                    cos_vals.append(fcos(angle))
                    sin_vals.append(fsin(angle))
    var sh = List[Int]()
    sh.append(S * N_HEADS)
    sh.append(HALF)
    return (
        Tensor.from_host(cos_vals, sh.copy(), dtype, ctx),
        Tensor.from_host(sin_vals, sh^, dtype, ctx),
    )


# Backward-compatible square-grid helper used by older smoke gates.
def build_klein_rope_tables_port[
    N_IMG: Int, N_TXT: Int, N_HEADS: Int, Dh: Int
](ctx: DeviceContext, dtype: STDtype) raises -> Tuple[Tensor, Tensor]:
    comptime assert Dh == 128, "Klein head dim must be 128"
    comptime S = N_IMG + N_TXT
    comptime HALF = Dh // 2
    var img_w = 1
    while img_w * img_w < N_IMG:
        img_w += 1
    if img_w * img_w != N_IMG:
        raise Error("build_klein_rope_tables_port: N_IMG must be a square grid")
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(KROPE_THETA)
    for tok in range(S):
        var p0 = 0
        var p1 = 0
        var p2 = 0
        var p3 = 0
        if tok >= N_TXT:
            var idx = tok - N_TXT
            p1 = idx // img_w
            p2 = idx % img_w
        else:
            p3 = tok
        for _h in range(N_HEADS):
            for axis in range(4):
                var pos = p0
                if axis == 1:
                    pos = p1
                elif axis == 2:
                    pos = p2
                elif axis == 3:
                    pos = p3
                for i in range(16):
                    var inv_freq = fexp(-log_theta * Float32(2 * i) / Float32(32))
                    var angle = Float32(pos) * inv_freq
                    cos_vals.append(fcos(angle))
                    sin_vals.append(fsin(angle))
    var sh = List[Int]()
    sh.append(S * N_HEADS)
    sh.append(HALF)
    return (
        Tensor.from_host(cos_vals, sh.copy(), dtype, ctx),
        Tensor.from_host(sin_vals, sh^, dtype, ctx),
    )


# ── the trained-adapter set: 144 Klein 9B LoRA adapters ───────────────────────
# 8 double × 12 (q,k,v,out,ff_in,ff_out per img/txt stream) + 24 single × 2 = 144,
# matching Serenity's SEPARATE nn.Linear wrapping (Flux2LoRASetup.py:57).
# Convenience wrapper over build_klein_lora_set with Klein 9B dims fixed. rank /
# alpha come from the TrainConfig (klein9b.json: lora_rank=16, lora_alpha=16).
def build_klein9b_lora_set(rank: Int, alpha: Float32) -> KleinLoraSet:
    return build_klein_lora_set(KNUM_DOUBLE, KNUM_SINGLE, KDIM, KF, rank, alpha)


# ── TRAINING forward (builds the backward tape) ──────────────────────────────
# img_tokens_t: [N_IMG, KIN_CH]  patchified+scaled+noised latent, packed
#               (BaseFlux2Setup.py pack_latents -> [N_IMG, in_ch]).
# txt_tokens_t: [N_TXT, KTXT_CH] the concatenated Qwen3 hidden states
#               (encode_text, QWEN3_HIDDEN_STATES_LAYERS = [9,18,27]).
# modvecs are pre-built from timestep/1000 (see model/klein/weights.mojo
# build_klein_*_modvecs driven by build_klein_vec_silu(timestep/1000)).
# Returns the KleinStackForward tape; .out is the flow prediction [N_IMG,KOUT_CH].
def klein_training_forward[
    N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    ctx: DeviceContext,
) raises -> KleinStackForward:
    return klein_stack_lora_forward_device_inputs_resident_moddev_rope[
        KH, KDh, N_IMG, N_TXT, S
    ](
        img_tokens_t, txt_tokens_t, base, dbw, sbw, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t,
        KDIM, KF, KIN_CH, KTXT_CH, KOUT_CH, KEPS, ctx,
    )


# ── HAND-CHAINED backward -> per-adapter d_A/d_B ─────────────────────────────
# d_out: dL/d(flow prediction) [N_IMG, KOUT_CH] (from the flow-matching loss; reference trainer
# target = latent_noise - scaled_latent_image, BaseFlux2Setup.py:159). Returns
# KleinLoraGrads (flat dbl/sgl d_A/d_B parallel to KleinLoraSet) consumed by
# klein_lora_adamw_step. Base weights are FROZEN (LoRA training).
def klein_backward[
    N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    saved: KleinStackForward,
    ctx: DeviceContext,
    compute_input_grads: Bool = False,
    compute_aux_grads: Bool = False,
) raises -> KleinLoraGrads:
    return klein_stack_lora_backward_resident_moddev_rope[
        KH, KDh, N_IMG, N_TXT, S
    ](
        d_out, img_tokens, txt_tokens, base, dbw, sbw, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t, saved,
        KDIM, KF, KIN_CH, KTXT_CH, KOUT_CH, KEPS, ctx,
        compute_input_grads, compute_aux_grads,
    )


# ── INFERENCE forward (NO tape; the sampler path) ────────────────────────────
# Flux2Sampler.__sample_base runs the transformer under @torch.no_grad(): no saved
# activations, no recompute checkpoints. Returns ONLY the flow prediction
# [N_IMG, KOUT_CH]; sampler then unpacks/steps the FlowMatchEulerDiscreteScheduler.
# Caller supplies a ScratchRingAllocator (ring-reused across denoise steps).
def klein_inference_forward[
    N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> List[Float32]:
    return klein_stack_lora_predict_resident_moddev_rope_scratch[
        KH, KDh, N_IMG, N_TXT, S
    ](
        img_tokens_t, txt_tokens_t, base, dbw, sbw, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t,
        KDIM, KF, KIN_CH, KTXT_CH, KOUT_CH, KEPS, ctx, scratch,
    )
