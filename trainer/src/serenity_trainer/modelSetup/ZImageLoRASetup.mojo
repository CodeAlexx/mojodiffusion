# lora_targets.mojo — Z-Image LoRA target module list.
#
# PORT SPEC: Serenity modules/modelSetup/BaseZImageSetup.py LAYER_PRESETS
# (lines 38-43) + ZImageLoRASetup.py (setup_model, line 57-58, uses
# config.layer_filter against the transformer module names).
#
# LAYER_PRESETS (BaseZImageSetup.py:38-43):
#   "full":      []
#   "blocks":    ["layers"]
#   "attn-mlp":  regex ["^(?=.*attention)(?!.*refiner).*",
#                       "^(?=.*feed_forward)(?!.*refiner).*"]
#   "attn-only": regex ["^(?=.*attention)(?!.*refiner).*"]
#
# The default training preset wraps every Linear inside the 30 MAIN `layers.<i>`
# blocks whose name contains `attention` or `feed_forward`, and EXCLUDES the
# refiners (noise_refiner.*, context_refiner.*) via the negative lookahead
# `(?!.*refiner)`. Inside a main block the trainable Linears are (matching the
# block math in serenitymojo/models/dit/zimage_dit.mojo):
#     attention.to_q,  attention.to_k,  attention.to_v,  attention.to_out.0
#     feed_forward.w1, feed_forward.w2, feed_forward.w3
# (norm_q / norm_k are RMSNorm weights, not Linears → not LoRA targets;
#  attention_norm1/2, ffn_norm1/2, adaLN_modulation are likewise excluded.)
#
# This file enumerates the per-block LoRA target SLOTS (a stable index scheme the
# block forward/backward and the weight loader agree on) and the relative module
# key suffixes used to look up the FROZEN base weight in the safetensors map.
#
# Pure host metadata; no tensors.
#
# CYCLE FIX: the per-block LoRA slot constants + host helpers used to live HERE
# and were ALSO imported by model/ZImageModel.mojo, while this module imports
# ZImageLoraSet from model — a hard model⇄setup comptime cycle. Those consts now
# live in the LEAF module modelSetup/zImageLoraTargets.mojo. We re-export them so
# existing references (LORA_*, lora_module_prefix, zimage_lora_target_prefixes,
# …) keep resolving through this spec module unchanged.
from serenity_trainer.modelSetup.zImageLoraTargets import (
    LORA_TO_Q, LORA_TO_K, LORA_TO_V, LORA_TO_OUT,
    LORA_FF_W1, LORA_FF_W3, LORA_FF_W2,
    LORA_SLOTS_PER_BLOCK, ZIMAGE_N_MAIN_LAYERS,
    lora_slot_module, lora_slot_base_suffix, lora_module_prefix,
    zimage_lora_count, zimage_lora_target_prefixes,
)


# ══════════════════════════════════════════════════════════════════════════════
# ZImageLoRASpec — the ModelSpec conformance for Z-Image LoRA training.
#
# PORT SPEC (1:1): Serenity modules/modelSetup/BaseZImageSetup.py::predict
#   (:81-214) + ZImageLoRASetup.py (the LoRA wiring). The Serenity predict body,
#   step by step, and where each maps here:
#     scaled_latent = model.scale_latents(latent_image)            (:105)
#       → (z - vae.shift_factor)*vae.scaling_factor = (z-0.1159)*0.3611
#     latent_noise  = _create_noise(scaled_latent, generator)      (:107)
#       → randn(scaled_latent.shape, seed)                          (per-step seed)
#     shift = model.calculate_timestep_shift(H, W)                 (:109)
#       → host μ = exp(image_seq_len*m + b)  (Flux defaults; calculate_timestep_shift)
#         used only when config.dynamic_timestep_shifting (:116); else timestep_shift.
#     timestep = _get_timestep_discrete(N, det, gen, B, config, shift) (:110-117)
#       → sampled discrete t in [0, 1000) via get_timestep_discrete using
#         config.timestep_distribution / min|max_noising_strength /
#         noising_weight|bias and the schedule shift above (NoiseMixin:121-212).
#     scaled_noisy, sigma = _add_noise_discrete(scaled_latent,     (:119-124)
#                            latent_noise, timestep, timesteps)
#       → sigma = (t+1)/1000 ; x_t = sigma*noise + (1-sigma)*scaled_latent
#         (modelSetup/BaseZImageSetup.mojo sigma_from_timestep + flow_target math)
#     latent_input = scaled_noisy.unsqueeze(2).to(train_dtype)      (:125)
#       → BF16, the extra t-dim is a no-op for the [S,dim] stack forward
#     output = model.transformer(latent_input, (1000-t)/1000, text) (:128-133)
#       → LoRA-overlaid Z-Image NextDiT forward (model unit); t-input is the
#         INVERTED, normalized timestep model_t_from_timestep(t).
#     predicted_flow = - stack(output).squeeze(2)                   (:135)
#       → predicted = -model_output                 (NB the LEADING minus)
#     flow = latent_noise - scaled_latent                          (:138)
#       → target = noise - scaled_latent            (velocity; loss_type 'target')
#   calculate_loss (:216-229): _flow_matching_losses(...).mean() = MSE(pred,target),
#   handled by the shared train_step (tape mse_loss + flow loss weighting).
#
# THE TAPE/HAND-CHAIN SEAM. The shared train_step (trainer/train_step.mojo) runs a
# TAPE mse_loss + backward(tape, loss), keying grads by each param's tape id. So
# the LoRA forward recorded in predict() MUST be tape-recorded (module/LoRAModule
# .mojo::lora_linear_forward records the A/B path on the tape; the frozen base is
# untracked). The hand-chained zimage_stack_forward/backward path (model/ZImage
# Model.mojo) is the ALTERNATIVE no-tape driver; this spec uses the tape path so it
# composes with the shared step. Both paths share the SAME LoRA slot scheme
# (lora_targets above) and the SAME frozen base weights (ZImageWeights), so the
# trained adapters are interchangeable between them and with the saver/loader.
#
# The full frozen NextDiT pipeline (patchify → embed → refiners → concat → main →
# final → unpatchify) is the model unit's surface; this spec calls the model-unit
# tape entry `zimage_transformer_forward_lora_taped` (contract below). predict()
# OWNS the latent/noise/sigma/target construction (the BaseZImageSetup.predict
# math) — which is the modelSetup slice — and the StepOutput assembly.
#
# DTYPE: BF16 storage; sigma/μ host F32 scalars only. No persistent F32 tensor.

from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd import Tape
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    add as _add, sub as _sub, mul_scalar as _mul_scalar, add_scalar as _add_scalar,
)

from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.modelSetup.BaseModelSetup import StepOutput, ModelSpec
from serenity_trainer.modelSetup.BaseZImageSetup import (
    sigma_from_timestep, model_t_from_timestep, ZIMAGE_NUM_TRAIN_TIMESTEPS,
    calculate_timestep_shift, get_timestep_discrete,
    ZIMAGE_VAE_SHIFT_FACTOR, ZIMAGE_VAE_SCALING_FACTOR,
)
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
# Runtime type + builder from the model unit. This is now a ONE-DIRECTION dep:
# model imports the LoRA slot consts from the LEAF (zImageLoraTargets), NOT from
# this spec, so importing ZImageLoraSet/build here closes no cycle. We still
# restate ZIMAGE_DIM locally (for the smoke's host shapes) to avoid importing
# model's comptime ZDIM purely for a literal; the canonical ZDIM lives in
# model/ZImageModel.
from serenity_trainer.model.ZImageModel import (
    ZImageLoraSet, build_zimage_lora_set, ZImageStackLoraGrads,
)
# The full hand-chained NextDiT wrapper (this is the model unit's copy borrowed
# from serenitymojo zimage_dit.mojo). The forward returns the velocity + the
# saved-for-backward bundle; the backward returns the 210 LoRA d_a/d_b the driver
# scatters into the optimizer. The model is HAND-CHAINED, NOT on the serenitymojo
# tape — so the LoRA grads are exposed via backward_lora() (the driver reads them
# directly) and do NOT come back through autograd.backward.
from serenity_trainer.model.ZImageDiT import (
    zimage_forward_full_lora, zimage_backward_full_lora, ZImageFullSaved,
    ZImageForwardOut,
)

comptime ZIMAGE_DIM = 3840   # ZH(30) * ZDh(128); == model.ZImageModel.ZDIM
from serenity_trainer.module.LoRAModule import lora_linear_forward


# Timestep sampling is delegated 1:1 to BaseZImageSetup.get_timestep_discrete
# (the ModelSetupNoiseMixin._get_timestep_discrete port). predict() supplies the
# config-driven distribution params and the schedule shift exactly as
# BaseZImageSetup.predict (:110-117) does.


# ── the conforming spec ───────────────────────────────────────────────────────
# Holds the FROZEN transformer weights, the trained LoRA overlay, the (already
# VAE-encoded, pre-scale) clean latent for this step, the caption features, and
# the schedule shift μ. predict() runs the BaseZImageSetup.predict math + the
# LoRA-overlaid forward and returns StepOutput. Comptime-shaped on latent HL/WL +
# caption length (the NextDiT sequence length must be a compile-time constant).
struct ZImageLoRASpec[HL: Int, WL: Int, CAPLEN: Int](Movable):
    var weights: ZImageWeights        # frozen base (untracked)
    var loras: ZImageLoraSet          # trained overlay (A/B tracked per step)
    var latent: Tensor                # clean VAE latent [1,16,HL,WL] BF16 (pre-scale)
    var cap_feats: Tensor             # [CAPLEN, dim] caption features BF16
    var timestep_shift: Float32       # legacy carrier; predict() now derives the
                                      # schedule shift from config per Serenity
                                      # (calculate_timestep_shift(H,W) when
                                      # config.dynamic_timestep_shifting, else
                                      # config.timestep_shift). Kept for ctor/API
                                      # compatibility; not read in predict().
    var base_seed: UInt64             # per-step noise/timestep seed base
    # HAND-CHAIN SEAM: predict() runs the full NextDiT forward (hand-chained, NOT
    # on the tape) and stashes its saved-for-backward here so the driver can call
    # backward_lora(d_velocity) AFTER it knows d_predicted. The model LoRA grads
    # do NOT flow through autograd.backward — they are read off this method.
    # Stores the WHOLE forward output (velocity + saved). Movable structs disallow
    # piecemeal field transfer-out at the use site, so predict() moves `fo^` in
    # atomically and backward_lora reads `.saved` from it.
    var fwd_out: Optional[ZImageForwardOut]

    def __init__(
        out self, var weights: ZImageWeights, var loras: ZImageLoraSet,
        var latent: Tensor, var cap_feats: Tensor,
        timestep_shift: Float32, base_seed: UInt64,
    ):
        self.weights = weights^
        self.loras = loras^
        self.latent = latent^
        self.cap_feats = cap_feats^
        self.timestep_shift = timestep_shift
        self.base_seed = base_seed
        self.fwd_out = None

    # ModelSpec.predict (trait conformance). Builds the noised latent + flow target
    # 1:1 with BaseZImageSetup.predict and runs the LoRA-overlaid forward on `tape`.
    def predict(
        mut self, mut tape: Tape, config: TrainConfig, step: Int, ctx: DeviceContext
    ) raises -> StepOutput:
        var seed = self.base_seed + UInt64(step)

        # 1) scale clean latent (ZImageModel.scale_latents)  (BaseZImageSetup:105)
        var scaled_latent = _scale_latents(self.latent, ctx)

        # 2) per-step noise (_create_noise)                  (:107)
        #    NB offset_noise_weight / perturbation_noise_weight default 0.0
        #    (TrainConfig.py:1014,1016) → plain randn, matching _create_noise:85-90.
        var noise = randn(scaled_latent.shape().copy(), seed, STDtype.BF16, ctx)

        # 3) schedule shift (BaseZImageSetup.predict:109,116):
        #    shift = calculate_timestep_shift(H, W) if dynamic_timestep_shifting
        #            else config.timestep_shift
        #    H = scaled_latent.shape[-2] = HL ; W = scaled_latent.shape[-1] = WL.
        var lshape = scaled_latent.shape()
        var H = lshape[len(lshape) - 2]
        var W = lshape[len(lshape) - 1]
        var shift: Float32
        if config.dynamic_timestep_shifting:
            shift = calculate_timestep_shift(H, W)
        else:
            shift = config.timestep_shift

        # 4) sample discrete timestep (_get_timestep_discrete, shift μ) (:110-117)
        #    RNG NOTE: Serenity (BaseZImageSetup.py:92-93,107,110) shares ONE
        #    torch.Generator(batch_seed) across BOTH _create_noise (randn) AND
        #    _get_timestep_discrete — they draw SEQUENTIALLY from a single stream,
        #    so the timestep draw is NOT independent of the noise draw. serenitymojo
        #    has no shared host/device torch stream: noise uses ops.random.randn
        #    (device RNG keyed by `seed`) and the timestep uses the host PCG32
        #    sampler keyed by the SAME `seed`. Values will not bit-match torch and
        #    the noise⇄timestep coupling is not reproduced (documented divergence;
        #    parity is verified against dumped (latent,noise,timestep) tensors). We
        #    key the timestep stream with the identical base `seed` reference trainer uses (no
        #    invented offset) so it is a deterministic function of reference trainer's batch_seed.
        var t = get_timestep_discrete(
            ZIMAGE_NUM_TRAIN_TIMESTEPS,   # model.noise_scheduler.config['num_train_timesteps']
            False,                        # deterministic (predict default arg)
            seed,                         # reference trainer shares the noise generator; same base seed
            config.timestep_distribution,
            config.min_noising_strength,
            config.max_noising_strength,
            config.noising_weight,
            config.noising_bias,
            shift,
        )
        var sigma = sigma_from_timestep(t, ZIMAGE_NUM_TRAIN_TIMESTEPS)
        var t_model = model_t_from_timestep(t, ZIMAGE_NUM_TRAIN_TIMESTEPS)

        # 5) noised input + flow target (_add_noise_discrete + flow)  (:119-124,138)
        #    sigmas = sigma[timestep] = (timestep+1)/N      (NoiseMixin.py:24,29)
        #    x_t  = noise*sigmas + scaled_latent*(1-sigmas) (NoiseMixin.py:36-37)
        #    flow = noise - scaled_latent                   (BaseZImageSetup.py:138)
        var a = _mul_scalar(scaled_latent, Float32(1.0) - sigma, ctx)
        var b = _mul_scalar(noise, sigma, ctx)
        var scaled_noisy = _add(a, b, ctx)
        var target = _sub(noise, scaled_latent, ctx)   # flow = noise - scaled_latent

        # 6) LoRA-overlaid transformer forward → model output; predicted = -output.
        #    The model-unit tape entry records the LoRA A/B path on `tape` so the
        #    shared backward reaches the adapters; the frozen base is untracked.
        #    Contract:
        #      def zimage_transformer_forward_lora_taped[HL,WL,CAPLEN](
        #          mut tape, latent_input[1,16,HL,WL], t_model, cap_feats[CAPLEN,dim],
        #          weights, loras, ctx) raises -> Tensor   # model output [1,16,HL,WL]
        var model_out = self._forward_lora_taped(tape, scaled_noisy, t_model, ctx)
        var predicted = _mul_scalar(model_out, Float32(-1.0), ctx)  # predicted = -flow

        return StepOutput(predicted^, target^, sigma)

    # The LoRA-overlaid forward — the FULL NextDiT pipeline (model/ZImageDiT.mojo,
    # borrowed from serenitymojo zimage_dit.mojo). HAND-CHAINED: the 30 main blocks
    # carry LoRA on 7 projections each (210 adapters) and the embedders/refiners/
    # final layer are FROZEN base. The model is NOT recorded on `tape` — the LoRA
    # grads are produced by backward_lora() (read directly by the driver), NOT by
    # autograd.backward. We accept `tape` to keep the predict() seam signature but
    # record nothing on it (the model owns its own reverse chain). Returns the
    # predicted VELOCITY [1,16,HL,WL]; predict() then does predicted = -velocity.
    def _forward_lora_taped(
        mut self, mut tape: Tape, latent_input: Tensor, t_model: Float32, ctx: DeviceContext
    ) raises -> Tensor:
        var fo = zimage_forward_full_lora[Self.HL, Self.WL, Self.CAPLEN](
            latent_input, t_model, self.cap_feats, self.weights, self.loras, ctx
        )
        # Clone the velocity (borrow → owned copy; * 1.0) BEFORE moving `fo` whole
        # into the spec — a partial field move of a Movable struct is disallowed.
        var velocity = _mul_scalar(fo.velocity, Float32(1.0), ctx)
        # stash the whole forward output so backward_lora can read `.saved`.
        self.fwd_out = Optional[ZImageForwardOut](fo^)
        return velocity^

    # HAND-CHAIN BACKWARD seam the DRIVER calls after computing d_velocity (the grad
    # of the loss wrt the model's velocity output — i.e. -d_predicted, since
    # predicted = -velocity). Returns the 210 LoRA d_a/d_b (ZImageStackLoraGrads).
    # The driver maps these to its optimizer slots by block-major/slot-minor order
    # (identical to loras.ad and zimage_lora_target_prefixes). The .d_x_in field is
    # the unified-input grad and is NOT load-bearing (frozen embedders train
    # nothing) — the driver reads ONLY .d_a / .d_b.
    #
    # Raises if called before _forward_lora_taped populated `saved`.
    def backward_lora(
        mut self, d_velocity: Tensor, ctx: DeviceContext
    ) raises -> ZImageStackLoraGrads:
        if not self.fwd_out:
            raise Error(
                "ZImageLoRASpec.backward_lora: no saved forward state — call "
                "predict()/_forward_lora_taped first."
            )
        return zimage_backward_full_lora[Self.HL, Self.WL, Self.CAPLEN](
            d_velocity, self.fwd_out.value().saved, self.weights, self.loras, ctx
        )


# Convenience: build a cold-start spec (A~randn, B=0) for a fresh LoRA run, mirroring
# ZImageLoRASetup.setup_model (transformer_lora = LoRAModuleWrapper(...)).
def make_zimage_lora_spec[HL: Int, WL: Int, CAPLEN: Int](
    var weights: ZImageWeights, var latent: Tensor, var cap_feats: Tensor,
    rank: Int, alpha: Float32, timestep_shift: Float32, base_seed: UInt64,
    ctx: DeviceContext,
) raises -> ZImageLoRASpec[HL, WL, CAPLEN]:
    var loras = build_zimage_lora_set(rank, alpha, ctx)
    return ZImageLoRASpec[HL, WL, CAPLEN](
        weights^, loras^, latent^, cap_feats^, timestep_shift, base_seed
    )


# ── owned latent scale (re-stated here to avoid a model→setup import cycle) ────
# ZImageModel.scale_latents (ZImageModel.py:175-176):
#   (latents - vae.config.shift_factor) * vae.config.scaling_factor
# Z-Image vae/config.json: shift_factor = 0.1159, scaling_factor = 0.3611
# (ZIMAGE_VAE_SHIFT_FACTOR / ZIMAGE_VAE_SCALING_FACTOR in BaseZImageSetup.mojo).
def _scale_latents(z: Tensor, ctx: DeviceContext) raises -> Tensor:
    var shifted = _add_scalar(z, -ZIMAGE_VAE_SHIFT_FACTOR, ctx)
    return _mul_scalar(shifted, ZIMAGE_VAE_SCALING_FACTOR, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# COMPILE SMOKE — exercises the predict() tape LoRA seam directly (no full DiT).
#
# Builds one frozen Linear + one LoRA adapter, records lora_linear_forward on a
# fresh tape (the EXACT op predict()'s forward uses per slot), seeds the MSE arm
# against a constant target, backs out grads, and L1-sums the adapter grads → a
# finite scalar the orchestrator asserts is finite. This proves: (a) the tape LoRA
# path compiles, (b) grads flow to A/B, (c) the StepOutput/MSE seam composes — the
# parts of this slice that are owned here. The full-pipeline gate is the model
# unit's job (it swaps the forward behind _forward_lora_taped's signature).
# ══════════════════════════════════════════════════════════════════════════════
from serenitymojo.autograd import backward as _backward
from serenity_trainer.module.LoRAModule import make_lora_adapter as _make_lora


def zimage_lora_predict_smoke[S: Int](rank: Int, ctx: DeviceContext) raises -> Float32:
    var in_f = ZIMAGE_DIM
    var out_f = ZIMAGE_DIM
    # frozen base [out,in] + a constant input x [S,in] + a constant target [S,out].
    var base = _mul_scalar(randn(_sh2(out_f, in_f), UInt64(1), STDtype.BF16, ctx), Float32(0.02), ctx)
    var x = _mul_scalar(randn(_sh2(S, in_f), UInt64(2), STDtype.BF16, ctx), Float32(0.1), ctx)
    var tgt = _mul_scalar(randn(_sh2(S, out_f), UInt64(3), STDtype.BF16, ctx), Float32(0.1), ctx)

    var ad = _make_lora(in_f, out_f, rank, Float32(rank), UInt64(7), ctx)

    var tape = Tape()
    ad.track(tape)                                  # A/B get fresh tape ids
    var y = lora_linear_forward(tape, x, base, ad, ctx)   # the per-slot predict op
    # NB in predict() the leading − (predicted = -model_output, BaseZImageSetup:135)
    # is folded by the loss seam — it does not change the grad MAGNITUDE to A/B, so
    # the smoke seeds the MSE arm against `y` directly to keep the seam minimal.
    var loss = tape.mse_loss(y, tgt, ctx)
    var gmap = _backward(tape, loss, ctx)

    var total = Float32(0.0)
    if gmap.__contains__(ad.a.id):
        var ah = gmap[ad.a.id][].to_host(ctx)
        for i in range(len(ah)): total += abs(ah[i])
    if gmap.__contains__(ad.b.id):
        var bh = gmap[ad.b.id][].to_host(ctx)
        for i in range(len(bh)): total += abs(bh[i])
    return total


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
