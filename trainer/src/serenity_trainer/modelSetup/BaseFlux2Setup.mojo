# BaseFlux2Setup.mojo — Klein (FLUX.2) flow-matching σ↔timestep map, the dynamic
# timestep shift, the VAE batch-norm latent scaling, and the discrete timestep
# sampler. Pure host scalar math + the BORROWED multinomial sampler.
#
# ── PORT SPEC (1:1) ───────────────────────────────────────────────────────────
#   * modules/modelSetup/BaseFlux2Setup.py::predict (:82-179)
#   * modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py::_add_noise_discrete
#       (:14-39)
#   * modules/modelSetup/mixin/ModelSetupNoiseMixin.py::_get_timestep_discrete
#       (:121-212)
#   * modules/model/Flux2Model.py:
#       calculate_timestep_shift (:267-278), scale_latents (:313-318),
#       unscale_latents (:321-326), pack/unpack/patchify/unpatchify (:253-310).
#
# ── _add_noise_discrete (FlowMatchingMixin.py:14-39) ──────────────────────────
#   num_timesteps   = timesteps.shape[-1]                         # = 1000  (:22)
#   all_timesteps   = arange(start=1, end=num_timesteps+1)        # [1..1000](:23)
#   sigma           = all_timesteps / num_timesteps               # (i+1)/N (:24)
#   one_minus_sigma = 1.0 - sigma                                          (:25)
#   sigmas          = sigma[timestep]                                      (:29)
#   x_t = noise*sigmas + scaled_latent*(1-sigmas)                         (:36-37)
# ⇒ for discrete index t (0-based): sigma(t) = (t+1)/num_timesteps.
#
# ── transformer t-input (BaseFlux2Setup.py:144) ──────────────────────────────
#   timestep = timestep / 1000     (the model's normalized t-input; NB Flux2 does
#                                   NOT invert t — UNLIKE Z-Image's (1000-t)/1000)
#
# ── velocity target (BaseFlux2Setup.py:159) ──────────────────────────────────
#   flow = latent_noise - scaled_latent_image
#
# ── guidance (BaseFlux2Setup.py:132-136) ─────────────────────────────────────
#   if model.transformer.config.guidance_embeds:                       (:132)
#       guidance = tensor([config.transformer.guidance_scale]).expand(B)
#   else: guidance = None                                              (:135-136)
# Serenity gates the guidance branch on the LOADED CHECKPOINT'S config value
# `model.transformer.config.guidance_embeds` (:132) — a runtime value, NOT a
# compile-time constant. The flagship FLUX.2-klein-base-9B checkpoint carries
# guidance_embeds=FALSE (verified vs transformer/config.json), so for that
# checkpoint the ELSE branch is taken and guidance = None. Only guidance-distilled
# variants (guidance_embeds=True, with guidance_in.* keys present) take the
# guidance branch, where guidance = config.transformer.guidance_scale and the
# transformer multiplies guidance ×1000 internally (transformer_flux2.py:1234)
# before the guidance_embedder, exactly as it multiplies timestep ×1000 (:1231).
# The Mojo port THREADS this checkpoint value as a runtime `guidance_embeds: Bool`
# (the default below is the klein-base-9B value FALSE; the real source of truth at
# transformer build time is weights.has_guidance(), i.e. whether guidance_in.*
# keys exist — the structural equivalent of config.guidance_embeds).
#
# DTYPE: BF16 tensor storage, F32 host scalars for σ / μ / sampling. No persistent F32.

from std.math import exp
from std.collections import Optional

# The aten-exact multinomial timestep sampler lives ONCE in the mixin port (the
# 1:1 ModelSetupNoiseMixin file). BaseFlux2Setup DELEGATES so the multinomial tie
# semantics, the Float64 config dtype, and all six distribution weight tables are
# sourced from the single verified implementation (same contract as
# BaseZImageSetup.mojo). See modelSetup/mixin/ModelSetupNoiseMixin.mojo.
from serenity_trainer.modelSetup.mixin.ModelSetupNoiseMixin import (
    _get_timestep_discrete_host as _ts_host,
)


# Klein noise scheduler: FlowMatchEulerDiscreteScheduler, num_train_timesteps=1000
# (diffusers default; Flux2 scheduler/scheduler_config.json carries num_train_timesteps
# = 1000). Used by _get_timestep_discrete (config['num_train_timesteps'], :116) and
# _add_noise_discrete (timesteps.shape[-1], FlowMatchingMixin.py:22).
comptime FLUX2_NUM_TRAIN_TIMESTEPS = 1000

# FlowMatchEulerDiscreteScheduler config defaults used by calculate_timestep_shift
# (Flux2Model.py:268-272). The Flux2 scheduler_config.json inherits the diffusers
# FlowMatchEulerDiscreteScheduler defaults (verified vs diffusers __init__):
#   base_image_seq_len = 256, max_image_seq_len = 4096,
#   base_shift = 0.5, max_shift = 1.15.
# patch_size = 2 is hardcoded in calculate_timestep_shift (Flux2Model.py:272).
comptime FLUX2_BASE_IMAGE_SEQ_LEN = 256
comptime FLUX2_MAX_IMAGE_SEQ_LEN  = 4096
comptime FLUX2_BASE_SHIFT         = Float32(0.5)
comptime FLUX2_MAX_SHIFT          = Float32(1.15)
comptime FLUX2_PATCH_SIZE         = 2

# AutoencoderKLFlux2: scaling is a BATCH-NORM, not a (shift,scale) pair like other
# VAEs (Flux2Model.scale_latents :313-318):
#   mean = vae.bn.running_mean ; std = sqrt(vae.bn.running_var + batch_norm_eps)
#   scale_latents(z)   = (z - mean) / std
#   unscale_latents(z) = z * std + mean
# These running_mean/running_var are CHECKPOINT params (per-channel, 128 channels)
# loaded by the model loader (Flux2ModelLoader), NOT compile-time constants.
# batch_norm_eps is the diffusers config default and equals 1e-4
# (autoencoder_kl_flux2.py:104 `batch_norm_eps: float = 1e-4`; consumed by
# Flux2Model.scale_latents via vae.config.batch_norm_eps). Must match
# KleinVAE.KLEIN_BN_EPS (1e-4).
comptime FLUX2_BATCH_NORM_EPS = Float32(1e-4)

# Default `guidance_embeds` for Klein checkpoints. The flagship
# FLUX.2-klein-base-9B carries guidance_embeds=FALSE (transformer/config.json),
# so the DEFAULT here is False — matching that checkpoint and Serenity's
# else-branch (BaseFlux2Setup.py:135-136 ⇒ guidance=None). This is only a default;
# Serenity reads the actual value from model.transformer.config.guidance_embeds
# (:132) at runtime, and the Mojo predict/sampler/LoRA paths take `guidance_embeds`
# as an overridable parameter threaded from the loaded checkpoint
# (weights.has_guidance() is the structural source of truth at transformer build).
# A guidance-distilled variant (guidance_in.* keys present) overrides this to True.
comptime FLUX2_GUIDANCE_EMBEDS = False


# The INTEGER guidance value the diffusers guidance_embedder sees, given the config
# guidance scale.  BaseFlux2Setup.predict (:133): guidance = tensor([guidance_scale]).
# The transformer multiplies guidance ×1000 internally (transformer_flux2.py:1234)
# before time_proj/guidance_embedder, so the embedder input is guidance_scale*1000.
# Mirrors model_t_from_timestep's integer-domain convention (the t_embedder sees the
# integer timestep t after the ×1000 re-scale).  `guidance_embeds` is the runtime
# checkpoint value (BaseFlux2Setup.py:132 model.transformer.config.guidance_embeds),
# threaded by the caller; it defaults to FLUX2_GUIDANCE_EMBEDS (False = klein-base-9B).
# Returns None when guidance_embeds=False ⇒ guidance=None (BaseFlux2Setup.py:135-136).
def guidance_embedder_value(
    guidance_scale: Float32, guidance_embeds: Bool = FLUX2_GUIDANCE_EMBEDS
) -> Optional[Float32]:
    if not guidance_embeds:
        return Optional[Float32](None)
    return Optional[Float32](guidance_scale * Float32(1000.0))


# σ for a discrete timestep index `t` in [0, num_timesteps).
#   sigma(t) = (t + 1) / num_timesteps          (FlowMatchingMixin.py:24)
def sigma_from_timestep(t: Int, num_timesteps: Int = FLUX2_NUM_TRAIN_TIMESTEPS) -> Float32:
    return Float32(t + 1) / Float32(num_timesteps)


# The transformer's normalized t-input for a discrete timestep index `t`.
#   timestep / 1000                              (BaseFlux2Setup.py:144)
# NB: Flux2 does NOT invert the timestep (Z-Image used (1000-t)/1000; Flux2 uses
# t/1000 directly). The discrete index t corresponds to the integer timestep value
# returned by _get_timestep_discrete (NoiseMixin.py:212 returns .int()), and the
# model input is that integer / 1000.
def model_t_from_timestep(t: Int, num_timesteps: Int = FLUX2_NUM_TRAIN_TIMESTEPS) -> Float32:
    return Float32(t) / Float32(num_timesteps)


# Inverse: discrete timestep index from a σ in (0, 1].  t = round(σ*N) - 1.
def timestep_from_sigma(sigma: Float32, num_timesteps: Int = FLUX2_NUM_TRAIN_TIMESTEPS) -> Int:
    var idx = Int(sigma * Float32(num_timesteps) + Float32(0.5)) - 1
    if idx < 0:
        return 0
    if idx >= num_timesteps:
        return num_timesteps - 1
    return idx


# SNR for flow-matching σ (loss-weight LossWeight.SIGMA path uses σ directly;
# min-SNR would use ((1-σ)/σ)²). For x_t = (1-σ)·x0 + σ·ε, SNR = ((1-σ)/σ)².
def snr_from_sigma(sigma: Float32) -> Float32:
    var s = sigma
    if s < Float32(1e-8):
        s = Float32(1e-8)
    var ratio = (Float32(1.0) - s) / s
    return ratio * ratio


# ── Flux2Model.calculate_timestep_shift (Flux2Model.py:267-278) ──────────────
# Python signature calculate_timestep_shift(self, latent_height, latent_width);
# BaseFlux2Setup.predict (:114) calls calculate_timestep_shift(latent_height,
# latent_width) where latent_height = latent_image.shape[-2], latent_width =
# latent_image.shape[-1] (:108-109). image_seq_len uses the PRODUCT
# (W//patch)*(H//patch), so argument order is irrelevant to the result.
#   patch_size    = 2                                            (:272)
#   image_seq_len = (latent_width//patch) * (latent_height//patch)(:274)
#   m  = (max_shift - base_shift) / (max_seq_len - base_seq_len) (:275)
#   b  = base_shift - m*base_seq_len                            (:276)
#   mu = image_seq_len*m + b                                    (:277)
#   return math.exp(mu)                                         (:278)
# NB the latent here is the PATCHIFIED latent (predict :107 patchify_latents),
# whose H/W are already halved vs the raw VAE latent. So patch_size=2 inside
# calculate_timestep_shift halves AGAIN: image_seq_len = (H_raw/2/2)*(W_raw/2/2).
def calculate_timestep_shift(latent_h: Int, latent_w: Int) -> Float32:
    var base_seq_len = Float32(FLUX2_BASE_IMAGE_SEQ_LEN)
    var max_seq_len  = Float32(FLUX2_MAX_IMAGE_SEQ_LEN)
    var base_shift   = FLUX2_BASE_SHIFT
    var max_shift    = FLUX2_MAX_SHIFT
    var patch_size   = FLUX2_PATCH_SIZE

    var image_seq_len = Float32((latent_w // patch_size) * (latent_h // patch_size))
    var m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    var b = base_shift - m * base_seq_len
    var mu = image_seq_len * m + b
    return exp(mu)


# ── ModelSetupNoiseMixin._get_timestep_discrete (NoiseMixin.py:121-212) ──────
# All six distributions ported 1:1 in the mixin host port; BaseFlux2Setup delegates.
#   if shift is None: shift = config.timestep_shift                       (:130-131)
#   deterministic: return int(N*0.5) - 1                                  (:133-139)
#   else:
#     min_timestep = int(N * config.min_noising_strength)                 (:141)
#     max_timestep = int(N * config.max_noising_strength)                 (:142)
#     num_timestep = max_timestep - min_timestep                         (:143)
#     UNIFORM/LOGIT_NORMAL/HEAVY_TAIL (continuous, :151-172) then SHIFT
#       t = N*shift*t / ((shift-1)*t + N)                                 (:172)
#     SIGMOID/COS_MAP/INVERTED_PARABOLA (discrete multinomial, :173-210)
#     return timestep.int()                                              (:212)
#
# predict() runs B=1 latents (batch['latent_image'].shape[0]; Flux2 predict :99),
# so we return the single sampled index. `seed` selects the RNG stream for this
# step (Serenity shares ONE torch.Generator(batch_seed) for noise+timestep;
# documented RNG-stream divergence — same as BaseZImageSetup.mojo).
def get_timestep_discrete(
    num_train_timesteps: Int,
    deterministic: Bool,
    seed: UInt64,
    timestep_distribution: Int,
    min_noising_strength: Float32,
    max_noising_strength: Float32,
    noising_weight: Float32,
    noising_bias: Float32,
    shift: Float32,
) raises -> Int:
    if deterministic:
        # int(num_train_timesteps * 0.5) - 1   (NoiseMixin.py:133-139)
        return Int(Float64(num_train_timesteps) * Float64(0.5)) - 1

    var host = _ts_host(
        num_train_timesteps,
        1,                                  # batch_size (B=1)
        timestep_distribution,
        Float64(min_noising_strength),
        Float64(max_noising_strength),
        Float64(noising_weight),
        Float64(noising_bias),
        Float64(shift),
        List[Float64](),                    # empty weights_cache → recompute
        seed,
    )
    # NO clamp (1:1 with reference trainer's unclamped timestep.int(), NoiseMixin.py:212).
    return host.values[0]
