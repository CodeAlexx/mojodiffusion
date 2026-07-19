# BaseZImageSetup.mojo — Z-Image flow-matching σ↔timestep map, the dynamic
# timestep shift, and the discrete timestep sampler.
#
# PORT SPEC (1:1):
#   * modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py
#       (_add_noise_discrete, lines 14-39)
#   * modules/modelSetup/mixin/ModelSetupNoiseMixin.py
#       (_get_timestep_discrete, lines 121-212)
#   * modules/model/ZImageModel.py
#       (scale_latents :175-176, unscale_latents :178-179,
#        calculate_timestep_shift :181-193)
#   * modules/modelSetup/BaseZImageSetup.py (predict :81-214)
#
# ── _add_noise_discrete (ModelSetupFlowMatchingMixin.py:14-39) ────────────────
#   num_timesteps   = timesteps.shape[-1]                       # = 1000   (:22)
#   all_timesteps   = arange(start=1, end=num_timesteps+1)      # [1..1000](:23)
#   sigma           = all_timesteps / num_timesteps             # (i+1)/N  (:24)
#   one_minus_sigma = 1.0 - sigma                                          (:25)
#   sigmas          = sigma[timestep]                                      (:29)
#   x_t = noise * sigmas + scaled_latent * (1 - sigmas)                    (:36-37)
# So for the discrete index t (0-based): sigma(t) = (t + 1) / num_timesteps.
#
# ── transformer t-input (BaseZImageSetup.py:130) ─────────────────────────────
#   t_model = (1000 - timestep) / 1000          (num_train_timesteps = 1000)
#
# ── velocity target (BaseZImageSetup.py:138) ─────────────────────────────────
#   flow = latent_noise - scaled_latent_image
#
# DTYPE: BF16 tensor storage, F32 host scalars for σ / μ / sampling.

from std.math import exp

# The canonical, aten-exact timestep sampler lives in the mixin port (the 1:1
# Serenity file-structure match). BaseZImageSetup.get_timestep_discrete now
# DELEGATES to it instead of re-deriving the math, so the multinomial tie
# semantics (aten lower_bound `cumdist[idx] >= u*sum`), the Float64 config dtype
# (no F32 boundary-rounding), and all six distribution weight tables are sourced
# from the single verified implementation. (Fixes: my earlier inline port used a
# forward-scan `threshold < cum` with the WRONG tie behaviour vs aten, and F32
# params.) See modelSetup/mixin/ModelSetupNoiseMixin.mojo:_get_timestep_discrete_host.
from serenity_trainer.modelSetup.mixin.ModelSetupNoiseMixin import (
    _get_timestep_discrete_host as _ts_host,
)


comptime ZIMAGE_NUM_TRAIN_TIMESTEPS = 1000

# FlowMatchEulerDiscreteScheduler defaults used by calculate_timestep_shift.
# ZImageModel.calculate_timestep_shift (:182) NOTE: these values are NOT in
# Z-Image's scheduler config and therefore fall back to the diffusers
# FlowMatchEulerDiscreteScheduler defaults (the "Flux" settings). Confirmed
# against diffusers 0.38.0.dev0 FlowMatchEulerDiscreteScheduler.__init__:
#   base_image_seq_len = 256, max_image_seq_len = 4096,
#   base_shift = 0.5, max_shift = 1.15.
# patch_size = 2 is hardcoded in calculate_timestep_shift (:187).
comptime ZIMAGE_BASE_IMAGE_SEQ_LEN = 256
comptime ZIMAGE_MAX_IMAGE_SEQ_LEN  = 4096
comptime ZIMAGE_BASE_SHIFT         = Float32(0.5)
comptime ZIMAGE_MAX_SHIFT          = Float32(1.15)
comptime ZIMAGE_PATCH_SIZE         = 2

# VAE config (zimage_base/vae/config.json):
#   shift_factor = 0.1159, scaling_factor = 0.3611
comptime ZIMAGE_VAE_SHIFT_FACTOR   = Float32(0.1159)
comptime ZIMAGE_VAE_SCALING_FACTOR = Float32(0.3611)


# σ for a discrete timestep index `t` in [0, num_timesteps).
#   sigma(t) = (t + 1) / num_timesteps          (ModelSetupFlowMatchingMixin.py:24)
def sigma_from_timestep(t: Int, num_timesteps: Int = ZIMAGE_NUM_TRAIN_TIMESTEPS) -> Float32:
    return Float32(t + 1) / Float32(num_timesteps)


# The transformer's normalized t-input for a discrete timestep index `t`.
#   t_model = (1000 - timestep) / 1000          (BaseZImageSetup.py:130)
# Generalized to num_timesteps for non-1000 schedules.
def model_t_from_timestep(t: Int, num_timesteps: Int = ZIMAGE_NUM_TRAIN_TIMESTEPS) -> Float32:
    return Float32(num_timesteps - t) / Float32(num_timesteps)


# Inverse: discrete timestep index from a σ in (0, 1].
#   t = round(sigma * num_timesteps) - 1
def timestep_from_sigma(sigma: Float32, num_timesteps: Int = ZIMAGE_NUM_TRAIN_TIMESTEPS) -> Int:
    var idx = Int(sigma * Float32(num_timesteps) + Float32(0.5)) - 1
    if idx < 0:
        return 0
    if idx >= num_timesteps:
        return num_timesteps - 1
    return idx


# SNR for flow-matching σ (used by loss weighting, ModelSetupDiffusionLossMixin).
# For flow matching x_t = (1-σ)·x0 + σ·ε, so the signal-to-noise ratio is
# ((1-σ)/σ)². Serenity's sigma-based weighting and the min-SNR path consume it.
def snr_from_sigma(sigma: Float32) -> Float32:
    var s = sigma
    if s < Float32(1e-8):
        s = Float32(1e-8)
    var ratio = (Float32(1.0) - s) / s
    return ratio * ratio


# ── ZImageModel.calculate_timestep_shift (ZImageModel.py:181-193) ────────────
# Python signature is calculate_timestep_shift(self, latent_width, latent_height)
# but BaseZImageSetup.predict (:109) calls it with
#   calculate_timestep_shift(scaled_latent_image.shape[-2],  # = H
#                            scaled_latent_image.shape[-1])   # = W
# i.e. (latent_width:=H, latent_height:=W). Since image_seq_len uses the PRODUCT
# (H//2)*(W//2) the naming swap is irrelevant to the result. We take (H, W) and
# compute exactly:
#   patch_size    = 2                                                      (:187)
#   image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)(:189)
#   m  = (max_shift - base_shift) / (max_seq_len - base_seq_len)           (:190)
#   b  = base_shift - m * base_seq_len                                     (:191)
#   mu = image_seq_len * m + b                                            (:192)
#   return math.exp(mu)                                                    (:193)
def calculate_timestep_shift(latent_h: Int, latent_w: Int) -> Float32:
    var base_seq_len = Float32(ZIMAGE_BASE_IMAGE_SEQ_LEN)
    var max_seq_len  = Float32(ZIMAGE_MAX_IMAGE_SEQ_LEN)
    var base_shift   = ZIMAGE_BASE_SHIFT
    var max_shift    = ZIMAGE_MAX_SHIFT
    var patch_size   = ZIMAGE_PATCH_SIZE

    var image_seq_len = Float32((latent_h // patch_size) * (latent_w // patch_size))
    var m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    var b = base_shift - m * base_seq_len
    var mu = image_seq_len * m + b
    return exp(mu)


# ── host RNG note ────────────────────────────────────────────────────────────
# The timestep sampler's host RNG now lives ONCE in the mixin port
# (modelSetup/mixin/ModelSetupNoiseMixin.mojo — _expand_key / _uniform_at /
# _standard_normal_at), which BaseZImageSetup delegates to. Serenity samples
# timesteps from torch.Generator(seed); serenitymojo has no host-callable torch
# stream, so VALUES WILL NOT BIT-MATCH torch's Mersenne/Philox RNG — the
# unavoidable RNG-stream difference. The FORMULAS that consume the draws
# (UNIFORM/LOGIT_NORMAL/HEAVY_TAIL + the COS_MAP/SIGMOID/INVERTED_PARABOLA
# multinomial) are 1:1 with Serenity; numeric parity is verified against dumped
# (latent, noise, timestep) tensors, not raw RNG bytes. The local PCG32→Box-Muller
# host RNG that used to live here was removed to keep a single sampler source.


# ── ModelSetupNoiseMixin._get_timestep_discrete (NoiseMixin.py:121-212) ──────
# ALL distributions ported 1:1:
#   * continuous: UNIFORM, LOGIT_NORMAL, HEAVY_TAIL          (NoiseMixin.py:145-172)
#   * discrete (torch.multinomial weight-table): SIGMOID, COS_MAP,
#     INVERTED_PARABOLA                                      (NoiseMixin.py:173-210)
#
#   if shift is None: shift = config.timestep_shift                        (:130-131)
#   deterministic: return int(N * 0.5) - 1                                 (:133-139)
#   else:
#     min_timestep = int(N * config.min_noising_strength)                  (:141)
#     max_timestep = int(N * config.max_noising_strength)                  (:142)
#     num_timestep = max_timestep - min_timestep                          (:143)
#     UNIFORM:       t = min + (max-min) * rand()                          (:151-153)
#     LOGIT_NORMAL:  bias = config.noising_bias                            (:155)
#                    scale = config.noising_weight + 1.0                   (:156)
#                    normal = Normal(bias, scale)                          (:158)
#                    t = sigmoid(normal) * num_timestep + min_timestep     (:159-160)
#     HEAVY_TAIL:    scale = config.noising_weight                         (:162)
#                    u = rand()                                            (:164-168)
#                    u = 1 - u - scale*(cos(pi/2*u)^2 - 1 + u)             (:169)
#                    t = u * num_timestep + min_timestep                   (:170)
#     # SHIFT (applied to the SCALED timestep), NoiseMixin.py:172:
#     t = N * shift * t / ((shift - 1) * t + N)        # CONTINUOUS ONLY
#     # ── discrete path (NoiseMixin.py:173-210), NO shift formula after ──
#     linspace            = linspace(0,1,num_timestep)                     (:180)
#     linspace            = linspace/(shift - shift*linspace + linspace)   (:181)
#     linspace_derivative = linspace(0,1,num_timestep)                     (:183)
#     linspace_derivative = shift/(shift + ld - ld*shift)^2                (:184)
#     COS_MAP:    weights = 2/(pi - 2*pi*ls + 2*pi*ls^2)                   (:189)
#                 weights *= linspace_derivative                           (:190)
#     SIGMOID:    bias    = config.noising_bias + 0.5                      (:194)
#                 weight  = config.noising_weight                          (:195)
#                 weights = linspace/(shift - shift*linspace + linspace)   (:197)
#                 weights = 1/(1+exp(-weight*(weights-bias)))              (:198)
#                 weights *= linspace_derivative                           (:199)
#     INVERTED_PARABOLA:
#                 bias    = config.noising_bias + 0.5                      (:203)
#                 weight  = config.noising_weight                          (:204)
#                 weights = clamp(-weight*((ls-bias)^2) + 2, min=0)        (:206)
#                 weights *= linspace_derivative                           (:207)
#     samples = multinomial(weights, 1, replacement=True) + min_timestep   (:209)
#     timestep = samples.long()                                            (:210)
#     return timestep.int()   # truncation toward zero                     (:212)
#
# batch_size is the latent batch; predict() runs B=1 latents (shape[0]==1), so we
# return the single sampled index. `seed` selects the RNG stream for this step.
#
# NOTE: torch.linspace(0,1,n) returns n points from 0 to 1 INCLUSIVE, i.e.
#   ls[i] = i/(n-1)  for n>1 ; for n==1 torch returns [0.0].
# torch.multinomial(w,1,replacement=True) draws index j with probability
#   w[j]/sum(w) ; equivalent to inverse-CDF of a single U[0,1) draw over the
# normalized weights (exact same distribution; RNG stream differs from torch,
# documented above).
# Z-Image's predict() runs B=1 latents, so this returns the SINGLE host int index.
# All six distributions + the deterministic branch are sourced from the verified
# mixin host port (`_ts_host` == _get_timestep_discrete_host). We pass Float64
# config values straight through (the mixin uses Float64 to match reference trainer's config
# dtype and avoid F32 boundary-rounding on int(N*strength) and the multinomial).
#
# `shift` arrives as Float32 from predict() (config.timestep_shift /
# calculate_timestep_shift are F32 here); widened to Float64 at the boundary —
# the mixin's continuous SHIFT formula and the discrete linspace/derivative all
# run in Float64. The deterministic branch returns int(N*0.5)-1 with NO clamp,
# the non-deterministic branch truncates via int() with NO clamp — both 1:1 with
# reference trainer (NoiseMixin.py:133-139, 212); the mixin only raises on num_timestep<1 for the
# discrete path, mirroring torch.linspace/torch.multinomial on an empty vector.
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

    # Delegate to the verified mixin host port (aten-exact multinomial, Float64
    # config dtype, all six distributions). batch_size=1 (B=1 predict()), empty
    # weights_cache (recompute the table this step — reference trainer caches across steps but the
    # table is a pure function of shift/weight/bias so per-step recompute is
    # numerically identical).
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
    # host.values has batch_size==1 entries; return the single timestep index.
    # NO clamp (1:1 with reference trainer's unclamped timestep.int(), NoiseMixin.py:212).
    return host.values[0]
