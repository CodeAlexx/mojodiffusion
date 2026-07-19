# BaseLensSetup.mojo — Lens flow-matching σ↔timestep map, the dynamic timestep
# shift, the discrete timestep sampler, and the patchify/pack/scale latent
# transforms. Mirrors modelSetup/BaseZImageSetup.mojo (the existing template).
#
# PORT SPEC (1:1):
#   * Serenity pr-1510 modules/modelSetup/BaseLensSetup.py::predict (:75-160)
#   * Serenity pr-1510 modules/model/LensModel.py
#       (patchify_latents :268-273, unpatchify_latents :276-281,
#        pack_latents :260-261, unpack_latents :264-266,
#        scale_latents :285-290, unscale_latents :292-297,
#        calculate_timestep_shift :250-258)
#   * Serenity modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py
#       (_add_noise_discrete, lines 14-39)  — IDENTICAL to the Z-Image port
#   * Serenity modules/modelSetup/mixin/ModelSetupNoiseMixin.py
#       (_get_timestep_discrete, lines 121-212) — model-agnostic; reuse the verified
#       mixin host port.
#
# ── _add_noise_discrete (ModelSetupFlowMatchingMixin.py:14-39) ────────────────
#   num_timesteps = timesteps.shape[-1]                            # = 1000
#   sigma         = arange(1, N+1) / N           → sigma(t)=(t+1)/N (:24)
#   x_t           = noise*sigmas + scaled_latent*(1-sigmas)        (:36-37)
#
# ── transformer t-input (BaseLensSetup.py:147) ───────────────────────────────
#   timestep = timestep / 1000        # NOTE: Lens uses t/1000 (NOT the Z-Image
#                                     # inverted (1000-t)/1000).
#
# ── velocity target (BaseLensSetup.py:155) ───────────────────────────────────
#   flow = latent_noise - scaled_latent_image          # predicted = transformer
#                                                       # output directly (NO minus).
#
# DTYPE: BF16 tensor storage; F32 host scalars for σ/μ/sampling; latent transforms
# materialize on host (F32 registers) then re-upload BF16 — no persistent F32.

from std.math import exp
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# Model-agnostic verified timestep host sampler (same one Z-Image delegates to:
# aten-exact multinomial, Float64 config dtype, all six distributions).
from serenity_trainer.modelSetup.mixin.ModelSetupNoiseMixin import (
    _get_timestep_discrete_host as _ts_host,
)


comptime LENS_NUM_TRAIN_TIMESTEPS = 1000

# FlowMatchEulerDiscreteScheduler config used by calculate_timestep_shift. Lens
# (LensModel.calculate_timestep_shift :250-254) reads these from
# self.noise_scheduler.config at runtime. VERIFIED against the actual Lens
# scheduler_config.json (microsoft/Lens scheduler/scheduler_config.json,
# _class_name=FlowMatchEulerDiscreteScheduler, _diffusers_version 0.37.1):
#   base_image_seq_len=256, max_image_seq_len=4096, base_shift=0.5, max_shift=1.15,
#   num_train_timesteps=1000, use_dynamic_shifting=true, time_shift_type=exponential
#   (-> exp(mu), matches calculate_timestep_shift). These equal the diffusers Flux
#   defaults, so the four constants below are correct 1:1 — NOT unverified.
# patch_size=2 is hardcoded (LensModel.py:255).
comptime LENS_BASE_IMAGE_SEQ_LEN = 256
comptime LENS_MAX_IMAGE_SEQ_LEN  = 4096
comptime LENS_BASE_SHIFT         = Float32(0.5)
comptime LENS_MAX_SHIFT          = Float32(1.15)
comptime LENS_PATCH_SIZE         = 2

# patchified in_channels: AutoencoderKLFlux2 latent has 32 channels; patchify
# packs 2×2 → 32*4 = 128 = transformer in_channels (transformer.py:389).
comptime LENS_VAE_LATENT_CHANNELS = 32
comptime LENS_IN_CHANNELS         = 128
# AutoencoderKLFlux2 batch_norm_eps (vae config.json batch_norm_eps). Lens uses
# batch-norm latent scaling (LensModel.scale_latents :285-290).
# OPEN RISK: confirm batch_norm_eps against the Lens vae/config.json.
comptime LENS_VAE_BATCH_NORM_EPS  = Float32(1.0e-5)


# σ for a discrete timestep index `t` in [0, num_timesteps).
#   sigma(t) = (t + 1) / num_timesteps          (ModelSetupFlowMatchingMixin.py:24)
def sigma_from_timestep(t: Int, num_timesteps: Int = LENS_NUM_TRAIN_TIMESTEPS) -> Float32:
    return Float32(t + 1) / Float32(num_timesteps)


# The transformer's normalized t-input for a discrete timestep index `t`.
#   timestep = timestep / 1000          (BaseLensSetup.py:147)
# NB: Lens passes t/1000 (NOT the Z-Image inverted (1000-t)/1000).
def model_t_from_timestep(t: Int, num_timesteps: Int = LENS_NUM_TRAIN_TIMESTEPS) -> Float32:
    return Float32(t) / Float32(num_timesteps)


# SNR for flow-matching σ (loss weighting; ModelSetupDiffusionLossMixin). For
# x_t=(1-σ)x0+σε the SNR is ((1-σ)/σ)².
def snr_from_sigma(sigma: Float32) -> Float32:
    var s = sigma
    if s < Float32(1e-8):
        s = Float32(1e-8)
    var ratio = (Float32(1.0) - s) / s
    return ratio * ratio


# ── LensModel.calculate_timestep_shift (LensModel.py:250-258) ────────────────
# def calculate_timestep_shift(self, latent_height, latent_width):
#     base_seq_len = scheduler.config.base_image_seq_len
#     max_seq_len  = scheduler.config.max_image_seq_len
#     base_shift   = scheduler.config.base_shift
#     max_shift    = scheduler.config.max_shift
#     patch_size   = 2
#     image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
#     m  = (max_shift - base_shift) / (max_seq_len - base_seq_len)
#     b  = base_shift - m * base_seq_len
#     mu = image_seq_len * m + b
#     return math.exp(mu)
# CALL SITE (BaseLensSetup.py:131): shift = model.calculate_timestep_shift(
#     latent_height, latent_width) where latent_height/width are the PATCHIFIED
#   latent dims (BaseLensSetup.py:127-129: latent_image = patchify_latents(...),
#   latent_height = latent_image.shape[-2], latent_width = latent_image.shape[-1]).
#   So the dims passed are H//2, W//2 already; this fn divides by patch_size AGAIN
#   (a Lens-specific double //2 — ported 1:1, do not "fix").
def calculate_timestep_shift(patched_latent_h: Int, patched_latent_w: Int) -> Float32:
    var base_seq_len = Float32(LENS_BASE_IMAGE_SEQ_LEN)
    var max_seq_len  = Float32(LENS_MAX_IMAGE_SEQ_LEN)
    var base_shift   = LENS_BASE_SHIFT
    var max_shift    = LENS_MAX_SHIFT
    var patch_size   = LENS_PATCH_SIZE

    var image_seq_len = Float32((patched_latent_w // patch_size) * (patched_latent_h // patch_size))
    var m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    var b = base_shift - m * base_seq_len
    var mu = image_seq_len * m + b
    return exp(mu)


# ── ModelSetupNoiseMixin._get_timestep_discrete (NoiseMixin.py:121-212) ──────
# Delegated 1:1 to the verified model-agnostic mixin host port (same call Z-Image
# uses). All six distributions + the deterministic branch sourced from there. See
# modelSetup/BaseZImageSetup.mojo for the full per-line annotation; the math is
# model-independent (num_train_timesteps, distribution, strengths, weight/bias,
# shift). RNG-STREAM CAVEAT: values will not bit-match torch.Generator; parity is
# verified against dumped (latent,noise,timestep) tensors (see PORT_MAP).
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
        1,                                  # batch_size (B=1 predict())
        timestep_distribution,
        Float64(min_noising_strength),
        Float64(max_noising_strength),
        Float64(noising_weight),
        Float64(noising_bias),
        Float64(shift),
        List[Float64](),                    # empty cache → recompute table
        seed,
    )
    return host.values[0]


# ══════════════════════════════════════════════════════════════════════════════
# LATENT TRANSFORMS (LensModel.py). Host-resident (BF16 boundary → F32 registers →
# BF16), 1:1 with the torch reshape/permute. B=1 (predict runs single latents).
# ══════════════════════════════════════════════════════════════════════════════

# patchify_latents (LensModel.py:268-273):
#   x[B,C,H,W] → view[B,C,H//2,2,W//2,2] → permute(0,1,3,5,2,4)
#             → reshape[B, C*4, H//2, W//2]
# B=1; returns [1, C*4, H//2, W//2].
def patchify_latents(z: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = z.shape()
    var C = sh[len(sh) - 3]
    var H = sh[len(sh) - 2]
    var W = sh[len(sh) - 1]
    var h2 = H // 2
    var w2 = W // 2
    var src = z.to_host(ctx)                       # F32 host, layout [C,H,W] (B=1)
    var out = List[Float32]()
    var n_out = C * 4 * h2 * w2
    for _ in range(n_out):
        out.append(Float32(0.0))
    # out channel = c*4 + p*2 + q  (p over H-sub, q over W-sub), spatial (i,j).
    for c in range(C):
        for p in range(2):
            for q in range(2):
                var oc = c * 4 + p * 2 + q
                for i in range(h2):
                    for j in range(w2):
                        var sidx = c * (H * W) + (i * 2 + p) * W + (j * 2 + q)
                        var oidx = oc * (h2 * w2) + i * w2 + j
                        out[oidx] = src[sidx]
    var osh = List[Int](); osh.append(1); osh.append(C * 4); osh.append(h2); osh.append(w2)
    return Tensor.from_host(out^, osh^, STDtype.BF16, ctx)


# unpatchify_latents (LensModel.py:276-281):
#   x[B,Cp,h,w] → reshape[B,Cp//4,2,2,h,w] → permute(0,1,4,2,5,3)
#             → reshape[B,Cp//4,h*2,w*2]
def unpatchify_latents(z: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = z.shape()
    var Cp = sh[len(sh) - 3]
    var h = sh[len(sh) - 2]
    var w = sh[len(sh) - 1]
    var C = Cp // 4
    var H = h * 2
    var W = w * 2
    var src = z.to_host(ctx)
    var out = List[Float32]()
    for _ in range(C * H * W):
        out.append(Float32(0.0))
    for c in range(C):
        for p in range(2):
            for q in range(2):
                var ic = c * 4 + p * 2 + q
                for i in range(h):
                    for j in range(w):
                        var sidx = ic * (h * w) + i * w + j
                        var oidx = c * (H * W) + (i * 2 + p) * W + (j * 2 + q)
                        out[oidx] = src[sidx]
    var osh = List[Int](); osh.append(1); osh.append(C); osh.append(H); osh.append(W)
    return Tensor.from_host(out^, osh^, STDtype.BF16, ctx)


# pack_latents (LensModel.py:260-261):
#   x[B,C,H,W] → reshape[B,C,H*W] → permute(0,2,1) = [B, H*W, C]
def pack_latents(z: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = z.shape()
    var C = sh[len(sh) - 3]
    var H = sh[len(sh) - 2]
    var W = sh[len(sh) - 1]
    var src = z.to_host(ctx)
    var seq = H * W
    var out = List[Float32]()
    for _ in range(seq * C):
        out.append(Float32(0.0))
    for c in range(C):
        for s in range(seq):
            out[s * C + c] = src[c * seq + s]
    var osh = List[Int](); osh.append(1); osh.append(seq); osh.append(C)
    return Tensor.from_host(out^, osh^, STDtype.BF16, ctx)


# unpack_latents (LensModel.py:264-266):
#   x[B,seq,C] → reshape[B,H,W,C] → permute(0,3,1,2) = [B,C,H,W]
def unpack_latents(z: Tensor, H: Int, W: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = z.shape()
    var C = sh[len(sh) - 1]
    var seq = H * W
    var src = z.to_host(ctx)
    var out = List[Float32]()
    for _ in range(C * seq):
        out.append(Float32(0.0))
    for s in range(seq):
        for c in range(C):
            out[c * seq + s] = src[s * C + c]
    var osh = List[Int](); osh.append(1); osh.append(C); osh.append(H); osh.append(W)
    return Tensor.from_host(out^, osh^, STDtype.BF16, ctx)


# scale_latents (LensModel.py:285-290) — batch-norm latent scaling:
#   mean = vae.bn.running_mean.view(1,-1,1,1)
#   std  = sqrt(vae.bn.running_var.view(1,-1,1,1) + vae.config.batch_norm_eps)
#   return (latents - mean) / std
# Operates on the PATCHIFIED latent [1, C=128, h, w]; bn_mean/bn_var are [128].
def scale_latents(
    z: Tensor, bn_mean: Tensor, bn_var: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var sh = z.shape()
    var C = sh[len(sh) - 3]
    var h = sh[len(sh) - 2]
    var w = sh[len(sh) - 1]
    var hw = h * w
    var src = z.to_host(ctx)
    var mh = bn_mean.to_host(ctx)
    var vh = bn_var.to_host(ctx)
    var out = List[Float32]()
    for _ in range(C * hw):
        out.append(Float32(0.0))
    for c in range(C):
        var m = mh[c]
        var s = _sqrtf(vh[c] + eps)
        for k in range(hw):
            var idx = c * hw + k
            out[idx] = (src[idx] - m) / s
    var osh = List[Int](); osh.append(1); osh.append(C); osh.append(h); osh.append(w)
    return Tensor.from_host(out^, osh^, STDtype.BF16, ctx)


# unscale_latents (LensModel.py:292-297):
#   return latents * std + mean
def unscale_latents(
    z: Tensor, bn_mean: Tensor, bn_var: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var sh = z.shape()
    var C = sh[len(sh) - 3]
    var h = sh[len(sh) - 2]
    var w = sh[len(sh) - 1]
    var hw = h * w
    var src = z.to_host(ctx)
    var mh = bn_mean.to_host(ctx)
    var vh = bn_var.to_host(ctx)
    var out = List[Float32]()
    for _ in range(C * hw):
        out.append(Float32(0.0))
    for c in range(C):
        var m = mh[c]
        var s = _sqrtf(vh[c] + eps)
        for k in range(hw):
            var idx = c * hw + k
            out[idx] = src[idx] * s + m
    var osh = List[Int](); osh.append(1); osh.append(C); osh.append(h); osh.append(w)
    return Tensor.from_host(out^, osh^, STDtype.BF16, ctx)


from std.math import sqrt as _sqrt_math
def _sqrtf(x: Float32) -> Float32:
    return _sqrt_math(x)
