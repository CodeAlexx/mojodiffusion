# ZImageSampler.mojo — 1:1 port of Serenity
#   modules/modelSampler/ZImageSampler.py  (ZImageSampler.__sample_base, :37-135).
#
# The Serenity denoising loop (ZImageSampler.py:80-125):
#   generator = manual_seed(seed)                                  (:51-55)
#   prompt_embedding = encode_text([prompt,neg] if cfg>1 else p)   (:69-74)
#   latent = randn(1, C, H//8, W//8) f32                           (:80-85)
#   noise_scheduler.set_timesteps(diffusion_steps)                 (:88)
#   timesteps = noise_scheduler.timesteps                          (:89)
#   for i, t in enumerate(timesteps):                              (:97)
#       inp = latent.unsqueeze(2).to(train_dtype)                  (:98)   # add a t-dim
#       inp = cat([inp]*batch_size)                                (:99)   # CFG dup
#       t_model = (1000 - t)/1000                                  (:104)  # inverted, norm'd
#       out = transformer(inp_list, t_model, prompt_embedding).sample      (:103-108)
#       noise_pred = - stack(out).squeeze(dim=2)                   (:110)  # NOTE the leading −
#       if cfg>1: noise_pred = neg + cfg*(pos - neg)               (:112-114)
#       latent = noise_scheduler.step(noise_pred, t, latent)[0]    (:116)
#   latents = unscale_latents(latent)                              (:124)
#   image = vae.decode(latents)[0]                                 (:125)
#
# The transformer's `.sample` is the model VELOCITY (BaseZImageSetup.py:130-135
# uses the SAME convention: model t-input = (1000 - timestep)/1000, predicted
# flow = - stack(transformer(...))). predicted noise_pred = -velocity, EXACTLY.
#
# The scheduler step is the diffusers FlowMatchEulerDiscreteScheduler ported 1:1
# in modelSampler/FlowMatchEulerDiscreteScheduler.mojo:
#   prev = sample + (sigma_next - sigma) * model_output
# (Z-Image config shift=6.0, num_train_timesteps=1000, no dynamic shifting.)
#
# BORROWED pieces (use, do not re-port):
#   • model/ZImageDiT.zimage_forward_full_lora[HL,WL,CAPLEN] — the full LoRA-overlaid
#     NextDiT forward (embed → refiners → main stack w/ 210 LoRA adapters → final →
#     unpatchify). Returns ZImageForwardOut whose `.velocity` [1,16,HL,WL] is `.sample`.
#   • model/ZImageVAE.{decode_latent, unscale_latents}. decode_latent's borrowed
#     decoder applies the z/scaling+shift rescale internally (_rescale ≡
#     unscale_latents, shift=0.1159, scaling=0.3611). Serenity does
#     unscale_latents THEN vae.decode; since the borrowed decode already rescales
#     once, we pass the RAW latent to decode_latent (a single rescale, matching the
#     net Serenity transform). See ZImageVAE.mojo:526-540 / :489-491.
#   • model/QwenTextEncoder.text_encode (caller-side; the sampler accepts the
#     precomputed cap_feats embeddings for the numeric gate).
#
# DTYPE: the persistent latent / Euler state is F32 (ZImageSampler.py:84
# dtype=torch.float32); only the transformer INPUT is cast to train_dtype (BF16)
# per step (ZImageSampler.py:98 `.to(train_dtype)`), done in _predict_noise since
# the borrowed forward propagates its input dtype. The scheduler step upcasts to
# F32 and accumulates in F32 (diffusers :486/:513), casting the prev_sample back
# to the velocity dtype at the end (:519). Per-step sigma deltas are host F32
# scalars (allowed scalar schedule math), applied via the scheduler tensor op.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar

from serenity_trainer.model.ZImageModel import ZImageLoraSet
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
from serenity_trainer.model.ZImageDiT import (
    ZImageInferCache,
    prepare_zimage_infer_cache,
    zimage_forward_full_infer,
    zimage_forward_full_infer_cached,
)
from serenity_trainer.model.ZImageVAE import ZImageDecoder, decode_latent
from serenity_trainer.modelSampler.FlowMatchEulerDiscreteScheduler import (
    FlowMatchEulerDiscreteScheduler, make_zimage_scheduler, ZIMAGE_DEFAULT_SHIFT,
)


# ── the sampler output: a decoded image tensor [1, 3, H, W] (BF16) ─────────────
struct ZImageSampleOutput(Movable):
    var image: Tensor

    def __init__(out self, var image: Tensor):
        self.image = image^


# ── one transformer evaluation → predicted noise (= -velocity) ─────────────────
# Wraps the borrowed forward and applies Serenity's leading − (ZImageSampler.py:110:
#   noise_pred = - torch.stack(output_list).squeeze(2)).
# t_model = (1000 - timestep)/1000 is computed by the caller from the scheduler's
# discrete timestep (ZImageSampler.py:104).
def _predict_noise[
    HL: Int, WL: Int, CAPLEN: Int,
](
    latent_nchw: Tensor,        # [1,16,HL,WL] F32 (persistent Euler state)
    t_model: Float32,           # (1000 - timestep)/1000
    cap_feats: Tensor,          # [CAPLEN, cap_feat_dim] BF16
    weights: ZImageWeights,
    loras: ZImageLoraSet,
    ctx: DeviceContext,
) raises -> Tensor:
    # Serenity casts latent_model_input to train_dtype (BF16) BEFORE the
    # transformer (ZImageSampler.py:98 `.to(train_dtype)`); the borrowed forward
    # propagates its input dtype, so we cast here to keep the transformer on the
    # BF16 train_dtype path. The persistent `latent` stays F32 in the caller.
    var lat_bf16 = cast_tensor(latent_nchw, STDtype.BF16, ctx)
    # Serenity samples under torch.no_grad(): NO saved activations / checkpoints.
    # Use the activation-free inference forward (velocity only). Same numerics as
    # the training forward, none of the saved-for-backward work.
    var velocity = zimage_forward_full_infer[HL, WL, CAPLEN](
        lat_bf16, t_model, cap_feats, weights, loras, ctx
    )                                                  # velocity == .sample
    return mul_scalar(velocity, Float32(-1.0), ctx)    # noise_pred = -velocity


def _predict_noise_cached[
    HL: Int, WL: Int, CAPLEN: Int,
](
    latent_nchw: Tensor,        # [1,16,HL,WL] F32 (persistent Euler state)
    t_model: Float32,
    cache: ZImageInferCache,
    weights: ZImageWeights,
    loras: ZImageLoraSet,
    ctx: DeviceContext,
) raises -> Tensor:
    var lat_bf16 = cast_tensor(latent_nchw, STDtype.BF16, ctx)
    var velocity = zimage_forward_full_infer_cached[HL, WL, CAPLEN](
        lat_bf16, t_model, cache, weights, loras, ctx
    )
    return mul_scalar(velocity, Float32(-1.0), ctx)


# ── the denoise driver (port of ZImageSampler.__sample_base) ───────────────────
# `cond`/`uncond` are the encoded prompt embeddings ([CAPLEN, dim] BF16): the
# CFG-positive (prompt) and CFG-negative (negative_prompt) caption features.
# uncond is only consumed when cfg>1 (ZImageSampler.py:69-74, batch_size=2 path).
# `weights` is the FROZEN transformer store; `loras` the trained adapter overlay;
# `vae` the borrowed Z-Image VAE decoder.
#
# HL = height//8, WL = width//8 (vae_scale_factor=8); the decoder's comptime
# LH/LW = HL//8 / WL//8 (it patchifies the latent further).
def sample_zimage[
    HL: Int, WL: Int, CAPLEN: Int,
](
    cond: Tensor,
    uncond: Tensor,
    seed: UInt64,
    diffusion_steps: Int,
    cfg_scale: Float32,
    timestep_shift: Float32,
    num_latent_channels: Int,
    weights: ZImageWeights,
    loras: ZImageLoraSet,
    vae: ZImageDecoder[HL // 8, WL // 8],
    ctx: DeviceContext,
) raises -> ZImageSampleOutput:
    # prepare latent image: randn(1, C, HL, WL) in F32 (ZImageSampler.py:80-85,
    # dtype=torch.float32). Serenity keeps latent_image in F32 across the whole
    # loop — only latent_model_input is cast to train_dtype (:98), which the
    # borrowed forward does internally. The Euler state must stay F32-precision;
    # the scheduler step upcasts to F32 and casts back to the velocity dtype
    # exactly like diffusers (:486/:519).
    var lat_sh = List[Int]()
    lat_sh.append(1); lat_sh.append(num_latent_channels); lat_sh.append(HL); lat_sh.append(WL)
    var latent = randn(lat_sh^, seed, STDtype.F32, ctx)

    # prepare timesteps  (ZImageSampler.py:88-89)
    var scheduler = make_zimage_scheduler(diffusion_steps, timestep_shift)
    var use_cfg = cfg_scale > Float32(1.0)
    var cond_cache = prepare_zimage_infer_cache[HL, WL, CAPLEN](cond, weights, ctx)
    var uncond_cache = prepare_zimage_infer_cache[HL, WL, CAPLEN](uncond, weights, ctx)

    # denoising loop (ZImageSampler.py:97-118)
    for i in range(diffusion_steps):
        # t_model = (1000 - timestep)/1000   (ZImageSampler.py:104). timestep is
        # the scheduler's discrete value (sigma*1000), NOT the loop index.
        var timestep = scheduler.timesteps[i]
        var t_model = (Float32(1000.0) - timestep) / Float32(1000.0)

        # positive (cond) branch → noise_pred = -velocity   (:103-110)
        var noise_pred = _predict_noise_cached[HL, WL, CAPLEN](
            latent, t_model, cond_cache, weights, loras, ctx
        )

        if use_cfg:
            # negative (uncond) branch   (:99 cat-dup → :113 chunk2)
            var neg_pred = _predict_noise_cached[HL, WL, CAPLEN](
                latent, t_model, uncond_cache, weights, loras, ctx
            )
            # noise_pred = neg + cfg*(pos - neg)   (ZImageSampler.py:112-114)
            var diff = sub(noise_pred, neg_pred, ctx)
            var scaled = mul_scalar(diff, cfg_scale, ctx)
            noise_pred = add(neg_pred, scaled, ctx)

        # scheduler.step(noise_pred, t, latent)[0]   (ZImageSampler.py:116)
        latent = scheduler.step(noise_pred, latent, i, ctx)

    # VAE decode (ZImageSampler.py:124-125). The borrowed decode_latent applies the
    # z/scaling+shift rescale internally (≡ unscale_latents), so the RAW latent goes
    # in (one net rescale, matching Serenity's unscale_latents → vae.decode).
    var image = decode_latent[HL // 8, WL // 8](vae, latent, ctx)
    return ZImageSampleOutput(image^)
