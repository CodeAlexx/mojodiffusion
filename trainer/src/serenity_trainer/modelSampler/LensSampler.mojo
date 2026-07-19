# LensSampler.mojo — 1:1 port of Serenity
#   modules/modelSampler/LensSampler.py  (LensSampler.__sample_base, pr-1510)
# structurally mirrored on modelSampler/ZImageSampler.mojo.
#
# ─────────────────────────────────────────────────────────────────────────────
# Serenity SOURCE (LensSampler.py:__sample_base, lines 38-158):
#   generator = manual_seed(seed)                                     (:51-57)
#   vae_scale_factor = 8 ; num_latent_channels = 32 ; patch_size = 2  (:63-65)
#   materialize_text_encoder(train_device)                            (:68)
#   batch_size = 2 if cfg>1 else 1                                    (:70)
#   prompt_features, prompt_mask = encode_text([prompt,neg]|[prompt]) (:71-74)
#   release_text_encoder(); torch_gc()                               (:76-77)
#   latent = randn(1, 32, H//8, W//8) f32                            (:80-85)
#   latent = patchify_latents(latent)                               (:87)  # ->128ch,H//16,W//16
#   latent = pack_latents(latent) ; image_seq_len = latent.shape[1]  (:89-90)
#   mu = compute_empirical_mu(image_seq_len, diffusion_steps)        (:91)
#   sigmas = np.linspace(1.0, 1/diffusion_steps, diffusion_steps)    (:95)
#   noise_scheduler.set_timesteps(N, mu=mu, sigmas=sigmas)           (:96)
#   timesteps = noise_scheduler.timesteps                           (:97)
#   img_shapes = [(1, H//8//patch, W//8//patch)]                    (:105)
#   for i, t in enumerate(timesteps):                              (:108)
#       latent_model_input = cat([latent]*batch_size)              (:109)
#       expanded_t = t.expand(latent_model_input.shape[0])         (:110)
#       noise_pred = transformer(
#           hidden_states=latent_model_input.to(train_dtype),       (:114)
#           encoder_hidden_states=[f.to(train_dtype) for f in prompt_features], (:115)
#           encoder_hidden_states_mask=prompt_mask,                 (:116)
#           timestep=expanded_t / 1000,                             (:117)
#           img_shapes=img_shapes)                                  (:118)
#       if batch_size == 2:                                         (:121)
#           cond, uncond = noise_pred.chunk(2)                      (:122)
#           comb = uncond + cfg*(cond - uncond)                     (:126)
#           cond_norm = norm(cond, dim=-1, keepdim)                 (:127)
#           comb_norm = norm(comb, dim=-1, keepdim)                 (:128)
#           scale = where(comb_norm>0, cond_norm/comb_norm.clamp(1e-12), 1) (:129)
#           noise_pred = comb * scale                               (:130)
#       latent = noise_scheduler.step(noise_pred, t, latent)[0]     (:132)
#   latent = unpack_latents(latent, H//8//patch, W//8//patch)       (:140-144)
#   latents = unscale_latents(latent)                              (:145)
#   latents = unpatchify_latents(latents)                          (:146)
#   decoded = vae.decode(latents)[0]                               (:148)
#   image = LensPipeline._to_pil(decoded)                          (:150)
#
# CFG NOTE: Lens uses norm-rescaled CFG (rescale combined pred to ||cond|| to
# prevent magnitude blowup at high guidance). Identical math to the borrowed
# serenitymojo pipeline cfg_norm_rescale_pair (lens_pipeline_1024_multistep.mojo).
# Here cond/uncond are the per-token noise predictions [1, S_img, 128]; the
# rescale is over the channel (last) dim, per token.
#
# SCHEDULER: Lens FlowMatch is the host scalar schedule in
# sampling/lens_flowmatch.mojo. set_timesteps(sigmas=linspace(1,1/N,N), mu) ==
# build_lens_shifted_sigmas: exponential-shift each raw sigma by mu. mu is the
# EMPIRICAL mu (compute_empirical_mu, image-token seq len), NOT the predict-path
# calculate_timestep_shift. The Euler step is prev = sample + (σ_next-σ)·pred.
#
# DTYPE: persistent latent / Euler state F32 (LensSampler.py:84 dtype=float32);
# the transformer INPUT is cast to train_dtype (BF16) per step (:114). The
# scheduler step accumulates in the latent storage dtype.
#
# ── CROSS-SLICE CONTRACT (SLICE A: model/LensDiT.mojo + model/LensModel.mojo +
#    model/LensVAE.mojo) — the expected ported-Lens API this driver consumes:
#   model.LensModel: pack_latents/unpack_latents/patchify_latents/unpatchify_latents
#                    (the staticmethod latent reshapes, ported 1:1 from LensModel.py)
#   model.LensModel: LensLoraSet (trained adapter overlay; B may be 0)
#   model.LensDiT:   lens_forward_full_infer[S_IMG,S_TXT](hidden[1,S_img,128],
#                       txt0..3 [1,S_txt,2880], mask[1,S_txt], timestep f32 (t/1000),
#                       weights, loras, ctx) -> Tensor[1,S_img,128]  (velocity/flow)
#   model.LensVAE:   LensDecoder[LH,LW] with .unscale_latents(latent_nchw) (VAE
#                    batch-norm un-normalize) and .decode(latent_nchw) -> image.

from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar

from serenity_trainer.modelLoader.LensModelLoader import LensWeights, LENS_ENC_DIM
# The latent (un)patchify/pack reshapes are @staticmethod on LensModel; the free
# function forms (identical math) live in BaseLensSetup — the SAME ones the verified
# predict/train_step path uses. The sampler is the INFERENCE path, so its LoRA
# overlay is the device List[LArc] (LArc = ArcPointer[LoraAdapter]) consumed by
# lens_forward_full_infer, NOT the host-list training LensLoraSet. The VAE seam is
# LensVAE[LH,LW] (decode + unscale_latents).
from serenity_trainer.modelSetup.BaseLensSetup import (
    pack_latents, unpack_latents, patchify_latents,
)
from serenity_trainer.model.LensDiT import lens_forward_full_infer, LArc
from serenity_trainer.model.LensVAE import LensVAE
from serenity_trainer.sampling.lens_flowmatch import (
    lens_compute_empirical_mu, build_lens_raw_sigmas, lens_exponential_shift,
    lens_euler_step,
)


# ── the sampler output: a decoded image tensor [1, 3, H, W] (BF16) ─────────────
struct LensSampleOutput(Movable):
    var image: Tensor

    def __init__(out self, var image: Tensor):
        self.image = image^


# ── norm-rescaled CFG (LensSampler.py:126-130, host-side) ──────────────────────
# comb = uncond + cfg*(cond - uncond) ; per-token scale = ||cond|| / max(||comb||,1e-12).
# cond/uncond: [1, S_img, IN_CH]. The norm is over the last (channel) dim, per token.
def cfg_norm_rescale_pair(
    cond: Tensor, uncond: Tensor, cfg_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    var cond_h = cond.to_host(ctx)
    var uncond_h = uncond.to_host(ctx)
    var sh = cond.shape()
    var n_tok = sh[1]
    var n_ch = sh[2]
    var total = n_tok * n_ch
    var out = List[Float32]()
    for _ in range(total):
        out.append(Float32(0.0))
    for tok in range(n_tok):
        var base = tok * n_ch
        var cond_sq = Float64(0.0)
        var comb_sq = Float64(0.0)
        var comb_row = List[Float32]()
        for j in range(n_ch):
            comb_row.append(Float32(0.0))
        for j in range(n_ch):
            var c = Float64(cond_h[base + j])
            var u = Float64(uncond_h[base + j])
            var cm = u + Float64(cfg_scale) * (c - u)
            comb_row[j] = Float32(cm)
            cond_sq += c * c
            comb_sq += cm * cm
        var cond_norm = Float64(0.0)
        if cond_sq > 0.0:
            cond_norm = cond_sq ** 0.5
        var comb_norm = comb_sq ** 0.5
        if comb_norm < 1.0e-12:
            comb_norm = 1.0e-12
        # torch.where(comb_norm>0, cond_norm/comb_norm, 1.0): comb_norm>=1e-12>0 here.
        var scale = cond_norm / comb_norm
        for j in range(n_ch):
            out[base + j] = Float32(Float64(comb_row[j]) * scale)
    var osh = List[Int]()
    osh.append(1); osh.append(n_tok); osh.append(n_ch)
    return Tensor.from_host(out, osh^, STDtype.BF16, ctx)


# ── one transformer evaluation → predicted flow (Lens predicts flow directly;
# NO leading minus, unlike Z-Image). LensSampler.py:113-119. ───────────────────
def _predict_flow[S_IMG: Int, S_TXT: Int](
    latent_packed: Tensor,    # [1, S_img, 128] F32 (persistent Euler state)
    t_over_1000: Float32,     # timestep/1000  (LensSampler.py:117)
    txt0: Tensor, txt1: Tensor, txt2: Tensor, txt3: Tensor,  # [1,S_txt,2880] BF16
    mask: Tensor,             # [1, S_txt]
    weights: LensWeights,
    loras: List[LArc],
    ctx: DeviceContext,
) raises -> Tensor:
    var hid_bf16 = cast_tensor(latent_packed, STDtype.BF16, ctx)   # :114 .to(train_dtype)
    return lens_forward_full_infer[S_IMG, S_TXT](
        hid_bf16, txt0, txt1, txt2, txt3, mask, t_over_1000, weights, loras, ctx
    )


# ── the denoise driver (port of LensSampler.__sample_base) ─────────────────────
# `cond_*` / `uncond_*` are the encoded prompt / negative-prompt 4-layer GPT-OSS
# features ([1, S_txt, 2880] BF16, one per selected layer) + the attention mask.
# uncond_* are only consumed when cfg>1 (LensSampler.py:70 batch_size=2). `weights`
# is the FROZEN transformer store; `loras` the trained adapter overlay; `vae` the
# Flux2 VAE decoder.
#
# Comptime shapes: S_IMG = (height//16)*(width//16) packed image tokens; S_TXT the
# (padded) text length; LH = height//8, LW = width//8 (the 32-ch latent grid the
# VAE decoder consumes after unscale+unpatchify).
def sample_lens[
    S_IMG: Int, S_TXT: Int, LH: Int, LW: Int, VLH: Int, VLW: Int,
](
    cond0: Tensor, cond1: Tensor, cond2: Tensor, cond3: Tensor, cond_mask: Tensor,
    uncond0: Tensor, uncond1: Tensor, uncond2: Tensor, uncond3: Tensor, uncond_mask: Tensor,
    seed: UInt64,
    diffusion_steps: Int,
    cfg_scale: Float32,
    var weights: LensWeights,
    loras: List[LArc],
    vae: LensVAE[VLH, VLW],
    ctx: DeviceContext,
) raises -> LensSampleOutput:
    var use_cfg = cfg_scale > Float32(1.0)
    var patch = 2
    # latent grid: 32-ch, H//8 = LH, W//8 = LW (LensSampler.py:80-85, F32).
    var lat_sh = List[Int]()
    lat_sh.append(1); lat_sh.append(32); lat_sh.append(LH); lat_sh.append(LW)
    var latent_nchw = randn(lat_sh^, seed, STDtype.F32, ctx)

    # patchify (32ch,LH,LW)->(128ch,LH//2,LW//2) then pack -> [1, S_img, 128]
    var patchified = patchify_latents(latent_nchw, ctx)
    var latent = pack_latents(patchified, ctx)                # [1, S_IMG, 128] F32

    # empirical mu over the IMAGE token seq len (LensSampler.py:90-91).
    var image_seq_len = S_IMG
    var mu = lens_compute_empirical_mu(image_seq_len, diffusion_steps)
    # set_timesteps(sigmas=linspace(1,1/N,N), mu): exponential-shift each raw sigma.
    var raw = build_lens_raw_sigmas(diffusion_steps)
    var sigmas = List[Float32]()
    for i in range(len(raw)):
        sigmas.append(lens_exponential_shift(raw[i], mu))

    # denoise loop (LensSampler.py:108-132). Lens passes sigma (= timestep) directly;
    # the transformer timestep input is sigma (timestep/1000 with timestep=sigma*1000
    # → sigma). We pass the shifted sigma as t_over_1000 (LensSampler.py:117 expanded
    # timestep is the scheduler sigma scaled to model units; for Lens FlowMatch the
    # model timestep == sigma).
    for i in range(diffusion_steps):
        var sigma_curr = sigmas[i]
        var sigma_next = Float32(0.0)
        if i + 1 < diffusion_steps:
            sigma_next = sigmas[i + 1]

        # cond (positive) branch flow prediction.
        var flow_cond = _predict_flow[S_IMG, S_TXT](
            latent, sigma_curr, cond0, cond1, cond2, cond3, cond_mask, weights, loras, ctx
        )
        var noise_pred: Tensor
        if use_cfg:
            var flow_uncond = _predict_flow[S_IMG, S_TXT](
                latent, sigma_curr, uncond0, uncond1, uncond2, uncond3, uncond_mask,
                weights, loras, ctx
            )
            noise_pred = cfg_norm_rescale_pair(flow_cond, flow_uncond, cfg_scale, ctx)
        else:
            noise_pred = flow_cond^

        # Euler step: prev = latent + (σ_next - σ)·noise_pred (lens_flowmatch).
        latent = lens_euler_step(latent, noise_pred, sigma_curr, sigma_next, ctx)

    # Free the resident DiT weights (~8GB) BEFORE the VAE decode — 1:1 with
    # Serenity LensSampler.py, which moves the transformer off-device
    # (transformer_to(temp_device)) before vae.decode. At 1024 the decode OOMs
    # if the DiT stays resident. Dropping `weights` releases its device tensors.
    _ = weights^

    # Decode tail (LensSampler.py:140-148). Serenity does, in order:
    #   latent_image = unpack_latents(latent, h_lat, w_lat)   # [1,128,LH//2,LW//2] packed, scaled
    #   latents = unscale_latents(latent_image)               # z*std + mean
    #   latents = unpatchify_latents(latents)                 # [1,32,LH,LW]
    #   decoded = vae.decode(latents)                         # plain AutoencoderKLFlux2 conv decode
    # In this port LensVAE.decode == KleinVaeDecoder.decode FUSES the single
    # inverse-BN (unscale) + packed unpatchify + conv decoder internally
    # (KleinVAE.mojo:766-810). So we apply unpack here and hand the PACKED, SCALED
    # latent [1,128,LH//2,LW//2] straight to vae.decode, yielding EXACTLY one
    # unscale + one unpatchify + one conv decode — 1:1 with Serenity. Calling
    # unscale_latents/unpatchify_latents here as well would double-transform.
    var unpacked = unpack_latents(latent, LH // patch, LW // patch, ctx)  # [1,128,LH//2,LW//2]
    var image = vae.decode(unpacked, ctx)                                 # [1,3,H,W]
    return LensSampleOutput(image^)
