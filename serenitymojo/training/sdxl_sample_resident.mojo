# sdxl_sample_resident.mojo — sample-during-training denoise for SDXL.
#
# Generates ONE sample image from the SDXL trainer's CURRENT state — the FROZEN
# base UNet weights (`SdxlRealWeights`) PLUS the live, in-place-updated per-ST
# LoRA stack (`List[SdxlLoraSet]`) — by running the SAME training forward
# (`sdxl_real_forward`) inside an eps-prediction Euler CFG denoise loop on the
# SDXL scaled-linear schedule. The result is an NCHW latent the caller VAE-decodes
# to a PNG.
#
# ── WHY this reuses the TRAINING forward (not the standalone SDXLUNet) ─────────
# The point of sampling-during-training is to see what the model produces WITH the
# LoRA currently being trained. `sdxl_real_forward[H, W]` (sdxl_real_train.mojo:289)
# threads the live `lora: List[SdxlLoraSet]` through every SpatialTransformer's
# LoRA projection, so calling it IS the model+LoRA forward — identical to the
# trainer's per-step forward (train_sdxl_real.mojo:524). Its returned `.out` is the
# eps prediction NHWC [1,L,L,4] — the SAME quantity the standalone sampler's
# `SDXLUNet.forward` returns. The standalone SDXLUNet (pipeline/sdxl_sample_cli.mojo)
# is a DIFFERENT struct compiled at LH=LW=128 that loads frozen base weights from
# disk with NO LoRA overlay wired — wrong for sampling-in-training — which is why
# this file does NOT reuse it.
#
# ── COMPTIME DIMS (load-bearing) ──────────────────────────────────────────────
# `sdxl_real_forward[H, W]` is compiled at the trainer's LATENT_HW (the small smoke
# default 16 -> 128px image; 64 -> 512px once activation checkpointing lands). This
# sampler is therefore ALSO parameterised [H, W] and the driver instantiates it with
# the SAME LATENT_HW the trainer compiles. The VAE decoder is loaded at [H, W] too
# (decode -> [1,3,8H,8W]). The Euler sigma table is built for `n_steps`.
#
# ── DENOISE (1:1 with sdxl_euler.mojo + pipeline/sdxl_sample_cli.mojo) ─────────
# SDXL is eps-prediction with the EulerDiscreteScheduler. Per step:
#   sigmas = build_sdxl_sigmas(n_steps)                 # [n_steps+1] high->0
#   x = noise * initial_noise_sigma(sigmas[0])          # scaled init latent
#   for i in range(n_steps):
#       c_in  = 1/sqrt(sigma^2+1)                        # input scale
#       x_in  = x * c_in
#       eps_c = unet(x_in, t_i, context, y)             # COND
#       eps_u = unet(x_in, t_i, ctx_uncond, y_uncond)   # UNCOND
#       eps   = eps_u + cfg*(eps_c - eps_u)             # CFG combine
#       x     = x + eps*(sigma_next - sigma)            # Euler eps-pred step
# This mirrors `_denoise` in pipeline/sdxl_sample_cli.mojo verbatim; the ONLY delta
# is the forward call (training forward with the live LoRA, NHWC) and that the
# latent `x` lives as a host List[Float32] in NCHW order (the trainer's latent math
# convention — train_sdxl_real.mojo:454-465) instead of a Tensor. The arithmetic is
# identical to sdxl_euler.sdxl_cfg + sdxl_euler.sdxl_euler_step.
#
# ── CONDITIONING (v1 decision — flagged) ──────────────────────────────────────
# The trainer cache holds the DATASET-caption embeds (text_embedding [1,77,2048] +
# pooled [1,1280] + time_ids [1,6]) — NOT arbitrary sample-prompt embeds (no
# pure-Mojo SDXL CLIP-L/G tokenizer in-tree; pipeline/sdxl_sample_cli.mojo:56-65).
# v1 reuses the SAME context/y the train step already built as the COND
# conditioning, and ZEROED context/y as the UNCOND (the standard empty-prompt CFG
# approximation — zeros, not a true empty-prompt CLIP encode). This is the SAME
# "conditioning already resident, no extra encoder/tokenizer" shortcut the chroma
# and flux sample templates took, and it exercises the real denoise + CFG + decode
# + PNG path with the live LoRA. Swapping in a real SDXL encode(prompt, negative)
# later is a drop-in replacement of (context, y, ctx_uncond, y_uncond) only.
#
# ── MEMORY NOTE ───────────────────────────────────────────────────────────────
# Each denoise step runs TWO full sdxl_real_forward passes (cond + uncond), each
# saving the full activation set the trainer's forward retains (it is the TRAINING
# forward — it always saves acts; there is no inference-only variant of it). This is
# ~2x the per-train-step forward memory PLUS it runs n_steps times, so one sample is
# ~2*n_steps full forwards. This is WHY sample cadence must be rare. The VAE decoder
# is loaded PER CALL so it adds zero resident memory between samples; the UNet
# weights + LoRA are BORROWED from the trainer (no new resident weights).

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.decoder2d import nchw_to_nhwc
from serenitymojo.models.vae.ldm_decoder import load_sdxl_ldm_decoder

from serenitymojo.sampling.sdxl_euler import (
    build_sdxl_sigmas, build_sdxl_timesteps,
    sdxl_initial_noise_sigma, sdxl_input_scale,
)
from serenitymojo.models.sdxl.sdxl_real_train import (
    SdxlRealWeights, sdxl_real_forward,
)
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import SdxlLoraSet


# ── tiny shape helper ─────────────────────────────────────────────────────────
def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def _zeros2(a: Int, b: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for _ in range(a * b):
        h.append(0.0)
    var s = List[Int](); s.append(a); s.append(b)
    return Tensor.from_host(h^, s^, STDtype.F32, ctx)


def _zeros3(a: Int, b: Int, c: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for _ in range(a * b * c):
        h.append(0.0)
    var s = List[Int](); s.append(a); s.append(b); s.append(c)
    return Tensor.from_host(h^, s^, STDtype.F32, ctx)


# ──────────────────────────────────────────────────────────────────────────────
# sdxl_sample_resident — eps-pred Euler CFG denoise on the FROZEN UNet weights +
# the LIVE per-ST LoRA. Runs entirely at the trainer's comptime LATENT_HW = H = W.
#
# Inputs (all on the trainer's DeviceContext):
#   w           frozen SDXL base weights (the trainer's resident SdxlRealWeights)
#   lora        live per-ST LoRA set list (updated in place by the optimizer)
#   context     [1,77,2048] COND cross-attention context (the cached caption's;
#               the SAME `context` the train step built — train_sdxl_real.mojo:485)
#   y           [1,2816]    COND ADM vector (pooled + sin_embed time_ids; the
#               SAME `y` the train step built — train_sdxl_real.mojo:480)
#   init_noise  [4*H*W] host floats in NCHW order — the t=high pure-noise latent
#               (the trainer's noise convention — train_sdxl_real.mojo:_host_noise)
#   n_steps     number of Euler steps (sampler default 30)
#   cfg         classifier-free guidance scale (sampler default 7.5)
#
# Returns the denoised latent as an NCHW device Tensor [1,4,H,W] (raw, un-rescaled;
# the VAE decode applies z/scale+shift internally — sdxl_sample_resident does NOT
# pre-scale, identical to pipeline/sdxl_sample_cli feeding the denoise output
# straight to vae.decode).
# ──────────────────────────────────────────────────────────────────────────────
def sdxl_sample_resident[H: Int, W: Int](
    w: SdxlRealWeights,
    lora: List[SdxlLoraSet],
    context: Tensor,       # [1,77,2048]
    y: Tensor,             # [1,2816]
    init_noise: List[Float32],   # [4*H*W] NCHW host floats
    n_steps: Int,
    cfg: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    if n_steps < 1:
        raise Error("sdxl_sample_resident: n_steps must be >= 1")
    var NLAT = 4 * H * W
    if len(init_noise) != NLAT:
        raise Error(
            String("sdxl_sample_resident: init_noise len ")
            + String(len(init_noise)) + " != " + String(NLAT)
        )

    # Euler schedule (high noise -> 0) + discrete timesteps (sdxl_euler.mojo).
    var sigmas = build_sdxl_sigmas(n_steps)
    var tsteps = build_sdxl_timesteps(n_steps)

    # x = noise * initial_noise_sigma(sigmas[0])  (NCHW host floats, in place).
    var init_sigma = sdxl_initial_noise_sigma(sigmas[0])
    var x = List[Float32]()
    for i in range(NLAT):
        x.append(init_noise[i] * init_sigma)

    # UNCOND conditioning = zeros (empty-prompt CFG approximation; see header).
    var ctx_uncond = _zeros3(1, context.shape()[1], context.shape()[2], ctx)
    var y_uncond = _zeros2(1, y.shape()[1], ctx)

    for i in range(n_steps):
        var sigma = sigmas[i]
        var sigma_next = sigmas[i + 1]
        var t_i = tsteps[i]
        var c_in = sdxl_input_scale(sigma)

        # x_in = x * c_in, NCHW host -> NCHW device -> NHWC (the forward's layout).
        var x_in_h = List[Float32]()
        for j in range(NLAT):
            x_in_h.append(x[j] * c_in)
        var x_in_nchw = Tensor.from_host(x_in_h^, _sh4(1, 4, H, W), STDtype.F32, ctx)
        var x_in_nhwc = nchw_to_nhwc(x_in_nchw, ctx)   # [1,H,W,4]

        # timestep tensor [1].
        var t_h = List[Float32](); t_h.append(t_i)
        var t_s = List[Int](); t_s.append(1)
        var t = Tensor.from_host(t_h^, t_s^, STDtype.F32, ctx)

        # COND + UNCOND forwards through the FROZEN base + LIVE LoRA (NHWC eps).
        var fwd_c = sdxl_real_forward[H, W](
            x_in_nhwc.clone(ctx), t.clone(ctx), y.clone(ctx), context.clone(ctx), w, lora, ctx
        )
        var eps_c = fwd_c.out.to_host(ctx)             # NHWC flat [H*W*4]
        var fwd_u = sdxl_real_forward[H, W](
            x_in_nhwc.clone(ctx), t.clone(ctx), y_uncond.clone(ctx), ctx_uncond.clone(ctx), w, lora, ctx
        )
        var eps_u = fwd_u.out.to_host(ctx)             # NHWC flat [H*W*4]

        # CFG combine + Euler eps-pred step, both in NCHW host order. eps is NHWC;
        # convert each element's index NHWC(h,w,c) -> NCHW(c,h,w) on the fly.
        var d_sigma = sigma_next - sigma
        for hh in range(H):
            for ww in range(W):
                for c in range(4):
                    var nhwc_i = (hh * W + ww) * 4 + c
                    var nchw_i = (c * H + hh) * W + ww
                    var e = eps_u[nhwc_i] + cfg * (eps_c[nhwc_i] - eps_u[nhwc_i])
                    x[nchw_i] = x[nchw_i] + e * d_sigma

    # denoised latent NCHW device tensor [1,4,H,W] (raw; VAE rescales internally).
    return Tensor.from_host(x^, _sh4(1, 4, H, W), STDtype.F32, ctx)


# ──────────────────────────────────────────────────────────────────────────────
# sdxl_decode_latent_to_png — SDXL VAE decode of the denoised NCHW latent + PNG.
# 1:1 with pipeline/sdxl_sample_cli.mojo Stage 4-5:
#   decode : load_sdxl_ldm_decoder[H,W](vae_path).decode(latent) -> [1,3,8H,8W]
#            (decode applies z = z/SDXL_SCALING + SDXL_SHIFT internally)
#   write  : save_png(..., SIGNED)  — the VAE output is in [-1,1]
# The decoder is loaded fresh PER CALL — sample cadence is rare — so this keeps
# zero extra resident memory between samples (the trainer already holds the UNet
# weights + LoRA + Adam state).
# ──────────────────────────────────────────────────────────────────────────────
def sdxl_decode_latent_to_png[H: Int, W: Int](
    latent_nchw: Tensor,   # [1,4,H,W] raw denoised latent
    vae_path: String,
    out_path: String,
    ctx: DeviceContext,
) raises:
    var latent_f32 = cast_tensor(latent_nchw, STDtype.F32, ctx)
    var vae = load_sdxl_ldm_decoder[H, W](vae_path, ctx)
    var rgb = vae.decode(latent_f32, ctx)
    save_png(rgb, out_path, ctx, ValueRange.SIGNED)
