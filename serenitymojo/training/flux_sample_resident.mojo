# flux_sample_resident.mojo — sample-during-training denoise for FLUX.1-dev.
#
# Generates ONE sample image from the FLUX trainer's CURRENT state — the frozen
# FluxStackBase + the block-swap-streamed double/single block weights (via the
# trainer's live TurboPlannedLoader) PLUS the live, in-place-updated LoRA set
# (FluxLoraSet) — by running the SAME training+LoRA forward
# (flux_stack_lora_forward_offload) inside a rectified-flow Euler denoise loop on
# the FLUX.1-dev sigma schedule. The result is a packed latent the caller
# unpacks + VAE-decodes to a PNG.
#
# ── WHY this reuses the TRAINING forward (not a fresh inference forward) ───────
# The point of sampling-during-training is to see what the model produces WITH
# the LoRA currently being trained. flux_stack_lora_forward_offload applies the
# live LoRA on top of the frozen base while streaming each block through the
# trainer's loader — so calling it IS the model+LoRA forward, identical to the
# trainer's per-step forward (train_flux_real.mojo:552). Its `.out` is the packed
# velocity prediction [N_IMG*OUT_CH] (rectified-flow velocity = noise - latent),
# the SAME quantity the inference CLI's DiT returns (flux_sample_cli.mojo:294).
#
# ── DENOISE (1:1 with the gated inference CLI) ────────────────────────────────
# flux_sample_cli.mojo:denoise() runs FLUX's guidance-distilled rectified-flow
# Euler loop in PACKED latent space:
#   sched = build_flux1_sigma_schedule(steps, N_IMG)   # steps+1 descending sigmas
#   for i in range(steps):
#       t_curr = sched[i]; t_prev = sched[i+1]
#       pred = DiT(img, txt, t=t_curr*1000, guidance=g*1000, vector=clip_pool, rope)
#       img  = img + (t_prev - t_curr) * pred           # Euler down the schedule
# FLUX.1-dev is GUIDANCE-DISTILLED: a SINGLE forward per step; the guidance scalar
# is a MODEL INPUT (a guidance vector fed into the embedder), NOT a classifier-free
# guidance multiplier. There is NO negative prompt / no uncond pass / no CFG mix —
# this is the load-bearing difference from the ideogram4 sample template, which
# runs two forwards + a CFG blend. (flux_sample_cli.mojo:43-46, 239-241.)
#
# t/guidance prescale: the BFL time_factor convention multiplies the timestep AND
# guidance scalar by 1000 before the embedder (the foundation t_embedder does NOT
# apply the 1000x internally) — matched here exactly as the trainer + CLI do
# (train_flux_real.mojo:540, flux_sample_cli.mojo:282/288).
#
# ── CONDITIONING (v1 decision — flagged) ──────────────────────────────────────
# The trainer cache holds per-sample DATASET-caption embeds (t5_embed [seq,4096]
# + clip_pool [768]) — NOT arbitrary sample-prompt embeds. v1 reuses a cached
# sample's already-padded txt_tokens [N_TXT*TXT_CH] + clip_pool [VEC_DIM] as the
# conditioning (the same option the ideogram4 template took). It exercises the
# real denoise + decode + PNG path with conditioning already resident in VRAM —
# no extra T5/CLIP encoder load, no tokenizer wiring (both of which would compete
# with the frozen base + streamed block + LoRA + Adam state for memory). Swapping
# in a real encode(prompt) later is a drop-in replacement of (txt_tokens, clip_pool)
# only — the denoise/decode is unchanged. The driver passes the conditioning it
# already loaded for the cache sample.
#
# ── MEMORY NOTE (streaming cost of sampling mid-train) ────────────────────────
# Each denoise step runs a FULL flux_stack_lora_forward_offload, which streams all
# 19+38 blocks through the loader — i.e. one sample of N steps re-streams the whole
# transformer N times (N x the per-train-step streaming cost). At inference defaults
# (steps=20) that is ~20 full block-swap passes per sampled image, on top of the
# trainer's resident base + LoRA + Adam state. This is WHY sample cadence must be
# rare (every N steps). The VAE decoder + rope are built/loaded PER CALL so they
# add zero resident memory between samples. No NEW resident weights are held: the
# sampler borrows the trainer's already-resident base + loader + lora.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import reshape, permute
from serenitymojo.image.png import save_png, ValueRange

from serenitymojo.models.flux.flux_stack import FluxStackBase
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, flux_stack_lora_forward_offload,
)
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule
from serenitymojo.pipeline.flux_tiled_decode import flux_tiled_decode


# ──────────────────────────────────────────────────────────────────────────────
# flux_sample_offload — rectified-flow Euler denoise on the frozen base + streamed
# blocks + live LoRA. Returns the PACKED denoised latent [N_IMG*OUT_CH] (host
# Float32), which the caller unpacks + VAE-decodes (flux_decode_packed_to_png).
#
# The comptime params bind the FLUX.1-dev arch + resolution shapes exactly as the
# trainer instantiates them (train_flux_real.mojo:108-132). The runtime args are
# the trainer's OWN resident objects (base/loader/lora) + the rope host tables it
# already built + the cached conditioning it already loaded — so the sampler adds
# NO new resident weights.
#
# Inputs (all sourced from the trainer; nothing re-loaded except rope/VAE per call):
#   base          frozen FluxStackBase (resident in the trainer)
#   loader        the trainer's live TurboPlannedLoader (streams blocks; MUT — its
#                 prefetch/await/done cursor advances exactly as in a train step)
#   lora          the live LoRA set (updated in place by the optimizer each step)
#   txt_tokens    [N_TXT*TXT_CH] F32 — padded T5 conditioning (cached caption)
#   clip_pool     [VEC_DIM]      F32 — CLIP-pooled conditioning (cached caption)
#   cos, sin      rope host tables [S*H, Dh//2] flattened (the trainer's `cos`/`sin`)
#   guidance      guidance scalar (NOT prescaled; this fn applies the *1000)
#   steps         number of Euler steps (inference default 20)
#   seed          RNG seed for the t=1 packed init noise
#   D..EPS        the trainer's comptime arch scalars, passed through to the forward
# ──────────────────────────────────────────────────────────────────────────────
def flux_sample_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, IN_CH: Int, OUT_CH: Int,
](
    base: FluxStackBase,
    mut loader: TurboPlannedLoader,
    lora: FluxLoraSet,
    txt_tokens: List[Float32],     # [N_TXT*TXT_CH]
    clip_pool: List[Float32],      # [VEC_DIM]
    cos: List[Float32],            # [S*H * Dh//2]
    sin: List[Float32],            # [S*H * Dh//2]
    guidance: Float32,
    steps: Int,
    seed: UInt64,
    D: Int, Fmlp: Int, TXT_CH: Int, T_DIM: Int, VEC_DIM: Int, EPS: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    if steps < 1:
        raise Error("flux_sample_offload: steps must be >= 1")

    # guidance pre-scaled *1000 (BFL time_factor; same as the trainer + CLI). FLUX
    # is guidance-distilled, so this is a MODEL INPUT, never a CFG multiplier.
    var guidance_list = List[Float32]()
    guidance_list.append(guidance * Float32(1000.0))
    var guidance_opt = Optional[List[Float32]](guidance_list^)

    # t=1 packed init noise [N_IMG*IN_CH] (rectified-flow starts from pure noise).
    # Deterministic Box-Muller PCG (same generator the trainer uses for its per-step
    # noise, train_flux_real.mojo:273) so the sample is reproducible from `seed`.
    var img = _host_gaussian(N_IMG * IN_CH, seed)

    # descending sigma schedule (steps+1 points, exact endpoints 1.0..0.0), 1:1
    # with the gated inference CLI.
    var sched = build_flux1_sigma_schedule(steps, N_IMG)

    for i in range(steps):
        var t_curr = sched[i]
        var t_prev = sched[i + 1]

        # timestep prescaled *1000 (BFL time_factor).
        var timestep = List[Float32]()
        timestep.append(t_curr * Float32(1000.0))

        # single guidance-distilled forward: frozen base + streamed blocks + LoRA.
        var fwd = flux_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt_tokens.copy(), timestep.copy(), guidance_opt,
            clip_pool.copy(), base, loader, lora, cos.copy(), sin.copy(),
            D, Fmlp, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
        )

        # Euler step in packed space: img = img + (t_prev - t_curr) * pred.
        var dt = t_prev - t_curr
        for j in range(len(img)):
            img[j] = img[j] + dt * fwd.out[j]

    return img^


# ──────────────────────────────────────────────────────────────────────────────
# flux_decode_packed_to_png — unpack + VAE tiled-decode + write PNG.
# 1:1 with the parity-gated decode tail (flux_sample_cli.mojo:417-431):
#   unpack : [1, N_IMG, 64] -> [1,16,LATENT_H,LATENT_W]   (channel-major depatchify)
#   decode : flux_tiled_decode (3x3 overlap+feathered blend) -> [1,3,8*LH,8*LW]
#   write  : save_png (SIGNED [-1,1] range, the VAE output convention)
# The decoder is loaded fresh inside flux_tiled_decode PER CALL — sample cadence is
# rare, so this keeps zero extra resident VAE memory between samples (the trainer
# already holds the frozen base + LoRA + Adam state). VAE_PATH is the FLUX ae.
# ──────────────────────────────────────────────────────────────────────────────
def flux_decode_packed_to_png[
    N_IMG: Int, IMG_H2: Int, IMG_W2: Int, LATENT_H: Int, LATENT_W: Int, IN_CH: Int,
](
    packed_host: List[Float32],    # [N_IMG * (IN_CH*4)]  packed velocity-free latent
    vae_path: String,
    out_path: String,
    ctx: DeviceContext,
) raises:
    # Re-upload the host packed latent [1, N_IMG, IN_CH*4] then unpack to NCHW.
    var psh = List[Int]()
    psh.append(1)
    psh.append(N_IMG)
    psh.append(IN_CH * 4)
    var packed = Tensor.from_host(packed_host.copy(), psh^, STDtype.F32, ctx)
    var latent = _unpack_latent[N_IMG, IMG_H2, IMG_W2, LATENT_H, LATENT_W, IN_CH](
        packed, ctx
    )
    var img = flux_tiled_decode[LATENT_H, LATENT_W](latent, vae_path, ctx)
    save_png(img, out_path, ctx, ValueRange.SIGNED)


# ── unpack [1, N_IMG, IN_CH*4] -> [1,IN_CH,LATENT_H,LATENT_W] ─────────────────
# Verbatim depatchify from flux_sample_cli.mojo:_unpack_latent (the inverse of the
# trainer's _pack_latents): reshape [1,h2,w2,c,2,2] -> permute (0,c,h2,2,w2,2) ->
# [1,c,2*h2,2*w2]. Token (ih,iw) carried [c, ph, pw] (c-major), so the unpack
# walks them back to the (hh,ww) pixel grid.
def _unpack_latent[
    N_IMG: Int, IMG_H2: Int, IMG_W2: Int, LATENT_H: Int, LATENT_W: Int, IN_CH: Int,
](packed: Tensor, ctx: DeviceContext) raises -> Tensor:
    var s6 = List[Int]()
    s6.append(1)
    s6.append(IMG_H2)
    s6.append(IMG_W2)
    s6.append(IN_CH)
    s6.append(2)
    s6.append(2)
    var t6 = reshape(packed, s6^, ctx)
    var p = List[Int]()
    p.append(0)
    p.append(3)
    p.append(1)
    p.append(4)
    p.append(2)
    p.append(5)
    var tp = permute(t6, p^, ctx)
    var sp = List[Int]()
    sp.append(1)
    sp.append(IN_CH)
    sp.append(LATENT_H)
    sp.append(LATENT_W)
    return reshape(tp, sp^, ctx)


# ── deterministic host gaussian (Box-Muller PCG) ─────────────────────────────
# Identical generator to train_flux_real.mojo:_host_noise so a sampled image is
# reproducible from `seed` and uses the SAME noise statistics the trainer trains
# against (the repo-wide Box-Muller convention; see the noise_stats_smoke gate).
def _host_gaussian(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^
