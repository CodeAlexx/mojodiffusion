# sd35_sample_resident.mojo — sample-during-training denoise for SD3.5-Large.
#
# Generates ONE sample latent from the SD3.5 trainer's CURRENT state — the FROZEN
# SD35StackBase (x_embedder/context_embedder/t_embedder/y_embedder/final_layer) +
# the block-swap-streamed joint-block weights (via the trainer's live
# TurboPlannedLoader) PLUS the live, in-place-updated LoRA set (SD35LoraSet) — by
# running the SAME training+LoRA forward (sd35_stack_lora_forward_offload) inside a
# CFG flow-match Euler denoise loop on the SD3.5-Large OneTrainer sigma schedule.
# The result is a (trainer-scaled) PACKED latent the caller unpacks + SD3-VAE-
# decodes + writes to PNG.
#
# ── WHY this reuses the TRAINING forward (not the inference _sd3_large_forward) ──
# The whole point of sampling-during-training is to see what the model produces
# WITH the LoRA currently being trained. sd35_stack_lora_forward_offload
# (models/sd35/sd35_stack_lora.mojo:803) routes every joint block through the LoRA
# block fwd while streaming it through the trainer's loader — so calling it IS the
# model+LoRA forward, identical to the trainer's per-step forward
# (train_sd35_real.mojo:733). Its returned `.out` is the final-layer velocity
# prediction [N_IMG, OUT_CH] — the SAME quantity the inference CLI's DiT returns
# (sd3_sample_cli.mojo:_sd3_large_forward), just carried as host List[Float32]
# (the offload forward's host carrier) instead of a Tensor. The standalone
# inference forward uses SD3MMDiTPreBlockGate + BlockLoader with NO LoRA adapters —
# wrong for sampling-in-training — which is why this file does NOT reuse it.
#
# ── SCHEDULE + STEP (1:1 with the gated inference path, host-float carrier) ─────
# sd3_sample_cli.mojo:_denoise runs SD3.5-Large's shifted rectified-flow CFG Euler:
#   sched  = SD3FlowMatchScheduler.large_default()    # OneTrainer set_timesteps,
#                                                       # 28 steps, shift 3.0
#   for step in range(NUM_STEPS):
#       sigma = sched.timestep(step); dt = sched.dt(step)   # dt < 0 (descending)
#       v_cond   = DiT(latent, sigma, context_cond, pooled_cond)
#       v_uncond = DiT(latent, sigma, context_uncond, pooled_uncond)
#       velocity = v_uncond + CFG*(v_cond - v_uncond)        # sd3_cfg
#       latent   = latent + velocity*dt                      # sd3_euler_step
# We mirror it EXACTLY; the only deltas are (a) the forward call is the training
# offload fwd with the live LoRA, and (b) latent/velocity live as host
# List[Float32] in the trainer's PACKED latent space [N_IMG, IN_CH] (the offload
# forward's carrier) instead of an NCHW Tensor — the CFG + Euler arithmetic is
# identical scalar-for-scalar. The schedule values come from the SAME
# build_sd3_onetrainer_sigmas the inference scheduler uses.
#
# sigma convention: sd35_stack_lora_forward_offload's _build_conditioning
# multiplies sigma by 1000 INTERNALLY (sd35_stack_lora.mojo:313), so we pass the
# schedule sigma in its [0,1] form — exactly as the trainer passes sigma_cont
# (train_sd35_real.mojo:722,734). We do NOT pre-multiply by 1000 here.
#
# ── CONDITIONING (v1 decision — flagged) ──────────────────────────────────────
# The trainer cache holds per-sample DATASET-caption embeds (split CLIP-L/CLIP-G/
# T5 hidden + pooled), already staged by the trainer into the combined
# txt_tokens [N_CTX*CTX_CH] + pooled_h [POOLED_DIM] the train step feeds the
# forward. There is no in-tree SD3 triple-encoder "encode from string" entry
# (sd3_sample_cli.mojo:60-64 takes a pre-encoded sidecar), so v1 reuses the
# CURRENT step's cached caption embeds as the COND conditioning and a ZEROED
# [N_CTX*CTX_CH]/[POOLED_DIM] vector as the UNCOND (the empty-conditioning CFG
# branch). This exercises the real denoise + CFG + decode + PNG path with
# conditioning already resident in VRAM — no extra CLIP/T5 encoder load, no
# tokenizer wiring (both of which would compete with the frozen base + streamed
# block + LoRA + Adam state for memory). Swapping in a real encode(prompt) later
# is a drop-in replacement of (cond_txt, cond_pooled) only — the denoise/decode is
# unchanged.
#
# ── MEMORY NOTE (streaming cost of sampling mid-train) ────────────────────────
# Each CFG step runs TWO full sd35_stack_lora_forward_offload passes, each
# streaming all 38 joint blocks through the loader — i.e. one sample of N steps
# re-streams the whole transformer 2N times. At inference defaults (28 steps) that
# is ~56 full block-swap passes per sampled image, on top of the trainer's
# resident base + LoRA + Adam state. This is WHY sample cadence must be rare. The
# SD3 VAE decoder is loaded fresh PER CALL (the embedded first_stage_model.decoder
# in the SD3.5 checkpoint), so it adds zero resident memory between samples. No
# NEW resident weights are held: the sampler borrows the trainer's already-
# resident base + loader + lora.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import permute, reshape
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.ldm_decoder import load_sd3_embedded_ldm_decoder
from serenitymojo.sampling.sd3_flow_match import build_sd3_onetrainer_sigmas

from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35StackBase, SD35LoraSet, sd35_stack_lora_forward_offload,
)
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader


# ──────────────────────────────────────────────────────────────────────────────
# sd35_sample_resident — CFG flow-match Euler denoise on the resident/streamed
# base + live LoRA. All compute stays in the trainer's PACKED latent space
# [N_IMG, IN_CH] (host List[Float32]) — exactly the space the train loop's
# `latent_packed`/`noisy`/`fwd.out` live in.
#
# Inputs (all on the trainer's DeviceContext / loader):
#   base             frozen stack base (embedders + final layer)
#   loader           the SAME TurboPlannedLoader the train loop streams blocks with
#   lora             live LoRA set (updated in place by the optimizer each step)
#   cond_txt         [N_CTX*CTX_CH] host floats — COND text conditioning
#   cond_pooled      [POOLED_DIM]   host floats — COND pooled (CLIP-L+G) conditioning
#   uncond_txt       [N_CTX*CTX_CH] host floats — UNCOND text conditioning (zeros)
#   uncond_pooled    [POOLED_DIM]   host floats — UNCOND pooled conditioning (zeros)
#   init_noise       [N_IMG*IN_CH]  host floats — the t=1 packed latent (pure noise)
#   n_steps          number of Euler steps (sampler default 28)
#   cfg              classifier-free guidance scale (sampler default 4.5)
#   shift            FlowMatch static shift (SD3.5-Large default 3.0)
#   D..QK_EPS        the trainer's comptime arch scalars, passed through to the fwd
#
# Returns the denoised packed latent [N_IMG*IN_CH] host floats in TRAINER-SCALED
# space (the caller unpacks + VAE-decodes; the SD3 VAE's internal z/scale+shift is
# exactly the inverse of the trainer's (latent-SHIFT)*SCALE, so NO manual denorm
# is needed — identical to sd3_sample_cli feeding the latent straight to decode).
#
# NOTE on the comptime split: sd35_stack_lora_forward_offload is parameterised
# [H, Dh, N_IMG, N_CTX, S] (comptime). This wrapper is therefore also
# parameterised; the driver instantiates it with the SAME H/Dh/N_IMG/N_CTX/S the
# trainer uses (1024px: H=38 Dh=64 N_IMG=4096 N_CTX=154 S=4250).
# ──────────────────────────────────────────────────────────────────────────────
def sd35_sample_resident[
    H: Int, Dh: Int, N_IMG: Int, N_CTX: Int, S: Int
](
    base: SD35StackBase,
    mut loader: TurboPlannedLoader,
    lora: SD35LoraSet,
    cond_txt: List[Float32],        # [N_CTX*CTX_CH]
    cond_pooled: List[Float32],     # [POOLED_DIM]
    uncond_txt: List[Float32],      # [N_CTX*CTX_CH]
    uncond_pooled: List[Float32],   # [POOLED_DIM]
    init_noise: List[Float32],      # [N_IMG*IN_CH]
    n_steps: Int,
    cfg: Float32,
    shift: Float32,
    D: Int, MLP: Int, IN_CH: Int, CTX_CH: Int, OUT_CH: Int,
    TIMESTEP_DIM: Int, POOLED_DIM: Int,
    eps: Float32, qk_eps: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    if n_steps < 1:
        raise Error("sd35_sample_resident: n_steps must be >= 1")

    # OneTrainer SD3.5 set_timesteps schedule (n_steps+1 shifted sigmas, last 0.0).
    # Exactly the schedule the gated inference CLI builds
    # (SD3FlowMatchScheduler.large_default -> build_sd3_onetrainer_sigmas).
    var sigmas = build_sd3_onetrainer_sigmas(n_steps, shift)

    var img = init_noise.copy()   # [N_IMG*IN_CH], evolves in place each step

    for step in range(n_steps):
        var t_curr = sigmas[step]
        var t_next = sigmas[step + 1]
        var dt = t_next - t_curr   # < 0 (down the schedule), 1:1 with the sampler

        # COND pass: frozen base + streamed blocks + LIVE LoRA, cached caption text.
        # sigma passed in [0,1] form; the forward applies *1000 internally.
        var cond_fwd = sd35_stack_lora_forward_offload[H, Dh, N_IMG, N_CTX, S](
            img.copy(), cond_txt.copy(), cond_pooled.copy(), t_curr,
            base, loader, lora,
            D, MLP, IN_CH, CTX_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
            eps, qk_eps, ctx,
        )
        # UNCOND pass: same trunk + LoRA, zeroed text + pooled conditioning.
        var uncond_fwd = sd35_stack_lora_forward_offload[H, Dh, N_IMG, N_CTX, S](
            img.copy(), uncond_txt.copy(), uncond_pooled.copy(), t_curr,
            base, loader, lora,
            D, MLP, IN_CH, CTX_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
            eps, qk_eps, ctx,
        )

        # CFG: pred = uncond + cfg*(cond - uncond)   (sd3_cfg, host-float form)
        # then Euler: img = img + dt*pred            (sd3_euler_step, host-float form)
        var n = len(img)
        for i in range(n):
            var pred = uncond_fwd.out[i] + cfg * (cond_fwd.out[i] - uncond_fwd.out[i])
            img[i] = img[i] + dt * pred

    return img^


# ──────────────────────────────────────────────────────────────────────────────
# sd35_unpack_latent — packed [N_IMG, IN_CH] host floats -> NCHW [1,LC,LH,LW].
# Inverse of the trainer's _pack_latents (train_sd35_real.mojo:568-579), which
# packs each token (ih,iw) as the 64-vector `for c, for ph, for pw` (channel-major,
# index = ((c*PACK+ph)*PACK+pw)) with token-major order ih*PGW+iw. The reshape
# [1,PGH,PGW,LC,PACK,PACK] -> permute [0,3,1,4,2,5] -> [1,LC,LH,LW] walks those
# [c,ph,pw] back to the (hh,ww) pixel grid (byte-identical to the chroma resident
# unpack, whose pack convention is the same channel-major form).
# ──────────────────────────────────────────────────────────────────────────────
def sd35_unpack_latent[
    LC: Int, LH: Int, LW: Int, PGH: Int, PGW: Int, PACK: Int, N_IMG: Int, IN_CH: Int
](
    packed: List[Float32],   # [N_IMG*IN_CH]
    ctx: DeviceContext,
) raises -> Tensor:
    var t = Tensor.from_host(packed.copy(), [1, N_IMG, IN_CH], STDtype.F32, ctx)
    var t6 = reshape(t, [1, PGH, PGW, LC, PACK, PACK], ctx)
    var tp = permute(t6, [0, 3, 1, 4, 2, 5], ctx)        # [1,LC,PGH,PACK,PGW,PACK]
    return reshape(tp, [1, LC, LH, LW], ctx)             # [1,LC,LH,LW]


# ──────────────────────────────────────────────────────────────────────────────
# sd35_decode_latent_to_png — unpack + SD3 VAE decode + write PNG.
# 1:1 with the gated decode tail (sd3_sample_cli.mojo:381-389):
#   unpack : packed -> [1,LC,LH,LW]  (sd35_unpack_latent, above)
#   decode : load_sd3_embedded_ldm_decoder[LH,LW](CKPT).decode(latent) -> [1,3,8LH,8LW]
#            (decode applies z = z/SD3_SCALING + SD3_SHIFT internally == the exact
#             inverse of the trainer's (latent-VAE_SHIFT)*VAE_SCALE, so the
#             trainer-scaled latent is fed straight in — no manual denorm)
#   write  : save_png(..., SIGNED)  — the VAE output is in [-1,1]
# The decoder is loaded fresh PER CALL from the checkpoint's embedded
# first_stage_model.decoder.* — sample cadence is rare (every N steps), so this
# keeps zero extra resident VAE memory between samples (the trainer already holds
# the base + streamed blocks + LoRA + Adam state).
# ──────────────────────────────────────────────────────────────────────────────
def sd35_decode_latent_to_png[
    LC: Int, LH: Int, LW: Int, PGH: Int, PGW: Int, PACK: Int, N_IMG: Int, IN_CH: Int
](
    packed: List[Float32],   # [N_IMG*IN_CH] denoised trainer-scaled latent
    ckpt_path: String,       # SD3.5 checkpoint (embedded VAE decoder)
    out_path: String,
    ctx: DeviceContext,
) raises:
    var latent = sd35_unpack_latent[LC, LH, LW, PGH, PGW, PACK, N_IMG, IN_CH](
        packed, ctx
    )
    var vae = load_sd3_embedded_ldm_decoder[LH, LW](ckpt_path, ctx)
    var rgb = vae.decode(latent, ctx)
    save_png(rgb, out_path, ctx, ValueRange.SIGNED)
