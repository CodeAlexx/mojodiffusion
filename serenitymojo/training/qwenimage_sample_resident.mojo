# qwenimage_sample_resident.mojo — sample-during-training denoise for Qwen-Image.
#
# Generates ONE sample latent from the Qwen-Image trainer's CURRENT state — the
# FROZEN base (`QwenOffloadBase`: img_in/txt_in/proj_out + timestep-MLP + norm_out)
# + the block-swap-streamed double-block weights (the trainer's live
# `TurboPlannedLoader`) PLUS the live, in-place-updated LoRA set (`QwenLoraSet`) —
# by running the SAME training+LoRA forward (qwenimage_stack_lora_forward_offload)
# inside a Qwen-Image true-CFG flow-match Euler denoise loop. The result is a
# packed latent [N_IMG, OUT_CH] the caller unpatchifies + Qwen-VAE-tiled-decodes
# to a PNG.
#
# ── WHY this reuses the TRAINING forward (not the standalone CLI forward) ──────
# The point of sampling-during-training is to see what the model produces WITH the
# LoRA currently being trained. qwenimage_stack_lora_forward_offload
# (qwenimage_stack_lora.mojo:556) streams every one of the 60 double blocks
# through the trainer's loader and applies the live LoRA on top of the frozen base
# — so calling it IS the trainer's per-step forward (train_qwenimage_real.mojo:600).
# Its returned `.out` is the proj_out velocity prediction [N_IMG*OUT_CH] (host
# Float32), the SAME quantity the inference CLI's DiT returns (the CLI's
# forward_cfg_mixed_text proj_out). The standalone CLI path uses a DIFFERENT class
# (QwenImageDitOffloaded) with NO LoRA adapters, so it would IGNORE the LoRA being
# trained — wrong for sampling-in-training — which is why this file does NOT reuse it.
#
# ── SCHEDULE + STEP (1:1 with qwenimage_sample_cli.mojo, the gated sampler) ────
# qwenimage_sample_cli.mojo:denoise() runs (STEPS=30, CFG=4.0):
#   sched  = Scheduler.qwen(STEPS, Float32(N_IMG))   # dynamic-exp shift schedule
#   sigmas = sched.sigmas()                          # STEPS+1 values, 1->0
#   for i in range(STEPS):
#       preds = DiT_cfg(x, pos, neg, sigmas[i])      # cond + uncond forwards
#       pred  = cfg_qwen(preds.pos, preds.neg, CFG)  # true-CFG + per-row L2 rescale
#       x     = sched.step(x, pred, i)               # Euler: x + pred*(sig[i+1]-sig[i])
# We mirror it EXACTLY; the deltas are (1) the forward call (training offload fwd
# with the live LoRA) and (2) pred/x live as host List[Float32] (the offload
# forward's carrier convention) instead of Tensors — the schedule + Euler + CFG
# arithmetic are byte-for-byte the CLI's. The CFG combine (true-CFG textbook form
# + the per-row L2 norm rescale) is replicated host-side here because the offload
# forward returns host floats; it reproduces sampling/flow_match.cfg_qwen exactly.
#
# silu_temb_h: Qwen's per-step modulation. The trainer feeds the DiT a SIGMA, then
# computes silu(time_text_embed(sigma)) once per step (train_qwenimage_real.mojo:593
# _build_silu_temb). This file does the SAME per denoise step from sigmas[i] — the
# inner sinusoidal embedding applies the *1000 scale (train_qwenimage_real.mojo:348).
#
# ── CONDITIONING (v1 decision — flagged) ──────────────────────────────────────
# The trainer cache holds per-sample DATASET-caption embeds (txt_embed
# [seq, TXT_CH=3584] — Qwen2.5-VL hidden, NOT a sample-prompt encode; there is no
# in-tree Qwen2.5-VL encoder load in the trainer). v1 reuses a cached caption's
# already-padded txt_tokens [N_TXT*TXT_CH] as the COND conditioning (the SAME
# `txt_tokens` the train step builds, train_qwenimage_real.mojo:566). The UNCOND is
# a zeroed [N_TXT*TXT_CH] vector (the trainer has no encoded negative). This
# exercises the real denoise + true-CFG + decode + PNG path with conditioning
# already resident in VRAM — no extra Qwen2.5-VL encoder load, no tokenizer wiring
# (both of which would compete with the resident base + streamed block + LoRA +
# Adam state for memory). Swapping in a real encode(prompt) later is a drop-in
# replacement of (cond_txt, uncond_txt) only — the denoise/decode is unchanged.
# The driver passes the conditioning it already loaded for the cache sample.
#
# ── MEMORY NOTE (streaming cost of sampling mid-train) ────────────────────────
# Each denoise step runs TWO full qwenimage_stack_lora_forward_offload passes (cond
# + uncond), and each streams all 60 double blocks through the loader — i.e. one
# sample of N steps re-streams the whole transformer 2*N times (2*N x the per-train-
# step streaming cost; the train step streams it once for forward + once for
# backward). At the sampler default (steps=30) that is ~60 full 60-block block-swap
# passes per sampled image, on top of the trainer's resident base + LoRA + Adam
# state. This is WHY sample cadence must be rare (every N steps). The VAE decoder is
# loaded fresh PER CALL inside qwenimage_tiled_decode (qwenimage_tiled_decode.mojo:94),
# so it adds zero resident memory between samples. No NEW resident weights are held:
# the sampler borrows the trainer's already-resident base + loader + lora + rope +
# norm_out/timestep-MLP weights.

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.layout import unpatchify
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.qwenimage_tiled_decode import qwenimage_tiled_decode
from serenitymojo.sampling.flow_match import Scheduler

from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    QwenLoraSet, QwenOffloadBase, QwenOffloadForward,
    qwenimage_stack_lora_forward_offload, compute_silu_temb,
)
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.ops.embeddings import timestep_embedding


# ── per-step silu(time_text_embed(sigma)) [1, D] host F32 ─────────────────────
# Identical to train_qwenimage_real.mojo:_build_silu_temb: build the sinusoidal
# timestep embedding (with the *1000 scale applied inside) then push it through the
# frozen timestep MLP and silu — using the trainer's resident te_lin1/te_lin2
# weights carried on the QwenOffloadBase. Reuses compute_silu_temb (the same MLP
# the trainer's forward calls per block), so the modulation a sampled step sees is
# bit-identical to a train step at that sigma.
def _silu_temb_for_sigma(
    base: QwenOffloadBase, sigma: Float32, timestep_dim: Int, D: Int,
    ctx: DeviceContext,
) raises -> List[Float32]:
    # sinusoidal embedding: t scaled by 1000 (Qwen convention; see
    # train_qwenimage_real.mojo:348 _sinusoidal_temb).
    var t_h = List[Float32]()
    t_h.append(sigma * Float32(1000.0))
    var t_tensor = Tensor.from_host(t_h, [1], STDtype.F32, ctx)
    var t_emb = timestep_embedding(
        t_tensor, timestep_dim, ctx, Float32(10000.0), STDtype.F32
    )
    var sin_emb_h = t_emb.to_host(ctx)            # [1, timestep_dim] flat
    return compute_silu_temb(base, sin_emb_h, timestep_dim, D, ctx)


# ── Qwen true-CFG combine + per-row L2 rescale (host) ─────────────────────────
# Reproduces sampling/flow_match.cfg_qwen EXACTLY, on the host-float velocity
# carriers the offload forward returns:
#   comb = v_uncond + scale*(v_cond - v_uncond)              # TEXTBOOK form
#   out  = comb * (||v_cond||_lastdim / ||comb||_lastdim)    # per-row norm rescale
# Inputs are flat [rows*dim] (rows=N_IMG tokens, dim=OUT_CH); the L2 norm reduces
# the trailing `dim` (per-token), matching pipeline_qwenimage.py:704-708 and the
# device cfg_qwen's _l2_ratio_lastdim (which itself computes the ratio host-side).
def _cfg_qwen_host(
    v_cond: List[Float32], v_uncond: List[Float32], scale: Float32, dim: Int,
) -> List[Float32]:
    var n = len(v_cond)
    var rows = n // dim
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    for r in range(rows):
        var base = r * dim
        var cs_sum = Float32(0.0)
        var ms_sum = Float32(0.0)
        # comb + accumulate per-row L2 of cond and comb
        for j in range(dim):
            var cv = v_cond[base + j]
            var uv = v_uncond[base + j]
            var comb = uv + scale * (cv - uv)
            out[base + j] = comb
            cs_sum += cv * cv
            ms_sum += comb * comb
        var cn = sqrt(cs_sum)
        var mn = sqrt(ms_sum)
        var ratio = Float32(1.0)
        if mn > Float32(0.0):
            ratio = cn / mn
        for j in range(dim):
            out[base + j] = out[base + j] * ratio
    return out^


# ──────────────────────────────────────────────────────────────────────────────
# qwenimage_sample_resident — Qwen-Image true-CFG flow-match Euler denoise on the
# trainer's resident/streamed base + live LoRA. All compute stays in the trainer's
# PACKED latent space [N_IMG, OUT_CH] (host List[Float32]) — exactly the space the
# train loop's noisy/target/fwd.out live in.
#
# Inputs (all sourced from the trainer; nothing re-loaded except the VAE per call):
#   base        frozen QwenOffloadBase (resident in the trainer)
#   loader      the trainer's live TurboPlannedLoader (streams blocks; MUT — its
#               prefetch/await/done cursor advances exactly as in a train step)
#   lora        the live LoRA set (updated in place by the optimizer each step)
#   cond_txt    [N_TXT*TXT_CH] host F32 — COND text conditioning (cached caption)
#   uncond_txt  [N_TXT*TXT_CH] host F32 — UNCOND text conditioning (zeros)
#   init_noise  [N_IMG*IN_CH]  host F32 — the t=1 packed latent (pure noise)
#   cos, sin    rope host tables (the trainer's cos_h/sin_h, F32, built once)
#   norm_out_w  [2D,D] BF16 — top-level norm_out.linear weight (the trainer's)
#   norm_out_b  [2D]   BF16 — top-level norm_out.linear bias
#   n_steps     number of Euler steps (sampler default 30)
#   cfg         true-CFG scale (sampler default 4.0)
#   D..eps      the trainer's comptime arch scalars, passed through to the forward
#
# Returns the denoised packed latent [N_IMG*OUT_CH] host floats (the caller
# unpatchifies + VAE-decodes). N_IMG is BOTH the image-token count AND the packed
# seq_len fed to the Qwen dynamic-shift schedule (qwenimage_sample_cli.mojo:224
# Scheduler.qwen(STEPS, Float32(N_IMG))).
# ──────────────────────────────────────────────────────────────────────────────
def qwenimage_sample_resident[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    base: QwenOffloadBase,
    mut loader: TurboPlannedLoader,
    lora: QwenLoraSet,
    cond_txt: List[Float32],       # [N_TXT*TXT_CH]
    uncond_txt: List[Float32],     # [N_TXT*TXT_CH]
    init_noise: List[Float32],     # [N_IMG*IN_CH]
    cos: List[Float32],
    sin: List[Float32],
    norm_out_w: List[BFloat16], norm_out_b: List[BFloat16],
    n_steps: Int,
    cfg: Float32,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, timestep_dim: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    if n_steps < 1:
        raise Error("qwenimage_sample_resident: n_steps must be >= 1")

    # Qwen-Image dynamic-exponential flow-match schedule (1->0, n_steps+1 points).
    # seq_len == N_IMG (the packed token count), 1:1 with qwenimage_sample_cli.mojo.
    var sched = Scheduler.qwen(n_steps, Float32(N_IMG))
    var sigmas = sched.sigmas()

    var img = init_noise.copy()    # [N_IMG*IN_CH], evolves in place each step

    for step in range(n_steps):
        var sigma = sigmas[step]

        # per-step modulation: silu(time_text_embed(sigma)) [1, D] (frozen MLP).
        var silu_temb_h = _silu_temb_for_sigma(base, sigma, timestep_dim, D, ctx)

        # COND pass: frozen base + streamed blocks + LIVE LoRA, cond text.
        var cond_fwd = qwenimage_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            img.copy(), cond_txt.copy(), silu_temb_h.copy(),
            base, loader, lora, cos.copy(), sin.copy(),
            norm_out_w, norm_out_b,
            D, F, in_ch, txt_ch, out_ch, eps, ctx,
        )
        # UNCOND pass: same trunk + LoRA, zeroed text conditioning.
        var uncond_fwd = qwenimage_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            img.copy(), uncond_txt.copy(), silu_temb_h.copy(),
            base, loader, lora, cos.copy(), sin.copy(),
            norm_out_w, norm_out_b,
            D, F, in_ch, txt_ch, out_ch, eps, ctx,
        )

        # Qwen true-CFG + per-row L2 rescale (cfg_qwen), then Euler in packed space.
        var pred = _cfg_qwen_host(cond_fwd.out, uncond_fwd.out, cfg, out_ch)
        var dt = sigmas[step + 1] - sigmas[step]   # < 0 (down the schedule)
        for i in range(len(img)):
            img[i] = img[i] + dt * pred[i]

    return img^


# ──────────────────────────────────────────────────────────────────────────────
# qwenimage_decode_packed_to_png — unpatchify + Qwen-VAE tiled-decode + write PNG.
# 1:1 with the gated decode tail (qwenimage_sample_cli.mojo:332-340):
#   unpatchify : packed [1, N_IMG, IN_CH] -> NCHW [1, LAT_C, LAT_H*PATCH, LAT_W*PATCH]
#                via ops.layout.unpatchify(channels=LAT_C, h=LAT_H*PATCH,
#                w=LAT_W*PATCH, patch=PATCH). IN_CH == LAT_C*PATCH*PATCH (64=16*2*2).
#   decode     : qwenimage_tiled_decode[VAE_H, VAE_W] (3x3 overlap+feathered blend)
#                -> [1, 3, 8*VAE_H, 8*VAE_W]  (VAE_H = LAT_H*PATCH = 64 at 512px).
#   write      : save_png(..., SIGNED)  — the Qwen VAE output is in [-1, 1].
# The decoder is loaded fresh inside qwenimage_tiled_decode PER CALL — sample
# cadence is rare, so this keeps zero extra resident VAE memory between samples
# (the trainer already holds the base + LoRA + Adam state). The latent must be BF16
# for the decoder (qwenimage_sample_cli.mojo:334 casts to BF16 before decode).
#
# Comptime geometry (512px / patch=2): LAT_H=32, LAT_W=32 (patch-grid), PATCH=2,
# LAT_C=16 (VAE latent channels), IN_CH=64. -> VAE input [1,16,64,64] -> [1,3,512,512].
# ──────────────────────────────────────────────────────────────────────────────
def qwenimage_decode_packed_to_png[
    N_IMG: Int, LAT_H: Int, LAT_W: Int, LAT_C: Int, PATCH: Int, IN_CH: Int,
](
    packed: List[Float32],    # [N_IMG*IN_CH] denoised packed latent
    vae_dir: String,
    out_path: String,
    ctx: DeviceContext,
) raises:
    comptime assert IN_CH == LAT_C * PATCH * PATCH, "IN_CH must equal LAT_C*patch^2"
    comptime VAE_H = LAT_H * PATCH    # latent height in VAE-input pixels (64 @ 512px)
    comptime VAE_W = LAT_W * PATCH

    # [N_IMG*IN_CH] host -> [1, N_IMG, IN_CH] tensor (BF16; the decoder's dtype).
    var seq = Tensor.from_host(packed.copy(), [1, N_IMG, IN_CH], STDtype.BF16, ctx)
    var latent = unpatchify(seq, LAT_C, VAE_H, VAE_W, PATCH, ctx)   # [1, LAT_C, VAE_H, VAE_W]
    var latent_bf16 = cast_tensor(latent, STDtype.BF16, ctx)
    var img = qwenimage_tiled_decode[VAE_H, VAE_W](latent_bf16, vae_dir, ctx)
    save_png(img, out_path, ctx, ValueRange.SIGNED)
