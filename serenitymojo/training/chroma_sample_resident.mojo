# chroma_sample_resident.mojo — sample-during-training denoise for Chroma1-HD.
#
# Generates ONE sample latent from the model's CURRENT state — the FROZEN base
# (`ChromaStackBase` x_embedder/context_embedder/proj_out) + the streamed block
# weights (`TurboPlannedLoader`) + the FROZEN approximator (`ChromaDitCache`)
# PLUS the live, in-place-updated LoRA adapters (`FluxLoraSet`) — by running the
# verified offload LoRA forward (chroma_stack_lora_forward_offload) inside a CFG
# Euler flow-match denoise loop. The result is a (scaled) latent the caller
# unpacks + FLUX-VAE-decodes + writes to PNG.
#
# ── WHY this reuses the TRAINING forward (not the standalone _chroma_forward) ──
# The whole point of sampling-during-training is to see what the model produces
# WITH the LoRA currently being trained. chroma_stack_lora_forward_offload
# (models/chroma/chroma_stack_lora.mojo:606) routes every block through the LoRA
# block fwd (chroma_double/single_block_lora_forward), so calling it IS the
# model+LoRA forward. Its returned `.out` is the proj_out velocity prediction
# [N_IMG, OUT_CH] — the SAME tensor the standalone sampler's _chroma_forward
# returns (both are the DiT's final proj_out), so the Euler step + CFG combine
# are byte-for-byte the chroma_sample_cli loop. The standalone _chroma_forward
# would IGNORE the LoRA (it uses ChromaShared+BlockLoader, no adapters) — wrong
# for sampling-in-training — which is why this file does NOT reuse it.
#
# ── SCHEDULE + STEP (1:1 with chroma_sample_cli.mojo, the gated sampler) ───────
# chroma_sample_cli.mojo Stage 6-7 runs (NUM_STEPS=30, GUIDANCE=4.0):
#   sigmas = build_flux1_sigma_schedule(NUM_STEPS, N_IMG)      # [NUM_STEPS+1], 1->0
#   for step in range(NUM_STEPS):
#       t_curr=sigmas[step]; t_next=sigmas[step+1]; dt=t_next-t_curr  # dt < 0
#       pred = uncond + GUIDANCE*(cond - uncond)                       # CFG combine
#       img_packed = img_packed + dt*pred                             # Euler down
# We mirror it EXACTLY; the only delta is the forward call (training offload fwd
# with the live LoRA) and that pred/img live as host List[Float32] (the offload
# forward's carrier convention) instead of a Tensor — the arithmetic is identical.
#
# t_model: the training step computes t_model = sigma_idx/1000 and feeds it to the
# approximator. The standalone sampler feeds the SIGMA itself as `timestep` to
# _build_approximator_input, which multiplies by 1000 internally
# (chroma_sample_cli.mojo:313 _sinusoid_values(timestep*1000)). The trainer's
# _pooled_modulation_tensor calls approx._approximator_input(t_model) — and
# ChromaDitCache builds the SAME sinusoid from t_model*1000. So to reproduce the
# sampler's per-step modulation we pass t_model = sigma (the schedule value) into
# _pooled_modulation_tensor each step — identical to the sampler passing t_curr.
#
# ── CONDITIONING (v1 decision — flagged) ──────────────────────────────────────
# The trainer cache holds pre-encoded DATASET-caption t5_embed [1,seq,4096], NOT
# arbitrary sample-prompt embeds (no SentencePiece T5 tokenizer in-tree yet —
# chroma_sample_cli.mojo:51-54). v1 reuses a cached caption's T5 features as the
# COND conditioning (already padded to [N_TXT,4096] host floats by the caller —
# the same `txt_tokens` the train step builds). The UNCOND is the zeroed text
# features the inference sampler uses for CFG (an all-zero [N_TXT*TXT_CH] vector).
# This exercises the real denoise + CFG + decode + PNG path with conditioning
# already resident — no extra T5 load, no tokenizer wiring (both of which would
# compete with the resident base + approximator + streamed blocks + LoRA + Adam
# for memory and add risk). Swapping in a real encode(prompt) later is a drop-in
# replacement of `cond_txt` only.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import permute, reshape
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.ldm_decoder import load_flux1_ldm_decoder
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule

from serenitymojo.models.chroma.chroma_stack_lora import (
    ChromaStackBase, chroma_stack_lora_forward_offload,
)
from serenitymojo.models.flux.flux_stack_lora import FluxLoraSet
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.models.dit.chroma_dit import ChromaDitCache


# ──────────────────────────────────────────────────────────────────────────────
# chroma_sample_resident — CFG Euler flow-match denoise on the resident/streamed
# base + live LoRA. All compute stays in the trainer's PACKED latent space
# [N_IMG, IN_CH] (host List[Float32]) — exactly the space the train loop's
# `latent_packed`/`noisy`/`fwd.out` live in.
#
# Inputs (all on the trainer's DeviceContext / loader):
#   base             frozen stack base (x_embedder/context_embedder/proj_out)
#   approx           frozen approximator (distilled_guidance_layer) cache
#   loader           the SAME TurboPlannedLoader the train loop streams blocks with
#   lora             live LoRA set (updated in place by the optimizer each step)
#   cond_txt         [N_TXT*TXT_CH] host floats — COND text conditioning
#   uncond_txt       [N_TXT*TXT_CH] host floats — UNCOND text conditioning (zeros)
#   init_noise       [N_IMG*IN_CH]  host floats — the t=1 packed latent (pure noise)
#   cos, sin         host RoPE tables (the trainer's, built once)
#   n_steps          number of Euler steps (sampler default 30)
#   cfg              classifier-free guidance scale (sampler default 4.0)
#   mod_index        the approximator mod table row count (MOD_INDEX=344)
#
# Returns the denoised packed latent [N_IMG*IN_CH] host floats in TRAINER-SCALED
# space (the caller unpacks + VAE-decodes; the FLUX VAE's internal z/scale+shift
# is exactly the inverse of the trainer's (latent-SHIFT)*SCALE, so NO manual
# denorm is needed — identical to chroma_sample_cli feeding the unpacked latent
# straight to vae.decode).
#
# NOTE on the comptime split: chroma_stack_lora_forward_offload is parameterised
# [H, Dh, N_IMG, N_TXT, S] (comptime). This wrapper is therefore also
# parameterised; the driver instantiates it with the SAME H/Dh/N_IMG/N_TXT/S the
# trainer uses (512px: H=24 Dh=128 N_IMG=1024 N_TXT=512 S=1536).
# ──────────────────────────────────────────────────────────────────────────────
def chroma_sample_resident[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    base: ChromaStackBase,
    approx: ChromaDitCache,
    mut loader: TurboPlannedLoader,
    lora: FluxLoraSet,
    cond_txt: List[Float32],      # [N_TXT*TXT_CH]
    uncond_txt: List[Float32],    # [N_TXT*TXT_CH]
    init_noise: List[Float32],    # [N_IMG*IN_CH]
    cos: List[Float32],
    sin: List[Float32],
    n_steps: Int,
    cfg: Float32,
    mod_index: Int,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    if n_steps < 1:
        raise Error("chroma_sample_resident: n_steps must be >= 1")

    # FLUX flow-match schedule (1->0, NUM_STEPS+1 points). image_seq_len = N_IMG
    # exactly as chroma_sample_cli.mojo:803 (build_flux1_sigma_schedule(.., N_IMG)).
    var sigmas = build_flux1_sigma_schedule(n_steps, N_IMG)

    var img = init_noise.copy()   # [N_IMG*IN_CH], evolves in place each step

    for step in range(n_steps):
        var t_curr = sigmas[step]
        var t_next = sigmas[step + 1]
        var dt = t_next - t_curr   # < 0 (down the schedule), 1:1 with the sampler

        # frozen approximator -> per-step modulation table, conditioned on the
        # SIGMA (== t_model the train loop would feed; see header). [1,MOD_IDX,D].
        var pooled_tensor = _pooled_modulation_tensor(approx, t_curr, ctx)
        var pooled = _host_f32(pooled_tensor, ctx)

        # COND pass: frozen base + streamed blocks + LIVE LoRA, cond text.
        var cond_fwd = chroma_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            img.copy(), cond_txt.copy(), pooled.copy(), mod_index,
            base, loader, lora, cos.copy(), sin.copy(),
            D, Fmlp, in_ch, txt_ch, out_ch, eps, ctx,
        )
        # UNCOND pass: same trunk + LoRA, zeroed text conditioning.
        var uncond_fwd = chroma_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            img.copy(), uncond_txt.copy(), pooled.copy(), mod_index,
            base, loader, lora, cos.copy(), sin.copy(),
            D, Fmlp, in_ch, txt_ch, out_ch, eps, ctx,
        )

        # CFG: pred = uncond + cfg*(cond - uncond)  (chroma_sample_cli.mojo:822-826)
        # then Euler: img = img + dt*pred           (chroma_sample_cli.mojo:828-830)
        var n = len(img)
        for i in range(n):
            var pred = uncond_fwd.out[i] + cfg * (cond_fwd.out[i] - uncond_fwd.out[i])
            img[i] = img[i] + dt * pred

    return img^


# ── frozen per-step modulation table [1, MOD_INDEX, D] BF16 (the train loop's
#    _pooled_modulation_tensor; ChromaDitCache builds the SAME sinusoid the
#    standalone sampler's _build_approximator_input does). ───────────────────────
def _pooled_modulation_tensor(
    approx: ChromaDitCache, t_model: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var approx_in = approx._approximator_input(t_model, ctx)
    return approx.approximator_forward(approx_in, ctx)


# Stage a tensor (BF16/F16/F32) to host Float32 for the host-float forward carrier.
def _host_f32(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    if t.dtype() == STDtype.F16:
        var hf = t.to_host_f16(ctx)
        var out = List[Float32]()
        for i in range(len(hf)):
            out.append(hf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


# ──────────────────────────────────────────────────────────────────────────────
# chroma_unpack_latent — packed [N_IMG, IN_CH] host floats -> NCHW [1,LC,LH,LW].
# 1:1 with chroma_sample_cli.mojo:_unpack_latent (reshape [1,PGH,PGW,LC,PACK,PACK]
# -> permute [0,3,1,4,2,5] -> [1,LC,LH,LW]). The trainer's _pack_latents packs the
# 64-vector channel-major as ((c*PACK+ph)*PACK+pw), token-major [ih,iw] — exactly
# this unpack's inverse (verified against train_chroma_real._pack_latents).
# ──────────────────────────────────────────────────────────────────────────────
def chroma_unpack_latent[
    LC: Int, LH: Int, LW: Int, PGH: Int, PGW: Int, PACK: Int, N_IMG: Int, IN_CH: Int
](
    packed: List[Float32],   # [N_IMG*IN_CH]
    ctx: DeviceContext,
) raises -> Tensor:
    var t = Tensor.from_host(packed.copy(), [1, N_IMG, IN_CH], STDtype.F32, ctx)
    var t6 = reshape(t, [1, PGH, PGW, LC, PACK, PACK], ctx)
    var tp = permute(t6, [0, 3, 1, 4, 2, 5], ctx)        # [1,LC,PGH,PACK,PGW,PACK]
    return reshape(tp, [1, LC, LH, LW], ctx)              # [1,LC,LH,LW]


# ──────────────────────────────────────────────────────────────────────────────
# chroma_decode_latent_to_png — unpack + FLUX VAE decode + write PNG.
# 1:1 with the gated decode tail (chroma_sample_cli.mojo:835-859):
#   unpack : packed -> [1,LC,LH,LW] (chroma_unpack_latent, above)
#   decode : load_flux1_ldm_decoder[LH,LW](VAE_PATH).decode(f32 latent) -> [1,3,8LH,8LW]
#            (decode applies z/scale+shift internally == inverse of trainer scale)
#   write  : save_png(..., SIGNED)  — the VAE output is in [-1,1]
# The decoder is loaded fresh PER CALL — sample cadence is rare (every N steps),
# so this keeps zero extra resident memory between samples (the trainer already
# holds the base + approximator + streamed blocks + LoRA + Adam state).
# ──────────────────────────────────────────────────────────────────────────────
def chroma_decode_latent_to_png[
    LC: Int, LH: Int, LW: Int, PGH: Int, PGW: Int, PACK: Int, N_IMG: Int, IN_CH: Int
](
    packed: List[Float32],   # [N_IMG*IN_CH] denoised scaled latent
    vae_path: String,
    out_path: String,
    ctx: DeviceContext,
) raises:
    var latent = chroma_unpack_latent[LC, LH, LW, PGH, PGW, PACK, N_IMG, IN_CH](
        packed, ctx
    )
    var latent_f32 = cast_tensor(latent, STDtype.F32, ctx)
    var vae = load_flux1_ldm_decoder[LH, LW](vae_path, ctx)
    var rgb = vae.decode(latent_f32, ctx)
    save_png(rgb, out_path, ctx, ValueRange.SIGNED)
