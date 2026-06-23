# Training-pipeline fixes + sampler verification — 2026-06-22

Follows the all-models audit (`TRAINING_PIPELINE_AUDIT_2026-06-22.md`). This session (1) fixed 8 models'
training pipelines to match their reference EXACTLY (OneTrainer or ai-toolkit), each build→skeptic→bugfix
verified, and (2) run-verified every standalone sampler that has weights on disk.

Discipline: every "fixed" item below COMPILES and was skeptic-verified against the reference by reading
both sides (file:line). **None has been run on real training data yet** — loss-trajectory parity is the
next phase (several model caches are missing/mismatched). Sampler verdicts ARE measured runtime results.

================================================================================
## PART 1 — FIX CAMPAIGN (8 models → reference, all compile-clean + skeptic-passed)
================================================================================
Reference per user directive: OneTrainer for flux/klein/zimage/ernie/sdxl/qwen; ai-toolkit for hidream/l2p.
LEFT ALONE: ltx2, sd35, wan22.

Cross-cutting fact resolved: OT LoRA scale = alpha/rank (LoRAModule.py:329); OT presets that DON'T set
lora_alpha inherit default **1.0** (TrainConfig.py:1144) → scale 1/16=0.0625. So flux/qwen/sdxl/ernie/zimage
use lora_alpha=1.0; klein keeps 16 (its preset sets it explicitly).

- **klein (9b/4b)** — config only: LR 4e-4→3e-5, +warmup 100, +CONSTANT scheduler, optimizer ADAMW explicit;
  klein4b eps 1e-6→1e-8, wd 0→0.01, shift 1.8→1.0. SKEPTIC: StableAdamW is INERT for the ADAMW case
  (only PRODIGY reads it) → Mojo plain decoupled AdamW is correct. Engine + LoRA + export already matched.
- **flux** — guidance 3.5→1.0 (OT trains flux-dev at guidance_scale=1.0; 3.5 is inference-only), lora_alpha→1.0,
  LR→3e-4, +warmup 200, LoRA-A kaiming init. Timestep already discretized like OT. PLUS the all-linear LoRA
  surface (504 modules: 418 block-proj + 86 stack-level mod/embedder/proj_out) to match OT's empty layer_filter
  (OT LoRAs every transformer Linear). NOTE: saved keys are OT split-qkv `lora_transformer_*` (correct vs OT);
  the serenitymojo Flux INFERENCE overlay wants Kohya fused `lora_unet_*` → a trained flux LoRA needs a
  diffusers→BFL converter to load at inference (follow-up).
- **sdxl** — LR 1e-4→3e-4 (+ relaxed the validate guard to cfg.lr>0), weight_decay 0→0.01, **lora_alpha 16→1.0**
  (caught by skeptic — was 16× too strong; comptime ALPHA + config both moved), +caption-dropout path
  (independent TE1/TE2 Bernoulli, default p=0 → byte-identical no-op), +warmup 200. eps-pred/DDPM/target/MSE/
  LoRA-set/Kohya-keys already matched OT.
- **qwen** — cache key txt_embed→text_embedding, TIMESTEP DISCRETIZATION (new DiscreteTimestep +
  sample_timestep_discrete_qwen — sigma=(idx+1)/1000 for noise, model_t=idx/1000 for the embedder; fixes a
  one-quantum offset), shift 3.0→1.0, lora_alpha→1.0, LR→3e-4, +warmup 200, LoRA-A kaiming. Layer-set (720) +
  export keys already matched.
- **zimage** — RoPE `build_positions` rewritten to match diffusers ZImageTransformer2DModel patchify_and_embed
  (cap_padded=ceil(valid_cap/32)*32, rows [0,cap_padded)→(i+1,0,0), image offset cap_padded+1). SKEPTIC found
  the OLD code was ALSO a latent inference bug (diverged from the verified NextDiT at real caption lengths), so
  the fix improves inference fidelity too. Residual: short-caption attn-mask gap (forward has no per-token mask).
- **ernie** — THE fix: unmasked text padding. Cache has text_real_len=3 but 253 nonzero PAD-embedding vectors
  were attending (sdpa_nomask). Now threads a per-key text-pad mask (image-first, -1e4 on padded text keys) via
  sdpa/sdpa_backward_masked across all 36 blocks + per-sample image-token RoPE axis-0 = real_len (matches
  transformer_ernie_image.py:388-401). No-op guard when real_len==N_TXT.
- **hidream-o1** (ai-toolkit) — removed the unconditional gauss-shift loss weighting (ref = plain MSE), sigma
  floor 0.001→0.002994012087583542, +5 resident-head LoRA adapters (new inline fwd+bwd through the frozen
  x_embedder/t_embedder/final_layer heads), noise_scale 7.5→8.0, eps→1e-6, +grad-clip 1.0. SKEPTIC: head
  fwd/bwd math correct by reading but UNGATED — needs a torch parity dump. Save-key `.model.` strip applied.
- **l2p** (ai-toolkit) — FATAL fixes: (1) cache reader for the real `pixel`/`cap_feats` keys (was reading
  Klein's latent/text_embedding → aborted), (2) wired the REAL frozen local_decoder MicroDiffusionModel U-Net
  fwd+bwd (killed the proxy-head surrogate). SKEPTIC also caught + I fixed an inverted adaLN timestep
  (t_value=sigma → 1-sigma, would have broken convergence). adaLN-LoRA (8th target) deferred (shared zimage
  infra — needs a separate L2P carrier).

Orchestrator compile-fixes: klein `-I MOJO-libs` (separate-process IPC pulls net/json); l2p `_absum_l2p`
made dtype-generic.

================================================================================
## PART 2 — SAMPLER RUN-VERIFICATION (24 GB GPU, measured)
================================================================================
Ran every standalone sampler with weights on disk; checked the output PNG is non-blank (PIL stddev).

**WORKS (4) — real non-blank 1024 images:** z-image · flux (streaming-offload fits 24 GB) · anima (after 2
dtype-cast fixes in anima_sample_cli.mojo; decodes via separate anima_decode_cli) · sd3.5.

**BROKEN (5):**
- klein — stale cap-cache (expected BF16[1,512,12288], on-disk caps wrong shape) → needs a re-precache.
- boogu — CUDA OOM mid-denoise (step 15), exceeds 24 GB resident.
- qwen-image — denoise OK (30/30) but `qwenimage_tiled_decode` is CPU-bound and HUNG 30 min (killed). Defect.
- sdxl — denoise OK, VAE decode OOM at 1024.
- chroma — denoise OK (DiT dropped first), FLUX-VAE decode OOM at 1024.

**SKIPPED (2, disk 100%/no-write rule):** ernie (needs a precache write), ideogram4 standalone (missing
ideogram4_fx_sampler.safetensors fixture).

**Dominant failure = the VAE DECODE stage** (3/5: qwen hang, sdxl+chroma OOM). Highest-leverage follow-up:
give qwen/sdxl/chroma a tiled/streamed GPU decode (qwen has a tiled path that's running on host).

================================================================================
## OPEN FOLLOW-UPS
================================================================================
- VAE-decode memory: tiled/streamed GPU decode for qwen/sdxl/chroma samplers.
- klein sampler: re-precache caps. boogu: offload to fit denoise.
- flux inference-loader: diffusers split-qkv → Kohya BFL fused-qkv converter (so trained flux LoRA loads).
- ernie validation sampler: same text-pad mask + image-RoPE fix as the trainer (sampler still unmasked).
- hidream +5-head: generate hidream_o1_train_step_ref.py torch dump to gate the new head bwd.
- l2p adaLN-LoRA: add as an 8th target via a separate L2P-only LoRA carrier.
- Real-run parity: build per-model caches, run N steps, compare loss-trajectory to the reference (the actual
  proof the fixes train correctly — not done; all current verification is compile + code-skeptic).
