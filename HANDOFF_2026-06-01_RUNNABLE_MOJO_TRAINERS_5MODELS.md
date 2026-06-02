# Runnable pure-Mojo LoRA trainers, 5 models — Handoff (2026-06-01)

> Read this FIRST after `/clear`. Self-contained: by the end you know goal, what
> actually RUNS, the honest scope caveats, the git/entanglement state, and the
> exact next actions. This is a `feature` (port) handoff. Work is in
> `/home/alex/mojodiffusion` (its OWN git repo — NOT the EriDiffusion repos).

## §0 — Read these in order (60 seconds)
1. This file.
2. `HANDOFF_2026-06-01_SDXL_ERNIE_ANIMA_TRAINING_PORT.md` — the PRIOR phase (parity gates: block→stack→LoRA-step). It scoped train_*_real loops as post-milestone; THIS handoff is that post-milestone work done.
3. `serenitymojo/training/train_klein_real.mojo` — the template every train_<m>_real mirrors.
4. Memory `~/.claude/projects/-home-alex-EriDiffusion/memory/project_mojo_training_port_3models_2026-06-01.md` (REAL TRAINERS section).
Then run §9 and you're ready.

## §1 — Goal in one sentence
**Port the SDXL/Ernie/Anima/Flux/Z-Image LoRA trainers to pure-Mojo `train_<model>_real` loops that actually RUN — translating the existing EriDiffusion-v2 Rust trainers, reusing the parity-verified Mojo LoRA stacks, proven by a real run (loss↓ + LoRA-B imprint on real weights).**

The prior session built+gated the LoRA *step math* at reduced dims; the user's ask was *runnable trainers*. This session delivered that for all 5 (with honest scope caveats below).

## §2 — Project recap
- `mojodiffusion`/`serenitymojo` = pure-Mojo (no `import max`) GPU port. All 5 models had inference forwards + the parity-gated LoRA step already.
- Port SOURCE (translate, don't invent): `EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_<m>.rs` + `prepare_<m>.rs` (working Rust trainers). Encoders to port from: `inference-flame/src/{vae/ldm_encoder,vae/qwenimage_encoder,vae/klein_vae,models/mistral3b_encoder}.rs`.
- Structure: shared `training/` reused by CALLING; per-model `models/<m>/`; missing prims → `ops/` (Tenet 1; none needed this session).

## §3 — What's known true ✅ (every run re-verified by the MAIN LOOP, not agent self-report)
All runs: real checkpoint, real cache, fixed-σ smoke (clean monotone loss), 0 nonfinite, under 78°C.

| Model | Depth/scope I RAN | Result (my run) | Adapters |
|---|---|---|---|
| **Flux** | **FULL** 19+38 blocks (11.9B) via offload | loss 1.4944→1.4932; B 0→4835 | 418/418 |
| **SDXL** | real-dims UNet, 16² latent crop | loss 0.6497→0.6186; B 0→9273 | 700/700 |
| **Ernie** | **FULL** 36 layers (streamed) | B 0→11491 (loss σ-NOISY, no fixed-σ) | 252/252 |
| **Z-Image** | reduced depth 4/30 | loss 445.29→445.14; B 0→5534 | 56/56 |
| **Anima** | reduced S_IMG=64 crop | loss 1.868→1.374; B 0→2723 | 280/280 |

Key supporting gates (also re-run by me):
- **Flux offload stack** `flux_stack_lora_forward/backward_offload`: equivalence vs resident **75/75 checks cos≥0.9999**, peak **2.79 GB/block** real flux1-dev. (THE blocker that made 12B fit 24 GB.)
- **Flux VAE encoder** `FluxVaeEncoder` (ae.safetensors = Flux.1-AE, 16ch/8): mu cos vs torch **0.9999985**. (NOT flux2-vae.)
- **SDXL real-dims** finite-diff self-consistency on real weights: worst |ratio−1| **0.0086** (Klein composition-defect check).

## §4 — What's open / honest scope limits (NOT "broken" — real remaining engineering)
1. **Memory caps full scope.** Z-Image runs 4/30 layers, Anima a 64-token crop — their stacks are F32-resident-only (Z-Image full F32 = 24.62 GB > 24 GB). Need BF16-resident or an offload stack (Flux's is the template). Flux/Ernie run full-size but SLOW: Flux streams the 24 GB checkpoint twice/step (~260 s/step); Ernie host-list path ~5–10 min/step. Speedup = device-resident-scratch block variants (Klein `*_resident_moddev_rope_scratch` analogue).
2. **Resolution.** SDXL 16² / Anima 64-tok crops: the stacks are comptime-templated on spatial/seq dims; full 512²/1024² needs raising the knob + activation checkpointing.
3. **Mojo prepare/text-encode not complete.** Image VAE-encode works (`FluxVaeEncoder`). Text path does NOT: T5xxl uses a SentencePiece **Unigram** tokenizer (in-tree Mojo tokenizer is byte-BPE only) and **Mistral-3B** (Ernie) isn't ported. So ALL runs reuse the real Rust-encoded **caches** for text (and most for latent). To encode from raw captions in pure Mojo: port a Unigram tokenizer + Mistral (`inference-flame/src/models/mistral3b_encoder.rs`).
4. **Ernie loss signal is weak** (σ-noisy; no fixed-σ mode) — only B-imprint verified. Add a FIXED_SIGMA toggle (like the other 4) for a clean monotone proof.

## §7 — Files this session (committed on branch `training-port-5models-lora`)
Commit `04c99f6` (runnable trainers) — all in `serenitymojo/`:
- `training/train_{flux,sdxl,ernie,anima,zimage}_real.mojo` — the 5 real loops.
- `pipeline/{flux,anima,zimage}_prepare.mojo` — prepare/cache (image VAE-encode + cache reader; text from Rust cache).
- `models/flux/{weights.mojo, flux_stack_lora.mojo (offload fns), parity/flux_offload_*}` — base loader + offload stack + gates.
- `models/sdxl/{sdxl_real_train.mojo, real_weights.mojo, parity/real_finitediff.mojo, sdxl_unet_stack_lora.mojo}` — real-dims fwd+bwd+LoRA.
- `models/zimage/{real_weights.mojo, weights.mojo}`; `vae/{flux_vae_encoder.mojo, vae_encode_general.mojo, parity/*}`; `TRAINING_PLAN_*.md`.
Commit `b231af5` (prior arc, also this branch) — the parity-gated LoRA steps + per-model dirs.
**UNCOMMITTED + ENTANGLED (left for codex, see §11/§13):** `offload/plan.mojo` (my `build_flux_block_plan` mixed with codex's `build_klein_block_plan`), `offload/vmm_cuda.mojo` + `vmm_cuda_smoke.mojo` (`cu_mem_get_info` for Flux mem-smoke), `offload/plan_smoke.mojo` (flux gate).

## §8 — Required environment
- Build from repo root: `cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm <path> -o /tmp/<n>`.
- **Flux + Ernie need `-Xlinker -lcuda`** (turbo/offload uses cuMemcpy/cuMemGetInfo). SDXL built with it too (harmless).
- Single shared 24 GB 3090, thermal cap 78°C. GPU is SERIAL — Flux (~20 GB) + Ernie (~12 GB) can't co-reside; run trainers one at a time.
- `mojo-syntax` + `mojo-gpu-fundamentals` skills for any Mojo (0.26.x+: inout→mut/out, alias→comptime, let→var). Pure Mojo, no `import max`.

## §9 — Orientation script (5 minutes — re-verify the fast one)
```bash
cd /home/alex/mojodiffusion
# Z-Image is the fastest real-run re-verify (~28 s/step, fits with headroom):
pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_zimage_real.mojo -o /tmp/zi && /tmp/zi 2
#   expect: PROG loss DECREASE + loraB_nonzero=56/56, nonfinite=0
cat serenitymojo/training/train_klein_real.mojo | head -40   # the template
# SDXL real-dims (fast, ~28 s/step):
pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_sdxl_real.mojo -o /tmp/sx && /tmp/sx 5
```
Weight paths (verified on disk): Flux `/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors` + `vaes/ae.safetensors`; SDXL `.../checkpoints/sdxl_unet_bf16.safetensors` + `vaes/OfficialStableDiffusion/sdxl_vae.safetensors`; Ernie `/home/alex/models/ERNIE-Image/transformer`; Z-Image `/home/alex/.serenity/models/zimage_base/transformer` (diffusers dir, UNFUSED — not the fused single file); Anima `/home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors`.
Caches reused: `EriDiffusion-v2/cache/{alina_zimage_512, boxjana_ernie_512_FIXED, eri2_flux_512_smoke, eri2_sdxl_512_smoke}`.

## §10 — Next action
Pick ONE axis (all real engineering, none blocked):
1. **Make Flux self-contained** — commit `offload/plan.mojo` + `offload/vmm_cuda.mojo` (decide with codex how to split the klein+flux block-plan additions in plan.mojo), else `train_flux_real` won't build from commits `b231af5`+`04c99f6` alone. (Cheapest, unblocks Flux build.)
2. **Z-Image/Anima full scope** — port a BF16-resident or offload LoRA stack (mirror Flux's `flux_stack_lora_*_offload` + the resident-vs-offload equivalence gate), then flip `MAIN_DEPTH=30` (zimage) / raise `LATENT_HW` (anima).
3. **End-to-end Mojo prepare** — port a SentencePiece Unigram tokenizer (T5xxl) + `mistral3b_encoder.rs` so prepare encodes raw captions instead of reusing Rust caches.
4. **Speed** — device-resident-scratch block variants for Flux/Ernie (cut the per-step checkpoint streaming).

## §11 — DO NOT
- **Don't claim a model "trains" from compile or one synthetic step** — the bar is a REAL run: loss↓ (fixed-σ) + LoRA-B 0→nonzero on real weights (the L2P lesson). Re-run the gate; don't trust agent self-report.
- **Don't stage `offload/plan.mojo` / `vmm_cuda.mojo` blindly** — they carry codex's uncommitted offload+Klein work mixed with this session's flux additions. Resolve the split first.
- **Don't use `klein_encoder`/`flux2-vae` for flux1-dev** — flux1-dev = `ae.safetensors` (Flux.1-AE, 16ch/8); `klein_encoder` is the FLUX.2 128ch/16 VAE (wrong latent space).
- **Don't run two trainers at once** — single 24 GB GPU; Flux+Ernie OOM together.
- **Don't "fix" the reduced depth/crop by hardcoding** — it's a memory limit; the fix is BF16/offload + checkpointing, not faking full scope.
- **Don't put model backward in `ops/`** (Tenet 1) — none was needed; reuse the verified arms.

## §12 — Key file paths
- `serenitymojo/training/train_{klein,flux,sdxl,ernie,anima,zimage}_real.mojo` — the loops (klein = original template).
- `serenitymojo/training/klein_dataset.mojo` — cache reader (model-agnostic).
- `serenitymojo/pipeline/{klein_prepare_alina,flux_prepare,anima_prepare,zimage_prepare}.mojo` — prepare.
- `serenitymojo/models/<m>/{<m>_stack_lora,block,config,weights}.mojo` — per-model verified stacks.
- `serenitymojo/models/flux/flux_stack_lora.mojo` — resident + `*_offload` LoRA stack.
- `serenitymojo/models/sdxl/{sdxl_real_train,real_weights}.mojo` — SDXL real-dims path.
- `serenitymojo/vae/flux_vae_encoder.mojo` — flux1-dev VAE encoder; `vae/vae_encode_general.mojo` — generic AutoencoderKL-2D math.
- `serenitymojo/offload/{turbo_planned_loader,plan}.mojo` — streaming (plan.mojo entangled, uncommitted).
- `serenitymojo/MAP.md` — repo map.
- Port sources: `EriDiffusion-v2/crates/eridiffusion-cli/src/bin/{train,prepare}_<m>.rs`; encoders `inference-flame/src/{vae,models}/*_encoder.rs`.

## §13 — Git state
- `mojodiffusion`: `04c99f6` (branch `training-port-5models-lora`) — **DIRTY** (~160 files). This branch = b231af5 (LoRA steps) + 04c99f6 (runnable trainers), both this session, NOT pushed, NOT merged to master (`28d67d7`). Dirty tree = codex's uncommitted Klein config-refactor + codex's offload work + the entangled `offload/plan.mojo`/`vmm_cuda.mojo` (mine+codex) left intentionally uncommitted.
- `flame-core`, `EriDiffusion-v2`, `inference-flame`: untouched read-only references.

## §14 — Why we know the runs are real
Each train_<m>_real was rebuilt from current source by the MAIN LOOP and run on the real checkpoint + real cache; the PROG output (loss↓ under fixed-σ + LoRA-B 0→nonzero across ALL adapters + nonfinite=0) was captured in-session, not taken from the builder agent's report (two builders mis-narrated "concurrent agents" earlier this arc — re-running is the rule). Flux's offload path is additionally proven equivalent to the resident path (75/75 checks cos≥0.9999), so the FULL-depth 12B run is the same computation as the gated math, just streamed.
