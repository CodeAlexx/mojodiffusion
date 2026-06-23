# All-models training-pipeline audit vs references — 2026-06-22

Stage-by-stage audit of every serenitymojo `train_*_real.mojo` training pipeline against its
reference oracle (OneTrainer preset / OneTrainer-anima fork / EriDiffusion-v2 Rust / torch
parity dumps). One auditor per model; each compared the 9 stages (data/caption, latent
encode+norm, noise, timestep+shift, add_noise+target-sign, loss, optimizer, LoRA layer-set,
save/export-keys). Every item below is MEASURED (file:line both sides) unless tagged HYPOTHESIS.
Discipline: TENET 4 — divergence = the code differs (measured); that a divergence *degrades the
trained LoRA* is HYPOTHESIS until a causation probe runs.

================================================================================
## CROSS-CUTTING PATTERNS (systemic — highest leverage)
================================================================================

1. **Learning rate wrong in nearly every model** — the Mojo `configs/<model>.json` use a generic
   LR, not the per-model preset. ideogram4 8× (4e-4 vs 5e-5) · klein 13× (4e-4 vs 3e-5) ·
   anima 3.3× (1e-4 vs 3e-5) · l2p 3× (3e-4 vs 1e-4) · ltx2 6× (3e-4 vs 5e-5). **SDXL is the
   outlier — 3× too LOW** (1e-4 vs 3e-4) with a `validate_sdxl_train_config` guard that REJECTS
   the oracle's 3e-4.
2. **No LR warmup anywhere** — OT/EDv2 presets warm up 50–100 steps; every Mojo trainer runs flat
   LR from step 1.
3. **Unmasked text/padding corrupts conditioning** — ideogram4 (pad slots forced `indicator=3`),
   **ernie** (cache has 3 real tokens but 253 nonzero PAD-embedding vectors all attend via
   `sdpa_nomask`), anima (context 256 vs oracle 512). Most likely "trains on garbage" class.
4. **LoRA layer-set divergences** — ideogram4 (trains adaln + fused-QKV), hidream (missing 5
   resident-head adapters, 252 vs 257), l2p (missing adaln_modulation — "biggest convergence gap"),
   wan22 (no dual-expert). **klein / flux / sdxl / anima / ernie / zimage MATCH their oracle.**
5. **Export-key mismatch → LoRA won't load at inference** — flux (saves diffusers split-qkv; the
   serenitymojo Flux loader expects Kohya BFL fused-qkv → 0 targets matched), hidream (5 missing
   keys), l2p (probe). klein/anima/ernie/sdxl/zimage/wan22 export keys MATCH/loadable.
6. **Scaffolds / never-run-on-real-data** — ltx2 + wan22 are SCAFFOLDS (stub raises / "do not run").
   anima, sdxl, flux run a `FIXED_SIGMA_SMOKE` single-sample gate that has NEVER trained on real
   data — their "PASS" proves the backward chain compiles, NOT parity.

**Faithful across the board (credit):** core diffusion math — target sign (noise−clean), add_noise
(σ·noise+(1−σ)·clean), flow-match/epsilon prediction type, plain-MSE loss, AdamW formula
(betas/eps/wd/bias-correction), Box-Muller unit-Gaussian noise (no repo Box-Muller bug), grad-clip
1.0 — all MATCH the references in the models where the path actually runs.

================================================================================
## PER-MODEL LEDGER
================================================================================

### ideogram4 (ref: OneTrainer-dxqb-ideogram) — REAL DEFECTS
- Forward EXONERATED (cos 0.99993, loss 0.961=torch @t=0.7); loss 1.1 = correct expected mean of
  per-t loss (curve min at t=0.7), NOT a bug.
- Caption DOUBLE-JSON-wrap (stager wraps already-JSON .txt again → 706 vs 630 tokens, diverges @tok 8).
- Padding unmasked (pad slots `indicator=3` not `text_mask*3`; garbage attends as K/V).
- VAE latent-norm wrong source (ai-toolkit constants vs VAE BatchNorm stats: ~8% scale + mirror
  patchify) — bites only if the real cache is built via the Mojo encode path (unconfirmed).
- LoRA trains `adaln_modulation` (OT frozen) + fused-QKV (OT split q/k/v). 204 vs 238 adapters.
- Optimizer LR 4e-4 vs 5e-5 (8×), no warmup, no grad clip.

### klein (ref: OneTrainer FLUX_2 preset) — ENGINE CLEAN
- All numeric stages faithful; LoRA learns AND loads (export keys byte-identical, 144 modules).
- LR 13× (4e-4 vs 3e-5) [HIGH]; no warmup vs OT 100-step [MED].
- HYPOTHESIS: OT preset uses StableAdamW + weight_decay_by_lr; Mojo plain decoupled AdamW.

### sdxl (ref: OneTrainer BaseStableDiffusionXLSetup) — MATH CLEAN
- Epsilon prediction + scaled-linear DDPM + target ε + plain MSE all CORRECT (no flow-match contamination).
- LR 1e-4 vs 3e-4 (3× too low; guard rejects the right value); weight_decay 0.0 vs 0.01.
- No caption dropout; FIXED_SMOKE single sample; LATENT_HW=16 size-conditioning self-inconsistency.
- LoRA layer-set + Kohya export keys MATCH/loadable; latent scale 0.13025 correct.

### flux (ref: EriDiffusion-v2 train_flux / OneTrainer BaseFluxSetup) — MATH CLEAN, LoRA WON'T LOAD
- All scalars faithful (caption no-wrap, VAE, pack, timestep, target, guidance, AdamW, clip).
- HIGH: saved LoRA keys (diffusers split-qkv) ≠ Mojo inference loader (Kohya BFL fused-qkv) → loads
  0 targets in serenitymojo Flux inference (ComfyUI/diffusers WOULD accept it).
- HIGH: never run on real data (cache dir empty). MED: resolution hardcoded 512 vs preset 768.

### ernie (ref: OneTrainer BaseErnieSetup) — 1 CRITICAL BUG
- CRITICAL: text pad tokens attend UNMASKED — cache `text_real_len=3` but 256 rows fed to attention,
  rows 3..255 are real Mistral PAD embeddings (L2≈28-32), no mask (`sdpa_nomask`). ~99% of text
  attention is noise OT masks out. Most likely behind high/wrong loss.
- Everything else (scale_latents, noise, timestep, add_noise, target, loss, optimizer, LoRA, save
  keys) MATCHES; LoRA loads. (batch 2 vs 1 = minor.)

### zimage (ref: EriDiffusion-v2 train_zimage) — 1 BUG
- RoPE caption-position scheme: Mojo prunes+rebuckets caption to 224/256 (image axis-0 pos 225/257)
  vs EDv2 full-512 (pos 513); pad tokens get (0,0,0) not (i+1,0,0). Different attention phase.
- Everything else matches (VAE shift/scale, noise, timestep, target, MSE, AdamW, LoRA, PEFT keys).

### anima (ref: OneTrainer anima fork `anima-pr1487` + EDv2 train_anima) — NOT A REAL RUN
- S1 CRITICAL: `FIXED_SIGMA_SMOKE=True` (FIXED_SIGMA=0.5, noise seed frozen, one fixed latent) →
  overfits one (sigma,noise,latent,caption). Must flip False for a real run.
- S2: LR 1e-4 vs 3e-5; timestep_shift 1.0 vs 3.0 (rect-flow needs the shift).
- S3 HYPOTHESIS: context len 256 vs 512. S4: one global context, no caption dropout.
- LoRA layer-set, export keys, target sign all MATCH.

### hidream_o1 (ref: ai-toolkit + EDv2 train_hidream_o1) — REAL DEFECTS
- SEV-1: EXTRA gauss-shift loss weighting `wt(t)` applied to loss AND grad; reference is plain MSE
  (ai-toolkit / DiffSynth comments it out / torch dump / Rust all unweighted). Reshapes every gradient.
- SEV-2: sigma floor 0.001 vs 0.002994012087583542 (the Rust side documents this as a fixed bug) —
  over-amplifies 1/sigma velocity weighting at low sigma.
- SEV-2: LoRA missing 5 resident-head adapters (252 vs 257); head linears frozen.
- SEV-3: noise_scale 7.5 vs 8.0; eps 1e-8 vs 1e-6; no grad clip vs 1.0.
- Caption (no double-wrap), pixel-space, target sign, Box-Muller, LoRA init all CORRECT.

### l2p (ref: EriDiffusion-v2 train_l2p) — 2 FATAL
- FATAL #1: cache-key mismatch (cache has `pixel`/`cap_feats`; Mojo KleinCache reads
  `latent`/`text_embedding`/`text_mask` + asserts rank-4) → trainer aborts on first peek_key. Never runs.
- FATAL #2: head replaced by a PROXY final-linear (xᵀ embedder weight) instead of the real frozen
  local_decoder U-Net → LoRA minimizes a surrogate objective unrelated to real L2P inference.
- SEV-2: timestep shift 3.0 (inference schedule) vs uniform unshifted training; missing adaln LoRA.

### ltx2 (ref: musubi-tuner) — SCAFFOLD
- No production AV training path. Legacy video-only smoke can't run (missing cache+ckpt), feeds NO
  text conditioning, 192 vs 1152 LoRA adapters. AV "trainer" is a contract stub that raises by design.
- To make real: wire ltx2_block_backward_av into a stack loop + AV data loader + text cross-attn +
  full 1152-adapter surface + shifted_logit_normal(stretched) timesteps + eri2 recipe scalars.

### wan22 (ref: EriDiffusion-v2 train_wan22) — SCAFFOLD
- `FIXED_SIGMA_SMOKE`, header "do not run." CRITICAL: no dual-expert (WAN2.2-T2V-A14B = two 14B
  transformers high/low-noise routed at boundary 0.875; Mojo loads ONE low-noise net).
- Off-by-one timestep (noise σ vs model t differ by 1/1000); wrong shift (1.0 vs WAN 5.0); cache
  schema mismatch (`t5_embed` vs `text_embedding`, flat vs 5D); no text mask; LoRA-A init narrow.
- Export key naming MATCHES (would cross-load into EDv2 sampler); LoRA layer-set + optimizer correct.

### chroma (ref: OneTrainer chroma) — SMOKE HARNESS + 2 CRITICAL
- FIXED_SIGMA_SMOKE=True (real ckpt+cache but fixed t=500/one sample).
- CRITICAL: T5 cache DEGENERATE — every file stores only 2-7 tokens/caption + `t5_attention_mask`
  ignored (Mojo reads t5_seq=shape[1], zero-pads to 512, never reads the mask, sdpa_nomask attends
  ~505 zero rows). Conditions on ~2 tokens of caption. Cache producer broken.
- CRITICAL: LoRA trains+saves MORE modules than OT's `attn,ff.net` filter (includes ff_context txt-MLP
  + single proj_mlp/proj_out) via `save_chroma_lora` instead of the existing `save_chroma_lora_for_layer_filter`.
- HIGH: timestep dist family wrong (OT INVERTED_PARABOLA w=7.7 vs Mojo logit-normal shift 1.15); timestep
  off-by-one (noise σ=(idx+1)/1000 vs model t=idx/1000). MED: LoRA scale 16× (alpha 16 vs OT 1.0, hard-asserted).
- VAE/pack/Box-Muller/MSE/AdamW/clip MATCH.

### qwenimage (ref: EriDiffusion-v2 train_qwenimage) — SMOKE HARNESS, runs on ZEROS
- FIXED_SIGMA_SMOKE=True committed default + NO qwen cache producer + cache dir missing → SYNTHETIC
  ALL-ZERO tokens. Header: "do NOT execute." Only proves loss↓/B-grows on zeros.
- CRITICAL: cache key `txt_embed` hard-coded (no fallback) but the producer writes `text_embedding` → hard fail.
- HIGH: timestep continuous vs discrete; timestep_shift 3.0 vs 1.0. MED: LoRA-A init uniform vs kaiming.
- Target sign/loss/optimizer/clip/LoRA layer-set + export keys (loads at inference)/noise all MATCH.

### sd35 (ref: OneTrainer BaseStableDiffusion3Setup) — SMOKE HARNESS + UNLOADABLE LoRA
- FIXED_SIGMA_SMOKE=True (single-sample/fixed-t); configured cache doesn't exist.
- CRITICAL: LoRA export keys Mojo-internal (`joint_blocks` vs diffusers `transformer_blocks`, fused-QKV
  not split, MLP naming) → WON'T LOAD into diffusers/ComfyUI/any SD3.5 pipeline. No conversion.
- HIGH: timestep embedder off-by-one (idx+1 vs idx). H2: noise PCG vs torch (probe trap).
- Flow-match math, VAE, add_noise, target sign, timestep shift 1.0 (MATCH — no 3.0 bug), loss, optimizer,
  LoRA rank/alpha, CLIP-L+G+T5 context assembly all MATCH.

================================================================================
## REINFORCED CROSS-CUTTING (now that all 14 are in)
================================================================================
- **MOST trainers are FIXED_SIGMA_SMOKE harnesses, not real training paths** — committed default True in
  ideogram4-adjacent + anima, sdxl, flux, chroma, sd35, qwenimage, wan22. They pin one sample + fixed
  timestep; their "loss decreased / PASS" proves the backward chain compiles, NOT learning or parity.
  Real-data behavior is UNEXERCISED. Genuinely-runnable + matching: klein (engine clean), ernie (1 mask
  bug), zimage (1 RoPE bug). **l2p is fail-loud broken; ltx2/wan22 are scaffolds.**
- **Cache producers missing / key-mismatched** for half the fleet: qwenimage (`txt_embed` vs `text_embedding`,
  no producer), sd35 (no cache), flux (empty dir, no producer wired), l2p (`pixel`/`cap_feats` vs
  `latent`/`text_embedding`, aborts), wan22 (`t5_embed` vs `text_embedding`, flat vs 5D), chroma (degenerate
  2-7-token T5 cache + mask ignored). A real run requires building/fixing the per-model cache FIRST.
- **Systemic timestep off-by-one**: noise σ=(idx+1)/1000 but model-timestep t=idx/1000 fed to the
  transformer (chroma, qwenimage, sd35, wan22). The references keep them consistent. Likely the Mojo
  inference convention leaking into training; small but systematic. Probe before fixing.
- **Timestep shift wrong in many**: anima (1.0 vs 3.0), qwenimage (3.0 vs 1.0), wan22 (1.0 vs 5.0),
  l2p (3.0 vs uniform), ltx2 (1.0 vs shifted-stretched), chroma (1.15 vs INVERTED_PARABOLA). sd35/klein/
  flux/zimage/ideogram4(after the 1.5→1.0 fix) = correct.

================================================================================
## RECOMMENDED FIX ORDER (causation probes first — do not assume)
================================================================================
1. ideogram4: drop the json.dumps double-wrap (1 line) + mask padding (indicator=text_mask*3) →
   re-encode a real giger caption through the full stager, run train-forward, compare vs torch ref
   feeding the caption once/unpadded. This is the causation probe for the #1 conditioning defect.
2. ernie: thread per-sample real_len into a text key-mask (cache already has it).
3. Recipe sweep (config-only, no code): per-model LR to preset + add warmup + (sdxl/anima) weight_decay.
4. hidream: gate the gauss-shift loss weight OFF; fix sigma floor 0.001→0.002994.
5. Export keys: reconcile flux save↔load scheme; decide hidream 257 vs 252.
6. flux/anima/sdxl: build a real cache + flip FIXED_SMOKE off → actual loss-trajectory parity.
7. l2p: fix cache-key read + replace proxy head. ltx2/wan22: scaffold → real path (large).

Method that found all of this: per-model auditor reads BOTH sides (Mojo + reference) and quotes
file:line; orchestrator owns all `-O2` compiles for any causation probe. Sampling-during-training
was wired in parallel (ideogram4/chroma/flux/anima/sdxl/qwenimage/sd35 compile clean this session;
klein/ernie/zimage already sampled; hidream/l2p/ltx2/wan22 still need it).
