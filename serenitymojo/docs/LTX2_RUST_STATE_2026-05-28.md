# LTX2 Rust State — Pre-Port Mapping (Planner A)
**Date:** 2026-05-28
**Scope:** Map current Rust LTX-2.3 inference state under `/home/alex/EriDiffusion/inference-flame/` to enable a 3-team Mojo port. READ-ONLY pre-port analysis.
**Companion:** Planner B (upstream architecture + checkpoint layout).

Target model: **LTX-2.3 22B** (Lightricks), audio+video joint DiT. NOT the 0.9.x LTX-Video line.

Primary sources read:
- `/home/alex/EriDiffusion/AUDIT_2026-05-03_LTX2_INFERENCE.md` (definitive inference audit)
- `/home/alex/EriDiffusion/inference-flame/LTX2_FULL_PARITY_PLAN.md` (Phase 1-4 parity status)
- `/home/alex/EriDiffusion/inference-flame/LTX_FEATURE_PARITY_PLAN.md`
- `/home/alex/EriDiffusion/inference-flame/LTX_FEATURE_PARITY.md` (Lightricks reference feature catalog)
- `/home/alex/EriDiffusion/inference-flame/alexltx.md` (4-bug fix history + arch notes)

Note: `AUDIT_LTX_2026-05-25.md`, `SKEPTIC_LTX_2026-05-25.md`, `FIXES_LTX_2026-05-25.md` are **trainer-side** audits for `train_ltx2.rs` (EriDiffusion-v2) — out of scope for this port; not cited as inference state.

---

## 1. Component inventory

Status legend:
- `parity-gated`: has dedicated `*_parity.rs` bin and a Python reference; AUDIT or PARITY_PLAN cites measured cos_sim / max_abs
- `compiles-only`: built and used in pipeline bins but no standalone parity test
- `partial-parity`: parity bin exists but with known caveats / failing layers
- `flagged-buggy`: audit explicitly calls out broken behavior
- `missing`: referenced but not implemented

| Component | Rust file | Lines | Parity status | Notes |
|---|---|---:|---|---|
| Video VAE decoder | `src/vae/ltx2_vae.rs` | 535 | `parity-gated` | Per-frame `cudnn_conv2d_bf16` (dodges Conv3d landmine) via `ltx2_video_vae_parity.rs` (AUDIT §5, §10). Has `#[deprecated]` `decode_legacy`. ~50 CausalConv3d instances × kD=3 launches/frame = perf debt. |
| Video VAE encoder | `src/vae/ltx2_encoder.rs` | 625 | `parity-gated` | Mirror of decoder, `ltx2_vae_encode_parity.rs` exists. Channel-expand last-channel×127 → 256, then `narrow(1,0,128)` for deterministic mean (audit:121). |
| Audio VAE | `src/vae/ltx2_audio_vae.rs` | 590 | `parity-gated` | `ltx2_audio_vae_parity.rs` ↔ `scripts/ltx2_audio_vae_decode_ref.py`. |
| Vocoder (BigVGAN-style) | `src/vae/ltx2_vocoder.rs` | 1119 | `parity-gated` | `ltx2_vocoder_parity.rs` ↔ `scripts/ltx2_vocoder_ref.py`. Largest VAE-side file. |
| Text conditioning / I2V mask | `src/models/ltx2_conditioning.rs` | 605 | `parity-gated` | `ltx2_conditioning_parity.rs` ↔ `ltx2_conditioning_mask_ref.py`. **Phase 3A done**: multi-keyframe masks bit-exact, latents at FMA noise; `image_cond_noise_scale` max_abs=3.58e-7 (PARITY_PLAN §Phase 3). |
| DiT (full transformer) | `src/models/ltx2_model.rs` | **5552** | `partial-parity` | `LTX2StreamingModel` + `LTX2TransformerBlock` + `LTX2Attention`. 48 dual-stream blocks. `ltx2_validate.rs` runs block-by-block vs `/home/alex/ltx2-refs/` but tolerance is **200.0 max_abs** (AUDIT finding 20 — "the threshold is enormous"). Structurally matches Lightricks but RoPE freq build is host-CPU (finding 5); RMSNorm forced to F32-slow path because BF16 fast kernel is buggy (finding 8). |
| Temporal upsampler | `src/models/ltx2_temporal_upsampler.rs` | 340 | `missing` (effectively) | File exists (340 lines) but AUDIT finding 15 says "not implemented" — no binary loads `ltx-2.3-temporal-upscaler-x2-1.0.safetensors`. May be stubbed/skeleton. |
| Spatial upsampler 2× | `src/models/ltx2_upsampler.rs` | 404 | `parity-gated` | `ltx2_latent_upsampler_parity.rs`: cos_sim=0.999951, max_abs=4.7e-2 (PARITY_PLAN §Phase 4). |
| Sigma scheduler | `src/sampling/ltx2_sampling.rs` | 225 | `partial-parity` + `flagged-buggy` | `LTX2_DISTILLED_SIGMAS` (8-step) + `LTX2_STAGE2_DISTILLED_SIGMAS` (3-step) — bit-exact to distilled checkpoint. `linear_quadratic_schedule` parity-verified (max_abs=0.0 for n=8/20/25/30). **But**: `build_dev_sigma_schedule` is a Flux-style exponential, NOT Lightricks's `linear_quadratic`, and `ltx2_generate.rs` calls the wrong one (AUDIT findings 1, 2, 22). `FlowMatchEulerDiscreteScheduler` (LTX-2 dev/0.9.8 default) is **not implemented**. |
| Guidance (CFG-star, STG) | `src/sampling/ltx2_guidance.rs` | 258 | `parity-gated` | `ltx2_guidance_parity.rs` PASS max_abs=0.0 BF16 for `cfg_star_rescale`, `build_skip_layer_mask`, `stg_rescale` (PARITY_PLAN §Phase 2). |
| Multiscale (AdaIN + tone map) | `src/sampling/ltx2_multiscale.rs` | 152 | `parity-gated` | `ltx2_adain_tone_map_parity.rs`: `adain_filter_latent` cos_sim=0.999991 @ factor=1.0; `tone_map_latents` cos_sim=0.999994 @ compression=1.0. F32-internal. |

**Totals:** 11 LTX2 source files. By status: 8 parity-gated, 1 partial-parity DiT (with weak threshold), 1 flagged-buggy scheduler (wrong dev fallback), 1 effectively-missing (temporal upsampler).

Supporting files (not LTX2-prefixed but required):
- `src/models/gemma3_encoder.rs` (~600 lines): Gemma-3 12B text encoder; loaded by Phase 2 negative prompt path. AUDIT layer-3 parity FAILS at cos_sim ≈ 0 vs Lightricks's `ltx_core.text_encoders.gemma.encode_text` (PARITY_PLAN §Phase 2 known gap).
- `src/models/feature_extractor.rs` (~150 lines): hidden-states → packed embed → `video_aggregate_embed` projection → 4096-dim context.
- `src/models/fp8_resident.rs` (~800 lines): GPU-resident FP8 weights. AUDIT finding 19 — possibly dead code per `alexltx.md`, superseded by flame-swap v2.
- `src/models/lora_loader.rs` (340 lines): LoRA fusion math; parity PASS 12/12 @ 0.999999 (PARITY_PLAN §Phase 1).

---

## 2. Dependency graph

```
ltx2_full_pipeline (bin, most complete)
ltx2_two_stage (bin, no audio decode)
  ├─ LTX2StreamingModel (models/ltx2_model.rs)
  │   ├─ Gemma3Encoder + feature_extractor (text → 4096-d context via video_aggregate_embed)
  │   │   └─ text_embedding_projection.video_aggregate_embed (loaded from LTX_CHECKPOINT=dev .safetensors)
  │   ├─ audio_aggregate_embed (text → 2048-d audio context, same source file)
  │   ├─ ltx2_conditioning (I2V / multi-keyframe mask + lerp; Phase 3A)
  │   ├─ AdaLayerNormSingle (timestep + prompt_timestep — both audio & video)
  │   ├─ LTX2TransformerBlock × 48 (dual-stream)
  │   │   ├─ LTX2Attention (video self-attn, audio self-attn, video CA, audio CA, A2V, V2A)
  │   │   │   ├─ RMSNorm Q/K (F32 slow path — finding 8)
  │   │   │   ├─ compute_rope_frequencies + apply_rotary_emb (BF16-fused; freq table host-CPU)
  │   │   │   └─ wmma flash attention (flame-core 74d7043; d∈{64,96,128})
  │   │   ├─ GELU-tanh FFN × 4
  │   │   ├─ scale_shift_table (9-param, indices 0-5 self/FFN, 6-8 CA Q-mod)
  │   │   ├─ prompt_scale_shift_table + audio_prompt_scale_shift_table (CA KV-mod; fix bcf9d00)
  │   │   └─ LoRA fusion via lora_loader (Phase 1 parity)
  │   ├─ Spatial upsampler 2× (models/ltx2_upsampler.rs, only between stages)
  │   └─ flame-swap v2 (block streamer; non-owning GPU views)
  ├─ Sampler (sampling/ltx2_sampling.rs — distilled 8+3 sigma tables)
  ├─ Guidance (sampling/ltx2_guidance.rs — CFG/CFG-star/STG; Phase 2 complete)
  └─ Video/Audio VAE + Vocoder (vae/ltx2_{vae,encoder,audio_vae,vocoder}.rs)

ltx2_generate (T2V single-stage CFG)
  └─ Same DiT path, BUT calls build_dev_sigma_schedule (BUGGY — AUDIT finding 2)

ltx2_generate_av / ltx2_generate_ms / ltx2_generate_kf / ltx2_i2v_gen
  └─ Variants on the above; each adds: AV joint, multi-scale, keyframes, I2V
```

---

## 3. Two-stage pipeline summary

Source of truth: `src/bin/ltx2_two_stage.rs` (539 lines) + `src/bin/ltx2_full_pipeline.rs` (484 lines). Pipeline mirrors `LTX-2/packages/ltx-pipelines/src/ltx_pipelines/distilled.py` (alexltx.md:383).

### Stage 1: half-resolution AV denoise (8 steps)
- **Input shapes** (default 512×320 target):
  - Video latent: `[B, 128, F_lat, 10, 8]` at half res (256×160) for `vae_scale_factor=[8,32,32]`, NUM_FRAMES=257 → F_lat=33
  - Audio latent: `[B, 128, T_audio]` (audio channels 128)
- **Conditioning**: Gemma-3 12B → `feature_extractor` → `video_aggregate_embed` (4096-dim) + `audio_aggregate_embed` (2048-dim). Both projections load from **LTX_CHECKPOINT** (`ltx-2.3-22b-dev.safetensors`), even though DiT loads from **MODEL_PATH** (`ltx-2.3-22b-distilled.safetensors`) — intentional dual-checkpoint pattern (AUDIT finding 4).
- **Forward**: 8 steps via `LTX2_DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]`.
- **Sampler**: Euler velocity step. Distilled: `guidance_scale=1, stg_scale=0` → no CFG, no STG, 1 forward/step. **8 forwards stage 1.**
- **Output**: stage-1 video latents (still at half res, denoised).

### Stage-boundary: spatial upsample 2× + AdaIN
- `un_normalize_latents` → `LTX2LatentUpsampler.forward(x)` (`models/ltx2_upsampler.rs`) → `adain_filter_latent` (match per-channel mean/std to stage-1 reference, factor=1.0) → `normalize_latents` again for stage-2 input.
- Audio passes through unchanged.
- New shape: `[B, 128, F_lat, 20, 16]` (spatial doubled).
- Stage-2 noise injection: `latent = (1-σ)·upscaled + σ·noise` where `σ = LTX2_STAGE2_DISTILLED_SIGMAS[0] = 0.909375` — **intentional 91% noise** (alexltx.md:216).

### Stage 2: full-resolution AV refine (3 steps)
- Same DiT, 3 steps via `LTX2_STAGE2_DISTILLED_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]`. **3 forwards stage 2.**
- Optional `tone_map_latents(compression=0.6)` (distilled default per Lightricks; flagged in PARITY_PLAN Phase 4 ⏸️ "deferred decode-time noise injection").

### Outputs
- `output/ltx2_twostage_video_latents.safetensors`
- `output/ltx2_twostage_audio_latents.safetensors`

### Decode
- **In Rust**: `ltx2_decode_av.rs` runs video VAE + audio VAE + vocoder → mp4 mux.
- **Currently in production via Python**: `decode_latents_av.py` calls Lightricks's reference (`LTX-2/packages/ltx-trainer/...`). `alexltx.md:159` shows the active path uses Python decode.

Total compute: **11 DiT forwards** for the full two-stage (8 stage-1 + 3 stage-2), distilled. Dev path would be ~120 (3 conds × 40 steps with CFG+STG), but dev sampling **does not work** (AUDIT finding 1).

---

## 4. Parity gate inventory

| Bin | Tests what | Reference Python script | Status per audit/parity-plan docs |
|---|---|---|---|
| `ltx2_video_vae_parity.rs` | Video VAE decoder | `ltx2_video_vae_decode_ref.py` | Parity-gated (AUDIT §"Architecture parity" row "VAE" = "matches"). |
| `ltx2_vae_encode_parity.rs` | Video VAE encoder | `ltx2_vae_encode_parity.py` | Parity-gated. |
| `ltx2_audio_vae_parity.rs` | Audio VAE | `ltx2_audio_vae_decode_ref.py` | Parity-gated; "AV is the product — audio is never optional" (PARITY_PLAN). |
| `ltx2_vocoder_parity.rs` | Vocoder | `ltx2_vocoder_ref.py` | Parity-gated. |
| `ltx2_sigma_parity.rs` | `linear_quadratic_schedule` | `ltx2_sigma_schedule_ref.py` | **PASS max_abs=0.0** for n=8/20/25/30 (PARITY_PLAN §Phase 1). Note: orphan code — no T2V bin actually uses it (AUDIT finding 22). |
| `ltx2_conditioning_parity.rs` | Multi-keyframe mask + lerp | `ltx2_conditioning_mask_ref.py` | **Phase 3A done**: masks bit-exact, latents at FMA noise; `image_cond_noise_scale` max_abs=3.58e-7. |
| `ltx2_neg_prompt_parity.rs` | Negative prompt encoding (Gemma+FE) | `ltx2_neg_prompt_ref.py` | **PARTIAL FAIL** — Layer 3 cos_sim ≈ 0 vs Lightricks's private `ltx_core.text_encoders.gemma.encode_text`. PARITY_PLAN §Phase 2 known gap: "self-consistent but absolute embedding differs". Acknowledged as "multi-session scope". |
| `ltx2_guidance_parity.rs` | `cfg_star_rescale`, STG mask, STG rescale | `ltx2_cfg_star_ref.py`, `ltx2_stg_mask_ref.py`, `ltx2_stg_rescale_ref.py` | **PASS max_abs=0.0 BF16** for all three (PARITY_PLAN §Phase 2). |
| `ltx2_latent_upsampler_parity.rs` | Spatial 2× upsampler | `ltx2_latent_upsampler_ref.py` | **PASS cos_sim=0.999951, max_abs=4.7e-2** (PARITY_PLAN §Phase 4). |
| `ltx2_adain_tone_map_parity.rs` | AdaIN + tone-map | `ltx2_adain_tone_map_ref.py` | **PASS cos_sim=0.999991/0.999994**; bit-exact at factor=0.0. |
| `ltx2_validate.rs` | Full 48-block DiT vs `/home/alex/ltx2-refs/` dumps | `ltx2_dit_forward_parity_ref.py` + `ltx2_dit_forward_parity_cmp.py` + `ltx2_dit_parity_inputs.py` | **Threshold is `output_err < 200.0` per block** — AUDIT finding 20 calls this "enormous", "could mask catastrophic block-level corruption". p95<1.0 is printed but NOT asserted. |
| `ltx2_fp8_stream_parity.rs` | FP8 dequant via flame-swap | (none listed) | Coverage uncertain. |
| `ltx2_connector_validate.rs` | text→video/audio aggregate projections | (uses `ltx2_av_block0_parity.py`) | Connector matches per AUDIT §"Architecture parity". |
| `ltx2_lora_wiring_check.rs` | LoRA wiring into AV BlockOffloader path | (none listed) | Smoke test; passes (PARITY_PLAN §Phase 1). |
| `ltx2_lora_fusion_correctness.rs` | bit-exact LoRA fusion video+audio | `lora_fusion_parity_ref.py` | **PASS bit-exact** disk-sync + BlockOffloader (PARITY_PLAN §Phase 1). |

### Parity infrastructure gaps
- **No standalone full-DiT-block parity bin with tight tolerance.** `ltx2_validate.rs` exists but at max_abs=200 it's a smoke filter, not a parity gate.
- **No scheduler parity bin for `FlowMatchEulerDiscreteScheduler`** (the LTX-2 dev sampler). This is the #1 critical gap per AUDIT.

---

## 5. Known gaps / incomplete pieces

Quoted verdicts from AUDIT / PARITY_PLAN. Highest impact first.

1. **`FlowMatchEulerDiscreteScheduler` not implemented** — AUDIT finding 1 (Critical): "LTX-2/LTX-2.3 default scheduler is `FlowMatchEulerDiscreteScheduler` with `use_dynamic_shifting=True, time_shift_type='exponential', base_shift=0.95, max_shift=2.05, shift_terminal=0.1`. The Rust port has none of these primitives — no exponential time-shift with `mu = m*tokens + b`, no `shift_terminal` stretching, no Karras/Beta paths." Means **dev-mode (40-step) sampling cannot be correct**, only distilled works.

2. **`ltx2_generate.rs` calls the WRONG sigma function** — AUDIT finding 2 (Critical): uses `build_dev_sigma_schedule(NUM_STEPS, num_tokens, 0.5, 1.15, 0.0)` — a Flux-style exponential whose doc-comment self-confesses "not what Lightricks uses". Constants `base_shift=0.5, max_shift=1.15` don't match LTX-2 config (`0.95, 2.05`). 1-line fix to call `linear_quadratic_schedule` instead but not done (AUDIT finding 22 — "orphaned correct code").

3. **Temporal upsampler not wired** — AUDIT finding 15 (Medium): `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` exists on disk but no binary loads it. The `src/models/ltx2_temporal_upsampler.rs` (340 lines) exists but is not used by any pipeline. Effectively missing for end-to-end use.

4. **Gemma3 encoder ≠ Lightricks's encoder** — PARITY_PLAN §Phase 2 known gap: `ltx2_neg_prompt_parity` Layer 3 fails at cos_sim ≈ 0. "Our in-tree `Gemma3Encoder` + `feature_extractor` outputs differ wildly from Lightricks's private `ltx_core.text_encoders.gemma.encode_text`. Not BF16 noise — different hidden-state layer / chat template / tokenizer." Self-consistent (pos and neg encoded same way) so CFG works, but **absolute embedding diverges from Lightricks**.

5. **No tokenizer in Rust** — alexltx.md:192: "There is currently no tokenizer in the Rust binary — Gemma tokenization is done offline in Python and dumped as JSON" (`/home/alex/ltx2-refs/bench20/tokens_*.json`). Users must pre-tokenize new prompts.

6. **`ltx2_validate.rs` block threshold = 200.0** — AUDIT finding 20: "the threshold is so loose that catastrophic block-level corruption could pass." p95<1.0 printed but not asserted.

7. **VAE per-frame Conv2d perf debt** — AUDIT finding 10: "~387 launches per CausalConv3d instance, and the decoder has ~50+ such convs." Correctness OK; perf workaround for the `cat→Conv3d` contiguity bug in flame-core.

8. **RMSNorm forced to F32-slow path** — AUDIT finding 8: "REVERTED to slow F32 manual path for diagnosis: the BF16 fast kernel was producing slightly different values from PyTorch." Compounds across 48 blocks.

9. **FP8 inference correctness uncertain** — AUDIT finding 19: Several bins (`ltx2_i2v_gen`, `ltx2_generate`, `ltx2_lora_*`) load `ltx-2.3-22b-distilled-fp8.safetensors`. `ltx2_two_stage.rs:22-31` warns FP8 produces gray output without scale-aware dequant. Dequant happens in `models/fp8_resident.rs` but AUDIT explicitly **did not extend coverage to FP8**.

10. **`ltx2_two_stage.rs` uses unsafe raw `cudaMemPoolTrimTo`** — AUDIT finding 21: bypasses flame-core helper that exists. Cosmetic inconsistency.

11. **`decode_timestep` + `decode_noise_scale` not wired** — PARITY_PLAN §Phase 4 ⏸️ deferred: timestep-conditioned VAE decoder primitive missing. Defaults `t=0.05, s=0.025` in Lightricks.

12. **Prompt enhancement not ported** — PARITY_PLAN §Phase 5 deferred (Florence-2 + Llama-3.2-3B). Not core for video gen but Lightricks default for short prompts.

13. **TeaCache not ported** — PARITY_PLAN §Phase 5 deferred.

The user's "Rust is incomplete" framing is borne out by gaps 1-5. The distilled-only fast path works; dev-mode 40-step sampling does not.

---

## 6. Mojo port team-split recommendation

Goal: 3 teams, minimal file overlap, each with a known-good Rust parity reference for a "block 0 / first chunk" scope. Recommended NOT to ship full LTX-2 in one campaign — gate on distilled-only stage-1 video first.

### Team 1 — DiT core (text → video latents, distilled stage-1 only)
**Owns:** `models/ltx2_model.rs` port → Mojo. Subset for first chunk: `LTX2TransformerBlock` (single block, video-only path), `LTX2Attention`, RoPE helpers, AdaLayerNormSingle, the 9-param `scale_shift_table`. Defer A2V/V2A, audio stream, CA-KV modulation until parity gate on video self-attn + video CA + FFN passes.
**Parity oracle:** `ltx2_validate.rs` outputs (tighten threshold to max_abs<5.0 or cos_sim≥0.999 against `/home/alex/ltx2-refs/` block dumps). Block-0 first.
**Caveats:**
- Rust uses F32 RMSNorm slow path (AUDIT finding 8) — Mojo can either replicate or fix in serenitymojo's RMSNorm kernel.
- RoPE freq table host-CPU loop in Rust — Mojo can do it on-GPU from day one.
- 48-block streaming/offload pattern (flame-swap v2) needs a Mojo analog — serenitymojo has `offload/block_loader.mojo` per repo gitStatus. Reuse.
- DO NOT scope dev-mode sampling — Rust has no working dev scheduler to gate against.

### Team 2 — VAE + Audio path (vae/* + vocoder)
**Owns:** `vae/ltx2_vae.rs` + `vae/ltx2_encoder.rs` + `vae/ltx2_audio_vae.rs` + `vae/ltx2_vocoder.rs` ports. Total ~2,869 lines.
**Parity oracle:** All four have parity bins (`ltx2_video_vae_parity`, `ltx2_vae_encode_parity`, `ltx2_audio_vae_parity`, `ltx2_vocoder_parity`) and Python references. These are the **safest, most parity-gated** Rust pieces.
**Caveats:**
- Rust uses per-frame Conv2d (AUDIT finding 10) to dodge a flame-core contiguity bug. In Mojo, can attempt single Conv3d from day one (serenitymojo has `vae/conv3d.mojo` per gitStatus) — but **gate on parity vs the same Python references** to catch divergence.
- Vocoder is 1119 lines (largest VAE file) — consider it a separate sub-deliverable inside Team 2's chunk plan.
- Audio path is "never optional" (PARITY_PLAN audio invariant) — Team 2 must ship audio VAE and vocoder together, not video-VAE-only.

### Team 3 — Sampler + guidance + conditioning + multiscale (orchestration)
**Owns:** `sampling/ltx2_sampling.rs`, `sampling/ltx2_guidance.rs`, `sampling/ltx2_multiscale.rs`, `models/ltx2_conditioning.rs`, `models/ltx2_upsampler.rs`, plus the two-stage pipeline binary (`pipeline/ltx2_pipeline_smoke.mojo` to-be-created).
**Parity oracles:**
- Distilled sigma tables: bit-exact constants (no kernel parity needed).
- `linear_quadratic_schedule`: PASS max_abs=0.0 in Rust.
- Guidance: `ltx2_guidance_parity` PASS max_abs=0.0 BF16 (cfg_star, STG mask, STG rescale).
- Conditioning: `ltx2_conditioning_parity` Phase 3A done.
- Spatial upsampler: `ltx2_latent_upsampler_parity` cos_sim=0.999951.
- AdaIN + tone-map: `ltx2_adain_tone_map_parity` cos_sim=0.999991+.
**Caveats:**
- **Skip `FlowMatchEulerDiscreteScheduler` for Block 0** — Rust doesn't have it, so no parity oracle exists. Distilled-only sampling is the gate.
- Skip `build_dev_sigma_schedule` entirely — it's flagged-buggy in Rust (AUDIT finding 2). Use `linear_quadratic_schedule` (parity-clean) and the distilled sigma tables.
- Temporal upsampler is effectively unimplemented in Rust → defer in Mojo too; document as gap.
- This team also owns the two-stage pipeline orchestration smoke (boundary noise injection σ=0.909375, AdaIN bracketing, etc.).

### Cross-team coordination
- **Text encoder (Gemma-3) is out of all three teams' scope for the first chunk**. AUDIT/PARITY_PLAN show it's the one parity oracle that's known-broken in Rust (Layer 3 cos_sim≈0). Use Rust's offline tokenized-prompt JSON path (`/home/alex/ltx2-refs/bench20/tokens_*.json`) + reproduce the projected 4096-d/2048-d embeddings as test fixtures. Defer Gemma3 Mojo port until LTX-2 stage-1 video works end-to-end.
- **Tokenizer is also deferred** (alexltx.md:192).
- **FP8 path is deferred** (correctness uncertain in Rust per AUDIT finding 19). Use BF16 distilled checkpoint for parity (`ltx-2.3-22b-distilled.safetensors`, 27.3 GB).

### Block 0 deliverable per team
- Team 1: One transformer block (video stream only) forward parity vs Rust `ltx2_validate` block-0 dump at cos_sim≥0.999 on BF16.
- Team 2: Video VAE decoder forward parity vs `ltx2_video_vae_parity` Python ref at cos_sim≥0.999 BF16; audio VAE encode/decode round-trip cos_sim≥0.999.
- Team 3: Distilled sigma table emission (bit-exact) + `linear_quadratic_schedule` parity (max_abs=0) + `cfg_star_rescale` parity (max_abs=0 BF16). All three are 1-day deliverables in Rust; Mojo should match.

---

## 7. Open questions for the main agent

1. **Distilled-only vs dev-mode scope for Mojo port.** Rust dev sampling is broken (no `FlowMatchEulerDiscreteScheduler`). Should Mojo also be distilled-only for first delivery, or implement the scheduler from upstream Python ref (Planner B's territory)? Recommendation: distilled-only.
2. **Audio path in scope for Block 0?** PARITY_PLAN says "audio is never optional" (invariant), but the simplest first chunk would be video-only. Tension between Lightricks/Rust convention and chunked-port methodology. If audio is mandatory from chunk 1, Team 1 must port `forward_audio_video` (not just `forward_video_only`) — much larger scope.
3. **LoRA in scope?** Phase 1 done in Rust (bit-exact). Likely defer to chunk 2; needs explicit decision.
4. **FP8 in scope?** Rust ships FP8 distilled but correctness is unverified (AUDIT finding 19). Recommendation: defer; require BF16 checkpoint for parity.
5. **Decode-time noise injection (`decode_timestep`, `decode_noise_scale`)** deferred in Rust Phase 4. Required for byte-exact Lightricks output. Include in Mojo or defer same?
6. **Tokenizer port.** Rust uses offline JSON. Does Mojo continue that pattern, or does the port include a real Gemma tokenizer? (Note: serenitymojo has `tokenizer/tokenizer.mojo` per gitStatus — check if Gemma-compatible.)
7. **`ltx2_validate.rs` 200.0 tolerance.** Is 200.0 max_abs acceptable as a Mojo parity gate too, or should the port hold itself to cos_sim≥0.999 / max_abs<5 (which Rust may not even pass currently)?
8. **Temporal upsampler.** Skip (matches Rust state) or implement (close a Rust gap as part of porting)?
9. **Two checkpoint files (distilled DiT + dev for connector).** AUDIT finding 4: pipeline loads DiT from `ltx-2.3-22b-distilled.safetensors` AND `text_embedding_projection.video_aggregate_embed` from `ltx-2.3-22b-dev.safetensors`. Mojo must implement the same dual-checkpoint load. Confirm both files are on the build machine.
10. **Existing serenitymojo modules to reuse vs port-from-Rust.** gitStatus shows recent edits in `ops/attention.mojo`, `ops/rope.mojo`, `ops/linear.mojo`, `models/vae/conv3d.mojo` — are these already LTX-compatible or do they need extension?

---

*End of Planner A report. Planner B covers upstream architecture + checkpoint layout. Together they should give the main agent enough to dispatch 3 builder+skeptic+bugfix loops via `/mojo-port`.*
