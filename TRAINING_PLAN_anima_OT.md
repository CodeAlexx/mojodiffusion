# Anima trainer — re-target to OneTrainer reference (OT-faithful, pure Mojo)

> Goal: a WORKING end-to-end Anima LoRA trainer faithful to `/home/alex/OneTrainer-anima-ref`,
> shipped 100% pure Mojo. Torch only as a dev-time parity ORACLE under `parity/` (never shipped).
> Decisions locked (USER 2026-06-02): LoRA scope = **attn-mlp** (already built+gated); milestone =
> **full end-to-end working trainer**; parity = small OT-faithful torch oracle (CosmosTransformer3DModel
> + OT recipe), shipped code pure Mojo.

## What is ALREADY DONE + GATE-VERIFIED (do NOT re-port — [[8M-token waste]] rule)
Block fwd+bwd, 28-block stack fwd+bwd, attn-mlp LoRA step — all parity cos ≥ 0.99999999
(`models/anima/{block,anima_stack,anima_stack_lora,lora_block,weights}.mojo` + `parity/`).
The transformer math == the Cosmos math OT calls (diffusers_to_original is a pure rename).
`training/train_anima_real.mojo` runs a fixed-σ smoke (RC=0). This job is the OT **recipe delta**, not a re-port.

## The 5 OT deltas (vs current inference-flame-anchored trainer)
| # | Delta | OT source (cite) | Current Mojo |
|---|---|---|---|
| 1 | context len **512** | `AnimaModel.py:23 PROMPT_MAX_LENGTH=512`; `AnimaBaseDataLoader.py:38` "(512,1024)" | `ANIMA_MAX_SEQ_LEN=256` |
| 2 | **scale_latents** before flow | `BaseAnimaSetup.py:108`; `AnimaModel.py:233` per-ch (x−mean)·(1/std) | implemented in `train_anima_ot.mojo`; sampler emits scaled latents |
| 3 | OT discrete timestep + σ | `ModelSetupNoiseMixin._get_timestep_discrete`; `_add_noise_discrete` σ=idx/N | sigmoid (rs) |
| 4 | OT MSE flow loss | `_flow_matching_losses` unmasked `mse_loss(pred,target)`; target=noise−scaled_latent | MSE (ok, re-verify) |
| 5 | OT LoRA save keys | `AnimaLoRASaver` raw `transformer_lora.state_dict()` (diffusers names) | kohya `lora_unet_blocks_*` |

VAE constants (delta 2): `/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/config.json`
`latents_mean`/`latents_std` (16 ch). READ from file — never hardcode from memory.
Timestep→model (`BaseAnimaSetup.py:137`): `transformer(timestep=timestep/1000, ...)`.

## Chunks (each = builder→skeptic→bugfix→gate; ONE mojo compile at a time)

### A — Recipe re-target of the LoRA STEP  [smallest, de-risks the oracle]
- S_TXT 256→512 (comptime param into `anima_stack_lora_forward_streamed[...]`); context input [1,512,1024].
- `scale_latents`: per-channel `(lat−mean)·(1/std)` (read 16-ch consts from VAE config) BEFORE flow build.
- OT timestep path: `_get_timestep_discrete` (distribution+bias/weight/shift FROM the Anima preset JSON
  `training_presets/#anima LoRA.json`) → σ=idx/num_train_timesteps; `noisy=noise·σ+scaled·(1−σ)`;
  `target=noise−scaled`; pass `timestep/1000` into the t-embedder sinusoidal.
- LoRA save: OT diffusers-state-dict key naming (mirror `transformer_lora.state_dict()` layout).
- **Gate (parity/anima_ot_step_oracle.py + anima_ot_step_parity.mojo):** torch oracle = installed
  `CosmosTransformer3DModel` (load converted Anima weights) OR the existing triangulated `stack_oracle`
  if conversion is infeasible — builder establishes feasibility as gate-zero. Diff `predicted_flow` + `loss`
  on FIXED (scaled_latent, noise, timestep, frozen ctx[1,512,1024]); cos ≥ 0.999. Non-degenerate inputs.

### B — Qwen-Image VAE ENCODER  [genuinely-new compute; biggest]
- Build `models/vae/qwenimage_encoder.mojo` (mirror `qwenimage_decoder.mojo` + `zimage_encoder.mojo`).
  Causal VAE base_dim=96, z_dim=16, dim_mult [1,2,4,4]. Output 5D [1,16,1,H/8,W/8].
- **Gate:** encode real image → latent std ≈ 0.96 (≈0.85 = HWC/CHW scramble); encode→decode round-trip
  sanity vs existing decoder; if a torch AutoencoderKLQwenImage is reachable, parity the encoder.

### C — Text→context @512  [data path]
- Reuse `models/text_encoder/{qwen3_encoder,t5_encoder}.mojo`. Tokenize prompt with BOTH (max_len 512).
  Run Qwen3 → zero pad positions → `net.llm_adapter` (6 blocks, fwd exists in `anima_dit.mojo` but at
  ANIMA_S_TXT=256 — extend to 512 query tokens; 1D-RoPE handles any len) → context [1,512,1024]. FROZEN.
- Cache per caption (reuse cap_cache pattern). **Gate:** finite + matches a torch Qwen3+adapter oracle at 512.

### D — End-to-end wire + learning verdict
- `prepare`: real image → (B) latent → scale_latents; caption → (C) context. `train`: cache → flow target →
  stack_lora fwd/bwd → global-norm clip → LoRA-AdamW → log loss+grad_norm; reuse validation sampler + lora_save.
- **Learning verdict (NON-NEGOTIABLE):** loss DROPS **and** a validation sample SHIFTS with the LoRA. Never loss alone. (NOT "L2P" — L2P is a separate model in this repo.)

## Prep findings (lead, 2026-06-02 — de-risk B/C)
- **B oracle is authoritative:** installed diffusers 0.38 imports `AutoencoderKLQwenImage` AND weights are on
  disk (`/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/diffusion_pytorch_model.safetensors`,
  config.json same dir). Build the encoder parity vs the REAL torch encoder (not just std≈0.96). `AutoencoderKLWan` = fallback.
- **C is length-agnostic:** `net.llm_adapter` = 6 blocks (cross_attn q/k/v/o_proj 1024²; mlp 1024→4096→1024 WITH bias;
  3 RMSNorms norm_{self_attn,cross_attn,mlp}; q/k/v_norm[64]) + `embed.weight[32128,1024]` (T5-id lookup = the queries).
  NO baked positional/length tensor → 1D-RoPE at runtime → 256→512 is a pure length change. 118 adapter tensors total.

## STATUS 2026-06-02 — ALL CHUNKS DONE + lead-reproduced
- A ✅ recipe gate cos 0.99999999999, loss rel-err 2.9e-7. RoPE bugfix: 3-axis table == live diffusers `CosmosRotaryPosEmbed` cos 1.0.
- B ✅ Qwen-Image VAE encoder cos 0.9999931 vs real torch `AutoencoderKLQwenImage` (byte-identical weights), std-matched.
- C ✅ 512 adapter cos 0.99999999998; real-Qwen3 full path cos 0.9999. (NOT yet wired into prepare — needs a tokenizer-id sidecar.)
- D ✅ end-to-end on REAL VAE latent: fixed-σ loss 1.876→0.677 monotone, 280/280 LoRA-B, sample shift mad 0.377/cos 0.924; real-σ trend −0.114, shift mad 0.252. LEARNING PASS (lead-reproduced).
- HONEST GAPS (post-smoke): D used the captured 256→512-padded context, not the C pipeline; sample-shift proves the LoRA changes output, NOT subject-correct learning (needs a longer run + visual); 1024² full-res not run (smoke S_IMG=256); no skeptic pass on the D integration.

## GAPS 1+2 CLOSED 2026-06-02 (builder)
- **GAP 1 (Chunk C wired into prepare):** caption = "a photo collage grid of a woman with a white cat, a smiling woman, a pink lotus flower, traditional Chinese ink paintings, a swan on a lake, and bowls of food" (the test_256.png teaser collage). HF tokenizer sidecar (Qwen2 38 tok + T5 49 tok @ max_len 512) → `pipeline/anima_text_context.mojo` (real Qwen3-0.6B → zero-pad → `net.llm_adapter`) → context_cond [1,512,1024] finite, mean_abs 0.0449 → `output/anima_gap2/anima_context_mojo.safetensors`. `train_anima_ot.CONTEXT_PATH` now points here (src_tokens=512, used as-is). PROVENANCE: latent = real Qwen-Image VAE encode of test_256.png (prepare cache); context = ENTIRELY Mojo pipeline (Qwen3+adapter). DIFF vs old captured sidecar: cos 0.038 (orthogonal), mean_abs 0.045 vs 0.0026 — wholly different (the old sidecar was a different caption captured at 256 tok).
- **GAP 2 (directional learning proof):** single-image overfit, fixed-σ (idx 500), 20 steps, S_IMG=256, 24.7 s/step, peak temp 64°C, RC=0. Loss 1.783→0.408 (Δ−1.375); first-half mean 1.330→second-half 0.563 (trend −0.767). 280/280 LoRA-B grew, 0 nonfinite. Sample-shift mad 0.365 / cos 0.920. **DIRECTIONAL (Tenet-4): L2(base→target)=109.53, L2(lora→target)=70.57 → lora CLOSER by 38.96 = TRUE.** 3 decoded PNGs (qwenimage_decoder wan21-keys, BF16) under `output/anima_gap2/{base_sample,lora_sample,target}.png`: target = faithful collage; base = near-empty striped field; lora = textured warm field (visibly shifted toward target). VERDICT: LEARNING PASS (all four gates).
- CAVEATS: 20-step overfit (loss had largely plateaued by ~step 24 in a longer run). Two earlier longer runs (90/80-step) were silently SIGKILLed mid-train (~step 27/44, likely host-RAM OOM from per-step List allocations — un-investigated; 20-step is safely under). 4-step Euler sample is too few steps to render the full target image (the lora PNG is a directional bias, not a reproduction) — expected at this step budget. 1024² still out of scope.

## STATUS 2026-06-03 -- Codex operational memory

- Use `/home/alex/OneTrainer-anima-ref` for Anima. Do not use the default
  `/home/alex/OneTrainer` tree for Anima parity decisions.
- OneTrainer Anima trains and samples in scaled latent space:
  `scaled = (raw - mean) * (1/std)`, target `noise - scaled`, training model
  timestep `timestep/1000`, sampler denoise from scaled Gaussian latent noise.
  The Mojo `QwenImageVaeDecoder.decode_wan21_keys` internally applies
  `z / inv_std + mean`, so feed scaled latents directly to decode. Do not
  pre-unscale unless intentionally debugging a non-standard latent.
- `models/anima/anima_text_context.mojo` now mirrors the OneTrainer conditioner
  masking: zero Qwen padding, T5 self-attn mask, Qwen cross-attn mask, and
  zero padded T5 output rows.
- `pipeline/anima_sample_cli.mojo` is the process-separated 1024 LoRA sampler.
  It accepts either a direct `context_cond` safetensors sidecar or the shared
  `serenity.sample_prompts.v1` JSON (`configs/anima_alina_samples.json`), reads
  precomputed context paths, applies PEFT LoRA as an overlay, and writes a
  scaled latent for fresh-process VAE decode.
- `training/train_anima_ot.mojo` now uses the mandatory Mojo progress display
  and writes PEFT plus optimizer-state sidecars for faithful resume. Remaining
  production proof is a completed long dataset run plus the standard 10-step
  save/resume-to-25 smoke.

## Discipline
- Pure Mojo shipped; torch only under `parity/`, run via `/home/alex/serenityflow-v2/.venv/bin/python` (or OT venv).
- Non-degenerate test data (sinusoidal/random), real H=16/Dh=128. `rm -f serenitymojo.mojopkg` before each serial build.
- Lead re-runs every agent gate on a clean build before trusting it. mojo-syntax skill for all Mojo (0.26.x: inout→mut/out, alias→comptime).
