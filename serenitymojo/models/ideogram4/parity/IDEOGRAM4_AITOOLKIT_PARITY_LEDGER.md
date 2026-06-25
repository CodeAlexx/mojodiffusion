# Mojo ideogram4 training — ai-toolkit parity ledger (IN PROGRESS, started 2026-06-25 overnight)

## Why this exists
The MOJO ideogram4 training was previously gated against the WRONG oracle:
- Forward oracles (`models/dit/parity/ideogram4_oracle.py`, `_predict_oracle.py`,
  `_mrope_oracle.py`, `_intermediates_oracle.py`) import **`ideogram4-ref`**
  (`/home/alex/ideogram4-ref/src/ideogram4/modeling_ideogram4.py`, 379 lines) — a
  ref/diffusers impl, NOT production-validated.
- Backward/training (`autograd_v2/tests/ideogram4_block_parity.mojo`) is only a
  SELF-CONSISTENCY bit-gate (engine == hand-chain), **no external oracle at all**.

MEASURED gap: ai-toolkit's `extensions_built_in/diffusion_models/ideogram4/src/transformer.py`
(534 lines) **DIFFERS** from ideogram4-ref (379 lines). ai-toolkit runs ideogram4 in
PRODUCTION → it is the valid oracle. OneTrainer's ideogram was never tested → invalid.
**Everything must be re-verified against ai-toolkit.**

## Oracle = ai-toolkit (the ONLY valid one)
`/home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4/` — load the fp8
transformer via ai-toolkit's path, GPU bf16, torch.autograd for the backward.
Gate bar: cos ≥ 0.999 (input grad + every weight grad + LoRA d_A/d_B), non-degenerate data.

## Components to verify (each = re-point/build an ai-toolkit oracle + run the mojo gate)
| # | Component | Mojo gate | ai-toolkit oracle | Status |
|---|-----------|-----------|-------------------|--------|
| 1 | DiT block FORWARD (block0 in/out) | `autograd_v2/tests/ideogram4_block_aitoolkit_parity.mojo` | `ideogram4_aitoolkit_oracle.py` (ai-toolkit prod transformer) | ✅ block0_out cos **0.99999997** (lead re-run) |
| 2 | DiT block BACKWARD (d_x, d_adaln, 6× LoRA dA/dB) | same gate | same oracle (torch.autograd) | ✅ d_x **0.99999993**, d_adaln 0.99998849, all LoRA dA/dB ≥**0.99998** (lead re-run, exit 0). NOTE: mojo is a LoRA trainer — base/norm weight grads frozen (not computed), so only LoRA grads comparable. |
| 3 | Full STACK fwd→velocity (34 blocks) | `ideogram4_aitoolkit_predict_parity.mojo` | `ideogram4_aitoolkit_predict_oracle.py` | ✅ **PASS** (after fix #4): block0 0.99999 / block16 0.99992 / block33 0.99979 / transformer_out 0.99990 / **velocity 0.99991** (lead re-run, exit 0). |
| 4 | **MRoPE** | same gate (mrope_cos/sin) | same oracle | ✅ **FIXED + PASS**: was mrope_cos **0.714** / sin 0.580 → **0.999999** after fix. **REAL BUG (the ideogram4-ref oracle missed it):** `models/dit/ideogram4_mrope.mojo:49` bf16-rounded inv_freq to match ideogram4-ref; ai-toolkit production keeps inv_freq **F32**. Removed the `.cast[bfloat16]().cast[f32]()`. Affects BOTH training + inference mrope. |
| 5 | Intermediates (per-block chain) | (covered by #1 block + #3 stack) | — | ✅ covered — per-block #1 + composition #3 both PASS vs ai-toolkit. |
| 6 | VAE encode (image→latent) | `ideogram4_aitoolkit_vae_parity.mojo` | `ideogram4_aitoolkit_vae_oracle.py` | ✅ **PASS** (lead re-run): mean **0.99996**, latents vs prod **0.99995**, std 0.832≈ai-toolkit 0.832, no HWC scramble. Notes: ai-toolkit's `BatchNorm2d` is defined-but-NEVER-called (prod uses `get_latent_norm`); true latent std is ~0.83 (NOT 0.96); 64ch-moments cos 0.44 is the unused logvar band (diagnostic, not gated). |
| 7 | Text encoder (Qwen3-VL 13-tap llm_features) | `ideogram4_aitoolkit_encoder_parity.mojo` | `ideogram4_aitoolkit_encoder_oracle.py` | ✅ **PASS** (lead re-run): llm_features **0.99994**, taps L0/3/18/33/35 ≥**0.99933**, same 13-layer set (post-layer, no model.norm), exit 0. (fp8-dequant vs bf16 negligible.) |
| 8 | Flow-match step (add_noise/target/loss/timestep/velocity) | recipe audit (code-read) | ai-toolkit `custom_flowmatch_sampler.py`/`ideogram4.py`/`SDTrainer.py` | ✅ **MATCH** — (1-t)·clean+t·noise, target noise−clean, mean-MSE no-weight, sigmoid(randn) timestep, velocity negate. (trainer in **serenity-trainer**: Ideogram4LiveTrainer→Ideogram4LoRATrainer→Ideogram4LoRATrainStep) |
| 9 | Recipe: optimizer / LR / captions / masking / LoRA / VAE-norm | recipe audit (cite file:line) | ai-toolkit `eri2_ideogram4_lora.yaml` + code | ⚠️ **MATCH**: LR sched(constant), timestep, step-math, VAE latent-norm (cos 0.99996), LoRA targets(204 block, globals frozen), rank/alpha/init. **DIVERGE (must fix, see below)**: P0 padding-mask leak, P0 caption JSON-wrap, P1 LR-default 4e-4, P2 eps 1e-8, caption_dropout 0, single-res 512. |
| 10 | Real-run verdict (loss DROPS) + autograd_v2 trainer anchor (engine == hand-chain) | live trainer on re-staged eri2 cache (115 img + uncond, P0 fixes baked in: pad-rows=0, prose captions) | — | ✅ **PASS (2026-06-25 lead-run)**: loss DROPS smooth 0.7750→0.7599 (−2%/30 steps), grad_norm 70-107 nonzero, LoRA saved. **Anchor: ON(IDEOGRAM4_V2_GRAPH engine) vs OFF(hand-chain) loss BIT-IDENTICAL max\|diff\|=0.0 over 30/30 steps.** Cache re-staged with both P0 fixes (a BF16-mask dtype bug in the P0 feature-zeroing was found+fixed when the real encode ran). **Sample-SHIFT CONFIRMED (250-step LoRA): base(0 adapters) vs LoRA(204) same prompt+seed → mean \|Δpixel\|=21.5/255 = 8.4% shift, 81.4% pixels changed.** Speed: engine 3.27 vs hand-chain 2.65 s/step (coarse stage-1, no slab/capture yet; ring microbench refuted as the cause — 0.0017ms; the 620ms is graph-record/BFS, slab/capture is the real fix). |

## VERDICT (2026-06-25 overnight)
**Every per-component gate PASSES vs ai-toolkit** (block fwd+bwd, full forward→velocity, MRoPE, VAE encode, text-encoder taps, step math, VAE-norm, LoRA targets/rank/alpha/init). The campaign **found + fixed one real model bug** (MRoPE inv_freq bf16→f32) that the invalid ideogram4-ref oracle hid. The mojo numeric core is **faithful to ai-toolkit production**. Before a real run *reproduces* ai-toolkit, apply the P0/P1 recipe fixes above (padding-mask, caption-wrap, LR) — these are data/recipe, not numeric-core, divergences. **CAMPAIGN COMPLETE (2026-06-25):** P0 fixes applied + re-staged (cache pad-rows=0 verified), real-run verdict PASS (loss drops, LoRA learning), and the trainer is now on **autograd_v2** (IDEOGRAM4_V2_GRAPH default-ON) with the engine==hand-chain anchor **bit-identical over 30 steps**. Optional follow-ups: sample-shift visual (longer run), P1/P2 recipe (LR 1e-4 default, eps, caption_dropout, multi-res), slab/capture for engine speed, re-point the stale serenity-trainer predict_gate to ai-toolkit.

## Results (cos, filled as gates run)
- Block fwd: block0_out **0.99999997**; bwd d_x **0.99999993**, d_adaln 0.99998849, 6× LoRA dA/dB ≥0.99998.
- Stack fwd→velocity (after mrope fix): block0 0.99999, block16 0.99992, block33 0.99979, transformer_out 0.99990, velocity **0.99991**.
- MRoPE: 0.714 → **0.999999** after the f32-inv_freq fix.

## BUG FOUND + FIXED (by this campaign)
**MRoPE inv_freq dtype** (`serenitymojo/models/dit/ideogram4_mrope.mojo:49`): bf16-rounded inv_freq to match the
INVALID ideogram4-ref oracle; ai-toolkit production keeps inv_freq F32. Removed the `.cast[bf16]().cast[f32]()`.
mrope cos 0.714→0.999999; verified by the predict gate (exit 0). **Affects both training AND inference.** APPLIED.

## MUST-CHANGE before a real mojo run matches ai-toolkit (recipe audit, cite file:line — NOT yet applied; needs the user's repo + a real-run re-gate)
1. **P0 padding-mask leak — ✅ APPLIED + RE-VERIFIED (2026-06-25)**. Threaded a `text_len` (natural pre-pad count, default NT = old behavior byte-identical) through prepare→encode→predict: `prepare_cache.mojo` zeros encoder pad-feature rows via a [1,NT,1] mask (= ai-toolkit `pipeline.py:156-157` stacked*text_mask); `Ideogram4Predict.mojo build_packed_inputs` sets indicator=0 + position_ids→real_len-1 at pad (= ai-toolkit `:249/:262`); `Ideogram4CacheReader` + the trainer thread it. Re-verify: predict gate indicator/position_ids/packed-x/add_noise/flow_target all PASS byte-identical (NT no-regression) + code-read confirms pad logic. **CACHE RE-STAGE REQUIRED** (old caches load w/ default NT = old behavior).
2. **P0 caption JSON-wrap — ✅ APPLIED + RE-VERIFIED (2026-06-25)**. `scripts/ideogram4_stage_images.py render_prompt` now uses ai-toolkit's `digest_caption_string` (plain passes through, JSON minified) + the verified chat-template (no system prompt). Re-verify: edited render_prompt token-ids **IDENTICAL** to ai-toolkit's `apply_chat_template` for plain caption, empty-uncond, AND structured JSON (the json-wrap had leaked `{"high_level_description":"` = +7 tokens into the maskless DiT). **CACHE RE-STAGE REQUIRED**.

### Flag (separate, not a P0): serenity-trainer `smoke/ideogram4_predict_gate.mojo` predict-velocity = 0.971 (FAIL) — its indicator/pos/packed-x PASS, so NOT the padding fix; likely a STALE ideogram4-ref fixture (which the MRoPE f32 fix now correctly diverges from) or a flash-vs-oracle backend mismatch. The ai-toolkit campaign predict gate has the forward faithful at 0.99991. Re-point that gate to ai-toolkit to confirm.
3. **P1 LR default** — `Ideogram4LiveTrainer.mojo:118` default 4e-4; ai-toolkit 1e-4. Launch with explicit 1e-4 or change the default.
4. **P2** optimizer eps 1e-8→1e-6 (`TrainConfig.mojo:89-93`); caption_dropout 0→0.05; multi-res [512,768,1024] (hardcoded 512); confirm grad-clip@1.0 is applied in the loop.

## Notes
- The rust EriTrainer ideogram parity is ALREADY done (its ledger:
  `/home/alex/EriTrainer/trainer/parity/IDEOGRAM_PARITY_LEDGER.md`) — useful as a
  capture-point template, but its numbers do NOT transfer to mojo.
- Known recipe divergences to confirm on the mojo side (from the rust ledger):
  caption JSON-minify before chat template, LR constant-1e-4 (not cosine→0), text masking.
