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
| 10 | L2P real-run verdict (loss DROPS + sample SHIFTS) | real train smoke (`train_ideogram4` on the eri2 cache) | — | ⬜ **PENDING** — the ONLY gate left. Needs the training `--cache-dir` (118-img eri2 cache, hidden behind the prior run's guard wrapper) + the P0 recipe fixes applied first (else loss won't match). |

## VERDICT (2026-06-25 overnight)
**Every per-component gate PASSES vs ai-toolkit** (block fwd+bwd, full forward→velocity, MRoPE, VAE encode, text-encoder taps, step math, VAE-norm, LoRA targets/rank/alpha/init). The campaign **found + fixed one real model bug** (MRoPE inv_freq bf16→f32) that the invalid ideogram4-ref oracle hid. The mojo numeric core is **faithful to ai-toolkit production**. Before a real run *reproduces* ai-toolkit, apply the P0/P1 recipe fixes above (padding-mask, caption-wrap, LR) — these are data/recipe, not numeric-core, divergences. Last gate = the L2P real-run once the cache + fixes are in.

## Results (cos, filled as gates run)
- Block fwd: block0_out **0.99999997**; bwd d_x **0.99999993**, d_adaln 0.99998849, 6× LoRA dA/dB ≥0.99998.
- Stack fwd→velocity (after mrope fix): block0 0.99999, block16 0.99992, block33 0.99979, transformer_out 0.99990, velocity **0.99991**.
- MRoPE: 0.714 → **0.999999** after the f32-inv_freq fix.

## BUG FOUND + FIXED (by this campaign)
**MRoPE inv_freq dtype** (`serenitymojo/models/dit/ideogram4_mrope.mojo:49`): bf16-rounded inv_freq to match the
INVALID ideogram4-ref oracle; ai-toolkit production keeps inv_freq F32. Removed the `.cast[bf16]().cast[f32]()`.
mrope cos 0.714→0.999999; verified by the predict gate (exit 0). **Affects both training AND inference.** APPLIED.

## MUST-CHANGE before a real mojo run matches ai-toolkit (recipe audit, cite file:line — NOT yet applied; needs the user's repo + L2P re-gate)
1. **P0 padding-mask leak** — `smoke/ideogram4_prepare_cache.mojo:91-98` pads ids to NT=256 w/ PAD_ID; `ideogram_qwen3vl.mojo:59-79` never zeros pad features; `Ideogram4Predict.mojo:141-147` writes indicator=3 for ALL 256. ai-toolkit zeros pad features (`pipeline.py:156-157`) + indicator=0 at pads (`pipeline.py:249`). Thread a text_mask → likely loss-magnitude culprit.
2. **P0 caption JSON-wrap** — `scripts/ideogram4_stage_images.py:43-44` wraps `{"high_level_description":cap}`; ai-toolkit chat-templates the RAW plain-text caption (`ideogram_caption.py:303-319`). Drop the wrapper for plain .txt.
3. **P1 LR default** — `Ideogram4LiveTrainer.mojo:118` default 4e-4; ai-toolkit 1e-4. Launch with explicit 1e-4 or change the default.
4. **P2** optimizer eps 1e-8→1e-6 (`TrainConfig.mojo:89-93`); caption_dropout 0→0.05; multi-res [512,768,1024] (hardcoded 512); confirm grad-clip@1.0 is applied in the loop.

## Notes
- The rust EriTrainer ideogram parity is ALREADY done (its ledger:
  `/home/alex/EriTrainer/trainer/parity/IDEOGRAM_PARITY_LEDGER.md`) — useful as a
  capture-point template, but its numbers do NOT transfer to mojo.
- Known recipe divergences to confirm on the mojo side (from the rust ledger):
  caption JSON-minify before chat template, LR constant-1e-4 (not cosine→0), text masking.
