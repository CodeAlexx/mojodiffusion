# SKEPTIC FINDINGS — Lens Phase F
**Date:** 2026-05-28  
**Verdict:** NOT-VERIFIED (zeroed-text oracle exists; REAL-PROMPT parity gate not achievable from Mojo side)

---

## Real-Parity Gate Attempt

### What exists
The Rust oracle (`lens_infer.rs`) already has a **full per-step parity harness** (`--parity` mode) that:
- Loads `captures/hidden_states_pre_step_NN.safetensors` (all 20 steps exist)
- Loads `captures/noise_pred_step_NN.safetensors` (all 20 steps exist)
- Runs the Rust DiT on the captured hidden state + zeroed text features
- Reports `cos / max_abs / mean_abs` per step vs the Python capture

**The captures are zeroed-text only** (`capture_metadata.json`: `"prompt": "(zeroed pre-cached text features, no encoder)"`). So the Rust `--parity` run compares Rust vs Python with the same zeroed input.

**The Mojo pipeline uses the same zeroed captures** (`captures_text_smoke/hidden_layer_{05,11,17,23}.safetensors`, metadata confirms `"seed": 42` but no prompt text — these are a 64-token zero sequence from the GPT-OSS encoder on a short dummy input, NOT real-prompt features).

**Parity-gate status:** The Rust `--parity` mode is the real oracle for zeroed-text math correctness. I cannot run it from the Mojo skeptic role without re-running `cargo run --bin lens_infer -- --parity`. No Mojo-side attempt can get RNG-matched noise against the Rust oracle because Mojo uses Box-Muller while Rust uses Philox4x32-10. No one-step capture with matched noise exists. **Real-prompt captures with matched initial noise do not exist on disk.**

Verdict on real-parity gate: **not achieved**. The gate requires the bugfixer to either run `LENS_DUMP_INIT_NOISE` + `LENS_DUMP_STEPS_DIR` from Rust to produce Philox-seeded tensors, then load them into a Mojo variant via `LENS_LOAD_INIT_NOISE`, or simply run `lens_infer --parity` (Rust vs Python, zeroed features) and report the cos scores.

---

## Per-Area Audit Table

| Area | Mojo (`lens_pipeline_1024_multistep.mojo`) | Rust (`lens_dit.rs`) | Status |
|------|---------------------------------------------|----------------------|--------|
| **48-block dual-stream forward** | Steps 1–8 match Rust exactly: silu(temb)→mod linear→6 chunks; RMSNorm1→modulate; QKV linear; QK-RMSNorm (eps=1e-5); interleaved-pair RoPE; joint SDPA (img first); split; out-proj; gate1 residual; RMSNorm2→modulate2; SwiGLU; gate2 residual | Same sequence, same eps | MATCH |
| **adaLN chunk order (6-way mod)** | Mojo: `_adaln_chunk(img_mod, 0..5)` = shift1,scale1,gate1,shift2,scale2,gate2. Rust: `halves[0].chunk(3)=(shift,scale,gate), halves[1].chunk(3)=(shift,scale,gate)` — same | Identical | MATCH |
| **Final AdaLayerNormContinuous** | Mojo: `scale=slice(0,DIM)`, `shift=slice(DIM,DIM)` → scale-first chunk. Rust `ada_layer_norm_continuous`: `scale=chunks[0], shift=chunks[1]` (comment says "SCALE IS FIRST"). Uses LayerNorm (not RMSNorm). Mojo uses `layer_norm` with ones/zeros. | Identical: scale-first, LayerNorm | MATCH |
| **Text concat order** | Mojo: `concat(2, ctx, n05, n11, n17, n23)` — layers 5,11,17,23 in order. Rust: `for (i, feat) in encoder_hidden_states.iter()` — iterates in caller-supplied order (5,11,17,23). Mojo text-math gate (`lens_dit_math.mojo:926-966`): layer 0=layer_05, 1=layer_11, 2=layer_17, 3=layer_23 — consistent. | Same order | MATCH |
| **Per-layer txt_norm eps** | Mojo: `TXT_NORM_EPS = 1e-5`. Rust: `rms_norm_bf16(feat, Some(&self.txt_norm[i]), 1e-5)` | 1e-5 both | MATCH |
| **Block RMSNorm eps** | Mojo: `BLOCK_NORM_EPS = 1e-6`. Rust: `1e-6` | MATCH |
| **QK-RMSNorm eps** | Mojo: `QK_NORM_EPS = 1e-5`. Rust: `1e-5` | MATCH |
| **RoPE construction** | Mojo: builds pos/neg host tables [4096,32], uses neg[-(h_hi):]||pos[:h_lo] for H/W axes, `max_vid_idx=max(LH/2,LW/2)=32` for text slice. Rust: identical logic in `LensEmbedRope` — confirmed by Rust unit tests passing. | MATCH |
| **Timestep embedding** | Mojo: pre-scales sigma×1000 before calling `timestep_embedding`, which is cos-first with no internal scale. Rust: `timestep_embedding(t, 256)` with `scale=1000` internally, `flip_sin_to_cos=True` (cos-first). Net result identical. | MATCH |
| **mu/schedule** | Both use dynamic shifting with mu=2.198, 20 linear sigmas from 1.0→0.05, Euler step. `capture_metadata.json` confirms mu=2.1980220725551165. | MATCH |
| **CFG** | Mojo pipeline skips CFG rescale (zeroed cond==uncond, comment says identity). Rust: runs `cfg_norm_rescale_pair` but with cond==uncond the scale==1.0. Numerically identical for zeroed embeds. **For real prompts: Mojo pipeline has NO CFG two-forward pass at all — single forward, no uncond.** | BUG for real prompts |
| **RNG** | Mojo: Box-Muller via `randn()`. Rust: Philox4x32-10 via `randn_torch()`. | MISMATCH — different noise |
| **Assertion style** | Mojo pipeline: no runtime assertions, shape mismatches raise implicitly. Math gates (`lens_dit_math.mojo`): strict raise-on-drift thresholds (mean_abs_diff<0.05/0.005). | Soft in pipeline, hard in math gates |

---

## Key Bugs Found

1. **No real-prompt CFG in Mojo pipeline.** `lens_pipeline_1024_multistep.mojo` runs a single DiT forward and passes zeroed text as both cond and uncond. There is no two-forward (cond+uncond) CFG loop. For any real-prompt run this means CFG is completely absent.

2. **text_smoke captures are NOT real-prompt features.** `captures_text_smoke/metadata.json` has no prompt field and a 64-token dummy input. The image was not prompt-steered at all.

3. **RNG mismatch prevents any matched-noise parity.** Mojo Box-Muller vs Rust Philox. Rust already has `LENS_LOAD_INIT_NOISE` env-var support to inject external noise tensors.

4. **Parity harness never run end-to-end.** The Rust `--parity` flag compares Rust DiT vs Python at all 20 steps with zeroed captures. This is the authoritative gate and **was not run as part of this build session**.

---

## Bugfixer Punch List

1. **Run `lens_infer --parity`** (Rust vs Python, 20-step zeroed features). This is already wired — just needs to be executed and cos scores reported. This is the only real verification that the DiT math is numerically correct.

2. **Wire real GPT-OSS text captures.** Generate real-prompt features via `lens_infer` (default mode, no `--use-cached-features`) and dump them with `LENS_DUMP_INIT_NOISE` + a 20-step `LENS_DUMP_STEPS_DIR`. Load initial noise into Mojo via a pipeline variant.

3. **Add two-forward CFG to Mojo pipeline.** The denoise loop must run `lens_forward(cond)` + `lens_forward(uncond)` and apply `cfg_norm_rescale` before the Euler step. Currently absent.

4. **Matched-noise one-step parity.** Use Rust `LENS_DUMP_INIT_NOISE` to dump Philox noise, then load it into Mojo pipeline. Run one DiT step, compare Mojo `noise_pred` vs Rust `noise_pred` for step 0. Target: cos ≥ 0.999.

5. **Do not accept "coherent image" as passing.** The current "coherent image" result is from zeroed text + mismatched RNG. Any of the 48 blocks could have an off-by-one in modulation indexing and produce a plausible-looking image from pure noise.
