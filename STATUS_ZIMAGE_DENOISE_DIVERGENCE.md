# STATUS — Z-Image pure-Mojo pipeline: denoise divergence (RESOLVED ENOUGH FOR GPU IMAGE)

**Date:** 2026-05-26 · **Project:** serenitymojo (`/home/alex/mojodiffusion`) — pure Mojo+MAX, inference-only, GPU-only Z-Image text→image. · **Companion memory:** `~/.claude/projects/-home-alex-EriDiffusion/memory/project_mojodiffusion_max_phase0.md`

> Update 2026-05-26: the root cause was a missed **post-CFG sign flip** in `diffusers/pipelines/z_image/pipeline_z_image.py`: after CFG combine, diffusers does `noise_pred = -noise_pred` before `scheduler.step(...)`. The Mojo pipeline and `parity_denoise.mojo` were applying the raw DiT velocity directly. CPU-only analysis of existing dumps resolves the step-0 anomaly: `noise + dt*raw_pred` gives std **1.00903**, while `noise + dt*(-raw_pred)` gives std **0.99898**, matching diffusers `lat_step_00` std **0.99901**. The GPU pipeline now defaults to Z-Image's native 1024x1024 target and writes `/home/alex/mojodiffusion/output/zimage_first_1024.png`. Durable notes for future agents are in `serenitymojo/docs/ZIMAGE_DENOISE_SIGN_CONVENTION.md`.

---

## 1. What works (do NOT re-verify — proven this session, GPU-bf16 vs diffusers)

| Stage | Result |
|---|---|
| Tokenizer + chat template | exact token ids (173 pos / 8 neg for the test prompt) |
| Qwen3-4B encoder cap_feats (layer-34, no final norm) | cos 0.99998 (cond) / 0.99999 (uncond) |
| cond DiT velocity (noise + lat_step_06/13/20/27, all t) | cos 0.9997–0.9999, magnitude ratio 1.00 |
| uncond DiT velocity (CAPLEN=8) | cos 0.9994–0.9999 |
| VAE decode | cos 0.99998 (512²/1024², prior) |
| **Euler update rule** | **exact** — diffusers `scheduling_flow_match_euler_discrete.py`: `prev_sample = sample + (σ_next−σ)·model_output` (sample upcast f32). For Z-Image, `model_output = -(vc + cfg·(vc−vu))` because `pipeline_z_image.py` negates after CFG. Verified numerically: `noise + dt·(-pred_raw)` matches diffusers `lat_step_00` at cos 0.9999986. |
| Pipeline conventions | timestep = (1−σ) (DiT does ×t_scale=1000 internally); CFG **code-form** `pred_raw = vc + cfg·(vc−vu)` [=5vc−4vu], then **negate before scheduler** (`noise_pred = -pred_raw`); extract layer 34; cap fed rank-2 [CAPLEN,2560]; sigmas = diffusers exact (shift=6, 30 steps); latent kept F32, bf16 only to feed DiT. |

Block-by-block GPU-bf16 parity (at lat_step_27/t=0.82): **every op cos ≥0.99994**; the full-forward velocity is cos 0.99974, with the only "drop" at the final 3840→64 Linear amplifying upstream bf16 residual. **SDPA is clean** (nr0_attn_out 0.9999971 — math-mode Dh=128 fallback is NOT the problem). My matmul is bit-faithful (oracle's fl_scaled → my linear → cos 0.9999951).

## 2. The bug (historical)
Free-running 30-step denoise from a fixed noise with the old no-negate code: latent std 1.0→**3.4** (diffusers→0.75). Per-step cos vs diffusers' trajectory: step0 0.9999 → step10 0.9965 → step20 0.959 → step28 0.28 (gradual, accelerating). Teacher-forced (oracle latent in each step) degrades far less (0.88 @ step28), because each per-step sign error was small early but accumulated.

Mechanistic correction: `⟨v_raw,x⟩/|x|² ≈ −0.92` is the raw DiT output captured by hooks. Diffusers sends `-v_raw` to the scheduler, so with `dt<0` the first update shrinks instead of growing.

## 3. Ruled OUT (each tested, did not fix)
- **CFG amplification** — diverges *identically* at CFG=0, 1.5, 4 (std ≈3.4). (The guidance term `vc−vu` is only 5.5% of |v| so its *direction* error is cos 0.81, but since CFG=0 also diverges, this is secondary not causal.)
- **Velocity precision (the main hypothesis)** — implemented full **F32 residual stream** through all 34 blocks + **F32 final layer** (device cast kernel, F32 residual adds, F32 weights in final layer). **No change: std still 3.36.** Reverted (added cast overhead, no benefit). → precision is NOT the cause.
- **Sigma schedule** — used diffusers' exact 31 sigmas (hardcoded). No change.
- **Latent precision** — F32 latent + F32 CFG combine. No change.
- **Update rule / timestep** — confirmed exact; the missing piece was the Z-Image pipeline's post-CFG negate before the scheduler.
- **Per-step velocity** — cos 0.9997–0.9999, magnitude 1.00, ⟨v,x⟩ matches within 0.005, err_proj small+mixed-sign (no systematic outward bias detectable at the single-step level).

## 4. Resolution note
The "radial bias" was an artifact of comparing the raw transformer output to scheduler latents. Diffusers' forward hook captures raw DiT output, but the pipeline negates after CFG and before Euler. The previous `noise + dt·pred` check used the wrong sign; with `-pred`, the step-0 magnitude gap disappears.

## 5. GPU rerun result
Command built and ran the pure-Mojo GPU pipeline:

```bash
cd /home/alex/mojodiffusion
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_pipeline.mojo -o /tmp/zimage_pipeline_1024
/tmp/zimage_pipeline_1024
```

Observed run:

- output: `/home/alex/mojodiffusion/output/zimage_first_1024.png`
- file format: PNG, RGB, 1024x1024, 3.1 MB
- image mean/min/max: mean `[93.89, 88.62, 80.24]`, min `[0, 0, 0]`, max `[255, 255, 255]`
- init latent: mean `-0.002746404`, std `1.0012124`, absmax `4.6118364`
- step-0 raw velocities:
  - cond mean `-0.10992641`, std `0.9794241`, absmax `4.59375`
  - uncond mean `-0.07749318`, std `1.0508091`, absmax `5.46875`
- final latent: mean `-0.1327351`, std `0.77510124`, absmax `3.8916383`

The final std moved from the historical bad ~3.36 down to 0.775, in the same
range as the diffusers trajectory (~0.75). This confirms the sign-boundary fix
is effective on GPU at the native 1024x1024 target. A required performance fix
was also made for 1024: `NextDiT._zero_mask` and VAE `AttnBlock` all-zero
attention masks now allocate on device and use `ctx.enqueue_memset`, instead of
constructing huge host `List[Float32]` masks.

- The hardcoded sigma list contains a duplicate terminal zero, so the printed
  final step is `0.0 -> 0.0`. It is harmless for this image but should be
  checked against the exact diffusers table before calling the 30-step schedule
  final.

## 6. Decision log (owner-tagged)
| Decision | Owner | Note |
|---|---|---|
| Pause on this bug; status-doc it | USER | "we may never finish this or come to it much later" |
| Revert F32-trunk experiment | USER (agent recommended) | didn't fix, added overhead |
| Keep F32 *latent* in pipeline | AGENT | correct diffusers convention, unrelated to the bug |
| Precision is NOT the cause | AGENT-tested | F32 trunk + final layer made no difference |
| Missed post-CFG negate found | AGENT-tested CPU+GPU | existing dumps prove `-pred_raw` matches `lat_step_00`; 1024 GPU rerun produced final latent std 0.775 |
| Hardcoded sigmas widened | AGENT | replaced 5-decimal schedule constants with full float32 diffusers values |
| Device-zero masks | AGENT | replaced host-built all-zero DiT/VAE attention masks with device memset; required for 1024 practicality |
| GPU image generated | AGENT-tested GPU | `/home/alex/mojodiffusion/output/zimage_first_1024.png`, PNG size 1024x1024 |

## 7. How to resume (cold-start)
- **Run the pipeline:** `cd /home/alex/mojodiffusion && pixi run mojo run -I . serenitymojo/pipeline/zimage_pipeline.mojo` (native 1024² default, CFG=4, gyroid prompt). Output PNG -> `output/zimage_first_1024.png`. Watch `[stat] final_latent std` — the 2026-05-26 1024 GPU rerun was **0.775** rather than the old ~3.4.
- **Diffusers oracle (GPU bf16):** venv = `/home/alex/serenityflow-v2/.venv/bin/python` (CUDA torch + diffusers 0.38 + transformers + ZImageTransformer2DModel). **NEVER fp32-CPU** (cos 0.5/layer) and **NEVER fp32-host-load a full model** (60GB OOM killed a session — load bf16 on GPU, `text_encoder=None` for denoise so the 12.3GB transformer fits alone, `torch.no_grad()`+`empty_cache()`).
- **Parity infrastructure** (all in `serenitymojo/pipeline/parity/`): `zimage_oracle_denoise.py` (per-step latents + velocities), `zimage_oracle_velt.py` (fresh single-sample velocities — use these, NOT batched-hook captures, which gave a false "0.38" artifact), `blkdbg_oracle.py` (block-by-block), and Mojo drivers `parity_{encoder,dit_velocity,velt,denoise,blkdbg,final,linop,imgtok}.mojo`. Reference bins already dumped: `noise.bin`, `cond.bin`, `uncond.bin`, `vf_*.bin`, `vfu_*.bin`, `lat_step_NN.bin`, `velc_NN.bin`. DiT debug methods (additive, forward unchanged): `debug_main_layer`, `debug_final_sub`, `debug_final_linear_only`.
- **The proven build loop is now a skill:** `/home/alex/mojodiffusion/.claude/skills/mojo-port/` (builder→skeptic→bugfix→parity, Mojo-pure; use for Klein/Qwen/etc.). Keep separate from EriDiffusion's Rust `/port-*` skills.
- **Pitfalls that wasted time here:** cos is magnitude-blind (check magnitude ratio + per-step too); a batched-CFG-hook velocity ≠ single-sample velocity (use fresh refs); the oracle's `lat_step_NN` are CFG=4 — don't compare CFG=0 steps against them; forward hooks capture raw DiT output, but Z-Image diffusers negates after CFG before scheduler.
- **Conventions to keep** (all verified): see §1 table. Reference = diffusers `transformer_z_image.py` + `scheduling_flow_match_euler_discrete.py`, read line-by-line.

## 8. The real takeaway
The **architecture is sound** — 9 components individually parity-verified to cos 0.999+, the encoder/VAE/tokenizer/DiT-forward all match diffusers, and the pipeline composes + runs on GPU with clean memory. The denoise failure was not precision; it was a sign convention at the boundary between raw DiT output and the diffusers scheduler. Everything needed to verify the fix is here + in memory.

## 9. Klein 9B handoff note (2026-05-26)
Klein 9B is not a full image pipeline yet, but two runnable Mojo slices now
exist and are documented in `serenitymojo/docs/SDXL_FLUX_KLEIN_PORT_STATUS.md`:

- `serenitymojo/pipeline/klein9b_text_smoke.mojo` loads dense BF16 Qwen3-8B from
  the HF cache and produces Klein conditioning `[1,512,12288]` for positive and
  negative prompts. The `.serenity` Qwen file is Comfy-quantized and needs a
  dequant loader before it can replace the dense HF shards.
- `serenitymojo/models/dit/klein_dit.mojo` plus
  `serenitymojo/pipeline/klein9b_dit_smoke.mojo` loads real BF16 tensors from
  `/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors` and
  runs a fast truncated 9B DiT path on GPU: shared projections/modulation,
  `double_blocks.0`, `single_blocks.0`, and final projection. Verified output
  was `[1,4,128]` with finite stats.
- `serenitymojo/pipeline/klein9b_dit_full_smoke.mojo` loads all 201 BF16 DiT
  tensors and runs every Klein 9B block (8 double + 24 single) on the same tiny
  token grid. Verified output was `[1,4,128]` with finite stats
  (`mean=0.034185875`, `std=0.2429931`, `absmax=0.69921875`). This proves the
  full transformer wiring runs, but not yet the 1024x1024 production memory path.

Build/run commands:

```bash
cd /home/alex/mojodiffusion
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_text_smoke.mojo -o /tmp/klein9b_text_smoke
/tmp/klein9b_text_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_dit_smoke.mojo -o /tmp/klein9b_dit_smoke
/tmp/klein9b_dit_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_dit_full_smoke.mojo -o /tmp/klein9b_dit_full_smoke
/tmp/klein9b_dit_full_smoke
```

Next practical Klein step: port the FLUX.2/Klein VAE decode, then wire the full
text-to-image pipeline with an offloaded or otherwise memory-safe 1024x1024 DiT
denoise path.
