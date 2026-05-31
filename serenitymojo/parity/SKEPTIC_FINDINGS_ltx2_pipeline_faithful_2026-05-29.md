# SKEPTIC FINDINGS — LTX-2 faithful HQ pipeline (2026-05-29)

Auditor: skeptic subagent. Scope: verify the staged HQ `generate()` flow wired into
`serenitymojo/pipeline/ltx2_t2v_av_hq.mojo::run_staged` executed EVERY stage of the
authoritative `pipeline.py` HQ recipe, and view the decoded frames for real sharpness.

## VERDICT: FAIL (partial). The two previously-missing stages (spatial upscale + stage-2
## refine) ARE now wired and ran, and the output is genuinely sharp at 1536x1024. BUT two
## load-bearing parts of the recipe are still NOT faithful: (1) NAG is NOT applied in either
## stage — it is referenced only in comments, never called; (2) res_2s is STILL the
## deterministic core, not the stochastic SDE+bongmath loop the HQ recipe runs (the exact
## gap a prior skeptic already flagged and which was NOT fixed). The brief required ALL
## stages; NAG and stochastic res2s are missing, so this fails the faithfulness bar.

---

## Stage-by-stage verification (against pipeline.py HQ generate())

| Stage | Required | Ran? | Evidence |
|-------|----------|------|----------|
| Stage 1 denoise | LTX2Scheduler 15-step token-shifted sigmas + res_2s | YES (code); fast-gate ran 2/15 | log `S1 step 1/15 sigma=1.0->0.969`; std=0.95 finite |
| **SPATIAL UPSAMPLE** | LatentUpsampler x2 + upsample_video (un_normalize/normalize w/ VAE stats) | YES | latent `[1,128,2,16,24] -> [1,128,2,32,48]` exact 2x; std 0.95->1.11 |
| **STAGE-2 REFINE** | forward-noise upscaled @ 0.909375 + 3-step res_2s | YES | log `S2 step 1/3 sigma=0.909375->0.725`; vx2 std=0.85 finite |
| DECODE | VAE -> frames at 2x | YES | frames NCDHW `[1,3,9,1024,1536]`; 9 PNGs at 1536x1024 |
| **NAG (both stages)** | NAGPatch around stage-1 AND stage-2 | **NO** | `ltx2_block_forward_av_nag` NEVER called in run_staged; only 3 comment lines mention NAG |
| res_2s stochasticity | SDE injection (sigma_up=sigma_next*0.5) + bongmath | **NO** | sampler is deterministic RK2 core; SDE/bongmath "intentionally omitted" (ltx2_sampling.mojo:345-355) |
| 4-pass temporal | temporal upscale + FOUR_PASS_STAGE4_SIGMAS | NO (not in the 2-stage HQ recipe; deferred — acceptable) | — |

## What IS genuinely fixed and verified (credit where due)

1. **The two missing stages are real.** Spatial upscale and stage-2 refine both execute
   end-to-end. The previously-shipped "Stage-1 only" bug IS resolved for those two stages.
2. **Upsampler is parity-gated, not faked.** `models/upsampler/ltx2_upsampler_smoke.mojo`
   compares against a genuine PyTorch dump (`parity/spatial_out.npy`, 262KB real reference)
   with a `cos >= 0.999` fail gate. Weights present and full-size
   (`ltx-2-spatial-upscaler-x2-1.0.safetensors` = 995 MB). This stage is trustworthy.
3. **Output IS visibly sharper.** New `output/ltx2_hq2/hq_frame02.png` (1536x1024): a
   coherent scene — hands holding a tool against a panel of switches, plausible five-finger
   anatomy, fine skin/metal texture, NO grit. Prior `output/ltx2_hq/hq_frame02.png`
   (768x512) is small, dark, grainy. On absolute quality the staged result is decisively
   better. (Caveat: different scene/seed, so it is NOT a controlled before/after of one gen.)

## Why it FAILS the faithfulness bar

### FINDING 1 (BLOCKER) — NAG not applied in either stage.
The brief explicitly requires "NAG was applied in both stages." It is not. The combine,
`ltx2_block_forward_av_nag`, and `NAGContext` are built in `models/dit/ltx2_nag.mojo`, but
`run_staged` never invokes any of them — grep of the pipeline shows NAG only in comments
(lines 964-967). The builder's stated reason is honest (the NAG negative baseline is the
Gemma encoding of `""`, and no null-encoding dump exists; serenitymojo has no runtime Gemma
encoder). That is a real blocker, but it is still a SKIPPED STAGE. The output was produced
WITHOUT the guidance NAG provides — not the faithful recipe.

### FINDING 2 (BLOCKER) — res_2s is the deterministic core, NOT the HQ stochastic loop.
This is the SAME gap a prior skeptic already documented in
`SKEPTIC_FINDINGS_ltx2_res2s_2026-05-29.md` (FINDING 1-3) and it was NOT fixed. The new
staged loop reuses `res2s_coefficients/substep/combine` — the deterministic RK2 update.
`ltx2_sampling.mojo:345-355` still asserts "SDE injection is a no-op in that mode ...
intentionally omitted" and "bongmath ... algebraically identity ... omitted." The prior
audit proved both claims FALSE: the HQ recipe sets `sigma_up = sigma_next*0.5` (active on
every interior step) and bongmath fires on 6/7 steps; a single real step diverges at
cos=0.875 / rel-L2=0.53 from the deterministic port. So BOTH stage-1 and stage-2 ran a
different sampler than the reference HQ recipe. Faithfulness is not met.

### FINDING 3 (PROCESS) — the frames I reviewed are the FAST GATE, not the full run.
`/tmp/staged_out.log` ends `=== HQ STAGED (fast gate) DONE ===` and shows only 2/15 stage-1
steps with `[decode] audio SKIPPED (fast gate)`. The 9 PNGs in `output/ltx2_hq2/`
(timestamped 06:32) are from that gate. The full 15-step run (PID 1027529) started 06:33
and at audit time was only ~2 steps deep (GPU 11.6 GB used, consistent). So the reviewed
frames came from a 2-step stage-1 that the upscale+refine rescued — impressive, but the
"final 15-step frames + mp4" the report promises did not yet exist. No audio/mux verified.

## Net assessment
- Missing-stage bug (upscale + refine): FIXED and verified. Good work.
- Upsampler parity: real, gated, trustworthy.
- Sharpness: real and decisive on absolute terms.
- NAG: SKIPPED (honest blocker, but still absent from the produced output).
- res_2s stochasticity: STILL the deterministic core — the recipe's sampler is not faithful,
  repeating an already-flagged, unfixed gap.
- Reviewed frames are from a 2-step fast gate, not the full run.

The brief's hard requirement was that EVERY stage of pipeline.py's HQ flow ran, including
NAG in both stages and stochastic res2s. Two of those are not met. **FAIL.**

## To pass
1. Implement the res_2s SDE injection (`sigma_up = sigma_next*0.5`, channel-normalized noise,
   seeds `noise_seed` / `noise_seed+10000`) + bongmath in the loop (per the prior res2s
   findings), used in BOTH stages.
2. Produce an empty-string Gemma+connector null-context dump and wire
   `ltx2_block_forward_av_nag` into both stage forwards (scale=11, alpha=0.25, tau=2.5).
3. Re-run the FULL 15-step + 3-step staged flow with audio decode/mux, then re-view the
   final frames.
