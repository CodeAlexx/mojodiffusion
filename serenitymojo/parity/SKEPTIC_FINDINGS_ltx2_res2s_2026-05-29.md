# SKEPTIC FINDINGS — LTX-2 res_2s sampler port (2026-05-29)

Auditor: skeptic subagent. Scope: verify the res_2s second-order sampler ported into
`serenitymojo/sampling/ltx2_sampling.mojo` against the Lightricks reference, re-run the
gate, and confirm fail-closed behavior.

## VERDICT: PARTIAL PASS — the coefficient/RK2 math is genuine and correct, but the
## ported "deterministic core" is NOT the algorithm the HQ recipe actually runs, and the
## user's real gate (a visibly sharp clip vs the soft MVP) was NEVER executed.

---

## What IS correct (verified)

1. **Genuine 2nd-order exponential RK, not Euler relabeled.** The port implements two
   distinct model evals per step (stage-1 @ `sigma`, stage-2 @ midpoint `sub_sigma`) and
   combines them with φ-function weights. This is structurally a 2-stage Runge-Kutta in
   log-sigma space, confirmed against `res2s.py::phi` / `get_res2s_coefficients` and the
   loop in `samplers.py::res2s_audio_video_denoising_loop`. It is NOT the Euler `step()`
   with a different name.

2. **Coefficient math matches the reference exactly.** Re-derived from scratch in Python
   (`math.exp`/`math.factorial`, float64) and against an independent host Float64 path:
   - `h = log(sigma/sigma_next)` — matches ref line 241 (`-log(sigma_next/sigma)`), diff 0.0
   - `a21 = c2·φ₁(-h·c2)`, `b2 = φ₂(-h)/c2`, `b1 = φ₁(-h)-b2` — all diff 0.0 vs host-ref
   - `sub_sigma = sqrt(sigma·sigma_next)` (geometric mean, the c2=0.5 hardcode) — matches
   - φ₁(0)=1, φ₂(0)=0.5 Taylor limits — exact
   - `combine_weights`: `x_next = x + h·(b1·eps1 + b2·eps2)` — matches ref lines 330-331
   The φ functions, midpoint sigma, eval count, and combine are all faithful.

3. **Smoke fail-closed CONFIRMED.** I injected `b1 += 0.05` into `res2s_coefficients`;
   the smoke raised `res2s scalar mismatch: b1 vs host-ref` and exited non-zero (EXIT=1).
   The gate is not a no-op rubber stamp. (Fault reverted; file clean.)

4. Smoke PASS reproduced verbatim: `x_next cos(mojo, host-ref) = 0.999999999999966`,
   all scalar diffs ≤ 1.5e-7 (f32 ingest noise on the sigma table).

---

## What is WRONG / MISLEADING (the load-bearing problems)

### FINDING 1 (CRITICAL) — the "SDE injection is a no-op in deterministic mode" claim is FALSE for the HQ recipe.

The builder report states: *"The pipeline's `Res2sDiffusionStep` SDE-injection branch is a
no-op in deterministic mode (returns `denoised_sample` unchanged when `sigma_up==0` or
`sigma_next==0`) ... so the SDE and bong-iteration paths are intentionally omitted."*

This is incorrect. The HQ pipeline (`ti2vid_two_stages_hq.py:152,162,225`) constructs a
plain `Res2sDiffusionStep()` and calls `res2s_audio_video_denoising_loop(...)` **without
overriding `legacy_mode` (default True) or `bongmath` (default True)**. The stepper sets
`sigma_up = sigma_next * 0.5` (diffusion_steps.py:84). The short-circuit at line 86 only
fires when `sigma_up == 0` OR `sigma_next == 0` — i.e. ONLY on the final step. For every
interior step `sigma_next > 0`, so `sigma_up > 0`, so the SDE branch (lines 90-94) RUNS
and injects noise: `x = alpha_ratio·(denoised_next + sigma_down·eps_next) + sigma_up·noise`.

Verified numerically across all 7 distilled stage-1 interior steps (substep SDE,
`sigma_up = sub_sigma·0.5`):
```
step 0: sub_sigma=0.9969 sigma_up=0.4984 alpha_ratio=0.8664 sigma_down=0.9964 NOOP=False
step 1: sub_sigma=0.9906 sigma_up=0.4953 alpha_ratio=0.8673 sigma_down=0.9892 NOOP=False
...
step 6: sub_sigma=0.5530 sigma_up=0.2765 alpha_ratio=0.9259 sigma_down=0.5173 NOOP=False
step 7: FINAL
```
`sigma_up ≈ 0.4-0.5` is a LARGE noise term and `alpha_ratio ≈ 0.87` rescales the signal.
**The HQ recipe is stochastic (an ancestral/SDE sampler), not deterministic.** Omitting
the SDE path does not "port the deterministic recipe" — it ports a *different algorithm*.

### FINDING 2 (CRITICAL) — bongmath is NOT an algebraic identity; it fires on 6 of 7 steps.

The builder claims bong-iteration "only re-derives x_anchor back from x_mid and is
algebraically identity for the deterministic x_anchor we carry." That identity only holds
if `x_mid == x_anchor + h·a21·eps_1`. But `x_mid` is FIRST overwritten by substep SDE
injection (samplers.py:285-298) BEFORE the bong loop (302-307). So
`x_anchor_new = x_mid_sde - h·a21·eps_1 ≠ x_anchor`. The bong gate `h < 0.5 and sigma > 0.03`
fires on steps 0-5 (h ranges 0.006→0.227); only step 6 (h=0.541) skips it.

### FINDING 3 (CRITICAL) — end-to-end divergence: cos = 0.875, well below the 0.999 gate.

I simulated one real interior step (sigma 0.909→0.725) of the actual HQ loop (substep SDE
+ bongmath + step SDE, legacy_mode=True) vs the ported deterministic core:
```
substep SDE moved x_mid by:        0.939  (≈ signal magnitude)
bongmath moved x_anchor by:        1.052
cos(det_core, full_HQ_loop_step) = 0.875   ← below the 0.999 the smoke claims
rel L2(det_core vs HQ loop)      = 0.529
```
The ported step does NOT reproduce the reference HQ step. The 0.999999 cosine in the
smoke is real but measures only the port's internal self-consistency (Mojo vs a host
re-implementation of *the same omitted-SDE formula*) — a tautology, not parity with the
algorithm the HQ recipe runs. Note the second model eval input also differs: the reference
feeds the *SDE-injected* `x_mid` into the stage-2 denoiser, the port feeds the deterministic one.

### FINDING 4 (BLOCKER) — the user's actual gate was never run.

The mandated gate was a "VISIBLY SHARP, coherent clip" decoded and compared against the
soft MVP. The smoke is a pure scalar/arithmetic unit test on a 2×2 synthetic tensor: it
never loads the DiT, never runs a sampling loop, never touches the VAE, never produces a
clip. No clip exists; no sharpness comparison was made. The HARD RULE gate is UNMET.

---

## Net assessment

- The φ/coefficient/combine math: faithful, exact, fail-closed. Keep it.
- The claim that this constitutes "the res_2s sampler for the HQ recipe": FALSE. The HQ
  recipe is stochastic (SDE injection at substep+step, active every interior step) with
  bong anchor refinement (active 6/7 steps). The port silently drops both and renames the
  remaining deterministic skeleton "res_2s deterministic core." A single real step diverges
  at cos 0.875 / rel-L2 0.53 from the reference.
- To actually meet the recipe, the port must implement `Res2sDiffusionStep.get_sde_coeff` +
  the substep/step SDE injection (`sigma_up = sigma_next·0.5`, channel-normalized noise,
  matched seeds `noise_seed`/`noise_seed+10000`) and the bongmath loop, then gate a full
  decoded clip against the MVP. Until then the HARD RULE gate is UNMET and no further
  reliance should be placed on res_2s producing the proven-good HQ output.

## Repro
- Reference read: `ltx2-official-ref/.../utils/res2s.py`, `.../utils/samplers.py`
  (`res2s_audio_video_denoising_loop`), `ltx-core/.../diffusion_steps.py`
  (`Res2sDiffusionStep`), `ltx-pipelines/.../ti2vid_two_stages_hq.py` (the only caller).
- Smoke: `pixi run mojo run -I . serenitymojo/pipeline/ltx2_res2s_smoke.mojo` → PASS (coeff-only).
- Fault inject (b1+=0.05) → smoke raises + EXIT=1 (fail-closed confirmed, then reverted).
- Numerical divergence + SDE-active tables: inlined Python repro of `Res2sDiffusionStep`
  + `get_res2s_coefficients` (heavy package import blocked by missing OpenImageIO).
