# SKEPTIC FINDINGS — LTX2 Team 3 (sampler + guidance)

Date: 2026-05-28
Auditor: Skeptic (adversarial, read-only source + GPU smoke run)
Verdict: **PASS** (all mandatory checks). 0 FAIL, 0 WARN-blocking, 2 informational WARN (BF16 re-verify follow-on, `__init__` export).

## Files audited (Mojo, NEW)
- `serenitymojo/sampling/ltx2_sampling.mojo` (264 lines)
- `serenitymojo/sampling/ltx2_guidance.mojo` (332 lines)
- `serenitymojo/pipeline/ltx2_sampler_smoke.mojo` (266 lines)

## Rust references read line-by-line
- `inference-flame/src/sampling/ltx2_sampling.rs` (226 lines)
- `inference-flame/src/sampling/ltx2_guidance.rs` (259 lines)
- `inference-flame/src/sampling/ltx2_multiscale.rs` (153 lines)
- `inference-flame/src/bin/ltx2_generate_av.rs` (combine, lines 590-676)  ← **Q1 oracle**
- `inference-flame/src/bin/ltx2_generate_kf.rs` (combine, lines 710-750)  ← Q1 confirm
- Python refs: `ltx2_sigma_schedule_ref.py`, `ltx2_cfg_star_ref.py`, `ltx2_stg_mask_ref.py`, `ltx2_stg_rescale_ref.py`

---

## A. Sigma schedule parity

### A1 — `linear_quadratic_schedule` port — **PASS**
Mojo `build_ltx2_distilled_sigma_schedule` (ltx2_sampling.mojo:99-157) is a line-for-line port of Rust `linear_quadratic_schedule` (ltx2_sampling.rs:32-62):
- `linear_steps = num_steps // 2`, `quadratic_steps = num_steps - linear_steps` ✓ (rs:36-37)
- `slope = threshold_noise / linear_steps` ✓ (rs:39)
- `tn_step_diff = linear_steps - threshold_noise * num_steps` ✓ (rs:40)
- `quadratic_coef = tn_step_diff / (linear_steps * quadratic_steps²)` ✓ (rs:41-42)
- `linear_coef = threshold_noise/linear_steps - 2*tn_step_diff / quadratic_steps²` ✓ (rs:43-44)
- `const_coef = quadratic_coef * linear_steps²` ✓ (rs:45)
- ascending linear branch `[0, linear_steps)` = `slope*i`; quad branch `[linear_steps, num_steps)` = `qc*i²+lc*i+cc` ✓ (rs:50-56). Branch boundary identical.
- `ascending.append(1.0)` ✓ (rs:57)
- `descending = [1-x]`, drop last (`sigma_schedule[:-1]`) ✓ (rs:59-61 `.pop()`)
All arithmetic in F32 in both (Rust `f32`, Mojo `Float32`). `num_steps==1` early-return `[1.0]` matches (rs:33-35, mojo:126-129).

### A2 — Hardcoded distilled tables byte-for-byte — **PASS**
See "Sigma tables" section. Both 9-val and 4-val tables match the Rust constants exactly.

### A3 — Cross-check vs `ltx2_sigma_schedule_ref.py` — **PASS**
Python imports Lightricks's actual `linear_quadratic_schedule(8, 0.025)` (script lines 30-32, 88-91). The Mojo 8-step output `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.7250001, 0.421875]` matches the distilled table first-8 within F32 epsilon (the `0.7250001` is F32 rounding of the quadratic-branch eval; the hardcoded table value `0.725` is bit-exact). Rust `bin/ltx2_sigma_parity` reports max_abs=0.0 for n=8/20/25/30 (per source doc rs:23-24), so the formula is parity-clean; the Mojo formula reproduces it.

### A4 — Euler step sign + final-step special case — **PASS**
Mojo `LTX2Scheduler.step` (ltx2_sampling.mojo:233-263):
- final (`sigma_next == 0`): `x - v*sigma` ✓ — matches Rust `euler_denoise_ltx2` rs:211-214 `x.sub(&velocity.mul_scalar(sigma))` AND the inline generate loop `ltx2_generate_av.rs:660` `video_x.sub(&video_vel.mul_scalar(sigma))`.
- interior: `x + v*(sigma_next - sigma)` ✓ — matches rs:216-218 `dt = sigma_next - sigma; x.add(&velocity.mul_scalar(dt))` AND av.rs:662-663.
Sign and special-case correct. Smoke check [6] confirms numerically: x=[10,20], v=[1,2], sigma=0.421875 → [9.578125, 19.15625] ✓.

---

## B. CFG-star

### B1 — `cfg_star_rescale` math + reduction axis — **PASS**
Mojo `cfg_star_alpha` (ltx2_guidance.mojo:164-200) vs Rust `cfg_star_rescale` (rs:92-143):
- `alpha = dot(text,uncond) / (||uncond||² + 1e-8)` ✓ — Rust rs:127 `(dot / (nsq + 1e-8))`; +1e-8 added to squared norm, NOT the ratio ✓ (matches Python ref line 31 `squared_norm = ... + 1e-8`).
- Per-batch: `flat_len = prod(dims[1:])`, loop `b in 0..batch` over the full flattened slab ✓ (rs:100-101,116-126). NOT per-token — reduction is over the entire `[flat_len]` per batch row. Correct axis.
- Both reduce in F64 (`dot`, `nsq` as `f64`/`Float64`), cast alpha to F32 ✓ (rs:119-127, mojo:190-198).
- alpha tensor `[batch,1,1,…]` broadcast multiply ✓ (rs:131-141, mojo:222-229).

### B2 — Q1 combine convention — **RESOLVED, PASS** (see Builder question resolutions below).

### B3 — Cross-check vs `ltx2_cfg_star_ref.py` — **PASS**
Python (lines 27-34) computes `dot_product / squared_norm` per-batch, `rescaled = alpha.view(B,1,…) * eps_uncond`. The Mojo `cfg_star_rescale` returns exactly this `rescaled` term. Smoke [2]: text=[1,2,3,4], uncond=[2,2,2,2] → dot=2+4+6+8=20, nsq=16, alpha=20/16=1.25 ✓, rescaled=1.25*2=2.5 ✓. Rust parity oracle is max_abs=0.0 BF16 on this op (source doc); Mojo reproduces the F32 math.

---

## C. STG

### C1 — `build_skip_layer_mask` + `stg_rescale` unbiased std — **PASS**
- Mask (mojo:96-130) shape `[num_layers][batch*num_conds]`, zeros at `ptb_index, ptb_index+num_conds, …` on `skip_block_list` rows, OOB block-idx `continue` ✓ — exact match to Rust rs:64-84 and Python ref `mask[block_idx, ptb_index::num_conds] = 0`.
- Smoke [4]: (4,1,3,skip=[1,3],ptb=2) → `[[1,1,1],[1,1,0],[1,1,1],[1,1,0]]` ✓ — matches Rust unit-test `build_mask_matches_lightricks_small` (rs:228-239) and `ltx2_stg_mask_ref.py` small case (lines 59-66).
- `stg_rescale` (mojo:289-331) unbiased (n-1) std in F64: `_unbiased_std_f64` divides by `nf-1.0` ✓ — matches Rust `unbiased_std_f64` rs:198-203 (`/ (n - 1.0)`) and torch `.std` default (Python ref line 28). factor = `scale*f + (1-scale)` ✓ (rs:183, mojo:322).
- Smoke [5]: pos=[1,2,3,4] std=√(5/3)≈1.291, guided=[2,4,6,8] std≈2.582, ratio=0.5, factor=0.7*0.5+0.3=0.65, out=[1.3,2.6,3.9,5.2] ✓.

### C2 — `SkipLayerStrategy` enum values — **PASS**
Mojo tags: AttentionSkip=0, AttentionValues=1, Residual=2, TransformerBlock=3 (mojo:44-59). Rust enum order is AttentionSkip, AttentionValues, Residual, TransformerBlock (rs:19-38) — same ordinal mapping. `from_str` aliases match exactly (`stg_av`/`attention_values` → AttentionValues, etc.; Rust rs:42-49, Mojo:67-80). Mojo additionally accepts uppercase spellings — superset, no conflict. Smoke confirms `from_str("stg_av")==1`.

---

## D. Scope discipline

### D1 — No dev-mode / FlowMatchEuler / build_dev_sigma_schedule ported — **PASS**
Grep of the two new sampling Mojo files: the only `build_dev_sigma_schedule` / `ltx2_scheduler_sigmas` hits are in the module-header OMISSION doc-block (ltx2_sampling.mojo:6-18), not ported code. No `FlowMatchEulerDiscreteScheduler` implementation. Header explicitly documents the omission with the Rust rationale (build_dev is Flux-style/FLAGGED-BUGGY; FlowMatchEuler unimplemented in Rust so no parity oracle; ltx2_scheduler_sigmas TODO-PARITY rs:81-82). Scope discipline clean.

### D2 — Multiscale/AdaIN deferral justified — **PASS**
`ltx2_multiscale.rs` `adain_filter_latent` (rs:41-107) uses `mean_dim(&[2,3,4])`, `sum_dims(&[2,3,4])` — per-(B,C) reductions over the (F,H,W) slab of a 5D NCFHW tensor. `tone_map_latents` (rs:123-152) needs `sigmoid`. serenitymojo's `ops/tensor_algebra.mojo` exposes only elementwise + broadcast (add/sub/mul/div/scalar/reshape/permute/concat/slice/gather) — NO axis-reduction. `ops/norm.mojo` has only fused rms/layer/group_norm kernels (not a general per-channel reduce), and no `sigmoid`. The deferral reason is genuine: AdaIN needs a per-channel (F,H,W) reduction primitive that does not exist yet.

---

## E. Smoke run (GPU)

GPU free: 22988 MiB. Rebuilt with `pixi run mojo build -I . serenitymojo/pipeline/ltx2_sampler_smoke.mojo -o /tmp/ltx2_sampler_smoke` (build exit 0).

```
=== LTX-2 distilled sampler + guidance smoke ===

[1] build_ltx2_distilled_sigma_schedule(8, 0.025)
      [ 0 ] = 1.0  ...  [ 6 ] = 0.7250001  expected 0.725  [ 7 ] = 0.421875
    PASS  (== distilled LTX2_DISTILLED_SIGMAS[:8], +0.0 terminator)
    stage2 sigmas: 0.909375 0.725 0.421875 0.0   stage2 PASS

[2] cfg_star_rescale: alpha = 1.25 (exp 1.25), rescaled = 2.5 2.5 2.5 2.5   PASS

[3] cfg_star full combine (scale=3.0)  out min/max/mean = -2.0 7.0 2.5
    (manual verify: 2.5 + 3*(cond-2.5) for cond=[1,2,3,4] = [-2,1,4,7] ✓)

[4] build_skip_layer_mask(4,1,3,skip=[1,3],ptb=2)
    [[1,1,1],[1,1,0],[1,1,1],[1,1,0]]   PASS
    single_cond = 1,0,1,0   from_str('stg_av')==1

[5] stg_rescale  out = 1.3 2.6 3.8999999 5.2 (exp 1.3,2.6,3.9,5.2)   PASS

[6] LTX2Scheduler.step final denoise  num_steps=8  out = 9.578125 19.15625   PASS

=== OVERALL: PASS ===
run_exit=0
```

The sigma-schedule assert PASSes; cfg_star + stg produce finite output; final-step denoise correct.

**WARN (informational, follow-on):** the smoke asserts in F32, but the Rust parity oracles (`ltx2_guidance_parity`) are validated at BF16 output (max_abs=0.0 BF16). A BF16 re-verify of `cfg_star_rescale` / `stg_rescale` against the dumped `*_ref.safetensors` `rescaled_bf16` / `out_bf16` tensors is recommended as a follow-on (not a blocker — the F32 math is bit-identical to the Rust F32 path; only the final dtype cast is unverified in Mojo).

---

## F. No regressions

### F1 — Only new files — **PASS (with note)**
`git status --short serenitymojo/sampling/ serenitymojo/pipeline/ltx2_sampler_smoke.mojo`:
- `?? serenitymojo/pipeline/ltx2_sampler_smoke.mojo`
- `?? serenitymojo/sampling/ltx2_guidance.mojo`
- `?? serenitymojo/sampling/ltx2_sampling.mojo`
- ` M serenitymojo/sampling/__init__.mojo` (doc-comment only — see WARN)
(Many other unrelated `??` sampling files exist from prior sessions; not part of this work.)

**WARN (informational):** `sampling/__init__.mojo` is modified but the diff adds ONLY doc-comments for OTHER modules (sd15_euler, flux1_dev, sd3_flow_match, etc.) — it does NOT add an entry for `ltx2_sampling` / `ltx2_guidance`. Not a regression; the new modules are importable by full path (the smoke does so). Minor: the ltx2 modules are undocumented in the package index.

### F2 — No flow_match shadow/break — **PASS**
The new files do NOT import `flow_match` (grep: only doc-comment references at ltx2_sampling.mojo:23,33,171 and ltx2_guidance.mojo:16,18). The Euler arithmetic and host-F64 reduction pattern are re-implemented via `ops.tensor_algebra` directly, not by calling flow_match. No shadowing, no duplication of flow_match symbols, flow_match untouched.

---

## Builder question resolutions

### Q1 — CFG-star combine convention — **RESOLVED. Builder's combine is CORRECT.**
**Definitive answer:** `out = rescaled_uncond + cfg_scale * (cond - rescaled_uncond)`.

The Rust `cfg_star_rescale` returns ONLY the rescaled uncond term; the combine is in the generate binaries. **File:line evidence:**
- `inference-flame/src/bin/ltx2_generate_av.rs:614-621`:
  ```rust
  if cfg_star {
      video_uncond = cfg_star_rescale(&video_cond, &video_uncond)?;  // uncond <- alpha*uncond
  }
  // noise_pred = uncond + cfg_scale * (cond - uncond)
  let v_guided = video_uncond.add(&video_cond.sub(&video_uncond)?.mul_scalar(cfg_scale)?)?;
  ```
- Confirmed identical at `inference-flame/src/bin/ltx2_generate_kf.rs:716-721`.

The anchor is the **rescaled uncond** (uncond is overwritten in-place by cfg_star_rescale before the combine), and the scale multiplies `(cond - rescaled_uncond)`. This is EXACTLY the Mojo `cfg_star` (ltx2_guidance.mojo:252-255):
```mojo
var rescaled = cfg_star_rescale(v_cond, v_uncond, ctx)   # alpha * v_uncond
var diff = sub(v_cond, rescaled, ctx)                    # cond - rescaled
var scaled = mul_scalar(diff, scale, ctx)                # scale * diff
return add(rescaled, scaled, ctx)                        # rescaled + scale*diff
```
The builder's inference from pipeline_ltx_video.py was right. No bug. Smoke [3] numerically confirms (out=[-2,1,4,7] for cond=[1,2,3,4], rescaled=2.5, scale=3).

### Q2 — F64 host reductions vs Rust precision — **RESOLVED. Precision matches. No drift concern at the reduction; minor follow-on at final cast.**
**File:line evidence:**
- CFG-star: Rust accumulates `dot`/`nsq` in `f64` (`ltx2_guidance.rs:119-126`, `let mut dot = 0.0f64`), casts alpha to `f32` (rs:127). Mojo does the identical thing (`cfg_star_alpha` mojo:190-198, `Float64` accumulators, `Float32(dot/(nsq+1e-8))`). **EXACT match** — both reduce in F64, not pure-F32 device-side. The source comment rs:109-110 even states "reduces in f64 for parity with the Python reference."
- STG: Rust `unbiased_std_f64` (rs:198-203) sums in `f64`, divides by `n-1.0`, `.sqrt()`; Mojo `_unbiased_std_f64` (mojo:269-286) identical. **EXACT match.**

The host-F64 reduction is the SAME path the Rust takes (Rust also pulls flats to host/CPU and reduces in f64 — rs:108-129). So there is NO host-vs-device F64 drift risk: both are host-F64. The only unverified piece is the final BF16 output cast (the multiply runs F32, then cast to input dtype) — this is the same WARN as in section E (BF16 re-verify follow-on). Not a blocker; the alpha/factor scalars are F64-reduced identically to Rust.

---

## Sigma tables (Rust vs Mojo)

### Distilled stage-1 (`LTX2_DISTILLED_SIGMAS`, ltx2_sampling.rs:9-11)
Rust:
```
[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]   (9 values)
```
Mojo `ltx2_distilled_sigmas()` (ltx2_sampling.mojo:50-67): identical 9 values. **BYTE-EXACT.**

### Stage-2 refine (`LTX2_STAGE2_DISTILLED_SIGMAS`, ltx2_sampling.rs:15-17)
Rust:
```
[0.909375, 0.725, 0.421875, 0.0]   (4 values)
```
Mojo `ltx2_stage2_distilled_sigmas()` (ltx2_sampling.mojo:70-82): identical 4 values. **BYTE-EXACT.**

Note: the *computed* `build_ltx2_distilled_sigma_schedule(8,0.025)` yields `0.7250001` at index 6 (F32 rounding of the quadratic-branch eval) — within tolerance of the hardcoded `0.725`. The hardcoded TABLE is bit-exact; the FORMULA reproduces it to F32 epsilon (parity-clean per Rust max_abs=0.0). The Euler sampler uses the hardcoded bit-exact table via `LTX2Scheduler.distilled()`.

---

## Bugfix Worklist (ordered)
1. (P3, follow-on) BF16 re-verify: add a smoke variant that loads `output/ltx2_cfg_star_ref.safetensors` (`rescaled_bf16`) and `output/ltx2_stg_rescale_ref.safetensors` (`out_bf16`) and diffs the Mojo BF16-cast output at 1e-3, matching the Rust `ltx2_guidance_parity` tolerance. The F32 math is already bit-clean; only the final BF16 cast is unverified in Mojo.
2. (P4, cosmetic) Add `ltx2_sampling` + `ltx2_guidance` entries to `serenitymojo/sampling/__init__.mojo` module index doc-block (currently they are the only un-indexed new modules).
3. (P5, future) When a device per-(B,C)-over-(F,H,W) reduction + `sigmoid` op lands, port `adain_filter_latent` / `tone_map_latents` from `ltx2_multiscale.rs` (currently correctly deferred).

No source modifications were made (read-only audit + smoke run).
