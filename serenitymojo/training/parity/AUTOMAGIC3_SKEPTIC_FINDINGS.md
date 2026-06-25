# Automagic3 Mojo port — adversarial skeptic findings (2026-06-24)

Reviewer: fresh-eyes skeptic (did NOT build it). Assumption: the port lies.

Scope read:
- REFERENCE: `/home/alex/ai-toolkit/toolkit/optimizers/automagic3.py`
- PORT: `serenitymojo/training/automagic3.mojo`
- WIRING: `serenitymojo/training/levers.mojo` (lazy-init, dispatch, validate),
  `training/train_config.mojo` (=9), `io/train_config_reader.mojo` (string map)
- CROSS-REF: `EriDiffusion-v2 .../optimizers.rs:557-763` (Automagic3)
- GATE: `automagic3_parity_probe.mojo` + `oracle.safetensors` + `gen_oracle.py`

Parity probe RE-RUN: **PASSES** (lr-traj rel ≤ 4e-8, final A/B/C cos ≥ 0.99999).
The math on the oracle's F32 path is faithful. The findings below are about
what the oracle does NOT exercise, plus one documentation lie.

---

## BLOCKER

### B1 — bf16 LoRA params get plain RNE writeback; the reference's STOCHASTIC ROUNDING (its documented improvement #4) is silently dropped → the weight effectively does not learn at LoRA lr.

This is the load-bearing one. The port header (automagic3.mojo:48-49, 552-553)
and the Rust note both claim *"our LoRA params are F32 / F32 writeback only"*.
That is FALSE for the actual wiring:

- `LoraAdapter.a` / `.b` are `List[BFloat16]` (train_step.mojo:134-135), NOT F32.
- The levers dispatch (levers.mojo:531-541) does
  `_levers_bf16_to_f32` → `automagic3_step_2d` → `_levers_writeback_bf16`.
- `_levers_writeback_bf16` (levers.mojo:332-336) is **plain RNE**:
  `p[i] = BFloat16(src[i])`.

The Python applies the update in fp32 then **stochastically rounds** back
(automagic3.py:536-544 → `_stochastic_copy_` → `_sr_truncate`, drop 16 bits).
Its own docstring (lines 154-159, improvement #4) says round-to-nearest
"silently discard[s] updates smaller than an ULP … so it actually keeps
learning instead of stalling."

Measured divergence (1000 consistent-direction steps, lr·|update|=3e-5/step,
w₀=0.1):
```
f32 reference : 0.070000   (what should happen)
bf16 + SR     : 0.066895   (Python automagic3)
bf16 + RNE    : 0.100098   (Mojo levers writeback)  <-- weight never moved
```
At the optimizer's own default start lr (1e-4) and the RMS+element clip cap of
1.0, the MAX per-step update is ≤ 1e-4. For any weight ≥ 0.1 the bf16 ULP
(≥ 4.9e-4) exceeds that, so under RNE the typical step rounds to nothing. The
controller will keep voting "step too small" and ramp the lr, but the weight
stays pinned until lr·update crosses ~½ ULP — a qualitatively different (and
broken) training trajectory than the reference. This is precisely the failure
mode SR was added to prevent.

Triage rationale BLOCKER (not FRAGILE): for THIS optimizer, SR is not an
incidental dtype detail — it is a named, deliberate algorithm feature, and the
optimizer's whole regime (tiny start lr, multiplicative controller) lives right
in the sub-ULP zone where it matters. A run wired through bf16 LoRA would
mistrain. The parity oracle is F32-only (gen_oracle.py: `DT=torch.float32`,
init·0.02, F32 params), so it CANNOT catch this — green gate, broken run.

Caveat on blast radius: `levers_optimizer_step` (the full GPU+bf16 path) has
**no production caller yet** (only the host-dispatch test + the F32 probe run
today). So this is a *latent* BLOCKER that fires the moment a trainer wires
automagic3 onto real bf16 adapters. Same `_levers_writeback_bf16` RNE is shared
by adafactor / schedule-free / adamw8bit, BUT automagic3 is the only one whose
reference implements SR *specifically to fix this*, so the divergence is a
dropped algorithm feature here, not just dtype noise.

Fix direction (do NOT apply): make `_levers_writeback_bf16` (or an automagic3
variant) stochastically round, mirroring `_sr_truncate` (add uniform noise into
the low 16 mantissa bits, mask, narrow), OR keep an F32 master copy of the LoRA
params and round only for the device view. Then add a bf16 oracle column.

---

## FRAGILE

### F2 — `up`/`down` use `elif` (Mojo) vs independent masks (Python); only correct because the two events are mutually exclusive. Holds for H≥2, but it's an unguarded assumption the oracle (H=8) never stresses.
Python (automagic3.py:508-511): `up` and `down` are separate boolean masks,
combined as `num = (w*up).sum() - (w*down).sum()`. Mojo (automagic3.mojo:341-344):
`if s1==h or s1==0: num+=w   elif flips==h-1: num-=w`. These agree ONLY if no
element is simultaneously all-agree AND perfect-alternation. For H≥2 that's
impossible (all-same ⟹ 0 flips; alternation ⟹ H-1 flips), so the `elif` is
safe. Flagged because (a) the default-clamp floor is H=2 and the H=2 corner is
not in any test, and (b) it's a behavioral coupling that a future H-related edit
could silently break. No bug today.

### F3 — `grad.nan_to_num_` is omitted (documented), but the Mojo has NO non-finite guard at all on the second-moment EMA.
automagic3.py:416 neutralises NaN/±inf grads in place before they poison the
EMA (NaN is sticky forever). The Mojo header (automagic3.mojo:20-21) says this
is "omitted — documented" on the assumption LoRA grads are finite. If an inf/NaN
grad ever reaches the step (loss spike, fp16 overflow upstream), `state.v` /
`row_var` / `col_var` become NaN permanently and silently corrupt the run, with
no nan-skip downstream of the levers step. Latent; depends on the grad source
being trustworthy. Lower-risk than B1 but worth a guard.

### F4 — eps placement is correct for BOTH paths; calling out because it is the exact landmine the header warns about and a future "simplification" could break it.
Verified MATCH:
- dim≥2: eps folded INSIDE the EMA add — Python `sq.mean(dim=-1).add_(eps)`
  (lines 437-438) == Mojo `row_mean + eps_f` inside the EMA (automagic3.mojo:246,
  256). The `_approx_sq_grad` reconstruction adds NO eps (Python 240-242; Mojo
  rfac/cfac 266-271 add none). Correct.
- 1D: eps added AFTER, in the rsqrt — Python `v.add(eps).rsqrt()` (line 456) ==
  Mojo `rsqrt(v + eps)` (automagic3.mojo:281). Correct.
This asymmetry (folded for 2D, post for 1D) is real and faithfully reproduced.
Not a bug — a fragile invariant.

---

## DISAGREE-WITH-CONCERN

### D5 — port header / Rust comment assert "LoRA params are F32". They are bf16 (see B1). The comment is actively misleading and is what would lull a maintainer into trusting the F32-only oracle.
automagic3.mojo:48 ("bf16 a/b cast -> f32 master -> RNE writeback") at least
admits the bf16 round-trip, but lines 552-553 of the Rust and the spirit of the
Mojo note ("F32 writeback only … not ported") frame the dropped SR as a no-op.
It is not a no-op (B1). Recommend correcting the comment regardless of whether
SR is implemented, so the next reader knows the oracle does not cover the real
dtype.

---

## CHECKED HARDEST AND FOUND CORRECT

1. **Factored 2nd moment** — axes (`mean over cols` → row[rows]; `mean over rows`
   → col[cols]), beta2 placement, eps-folded-vs-post, and the
   `rsqrt(row/mean(row)) ⊗ rsqrt(col)` rank-1 reconstruction all MATCH
   automagic3.py:432-456. `mean(row)` is the scalar mean over the [rows] vector
   (last dim of the 1-D row state) — Mojo `rmean` (automagic3.mojo:262-265) is
   exactly that. Row-major `g[r*cols+c]` verified for BOTH orientations
   (A[16,64] and B[64,16] both pass cos≥0.99999).

2. **Two-stage trust region** — RMS-clip (`update /= max(RMS/clip, 1)`) THEN
   element clamp ±clip, in that order, with RMS computed on the pre-clip update
   in BOTH (Python 462→467; Mojo 285-300). RMS = ‖u‖₂/√n over all n elements.
   Reductions accumulate in Float64 (automagic3.mojo:285-288). MATCH.

3. **Sign-history controller** — the novel part:
   - chronological roll: Python `roll(hist, -hist_idx)` ≡ Mojo
     `(hist_idx + k) % h` (verified numerically, identical order).
   - up = all-agree (s1==H or s1==0), down = flips==H-1, weight = |update|,
     pooled across the WHOLE group (A+B together via the shared `ctl`), NOT
     per-tensor — MATCH (automagic3.py:506-524 vs automagic3.mojo:320-345 +
     levers reset/loop/apply).
   - `lr *= exp(clamp(num/max(den,1e-30), ±1))`, clamp [1e-30,1e3] —
     MATCH (automagic3.py:587-597 vs automagic3.mojo:189-203). signal is
     clamped to [-1,1] BEFORE exp in both → exp∈[0.37,2.72], no overflow path.
   - warmup: controller abstains until fill==H (first H steps); param update
     ALWAYS applies. The probe confirms lr is flat for steps 0-6 and first moves
     at step 7 (after the 8th sign at H=8) — off-by-one is CORRECT (fill reaches
     H on the step that stores the Hth plane, vote fires that same step).
   - ONE shared lr per group, accumulated once across [start,end) and nudged
     ONCE after the loop (levers.mojo:528-542). No per-tensor leak. step_lr
     correctly IGNORED (controller self-adapts).

4. **Decoupled WD** — `update += wd*p` (after the vote/clamp, so it does NOT
   affect the sign bit or vote weight) then `p -= lr*update`. Algebraically
   `p_new = p - lr·update - lr·wd·p`, identical to Python (533-535) and Rust
   (735-740). The oracle uses wd=0 so this path is untested numerically, but the
   algebra and ordering are verified correct.

5. **Levers wiring** — lazy-init builds 2 states/adapter with the right ctors:
   `Automagic3State(rank, in_f, h)` for A[rank,in_f], `(out_f, rank, h)` for
   B[out_f,rank] (levers.mojo:387-393) — shapes match the adapter. eps uses the
   algorithm default 1e-30 (NOT cfg.eps=1e-8 which is the AdamW denom — correctly
   overridden, levers.mojo:521-523). beta2 falls back to 0.999, clip to 1.0,
   both overridable. No silent AdamW fallback (validate + dispatch both list
   AUTOMAGIC3 explicitly; unknown tags raise). Config tag=9 and the string map
   (AUTOMAGIC3 / AUTOMAGIC_3 / AUTOMAGIC-3) are wired.

6. **≥3D** — genuinely moot: LoRA adapters are always 2D, `Automagic3State` only
   has 2D-factored and 1D ctors, and the levers lazy-init only ever builds those.
   No code path constructs a ≥3D state. Confirmed NOT a risk for the wired use.

7. **rows==cols** — no special-casing on rows vs cols anywhere; the factored math
   is symmetric. (Oracle never uses square, but there is no shape-dependent
   branch that could break it.)
