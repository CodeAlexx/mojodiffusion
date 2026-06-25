# Skeptic findings — krea2 chunk 1 (Krea2Config + 3-axis interleaved RoPE)

Date: 2026-06-24
Reviewer: fresh-eyes skeptic (did NOT write the code)
Scope: `serenitymojo/models/dit/krea2_dit.mojo`, `serenitymojo/models/dit/parity/krea2_rope_probe.mojo`
Reference (read line-by-line): `/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src/mmdit.py` + `krea2.py`

## Method
- Read all four files line-by-line (builder, probe, reference mmdit.py, reference krea2.py).
- Read the two foundation ops the builder reuses/forks: `ops/rope.rope_interleaved`, `ops/rope_tables.build_multiaxis_rope_tables`.
- Re-ran the probe myself: **exit 0, "PROBE OK"** (builder's claim is honest).
- Independently reproduced the torch reference `rope()`/`ropeapply()`/`PositionalEncoding.forward` in Python on the probe's exact tiny inputs (L=8, axes=[32,48,48], theta=1e3) and cross-checked cos/sin AND applied q_rot against the Mojo kernel output. **They match to F32 precision (~3e-8).**

## Verdict-relevant verification (all PASS)
These were the highest-risk silent-bug candidates; I confirmed each against the torch reference, not against the builder's prose:

1. **Per-axis omega denominator is correct.** Kernel uses `ha = axes_half[a]` (16/24/24) as the exponent denominator, NOT a global half=64. Reference `rope(pos, dim)` uses `scale = arange(0,dim,2)/dim` where `dim` is the per-axis dim → exponent `-i/(dim/2) = -i/half_a`. Match.
2. **Axis order [global, h, w] and split [32,48,48] → column ranges [0:16),[16:40),[40:64) (half-units).** Reproduced the reference `torch.cat(..., dim=-3)` freqs layout: pair 0 = axis global (cos(5·1)=0.2837), pair 16 = axis h (cos(2)=−0.4161), pair 40 = axis w (cos(1)=0.5403). Mojo kernel's offset walk `[0,16,40]` produces identical owning-axis/local-i for every column. Match.
3. **theta = 1e3, not 1e4.** Confirmed in source: `KREA2_MMDIT_CONFIG` (krea2.py:55) does NOT contain `theta`, so `SingleMMDiTConfig(**mmdit_kwargs)` takes the dataclass default `theta: float = 1e3` (mmdit.py:102). `SingleStreamDiT.__init__` passes `theta=config.theta` (mmdit.py:356), overriding both `rope()`'s 1e4 default (mmdit.py:31) and `PositionalEncoding.__init__`'s 1e2 default (mmdit.py:137). Builder's three-way theta claim is correct.
4. **Interleaved pairs, not half-split.** Reference `ropeapply` does `xq.reshape(*shape, -1, 1, 2)` → adjacent pairs `(x[2p], x[2p+1])` rotated by freq entry `p`. This is exactly `rope_interleaved`'s convention (rope.mojo:7-9, 58-59). NOT half-split. Match — no BLOCKER here.
5. **2×2 sign convention.** Reference `freqs[...,0,:]=[cos,−sin]`, `freqs[...,1,:]=[sin,cos]`; `out0=cos·x0−sin·x1`, `out1=sin·x0+cos·x1`. Identical to `rope_interleaved` (rope.mojo:58-59). Match (verified numerically on q_rot token0 and token5).

## Findings

### F1 — Docstring misstates the reference's trig precision (STYLE)
- **Where:** `krea2_dit.mojo:30-31` (comment) and `:188` docstring.
- **What:** The comment claims *"The torch reference computes omega in F64 and trig in F32 with proper reduction, so this matches it."* I verified the reference: `rope()` computes `scale`/`omega` in F64, the `einsum` angle in F64, and **`torch.cos/torch.sin` in F64**, only `.float()`-casting at the very end (mmdit.py:32-39). So the reference does trig in **F64**, not F32. The kernel does F64 range-reduction + F32 trig on the small remainder, which agrees with F64 trig to ~1e-7 (measured) — so the *implementation* is fine, but the *justification prose* is factually wrong about the reference. Harmless to numerics; fix the comment so a future reader doesn't propagate the wrong model of the oracle.
- **Severity:** STYLE.

### F2 — `log(theta)` computed in F32, defeating the kernel's own F64 rationale (FRAGILE)
- **Where:** `build_krea2_rope`, `krea2_dit.mojo:237` (`var lt = log(theta)`, where `theta: Float32`), consumed at `:158` as `Float64(log_theta)`.
- **What:** The entire reason this kernel forks from `build_multiaxis_rope_tables` is F64 precision. But the single most-reused constant, `log(theta)`, is computed in **F32** (`theta` is `Float32`, `log` returns `Float32`) and only *then* widened to F64. Measured worst-case impact at the production max position (pos≈6400, low-freq axis): omega rel-err ≈ **9.2e-8**, angle abs-err ≈ **3.2e-5 rad**, vs ≈3.6e-13 rad if `log` ran in F64. That's ~1e8× larger error than a free F64 `log` would give. It is a *systematic bias*, not noise.
- **Why only FRAGILE, not BLOCKER:** 3.2e-5 rad → cos/sin error ~3e-5, far inside any cos≥0.999 gate and ~100× below the bf16 storage quantum (~4e-3). The probe's torch cross-check still matches (the tiny L=8 positions never reach the regime where it bites). It will not fail parity, but it silently throws away the precision the fork was built to gain.
- **Minimal fix:** compute the log in F64: `var lt = log(Float64(theta))` and change the kernel param to `log_theta: Float64` (drop the inner `Float64(log_theta)` widen). One-line, free precision win.
- **Severity:** FRAGILE.

### F3 — Probe self-check is a tautology; never compares to the oracle (FRAGILE — test quality)
- **Where:** `krea2_rope_probe.mojo:104-158`.
- **What:** The "host reference self-check" recomputes the rope on the host using **the same F64-omega + same range-reduction math** as the kernel and asserts they agree (max_err < 1e-5). That only proves the GPU kernel matches a CPU re-run of the identical formula — it cannot catch a wrong axis order, wrong theta, wrong split, or wrong sign convention, because the host recompute would share the same mistake. The probe gates "kernel == my own restatement of the kernel," not "kernel == torch." (I closed this gap manually with a Python torch cross-check, which passed — but the committed probe does not.)
- **Why FRAGILE not STYLE:** the parity gate for this chunk is, as written, blind to every class of reference-fidelity bug. A real regression in axis order or theta would still print "PROBE OK".
- **Minimal fix:** dump the reference cos/sin (and an applied q_rot row) from the torch `rope()`/`ropeapply()` on the SAME tiny inputs into a fixture, and assert the kernel matches the *fixture*, not a host recompute of itself. (The numbers I generated: torch `cos[token5,0..3] = 0.2837, −0.9945, −0.5122, 0.2002`; `q_rot[token5,0..3] = 0.0841, −0.0377, −0.0701, −0.0979` — kernel matches all.)
- **Severity:** FRAGILE.

### F4 — bf16/f16 table branches are compiled but unexercised and likely-dead (STYLE)
- **Where:** `krea2_dit.mojo:250-271` (bf16/f16 store branches); probe only calls with `STDtype.F32` (`krea2_rope_probe.mojo:53`).
- **What:** The reference always materializes freqs in F32 (`rope()` ends `.to.float()`), and `rope_interleaved` consumes F32 tables even for bf16/f16 q/k via its `_f32_tables` path (rope.mojo:418-438). So the production krea2 path will use F32 tables; the bf16/f16 *table* branches here are speculative and never hit. Storing cos/sin in bf16 before the apply WOULD lose precision (the prompt's concern is real), but the right answer is "don't" — which the F32 default already enforces. Not a bug, just unused surface that the probe gives zero coverage to. If kept, a future caller could pass bf16 and silently degrade; consider removing the non-F32 branches or asserting F32 until a real consumer needs otherwise.
- **Severity:** STYLE.

## Mojo-correctness sweep (all clean)
- `comptime` used throughout (no `alias`). ✓
- No variable named `ref`. ✓
- `build_krea2_rope` and `apply_krea2_rope` are `raises` (`:176`, `:287`); GPU kernel `_krea2_rope_kernel` is a non-raising `def`, matching `_multiaxis_rope_kernel` in rope_tables.mojo. ✓
- `enqueue_function[F, F]` double-arg form matches the established rope_tables.mojo pattern. ✓
- I32 axes uploaded via host-buffer bitcast — byte-identical to rope_tables.mojo. ✓
- No reimplementation of `rope_interleaved` — the apply correctly reuses the existing op (`:298-299`). ✓
- The fork of `build_multiaxis_rope_tables` (rather than reuse) IS justified: the existing builder does plain F32 trig with no range reduction (rope_tables.mojo:86-89), which is wrong for krea2's global axis at thousands of radians. The fork adds F64 omega + mod-2π reduction. Justified, not needless duplication. (Caveat: see F2 — the fork's precision is partly self-sabotaged by the F32 `log`.)
- Range-reduction logic `k=floor(angle/2π+0.5); reduced=angle−k·2π` matches the established idiom in `ideogram4_mrope.mojo:54-57`. ✓

## Verdict
**0 BLOCKERS.** The math is faithful to the reference — axis order, per-axis split, theta=1e3, interleaved pairing, and the 2×2 sign all verified numerically against torch. Three non-blocking issues: F2 (FRAGILE — F32 `log` wastes the kernel's F64 rationale, free one-line fix), F3 (FRAGILE — the probe's self-check is a tautology, can't catch reference-fidelity bugs), F1/F4 (STYLE — docstring misstates the oracle; unused bf16 table branches). Chunk 1 is correct as committed; recommend F2 + F3 before this becomes the load-bearing gate for chunk 2.
