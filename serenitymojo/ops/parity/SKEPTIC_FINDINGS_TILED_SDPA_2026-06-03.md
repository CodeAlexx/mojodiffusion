# SKEPTIC FINDINGS — Tiled (online-softmax) SDPA — 2026-06-03

Component: `sdpa_tiled` / `sdpa_nomask_tiled` (serenitymojo/ops/attention.mojo).
Reviewer ran every check below ITSELF on RTX 3090 Ti (sm_86). Verdict: **clean**.

## §0 Receipts
- `serenitymojo/MAP.md`: serenitymojo is pure-Mojo+MAX, inference-only, GPU-only diffusion lib; ops hand-rolled on the Mojo SDK; BF16 storage / F32 accumulation.
- `ops/attention.mojo` `_sdpa_math` (parity ref): cuBLAS QKᵀ → `_scale_mask_f32` (scale, then +mask) → `_softmax_rows_f32` (F32 last-dim softmax) → P·V; mask read as `[B*H*S, S]` view of `[B,H,S,S]`, `mask[r=(b*H+h)*S+i, c=j]`.
- `mojo-gpu-fundamentals/SKILL.md` + mojo-syntax: one-thread-per-output, `stack_allocation` register arrays, `comptime` params, `def` raises — kernel matches idioms.
- `ops/softmax.mojo`: same two-pass max/exp-sum F32 reduction the math path mirrors; online recurrence is its streaming-exact equivalent.

## RECURRENCE — re-derived from scratch — CORRECT (recurrenceCorrect: true)
Kernel `_sdpa_online_f32` lines 579-585:
- `m_new = max(m, s)` ✓
- `corr = exp(m - m_new)` ✓ (rescale for OLD state)
- `p = exp(s - m_new)` ✓ — new term uses the NEW max, not old `m`
- `l = l*corr + p` ✓ — l rescaled
- `acc[d] = acc[d]*corr + p*V[j]` ✓ — acc rescaled by the SAME corr (the classic "rescaled l but not acc" bug is ABSENT)
- `m = m_new`; final `o = acc * (1/l)` ✓ — divide is acc/l
F32 throughout (qreg/acc/m/l/dot all Float32; V read F32). scale applied before max; mask added before max (line 576-578) so masked entries cannot poison `m`.

## PARITY RE-RUN BY REVIEWER (reRanParity: true, correctnessCos: 1.0)
- `sdpa_tiled_probe.mojo` (S=1536=3×_KV_BLOCK multi-block, Dh=128): NOMASK cos=1.0 ratio=1.0; MASKED(random [B,H,S,S]) cos=1.0 ratio=1.0; LARGE-S S=8192 Dh=128 completed, finite, shape [1,8192,2,128], no OOM. GATE1+GATE2 PASS, exit 0.
- Reviewer-authored `_skeptic_tiled_causal.mojo` (S=1536):
  - **CAUSAL** mask (strictly non-symmetric, 0 if j≤i else -1e30 — catches i/j transpose): cos=1.0, maxabs=3.0e-7, 0 NaN. Mask indexing `mask[qrow,j]` matches `_sdpa_math`'s `mask[r,c]` exactly (verified: both view `[B,H,S,S]` as `[B*H*S,S]`, qrow≡scores-row, j≡col). A transposed read would have failed this; it passed.
  - **MASKEDROW** (row 0 all -1e30): cos=1.0, maxabs=3.0e-7, **0 NaN in tiled AND math**; row0 outputs agree to ~5e-8. With finite-large-negative bias, exp(0)=1 for the max element ⇒ l≥1 ⇒ no 0/0. Reference does the identical thing.
  - **DH64** tiled (Dh=64, not 128-only): cos=1.0, maxabs=3.4e-7, 0 NaN.

## EXISTING PATHS UNTOUCHED (existingPathsUntouched: true)
- `git diff HEAD -- ops/attention.mojo`: **0 deletions**, +290 added; fn count 15→19 = exactly the 4 new entries (`_sdpa_online_f32`, `_sdpa_tiled`, `sdpa_tiled`, `sdpa_nomask_tiled`). No existing fn body changed. (The repo-wide "2 deletions" in `--stat` are in MAP.md/docs, not code.)
- Pre-existing `models/vae/sdpa_probe.mojo` re-run by reviewer: max_abs_diff 1.86e-8, "SDPA PROBE PASS", exit 0.

## EDGE CASES
- **Fully-masked row** (fullyMaskedRowGuarded: true *for finite masks*): finite large-negative (-1e30, which is what every real model builds — acestep_dit.mojo etc. use "large negative", NOT IEEE inf) → no NaN, matches math. **FRAGILE-shared (not BLOCKER):** a *literal IEEE -inf* fully-masked row → NaN — but reviewer confirmed the trusted `_sdpa_math` reference NaNs IDENTICALLY on the same input (exp(-inf−(−inf))=NaN in both). So it is a pre-existing property of the math path, not a tiled regression, and no current caller hits it. Non-masked rows in the same tensor stay correct.
- **Dh > 128**: `_sdpa_tiled` raises `comptime if Dh > _DH_MAX` (line 608) BEFORE any launch; both public entries route through it. `qreg`/`acc` are `_DH_MAX`(128)-sized and loops run `range(Dh)` only ⇒ no register overflow for Dh≤128; Dh>128 cannot reach the kernel. No silent corruption.
- **Dh < 128** (e.g. 64): handled — DH64 test cos=1.0.

## TAGGED ISSUES
- **FRAGILE-shared:** literal IEEE `-inf` fully-masked row → NaN in BOTH tiled and `_sdpa_math`. Not a regression; no caller uses literal -inf. If a future caller ever does, add `if l==0 or l!=l: out=0` to BOTH paths together. (where: `_sdpa_online_f32` line 588-591 + `_softmax_rows_f32` line 238.)
- **STYLE:** the Dh>128 guard is a runtime `raise` in a `comptime if` (correct, fires pre-launch) rather than `constrained[Dh<=_DH_MAX]()`. A `constrained` would reject at compile time. Cosmetic.
- **STYLE:** `_DH_MAX`-sized stack arrays are over-allocated for Dh=64 (64 unused slots). Harmless register pressure; intentional per comment.

## Verdict
No blockers. Recurrence is textbook-exact, both l and acc rescaled, parity cos=1.0 on multi-block nomask/random/causal masks and Dh∈{64,128}, large-S 8192 no-OOM, existing paths byte-identical and still passing, masked-row safe for all real (finite) masks.
