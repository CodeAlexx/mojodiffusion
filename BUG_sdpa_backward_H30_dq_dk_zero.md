> # ⛔ RESOLVED / RETRACTED 2026-05-30 — THIS BUG IS NOT REAL.
> The H=30 d_q/d_k zero was a DEGENERATE-TEST-DATA artifact, not a kernel defect.
> The V-fill ((i*3)%9) aliases the H*Dh=3840 stride at H=30 → V constant across the
> sequence → grad_scores is mathematically zero → d_q/d_k=0 is the CORRECT answer
> (torch agrees). The UNMODIFIED kernel passes cos>=0.999 at H=30/6/32 with
> NON-DEGENERATE inputs: ops/parity/sdpa_bwd_nondegen_parity.mojo. Do NOT re-chase.
> See memory project_mojo_sdpa_h30_blocker_false_2026-05-30 + docs/MOJO_KERNELS.md.
> (Original (incorrect) report preserved below for history.)

# BUG (CONFIRMED, must-fix before T5) — sdpa_backward zeros d_q/d_k at H=30

**Status:** OPEN, lead-verified (clean run, 4 seqs). Blocks Z-Image training.
**File:** `serenitymojo/ops/attention_backward.mojo` (`sdpa_backward`).
**Found:** wave-4 real-seq agent; **confirmed by lead** 2026-05-30.

## Symptom (MEASURED — my clean run, RC=1)
At Z-Image's REAL attention dims `B=1, H=30, Dh=128`, every seq S∈{256,384,1152,2304}:
```
d_q vs torch: cos ≈ -0.008..0.09   max_abs ~1e-12  → NUMERICALLY ZERO   FAIL
d_k vs torch: cos ≈ -0.135..0.09   max_abs ~1e-12  → NUMERICALLY ZERO   FAIL
d_v vs torch: cos = 0.99999999                                          PASS
```
Gate: `serenitymojo/ops/parity/sdpa_bwd_realseq_parity.mojo` (oracle `sdpa_bwd_realseq_oracle.py`).

## Why it was missed
The existing `sdpa_bwd_parity.mojo` only tested **H=32 and H=8** (32-aligned). Z-Image uses
**H=30** (config: dim=3840, n_heads=30, head_dim=128; zimage_dit.mojo:98, sdpa call :384).
The toy gate was a FALSE GREEN for the real head count. d_v passes because it uses a different
matmul arm (`attnᵀ@d_out`); the broken arms are the d_q/d_k path (`grad_scores@K` /
`grad_scoresᵀ@Q`).

## Isolation (agent, plausible — NOT yet lead-confirmed at this granularity)
Trigger is the HEAD COUNT, not seq length:
- `[1,8,32,128]` → ‖d_q‖≈0.0143 correct (32-aligned)
- `[1,8,30,128]` → ‖d_q‖≈4e-10 collapsed (H=30)
A numpy F64/F32 reimpl of the SAME decomposed math at H=30 gives correct d_q/d_k → the MATH is
right; the Mojo kernel's per-head d_q/d_k arm is defective for non-32-divisor H. HYPOTHESIS:
the per-head matmul loop / grid sizing / score-buffer row indexing assumes H divides 32.

## WHY THIS IS DANGEROUS
SILENT failure: loss still moves (via the correct d_v / v-projection path) while ALL q/k
attention-projection gradients are dead. A Z-Image block trained today would half-learn — no
crash, no NaN, just a broken model. This is the worst failure class.

## Impact on prior "DONE" claims (correction)
- "Both Tier-5 kernels cleared / sdpa-bwd done" was a FALSE GREEN for Z-Image (H=30). True only
  for 32-aligned H.
- `block_composed_parity` (VERDICT SOUND) used H=2 — its composition logic is correct, but it
  does NOT cover H=30, so it does not exonerate this bug. Both true simultaneously.
- The FFN/MLP training path (linear/norm/swiglu/mse, train_skeleton, optim convergence,
  mixed-precision, zimage FFN sub-path) is UNAFFECTED — no attention in those.

## FIX (Tenet 1 — belongs in the primitive, NOT a trainer)
Fix the d_q/d_k per-head arm in `attention_backward.mojo::sdpa_backward` so it is correct for
arbitrary H (not just 32-divisors). Then:
1. Re-gate `sdpa_bwd_realseq_parity.mojo` (H=30) → cos≥0.999 on d_q/d_k/d_v.
2. ADD H=30 (and another non-aligned H, e.g. H=6) to the main `sdpa_bwd_parity.mojo` so the
   regression can't recur.
3. Re-run `block_composed_parity` at H=30 (not H=2) to confirm the block composes at real dims.
NOTE: the lead's grep for `BH`/`grad_scores`/`for bh` in the on-disk file did NOT match the
expected symbol names — the on-disk kernel may differ from the session-start version. READ the
actual current `sdpa_backward` body fully before editing (the file also has NUL-byte display
corruption under cat/grep — use python `open(...,errors='replace')` to read it).

## DO NOT
- Do NOT fix in a trainer/model — it's a primitive bug.
- Do NOT trust the toy `sdpa_bwd_parity.mojo` green — it tests the wrong H.
- Do NOT attempt a blind fix without reading the current kernel body (lead deferred the fix for
  exactly this reason — a wrong edit to attention backward is silent-corruption risk).
