# Skeptic Findings — Turbo Phase 2: Residency / Budget / Scheduling

**Date:** 2026-05-28  
**Verdict: PASS-WITH-CAVEATS**  
**Auditor:** ERI Skeptic Agent

---

## Fail-Closed Experiment (Headline)

**The smoke IS fail-closed.** Mutation test: patched `eviction_candidates` to never
exclude refcount>0 blocks (`if entry.refcount != 0:` → `if False:`), rebuilt, reran.
Result: 3 FAILs (A3 ×2, A6 ×1), nonzero exit, `raise` fired in `main()`.
The "57 PASS / exit 0" report from the builder is not theater — actual failures surface.

---

## Per-Concern Table

| # | Concern | Holds? | Evidence |
|---|---------|--------|----------|
| 1 | **State machine fidelity** | YES | `_transition_allowed` encodes all 6 transitions from stagehand `VALID_TRANSITIONS` exactly. All illegal paths tested in A1. |
| 2 | **Budget invariant + watermarks** | YES (minor caveat) | `high>low` enforced, `can_prefetch` = `not above_high`, `should_evict` = `above_high`. Matches stagehand `budget.py`. Caveat: bytes counted only at GPU_READY arrival, not at PREFETCHING arrival — stagehand uses a live hardware counter that captures the allocation earlier. Low risk for sequential single-slot use, real risk if multiple blocks are simultaneously PREFETCHING. |
| 3 | **Refcount discipline** | YES | `eviction_candidates` checks `refcount != 0` and skips; mutation test proved it raises on failure. `release()` raises on underflow. |
| 4 | **Eviction scoring formula** | PARTIAL — semantic divergence | Mojo cursor = block-just-finished (pre-increment). Stagehand `score_for_eviction` uses `self._cursor` which is post-increment (= i+1 after block i). Mojo distances are uniformly +1 vs stagehand. For equal-size blocks the sort order is preserved; for mixed sizes the order can invert. Concrete example: dist_A=1,size_A=50 vs dist_B=2,size_B=30 → stagehand evicts B (60>50), Mojo evicts A (100>90). The smoke only tests Mojo's own arithmetic — A4 expected values are tuned to Mojo, not cross-checked against stagehand. |
| 5 | **CFG next_use_distance** | INTERNAL ONLY | Mojo models cursor as a position in the expanded 2N-length list `[0,0,1,1,...]`. Stagehand has no branch_count in `score_for_eviction` — its cursor is always a block index 0..N-1 regardless of CFG. The two models are incommensurable; distances from Mojo CFG are ~2× those from stagehand. Smoke A5 validates Mojo-internal consistency only. |
| 6 | **Prefetch window exclusion** | YES | Window built as `[cursor+1, cursor+W]`. Stagehand uses `cursor_for_policy = self._cursor - 1` then `+1..+W`, which is identical when Mojo cursor = stagehand cursor-1. No off-by-one. Window exclusion enforced in both `eviction_candidates` and confirmed not in `prefetch_targets`. |

---

## Bugfixer Punch List

**P1 — EVICTION SCORE CURSOR OFF-BY-ONE (semantic divergence, not crash)**
- File: `residency.mojo`, `next_use_distance()`
- Stagehand `score_for_eviction` uses post-increment cursor (`self._cursor`, value = i+1 after block i), formula `(exec_order - cursor) % N`.
- Mojo searches from `cursor+1` which gives distances +1 vs stagehand for all blocks.
- Fix: use `(block_index - (cursor + 1)) % n` with `0 → n` fixup instead of the forward search, OR document that Mojo cursor convention is pre-increment and the divergence is intentional.
- Smoke A4 expected values must be updated if formula is corrected to match stagehand.

**P2 — CFG DISTANCE MODEL HAS NO STAGEHAND REFERENCE**
- Stagehand never uses a branch-doubled cursor in scoring. The Mojo CFG expansion produces distances 2× stagehand for cfg_paired mode.
- Fix: either align to stagehand semantics (drop branch_count from `next_use_distance`, treat cursor as block index), or explicitly document the deliberate divergence and add a cross-check test.

**P3 — BUDGET ACCOUNTING TIMING (minor)**
- Budget bytes added at PREFETCHING→GPU_READY, not at PREFETCHING entry. A block with a 1+ GB allocation in flight during PREFETCHING is invisible to `can_prefetch()`.
- Fix: call `_budget.add(entry.byte_count)` at HOST_STAGED→PREFETCHING and remove at EVICTING/GPU_FREEING entry instead of GPU_READY entry. Or add partial-credit accounting for PREFETCHING state.

**P4 — SMOKE TESTS ONLY VALIDATE MOJO-INTERNAL CONSISTENCY (not stagehand parity)**
- A4 expected eviction order is calculated from Mojo's cursor-pre-increment convention. It would fail if corrected to stagehand convention.
- Fix: add one cross-check test that computes expected values using the stagehand formula `(exec_order - (cursor+1)) % N` and asserts the same order.
