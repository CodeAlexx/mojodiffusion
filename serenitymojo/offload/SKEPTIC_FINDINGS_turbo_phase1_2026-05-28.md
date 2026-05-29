# SKEPTIC FINDINGS: TurboBlockLoader Phase 1 — 2026-05-28

**Verdict: PASS-WITH-CAVEATS**

Byte parity is genuine and deterministic across 3 smoke runs. The implementation is structurally correct. However the smoke has two significant coverage gaps: CHECK 3 is an assertion-by-reference (not a live ablation), and CHECK 2's overlap is collapsed by a rogue `ctx.synchronize()` that makes the fence test trivial. The implementation ships correct bytes; it does not ship a verified async overlap.

---

## Per-Concern Table

| # | Concern | Holds? | Evidence |
|---|---------|--------|----------|
| 1 | Byte parity vs independent sync read | YES | `_verify_block_bytes` reads turbo and `BlockLoader.load_block` results independently via D2H + `ctx.synchronize()`, byte-for-byte for 15 tensors each in layers.0 and layers.1. Genuine. |
| 2 | Fence is load-bearing (ablation run here) | INCONCLUSIVE | See headline below. Neither the smoke's CHECK 3 nor two /tmp ablations detected a race in 5 trials. Root cause: every ablation path contains a `ctx.synchronize()` that drains the copy stream before any readback — the race window is never open at read time. |
| 3 | Copy-kernel grid/block correctness | YES | For n=361820672: grid=1413362, threads=361820672 (exact fit). `if i < n` bounds check present. Per-tensor offsets packed contiguously; `create_sub_buffer(rec.offset, rec.nbytes)` matches fill offsets exactly. No off-by-one. |
| 4 | Host-write → kernel-read ordering | YES | CPU fills pinned slab before `copy_stream.enqueue_function()` returns. CUDA pinned memory is coherent; CPU writes before kernel dispatch are visible to the kernel. No barrier needed; the sequencing in `prefetch()` is correct. |
| 5 | Slot-reuse safety (A not corrupted while B staged) | WEAK PASS | `dummy_buf` is a fresh `float32` buffer unrelated to slot-A's device slab. Nothing writes slot-A in CHECK 2. Trivially true. Slot-A content is never read back after CHECK 2 to confirm it is intact. |
| 6 | Block shape/dtype parity | YES (structurally) | `tv.shape` and `tv.dtype` come from the same `ShardedSafeTensors.tensor_view()` as `BlockLoader`. Smoke only asserts `nbytes` equality, not shape equality, but the byte-parity check subsumes this for the offload seam. |
| 7 | Slab sizing / no leak | YES | `slab_capacity` = max block bytes; both slabs allocated once in `open()`. `prefetch()` does no device alloc. `await_block()` uses `create_sub_buffer` (non-owning views). Smoke calls `unload_block()` for all blocks. |

---

## Headline: Fence-Removal Ablation (Concern 2)

**Ablation result: NO RACE DETECTED — 5/5 trials pass with fence removed.**

Two /tmp ablations were built and run:
- `/tmp/turbo_nofence_ablation`: removes `enqueue_wait_for` calls; calls `ctx.synchronize()` before readback — passes trivially.
- `/tmp/turbo_race_ablation`: dispatches H2D copy on `copy_stream`, immediately enqueues slow dummy kernel + D2H on default stream, then calls `ctx.synchronize()` before byte compare. 5/5 trials: no mismatch.

The same flaw is present in the smoke's CHECK 2: `ctx.synchronize()` at line 191 (between `turbo.prefetch()` and the dummy kernel dispatch) fully drains the copy stream. By the time `await_block()` enqueues `enqueue_wait_for(ev1)`, the event has already fired — the fence is never tested under real concurrency.

**Why the fence still matters:** The Phase-0 probe (`turbo_probe_smoke.mojo` CHECK 4) demonstrated a genuine race (checksum 4193700-4193850 vs 4194304) using a structurally different setup: H2D via `enqueue_copy` on the DEFAULT stream + checksum on an EXPLICIT stream with no fence. The turbo loader inverts this (H2D on explicit copy stream, compute/readback on default stream). Whether this inversion also produces a detectable race at RTX 3090 Ti load patterns is unproven — the smoke does not test it.

**Practical risk:** The fence is correct per CUDA spec and load-bearing on any GPU where the two streams genuinely interleave. Removing it is unsafe. But the smoke cannot catch a regression that removes it.

---

## Punch List for Bugfixer

1. **Fix CHECK 2 overlap collapse (HIGH).** Move `ctx.synchronize()` (turbo_loader_smoke.mojo line 191) to after `ctx.enqueue_function[dummy]` dispatch — or eliminate it entirely if `enqueue_create_buffer` no longer requires a sync. The copy kernel must still be in-flight on the copy stream when the dummy starts on the default stream.

2. **Implement a live fence ablation in the smoke (HIGH).** Add a CHECK 3 variant that temporarily bypasses `enqueue_wait_for` (a flag or a second code path) and asserts that bytes are WRONG — proving the fence is the causal guard. Without this, a future refactor that removes the fence line passes all checks silently.

3. **Strengthen CHECK 5 slot-A integrity (MEDIUM).** After `await_block("layers.2")`, read back slot-0 device slab bytes and compare against the reference for layers.0. Confirm slot-A is intact after the copy stream wrote into slot-B concurrently.

4. **Add explicit shape equality assertion in CHECK 1 (LOW).** `_verify_block_bytes` checks `nbytes` but not individual dimension values. Add `rec.shape[] == sync_tensor.shape()` assertion to guard against transposed-but-same-size regressions.

5. **Document `ctx.compile_function` call per prefetch (LOW / future perf).** `prefetch()` calls `ctx.compile_function[_h2d_copy_kernel, _h2d_copy_kernel]()` on every call. In MAX 26.3 this should be JIT-cached, but should be pre-compiled once in `open()` and stored as a struct field, matching the probe's pattern and avoiding any potential re-JIT overhead in the hot path.
