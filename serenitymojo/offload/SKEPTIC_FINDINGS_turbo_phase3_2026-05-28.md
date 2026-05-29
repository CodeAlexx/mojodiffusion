# Skeptic Findings — Turbo Phase 3 Parity Audit

**Date:** 2026-05-28
**Verdict: PASS-WITH-CAVEATS**
**Auditor:** ERI Debugger (adversarial skeptic)

---

## MUTATION TEST — HEADLINE RESULT

**Corrupted copy kernel (`dst[i] = src[i] ^ 0xFF`) → parity FAILS.**

```
[check 2] cosine similarity: 0.089748696  (threshold >= 0.999)
Unhandled exception caught during execution: PARITY FAIL: cosine similarity 0.089748696 < 0.999
EXIT_CODE=1
```

The parity test is *not* a no-op. Corrupting the turbo copy kernel
produces cos ≈ 0.09 (garbage), the `_check` guard raises, and the
process exits nonzero. This confirms the test genuinely exercises
the turbo bytes, not a trivial or sync-fallback path.

---

## Per-Concern Table

| # | Concern | Holds? | Evidence |
|---|---------|--------|----------|
| 1 | Turbo path uses TurboBlockLoader async copy-kernel (not sync fallback) | **YES** | `TurboPlannedLoader.await_block` (turbo_planned_loader.mojo:254) calls `self._turbo.await_block(load_prefix, ctx)` which is `TurboBlockLoader.await_block`. That method (turbo_loader.mojo:349-352) calls `ctx.stream().enqueue_wait_for(ev0/ev1)` — the event was recorded by `copy_stream.record_event` after the copy kernel dispatched on `copy_stream`. No call to `BlockLoader.load_block` or `PlannedBlockLoader` anywhere in the turbo path. Confirmed by mutation test (corrupting `_h2d_copy_kernel` breaks parity → kernel is load-bearing). |
| 2 | `Klein9BOffloadedTurbo` uses `TurboPlannedLoader` (not sync loader) | **YES** | klein_dit.mojo:847 declares `var loader: TurboPlannedLoader`. `forward_full` (line 942-958) calls `self.loader.prefetch/prefetch_next/await_block` which route through `TurboPlannedLoader`. Block math (`_run_double`, `_run_single`) is byte-for-byte identical to `Klein9BOffloaded`'s block math (same `Klein9BDiT._double_block/_single_block` methods called with same arguments). |
| 3 | Parity test is genuine and fail-closed | **YES, with one caveat** | Inputs: `_linspace(-0.1, 0.1)` img (1×4×128 BF16), `_linspace(-0.05, 0.05)` txt (1×8×12288 BF16) — non-trivial, non-zero. Output: 512-element F32 vector. `_check` raises `Error` on failure (line 56-57). Mutation test confirmed nonzero exit on failure. **Caveat:** `txt_vals` shape uses `txt_shape.copy()` for run 1 (line 171) then `txt_shape.copy()` again for run 2 (line 201) — both copies are valid. However, `txt_shape` was already moved into `_linspace` after the first copy; the second `txt_shape.copy()` may share state depending on Mojo List copy semantics. Verified no issue in practice (output elements = 512, not 0), but this is a fragility. |
| 4 | Mutation test: corrupted turbo breaks parity | **YES — CONFIRMED** | `dst[i] = src[i] ^ 0xFF` in `_h2d_copy_kernel` → cos=0.089, raise, EXIT_CODE=1. Test is fail-closed against real byte corruption. |
| 5 | Real weights loaded (not zero/synthetic) | **YES** | Checkpoint exists: 18.2 GB at `/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors`. Output shape 1×4×128 with MAD=0 across 512 BF16 elements processed through 32 blocks with real weights. |

---

## Code Path Trace (Concern 1)

```
Klein9BOffloadedTurbo.forward_full[4,8,12]
  → self.loader.prefetch_next(bi)                 [turbo_planned_loader.mojo:203-205]
  → self.loader.await_block(bi, ctx)              [turbo_planned_loader.mojo:219]
      → self._dispatch_pending(ctx)               [line 251]
          → self._turbo.prefetch(prefix, ctx)     [line 217]
              → copy_stream.enqueue_function(     [turbo_loader.mojo:281-288]
                    _h2d_copy_kernel, host_ptr, dev_ptr, n_bytes)
              → copy_stream.record_event(ev0/1)   [line 290/305]
      → self._turbo.await_block(load_prefix, ctx) [turbo_planned_loader.mojo:254]
          → ctx.stream().enqueue_wait_for(ev0/1)  [turbo_loader.mojo:350-352]
          → returns Block of sub-buffer Tensor views
      → return PlannedBlockHandle(index, prefix, block^)
```

No `BlockLoader.load_block` or synchronous `PlannedBlockLoader` is on
this path. The async copy-kernel is the sole H2D mechanism.

---

## Bugfixer Punch List

**P1 — Input shape fragility (smoke, non-fatal now):**
`txt_shape.copy()` is called for turbo run at line 201 of
`klein_turbo_parity_smoke.mojo`. After `_linspace` has moved
`txt_shape` in the first call (line 171), the second call copies from
whatever state remains. Works now (presumably `_linspace` takes `var`
not `owned`, so original is intact) but is semantically fragile.
Recommendation: explicitly reconstruct `txt_shape2` rather than
calling `.copy()` on a potentially-moved variable.

**P2 — `TurboPlannedLoader` default config in load() (minor):**
`Klein9BOffloadedTurbo.load()` (klein_dit.mojo:853) passes
`OffloadConfig.synchronous_cfg_paired()` to `TurboPlannedLoader.open()`.
For the single-pass `forward_full` path this config causes extra
residency bookkeeping state transitions (for CFG-paired branches
that never run). Harmless correctness-wise; a dedicated
`OffloadConfig.single_pass()` would be cleaner.

**P3 — No wall-clock overlap verification:**
The test reports `async_enabled: True` (hardcoded `return True` in
`TurboBlockLoader.async_enabled()`). There is no timing gate — overlap
is structural only. This is explicitly documented and acceptable for
Phase 3, but a future phase should add a performance regression gate.

---

## Summary

The parity claim is **real**. The turbo path genuinely exercises
`TurboBlockLoader`'s async copy-kernel; corrupting that kernel breaks
parity with cos≈0.09 and a nonzero exit. The test is fail-closed.
Real 18 GB checkpoint weights are loaded. Block math is identical
between sync and turbo paths. Two minor code hygiene issues (P1, P2)
and one future-work item (P3) noted; none are blocking.
