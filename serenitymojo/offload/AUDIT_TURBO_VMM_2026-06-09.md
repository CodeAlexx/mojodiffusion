# Audit — Turbo / VMM offload path (inference focus)

**Date:** 2026-06-09
**Method:** read-only source audit. The GPU was in use, so **nothing was
executed**. Every claim about *static structure* (what the code does call /
allocate) is cited to `file:line` and is confirmed by reading. Every claim about
*runtime behavior* (overlap, timing, VRAM) is labeled **HYPOTHESIS** — it follows
from the code by inspection but has **not** been measured this session and must be
confirmed on-GPU before being treated as fact (Tenet 4).

---

## 1. Inventory — what exists in `serenitymojo/offload/`

| Component | File | Status (structural) |
|---|---|---|
| CUDA VMM driver FFI (`cuMemAddressReserve/Create/Map/SetAccess`, events) | `vmm_cuda.mojo` | Complete FFI surface. |
| VMM slab allocator (reserve VA, define regions, `ensure_resident`/refcount/`evict`/`destroy`) | `vmm_slab.mojo` | Implemented. **Wired into nothing** — no `VmmSlabAllocator` reference outside `vmm_slab*` (grep). |
| Async double-buffer weight loader | `turbo_loader.mojo` (`TurboBlockLoader`) | Implemented: 2 pinned-host + 2 device slabs, `cuMemcpyHtoDAsync_v2` on an explicit copy stream, H2D-done + compute-done events. |
| Plan-aware async wrapper | `turbo_planned_loader.mojo` (`TurboPlannedLoader`) | Implemented; exposes both the legacy `prefetch`/`prefetch_next` (no ctx) and the fixed `prefetch_with_ctx`/`prefetch_next_with_ctx` + `mark_active_block_done`. |
| Synchronous backend | `block_loader.mojo` | Baseline. |
| Residency / budget / eviction | `residency.mojo`, `plan.mojo` | Wired but **idle** on the turbo path: budget set to 128 GB hi / 64 GB lo so `can_prefetch()` is always true and no eviction runs (`turbo_planned_loader.mojo:127-130`). |

**Adoption split (grep, this session):**
- The fixed overlap contract (`prefetch_with_ctx` + `mark_active_block_done`) is
  adopted by the **training** stacks: `chroma/qwenimage/sd35/flux/klein/wan22/ltx2
  _stack_lora.mojo`.
- The Klein **inference** path (`Klein9BOffloadedTurbo`, `klein_dit.mojo`, driven by
  `klein9b_pipeline_multistep_turbo.mojo:149,172`) does **not** call either — it uses
  the legacy pattern only.

---

## 2. FINDING 1 (headline) — the turbo *inference* loop does not overlap

**Structural fact.** Every Klein turbo forward loop uses this shape
(`klein_dit.mojo:980-983`, `:1071-1074`, and the `Klein9BDiT` variants `:713-716`,
`:806-809`):

```mojo
self.loader.prefetch(0)                 # TurboPlannedLoader.prefetch → _pending_idx=0, NO GPU dispatch
for bi in range(...):
    self.loader.prefetch_next(bi)       # _pending_idx = bi+1  (overwrites)
    var handle = self.loader.await_block(bi, ctx)
    ... run block bi ...
```

`prefetch`/`prefetch_next` here only set `_pending_idx` (no `ctx`, no copy
dispatch) — `turbo_planned_loader.mojo:175-193`. The GPU copy is dispatched inside
`await_block` via `_dispatch_pending` → `prefetch_with_ctx(_pending_idx)`
(`turbo_planned_loader.mojo:231-237, 271`), i.e. it dispatches the **pending**
index, which is always `bi+1`, **then** calls `_turbo.await_block(bi)`.

**HYPOTHESIS (static trace, not executed) — every block is staged synchronously,
zero copy/compute overlap, plus wasted H2D.** Trace of the two-slot rotation
(`TurboBlockLoader`: `_idle_slot = 1 - active_slot`, `:361-363`; `active_slot`
starts 0; `await_block` sync-fallback at `:554-565`):

```
bi=0: pending=1 → _turbo.prefetch(1) stages block1 into idle slot1.
      _turbo.await_block(0): prefix0="",prefix1=block1 → block0 NOT staged
        → fallback prefetch(0): idle slot1 → OVERWRITES block1 with block0.
        await fences default stream on slot1 event (block0, just dispatched).
        active=1.   ⇒ block1's copy was wasted; block0 staged serially.
bi=1: pending=2 → _turbo.prefetch(2): idle slot0 → stages block2.
      _turbo.await_block(1): prefix0=block2,prefix1=block0 → block1 NOT staged
        → fallback prefetch(1): idle slot0 → OVERWRITES block2 with block1.
        active=0.   ⇒ block2's copy wasted; block1 staged serially.
...repeats: the prefetched bi+1 is always clobbered by the sync fallback of bi.
```

Because the *pending* index is always `bi+1` but `await_block` needs `bi`, the
needed block is never the staged one, so `await_block` takes the synchronous
fallback **every iteration**, and the fence (`enqueue_wait_for`) waits on a copy
that was dispatched microseconds earlier → effectively serial H2D-then-compute.
This matches the documented 2026-05-31 Klein finding (“turbo existed but didn’t
overlap”, ~46 s/step) in `docs/FULL_PORT_OFFLOAD_TURBO_2026-05-27.md:253-291`.

**Confidence:** high on the *structure* (the loop and dispatch order are as quoted);
the “effectively serial / wasted copy” conclusion is a HYPOTHESIS until timed on-GPU
(see §7). The earlier phase-3 skeptic explicitly left this open as **P3 — “no
wall-clock overlap verification … overlap is structural only”**
(`SKEPTIC_FINDINGS_turbo_phase3_2026-05-28.md:79-83`).

---

## 3. FINDING 2 — the compute-done fence is never armed on the inference path

`TurboBlockLoader` guards slot reuse with per-slot `compute_done` events: a prefetch
only waits on `compute_done{0,1}` if `compute_recorded{0,1}` is true
(`turbo_loader.mojo:448-453`), and those flags are set **only** by
`mark_active_slot_compute_done` (`:602-617`), surfaced as
`TurboPlannedLoader.mark_active_block_done` (`turbo_planned_loader.mojo:214-221`).

**Structural fact:** `klein_dit.mojo` calls `mark_active_block_done` **zero** times
(grep count 0). The training stacks do call it.

**HYPOTHESIS:** on the inference path the copy stream never waits for default-stream
compute to finish reading a slab before the next prefetch could overwrite it. Today
this is *probably masked* because Finding 1 makes the path effectively serial (the
`await` fence orders everything), so there is likely no live corruption — but it is a
**latent race** that would bite the moment real overlap is enabled (e.g. by adopting
the fix in Finding 1 without also adding the fence). Must be verified on-GPU.

---

## 4. FINDING 3 — the VMM slab allocator is parked (dead code)

`vmm_cuda.mojo` + `vmm_slab.mojo` implement a real reserved-VA slab with on-demand
physical mapping, refcount, and eviction. **Nothing references `VmmSlabAllocator`**
outside its own smoke (grep empty). The design doc’s backend progression ends at a
`TurboVmmBackend` (“reserved virtual address slab + smaller physical pool +
event-gated slot reuse”, `docs/FULL_PORT_OFFLOAD_TURBO_2026-05-27.md:196-217,
292-310`), but `turbo_loader` instead uses two **fixed, full-size** device slabs
(`:279-282`). So the VMM layer is built but never became a `TurboBlockLoader`
backend.

Note: for a strict *double-buffer* loader, VMM would not reduce device VRAM below
~`2 × max_block_bytes` anyway — the VMM payoff is a *residency cache* (many blocks
resident behind one VA with eviction), which is exactly what `residency.mojo` models
but is currently wired idle (Finding inventory). So “use VMM” is only worthwhile
together with enabling the residency/eviction budget.

---

## 5. FINDING 4 — VRAM footprint and minor notes

- **Weight-streaming VRAM** = `2 × max_block_bytes` on device (`dev0`+`dev1`) plus
  `2 × max_block_bytes` pinned host (`host0`+`host1`), all sized to the **largest**
  block (`turbo_loader.mojo:279-282`; `slab_bytes()` reports `capacity*4`,
  `:622-624`). This is the expected double-buffer cost, not a leak.
- The persistent full-model pinned store is **off** by default
  (`TURBO_USE_PERSISTENT_BLOCK_STORE = False`, `turbo_loader.mojo:44`; `block_store`
  allocated as 1 byte, `:276`) — good, keeps startup bounded.
- The copy path is the **CUDA DMA** branch (`TURBO_USE_CUDA_DMA_COPY = True`,
  `:43`); the SM byte-copy kernel is a fallback only. Good (no SM time spent copying
  weights) — assuming the DMA branch is the one actually taken at runtime (HYPOTHESIS,
  comptime flag says so).
- The dtype guard in `await_block` is a **no-op `pass`** (`turbo_planned_loader.mojo:
  260-263`) — the “loud assertion” described in the header is not actually emitted. Safe
  for Klein (BF16 on disk) but would **silently** raw-copy a non-BF16 checkpoint under
  `force_bf16`. Low priority, but the comment overstates the protection.
- `Klein9BOffloadedTurbo.load` opens the loader with
  `OffloadConfig.synchronous_cfg_paired()` (`klein_dit.mojo:891`); phase-3 P2 notes
  this drives extra residency bookkeeping for the single-pass path
  (`SKEPTIC_FINDINGS_turbo_phase3_2026-05-28.md:71-77`). Harmless.

---

## 6. What would actually help inference (ranked) — each needs on-GPU measurement

1. **Adopt the fixed overlap contract in the Klein inference loops** (highest value,
   lowest risk; the training stacks already prove the pattern compiles + runs):
   ```mojo
   self.loader.prefetch_with_ctx(0, ctx)        # dispatch block 0 up front
   for bi in range(count):
       var handle = self.loader.await_block(bi, ctx)
       self.loader.prefetch_next_with_ctx(bi, ctx)   # dispatch bi+1 NOW
       ... run block bi ...
       self.loader.mark_active_block_done(ctx)       # arm slot-reuse fence
   ```
   This removes the per-block sync fallback (Finding 1) and the wasted clobbered copy
   (Finding 4-intro), and arms the fence (Finding 2). Apply to `forward_full`,
   `forward_full_cfg` in both `Klein9BOffloadedTurbo` and `Klein9BDiT`. **Expected**
   to recover copy/compute overlap — HYPOTHESIS until timed.
2. **Add a wall-clock overlap + peak-VRAM gate** (settles phase-3 P3): time
   copy-stream vs default-stream, and read `cu_mem_get_info()` peak
   (`vmm_cuda.mojo:119-135`) across a run. Without this, “turbo” cannot be claimed to
   help at all.
3. **Only if (1)+(2) show streaming is the bottleneck:** wire `VmmSlabAllocator` +
   the residency/eviction budget as a real `TurboVmmBackend` to hold N>2 resident
   blocks with eviction (the doc’s step 4). This is the big lift and should not start
   before measurement justifies it.

A new from-scratch “memory lib” is **not** warranted — this substrate already exceeds
what a greenfield lib would provide; the gaps are *adoption and measurement*, not
missing primitives.

---

## 7. Verification plan (run when the GPU is free — do not claim results before)

1. Baseline timing of `klein9b_pipeline_multistep_turbo` as-is (per-step ms, peak VRAM
   via `cu_mem_get_info`). Confirms/【refutes Finding 1’s “serial” HYPOTHESIS.
2. Apply the §6.1 loop change; re-time. A drop in per-step time = overlap recovered.
3. Add the §6.2 perf/VRAM gate as a committed smoke so regressions are caught.
4. Stress the fence (Finding 2): with overlap enabled, run enough steps to force slot
   reuse and check output parity vs the synchronous `PlannedBlockLoader` path
   (the existing `klein_turbo_parity_smoke` is the oracle shape).

Until step 1 runs, treat Findings 1–2 as **HYPOTHESES** with strong static support,
not confirmed runtime behavior.
