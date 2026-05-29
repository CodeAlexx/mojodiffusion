# Skeptic Findings — turbo_probe_smoke Phase-0 Audit
**Date:** 2026-05-28  
**Verdict:** FEASIBLE-WITH-CAVEATS

---

## Per-Check Table

| Check | Claim | Holds? | Evidence |
|-------|-------|--------|----------|
| 1 | `enqueue_create_host_buffer` works | YES — solid | Alloc + write + read back succeeds; no false-positive path. |
| 2 | `create_sub_buffer` compiles and has correct `len()` | YES — solid | `len(sub)==256` verified; the primitive exists and returns a view. |
| 3 | Copy-stream H2D *overlaps* compute-stream burn kernel | NO — unproven | "Both completed" is the trivially true claim. There is no timing signal. The check merely proves two async enqueues don't crash; it says nothing about simultaneity. |
| 4 | `DeviceEvent` fences cross-stream ordering | NO — not fail-closed | See fence-removal experiment below. |

---

## Fence-Removal Experiment (Item A — the critical test)

Three variants were built and each run 5 times:

| Variant | `enqueue_wait_for` | `copy_ctx.synchronize()` | All runs correct? |
|---------|--------------------|--------------------------|-------------------|
| Original | present | present | Yes (5/5) |
| No fence | **removed** | present | **Yes (5/5)** |
| No order | **removed** | **removed** | **Yes (5/5)** |

**Conclusion:** CHECK 4 passes whether or not the event fence exists. The race is benign because the payload is 256 floats (1 KB). A 1 KB H2D DMA on a modern PCIE link finishes in single-digit microseconds; by the time the CPU has enqueued the checksum kernel and the GPU scheduler has dispatched it, the copy is already complete. The event/fence machinery is exercised syntactically but never stressed. This is a coincidental win, not proof of ordering.

---

## Per-Concern Verdicts

**B. Does CHECK 3 prove overlap?** No. "Both completed" after two async enqueues is always true. The only overlap signal would be wall-clock timing (total < max(burn, copy)) — absent here because no timer is available. CHECK 3 should be honestly retitled "two contexts can coexist without crash."

**C. Two contexts = two real streams?** Likely yes at the stream level (two `DeviceContext()` calls), but it is not confirmed whether they map to the same CUDA context (just different streams) or to two separate CUDA contexts. The probe works either way due to unified virtual addressing, but the probe does not confirm which case holds. No hidden `synchronize()` call serializes the burn+copy issue — those are genuinely concurrent at the CPU enqueue level.

**D. Burn loop optimized away?** No. The final `buf[i] = v` write to a device buffer is a store that a compiler cannot eliminate under memory-model rules. BURN_ITERS=2048 with FMA is adequate to make the kernel non-trivial in wall time.

**E. Float32 checksum exactness?** Confirmed exact. Sequential accumulation of integers 1..256 stays within Float32 precision (max value 32896 < 2^15; no rounding). The 0.5 tolerance is sound.

---

## Punch List for Bugfixer

1. **Make CHECK 4 fail-closed.** The probe must produce a wrong/garbage checksum when the fence is absent. Use a large copy (e.g. 16 MB, not 1 KB) so the GPU cannot "accidentally" finish the copy before the checksum kernel is dispatched. Alternatively, poison the buffer first (fill with zeros on compute_ctx), then enqueue the H2D copy on copy_ctx with the fence — without the fence, the checksum should read zeros; with it, 32896. Only then is ordering actually proven.

2. **Strengthen CHECK 3 or retitle it.** Either add a timing assertion (`total_wall < 0.9 * burn_alone_wall`) or rename the check to "two DeviceContexts coexist without crash (overlap not measured)." Claiming overlap without a timer is misleading.

3. **Clarify DeviceContext identity.** Add a comment (or a runtime log of stream/context IDs if the API exposes them) confirming whether two `DeviceContext()` instances produce two independent CUDA streams within one CUDA context, or two separate CUDA contexts. This matters for the real offload design (peer access, memory ownership semantics).

4. **Cross-context buffer ownership.** `sub` is allocated via `compute_ctx`; `copy_ctx.enqueue_copy` writes to it. This works now but should be documented as a supported pattern (or guarded with a note about unified addressing dependency).

