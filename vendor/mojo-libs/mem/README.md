# mem — fast memory allocators for Mojo (100% Mojo)

Custom allocators that sit on top of `alloc`/`UnsafePointer` and hand memory out
in patterns the system allocator can't: bump-once-free-all arenas, O(1)
fixed-size pools, a size-class slab allocator, a growable chunked arena, a byte
ring buffer, aligned buffers, and allocation accounting. Built for **heavy
allocation churn and speed** — the hot paths are `@always_inline` freelist
pops/pointer bumps with no per-object allocator metadata.

> The package directory is **`mem/`** (not `memory/`, which collides with
> `std.memory`). Import as `from mem.arena import Arena`, etc.

## Modules

| Module | What it is |
|---|---|
| `aligned.mojo` | `align_up` / `align_up_pow2` / `align_down_pow2` / `is_power_of_two` (all `@always_inline`), and **`AlignedBuffer`** — a heap buffer whose data pointer is aligned to an arbitrary power-of-two (32/64/…) for SIMD / cache-line work, freed once on drop. |
| `arena.mojo` | **`Arena`** — a fixed-capacity bump/region allocator. `alloc_bytes(n, align)` / `alloc_array[T](count)` bump an offset; `reset()` reclaims everything in O(1). Tracks `used`/`peak_used`/`remaining`/`alloc_count`. |
| `arena_growable.mojo` | **`GrowableArena`** — a chunked bump allocator that **grows without a fixed cap**. Over-large requests get their own chunk; `reset()` **keeps** chunks so steady-state has zero new allocations (`shrink_to_first()` gives memory back). |
| `pool.mojo` | **`Pool`** — fixed-size object pool over one cache-line-aligned block with an intrusive freelist. O(1) `acquire`/`release`; `release` validates the pointer (range + alignment) and raises on a foreign pointer. `reset()` frees everything. |
| `slab.mojo` | **`SlabAllocator`** — a general size-class allocator (15 classes 16 B…4 KiB), per-class intrusive freelists, blocks carved from 1 MiB chunks. 8-byte header records the class so `free(ptr)` is O(1). Allocations over the largest class fall back to a direct malloc (documented, not pooled). |
| `ring.mojo` | **`ByteRing`** — a fixed-capacity circular byte FIFO (head/tail/count). `push_byte`/`pop_byte` (+ `try_*`), and bulk `push`/`pop` that copy in ≤2 contiguous runs across the wrap. Single-threaded (not atomic). |
| `stats.mojo` | **`MemStats`** (live/peak/total + counts, leak detection) and **`TrackingAllocator`** (a raw-alloc wrapper that records every alloc/free via an 8-byte size header). A `comptime STATS_ENABLED` flag compiles the recording away when off. |
| `bench.mojo` | `Timer` (monotonic ns) + `report` / `report_speedup` for the microbenchmarks. |

## Example

```mojo
from mem.arena import Arena

var a = Arena.with_capacity(1 << 20)        # 1 MiB region
for _ in range(frames):
    var scratch = a.alloc_array[Float32](4096)   # bump, no syscall
    var bytes   = a.alloc_bytes(123, 64)         # 64-byte aligned
    ... use scratch / bytes ...
    a.reset()                                    # reclaim all, O(1)
```

## Verified + measured (this machine, RTX-class host CPU)

Every module has a correctness test; the allocators also carry microbenchmarks
that accumulate a data-dependent `sink` so the optimizer can't delete the work.
**170 assertions across 6 test files, all passing.**

| Test | Assertions | Benchmark (measured) |
|---|---|---|
| `arena_test` (arena + aligned) | 29/29 | — |
| `pool_test` | 19/19 | acquire+release **6.08 ns/op** vs raw alloc/free 17.82 ns/op → **2.93×** |
| `slab_test` | 28/28 | mixed-size alloc/free **7.58 ns/op** vs raw 18.99 ns/op → **2.51×** |
| `arena_growable_test` | 19/19 | bump+reset **3.36 ns/op** vs raw 17.73 ns/op → **~5.3×** |
| `ring_test` | 43/43 | push+pop byte **1.17 ns/op (854 Mops/s)**; bulk 256 B **~37 GB/s** |
| `stats_test` | 32/32 | — (comptime-off path compiles away) |

```bash
# from the repo root (-I .)
pixi run mojo run -I . mem/tests/arena_test.mojo
pixi run mojo run -I . mem/tests/pool_test.mojo
pixi run mojo run -I . mem/tests/slab_test.mojo
pixi run mojo run -I . mem/tests/arena_growable_test.mojo
pixi run mojo run -I . mem/tests/ring_test.mojo
pixi run mojo run -I . mem/tests/stats_test.mojo
```

## Honest notes

- **Speedups are real but not magic.** glibc malloc has a fast tcache path for
  the exact churn the pool/slab target, so the win is ~2.5–3× (and ~5× for the
  arena's bump+reset), not orders of magnitude — measured above, not asserted.
- **Single-threaded.** None of these are atomic/thread-safe; cross-thread use
  needs external synchronization (or atomic head/tail for an SPSC ring).
- **`SlabAllocator`** large path (> 4 KiB) is a direct malloc/free fallback, not
  pooled; the large-free uses a linear scan of outstanding large allocations.
- **`GrowableArena.reset()`** retains chunks by design (the steady-state win);
  call `shrink_to_first()` to return memory to the OS.
- Allocations are raw `UnsafePointer` regions — the caller owns lifetime/typing
  within an arena/pool; only the allocator's backing blocks are freed on drop.
