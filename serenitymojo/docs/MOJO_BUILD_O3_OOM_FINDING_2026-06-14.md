# Mojo build OOM root cause — it's `-O3`, not the monolith (MEASURED 2026-06-14)

## TL;DR
The ~48–60 GB compile blow-up that has repeatedly OOM-killed the GNOME desktop —
and that the repo lore blamed on "the monolith" / "too many threads (-j0)" / "the
whole graph executor" — is actually driven by **`-O3`, Mojo's DEFAULT `mojo build`
optimization level** (aggressive whole-program inlining / interprocedural opt over
the ~322k-LOC import tree). It is NOT the file size and NOT the thread count.

**FIX: pass `--optimization-level 2`.** Same binary, ~20× less compiler RAM, faster
build, full real optimization.

## The measurement
Single-threaded, identical flags, only `--optimization-level` varied.

### zimage worker (`serenity_worker_zimage.mojo`)
| Opt level | Peak RAM | Build time | Binary | Result |
|---|---|---|---|---|
| `-O3` (default) | **48 GB** | OOM-loops forever | — | kills GNOME session via systemd-oomd |
| **`-O2`** | **2.0 GB** | **85 s** | 4.7 MB | builds ✓ |
| `-O1` | 2.0 GB | 91 s | 5.0 MB | builds ✓ |
| `-O0` | 5.0 GB | ~150 s | 36 MB (no DCE, bloated) | builds ✓ |

### serenity_daemon (the 3709-line "monolith")
| Opt level | Peak RAM | Build time | Binary |
|---|---|---|---|
| `-O3` (default) | ~60 GB (prior incident logs, 2026-06-13) | OOM | — |
| **`-O2`** | **3.8 GB (MEASURED)** | 225 s | 9.8 MB |

The "monolith" that was blamed for the 60 GB desktop kill builds in **3.8 GB at -O2**.

## Why `.mojopkg` / file-splitting does NOT help
Per Modular docs, a `.mojopkg` stores "non-elaborated parametric bytecode"; the code
"becomes an arch-specific executable only after it's imported into a Mojo program
that's then compiled with `mojo build`." So packaging defers ALL the heavy
elaboration + optimization to the same final build. A forum user who split a large
FFI surface reported "minimal success." The lever is the optimizer, not the layout.
- https://docs.modular.com/mojo/cli/package/
- https://forum.modular.com/t/very-slow-compile-times-with-ffi-functions/1728

## Why it killed the *terminal* specifically (two different OOM mechanisms)
The per-build cgroup cap in `scripts/mem_safe.sh` (`systemd-run --user --scope
MemoryMax=…`) protects the *machine*: a runaway build is cgroup-OOM-killed inside
its own scope. But it does NOT protect the *terminal*: a 48 GB build on a 62 GB box
drives the whole `user@1000.service` slice past **systemd-oomd's PSI memory-pressure
threshold (50% for >20 s)**, and oomd then reaps the biggest leaf scopes — which
included the GNOME Terminal hosting Claude Code (measured 2026-06-14 17:18:21).
With `-O2` (2–4 GB) there is no pressure, so oomd never fires.

## What was changed (this finding)
`pixi.toml` heavy `mojo build` tasks now pass `--optimization-level 2`:
- `build-daemon`, `build-daemon-safe`
- `build-worker-zimage-safe`, `build-worker-zimage-raw`
- `build-worker-ideogram4-raw`
- `build-video-smoke`

The `mem_safe.sh` caps + `-j` throttles are now **backstop only**, not the fix.

## Open item (NOT yet measured — Tenet 4)
`-O2`-vs-`-O3` *runtime* parity has not been benchmarked. Partial evidence: the
`-O2` zimage worker produced a correct Z-Image PNG end-to-end this session, and the
GPU kernels run through MAX + cuDNN (FFI), so the host `-O` level governs only host
orchestration, not the GPU compute. A head-to-head speed check vs `-O3` is blocked
because `-O3` cannot build on this box. Treat `-O2` as the standard; confirm runtime
with a real timed generation if perf is ever in question.
