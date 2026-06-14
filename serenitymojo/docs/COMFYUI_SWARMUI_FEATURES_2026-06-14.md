# Adding missing ComfyUI/SwarmUI parts to the serenity stack

Initiative started 2026-06-14. Plan: `~/.claude/plans/snappy-cuddling-crane.md`.
Four zimage-verifiable phases, one at a time, each with a REAL end-to-end image gate
(Tenet 4: measurement, not assertion) before advancing. Team model per phase:
builder / bug-fixer / skeptic (Workflow), with the GPU measurement gate run in the
main loop.

Stack: Rust control plane `serenity-server` (:7801) → pure-Mojo `serenity_worker_zimage`
→ images. The only built executable real-image backend is **zimage** (512² / 1024²).

---

## Phase 1 — Hires-fix / Upscale (2-pass)  ✅ DONE & VERIFIED (2026-06-14)

**Prerequisite found by measurement:** the Rust server's `post_generate` dropped
`init_image`/`mask_image` (not in `GenerateRequest`, not copied to `JobParams`), so
img2img/inpaint never reached the worker. Wired them through (server-side only;
`JobParams` already had the fields). Measured: job-0826 (init ignored, clean t2i car)
vs job-0827 (init honored, lighthouse-derived) confirmed the fix.

**Design:** a job with `hires_scale > 1` runs as **two worker passes under one job_id**:
base pass → Lanczos upscale of the base PNG → img2img refine pass at the upscaled res.
- Hires params (`hires_scale`, `hires_denoise`) live on `jobs::JobEntry` (control plane),
  **never** in the frozen worker wire `JobParams`.
- Driver: `drive_one_pass(publish_done)` suppresses the base pass's terminal Done (record
  stays `running`); only the refine pass publishes Done. `drive_hires_two_pass` orchestrates.
- The refine target snaps to a worker-supported square grid (512/1024) so every hires job
  is on a valid encoder grid; `hires_scale` is clamped `[1,4]` + NaN-guarded (an unbounded
  scale would drive a multi-TB image-crate alloc → desktop OOM).
- Upscale temp is named `hires_src_<job>.png` (invisible to the `job-*.png` gallery scan)
  and deleted after the refine pass.

**Worker change:** the zimage img2img VAE encoder was hard-pinned to `ZImageVaeEncoder[64,64]`
(512-only). Added a `[128,128]` dispatch so the refine pass encodes a 1024 init. The decoder
at 1024 was already verified (txt2img 1024 works); the encoder at 1024 is now verified by the
hires output. Capped rebuild (`build-worker-zimage-raw` in a 48G systemd scope).

**UI:** `GenParams` gained `hires_scale`/`hires_denoise` (+ store mirrors + commit) and a
"Hires-fix" slider block in `_section_sampling` (built by the team's UI agent).

**Files:**
- `serenity-server/crates/server/src/main.rs` — `GenerateRequest` (init/mask/hires fields),
  `post_generate` (copy + clamp), `run_worker_driver` (promote hires, branch),
  `drive_one_pass` / `drive_hires_two_pass` / `upscale_png` (new).
- `serenity-server/crates/server/src/jobs.rs` — `JobEntry.hires_scale/hires_denoise`.
- `serenitymojo/serve/zimage_backend.mojo` — encoder size dispatch (init-image encode).
- `MojoUI/mojoui/app/genparams.mojo` — hires fields + store mirrors.
- `serenityUI/src/sections.mojo` — Hires-fix slider block.

**Skeptic findings (all fixed):** blocker — `jobs.rs` `entry()` test fixture missing new
fields (`cargo test` broke); major — temp file matched gallery scan + never cleaned up;
major — `hires_scale` unbounded (OOM). Plus the encoder-grid snap (found by my own measurement).

**Verification (measured):**
- curl: job-0829/0831 = detailed native 1024² owl / lighthouse, full 2-pass in server log,
  genparams tEXt preserved. Extreme `hires_scale:100` → clamp 4 → snap 1024, no crash/OOM.
- GUI button path (`--selftest-ui`): submitted `hires_scale:2.0` @512 → server ran 2-pass
  (`base_w=512 hires_w=1024`) → job-0835 = 1024 hires lighthouse.
- `cargo test -p serenity-server`: 14 passed. serenityUI `--selftest`: ALL PASS (genparams
  round-trips the hires fields through PNG tEXt). No regressions.

---

## Phase 2 — Grid / XYZ-plot generator     ✅ server DONE & VERIFIED (2026-06-14)

**`POST /v1/grid`** (new `crates/server/src/grid.rs`): sweep one axis
(`seed|cfg|steps|sampler|scheduler`) across N values (≤16), run one zimage job per
value, composite the cells into a single labeled grid PNG.
- Enqueue replicates `post_generate` exactly (new job_id via `next_id.fetch_add`,
  register a `JobChannel`, push a queued `JobEntry`, `DriverCtl::Wake`); cells ride the
  normal driver path and interleave safely with other jobs.
- Wait loop polls `st.jobs` records keyed on the SPECIFIC cell ids; the `st.jobs` mutex
  is never held across the `tokio::sleep` await. Budget = 180s × cells (capped 1h).
- Composite via the `image` crate (320px cells, dark bg) + a self-contained 5×7 bitmap
  font for the `axis=value` label band. A failed/missing cell → placeholder tile (never
  aborts the grid). Saved as `<out_dir>/grid-NNNN.png`.
- `main.rs`: `mod grid;` + `.route("/v1/grid", post(grid::post_grid))`; widened a few
  crate items to `pub(crate)`.

**Skeptic:** no blocker/major (independently verified: no lock-across-await, no channel
leak, keys on specific ids, inputs clamped to 16, graceful per-cell degradation,
bounds-checked font, atomic ids). Applied the timeout-scaling fix; nits (spawn_blocking,
redundant Vec) noted.

**Verified (measured):** `cfg=[1,3,5,7]` → `grid-0841.png` = a 2×2 grid of lighthouses
labeled `cfg=1.0/3.0/5.0/7.0`, cells visibly distinct across the sweep; `seed=[10,20]`
grid also produced. `cargo build` clean.

GUI grid section (axis dropdown + values field + Generate-Grid button) = the remaining
exposure step (the feature is API-complete and measured).
## Phase 3 — Inpaint UI panel               ⏳ (backend already executes mask denoise)
## Phase 4 — ComfyUI node-dispatch completeness  ⏳ (executor + capped worker rebuild)
