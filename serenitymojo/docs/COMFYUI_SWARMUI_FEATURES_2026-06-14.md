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
## Phase 3 — Inpaint UI panel               ✅ DONE & VERIFIED (2026-06-14)

The zimage worker already executes mask-based denoise (`SetLatentNoiseMask`,
`load_comfy_latent_preserve_mask` → `inpaint_preserve_mask`); the server forwards
`mask_image`/`lanpaint_mask_channel` (wired in the Phase 1 prereq). Phase 3 = the UI.

- `GenParams` gained `mask_image` + `mask_channel` (serialized under keys `mask_image`
  and `lanpaint_mask_channel`), mirroring the `init_image` pattern (store mirror + commit).
- `_section_inpaint` (serenityUI): mask-path field + Validate + a channel combobox
  (`load_image_mask|red|green|blue|alpha|luminance`) + the "needs Init image; white =
  regenerate" hint, modeled on `_section_init_image`. Reuses the shared creativity slider.

**Verified (measured):**
- curl: a forest init + a white-center mask + prompt "a bright red barn" → the **center
  regenerated to a red barn while the forest edges were preserved** (mask convention:
  white = regenerate). Server log: `preserve pixels 3072/4096`.
- UI path: `--selftest` inpaint assertion — genparams carry `mask_image` +
  `lanpaint_mask_channel`, a masked-denoise job (init + red-channel mask + creativity 0.85)
  completes. ALL PASS.

## Phase 4 — ComfyUI node-dispatch completeness  ✅ DONE & VERIFIED (2026-06-14)

KEY FINDING: the live lowering is **Rust** (`serenity-server/crates/graph`), not the Mojo
`workflow_graph.mojo` (that is the parity ORACLE). Nodes that lower to existing flat params
need NO worker rebuild — the worker runs flat params, only the server's graph→params lowering
changes. (So Phase 4 was Rust-only, contrary to the original plan's "capped worker rebuild".)

Added to the Rust graph crate (`nodes.rs` allowlist + `import.rs` ports/widgets + `execute.rs`
dispatch + `execute_handlers.rs`):
- **KSamplerAdvanced** — maps to the same flat params as KSampler, but seeds from `noise_seed`
  and derives `creativity = clamp(1 - start_at_step/steps)` (start_at_step=0 → full txt2img).
- **ConditioningConcat** — joins the two input prompt texts (`"to, from"`).
- **ConditioningCombine / ConditioningAverage** — FAIL-LOUD 501: batching / weighted-blend
  can't be lowered to a single text prompt without silently dropping the second conditioning
  (chose a clear 501 over a silently-wrong image; use Concat to join).
- KSamplerAdvanced FAIL-LOUD 501 on `add_noise=disable` or early `end_at_step`
  (return-with-leftover-noise) — not representable in the zimage flat denoise model.

**Team:** builder/bug-fixer/skeptic; build green, 18/18 graph tests pass. The bug-fixer
correctly flagged that the Rust lowering is now AHEAD of the Mojo oracle (which 501s on these
nodes) and refused to fabricate a fake parity ref — so the Mojo `serenity_lower` oracle should
gain these nodes if strict Rust↔Mojo lockstep is required (follow-up). Skeptic's fail-loud
recommendations were applied.

**Verified (measured, live server):**
- KSamplerAdvanced workflow → lowered (`seed` from noise_seed, `creativity` 1.0 from
  start_at_step 0) → ran on zimage → a real serene-mountain-lake image (job-0859/0860).
  (Caught a test bug first: a top-level `prompt` override beats the workflow's CLIPTextEncode
  via `set_if_missing` — omit it so the graph's prompt wins.)
- ConditioningConcat workflow → joined prompt "a red vintage car, on a snowy mountain road"
  → real image (job-0861).
- Fail-loud confirmed: add_noise=disable / end_at_step<steps / ConditioningCombine all return
  clear `[501]` messages.

---

ALL FOUR PHASES DONE & VERIFIED. Verification images:
github.com/CodeAlexx/samples/tree/main/comfyui_swarmui_2026-06-14

---

## Follow-on: Ideogram4 KJ workflow (the "kj nodes") — worker DONE & VERIFIED (2026-06-14)

User goal: make the ideogram4 Comfy export (with `Ideogram4PromptBuilderKJ`) run end-to-end.
Findings + work:
- The KJ export ALREADY lowers via the live Rust path (`apply_ideogram4_comfy_ui_export`,
  import.rs:543) — KJ builder + ResolutionSelector + the RES4LYF UUID nodes (ClownsharKSampler,
  Dual Model CFG Guider) are inert UI nodes; the prompt comes from the top level. No 501.
- The only missing piece was the ideogram4 SERVING worker. Added a standalone
  `serenitymojo/serve/serenity_worker_ideogram4.mojo` (mirrors serenity_worker_zimage, swaps in
  `Ideogram4Backend`) + pixi `build-worker-ideogram4-raw` (capped, outer systemd scope). Launch:
  `serenity-server --worker output/bin/serenity_worker_ideogram4` (NO --kind — standalone takes
  just <fd>). Single-worker server, so it's a separate instance from the zimage one.
- VERIFIED PARAMS (measured): the backend defaults are **cfg 7.0** + scheduler
  **ideogram4_logitnormal** + **48 steps** (ideogram4_backend.mojo:330,340). cfg 1.0/simple →
  garbage; correct params → coherent image. The model tokenizes the RAW JSON caption string
  (ideogram4_backend.mojo:480) — so prompt serialization must be byte-exact.
- VERIFIED (measured): structured JSON caption → real sailboat image (job-0865). Byte-exact
  passthrough CONFIRMED (sent==worker-received, 837 bytes). A benign golden-retriever prompt hit
  the model's safety-filter false-positive (documented behavior, gray "Image blocked" screen).

### JSON-fidelity finding (web-researched + measured)
The image varies "through the prompt node" because ideogram4 tokenizes the raw string. The
trailing-comma/2-space JSON is NOT the KJ Prompt Builder (its source uses json.dumps or 4-space
pretty — never trailing commas) — it's the **LLM/QwenVL expansion** emitting sloppy JSON. The
serenity transport is byte-exact (verified). FIX for the magic-prompt to build: normalize the
LLM output (parse tolerantly → re-serialize canonically `separators=(",",":")`, ensure_ascii=False).

### Remaining (large): magic-prompt + bbox builder
User wants both, with **Qwen3-VL-8B** (vision). The repo already has a pure-Mojo TEXT magic-prompt
(`pipeline/ideogram4_magic.mojo`, Qwen3-8B local). Qwen3-VL-8B (Abliterated-Caption-it) weights are
on disk but the VISION TOWER is NOT ported — that's a real model port (the long pole). The bbox
editor is a new SerenityUI canvas. These are multi-session.
## Phase 4 — ComfyUI node-dispatch completeness  ⏳ (executor + capped worker rebuild)
