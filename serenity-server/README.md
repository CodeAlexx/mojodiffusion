# Serenity Studio server runbook

This is the authoritative run and debugging procedure for the Serenity Studio
browser application in this repository. The browser source is
`serenity-server/canvas/`; the Rust server serves those files directly and owns
HTTP, WebSocket, queues, graph lowering, GPU leases, and Mojo worker lifecycle.
Model execution remains in the Mojo workers under `output/bin/`.

The browser Video Edit tab is the one deliberate exception to that execution
architecture: it uses the vendored Genesis Rust/C/FFmpeg/OpenCL compositor as a
separate `gcompose` sidecar. It does not use Mojo and it never launches
Genesis's native egui application. The boundary, API, build, and measured smoke
evidence are documented in
`../docs/SERENITY_GENESIS_VIDEO_EDITOR_2026-07-23.md`.

Run every command from the repository root. Do not use another checkout, a
standalone static-file server, a retired Mojo daemon, or the trainer web UI as a
substitute for Serenity Studio.

## Inspect before launching or building

```bash
git rev-parse --show-toplevel
git worktree list --porcelain
git status --short
test -x serenity-server/target/release/serenity-server
find output/bin -maxdepth 1 -type f -executable -name 'serenity_worker_*' -printf '%f\n' | sort
ps -eo pid,lstart,args | rg 'serenity-server|serenity_worker_' | rg -v 'rg '
```

There must be one canonical worktree at `/home/alex/mojodiffusion`. Do not
start a second server on the same port. Do not build when the required server
and worker binaries already exist.

## Launch the existing production UI

Choose a real installed worker; `serenity_worker_stub` is test-only. The server
will switch to other installed family workers when a request selects them.

```bash
mkdir -p output/run_serenity_ui
serenity-server/target/release/serenity-server \
  --worker output/bin/serenity_worker_zimage \
  --port 7801 \
  --out-dir output/run_serenity_ui
```

Leave that foreground process attached so its Rust and Mojo logs remain
visible. Open `http://127.0.0.1:7801/`. The UI is served by this process; there
is no separate npm, Vite, or static-server step.

In another terminal, prove the exact running process before browser debugging:

```bash
curl -fsS http://127.0.0.1:7801/v1/health
curl -fsS http://127.0.0.1:7801/v1/capabilities
curl -fsS http://127.0.0.1:7801/
curl -fsS http://127.0.0.1:7801/video_edit/status
```

If port 7801 is occupied, inspect its owner. Do not kill it or silently select
another port. A port change must be explicit and the browser URL and all test
environment variables must use the same port.

## Build only when the measured binary is missing or stale

Building Rust does not build Mojo workers:

```bash
pixi run cargo build --release \
  --manifest-path serenity-server/Cargo.toml \
  --bin serenity-server
```

Build the separate browser video-editor worker when
`output/bin/genesis-gcompose` is missing or stale:

```bash
pixi run build-genesis-video-editor
```

For an output directory directly under the repository's `output/` directory,
the server resolves that worker automatically. An unusual deployment can set
`SERENITY_GENESIS_WORKER` and `SERENITY_GENESIS_ASSETS` explicitly.

With the server running on port 7811 and the verified LTX source movie present,
run the real browser edit/export acceptance gate:

```bash
pixi run check-genesis-video-editor
```

The gate imports real video and audio through the visible toolbar, proves the
thumbnail strip, music waveform, synchronized native-clock playback with
changing large-preview pixels, and the Genesis-style media bin,
Program/Source monitors, timeline editing toolbar, persistent
Properties/Filters/Scopes/Audio dock (including parameter controls), and track
mixer. It exercises marker,
snap, copy/paste/undo, audio mute/unmute, and Saturation enable/disable actions,
exports H.264 plus AAC from the browser, probes the movie, and writes its
screenshot, contact sheet, and JSON result under
`output/checks/genesis_browser/`.

Properties auto-opens the clip under the playhead when a project loads and
falls back to editable project settings when there is no clip selection.

To inspect a particular persisted project through the same browser gate:

```bash
GENESIS_INSPECT_PROJECT_ID=project-id pixi run check-genesis-video-editor
```

Mojo worker builds are separate, GPU-architecture-specific operations. Never
run a broad `pixi run build-*` merely to debug the browser. Identify the exact
missing worker and use its existing `pixi.toml` or
`scripts/verify_renamed_entrypoints.sh` entry only when the user authorizes that
build.

## Reproduce API behavior before editing

Preflight and generation must receive the same request. Example wiring probe:

```bash
request='{"model":"zimage","prompt":"test","width":512,"height":512,"steps":1,"cfg":4,"seed":42,"sampler":"euler","scheduler":"simple"}'

curl -sS http://127.0.0.1:7801/v1/preflight \
  -H 'Content-Type: application/json' \
  --data "$request"
```

Do not submit `/v1/generate` unless the user requested a generation and
preflight admits the exact body. A browser/UI repair must first capture:

- the browser URL and selected model/template;
- the request body sent to `/v1/preflight` and `/v1/generate`;
- the HTTP status and response body;
- the terminal server/worker log lines;
- `/v1/health`, `/v1/capabilities`, and the relevant job/result record; and
- the artifact and result-manifest paths under `output/run_serenity_ui/`.

## Trace a browser failure

1. Confirm the page came from the current server with the health and root
   checks above.
2. Read `canvas/CONTRACT.md`, the deployed JavaScript, and its checked-in
   `.map`; do not infer behavior from an old TypeScript or build directory.
3. Follow the same body through browser assembly, `/v1/preflight`, workflow
   lowering, production admission, worker dispatch, queue/progress, result
   manifest, and gallery.
4. Classify the failure before changing code: browser request, Rust
   control-plane contract, missing/stale worker binary, missing model artifact,
   or Mojo runtime.
5. Capture a failing baseline, make only the authorized repair, repeat the same
   request, and inspect the actual image/video when a generation was requested.

Useful read-only endpoints include `/v1/health`, `/v1/capabilities`,
`/v1/jobs`, `/v1/gallery`, `/object_info`, `/templates`, and the job/result
routes returned by submission. Browser progress uses `/ws`; per-job progress is
also available through the server's documented progress route.

## Verification after a UI change

Run the narrow checks for the touched surface:

```bash
pixi run cargo test --manifest-path serenity-server/Cargo.toml --workspace
node --check serenity-server/canvas/js/<touched-file>.js
python3 -m json.tool serenity-server/canvas/js/<touched-file>.js.map >/dev/null
```

Also run the relevant `scripts/check_*` product contract and a real browser
smoke for the changed route. A static page load, compile, dropdown entry, or
stub-worker result is not production evidence. Generation evidence must bind
the exact request, model, seed, steps, worker, output, manifests, timing, VRAM,
and visual inspection.

## Required handoff

Every machine or agent handoff must state:

- canonical repository, commit, worktree count, and dirty paths;
- whether a server, worker, trainer, or GPU process is still active;
- exact server launch command, port, worker, output root, and browser URL;
- exact files owned and changed;
- commands run with exit codes;
- request bodies and model/template rows proven;
- output and manifest paths plus visual inspection result;
- artifact, code, or runtime blockers; and
- the single exact next command to run.

Never hand off “start the UI,” “run the server,” or “continue debugging” without
those concrete values. A future agent must not spend tokens reconstructing a
known launch procedure.
