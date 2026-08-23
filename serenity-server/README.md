# Serenity Studio server runbook

This is the authoritative run and debugging procedure for the Serenity Studio
browser application in this repository. The browser source is
`serenity-server/canvas/`; the Rust server serves those files directly and owns
HTTP, WebSocket, queues, graph lowering, GPU leases, and Mojo worker lifecycle.
Model execution remains in the Mojo workers under `output/bin/`.

The live job driver never terminates a generation because `/proc/<pid>/io`
reports physical reads during sampling. Disk-read assertions are profiling and
performance evidence, not a user-facing completion gate; worker failures,
invalid requests, IPC errors, and missing output artifacts still fail loudly.

Video routing is organized as a thin shared shell in
`crates/server/src/video.rs` plus backend modules in
`crates/server/src/video/`. MiniMax H3, LTX2, Wan 2.2, Scail2, and Bernini own
their admission and orchestration in separate modules; `probe.rs` contains
shared media inspection and `tests.rs` contains route-level coverage. Keep
model inference in Mojo: this Rust layer owns capability publication, request
validation, queues, process lifecycle, and artifact delivery.

MiniMax H3 follows an installed-software contract. Its canonical model appears
and remains selectable when the checkpoint directory, compiled runner, model
files, and linked GPU runtime libraries are present. Deleted generated videos,
benchmark reports, product-gate JSON, conditioning caches, modulation caches,
and INT8 resident caches do not hide or disable the model. BF16 starts directly;
the first INT8 or INT8 Fast request builds its selected acceleration cache in a
separate GPU-only phase, then reuses that cache on later requests.

The browser Video Edit tab is the one deliberate exception to that execution
architecture: it uses the vendored Genesis Rust/C/FFmpeg/OpenCL compositor as a
separate `gcompose` sidecar. It does not use Mojo and it never launches
Genesis's native egui application. The boundary, API, build, and measured smoke
evidence are documented in
`../docs/SERENITY_GENESIS_VIDEO_EDITOR_2026-07-23.md`.

## User checkpoint registry

Serenity Studio recursively discovers `.safetensors` beneath
`~/.serenity/models/checkpoints/`, `diffusion_models/`, `unet/`, and `dits/`.
Browser registry and capability reads are deduplicated and retry boundedly
across a transient local-server rebuild, so an already-open tab does not strand
its model picker empty. Persistent failures still surface as unavailable.
It also recursively discovers complete Diffusers bundles anywhere below the
model root by their `model_index.json`; a folder such as `microsoft_lens/` or a
future nested user folder does not require a source-code allowlist. The Models
and Generate refresh actions rescan the registry.

The Models tab treats full checkpoints and standalone diffusion/UNet weights
as base checkpoints. It also recursively inventories the SwarmUI-style
auxiliary model sets under `vaes/`, `loras/`, `Embeddings/`, `controlnets/`,
`text_encoders/`, `clip/`, and `clip_vision/`, plus Serenity's `ipadapters/`
and upscaler roots. Feature adapters and compiled runtime support weights are
shown in their own filters. Auxiliary artifacts remain separate from
base-model generation routing.

Architecture routing uses safetensors ModelSpec/compatible metadata first,
tensor-key signatures second, and legacy filename hints only as a fallback.
The generate API resolves the selected registry identity to an exact scanned
path and carries it over the Rust-to-Mojo job protocol; it does not accept an
arbitrary filesystem path.

When metadata is absent or wrong, open the checkpoint in the Models tab and
set **Model Type**. Serenity stores the override by registry-relative model
name in `~/.serenity/models/model_type_overrides.json`; **Reset to Auto**
returns to header, sidecar, and filename detection without modifying the
downloaded checkpoint. The override controls classification and routing but
does not claim that an architecture-specific worker can consume arbitrary
weights when that loader is not implemented.

The selected-checkpoint loaders currently cover:

- Krea 2 diffusion-model safetensors;
- extracted SDXL UNets and ordinary full SDXL checkpoints. FP16 and BF16 UNet
  tensors are normalized to the worker's BF16 execution dtype while loading.

For a full SDXL checkpoint, the selected file currently supplies the denoiser;
the installed shared SDXL CLIP-L, CLIP-G, tokenizers, and VAE remain in use.
Model cards expose this as `selected_checkpoint_scope: "denoiser"`.

Other architecture workers retain their finite bundled profiles. A discovered
file that its worker does not consume is disabled with a reason instead of
silently generating with different canonical weights. Unknown architectures
remain visible but are not reported as runnable merely because a file exists.
The bundled `microsoft_lens` Diffusers directory is admitted through the
Mojo-native `serenity_worker_lens` route at 1024x1024 only, with Euler/Simple,
CFG 5, optional negative text, and no LoRA or image-conditioning support. Lens
is recycled after every terminal job because its VAE path otherwise retains
about 14 GiB on the measured 24 GiB product GPU.
The LTX 2.3 route accepts a registry-classified, complete single-file LTX 2.3
checkpoint for ordinary generation and I2V; when detection is ambiguous, the
user must confirm **Model Type: LTX 2 / 2.3**. The official partial FP8
diffusion file remains available for its admitted standard path. Retake and
Extend deliberately select the complete
`ltx-2.3-22b-distilled-1.1` BF16 checkpoint because the creator topology needs
the bundled video and audio VAE encoders. Standalone VAE, text-encoder,
upscaler, and adapter artifacts remain visible as components rather than base
generators.

On 2026-07-27 this contract was product-gated with an ordinary FP16 SDXL full
checkpoint and with a Krea 2 checkpoint exposed through an arbitrary nested
filename. Both worker logs named the exact selected path and both runs produced
non-uniform 1024x1024 PNGs; the temporary Krea registry fixture was removed.

## LoRAs, samplers, and schedulers

LoRAs are recursively discovered beneath the configured model roots and shown
case-insensitively A-Z in Models, Generate, and Canvas. Selecting an adapter
passes its exact registry path and user multiplier to the model worker. Product
loading is connected for SDXL, Ideogram 4, SD 3/3.5, Qwen Image, Anima,
Flux.1, Chroma, Klein/Flux.2, Krea 2, Z-Image, and LTX-2. A custom adapter
without architecture metadata may reach the selected worker and is validated
against that worker's real tensor targets; positively identified cross-family
adapters are rejected with the conflicting architecture instead of being
silently ignored. Per-family capability metadata reports whether one or
multiple adapters can actually be composed.

Serenity publishes the same 44 sampler and 16 scheduler identifiers as the
bundled SwarmUI source for discovery/workflow compatibility. Canvas and
Generate do not expose that entire catalog blindly: both hydrate the selected
model's executable subset from the same `/v1/capabilities` document. Flux and
Chroma currently execute Euler/flow-match Euler and genuine DPM++ 2M over nine
creator-compatible schedules: Normal, Karras, Exponential, Simple,
DDIM Uniform, SGM Uniform, Beta, Linear Quadratic, and KL Optimal. Other
families publish their own measured implementations and defaults. Requested and
executed sampler/scheduler names are recorded separately in worker result
manifests; a public catalog name is not evidence that its algorithm executed.

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
Properties/Filters/Scopes/Audio dock, and track mixer. The right-dock gate
requires 143 native Genesis controls, replacement/group/track/subtitle editing,
named export/resolution presets, program-grade key actions, track removal, 25
searchable filter groups, RGB histogram/luma waveform/vectorscope/RGB parade,
Genesis-assembled stereo meters/spectrum/program waveform, a decoded source
envelope, and level/pan/mute/solo mixer controls. It discovers and groups every
available `.cube` file from Serenity uploads, adjacent workspace libraries,
and optional `SERENITY_VIDEO_LUT_DIRS`/`GENESIS_LUT_DIR` paths. It exercises
marker, snap, copy/paste/undo, audio mute/unmute, a LUT apply/exact-clear cycle,
and a native Saturation add/remove/persistence cycle, exports H.264 plus AAC
from the browser, probes the movie, and writes its screenshot, contact sheet,
and JSON result under
`output/checks/genesis_browser/`.

Properties auto-opens the clip under the playhead when a project loads and
falls back to editable project settings when there is no clip selection.

## Verify the desktop-style Generate screen

Generate is a capability-driven image and admitted text-to-video workspace:
filterable parameters on the left, a large result viewer in the center, Current
Batch on the right, the prompt and Generate action below the viewer, and
History/Presets/Models/LoRAs in the lower library. Inpainting and source-image
editing remain Canvas-owned.

The browser preflights and then submits the same flat request directly to
`/v1/generate`. It exposes only controls admitted by the selected model's
`/v1/capabilities` profile, including
compiled resolution choices, sampler and noise scheduler, CFG or distilled
guidance, seed, batch count, compatible LoRAs, variation seed where the worker
implements it, and init image plus creativity where img2img is admitted.
Unsupported VAE override, refiner, ControlNet, upscale, hires, video, and mask
fields are absent unless the selected server-published profile admits them.

With the server running, execute the focused browser contract:

```bash
SERENITY_BASE_URL=http://127.0.0.1:7811 \
  node scripts/check_serenity_generate_ui.js
```

The gate verifies the three-column layout and server-owned History, runnable
model filtering, Krea parameter reuse, Z-Image's sampler/scheduler surfaces,
exact Generate-to-Workflow values, admitted LTX/Sulphur/WAN profiles, identical
image preflight/generate bodies, and the flat `/v1/generate` request. The full
design and current real-generation evidence are recorded in
`../docs/SERENITY_GENERATE_UI_2026-07-24.md`. The shared resolution ladder those "compiled resolution choices" come from — the one `serenitymojo/training/aspect_buckets.mojo` file every image backend derives its valid sizes from, and how the Generate tab hydrates them via `/v1/capabilities` — is mapped in `../../serenitymojo/MAP.md` (2026-08-22: SERENITY GENERATE UI + SHARED RESOLUTION LADDER).

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
2. Read `canvas/CONTRACT.md` and the deployed JavaScript; do not infer behavior
   from an old TypeScript, identity source map, or build directory.
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
