# serenity-server — Rust control plane for serenitymojo

Purpose-built Rust replacement for the pure-Mojo daemon's **control plane**
(HTTP/WS/JSON/jobs/graph/dispatch). It drives the **unchanged Mojo inference
workers** over the existing AF_UNIX newline-JSON IPC. Inference stays pure Mojo+MAX.

Lives inside the Mojo repo (`github.com/CodeAlexx/mojodiffusion`), NOT any Python
"serenity" tree. Design: `serenitymojo/docs/SERENITY_RUST_CONTROL_PLANE_PLAN_2026-06-13.md`.

## Crates
- **`wire`** — FROZEN IPC contract (1:1 with `serenitymojo/serve/ipc_codec.mojo` +
  `backend.mojo`). `JobParams`, `WorkerEvent`, `CANCEL_LINE`. Do not edit field
  names/types/defaults without changing the Mojo side in lockstep. `cargo test -p serenity-wire`.
- **`ipc`** — spawn + drive a worker (re-host of `proc_ipc.mojo` +
  `process_isolated_backend.mojo` parent half).
- **`server`** — axum HTTP/WS, job queue, worker-driver thread.

## Current product surface

A Rust server drives the **already-built** Mojo workers end-to-end:
`POST /v1/generate` → worker runs → `GET /v1/progress` (WS) streams events →
a PNG with a `serenity.genparams.v1` tEXt chunk lands in the out dir.
On every completed job the Rust control plane also writes
`<output>.serenity_server_result.json` with schema
`serenity.server_result.v1`. That manifest records the workflow
client/schema/executor/route plan, actual output dimensions and file size,
submitted sampler/seed/steps/cfg settings, and whether a worker-specific sidecar
such as `*.zimage_daemon_result.json` was present. Treat this as workflow
completion evidence; sampler parity and visual quality remain separate gates.
The schema and sidecar discovery live in `crates/server/src/result_manifest.rs`,
and the API exposes the refs through `result_manifests` on `/v1/job/:id`,
`/v1/jobs`, `/v1/gallery/:id`, and each `/v1/gallery` item. `/v1/job/:id/result`
also reports `output_location` with `root_kind:"ui_workflow_gallery"`,
`root`, `inside_root`, and `relative_path` so callers can prove a completed
artifact landed in the product gallery root instead of a check directory.
The same `output_location` block is attached to `/v1/job/:id`, `/v1/jobs`,
`/v1/gallery/:id`, and each `/v1/gallery` item when an output path exists.

Current admitted image workers are Z-Image, Ideogram4, SDXL, Anima, SD3,
Klein/Flux2, and SenseNova-U1. Generic Flux.1-dev is blocked from
`/v1/generate` after the real browser workflow gate reached the Flux worker at
1024x1024/20 steps and failed with CUDA OOM at 15/20 on 2026-06-17. Chroma is
also blocked because the current Mojo path requires pre-encoded T5 sidecars and
has no production-admitted Rust-server worker route. Both remain visible through
blocked capability profiles so the UI/API fail loudly instead of falling back to
another image model. SenseNova is intentionally bounded to shape-dispatched
512x512 and 1024x1024 txt2img, no negative prompt, no LoRA,
no img2img/inpaint/masks, no VAE override, and no variation noise; it dispatches
to `serenity_worker_sensenova`. The installed worker was rebuilt through the
capped `build-worker-sensenova-raw` path and writes
`<png>.sensenova_daemon_result.json` on completion. Current 1024 evidence:
`output/run_serenity_ui/job-0037.png`, SenseNova-U1, workflow route `image`,
30 steps, seed `2026061730`, valid 1024x1024 RGB PNG, visual-health pass, and
both server and worker manifests present. Microsoft Lens still has a compiled
worker but remains blocked from `/v1/generate` until its render/OOM/parity gate
is accepted.

`POST /v1/preflight` accepts the same flat or workflow-shaped request body as
`/v1/generate`, runs the same production prequeue gate, and returns
`serenity.generate.preflight.v1`:

- `admitted`: whether `/v1/generate` would enqueue the request
- `same_gate_as_generate`: always true for this endpoint contract
- `block_profile`: local Mojodiffusion block/offload profile for the model family
- `artifact_profile`: local model/tokenizer/text-encoder/VAE presence report,
  checked before CUDA and before enqueue
- `output_root`: the actual `ui_workflow_gallery` root the server will pass to
  `/v1/generate`, including the `job-XXXX.png` artifact pattern
- `limits`: current production-surface limits, including no external repo runtime dependency

`POST /v1/generate` rejects unsupported workflow/raw/prequeue requests before
enqueue with `serenity.generate.error.v1`. The body preserves the top-level
`error` string for simple clients and adds `same_gate_as_preflight:true`,
`enqueue_blocked:true`, `capability_profile`, and `rejection_stage` for workflow
failures. Unsupported graph nodes still return HTTP 501. If a workflow lowers to
an image route but fails model admission, size/sampler policy, artifact gates, or
disabled feature policy, the response keeps `workflow_plan` and marks
`rejection_stage:"workflow_capability"` instead of collapsing into a flat
request error. Raw disabled features and post-parse prequeue failures return
HTTP 400 unless their parser stage has a more specific status.

Workflow bodies are lowered through the Rust graph IR before the image job shape
is built. The lowered request carries `workflow_route_kind` and
`workflow_plan` (`serenity.workflow_plan.v1`) with terminal nodes such as
`SaveImage`, `SaveVideo`, and `SaveAudioOpus`. The server admits only image
routes into the current `/v1/generate` queue. Non-image, mixed, or unresolved
workflow routes fail at `rejection_stage:"workflow_route"` with the route plan
preserved and `enqueue_blocked:true`, before `JobParams` or an image worker are
selected.

The graph executor does not require every workflow to contain a prompt node, a
VAE node, or an image terminal. Prompt, VAE, image, video, and audio behavior are
operator contracts in the IR. For example, LTX-style no-VAE video graphs lower
to a video route plan and fail loudly at the route gate until a video product
dispatcher is admitted; they are not forced through the image prompt/VAE path.

`POST /v1/grid` uses the same production gate before it creates any cell jobs.
Direct grid requests with active disabled surfaces, and per-cell sampler/model
requests rejected by prequeue validation, also return
`serenity.generate.error.v1` with `enqueue_blocked:true` and a
`capability_profile`. If a grid request includes a `workflow` body, the grid
route lowers it through `serenity-graph` first and applies the same
`workflow_route` gate as `/v1/generate`; valid image routes can become grid cell
jobs, while unsupported or non-image workflow routes fail before any cell is
enqueued. Top-level flat model/prompt fields do not bypass the workflow body.
The browser Grid XYZ panel submits inherited generation settings as
`workflow:{params:{...}}` plus top-level axis keys, so the grid UI uses the same
workflow-lowering route as direct grid API callers. It also uses the shared
prompt-syntax submit resolver before posting, preserving raw prompts while
sending resolved base prompt text and a concrete seed in `workflow.params`.
If a grid workflow lowers into a disabled surface, such as `init_image`, the
error preserves workflow route context with `rejection_stage:"workflow_capability"`
instead of being reported as a raw flat-grid rejection.

Concrete VAE overrides are blocked by preflight/generate today. Production
routes decode through each model manifest's baked local VAE artifact; `Automatic`
or an empty VAE value is accepted as that default.

Model-family admission, preflight reporting, and worker dispatch use the same
Rust classifier. When a request needs a different backend worker, the driver
swaps to the matching `serenity_worker_<family>` binary. If that target binary is
missing, the job fails loudly instead of running on the previous resident model.
The classifier accepts common production aliases, including `flux-2`/`klein`
for the bounded Klein 9B `flux2` route, `chroma` for the blocked Chroma route,
generic `flux` for the blocked Flux.1-dev route, and
`sdxl`/`sd_xl`/`sd-xl`/`sd xl` for SDXL. The browser adapter uses the same
backend mapping and `/v1/capabilities`
defaults before it enforces model limits, so model selection adjusts
width/height, steps, CFG, sampler, and scheduler before the workflow graph is
submitted. Current SenseNova browser defaults are 1024x1024, 30 steps,
`cfg:4`, `scheduler:"simple"`.

Ideogram-4 requests may provide `prompt`, `prompt_raw`, or `prompt_json`.
For `prompt_json`, string values are used directly and JSON object/array values
are serialized, preserving authored structured prompt fields such as bbox arrays.
The Rust server mirrors that value into both `prompt` and `prompt_raw` before the
same prequeue gate used by `/v1/generate`. The graph importer has the same
top-level `prompt_json` override for bounded raw Ideogram Comfy exports. The
default Ideogram scheduler remains `ideogram_logitnormal`; the bounded Comfy
`simple`/flowmatch scheduler path is admitted explicitly and broader scheduler
names still fail loud.

The canvas adapter calls `/v1/preflight` before `/v1/generate` with the exact
body it will submit. The main Generate button assembles a Comfy API prompt graph
and posts it as `workflow` with `workflow_client:"serenity.canvas.generate_ws"`;
prompt-syntax expansion patches that graph before preflight/generate so resolved
prompts, concrete seed, and prompt LoRAs reach the Rust workflow lowerer. A
blocked preflight is shown in the generate status and is not enqueued. Canvas
image/reference/mask layers are represented as workflow nodes, not browser-side
route guesses; current image-conditioning workflows lower and then fail loud at
`rejection_stage:"workflow_capability"` until that route is production-admitted.
Display labels such as `Z-Image (base)` are normalized before they enter
workflow `ckpt_name` or `workflow.params.model`. Legacy canvas callers that
invoke `api.submitPrompt(null)` also submit through `workflow:{params:{...}}`,
so the browser adapter no longer has a flat `/v1/generate` fallback.
Gallery context-menu and lightbox img2img/upscale actions follow the same
workflow-first rule: they create raster layer state and, for upscale, wait for
that layer before requesting generation. The browser does not hide or reject
those actions based on a guessed route shape; unsupported image-conditioning
graphs are rejected by Rust workflow preflight/capability gates.
The Refine / Upscale panel also authors workflow intent now. Active settings are
emitted as a `SerenityRefinerUpscaleIntent` graph node; Rust graph lowering
turns that node into the existing refiner/upscale/hires capability surface, and
the request fails at `rejection_stage:"workflow_capability"` until a real
two-pass executor is product-admitted.

The Workflows tab follows the same rule. Its built-in visual nodes compile to a
Comfy API prompt graph with a `SaveImage` terminal before submit. Legacy
`wfLower` adapters submit as `workflow: {params: ...}` instead of falling back to
flat `/v1/generate`; that adapter preserves `prompt_json` bbox prompts and maps
the UI `loras` alias to the canonical wire `lora` field. The active Ideogram
bbox workflow lowerer emits structured `prompt_json` plus `prompt_raw`, so bbox
arrays remain machine-readable through Rust preflight, generate, and PNG
genparams.

This endpoint is a server-side model/memory/block tool. It does not call SFv2,
EriDiffusion, Serenity Python, or any outside repo. Model artifacts may live in
the local `.serenity` model store, but missing artifacts fail the requested model
before the worker is launched.

Current Generate-screen evidence from 2026-06-16:

- `output/run_serenity_ui/job-0016.png`: Z-Image `z_image_base_bf16`, browser
  Generate button, 1024x1024, 16 steps, `cfg:5`, `scheduler:"simple"`, seed
  `2026061653`. Visually inspected as a sharp red teapot on a wooden table with
  a blue cloth backdrop. The browser valid-image gate measured
  `avg_stddev:68.639`, `luminance_range:249.333`, `edge_energy:4.314`, and
  `color_bins:1104`. Both server and `zimage` worker result manifests are
  present.
- `output/run_serenity_ui/job-0017.png`: SDXL `sd_xl_base_1.0`, browser
  Generate button, 1024x1024, 20 steps, `cfg:7`, `scheduler:"normal"`, seed
  `2026061654`. The driver swapped `zimage -> sdxl`; the result is a coherent
  lantern portrait. The browser valid-image gate measured
  `avg_stddev:52.72`, `luminance_range:254.667`, `edge_energy:2.963`, and
  `color_bins:696`. Both server and `sdxl` worker result manifests are present.
- `output/run_serenity_ui/job-0018.png`: Klein 9B/Flux2
  `flux-2-klein-base-9b_fp8_e4m3fn`, browser Generate button,
  `serenity.canvas.generate_ws` workflow graph, 512x512, 4 steps,
  `scheduler:"simple"`, seed `2026061655`. The driver swapped `sdxl -> flux2`;
  the result is a coherent dark-coat portrait. The browser valid-image gate
  measured `avg_stddev:27.285`, `luminance_range:212.333`,
  `edge_energy:1.992`, and `color_bins:256`.
- `output/run_serenity_ui/job-0003.png`: Z-Image, 1024x1024, 8 steps, real
  workflow UI path.
- `output/run_serenity_ui/job-0004.png`: Klein 9B/Flux2, 512x512, 4 steps, real
  workflow UI path. This proves dispatch and generation, not final Klein
  quality.
- `output/run_serenity_ui/job-0006.png`: SDXL `sd_xl_base_1.0`, 1024x1024,
  20 steps, real workflow UI path and coherent image output. The earlier
  4-step `job-0005.png` is only a smoke artifact.
- `output/run_serenity_ui/job-0008.png`: Z-Image, 512x512, 4 steps, generated
  by a real Playwright click on the browser `#btn-generate` button after the
  server-result manifest change. Its
  `output/run_serenity_ui/job-0008.png.serenity_server_result.json` confirms
  `workflow_client:"serenity.canvas.generate_ws"`,
  `workflow_schema:"serenity.workflow_graph.v1"`,
  `workflow_executor:"serenity.workflow_graph.executor.v1"`, image route,
  actual output `512x512`, and worker sidecar
  `output/run_serenity_ui/job-0008.png.zimage_daemon_result.json`.
- `output/run_serenity_ui/job-0011.png`: SDXL `sd_xl_base_1.0`, browser
  Generate button, 1024x1024, 20 steps, `cfg:7`, `scheduler:"normal"`, seed
  `2026061631`. Visually inspected as a coherent portrait. The browser
  valid-image gate measured `avg_stddev:38.112`, `luminance_range:245.667`,
  `edge_energy:3.263`, and `color_bins:314`.
- `output/run_serenity_ui/job-0013.png`: Klein 9B/Flux2
  `flux-2-klein-base-9b_fp8_e4m3fn`, browser Generate button,
  `serenity.canvas.generate_ws` workflow graph, 512x512, 4 steps,
  `scheduler:"simple"`, seed `2026061642`. Visually inspected as a coherent
  portrait. Current limits: Klein progress stays at `0/4` until completion, and
  the installed worker still lacks a worker-side result manifest until the
  capped Mojo worker build picks up the new `klein_runtime_backend.mojo`
  sidecar source.

The real-image regression gate is
`scripts/check_serenity_browser_generate_valid_image.js`; it clicks the browser
Generate button, waits for the job, decodes the saved PNG, and fails
placeholder/blank/flat outputs. The edge-energy hard floor is `1.0`; it is a
validity guard, not an aesthetic sharpness score.

The supported text-to-image matrix gate is
`scripts/check_serenity_supported_generate_matrix.js`:

```bash
NODE_PATH=${TMPDIR:-/tmp}/mojodiffusion-playwright-tools/node_modules \
PLAYWRIGHT_CHROMIUM_EXECUTABLE=/usr/bin/google-chrome \
SERENITY_MATRIX_REPORT=output/checks/serenity_supported_generate_matrix_latest.json \
node scripts/check_serenity_supported_generate_matrix.js
```

It clicks the browser Generate button for Z-Image, SDXL, and Klein/Flux2, then
checks `/v1/job/:id/result` for visual-health pass, dimensions, result manifests,
and worker manifests where the installed worker emits them. Latest matrix
evidence:

- `job-0019`: Z-Image `z_image_base_bf16`, 1024x1024, 16 steps,
  `avg_stddev:69.598`, `luminance_range:252`, `edge_energy:4.308`,
  `color_bins:1056`, worker schema `serenity.zimage.daemon_result.v1`.
- `job-0020`: SDXL `sd_xl_base_1.0`, 1024x1024, 20 steps,
  `avg_stddev:35.227`, `luminance_range:253.667`, `edge_energy:2.073`,
  `color_bins:633`, worker schema `serenity.sdxl.daemon_result.v1`.
- `job-0021`: Klein `flux-2-klein-base-9b_fp8_e4m3fn`, 512x512, 4 steps,
  `avg_stddev:29.085`, `luminance_range:232`, `edge_energy:2.144`,
  `color_bins:338`, worker result manifest not yet emitted by the installed
  worker.

The API result document exposes the same signal at `/v1/job/:id/result` as
`visual_health` (`serenity.visual_health.v1`). The rebuilt server reports
`pass` for current coherent artifacts:

- `job-0016`: 1024x1024 Z-Image, `avg_stddev:68.639`,
  `luminance_range:249.333`, `edge_energy:4.314`, `color_bins:1104`.
- `job-0017`: 1024x1024 SDXL, `avg_stddev:52.72`,
  `luminance_range:254.667`, `edge_energy:2.963`, `color_bins:696`.
- `job-0018`: 512x512 Klein, `avg_stddev:27.285`,
  `luminance_range:212.333`, `edge_energy:1.992`, `color_bins:256`.
- `job-0011`: 1024x1024 SDXL, `avg_stddev:38.112`,
  `luminance_range:245.667`, `edge_energy:3.263`, `color_bins:314`.
- `job-0013`: 512x512 Klein, `avg_stddev:25.769`,
  `luminance_range:217.667`, `edge_energy:3.032`, `color_bins:441`.

Block/offload profile metadata lives in Rust at
`crates/server/src/block_profiles.rs`. The profile contract deliberately marks
the control plane owner as Rust and the runtime owner as Mojo: Rust owns
admission, memory/block reporting, worker dispatch, and fail-loud product
limits; Mojo owns the CUDA/MAX model execution path.

`serenitymojo/serve/serenity_daemon.mojo` is the legacy pure-Mojo control-plane
monolith. It is still useful as a reference and compatibility bridge, but new
production control-plane behavior should land in this Rust server first.

## Rules for contributors (human or agent)
- `cargo build` / `cargo test` are fine (incremental, never OOMs).
- **NEVER run `mojo build` or `pixi run build-*`** — building Mojo on this box has
  OOM-killed the GNOME session. The stub worker is already built; Mojo builds are
  done by the orchestrator only, capped (`build-worker-stub-safe`).
- Code against the FROZEN `wire` + `ipc` public APIs; if a signature must change,
  document it rather than editing across crate boundaries mid-parallel-work.

## Run (once implemented)
```
cargo build
./target/debug/serenity-server --worker ../output/bin/serenity_worker_stub --out-dir ../output/run_serenity_ui
# Optional: add `--kind stub` to override the `/v1/health` backend label only.
# The standalone Mojo workers are fd-only: `serenity_worker_* <fd>`.
# Legacy monolith worker mode is explicit: `--daemon-worker-kind zimage`.
# then:
#   POST /v1/preflight {model:"qwen-image",prompt:"hi",steps:1,width:1024,height:1024}
#   POST /v1/generate  {model:"zimage",prompt:"hi",steps:1,width:1024,height:1024}
#   WS /v1/progress?job=<id>
```

Product UI/workflow generation evidence belongs under
`output/run_serenity_ui`. The server default uses that directory, and the
browser valid-image gate rejects artifacts that land outside it unless the gate
is given an explicit `SERENITY_VALID_IMAGE_OUTPUT_DIR`.
