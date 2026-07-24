# Serenity Canvas / InvokeAI design integration plan

Status: COMPLETE (practical parity scope)
Updated: 2026-07-21

## Ground truth

- The target product is the Serenity web Canvas served by
  `serenity-server/canvas/` on port 7811. The native Mojo desktop application in
  `/home/alex/serenityUI` remains parked.
- The design and interaction reference is InvokeAI's Canvas, primarily
  `features/ui/layouts/CanvasWorkspacePanel.tsx`,
  `features/ui/layouts/canvas-tab-auto-layout.tsx`, and
  `features/controlLayers/` in `/home/alex/invokeai-ref`.
- Serenity's Workflow node graph is a separate screen. Canvas work must not
  replace or overload the Workflow editor.
- Reproduce the useful product behavior and layout in Serenity's existing
  Konva/vanilla-JS architecture; do not transplant InvokeAI's React/Redux
  implementation wholesale.

## Existing Serenity foundation

The current Canvas has an infinite Konva surface, typed entities, bounding box,
modern paint/shape/gradient/lasso tools, capability-gated generation,
reference/control project data, staging, video overlays, undo/redo, project
persistence, gallery boards, and a status bar. SAM3 text, click, and exemplar
masking is connected through an isolated local CUDA service that releases its
model after an idle window. The
main files are `canvas-tab.js`, `layer-types.js`,
`canvas-tools.js`, `canvas-compositor.js`, `canvas-refimages.js`,
`canvas-sam.js`, `canvas-staging.js`, `canvas-video.js`, and `canvas-tab.css`.

## C0 measured parity inventory (2026-07-20)

Reference baseline: official InvokeAI `main` at
`82e26811264701477683cfc937d05c6977c5ecda` (2026-07-17). The current layout is
defined by `features/ui/layouts/canvas-tab-auto-layout.tsx`,
`CanvasTabLeftPanel.tsx`, `CanvasWorkspacePanel.tsx`, and
`features/controlLayers/components/CanvasLayersPanelContent.tsx`. The checked
upstream product screenshot is
`/home/alex/invokeai-ref/docs/src/content/docs/assets/invoke-webui-canvas.png`.

Serenity browser evidence was captured from the live web UI on `:7811` at
1600x900 and 1280x800. Both viewports loaded without console or page errors.
After the first C1 layout slice, the drawable stage measures 928x856 and
678x756 respectively, with all visible controls inside the viewport.

| Surface | Serenity state | Evidence and remaining gap |
| --- | --- | --- |
| Three-panel workspace | Working | CSS now places generation/settings left, the Konva workspace center, and entities/layers right; `canvas-tab.css`. |
| Center tool rail and bbox controls | Working | Select/bbox, move, brush, eraser, mask, rectangle, gradient, fill, text, lasso, pencil, speech bubble, clone, SAM, picker, pan, view, undo, and redo are visible; `canvas-tab.js:470`, `canvas-tools.js`. |
| Modern Shapes tool | Working | Rectangle, oval, polygon, and freehand shapes support additive/subtractive edits on draw and mask entities, with Enter/double-click commit and Escape cancellation; `canvas-tools.js`. |
| Gradient tool | Working | Linear/radial modes, two colors, transparent-end/clip choice, preview, commit, and cancellation are wired; `canvas-tools.js`. |
| Lasso tool | Working | Freehand/polygon selection, additive/subtractive mask edits, auto-mask, Enter/double-click commit, and Escape cancellation are wired; `canvas-tools.js`. |
| Entity families | Working, capability-gated | Draw, inpaint mask, regional guidance, control, adjustment, and text entities are editable and persistent. Backend-only effects are admitted from `/v1/capabilities` and fail before upload/GPU when unsupported; `layer-types.js`, `canvas-tab.js`. |
| Entity list operations | Working | Select, visibility, lock, inline rename, opacity, drag reorder, duplicate, merge-down, flatten, and delete are wired; `canvas-tab.js:1338-1595`. |
| Entity transforms and filters | Working | Unified transformer/move, bbox crop, PNG save, adjustment/filter entity, and draw-layer lock transparency are available and undoable; `canvas-tab.js`, `canvas-tools.js`, `canvas-refimages.js`. |
| Context menus | Working | Rename, transform, duplicate, arrange, lock, transparency lock, copy/save/crop, run-workflow, filter, convert, merge/flatten/delete, and mask morphology are type-aware; `canvas-refimages.js`. |
| Bounding box and view | Working | Resize handles, numeric dimensions, snapping, aspect lock, fit, reset, pan, zoom, and status readout are wired; `canvas-tab.js:802-1128`, `canvas-statusbar.js`. |
| Reference images | Working, capability-gated | Add/remove, preview, method, weight, start/end range, project persistence, and selected-model compatibility are wired. Method=Style selects the stacked source/style workspace and routes the reference through Mojo vision analysis into the selected Krea2 Raw, Krea2 Turbo, or Ideogram4 FlowEdit target conditioning; other reference methods remain capability-gated; `canvas-refimages.js`, `canvas-tab.js`. |
| Control/regional guidance | Working locally, capability-gated | Entity UI and compositor metadata persist. Current backends expose their unsupported reason and Canvas rejects unsupported generation before upload or CUDA allocation; `canvas-tab.js`, `canvas-compositor.js`. |
| SAM object selection | Working | Official `facebook/sam3` Transformers weights live outside the repository under `~/.serenity/models/sam3`; `serenity-sam3.service` provides text, labeled-click, and exemplar masks on port 7812, unloads after eight idle seconds, and `/canvas/sam3/status` admits the tool only while that service is reachable. Before every browser request, `/canvas/sam3/prepare` keeps the measured-safe Z-Image resident worker but reaps incompatible workers so Krea2's measured 20.6 GiB allocation cannot OOM SAM; active generation returns a visible conflict. Mask PNGs are converted to transparent red Canvas mask entities rather than treating opaque black mask pixels as selected; `canvas-sam.js`, `server/tools/sam3_service.py`, `server/src/main.rs`. |
| Generation controls | Working | Positive/negative prompts, sampler, scheduler, seed, batch, denoise, steps, CFG/guidance, model, bbox, video frames/FPS/audio, conditioning/noise artifacts, and multi-LoRA stack are wired with capability-driven visibility; `canvas-tab.js`. |
| Progress and busy state | Working | Header/status line, phase label, progress bar, load/conditioning/denoise/decode/mux state, terminal errors, and Canvas-local interrupt are wired from server events/status; `canvas-tab.js`, `canvas-statusbar.js`. |
| Staging loop | Working | Every Canvas result enters staging with candidate navigation, compare, full/partial accept, new-layer accept, reject, regenerate, and metadata preservation; `canvas-staging.js`. |
| Gallery and boards | Working | The Canvas right rail loads completed image/video jobs, stages a selected result, and supports persistent local board creation, assignment, and filtering; `canvas-tab.js`. |
| Source/result editing workspace | Working | Create mode keeps the dominant Canvas. FlowEdit, `Masked Edit - LanPaint`, and DynaEdit modes open two equal landscape panes. Style mode puts the selected/current Canvas source above the style reference in the left 34% and a 1024x1024 result Canvas in the larger right pane; the tool rail follows the result pane and does not obscure either reference. The lower reference header exposes an Entire image checkbox: it defaults checked and sends `auto_mask=false` for coherent full-frame stylization; unchecking restores FlowEdit's localized auto-mask. Source images/videos load through the native file picker or drop target; `canvas-tab.js`, `canvas-tab.css`. |
| Image edit / inpaint | Working on bounded compiled routes | Krea2 Raw and Turbo FlowEdit admit compiled 512x512 and 1024x1024 graphs; Ideogram4 admits 1024x1024. All preserve UI-authored source/target prompts, seed, step window, mask settings, and the selected checkpoint. Turbo visibly selects its distilled 8-step, CFG-zero profile and the Mojo velocity path skips the unused unconditional forwards. The shared capability-filtered `Masked Edit - LanPaint` workspace currently enables Krea2 Turbo 1024, Krea2 Raw 1024, and canonical Z-Image Base 1024 without cloning model-specific screens. Krea2 exposes the complete damped inner-loop graph with one optional LoRA and visible lambda, step-size, beta, friction, prompt-mode, early-stop, threshold, patience, and mask-blend controls. Z-Image hides those inapplicable inner controls and exposes its bounded `VAEEncode` plus `SetLatentNoiseMask` path with visible 4-step, CFG 1, denoise 0.65 controls and the shared LoRA chain. The same selector inventories the other upstream LanPaint families as disabled entries. Graph-ready image families share one model-preserving LanPaint tail transform, but they remain fail-loud until their Mojo backend and local checkpoint pass the runtime gate; no missing weights are downloaded. Registry-only Z-Image Turbo is disabled because the current resident worker loads canonical Base. See `docs/SERENITY_LANPAINT_MODEL_MATRIX_2026-07-22.md`; `workflow-builder.js`, `canvas-tab.js`, `graph/src/execute.rs`, `server/src/capabilities.rs`, `serenitymojo/serve/{krea2,zimage}_backend.mojo`. |
| DynaEdit video | UI boundary only | The pure-Mojo implementation remains `pipeline/lingbot_flowedit.mojo --dynaedit`, but it currently consumes staged tensor/prompt files and has no web request runner. Canvas accepts a source video for preview and shows the exact unavailable reason; it never falls back to LTX2 or Wan. |
| Run workflow on layer | Working when admitted | The type-aware context action sends an active raster layer through the current image-to-image workflow and returns it to staging. Z-Image is now the admitted img2img/inpaint family. Unsupported selected backends fail locally with the capability reason; `canvas-tab.js`. |
| Project/session persistence | Working | New/save/load are exposed; the v3 archive covers layers/pixels, bbox, references, LoRAs, all generation fields, and tool settings, with browser round-trip evidence; `layer-types.js`, `canvas-tab.js`. |
| Settings and hotkeys | Working | Tool/clipboard/history/view/nudge/delete shortcuts, an in-product shortcut dialog, project controls, bbox controls, and persisted tool settings are wired; `canvas-tab.js`. |
| LTX2 plus LoRA | Working | Canvas dynamically hydrates the active LTX2 workflow template, exposes the compatible registry-backed LoRA stack and exact compiled request controls, writes project/staging metadata, and reaches the pure-Mojo runner. Missing/wrong-architecture adapters reject before GPU. Exact LoRA and fixed-seed base movies completed and are non-identical. |

The permanent live browser gate is
`scripts/check_serenity_canvas_invoke_parity.js`. It verifies the landscape
workspace, two-pane source/result editor, native source file loading, FlowEdit
request preservation, the capability-filtered Krea2 Turbo/Raw and Z-Image Base
masked-edit Engine selector, per-engine visible control surfaces, the disabled
upstream family inventory, deferred model-specific LanPaint graph tails, Z-Image LoRA
graph, the full visible Krea2 LanPaint graph, and real server preflight for both
masked backends; it also covers the stacked Style reference layout and real 1024x1024 Krea2 Raw
preflight, DynaEdit fail-loud boundary, live SAM3 admission and GPU-prepare
handshake,
transparency-locked paint, persistent gallery boards, and the exact LTX2 plus
LoRA request. The
historical Design-B suite behind `SERENITY_LEGACY_CANVAS_TEST=1` is archaeology,
not the deployed Canvas contract.

## Delivery phases

### C0 — measured parity inventory

- Capture current Serenity and InvokeAI Canvas screenshots at the same landscape
  viewport.
- Build a feature/layout matrix: workspace chrome, tool rail, layer/entity list,
  bbox controls, settings, HUD/alerts, staging, context menus, keyboard controls,
  generation controls, and result acceptance.
- Mark every Serenity item as working, partial, missing, or intentionally out of
  scope. Do not infer behavior from labels alone.

Gate: one checked inventory with exact source paths and browser evidence.

### C1 — workspace layout and visual hierarchy

- Align the Serenity Canvas workspace with the InvokeAI arrangement: persistent
  tool controls, dominant center canvas, clear layer/entity rail, compact bbox
  and view controls, visible busy/progress state, and an unobstructed staging
  area.
- Preserve landscape responsiveness and prevent panels/toolbars from shrinking
  the drawable viewport below a usable size.
- Keep all model readiness and job state sourced from the server.

Gate: browser smoke at 1600x900 plus a narrower viewport; no clipped controls or
hidden canvas actions.

### C2 — entity and layer interaction parity

- Normalize raster/draw, inpaint mask, regional guidance, control/reference,
  adjustment, and text entities around consistent visibility, lock, rename,
  reorder, duplicate, delete, transform, and context-menu behavior.
- Make the selected entity and active tool unmistakable.
- Preserve session import/export and undo/redo across every mutating action.

Gate: add/edit/reorder/undo/redo/save/reload round-trip for each supported layer
family.

### C3 — generation and staging loop

- Keep generation controls usable without leaving Canvas.
- Show actual submission, model/conditioning load, denoise step, decode/mux,
  completion, interruption, and error state.
- Match InvokeAI's useful staging loop: navigate candidates, compare, accept,
  reject, regenerate, and create a new layer from a result.

Gate: real server job produces an artifact, enters staging, and can be accepted
onto the canvas with metadata preserved.

### C4 — LTX2 and LoRA Canvas integration

2026-07-20 milestone: Canvas authors `serenity.genparams.v1` directly from its
UI and active workflow template. The request carries exact frame count, FPS,
conditioning artifacts, seed/noise fixture, audio choice, and arbitrary
registry-backed LoRA rows. Rust is only the web control plane and pre-GPU
registry/admission boundary; inference remains the existing pure-Mojo LTX2
request binary.

The exact step-3000 request completed as `video-0020`: 512x768, 121 frames,
25 FPS, 20 Res2S steps, seed 42, one `ltx2_eri2_step3000` adapter at 1.0. Its
H.264 artifact contains 121 decoded frames over 4.84 seconds. The result
manifest reports 615.11 seconds wall time and 16,444.8 MiB peak VRAM. A temporal
contact sheet was visually inspected for identity and motion consistency. This
route remains explicitly experimental; it does not claim sampler, speed, or
audio parity.

Canvas then submitted the fixed-seed base comparison as `video-0023` with the
same prompt/trigger, conditioning, noise, geometry, schedule, sampler, and seed,
but zero trained adapters. It completed in 511.09 seconds at 15,924.9 MiB peak
VRAM. Its valid 121-frame H.264 artifact differs from the LoRA artifact both by
SHA-256 and by decoded frame content; a stacked temporal contact sheet confirms
a material identity/background/trajectory change while both runs remain
temporally coherent. The interrupted `video-0022` attempt is not evidence: the
desktop user session SIGKILLed its server process at step 1. The successful
rerun was isolated under the `serenity-canvas-codex.service` user unit.

- Integrate the completed 5080 LTX2 inference runner/worker rather than creating
  a second implementation.
- Add an LTX2 model surface to Canvas with supported width, height, frame count,
  FPS, steps, seed, scheduler, conditioning/noise, and audio controls. Compiled
  dtype/quantization/offload choices are reported as runner-owned rather than
  exposed as non-functional controls.
- Add a real LoRA loader: file/model-registry selection, one or more LoRA paths
  if the backend supports stacking, per-LoRA scale, remove/disable controls, and
  fail-loud validation before CUDA work.
- Carry the selected LoRA path and scale through the canonical SerenityUI
  request, pure-Mojo request runner, LTX2 overlay path, result manifest, and
  gallery/staging metadata. Mojo and Rust remain separate product stacks; this
  route has no Rust runtime dependency.
- Distinguish base LTX2, video-only LoRA, and audio/video LoRA compatibility.
  Reject unsupported combinations rather than silently ignoring adapters.

Gate: preflight admits an installed compatible adapter, a real LTX2+LoRA run
writes a valid MP4, metadata proves the exact adapter and scale applied, and a
paired fixed-seed base-vs-LoRA run shows a non-identical result.

Gate result: PASS. LoRA artifact `video-0020` has SHA-256
`efe9a5dd6e9e9014b79fe9eb3498576949ca5a46bd7f932bd155c30fabf829f1`;
base artifact `video-0023` has SHA-256
`d09b490eeddfe19431de4430b741561ca81f5c287dc32a31938b4536983d3ad2`.

2026-07-23 production-speed update: Canvas now exposes an explicit LTX2 guidance
mode. The default compiled route is Creator fast-distilled Euler: 8 stage-1
evaluations with `ltx2_distilled`, followed by the official 3-step stage 2.
The dev route remains explicit CFG-star Res2S with the LTX2 schedule and a
bounded 1-20 step range. Canvas, workflow lowering, and the Rust control plane
all preserve these authored values and fail before GPU work when the mode,
sampler, scheduler, or step count disagree.

Playwright submitted the exact fixed-seed step-3000 LoRA request through the
live Canvas as `video-0409`. It produced H.264 512x768, 121 frames, 25 FPS,
4.84 seconds duration in 52.36 seconds wall time: 41.06 seconds denoise and
7.42 seconds video decode. Frames 0/30/60/90/120 were visually inspected for
identity, motion, and convergence. Nsight plus an isolated target-shape VAE
gate found that the generic SDK path spent about 18.6 seconds exhaustively
autotuning Conv3d in every fresh process. The local BF16 NDHWC/FCQRS cuDNN
heuristic shim reduced isolated cold decode from 25.95 to 7.32 seconds. The
optimized run's final latent, raw RGB stream, and MP4 SHA-256 values are
byte-identical to the previously accepted `video-0406` outputs.

### C5 — source/result editing and masking

- Keep create mode full-width. Editing modes split only the center workspace,
  preserving the left generation controls and right entity/gallery rail.
- Load the source through the browser's native file manager or drop target and
  keep an immutable source preview beside the editable result/mask Canvas.
- Build FlowEdit from visible request controls. Krea2 Raw and Turbo expose their
  compiled 512x512 and 1024x1024 edit shapes; Turbo defaults to its creator
  8-step, CFG-zero schedule. Ideogram4 stays at 1024x1024 and requires structured
  JSON source and target captions. Missing weights fail during preflight.
- Admit only the Z-Image worker's verified init-image and mask semantics.
  Generic img2img/inpaint remains rejected for every other family, and UniPC
  img2img remains rejected until it has sliced-sigma parity evidence.
- Install SAM3 as an accessory segmentation service, not a diffusion backend.
  Keep it dependency-isolated and release its CUDA model before Mojo generation.
- Keep DynaEdit visible but unavailable until ordinary uploaded video can be
  staged into its pure-Mojo runner contract and results returned through the web
  job/event path.

Gate: landscape browser evidence for equal source/result panes and file loading;
FlowEdit graph lowering; Z-Image masked preflight plus decoded real output;
visually inspected SAM3 mask; DynaEdit rejection before upload/GPU.

Krea2 Raw 1024 evidence: `job-0054` completed 28 steps with the compiled
1024x1024 Mojo FlowEdit arm and the user-supplied neon anime style reference.
Visual inspection confirmed one subject, preserved arms-crossed pose, black
dress, framing, and apartment, with anime linework plus pink/cyan rim lighting;
the earlier Turbo attempts were rejected as quality failures. The Raw checkpoint
and its 13 GB W8A8 sidecar are local. `job-0058` then proved live progress IPC at
1/4, 3/4, and 4/4 before completion. Playwright verifies that Entire image
disables `auto_mask` in the submitted graph and is checked by default.

Krea2 Turbo 1024 evidence: the old Raw-like 28-step/CFG 5.5 request
`job-0114` took about 362.6 seconds and produced a fragmented masked result.
The corrected Turbo profile plus the CFG-zero Mojo forward shortcut completed
`job-0115` in 74.53 seconds (43.96-second denoise), and `job-0126` produced a
coherent full-frame watercolor/anime edit. The fully cold `job-0126` switch was
138.35 seconds because text/model files were not page-cached; this is recorded
separately from warm-cache latency. Nsight Systems emitted the requested CUDA
captures, but the installed 2023.4.4 importer hit its known `Wrong event order`
failure, so phase timers and wall time are the accepted measurements for this
run. After `job-0126`, `/canvas/sam3/prepare` released the measured 20,608 MiB
Krea2 worker and SAM returned ten masks in 1.82 seconds.

The resident-worker iteration path now mirrors unchanged-node reuse: four
conditioning bins, the normalized source latent, and the matching int8 DiT are
retained only while their keys remain unchanged. Warm seed `job-0141` took
77.21 seconds inside Mojo; exact Regenerate `job-0142` then hit all three caches
and completed in 52.56 seconds inside Mojo / 57 seconds through the polled HTTP
job lifecycle. A changed target instruction in `job-0143` visibly released the
20.7 GiB DiT before the text encoder, reused only the unchanged source latent,
rebuilt safely, and completed in 59.19 seconds inside Mojo / 62 seconds through
the HTTP lifecycle. All three 1024x1024 results passed visual-health checks and
were inspected as coherent full-frame watercolor edits. Cold 138.35-second and
warm/cached timings remain separate measurements.

Z-Image edit latency evidence after removing the contradictory Rust per-job
recycle: `job-0124` completed warm img2img in 1.60 seconds end-to-end (1.15
seconds in the Mojo manifest), and masked `job-0125` completed in 4.22 seconds
while preserving 1,718/4,096 latent mask pixels. Both outputs and the mask were
visually inspected. SD3 keeps its measured per-job recycle policy; Z-Image now
uses the resident lifecycle implemented by its Mojo backend.
The SAM prepare handshake keeps that measured-safe worker: PID 462913 stayed at
14,918 MiB through a real 1.58-second SAM request, and the following 16-step
production-default inpaint `job-0133` completed in 7.08 seconds end-to-end.

## Acceptance evidence

- Static UI/request contract checks.
- Rust server and Mojo worker builds at the repository's safe optimization
  level.
- Preflight before GPU allocation, including missing/incompatible LoRA rejects.
- Real progress events covering load, denoise, decode/mux, and terminal state.
- MP4 dimensions, duration, frame count, codec/container, file size, and result
  manifest inspection.
- Measured conditioning, denoise, decode/mux, total wall time, and peak/post-job
  VRAM where available.
- Visual inspection of generated image/video artifacts and masks.

## Final acceptance record (2026-07-21)

- `node scripts/check_serenity_canvas_invoke_parity.js`: PASS; 1600x900 Canvas
  center measured 928x856 and split into equal 464px source/result panes, no
  browser/page errors, SAM3 admitted, native source loading worked, transparency
  clipping and gallery boards persisted, and the exact LTX2 LoRA request was
  preserved.
- The same browser gate selected Reference Images Method=Style, measured the
  315.5px stacked source/style column beside the 612.5px result pane, verified a
  1024x1024 bounding box, proved one vision-analysis request, rejected duplicate
  IP-Adapter injection, and passed the real `/v1/preflight` gate with all 13
  Ideogram4 artifacts present. `output/bin/qwen3vl_caption` was rebuilt at `-O2`;
  a real `/v1/caption` GPU request returned a grounded caption.
- `python3 scripts/check_canvas_preflight_submit_contract.py`: PASS.
- `node scripts/check_canvas_flowedit_conditioning.js`: PASS; Turbo remained
  selected at 1024x1024 with the visible 8-step/CFG-zero request.
- Rust server: 122/122 tests passed. The live browser gate also exercised the
  SAM GPU-prepare handshake and returned a real mask after the Krea2 pressure
  test. Video execution was intentionally excluded from this performance pass.
- `cargo test --manifest-path serenity-server/Cargo.toml -p serenity-server
  --no-fail-fast`: 120 passed, 0 failed.
- Mojo request binary rebuilt at optimization level 2 with the repository's
  CUDA/cuDNN link contract; unsupported checkpoint and incompatible/missing
  LoRA probes rejected before GPU allocation.
- C2 entity, C3 staging, C4 tools, project/gallery, and LTX2 project/request
  Playwright gates all passed with no console or page errors.
- `video-0020/result.json` and `video-0023/result.json` prove exact request,
  sampler, schedule, frame, timing, and VRAM metadata. `ffprobe` independently
  counted 121 H.264/yuv420p frames at 512x768 and 25 FPS for the base artifact.
- LoRA and base temporal contact sheets were visually inspected. Neither movie
  shows the earlier ladder/neck corruption; the LoRA result preserves the
  previously user-confirmed trained identity.
- Z-Image masked job `job-0025` completed through the real Mojo worker at
  512x512/4 steps. The worker consumed the SAM3 grayscale PNG through the
  explicit red channel, measured 1718/4096 preserved latent pixels (mean
  0.4213), wrote both result manifests, and produced a visually inspected PNG
  with the subject region changed while the unmasked scene remained anchored.
  The preceding `job-0024` alpha-default attempt is not acceptance evidence; its
  4096/4096 preserve count exposed the mask-channel bug fixed by `ImageToMask`.
