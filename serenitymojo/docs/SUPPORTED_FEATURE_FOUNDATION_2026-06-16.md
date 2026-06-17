# Supported Feature Foundation

Date: 2026-06-16

Purpose: keep feature admission out of per-model folklore. The Rust server now
owns a machine-readable `/v1/capabilities` contract for the current
`/v1/generate` product route. UI controls, preflight checks, and readiness
scripts should use this contract before exposing or accepting a feature.

This is not a parity claim. It is the shared admission layer that says what is
currently supported and what must fail before enqueue.

## Contract

- Endpoint: `GET /v1/capabilities`
- Schema: `serenity.capabilities.v1`
- Gate source:
  `serenity-server/crates/server/src/capabilities.rs::validate_generate_prequeue`
- Product route: `/v1/generate`
- Preflight route: `/v1/preflight`
- Per-request profile schema: `serenity.capability_profile.v1`, returned in
  `/v1/preflight` reports and structured `/v1/generate` errors as
  `capability_profile`
- Generate error schema: `serenity.generate.error.v1` for fail-loud product
  rejections before enqueue
- Unsupported policy: `fail_loud`
- Runtime dependency on external repos: `false`

Implementation note: the capability classifier, sampler/scheduler normalization,
Ideogram prompt JSON handling, raw unsupported-surface guard, prequeue validator,
and `/v1/capabilities` payload builder live in
`serenity-server/crates/server/src/capabilities.rs`. Keep this out of
`main.rs`; the Rust server entrypoint should route and orchestrate, not grow a
second admission system.

## Workflow IR Route Plan

Workflow requests are not flat txt2img requests with extra metadata. They lower
through `serenity-graph` first, and the graph result becomes the routing source
of truth:

- `workflow_route_kind`: `image`, `video`, `audio_video`, `mixed`, or `unknown`
- `workflow_plan.schema`: `serenity.workflow_plan.v1`
- `workflow_plan.terminal_nodes`: the graph terminals that determine the route,
  including `SaveImage`, `SaveVideo`, and `SaveAudioOpus`
- `workflow_plan.source`, `node_count`, and `edge_count`

The current `/v1/generate` queue admits image routes only. A workflow route that
lowers to video, audio-video, mixed, or unknown fails before image `JobParams`
construction with `rejection_stage:"workflow_route"` and
`enqueue_blocked:true`. The response preserves the workflow plan so the next
dispatcher can be wired without guessing from UI fields.

The route plan is intentionally node/operator based. The executor no longer has
a global "must contain prompt" rule, and the server does not assume every graph
has a VAE or image terminal. No-VAE LTX-style video graphs can now lower to a
video plan through `LTXVLoader`, `LTXVSampler`, `SaveVideo`, optional
`LoadAudio`, and `SaveAudioOpus`; they fail loud only because the image product
queue is not the right dispatcher.

## Result Manifest Contract

Every completed Rust-control-plane job now writes a server-side result manifest
next to the output artifact:

- Path: `<output>.serenity_server_result.json`
- Schema: `serenity.server_result.v1`
- Writer: `serenity-server/crates/server/src/main.rs` on `WorkerEvent::Done`
- Purpose: record workflow dispatch and artifact evidence without depending on
  every Mojo worker having its own backend-specific sidecar yet

The manifest records job id, model, output path, actual PNG dimensions and file
size, submitted seed/steps/cfg/sampler/scheduler, workflow
client/schema/executor/source/route plan, and whether a worker sidecar such as
`*.zimage_daemon_result.json` or `*.sdxl_daemon_result.json` was found.

The server also exposes these paths through API metadata:

- `/v1/job/:id` and `/v1/jobs`: top-level `result_manifests` and
  `metadata.result_manifests` when metadata is present
- `/v1/gallery/:id`: top-level `result_manifests` and
  `metadata.result_manifests`
- `/v1/gallery`: each item carries the same gallery fields

This is an evidence contract, not a visual-quality or sampler-parity claim. The
readiness label is `server_completion_evidence`, and the manifest explicitly
notes that sampler parity and visual quality require separate gates. If no
worker-specific sidecar exists, the manifest still writes and includes
`worker_specific_result_manifest_missing` in `readiness.limits`.

The contract lists each admitted image backend with:

- production status and worker binary
- default width, height, steps, CFG, sampler, and scheduler
- admitted image sizes
- supported sampler/scheduler subsets
- feature support for text-to-image, CFG, negative prompts, bbox prompt JSON,
  LoRA limits, prompt weights, image conditioning, VAE override, refiner,
  upscale, outpaint, ControlNet, video, inpaint, and image-to-image

`/v1/preflight` embeds the selected model's `capability_profile` from the same
tables. Admitted models receive their backend entry plus selected-model
metadata. Known blocked models such as Qwen image, Klein/Flux2 4B, and generic
Flux2-dev receive a fail-loud blocked profile with empty sampler/scheduler
support and all features marked unsupported. The bounded Klein 9B txt2img route
is admitted separately as `flux2`. This lets tools inspect the exact request
surface even when the request is rejected before enqueue.

Workflow-lowering failures from `/v1/preflight` also return the selected
`capability_profile` when a top-level model is present. The response marks
`rejection_stage:"workflow_lowering"` and preserves the graph error status, such
as HTTP 501 for an unsupported Comfy node.

Workflow-lowering failures from `/v1/generate` preserve the same graph HTTP
status while returning `serenity.generate.error.v1` with `admitted:false`,
`same_gate_as_preflight:true`, `enqueue_blocked:true`,
`rejection_stage:"workflow_lowering"`, and the selected `capability_profile`
when a top-level model is present.

`/v1/grid` is not a separate capability surface. It expands to one generate job
per cell, so direct grid requests use the same guards before any cell is
enqueued. If the request includes `workflow`, grid first lowers it through the
same graph IR and route gate as `/v1/generate`; flat top-level fields cannot
bypass a workflow body. Active disabled raw surfaces return
`serenity.generate.error.v1` before axis expansion; per-cell prequeue failures
return the same schema with `rejection_stage:"grid_cell_prequeue"` and
`grid.one_generate_job_per_cell:true`.

## Canvas Consumer Contract

The browser route also uses this contract through
`serenity-server/canvas/js/api.js`. The API adapter caches
`/v1/capabilities`, emits `capabilities:loaded`, and exposes shared helpers for
model-to-backend mapping, default generation settings, and feature support
checks. UI modules must consume those helpers rather than walking the payload
independently.

Current consumers:

- `param_rail.js`: capability defaults, sampler/scheduler subsets, production
  sizes, and LoRA limits. Model aliases are normalized before control defaults
  are enforced. This is required for `flux-2`/`klein` to receive `flux2`
  512x512 limits, for `sd_xl_base_1.0` to receive SDXL's `normal` scheduler and
  1024x1024 limit, and for `sensenova-u1` to select 1024x1024, 30 steps,
  `cfg:4`, and `scheduler:"simple"` instead of stale browser fallbacks.
- `prompt_bar.js`: negative-prompt availability.
- `refiner_upscale.js`: workflow-intent refiner/upscale/hires controls.
- `gallery_pro.js`: workflow-state img2img/upscale actions.
- `api.js` and `generate_ws.js`: the main Generate button assembles a Comfy API
  prompt graph and submits it as `workflow` with
  `workflow_client:"serenity.canvas.generate_ws"`; preflight and generate receive
  the same workflow body. Structured `serenity.generate.error.v1` bodies are
  preserved, generate-route rejection is distinct from backend/network failure,
  and `generate:rejected` carries the preserved capability profile. Canvas
  raster/reference/mask layers are expressed as workflow nodes and uploaded
  paths inside `LoadImage`, `VAEEncode`, and mask nodes. The browser no longer
  blocks those shapes before preflight; current unsupported image-conditioning
  workflows lower through Rust and fail at `rejection_stage:"workflow_capability"`.
  Display labels such as `Z-Image (base)` are normalized to model ids before
  entering `CheckpointLoaderSimple` or `workflow.params`. Legacy canvas
  `api.submitPrompt(null)` calls are wrapped as `workflow.params` instead of
  posting flat model/prompt fields.
- `gallery_pro.js`: context-menu and lightbox img2img/upscale actions create
  raster layer workflow state instead of hiding or rejecting the action in the
  browser. Upscale waits for the image layer to load before emitting
  `generate:request`, then the normal Generate path submits the resulting graph.
  Unsupported image-conditioning routes still fail at Rust workflow preflight or
  `workflow_capability`; Gallery is not an admission authority.
- `refiner_upscale.js` and `generate_ws.js`: active Refine / Upscale settings
  become a `SerenityRefinerUpscaleIntent` workflow node. The Rust graph executor
  lowers that node into refiner/upscale/hires metadata so the existing
  capability gate can reject it as `workflow_capability` with the image route
  plan preserved. The browser no longer clears or disables this panel based on a
  guessed txt2img-only route.
- `prompt_syntax.js`: Swarm-style wildcard/random/LoRA syntax is resolved before
  submit and patched into the outgoing workflow graph, including CLIPTextEncode
  text, KSampler seed, and prompt LoRA loader chain. Workflow params adapters
  also receive resolved prompt text, concrete seed, and LoRA stack fields before
  submit.
- `workflows.js`: the Workflows tab built-in nodes compile to a Comfy API
  prompt graph with a `SaveImage` terminal before submit. Legacy `wfLower`
  adapters stay inside the workflow envelope as `{params: ...}` instead of
  bypassing the graph lowerer; the Rust adapter preserves Ideogram
  `prompt_json`/bbox payloads and maps UI `loras` to canonical wire `lora`.
  Runtime-created workflow graphs use the same capability-backed defaults for
  missing width, height, steps, CFG, sampler, and scheduler values.
- `ideogram4_nodes.js`: the active Ideogram bbox workflow lowerer emits
  structured `prompt_json`, `prompt_raw`, and an explicit empty negative prompt
  through `workflow.params`. The old standalone bbox builder has the same
  workflow-envelope submit shape if it is wired back in.
- `grid_xyz.js`: the browser Grid XYZ panel posts axis keys at the top level,
  but inherited generation settings live under `workflow.params` with
  `workflow_client:"serenity.canvas.grid_xyz"`. `/v1/grid` lowers that workflow
  before axis collection and per-cell prequeue validation. Before submit, Grid
  XYZ uses the shared `prompt_syntax.js` submit resolver, so the base prompt,
  seed, raw prompt, and LoRA stack match the Generate path.

The generate adapter does not synthesize a production two-pass executor or
outpaint executor while those surfaces are unsupported. Refine / Upscale is
represented as workflow intent and rejected by Rust capability gates; image,
reference, and mask layers can also be submitted as workflow nodes, but that is
not image-to-image admission by itself. The workflow node surface, route
executor, and Rust prequeue capability gate must all be widened before those
controls can become live.

## Current Product Limits

The Rust `/v1/generate` route currently admits bounded image generation only:

- one image per queued job
- image routes may come from a flat body or from an image workflow terminal
- production image evidence must use admitted route dimensions; 256px-scale
  artifacts are smoke/MVP evidence only and must not be cited as a working
  production generator
- product matrix prompts must be representative user prompts, not object-only
  blank/noise sentinels. The current browser matrix records the exact prompt and
  uses the surreal Dutch-master landscape-burden prompt as `production_prompt_v2`.
  Its automated image gate only rejects corrupted, blank, flat, or placeholder
  artifacts; prompt adherence and aesthetic quality still require visual review.
- no image-to-image product route admitted yet
- no inpaint
- no image conditioning
- no VAE override
- no hires two-pass
- no refiner
- no upscale
- no outpaint
- no ControlNet
- no video queue execution through `/v1/generate`; video workflows route-plan and
  fail at `workflow_route` before image enqueue

The Workflows tab is admitted for the current built-in image graph shape and for
legacy params adapters that lower through `workflow.params`. This is not a claim
that every possible future Workflows node is production-admitted; new node
families still need graph importer support, route planning, capability gating,
and product evidence.

Those features are represented as `supported:false` with `policy:"fail_loud"`.
The canvas adapter keeps forwarding txt2img sentinels for those fields, and
direct API posts still hit the same prequeue validator.

Meaningful raw requests for disabled surfaces are rejected before enqueue. The
guard permits browser/default sentinels such as `null`, `false`, empty strings,
`upscale_by:1.0`, and `denoise:1.0`, but rejects active ControlNet, refiner,
upscale, outpaint, non-Ideogram `prompt_json`, and image-conditioning fields.
It also rejects workflow-lowered disabled fields such as `init_image`,
`mask_image`, `conditioning_mask_image`, `inpaint_conditioning_image`,
`reference_image`, `outpaint_left`, `threshold_mask_value`, and LanPaint
metadata before those values can become ignored PNG metadata.
For `/v1/preflight`, these return `admitted:false`; for `/v1/generate`, they
return HTTP 400 with `serenity.generate.error.v1`, `enqueue_blocked:true`, and
the selected capability profile, without changing `/v1/jobs`.
When the disabled field was produced by workflow lowering, the report keeps the
workflow route context and uses `rejection_stage:"workflow_capability"` so the
caller can distinguish "graph lowered correctly, feature not admitted" from
`workflow_lowering` and `workflow_route` failures.
Direct `/v1/grid` posts use the same no-enqueue contract. If a grid workflow
lowers into a disabled surface, Grid now uses the same `workflow_capability`
stage and preserves `workflow_plan` instead of collapsing the error into a raw
flat-grid rejection.

## Admitted Backend Highlights

- Z-Image: `512x512` and `1024x1024`, negative prompt admitted, LoRA overlays
  admitted, sampler subset `euler`, `flowmatch_euler`, `dpmpp_2m`, `uni_pc`,
  `uni_pc_bh2`, scheduler subset `simple`, `sgm_uniform`. Karras is not
  advertised and is rejected before enqueue.
- Ideogram4: `1024x1024`, bbox `prompt_json` admitted, negative prompt and LoRA
  rejected, scheduler subset `ideogram_logitnormal`, `simple`.
- SDXL, Anima, SD3, Flux: bounded `1024x1024` routes with backend-specific
  sampler/scheduler subsets. SDXL aliases include `sdxl`, `sd_xl`, `sd-xl`,
  `sd xl`, `stable-diffusion-xl`, and `animagine`. Flux admits at most one LoRA
  and rejects negative prompts.
- Qwen, Z-Image L2P, and video execution remain blocked or routed elsewhere
  until separate product gates pass. Flux2/Klein has a bounded workflow-IR image
  route that can generate through the Rust server, but it still lacks strict
  timing/VRAM result-sidecar evidence before a production-quality claim.

## Workflow Evidence

Live no-GPU route gate:

- `output/checks/workflow_route_live_http/report.json`
- LTX-style no-VAE workflow lowered to `workflow_route_kind:"video"` with a
  `SaveVideo` terminal.
- `/v1/preflight`: HTTP 200, `admitted:false`,
  `rejection_stage:"workflow_route"`.
- `/v1/generate`: HTTP 501, `schema:"serenity.generate.error.v1"`,
  `enqueue_blocked:true`.
- `/v1/jobs` remained empty.
- `/v1/grid` workflow-bypass gate:
  `output/checks/serenity_server_t2i_product_gate_grid_workflow_route.json`
  includes `zimage_grid_workflow_unsupported_node_generate_error` with HTTP 501,
  `schema:"serenity.generate.error.v1"`,
  `rejection_stage:"workflow_lowering"`, and unchanged job count.
- Canvas workflow submit gate:
  `output/checks/serenity_server_t2i_product_gate_canvas_workflow_submit.json`
  includes `zimage_browser_workflow_image_route_profile` with HTTP 200,
  `workflow_route_kind:"image"`, `workflow_plan.source:"comfy_api_prompt_graph"`,
  and a `SaveImage` terminal. The browser smoke also proves `/v1/preflight` and
  `/v1/generate` receive identical workflow bodies with no flat prompt/model
  leakage.
- Workflow capability gate:
  `output/checks/serenity_server_t2i_product_gate_workflow_capability_route.json`
  includes `zimage_browser_workflow_img2img_capability_profile` with HTTP 200,
  `admitted:false`, `rejection_stage:"workflow_capability"`,
  `workflow_route_kind:"image"`, `workflow_plan.source:"comfy_api_prompt_graph"`,
  and a `SaveImage` terminal. The same browser smoke proves a raster layer
  submits `LoadImage` and `VAEEncode` with an uploaded `/uploads/` path, and
  that `Z-Image (base)` is normalized to `z-image` before workflow submit. It
  also proves `api.submitPrompt(null)` sends `workflow.params`, preserves
  `prompt_raw`, resolves `<random:...>`, and does not leak flat top-level
  generate fields. Gallery Pro is included in the same browser smoke: a saved
  result exposes img2img/upscale actions, the upscale action waits for raster
  layer creation, and the outgoing graph contains `LoadImage`/`VAEEncode` inside
  the workflow envelope.
- Refine/Upscale workflow capability gate:
  `output/checks/serenity_server_t2i_product_gate_refiner_upscale_workflow.json`
  includes `zimage_browser_workflow_refiner_upscale_capability_profile` with
  HTTP 200, `rejection_stage:"workflow_capability"`,
  `workflow_route_kind:"image"`, `workflow_plan.source:"comfy_api_prompt_graph"`,
  and a `SaveImage` terminal. The error remains explicit:
  hires two-pass depends on img2img refine and is disabled until the production
  executor is admitted.
- Workflow params fallback gate:
  `output/checks/serenity_server_t2i_product_gate_workflow_params_fallback.json`
  includes `zimage_browser_workflow_params_route_profile` with HTTP 200,
  `workflow_route_kind:"image"`, and
  `workflow_plan.source:"flat_params_adapter"`.
- Workflows tab submit gate:
  `output/checks/serenity_server_t2i_product_gate_workflows_tab_submit.json`
  includes `zimage_browser_workflow_image_route_profile` and
  `zimage_workflows_tab_image_route_profile` with HTTP 200,
  `workflow_route_kind:"image"`, `workflow_plan.source:"comfy_api_prompt_graph"`,
  and `SaveImage` terminals. The browser smoke captures the Workflows tab
  preflight/generate POST bodies and shows only `workflow` and
  `workflow_client` top-level keys.
- Ideogram bbox workflow submit gate:
  `output/checks/serenity_server_t2i_product_gate_ideogram_bbox_workflow_submit.json`
  includes `ideogram4_bbox_workflow_params_route_profile` with HTTP 200,
  `workflow_route_kind:"image"`, and
  `workflow_plan.source:"flat_params_adapter"`. The browser smoke injects an
  active Workflows-tab Ideogram bbox node and verifies the submitted
  `workflow.params.prompt_json.compositional_deconstruction.elements[0].bbox`
  remains `[120,180,760,820]`.
- Grid workflow submit gate:
  `output/checks/serenity_server_t2i_product_gate_grid_workflow_submit.json`
  includes `zimage_grid_workflow_params_success` with HTTP 200, one cell, and a
  real composite grid path under
  `output/checks/serenity_server_t2i_product_gate_grid_workflow_submit_run/`.
  The browser smoke captures Grid XYZ POST bodies with only `workflow`,
  `workflow_client`, `x_axis`, and `x_values` as top-level keys, and confirms
  `<random:...>` is resolved in `workflow.params.prompt` while the raw prompt is
  preserved in `workflow.params.prompt_raw`.
- Grid workflow capability gate:
  `output/checks/serenity_server_t2i_product_gate_grid_workflow_capability.json`
  includes `zimage_grid_workflow_img2img_capability_error` with HTTP 400,
  `rejection_stage:"workflow_capability"`, `workflow_route_kind:"image"`, and
  `workflow_plan.source:"flat_params_adapter"`. The same report also keeps
  `zimage_grid_workflow_params_success` passing with one grid artifact.

Live image preflight:

- `output/checks/workflow_image_preflight_route_live_http/report.json`
- Klein workflow lowered to `workflow_route_kind:"image"` with a `SaveImage`
  terminal and `backend:"flux2"`.

Live image generation:

- Request:
  `output/checks/klein_workflow_ir_lora_4step_working_lora_run/request.json`
- Report:
  `output/checks/klein_workflow_ir_lora_4step_working_lora_run/report.json`
- Output:
  `output/checks/klein_workflow_ir_lora_4step_working_lora_run/job-0001.png`
- Result: HTTP 200, `job-0001` done, valid 512x512 RGB PNG,
  `idat_sha256:d55ba7d752a52669b0376126acfd5238dfd4807034eff93720a5c21225c075c8`.
- PNG metadata preserves `workflow_executor:"serenity.workflow_graph.executor.v1"`,
  `workflow_route_kind:"image"`, and a `SaveImage` terminal plan.

This is a real workflow-IR generation proof, not a full production-quality claim
for Klein. The current blockers are timing/VRAM result-sidecar evidence and a
stronger runtime progress heartbeat.

## Browser Generate Evidence - 2026-06-16

These runs used the actual canvas Generate button, not flat direct posts. The
browser assembled a 7-node image workflow and submitted it as
`workflow_client:"serenity.canvas.generate_ws"`; Rust preflight and generate saw
the same workflow body.

- Z-Image UI workflow: `output/run_serenity_ui/job-0003.png`
  - 1024x1024 RGB PNG, `steps:8`, `seed:20260616`, `scheduler:"simple"`.
  - Verified real image output after removing earlier stub outputs.
- Klein 9B UI workflow: `output/run_serenity_ui/job-0004.png`
  - Browser model `flux-2-klein-base-9b_fp8_e4m3fn` normalized to backend
    `flux2`, 512x512, `scheduler:"simple"`, `steps:4`.
  - Rust worker driver swapped `zimage -> flux2`; `serenity_worker_klein` ran
    Qwen3 encode, streamed 32 Klein blocks, denoised 4/4 steps, and saved a real
    512x512 RGB PNG with `serenity.genparams.v1`.
  - This proves workflow/UI/worker dispatch. It is not a production-quality
    Klein image claim; the sample is low quality and Klein still needs stronger
    progress/timing/VRAM sidecar evidence.
- SDXL UI workflow alias gate: `output/run_serenity_ui/job-0005.png`
  - Browser model `sd_xl_base_1.0` normalized to backend `sdxl`, 1024x1024,
    `scheduler:"normal"`, `steps:4`.
  - The output was a patterned smoke artifact, so it is not quality evidence.
- SDXL UI workflow quality check: `output/run_serenity_ui/job-0006.png`
  - Same `sd_xl_base_1.0` Generate-screen path at 1024x1024, `steps:20`,
    `cfg:7`, `scheduler:"normal"`, `seed:2026061620`.
  - Rust worker driver ran `serenity_worker_sdxl`, CLIP-L/G encode, SDXL UNet,
    tiled VAE decode, result manifest
    `output/run_serenity_ui/job-0006.png.sdxl_daemon_result.json`, and saved a
    coherent 1024x1024 RGB PNG with `serenity.genparams.v1`.

Regression gates added/updated:

- `scripts/check_canvas_browser_controls.js` verifies browser capability
  normalization for Klein (`flux2`, 512x512, `simple`), SenseNova
  (`sensenova`, 1024x1024, 30 steps, `simple`), and `sd_xl_base_1.0`
  (`sdxl`, 1024x1024, `normal`).
- `scripts/check_canvas_preflight_submit_contract.py` pins the browser alias
  markers and workflow-only submit contract.
- Rust tests cover `sd_xl_base_1.0` in worker dispatch and prequeue admission.

## Verification

Primary gate:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_serenity_server_t2i_product_gate.py \
  --models "" \
  --server-bin serenity-server/target/debug/serenity-server \
  --worker-bin output/bin/serenity_worker_stub \
  --out-dir output/checks/serenity_server_t2i_product_gate_prequeue_after_workflow_plan_run \
  --write-report output/checks/serenity_server_t2i_product_gate_prequeue_after_workflow_plan.json \
  --poll-interval 1
```

The report must include:

- `/v1/capabilities` HTTP 200
- `coverage.ok:true`
- no `failed_capability_cases`
- no `failed_capability_rejection_cases`
- no `failed_preflight_capability_profile_cases`
- a populated `capability_rejections` matrix generated from the live
  `/v1/capabilities` payload; the current no-CUDA prequeue report contains 88
  fail-loud cases
- a populated `preflight_capability_profiles` set proving admitted and blocked
  request profiles from `/v1/preflight`, including workflow-lowering rejection
  profiles
- Z-Image Karras prequeue rejection with unchanged job count
- `/v1/samplers` support that matches the capability sampler subsets

Representative required generated rejection cases:

- `zimage_image_to_image_disabled`
- `zimage_controlnet_disabled`
- `zimage_bbox_prompt_json_disabled`
- `sdxl_vae_override_disabled`
- `sdxl_image_conditioning_disabled`
- `ideogram4_negative_prompt_disabled`
- `flux_multi_lora_disabled`
- `flux_outpaint_lowered_fields_disabled`

Representative required preflight profile cases:

- `zimage_admitted_profile`
- `qwen_blocked_profile`
- `klein_blocked_profile`
- `ideogram_negative_prompt_profile`
- `zimage_raw_controlnet_profile`
- `zimage_workflow_unsupported_node_profile`
- `zimage_browser_workflow_img2img_capability_profile`

Static/UI guards:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_canvas_preflight_submit_contract.py
NODE_PATH=${TMPDIR:-/tmp}/mojodiffusion-playwright-tools/node_modules \
  node scripts/check_canvas_browser_controls.js
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_swarmui_sampler_surface.py \
  --write-readiness output/checks/swarmui_sampler_surface_readiness.json
```

## Non-Claims

- `/v1/capabilities` does not prove sampler parity.
- A listed sampler/scheduler subset still needs artifact, timing, VRAM, and
  requested/executed metadata evidence before parity can be accepted.
- Disabled controls are not hidden readiness; they are unsupported product
  features with fail-loud admission.
- This route-plan work does not implement image-to-image. It keeps image-to-image
  from being hard-wired into the server or browser shape so a future
  workflow-specific path can be admitted by IR/operator behavior instead of UI
  assumptions.

## Current Real-Image Gate - 2026-06-16

Tests and preflight profiles are not the acceptance gate by themselves. Current
Generate-screen evidence includes real saved images inspected from disk:

- `output/run_serenity_ui/job-0016.png`: Z-Image `z_image_base_bf16`,
  browser Generate button, 1024x1024, 16 steps, `cfg:5`,
  `scheduler:"simple"`, seed `2026061653`. The image is a sharp red ceramic
  teapot on a wooden table with a blue cloth backdrop, matching the prompt and
  not a placeholder. The browser valid-image gate measured
  `avg_stddev:68.639`, `luminance_range:249.333`, `edge_energy:4.314`,
  `color_bins:1104`. `/v1/job/job-0016/result` includes both the server
  manifest and `job-0016.png.zimage_daemon_result.json`.
- `output/run_serenity_ui/job-0017.png`: SDXL `sd_xl_base_1.0`, browser
  Generate button, 1024x1024, 20 steps, `cfg:7`, `scheduler:"normal"`, seed
  `2026061654`. The image is a coherent woman holding a brass lantern in a dark
  green velvet dress. The worker driver swapped `zimage -> sdxl`. The browser
  valid-image gate measured `avg_stddev:52.72`, `luminance_range:254.667`,
  `edge_energy:2.963`, `color_bins:696`. `/v1/job/job-0017/result` includes
  both the server manifest and `job-0017.png.sdxl_daemon_result.json`.
- `output/run_serenity_ui/job-0018.png`: Klein/Flux2
  `flux-2-klein-base-9b_fp8_e4m3fn`, browser Generate button, workflow graph
  from `serenity.canvas.generate_ws`, 512x512, 4 steps, `cfg:4`,
  `scheduler:"simple"`, seed `2026061655`. The image is a coherent centered
  portrait in a dark coat. The worker driver swapped `sdxl -> flux2`. The
  browser valid-image gate measured `avg_stddev:27.285`,
  `luminance_range:212.333`, `edge_energy:1.992`, `color_bins:256`.
  `/v1/job/job-0018/result` includes the server manifest and workflow plan.
- `output/run_serenity_ui/job-0011.png`: SDXL `sd_xl_base_1.0`, browser
  Generate button, 1024x1024, 20 steps, `cfg:7`, `scheduler:"normal"`, seed
  `2026061631`. The image is a coherent portrait. The valid-image gate measured
  `avg_stddev:38.112`, `luminance_range:245.667`, `edge_energy:3.263`,
  `color_bins:314`. `/v1/job/job-0011/result` includes both the server manifest
  and `job-0011.png.sdxl_daemon_result.json`.
- `output/run_serenity_ui/job-0013.png`: Klein/Flux2
  `flux-2-klein-base-9b_fp8_e4m3fn`, browser Generate button, workflow graph
  from `serenity.canvas.generate_ws`, 512x512, 4 steps, `cfg:4`,
  `scheduler:"simple"`, seed `2026061642`. The image is a coherent portrait,
  not noise. Pixel stats on the saved PNG: `avg_stddev:25.769`,
  `luminance_range:219`, `edge_energy_find_edges_mean:7.743`,
  `sampled_color_bins:311`. `/v1/job/job-0013/result` includes the server
  manifest and workflow plan.

The browser real-image gate is `scripts/check_serenity_browser_generate_valid_image.js`.
It clicks `#btn-generate`, captures preflight/generate evidence, waits for the
job, decodes the saved PNG, and rejects placeholder/blank/flat outputs. It is
the gate to run when the question is whether Generate still produces usable
images. The edge-energy hard floor is `1.0`; edge detail is a validity guard,
not an aesthetic sharpness score.

UI/workflow generation evidence belongs under `output/run_serenity_ui`. The
Rust server default now uses that path, and the browser gate rejects returned
PNG paths outside that root unless `SERENITY_VALID_IMAGE_OUTPUT_DIR` is set
explicitly for a deliberate alternate evidence run. `output/checks` remains for
script reports and isolated checks, not the primary Generate gallery artifact.
`/v1/job/:id/result` now exposes `output_location.root_kind`,
`output_location.root`, `output_location.inside_root`, and
`output_location.relative_path`; server result sidecars also persist the same
location block under `output.location`. The lighter jobs and gallery endpoints
also attach the same `output_location` block when an output path exists, so
lists, single job records, gallery items, and result documents all identify
whether an artifact belongs to the UI workflow gallery root. `/v1/capabilities`
advertises the static `output_contract`, and `/v1/preflight` reports the actual
server `output_root` that `/v1/generate` will use for the submitted request.

The supported text-to-image matrix gate is
`scripts/check_serenity_supported_generate_matrix.js`. It runs the browser gate
sequentially for every `/v1/capabilities` backend admitted for text-to-image:
Z-Image, Ideogram4, SDXL, Anima, SD3, Flux.1-dev, Klein/Flux2, and SenseNova.
It then verifies the product result endpoint for each generated job. The matrix
now also requires
`/v1/job/:id/result.output_location.inside_root:true` and the matching server
sidecar `output.location.inside_root:true`, so a generated image outside
`output/run_serenity_ui` cannot be accepted as product UI evidence. The latest
report is
`output/checks/serenity_supported_generate_matrix_latest.json` with schema
`serenity.supported_generate_matrix.v1`. It passed on 2026-06-16 with:

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

SenseNova-U1 is now admitted by the Rust `/v1/generate` prequeue gate for
shape-dispatched 512x512 and 1024x1024 txt2img. It dispatches to
`output/bin/serenity_worker_sensenova`, uses
`/home/alex/.serenity/models/sensenova_u1`, rejects negative prompts, LoRA,
img2img/inpaint/masks, VAE overrides, variation noise, hires, and non-admitted
sizes before enqueue. The source worker now writes
`<png>.sensenova_daemon_result.json` plus the PNG `serenity.genparams.v1` tEXt
chunk, and the installed worker was rebuilt through the capped
`build-worker-sensenova-raw` path. Live workflow/product evidence:

- `output/run_serenity_ui/job-0037.png`: SenseNova-U1 `sensenova-u1`,
  workflow-lowered route `image` from `flat_params_adapter`, 1024x1024,
  30 steps, `cfg:4`, `scheduler:"simple"`, seed `2026061730`. Visually
  inspected as a coherent photo-like image of a woman carrying a miniature
  landscape on her back. The server visual-health gate measured
  `avg_stddev:52.07`, `luminance_range:249.0`, `edge_energy:3.293`, and
  `color_bins:712`. Both server and `sensenova` worker result manifests are
  present. Worker sidecar timings recorded `total_wall_seconds:2267.463`,
  `denoise_seconds:2065.749`, `denoise_seconds_per_step:68.858`,
  `text_encode_seconds:200.716`, and `peak_vram_mib:4559.75`.

- `output/run_serenity_ui/job-0034.png`: SenseNova-U1 `sensenova-u1`,
  browser Generate button, 512x512, 12 steps, `cfg:4`, `scheduler:"simple"`,
  seed `2026061668`. Visually inspected as a coherent painterly portrait of a
  woman bent under a load carrying a miniature landscape/tree scene on her back.
  The browser valid-image gate measured `avg_stddev:43.965`,
  `luminance_range:227.667`, `edge_energy:2.921`, and `color_bins:508`.
  Both server and `sensenova` worker result manifests are present. Worker
  sidecar timings recorded `total_wall_seconds:384.162`, `denoise_seconds:258.171`,
  `text_encode_seconds:101.816`, and `peak_vram_mib:4291.063`.

Microsoft Lens remains blocked from `/v1/generate` admission. The GPT-OSS MoE
scatter crash noted in the Lens worker has been fixed by casting expert
`down_out` to F32 for the internal gated scatter accumulation while returning
BF16 at the layer boundary, but Lens still lacks accepted render/OOM/parity
evidence and should remain fail-loud until that gate passes.

The Rust result API now exposes the same contract as product data. Every
`/v1/job/:id/result` response includes `visual_health` with schema
`serenity.visual_health.v1`: dimensions, RGB stddev, luminance range, edge
energy, color bins, thresholds, and explicit failure reasons. Future server
result manifests also persist this block when a job completes. Live evidence:

- `job-0016`: `visual_health.status:"pass"`, 1024x1024,
  `avg_stddev:68.639`, `luminance_range:249.333`, `edge_energy:4.314`,
  `color_bins:1104`.
- `job-0017`: `visual_health.status:"pass"`, 1024x1024,
  `avg_stddev:52.72`, `luminance_range:254.667`, `edge_energy:2.963`,
  `color_bins:696`.
- `job-0018`: `visual_health.status:"pass"`, 512x512,
  `avg_stddev:27.285`, `luminance_range:212.333`, `edge_energy:1.992`,
  `color_bins:256`.
- `job-0011`: `visual_health.status:"pass"`, 1024x1024,
  `avg_stddev:38.112`, `luminance_range:245.667`, `edge_energy:3.263`,
  `color_bins:314`.
- `job-0013`: `visual_health.status:"pass"`, 512x512,
  `avg_stddev:25.769`, `luminance_range:217.667`, `edge_energy:3.032`,
  `color_bins:441`.

Klein still has two limits:

- Progress does not stream per denoise step through `/v1/job`; the worker logs
  denoise steps but the browser and job API sit at `0/4` until completion.
- `serenitymojo/serve/klein_runtime_backend.mojo` now writes source-side
  `<png>.klein_daemon_result.json` manifests, but the installed worker was not
  rebuilt in this session because interactive Mojo builds are forbidden here.
  Existing Klein artifacts therefore have server-side result evidence but no
  worker sidecar until the capped worker build/install path runs.
