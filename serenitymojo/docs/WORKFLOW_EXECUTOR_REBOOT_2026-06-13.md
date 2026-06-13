# Workflow Executor Reboot Notes - 2026-06-13

This is the short restart file for the Serenity native workflow executor work.
A longer local handoff also exists at:

```text
serenitymojo/docs/HANDOFF_WORKFLOW_EXECUTOR_2026-06-13.md
```

That file is intentionally local because this repo ignores `*HANDOFF*.md`.

## North Star

Serenity should be one integrated app/runtime that is faster and cleaner than
SwarmUI plus ComfyUI together. ComfyUI and SwarmUI are import/oracle/reference
layers, not the product runtime. The product path is SerenityUI/MojoUI graph IR
plus a Mojo graph executor plus mojodiffusion model backends.

For LanPaint and inpaint work, the north star is the same: use the
LanPaint/Comfy/SerenityFlow Python behavior as the oracle, lower only the
understood workflow semantics into Mojo metadata or typed graph values, and keep
real generation inside mojodiffusion backends. Graph/daemon/IPC plumbing may
recognize inpaint-related shape and metadata before the backend is complete, but
it must not pretend that ordinary img2img or txt2img is mask-aware denoise.

## Key Files

- Backend executor: `serenitymojo/serve/workflow_graph.mojo`
- Daemon integration: `serenitymojo/serve/serenity_daemon.mojo`
- Product smoke: `scripts/check_workflow_graph_product_contract.py`
- Klein LoRA product smokes:
  `scripts/check_klein_lora_daemon_smoke.py`,
  `scripts/check_klein_lora_reference_daemon_smoke.py`
- Static surface gate: `scripts/check_workflow_node_surface.py`
- LanPaint oracle surface gate: `scripts/check_lanpaint_oracle_surface.py`
- LanPaint canvas product smoke: `scripts/check_lanpaint_canvas_daemon_smoke.py`
- Shared backend contract: `serenitymojo/serve/backend.mojo`
- Worker IPC contract: `serenitymojo/serve/ipc_codec.mojo`
- Image/mask I/O helpers: `serenitymojo/serve/image_io.mojo`
- Mask math substrate: `serenitymojo/sampling/inpaint.mojo`
- Klein daemon bridge: `serenitymojo/serve/klein_backend.mojo`
- Klein staged sampler and LoRA loader:
  `serenitymojo/sampling/klein_sample_cli.mojo`,
  `serenitymojo/sampling/klein_sampler.mojo`,
  `serenitymojo/models/klein/klein_stack_lora.mojo`
- Product readiness: `output/checks/workflow_graph_product_readiness.json`
- Klein LoRA readiness evidence:
  `output/checks/klein9b_lora_daemon_smoke.json`,
  `output/checks/klein9b_lora_reference_edit_daemon_smoke.json`
- LanPaint canvas readiness evidence:
  `output/checks/lanpaint_canvas_daemon_smoke.json`
- Static readiness: `output/checks/workflow_node_surface_readiness.json`

## Current Evidence

The daemon compiled with:

```bash
cd /home/alex/mojodiffusion
pixi run build-daemon
```

2026-06-13 Z-Image scheduler update: current Comfy `sgm_uniform` semantics are
now ported for the bounded Z-Image Euler/flow-match Euler, DPM++ 2M, `uni_pc`,
and `uni_pc_bh2` paths.
The oracle is SwarmUI's embedded Comfy checkout:
`normal_scheduler(sgm=True)` includes the max sigma endpoint, converts through
`ModelSamplingDiscreteFlow.sigma`, and appends one terminal `0.0`. The Mojo
helper records the same shifted flow values and the backend scales txt2img
initial noise by `sigmas[0]`. Comfy puts both `uni_pc` and `uni_pc_bh2` through
`DISCARD_PENULTIMATE_SIGMA_SAMPLERS` schedule prep before SigmaConvert UniPC
math, so the Z-Image port now uses that same wording for both variants while
leaving accepted sampler parity false.

Focused runtime evidence:

```bash
python3 scripts/check_zimage_daemon_product_contract.py \
  --daemon output/bin/serenity_daemon \
  --timeout 900 --steps 1 \
  --skip-dpmpp2m-smoke --skip-generic-unipc-smoke --skip-unipc-smoke \
  --skip-multi-image-smoke --skip-variation-smoke --skip-img2img-smoke \
  --skip-multi-lora-smoke \
  --write-readiness output/checks/zimage_unipc_sgm_uniform_product_readiness.json
```

Proof jobs:

- `job-0357`: unsupported sampler fail-loud smoke.
- `job-0358`: historical `uni_pc` + `sgm_uniform` rejection smoke,
  superseded by the bounded UniPC `sgm_uniform` port wording above.
- `job-0359`: baseline `euler` + `flowmatch`, manifest
  `schedule_source:"zimage_comfy_simple_sigmas"`.
- `job-0360`: `euler` + `sgm_uniform`, manifest
  `executed_scheduler:"sgm_uniform_flowmatch"`,
  `sigma_trace:[1.0,0.9477647,0.85859877,0.67192113,0.0]`,
  `txt2img_initial_noise_scale:1.0`, and
  `schedule_source:"zimage_comfy_sgm_uniform_sigmas"`.
- `job-0361`: unsupported sampler fail-loud smoke from the current focused
  UniPC `sgm_uniform` gate.
- `job-0362`: baseline `euler` + `flowmatch` artifact from the current focused
  gate.
- `job-0363`: `euler` + `sgm_uniform` artifact from the current focused gate.
- `job-0364`: `uni_pc` + `sgm_uniform`, manifest
  `executed_scheduler:"sgm_uniform_flowmatch"`,
  `sigma_trace:[1.0,0.96028626,0.90089285,0.8023738,0.0]`,
  `txt2img_initial_noise_scale:0.70710677`,
  `solver_variant:"bh1"`, `sigma_parameterization:"SigmaConvert"`,
  and
  `schedule_source:"zimage_comfy_sgm_uniform_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps"`.
- `job-0365`: `uni_pc_bh2` + `sgm_uniform`, same prepared sigma trace and
  initial/final scaling as `job-0364`, with `solver_variant:"bh2"` and
  `sigma_parameterization:"SigmaConvert"`.
- `job-0366`: focused baseline artifact for the rebuilt bh2 simple-flowmatch
  gate.
- `job-0367`: `uni_pc_bh2` + `flowmatch`, manifest
  `executed_scheduler:"simple_flowmatch"`,
  `sigma_trace:[1.0,0.923077,0.8181818,0.6666667,0.0]`,
  `txt2img_initial_noise_scale:0.70710677`,
  `solver_variant:"bh2"`, `solver_order:3`,
  `sigma_parameterization:"SigmaConvert"`, and
  `schedule_source:"zimage_comfy_simple_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps"`.

The workflow product smoke passed after expanding the product evidence from one
Z-Image template to the current supported SerenityFlow t2i template set, bounded
SerenityFlow Klein edit templates, a `SetLatentNoiseMask` metadata graph, and
the Ideogram4 v4 visual export:

```bash
python3 scripts/check_workflow_graph_product_contract.py \
  --daemon output/bin/serenity_daemon \
  --write-readiness output/checks/workflow_graph_product_readiness.json \
  --json
```

Proof jobs from the latest product smoke against `output/bin/serenity_daemon`:

- `job-0293`: native linked Serenity typed graph
- `job-0294`: `LoadImage -> VAEEncode -> KSampler` metadata plumbing under stub
- `job-0295`: `LoraLoaderModelOnly -> KSampler` metadata lowering under stub
- `job-0296`: `LoadImage` mask slot -> `SetLatentNoiseMask` -> `mask_image`
  metadata plumbing under stub
- `job-0297`: raw Comfy API prompt import
- `job-0298`: SerenityFlow `zimage_t2i.json`
- `job-0299`: SerenityFlow `qwen_image_t2i.json`
- `job-0300`: SerenityFlow `klein9b_t2i.json`
- `job-0301`: SerenityFlow `klein4b_t2i.json`
- `job-0302`: SerenityFlow `flux2_dev_t2i.json`
- `job-0303`: SerenityFlow `klein9b_edit.json`
- `job-0304`: SerenityFlow `klein4b_edit.json`
- `job-0305`: Ideogram4 v4 visual Comfy export
  `/home/alex/Downloads/ideogram4_basic_txt2img_workflow_by_AI_Characters_v4.json`

The SerenityFlow jobs proved model/prompt/negative/size/seed/steps/cfg/sampler/
scheduler/sigma_shift metadata from real API-prompt graphs. The mask graph
proved `mask_image=/tmp/serenity_graph_mask.png` survives graph lowering,
daemon parse, PNG genparams, and readiness JSON. The Ideogram4 job proved the
bounded visual-export importer extracts the v4 txt2img subgraph:
`ideogram-4-fp8`, `1024x1024`, seed override `424242`, `euler`/`simple`, 48
quality steps, `sigma_shift=5`, `cfg=7`, and CFGOverride metadata.

Do not add Ideogram3 as a product target. Any `v3` in the Ideogram workflow
filenames is a workflow revision, not an Ideogram 3 model family.

The LanPaint canvas product smoke passed against the stub daemon:

```bash
python3 scripts/check_lanpaint_canvas_daemon_smoke.py \
  --daemon output/bin/serenity_daemon \
  --write-report output/checks/lanpaint_canvas_daemon_smoke.json
```

It posted `/home/alex/LanPaint/example_workflows/SDXL_Inpaint.json` as a Comfy
UI visual canvas graph and completed `job-0311`. The report proves graph import,
typed lowering, daemon parse, PNG genparams, and readiness JSON carry
`workflow_source="comfy_ui_canvas_graph"`, `lanpaint_num_steps=5`,
`lanpaint_prompt_mode="Image First"`, `lanpaint_mask_blend_overlap=9`,
`mask_image`, and `init_image`. This is a no-heavy metadata/product-path smoke;
it does not prove real mask-aware denoise, mask blend execution, inpaint image
quality, or full LanPaint parity.

The first Z-Image `SetLatentNoiseMask` img2img runtime slice is now present. The
bounded Z-Image `LanPaint_MaskBlend` final decoded pixel slice is also present:
it max-pools the image-space mask, Gaussian smooths it with
`sigma=(blend_overlap-1)/4`, then composites
`image1*(1-mask)+image2*mask`. For the LanPaint `ImageScale(area)` ->
`MaskBlend.image1` role, the base/original image is resized to the decoded
output size with Comfy/PyTorch `area` semantics before the blend. This does not
change the full LanPaint sampler boundary below: the sampler inner-loop remains
fenced by `reject_unsupported_lanpaint_sampler_params(...)`, and backends
without the Z-Image split still reject unsupported LanPaint metadata through
`reject_unsupported_lanpaint_params(...)`.

Validation passed:

```bash
pixi run mojo run -I . -I /home/alex/MOJO-libs serenitymojo/serve/image_io_mask_smoke.mojo
pixi run build-daemon
python3 scripts/check_lanpaint_oracle_surface.py --write-report output/checks/lanpaint_oracle_surface.json
python3 scripts/check_workflow_node_surface.py --write-readiness output/checks/workflow_node_surface_readiness.json
```

Static surface gate command:

```bash
python3 scripts/check_workflow_node_surface.py \
  --write-readiness output/checks/workflow_node_surface_readiness.json

pixi run mojo run -I . serenitymojo/sampling/parity/inpaint_parity.mojo
pixi run mojo run -I . serenitymojo/sampling/parity/img2img_refpack_parity.mojo
pixi run mojo run -I . serenitymojo/sampling/parity/klein_reference_latent_bridge_smoke.mojo
```

Current static gate coverage:

- `arbitrary_comfy_swarm_graph_execution_ready = false`
- source checks cover the Comfy API prompt adapter, Comfy UI visual canvas
  adapter, bounded typed image graph allowlist, mask source metadata,
  `image_io.mojo` mask helpers, bounded LanPaint final-pixel blend helpers,
  LanPaint field lowering, explicit `lanpaint_*` `JobParams`/daemon/IPC fields,
  Z-Image preserve-mask application, Z-Image final decoded `LanPaint_MaskBlend`,
  the LanPaint `ImageScale(area)` -> `MaskBlend.image1` base-image resize
  oracle, and backend fail-loud rejection for unsupported LanPaint sampler
  fields.
- readiness checks read `output/checks/lanpaint_canvas_daemon_smoke.json` and
  expect the `job-0311` LanPaint canvas metadata smoke.
- `inpaint_parity.mojo` passes mask-blend and LanPaint overdamped-step tensor
  parity against the synthetic oracle.
- `img2img_refpack_parity.mojo` passes the raw Klein edit pack/id convention:
  target/noise ids first, reference ids second, and reference `t=10`.
- `klein_reference_latent_bridge_smoke.mojo` passes 512 and 1024 no-heavy
  bridge plans:
  - 512: target/ref tokens `[1,1024,128]`, combined `[1,2048,128]`,
    combined ids `[2048,4]`, edit sequence `512 + 2048 = 2560`
  - 1024: target/ref tokens `[1,4096,128]`, combined `[1,8192,128]`,
    combined ids `[8192,4]`, edit sequence `512 + 8192 = 8704`

## Supported Executor Slice

Currently accepted graph nodes:

- loaders: `CheckpointLoaderSimple`, `UNETLoader`, `DiffusionModelLoader`,
  `LoraLoaderModelOnly`, `CLIPLoader`, `DualCLIPLoader`, `TripleCLIPLoader`,
  `VAELoader`
- conditioning: `CLIPTextEncode`, `CLIPTextEncodeFlux`, `ConditioningZeroOut`,
  `FluxGuidance`
- latent/image: `LoadImage`, `EmptyLatentImage`, `EmptySD3LatentImage`,
  `EmptyFlux2LatentImage`, `ImageToMask`, `MaskToImage`, `VAEEncode`,
  `SetLatentNoiseMask`, `GetImageSize`, `ImageScale`,
  `ImageScaleToTotalPixels`, `ReferenceLatent`, `VAEDecode`
- embedded subgraph wrappers: `6007e698-2ebd-4917-84d8-299b35d7b7ab`,
  `f07d2d08-2bc5-4dd8-a9f0-f2347c6b5cca`
- model/sampler/sink: `ModelSamplingAuraFlow`, `ModelSamplingSD3`,
  `DifferentialDiffusion`, `KSampler`, `LanPaint_KSampler`,
  `LanPaint_KSamplerAdvanced`, `CFGGuider`, `BasicGuider`, `Flux2Scheduler`,
  `BasicScheduler`, `RandomNoise`, `KSamplerSelect`, `SamplerCustomAdvanced`,
  `LanPaint_SamplerCustomAdvanced`, `LanPaint_MaskBlend`, `SaveImage`,
  `PreviewImage`, `MarkdownNote`, `Note`

Do not overclaim this as full graph parity. `SetLatentNoiseMask` now carries
path-backed mask metadata plus mask source metadata: `LoadImage` exposes a typed
`MASK` slot with source `load_image_mask` for Comfy's inverted-alpha mask, while
`ImageToMask` red/green/blue channels remain raw RGB values with no
thresholding. `ImageToMask(alpha)` fails loud from graph lowering because Comfy
`LoadImage.IMAGE` is RGB-only; use `LoadImage.MASK` for alpha-derived masks.
`SetLatentNoiseMask` can attach that mask to a `LATENT`, and
`KSampler`/`SamplerCustomAdvanced` can lower it into `JobParams.mask_image` plus
`lanpaint_mask_channel`.

Z-Image consumes that mask only for the bounded img2img preserve-region slice:
it stores the encoded init latent, seeded noise, and latent preserve mask, then
applies preserve-region blending after every sampler update using `sigma_next`.
Result metadata records `mask_image`, `mask_channel`, `inpaint_mask_applied`,
`inpaint_preserve_active_pixels`, and `inpaint_preserve_mean`.

The visual Comfy UI canvas adapter (`looks_like_comfy_ui_canvas_graph`,
`comfy_ui_canvas_to_typed_graph`, `apply_comfy_ui_canvas_graph`) can lower
LanPaint sampler, mask-conversion, and mask-blend nodes into flat
`lanpaint_*`, `mask_image`, and `init_image` metadata. Full LanPaint sampler
inner-loop semantics are still not wired into a backend. Z-Image consumes only
the bounded final decoded `LanPaint_MaskBlend` pixel blend, with Comfy/PyTorch
`area` resize limited to the base/original `MaskBlend.image1` role; Qwen,
Ideogram4, and Klein must continue to fail loud for unsupported `lanpaint_*`
runtime metadata.
`LoraLoaderModelOnly` only lowers model-side LoRA metadata into the flat request
contract; full `LoraLoader`, CLIP-side LoRA, LoRA stacks, KJNodes utilities,
ControlNet, IPAdapter, arbitrary custom Comfy nodes, and unsupported LanPaint
utility nodes such as `ImagePadForOutpaint` are still unsupported and should
fail loud.

## Important Non-Claims

- Z-Image plain t2i template import works through stub product smoke.
- `LoadImage/VAEEncode` metadata plumbing works under stub.
- `SetLatentNoiseMask` metadata plumbing works under stub and Z-Image has a
  bounded img2img preserve-mask runtime path. This is not full inpaint,
  arbitrary mask tensor, or LanPaint parity.
- `LanPaint_KSampler`, `LanPaint_KSamplerAdvanced`,
  `LanPaint_SamplerCustomAdvanced`, `LanPaint_MaskBlend`, `ImageToMask`, and
  `MaskToImage` are accepted graph-lowering nodes now. They carry metadata and
  path-backed image/mask handles; LanPaint sampler nodes still do not execute
  LanPaint denoise semantics in a real backend. Z-Image `LanPaint_MaskBlend` is
  bounded to final decoded pixel compositing only, with area resize only for
  the `MaskBlend.image1` base/original-image role, not sampler parity.
- This is not accepted general Z-Image i2i parity.
- This is not accepted Z-Image/Qwen/Klein/Ideogram inpaint parity.
- Z-Image init-image encode code may be useful substrate for refiner/LanPaint/
  inpaint work; do not delete it based on the i2i confusion.
- Klein/Flux2 edit templates are the next real edit/i2i target.

## Klein 9B AI Toolkit LoRA Txt2Img Evidence

User prompt tested:

```text
srx_ottoman, desert caravan approaching an ancient city, camel riders and walking travelers in traditional robes and turbans, rocky desert road, fortified historical city in the distance, domes and a tall minaret, warm sunlight, dusty air, epic cinematic historical mood, highly detailed
```

LoRA:

```text
/home/alex/Downloads/flux2_klein_9b_imperial_historical_lora.safetensors
```

What changed:

1. `KleinBackend` now accepts exactly one LoRA at weight `1.0`, resolves bare
   names through `LORAS_DIR`, and passes the resolved path to the existing
   `klein_sample_cli` `lora_path` argv instead of `base`.
2. `klein_stack_lora.mojo` now recognizes the AI Toolkit/Comfy Flux2-Klein key
   layout in that file:
   `diffusion_model.double_blocks.{i}.{img,txt}_attn.qkv/proj`,
   `{img,txt}_mlp.{0,2}`, and
   `diffusion_model.single_blocks.{i}.linear1/linear2`.
3. The loader splits fused `qkv` LoRA `B` rows into live q/k/v adapters, maps
   single `linear1` to the full `to_qkv_mlp` slot and `linear2` to the full
   `to_out` slot, and rejects shape-incompatible files before sampling.

Evidence:

- Repeatable checker:
  `python3 scripts/check_klein_lora_daemon_smoke.py --write-report output/checks/klein9b_lora_daemon_smoke.json`.
- Checker report: `output/checks/klein9b_lora_daemon_smoke.json` with
  `ready:true`, `blockers:[]`, and log markers proving the LoRA sample argv,
  AI Toolkit loader count, and staged sample completion.
- Daemon job: `job-0308`, state `done`, `step=4`, `total=4`.
- Output PNG: `output/serenity_daemon/job-0308.png`, 512x512, IDAT sha256
  `8f69b85173efbc80fa690f15dd1bfa75c9d16e596b670c854f75d72a9c4597ed`.
- Manifest:
  `output/serenity_daemon/job-0308.png.klein_daemon_result.json` records
  `lora_count:1`, the resolved `lora_path`, and `lora_weight:1.0`.
- PNG `serenity.genparams.v1` records the exact prompt, seed `424242`,
  `steps:4`, `cfg:3.5`, `sampler:"euler"`, `scheduler:"simple"`, and the LoRA
  array with weight `1.0`.
- Daemon log `output/checks/klein9b_lora_daemon_smoke_38373.log` shows:
  `klein_sample_cli ... /home/alex/Downloads/flux2_klein_9b_imperial_historical_lora.safetensors ...`
  and `[klein][lora] loaded Flux2/Klein double_blocks adapters: 144`.
- The checker records a nonblank 512x512 PNG artifact for this prompt/LoRA
  smoke. This is execution evidence, not image-quality or content-parity
  evidence.
- Static readiness now verifies this report through
  `check_klein_lora_daemon_smoke_report`.

Limits:

- This proves bounded Klein 9B txt2img single-LoRA execution through the product
  daemon path. It does not prove multi-LoRA, non-1.0 LoRA weights, CLIP-side
  LoRA, LoRA edit routing, arbitrary Comfy graph parity, or exact OneTrainer /
  SerenityFlow trajectory parity.
- The Python/SerenityFlow oracle path represents Klein LoRA as a single
  `--lora` path and has no `--lora-weight`; keep weight support fail-loud until
  the oracle exposes a comparable control.

## Klein 9B Edit-LoRA Graph Evidence

The SerenityFlow/Comfy workflow
`/home/alex/serenityflow-v2/serenityflow/workflows/klein9b_edit_lora.json`
was run through the daemon as a bounded 512x512/1-step product smoke after
patching its placeholder `LoraLoaderModelOnly.lora_name` to:

```text
/home/alex/Downloads/flux2_klein_9b_imperial_historical_lora.safetensors
```

Repeatable checker:

```bash
python3 scripts/check_klein_lora_reference_daemon_smoke.py \
  --write-report output/checks/klein9b_lora_reference_edit_daemon_smoke.json
```

Evidence:

- Checker report:
  `output/checks/klein9b_lora_reference_edit_daemon_smoke.json` with
  `ready:true`, `blockers:[]`, `patched_lora_nodes:1`,
  `workflow_node_count:19`, and `workflow_edge_count:22`.
- Daemon job: `job-0309`, state `done`, `step=1`, `total=1`.
- Output PNG: `output/serenity_daemon/job-0309.png`, 512x512, IDAT sha256
  `17cce2f5ea03eb15c79c6f75f2ffeb90f70e412f51ef09edc6d87c05c495fca1`.
- Manifest:
  `output/serenity_daemon/job-0309.png.klein_daemon_result.json` records
  `mode:"reference_latent_edit"`, `lora_count:1`, resolved `lora_path`,
  `lora_weight:1.0`, `reference_image:job-0141.png`,
  `reference_latent_count:2`, `edit_denoise:0.45`, `edit_shift:2.02`, and
  `reference_t_offset:10.0`.
- PNG `serenity.genparams.v1` records
  `workflow_source:"comfy_api_prompt_graph"`, `workflow_node_count:19`,
  `workflow_edge_count:22`, prompt `"change the dress to blue"`, scheduler
  `"flux2"`, the LoRA array, and `reference_latent_count:2`.
- Daemon log `output/checks/klein9b_lora_reference_edit_daemon_smoke_56355.log`
  shows one `klein_sample_cli` command containing both the LoRA path and the
  edit tail `- <reference_image> 0.45 2.02 10.0`, plus
  `ReferenceLatent edit`, `[klein][lora] loaded Flux2/Klein double_blocks adapters: 144`,
  `[Klein-edit] denoise step 1/1`, and `DONE staged sample`.
- Static readiness verifies this report through
  `check_klein_lora_reference_daemon_smoke_report`.

This proves the bounded SerenityFlow/Comfy Klein edit-LoRA graph lowers through
`LoraLoaderModelOnly` and `ReferenceLatent` into one real Klein daemon sampler
invocation. It does not prove full-quality multi-step edit aesthetics, exact
SerenityFlow trajectory parity, multi-LoRA, non-1.0 LoRA weights, CLIP-side
LoRA, or arbitrary Comfy graph execution.

## Mask / Inpaint Metadata Update

Current bounded contract:

1. `JobParams` carries `mask_image`.
2. `/v1/generate` parses `mask_image` and persists it in
   `serenity.genparams.v1`.
3. Worker IPC encodes/decodes `mask_image`.
4. Comfy API prompt slot lowering maps `LoadImage` output slot 1 to typed
   `MASK` with source `load_image_mask`, matching Comfy's `LoadImage` inverted
   alpha mask.
5. `ImageToMask` red/green/blue channels remain raw RGB channel values with no
   thresholding. `ImageToMask(alpha)` fails loud in the graph importer because
   Comfy `LoadImage.IMAGE` is RGB-only; alpha-derived masks must use
   `LoadImage.MASK`. `MaskToImage -> ImageToMask(red)` preserves mask source
   metadata instead of collapsing the chain into a bare path.
6. `SetLatentNoiseMask(samples: LATENT, mask: MASK) -> LATENT` propagates the
   source latent plus mask path to downstream `KSampler` or
   `SamplerCustomAdvanced`.
7. Comfy UI visual canvas lowering maps widgets and links through
   `comfy_ui_canvas_to_typed_graph`, including `ImageToMask`, `MaskToImage`,
   `LanPaint_KSampler`, `LanPaint_KSamplerAdvanced`,
   `LanPaint_SamplerCustomAdvanced`, and `LanPaint_MaskBlend`.
8. `JobParams`, daemon genparams, and worker IPC carry explicit `lanpaint_*`
   fields: mask channel/blend overlap, LanPaint step count, lambda, step size,
   beta, friction, prompt/inpainting modes, add-noise controls, step bounds,
   leftover-noise behavior, early stop, inner threshold, and inner patience.
9. Shared mask helpers in `serenitymojo/serve/image_io.mojo` are
   `decode_comfy_mask`, `resize_mask_bilinear`,
   `load_comfy_latent_preserve_mask`, `resize_mask_nearest_exact`,
   `binarize_lanpaint_denoise_mask`, and
   `load_lanpaint_latent_preserve_mask`, plus the bounded final-pixel
   `LanPaint_MaskBlend` helpers `smooth_lanpaint_blend_mask`,
   `load_lanpaint_pixel_blend_mask`, and
   `apply_lanpaint_mask_blend_signed_chw`. Z-Image uses the Comfy soft bilinear
   path for `SetLatentNoiseMask`; the LanPaint blend mask uses nearest-exact
   image resize, max-pool, and Gaussian smoothing with
   `sigma=(blend_overlap-1)/4`; the LanPaint base/original image uses
   Comfy/PyTorch `area` resize only for the `ImageScale(area)` ->
   `MaskBlend.image1` role.
10. Z-Image consumes `mask_image` for the bounded `SetLatentNoiseMask` img2img
    slice by storing encoded init latent, seeded noise, and the latent preserve
    mask, then applying the preserve blend after each sampler update using
    `sigma_next`.
11. Z-Image consumes `lanpaint_mask_blend_overlap` only for the bounded final
    decoded `LanPaint_MaskBlend` pixel slice: it decodes the base/original
    `init_image`, area-resizes it to the decoded output size when needed,
    loads/smooths the image-space mask, and writes manifest fields
    `lanpaint_mask_blend_applied`, `lanpaint_mask_blend_overlap`, and
    `lanpaint_mask_blend_mean`. This is the narrow LanPaint
    `ImageScale(area)` -> `MaskBlend.image1` role only, not arbitrary graph-side
    image resize execution.
12. Qwen, Ideogram4, and Klein still call
    `reject_unsupported_mask_image_params(...)` for mask runtime metadata.
13. Z-Image calls `reject_unsupported_lanpaint_sampler_params(...)` for full
    LanPaint sampler-loop fields. Qwen, Ideogram4, and Klein call
    `reject_unsupported_lanpaint_params(...)` for unsupported LanPaint metadata.

This keeps unsupported inpaint/LanPaint workflows from silently degrading into
plain img2img or txt2img while allowing the narrow Z-Image preserve-mask and
final-pixel MaskBlend slices.

Generic scheduler update: `BasicScheduler -> SamplerCustomAdvanced` now carries
Comfy scheduler, step, and denoise metadata through the `SIGMAS` handle and
maps it into flat `scheduler`, `steps`, and `creativity` only when the connected
sampler consumes that handle. The product smoke records this separately from
the older Flux2Scheduler/Klein path.

Current LanPaint boundary:

- `scripts/check_lanpaint_oracle_surface.py` preserves the oracle shape from
  `/home/alex/LanPaint/src/LanPaint/nodes.py`, representative LanPaint workflow
  exports, and SerenityFlow's `SetLatentNoiseMask`.
- `serenitymojo/sampling/inpaint.mojo` plus
  `serenitymojo/sampling/parity/inpaint_parity.mojo` cover only the weight-free
  mask blend helper and one supplied-score overdamped LanPaint step. This is a
  math substrate, not a backend integration.
- `LanPaint_KSampler`, `LanPaint_SamplerCustomAdvanced`,
  `LanPaint_KSamplerAdvanced`, `LanPaint_MaskBlend`, `ImageToMask`, and
  `MaskToImage` are supported only for graph lowering and metadata propagation.
  The sampler nodes are not accepted as real LanPaint backend execution
  semantics. Z-Image `LanPaint_MaskBlend` is accepted only as final decoded
  pixel blending: max-pool, Gaussian smooth with
  `sigma=(blend_overlap-1)/4`, then `image1*(1-mask)+image2*mask`; base-image
  resize is limited to Comfy/PyTorch `area` semantics for the
  `ImageScale(area)` -> `MaskBlend.image1` role.
- Acceptance for real LanPaint/inpaint execution requires backend-specific
  mask-aware denoise, outpaint behavior where applicable, and parity checks
  against the Python oracle. Until then, any request that reaches a real backend
  with LanPaint sampler runtime metadata must fail loud; the Z-Image
  `SetLatentNoiseMask` path is only a preserve-mask img2img slice and the
  Z-Image `LanPaint_MaskBlend` path is only final-pixel compositing, not
  `LanPaint_KSampler` runtime parity.

## Oracle Search Findings

LanPaint oracles found under `/home/alex/LanPaint/example_workflows`:

- direct port candidates: `Qwen_Image_Inpaint.json`, `Qwen_Image_Outpaint.json`,
  `Z_image_Inpaint.json`, `SDXL_Inpaint.json`
- complex but high-value: `Flux2_Klein_inpainting.json`,
  `Flux.2.Dev_Inpaint.json`
- node classes: `LanPaint_KSampler`, `LanPaint_KSamplerAdvanced`,
  `LanPaint_SamplerCustomAdvanced`, `LanPaint_MaskBlend`,
  `SetLatentNoiseMask`, `ImagePadForOutpaint`, `ReferenceLatent`

Klein edit oracles:

- `/home/alex/serenityflow-v2/serenityflow/workflows/klein9b_edit.json`
- `/home/alex/serenityflow-v2/serenityflow/workflows/klein4b_edit.json`

Those are the cleanest next runtime targets because the graph importer already
recognizes `ReferenceLatent`, `VAEEncode`, `LoadImage`,
`EmptyFlux2LatentImage`, `Flux2Scheduler`, `BasicScheduler`, `KSamplerSelect`, and
`SamplerCustomAdvanced`.

Runtime gap for Klein edit has moved from "no execution route" to bounded
real-weight direct and daemon smoke evidence: the staged sampler now has a
compiled ReferenceLatent edit branch that decodes/resizes the source image,
VAE-encodes it, packs reference tokens, builds edit-specific RoPE for
target+reference ids, updates only target tokens during sampling, and decodes
only the target latent. This is still not accepted image-quality, pixel, or
trajectory parity.

## Klein ReferenceLatent Bridge Update

New no-heavy bridge files:

- `serenitymojo/sampling/klein_reference_latent_bridge.mojo`
- `serenitymojo/sampling/parity/klein_reference_latent_bridge_smoke.mojo`

The bridge now validates daemon `JobParams` for the bounded SerenityFlow Klein
edit shape:

- `reference_image` required
- `reference_latent_count > 0`
- `reference_latent_method == "index"` when present
- `init_image` must match `reference_image` if both are present
- size is currently 512x512 or 1024x1024

It then computes the reusable runtime plan:

- `latent_h = height / 16`, `latent_w = width / 16`
- `target_tokens = latent_h * latent_w`
- `reference_tokens = target_tokens`
- `combined_image_tokens = target_tokens + reference_tokens`
- `text_tokens = 512`
- `edit_sequence_tokens = text_tokens + combined_image_tokens`
- reference token start is `target_tokens`
- reference id offset is `t=10.0`

It also exposes builders for:

- `build_klein_reference_combined_img_ids(plan, ctx)` -> `[2*N,4]`
- `build_klein_reference_combined_tokens(noise_latent, reference_latent, plan, ctx)`
  -> `[1,2*N,128]`

This bridge removes the shape/id ambiguity that the real sampler wrapper now
uses. It remains a no-heavy gate; pixel parity still requires the staged edit
runner to be exercised with real weights.

## Klein ReferenceLatent Execution Bridge Update

New compiled runtime route:

- `serenitymojo/sampling/klein_sampler.mojo`
  - adds `_rope_host_reference[...]` for the oracle edit ids:
    target `[0,row,col,0]`, reference `[10,row,col,0]`, text `[0,0,0,tok]`
  - adds `_reference_latent_tokens[...]` to pack encoded
    `[1,128,LH,LW]` latents to the existing LoRA stack's `[N,128]` token shape
  - adds `_denoise_lora_reference_from_initial[...]`
    - concatenates `[target_tokens ; reference_tokens]` every step
    - forwards with `N_IMG=2*target_tokens`
    - slices prediction back to target tokens only
    - keeps reference tokens fixed
    - uses `build_flux2_img2img_sigmas(..., shift=2.02, denoise=creativity)`
  - adds public `klein_sample_with_reference_latent[...]`
- `serenitymojo/sampling/klein_sample_cli.mojo`
  - keeps existing txt2img and initial-noise replay paths
  - adds optional argv:
    `argv[7]=reference_image`, `argv[8]=denoise`, `argv[9]=shift`,
    `argv[10]=reference_t_offset`
  - decodes PNG/JPEG/WebP with `decode_image_any`, resizes with
    `resize_bilinear`, converts to signed RGB NCHW, and encodes with
    `KleinVaeEncoder[512,512]` or `[1024,1024]`
  - dispatches 4B/9B edit specializations for 512 and 1024
- `serenitymojo/serve/klein_backend.mojo`
  - validates ReferenceLatent requests with `plan_klein_reference_latent_bridge`
  - routes bounded ReferenceLatent jobs to the staged sampler CLI instead of
    rejecting before model load
  - writes `mode:"reference_latent_edit"` and edit metadata into the Klein
    daemon result manifest

This is not yet accepted Klein edit pixel parity. It is a compiled backend path
that now reaches the existing inference stack instead of a metadata-only stub.

Live real-weight direct CLI smoke evidence:

- Built binaries used:
  - `output/bin/klein_sample_cli`
  - `output/bin/serenity_daemon`
- Cap-cache was generated for
  `output/checks/klein4b_reference_edit_smoke.json` with variant `4b`.
- First command used `/home/alex/serenityflow-v2/input/input.png` at
  denoise `1.0` and wrote
  `output/checks/klein4b_reference_edit_smoke.png`.
  That source image is all-black RGBA, so the dark 512x512 output is not useful
  quality evidence.
- Second command used `output/serenity_daemon/job-0268.png` at denoise `0.45`
  and wrote
  `output/checks/klein4b_reference_edit_smoke_job0268_d045.png`.
  Mean abs diff from the 512x512 source was `[13.349, 5.089, 5.372]`.
- Third command used `output/serenity_daemon/job-0141.png` at denoise `0.45`
  and wrote
  `output/checks/klein4b_reference_edit_smoke_job0141_d045.png`.
  Mean abs diff from the resized source was `[5.6, 6.013, 8.634]`.
- The meaningful-reference direct smoke logged:
  `Klein-edit denoise step 1/1 | sigma 0.8829 | 1.8s/step`.

This proves the staged sampler can VAE-encode a reference image, append fixed
ReferenceLatent tokens, denoise target tokens, decode a valid 512x512 PNG, and
return GPU memory to baseline. It does not prove daemon HTTP execution,
multi-step aesthetics, Swarm/Comfy pixel parity, or accepted speed/VRAM parity.

Live real-weight daemon smoke evidence:

- Command runner: temporary Python harness started
  `output/bin/serenity_daemon dispatch <free-port>`.
- Request: SerenityFlow `klein4b_edit.json` as a Comfy API prompt graph, with
  top-level bounds `width=512`, `height=512`, `steps=1`, `creativity=0.45`,
  `seed=42`, and reference/init image
  `output/serenity_daemon/job-0141.png`.
- Daemon job: `job-0278`, state `done`, `step=1`, `total=1`.
- Output PNG: `output/serenity_daemon/job-0278.png`, 512x512 RGB.
- Embedded `serenity.genparams.v1` proves:
  - `model="flux2-klein-4b.safetensors"`
  - `prompt="change the dress to blue"`
  - `scheduler="flux2"`, `sampler="euler"`
  - `reference_image="/home/alex/mojodiffusion/output/serenity_daemon/job-0141.png"`
  - `reference_latent_method="index"`, `reference_latent_count=2`
  - `workflow_source="comfy_api_prompt_graph"`
  - `workflow_node_count=18`, `workflow_edge_count=21`
- Klein result manifest:
  `output/serenity_daemon/job-0278.png.klein_daemon_result.json`
  with `mode:"reference_latent_edit"`, `variant:"4b"`,
  `edit_denoise:0.45`, `edit_shift:2.02`, `reference_t_offset:10.0`,
  and sidecar binaries:
  `output/bin/klein_precache_sample_prompts`,
  `output/bin/klein_sample_cli`.
- Artifact stats vs resized `job-0141` source:
  mean abs diff `[8.849, 7.451, 6.928]`.
- Daemon log:
  `output/checks/klein4b_reference_edit_daemon_<port>.log` shows:
  cap-cache precache wrote `caps_pos.bin` and `caps_neg.bin`, sample command
  included `'-' '<reference_image>' '0.45' '2.02' '10.0'`, the sampler logged
  `Klein-edit denoise step 1/1 | sigma 0.8829 | 1.8s/step`, and the daemon
  exited cleanly.
- Machine-readable report:
  `output/checks/klein4b_reference_edit_daemon_smoke.json`.
- Repro runner:
  `python3 scripts/check_klein_reference_daemon_smoke.py --case 4b`
  starts `serenity_daemon dispatch`, posts the bounded Klein 4B edit workflow,
  verifies PNG genparams and the Klein daemon manifest, and rewrites the report.
- The same runner now supports
  `python3 scripts/check_klein_reference_daemon_smoke.py --case 9b`.
- 9B daemon job: `job-0279`, state `done`, `step=1`, `total=1`.
- 9B output PNG: `output/serenity_daemon/job-0279.png`, 512x512 RGB.
- 9B report:
  `output/checks/klein9b_reference_edit_daemon_smoke.json`,
  `ready:true`, `blockers:[]`, `case:"9b"`.
- 9B manifest:
  `output/serenity_daemon/job-0279.png.klein_daemon_result.json`
  with `mode:"reference_latent_edit"`, `variant:"9b"`,
  `model:"flux2-klein-9b.safetensors"`,
  `config_path:"/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json"`,
  `edit_denoise:0.45`, `edit_shift:2.02`, `reference_t_offset:10.0`.
- 9B artifact stats vs resized `job-0141` source:
  mean abs diff `[7.964, 6.467, 7.203]`.
- 9B daemon log shows:
  Qwen3-8B cap-cache precache wrote `caps_pos.bin`/`caps_neg.bin`,
  the sample command used `klein9b.json`, and the sampler logged
  `Klein-edit denoise step 1/1 | sigma 0.8829 | 3.4s/step`.
- Readiness checker now verifies this report:
  `scripts/check_workflow_node_surface.py` includes
  `check_klein_reference_daemon_smoke_report` for both 4B and 9B.

This closes the bounded real daemon execution gap for Klein 4B and 9B
ReferenceLatent edit. It still does not prove multi-step aesthetics,
LoRA edit routing, exact SerenityFlow/Comfy/Swarm trajectory parity,
LanPaint/inpaint semantics, or arbitrary Comfy graph execution.

Oracle direction after the daemon smokes:

- Do not use Rust as the Klein edit oracle for this work item.
- Use SerenityFlow's Python Comfy-compatible path as the oracle:
  `/home/alex/serenityflow-v2/serenityflow/nodes/latent.py`
  `VAEEncode`, `/home/alex/serenityflow-v2/serenityflow/nodes/conditioning_extra.py`
  `ReferenceLatent`, and
  `/home/alex/serenityflow-v2/serenityflow/bridge/sampling.py`.
- Python behavior to preserve:
  `VAEEncode` calls `vae.encoder.encode(image)`, `ReferenceLatent` attaches
  that latent to conditioning, and the sampler patchifies/concatenates the
  reference latent later with reference image ids at `t_offset=10.0`.
- The existing Mojo Klein inference path is the implementation base:
  `serenitymojo/sampling/klein_sample_cli.mojo`,
  `serenitymojo/sampling/klein_sampler.mojo`, and
  `serenitymojo/serve/klein_backend.mojo`.
- `scripts/check_workflow_node_surface.py` now guards the Python oracle markers
  in `VAEEncode`, `ReferenceLatent`, and `bridge/sampling.py` so a later run
  does not drift back to a non-Python Klein edit oracle.
- Already-guarded invariants:
  target/reference token order is `[target ; reference]`, target ids
  `[0,row,col,0]`, reference ids `[10,row,col,0]`, reference tokens stay fixed,
  only the first `N_TARGET` prediction slice updates, and the schedule is
  `build_img2img_sigmas(steps, 2.02, denoise)` with direct velocity Euler.

## Klein Staged Daemon Bridge Update

Current change moves Klein txt2img from a fail-loud-only daemon route to a
process-separated staged bridge with pollable child commands:

1. `KleinBackend` writes per-job
   `output/serenity_daemon/klein_jobs/<job_id>/sample_prompts.json` with
   `serenity.sample_prompts.v1` caps paths.
2. It launches `output/bin/klein_precache_sample_prompts <sample_json> <9b|4b>`
   through `ExternalCommand` to generate Qwen3 cap-cache files, then validates those caps with
   `validate_klein_cap_cache_header`.
3. It launches `output/bin/klein_sample_cli <klein*.json> <base|lora_path>
   <sample_json> job <out_png>` for txt2img, or appends
   `- <reference_image> <creativity> 2.02 10.0` for bounded ReferenceLatent edit,
   through the same nonblocking fork/exec/waitpid runner.
4. It rewraps the sampler PNG with `serenity.genparams.v1` and writes
   `<out_png>.klein_daemon_result.json`.

Files touched for this bridge:

- `serenitymojo/serve/klein_backend.mojo`
- `serenitymojo/serve/external_command.mojo`
- `serenitymojo/serve/external_command_smoke.mojo`
- `serenitymojo/pipeline/klein9b_precache_sample_prompts.mojo`
- `pixi.toml`
- `scripts/check_klein_lora_daemon_smoke.py`
- `scripts/check_klein_lora_reference_daemon_smoke.py`
- `scripts/check_workflow_node_surface.py`

New build tasks:

```bash
pixi run build-klein-precache
pixi run build-klein-sampler
pixi run build-daemon
```

Built artifacts now exist:

- `output/bin/klein_precache_sample_prompts`
- `output/bin/klein_sample_cli`
- `output/bin/serenity_daemon`

Checks run after this bridge:

```bash
python3 -m py_compile \
  scripts/check_workflow_node_surface.py \
  scripts/check_workflow_graph_product_contract.py

pixi run build-klein-precache
pixi run build-klein-sampler
pixi run mojo run -I . -I /home/alex/MOJO-libs \
  serenitymojo/serve/external_command_smoke.mojo
pixi run build-daemon

pixi run mojo run -I . serenitymojo/sampling/parity/img2img_refpack_parity.mojo
pixi run mojo run -I . serenitymojo/sampling/parity/klein_reference_latent_bridge_smoke.mojo

python3 scripts/check_workflow_graph_product_contract.py \
  --daemon output/bin/serenity_daemon \
  --write-readiness output/checks/workflow_graph_product_readiness.json \
  --json

python3 scripts/check_klein_lora_daemon_smoke.py \
  --write-report output/checks/klein9b_lora_daemon_smoke.json

python3 scripts/check_klein_lora_reference_daemon_smoke.py \
  --write-report output/checks/klein9b_lora_reference_edit_daemon_smoke.json

python3 scripts/check_workflow_node_surface.py \
  --write-readiness output/checks/workflow_node_surface_readiness.json
```

Latest product-smoke jobs against stub mode:

- `job-0293`: native linked graph
- `job-0294`: img2img metadata graph
- `job-0295`: model-only LoRA metadata graph
- `job-0296`: mask metadata graph
- `job-0297`: Comfy API prompt graph
- `job-0298`: SerenityFlow `zimage_t2i`
- `job-0299`: SerenityFlow `qwen_image_t2i`
- `job-0300`: SerenityFlow `klein9b_t2i`
- `job-0301`: SerenityFlow `klein4b_t2i`
- `job-0302`: SerenityFlow `flux2_dev_t2i`
- `job-0303`: SerenityFlow `klein9b_edit`
- `job-0304`: SerenityFlow `klein4b_edit`
- `job-0305`: Ideogram4 visual export

Previous isolated-mode no-heavy-model admission smoke wrote
`output/checks/klein_bridge_admission_smoke.json`, but the ReferenceLatent row
is now stale because the backend no longer rejects bounded edit jobs before
model load:

- `job-0263`: Klein ReferenceLatent/edit request failed before model load;
  `output_path=""`.
- `job-0264`: invalid Klein size failed before model load; `output_path=""`.
- `job-0265`: Flux2-dev request failed before model load; `output_path=""`.

Do not overclaim this as accepted Klein pixel parity. Real-weight direct CLI and
dispatch-mode daemon ReferenceLatent edit smokes have run for 4B and 9B, and
Klein 9B single-LoRA txt2img now has a checked 4-step daemon artifact
(`job-0308`) in `output/checks/klein9b_lora_daemon_smoke.json`.
The SerenityFlow/Comfy Klein 9B edit-LoRA graph now has a checked 1-step daemon
artifact (`job-0309`) in
`output/checks/klein9b_lora_reference_edit_daemon_smoke.json`. Multi-LoRA,
non-1.0 LoRA weights, exact oracle trajectory comparison, multi-step edit
aesthetics, and matched speed/VRAM evidence are still missing.
`scripts/check_klein_sampler_parity_contract.py` remains report-only with known
blockers: missing configured sample caps, no accepted seeded trajectory parity,
no accepted VAE/final-PNG parity, and no matched speed/VRAM evidence.

## Next Work

1. Keep `scripts/check_klein_lora_daemon_smoke.py` as the heavy regression gate
   for Klein 9B single-LoRA txt2img; rerun it after backend/LoRA loader edits
   before trusting `workflow_node_surface_readiness.json`.
2. Keep `scripts/check_klein_lora_reference_daemon_smoke.py` as the heavy
   regression gate for the SerenityFlow Klein edit-LoRA graph; rerun it after
   workflow lowering, ReferenceLatent, or LoRA loader edits.
3. Verify cancel/status behavior during a real long-running Klein sidecar job;
   the runner is pollable, but this still needs live-model evidence.
4. Compare the real edit path against SerenityFlow's Python Comfy-compatible
   oracle: `VAEEncode` -> `ReferenceLatent` -> sampler patchify/concat,
   reference ids with `t_offset=10.0`, fixed reference tokens, and target-only
   slice/update.
5. Extend the no-heavy Python-oracle checker only when SerenityFlow/Comfy
   behavior changes; current guards cover normal VAE encode, ReferenceLatent as
   conditioning metadata, reference patchify/concat, `t_offset=10.0`, and text
   ids using `txt_ids_dims=[3]`.
6. Keep text-id RoPE tied to the Python/Comfy `txt_ids_dims=[3]` convention in
   `serenityflow/bridge/sampling.py`; do not switch it based on non-Python
   sources.
7. Move the remaining LanPaint/inpaint work from metadata-only graph lowering to
   real runtime implementation only from SerenityFlow/LanPaint oracles:
   `LanPaint_KSampler`, `LanPaint_SamplerCustomAdvanced`,
   `LanPaint_KSamplerAdvanced`, outpaint padding, and real sampler-loop
   `mask_image`/`lanpaint_*` consumption in the relevant backend. The current
   Z-Image `LanPaint_MaskBlend` work is only final decoded pixel compositing.
8. Keep backend mask-aware denoise fail-loud until the lowering, runtime
   implementation, and oracle parity gates all exist; metadata plumbing alone
   is not LanPaint/inpaint parity, and final-pixel MaskBlend alone is not
   `LanPaint_KSampler` runtime parity.
9. Only after temp build and checks pass, run `pixi run build-daemon` to update
   `output/bin/serenity_daemon`.
