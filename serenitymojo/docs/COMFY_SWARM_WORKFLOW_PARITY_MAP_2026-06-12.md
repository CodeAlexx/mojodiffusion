# Comfy/Swarm Workflow Parity Map - 2026-06-12

Scope: static audit of the current `serenitymojo/serve` daemon workflow and
node surface. This is not a runtime generation claim, not a Qwen full-generation
claim, not an accepted inpaint/LanPaint parity claim, and not an accepted video
parity claim. Product runtime remains Mojo-native; Python is used only for this
audit/checker.

Current verdict: the daemon has flat `/v1/generate` support plus a typed linked
graph executor for the supported image chain. It accepts
`CheckpointLoaderSimple -> CLIPTextEncode -> EmptyLatentImage -> KSampler ->
VAEDecode -> SaveImage` style workflows, carries typed MODEL/CLIP/VAE/
CONDITIONING/LATENT/IMAGE values, carries a path-backed MASK value for
`SetLatentNoiseMask` with source metadata, lowers selected LanPaint visual-canvas
nodes into explicit metadata, and validates graph edges before enqueue. Z-Image
now consumes that mask for the first bounded `SetLatentNoiseMask` img2img
preserve-region runtime slice. It is not an arbitrary ComfyUI graph executor.
Unsupported graph nodes must fail loudly with HTTP 501 instead of being ignored
or silently flattened. In exact checker terms: not an arbitrary ComfyUI graph
executor.

Primary source files:

- `serenitymojo/serve/serenity_daemon.mojo`
- `serenitymojo/serve/workflow_graph.mojo`
- `serenitymojo/serve/backend.mojo`
- `serenitymojo/serve/ipc_codec.mojo`
- `serenitymojo/serve/zimage_backend.mojo`
- `serenitymojo/serve/qwenimage_backend.mojo`
- `serenitymojo/serve/image_io.mojo`
- `serenitymojo/serve/model_scan.mojo`
- `serenitymojo/sampling/inpaint.mojo`
- `scripts/check_lanpaint_oracle_surface.py`
- `scripts/check_lanpaint_canvas_daemon_smoke.py`
- `serenitymojo/docs/SWARMUI_PRODUCT_PATH_LEDGER_2026-06-12.md`
- `serenitymojo/docs/SWARMUI_PARITY_ROADMAP_2026-06-12.md`

## Supported Adapter Markers

The accepted linked graph node types are the current bounded allowlist:

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
  `RandomNoise`, `KSamplerSelect`, `SamplerCustomAdvanced`,
  `LanPaint_SamplerCustomAdvanced`, `LanPaint_MaskBlend`, `SaveImage`,
  `PreviewImage`, `MarkdownNote`, `Note`

The adapter also accepts `workflow.params` and `workflow.genparams` as flat
genparams passthrough forms. For linked `workflow.nodes`/`workflow.edges`
bodies, accepted fields are resolved through the typed graph value store and
then mapped into the existing flat `JobParams`/`serenity.genparams.v1` product
path. Arbitrary node families, custom outputs, and advanced graph behaviors are
not implemented.

Current graph mask support is path/source metadata, not a graph tensor. `LoadImage`
can expose a typed `MASK` output with source `load_image_mask`, which means the
Comfy `LoadImage` inverted-alpha mask. Raw `ImageToMask` channels remain raw
channel values with no thresholding, and `MaskToImage -> ImageToMask(red)` keeps
that source metadata instead of collapsing into a bare path. `SetLatentNoiseMask`
can propagate the mask as `mask_image` plus `lanpaint_mask_channel`, and
`/v1/generate` plus worker IPC preserve it.

Z-Image consumes that metadata only for the bounded img2img preserve-region slice:
it decodes the Comfy mask, resizes it to latent shape with the standard Comfy
bilinear mask path, converts it to a latent preserve mask, stores the encoded
init latent plus seeded noise, and reapplies the preserve region after every
sampler update using `sigma_next`. Result metadata records `mask_image`,
`mask_channel`, `inpaint_mask_applied`, `inpaint_preserve_active_pixels`, and
`inpaint_preserve_mean`.

Current LanPaint support remains metadata plumbing only: visual Comfy UI canvas
graphs lower selected LanPaint sampler/blend/mask-conversion nodes into
path-backed image/mask handles and canonical `lanpaint_*` fields. Full LanPaint
sampler inner-loop and blend semantics remain rejected by
`reject_unsupported_lanpaint_params(...)`.

## Node Surface Table

| Node family | Comfy/SwarmUI expectation | Current Mojo support | Blocker | Acceptance gate |
|---|---|---|---|---|
| Workflow envelope / graph executor | Accept Comfy-style workflow JSON, execute nodes by links in graph order, carry typed MODEL/CLIP/VAE/LATENT/IMAGE/MASK/CONDITIONING values. | `/v1/generate` accepts flat JSON, `workflow.params`, `workflow.genparams`, Comfy API prompt objects, Comfy UI visual canvas graphs, and linked `workflow.nodes`/`workflow.edges` for the supported image chain. `looks_like_comfy_ui_canvas_graph`, `comfy_ui_canvas_to_typed_graph`, and `apply_comfy_ui_canvas_graph` lower visual nodes/links into the typed executor. The graph executor validates typed inputs and runs nodes in dependency order before flattening to `JobParams`; MASK is currently path/source metadata, not a graph tensor. | Advanced Comfy/Swarm node families, custom node classes beyond the bounded allowlist, graph-side utility outputs, and full graph API parity are not represented. | Static gate proves supported markers and 501 paths; runtime graph gates submit linked shuffled fixtures, a Comfy API prompt fixture, and a LanPaint Comfy UI canvas fixture, verify typed edges drive prompt/negative/model/latent/image/mask/LanPaint metadata, and reject unsupported/wrong-typed links. |
| Checkpoint / CLIP / VAE loader nodes | `CheckpointLoaderSimple` returns MODEL, CLIP, and VAE handles; separate `CLIPLoader`, `DualCLIPLoader`, `VAELoader` nodes can override components. | `CheckpointLoaderSimple.fields.ckpt_name` emits typed MODEL/CLIP/VAE placeholders and maps MODEL to flat `model`. `/v1/models` scans checkpoints and LoRAs with arch tags. Text encoders and VAEs are internal to the selected backend. | No separated CLIP/VAE override nodes, no graph-selectable resident component split, and only Z-Image, Qwen, bounded Ideogram4, and Klein daemon backends are routed today. | Loader nodes create typed resident handles or fail 501 before CUDA for unsupported component splits; `/v1/models` reports compatibility per model/backend. |
| Text encode | `CLIPTextEncode` consumes a CLIP handle and emits conditioning. Positive/negative prompts are link-driven, and prompt weights affect conditioning. | `CLIPTextEncode` now requires a typed CLIP link and emits CONDITIONING text. `KSampler` consumes positive and negative CONDITIONING by link, so prompt polarity is not inferred from node title. Prompt syntax metadata is recorded, but `conditioning_weights_applied=false`. | Weighted spans are still not applied to model conditioning math. Advanced conditioning combine/region nodes are unsupported. | Positive and negative text encode nodes emit typed conditioning handles; prompt weighting either changes conditioning or is rejected/labelled per request. |
| Empty latent / batch | `EmptyLatentImage` emits a LATENT tensor with width, height, and batch size. Batch size produces multiple images or a batched denoise. | `EmptyLatentImage` emits a typed LATENT placeholder with width/height/images. `batch_size` maps to `images`, and the daemon serializes `images=N` into indexed output jobs with seed offsets. | True Comfy latent-batch execution inside one backend call is not implemented. | `images > 1` emits indexed artifacts/gallery rows through the product queue. Separate latent batch semantics need a backend batch path or fail-loud gating. |
| Sampler nodes | `KSampler` consumes MODEL, conditioning, and LATENT; sampler/scheduler names select real algorithms; denoise controls txt2img/img2img strength. | `KSampler` requires typed MODEL, positive CONDITIONING, negative CONDITIONING, and LATENT inputs. It maps `steps`, `seed`, `cfg`, `sampler_name`, `scheduler`, and `denoise` into product params. VAEEncode-derived latents carry `LoadImage` into flat `init_image`. Z-Image has bounded Euler, DPM++ 2M, generic UniPC, UniPC bh2, and flat/img2img creativity artifact evidence; unsupported sampler/scheduler pairs fail loud through the shared registry. | Ancestral/SDE/Karras/CFG++ and many catalog names are not accepted runtime algorithms. Graph img2img currently maps metadata into the existing flat backend path, not a full Comfy latent tensor executor. | Unsupported sampler/scheduler names fail before enqueue, or a real sampler registry maps names to measured Mojo samplers with artifact/timing evidence. |
| VAE decode / SaveImage | `VAEDecode` consumes LATENT plus VAE and emits IMAGE; `SaveImage` consumes IMAGE and honors save options such as prefix/path. | `VAEDecode` requires typed LATENT and VAE inputs and emits typed IMAGE. `SaveImage` requires IMAGE. Real backends still decode internally and write PNG with `serenity.genparams.v1`. | SaveImage prefix/path behavior and graph-selected VAE decode implementation are not independent product nodes. | Decode and save nodes consume typed graph values; SaveImage options are reflected in output paths and metadata, or unsupported options fail 501. |
| LoRA loader / stack | `LoraLoader`, LoRA stack nodes, and prompt LoRA tags apply one or more adapters to the model/CLIP with per-adapter weights and compatibility checks. | Flat `lora: [{name, weight}]` is parsed. `<lora:name:weight>` prompt tags are extracted. `/v1/models` scans LoRA files. `LoraLoaderModelOnly` lowers into the same flat metadata. Z-Image has explicit multi-LoRA runtime stacking for proven PEFT/Comfy formats. Klein 9B has bounded single-LoRA txt2img support for AI Toolkit/Comfy Flux2-Klein keys at weight `1.0`; `scripts/check_klein_lora_daemon_smoke.py` proved txt2img through the daemon as `job-0308`, and `scripts/check_klein_lora_reference_daemon_smoke.py` proved SerenityFlow's `klein9b_edit_lora.json` graph lowers `LoraLoaderModelOnly` plus `ReferenceLatent` into one real edit sampler invocation as `job-0309`. Qwen and Ideogram4 reject LoRA. | Full `LoraLoader` CLIP-side semantics, LoRA stacks, non-1.0 Klein weights, Klein multi-LoRA, LoKr, and unproven formats remain rejected. | LoRA graph nodes are supported with compatibility metadata, multi-adapter behavior where the backend can prove it, and fail-loud rejection where it cannot. |
| Image load / resize / img2img | `LoadImage`, resize/crop nodes, `VAEEncode`, img2img, mask and inpaint nodes work as graph nodes. | Flat `init_image` and `creativity` are accepted. `LoadImage` now emits typed IMAGE and MASK handles; `ImageToMask`, `MaskToImage`, `GetImageSize`, `ImageScale`, and `ImageScaleToTotalPixels` lower as path-backed metadata operations with mask source metadata; `VAEEncode` converts IMAGE+VAE into a LATENT handle carrying `init_image`; `SetLatentNoiseMask` can attach a mask path/source to that LATENT; and `KSampler.denoise` maps to `creativity`. Z-Image validates the init path, decodes PNG/JPEG/WebP through MOJO-libs, resizes bilinear to the job size, VAE-encodes, starts denoise at the creativity sigma, and for `SetLatentNoiseMask` img2img applies preserve-region blending after each sampler update using `sigma_next`. Runtime smokes prove graph `LoadImage -> VAEEncode -> KSampler`, `LoadImage -> SetLatentNoiseMask -> KSampler`, and the LanPaint SDXL visual canvas metadata path reach PNG genparams. Qwen and Ideogram4 reject img2img. | Resize/crop and most mask conversion remain path/metadata passthrough; true graph-side image/mask tensor transforms are still absent. Graph img2img currently uses metadata handles to route into the existing backend `init_image` path, with only the bounded Z-Image preserve-mask runtime slice consuming `mask_image`. | Image nodes decode/resize into typed graph values, `VAEEncode` feeds sampler latents, bounded Z-Image preserve-mask behavior remains covered by helper/static gates, and unsupported backends reject before CUDA. |
| Mask / inpaint / LanPaint | `SetLatentNoiseMask`, `ImageToMask`, `MaskToImage`, LanPaint sampler nodes, mask blend, and outpaint padding alter denoise and final compositing. | `JobParams.mask_image`, explicit `lanpaint_*` fields, `/v1/generate` genparams, and worker IPC preserve mask and LanPaint metadata. `SetLatentNoiseMask(samples: LATENT, mask: MASK) -> LATENT` lowers mask path/source metadata for downstream samplers. Z-Image consumes that path for the bounded img2img preserve-mask slice using `decode_comfy_mask`, `resize_mask_bilinear`, and `load_comfy_latent_preserve_mask`; manifests record `mask_image`, `mask_channel`, `inpaint_mask_applied`, `inpaint_preserve_active_pixels`, and `inpaint_preserve_mean`. `resize_mask_nearest_exact`, `binarize_lanpaint_denoise_mask`, and `load_lanpaint_latent_preserve_mask` pin the LanPaint hard-mask preparation substrate for the later full LanPaint path. `LanPaint_KSampler`, `LanPaint_KSamplerAdvanced`, `LanPaint_SamplerCustomAdvanced`, and `LanPaint_MaskBlend` are accepted graph-lowering nodes that copy sampler/blend widgets into canonical metadata; they do not execute the LanPaint inner loop or final blend. `serenitymojo/sampling/inpaint.mojo` has a mask-blend helper and one supplied-score LanPaint overdamped step with synthetic parity coverage. | Full LanPaint sampler/blend/outpaint semantics are not backend-executed. Qwen, Ideogram4, and Klein still reject `mask_image`; Z-Image still rejects unsupported `lanpaint_*` runtime metadata. `ImagePadForOutpaint`, `ThresholdMask`, `LanPaint_SamplerCustom`, and model-specific edit text encoders remain unsupported/fail-loud. | Backend-specific LanPaint inner-loop/blend/outpaint behavior and parity checks against the LanPaint/SerenityFlow Python oracle must pass before any real backend accepts `lanpaint_*` runtime metadata. Until then, real backends fail loud through `reject_unsupported_lanpaint_params(...)`; non-Z-Image mask runtime controls fail loud through `reject_unsupported_mask_image_params(...)`. |
| Upscale utility nodes | `UpscaleModelLoader`, `ImageUpscaleWithModel`, tiled upscale, and postprocess utility workflows emit real upscaled artifacts. | LTX2 upsampler and Flux/tiling primitives exist elsewhere, but no accepted daemon workflow node or utility endpoint was found. | No product endpoint, no node markers, no artifact/timing/VRAM evidence. | Daemon/UI utility path emits real before/after artifacts with dimensions, timings, VRAM when GPU is used, and metadata. |
| Control / IP-adapter / regional | ControlNet, IP-Adapter, regional prompting, and conditioning-combine nodes alter generation through model-specific adapters. | `ReferenceLatent` has a bounded Klein edit path, but ControlNet, IP-Adapter, and regional prompting have no accepted daemon graph surface. Some model/parity pieces exist outside the product daemon path, but they are not workflow parity. | No product node family for ControlNet/IPAdapter/regional conditioning, no typed CONTROL value, no per-model compatibility matrix, no artifact evidence. | Each node family has a backend-specific Mojo path, preflight compatibility, and measured artifacts; otherwise node requests fail 501. |
| Video nodes | Video loaders, image sequence nodes, video sampler nodes, `VideoCombine`, and VHS-style nodes emit MP4s with frame/duration/audio metadata. | `/v1/video` exposes a bounded LTX2 staged dev smoke runner when `output/bin/ltx2_video_smoke_runner` is built, records `accepted_video_parity:false`, and `/v1/video/probe` can inspect MP4 artifacts. Current daemon evidence proves both video-only and audio-enabled A/V smoke artifacts with frame/duration/audio/muxing/VRAM fields, and the runner now emits `ltx2_runner_timings.json` surfaced as daemon `stage_timings`. | This is still a bounded DEV-smoke runner, not a graph-native video node family or full SwarmUI video parity. Qwen/video remain non-production targets for this slice. | A video backend emits MP4 plus frame count, duration, resolution, muxing, audio behavior, timings, peak VRAM, and readiness label. |
| Failure semantics | Unsupported graph features fail before execution, name the unsupported feature, and never silently no-op. Bad requests are distinct from unsupported features. | Workflow parser raises `[501]` for unsupported graph shape, missing `type_id`, unsupported node type, and missing prompt node. `/v1/generate` converts that sentinel to HTTP 501. Bad schema/type errors return 422. Bounded Ideogram4 unsupported controls now return HTTP 422 before queue fanout; other backend admission failures can still become failed jobs. | Backend-specific unsupported settings are not uniformly preflighted before enqueue/CUDA. Graph unsupported behavior is static/HTTP-level, not full runtime graph validation. | Static checker continues to gate supported markers and 501 sentinel handling; runtime smoke posts unsupported nodes and backend-incompatible options and verifies fail-loud responses/states. |

## Exact Current Blockers

- No arbitrary ComfyUI graph executor beyond the bounded typed linked image
  chain.
- No tensor-backed graph MASK value: current MASK support is path/source metadata.
  Z-Image consumes it only for the bounded `SetLatentNoiseMask` img2img
  preserve-mask slice. LanPaint graph nodes lower metadata only. No advanced
  typed graph values for CONTROL, UPSCALE_MODEL, IMAGE batches, refiner handles,
  or video frames.
- No separate CLIP/VAE loader override nodes.
- `batch_size` maps to serial `images=N` outputs, not true Comfy latent-batch execution.
- Many `sampler` and `scheduler` catalog names are still unsupported runtime algorithms.
- `SaveImage` prefix/path options and graph-selected VAE decode are not independent product nodes.
- `LoraLoaderModelOnly` graph nodes lower to flat LoRA metadata. Z-Image supports proven multi-LoRA runtime formats. Klein 9B supports one AI Toolkit/Comfy Flux2-Klein LoRA at weight `1.0` on txt2img and the bounded SerenityFlow edit-LoRA graph. Full `LoraLoader`, CLIP-side LoRA, LoRA stacks, and Qwen/Ideogram4 LoRA remain unsupported/fail-loud.
- Graph `LoadImage -> VAEEncode -> KSampler` now maps `init_image`/denoise into
  the existing backend path, but resize/crop and true graph-side image/mask
  tensor transforms remain unsupported.
- `SetLatentNoiseMask` can carry `mask_image` plus source/channel metadata.
  Z-Image consumes it only for bounded img2img preserve-region blending; Qwen,
  Ideogram4, and Klein still fail loud for mask runtime metadata.
- `LanPaint_KSampler`, `LanPaint_KSamplerAdvanced`,
  `LanPaint_SamplerCustomAdvanced`, `LanPaint_MaskBlend`, `ImageToMask`, and
  `MaskToImage` are accepted for graph lowering only; they do not execute real
  LanPaint inpaint or blend semantics in a backend.
- Real backends Z-Image, Qwen, Ideogram4, and Klein reject unsupported
  `lanpaint_*` metadata through `reject_unsupported_lanpaint_params(...)`.
- Upscale, ControlNet, IP-Adapter, outpaint padding, unsupported LanPaint
  utility nodes, and regional prompting nodes are missing from the accepted
  product daemon graph surface.
- Video generation nodes remain non-accepted despite bounded LTX2 MP4 evidence;
  graph-native/full video parity still needs broader product gates.
- Some backend incompatibilities fail after enqueue/start rather than through a uniform no-CUDA request preflight.

## Static Gate

Run:

```bash
python3 scripts/check_workflow_node_surface.py --write-readiness output/checks/workflow_node_surface_readiness.json
```

The checker is intentionally no-CUDA except for reading the latest stub product
smoke report plus bounded Klein daemon smoke reports. Passing it means the
bounded typed image graph path is present and the latest checked product smokes
passed. It must not be read as arbitrary ComfyUI/SwarmUI workflow execution
parity.

LanPaint/inpaint has a separate no-heavy oracle surface checker:

```bash
python3 scripts/check_lanpaint_oracle_surface.py \
  --write-report output/checks/lanpaint_oracle_surface.json
```

That checker preserves the Python oracle and fail-loud boundary. It does not
prove image quality, full LanPaint parity, or general backend inpaint parity; the
only documented runtime scope here is the bounded Z-Image `SetLatentNoiseMask`
img2img slice.

Shared mask helper smoke:

```bash
pixi run mojo run -I . -I /home/alex/MOJO-libs serenitymojo/serve/image_io_mask_smoke.mojo
```

LanPaint visual canvas lowering has a separate no-heavy product smoke:

```bash
python3 scripts/check_lanpaint_canvas_daemon_smoke.py \
  --daemon output/bin/serenity_daemon \
  --write-report output/checks/lanpaint_canvas_daemon_smoke.json
```

Latest report: `output/checks/lanpaint_canvas_daemon_smoke.json`, ready with
`job-0311`. It posts `/home/alex/LanPaint/example_workflows/SDXL_Inpaint.json`
to the stub backend and proves PNG genparams carry
`workflow_source="comfy_ui_canvas_graph"`, `lanpaint_num_steps=5`,
`lanpaint_prompt_mode="Image First"`, `lanpaint_mask_blend_overlap=9`,
`mask_image`, and `init_image`. It does not prove real inpaint parity or
quality.
