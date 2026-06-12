# Comfy/Swarm Workflow Parity Map - 2026-06-12

Scope: static audit of the current `serenitymojo/serve` daemon workflow and
node surface. This is not a runtime generation claim, not a Qwen full-generation
claim, and not an accepted video parity claim. Product runtime remains
Mojo-native; Python is used only for this audit/checker.

Current verdict: the daemon has flat `/v1/generate` support plus a typed linked graph executor for the supported text-to-image chain. It accepts
`CheckpointLoaderSimple -> CLIPTextEncode -> EmptyLatentImage -> KSampler ->
VAEDecode -> SaveImage` style workflows, carries typed MODEL/CLIP/VAE/
CONDITIONING/LATENT/IMAGE values, and validates graph edges before enqueue. It
is not an arbitrary ComfyUI graph executor. Unsupported graph nodes must fail
loudly with HTTP 501 instead of being ignored or silently flattened.

Primary source files:

- `serenitymojo/serve/serenity_daemon.mojo`
- `serenitymojo/serve/backend.mojo`
- `serenitymojo/serve/zimage_backend.mojo`
- `serenitymojo/serve/qwenimage_backend.mojo`
- `serenitymojo/serve/image_io.mojo`
- `serenitymojo/serve/model_scan.mojo`
- `serenitymojo/docs/SWARMUI_PRODUCT_PATH_LEDGER_2026-06-12.md`
- `serenitymojo/docs/SWARMUI_PARITY_ROADMAP_2026-06-12.md`

## Supported Adapter Markers

The accepted linked graph node types are:

- `CheckpointLoaderSimple`
- `CLIPTextEncode`
- `EmptyLatentImage`
- `KSampler`
- `VAEDecode`
- `SaveImage`

The adapter also accepts `workflow.params` and `workflow.genparams` as flat
genparams passthrough forms. For linked `workflow.nodes`/`workflow.edges`
bodies, accepted fields are resolved through the typed graph value store and
then mapped into the existing flat `JobParams`/`serenity.genparams.v1` product
path. Arbitrary node families, custom outputs, and advanced graph behaviors are
not implemented.

## Node Surface Table

| Node family | Comfy/SwarmUI expectation | Current Mojo support | Blocker | Acceptance gate |
|---|---|---|---|---|
| Workflow envelope / graph executor | Accept Comfy-style workflow JSON, execute nodes by links in graph order, carry typed MODEL/CLIP/VAE/LATENT/IMAGE/CONDITIONING values. | `/v1/generate` accepts flat JSON, `workflow.params`, `workflow.genparams`, and linked `workflow.nodes`/`workflow.edges` for the supported t2i chain. The graph executor validates typed inputs and runs nodes in dependency order before flattening to `JobParams`. | Advanced Comfy/Swarm node families, custom node classes, graph-side utility outputs, and full graph API parity are not represented. | Static gate proves supported markers and 501 paths; runtime graph gate submits a linked shuffled fixture, verifies typed edges drive prompt/negative/model/latent params, and rejects unsupported/wrong-typed links. |
| Checkpoint / CLIP / VAE loader nodes | `CheckpointLoaderSimple` returns MODEL, CLIP, and VAE handles; separate `CLIPLoader`, `DualCLIPLoader`, `VAELoader` nodes can override components. | `CheckpointLoaderSimple.fields.ckpt_name` emits typed MODEL/CLIP/VAE placeholders and maps MODEL to flat `model`. `/v1/models` scans checkpoints and LoRAs with arch tags. Text encoders and VAEs are internal to the selected backend. | No separated CLIP/VAE override nodes, no graph-selectable resident component split, and only Z-Image/Qwen daemon backends are routed today. | Loader nodes create typed resident handles or fail 501 before CUDA for unsupported component splits; `/v1/models` reports compatibility per model/backend. |
| Text encode | `CLIPTextEncode` consumes a CLIP handle and emits conditioning. Positive/negative prompts are link-driven, and prompt weights affect conditioning. | `CLIPTextEncode` now requires a typed CLIP link and emits CONDITIONING text. `KSampler` consumes positive and negative CONDITIONING by link, so prompt polarity is not inferred from node title. Prompt syntax metadata is recorded, but `conditioning_weights_applied=false`. | Weighted spans are still not applied to model conditioning math. Advanced conditioning combine/region nodes are unsupported. | Positive and negative text encode nodes emit typed conditioning handles; prompt weighting either changes conditioning or is rejected/labelled per request. |
| Empty latent / batch | `EmptyLatentImage` emits a LATENT tensor with width, height, and batch size. Batch size produces multiple images or a batched denoise. | `EmptyLatentImage` emits a typed LATENT placeholder with width/height/images. `batch_size` maps to `images`, and the daemon serializes `images=N` into indexed output jobs with seed offsets. | True Comfy latent-batch execution inside one backend call is not implemented. | `images > 1` emits indexed artifacts/gallery rows through the product queue. Separate latent batch semantics need a backend batch path or fail-loud gating. |
| Sampler nodes | `KSampler` consumes MODEL, conditioning, and LATENT; sampler/scheduler names select real algorithms; denoise controls txt2img/img2img strength. | `KSampler` requires typed MODEL, positive CONDITIONING, negative CONDITIONING, and LATENT inputs. It maps `steps`, `seed`, `cfg`, `sampler_name`, `scheduler`, and `denoise` into product params. Z-Image has bounded Euler, DPM++ 2M, and UniPC bh2 evidence; unsupported sampler/scheduler pairs fail loud through the shared registry. | Generic UniPC/order variants, ancestral/SDE/Karras/CFG++ and many catalog names are not accepted runtime algorithms. `denoise` only maps to Z-Image img2img through `creativity`. | Unsupported sampler/scheduler names fail before enqueue, or a real sampler registry maps names to measured Mojo samplers with artifact/timing evidence. |
| VAE decode / SaveImage | `VAEDecode` consumes LATENT plus VAE and emits IMAGE; `SaveImage` consumes IMAGE and honors save options such as prefix/path. | `VAEDecode` requires typed LATENT and VAE inputs and emits typed IMAGE. `SaveImage` requires IMAGE. Real backends still decode internally and write PNG with `serenity.genparams.v1`. | SaveImage prefix/path behavior and graph-selected VAE decode implementation are not independent product nodes. | Decode and save nodes consume typed graph values; SaveImage options are reflected in output paths and metadata, or unsupported options fail 501. |
| LoRA loader / stack | `LoraLoader`, LoRA stack nodes, and prompt LoRA tags apply one or more adapters to the model/CLIP with per-adapter weights and compatibility checks. | Flat `lora: [{name, weight}]` is parsed. `<lora:name:weight>` prompt tags are extracted. `/v1/models` scans LoRA files. Z-Image has explicit multi-LoRA runtime stacking for proven PEFT/Comfy formats; Qwen rejects LoRA. | No `LoraLoader`/stack graph node support. Qwen has no LoRA path. LoKr and unproven formats remain rejected. | LoRA graph nodes are supported with compatibility metadata, multi-adapter behavior where the backend can prove it, and fail-loud rejection where it cannot. |
| Image load / resize / img2img | `LoadImage`, resize/crop nodes, `VAEEncode`, img2img, mask and inpaint nodes work as graph nodes. | Flat `init_image` and `creativity` are accepted. Z-Image validates the path, decodes PNG/JPEG/WebP through MOJO-libs, resizes bilinear to the job size, VAE-encodes, and starts denoise at the creativity sigma. Qwen rejects img2img. | No `LoadImage`, resize/crop, mask, `VAEEncode`, or inpaint graph nodes. No typed IMAGE/MASK/LATENT flow. | Image nodes decode/resize into typed graph values, `VAEEncode` feeds sampler latents, mask/inpaint behavior is proven by artifact gates, and unsupported backends reject before CUDA. |
| Upscale utility nodes | `UpscaleModelLoader`, `ImageUpscaleWithModel`, tiled upscale, and postprocess utility workflows emit real upscaled artifacts. | LTX2 upsampler and Flux/tiling primitives exist elsewhere, but no accepted daemon workflow node or utility endpoint was found. | No product endpoint, no node markers, no artifact/timing/VRAM evidence. | Daemon/UI utility path emits real before/after artifacts with dimensions, timings, VRAM when GPU is used, and metadata. |
| Control / reference / IP-adapter / regional | ControlNet, IP-Adapter, reference image, regional prompting, and conditioning-combine nodes alter generation through model-specific adapters. | No accepted daemon graph surface. Some model/parity pieces exist outside the product daemon path, but they are not workflow parity. | No product node family, no typed conditioning/control value, no per-model compatibility matrix, no artifact evidence. | Each node family has a backend-specific Mojo path, preflight compatibility, and measured artifacts; otherwise node requests fail 501. |
| Video nodes | Video loaders, image sequence nodes, video sampler nodes, `VideoCombine`, and VHS-style nodes emit MP4s with frame/duration/audio metadata. | `/v1/video` exposes a bounded LTX2 staged dev smoke runner when `output/bin/ltx2_video_smoke_runner` is built, records `accepted_video_parity:false`, and `/v1/video/probe` can inspect MP4 artifacts. Current daemon evidence proves both video-only and audio-enabled A/V smoke artifacts with frame/duration/audio/muxing/VRAM fields, and the runner now emits `ltx2_runner_timings.json` surfaced as daemon `stage_timings`. | This is still a bounded DEV-smoke runner, not a graph-native video node family or full SwarmUI video parity. Qwen/video remain non-production targets for this slice. | A video backend emits MP4 plus frame count, duration, resolution, muxing, audio behavior, timings, peak VRAM, and readiness label. |
| Failure semantics | Unsupported graph features fail before execution, name the unsupported feature, and never silently no-op. Bad requests are distinct from unsupported features. | Workflow parser raises `[501]` for unsupported graph shape, missing `type_id`, unsupported node type, and missing prompt node. `/v1/generate` converts that sentinel to HTTP 501. Bad schema/type errors return 422. Backend admission failures become failed jobs. | Backend-specific unsupported settings are not uniformly preflighted before enqueue/CUDA. Graph unsupported behavior is static/HTTP-level, not full runtime graph validation. | Static checker continues to gate supported markers and 501 sentinel handling; runtime smoke posts unsupported nodes and backend-incompatible options and verifies fail-loud responses/states. |

## Exact Current Blockers

- No arbitrary ComfyUI graph executor beyond the typed linked t2i chain.
- No advanced typed graph values for MASK, CONTROL, UPSCALE_MODEL, IMAGE batches,
  refiner handles, or video frames.
- No separate CLIP/VAE loader override nodes.
- `batch_size` maps to serial `images=N` outputs, not true Comfy latent-batch execution.
- Many `sampler` and `scheduler` catalog names are still unsupported runtime algorithms.
- `SaveImage` prefix/path options and graph-selected VAE decode are not independent product nodes.
- LoRA graph nodes are unsupported; Z-Image supports proven multi-LoRA runtime formats and Qwen rejects LoRA.
- Image load/resize exists only through flat Z-Image `init_image`; no graph image node family exists.
- Upscale, ControlNet, IP-Adapter, reference, mask/inpaint, and regional prompting nodes are missing from the product daemon graph surface.
- Video generation nodes are non-accepted until a daemon video runner emits measured MP4 artifacts with positive VRAM evidence.
- Some backend incompatibilities fail after enqueue/start rather than through a uniform no-CUDA request preflight.

## Static Gate

Run:

```bash
python3 scripts/check_workflow_node_surface.py --write-readiness output/checks/workflow_node_surface_readiness.json
```

The checker is intentionally no-CUDA except for reading the latest stub product
smoke report. Passing it means the supported typed t2i graph path is present and
the latest linked-graph smoke passed. It must not be read as arbitrary
ComfyUI/SwarmUI workflow execution parity.
