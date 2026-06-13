# SwarmUI Sampler Parity Map - 2026-06-12

Scope: sampler and scheduler product-surface parity for the Mojo runtime path in
`/home/alex/mojodiffusion`. This is a surface audit, not runtime parity
acceptance. Full Qwen generation remains bounded; video has measured LTX2
DEV-smoke artifacts but no accepted graph-native/full video parity.

SwarmUI/ComfyUI baseline sources used:

- `/home/alex/SwarmUI/dlbackend/ComfyUI/comfy/samplers.py`
- `/home/alex/SwarmUI/dlbackend/ComfyUI/nodes.py`
- `/home/alex/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ComfyUIBackendExtension.cs`
- `/home/alex/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs`
- `/home/alex/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmKSampler.py`
- `/home/alex/SwarmUI/src/Text2Image/T2IParamTypes.cs`

Mojo surface sources used:

- `serenitymojo/serve/serenity_daemon.mojo`
- `serenitymojo/serve/backend.mojo`
- `serenitymojo/serve/zimage_backend.mojo`
- `serenitymojo/serve/qwenimage_backend.mojo`
- `serenitymojo/serve/dispatch_backend.mojo`
- `serenitymojo/sampling/*.mojo`
- `serenitymojo/docs/SAMPLER_PRODUCT_HARNESS_2026-06-05.md`

## Summary

The daemon currently preserves many SwarmUI/ComfyUI sampler-facing request
fields in canonical metadata, threads the sampler-facing fields through typed
`JobParams`/worker IPC, and can map a constrained Comfy KSampler graph into flat
generation parameters. The production backends now fail loud for unsupported
sampler/scheduler names and unsupported advanced surfaces instead of silently
storing unsupported controls as metadata. Z-Image and Qwen now apply
`variation_seed`/`variation_strength` to initial latent noise through a pure-Mojo
Swarm-style CHW slerp helper. `serenitymojo/sampling/sampler_registry.mojo`
now exposes a versioned SwarmUI/Comfy catalog, model-aware backend support
matrix, default aliases, and fail-loud unsupported policy through `/v1/samplers`.
Z-Image now has bounded DPM++ 2M, generic Comfy UniPC bh1/order<=3, and UniPC
bh2 daemon denoise loops on the simple flow-match sigma schedule, plus bounded
Comfy-current `sgm_uniform` scheduler support for Euler/flow-match Euler and
DPM++ 2M paths only. Z-Image and Qwen still use model-specific flow-match
schedules internally and honor only a subset of the requested surface. Ideogram4
now has a bounded native daemon path that accepts Euler/flow-match sampler
aliases and executes
`ideogram4_logitnormal_euler` on the logit-normal schedule only. Multi-image
batch semantics, ancestral/SDE/Karras execution, refiner/upscale/ControlNet, and
arbitrary Comfy graph execution are still blockers.

2026-06-12 endpoint refresh: `output/checks/samplers_endpoint_smoke.json` now
matches the compiled `/v1/samplers` registry for Z-Image `dpmpp_2m` and
`uni_pc_bh2` support. Generic `uni_pc` is now admitted for Z-Image as a
distinct bh1/SigmaConvert path in code, including Comfy penultimate-sigma
discard and final-zero replacement. The focused product checker now records
`job-0077` as bounded runtime artifact evidence for that path. This still does
not change `accepted_sampler_parity:false`.

2026-06-13 scheduler refresh: Z-Image now admits `sgm_uniform` as
`sgm_uniform_flowmatch` for `euler`/`flowmatch_euler` and `dpmpp_2m` only. The
builder follows current Comfy `normal_scheduler(sgm=True)` semantics for
`ModelSamplingDiscreteFlow`: include the max sigma endpoint, apply the flow
shift through the model sampling object, append exactly one terminal `0.0`, and
scale txt2img initial noise by `sigmas[0]`. `uni_pc` and `uni_pc_bh2` with
`sgm_uniform` remain fail-loud until there is separate product artifact evidence,
even though current Comfy allows that sampler/scheduler combination. This still
does not change `accepted_sampler_parity:false`.

2026-06-12 Ideogram4 endpoint/runtime refresh: `/v1/samplers` now includes an
`ideogram4` backend entry with bounded `euler`/flow-match sampler aliases and
`logitnormal` scheduler aliases. `job-0106` is the latest bounded runtime
artifact for that path and records PNG `serenity.genparams.v1`, gallery
readback, `executed_sampler:"ideogram4_logitnormal_euler"`,
`executed_scheduler:"ideogram4_logitnormal"`, `accepted_sampler_parity:false`,
and `accepted_speed_parity:false`.

2026-06-12 img2img refresh: Z-Image flat `init_image`/`creativity` now has a
bounded daemon artifact gate. `job-0088` through `job-0090` reuse `job-0087.png`
as an init image and record `img2img_applied:true`, `denoise_start_step`,
`steps_executed`, `denoise_update_steps`, timings, and positive VRAM. Duplicate
terminal zero sigma intervals are now treated as no-op sentinels, so
`creativity=0.0` records zero denoise updates. That 2026-06-12 gate did not
accept mask/inpaint, graph `LoadImage`/`VAEEncode`, or full img2img parity.

2026-06-13 mask refresh: Z-Image now has the first bounded Comfy
`SetLatentNoiseMask` img2img runtime slice. The graph carries mask source
metadata, `LoadImage MASK` maps to Comfy inverted alpha via
`load_image_mask`, `ImageToMask` red/green/blue channels remain
raw/no-threshold, and `ImageToMask(alpha)` fails loud from graph lowering
because Comfy `LoadImage.IMAGE` is RGB-only. The backend stores encoded init
latent, seeded noise, and a latent preserve mask.
After each sampler update it reapplies the preserve region using `sigma_next`.
Manifests record `mask_image`, `mask_channel`, `inpaint_mask_applied`,
`inpaint_preserve_active_pixels`, and `inpaint_preserve_mean`. The bounded
Z-Image `LanPaint_MaskBlend` slice is final decoded image blending only:
max-pool the image-space mask, Gaussian smooth it with
`sigma=(blend_overlap-1)/4`, then composite
`image1*(1-mask)+image2*mask`. For the LanPaint `ImageScale(area)` ->
`MaskBlend.image1` role, the base/original `init_image` is resized to decoded
output size with Comfy/PyTorch `area` semantics instead of requiring the file to
already match output dimensions. Full `LanPaint_KSampler` sampler-loop runtime
parity is still rejected by `reject_unsupported_lanpaint_sampler_params(...)`
on Z-Image, and unsupported LanPaint metadata remains rejected by
`reject_unsupported_lanpaint_params(...)` on other backends.

`accepted_sampler_parity` must remain false until a real backend dispatch path
proves the requested sampler, scheduler, seed, variation, image count, denoise,
CFG, negative prompt, and output metadata behavior with artifacts and runtime
manifests.

## Parity Map

| Feature | SwarmUI/Comfy expectation | Current Mojo surface | Blocker | Acceptance gate |
| --- | --- | --- | --- | --- |
| Sampler name catalog | Comfy `SAMPLER_NAMES` includes Euler, ancestral Euler, Heun, LMS, DPM, DPM++, DDPM, LCM, IPNDM, DEIS, res multistep, gradient estimation, ER-SDE, seeds, SA-Solver, `ddim`, `uni_pc`, and `uni_pc_bh2`. SwarmUI exposes these plus CFG++ display variants. | `sampler_registry.mojo` exposes the Comfy/Swarm catalog and `/v1/samplers` support matrix. Daemon `JobParams` carries `sampler`; Z-Image accepts `euler`/`flowmatch_euler`, bounded `dpmpp_2m`, bounded generic `uni_pc` bh1/order<=3, and bounded `uni_pc_bh2` aliases. `euler`/`flowmatch_euler` and `dpmpp_2m` can run on the admitted Z-Image `simple` or `sgm_uniform` schedules; both UniPC variants are currently simple-flowmatch only and reject `sgm_uniform`. Qwen still accepts only `euler`/`flowmatch_euler`. Ideogram4 accepts bounded `euler`/flow-match aliases that execute as `ideogram4_logitnormal_euler`. Unsupported sampler names fail loud through shared registry admission. | Registry/admission exists, and Z-Image DPM++ 2M plus both UniPC variants and Ideogram4 logit-normal Euler are wired, but ancestral/SDE/CFG++/Karras/etc. names are not wired into real daemon denoise loops yet. `sgm_uniform` does not broaden UniPC parity until separate artifacts prove that combo. | `/v1/generate` and constrained KSampler graph requests must validate supported sampler names, fail loud on unsupported names or unsupported sampler/scheduler pairs, execute accepted names as distinct algorithms, and runtime manifests must show the executed sampler per artifact. |
| Scheduler name catalog | Comfy schedules include `simple`, `sgm_uniform`, `karras`, `exponential`, `ddim_uniform`, `beta`, `normal`, `linear_quadratic`, and `kl_optimal`. SwarmUI also exposes `turbo`, `align_your_steps`, `ltxv`, `ltxv-image`, and `flux2`. | `sampler_registry.mojo` includes the Comfy/Swarm scheduler catalog, backend defaults, and supported aliases. Z-Image accepts `simple`/flowmatch aliases as `simple_flowmatch` and bounded `sgm_uniform` as `sgm_uniform_flowmatch` for Euler/DPM++ paths. Its `sgm_uniform` builder follows current Comfy `normal_scheduler(sgm=True)` semantics: max sigma endpoint included, one terminal zero appended, flow shift through `ModelSamplingDiscreteFlow`, and txt2img initial noise scaled by `sigmas[0]`. `uni_pc`/`uni_pc_bh2` plus `sgm_uniform` fails loud pending product artifact evidence. Qwen accepts `simple`/flowmatch/`qwen`; Ideogram4 accepts only `logitnormal`/`logit_normal`/`ideogram_logitnormal`/`ideogram4_logitnormal`; other names fail loud before model work. | Request scheduler discovery and admission exist, but most catalog schedulers are unsupported for daemon product backends until a model-specific schedule builder has artifact evidence. Comfy allowing `uni_pc` with `sgm_uniform` is not product evidence for Z-Image. | Backend dispatcher must map scheduler names to model-compatible schedule builders or reject unsupported pairs with 4xx/501 plus metadata. |
| KSampler request mapping | Comfy `KSampler` expects `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, positive/negative conditioning, latent image, and `denoise`. `KSamplerAdvanced` adds start/end step behavior. | `serenity_daemon.mojo` maps supported typed graph nodes: `KSampler` consumes typed MODEL, positive CONDITIONING, negative CONDITIONING, and LATENT values; `sampler_name` maps to `sampler`, `scheduler` is preserved, `denoise` maps to `creativity`, `batch_size` maps to serial `images`, and flat genparams remain accepted. Unknown graph nodes and wrong typed links fail 501. | Mapping is limited to the supported t2i graph; it is not arbitrary Comfy graph execution and does not prove every sampler/scheduler catalog entry has backend execution. | Static graph tests plus daemon product smoke must prove mapped params survive into artifact metadata, wrong typed links fail loud, and unsupported nodes fail loud. Runtime parity requires backend execution evidence. |
| CFG and negative prompt | KSampler combines positive and negative conditioning using CFG, with model-specific conventions where needed. Prompt weights and LoRA prompt tags affect conditioning in SwarmUI. | `prompt`, `negative`, `cfg`, and prompt syntax metadata are parsed. Z-Image uses its current cond-anchored CFG form with sign convention. Qwen uses Qwen scheduler/CFG helpers. Prompt weights are parsed and persisted but not proven applied to conditioning math. | CFG formulas are model-specific and not normalized behind a product sampler contract. Prompt syntax is not full conditioning parity. | Per model: negative prompt A/B artifact check, CFG=1 vs CFG>1 behavior check, prompt-weight conditioning check, and manifest fields for formula/sampler/scheduler. |
| Img2img denoise / creativity | Comfy denoise truncates the sigma schedule when `denoise < 1.0`. SwarmUI `Init Image Creativity` is the fraction of steps run after skipping initial steps. Masks, inpaint, and image-to-image workflows are product features. | Z-Image daemon backend accepts `init_image`, maps `creativity` into a schedule start step, skips duplicate terminal zero no-op intervals, and has bounded artifact evidence for creativity `0.0`, `0.5`, and `1.0` (`job-0088` through `job-0090`). Z-Image also consumes `SetLatentNoiseMask` for the bounded img2img preserve-mask slice: shared `image_io.mojo` helpers decode the Comfy mask, resize it with the standard soft bilinear mask path, invert it into a latent preserve mask, and blend preserved regions after every sampler update using `sigma_next`. Z-Image consumes `LanPaint_MaskBlend` only as a bounded final decoded pixel blend using the LanPaint max-pool/Gaussian mask and `image1*(1-mask)+image2*mask` formula, with Comfy/PyTorch `area` resize for the base/original `ImageScale(area)` -> `MaskBlend.image1` role. Hard nearest/binary helpers are present for the later LanPaint sampler path, not full LanPaint runtime parity. Qwen and Ideogram4 explicitly reject img2img. Sampling helper files exist for refpack/inpaint, but full img2img/inpaint/LanPaint parity is not accepted. | Denoise semantics are not generalized across backends; mask support is limited to Z-Image `SetLatentNoiseMask` img2img preserve blending plus final-pixel `LanPaint_MaskBlend`; graph-side image/mask tensor transforms are absent except the bounded MaskBlend image1 area resize; full LanPaint sampler inner-loop remains blocked; Qwen/Ideogram4 img2img blocked. | For each supported image backend: run artifact checks with known init image, denoise/creativity values, mask metadata where supported, dimensions, denoise start/update semantics, timings, and VRAM. Unsupported backends must fail loud. |
| Seed behavior | Swarm/Comfy seed controls noise; Swarm batch elements use deterministic seed offsets. Seed `-1` can randomize in UI. | Z-Image and Qwen use request seed for initial noise. Daemon validates unsigned seeds and stores them. `images=N` expands to serial jobs with `seed+i`; variation noise uses `variation_seed+image_index`. Stub path uses deterministic output color. | No random seed policy equivalent to SwarmUI `-1`. | Repeated same-seed requests must produce byte/comparable latent determinism per backend; multi-image requests must document and test per-image seed policy. |
| Variation seed and strength | `SwarmKSampler` blends base noise and variation-seed noise with spherical interpolation when `var_seed_strength > 0`. | `variation_seed` and `variation_strength` are parsed, stored in canonical metadata, typed through `JobParams`/IPC, and accepted through constrained workflow params. `serenitymojo/sampling/variation_noise.mojo` implements pure-Mojo CHW slerp; Z-Image applies it before BF16 latent upload and records `variation_seed`, `variation_strength`, and `variation_applied` in its manifest. Qwen applies the same helper on the optional variation path while keeping its zero-variation GPU `randn` path unchanged. | Runtime behavior exists for image backends, but full sampler parity is still false until nontrivial sampler/scheduler algorithms have per-sampler artifact gates. | Artifact test where same base seed plus different variation seed/strength changes noise/output as expected; manifest must record applied variation mode. |
| Images and batch size | SwarmUI has user-facing `Images` count and separate internal `Batch Size`. Comfy latent batch size can produce multiple outputs. | Daemon parses `images=N`, expands it into N serial queued jobs, offsets seeds as `seed+i`, and records `image_index`/`image_count` in `JobParams`, worker IPC, PNG metadata, job JSON, and Z-Image manifests. | Product Images count is covered by serial artifacts; true Comfy latent-batch execution is not implemented as one backend batch. | `/v1/generate images=N` must produce N artifacts, N metadata records, deterministic per-image seeds, progress/job states, and gallery entries. If separate batch-size semantics are exposed, they need a backend batch path or fail-loud gating. |
| Hires, upscale, and refiner | SwarmUI exposes hires/upscale flows, refiner model, refiner steps/CFG, refiner sampler/scheduler, and refiner step-swap or post-apply methods. | Mojo has model/runtime primitives and Z-Image 1024/full paths under active development, but no accepted daemon refiner/upscale product surface in this sampler map. | No request schema, backend chain, artifact metadata, or acceptance smoke for hires/upscale/refiner. | Add explicit API fields, fail-loud unsupported matrix, and artifact checks proving low-res output, upscale/refiner stage, final dimensions, metadata, timings, and VRAM. |
| Control, IP-Adapter, regional, masks | SwarmUI supports ControlNet slots, image inputs, strength, start/end fractions, masks, regional prompting, and related conditioning controls. | Z-Image has a bounded `SetLatentNoiseMask` img2img preserve-mask path with manifest fields for mask source/result stats and a bounded final-pixel `LanPaint_MaskBlend` path. Other Mojo files cover inpaint/mask helpers and model-specific control experiments, but the daemon sampler request surface does not expose ControlNet/IP-Adapter/regional conditioning. | No typed daemon request model, model compatibility matrix, or product runtime path for control conditioning. Full inpaint and LanPaint sampler-loop semantics are still unaccepted. | API must accept validated control inputs, connect them to model conditioning, produce artifact metadata, and reject unsupported controls per backend. |
| Output metadata and reuse | SwarmUI keeps generation parameters for reuse, gallery display, and PNG metadata. | Daemon writes canonical `serenity.genparams.v1` metadata and job/gallery state. Z-Image manifests include requested/executed sampler fields, readiness labels, timings, peak VRAM, mask fields (`mask_image`, `mask_channel`, `inpaint_mask_applied`, `inpaint_preserve_active_pixels`, `inpaint_preserve_mean`), and `accepted_sampler_parity:false`. Ideogram4 now embeds PNG `serenity.genparams.v1` for the bounded one-step artifact and keeps sampler/timing/VRAM evidence in the sidecar manifest. | Metadata preservation and fail-loud admission are present, but sampler/scheduler acceptance cannot claim registry parity until distinct algorithms are executed and proven. Ideogram4 metadata is only proven for the bounded one-step product smoke. | Every artifact must include requested and executed sampler/scheduler, denoise, seed/variation, image index/count, dimensions, timings, VRAM, backend, readiness, and acceptance booleans. |
| Workflow graph coverage | SwarmUI emits Comfy workflows and custom Swarm nodes for samplers, schedules, refiners, previews, and advanced flows. | Daemon supports a typed linked workflow graph for the supported t2i chain and rejects unknown nodes, wrong typed links, and cyclic graphs with 501. It is not an arbitrary Comfy executor. | Advanced sampler/refiner/upscale/control/video graphs are not represented. | Maintain a supported-node matrix, add product tests for every accepted node/edge, and reject unknown nodes with explicit unsupported details. |
| Model/backend mapping | SwarmUI routes sampler choices across many model families, with family-specific defaults such as Anima ER-SDE/simple, Flux2/flux2, Z-Image/simple, Qwen/simple. | `dispatch_backend.mojo` currently routes real daemon work to Z-Image, Qwen, and bounded Ideogram4 backends, with stub backends for smoke. Sampling modules exist for Klein/Flux2, SDXL, SD15, SD3, Chroma, ERNIE, Anima, and LTX2, but they are not daemon sampler products. | Klein/other model samplers are not dispatchable through the daemon product surface. Ideogram4 remains bounded and experimental. | Add backend dispatch entries only when each model has artifact, timing, VRAM, metadata, and sampler/scheduler failure-mode evidence. |
| Qwen and video quarantine | SwarmUI can generate Qwen image and video workflows through Comfy backends when resources and nodes are available. | Qwen daemon backend exists but full generation is not a safe target for this task. `/v1/video` now has bounded LTX2 DEV-smoke MP4 evidence, including video-only and audio-enabled paths, but not accepted graph-native/full video parity. | Memory/runtime readiness for full Qwen is unproven here; video remains bounded smoke evidence, not production parity. | Do not accept Qwen full sampler parity or video sampler parity until separate product gates prove real artifacts, resource evidence, metadata, quality, and graph/workflow behavior. |
| Readiness labels | SwarmUI users expect real outputs when a control is offered. Unsupported controls should be disabled or fail clearly. | Product docs and Z-Image manifests use readiness and acceptance booleans. Static checkers can validate markers but cannot prove runtime. | Current sampler surface mixes preserved metadata with executable behavior. | UI/API must expose only supported sampler controls per model or return precise unsupported errors. Static checker plus runtime artifact gates must both pass before readiness labels change. |

## Current Mapping Notes

- Flat `/v1/generate` params accepted by the daemon include `prompt`,
  `negative`, `model`, `width`, `height`, `steps`, `seed`, `cfg`, `sampler`,
  `scheduler`, `variation_seed`, `variation_strength`, `images`, `init_image`,
  `creativity`, and `lora`.
- Constrained Comfy graph mapping accepts `workflow.params`,
  `workflow.genparams`, and a typed linked workflow graph for checkpoint, CLIP
  prompt text, empty latent size/batch, KSampler, VAE decode, and save image.
  Unknown nodes, wrong typed links, and cyclic graphs are unsupported graph
  requests, not silent fallbacks.
- `JobParams` currently carries executable fields for prompt, negative, size,
  steps, seed, CFG, sampler, scheduler, variation seed/strength, image count,
  init image, creativity, LoRA, metadata JSON, and output dir. Worker IPC carries
  the same fields.
- Z-Image backend executes its internal Euler-like simple flow-match schedule,
  bounded `sgm_uniform_flowmatch` for Euler/flow-match Euler and DPM++ 2M,
  bounded DPM++ 2M on admitted Z-Image sigma schedules, generic Comfy UniPC
  bh1/order<=3 with SigmaConvert/penultimate-sigma discard/final-zero
  replacement/initial-noise scaling on simple flow-match only, bounded UniPC bh2
  order-2 on the simple Z-Image sigma trace, and real
  CFG/negative/img2img subset. It rejects unsupported sampler controls, supports
  proven multi-LoRA runtime stacking for accepted formats, records
  requested/executed sampler fields plus DPM++/UniPC trace fields, and records
  sampler parity as not accepted.
- Z-Image `sgm_uniform` uses current Comfy `normal_scheduler(sgm=True)` behavior
  for `ModelSamplingDiscreteFlow`: the max sigma endpoint is retained, the flow
  shift is applied through the model-sampling conversion, one terminal `0.0` is
  appended, and txt2img initial noise is scaled by `sigmas[0]`. This schedule is
  bounded to Euler/flow-match Euler and DPM++ 2M. `uni_pc` and `uni_pc_bh2` with
  `sgm_uniform` remain fail-loud until product artifact evidence exists for that
  exact combo, despite Comfy permitting it.
- DPM++ 2M runtime evidence exists for `job-0036`:
  `output/serenity_daemon/job-0036.png` and
  `output/serenity_daemon/job-0036.png.zimage_daemon_result.json`. The manifest
  records `dpmpp_update_steps:3`, `dpmpp_second_order_steps:2`,
  `denoise_seconds_per_step:0.3180188945`, `peak_vram_mib:21571.8125`, and
  `accepted_sampler_parity:false`.
- UniPC bh2 runtime evidence exists for `job-0040`:
  `output/serenity_daemon/job-0040.png` and
  `output/serenity_daemon/job-0040.png.zimage_daemon_result.json`, produced by
  `python3 scripts/check_zimage_daemon_product_contract.py
  --skip-dpmpp2m-smoke --skip-generic-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-img2img-smoke --skip-multi-lora-smoke
  --write-readiness output/checks/zimage_unipc_bh2_product_readiness.json`.
  The manifest records `requested_sampler:"uni_pc_bh2"`,
  `requested_scheduler:"flowmatch"`, `executed_sampler:"uni_pc_bh2"`,
  `executed_scheduler:"simple_flowmatch"`, `solver_type:"bh2"`,
  `solver_order:2`, `schedule_source:"zimage_build_sigmas"`,
  `unipc_update_steps:3`, `unipc_corrector_steps:2`,
  `unipc_second_order_steps:2`, `denoise_seconds_per_step:0.32013946925`,
  `peak_vram_mib:21727.5625`, and `accepted_sampler_parity:false`.
- Generic `uni_pc` runtime evidence exists for `job-0077`:
  `output/serenity_daemon/job-0077.png` and
  `output/serenity_daemon/job-0077.png.zimage_daemon_result.json`, produced by
  `python3 scripts/check_zimage_daemon_product_contract.py --timeout 900 --steps
  1 --skip-unsupported-smoke --skip-dpmpp2m-smoke --skip-unipc-smoke
  --skip-multi-image-smoke --skip-variation-smoke --skip-img2img-smoke
  --skip-multi-lora-smoke --write-readiness
  output/checks/zimage_daemon_generic_unipc_readiness.json`.
  The product checker submits a 512x512 4-step
  `sampler:"uni_pc"`, `scheduler:"flowmatch"` artifact smoke and validates
  `algorithm:"uni_pc"`, `solver_type:"bh1"`, `solver_variant:"bh1"`,
  `solver_order:3`, `sigma_parameterization:"SigmaConvert"`,
  `schedule_source:"zimage_build_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps"`,
  requested/executed sampler/scheduler fields, and UniPC update/corrector
  counters. The manifest records `steps_executed:4`,
  `unipc_update_steps:4`, `unipc_corrector_steps:3`,
  `unipc_second_order_steps:2`, `denoise_seconds_per_step:0.29673903050000006`,
  `total_wall_seconds:3.872930771`, `peak_vram_mib:21379.875`, and
  `accepted_sampler_parity:false`.
- Ideogram4 bounded logit-normal Euler evidence exists for `job-0106`:
  `output/serenity_daemon/job-0106.png` and
  `output/serenity_daemon/job-0106.png.ideogram4_daemon_result.json`, validated
  by `python3 scripts/check_ideogram4_daemon_product_contract.py --artifact
  output/serenity_daemon/job-0106.png --json`. The PNG embeds
  `serenity.genparams.v1`, `/v1/gallery/read` returns reusable params, and the
  sidecar manifest records
  `requested_sampler:"euler"`, `requested_scheduler:"logitnormal"`,
  `executed_sampler:"ideogram4_logitnormal_euler"`,
  `executed_scheduler:"ideogram4_logitnormal"`, sigma trace
  `[0.9994472,0.00012339458]`, fixed text window `1024`,
  `denoise_seconds_per_step:6.174464852`,
  `total_wall_seconds:221.428687069`, `peak_vram_mib:22088.6875`,
  `accepted_sampler_parity:false`, and `accepted_speed_parity:false`. This is a
  one-step artifact smoke with PNG metadata, not quality, speed, or sampler
  parity.
- Ideogram4 bounded fail-loud option evidence exists in
  `output/checks/ideogram4_daemon_product_readiness.json`, produced by
  `python3 scripts/check_ideogram4_daemon_product_contract.py
  --fail-loud-smoke --write-readiness
  output/checks/ideogram4_daemon_product_readiness.json --json`. The smoke
  covers negative prompt, LoRA, prompt LoRA tag, init image, non-default
  creativity/denoise, variation, unsupported size, unsupported sampler,
  unsupported scheduler, and bad CFG. Every case returned HTTP `422` before job
  fanout; `/v1/jobs` stayed unchanged and no Ideogram text/DiT/VAE load markers
  appeared.
- Z-Image img2img/creativity runtime evidence exists for `job-0088` through
  `job-0090`, produced by `python3 scripts/check_zimage_daemon_product_contract.py
  --timeout 900 --steps 1 --skip-unsupported-smoke --skip-dpmpp2m-smoke
  --skip-generic-unipc-smoke --skip-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-multi-lora-smoke --write-readiness
  output/checks/zimage_img2img_creativity_readiness.json`. The checker reuses
  baseline `job-0087.png` as `init_image` and validates PNG metadata, sidecar
  manifest fields, `img2img_applied:true`, `denoise_start_step`,
  `steps_executed`, `denoise_update_steps`, sigma-derived start semantics,
  timings, and VRAM for creativity `0.0`, `0.5`, and `1.0`. The report keeps
  `accepted_img2img_parity:false`.
- Qwen backend executes its internal Qwen schedule at 1024x1024 only. It rejects
  unsupported sampler controls, LoRA, and img2img. Full generation is out of
  scope for this task.
- Stub backends are useful for API plumbing but do not count as sampler runtime
  parity.

## Minimum Acceptance Sequence

1. Keep the `/v1/samplers` support matrix current per backend.
2. Implement exact Comfy `uni_pc` semantics separately from `uni_pc_bh2`; do
   not alias generic `uni_pc` to bh2 until artifact evidence passes.
3. Keep `uni_pc`/`uni_pc_bh2` with `sgm_uniform` fail-loud until that exact
   sampler/scheduler combo has product artifact evidence; Comfy catalog
   allowance alone is not enough.
4. Execute every accepted sampler/scheduler as the requested algorithm, or
   reject it before expensive model work.
5. Thread executed sampler/scheduler, variation, image index/count, and denoise
   semantics into backend manifests and PNG metadata.
6. Keep the multi-output `images > 1` daemon path covered by runtime artifact
   gates and add a separate batched-latent path only if the UI exposes Comfy
   batch-size semantics.
7. Add runtime artifact gates per backend that verify dimensions, metadata,
   timings, VRAM, readiness label, and acceptance booleans.
8. Only then flip `accepted_sampler_parity` for the specific model/sampler pair
   that has evidence.
