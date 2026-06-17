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
- `serenitymojo/docs/SUPPORTED_FEATURE_FOUNDATION_2026-06-16.md`

## Summary

The daemon currently preserves many SwarmUI/ComfyUI sampler-facing request
fields in canonical metadata, threads the sampler-facing fields through typed
`JobParams`/worker IPC, and can map a constrained Comfy KSampler graph into flat
generation parameters. The production backends now fail loud for unsupported
sampler/scheduler names and unsupported advanced surfaces instead of silently
storing unsupported controls as metadata. Z-Image now applies
`variation_seed`/`variation_strength` to initial latent noise through a pure-Mojo
Swarm-style CHW slerp helper; Qwen helper code still contains equivalent wiring,
but Qwen generation is metadata/preflight-only in this product slice and is
rejected before enqueue by both the Mojo daemon and Rust server. `serenitymojo/sampling/sampler_registry.mojo`
now exposes a versioned SwarmUI/Comfy catalog, model-aware backend support
matrix, default aliases, and fail-loud unsupported policy through `/v1/samplers`.
Z-Image now has bounded DPM++ 2M, generic Comfy UniPC bh1/order<=3, and UniPC
bh2 daemon denoise loops on the simple flow-match sigma schedule, plus bounded
Comfy-current `sgm_uniform` scheduler support for Euler/flow-match Euler,
DPM++ 2M, generic UniPC, and UniPC bh2 paths. Z-Image still uses a model-specific
flow-match schedule internally and honors only a subset of the requested surface;
Qwen is advertised only as inventory with no admitted sampler/scheduler until
artifact, timing, VRAM, and sampler evidence exists. Ideogram4
now has a bounded native daemon path that accepts Euler/flow-match sampler
aliases and executes
`ideogram4_logitnormal_euler` on logit-normal aliases, and now admits
`simple`/`flowmatch` scheduler aliases as the bounded
`ideogram4_simple_flowmatch` path. Multi-image batch semantics,
ancestral/SDE/Karras execution, refiner/upscale/ControlNet, and arbitrary Comfy
graph execution are still blockers.

2026-06-12 endpoint refresh: `output/checks/samplers_endpoint_smoke.json` now
matches the compiled `/v1/samplers` registry for Z-Image `dpmpp_2m` and
`uni_pc_bh2` support. Generic `uni_pc` is now admitted for Z-Image as a
distinct bh1/SigmaConvert path in code, including Comfy penultimate-sigma
discard and final-zero replacement. The focused product checker now records
`job-0077` as bounded runtime artifact evidence for that path. This still does
not change `accepted_sampler_parity:false`.

2026-06-13 scheduler refresh: Z-Image now admits `sgm_uniform` as
`sgm_uniform_flowmatch` for `euler`/`flowmatch_euler`, `dpmpp_2m`, `uni_pc`,
and `uni_pc_bh2`. The builder follows current Comfy `normal_scheduler(sgm=True)` semantics for
`ModelSamplingDiscreteFlow`: include the max sigma endpoint, apply the flow
shift through the model sampling object, append exactly one terminal `0.0`, and
scale txt2img initial noise by `sigmas[0]`. The Comfy oracle puts both
`uni_pc` and `uni_pc_bh2` through `DISCARD_PENULTIMATE_SIGMA_SAMPLERS` schedule
prep before UniPC SigmaConvert math, so this port uses the same sigma-prep
wording for both `sgm_uniform` UniPC variants. This still does not change
`accepted_sampler_parity:false`.

2026-06-12 Ideogram4 endpoint/runtime refresh: `/v1/samplers` now includes an
`ideogram4` backend entry with bounded `euler`/flow-match sampler aliases and
`logitnormal` scheduler aliases. `job-0106` is the latest bounded runtime
artifact for that path and records PNG `serenity.genparams.v1`, gallery
readback, `executed_sampler:"ideogram4_logitnormal_euler"`,
`executed_scheduler:"ideogram4_logitnormal"`, `accepted_sampler_parity:false`,
and `accepted_speed_parity:false`.

2026-06-16 Ideogram4 admission refresh: `sampler_registry.mojo` now admits
`simple`/`flowmatch`/`flow_match`/`simple_flowmatch` scheduler aliases for
Ideogram-4 and reports `executed_scheduler:"ideogram4_simple_flowmatch"`.
The Rust server sampler asset and prequeue gate mirror those aliases, so
`/v1/preflight` and `/v1/generate` fail loud or admit the same bounded
Ideogram scheduler set before enqueue. This is admission/static evidence only
until a new Ideogram runtime artifact records the simple/AuraFlow path in its
sidecar.

2026-06-16 SDXL conditioning refresh: the Rust-server product path now has a
bounded same-seed SDXL conditioning gate in
`output/checks/sdxl_conditioning_gate.json`. It proves `cfg=1.0` versus
`cfg=5.0` changes the PNG payload, and a non-empty negative prompt changes the
payload again at `cfg=5.0`, while preserving `serenity.genparams.v1`, sidecar
timings, peak VRAM, and requested/executed sampler metadata. This is bounded
artifact evidence and keeps `accepted_conditioning_parity:false`.

2026-06-16 SD3 conditioning refresh: the same Rust-server conditioning gate now
passes for SD3 in `output/checks/sd3_conditioning_gate.json`. It proves
`cfg=1.0` versus `cfg=4.5` changes the PNG payload, and a non-empty negative
prompt changes the payload again at `cfg=4.5`, while preserving
`serenity.genparams.v1`, timing/VRAM sidecars, requested/executed sampler
metadata, and the `vae_decode_tile_grid:"5x5_lowmem"` decode marker. This is
bounded artifact evidence and keeps `accepted_conditioning_parity:false`.

2026-06-16 Anima conditioning refresh: the same Rust-server conditioning gate
passes for Anima in `output/checks/anima_conditioning_gate.json`. It proves
`cfg=1.0` versus `cfg=4.5` changes the PNG payload, and a non-empty negative
prompt changes the payload again at `cfg=4.5`, while preserving
`serenity.genparams.v1`, timing/VRAM sidecars, and requested/executed sampler
metadata. This closes bounded CFG/negative artifact evidence for the currently
admitted image routes that actually support negative prompts. Flux and Ideogram
remain intentionally negative-prompt-blocked in this product slice.

2026-06-16 Z-Image Karras fail-loud refresh: the Rust-server prequeue gate now
includes a Z-Image `scheduler:"karras"` request in
`output/checks/serenity_server_t2i_product_gate_prequeue_latest.json`. It proves
the current product route rejects Karras before job fanout, preserves unchanged
job count, and `/v1/samplers` keeps `karras` out of Z-Image
`supported_schedulers`. This is not Karras runtime support; it keeps the
unsupported scheduler visible and fail-loud until a real scheduler builder and
artifact gate exist.

2026-06-16 supported-feature foundation: the Rust server now exposes
`GET /v1/capabilities` (`serenity.capabilities.v1`) as the shared feature
admission contract for `/v1/generate`. It mirrors the same prequeue gate used by
`/v1/preflight` and records admitted dimensions, sampler/scheduler subsets,
negative-prompt support, LoRA limits, Ideogram bbox prompt JSON, and disabled
features. The current Rust route remains txt2img-only; image-to-image, inpaint,
image conditioning, VAE override, hires two-pass, refiner, upscale, outpaint,
ControlNet, and video are unsupported fail-loud features rather than hidden
model-specific exceptions.

2026-06-16 foundation extraction update: the Rust admission code moved into
`serenity-server/crates/server/src/capabilities.rs` so `main.rs` stays a control
plane entrypoint rather than a capability registry. The no-CUDA product gate now
generates fail-loud test cases from `/v1/capabilities`; the current prequeue
report records 88 generated `capability_rejections` and no
`failed_capability_rejection_cases`, including image-to-image, ControlNet, VAE
override, outpaint, non-Ideogram bbox prompt JSON, Ideogram negative prompt, and
Flux multi-LoRA rejection cases.

2026-06-16 workflow-lowered field guard update: graph-derived disabled fields
now hit the same Rust front-door guard as flat requests after workflow lowering.
Meaningful `init_image`, `mask_image`, `conditioning_mask_image`,
`inpaint_conditioning_image`, `reference_image`, `outpaint_left`,
`threshold_mask_value`, and LanPaint metadata are rejected with HTTP 400 before
job creation. The no-CUDA report includes passing
`*_image_conditioning_disabled` and `*_outpaint_lowered_fields_disabled` cases
with unchanged job counts. This is deliberately not image-to-image or outpaint
admission.

2026-06-16 preflight profile update: `/v1/preflight` embeds
`capability_profile` (`serenity.capability_profile.v1`) for the selected model.
The same no-CUDA report now includes six green `preflight_capability_profiles`
cases and no `failed_preflight_capability_profile_cases`, covering admitted
Z-Image/Ideogram profiles, blocked Qwen plus Klein/Flux2 profiles, raw disabled
field rejection, and workflow-lowering rejection. Sampler tools should consume
this selected-request profile before exposing controls or interpreting a
rejection.

2026-06-16 workflow preflight profile update: the added
`zimage_workflow_unsupported_node_profile` preserves HTTP 501 for an unsupported
Comfy `ControlNetApply` node while returning `capability_profile.backend:"zimage"`
and `rejection_stage:"workflow_lowering"`.

2026-06-16 canvas capability-consumer update: the browser adapter now caches
`/v1/capabilities` in `serenity-server/canvas/js/api.js` and exposes shared
model/backend/feature helpers. The param rail, prompt bar, refiner/upscale
panel, and gallery action visibility consume those helpers instead of keeping
separate payload walkers. This keeps UI exposure tied to the same fail-loud
contract, while image-to-image and upscale remain hidden because the current
capability payload marks them unsupported.

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
| Sampler name catalog | Comfy `SAMPLER_NAMES` includes Euler, ancestral Euler, Heun, LMS, DPM, DPM++, DDPM, LCM, IPNDM, DEIS, res multistep, gradient estimation, ER-SDE, seeds, SA-Solver, `ddim`, `uni_pc`, and `uni_pc_bh2`. SwarmUI exposes these plus CFG++ display variants. | `sampler_registry.mojo` exposes the Comfy/Swarm catalog and `/v1/samplers` support matrix. Daemon `JobParams` carries `sampler`; Z-Image accepts `euler`/`flowmatch_euler`, bounded `dpmpp_2m`, bounded generic `uni_pc` bh1/order<=3, and bounded `uni_pc_bh2` aliases. `euler`/`flowmatch_euler`, `dpmpp_2m`, and both UniPC variants can run on the admitted Z-Image `simple` or `sgm_uniform` schedules; UniPC `sgm_uniform` uses Comfy `DISCARD_PENULTIMATE_SIGMA_SAMPLERS` schedule prep plus SigmaConvert wording. Qwen remains listed only as inventory/preflight metadata with no admitted sampler. Ideogram4 accepts bounded `euler`/flow-match aliases that execute as `ideogram4_logitnormal_euler`, with the scheduler admission deciding between `ideogram4_logitnormal` and `ideogram4_simple_flowmatch`. Unsupported sampler names fail loud through shared registry admission. | Registry/admission exists, and Z-Image DPM++ 2M plus both UniPC variants and Ideogram4 logit-normal Euler are wired, but ancestral/SDE/CFG++/Karras/etc. names are not wired into real daemon denoise loops yet. Qwen has no admitted sampler in this product slice. Ideogram4 simple/AuraFlow admission is static until a runtime artifact proves it. `sgm_uniform` support for DPM++ 2M and UniPC is bounded artifact evidence and does not broaden accepted sampler parity. | `/v1/generate` and constrained KSampler graph requests must validate supported sampler names, fail loud on unsupported names or unsupported sampler/scheduler pairs, execute accepted names as distinct algorithms, and runtime manifests must show the executed sampler per artifact. |
| Scheduler name catalog | Comfy schedules include `simple`, `sgm_uniform`, `karras`, `exponential`, `ddim_uniform`, `beta`, `normal`, `linear_quadratic`, and `kl_optimal`. SwarmUI also exposes `turbo`, `align_your_steps`, `ltxv`, `ltxv-image`, and `flux2`. | `sampler_registry.mojo` includes the Comfy/Swarm scheduler catalog, backend defaults, and supported aliases. Z-Image accepts `simple`/flowmatch aliases as `simple_flowmatch` and bounded `sgm_uniform` as `sgm_uniform_flowmatch` for Euler/DPM++/UniPC paths; `scheduler:"karras"` fails loud before enqueue in the Rust-server route and is absent from Z-Image `supported_schedulers`. Its `sgm_uniform` builder follows current Comfy `normal_scheduler(sgm=True)` semantics: max sigma endpoint included, one terminal zero appended, flow shift through `ModelSamplingDiscreteFlow`, and txt2img initial noise scaled by `sigmas[0]`. `uni_pc` and `uni_pc_bh2` are prepared through Comfy DISCARD_PENULTIMATE sigma prep before SigmaConvert UniPC math. Qwen remains metadata/preflight-only and `/v1/samplers` publishes it with no admitted scheduler. Ideogram4 accepts `logitnormal`/`logit_normal`/`ideogram_logitnormal`/`ideogram4_logitnormal` plus bounded `simple`/`flowmatch`/`flow_match`/`simple_flowmatch`; other names fail loud before model work. | Request scheduler discovery and admission exist, but most catalog schedulers are unsupported for daemon product backends until a model-specific schedule builder has artifact evidence. Z-Image UniPC `sgm_uniform` and Ideogram4 simple/AuraFlow are bounded support, not accepted sampler parity. | Backend dispatcher must map scheduler names to model-compatible schedule builders or reject unsupported pairs with 4xx/501 plus metadata. |
| KSampler request mapping | Comfy `KSampler` expects `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, positive/negative conditioning, latent image, and `denoise`. `KSamplerAdvanced` adds start/end step behavior. | `serenity_daemon.mojo` maps supported typed graph nodes: `KSampler` consumes typed MODEL, positive CONDITIONING, negative CONDITIONING, and LATENT values; `sampler_name` maps to `sampler`, `scheduler` is preserved, `denoise` maps to `creativity`, `Empty*LatentImage.batch_size` is accepted only as `1`, and flat genparams remain accepted. Unknown graph nodes, wrong typed links, and unsupported latent-batch graph semantics fail 501. | Mapping is limited to the supported t2i graph; it is not arbitrary Comfy graph execution and does not prove every sampler/scheduler catalog entry has backend execution. | Static graph tests plus daemon product smoke must prove mapped params survive into artifact metadata, wrong typed links fail loud, and unsupported nodes fail loud. Runtime parity requires backend execution evidence. |
| CFG and negative prompt | KSampler combines positive and negative conditioning using CFG, with model-specific conventions where needed. Prompt weights and LoRA prompt tags affect conditioning in SwarmUI. | `prompt`, `negative`, `cfg`, and prompt syntax metadata are parsed. Z-Image uses its current cond-anchored CFG form with sign convention. `output/checks/zimage_conditioning_readiness.json` proves same-seed Z-Image CFG and negative-prompt changes alter product PNG payloads while preserving manifest guidance/negative and executed sampler/scheduler metadata. `output/checks/sdxl_conditioning_gate.json`, `output/checks/sd3_conditioning_gate.json`, and `output/checks/anima_conditioning_gate.json` prove the same bounded behavior through the Rust-server SDXL, SD3, and Anima product paths. Qwen helper code contains Qwen scheduler/CFG helpers, but product Qwen requests are rejected before enqueue. Flux and Ideogram intentionally reject or do not use negative prompts in the bounded routes. Weighted prompt syntax now fails loud before enqueue through `scripts/check_weighted_prompt_fail_loud.py` while prompt-syntax metadata still records `conditioning_weights_applied:false`. | Z-Image, SDXL, SD3, and Anima have bounded CFG/negative artifact evidence, but CFG formulas are model-specific and not normalized behind a product sampler contract. Prompt-weight conditioning math is still missing; the current accepted behavior is prequeue rejection rather than silent persistence. Qwen helper code is not accepted CFG evidence while Qwen remains metadata/preflight-only. | Per model: negative prompt A/B artifact check, CFG=1 vs CFG>1 behavior check, prompt-weight conditioning check, and manifest fields for formula/sampler/scheduler. |
| Img2img denoise / creativity | Comfy denoise truncates the sigma schedule when `denoise < 1.0`. SwarmUI `Init Image Creativity` is the fraction of steps run after skipping initial steps. Masks, inpaint, and image-to-image workflows are product features. | Z-Image daemon backend accepts `init_image`, maps `creativity` into a schedule start step, skips duplicate terminal zero no-op intervals, and has bounded artifact evidence for creativity `0.0`, `0.5`, and `1.0` (`job-0088` through `job-0090`). Z-Image also consumes `SetLatentNoiseMask` for the bounded img2img preserve-mask slice: shared `image_io.mojo` helpers decode the Comfy mask, resize it with the standard soft bilinear mask path, invert it into a latent preserve mask, and blend preserved regions after every sampler update using `sigma_next`. Z-Image consumes `LanPaint_MaskBlend` only as a bounded final decoded pixel blend using the LanPaint max-pool/Gaussian mask and `image1*(1-mask)+image2*mask` formula, with Comfy/PyTorch `area` resize for the base/original `ImageScale(area)` -> `MaskBlend.image1` role. Hard nearest/binary helpers are present for the later LanPaint sampler path, not full LanPaint runtime parity. Qwen and Ideogram4 explicitly reject img2img. Sampling helper files exist for refpack/inpaint, but full img2img/inpaint/LanPaint parity is not accepted. | Denoise semantics are not generalized across backends; mask support is limited to Z-Image `SetLatentNoiseMask` img2img preserve blending plus final-pixel `LanPaint_MaskBlend`; graph-side image/mask tensor transforms are absent except the bounded MaskBlend image1 area resize; full LanPaint sampler inner-loop remains blocked; Qwen/Ideogram4 img2img blocked. | For each supported image backend: run artifact checks with known init image, denoise/creativity values, mask metadata where supported, dimensions, denoise start/update semantics, timings, and VRAM. Unsupported backends must fail loud. |
| Seed behavior | Swarm/Comfy seed controls noise; Swarm batch elements use deterministic seed offsets. Seed `-1` can randomize in UI. | Z-Image uses request seed for initial noise. Daemon validates unsigned seeds and stores them. `images=N` expands to serial jobs with `seed+i`; variation noise uses `variation_seed+image_index`. Qwen helper code also has request-seed plumbing, but Qwen generation is not admitted in this product slice. Stub path uses deterministic output color. | No random seed policy equivalent to SwarmUI `-1`. Qwen seed behavior has no accepted product artifact while Qwen remains metadata/preflight-only. | Repeated same-seed requests must produce byte/comparable latent determinism per admitted backend; multi-image requests must document and test per-image seed policy. |
| Variation seed and strength | `SwarmKSampler` blends base noise and variation-seed noise with spherical interpolation when `var_seed_strength > 0`. | `variation_seed` and `variation_strength` are parsed, stored in canonical metadata, typed through `JobParams`/IPC, and accepted through constrained workflow params. `serenitymojo/sampling/variation_noise.mojo` implements pure-Mojo CHW slerp; Z-Image applies it before BF16 latent upload and records `variation_seed`, `variation_strength`, and `variation_applied` in its manifest. Qwen helper code contains the same optional variation path, but `/v1/generate` rejects Qwen before enqueue and `/v1/samplers` exposes Qwen with no admitted sampler/scheduler. | Runtime behavior exists for Z-Image, but full sampler parity is still false until nontrivial sampler/scheduler algorithms have per-sampler artifact gates. Qwen variation helper code is inventory, not product evidence. | Artifact test where same base seed plus different variation seed/strength changes noise/output as expected; manifest must record applied variation mode for each admitted backend. |
| Images and batch size | SwarmUI has user-facing `Images` count and separate internal `Batch Size`. Comfy latent batch size can produce multiple outputs. | Daemon parses flat `images=N`, expands it into N serial queued jobs, offsets seeds as `seed+i`, and records `image_index`/`image_count` in `JobParams`, worker IPC, PNG metadata, job JSON, and Z-Image manifests. Linked workflow `Empty*LatentImage.batch_size>1` and `RepeatLatentBatch` now fail before enqueue with HTTP 501; `output/checks/latent_batch_fail_loud.json` proves the prequeue rejection and unchanged job count. | Product Images count is covered by serial artifacts; true Comfy latent-batch execution is not implemented as one backend batch. | `/v1/generate images=N` must produce N artifacts, N metadata records, deterministic per-image seeds, progress/job states, and gallery entries. Separate batch-size semantics require a backend batch path before they can be accepted. |
| Hires, upscale, and refiner | SwarmUI exposes hires/upscale flows, refiner model, refiner steps/CFG, refiner sampler/scheduler, and refiner step-swap or post-apply methods. | Mojo has model/runtime primitives and Z-Image 1024/full paths under active development, but no accepted daemon refiner/upscale product surface in this sampler map. | No request schema, backend chain, artifact metadata, or acceptance smoke for hires/upscale/refiner. | Add explicit API fields, fail-loud unsupported matrix, and artifact checks proving low-res output, upscale/refiner stage, final dimensions, metadata, timings, and VRAM. |
| Control, IP-Adapter, regional, masks | SwarmUI supports ControlNet slots, image inputs, strength, start/end fractions, masks, regional prompting, and related conditioning controls. | Z-Image has a bounded `SetLatentNoiseMask` img2img preserve-mask path with manifest fields for mask source/result stats and a bounded final-pixel `LanPaint_MaskBlend` path. Other Mojo files cover inpaint/mask helpers and model-specific control experiments, but the daemon sampler request surface does not expose ControlNet/IP-Adapter/regional conditioning. | No typed daemon request model, model compatibility matrix, or product runtime path for control conditioning. Full inpaint and LanPaint sampler-loop semantics are still unaccepted. | API must accept validated control inputs, connect them to model conditioning, produce artifact metadata, and reject unsupported controls per backend. |
| Output metadata and reuse | SwarmUI keeps generation parameters for reuse, gallery display, and PNG metadata. | Daemon writes canonical `serenity.genparams.v1` metadata and job/gallery state. Z-Image manifests include requested/executed sampler fields, readiness labels, timings, peak VRAM, mask fields (`mask_image`, `mask_channel`, `inpaint_mask_applied`, `inpaint_preserve_active_pixels`, `inpaint_preserve_mean`), and `accepted_sampler_parity:false`. Ideogram4 now embeds PNG `serenity.genparams.v1` for the bounded one-step artifact and keeps sampler/timing/VRAM evidence in the sidecar manifest. | Metadata preservation and fail-loud admission are present, but sampler/scheduler acceptance cannot claim registry parity until distinct algorithms are executed and proven. Ideogram4 metadata is only proven for the bounded one-step product smoke. | Every artifact must include requested and executed sampler/scheduler, denoise, seed/variation, image index/count, dimensions, timings, VRAM, backend, readiness, and acceptance booleans. |
| Workflow graph coverage | SwarmUI emits Comfy workflows and custom Swarm nodes for samplers, schedules, refiners, previews, and advanced flows. | Daemon supports a typed linked workflow graph for the supported t2i chain and rejects unknown nodes, wrong typed links, and cyclic graphs with 501. It is not an arbitrary Comfy executor. | Advanced sampler/refiner/upscale/control/video graphs are not represented. | Maintain a supported-node matrix, add product tests for every accepted node/edge, and reject unknown nodes with explicit unsupported details. |
| Model/backend mapping | SwarmUI routes sampler choices across many model families, with family-specific defaults such as Anima ER-SDE/simple, Flux2/flux2, Z-Image/simple, Qwen/simple. | `dispatch_backend.mojo` routes bounded daemon work to Z-Image, bounded Ideogram4, Flux2/Klein through the staged process-separated bridge, plus fixed SDXL/Anima sample-CLI wrappers. The Rust `serenity-server` path maps SDXL, Anima, and SD3/SD3.5 through admitted per-kind Mojo workers. Flux1-dev remains in sampler/model inventory but is blocked from `/v1/generate` after the 1024x1024/20-step browser workflow gate failed with CUDA OOM at 6/20 on 2026-06-17. Local result-sidecar writers exist in the SDXL/Anima/SD3/Flux workers, but Flux sidecar capability is inventory until the memory gate passes. Older lowmem one-step gates in `output/checks/all_admitted_manifest_gate_lowmem.json` remain bounded artifact evidence, not current Flux production admission. | Currently admitted Rust-server image families have bounded artifact evidence, but this is still not broad sampler parity. Standalone Mojo dispatch intentionally does not import heavyweight SD3/Flux stacks. SD15, Chroma, ERNIE, full video, full Qwen, Flux1-dev, and wider sampler variants remain missing or explicitly disabled. The old all-admitted lowmem evidence proves bounded artifact routes, not broad sampler parity, quality parity, img2img, speed parity, or current 20-step Flux readiness. | Accept model coverage only when each admitted model has artifact, timing, VRAM, metadata, sampler/scheduler, and failure-mode evidence; keep heavyweight workers process-separated unless build-memory gates justify widening standalone dispatch. |
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
  bounded `sgm_uniform_flowmatch` for Euler/flow-match Euler, DPM++ 2M, and
  both UniPC variants,
  bounded DPM++ 2M on admitted Z-Image sigma schedules, generic Comfy UniPC
  bh1/order<=3 with SigmaConvert/penultimate-sigma discard/final-zero
  replacement/initial-noise scaling, bounded UniPC bh2/order<=3 through the
  same Comfy SigmaConvert/discard-prep path, and real
  CFG/negative/img2img subset. It rejects unsupported sampler controls, supports
  proven multi-LoRA runtime stacking for accepted formats, records
  requested/executed sampler fields plus DPM++/UniPC trace fields, and records
  sampler parity as not accepted.
- Z-Image `sgm_uniform` uses current Comfy `normal_scheduler(sgm=True)` behavior
  for `ModelSamplingDiscreteFlow`: the max sigma endpoint is retained, the flow
  shift is applied through the model-sampling conversion, one terminal `0.0` is
  appended, and txt2img initial noise is scaled by `sigmas[0]`. For `uni_pc`
  and `uni_pc_bh2`, Comfy applies `DISCARD_PENULTIMATE_SIGMA_SAMPLERS` schedule
  prep before the UniPC SigmaConvert path; this port documents `sgm_uniform`
  support for both variants through that same sigma-prep/SigmaConvert boundary.
- Z-Image UniPC `sgm_uniform` runtime evidence exists for `job-0364` and
  `job-0365`, produced by
  `python3 scripts/check_zimage_daemon_product_contract.py --daemon
  output/bin/serenity_daemon --timeout 900 --steps 1 --skip-dpmpp2m-smoke
  --skip-dpmpp2m-sgm-uniform-smoke --skip-generic-unipc-smoke
  --skip-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-img2img-smoke --skip-multi-lora-smoke
  --write-readiness output/checks/zimage_unipc_sgm_uniform_product_readiness.json`.
  `job-0364` records generic `uni_pc`/`bh1`; `job-0365` records
  `uni_pc_bh2`/`bh2`. Both manifests record
  `executed_scheduler:"sgm_uniform_flowmatch"`,
  `sigma_trace:[1.0,0.96028626,0.90089285,0.8023738,0.0]`,
  `txt2img_initial_noise_scale:0.70710677`,
  `sigma_parameterization:"SigmaConvert"`, and
  `schedule_source:"zimage_comfy_sgm_uniform_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps"`.
  This is bounded evidence and leaves `accepted_sampler_parity:false`.
- Z-Image DPM++ 2M `sgm_uniform` runtime evidence exists for `job-0880`,
  produced by
  `python3 scripts/check_zimage_daemon_product_contract.py --daemon
  output/bin/serenity_daemon --timeout 900 --width 512 --height 512 --steps 1
  --cfg 1.0 --min-free-vram-mib 21000 --skip-unsupported-smoke
  --skip-sgm-uniform-smoke --skip-sgm-uniform-unipc-smoke
  --skip-sgm-uniform-unipc-bh2-smoke --skip-dpmpp2m-smoke
  --skip-generic-unipc-smoke --skip-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-img2img-smoke --skip-multi-lora-smoke
  --write-readiness output/checks/zimage_dpmpp2m_sgm_uniform_readiness.json`.
  It emitted baseline `job-0879` and DPM++ 2M `sgm_uniform` `job-0880`.
  `job-0880` records `requested_sampler:"dpmpp_2m"`,
  `requested_scheduler:"sgm_uniform"`, `executed_sampler:"dpmpp_2m"`,
  `executed_scheduler:"sgm_uniform_flowmatch"`,
  `sigma_trace:[1.0,0.9477647,0.85859877,0.67192113,0.0]`,
  `schedule_source:"zimage_comfy_sgm_uniform_sigmas"`,
  `dpmpp_update_steps:4`, `dpmpp_second_order_steps:3`,
  `denoise_seconds_per_step:0.31631403375`, `peak_vram_mib:22239.0625`, and
  `accepted_sampler_parity:false`.
- DPM++ 2M runtime evidence exists for `job-0036`:
  `output/serenity_daemon/job-0036.png` and
  `output/serenity_daemon/job-0036.png.zimage_daemon_result.json`. The manifest
  records `dpmpp_update_steps:3`, `dpmpp_second_order_steps:2`,
  `denoise_seconds_per_step:0.3180188945`, `peak_vram_mib:21571.8125`, and
  `accepted_sampler_parity:false`.
- UniPC bh2 runtime evidence exists for `job-0367`:
  `output/serenity_daemon/job-0367.png` and
  `output/serenity_daemon/job-0367.png.zimage_daemon_result.json`, produced by
  `python3 scripts/check_zimage_daemon_product_contract.py --daemon
  output/bin/serenity_daemon --timeout 900 --steps 1 --skip-unsupported-smoke
  --skip-sgm-uniform-smoke --skip-sgm-uniform-unipc-smoke
  --skip-sgm-uniform-unipc-bh2-smoke --skip-dpmpp2m-smoke
  --skip-dpmpp2m-sgm-uniform-smoke
  --skip-generic-unipc-smoke --skip-multi-image-smoke --skip-variation-smoke
  --skip-img2img-smoke --skip-multi-lora-smoke --write-readiness
  output/checks/zimage_unipc_bh2_comfy_product_readiness.json`.
  The manifest records `requested_sampler:"uni_pc_bh2"`,
  `requested_scheduler:"flowmatch"`, `executed_sampler:"uni_pc_bh2"`,
  `executed_scheduler:"simple_flowmatch"`, `solver_type:"bh2"`,
  `solver_variant:"bh2"`, `solver_order:3`,
  `sigma_parameterization:"SigmaConvert"`,
  `schedule_source:"zimage_comfy_simple_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps"`,
  `txt2img_initial_noise_scale:0.70710677`, `unipc_update_steps:4`,
  `unipc_corrector_steps:3`, `unipc_second_order_steps:2`,
  `peak_vram_mib:21485.2`, and `accepted_sampler_parity:false`.
- Generic `uni_pc` runtime evidence exists for `job-0077`:
  `output/serenity_daemon/job-0077.png` and
  `output/serenity_daemon/job-0077.png.zimage_daemon_result.json`, produced by
  `python3 scripts/check_zimage_daemon_product_contract.py --timeout 900 --steps
  1 --skip-unsupported-smoke --skip-dpmpp2m-smoke
  --skip-dpmpp2m-sgm-uniform-smoke --skip-unipc-smoke
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
  --skip-dpmpp2m-sgm-uniform-smoke --skip-generic-unipc-smoke
  --skip-unipc-smoke --skip-multi-image-smoke --skip-variation-smoke
  --skip-multi-lora-smoke --write-readiness
  output/checks/zimage_img2img_creativity_readiness.json`. The checker reuses
  baseline `job-0087.png` as `init_image` and validates PNG metadata, sidecar
  manifest fields, `img2img_applied:true`, `denoise_start_step`,
  `steps_executed`, `denoise_update_steps`, sigma-derived start semantics,
  timings, and VRAM for creativity `0.0`, `0.5`, and `1.0`. The report keeps
  `accepted_img2img_parity:false`.
- Z-Image CFG/negative conditioning runtime evidence exists for `job-0876`
  through `job-0878`, produced by
  `python3 scripts/check_zimage_conditioning_daemon_smoke.py --timeout 1200
  --poll-interval 1 --min-free-vram-mib 20000 --write-readiness
  output/checks/zimage_conditioning_readiness.json`. The checker submits three
  same-seed 512x512 4-step jobs: `cfg=1.0, negative=""`, `cfg=4.0,
  negative=""`, and `cfg=4.0` with a non-empty negative prompt. It validates
  PNG metadata, jobs/gallery state, `serenity.zimage.daemon_result.v1`
  manifests, `flowmatch_euler`/`simple_flowmatch` execution metadata,
  visual-health thresholds, and distinct IDAT hashes for CFG and negative-prompt
  deltas. The report keeps `accepted_conditioning_parity:false` and
  `accepted_sampler_parity:false`.
- SDXL CFG/negative conditioning runtime evidence exists for `job-0001` through
  `job-0003`, produced by
  `python3 scripts/check_serenity_server_conditioning_gate.py --models sdxl
  --server-bin serenity-server/target/debug/serenity-server --worker-bin
  output/bin/serenity_worker_zimage --out-dir
  output/checks/sdxl_conditioning_gate --write-report
  output/checks/sdxl_conditioning_gate.json --timeout-per-case 1800
  --poll-interval 2`.
  The checker submits three same-seed 1024x1024 one-step jobs through the Rust
  `/v1/preflight` + `/v1/generate` path. It validates PNG
  `serenity.genparams.v1`, `serenity.sdxl.daemon_result.v1` sidecars, timing,
  peak VRAM, `requested_sampler:"euler"`, `requested_scheduler:"normal"`,
  `executed_sampler:"sdxl_euler"`, and `executed_scheduler:"normal"`.
  IDAT hashes were distinct:
  `cfg_low_empty_negative=127a0b99bf2b0d3e6e981adb5cc4d362e4c2636efab2ff7bd10e3ad24dd0a0b5`,
  `cfg_high_empty_negative=4509c2b8e1291f01ebbe6bf3f466dd1d71a2de8399845962558c0dc88c106180`,
  and
  `cfg_high_with_negative=2de36eb412c5cc412a2c8d04a0e9eeb57c914001895eaeb438d2adc23303dd28`.
  Peak VRAM was about `14622-14627 MiB`; `job-0001` recorded
  `total_wall_seconds=65.714914246` including first-load cost, while `job-0002`
  and `job-0003` recorded about `10.1s` total wall time each. The report keeps
  `accepted_conditioning_parity:false` and `accepted_sampler_parity:false`.
- SD3 CFG/negative conditioning runtime evidence exists for `job-0001` through
  `job-0003`, produced by
  `python3 scripts/check_serenity_server_conditioning_gate.py --models sd3
  --server-bin serenity-server/target/debug/serenity-server --worker-bin
  output/bin/serenity_worker_zimage --out-dir
  output/checks/sd3_conditioning_gate --write-report
  output/checks/sd3_conditioning_gate.json --timeout-per-case 1800
  --poll-interval 2`.
  The checker submits three same-seed 1024x1024 one-step jobs through the Rust
  `/v1/preflight` + `/v1/generate` path. It validates PNG
  `serenity.genparams.v1`, `serenity.sd3.daemon_result.v1` sidecars, timing,
  peak VRAM, `requested_sampler:"euler"`, `requested_scheduler:"simple"`,
  `executed_sampler:"sd3_flowmatch_euler"`,
  `executed_scheduler:"sd3_simple_flowmatch"`, and
  `vae_decode_tile_grid:"5x5_lowmem"`.
  IDAT hashes were distinct:
  `cfg_low_empty_negative=4ac45e63fcd3556b23519f1ddc70d451486abe8bf1223e934a4280f6725dbdf2`,
  `cfg_high_empty_negative=43ddbf8aca9320a3d9e5cd439aac418990180910795a7d658b5645727cace83c`,
  and
  `cfg_high_with_negative=b74e307ecda6f26af9c7060d079ef244f890f78443d93d456d3d1572056f4b25`.
  Peak VRAM was about `18451-19092 MiB`; `job-0001` recorded
  `total_wall_seconds=145.002977806` including cold text/model load,
  `job-0002` recorded `23.027163543`, and `job-0003` recorded
  `22.699221252`. The report keeps `accepted_conditioning_parity:false` and
  `accepted_sampler_parity:false`.
- Anima CFG/negative conditioning runtime evidence exists for `job-0001`
  through `job-0003`, produced by
  `python3 scripts/check_serenity_server_conditioning_gate.py --models anima
  --server-bin serenity-server/target/debug/serenity-server --worker-bin
  output/bin/serenity_worker_zimage --out-dir
  output/checks/anima_conditioning_gate --write-report
  output/checks/anima_conditioning_gate.json --timeout-per-case 1800
  --poll-interval 2`.
  The checker submits three same-seed 1024x1024 one-step jobs through the Rust
  `/v1/preflight` + `/v1/generate` path. It validates PNG
  `serenity.genparams.v1`, `serenity.anima.daemon_result.v1` sidecars, timing,
  peak VRAM, `requested_sampler:"euler"`, `requested_scheduler:"normal"`,
  `executed_sampler:"anima_euler"`, and `executed_scheduler:"normal"`.
  IDAT hashes were distinct:
  `cfg_low_empty_negative=8b4db45968cce27e7fe99927ef1af4d52d955b681cef68d2f9b3fd1d50c8afa4`,
  `cfg_high_empty_negative=2ee6e3264d15fb177066d27ad29081e986442be342dbed63faf0e37ba70c71fa`,
  and
  `cfg_high_with_negative=7fec4d642f9a127ced2e894474da9d570cd954f7f4fd89f89b8c1917f35404b9`.
  Peak VRAM was about `10598-12132 MiB`; total wall times were
  `102.882308372`, `86.731115867`, and `88.064855635`, dominated by
  `82-84s` Qwen/Wan VAE decode. The report keeps
  `accepted_conditioning_parity:false` and `accepted_sampler_parity:false`.
- Weighted prompt syntax now has a no-GPU daemon fail-loud gate:
  `python3 scripts/check_weighted_prompt_fail_loud.py --write-readiness
  output/checks/weighted_prompt_fail_loud.json`. It starts the compiled daemon
  in `stub` mode, posts `(red ceramic cube:1.30)`, requires HTTP 422 with
  `conditioning_weights_applied=false` in the error, and verifies `/v1/jobs`
  did not grow. This is not prompt-weight parity; it prevents silent
  persistence until conditioning-weight math has artifact evidence.
- Flux2/Klein bounded daemon evidence now exists for the staged route:
  `output/checks/klein9b_lora_daemon_smoke.json`,
  `output/checks/klein4b_reference_edit_daemon_smoke.json`,
  `output/checks/klein9b_reference_edit_daemon_smoke.json`, and
  `output/checks/klein9b_lora_reference_edit_daemon_smoke.json`. These reports
  validate real PNGs `job-0867`, `job-0868`, `job-0871`, and `job-0872`, the
  `serenity.klein_daemon_result.v1` manifests, PNG `serenity.genparams.v1`
  metadata, visual-health thresholds, the process-separated Qwen3 cap-cache
  precache, the existing staged Klein sampler CLI, and bounded
  ReferenceLatent/LoRA routes. This is real daemon evidence for the bounded
  Flux2/Klein route, not accepted full sampler parity.
- Qwen helper code contains an internal Qwen schedule and local fail-loud guards
  for unsupported sampler controls, LoRA, and img2img, but Qwen generation is
  metadata/preflight-only in this product slice. Both the Rust server and Mojo
  daemon reject Qwen before enqueue; full generation is out of scope for this
  task until artifact, timing, VRAM, and sampler gates pass.
- Stub backends are useful for API plumbing but do not count as sampler runtime
  parity.

## 2026-06-16 Rust-Server Worker Gate Refresh

The Rust server now has bounded manifest-backed artifact evidence for every
currently admitted image family in one strict managed run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_serenity_server_t2i_product_gate.py --all-admitted --server-bin serenity-server/target/debug/serenity-server --worker-bin output/bin/serenity_worker_zimage --out-dir output/checks/all_admitted_manifest_gate_lowmem --write-report output/checks/all_admitted_manifest_gate_lowmem.json --timeout-per-model 1800 --poll-interval 2 --strict-production
```

Result: `artifact_passed` and `manifest_backed` both contain `zimage`, `sdxl`,
`anima`, `sd3`, `flux`, and `ideogram4`; no failed models, prequeue cases, or
sampler cases were reported.

Model/backend notes:

- SD3 executes the bounded `euler`/`simple` flow-match route as
  `sd3_flowmatch_euler` / `sd3_simple_flowmatch`. The all-admitted artifact
  `output/checks/all_admitted_manifest_gate_lowmem/job-0008.png` is `1024x1024`
  and its sidecar records `released_resident_mmdit_before_vae:true`,
  `vae_decode_tile_grid:"5x5_lowmem"`, `denoise_seconds=17.719684853`,
  `vae_decode_seconds=2.973632038`, and `peak_vram_mib=18809.3125`.
- Flux has historical bounded `euler`/`simple` one-step artifact evidence in
  `output/checks/all_admitted_manifest_gate_lowmem/job-0009.png`, with sidecar
  fields `released_resident_dit_before_unpack:true`,
  `vae_decode_tile_grid:"5x5_lowmem"`, and `peak_vram_mib=22081.25`. It is not
  currently admitted: the 1024x1024/20-step browser workflow gate failed with
  CUDA OOM at 6/20 on 2026-06-17.
- SDXL, Anima, and Ideogram4 also produced `1024x1024` manifest-backed
  artifacts in the same run. ZImage produced the bounded `512x512` product-gate
  artifact.

This refresh changes the model-dispatch evidence level for currently admitted
Rust-server image families, but it does not flip sampler parity. The sidecars
still record `accepted_sampler_parity:false`; broader Comfy catalog samplers,
prompt-weight conditioning, latent batch execution, img2img/inpaint, advanced
workflow surfaces, Qwen full generation, and video remain separate acceptance
work.

## Minimum Acceptance Sequence

1. Keep the `/v1/samplers` support matrix current per backend.
2. Implement exact Comfy `uni_pc` semantics separately from `uni_pc_bh2`; do
   not alias generic `uni_pc` to bh2 until artifact evidence passes.
3. Keep `uni_pc`/`uni_pc_bh2` with `sgm_uniform` marked bounded despite current
   `job-0364`/`job-0365` product artifacts; Comfy catalog allowance and bounded
   combo evidence are not enough to flip accepted sampler parity for the wider
   catalog.
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
