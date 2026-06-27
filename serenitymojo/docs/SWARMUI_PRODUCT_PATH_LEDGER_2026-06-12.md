# SwarmUI Product Path Ledger

Date: 2026-06-12

Purpose: turn `SWARMUI_PARITY_AUDIT_MOJOLIB_MOJODIFFUSION_2026-06-12.md`
into buildable work. This ledger is not a readiness claim. It is the product
contract for making the Mojo stack behave like a local SwarmUI-class generator
while preserving the repo standard: real artifacts, timings, VRAM, dtype
boundaries, and honest labels.

Source acceptance docs:

- `/home/alex/serenityUI/SWARMUI_GAP_AUDIT_2026-06-10.md`
- `/home/alex/serenityUI/GENSCREEN_PARITY_PLAN.md`
- `/home/alex/serenityUI/SERENITYUI_TODO.md`
- `serenitymojo/docs/SUPPORTED_FEATURE_FOUNDATION_2026-06-16.md`

No-CUDA gate:

```bash
python3 scripts/check_swarmui_product_path_contract.py
python3 scripts/check_swarmui_product_path_contract.py --strict
python3 scripts/check_swarmui_product_path_contract.py --strict-all
```

`--strict` fails while P0 product-path blockers remain. `--strict-all` fails
until every SwarmUI-level blocker in this ledger has evidence.

## Acceptance Gates

A feature is accepted only when the product path has current evidence:

- Daemon/API/UI path, not only a standalone parity helper.
- Real output artifact with verified dimensions and metadata.
- Progress and cancellation behavior for daemon-backed generation.
- Persistent job/gallery state when the feature touches history or reuse.
- Text/conditioning seconds, denoise seconds and seconds/step, VAE/decode
  seconds, total wall time, and peak VRAM.
- BF16/F16/FP8 storage dtype boundaries preserved unless the reference requires
  otherwise.
- Fail-loud unsupported options before CUDA setup where possible.
- Explicit readiness label: `not ready`, `smoke runnable`, `experimental`, or
  `production ready`.
- Pure Mojo product runtime. Python may be used for dev support, parity oracles,
  inspection, and audit tooling, but not as the shipped generation pipeline.

For video, also require frame count, duration, resolution, muxing, and audio
behavior verification.

2026-06-16 foundation update: feature admission for the Rust `/v1/generate`
route is now centralized in `GET /v1/capabilities` with schema
`serenity.capabilities.v1`. The endpoint and shared prequeue guard now live in
`serenity-server/crates/server/src/capabilities.rs`, not as more growth in
`main.rs`. The same tables gate `/v1/preflight` and `/v1/generate`, and they
record admitted dimensions, sampler/scheduler subsets, negative-prompt support,
LoRA limits, Ideogram bbox prompt JSON, and disabled features. The current route
remains txt2img-only: image-to-image, inpaint, image conditioning, VAE override,
hires two-pass, refiner, upscale, outpaint, ControlNet, and video are
`supported:false` with `policy:"fail_loud"`.

The no-CUDA product gate now also derives fail-loud cases from the live
capability payload. The current
`output/checks/serenity_server_t2i_product_gate_prequeue_latest.json` report has
88 generated `capability_rejections` and no
`failed_capability_rejection_cases`, covering disabled fields such as
image-to-image, ControlNet, VAE override, bbox `prompt_json` on non-Ideogram
models, Ideogram negative prompts, Flux multi-LoRA, workflow-lowered image
conditioning fields, and workflow-lowered outpaint/LanPaint fields.

`/v1/preflight` now also returns a per-request `capability_profile` with schema
`serenity.capability_profile.v1`. The no-CUDA report records green profile
cases including Z-Image admitted, Qwen bounded txt2img admitted, Klein/Flux2
blocked, Ideogram negative-prompt rejection with an admitted profile, and
Z-Image raw ControlNet rejection with an admitted profile, plus Z-Image workflow
unsupported-node rejection with an admitted profile and
`rejection_stage:"workflow_lowering"`. This keeps tools from guessing which
feature matrix applied to a rejected request.

Canvas-side admission now consumes the same capability contract through
`serenity-server/canvas/js/api.js`. The adapter caches `/v1/capabilities`,
emits `capabilities:loaded`, and exposes shared feature helpers used by the
param rail, prompt bar, refiner/upscale panel, and gallery actions. The browser
still submits txt2img sentinel values for disabled surfaces; image-to-image and
upscale gallery actions stay hidden until the server capability payload and flat
submit body are both widened.

Workflow-derived disabled fields now hit the same front-door Rust guard after
graph lowering. Active `init_image`, `mask_image`, `conditioning_mask_image`,
`inpaint_conditioning_image`, `reference_image`, `outpaint_left`,
`threshold_mask_value`, and LanPaint metadata return HTTP 400 before job
creation. The current no-CUDA report includes passing
`*_image_conditioning_disabled` and `*_outpaint_lowered_fields_disabled` cases
with unchanged `/v1/jobs` counts.

## P0.1 Image Fast Path

Goal: image generation must use the newest mojodiffusion runtime kernels in the
SwarmUI-facing generation path.

Current blockers:

- 2026-06-27 update: Qwen-Image txt2img is product-admitted only for the bounded
  1024x1024 Euler/simple route. The runtime now keeps CFG on-device, avoids
  per-call bias clones in Qwen linear projections, uses the shared no-affine
  LayerNorm path, repacks Qwen middle text padding into a cuDNN flash-SDPA tail
  pad for BF16 product attention, and can opt into resident block pinning with
  `QWENIMAGE_PIN_RESIDENT_BYTES`. `sdpa_qwen_keymask_parity.mojo` now gates the
  flash repack/scatter helper against the existing keymask implementation on
  semantic rows. The Qwen sample CLI now uses the same process-death principle
  as the worker encoder path: a self-reexec `encode-child` writes BF16 caps and
  exits before the parent loads DiT, with observed parent VRAM before DiT at
  `1558 MiB` used / `22520 MiB` free for the one-step 1024px smoke. This is not
  accepted speed parity, and Qwen-Image-Edit, LoRA, img2img, and broad
  sampler/scheduler aliases still fail loud.
- Z-Image source routing now uses the cuDNN flash helper for no-saved inference
  forwards and has current daemon artifact/timing/VRAM evidence. Speed and full
  sampler parity are still not accepted.
- Existing speed docs record Z-Image paired Mojo at about `5.2067s/step` and
  `22238 MiB` versus OneTrainer at about `2.1440s/step` and `14340 MiB`.

Current fixes:

- Z-Image no-saved product forwards route through `_zimage_sdpa_product_fwd`,
  which calls the cuDNN flash SDPA shim. `zimage_refine_x_seq` and the
  refined-main stack now use no-saved device refiner forwards for generation.
- The flash SDPA parity/speed gate passes on the accepted real shapes, including
  Z-Image padded `B=1,S=1248,H=30,Dh=128` and `B=2,S=1248,H=30,Dh=128`.
- Z-Image result manifests no longer hardcode `peak_vram_mib: 0`; the generator
  samples the active `DeviceContext` memory high-water mark and the AOT build
  passes. Current daemon smoke evidence (`job-0028`) records positive timings,
  requested/executed sampler fields, and `peak_vram_mib`; speed parity is still
  not accepted.
- Qwen's tiny attention gate passes:
  `pixi run mojo run -I .
  serenitymojo/ops/parity/sdpa_qwen_keymask_parity.mojo`. It compares F32,
  BF16, and no-pad cases against the old full-mask SDPA reference with cos
  `0.99999999999999+` and max abs <= `2.24e-08`.

Required implementation:

- Replace remaining product sampler attention math-mode calls with the accepted
  fast path: CUDLASS/CUTLASS-class kernels, the cuDNN flash shim, or a
  repo-approved equivalent.
- Batch CFG cond/uncond instead of running two serial main-stack passes.
- Keep BF16 storage at tensor boundaries; F32 only inside compute.
- Preserve the same LoRA target mapping used by training and sampler validation.
- Keep peak VRAM measured in the Mojo product generation runner and write a
  positive value into the result manifest.

Acceptance evidence:

- `python3 scripts/check_swarmui_product_path_contract.py --strict` no longer
  reports image-fast-path P0 blockers.
- 2026-06-12 current checker:
  `python3 scripts/check_swarmui_product_path_contract.py --write-readiness
  output/checks/swarmui_product_path_readiness.json` reports
  `checks=78 passed=78 p0=0 p1=0 p2=0` after implementing the LTX2 fast
  attention route, cuDNN upsampler/video-VAE/audio-VAE decode gates, bounded
  video/A-V artifact gates, structured LTX2 runner stage timing manifests, and
  bounded Ideogram4 PNG metadata and prequeue fail-loud option markers.
  Product P0 and tracked P1 gates are ready.
  Full SwarmUI all-level parity remains blocked by Qwen-Edit/wider Qwen sampler
  coverage, full video parity beyond DEV-smoke artifacts, advanced workflow node families, sampler breadth,
  and Z-Image speed parity. `images=N` now
  emits indexed serial jobs, variation noise has a runtime artifact gate, and
  `/v1/samplers` exposes a pure-Mojo SwarmUI/Comfy sampler support registry.
- 2026-06-12 Ideogram4 fast-attention update: the resident and reference
  Ideogram4 DiT attention calls now route through `ideogram4_sdpa_product_fwd`
  instead of direct `sdpa_nomask[1,S,18,256]`. The forward-only Dh=256 cuDNN
  flash gate in `serenitymojo/ops/tests/sdpa_flash_parity.mojo` passed with
  `ideogram4_fwd_aligned` (`S=1024`, flash `0.5990126 ms`, math `5.4428871 ms`,
  `9.086x`) and `ideogram4_fwd_pad` (`S=1153`, flash `0.9248282 ms`, math
  `7.141845 ms`, `7.722x`). Full DiT fixture probes also passed after the
  wiring: `chunk6 full DiT velocity parity` and resident fp8 DiT both reported
  `cos=0.9995574960620331`, `max_abs=0.4228515625`, `PASS`. This clears the
  immediate Ideogram4 slow-SDPA blocker. The backend/artifact gate now has only
  bounded one-step smoke evidence, not full backend parity.
  Z-Image now has bounded DPM++ 2M, generic UniPC bh1/order<=3, and UniPC
  bh2 wiring on admitted simple and `sgm_uniform` schedules, including
  dedicated DPM++ 2M + `sgm_uniform` artifact evidence; true Comfy-style latent
  batch execution remains unimplemented and now fails loud for
  `Empty*LatentImage.batch_size>1` and `RepeatLatentBatch` before enqueue. The
  remaining ancestral, SDE, Karras, CFG++, and advanced daemon denoise loops
  remain sampler/runtime gaps.
- 2026-06-12 Ideogram4 bounded daemon artifact evidence:
  `python3 scripts/check_ideogram4_daemon_product_contract.py --artifact
  output/serenity_daemon/job-0106.png --json` passes with
  `bounded_artifact_ready:true` and `runtime_acceptance:false`. It validates
  `output/serenity_daemon/job-0106.png`, PNG `serenity.genparams.v1`, gallery
  readback, plus
  `output/serenity_daemon/job-0106.png.ideogram4_daemon_result.json`; the PNG is
  `1024x1024`. The sidecar manifest records backend
  `ideogram4_daemon`, model `ideogram-4-fp8`, readiness `experimental`,
  `requested_sampler:"euler"`, `requested_scheduler:"logitnormal"`,
  `executed_sampler:"ideogram4_logitnormal_euler"`,
  `executed_scheduler:"ideogram4_logitnormal"`, sigma trace
  `[0.9994472,0.00012339458]`, prompt tokens `15`, fixed text window `1024`,
  `lora_count:0`, `variation_applied:false`,
  `dtype:"fp8_transformer_bf16_activations_f32_latent"`,
  `accepted_sampler_parity:false`, and `accepted_speed_parity:false`. Mojo
  timings were `load_seconds=75.44370386599999`,
  `text_encode_seconds=135.538652776`, `prepare_seconds=1.928370351`,
  `denoise_seconds=6.174464852`, `vae_decode_seconds=2.190651774`,
  `total_wall_seconds=221.428687069`, and `peak_vram_mib=22088.6875`.
  Transformers are resident across denoise steps but not across jobs on the
  24GB-class GPU. This is one-step path/resource evidence with PNG metadata; it
  does not prove 20-step quality, sampler breadth, feature support for negative
  prompts/LoRA/img2img/variation, speed parity, or full SwarmUI backend
  acceptance.
- 2026-06-12 Ideogram4 bounded fail-loud option smoke passed:
  `python3 scripts/check_ideogram4_daemon_product_contract.py
  --fail-loud-smoke --write-readiness
  output/checks/ideogram4_daemon_product_readiness.json --json`. It proved
  HTTP `422` prequeue rejection for negative prompt, LoRA, prompt LoRA tag, init
  image, non-default creativity/denoise, variation, unsupported size,
  unsupported sampler, unsupported scheduler, and nonpositive CFG. `/v1/jobs`
  stayed at `126` rows, each request completed in under `0.001s`, and the log
  recorded `expensive_markers_seen:[]` for Ideogram Qwen/DiT/VAE load markers.
- 2026-06-16 Ideogram4 structured prompt update: `/v1/generate` now normalizes
  top-level `prompt_json` for the Ideogram backend before generic prompt syntax
  handling. Strings are used directly; JSON object/array values are serialized
  into `prompt` and `prompt_raw`, preserving
  `compositional_deconstruction.elements[*].bbox` arrays for the model. The
  Rust server now mirrors the same normalization for `/v1/preflight` and
  `/v1/generate`, and admits the bounded Ideogram `simple` scheduler before
  enqueue. The bounded workflow importer accepts the same `prompt_json` override
  for raw Ideogram Comfy exports and still fails loud on prompt-builder subgraphs
  unless a top-level prompt override is supplied. The static guards are
  `python3 scripts/check_ideogram4_daemon_product_contract.py`,
  `cargo test -p serenity-server -p serenity-graph`, and
  `python3 scripts/check_workflow_node_surface.py`.
  No new Ideogram PNG was generated in this update, so runtime acceptance still
  rests on the older one-step artifact and remains `experimental`.
- `/v1/samplers` endpoint smoke saved
  `output/checks/samplers_endpoint_smoke.json`; it returned schema
  `serenity.samplers.v1`, `accepted_sampler_parity:false`, 45 catalog samplers,
  15 catalog schedulers, and support entries for `zimage`, `qwenimage`, and
  `ideogram4`.
  Refreshed evidence from the compiled stub daemon now shows Z-Image endpoint
	  support for `euler`, `flowmatch_euler`, `flow_match_euler`, `dpmpp_2m`,
	  `dpm++ 2m`, `uni_pc`, and `uni_pc_bh2`; Qwen advertises the bounded
	  `qwenimage_flowmatch_euler` / `qwenimage_simple_flowmatch` txt2img route.
  Ideogram4 exposes bounded `euler`/flow-match aliases that execute as
  `ideogram4_logitnormal_euler`, with `logitnormal`/`logit_normal`/
  `ideogram_logitnormal`/`ideogram4_logitnormal` scheduler aliases executing as
  `ideogram4_logitnormal` and `simple`/`flowmatch`/`flow_match`/
  `simple_flowmatch` scheduler aliases executing as the bounded
  `ideogram4_simple_flowmatch` path.
  Current Z-Image `sgm_uniform` wording covers Euler/flow-match Euler, DPM++ 2M,
  `uni_pc`, and `uni_pc_bh2`: the Comfy oracle applies
  `DISCARD_PENULTIMATE_SIGMA_SAMPLERS` prep for both UniPC names before
  SigmaConvert math. This is bounded support and leaves
  `accepted_sampler_parity:false`.
- 2026-06-13 Z-Image UniPC `sgm_uniform` runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py --daemon
  output/bin/serenity_daemon --timeout 900 --steps 1 --skip-dpmpp2m-smoke
  --skip-dpmpp2m-sgm-uniform-smoke --skip-generic-unipc-smoke
  --skip-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-img2img-smoke --skip-multi-lora-smoke
  --write-readiness output/checks/zimage_unipc_sgm_uniform_product_readiness.json`
  passed. It emitted unsupported sampler `job-0361`, baseline `job-0362`,
  Euler `sgm_uniform` `job-0363`, generic UniPC `sgm_uniform` `job-0364`, and
  UniPC bh2 `sgm_uniform` `job-0365`. `job-0364` and `job-0365` both record
  `executed_scheduler:"sgm_uniform_flowmatch"`,
  `sigma_trace:[1.0,0.96028626,0.90089285,0.8023738,0.0]`,
  `txt2img_initial_noise_scale:0.70710677`,
  `sigma_parameterization:"SigmaConvert"`,
  `schedule_source:"zimage_comfy_sgm_uniform_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps"`,
  `unipc_update_steps:4`, `unipc_corrector_steps:3`, and
  `unipc_second_order_steps:2`. `job-0364` records `solver_variant:"bh1"` and
  `job-0365` records `solver_variant:"bh2"`. The readiness report is ready with
  no blockers, and still leaves `accepted_sampler_parity:false`.
- 2026-06-16 Z-Image DPM++ 2M `sgm_uniform` runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py --daemon
  output/bin/serenity_daemon --timeout 900 --width 512 --height 512 --steps 1
  --cfg 1.0 --min-free-vram-mib 21000 --skip-unsupported-smoke
  --skip-sgm-uniform-smoke --skip-sgm-uniform-unipc-smoke
  --skip-sgm-uniform-unipc-bh2-smoke --skip-dpmpp2m-smoke
  --skip-generic-unipc-smoke --skip-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-img2img-smoke --skip-multi-lora-smoke
  --write-readiness output/checks/zimage_dpmpp2m_sgm_uniform_readiness.json`
  passed. It emitted baseline `job-0879` and DPM++ 2M `sgm_uniform`
  `job-0880`. `job-0880` wrote `output/serenity_daemon/job-0880.png` and
  `output/serenity_daemon/job-0880.png.zimage_daemon_result.json`; the manifest
  records `requested_sampler:"dpmpp_2m"`,
  `requested_scheduler:"sgm_uniform"`, `executed_sampler:"dpmpp_2m"`,
  `executed_scheduler:"sgm_uniform_flowmatch"`,
  `sigma_trace:[1.0,0.9477647,0.85859877,0.67192113,0.0]`,
  `schedule_source:"zimage_comfy_sgm_uniform_sigmas"`,
  `dpmpp_update_steps:4`, `dpmpp_second_order_steps:3`,
  `denoise_seconds_per_step:0.31631403375`, `peak_vram_mib:22239.0625`, and
  `accepted_sampler_parity:false`.
- 2026-06-16 Z-Image `karras` fail-loud evidence:
  `output/checks/serenity_server_t2i_product_gate_prequeue_latest.json`
  includes `zimage_karras_scheduler`. The Rust server rejects
  `scheduler:"karras"` before job fanout with HTTP 400, keeps job count
  unchanged, and `/v1/samplers` omits `karras` from Z-Image
  `supported_schedulers`. This is not Karras runtime support; it prevents a
  silent unsupported scheduler path until a real scheduler builder and
  artifact/timing/VRAM gate exist.
- 2026-06-12 DPM++ 2M runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py
  --skip-unsupported-smoke --skip-dpmpp2m-sgm-uniform-smoke
  --skip-generic-unipc-smoke --skip-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-img2img-smoke --skip-multi-lora-smoke
  --write-readiness
  output/checks/zimage_dpmpp2m_product_readiness.json`
  passed. It emitted baseline `job-0035` and DPM++ `job-0036`. `job-0036`
  wrote `output/serenity_daemon/job-0036.png` and
  `output/serenity_daemon/job-0036.png.zimage_daemon_result.json`; the manifest
  records `requested_sampler:"dpmpp_2m"`,
  `requested_scheduler:"flowmatch"`, `executed_sampler:"dpmpp_2m"`,
  `executed_scheduler:"simple_flowmatch"`, `steps_executed:4`,
  `dpmpp_update_steps:3`, `dpmpp_second_order_steps:2`,
  `denoise_seconds_per_step:0.3180188945`, `peak_vram_mib:21571.8125`, and
  `accepted_sampler_parity:false`. The duplicate terminal `0.0 -> 0.0` sigma
  interval is skipped by the DPM++ branch and not counted as an update.
- 2026-06-13 UniPC bh2 runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py
  --daemon output/bin/serenity_daemon --timeout 900 --steps 1
  --skip-unsupported-smoke --skip-sgm-uniform-smoke
  --skip-sgm-uniform-unipc-smoke --skip-sgm-uniform-unipc-bh2-smoke
  --skip-dpmpp2m-smoke --skip-dpmpp2m-sgm-uniform-smoke
  --skip-generic-unipc-smoke --skip-multi-image-smoke --skip-variation-smoke
  --skip-img2img-smoke --skip-multi-lora-smoke
  --write-readiness output/checks/zimage_unipc_bh2_comfy_product_readiness.json`
  passed after moving bh2 onto the Comfy SigmaConvert/discard-prep path. It
  emitted baseline `job-0366` and UniPC bh2 `job-0367`. `job-0367` wrote
  `output/serenity_daemon/job-0367.png` and
  `output/serenity_daemon/job-0367.png.zimage_daemon_result.json`; the manifest
  records `requested_sampler:"uni_pc_bh2"`,
  `requested_scheduler:"flowmatch"`, `executed_sampler:"uni_pc_bh2"`,
  `executed_scheduler:"simple_flowmatch"`, `solver_type:"bh2"`,
  `solver_variant:"bh2"`, `solver_order:3`,
  `sigma_parameterization:"SigmaConvert"`,
  `schedule_source:"zimage_comfy_simple_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps"`,
  `txt2img_initial_noise_scale:0.70710677`, `unipc_update_steps:4`,
  `unipc_corrector_steps:3`, `unipc_second_order_steps:2`,
  `peak_vram_mib:21485.2`, and `accepted_sampler_parity:false`.
  Generic `uni_pc` is not an alias for this accepted `uni_pc_bh2` bh2 variant;
  the focused Mojo semantic gate
  `serenitymojo/sampling/parity/comfy_unipc_semantics_gate.mojo` proves generic
  `uni_pc` is `bh1`, order `min(3,len(sigmas)-2)`, SigmaConvert, Comfy
  penultimate-sigma discard, final-zero-replacement, and initial-noise scaling.
- 2026-06-12 Z-Image generic UniPC runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py --timeout 900 --steps
  1 --skip-unsupported-smoke --skip-dpmpp2m-smoke
  --skip-dpmpp2m-sgm-uniform-smoke --skip-unipc-smoke
  --skip-multi-image-smoke --skip-variation-smoke --skip-img2img-smoke
  --skip-multi-lora-smoke --write-readiness
  output/checks/zimage_daemon_generic_unipc_readiness.json`
  passed. It emitted bounded artifact
  `output/serenity_daemon/job-0077.png` plus
  `output/serenity_daemon/job-0077.png.zimage_daemon_result.json`; the manifest
  records `requested_sampler:"uni_pc"`, `executed_sampler:"uni_pc"`,
  `executed_scheduler:"simple_flowmatch"`, `solver_type:"bh1"`,
  `solver_variant:"bh1"`, `solver_order:3`,
  `sigma_parameterization:"SigmaConvert"`,
  `schedule_source:"zimage_build_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps"`,
  `steps_executed:4`, `unipc_update_steps:4`, `unipc_corrector_steps:3`,
  `unipc_second_order_steps:2`, `denoise_seconds_per_step:0.29673903050000006`,
  `total_wall_seconds:3.872930771`, `peak_vram_mib:21379.875`, and
  `accepted_sampler_parity:false`.
- 2026-06-12 Z-Image img2img/creativity runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py --timeout 900
  --steps 1 --skip-unsupported-smoke --skip-dpmpp2m-smoke
  --skip-dpmpp2m-sgm-uniform-smoke --skip-generic-unipc-smoke
  --skip-unipc-smoke --skip-multi-image-smoke --skip-variation-smoke
  --skip-multi-lora-smoke --write-readiness
  output/checks/zimage_img2img_creativity_readiness.json` passed after
  `pixi run build-daemon`. It emitted baseline
  `output/serenity_daemon/job-0087.png` plus img2img artifacts
  `job-0088.png`, `job-0089.png`, and `job-0090.png`, reusing `job-0087.png`
  as `init_image`. The checker validates PNG `serenity.genparams.v1`, sidecar
  manifests, `init_image`, `creativity`, `img2img_applied:true`,
  `denoise_start_step`, `steps_executed`, `denoise_update_steps`,
  sigma-derived start semantics, timings, and positive VRAM. Evidence:
  creativity `0.0` -> start step `8`, `steps_executed:0`,
  `total_wall_seconds=3.164280578`; creativity `0.5` -> start step `6`,
  `steps_executed:1`, `total_wall_seconds=3.527565501`; creativity `1.0` ->
  start step `0`, `steps_executed:7`, `total_wall_seconds=5.312134054`.
  Peak VRAM was `21393.25 MiB`, and the
  report keeps `accepted_img2img_parity:false`, `accepted_sampler_parity:false`,
  and `accepted_speed_parity:false`. This is bounded flat-parameter Z-Image
  img2img evidence, not mask/inpaint, graph `LoadImage`/`VAEEncode`, full
  denoise semantics, or quality parity.
- Latest full Z-Image daemon product gate after terminal-zero scheduler
  accounting:
  `python3 scripts/check_zimage_daemon_product_contract.py --timeout 900
  --steps 1 --skip-dpmpp2m-sgm-uniform-smoke
  --write-readiness output/checks/zimage_daemon_product_readiness.json`
  passed. It covered unsupported sampler failure (`job-0091`), baseline
  `job-0092`, img2img jobs `job-0093` through `job-0095`, DPM++ 2M `job-0096`,
  generic UniPC `job-0097`, UniPC bh2 `job-0098`, variation `job-0099`,
  multi-image, and multi-LoRA `job-0101`. The latest img2img manifest evidence
  records `0.0 -> denoise_start_step:8, steps_executed:0`,
  `0.5 -> denoise_start_step:6, steps_executed:1`, and
  `1.0 -> denoise_start_step:0, steps_executed:7`, with
  `accepted_img2img_parity:false` and `accepted_speed_parity:false`.
- 2026-06-12 Z-Image multi-LoRA runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py
  --skip-unsupported-smoke --skip-dpmpp2m-smoke
  --skip-dpmpp2m-sgm-uniform-smoke --skip-generic-unipc-smoke
  --skip-unipc-smoke --skip-multi-image-smoke --skip-variation-smoke
  --skip-img2img-smoke --write-readiness
  output/checks/zimage_multi_lora_product_readiness.json` passed. It emitted
  baseline `job-0044`, single-LoRA `job-0045`, and stacked-LoRA `job-0046`.
  `job-0046` wrote `output/serenity_daemon/job-0046.png`; the manifest records
  `lora_count:2`, `lora_merge_strategy:"rank_concat_scaled_b"`, weights
  `0.65` and `0.35`, resolved paths for `EriZimageLora.safetensors` and
  `gigerRegularLora.safetensors`, `peak_vram_mib:21799.3125`, and a stacked
  IDAT hash distinct from both the baseline and single-LoRA outputs. This
  accepts Z-Image PEFT/Comfy LoRA stacking only; Qwen LoRA and Z-Image
  LoKr/LyCORIS conversion remain unaccepted.
- A daemon-driven Z-Image run emits PNG + result manifest with positive timings
  and peak VRAM.
- The manifest states whether speed parity is accepted. It must remain false
  until paired baseline evidence says otherwise.

## P0.2 Daemon Product Gate

Goal: the daemon is the product generation gate, not a side experiment.

Current status:

- The daemon exposes `/v1/generate`, `/v1/jobs`, `/v1/job/<id>`,
  `/v1/cancel/<id>`, `/v1/models`, `/v1/health`, and WebSocket progress.
- Stub generation was smoke-tested with PNG metadata and `jobs.db` readback.
- Real Z-Image daemon generation now has current GPU evidence. Qwen daemon
  txt2img is admitted only for bounded 1024x1024 Euler/simple; Qwen-Image-Edit,
  LoRA, img2img, and wider sampler/scheduler controls remain fail-loud.
- Z-Image flat `init_image`/`creativity` now has bounded runtime artifact
  evidence through the daemon. Full image-node/mask/inpaint graph parity remains
  unaccepted.
- Ideogram4 now has a native daemon backend with a bounded one-step 1024x1024
  artifact and sidecar manifest. It is still experimental and does not prove
  full metadata, sampler, speed, or quality parity.
- Stub mode currently links CUDA and the cuDNN SDPA cshim because real backends
  are imported.

Required implementation:

- Add a standard daemon smoke command or script that builds with the exact link
  flags and exercises health, model scan, generate, jobs, PNG metadata, and DB
  readback.
- Add real-backend smoke mode for admitted image routes only. Qwen must stay
  scoped to the bounded 1024x1024 Euler/simple gate until artifact, timing, VRAM,
  quality, and sampler evidence justify widening it.
- Exercise WebSocket progress and cancel against a real backend.
- Keep process isolation as the default path for model switching after GPU VRAM
  reclaim is proven.

Acceptance evidence:

- Product smoke report with command lines, artifact paths, dimensions, metadata,
  timings, VRAM, and daemon endpoint responses.
- Current GPU runtime gate command:
  `python3 scripts/check_zimage_daemon_product_contract.py --cancel-smoke
  --write-readiness output/checks/zimage_daemon_product_readiness.json`.
  This starts `output/bin/serenity_daemon zimage <port>`, listens on
  `/v1/progress`, submits a bounded 512x512 Z-Image job through
  `/v1/generate`, validates the PNG `serenity.genparams.v1`, `jobs.db`,
  gallery/read endpoints, manifest timings, positive `peak_vram_mib`,
  unsupported-sampler fail-loud behavior, DPM++ 2M on simple and `sgm_uniform`
  schedules, variation noise output change, multi-image serial output, and a
  running-job cancel smoke.
- 2026-06-12 current run passed: `job-0028` emitted
  `output/serenity_daemon/job-0028.png` plus
  `output/serenity_daemon/job-0028.png.zimage_daemon_result.json`; timings were
  `load_seconds=2.4113649049999997`, `text_encode_seconds=2.63576205`,
  `denoise_seconds=1.221294356`, `vae_decode_seconds=0.696861333`,
  `total_wall_seconds=7.447758765`, and `peak_vram_mib=21510.625`.
  The manifest records `requested_sampler:"euler"`,
  `requested_scheduler:"flowmatch"`, `executed_sampler:"flowmatch_euler"`,
  `executed_scheduler:"simple_flowmatch"`, `variation_applied:false`,
  `image_index:0`, and `image_count:1`.
  The same gate proved variation noise as `job-0029` with
  `variation_seed:20261319`, `variation_strength:0.55`, and a changed PNG IDAT
  payload hash (`fecbe64e...` baseline to `59a22b92...` variation). It also
  proved `images=2` serial output as `job-0030` and `job-0031`
  with seeds `20260812` and `20260813`, `image_index` values `0` and `1`, and
  `image_count:2` in PNG metadata and result manifests. The historical
  unsupported sampler smoke targeted `uni_pc_bh2` before that alias was
  promoted; the current focused UniPC gate targets generic `uni_pc` and proved
  fail-loud behavior as `job-0038`. `job-0032` was cancelled through
  `/v1/cancel/<id>` while running.
- 2026-06-12 1024 tiled-decode run passed:
  `python3 scripts/check_zimage_daemon_product_contract.py --width 1024
  --height 1024 --steps 1 --skip-unsupported-smoke --skip-dpmpp2m-smoke
  --skip-dpmpp2m-sgm-uniform-smoke --skip-generic-unipc-smoke
  --skip-unipc-smoke --skip-multi-image-smoke --skip-variation-smoke
  --skip-img2img-smoke --skip-multi-lora-smoke
  --write-readiness
  output/checks/zimage_1024_product_readiness.json`. It emitted
  `output/serenity_daemon/job-0073.png` plus
  `output/serenity_daemon/job-0073.png.zimage_daemon_result.json`, with PNG
  `serenity.genparams.v1`, gallery/read endpoints, a jobs DB row, and manifest
  timings. The manifest records `resolution:1024x1024`,
  `requested_sampler:"euler"`, `executed_sampler:"flowmatch_euler"`,
  `requested_scheduler:"flowmatch"`, `executed_scheduler:"simple_flowmatch"`,
  `load_seconds=2.49144893`, `text_encode_seconds=3.132651268`,
  `denoise_seconds=2.078721406`, `vae_decode_seconds=3.266865353`,
  `total_wall_seconds=11.438153728`, and `peak_vram_mib=21497.3125`.
  This proves the daemon 1024 path uses the tiled 64-latent VAE decoder instead
  of the former whole-frame 128-latent decoder. It is still a 1-step
  experimental path/resource proof, not quality, sampler, or speed parity.
- 2026-06-12 Ideogram4 bounded daemon run passed:
  `python3 scripts/check_ideogram4_daemon_product_contract.py --artifact
  output/serenity_daemon/job-0106.png --json`. It validated
  `output/serenity_daemon/job-0106.png`, embedded PNG
  `serenity.genparams.v1`, `/v1/gallery/read` params readback, and the sidecar
  manifest `output/serenity_daemon/job-0106.png.ideogram4_daemon_result.json`,
  with `1024x1024` dimensions, `readiness_label:"experimental"`,
  `executed_sampler:"ideogram4_logitnormal_euler"`,
  `executed_scheduler:"ideogram4_logitnormal"`, fixed `1024` token text window,
  positive timings, and `peak_vram_mib=22088.6875`. The checker intentionally
  reports `runtime_acceptance:false`; this smoke does not accept quality,
  sampler, or speed parity.
- 2026-06-12 Z-Image img2img/creativity run passed:
  `python3 scripts/check_zimage_daemon_product_contract.py --timeout 900
  --steps 1 --skip-unsupported-smoke --skip-dpmpp2m-smoke
  --skip-dpmpp2m-sgm-uniform-smoke --skip-generic-unipc-smoke
  --skip-unipc-smoke --skip-multi-image-smoke --skip-variation-smoke
  --skip-multi-lora-smoke --write-readiness
  output/checks/zimage_img2img_creativity_readiness.json`. It validated
  baseline `job-0087` and img2img jobs `job-0088` through `job-0090` at
  creativity `0.0`, `0.5`, and `1.0` with 8-step img2img settings. The manifest
  records `init_image`, `creativity`, `img2img_applied:true`,
  `denoise_start_step`, `steps_executed`, `denoise_update_steps`, timings, and
  `peak_vram_mib`.
  Acceptance remains bounded: `accepted_img2img_parity:false`.
- No stale source prose describes the daemon as a skeleton.

## P0.3 Video Product Path

Goal: video generation works before UI expansion makes it look complete.

Current blockers:

- Full video parity remains bounded by DEV-smoke quality and sampler/runtime
  scope; do not claim HQ SwarmUI video parity from the artifact gate alone.
- The known F32 random stage-2 video/audio noise fallback in
  `ltx2_t2v_av_hq.mojo` was fixed after this ledger was created; fallback noise
  now stores BF16 and `_refhq_noise_blend` casts internally for F32 math.
- 2026-06-12 update: the daemon now exposes `/v1/video` as an explicit
  readiness/status contract, exposes `/v1/video/probe?path=<mp4>` for real MP4
  metadata (`mp4`, `frame_count`, `duration`, muxing, and audio behavior), and
  `POST /v1/video` can launch the compiled bounded LTX2 staged dev runner when
  `output/bin/ltx2_video_smoke_runner` exists. Both the video-only and
  audio-enabled A/V artifact gates now emit verified MP4s plus positive peak
  VRAM evidence; full video parity still remains separate because the runner
  labels the output as DEV SMOKE quality.
- 2026-06-12 module split: `/v1/video` routing stays in
  `serenity_daemon.mojo`, but the readiness/result/probe implementation moved
  to `serenitymojo/serve/video_api.mojo`. The daemon dropped from 3614 to 3232
  lines while keeping the public endpoint contract unchanged.

Required implementation:

- Keep LTX2/NAVA dtype boundaries guarded so latents/noise return the proper
  storage dtype.
- Keep the bounded product video runner buildable and fail loud when the runner
  is missing or an unsupported runner name is requested.
- Verify MP4 duration, frame count, resolution, muxing, and audio behavior.
- Record stage timings and peak VRAM.

Acceptance evidence:

- 2026-06-12 bounded-runner evidence:
  `pixi run build-daemon` passes after adding and then extracting the video
  route/probe/bounded-runner implementation into `serve/video_api.mojo`.
- Post-split route smoke on `output/bin/serenity_daemon stub 7815`:
  `GET /v1/video` returned `serenity.video_status.v1` with
  `bounded_daemon_smoke`; `GET /v1/video/probe?path=output/serenity_daemon/video-0072/ltx2_t2v_av_stage2_dev_smoke.mp4`
  returned `serenity.video_probe.v1`, `768x512`, `121` frames,
  `audio_behavior=audio_stream_present`, and `muxing=probe_ok`.
- `pixi run build-video-smoke` passes and emits
  `output/bin/ltx2_video_smoke_runner`.
- `python3 scripts/check_ltx2_dtype_contract.py --scope all` passes.
- `python3 scripts/check_swarmui_product_path_contract.py --write-readiness
  output/checks/swarmui_product_path_readiness.json` now reports
  `checks=78 passed=78 p0=0 p1=0 p2=0`. This keeps full video parity blocked
  by evidence while proving the bounded daemon runner route, LTX2 fast SDPA
  route, cuDNN latent upsampler, direct-FCQRS video VAE decode, structured
  runner timing manifest, and tracked UI/gallery/reuse/state P1 gate are wired.
- 2026-06-12 fast-path update: LTX2 AV attention now routes both square
  self-attention and rectangular text/cross-modal attention through the BF16
  cuDNN SDPA shim (`sdpa_flash_train_fwd` /
  `sdpa_flash_train_fwd_rect`) via `_ltx2_sdpa_product_fwd*`; the standalone
  `build-video-smoke` task links `libserenity_cudnn_sdpa`.
- 2026-06-12 upsampler update: the LTX2 latent upsampler now loads checkpoint
  Conv3d weights as FCQRS and Conv2d weights as zero-copy FCQRS views, then
  dispatches both through `conv3d_fcqrs_cudnn`. The old QRSCF/RSCF host
  transpose helpers and naive conv wrapper are removed from the upsampler.
  `output/bin/ltx2_upsampler_smoke` passed with spatial cosine `0.99989617`
  and temporal cosine `0.9995859`.
- 2026-06-12 VAE decode update: the LTX2 video VAE decoder keeps checkpoint
  Conv3d weights in FCQRS/OIDHW layout and passes the resident tensors directly
  to `conv3d_fcqrs_cudnn`; the dead host-side QRSCF transpose helper is removed.
  The isolated target-shape smoke
  `output/bin/ltx2_vae_decode_hq121_smoke` completed `[1,3,121,512,768]`
  decode in `real 24.14s`.
- 2026-06-12 audio VAE update: the LTX2 audio VAE decoder now views checkpoint
  Conv2d OIHW weights as cuDNN FCQRS `[Cout,Cin,kh,kw,1]` tensors and dispatches
  causal conv2d through `conv3d_fcqrs_cudnn`. The old per-conv host
  OIHW-to-QRSCF transpose helper is removed. The isolated audio VAE parity smoke
  `output/bin/ltx2_audio_vae_smoke` passed against the Python oracle with
  cosine `0.999996`, max abs diff `0.03125`, decoded shape `[1,2,29,64]`, and
  `real 1.05s`.
- Historical measured daemon evidence after the LTX2 fast-path and upsampler
  patches:
  `python3 scripts/check_ltx2_video_daemon_product_contract.py --timeout 180
  --write-readiness output/checks/ltx2_video_daemon_readiness.json` reports
  product wiring ready, positive external peak VRAM delta `6639 MiB`, and
  timeout after `[Stage2] done -> decoding at 2x resolution`. This historical
  run moved past the prior upsampler/VAE-load stall but stopped before MP4
  artifact emission, so it was not accepted video parity.
- Current measured daemon video-only evidence:
  `python3 scripts/check_ltx2_video_daemon_product_contract.py --timeout 520
  --weight-mode resident --audio-mode noaudio --strict-artifact --write-readiness
  output/checks/ltx2_video_daemon_resident_readiness.json` passed. It emitted
  `output/serenity_daemon/video-0074/ltx2_t2v_stage2_dev_smoke.mp4`, probe
  metadata `width=768`, `height=512`, `frame_count=121`, `duration=5.041667`,
  `fps=24.0`, `video_codec=h264`, `muxing=probe_ok`,
  `audio_behavior=video_only_no_audio_stream`, and external peak VRAM delta
  `10476 MiB` (`11226 MiB` peak used). The result manifest records
  `mode:"staged lora resident noaudio nonag"`, `weight_mode:"resident"`,
  `total_wall_seconds=174.955233998`, `audio_mode:"noaudio"`,
  `accepted_video_parity:false`, and `accepted_sampler_parity:false`. It also
  records `runner_timing_path`, `runner_timings`, and `stage_timings`;
  measured no-audio timings include
  `stage1_denoise_seconds=57.315346847999535`,
  `stage2_denoise_seconds=75.27145183599896`,
  `video_decode_seconds=23.61521103099949`,
  `frame_png_write_seconds=1.7494108600003528`, and
  `video_mux_seconds=0.6264517420004267`. Compared with the previous stream
  no-audio gate (`total_wall_seconds=202.353546797`,
  `stage1_denoise_seconds=73.01245590500002`,
  `stage2_denoise_seconds=90.72237481299999`, `peak used=10845 MiB`), resident
  mode plus the no-sync resident materializer and clone-pair fence coalescing
  improves the bounded wall time by about `27.4s` at about `381 MiB` higher
  sampled peak used.
- Current LTX2 resident raw-FP8 loader evidence:
  `pixi run mojo build --target-accelerator sm_86 -I .
  serenitymojo/pipeline/ltx2_fp8_resident_smoke.mojo -o
  output/bin/ltx2_fp8_resident_smoke` passes, and
  `output/bin/ltx2_fp8_resident_smoke` passes outside the sandbox. It preloads
  block 4, reports `resident bytes: 386924928 (369 MiB)`, sees
  `fp8 tensor count: 34`, materializes representative video/audio weights as
  BF16, then synchronizes once at the end. The resident materializer now uses
  `fp8_e4m3_dequant_to_bf16_no_sync` to avoid a per-FP8-tensor host/device
  fence; streamed loads keep `fp8_e4m3_dequant_to_bf16` and its synchronization.
- `output/bin/fp8_dequant_smoke` still passes all bit-exact checks after the
  split wrapper: all 256 E4M3 byte values at scales `1.0`, `0.5`, `2.0`, and
  the real checkpoint scale, plus the real block-4 `attn1.to_q` slice against
  the torch reference (`5/5`).
- Current LTX2 clone-pair profile evidence:
  `pixi run build-video-smoke` passes after coalescing staged/refhq paired
  video/audio block-output clones into one same-stream fence. The profile command
  `output/bin/ltx2_video_smoke_runner staged lora resident noaudio nonag profile
  output/ltx2_profile_after_clone_pair 1` emitted
  `output/ltx2_profile_after_clone_pair/ltx2_t2v_stage2_dev_smoke.mp4`
  (`768x512`, `121` frames, `5.041667s`, H.264, no audio stream) and
  `ltx2_runner_timings.json` with `total_runner_seconds=171.75974076999955`,
  `stage1_denoise_seconds=56.413680857000145`,
  `stage2_denoise_seconds=74.39678424900012`, and
  `video_decode_seconds=23.28128546500011`. Compared with the immediately
  previous post-no-sync profile (`total_runner_seconds=314.8831040389996`), this
  is a `1.833x` same-profile runner-speed improvement. This is speed evidence
  for the bounded staged runner, not full video parity.
- Current measured daemon A/V evidence:
  `python3 scripts/check_ltx2_video_daemon_product_contract.py --timeout 520
  --audio-mode audio --strict-artifact --write-readiness
  output/checks/ltx2_video_daemon_audio_readiness.json` passed. It emitted
  `output/serenity_daemon/video-0072/ltx2_t2v_av_stage2_dev_smoke.mp4` plus
  `output/serenity_daemon/video-0072/dev_audio.wav`. Probe metadata reports
  `width=768`, `height=512`, `frame_count=121`, `duration=5.041667`, `fps=24.0`,
  `video_codec=h264`, `audio_codec=aac`, `audio_duration=5.034`,
  `audio_behavior=audio_stream_present`, `stream_count=2`, and `muxing=probe_ok`.
  External peak VRAM delta was `9880 MiB` (`10902 MiB` peak used), and the
  readiness report preserves result `total_wall_seconds=202.223939339`,
  runner `total_runner_seconds=201.75603515000148`, `audio_mode:"audio"`,
  `accepted_video_artifact:true`, `accepted_av_artifact:true`,
  `accepted_video_parity:false`, and `accepted_sampler_parity:false`.
  The runner also writes
  `output/serenity_daemon/video-0072/ltx2_runner_timings.json`; during the A/V
  run the daemon result surfaced `runner_timings` and `stage_timings`, and the
  checker reports `claims_stage_timing_gate:true`. Current measured A/V stage
  timings from `output/checks/ltx2_video_daemon_audio_readiness.json` include
  `stage1_denoise_seconds=70.47008886299955`,
  `stage2_denoise_seconds=89.73880778400053`,
  `video_decode_seconds=24.576268509999863`,
  `audio_vae_seconds=0.2347105040007591`,
  `vocoder_seconds=3.0844138479988032`, and
  `audio_mux_seconds=0.7538969699999143`.
- `POST /v1/video` accepts only `runner:"ltx2_staged_dev_smoke"` and clamps
  `steps` to `1..3`. It accepts `audio_mode:"noaudio"` or `"audio"`, defaulting
  the bounded artifact gate to video-only. It accepts `weight_mode:"resident"`
  or `"stream"` and now defaults to the resident warm-range path instead of
  forcing stream mode. It runs
  `output/bin/ltx2_video_smoke_runner staged lora <weight_mode> <audio_mode>
  nonag output/serenity_daemon/<video-id> <steps>`, writes
  `ltx2_video_runner.log`, and writes `ltx2_video_result.json` with
  `accepted_video_artifact`, `accepted_av_artifact`,
  `accepted_video_parity:false`, `accepted_sampler_parity:false`,
  `weight_mode`, `total_wall_seconds`, output paths, stage timings, and MP4
  probe fields when the runner succeeds. This logic now lives in
  `serenitymojo/serve/video_api.mojo`; the daemon only dispatches the route.
- Mojo artifact probe smoke:
  `pixi run mojo run -I . -I /home/alex/MOJO-libs
  serenitymojo/components/artifacts_smoke.mojo` produced
  `output/artifacts_smoke_seq.mp4`; `/v1/video/probe?path=...` reported
  `width=16`, `height=16`, `frame_count=2`, `duration=0.5`, `fps=4`,
  `video_codec=h264`, `has_audio=false`, and `muxing=probe_ok`.
- Remaining acceptance gap: the bounded LTX2 runner now accepts the audio-enabled
  A/V artifact path and emits structured stage timings, but it is still a
  one-step DEV SMOKE. SwarmUI-level video parity still requires an accepted
  non-smoke/HQ runtime claim and model/sampler parity evidence. The current
  artifact gates remain labeled
  `accepted_video_parity:false`.

## P1.1 Gallery And Reuse Params

Goal: generated and imported images act like SwarmUI gallery items.

Current status:

- PNG tEXt metadata, `jobs.db`, gallery list/read endpoints, arbitrary PNG
  metadata readback, search/filter/sort, lazy pure-Mojo thumbnail cache,
  favorite state, delete, presets/state, reuse generate, and queue mutation are
  runtime-proven by `scripts/check_ui_gallery_reuse_state_contract.py`.
- Reuse provenance metadata, restart-safe job history in `/v1/jobs` and
  `/v1/job/<id>`, indexed external PNG import, gallery rename, and manual
  ordering are now runtime-proven by the same checker.

Required implementation:

- Add preset-derived provenance metadata before claiming preset-source parity.
- Add browser-side control restoration evidence if a frontend claim is made.

Acceptance evidence:

- Generate image, restart daemon/UI, gallery still shows it.
- Read an external PNG with genparams, reuse those params into a new job, and
  record provenance in the new artifact metadata.
- Indexed import, job history, and gallery ordering/rename pass the runtime
  contract; this tracked P1 is clear.

## P1.2 Model And LoRA Browser

Goal: local model/LoRA browsing feels like a production generator, not a fixed
dropdown.

Current status:

- `/v1/models` scans checkpoints and LoRAs from disk and detects families.
- `/v1/models` exposes browser query params, model cards, LoRA `target_arch`,
  family compatibility metadata, and LoRA search/filter/sort.
- Z-Image accepts multiple loader-supported PEFT/Comfy LoRAs in the daemon and
  records the stack in manifest/PNG metadata.
- Browser UX pieces still not accepted: real preview thumbnails/user notes/model
  favorite persistence, UI restoration, stack reorder/enable controls, Qwen
  LoRA, and Z-Image LoKr/LyCORIS conversion.

Required implementation:

- Add UI multi-LoRA stack controls and extend runtime support to Qwen and
  Z-Image LoKr/LyCORIS only when tensor-target conversion is proven.
- Keep fail-loud incompatible model/LoRA combinations and expand compatibility
  beyond family-level metadata when tensor-target inventories are available.

Acceptance evidence:

- Model scan report includes paths, family tags, LoRA count, and compatibility
  decisions.
- Real generation proves selected model + LoRA stack is honored.

## P1.3 Presets, State, Prompt Syntax, Queue

Goal: repeated generation workflows survive restart and match SwarmUI behavior.

Current blockers:

- Named presets/state endpoints are accepted as of 2026-06-12:
  `/v1/state` and `/v1/presets` persist Mojo-written JSON under
  `output/serenity_daemon/state/` and survived a stub daemon restart smoke.
- Prompt syntax has a daemon parser as of 2026-06-12: raw prompt, resolved
  prompt, weighted spans, LoRA tags, random choices, and wildcard records are
  stored in `serenity.genparams.v1` as `prompt_syntax`. Weighted spans are not
  yet applied to model conditioning math.
- Queue reorder/remove is accepted for active queued jobs as of 2026-06-12;
  interrupt-current semantics still rely on `/v1/cancel/<id>` and need a
  dedicated acceptance smoke against a real backend.

Required implementation:

- Add versioned named presets and last-state persistence. DONE 2026-06-12:
  `serenity.ui_state.v1` and `serenity.presets.v1` are exposed through daemon
  APIs and persisted in the product output state directory.
- Gate prompt syntax either in the UI parser or daemon parser. DONE 2026-06-12:
  daemon parser handles `(text:weight)`, `<lora:name:weight>`,
  `<random:a,b>`, `__name__`, and `<wildcard:name>` with explicit
  `conditioning_weights_applied=false` metadata.
- Add queue reorder/remove and interrupt-current behavior with immutable job
  param snapshots. PARTIAL 2026-06-12: queued-only `/v1/reorder[/<id>]` and
  `/v1/remove[/<id>]` are implemented and stub-smoked.

Acceptance evidence:

- Restart restores last state and named presets. DONE 2026-06-12: wrote state
  and preset on port `18115`, restarted on `18116`, read both back, then
  deleted the smoke preset through `DELETE /v1/presets/<name>`.
- Prompt parser fixture suite passes and generated params store raw + resolved
  prompt. DONE 2026-06-12: stub daemon job `job-0008` wrote PNG metadata with
  `prompt_raw`, resolved `prompt`, `prompt_syntax.weighted`, `random`,
  `wildcards`, and extracted LoRA tag.
- Queue operations have daemon/API tests.
- 2026-06-12 runtime checker:
  `python3 scripts/check_ui_gallery_reuse_state_contract.py --write-readiness
  output/checks/ui_gallery_reuse_state_readiness.json` reports
  `checks=19 passed=19 p0=0 p1=0 p2=0`; it proves reuse provenance, jobs
  history after restart, indexed external import, gallery rename/manual order,
  queue mutation, presets/state restart, favorites, delete, and reuse generate.

## P2 Workflow Graphs And Advanced Controls

Goal: future SwarmUI graph parity without blocking the core local generator.

Current status:

- 2026-06-12 update: daemon `/v1/generate` accepts typed linked workflow graphs
  for the supported text-to-image chain. Supported inputs are
  `workflow.params` / `workflow.genparams` or graph bodies using
  `CheckpointLoaderSimple`, `CLIPTextEncode`,
  `EmptyLatentImage`, `KSampler`, `VAEDecode`, and `SaveImage`. The graph
  executor carries MODEL/CLIP/VAE/CONDITIONING/LATENT/IMAGE placeholders,
  validates typed edges, runs supported nodes in dependency order, and maps the
  result into the existing flat genparams product path.
- Unknown graph formats, unsupported node types, wrong typed links, or cyclic
  linked graphs fail with HTTP 501 and name the unsupported feature. This is a
  fail-loud supported-t2i graph executor, not arbitrary SwarmUI/Comfy graph
  execution.
- ControlNet/IP-Adapter/regional prompting/live preview/upscaler utility paths
  have partial model pieces but no accepted full product surface in this audit.

Required implementation:

- Keep unsupported workflow nodes fail-loud until a real graph executor exists.
- Add utility tabs only when the backend emits real artifacts and result JSON.
- Do not treat model-file presence as UI/product parity.

Acceptance evidence:

- `pixi run build-daemon` passes with the workflow adapter.
- Runtime smoke on `output/bin/serenity_daemon stub`: POSTing a shuffled linked
  t2i graph queued `job-0047`, completed it, wrote
  `output/serenity_daemon/job-0047.png`, and PNG readback showed
  `serenity.genparams.v1` with link-driven prompt/negative prompt, model,
  width/height, steps, seed, cfg, sampler, scheduler, and creativity mapped from
  the typed graph. The PNG IDAT hash was
  `65fce6eb3887ce1186d34b2a22ea8a6229c9fc308ef55ba3a9513271c196fa5e`.
- Unsupported graph smoke returned HTTP 501:
  `{"detail":"unsupported workflow graph node type: NotSupported"}`.
- Wrong typed link smoke returned HTTP 501 with a `workflow graph input`
  expected-type error.
- `python3 scripts/check_swarmui_product_path_contract.py --write-readiness
  output/checks/swarmui_product_path_readiness.json` reports
  `checks=90 passed=90 p0=0 p1=0 p2=0`. Product P0 and tracked P1 are ready.
  Full SwarmUI all-level parity still remains blocked by Qwen-Edit/wider Qwen
  sampler coverage, full video parity beyond DEV-smoke artifacts, advanced
  workflow node families, sampler breadth,
  and Z-Image speed parity.

## 2026-06-16 Worker Result Sidecars for Admitted Image Workers

Current status:

- `serenitymojo/serve/product_manifest.mojo` provides local Mojo helpers for
  result-sidecar JSON escaping, peak VRAM calculation, and text-file writes.
- SDXL, Anima, SD3, and Flux worker backends now emit backend result manifests
  with schema, readiness label, requested/executed sampler and scheduler, phase
  timings, peak VRAM, output paths, and non-acceptance booleans.
- SDXL, Anima, SD3, and Flux also keep `serenity.genparams.v1` PNG metadata
  markers in the static product-path contract. Anima was moved from plain PNG
  save to `encode_png_with_text`.
- Optimized patched workers were built into `output/bin/serenity_worker_sdxl`,
  `output/bin/serenity_worker_anima`, `output/bin/serenity_worker_sd3`, and
  `output/bin/serenity_worker_flux`.

Runtime evidence:

- SDXL strict Rust-server gate:
  `output/checks/sdxl_manifest_gate.json`.
  The PNG is `1024x1024`, carries `serenity.genparams.v1`, and has sidecar
  schema `serenity.sdxl.daemon_result.v1` with
  `peak_vram_mib=12345.0625`, `total_wall_seconds=212.088807925`,
  `denoise_seconds=32.122987534`, and
  `vae_decode_seconds=72.143815548`.
- Anima strict Rust-server gate with optimized worker:
  `output/checks/anima_manifest_gate_o2.json`.
  The PNG is `1024x1024`, carries `serenity.genparams.v1`, and has sidecar
  schema `serenity.anima.daemon_result.v1` with
  `peak_vram_mib=11147.3125`, `total_wall_seconds=93.019964106`,
  `denoise_seconds=2.231944518`, and
  `vae_decode_seconds=88.135556873`.

Limits:

- Superseded by the next section: SD3/Flux live strict gates and the
  all-admitted Rust-server gate have now passed for the bounded product path.
- Current correction, 2026-06-17: the historical Flux lowmem artifact gate is no
  longer sufficient for admission. The real 1024x1024/20-step browser workflow
  gate failed with CUDA OOM at 6/20, so Flux.1-dev is currently blocked from
  `/v1/generate` until the memory gate passes.

## 2026-06-16 All-Admitted Rust-Server Artifact Gate

The previous SD3/Flux live-gate gap is closed for the bounded Rust-server
product path.

Runtime evidence:

- SD3 strict gate:
  `output/checks/sd3_manifest_gate_lowmem.json`.
  The PNG is `1024x1024`, carries `serenity.genparams.v1`, and has sidecar
  schema `serenity.sd3.daemon_result.v1`. The sidecar records
  `released_resident_mmdit_before_vae:true`,
  `vae_decode_tile_grid:"5x5_lowmem"`, `peak_vram_mib=18799.25`,
  `total_wall_seconds=97.526130155`, `denoise_seconds=59.260229214`, and
  `vae_decode_seconds=3.263787052`.
- Flux strict gate:
  `output/checks/flux_manifest_gate_lowmem.json`.
  The PNG is `1024x1024`, carries `serenity.genparams.v1`, and has sidecar
  schema `serenity.flux.daemon_result.v1`. The sidecar records
  `released_resident_dit_before_unpack:true`,
  `vae_decode_tile_grid:"5x5_lowmem"`, `peak_vram_mib=18179.0625`,
  `total_wall_seconds=18.618336994`, `denoise_seconds=12.392615909`, and
  `vae_decode_seconds=3.719806555`.
- All-admitted strict gate:
  `output/checks/all_admitted_manifest_gate_lowmem.json`.
  The gate produced manifest-backed artifacts for `zimage`, `sdxl`, `anima`,
  `sd3`, `flux`, and `ideogram4`; `artifact_only`, `failed_models`,
  `failed_prequeue_cases`, and `failed_sampler_cases` were empty.

All-admitted metrics:

| Model | Artifact | Schema | Size | Peak VRAM MiB | Total wall seconds |
| --- | --- | --- | --- | ---: | ---: |
| ZImage | `output/checks/all_admitted_manifest_gate_lowmem/job-0005.png` | `serenity.zimage.daemon_result.v1` | `512x512` | `22249.3125` | `115.225474649` |
| SDXL | `output/checks/all_admitted_manifest_gate_lowmem/job-0006.png` | `serenity.sdxl.daemon_result.v1` | `1024x1024` | `12327.25` | `26.981791022` |
| Anima | `output/checks/all_admitted_manifest_gate_lowmem/job-0007.png` | `serenity.anima.daemon_result.v1` | `1024x1024` | `10519.3125` | `109.493639988` |
| SD3 | `output/checks/all_admitted_manifest_gate_lowmem/job-0008.png` | `serenity.sd3.daemon_result.v1` | `1024x1024` | `18809.3125` | `25.678156276` |
| Flux | `output/checks/all_admitted_manifest_gate_lowmem/job-0009.png` | `serenity.flux.daemon_result.v1` | `1024x1024` | `22081.25` | `87.31288534` |
| Ideogram4 | `output/checks/all_admitted_manifest_gate_lowmem/job-0010.png` | `serenity.ideogram4.daemon_result.v1` | `1024x1024` | `22225.3125` | `272.823248619` |

Static readiness was refreshed after the runtime gates:

- `output/checks/swarmui_product_path_readiness.json`: `90/90`, P0/P1/P2
  blockers at `0`.
- `output/checks/swarmui_sampler_surface_readiness.json`: `37/37` markers,
  with six broad surface blockers remaining.

Limits:

- This is accepted bounded artifact evidence for the currently admitted
  Rust-server image families. It is not accepted full SwarmUI all-level parity.
- The sidecars still record `accepted_sampler_parity:false` and
  `accepted_speed_parity:false` where applicable.
- Qwen-Edit, wider Qwen sampler coverage, SD15, Chroma, ERNIE, full video,
  advanced workflow surfaces, and broader sampler variants remain blocked or
  explicitly disabled.

## 2026-06-16 SDXL Conditioning Artifact Gate

New reusable checker:

- `scripts/check_serenity_server_conditioning_gate.py`
  - Launches the Rust server and submits same-seed low-CFG, high-CFG, and
    high-CFG-with-negative jobs through `/v1/preflight` and `/v1/generate`.
  - Requires PNG `serenity.genparams.v1`, timing/VRAM sidecars, manifest
    conditioning fields, and distinct IDAT hashes.
  - Keeps `accepted_conditioning_parity:false` and
    `accepted_sampler_parity:false`; this is bounded artifact evidence, not full
    conditioning parity.

Runtime evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_serenity_server_conditioning_gate.py --models sdxl --server-bin serenity-server/target/debug/serenity-server --worker-bin output/bin/serenity_worker_zimage --out-dir output/checks/sdxl_conditioning_gate --write-report output/checks/sdxl_conditioning_gate.json --timeout-per-case 1800 --poll-interval 2
```

Results:

- Report: `output/checks/sdxl_conditioning_gate.json`.
- Artifacts:
  - `output/checks/sdxl_conditioning_gate/job-0001.png`
    (`cfg=1.0`, empty negative).
  - `output/checks/sdxl_conditioning_gate/job-0002.png`
    (`cfg=5.0`, empty negative).
  - `output/checks/sdxl_conditioning_gate/job-0003.png`
    (`cfg=5.0`, negative `red cube, red object, ceramic cube`).
- All three artifacts are `1024x1024`, carry `serenity.genparams.v1`, and have
  `serenity.sdxl.daemon_result.v1` sidecars with requested/executed sampler
  metadata, timings, and peak VRAM.
- IDAT hashes are distinct:
  - low CFG: `127a0b99bf2b0d3e6e981adb5cc4d362e4c2636efab2ff7bd10e3ad24dd0a0b5`
  - high CFG: `4509c2b8e1291f01ebbe6bf3f466dd1d71a2de8399845962558c0dc88c106180`
  - high CFG + negative:
    `2de36eb412c5cc412a2c8d04a0e9eeb57c914001895eaeb438d2adc23303dd28`
- Timings/VRAM:
  - `job-0001`: `total_wall_seconds=65.714914246`,
    `denoise_seconds=2.248967759`, `vae_decode_seconds=4.418024194`,
    `peak_vram_mib=14626.75`.
  - `job-0002`: `total_wall_seconds=10.118343411`,
    `denoise_seconds=2.134382905`, `vae_decode_seconds=3.812047414`,
    `peak_vram_mib=14626.8125`.
  - `job-0003`: `total_wall_seconds=10.136425081`,
    `denoise_seconds=2.134803262`, `vae_decode_seconds=3.828932893`,
    `peak_vram_mib=14621.625`.

The sampler surface checker now tracks this report and refreshes to `38/38`
markers. The conditioning blocker remains because prompt-weight math and
per-model negative/CFG evidence for the remaining admitted families are not
complete.

## 2026-06-16 SD3 Conditioning Artifact Gate

Runtime evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_serenity_server_conditioning_gate.py --models sd3 --server-bin serenity-server/target/debug/serenity-server --worker-bin output/bin/serenity_worker_zimage --out-dir output/checks/sd3_conditioning_gate --write-report output/checks/sd3_conditioning_gate.json --timeout-per-case 1800 --poll-interval 2
```

Results:

- Report: `output/checks/sd3_conditioning_gate.json`.
- Artifacts:
  - `output/checks/sd3_conditioning_gate/job-0001.png`
    (`cfg=1.0`, empty negative).
  - `output/checks/sd3_conditioning_gate/job-0002.png`
    (`cfg=4.5`, empty negative).
  - `output/checks/sd3_conditioning_gate/job-0003.png`
    (`cfg=4.5`, negative `red cube, red object, ceramic cube`).
- All three artifacts are `1024x1024`, carry `serenity.genparams.v1`, and have
  `serenity.sd3.daemon_result.v1` sidecars with requested/executed sampler
  metadata, timings, peak VRAM, and `vae_decode_tile_grid:"5x5_lowmem"`.
- IDAT hashes are distinct:
  - low CFG: `4ac45e63fcd3556b23519f1ddc70d451486abe8bf1223e934a4280f6725dbdf2`
  - high CFG: `43ddbf8aca9320a3d9e5cd439aac418990180910795a7d658b5645727cace83c`
  - high CFG + negative:
    `b74e307ecda6f26af9c7060d079ef244f890f78443d93d456d3d1572056f4b25`
- Timings/VRAM:
  - `job-0001`: `total_wall_seconds=145.002977806`,
    `denoise_seconds=59.192672489`, `vae_decode_seconds=3.179079135`,
    `peak_vram_mib=19091.75`.
  - `job-0002`: `total_wall_seconds=23.027163543`,
    `denoise_seconds=17.439111197`, `vae_decode_seconds=2.894952377`,
    `peak_vram_mib=19091.75`.
  - `job-0003`: `total_wall_seconds=22.699221252`,
    `denoise_seconds=17.222514973`, `vae_decode_seconds=2.757718454`,
    `peak_vram_mib=19079.0`.

The sampler surface checker now tracks both SDXL and SD3 conditioning reports
and refreshes to `39/39` markers. The conditioning blocker remains because
prompt-weight math and per-model negative/CFG evidence for the remaining
admitted families are not complete.

## 2026-06-16 Anima Conditioning Artifact Gate

Runtime evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_serenity_server_conditioning_gate.py --models anima --server-bin serenity-server/target/debug/serenity-server --worker-bin output/bin/serenity_worker_zimage --out-dir output/checks/anima_conditioning_gate --write-report output/checks/anima_conditioning_gate.json --timeout-per-case 1800 --poll-interval 2
```

Results:

- Report: `output/checks/anima_conditioning_gate.json`.
- Artifacts:
  - `output/checks/anima_conditioning_gate/job-0001.png`
    (`cfg=1.0`, empty negative).
  - `output/checks/anima_conditioning_gate/job-0002.png`
    (`cfg=4.5`, empty negative).
  - `output/checks/anima_conditioning_gate/job-0003.png`
    (`cfg=4.5`, negative `red cube, red object, ceramic cube`).
- All three artifacts are `1024x1024`, carry `serenity.genparams.v1`, and have
  `serenity.anima.daemon_result.v1` sidecars with requested/executed sampler
  metadata, timings, and peak VRAM.
- IDAT hashes are distinct:
  - low CFG: `8b4db45968cce27e7fe99927ef1af4d52d955b681cef68d2f9b3fd1d50c8afa4`
  - high CFG: `2ee6e3264d15fb177066d27ad29081e986442be342dbed63faf0e37ba70c71fa`
  - high CFG + negative:
    `7fec4d642f9a127ced2e894474da9d570cd954f7f4fd89f89b8c1917f35404b9`
- Timings/VRAM:
  - `job-0001`: `total_wall_seconds=102.882308372`,
    `denoise_seconds=2.060830602`, `vae_decode_seconds=82.225633783`,
    `peak_vram_mib=10633.3125`.
  - `job-0002`: `total_wall_seconds=86.731115867`,
    `denoise_seconds=2.138339561`, `vae_decode_seconds=82.530627286`,
    `peak_vram_mib=10598.3125`.
  - `job-0003`: `total_wall_seconds=88.064855635`,
    `denoise_seconds=2.134478508`, `vae_decode_seconds=83.785780459`,
    `peak_vram_mib=12132.4375`.

The sampler surface checker now tracks Z-Image, SDXL, SD3, and Anima
conditioning reports and refreshes to `40/40` markers. The conditioning blocker
remains because prompt-weight math is still rejected/fail-loud rather than
implemented, and Flux/Ideogram intentionally do not accept negative prompts in
their current bounded production routes.

## Build Order

1. Harden the bounded real-backend image smokes into acceptance-grade paths:
   Z-Image still needs speed/quality parity, Ideogram4 needs multi-step proof
   and broader request-surface coverage beyond the bounded PNG metadata/gallery
   and fail-loud option smokes, and Qwen must stay within the bounded 1024x1024
   Euler/simple route until memory, artifact, timing, VRAM, quality, and sampler
   evidence says wider execution is safe.
2. Replace Z-Image's two serial CFG main-stack passes with a measured faster
   path before accepting image speed parity.
3. Promote the bounded LTX2 daemon video runner beyond DEV-smoke by adding
   graph-native request coverage, quality/HQ acceptance evidence, and stable
   video/audio metadata gates.
4. Extend the typed t2i workflow graph into advanced node families only when
   each node family has artifact-backed product gates.
5. Finish model/LoRA browser UI stack controls, Qwen/LoKr LoRA support, and
   compatibility beyond family-level metadata.
6. Add advanced utility surfaces only when the backend emits real artifacts and
   result JSON.
