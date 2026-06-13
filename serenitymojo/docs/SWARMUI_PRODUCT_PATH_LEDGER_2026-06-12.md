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

## P0.1 Image Fast Path

Goal: image generation must use the newest mojodiffusion runtime kernels in the
SwarmUI-facing generation path.

Current blockers:

- 2026-06-12 update: Qwen-Image product forward no longer calls masked `sdpa`.
  It routes through `sdpa_qwen_keymask`, an online-softmax key-mask path that
  preserves Qwen's middle-of-sequence text padding semantics without allocating
  `[B,H,S,S]` masks or F32 score slabs. This is statically and op-parity gated;
  full Qwen generation was not run because Qwen remains too large/slow/OOM-risky
  for this slice.
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
  Full SwarmUI all-level parity remains blocked by Qwen full generation,
  full video parity beyond DEV-smoke artifacts, advanced workflow node families, sampler breadth,
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
  bh2/simple-flowmatch wiring; true Comfy-style latent batch execution and the
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
- `/v1/samplers` endpoint smoke saved
  `output/checks/samplers_endpoint_smoke.json`; it returned schema
  `serenity.samplers.v1`, `accepted_sampler_parity:false`, 45 catalog samplers,
  15 catalog schedulers, and support entries for `zimage`, `qwenimage`, and
  `ideogram4`.
  Refreshed evidence from the compiled stub daemon now shows Z-Image endpoint
  support for `euler`, `flowmatch_euler`, `flow_match_euler`, `dpmpp_2m`,
  `dpm++ 2m`, `uni_pc`, and `uni_pc_bh2`; Qwen remains `euler`/flow-match only.
  Ideogram4 exposes bounded `euler`/flow-match aliases that execute as
  `ideogram4_logitnormal_euler`, with only `logitnormal`/`logit_normal`/
  `ideogram_logitnormal`/`ideogram4_logitnormal` scheduler aliases accepted.
- 2026-06-12 DPM++ 2M runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py
  --skip-unsupported-smoke --skip-generic-unipc-smoke --skip-unipc-smoke
  --skip-multi-image-smoke --skip-variation-smoke --skip-img2img-smoke
  --skip-multi-lora-smoke --write-readiness
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
- 2026-06-12 UniPC bh2 runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py
  --skip-dpmpp2m-smoke --skip-generic-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-img2img-smoke --skip-multi-lora-smoke
  --write-readiness output/checks/zimage_unipc_bh2_product_readiness.json`
  passed. It emitted unsupported generic UniPC failure `job-0038`, baseline
  `job-0039`, and UniPC bh2 `job-0040`. `job-0040` wrote
  `output/serenity_daemon/job-0040.png` and
  `output/serenity_daemon/job-0040.png.zimage_daemon_result.json`; the manifest
  records `requested_sampler:"uni_pc_bh2"`,
  `requested_scheduler:"flowmatch"`, `executed_sampler:"uni_pc_bh2"`,
  `executed_scheduler:"simple_flowmatch"`, `solver_type:"bh2"`,
  `solver_order:2`, `schedule_source:"zimage_build_sigmas"`,
  `unipc_update_steps:3`, `unipc_corrector_steps:2`,
  `unipc_second_order_steps:2`, `denoise_seconds_per_step:0.32013946925`,
  `peak_vram_mib:21727.5625`, and `accepted_sampler_parity:false`.
  Generic `uni_pc` is not an alias for this accepted `uni_pc_bh2` bh2/order-2
  flow path; the focused Mojo semantic gate
  `serenitymojo/sampling/parity/comfy_unipc_semantics_gate.mojo` proves generic
  `uni_pc` is `bh1`, order `min(3,len(sigmas)-2)`, SigmaConvert, Comfy
  penultimate-sigma discard, final-zero-replacement, and initial-noise scaling.
- 2026-06-12 Z-Image generic UniPC runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py --timeout 900 --steps
  1 --skip-unsupported-smoke --skip-dpmpp2m-smoke --skip-unipc-smoke
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
  --skip-generic-unipc-smoke --skip-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-multi-lora-smoke --write-readiness
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
  --steps 1 --write-readiness output/checks/zimage_daemon_product_readiness.json`
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
  --skip-unsupported-smoke --skip-dpmpp2m-smoke --skip-generic-unipc-smoke
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
- Real Z-Image daemon generation now has current GPU evidence. Full Qwen
  daemon generation was not run because Qwen remains too large/slow/OOM-risky
  for this slice.
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
- Add real-backend smoke mode for Z-Image and Qwen with bounded settings.
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
  unsupported-sampler fail-loud behavior, variation noise output change,
  multi-image serial output, and a running-job cancel smoke.
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
  --skip-generic-unipc-smoke --skip-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-img2img-smoke --skip-multi-lora-smoke
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
  --skip-generic-unipc-smoke --skip-unipc-smoke --skip-multi-image-smoke
  --skip-variation-smoke --skip-multi-lora-smoke --write-readiness
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
  `checks=78 passed=78 p0=0 p1=0 p2=0`. Product P0 and tracked P1 are ready.
  Full SwarmUI all-level parity still remains blocked by Qwen full generation,
  full video parity beyond DEV-smoke artifacts, advanced workflow node families, sampler breadth,
  and Z-Image speed parity.

## Build Order

1. Harden the bounded real-backend image smokes into acceptance-grade paths:
   Z-Image still needs speed/quality parity, Ideogram4 needs multi-step proof
   and broader request-surface coverage beyond the bounded PNG metadata/gallery
   and fail-loud option smokes, and Qwen remains bounded until memory evidence
   says a full run is safe.
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
