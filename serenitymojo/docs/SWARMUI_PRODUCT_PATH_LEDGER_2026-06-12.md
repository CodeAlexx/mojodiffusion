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
  `checks=56 passed=55 p0=0 p1=1 p2=0`. Product P0 is ready after the bounded
  video smoke runner wiring, model/gallery API slice, and sampler registry
  wiring, but full SwarmUI all-level parity remains blocked. `images=N` now
  emits indexed serial jobs, variation noise has a runtime artifact gate, and
  `/v1/samplers` exposes a pure-Mojo SwarmUI/Comfy sampler support registry.
  Z-Image now has bounded DPM++ 2M and UniPC bh2/simple-flowmatch wiring; true
  Comfy-style latent batch execution and the remaining generic UniPC/order-3,
  ancestral, SDE, Karras, CFG++, and advanced daemon denoise loops remain
  sampler/runtime gaps.
- `/v1/samplers` endpoint smoke saved
  `output/checks/samplers_endpoint_smoke.json`; it returned schema
  `serenity.samplers.v1`, `accepted_sampler_parity:false`, 45 catalog samplers,
  15 catalog schedulers, and support entries for `zimage` and `qwenimage`.
- 2026-06-12 DPM++ 2M runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py
  --skip-unsupported-smoke --skip-multi-image-smoke --skip-variation-smoke
  --write-readiness output/checks/zimage_dpmpp2m_product_readiness.json`
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
  --skip-dpmpp2m-smoke --skip-multi-image-smoke --skip-variation-smoke
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
- 2026-06-12 Z-Image multi-LoRA runtime evidence:
  `python3 scripts/check_zimage_daemon_product_contract.py
  --skip-unsupported-smoke --skip-dpmpp2m-smoke --skip-unipc-smoke
  --skip-multi-image-smoke --skip-variation-smoke --write-readiness
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
- No stale source prose describes the daemon as a skeleton.

## P0.3 Video Product Path

Goal: video generation works before UI expansion makes it look complete.

Current blockers:

- Video models are known broken.
- The known F32 random stage-2 video/audio noise fallback in
  `ltx2_t2v_av_hq.mojo` was fixed after this ledger was created; fallback noise
  now stores BF16 and `_refhq_noise_blend` casts internally for F32 math.
- 2026-06-12 update: the daemon now exposes `/v1/video` as an explicit
  readiness/status contract, exposes `/v1/video/probe?path=<mp4>` for real MP4
  metadata (`mp4`, `frame_count`, `duration`, muxing, and audio behavior), and
  `POST /v1/video` can launch the compiled bounded LTX2 staged dev runner when
  `output/bin/ltx2_video_smoke_runner` exists. This is still not accepted video
  generation: no current daemon run has produced a verified MP4 plus positive
  peak VRAM evidence.

Required implementation:

- Keep LTX2/NAVA dtype boundaries guarded so latents/noise return the proper
  storage dtype.
- Keep the bounded product video runner buildable and fail loud when the runner
  is missing or an unsupported runner name is requested.
- Verify MP4 duration, frame count, resolution, muxing, and audio behavior.
- Record stage timings and peak VRAM.

Acceptance evidence:

- 2026-06-12 bounded-runner evidence:
  `pixi run build-daemon` passes after adding the video route, probe, and
  bounded runner launch path.
- `pixi run build-video-smoke` passes and emits
  `output/bin/ltx2_video_smoke_runner`.
- `python3 scripts/check_ltx2_dtype_contract.py --scope all` passes.
- `python3 scripts/check_swarmui_product_path_contract.py --write-readiness
  output/checks/swarmui_product_path_readiness.json` now reports
  `checks=56 passed=55 p0=0 p1=1 p2=0`. This clears the product P0 blocker by
  wiring a bounded daemon runner, not by accepting full video parity.
- `POST /v1/video` accepts only `runner:"ltx2_staged_dev_smoke"` and clamps
  `steps` to `1..3`. It runs
  `output/bin/ltx2_video_smoke_runner staged lora stream audio nonag
  output/serenity_daemon/<video-id> <steps>`, writes
  `ltx2_video_runner.log`, and writes `ltx2_video_result.json` with
  `accepted_video_parity:false`, `accepted_sampler_parity:false`,
  `total_wall_seconds`, output paths, and MP4 probe fields when the runner
  succeeds.
- Mojo artifact probe smoke:
  `pixi run mojo run -I . -I /home/alex/MOJO-libs
  serenitymojo/components/artifacts_smoke.mojo` produced
  `output/artifacts_smoke_seq.mp4`; `/v1/video/probe?path=...` reported
  `width=16`, `height=16`, `frame_count=2`, `duration=0.5`, `fps=4`,
  `video_codec=h264`, `has_audio=false`, and `muxing=probe_ok`.
- Remaining acceptance gap: the bounded LTX2 runner must still be executed
  through the daemon and emit a real MP4 with frame/duration/mux/audio fields,
  successful exit code, timings, and positive peak VRAM before SwarmUI-level
  video generation parity can be claimed.

## P1.1 Gallery And Reuse Params

Goal: generated and imported images act like SwarmUI gallery items.

Current status:

- PNG tEXt metadata, `jobs.db`, gallery list/read/import endpoints, search/filter/sort,
  lazy pure-Mojo thumbnail cache, favorite state, and delete are implemented in
  the daemon API.
- Full UI reuse and imported-external-file fixture coverage are not accepted in
  this audit.

Required implementation:

- Wire gallery params back into UI controls with a restart/reuse gate.
- Add an external PNG fixture for arbitrary-path `serenity.genparams.v1` import.
- Add rename and persistent UI selection if required by the product UI.

Acceptance evidence:

- Generate image, restart daemon/UI, gallery still shows it.
- Import an external PNG with genparams and reuse those params into a new job.

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

## P2 Workflow Graphs And Advanced Controls

Goal: future SwarmUI graph parity without blocking the core local generator.

Current status:

- 2026-06-12 update: daemon `/v1/generate` accepts constrained workflow bodies.
  Supported inputs are `workflow.params` / `workflow.genparams` or SerenityUI
  native t2i node graphs using `CheckpointLoaderSimple`, `CLIPTextEncode`,
  `EmptyLatentImage`, `KSampler`, `VAEDecode`, and `SaveImage`. The adapter maps
  known nodes into the existing flat genparams product path.
- Unknown graph formats or node types fail with HTTP 501 and name the
  unsupported node. This is a fail-loud adapter, not arbitrary SwarmUI/Comfy
  graph execution.
- ControlNet/IP-Adapter/regional prompting/live preview/upscaler utility paths
  have partial model pieces but no accepted full product surface in this audit.

Required implementation:

- Keep unsupported workflow nodes fail-loud until a real graph executor exists.
- Add utility tabs only when the backend emits real artifacts and result JSON.
- Do not treat model-file presence as UI/product parity.

Acceptance evidence:

- `pixi run build-daemon` passes with the workflow adapter.
- Runtime smoke on `output/bin/serenity_daemon stub 18120`: POSTing a wrapped
  SerenityUI native node graph from
  `/home/alex/.cache/serenityui/klein9b_nodegraph.workflow.json` queued
  `job-0009`, completed it, wrote `output/serenity_daemon/job-0009.png`, and
  gallery readback showed `serenity.genparams.v1` with prompt, negative prompt,
  width/height, steps, seed, cfg, sampler, scheduler, and creativity mapped from
  the graph/body.
- Unsupported graph smoke returned HTTP 501:
  `{"detail":"unsupported workflow graph node type: NotSupported"}`.
- `python3 scripts/check_swarmui_product_path_contract.py --write-readiness
  output/checks/swarmui_product_path_readiness.json` now reports
  `checks=56 passed=55 p0=0 p1=1 p2=0`.

## Build Order

1. Add daemon real-backend image product smokes with artifact/timing/VRAM
   readback for Z-Image and only bounded Qwen gates until memory evidence says a
   full Qwen run is safe.
2. Replace Z-Image's two serial CFG main-stack passes with a measured faster
   path before accepting image speed parity.
3. Build one daemon-backed video runner that emits a real MP4 plus
   frame/duration/muxing/audio/timing/VRAM evidence.
4. Extend constrained workflow support into a real graph executor only when each
   node family has artifact-backed product gates.
5. Finish model/LoRA browser UI stack controls, Qwen/LoKr LoRA support, and
   compatibility beyond family-level metadata.
6. Add advanced utility surfaces only when the backend emits real artifacts and
   result JSON.
