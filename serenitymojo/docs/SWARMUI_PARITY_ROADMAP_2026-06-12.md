# SwarmUI Parity Roadmap - 2026-06-12

Goal: reach SwarmUI-level parity with pure Mojo product/runtime paths across
`mojodiffusion` + `MOJO-libs`, starting from what is real in the repo today.

This is the controlling roadmap. Audits and status docs are evidence; this file
is the execution order.

## Hard Rules

- Product generation paths must be Mojo-native. Python is allowed for dev
  tooling, artifact inspection, parity oracles, and audits only.
- Do not call a smoke test production.
- Do not run full Qwen generation until bounded memory evidence says it is safe.
  Qwen remains too large/slow/OOM-prone; bounded op/static gates are acceptable.
- Current video models are not accepted. The daemon has a bounded LTX2 staged
  smoke runner and MP4 probe, but video generation remains quarantined until a
  real daemon run emits a measured MP4 artifact with VRAM evidence.
- Claims require current evidence: artifact path, dimensions, metadata, timings,
  peak VRAM, and readiness label.

## Current Evidence Snapshot

Current gate:

```bash
python3 scripts/check_swarmui_product_path_contract.py --write-readiness output/checks/swarmui_product_path_readiness.json
```

Current result after adding variation noise runtime behavior, the
sampler/scheduler registry, the bounded LTX2 video smoke runner, the
model/gallery API slice, bounded Z-Image UniPC bh2, Z-Image multi-LoRA runtime
stacking, typed linked workflow execution for the supported t2i graph, and the
runtime UI/gallery/reuse/state contract:
`59` checks, `59` passed, `P0=0`, `P1=0`, `P2=0`. Product P0 and tracked P1
gates are ready. Full SwarmUI all-level parity is still blocked.

Current high-risk runtime gaps:

- Qwen full daemon generation was not run; only the masked-attention fast path
  and tiny parity gate are proven.
- Video generation is still not accepted; `/v1/video` can launch the bounded
  `ltx2_staged_dev_smoke` runner and `/v1/video/probe` inspects MP4 artifacts,
  but no current daemon run has produced MP4/timing/positive-VRAM evidence.
- Sampler fields now reach typed `JobParams`/worker IPC, `/v1/samplers` exposes
  a SwarmUI/Comfy catalog and per-backend support matrix, and unsupported names
  fail loud before model work. Z-Image has bounded DPM++ 2M/simple-flowmatch
  runtime evidence (`job-0036`) and bounded UniPC bh2/simple-flowmatch runtime
  evidence (`job-0040`), but generic UniPC/order-3, ancestral, SDE, Karras,
  CFG++, and other daemon denoise loops are still not accepted sampler parity.
- Variation noise is implemented for image backends and proven for Z-Image by a
  daemon artifact gate; this is not full sampler/scheduler parity.
- `images=N` now emits serial indexed daemon jobs with seed offsets and
  metadata. True Comfy-style batched latent execution remains unimplemented.
- UI/gallery/reuse/state has a stub-daemon runtime contract that now proves
  reuse provenance, restart-safe job history, indexed external gallery import,
  gallery rename/manual order, presets/state restart, favorite/delete, queue
  mutation, and reuse generate.
- Workflow support now keeps advanced Comfy/Swarm node families beyond the typed t2i graph as an explicit remaining blocker.
- The accepted workflow path is a typed linked t2i graph executor for
  `CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentImage`, `KSampler`,
  `VAEDecode`, and `SaveImage`; unsupported graph families must keep failing
  loudly.
- Z-Image speed parity is not accepted; current evidence still needs a paired
  baseline and optimized CFG/main-stack path.

Current artifact evidence found in this working tree:

- Stub metadata proof:
  - `output/serenity_daemon/job-0001.png`
  - `output/serenity_daemon/jobs.db`
- Real Z-Image daemon product proof:
  - `output/serenity_daemon/job-0004.png`
  - `output/serenity_daemon/job-0004.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0010.png`
  - `output/serenity_daemon/job-0010.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0015.png`
  - `output/serenity_daemon/job-0015.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0023.png`
  - `output/serenity_daemon/job-0023.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0024.png`
  - `output/serenity_daemon/job-0024.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0025.png`
  - `output/serenity_daemon/job-0025.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0028.png`
  - `output/serenity_daemon/job-0028.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0029.png`
  - `output/serenity_daemon/job-0029.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0030.png`
  - `output/serenity_daemon/job-0030.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0031.png`
  - `output/serenity_daemon/job-0031.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0036.png`
  - `output/serenity_daemon/job-0036.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0040.png`
  - `output/serenity_daemon/job-0040.png.zimage_daemon_result.json`
  - `output/serenity_daemon/job-0047.png`

`job-0028` is the current 512x512, 1-step experimental artifact from the daemon
product path, produced by the repeatable runtime gate. It proves the Z-Image
daemon path, PNG `serenity.genparams.v1`, jobs DB row, gallery/read endpoints,
timings, positive peak VRAM, variation noise output changes, indexed multi-image
serial output, and running-job cancel smoke (`job-0032`). It also proves
unsupported sampler fail-loud behavior (`job-0027`). It is not full sampler
parity or speed parity.

`job-0040` is the current bounded UniPC bh2 sampler artifact from the daemon
product path. It proves `requested_sampler:"uni_pc_bh2"` executes as
`executed_sampler:"uni_pc_bh2"` with `solver_type:"bh2"`, `solver_order:2`,
`schedule_source:"zimage_build_sigmas"`, `unipc_update_steps:3`,
`unipc_corrector_steps:2`, `unipc_second_order_steps:2`,
`denoise_seconds_per_step:0.32013946925`, and `peak_vram_mib:21727.5625`.
Generic `uni_pc` remains fail-loud (`job-0038`) until its exact Comfy/Swarm
semantics are separately mapped and proven.

`job-0047` is the current typed linked workflow graph smoke artifact from the
stub daemon product path. It proves linked positive/negative conditioning, model,
latent dimensions, sampler fields, scheduler, seed, CFG, denoise/creativity, PNG
metadata, and fail-loud wrong-type-link/unsupported-node behavior for the
supported t2i graph. It is not advanced Comfy/Swarm graph parity.

## What We Have

### Product/Daemon

- `serenitymojo/serve/serenity_daemon.mojo`
  - localhost generation API
  - job DB path
  - canonical genparams passthrough
  - process-isolated worker support
- `serenitymojo/serve/zimage_backend.mojo`
  - real Z-Image backend surface
  - MOJO-libs PNG metadata output
  - init-image decode/resize hooks
  - VRAM trim hooks
- `serenitymojo/serve/model_scan.mojo`
  - model/LoRA scanner foundation
- Implemented daemon API surfaces in this worktree:
  - gallery list/item/read-params endpoints
  - queued-job reorder/remove endpoints
  - presets/state endpoints
  - prompt syntax parser records raw/resolved prompt metadata

Remaining product gaps:

- imported-image reuse workflow into UI controls and gallery rename
- Qwen LoRA, Z-Image LoKr/LyCORIS conversion, and UI LoRA stack controls
- arbitrary graph execution beyond the constrained native t2i adapter
- real video generation backend

### Image I/O And Gallery Tools

- `/home/alex/MOJO-libs/image/`
  - PNG/JPEG/WebP decode/encode
  - resize, color, filters, EXIF, ICC, CMYK
  - GPU color/filter ops
- Product use:
  - `serve/image_io.mojo` decodes user images via MOJO-libs.
  - `serve/zimage_backend.mojo` writes metadata PNGs.
  - `serve/stub_backend.mojo` proves PNG metadata and DB path.

Missing adoption:

- full imported-image reuse workflow
- gallery rename if the UI needs it

### Memory/Offload Tools

- `serenitymojo/scratch_ring.mojo`
  - GPU scratch ring for frame-scoped Tensor allocation
  - widely used in Klein/autograd/backward paths
- `serenitymojo/offload/turbo_planned_loader.mojo`
  - async block streaming
  - explicit overlap API: `prefetch_with_ctx`,
    `prefetch_next_with_ctx`, `mark_active_block_done`
- `serenitymojo/offload/vmm_cuda.mojo`
  - CUDA memory info and pool trim
- `serenitymojo/offload/vmm_slab.mojo`
  - VMM slab allocator primitive
  - implemented but mostly parked
- `/home/alex/MOJO-libs/mem/`
  - CPU arenas/pools/slabs/ring/stats

Important finding:

- The tools exist. The gap is adoption and measurement in product paths, not a
  missing generic memory library.
- No `cutlass` or `cudlass` implementation was found under
  `/home/alex/mojodiffusion` or `/home/alex/MOJO-libs`. The current in-tree fast
  attention primitive is the cuDNN v9 flash SDPA shim in
  `serenitymojo/ops/attention_flash.mojo`.

### Fast Kernels

- `serenitymojo/ops/attention_flash.mojo`
  - cuDNN v9 BF16 flash SDPA forward/backward
  - already verified by the flash parity/speed gate
- `serenitymojo/ops/attention.mojo`
  - `sdpa_tiled` / `sdpa_nomask_tiled` for large-S online softmax
  - legacy math fallbacks still exist and must not be confused with product
    speed parity
- Z-Image no-saved product forwards now have a flash dispatch route.

## Model Status For Product Parity

| Model/Family | Current status | Next action |
|---|---|---|
| Z-Image | Best first SwarmUI product target. Has product backend, metadata path, flash SDPA route, sampler manifests. Speed parity still not accepted. | Make daemon-driven real Z-Image generation produce PNG + manifest with dimensions, timings, peak VRAM, progress/cancel, and readiness label. |
| Klein | Best memory/offload mechanics target. Uses `ScratchRingAllocator` and `TurboPlannedLoader`; product smokes exist in docs. Heavy. | Use after Z-Image product artifact, or if the slice is explicitly memory/offload measurement. |
| SDXL / SD3 / Flux / Chroma / ERNIE | Useful image candidates, but several are cached-input, staged, or contract smokes rather than full SwarmUI product paths. | Promote only after Z-Image product loop proves the daemon/gallery/runtime pattern. |
| Qwen | Too large/slow/OOM-prone for full generation in this slice. Static masked-SDPA blocker is fixed with `sdpa_qwen_keymask`; tiny op parity passes, but no full Qwen artifact was run. | Keep Qwen on bounded gates only until memory/offload evidence says a real daemon run is safe. |
| Video / LTX2 / NAVA / Wan | Not accepted. Components exist. Daemon now has a bounded `/v1/video` LTX2 staged dev smoke runner and `/v1/video/probe` MP4 metadata reader, but no daemon run has emitted accepted MP4/timing/positive-VRAM evidence. | Execute the bounded runner through the daemon, inspect the MP4/audio/probe/result manifest, add peak VRAM evidence, and keep `accepted_video_parity:false` until the artifact gate passes. |

## Milestones

### M0 - Baseline And Control Gates

Purpose: stop winging.

Acceptance:

- This roadmap exists and is kept current.
- `scripts/check_swarmui_product_path_contract.py` remains the global product
  gate.
- Qwen/video blockers stay visible, but are marked parked for the next slice.
- `serenitymojo/docs/IMAGE_MEMORY_TOOLING_MAP_2026-06-12.md` records available
  tools and adoption gaps.

Status: in progress.

### M1 - Z-Image Product Generation Evidence

Purpose: prove one real image model can behave like a SwarmUI backend.

Tasks:

- Run Z-Image through the daemon/process-isolated backend, not only a CLI.
- Keep model resident across consecutive jobs or prove process isolation and
  model switch VRAM reclaim.
- Emit progress events and support cancel.
- Write PNG with `serenity.genparams.v1`.
- Write result manifest with:
  - prompt, seed, width, height, steps, CFG
  - text encode seconds
  - denoise seconds and seconds/step
  - VAE decode seconds
  - total wall seconds
  - positive peak VRAM
  - artifact paths
  - accepted_sampler_parity and accepted_speed_parity flags
- Inspect the PNG dimensions and metadata.

Acceptance:

- A real Z-Image PNG exists from the daemon product path.
- A manifest exists and carries positive timings + peak VRAM.
- The result is explicitly labeled `experimental` unless paired speed/sampler
  parity also passes.

Status: accepted for this slice on `job-0028`. Manifest metrics:
`load_seconds=2.4113649049999997`, `text_encode_seconds=2.63576205`,
`denoise_seconds=1.221294356`, `vae_decode_seconds=0.696861333`,
`total_wall_seconds=7.447758765`, `peak_vram_mib=21510.625`.
The manifest records `requested_sampler:"euler"`,
`requested_scheduler:"flowmatch"`, `executed_sampler:"flowmatch_euler"`,
`executed_scheduler:"simple_flowmatch"`, `variation_applied:false`,
`image_index:0`, and `image_count:1`.
The same gate proves variation noise as `job-0029` with
`variation_seed:20261319`, `variation_strength:0.55`, and changed output IDAT
hash (`fecbe64e...` baseline to `59a22b92...` variation). It also proves
`images=2` serial output as `job-0030` and `job-0031` with seed offsets
`20260812` and `20260813`, `image_index` values `0` and `1`, and
`image_count:2` in PNG metadata and result manifests.

Repeatable GPU gate:

```bash
python3 scripts/check_zimage_daemon_product_contract.py --cancel-smoke \
  --write-readiness output/checks/zimage_daemon_product_readiness.json
```

This starts the compiled Mojo daemon in `zimage` mode, submits a bounded
512x512 job through `/v1/generate`, listens to `/v1/progress`, validates PNG
metadata, `jobs.db`, gallery/read endpoints, manifest timings/VRAM, and then
checks running-job cancel behavior. Passing this gate still does not accept
sampler or speed parity.

### M2 - Z-Image Runtime Speed/Memory Work

Purpose: use the stack's actual tools where measurement says they help.

Tasks:

- Profile current Z-Image product run by stage.
- Confirm whether bottleneck is attention, allocation churn, text encode,
  reload, CFG duplication, VAE, or host sync.
- Keep the cuDNN flash path for product no-saved forwards.
- Decide from measurement whether to:
  - add scratch-ring allocation to the sampler path
  - reduce CFG duplicate work
  - keep more model state resident
  - split subprocess lifecycle differently
  - defer VMM because it is not yet justified

Acceptance:

- Before/after timing and VRAM are recorded.
- No speed claim is accepted without comparable OneTrainer or prior accepted
  baseline evidence.

### M3 - Gallery And Reuse Params

Purpose: close a major SwarmUI experience gap after real image output works.

Tasks:

- Add `/v1/gallery`. DONE 2026-06-12: lists `output/serenity_daemon/job-*.png`.
- Add PNG metadata readback via MOJO-libs `read_png_text`. DONE 2026-06-12:
  `/v1/gallery/<id>` and `/v1/gallery/read?path=<png>` return
  `params_json` plus parsed `params` when valid.
- Index output images in jobs DB.
- Decode/resize thumbnails with MOJO-libs image tools. DONE 2026-06-12:
  gallery items lazily cache 256px PNG thumbnails under
  `output/serenity_daemon/thumbnails`.
- Add gallery `search`/`filter`/`sort`, favorite state, and delete. DONE
  2026-06-12: `/v1/gallery` query params, `POST /v1/gallery/<id>/favorite`,
  and `DELETE /v1/gallery/<id>`.
- Add reuse-params response shape that can feed the UI controls. DONE
  2026-06-12: the runtime checker posts provenance-bearing gallery params back
  into `/v1/generate` and proves metadata round-trip.

Acceptance:

- Import a generated PNG and read `serenity.genparams.v1`. DONE 2026-06-12:
  `job-0004` Z-Image artifact was read by `/v1/gallery/job-0004`.
- Read an external PNG with the same metadata key. DONE 2026-06-12:
  `scripts/check_ui_gallery_reuse_state_contract.py` copies a generated PNG
  outside `output/serenity_daemon` and proves `/v1/gallery/read?path=...`.
- Indexed external gallery import. DONE 2026-06-12:
  `POST /v1/gallery/import` imports an external PNG into the gallery index.
- Reuse params into a new request. DONE 2026-06-12: normalized gallery params
  generate a second artifact and reused output metadata records source
  provenance.

### M4 - Model And LoRA Browser

Purpose: make the backend usable like SwarmUI without hardcoded dropdowns.

Tasks:

- Scan checkpoint and LoRA dirs.
- Cache metadata/thumbnails for UI cards. PARTIAL 2026-06-12: API cards and
  metadata exist; real model/LoRA preview thumbnails and user notes remain.
- Expose search/filter endpoints. DONE 2026-06-12: `/v1/models` supports model
  and LoRA search/filter/sort plus family compatibility metadata.
- Support UI multi-LoRA stack controls and extend LoRA runtime beyond accepted
  Z-Image PEFT/Comfy stacks when tensor-target conversion is proven.
- Preserve the no-fused-LoRA rule unless a separate accepted merge path exists.

Acceptance:

- UI/backend can list checkpoints and LoRAs from disk.
- Z-Image can select a model/LoRA stack without hand-editing a config file.

### M5 - Queue, Prompt Syntax, Presets, State

Purpose: product UX parity after the generation core is real.

Tasks:

- Queue reorder/remove. DONE 2026-06-12: `/v1/reorder[/<id>]` reorders active
  queued jobs by `position` or `before_id`; `/v1/remove[/<id>]` removes an
  active queued job before execution.
- Interrupt current job.
- Persist named presets and last state. DONE 2026-06-12:
  `/v1/state` stores `serenity.ui_state.v1` at
  `output/serenity_daemon/state/last_state.json`; `/v1/presets` stores named
  `serenity.genparams.v1` parameter presets at
  `output/serenity_daemon/state/presets.json`. Restart smoke passed on stub
  daemon ports `18115` -> `18116`.
- Parse prompt syntax in product path. DONE 2026-06-12:
  - `(text:weight)` is parsed into `prompt_syntax.weighted` and resolved plain
    text for the backend.
  - `<lora:name:weight>` is extracted into the job LoRA list and
    `prompt_syntax.lora_tags`.
  - `<random:a,b>` is deterministically resolved from the request seed and
    recorded in `prompt_syntax.random`.
  - `__name__` and `<wildcard:name>` are recorded in `prompt_syntax.wildcards`;
    local wildcard text files are expanded when present, otherwise unresolved
    wildcard names are preserved in metadata.

Acceptance:

- Product gate P1 blockers are currently clear. Runtime evidence:
  `checks=19 passed=19 p0=0 p1=0` in
  `output/checks/ui_gallery_reuse_state_readiness.json`; queue mutation,
  presets/state restart, gallery readback, favorites, delete, reuse generate,
  provenance, indexed import, job history, and gallery rename/manual order pass.
  Prompt conditioning weights are parsed and persisted but not yet applied as
  weighted text conditioning in model math.
- PNG metadata, presets, and request JSON use the same canonical fields.

### M6 - Expand Image Model Coverage

Purpose: repeat the proven product pattern on additional image models.

Order:

1. Klein if the goal is memory/offload mechanics and low-VRAM behavior.
2. SDXL/Flux/Chroma/ERNIE if the goal is model breadth.
3. Qwen only after a bounded low-memory plan is written and accepted.

Acceptance for each model:

- pure Mojo product path
- real artifact
- dimensions + metadata
- timings + peak VRAM
- readiness label
- fail-loud unsupported options

### M7 - Video Rebuild

Purpose: restore video only after image parity has a stable product backbone.

Tasks:

- Pick one video vertical.
- Establish model identity and current failure mode.
- Produce a bounded real artifact.
- Verify:
  - MP4 path
  - frame count
  - duration
  - resolution
  - muxing
  - audio behavior if applicable
  - timings
  - peak VRAM

Acceptance:

- No video feature leaves `not ready` until the artifact gate passes.

## Immediate Next Slice

Do not implement full Qwen generation. Product P0 is ready after wiring the
bounded video smoke gate, but the next video slice must execute that runner
through the daemon and capture MP4 frame/duration/mux/audio, timing, and VRAM
evidence. It is still not a full HQ parity claim.

Next implementation slice:

1. Execute the bounded daemon-backed LTX2 video smoke and record exact
   artifact/runtime blocker evidence if it OOMs, stalls, or fails probe.
2. Promote the next sampler only with artifact/timing/VRAM evidence. Do not
   promote generic `uni_pc` by aliasing it to bh2; map its exact Comfy/Swarm
   semantics first.
3. Measure the current Z-Image product bottleneck by stage with the daemon gate
   as the control artifact, then replace the measured slow path first.
4. Keep Qwen on bounded op/static gates until memory/offload evidence says full
   generation is safe.
5. Keep video non-accepted until a daemon-backed runner emits MP4 frame/
   duration/muxing/audio/timing/positive-VRAM evidence.
6. Rerun the product gate and update this roadmap.

This gives us SwarmUI parity progress without pretending Qwen or video are
production usable today.
