# SwarmUI Parity Audit: MOJO-libs + mojodiffusion

Date: 2026-06-12

Scope:

- Source audit baseline: `/home/alex/serenityUI/SWARMUI_GAP_AUDIT_2026-06-10.md`.
- Acceptance checklist baseline: `/home/alex/serenityUI/GENSCREEN_PARITY_PLAN.md`.
- Current SerenityUI status baseline: `/home/alex/serenityUI/SERENITYUI_TODO.md`.
- Audited roots: `/home/alex/MOJO-libs` and `/home/alex/mojodiffusion`.
- Goal: parity with SwarmUI at all applicable local/native generation layers.
- Not in scope unless the product goal changes: public multi-user server, accounts,
  webhooks, extension marketplace, distributed workers, mobile UI, and generic model
  downloader.

## Source Acceptance Standard

The older SerenityUI docs are the acceptance frame for this audit:

- Function over visual parity: keep web-server/multi-user features out unless
  they serve the native single-user generator.
- Product path over helper path: generation must go through the daemon/API/UI
  surface, with CLI fallback allowed but not the accepted SwarmUI-class path.
- Real artifacts over labels: images need dimensions, PNG metadata, timings,
  peak VRAM, progress/cancel behavior, and persisted job/gallery state.
- Video needs the same, plus MP4 duration, frame count, resolution, muxing, and
  audio behavior.
- Performance is in scope: resident weights, model switching, smooth per-step
  progress, bounded VRAM, and no per-step host stalls or reload churn.
- Runtime must remain pure Mojo. Python is acceptable for dev checks, parity
  oracles, inspection, and audits, but not as the product generation pipeline.

Verdict:

The June 10 SwarmUI gap audit is already stale in several important places.
`mojodiffusion` now has a real Mojo localhost daemon, model and LoRA scanning,
progress/cancel surfaces, job DB indexing, PNG metadata, process-isolated workers,
and Z-Image/Qwen image generation backends. `MOJO-libs` is also broader than its
headline docs: it provides the HTTP/WebSocket/JSON/image/SQLite/FFmpeg-style pieces
needed to build a native single-user SwarmUI-class app without turning Python into
the product runtime.

However, this is not SwarmUI parity yet. The biggest blocker is no longer just
"missing UI controls." The product image/video generation path is not consistently
using the newest mojodiffusion fast kernels. Image model load + denoise remains too
slow for SwarmUI-level use, video models are broken, and the current daemon/UI path
does not yet prove that CUDLASS/CUTLASS-class or cuDNN flash kernels are the
generation path rather than only a trainer/parity path.

## Map Files And Source Maps

I did not find literal `*.map` files under the three requested audit roots with:

```bash
rg --files /home/alex/mojodiffusion /home/alex/MOJO-libs /home/alex/serenityUI | rg '\.map$|\.MAP$'
find /home/alex/mojodiffusion /home/alex/MOJO-libs /home/alex/serenityUI -name '*.map' -o -name '*.MAP'
```

Relevant map-style docs do exist and were used:

- `/home/alex/mojodiffusion/serenitymojo/MAP.md`
- `/home/alex/mojodiffusion/T5_ZIMAGE_TRAINING_MAP.md`
- `/home/alex/mojodiffusion/docs/MOJO_KERNELS.md`
- `/home/alex/mojodiffusion/docs/MOJO_MODULES.md`
- `/home/alex/mojodiffusion/TODO.md`
- `/home/alex/mojodiffusion/serenitymojo/docs/HANDOFF_2026-06-12.md`
- `/home/alex/mojodiffusion/serenitymojo/docs/SAMPLER_PRODUCT_HARNESS_2026-06-05.md`
- `/home/alex/mojodiffusion/serenitymojo/docs/ONETRAINER_CORE_PORT_STATUS_2026-06-06.md`
- `/home/alex/mojodiffusion/MOJO_TRAINER_USE_STATUS.md`
- `/home/alex/mojodiffusion/serenitymojo/docs/SWARMUI_PRODUCT_PATH_LEDGER_2026-06-12.md`

Important doc conflict: older map docs still describe `sdpa_nomask` math-mode as
the broad inference fallback, while newer ledgers say flash/graph work landed for
trainer paths. The active task board still lists "Inference speed audit - all
models image+video" and says to start by switching sampler attention to flash and
measuring. Therefore, the audit treats new kernels as present but not proven wired
into SwarmUI-facing image/video generation.

Machine-checkable product gate added from this audit:

```bash
python3 scripts/check_swarmui_product_path_contract.py
python3 scripts/check_swarmui_product_path_contract.py --strict
python3 scripts/check_swarmui_product_path_contract.py --strict-all
```

Default mode is report-only. `--strict` fails on P0 product-path blockers; at the
time of this audit those blockers are expected because image generation is still
not proven on the fast path and video is still broken.

## Verification Run This Pass

Commands that passed:

```bash
pixi run mojo --version
# Mojo 1.0.0b1 (a9591de6)

timeout 180 pixi run mojo run -I /home/alex/MOJO-libs /home/alex/MOJO-libs/json/tests/json_test.mojo
# 26 passed, 0 failed

timeout 180 pixi run mojo run -I /home/alex/MOJO-libs /home/alex/MOJO-libs/http/tests/ws_reassembly_test.mojo
# 12 passed, 0 failed

timeout 180 pixi run mojo run -I /home/alex/MOJO-libs /home/alex/MOJO-libs/sqlite/tests/writer_test.mojo
python3 /home/alex/MOJO-libs/sqlite/tests/writer_check.py
# Mojo writer tests and Python sqlite3 integrity checks passed

timeout 300 pixi run mojo build -I . -I /home/alex/MOJO-libs \
  serenitymojo/serve/serenity_daemon.mojo \
  -o /tmp/serenity_daemon_flash_audit \
  -Xlinker -lm -Xlinker -lcuda \
  -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
  -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
# Build passed

timeout 300 pixi run mojo build -I . \
  serenitymojo/pipeline/zimage_generate.mojo \
  -o /tmp/zimage_generate_flash_product_check \
  -Xlinker -lm -Xlinker -lcuda \
  -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
# Build passed

timeout 600 bash -lc 'rm -f /tmp/sdpa_flash_par && pixi run mojo build -I . \
  -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib \
  -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib \
  -Xlinker -lserenity_cudnn_sdpa serenitymojo/ops/tests/sdpa_flash_parity.mojo \
  -o /tmp/sdpa_flash_par && \
  LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
  /tmp/sdpa_flash_par'
# ALL GATES PASS. Z-Image B1 padded fwd+bwd speedup: 34.39x; B2 padded: 35.83x.

python3 scripts/check_swarmui_product_path_contract.py --write-readiness /tmp/swarmui_product_path_readiness.json
# 33 checks, 26 passed, p0=2, p1=4, p2=1
```

Runtime daemon smoke in `stub` mode passed:

- `GET /v1/health` returned ok.
- `GET /v1/models` returned a real scan of local checkpoints and LoRAs.
- `POST /v1/generate` accepted model, prompt, dimensions, steps, seed, CFG,
  variation fields, image count, and LoRA metadata.
- Job completed and wrote `output/serenity_daemon/job-0001.png`.
- PNG readback showed `32 x 32`, RGB, and `serenity.genparams.v1` tEXt metadata.
- `jobs.db` readback showed the finished job and canonical params JSON.

Commands that failed or exposed blockers during the initial pass:

```bash
python3 scripts/check_ltx2_dtype_contract.py
```

This initially failed on `serenitymojo/pipeline/ltx2_t2v_av_hq.mojo:2334-2335`
because the fallback path created F32 random video/audio latents/noise. That
specific dtype-boundary issue was fixed in this follow-up by making fallback
stage-2 noise BF16 at storage boundaries and casting internally for F32 blend
math. The guard now passes. Video is still not product-parity because there is no
accepted daemon-backed MP4/audio artifact path with duration, frame count,
resolution, muxing, timings, and VRAM evidence.

`sqlite/tests/format_test.mojo` also failed because it expected `/tmp/sq_phase0.db`;
the available fixture script creates different fixture names. Treat this as stale
test/fixture wiring, not proof the SQLite subset is broken.

## Layer Matrix

### Foundation Libraries

Status: strong base, not full database/browser platform.

`MOJO-libs` covers the pieces needed by a native single-user generator UI:

- JSON parser/tree/tape/schema helpers.
- Sockets, epoll, TLS, HTTP/1.1, HTTP/2, WebSocket, routing, compression, static
  files, multipart, cookies, and HTTP client pieces.
- PNG/JPEG/WebP decode/encode, resize/filter paths, EXIF/ICC/CMYK handling.
- PNG tEXt encode/read helpers used by the daemon metadata path.
- FFmpeg-style audio decode/mux wrappers.
- Pure-Mojo SQLite-format read/write/select subset.

Limits:

- The SQLite engine is an interoperability subset, not full SQLite.
- FFmpeg wrappers are not the same as a full media authoring stack.
- Some MOJO-libs HTTP/net code emits Mojo deprecation warnings under 1.0.0b1.

### Resident Generation Daemon

Status: partially implemented, real product shape exists.

`serenitymojo/serve/serenity_daemon.mojo` exposes:

- `POST /v1/generate`
- `GET /v1/jobs`
- `GET /v1/job/<id>`
- `POST /v1/cancel/<id>`
- `GET /v1/models`
- `GET /v1/health`
- WebSocket `/v1/progress`

The daemon uses `MOJO-libs` HTTP/WebSocket/JSON/SQLite pieces, stores job state in
`output/serenity_daemon/jobs.db`, writes PNG metadata, and has process-isolated
worker support so model switches can reclaim VRAM by killing the child process.

Limits:

- The daemon build currently needs explicit `-lm -lcuda` plus the
  `serenity_cudnn_sdpa` cshim link flags.
- Stub mode still links real CUDA symbols because real backends are imported.
- The daemon runtime smoke was CPU/stub only; real Z-Image/Qwen daemon generation
  still needs a current GPU artifact, timing, VRAM, and cancellation proof.
- Runtime WebSocket progress was unit-tested at the library layer, not exercised
  in the daemon smoke.

### Model And LoRA Browser

Status: daemon side implemented, UX/cache layer incomplete.

`serenitymojo/serve/model_scan.mojo` scans real checkpoint and LoRA directories,
including `.safetensors` header probes, and tags families such as zimage, anima,
flux, flux-2/klein, chroma, sd3, sdxl, ltx2, qwen-image, and wan.

Limits:

- No accepted SwarmUI-class browser UX in the audited roots: thumbnails, search,
  sort, metadata cache, favorites, compatibility warnings, and model cards remain
  product work.
- Z-Image daemon LoRA support is currently capped to one overlay in the audited
  backend path.

### Gallery, Metadata, Reuse Params

Status: backend primitives exist, UI parity incomplete.

Evidence:

- `MOJO-libs/image/png.mojo` supports PNG tEXt write/read.
- The daemon writes `serenity.genparams.v1` into generated PNGs.
- The daemon indexes jobs in a pure-Mojo SQLite-format DB.
- Stub runtime smoke verified both PNG metadata and `jobs.db` readback.

Limits:

- No complete gallery UX was audited here: persistent grid, import existing image,
  read params from arbitrary PNG, reuse params into controls, delete/rename/star,
  and thumbnail cache remain product work.

### Presets And State Persistence

Status: request schema carries many parameters; named SwarmUI-style presets are
not complete in the audited product path.

The daemon preserves prompt, negative, model, width, height, steps, seed, CFG,
sampler, scheduler, variation seed/strength, image count, init image, creativity,
and LoRA array in canonical params JSON.

Limits:

- Named presets, last-state restoration, UI control persistence, style presets,
  and workflow-level preset versioning were not found as a complete audited layer.

### Queue, Progress, Cancel

Status: core queue exists, SwarmUI semantics incomplete.

The daemon has a serial worker, job list, job status, progress events, and cancel
entrypoint. Process isolation adds a credible way to interrupt/replace large
resident model workers.

Limits:

- Queue reorder/remove, interrupt-current semantics, per-job immutable param
  snapshots in the UI, and progress stage UI need product verification.

### Image Model Generation Speed

Status: not parity; this is a P0 blocker.

The user correction is accurate: image models take too long to generate and the
product generation path is not proven to use the new mojodiffusion kernels.

Evidence:

- `/home/alex/mojodiffusion/serenitymojo/docs/SAMPLER_PRODUCT_HARNESS_2026-06-05.md`
  records Z-Image paired sampler evidence at 1024x1024, seed 42, 28 steps, CFG 3.5:
  OneTrainer denoise `2.144007647s/step`, VAE `1.0628398s`, peak `14340 MiB`;
  paired Mojo denoise `5.206695175s/step`, text encode `33.874484788s`, VAE
  `1.809631493s`, peak `22238 MiB`.
- `/home/alex/mojodiffusion/MOJO_TRAINER_USE_STATUS.md` says Z-Image generation
  remains too slow for production speed parity and identifies two serial CFG
  main-stack passes while OneTrainer batches cond/uncond.
- `/home/alex/mojodiffusion/serenitymojo/docs/HANDOFF_2026-06-12.md` lists an
  active inference speed audit for all image+video models and says samplers still
  run math-mode SDPA while flash is wired only into trainers.
- Source inspection confirms the product Z-Image/Qwen image paths still call
  `sdpa_nomask` from `serenitymojo.ops.attention`, whose implementation returns
  `_sdpa_math`. `sdpa_nomask_slab` is also documented as the same math-mode path
  with only allocation changed.
- `serenitymojo/ops/attention_flash.mojo` exists and exposes cuDNN v9 flash SDPA,
  but its header describes a training fwd+bwd shim. `models/zimage/lora_block.mojo`
  explicitly says the v3 forward keeps math SDPA and the flash scope is graph
  recompute/backward.

Required parity move:

- Move image model load + denoise off the current slow sampler path and onto the
  CUDLASS/CUTLASS-class fast kernel path, or the repo's concrete equivalent
  (`attention_flash`, StepSlab/V2 graph, batched CFG, resident fp8/BF16 weights,
  fused/batched LoRA, and allocator/sync elimination).
- Add a generation-path gate that proves the daemon/UI path, not only trainers,
  uses the fast kernels.
- Required output for each image model: prompt, seed, model, dimensions, steps,
  CFG, sampler/scheduler, text seconds, denoise seconds and seconds/step, VAE
  seconds, total wall time, peak VRAM, artifact path, PNG dimensions, metadata,
  and accepted/not-accepted parity label.

### Image Load Path

Status: functional, not product-speed gated.

`serenitymojo/serve/image_io.mojo` provides pure-Mojo PNG/JPEG/WebP init-image
decode helpers for img2img and UI thumbnail paths. Z-Image has an img2img path
that decodes, resizes, VAE-encodes, and blends init latents.

Limits:

- Image input decode and VAE encode/decode timings are not part of the daemon
  product gate yet.
- If "image loading" refers to model/weight loading, the current path still needs
  a resident/prewarmed CUDLASS-class load plan: no per-job rehydrate, no repeated
  F32 conversion, no avoidable host/device churn.

### img2img / Inpaint

Status: partial.

Z-Image has a real img2img init-image path. Qwen-image rejects img2img. A
SwarmUI-class inpaint/mask editor, masked latent logic, and model-family coverage
are not complete.

### Video / Audio

Status: broken/not parity.

The user explicitly reported video models are broken. The known LTX2 HQ F32
random-noise fallback was fixed and `scripts/check_ltx2_dtype_contract.py` now
passes, but NAVA/LTX2 still do not have an accepted product video path. Components
and docs are present; a daemon/API/UI path that produces and verifies real MP4
with audio is not.

Required video gate:

- BF16/FP8/F16 dtype boundary compliance.
- Real MP4 output with verified duration, frame count, resolution, muxing, and
  audio behavior.
- Per-stage timings: text/conditioning, denoise, VAE/video decode, audio decode
  or vocoder, mux.
- Peak VRAM and explicit limits.
- Daemon/API/UI path, not only standalone parity smoke.

### Prompt Syntax

Status: not SwarmUI parity.

The daemon preserves prompts and structured params, but no complete parser was
found for SwarmUI-style weighted prompt syntax, `<lora:name:weight>`,
`<random:a,b>`, wildcards, regional prompts, or autocomplete. If the UI resolves
these before daemon submit, that needs to be documented and gated.

### Batch, Aspect Presets, Variations

Status: request fields exist, semantics incomplete.

The daemon accepts width/height, variation seed/strength, and image count in the
request JSON. The stub smoke preserved those fields. This is not enough for
SwarmUI parity:

- `images` needs real multi-output batch semantics and gallery indexing.
- Aspect preset controls and validated model-size buckets need UI/backend parity.
- Variation seed/strength must be honored by real backends, not merely stored.

### Upscalers

Status: primitives present, utility product path incomplete.

LTX2 upsampler-related files and Flux/tiling pieces exist, and model scanning can
see relevant families. No complete SwarmUI-style utility tab/endpoint was accepted
in this audit.

### Workflow Graphs / Nodes

Status: explicitly not implemented in daemon.

The daemon rejects `workflow` with 501/reserved behavior. That is honest, but it
means there is no SwarmUI workflow graph parity yet.

## Current P0 Build Order

1. Wire image generation to the new kernel path.
   - Z-Image no-saved inference forwards now route through the cuDNN flash shim
     and the flash SDPA gate passes on real Z-Image shapes.
   - Replace the remaining Qwen product sampler masked `sdpa` math-mode calls
     with an accepted fast path that handles its text-padding mask correctly.
   - Batch CFG cond/uncond instead of two serial main-stack passes.
   - Keep BF16 storage boundaries; F32 only inside compute.
   - Keep generation result manifests strict: timings plus positive Mojo-side
     peak VRAM from a real run.
   - Make `python3 scripts/check_swarmui_product_path_contract.py --strict`
     green for image-fast-path blockers before claiming SwarmUI product parity.

2. Make daemon generation the product gate.
   - Build a standard daemon binary/script with required link flags.
   - Add real Z-Image and Qwen daemon smoke runs with artifact inspection.
   - Exercise WebSocket progress and cancel against a real backend.
   - Keep process isolation for model switches and prove VRAM reclaim with GPU.

3. Fix video before UI expansion.
   - Treat LTX2/NAVA as broken until dtype guard, real artifact, frame/audio, and
     timing/VRAM gates pass.
   - Do not label video output production if it is a tiny smoke, no-audio clip,
     wrong dtype boundary, or missing mux inspection.

4. Build the gallery/reuse-param loop on the existing backend primitives.
   - Use PNG tEXt and `jobs.db`.
   - Add image import, read params, reuse into controls, thumbnails, delete/star,
     and persistent sorting/filtering.

5. Finish model/LoRA browser UX and compatibility.
   - Add search, filter, thumbnails, metadata cache, model cards, family tags,
     compatibility warnings, and multi-LoRA stack support.

6. Add presets/state persistence.
   - Persist last UI state.
   - Add named presets/styles with schema versioning.
   - Keep request JSON canonical and compatible with PNG metadata.

7. Complete queue semantics.
   - Reorder/remove queued jobs.
   - Interrupt current job.
   - Preserve per-job param snapshots in UI and DB.

8. Then add img2img/inpaint mask editor, upscaler tab, prompt syntax, wildcards,
   aspect/batch workflows, live previews, ControlNet/IP-Adapter/regional prompting,
   and workflow graph support.

## Engineering Debt To Fix While Building

- Rename or update stale daemon comments that still describe substantial files as
  "skeleton" when they now carry real product logic.
- Split stub build from CUDA-linked real backend imports, or document that the
  daemon always links CUDA.
- Fix Mojo 1.0.0b1 deprecation warnings in MOJO-libs HTTP/net and the null-pointer
  warning in `serenitymojo/serve/proc_ipc.mojo`.
- Repair the stale SQLite format-test fixture path.
- Make source docs agree on the current inference kernel state. The active TODO,
  MAP files, speed ledgers, and source call sites currently require reconciliation.

## Bottom Line

This is an impressive Mojo stack and it is materially ahead of the June 10 docs.
The native infrastructure is real. The missing piece for SwarmUI parity is now
product integration and hard generation evidence, especially image/video speed and
kernel routing. The next credible milestone is not another standalone smoke; it is
a daemon-driven Z-Image generation that uses the fast CUDLASS/CUTLASS-class path,
reports timings/VRAM, writes gallery metadata, and produces a real inspectable PNG
fast enough to compare against the OneTrainer/SwarmUI baseline.
