# HANDOFF - SwarmUI Parity Push - 2026-06-12

This is the pick-up document for the active goal: **full SwarmUI parity** for
`mojodiffusion` / `serenityUI` product paths, with runtime/product generation
kept Mojo-native. Python is allowed for development support, static checkers,
artifact inspection, and parity oracles only.

Current branch:

```bash
git -C /home/alex/mojodiffusion status --short --branch
```

Expected branch at handoff time:

```text
training-port-5models-lora...origin/training-port-5models-lora
```

Last pushed commit before the LTX2 resident no-sync slice:

```text
43692de Route Ideogram4 attention through flash SDPA
```

Canonical output root:

```text
/home/alex/mojodiffusion/output/
```

Daemon artifacts live under:

```text
output/serenity_daemon/
```

Do not write new generated artifacts into `mojodiffusion/` root, `serenitymojo/`,
or model source directories. Use `output/bin`, `output/checks`,
`output/serenity_daemon`, or a specific `output/<gate-name>` directory.

## Non-Negotiable Rules

- Do not mark full SwarmUI parity complete from smoke artifacts.
- Do not call a tiny/noisy/broken artifact production-ready.
- Do not claim video parity while outputs are labeled `DEV SMOKE ONLY` or
  `accepted_video_parity:false`.
- Do not run full Qwen generation without bounded VRAM/runtime evidence; Qwen is
  still too large/slow/OOM-prone for blind runs.
- Do not move product/runtime generation to Python. Python can inspect, compare,
  generate oracle dumps, or run static checkers.
- Preserve dtype boundaries. For LTX/LTX2, BF16/F16/FP8 storage must not be
  upcast to F32 at tensor boundaries unless a nearby comment explains the exact
  reference reason.
- For video artifacts, inspect frame count, resolution, duration, muxing, audio
  behavior, timing, and VRAM before making any readiness claim.
- For image artifacts, inspect dimensions, PNG metadata
  `serenity.genparams.v1`, job DB/gallery state, timing, VRAM, and visual sanity.
  A gray/noisy texture output is not SwarmUI parity.
- Keep the full goal intact. P0/P1 tracked product gates being green is not full
  SwarmUI parity.

## Current High-Level Status

Tracked static/product path gate:

```bash
python3 scripts/check_swarmui_product_path_contract.py \
  --write-readiness output/checks/swarmui_product_path_readiness.json
```

Current expected status after the Ideogram4 fast-attention slice:

```text
checks=75 passed=75 p0=0 p1=0 p2=0
P0 product path: READY
tracked P0/P1 product gates: READY
SwarmUI all-level parity: BLOCKED
```

Meaning in plain English:

- The current tracked product-path checklist is green.
- Full SwarmUI parity is still false.
- The remaining work is real backend depth, quality, sampler/workflow breadth,
  and runtime performance, not just more labels.

## Latest Completed Slice: Ideogram4 Dh=256 Fast SDPA

Pushed commit:

```text
43692de Route Ideogram4 attention through flash SDPA
```

Files changed by that pushed slice:

- `serenitymojo/models/dit/ideogram4_dit.mojo`
- `serenitymojo/models/dit/ideogram4_resident.mojo`
- `serenitymojo/ops/tests/sdpa_flash_parity.mojo`
- `scripts/check_swarmui_product_path_contract.py`
- `serenitymojo/docs/IDEOGRAM4_STATUS.md`
- `serenitymojo/docs/SERENITYMOJO_MODULES.md`
- `serenitymojo/docs/SWARMUI_MODEL_GALLERY_LORA_PARITY_MAP_2026-06-12.md`
- `serenitymojo/docs/SWARMUI_PARITY_ROADMAP_2026-06-12.md`
- `serenitymojo/docs/SWARMUI_PRODUCT_PATH_LEDGER_2026-06-12.md`

What changed:

- Added `ideogram4_sdpa_product_fwd` in
  `serenitymojo/models/dit/ideogram4_dit.mojo`.
- Reference and resident Ideogram4 DiT attention now call
  `ideogram4_sdpa_product_fwd[1,S,18,256]`.
- Direct product `sdpa_nomask[1,S,18,256]` was removed from the Ideogram4 DiT
  forward path.
- Added forward-only Dh=256 cuDNN SDPA parity/speed cases in
  `serenitymojo/ops/tests/sdpa_flash_parity.mojo`:
  `ideogram4_fwd_aligned` and `ideogram4_fwd_pad`.
- The checker now guards the Ideogram4 fast-attention dispatch and absence of
  direct Dh=256 math SDPA in both reference and resident paths.

Verified evidence:

```bash
pixi run mojo build --target-accelerator sm_86 -I . \
  -Xlinker -lm -Xlinker -lcuda \
  -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
  -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
  serenitymojo/ops/tests/sdpa_flash_parity.mojo \
  -o output/bin/sdpa_flash_parity
```

```bash
LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
  output/bin/sdpa_flash_parity
```

Key results:

- `ideogram4_fwd_aligned`, `S=1024`: flash `0.5990126 ms`, math
  `5.4428871 ms`, speedup `9.086x`, PASS.
- `ideogram4_fwd_pad`, `S=1153`: flash `0.9248282 ms`, math `7.141845 ms`,
  speedup `7.722x`, PASS.
- Existing Klein/Z-Image SDPA cases also passed.
- Final line: `ALL GATES PASS`.

Fixture/probe evidence:

```bash
HF_HUB_OFFLINE=1 python3 serenitymojo/models/dit/parity/ideogram4_oracle.py A
HF_HUB_OFFLINE=1 python3 serenitymojo/models/dit/parity/ideogram4_oracle.py P
```

```bash
pixi run mojo build --target-accelerator sm_86 -I . \
  -Xlinker -lm -Xlinker -lcuda \
  -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
  -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
  serenitymojo/models/dit/parity/chunk6r_resident_probe.mojo \
  -o output/bin/ideogram4_chunk6r_resident_probe
```

```bash
output/bin/ideogram4_chunk6r_resident_probe
```

Result:

```text
resident fp8 DiT vs fixture:
ParityResult(cos=0.9995574960620331, max_abs=0.4228515625, n=33280, PASS)
```

Standalone full DiT probe:

```bash
pixi run mojo build --target-accelerator sm_86 -I . \
  -Xlinker -lm -Xlinker -lcuda \
  -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
  -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
  serenitymojo/models/dit/parity/chunk6_dit_probe.mojo \
  -o output/bin/ideogram4_chunk6_dit_probe
```

```text
chunk6 full DiT velocity parity:
ParityResult(cos=0.9995574960620331, max_abs=0.4228515625, n=33280, PASS)
```

Important non-claim:

- Ideogram4 is **not** an accepted SwarmUI backend yet.
- It still needs a daemon backend that emits a real artifact with dimensions,
  `serenity.genparams.v1`, timing, positive peak VRAM, job DB/gallery evidence,
  and fail-loud unsupported option behavior.

## Current LTX2 Slice: Resident FP8 No-Sync Dequant

Intent:

The LTX2 staged resident path already preloads raw FP8 blocks into VRAM, but
each denoise forward still materialized resident raw FP8 tensors as BF16 through
the same synchronized dequant API used by streamed loads. Block 4 has 34 FP8
tensors. Blocks 4-12 are resident in the bounded daemon path, and the 1-step
gate performs multiple DiT forwards. The old shape meant hundreds to thousands
of avoidable host/device fences in the denoise path.

Patch direction:

- Keep `fp8_e4m3_dequant_to_bf16` synchronized for public/default, streamed,
  Qwen, tests, and debug paths.
- Add `fp8_e4m3_dequant_to_bf16_no_sync`.
- Use the no-sync API **only** inside
  `LTX2BlockStream._load_resident_block_bf16`.
- Rely on same-`DeviceContext` stream ordering and a downstream/final sync
  before host readback.
- Do not change math or dtype. Raw FP8 still dequants to BF16, then the existing
  BF16/cuBLAS path consumes it.

Files touched by this slice:

- `serenitymojo/ops/fp8.mojo`
  - Factored the per-tensor dequant wrapper into
    `_fp8_e4m3_dequant_to_bf16_impl(..., sync_after_launch: Bool)`.
  - Kept `fp8_e4m3_dequant_to_bf16(...)` as synchronized API.
  - Added `fp8_e4m3_dequant_to_bf16_no_sync(...)`.
- `serenitymojo/offload/ltx2_block_stream.mojo`
  - Imports the no-sync API.
  - `_load_resident_block_bf16` uses the no-sync API for resident raw FP8.
  - `load_block_bf16` streamed path still uses the synchronized API.
- `scripts/check_ltx2_dtype_contract.py`
  - `--scope all` now includes `serenitymojo/offload/ltx2_block_stream.mojo`
    and `serenitymojo/ops/fp8.mojo`.
  - Adds a resident FP8 contract check:
    - no-sync API must exist,
    - resident materializer must use it,
    - streamed loader must not use it.
- `scripts/ltx2_parity_gate.py`
  - Adds `resident_fp8_loader` gate.
- `scripts/check_swarmui_product_path_contract.py`
  - Updates known all-level blockers so they no longer falsely say video has no
    MP4 evidence or Ideogram4 is still on slow SDPA.
- Docs updated:
  - `serenitymojo/docs/SERENITYMOJO_MODULES.md`
  - `serenitymojo/docs/SWARMUI_PARITY_ROADMAP_2026-06-12.md`
  - `serenitymojo/docs/SWARMUI_PRODUCT_PATH_LEDGER_2026-06-12.md`
  - `serenitymojo/docs/LTX2_TODO.md`
  - `serenitymojo/parity/LTX2_PARITY_MATRIX_2026-06-04.md`
  - this handoff doc.

Focused verification run for this slice:

Build FP8 dequant smoke:

```bash
pixi run mojo build --target-accelerator sm_86 -I . \
  serenitymojo/pipeline/fp8_dequant_smoke.mojo \
  -o output/bin/fp8_dequant_smoke
```

Build resident loader smoke:

```bash
pixi run mojo build --target-accelerator sm_86 -I . \
  serenitymojo/pipeline/ltx2_fp8_resident_smoke.mojo \
  -o output/bin/ltx2_fp8_resident_smoke
```

Sandbox note:

Running these GPU binaries inside the managed sandbox failed with:

```text
Failed to initialize NVML: 9
```

Rerun outside the sandbox is required for GPU/NVML runtime checks.

FP8 bit-exact smoke outside sandbox:

```bash
output/bin/fp8_dequant_smoke
```

Result:

```text
Test A (all 256 E4M3 bytes, scale=1.0) PASS ( 256 / 256 bit-exact)
Test B1 (all 256 bytes, scale=0.5) PASS ( 256 / 256 bit-exact)
Test B2 (all 256 bytes, scale=2.0) PASS ( 256 / 256 bit-exact)
Test B3 (all 256 bytes, real scale) PASS ( 256 / 256 bit-exact)
Test C (real ckpt block.4.attn1.to_q, torch ref) PASS ( 64 / 64 )
fp8 dequant smoke summary: 5 / 5
```

Resident loader smoke outside sandbox:

```bash
output/bin/ltx2_fp8_resident_smoke
```

Result:

```text
=== LTX2 FP8 resident loader smoke ===
[resident] preload block 4 only
[resident] bytes: 386924928  ( 369  MiB )
[block4] fp8 tensor count: 34
[block4] attn1.to_q dtype: BF16
[block4] audio_attn1.to_q dtype: BF16
FP8 RESIDENT GATE PASS
```

Dtype sidecar guard:

```bash
python3 scripts/check_ltx2_dtype_contract.py
```

Result:

```text
LTX2 dtype/RNG contract static guard: pass (scope=sidecar)
```

Dtype all-runtime guard:

```bash
python3 scripts/check_ltx2_dtype_contract.py --scope all
```

Result:

```text
LTX2 dtype/RNG contract static guard: pass (scope=all)
```

Named LTX2 parity gate:

```bash
python3 scripts/ltx2_parity_gate.py --only resident_fp8_loader --fail-fast
```

Result:

```text
PASS resident_fp8_loader
failures: 0
skipped: 0
```

Static Python/script checks:

```bash
python3 -m py_compile \
  scripts/check_ltx2_dtype_contract.py \
  scripts/check_swarmui_product_path_contract.py \
  scripts/ltx2_parity_gate.py
```

Result: pass.

SwarmUI product-path contract:

```bash
python3 scripts/check_swarmui_product_path_contract.py \
  --strict \
  --write-readiness output/checks/swarmui_product_path_readiness.json
```

Result:

```text
checks=75 passed=75 p0=0 p1=0 p2=0
P0 product path: READY
tracked P0/P1 product gates: READY
SwarmUI all-level parity: BLOCKED
```

Video runner build:

```bash
pixi run build-video-smoke
```

Result:

- Passed.
- Emits `output/bin/ltx2_video_smoke_runner`.
- Warnings were existing string `len(...)` deprecation warnings in
  `ltx2_block_stream.mojo` and a pre-existing unused-assignment warning in
  `ops/fp8.mojo`.

Optional but valuable if GPU time is available:

```bash
output/bin/ltx2_video_smoke_runner \
  staged lora resident noaudio nonag profile output/ltx2_profile_after_no_sync 1
```

Then compare against the previous resident no-audio daemon evidence:

```text
total_wall_seconds=186.882605556
stage1_denoise_seconds=57.9751922939995
stage2_denoise_seconds=76.74856130400076
video_decode_seconds=28.060338318000504
peak used=11490 MiB
peak delta=10501 MiB
```

If you run the daemon-level video checker, use:

```bash
python3 scripts/check_ltx2_video_daemon_product_contract.py \
  --timeout 520 \
  --weight-mode resident \
  --audio-mode noaudio \
  --strict-artifact \
  --write-readiness output/checks/ltx2_video_daemon_resident_readiness.json
```

This is expensive, but it is the product-path gate.

## Current LTX2 Video Evidence

Bounded video-only daemon artifact:

```text
output/serenity_daemon/video-0074/ltx2_t2v_stage2_dev_smoke.mp4
```

Known evidence from the current docs/ledger:

- `width=768`
- `height=512`
- `frame_count=121`
- `duration=5.041667`
- `fps=24.0`
- `video_codec=h264`
- `muxing=probe_ok`
- `audio_behavior=video_only_no_audio_stream`
- `mode:"staged lora resident noaudio nonag"`
- `weight_mode:"resident"`
- `total_wall_seconds=186.882605556`
- `runner total_runner_seconds=186.00316528799885`
- external peak VRAM delta `10501 MiB`
- peak used `11490 MiB`
- `accepted_video_parity:false`
- `accepted_sampler_parity:false`

Measured timings:

- `stage1_denoise_seconds=57.9751922939995`
- `stage2_denoise_seconds=76.74856130400076`
- `video_decode_seconds=28.060338318000504`
- `frame_png_write_seconds=1.718876539998746`
- `video_mux_seconds=1.1828436699997837`

Prior stream no-audio comparison:

- `total_wall_seconds=202.353546797`
- `stage1_denoise_seconds=73.01245590500002`
- `stage2_denoise_seconds=90.72237481299999`
- `peak used=10845 MiB`

Interpretation:

- Resident mode already improved bounded wall time by about `15.5s` over stream.
- The current no-sync resident dequant patch should target denoise load/fence
  time. Do not claim a new speedup until a fresh profile or daemon run measures
  it.

Audio-enabled A/V daemon artifact:

```text
output/serenity_daemon/video-0072/ltx2_t2v_av_stage2_dev_smoke.mp4
output/serenity_daemon/video-0072/dev_audio.wav
```

Known evidence:

- `stream_count=2`
- `audio_codec=aac`
- `audio_duration=5.034`
- `audio_behavior=audio_stream_present`
- `total_wall_seconds=202.223939339`
- runner `total_runner_seconds=201.75603515000148`
- external peak VRAM delta `9880 MiB`

Interpretation:

- A/V artifact path exists.
- This is still smoke/dev output, not full video parity.

## Current Z-Image Evidence

Current 1024x1024 Z-Image daemon product proof:

```text
output/serenity_daemon/job-0073.png
output/serenity_daemon/job-0073.png.zimage_daemon_result.json
```

Known evidence:

- `1024x1024`
- 1 step experimental artifact
- tiled 64-latent VAE decode path instead of former whole-frame 128-latent
  decoder path
- PNG `serenity.genparams.v1`
- `jobs.db`
- gallery/read endpoints
- manifest timings
- positive VRAM evidence

Measured timings:

- `load_seconds=2.49144893`
- `text_encode_seconds=3.132651268`
- `denoise_seconds=2.078721406`
- `vae_decode_seconds=3.266865353`
- `total_wall_seconds=11.438153728`
- `peak_vram_mib=21497.3125`

Interpretation:

- This proves a real product path and resource behavior.
- It is not quality, sampler, or speed parity.

## Current Sampler / Workflow Evidence

Accepted bounded sampler facts:

- Z-Image DPM++ 2M/simple-flowmatch has bounded daemon artifact evidence.
- Z-Image UniPC bh2/simple-flowmatch has bounded daemon artifact evidence.
- `images=N` emits serial indexed daemon jobs with seed offsets and metadata.
- Variation noise is implemented for image backends and proven for Z-Image by a
  daemon artifact gate.
- `/v1/samplers` exposes a SwarmUI/Comfy catalog and backend support matrix.
- Unsupported sampler names fail loud before model work.

Known non-claims:

- Generic `uni_pc` is not accepted by aliasing it to `uni_pc_bh2`.
- Generic UniPC/order-3, ancestral, SDE, Karras, CFG++, and other daemon denoise
  loops remain incomplete.
- True Comfy-style batched latent execution is not implemented.

Workflow:

- Typed linked t2i graph executor exists for:
  - `CheckpointLoaderSimple`
  - `CLIPTextEncode`
  - `EmptyLatentImage`
  - `KSampler`
  - `VAEDecode`
  - `SaveImage`
- Advanced Comfy/Swarm node families remain explicit blockers and must fail
  loudly until implemented.

## Current UI / Gallery / State Evidence

The tracked UI/gallery/reuse/state runtime checker is green.

Known accepted API slice:

- gallery read/import
- PNG metadata readback
- reuse provenance
- restart-safe job history
- indexed external gallery import
- gallery rename/manual order
- presets/state restart
- favorite/delete
- queue mutation
- reuse generate

Known non-claim:

- This is API-level parity, not full UI visual/interaction parity.

## Remaining Full SwarmUI Blockers

Do not mark the goal complete until these are actually done and proven:

1. Qwen full daemon generation/resource proof.
   - Full generation is parked until bounded VRAM/runtime evidence says it is
     safe.
   - Current Qwen evidence is fast-path/static/tiny parity only.
2. Full LTX2 video parity.
   - Current daemon can emit bounded smoke MP4s with timings/VRAM.
   - Full SwarmUI/HQ parity still needs non-smoke quality, duration, audio,
     workflow, option coverage, and visual/audio acceptance.
3. Ideogram4 daemon backend.
   - Fast attention is wired.
   - No accepted daemon backend artifact yet.
4. Z-Image speed parity.
   - Product path works.
   - Paired baseline/optimized CFG/main-stack speed evidence is still needed.
5. Sampler/scheduler breadth.
   - Many SwarmUI/Comfy sampler variants remain surface-only or blocked.
6. Advanced workflow node families.
   - Control/IP/regional/mask/upscale/video graph nodes are not accepted.
7. Model/gallery/LoRA breadth.
   - Z-Image multi-LoRA PEFT/Comfy is accepted for its path.
   - Qwen LoRA is still zero-LoRA.
   - Z-Image LoKr/LyCORIS overlay conversion is still not complete.
8. Video model failures beyond LTX2 smoke.
   - User explicitly stated no video models work as production paths.
   - Treat LTX2 smoke as a gate, not the finish line.
9. Quality.
   - Gray/noisy/texture-like outputs are not acceptable proof.
   - Artifact existence is only evidence that a path runs, not that it is
     usable.

## Suggested Next Work Order

Immediate continuation after the LTX2 resident no-sync slice is committed:

1. If GPU time permits, run a fresh resident no-sync profile:
   - `output/bin/ltx2_video_smoke_runner staged lora resident noaudio nonag profile output/ltx2_profile_after_no_sync 1`
   - Compare `stage1_denoise`, `stage2_denoise`, total wall, and peak VRAM
     against video-0074.
2. Next real speed work after this slice:
   - remove remaining hot linear/FP8 clone/sync points in LTX2 staged/refhq,
   - profile with `nsys` only after the no-sync resident path is verified,
   - then decide whether CUTLASS/cuBLASLt FP8 is actually worth lifting into
     this path.
3. Next image backend work:
   - implement the Ideogram4 daemon backend and artifact gate.
   - Use the same product acceptance shape as Z-Image: PNG, metadata, job DB,
     gallery, timing, VRAM, fail-loud unsupported options.
4. Next sampler/workflow breadth work:
   - promote only sampler variants with runtime artifact/timing/VRAM evidence,
   - keep unsupported Comfy/Swarm nodes fail-loud until they have typed graph
     execution and product-path tests.

## Commands To Keep Handy

Build daemon:

```bash
pixi run build-daemon
```

Product contract:

```bash
python3 scripts/check_swarmui_product_path_contract.py \
  --strict \
  --write-readiness output/checks/swarmui_product_path_readiness.json
```

LTX2 dtype guard:

```bash
python3 scripts/check_ltx2_dtype_contract.py --scope all
```

LTX2 parity gate list:

```bash
python3 scripts/ltx2_parity_gate.py --list
```

LTX2 resident FP8 loader gate:

```bash
python3 scripts/ltx2_parity_gate.py --only resident_fp8_loader --fail-fast
```

Build video runner:

```bash
pixi run build-video-smoke
```

Video daemon product gate:

```bash
python3 scripts/check_ltx2_video_daemon_product_contract.py \
  --timeout 520 \
  --weight-mode resident \
  --audio-mode noaudio \
  --strict-artifact \
  --write-readiness output/checks/ltx2_video_daemon_resident_readiness.json
```

Z-Image daemon product gate:

```bash
python3 scripts/check_zimage_daemon_product_contract.py \
  --write-readiness output/checks/zimage_daemon_readiness.json
```

UI/gallery/reuse/state contract:

```bash
python3 scripts/check_ui_gallery_reuse_state_contract.py \
  --write-readiness output/checks/ui_gallery_reuse_state_readiness.json
```

Workflow graph contract:

```bash
python3 scripts/check_workflow_graph_product_contract.py
```

Diff hygiene:

```bash
git -C /home/alex/mojodiffusion diff --check
```

## Files To Read First In A Fresh Session

1. `serenitymojo/docs/HANDOFF_SWARMUI_PARITY_2026-06-12.md`
2. `serenitymojo/docs/SWARMUI_PARITY_ROADMAP_2026-06-12.md`
3. `serenitymojo/docs/SWARMUI_PRODUCT_PATH_LEDGER_2026-06-12.md`
4. `serenitymojo/docs/SWARMUI_MODEL_GALLERY_LORA_PARITY_MAP_2026-06-12.md`
5. `serenitymojo/docs/SWARMUI_SAMPLER_PARITY_MAP_2026-06-12.md`
6. `serenitymojo/docs/COMFY_SWARM_WORKFLOW_PARITY_MAP_2026-06-12.md`
7. `serenitymojo/parity/LTX2_PARITY_MATRIX_2026-06-04.md`
8. `serenitymojo/docs/LTX2_TODO.md`
9. `serenitymojo/docs/IDEOGRAM4_STATUS.md`
10. `scripts/check_swarmui_product_path_contract.py`
11. `scripts/check_ltx2_dtype_contract.py`
12. `scripts/ltx2_parity_gate.py`

## Agent Notes

The most useful read-only agent finding for the current LTX2 slice:

- `fp8_e4m3_dequant_to_bf16` synchronized after every FP8 tensor.
- `_load_resident_block_bf16` was calling it for resident raw FP8 blocks.
- Resident raw FP8 blocks are already held by `self.resident_blocks`, so input
  lifetime is stable.
- Same-stream ordering should make the no-sync materialized BF16 tensors safe
  for later kernels, as long as the no-sync helper stays resident-only.
- Estimated fence removal for the bounded no-audio one-step gate:
  `34 FP8 tensors * 9 resident blocks * 4 DiT forwards = 1224` potential sync
  points removed.

The risk:

- If no-sync dequant leaks into streamed/non-resident paths, lifetime and debug
  semantics become harder to reason about.
- The dtype guard now exists to prevent that leak.

## Verification Record For Current Slice

These checks passed for the LTX2 no-sync slice:

```bash
python3 scripts/check_ltx2_dtype_contract.py --scope all
python3 scripts/ltx2_parity_gate.py --only resident_fp8_loader --fail-fast
python3 -m py_compile scripts/check_ltx2_dtype_contract.py scripts/check_swarmui_product_path_contract.py scripts/ltx2_parity_gate.py
python3 scripts/check_swarmui_product_path_contract.py --strict --write-readiness output/checks/swarmui_product_path_readiness.json
```

Before any future commit, still run:

```bash
git -C /home/alex/mojodiffusion diff --check
git -C /home/alex/mojodiffusion status --short --branch
```

This slice should stage only source/checker/docs files, not generated binaries
or fixture data. Intended file list:

```bash
git add \
  scripts/check_ltx2_dtype_contract.py \
  scripts/check_swarmui_product_path_contract.py \
  scripts/ltx2_parity_gate.py \
  serenitymojo/docs/HANDOFF_SWARMUI_PARITY_2026-06-12.md \
  serenitymojo/docs/LTX2_TODO.md \
  serenitymojo/docs/SERENITYMOJO_MODULES.md \
  serenitymojo/docs/SWARMUI_PARITY_ROADMAP_2026-06-12.md \
  serenitymojo/docs/SWARMUI_PRODUCT_PATH_LEDGER_2026-06-12.md \
  serenitymojo/offload/ltx2_block_stream.mojo \
  serenitymojo/ops/fp8.mojo \
  serenitymojo/parity/LTX2_PARITY_MATRIX_2026-06-04.md
```

Expected commit title:

```bash
git commit -m "Fence LTX2 resident FP8 dequant synchronization"
git push
```

Do not mark the active goal complete after this commit. It is a real movement
toward full SwarmUI parity, but full parity remains open.
