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

Current expected status after the LTX2 clone-pair, Z-Image generic UniPC, and
bounded Ideogram4 daemon artifact slices:

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

## Completed Slice: Ideogram4 Dh=256 Fast SDPA

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

- Ideogram4 fast SDPA by itself is **not** accepted SwarmUI backend parity.
- A bounded daemon backend artifact now exists and the latest artifact embeds
  reusable PNG metadata, but it is still only a one-step path/resource smoke. It
  still needs multi-step quality/resource proof, speed/residency work, and
  broader request-surface coverage before any full backend claim.

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

This build was followed by the clone-pair speed slice below. The current
representative staged resident no-audio profile is:

```text
output/ltx2_profile_after_clone_pair/ltx2_runner_timings.json
total_runner_seconds=171.75974076999955
stage1_denoise_seconds=56.413680857000145
stage2_denoise_seconds=74.39678424900012
video_decode_seconds=23.28128546500011
connector_seconds=5.725665003001268
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

## Follow-Up LTX2 Slice: Block Output Clone Fence Coalescing

Intent:

The staged/refhq DiT loops cloned the video and audio block outputs with two
separate `_clone(...)` calls after every transformer block. `_clone` enqueues one
D2D copy and immediately fences with `ctx.synchronize()`, so each block paid two
host/device fences at the output handoff. The new helper enqueues both video and
audio D2D copies on the same stream, then fences once.

Files touched by this follow-up slice:

- `serenitymojo/pipeline/ltx2_t2v_av_hq.mojo`
  - Adds `_TensorPair` and `_clone_pair(video, audio, ctx)`.
  - Uses `_clone_pair` in staged `_model_forward_p` for both NAG and non-NAG
    paths.
  - Uses `_clone_pair` in `_refhq_forward_flat`.

Verification:

```bash
python3 scripts/check_ltx2_dtype_contract.py --scope all
python3 scripts/ltx2_parity_gate.py --only resident_fp8_loader --fail-fast
python3 -m py_compile scripts/check_ltx2_dtype_contract.py scripts/check_swarmui_product_path_contract.py scripts/ltx2_parity_gate.py
pixi run build-video-smoke
```

All passed. The build still emits the existing `len(String)` deprecation
warnings from `ltx2_block_stream.mojo` and the existing unused-assignment warning
in `ops/fp8.mojo`.

Profile command:

```bash
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' \
  output/bin/ltx2_video_smoke_runner staged lora resident noaudio nonag profile \
  output/ltx2_profile_after_clone_pair 1
```

Result:

```text
elapsed=2:52.29 user=159.92 sys=5.38 maxrss=28500504KB
```

`output/ltx2_profile_after_clone_pair/ltx2_runner_timings.json`:

```text
total_runner_seconds=171.75974076999955
stage1_denoise_seconds=56.413680857000145
stage2_denoise_seconds=74.39678424900012
video_decode_seconds=23.28128546500011
connector_seconds=5.725665003001268
```

The immediately previous post-no-sync profile
`output/ltx2_profile_after_no_sync/ltx2_runner_timings.json` was:

```text
total_runner_seconds=314.8831040389996
stage1_denoise_seconds=124.44027067600109
stage2_denoise_seconds=79.05193418400086
video_decode_seconds=23.63663906399961
connector_seconds=62.52942450699993
```

Same-profile comparison:

```text
total_runner_seconds: 314.883104 -> 171.759741 (1.833x)
stage1_denoise_seconds: 124.440271 -> 56.413681 (2.206x)
stage2_denoise_seconds: 79.051934 -> 74.396784 (1.063x)
connector_seconds: 62.529425 -> 5.725665 (10.921x)
```

Artifact probe:

```text
output/ltx2_profile_after_clone_pair/ltx2_t2v_stage2_dev_smoke.mp4
768x512, 121 frames, 5.041667s, 24fps, H.264, no audio stream
```

Interpretation:

- The patch is a real speed improvement in the profiled staged resident path.
- The remaining measured denoise bottleneck is still per-block load/materialize
  overhead, not attention: after this patch, representative forward profile
  lines show `load≈22s`, `lora≈0.8s`, and `block≈3.7-13.2s`.
- This still does not accept full video parity. The artifact remains a bounded
  `DEV SMOKE ONLY` output.

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
- `total_wall_seconds=174.955233998`
- runner `total_runner_seconds=174.44195223299903`
- external peak VRAM delta `10476 MiB`
- peak used `11226 MiB`
- `accepted_video_parity:false`
- `accepted_sampler_parity:false`

Measured timings:

- `stage1_denoise_seconds=57.315346847999535`
- `stage2_denoise_seconds=75.27145183599896`
- `video_decode_seconds=23.61521103099949`
- `frame_png_write_seconds=1.7494108600003528`
- `video_mux_seconds=0.6264517420004267`

Prior stream no-audio comparison:

- `total_wall_seconds=202.353546797`
- `stage1_denoise_seconds=73.01245590500002`
- `stage2_denoise_seconds=90.72237481299999`
- `peak used=10845 MiB`

Interpretation:

- Resident mode plus the no-sync resident materializer and clone-pair fence
  coalescing now improve the bounded wall time by about `27.4s` over the prior
  stream no-audio gate.
- This is still bounded video-smoke speed evidence, not full SwarmUI/HQ video
  parity.

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

## Current Ideogram4 Backend Evidence

Bounded native Ideogram4 daemon proof:

```text
output/serenity_daemon/job-0106.png
output/serenity_daemon/job-0106.png.ideogram4_daemon_result.json
```

Repeatable checker:

```bash
python3 scripts/check_ideogram4_daemon_product_contract.py --artifact \
  output/serenity_daemon/job-0106.png --json
```

Known evidence:

- Native `Ideogram4Backend` path exists; it does not shell out to Python.
- PNG dimensions: `1024x1024`.
- PNG tEXt key: `serenity.genparams.v1`.
- Sidecar manifest schema: `serenity.ideogram4.daemon_result.v1`.
- `readiness_label:"experimental"`.
- `requested_sampler:"euler"`, `requested_scheduler:"logitnormal"`.
- `executed_sampler:"ideogram4_logitnormal_euler"`,
  `executed_scheduler:"ideogram4_logitnormal"`.
- Fixed `1024` token text window, `prompt_tokens:15`, `lora_count:0`,
  `variation_applied:false`.
- `accepted_sampler_parity:false`, `accepted_speed_parity:false`.
- The contract checker reports `bounded_artifact_ready:true` and
  `runtime_acceptance:false`.
- `/v1/gallery/read?path=/home/alex/mojodiffusion/output/serenity_daemon/job-0106.png`
  returns `has_params:true`, `metadata_key:"serenity.genparams.v1"`, and
  params matching the request.
- PNG IDAT SHA256:
  `95916a866934ec86df2cb18baefc392aad8d4dbabca918df7df7b759f658a215`.
- Bounded fail-loud unsupported-option smoke:
  `python3 scripts/check_ideogram4_daemon_product_contract.py
  --fail-loud-smoke --write-readiness
  output/checks/ideogram4_daemon_product_readiness.json --json`.
  Latest run proved HTTP `422` prequeue rejection for negative prompt, LoRA,
  prompt LoRA tag, init image, non-default creativity/denoise, variation,
  unsupported size, unsupported sampler, unsupported scheduler, and nonpositive
  CFG. `/v1/jobs` stayed at `126` rows, and the daemon log had
  `expensive_markers_seen:[]` for Ideogram text/DiT/VAE load markers.

Measured timings:

- `load_seconds=75.44370386599999`
- `text_encode_seconds=135.538652776`
- `prepare_seconds=1.928370351`
- `denoise_seconds=6.174464852`
- `vae_decode_seconds=2.190651774`
- `total_wall_seconds=221.428687069`
- `peak_vram_mib=22088.6875`

Known non-claims:

- This is one-step smoke evidence, not 20-step quality acceptance.
- PNG tEXt metadata and gallery readback are proven for the bounded one-step
  artifact only.
- Transformers are resident across denoise steps, not across jobs on the
  24GB-class GPU.
- Negative prompt, LoRA, init image/img2img, creativity/denoise, variation,
  unsupported sampler/scheduler, unsupported size, and bad CFG now fail loud for
  the bounded request set, but support for those features is still absent.
- Speed parity, quality parity, sampler parity, and broad request-surface parity
  remain unaccepted.

## Current Sampler / Workflow Evidence

Accepted bounded sampler facts:

- Z-Image DPM++ 2M/simple-flowmatch has bounded daemon artifact evidence.
- Z-Image UniPC bh2/simple-flowmatch has bounded daemon artifact evidence.
- Z-Image generic `uni_pc` now has bounded daemon artifact evidence. The product
  checker emitted `output/serenity_daemon/job-0077.png` plus
  `output/serenity_daemon/job-0077.png.zimage_daemon_result.json` from a
  512x512 4-step `sampler:"uni_pc"`, `scheduler:"flowmatch"` smoke. The manifest
  records `solver_type:"bh1"`, `solver_variant:"bh1"`, `solver_order:3`,
  `sigma_parameterization:"SigmaConvert"`,
  `schedule_source:"zimage_build_sigmas+comfy_discard_penultimate+comfy_unipc_timesteps"`,
  `unipc_update_steps:4`, `unipc_corrector_steps:3`,
  `unipc_second_order_steps:2`, `denoise_seconds_per_step:0.29673903050000006`,
  `peak_vram_mib:21379.875`, and `accepted_sampler_parity:false`.
- Z-Image flat img2img/creativity now has bounded daemon artifact evidence. The
  product checker emitted baseline `output/serenity_daemon/job-0087.png` and
  img2img outputs `job-0088.png`, `job-0089.png`, and `job-0090.png`, reusing
  `job-0087.png` as `init_image`. The 8-step img2img smoke records
  `img2img_applied:true`, `denoise_start_step`, `steps_executed`,
  `denoise_update_steps`, timings, and positive VRAM for creativity `0.0`,
  `0.5`, and `1.0`. Current evidence:
  `0.0 -> denoise_start_step:8, steps_executed:0`,
  `0.5 -> denoise_start_step:6, steps_executed:1`, and
  `1.0 -> denoise_start_step:0, steps_executed:7`; peak VRAM was
  `21393.25 MiB`. Duplicate terminal zero sigma intervals are now treated as
  no-op sentinels. This remains bounded flat-parameter evidence only and keeps
  `accepted_img2img_parity:false`.
- Latest full Z-Image daemon product gate after this scheduler-accounting patch:
  `python3 scripts/check_zimage_daemon_product_contract.py --timeout 900
  --steps 1 --write-readiness output/checks/zimage_daemon_product_readiness.json`
  passed. It covered unsupported sampler failure (`job-0091`), baseline
  `job-0092`, img2img `job-0093` through `job-0095`, DPM++ 2M `job-0096`,
  generic UniPC `job-0097`, UniPC bh2 `job-0098`, variation `job-0099`,
  multi-image, and multi-LoRA `job-0101`. The latest img2img evidence keeps
  `0.0 -> denoise_start_step:8, steps_executed:0`,
  `0.5 -> denoise_start_step:6, steps_executed:1`, and
  `1.0 -> denoise_start_step:0, steps_executed:7`.
- `images=N` emits serial indexed daemon jobs with seed offsets and metadata.
- Variation noise is implemented for image backends and proven for Z-Image by a
  daemon artifact gate.
- `/v1/samplers` exposes a SwarmUI/Comfy catalog and backend support matrix.
- `/v1/samplers` now includes `ideogram4` with bounded `euler`/flow-match
  sampler aliases executing as `ideogram4_logitnormal_euler`, and only
  `logitnormal`/`logit_normal`/`ideogram_logitnormal`/
  `ideogram4_logitnormal` scheduler aliases.
- Unsupported sampler names fail loud before model work.

Known non-claims:

- Generic `uni_pc` is not accepted by aliasing it to `uni_pc_bh2`; it now has a
  distinct bounded bh1 artifact.
- Generic `uni_pc` still does not make `accepted_sampler_parity:true`; the
  remaining sampler catalog still needs distinct artifact/timing/VRAM evidence.
- Ancestral, SDE, Karras, CFG++, and other daemon denoise loops remain
  incomplete.
- Z-Image img2img evidence does not cover graph `LoadImage`/`VAEEncode`,
  masks, inpaint, or full quality parity.
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
3. Ideogram4 full backend parity.
   - Fast attention is wired.
   - A bounded native daemon artifact with PNG metadata/gallery readback exists
     as `job-0106`.
   - A bounded prequeue fail-loud option gate exists in
     `output/checks/ideogram4_daemon_product_readiness.json`.
   - Quality, sampler, speed, LoRA support, img2img support, variation support,
     and broad request-surface parity remain unaccepted.
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

Immediate continuation after the LTX2 clone-pair and img2img scheduler-accounting
slices:

1. Next real LTX2 speed work:
   - attack remaining per-block load/materialize overhead in LTX2 staged/refhq,
   - profile with `nsys` only after a stable clone-pair baseline is retained,
   - then decide whether CUTLASS/cuBLASLt FP8 is actually worth lifting into
     this path.
2. Next image backend work:
   - harden Ideogram4 beyond `job-0106`: multi-step quality and resource
     evidence, speed/residency measurement, and broader request-surface gates.
   - Use the same product acceptance shape as Z-Image before making any full
     backend claim: PNG metadata, job DB/gallery, timing, VRAM, and fail-loud
     unsupported options.
3. Next sampler/workflow breadth work:
   - keep Z-Image img2img accounting tied to real update intervals
     (`job-0088` through `job-0090` prove the current bounded gate),
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
