# SCAIL-2 Mojo intake — 2026-07-19

## Proven scope

This intake proves the official single-segment SCAIL-2 base-animation path on
an RTX 5080 (16 GB): 896x512, 65 frames, 40 UniPC steps, shift 3, CFG 5, seed
42. Production input ingest, conditioning, denoising, the streamed FP8
transformer, Wan VAE decode, and automatic reference-audio handling run in
Mojo. Python is limited to one-time development conversion and parity oracles.

The admitted base-animation profile is wired into Serenity Studio's Generate
surface and `/v1/video`. The lower-level CLIs emit checksum-bound stage,
conditioning, prompt, run, or result sidecars for development evidence. Those
artifacts live outside the repository under `$SERENITY_HOME/runs/`; they are
not user output.

Replacement mode and one-to-three paired additional-reference inputs are wired
through Serenity Studio and `/v1/video` and have real product-geometry gates
below. Pose-driven raw-video preprocessing and long-video segment chaining are
represented by the pinned creator source or sequence contracts, but are
**not** claimed as product-ready by these single-segment gates.

Raw-video preprocessing is a separate creator product at SCAIL-Pose commit
`519c7f54cb972e7f92684213b7ef6c3e05a8f3b2`. Its animation and replacement
entrypoints require a Python vision environment and gated `facebook/sam3`
weights; pose-driven mode additionally requires NLF and DWPose/YOLOX weights.
Those dependencies are not part of the seven-stage Mojo inference runtime.
Until an installed preprocessor is admitted separately, Serenity requires the
reference mask and driving-mask video rather than pretending it can derive
them from raw media.

## Pinned provenance

- Creator repository commit:
  `5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5`
- Hugging Face model revision:
  `150cc0ca4e98e50e60b9295dacde39442fdccab2`
- Official transformer checkpoint size: `65,582,563,395` bytes
- Official transformer checkpoint SHA-256:
  `d6c73e94c57eb36e6351c800d1228e41ed7e45db1ccf410dd875bcfdd2945e7f`
- Official UMT5 SHA-256:
  `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d`

The model card declares the released model artifacts under MIT. The creator
source repository declares Apache-2.0; these are separate license records and
must not be collapsed into one claim. The original SAT/reference source is
pinned to branch `sat-scail2`, commit
`3dee8c4808057690393a189836c508b966ec282b`.

The creator files used as authority were `generate.py`, `wan/scail.py`,
`wan/modules/model_scail2.py`, the Wan VAE/UMT5/CLIP/attention modules,
`wan/utils/scail_utils.py`, `wan/utils/fm_solvers_unipc.py`, both SCAIL config
families, and the checkpoint/LoRA converters. The SAT-side architecture and
training evidence came from `dit_video_crossattn_sc_xc.py`,
`diffusion_video.py`, `data_video.py`, the rectified-flow loss/sigma modules,
and `Wan2.1-i2v-14Bsc-pose-xc-latent.yaml`. The old local Wan2.1 SCAIL workflow
is not a SCAIL-2 oracle.

At the pinned Hub revision, the published download inventory was 12 files and
about 82.5 GB: a 65.6 GB SAT transformer checkpoint, a 507.6 MB Wan2.1 VAE, a
2.5 GB visual OpenCLIP checkpoint, an 11.4 GB UMT5 encoder plus tokenizer
files, and two optional roughly 1.2 GB LoRAs (bias-aware DPO and replacement
relighting). The optional LoRAs are not part of the base-runtime admission.

The public SAT tree preserves training internals but does not publish a
supported training entry point, complete dataset contract, reproducible
optimizer recipe, or loss/gradient baseline. This intake therefore makes no
SCAIL-2 training-readiness claim.

`scripts/scail2_convert_checkpoint.py` verifies the transformer SHA before
opening it. It uses PyTorch's restricted `weights_only=True` loader with only
the measured NumPy RNG-state globals allowlisted; it never falls back to
unrestricted pickle loading. Canonical shards and the FP8 cache both require
the exact provenance tuple above and refuse unbound or mismatched-provenance
resumes. Every canonical and FP8 shard has a content checksum; resume also
validates the exact source-derived names and shapes (canonical) or the exact
SCAIL-2 name/shape/dtype/count schema (FP8). Digest comparison is independent
of the configured model-root path, so a valid cache can move between machines.

## Runtime artifacts

Use configurable roots; the Mojo CLIs take all artifact paths as arguments.
The measured local layout was:

```bash
export SCAIL_OFFICIAL="$HOME/.serenity/models/checkpoints/SCAIL-2"
export SCAIL_MOJO="$HOME/.serenity/models/checkpoints/SCAIL-2-Mojo"
export SERENITY_HOME="${SERENITY_HOME:-$HOME/.serenity}"
export SCAIL_OUTPUT="${SCAIL2_EVIDENCE_ROOT:-$SERENITY_HOME/runs/scail2-gates}"
export SCAIL_VAE="/configured/model/root/wan_2.1_vae.safetensors"
```

Canonical conversion:

```bash
python scripts/scail2_convert_checkpoint.py \
  "$SCAIL_OFFICIAL/model/1/fsdp2_rank_0000_checkpoint.pt" \
  "$SCAIL_MOJO/transformer"
```

FP8 stream preparation:

```bash
"${SCAIL_BIN:-/tmp/scail2-bin}/scail2_prepare_fp8_cache" \
  "$SCAIL_MOJO/transformer" \
  "$SCAIL_MOJO/transformer_fp8"
```

The cache contains 40 independently loadable blocks. Each block has 12
row-scaled E4M3 matrices plus 20 exact BF16 tensors; the 27 shared BF16 tensors
are published only after every block validates.

## Production run

The standard product build is `pixi run build-scail2`, which installs the seven
SCAIL-2 runners under ignored `output/bin/`. The following `/tmp` build is only
for isolated development:

```bash
pixi run build-cshim
pixi run bash -lc '
set -euo pipefail
mkdir -p /tmp/scail2-bin
for src in \
  serenitymojo/pipeline/scail2_stage_inputs.mojo \
  serenitymojo/pipeline/scail2_encode_prompt.mojo \
  serenitymojo/pipeline/scail2_encode_clip.mojo \
  serenitymojo/pipeline/scail2_encode_vae.mojo \
  serenitymojo/pipeline/scail2_prepare_fp8_cache.mojo \
  serenitymojo/pipeline/scail2_animation.mojo \
  serenitymojo/pipeline/scail2_decode.mojo
do
  name=$(basename "$src" .mojo)
  mojo build -I . -I vendor/mojo-libs \
    -Xlinker -lm -Xlinker -ldl \
    -Xlinker -L"$CONDA_PREFIX/targets/x86_64-linux/lib/stubs" \
    -Xlinker -lcuda \
    -Xlinker -L"$CONDA_PREFIX/lib" \
    -Xlinker -rpath-link -Xlinker "$CONDA_PREFIX/lib" \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    "$src" -o "/tmp/scail2-bin/$name"
done
'
export SCAIL_BIN=/tmp/scail2-bin
export LD_LIBRARY_PATH="$PWD/serenitymojo/ops/cshim/lib:$PWD/.pixi/envs/default/lib"
```

Conditioning stages:

```bash
"$SCAIL_BIN/scail2_stage_inputs" \
  REFERENCE_IMAGE REFERENCE_MASK \
  DRIVING_OR_POSE_VIDEO DRIVING_MASK_VIDEO \
  "$SCAIL_OUTPUT/animation_001_stage.safetensors" 512 896 65

python scripts/scail2_convert_clip.py \
  "$SCAIL_OFFICIAL/models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth" \
  "$SCAIL_MOJO/clip_visual/model.safetensors"

"$SCAIL_BIN/scail2_encode_prompt" \
  UMT5_MODEL_DIR TOKENIZER_JSON POSITIVE_PROMPT NEGATIVE_PROMPT \
  "$SCAIL_OUTPUT/animation_001_text_context.safetensors"

"$SCAIL_BIN/scail2_encode_clip" \
  "$SCAIL_MOJO/clip_visual/model.safetensors" \
  "$SCAIL_OUTPUT/animation_001_stage.safetensors" \
  "$SCAIL_OUTPUT/animation_001_clip_context.safetensors"

"$SCAIL_BIN/scail2_encode_vae" ref \
  "$SCAIL_OUTPUT/animation_001_stage.safetensors" "$SCAIL_VAE" \
  "$SCAIL_OUTPUT/animation_001_reference_latent.safetensors"

"$SCAIL_BIN/scail2_encode_vae" pose \
  "$SCAIL_OUTPUT/animation_001_stage.safetensors" "$SCAIL_VAE" \
  "$SCAIL_OUTPUT/animation_001_pose_latent.safetensors"
```

Denoise and decode:

```bash
"$SCAIL_BIN/scail2_animation" \
  "$SCAIL_OUTPUT/animation_001_stage.safetensors" \
  "$SCAIL_OUTPUT/animation_001_reference_latent.safetensors" \
  "$SCAIL_OUTPUT/animation_001_pose_latent.safetensors" \
  "$SCAIL_OUTPUT/animation_001_text_context.safetensors" \
  "$SCAIL_OUTPUT/animation_001_clip_context.safetensors" \
  "$SCAIL_MOJO/transformer_fp8" \
  "$SCAIL_OUTPUT/animation_001_latent.safetensors" 40 42

"$SCAIL_BIN/scail2_decode" \
  "$SCAIL_OUTPUT/animation_001_latent.safetensors" \
  "$SCAIL_VAE" \
  "$SCAIL_OUTPUT/animation_001_result" \
  OPTIONAL_REFERENCE_OR_DRIVING_VIDEO
```

`scail2_animation` enables synchronous device allocation before constructing
its `DeviceContext`. This is required for the 39,872-token sequence on a 16 GB
RTX 5080: the default async pool retained cumulative temporaries and OOMed,
while the automatic bounded path completed at a measured peak near 12.86 GB.
Users do not set an environment variable or babysit the run.

`scail2_decode` probes the optional source itself. If it contains audio, the
output receives AAC audio from that reference; if it is silent, the output is
video-only. SCAIL-2 does not synthesize audio. The official README explicitly
states that audio comes from reference videos.

## Serenity Studio product path

Build the seven canonical product entrypoints with:

```bash
pixi run build-scail2
```

After the machine-local product report passes, `SCAIL-2-Mojo` appears in the
Generate model picker. The Generate tab uploads the reference image, reference
mask, driving video, and driving-mask video. One Generate action sends:

```json
{
  "model": "scail2",
  "prompt": "the girl is dancing",
  "negative_prompt": "",
  "reference_image": "/absolute/path/reference.png",
  "reference_mask": "/absolute/path/reference_mask.png",
  "driving_video": "/absolute/path/driving.mp4",
  "driving_mask_video": "/absolute/path/driving_mask.mp4",
  "width": 896,
  "height": 512,
  "frames": 65,
  "fps": 16,
  "steps": 40,
  "guidance": 5.0,
  "seed": 42,
  "quant": "fp8"
}
```

The `/v1/video` SCAIL-2 arm automatically runs input staging, UMT5, CLIP,
reference VAE encode, pose VAE encode, FP8 denoise, fresh-process VAE decode,
MP4 mux, and driving-video audio preservation. Logs, frames, intermediate
tensors, timings, exit codes, peak VRAM, and provenance sidecars stay in
`$SERENITY_HOME/runs/scail2/video-NNNN/`. A successful run publishes exactly
one user-output file: `OUTPUT_ROOT/video-NNNN/scail2_animation.mp4`. A failed
or rejected run publishes no user-output files. No assistant-authored cache or
manual conditioning step is part of the product run.

Regenerate admission only from verified artifacts:

```bash
python3 scripts/check_scail2_product_gate.py --visual-inspection-passed
```

## Measured gates

- Creator preprocessing parity: exact for representative pose, driving, and
  mask tensors; staged artifact SHA-256
  `2fab668c036bbabb76c2136c3d483766f7cf614115b4da0ea1924bea27fd435a`.
- UniPC schedule and integer timesteps: max absolute error `0`; six-step
  trajectory max absolute error `1.1920929e-7`.
- RoPE adversarial minimum cosine: `0.9999989`; fixture checksums pass.
- Official Wan VAE reuse: `194/194` tensors byte-identical.
- Official UMT5 reuse: `242/242` tensors byte-identical,
  `11,361,820,672` tensor bytes.
- Real block 0, Mojo FP8 vs Python FP8: cosine `0.9999971794`.
- Real block 0, Mojo FP8 vs canonical: cosine `0.9999314142`.
- Shared patch/mask embedding and head video slice: exact.
- Shared image MLP cosine: `0.9999863742`.
- Automatic bounded one-step result is byte-identical to the explicit
  synchronous-allocation gate.
- Current-binary base-animation one-step SHA-256:
  `e959bd00f99160b4db03656870558390de33e0bf4e6ce57761dcf20045436651`.
- Current-binary base-animation 40-step latent SHA-256:
  `7281f726de405cc5ef7931db45a9541e86a782537911c9cc9c735fc14286959b`.
- Current-binary base-animation decoded MP4 SHA-256:
  `af292cfe7c8c18a66e3348f57797c21350257c35bf0528533b2ff6803a20234c`.
- Final video: H.264 `896x512`, exactly `65` frames, `4.062500` seconds.
  Start/middle/end frames were visually inspected and are coherent, varied,
  and free of flat/half-frame corruption.
- Current-binary three-reference 40-step latent SHA-256:
  `e7d64b2efb0ca7e00b5854a309e014ddbccaa2c6755f579d521601b5f8839601`.
  The run completed in `1878.54 s` at a measured `13,920 MiB` peak VRAM.
  Its decoded H.264 MP4 SHA-256 is
  `d32ebe94500b980495280315681300bed1d300a3a337e0f7e346980b956d3a74`;
  it is `896x512`, 65 frames at 16 fps, and its start/middle/end frames passed
  visual inspection.
- The current-binary replacement one-step latent is byte-identical to the
  accepted pre-compact-RoPE result, SHA-256
  `5b847f3bd11013905f8f244627f1dab236901db1f1f979bd14089b7a90878941`.
  The accepted 40-step replacement latent SHA-256 is
  `1e5db48720155b83d46ba55505c24648837b35ae3ae5c3bf9616da82cd420dbd`;
  its decoded audio-bearing MP4 SHA-256 is
  `9ffa5e43393539ad1c2d94ff41a44c8a2168b1577e0ddaa3c7cf6fcfc82f430c`.
- Automatic audio gate: the audio-bearing replacement reference produced H.264
  plus AAC 48 kHz mono; the official silent samples produced H.264 only.
- Repository contract and Rust workspace tests pass.

### RTX 5080 production performance (`nsys`, 2026-07-19)

Profile the real production geometry and both CFG branches; do not substitute a
small-token microbenchmark:

```bash
MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output="$SCAIL_OUTPUT/nsys/PROFILE_NAME" \
  "$SCAIL_BIN/scail2_animation" \
  STAGE REFERENCE_LATENT POSE_LATENT TEXT_CONTEXT CLIP_CONTEXT \
  FP8_CACHE OUTPUT_LATENT 1 42

nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
  "$SCAIL_OUTPUT/nsys/PROFILE_NAME.nsys-rep"
```

Measured on an RTX 5080 with driver `595.71.05`:

- Initial native-FP8 one-step trace: block compute `40.0544 s`.
- Skip the no-op activation pad copy: `39.3423 s`.
- Fuse per-row activation absmax reduction and FP8 encode: `37.6794 s`.
- Use Mojo's native `DType.float8_e4m3fn` cast instead of software E4M3
  encoding: `36.5631 s` under `nsys`, and `36.4095 s` in the direct AOT run.
- Every accepted one-step optimization produced the same latent bytes:
  SHA-256 `e959bd00f99160b4db03656870558390de33e0bf4e6ce57761dcf20045436651`.
- The real block gate remains unchanged: native FP8 versus canonical cosine
  `0.9998444131750903`.
- Runtime reaches denoise in about `9 s`; it previously spent roughly
  `85–100 s` re-hashing the complete FP8 cache twice. Cache creation/resume
  still performs full payload SHA-256 verification. Generation checks pinned
  provenance, checksum-sidecar syntax, and every safetensors name, dtype, and
  shape without rescanning all payload bytes.
- The remaining dominant kernel is official global self-attention: 80 cuDNN
  BF16 flash-attention calls total `23.5414 s`, or `67.5%` of GPU kernel time.
  CFG batch-2, the cuDNN open-source engine, all-plan autotuning, and FP8
  attention were measured and rejected (no speedup, unsupported graph, or
  unacceptable trajectory drift respectively). Do not re-enable them without
  a new parity result.
- The current compact-RoPE 40-step AOT run completed in `1528.62 s`
  (`25m 28.62s`)
  with `1.42 GB` peak process RSS. The prior native-FP8 run took `29.10 min`.
  The optimized latent is byte-identical to that accepted run, SHA-256
  `7281f726de405cc5ef7931db45a9541e86a782537911c9cc9c735fc14286959b`.
- Fresh-process decode completed in `29.69 s`. Its H.264 MP4 is `896x512`,
  65 frames at 16 fps, 4.0625 seconds, SHA-256
  `af292cfe7c8c18a66e3348f57797c21350257c35bf0528533b2ff6803a20234c`.
  The selected official reference is silent, so the verified result correctly
  contains no audio stream. Start/middle/end frames were visually inspected.

The accepted trace is
`$SCAIL_OUTPUT/nsys/scail2_nativecast_quant_1step.nsys-rep`; its exported
kernel table is `scail2_nativecast_quant_1step.stats.csv` in the same directory.

## Source map

- Checkpoint conversion: `scripts/scail2_convert_checkpoint.py`
- Production input staging: `serenitymojo/pipeline/scail2_stage_inputs.mojo`
- Pinned creator preprocessing oracle: `scripts/scail2_stage_inputs.py`
- UMT5/VAE equivalence gates: `scripts/scail2_verify_{umt5,vae}_equivalence.py`
- Model/runtime: `serenitymojo/models/scail2/`
- Conditioning: `serenitymojo/pipeline/scail2_encode_{prompt,clip,vae}.mojo`
- Cache: `serenitymojo/pipeline/scail2_prepare_fp8_cache.mojo`
- Denoise: `serenitymojo/pipeline/scail2_animation.mojo`
- Decode/audio: `serenitymojo/pipeline/scail2_decode.mojo`
- Scheduler: `serenitymojo/sampling/scail2_unipc.mojo`
- Numeric gates: `serenitymojo/models/scail2/parity/` and
  `serenitymojo/sampling/parity/scail2_unipc_parity.mojo`
