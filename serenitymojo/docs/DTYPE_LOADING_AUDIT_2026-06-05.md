# Checkpoint Dtype Loading Audit - 2026-06-05

Status: handoff for the OneTrainer-parity Mojo trainer port.

## Trainer Direction

The OneTrainer-parity Mojo port is the supported trainer path. Older
mojodiffusion trainer experiments are reference or compatibility surfaces unless
they use the same `serenitymojo` model loaders, offload runtime, sampler, save
and resume, progress display, and dtype contract.

Inference and training must share the same model/runtime primitives. A fix in a
loader or core op is expected to benefit both train and sample paths, not create
a train-only fork.

## Safetensors Result

The core safetensors reader is not double-loading checkpoint payloads:

- `serenitymojo/io/safetensors.mojo` reads the header eagerly, mmaps the data
  segment, and exposes `tensor_bytes` spans.
- `serenitymojo/io/sharded.mojo` opens each unique shard once and maps tensor
  names to shard handles.
- `Tensor.from_view` preserves the view dtype and copies raw bytes to device.

The F32 expansion bug was in model loaders and offload adapters that converted
checkpoint tensors through host-F32 lists or F32 device tensors.

## Fixed In This Pass

- Core ops: `rms_norm` and `rms_norm_backward` now support F32 activations with
  BF16/F16 norm weights while returning activation grads in activation dtype and
  norm grads in norm-weight dtype. RoPE forward/backward supports F32 tables with
  BF16/F16 activation storage.
- Flux, Chroma, Klein, Qwen-Image, ZImage, L2P, Anima, and Ernie loaders now
  preserve checkpoint dtype for block weights, biases, norm scales, stack-base
  projections, and final projections touched by the official trainer path.
- Chroma row-stacking now happens on device with `concat`, so separated q/k/v
  checkpoint tensors are fused without host-F32 roundtrips.
- Flux, Klein, and Qwen stack/block structs gained direct `ArcPointer[Tensor]`
  constructors so loaders can pass device tensors without rebuilding host lists.
- ZImage final projection and Ernie patch/final stack-base tensors now preserve
  checkpoint dtype instead of using F32 helper loaders.
- LTX2 production offload remains BF16-preserving. Its legacy host-list extractor
  is not the supported production trainer path.

Allowed F32 boundaries remain: GEMM accumulation, reductions, schedules,
timestep/noise scalars, RoPE tables, generated modulation vectors, host AdamW
master state, host grad readback, debug/parity oracles, and file formats that
require F32.

## Remaining Non-Official F32 Surfaces

The wide scan still finds host-F32 or F32-native loaders in older or unfinished
families:

- `serenitymojo/models/sd35/weights.mojo`
- `serenitymojo/models/sd35/sd35_stack_lora.mojo`
- `serenitymojo/models/sdxl/weights.mojo`
- `serenitymojo/models/sdxl/real_weights.mojo`
- `serenitymojo/models/wan22/wan22_stack_lora.mojo`
- legacy `load_ltx2_block_weights_from_block`

Do not treat those as official production trainer paths until they are ported to
the same checkpoint-dtype storage contract.

## Verification

Target accelerator used here: `sm_86` for the local RTX 3090 Ti.

Passed:

- `python3 scripts/check_ltx2_dtype_contract.py`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm serenitymojo/models/flux/parity/load_flux_weights_smoke.mojo -o /tmp/load_flux_weights_smoke_dtype`
- `/tmp/load_flux_weights_smoke_dtype` outside sandbox: Flux real checkpoint shape smoke PASS.
- `/tmp/norm_bwd_parity_dtype_sm86`: all norm backward gates PASS at cosine >= 0.999.
- `/tmp/qwenimage_real_smoke_dtype`: Qwen-Image real-dim forward/backward finite, LoRA grads nonzero, AdamW applied.
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_chroma_real.mojo -o /tmp/train_chroma_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib -Xlinker -lsqlite3 serenitymojo/training/train_klein_real.mojo -o /tmp/train_klein_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_zimage_real.mojo -o /tmp/train_zimage_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_l2p_real.mojo -o /tmp/train_l2p_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_anima_real.mojo -o /tmp/train_anima_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_ernie_real.mojo -o /tmp/train_ernie_real_dtype`

The Flux smoke had to run outside the sandbox because CUDA `DeviceContext`
initialization failed under sandbox NVML.
