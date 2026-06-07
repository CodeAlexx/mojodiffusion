---
name: mojo-model-runtime-port
description: Build a shared Mojo-native model runtime before or while porting a model for training. Use when adding image/video/model inference, sampler CLIs, a Mojo-only inference UI path, or preparing reusable loader/block/conditioning/VAE/scheduler/offload code for the OneTrainer-to-Mojo trainer.
---

# Mojo Model Runtime Port

## Overview

Use this when a model needs both production inference and future training
support. The goal is one Mojo runtime surface that can generate real images or
video now, then feed the trainer without duplicating loaders, blocks,
conditioning, samplers, dtype policy, or offload behavior.

This is a good idea only if inference is treated as the first product gate, not
as a shortcut. Inference code may skip training-only activation saves, but it
must share model math and tensor contracts with the trainer.

## Use With

- `ot-mojo-model-intake` for OneTrainer reference inventory.
- `ot-mojo-sampler-port` for sampler-specific parity.
- `ot-mojo-training-port` when turning the same runtime into a train loop.
- `ot-mojo-parity-gates` before any readiness claim.

For non-OneTrainer inference work, still apply the Mojo-native runtime,
artifact, dtype, speed, and UI rules. Do not use Python as the product path.

## Build Order

1. Identify the model contract:
   - checkpoint/config format, model class, tensor keys, storage dtypes
   - image/video dimensions, latent packing, frame count, patch geometry
   - tokenizer/text encoders, conditioning templates, negative/CFG behavior
   - scheduler/timestep/sigma rules, VAE or video decoder, postprocess/muxing
   - LoRA/full-finetune target names and save/resume needs if training applies
2. Build shared runtime modules:
   - config reader and fail-loud preflight before `DeviceContext()`
   - checkpoint loader preserving BF16/F16/FP8 storage dtype at boundaries
   - block/stack forward math, attention/norm/MLP kernels, positional encodings
   - conditioning, scheduler, VAE/decode, output writer, offload policy
   - model artifact manifest with exact source file/key names
3. Build inference first:
   - a Mojo CLI or runner that loads the real model and emits real artifacts
   - request/result JSON suitable for a Mojo-only inference UI
   - timings for text/conditioning, denoise, VAE/decode, postprocess/mux
   - peak VRAM and output dimensions/duration/frame-count verification
4. Add trainer bridge points:
   - train/recompute/backward entry points using the same block math
   - activation checkpoint/offload plan that does not create F32 boundaries
   - cache tensor schemas compatible with the runtime conditioning/latent rules
   - LoRA/full-finetune update hooks, save/resume, and validation sampler calls
5. Add parity and readiness gates:
   - dtype guard, no accidental F32 storage boundary
   - inference artifact guard and dimension/duration guard
   - sampler trajectory/image/speed/VRAM gates when a reference exists
   - loss, grad, optimizer update, save/resume gates for training

## Shared Surface Rules

- Prefer one source of truth for model config, weight names, tensor dtypes, block
  math, scheduler, VAE/decode, and conditioning.
- Inference can expose `predict` or `generate` paths that avoid training tape.
  Training can expose recompute/backward variants, but both must call the same
  kernels or a clearly paired implementation.
- LoRA application must use the same target-name mapping for inference, sample
  validation, training, save, and resume.
- UI-facing runners must use machine-readable request/result manifests and
  progress events rather than parsing console logs.
- Large-model inference and validation sampling may run process-separated to
  release trainer VRAM before sampling.

## Dtype And Memory

- Preserve checkpoint, activation, latent, conditioning, VAE, LoRA, and cache
  storage dtypes at tensor boundaries.
- F32 is allowed for accumulators, reductions, attention scores, norm math,
  scheduler scalars, host stats, and oracle/debug artifacts only.
- Any intentional product F32 boundary needs a nearby comment naming the
  reference reason.
- Offload/checkpointing must be a first-class runtime policy, not a trainer-only
  afterthought. It should lower VRAM for inference validation and training.

## Inference UI Contract

A model is UI-ready only when the Mojo runner has:

- preflight errors before CUDA for unsupported dtype/model/settings
- request JSON with model path, prompt, seed, dimensions, steps, guidance, dtype,
  output paths, LoRA paths/scales, and offload policy
- result JSON with artifact paths, timings, peak VRAM if measured, dimensions,
  frame count/duration for video, and explicit readiness labels
- cancellation/timeout behavior or a documented blocker
- no Python dependency in the product inference path

## Do Not Claim

- A tiny artifact is not production inference.
- A generated image/video is not sampler parity.
- A sampler smoke is not train readiness.
- A shared file name is not shared behavior; prove the trainer consumes the same
  runtime contracts.
- A fast path that upcasts storage tensors to F32 is a bug unless the reference
  requires it.
