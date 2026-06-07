---
name: ot-mojo-production-use
description: Standard workflow for deciding whether a OneTrainer-to-Mojo model can be used for real training or sampling. Use for production readiness, presets, running a training job, or answering "is this ready?".
---

# ot-mojo-production-use

Use when someone asks if a model is ready, wants to train, or wants a preset.

## Production Ready Means

A model is production ready only when the product path, not just a parity helper,
has evidence for:

- correct OneTrainer config/preset import
- cache/preflight before CUDA
- loss and gradient parity
- optimizer update and LR parity
- save/resume parity
- sampler output and speed/VRAM parity
- dtype-boundary compliance
- fail-loud unsupported options
- documented limits

## Run Decision

Before running real training:

1. Check the model status doc and readiness guard.
2. Confirm checkpoint/cache/sample paths exist.
3. Confirm dtype policy and offload/checkpoint policy.
4. Confirm the preset is LoRA or full finetune and whether that method is
   actually supported.
5. Run a short bounded smoke if the user needs a runtime sanity check.
6. Do not run a long job if strict blockers remain unless the user explicitly
   accepts a smoke/experimental run.

## Required Metrics

Training:

- loss per step
- grad norm
- forward/backward/optimizer seconds
- total seconds/step
- peak VRAM
- saved tensor dtype counts

Sampling:

- text encode seconds
- denoise seconds and seconds/step
- VAE decode seconds
- total wall time
- peak VRAM
- image dimensions and format

## Labels To Use

- `not ready`: missing core parity or product path.
- `smoke runnable`: can run bounded tests, outputs not trusted.
- `experimental`: can train/sample but missing parity or speed gates.
- `production ready`: all required gates pass with current artifacts.

Never call a model production ready based on docs, templates, or old memory.
