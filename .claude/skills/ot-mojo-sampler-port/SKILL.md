---
name: ot-mojo-sampler-port
description: Standard workflow for porting and validating OneTrainer sampler components for each model: text conditioning, initial noise, scheduler, denoise trajectory, VAE encode/decode, image artifact, speed, VRAM, and train-loop sample cadence.
---

# ot-mojo-sampler-port

Every model needs its own sampler. Do not reuse another model's sampler unless
OneTrainer does.

## Sampler Surfaces

Inventory and port these from OneTrainer:

- sampler class and scheduler
- prompt/template/tokenizer/text-encoder path
- negative/CFG behavior
- initial noise generation and shape/packing
- timestep/sigma schedule
- latent patch/pack/unpack rules
- VAE scale/shift and decode/postprocess
- image dimensions, quantization constraints, output format
- train-loop sample cadence and save-before-sample behavior

## Process Separation

For large models, in-process validation sampling can OOM. Use process-separated
request/result manifests when needed:

- request schema: queued sample inputs, LoRA/state/sample paths, output PNG,
  result manifest, no parity claim
- supervisor: runs after trainer exits or memory is released
- result schema: Mojo-side timings and artifact paths, still no parity claim
- external support tool may poll `nvidia-smi` for peak VRAM

## Parity Bundle

Accepted sampler parity needs paired OneTrainer and Mojo artifacts:

- prompt, negative prompt, seed, width, height, steps, guidance, dtype
- text conditioning tensors/masks/pooled outputs as applicable
- raw noise and post-pack/post-patch noise
- scheduler timesteps/sigmas and per-step trace
- predicted velocity/noise trajectory where practical
- final latent and VAE pre-decode tensor
- final PNG or image tensor
- denoise seconds/step, VAE decode seconds, total wall time, peak VRAM
- numeric comparisons with tolerances

## Labels

- `smoke`: real image or tiny output proves wiring only.
- `artifact evidence`: headers/paths/metadata prove files exist and are shaped.
- `sampler parity`: paired OT/Mojo trajectory/image evidence passes.
- `speed parity`: paired OT/Mojo timing plus positive VRAM evidence passes.

Never collapse these labels.

## Required Commands

Each model should have:

- no-CUDA sampler manifest guard
- strict sampler parity or blocker guard
- strict speed/VRAM guard
- train-loop sample-cadence guard
- PNG dimension/format check for generated outputs
