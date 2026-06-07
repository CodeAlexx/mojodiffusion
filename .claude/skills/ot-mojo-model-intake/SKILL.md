---
name: ot-mojo-model-intake
description: Standard intake/audit workflow before porting a new OneTrainer model into mojodiffusion. Use when adding or resuming a model vertical and before coding loaders, training, samplers, presets, or parity gates. Enforces OneTrainer-only reference, model identity, artifact naming, dtype, speed/loss baseline, and missing-surface inventory.
---

# ot-mojo-model-intake

Use this first for every new model vertical. The output is an evidence-backed
intake note and blocker list, not implementation.

## Reference Scope

- Primary reference: `/home/alex/OneTrainer`.
- Anima exception: `/home/alex/OneTrainer-anima-ref`.
- Do not use random web/HF/diffusers docs as parity authority unless the user
  explicitly changes scope. Checkpoint headers and local model files can be used
  to verify shapes/dtypes.
- Record exact OneTrainer source file names, classes, methods, and preset/config
  paths. Mojo files and manifests must keep those names visible enough that a
  later agent can map OT -> Mojo without guessing.

## Intake Checklist

1. Identify the model target:
   - OneTrainer `ModelType`
   - training methods registered by OneTrainer
   - target variant/class, e.g. Flux2/Klein, Flux.1 dev, SD3.5, Chroma, Z-Image
   - local checkpoint paths, VAE, tokenizers/text encoders, scheduler, cache
2. Inventory OneTrainer files:
   - `modules/model/*`
   - `modules/modelLoader/*`
   - `modules/modelSetup/*`
   - `modules/modelSampler/*`
   - `modules/dataLoader/*`
   - saver/converter files for LoRA/full finetune
3. Capture baseline facts:
   - 100-step OneTrainer baseline when available: loss, grad norm, step time,
     peak VRAM, dtype, optimizer, LR schedule, batch/resolution/cache state
   - one-step dump availability: step tensors, adapter tensors, meta JSON
   - sampler baseline: prompt, seed, resolution, steps, guidance, dtype,
     denoise time, VAE time, output artifact, peak VRAM
4. Verify dtype contract before coding:
   - checkpoint/cache/activation/LoRA storage dtype
   - FP8/F16/BF16 special cases
   - allowed F32 compute internals only
   - no host F32 tensor boundary unless explicitly justified
5. Inventory missing Mojo surfaces:
   - config/preset
   - weights loader
   - stack/block forward and backward
   - LoRA keys/save/resume
   - full finetune inventory/save/load/resume if OneTrainer registers it
   - sampler components: scheduler, text conditioning, noise, VAE, image output
   - parity scripts/manifests

## Required Intake Output

Create or update a concise repo note or status section with:

- `reference_files`: exact OneTrainer files and classes inspected.
- `baseline_numbers`: loss/grad/speed/VRAM, or explicit "missing".
- `target_artifacts`: paths expected under `/home/alex/onetrainer-mojo/parity`.
- `mojo_surfaces`: present/missing per surface.
- `blockers`: separated into train, sampler, dtype, speed/VRAM, resume, full FT.
- `next_gate`: the next command that would prove progress.

## Do Not Claim

- A compile is not parity.
- A smoke image is not sampler parity.
- A zero-lr update dump is not nonzero AdamW parity.
- Generator-only speed without matched OT fields and positive VRAM is not speed
  parity.
- Full finetune scaffolding is not full finetune support.
