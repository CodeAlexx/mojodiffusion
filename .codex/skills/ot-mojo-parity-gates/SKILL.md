---
name: ot-mojo-parity-gates
description: Standard verification workflow for OneTrainer-to-Mojo parity. Use when testing or accepting any model training/sampler port. Requires current-state evidence for loss, gradients, optimizer update, save/resume, sampler output, speed, VRAM, and dtype boundaries.
---

# ot-mojo-parity-gates

Use this before saying a model is ready.

## Evidence Levels

- `compile`: code builds only.
- `smoke`: bounded runtime/wiring; not parity.
- `artifact consumer`: opens OneTrainer artifacts and validates names/shapes.
- `loss bridge`: recomputes dumped loss from dumped tensors.
- `state-init`: zero-lr optimizer state init; not nonzero update parity.
- `update-bearing`: OneTrainer dump has nonzero update deltas.
- `Mojo replay`: Mojo runs forward/backward/optimizer on matching inputs.
- `production parity`: product path matches OT and records speed/VRAM.

Say the exact level. Do not upgrade labels.

## Training Acceptance

For a model training vertical, require:

1. OneTrainer 100-step baseline with loss, grad norm, speed, VRAM.
2. OneTrainer one-step dump with step tensors, adapter/full-weight tensors, meta.
3. Mojo loss replay on byte-identical inputs.
4. Mojo backward grad comparison for every trainable tensor or accepted subset
   with explicit missing targets.
5. AdamW/LR/optimizer-state update parity.
6. Save key/dtype parity.
7. Resume continuation parity.
8. Runtime dtype-boundary guard.
9. Product loop run with real artifacts, speed, and VRAM.

## Sampler Acceptance

Require paired OT/Mojo:

- prompt/seed/resolution/steps/guidance/dtype
- conditioning/noise/scheduler trace
- denoise trajectory/final latent/VAE/image
- speed and peak VRAM

PNG dimension match is not pixel parity. A generated image is not speed parity.

## Expected Failures

Strict gates should fail when evidence is missing. Keep those failures:

- missing paired sampler manifest
- missing positive-lr dump
- missing full-finetune model-specific hooks
- missing positive VRAM in speed metadata
- missing dtype-boundary cleanup

If a gate passes by weakening requirements, revert or tighten it.

## Reporting

Final reports must include:

- commands run and exit status
- exact loss/speed/VRAM numbers
- artifact paths
- accepted evidence level
- remaining blockers
- any reason F32 appears at a storage boundary
