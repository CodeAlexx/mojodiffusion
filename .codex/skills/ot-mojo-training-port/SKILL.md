---
name: ot-mojo-training-port
description: Standard workflow for implementing a OneTrainer-to-Mojo training vertical: config, cache, loaders, forward/backward, LoRA, full finetune fail-loud or support, optimizer, save/resume, and product train loop. Use after ot-mojo-model-intake.
---

# ot-mojo-training-port

Use after `ot-mojo-model-intake`. Keep runtime Mojo-native; Python is allowed
for oracle dumps, static guards, and tooling only.

## Build Order

1. Config and preset:
   - parse OneTrainer fields through `train_config_reader.mojo`
   - preserve model type, training method, dtype, optimizer, LR, cache/output,
     sample/save/backup, validation, EMA, masked/prior flags, checkpoint/offload
   - unsupported combinations must fail before `DeviceContext()`
2. Cache/data path:
   - mirror OneTrainer cache fields and tensor shapes
   - preserve cached storage dtype at boundaries
   - raw `only_cache` must either work or fail loud before CUDA
3. Weights/stack:
   - load checkpoint tensors in storage dtype
   - use F32 only inside compute kernels/reductions
   - map every OneTrainer key to a Mojo struct field or a fail-loud blocker
4. Backward:
   - implement block backward before stack backward
   - recompute or offload activations intentionally; never clone large F32
     activation boundaries for convenience
   - add op backward arms only when existing ops cannot cover the OT math
5. LoRA:
   - map OneTrainer target names exactly
   - save raw OT-compatible keys
   - preserve BF16/F16 LoRA tensor storage
   - resume LoRA and AdamW state with deterministic key/order checks
6. Full finetune:
   - if OneTrainer registers `TrainingMethod.FINE_TUNE`, either implement real
     full-weight update/save/load/resume or add a fail-loud preflight
   - scaffolding alone must keep `full_finetune_ready=false`
7. Product loop:
   - bind through shared OneTrainer loop policy
   - apply save/sample cadence from sample config
   - support resume or fail loud
   - print loss, grad norm, step time, phase timing, and artifact paths

## Required Evidence

- OneTrainer baseline and one-step artifacts are named in the local status doc.
- Mojo product loop consumes equivalent config/cache/prompt/seed/model artifacts.
- Loss replay, backward grads, AdamW update, optimizer state, save/resume each
  have gates or explicit blockers.
- Speed is measured as steady-state seconds/step with peak VRAM, not guessed.

## Dtype Contract

- BF16/F16/FP8 checkpoint and cache tensors must not be stored as F32 at model
  boundaries.
- F32 is allowed for accumulators, reductions, schedules/sigmas, AdamW moments,
  debug/oracle dumps, and scalar host statistics.
- If an op computes in F32, return the input/storage dtype unless OneTrainer
  requires otherwise.

## Agent Rules

When using agents, split by disjoint file ownership:

- builder: bounded implementation slice
- verifier: parity/static checks and expected failures
- skeptic: dtype, naming, and false-claim audit

The lead reruns every command before accepting the result.
