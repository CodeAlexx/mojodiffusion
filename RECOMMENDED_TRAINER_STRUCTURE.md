# Recommended modular trainer structure — serenitymojo

> Synthesized from 3 analyses: OneTrainer modular seams, EDv2 duplication inventory,
> Mojo current-state + language constraints. Goal: OneTrainer-style modularity — a
> shared training PIPELINE written once, with MODEL code as the only per-model surface.
> Avoids EDv2's flaw (60–70% of each of 17 trainers is duplicated boilerplate).
>
> STATUS 2026-05-30: Stage 1 IMPLEMENTED + compile-verified (RC=0, both entry points).
> IMPORTANT PLACEMENT CORRECTION discovered during implementation: this repo already
> has a `pipeline/` dir meaning INFERENCE pipelines+smokes (not a package), and
> `training/` is already documented as the model-agnostic training library ("Inference
> path never imports this"). So the shared pipeline lives in **`training/`**, NOT a new
> `pipeline/` dir (which would collide with inference). The split is `training/` (shared)
> vs `models/<model>/` (per-model). The agnostic primitives (optim/schedule/loop/
> dit_block/checkpoint) were deliberately LEFT in `training/` (already their correct
> home) — NOT relocated — so the proven cos≥0.999 gate suite was untouched (zero
> regression risk).

## The principle

**The loop never reads a tensor's meaning; the model never iterates epochs.**
Everything model-specific funnels through a narrow seam. From the EDv2 evidence the
genuinely per-model surface is only 4 things: **(1) the block forward/backward,
(2) the σ-map (timestep→sigma, ~2 lines), (3) the LoRA target-module map,
(4) the weight-key layout.** Everything else is shared.

## Directory layout (pipeline vs models — the separation you asked for)

```
serenitymojo/
  pipeline/                  # MODEL-AGNOSTIC — written ONCE, no model knowledge
    train_config.mojo        # TrainConfig (runtime scalars: lr, shift, rank, alpha, eps,
                             #   steps, warmup, ...) + CommonArgs  → kills EDv2 offender #1
    train_step.mojo          # the shared step recipe, comptime-shaped over [B,S,H,Dh]:
                             #   flow_target → lora_fwd → block_fwd → mse → block_bwd →
                             #   lora_bwd → optimizer_step.  (the 85% dup, lifted once)
    driver.mojo              # the loop skeleton: for step in start..steps { loss accum,
                             #   %save / %sample gate }  → kills EDv2 offender #3a
    optimizer_step.mojo      # clip → lr_schedule → step → zero_grad → ema  (already
                             #   optim.mojo)  → kills EDv2 offender #2
    flow_target.mojo         # build_flow_target(latent,noise,sigma,sign) (already
                             #   schedule.mojo); σ-map is INJECTED per model
    lora.mojo                # LoRA adapter math (already model-agnostic)
    loop_state.mojo          # TrainState: F32 masters + AdamW m/v + accum (already loop.mojo)
    checkpoint.mojo          # save/resume byte-exact (already checkpoint*.mojo)
    logger.mojo              # board/log_step(loss, grad_norm, lr, sps)
  models/                    # PER-MODEL — category (c) content ONLY
    klein/
      config.mojo            # klein_4b()/klein_9b() dims + recipe defaults (lr/shift/alpha)
      block.mojo             # single-stream (=proven dit_block) AND double-stream fwd/bwd
                             #   ← double-stream is the ONE genuinely-new compute unit
      sigma_map.mojo         # (floor(t)+1)/1000
      lora_targets.mojo      # split-qkv map + PEFT save-key convention (move from lora.mojo)
      weights.mojo           # safetensors → BlockWeights (fused-qkv → wq/wk/wv split)
    zimage/
      config.mojo            # zimage() dims (3840/30/128, 30 layers)
      block.mojo             # single-stream modulated block (= proven dit_block directly)
      sigma_map.mojo         # (idx+1)/1000
      lora_targets.mojo  weights.mojo
  ops/  autograd.mojo  tensor.mojo  io/   # unchanged primitives
```

## The seam, shaped for Mojo (not OOP)

Mojo has no inheritance and attention dims must be comptime, so the OneTrainer
`predict()`-on-a-base-class pattern is translated, not copied:

- **`struct StepOutput { predicted, target, timestep }`** — a TYPED seam (not a
  stringly-typed dict). The shared `train_step` consumes only this.
- **A per-model `struct` conforming to `trait ModelSpec`** exposing the 4-item surface
  (block fwd/bwd, sigma_map, lora_target_map, weight_loader). The step calls these
  methods DIRECTLY (no stored closure — Mojo can't store heterogeneous captured
  closures; "grads-as-input, not callbacks").
- **Dispatch = comptime monomorphization.** `train_step` is `[B,S,H,Dh]`-parameterized
  because the SDPA kernel is comptime-shaped. Top-level:
  `match config.model_type: Klein9B → train_step[1, S, 32, 128](klein_spec, cfg); ...`
  Each model-shape is one monomorphization, selected once at main(). No loop branching.
- **Block-kind fork (single vs double stream) is a comptime branch / two trait
  conformances**, NOT a subclass.

## What this buys vs EDv2

| EDv2 offender | Lines duped | Becomes |
|---|---|---|
| `Args` struct (~47/70 fields ×17) | ~4,000+ | one `CommonArgs` in pipeline/train_config + tiny per-model arg |
| optimizer/clip/lr/ema epilogue ×14 | ~14×30 | `pipeline/optimizer_step.mojo` |
| `main()` loop + flow-target ×17 | ~17×60 | `pipeline/driver.mojo` + `flow_target.mojo` |
| per-model train_*.mojo (Mojo, 85% dup) | ~900/file | thin `models/<m>/` descriptor + shared `train_step` |

## Forced trade-off (not optional, you should know it)

Comptime attention dims mean **each distinct (S,H,Dh) is a separate compile**
(monomorphization). Cost: binary size + compile time grow with the number of trained
shapes. Benefit: full comptime type-safety + no runtime shape dispatch. This is a Mojo
language constraint, not a design choice — the runtime `TrainConfig` carries the recipe
(lr/shift/eps/rank), but the shapes must cross as comptime params.

## Prerequisite that's real regardless of structure

Klein = **8 double-stream + 24 single-stream** blocks. The proven `dit_block` is
single-stream only. The **double-stream block fwd/bwd is the one genuinely-new compute
unit** Klein needs (its own parity gate before use). Z-Image (30 single-stream) maps to
the proven block directly — and per the H=30 finding (2026-05-30), is NOT blocked by
sdpa_backward.

## Migration order (cheapest-first)

1. Create `pipeline/` + `models/{klein,zimage}/` dirs; move the already-agnostic
   primitives (optim/schedule/loop/checkpoint/lora) under `pipeline/` (rename only).
2. Lift the duplicated step body from the two `train_*.mojo` into `pipeline/train_step.mojo`.
3. Reduce `train_klein.mojo`/`train_zimage.mojo` to thin `models/<m>/` descriptors +
   a comptime `match` entry point.
4. Build the Klein double-stream block (new compute unit) with its parity gate.
5. Add real weight loaders (safetensors → BlockWeights) per model.

Do this NOW while only 2 models exist — the per-model cost of the duplication compounds
with every model added (EDv2 reached 17).
