---
name: mojo-train-port
description: PRELIMINARY. The placement map + workflow for adding a model's TRAINING (LoRA or full fine-tune) to serenitymojo (/home/alex/mojodiffusion). Tells you exactly WHERE training code goes (shared training/ pipeline vs per-model models/<model>/), WHICH existing files to reuse vs build, the seam contracts, and HOW to verify (per-block torch-autograd parity, non-degenerate data). Use when asked to "port/add training for <model>" from an EriDiffusion-v2 trainer or a HF page. Sibling to `mojo-port` (which is INFERENCE-only). Worked example: Klein-9B LoRA (built+verified 2026-05-30).
---

# mojo-train-port

How to add a model's **training** to serenitymojo without rediscovering the layout.
The structure mirrors OneTrainer's modularity (audited 2026-05-30): a **shared
training pipeline written once** + a **thin per-model surface**. The job of porting
a new model's training = fill the per-model surface and reuse everything else.

> **SCOPE.** Pure Mojo + MAX, GPU, TRAINING (autograd/LoRA/full-FT). Reference impls
> (EriDiffusion-v2 Rust trainers, diffusers/PyTorch) are **parity oracles + architecture
> references only** — never imported by shipped Mojo. For INFERENCE porting use the
> sibling `mojo-port` skill. PRELIMINARY: Klein is the only fully-worked example; the
> per-model surface generalizes but the doc will sharpen as more models land.

## §0 — Receipt-of-reading (before writing or spawning agents)

State in one sentence each:
1. `serenitymojo/docs/MOJO_MODULES.md` — the current module map (shared vs per-model).
2. `serenitymojo/docs/MOJO_AUTOGRAD_INTERNALS.md` — the hand-chained block→stack
   composition + the d_x→d_y handoff contract.
3. `RECOMMENDED_TRAINER_STRUCTURE.md` — the design rationale for the split.
4. `serenitymojo/docs/MOJO_REUSABLE_INFERENCE_COMPONENTS.md` — what already exists to
   reuse (text encoders, tokenizer, VAE decoder, cache).
5. The **reference** to mirror: the EriDiffusion-v2 `train_<model>.rs` + the model's
   inference forward (`serenitymojo/models/dit/<model>_dit.mojo`) — name the exact
   block forward you must reproduce a backward for.
6. `mojo-syntax` skill + `mojo-port`'s "Mojo gotchas" section (the idioms apply here too).

## The structure (THE placement map)

```
serenitymojo/
  training/                        # SHARED pipeline — written ONCE, model-agnostic. REUSE, don't fork.
    train_config.mojo              #   TrainConfig (lr, shift, rank, alpha, eps, dims) — one struct
    train_step.mojo                #   shared LoRA step recipe + LoraAdapter/_lora_fwd/_lora_bwd/_lora_adamw
    optim.mojo schedule.mojo loop.mojo checkpoint*.mojo   # AdamW, flow-match/timestep, F32-master loop, ckpt
    dit_block.mojo                 #   the PROVEN single-stream block (template for new blocks)
    klein_dataset.mojo             #   dataloader (write_sample / KleinCache / load_batch) — generalize per model
    validation_sampler.mojo lora_save.mojo   # harness: sample (L2P gate), PEFT save/resume
  models/<model>/                  # PER-MODEL surface — THIS is what a new port fills (~6 files)
    config.mojo                    #   <model>_<variant>() -> TrainConfig  (dims+recipe; CONFIRM from real safetensors header)
    <model>_block.mojo             #   block fwd (saving acts) + bwd, + *_lora variants (the real new compute)
    <model>_stack.mojo             #   full-depth fwd+bwd: input proj → modulation → N blocks → final layer; per-block RECOMPUTE
    <model>_stack_lora.mojo        #   LoRA set across all blocks + adamw_step + save (if LoRA)
    weights.mojo                   #   safetensors -> the block weight structs (the loader)
    train.mojo                     #   thin entry point
    parity/                        #   <block>_oracle.py (torch) + <block>_parity.mojo gates + *_real_smoke.mojo
  io/train_config_reader.mojo      # OneTrainer JSON -> TrainConfig (shared)
  models/vae/<model>_encoder.mojo  # if the model's VAE encoder isn't built (image->latent for the data path)
```

## What VARIES per model (build these) vs what to REUSE (don't rebuild)

**Build (the per-model surface):**
1. **config** — dims + recipe. CONFIRM dims from the real `.safetensors` header (read it
   with python struct, don't trust the plan). Recipe from the EDv2 `train_<model>.rs` /
   OneTrainer JSON (cite file:line — source-fidelity gate).
2. **block fwd+bwd** — the one genuinely-new compute. Mirror the inference forward
   (`models/dit/<model>_dit.mojo`), add activation-saving + a hand-chained backward
   (template: `training/dit_block.mojo` and Klein's `models/klein/{double,single}_block.mojo`).
   ALL the backward ARMS you need almost certainly already exist (`ops/*_backward.mojo`:
   modulate, rms/layer_norm, linear, sdpa, swiglu, rope(interleaved/halfsplit), cat/slice/
   reshape, gate_residual). Check names carefully (e.g. `cat_backward` not `concat_backward`,
   `gate_residual_backward` not `residual_gate_backward`). If an arm is truly missing,
   build it in `ops/<area>_backward.mojo` with its own parity gate FIRST.
3. **stack** — compose blocks at full depth; per-block recompute for 24GB.
4. **weights loader** — safetensors keys → block weight structs (mirror `models/klein/weights.mojo`).
5. **LoRA targets** (if LoRA) — which projections; the inverse of `lora.mojo::_map_<model>_trainer`.
6. **sigma_map** (~2 lines) if the model's timestep→sigma differs.

**Reuse (do NOT rebuild — see MOJO_REUSABLE_INFERENCE_COMPONENTS.md):**
- Text encoders: `Qwen3Encoder.encode_klein` / `T5Encoder` / `Qwen25VLEncoder` / `ClipEncoder` + `Qwen3Tokenizer`.
- VAE decoder (and encoder if it exists); `cap_cache` for the latent/text cache.
- The shared `training/` pipeline (step recipe, LoRA math, optim, schedule, loop, ckpt, sampler, save, config reader).

## Seam contracts + Mojo constraints (a design MUST respect)

- **Attention dims (B,S,H,Dh) are COMPTIME** (the SDPA kernel is comptime-shaped) → the
  stack/step is `[..]`-parameterized; the runtime config carries only the recipe. Each
  model-shape is one monomorphization, selected by a top-level `match`.
- **Move-only Tensor** → the block API crosses as host `List[Float32]`; collections are
  `List[ArcPointer[Tensor]]`; multi-return via a `Movable` struct (NOT a tuple); `.copy()`/`^`
  struct fields before moving (even `Copyable` structs need it in some spellings).
- **No storable closures** → "grads-as-input": the loop drives fwd/bwd and hands grads
  back; you cannot inject a model as a stored callback.
- **Inter-block handoff**: a block's `d_x` = the next block's `d_y` (the proven composition contract).
- **Modulation** (AdaLN) is usually SHARED across blocks (one img_mod/single_mod feeds all
  blocks of a kind); ModVecs come from `linear(silu(t_embedder(timestep)), *_modulation.lin)`.

## Verification discipline (NON-NEGOTIABLE — the gate)

- **Per-block parity vs torch autograd**, cos ≥ 0.999 on the input grad + every weight grad
  (+ every LoRA d_A/d_B). Oracle = a torch model with the same math, `torch.autograd` for
  the reference, run as a SEPARATE command (`/home/alex/serenityflow-v2/.venv/bin/python`).
- **NON-DEGENERATE test data, always** (sinusoidal/random — NEVER modular fills like
  `(i*k)%9`). Structured fills alias at real model strides (`H*Dh`) → constant inputs whose
  TRUE gradient is zero → `cos(≈0,≈0)` reads as a spurious FAIL. This faked a whole "H=30
  sdpa bug" in 2026-05-30. Verify at REAL head count (e.g. H=32), small Dh/N for oracle speed.
- **Real-depth FINITE smoke** on real weights (full block count, D, with checkpointing,
  no OOM) — torch can't do full-real-dim parity cheaply, so: small-dim torch parity +
  real-depth finite + per-block real-H parity = composition proven.
- **VAE encoder gate**: encoded real-image latent std ≈ 0.96 (≈0.85 = HWC↔CHW scramble).
- **The lead re-runs every agent claim** on a clean serial build before counting it
  (`rm -f serenitymojo.mojopkg` first; `pixi run mojo run -I .`).

## Porting workflow (given "port <model> training from EDv2/HF")

1. **Confirm dims** from the real `.safetensors` header + the recipe from EDv2 `train_<model>.rs`
   / OneTrainer JSON (cite). Write `models/<model>/config.mojo`.
2. **Identify the block kind(s)** from the inference `models/dit/<model>_dit.mojo` forward.
   Map every op to its backward arm (most exist). Build any missing arm + gate it first.
3. **Build block fwd(save-acts)+bwd (+LoRA variants)** mirroring `dit_block.mojo`/Klein.
   Parity-gate each block vs torch (cos ≥ 0.999, non-degenerate, real H).
4. **Build the stack** (full depth, per-block recompute) + its parity + a real-weight finite smoke.
5. **Loader** (`weights.mojo`) + a real-weight finite forward.
6. **Data path**: reuse `klein_dataset`/`cap_cache` + the right text encoder; build the
   model's VAE encoder only if missing (gate std≈0.96).
7. **Wire `train.mojo`** (or a `train_<model>_real.mojo` loop): cache→flow-match target→
   stack fwd/bwd→AdamW/LoRA-adamw→log loss+grad_norm; reuse `validation_sampler` + `lora_save`.
8. **Real-run gate (the L2P verdict)**: loss DROPS **and a sample SHIFTS** with the LoRA —
   never loss alone.

Spawn agents the same way `mojo-port` does (builder writes + compiles, skeptic attacks,
bugfix repairs, lead re-verifies). Only ONE agent compiles at a time (concurrent `mojo`
compiles corrupt the shared cache).

## Reference docs (read, don't re-derive)
`serenitymojo/docs/MOJO_{MODULES,AUTOGRAD_INTERNALS,KERNELS,CONVENTIONS,DIAGNOSTICS,REUSABLE_INFERENCE_COMPONENTS}.md`,
`RECOMMENDED_TRAINER_STRUCTURE.md`, the Klein worked example under `models/klein/`.
