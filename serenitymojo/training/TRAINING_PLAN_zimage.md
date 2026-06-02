# TRAINING_PLAN_zimage.md — Z-Image (NextDiT) LoRA trainer, pure Mojo

Status: BUILDING (2026-06-01). Translation of the working Rust trainer
`EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_zimage.rs` onto the
parity-verified Mojo Z-Image LoRA stack.

## What we TRANSLATE FROM (read line-by-line)
- `train_zimage.rs` — per-step recipe: cache load → logit-normal sigma →
  flow-match (x_t, v-target) → forward → MSE → backward → clip(1.0) → AdamW.
  Recipe scalars: lr=3e-4, rank=16, alpha=1.0, timestep_shift=1.0 (released
  preset), VAE shift=0.1159 scale=0.3611 (applied to cached latent at train).
- `zimage_dit.mojo` (inference) — SOURCE for embedders + modvecs + rope +
  final-layer (t_embedder sinusoid→mlp, cap_embedder=RMSNorm+Linear,
  x_embedder=patchify+Linear, adaLN modvec chunk4, rope positions, f_scale).

## What we REUSE (parity-verified this session)
- `models/zimage/zimage_stack_lora.mojo` — build_zimage_lora_set,
  zimage_stack_lora_forward/backward, zimage_lora_adamw_step, save_zimage_lora.
  Verified by `parity/lora_step_smoke.mojo` (B 0→nonzero, grads finite,
  save/load byte-exact) + `parity/lora_stack_parity.mojo`.
- `training/schedule.mojo` — sample_timestep_logit_normal, flow_match_noise_target
  (x_t=(1-σ)latent+σnoise, target=noise-latent — Mojo-stack convention).
- `train_klein_real.mojo` — loop template (per-step timing, grad clip, board,
  PROG line, save cadence).

## Weight source (RESOLVED 2026-06-01)
The DIFFUSERS directory `/home/alex/.serenity/models/zimage_base/transformer/`
has the UNFUSED `to_q/to_k/to_v/to_out.0` + `all_x_embedder.2-1` +
`all_final_layer.2-1` + `*_norm.weight` names that the existing Mojo
`weights.mojo` loader and LoRA target map already expect. (The single-file
`z_image_base_bf16.safetensors` is the FUSED comfy format — NOT used.)
Arch: D=3840 (H=30,Dh=128), F=10240, NR=2, CR=2, MAIN=30, out_ch=64
(patchified, 16ch×2×2), cap_dim=2560, adaln_dim=256, t_scale=1000, theta=256,
rope_axes=(32,48,48).

## Cache (FAST PATH — VAE encoder port SKIPPED)
`/home/alex/EriDiffusion/EriDiffusion-v2/cache/alina_zimage_512` — 51 files,
each {latent F32 [1,16,64,64], text_embedding F32 [1,512,2560], text_mask
[1,512]}. Latents are RAW posterior.mode(); trainer applies (lat-shift)*scale.

## Build steps
1. weights.mojo: add `load_zimage_block_weights_prefixed(prefix)` for
   noise_refiner.{i}/context_refiner.{i}/layers.{i} (currently layers-only).
   [DONE / pending]
2. zimage_real_weights.mojo: embedder + modvec + rope + final-layer loaders
   that translate zimage_dit.mojo's host math into the stack's inputs
   (x_seq, cap_seq, nr_mod[], main_mod[], f_scale, rope tables, final_lin).
3. zimage_prepare.mojo: cache reader (latent/text_embedding/text_mask). Since
   cache exists, this just reads + reshapes.
4. train_zimage_real.mojo: the loop. 512px → img grid 32×32 patch2 → 256 img
   tokens → pad32 → 256; cap 512 → pad32 → 512; S=768.
5. Real smoke: 2-step then 5-step; show PROG loss↓ + LoRA-B 0→nonzero.

## Resolution / token math (512px)
latent [16,64,64] → patchify(p=2) → (32×32)=1024 img tokens, ch=64.
WAIT: 64/2=32 grid → 32*32=1024 img tokens (img_pad: 1024%32==0 → 0 pad).
cap 512 (already mult of 32). S = 1024 + 512 = 1536.

## HARD DTYPE RULE (2026-06-02)

Z-Image training is BF16/BP16 for the base model weights. OneTrainer does not
train a full-F32 Z-Image model, and this Mojo trainer must not try to do that
either. A full-F32 Z-Image base/model load is invalid for the local 24 GB target
and will OOM.

Allowed F32 uses are narrow: scalar loss/reduction math, optimizer masters,
transient accumulators, activations currently carried by the parity stack, and
small norm-vector compatibility tensors while `rms_norm` requires matching
input/weight dtype. Large block projection and MLP weights (`to_q`, `to_k`,
`to_v`, `to_out`, `w1`, `w3`, `w2`) must stay in checkpoint dtype through
`load_zimage_block_weights_prefixed_mixed` and/or an offloaded BF16-preserving
path.

## STATUS (2026-06-02): full-depth tensor-resident LoRA path running.

Runtime/API guide for future work: `docs/MOJO_TRAINER_RUNTIME_API_GUIDE.md`.
That guide records the offloader and scratch-ring usage rules that came out of
this Z-Image speed pass.

### Files
- `models/zimage/weights.mojo` — +load_zimage_block_weights_prefixed (nr/cr/main)
  for parity plus `load_zimage_block_weights_prefixed_mixed` for training.
- `models/zimage/real_weights.mojo` — NEW. ZImageRealAux + embedder/modvec/
  rope/f_scale builders translating zimage_dit.mojo host math.
- `pipeline/zimage_prepare.mojo` — NEW. cache reader/inspector (VAE port skipped;
  cache exists). Verified: 51 samples, latent [1,16,64,64], text_emb [1,512,2560].
- `training/train_zimage_real.mojo` — NEW. The real loop.

### Historical reduced-depth probe (real base weights + real cache, MAIN_DEPTH=4)
Overfit-correctness probe (fixed sample+timestep), 10 steps:
```
step1 loss=445.291 loraB_sum=1251 nonzero=56/56 nonfinite=0
...
step10 loss=444.384 loraB_sum=11473 nonzero=56/56 nonfinite=0
loss 445.291 -> 444.384 (MONOTONIC DECREASE every step); LoRA-B 0 -> 11473.
```
This is the canonical trainer-correctness gate: fixed input + correct backward
=> loss MUST drop monotonically. It does. LoRA-B grows 0->nonzero on all 56
adapters; grads finite. Checkpoint saved (37MB, 56 adapters, PEFT names).
Multi-sample mode (OVERFIT_PROBE=False) runs too but loss oscillates per-step
(different timestep+sample each step) — expected variance, not divergence.

Do not use the `~445` reduced-depth loss as a baseline. It is a truncated-stack
wiring/backward smoke only. For full-depth Z-Image, `~445` is a broken run; the
target scale is the OneTrainer baseline below.

### OneTrainer 100-Step Baseline (512, Klein/Alina dataset)

OneTrainer Z-Image LoRA baseline used the local `Otpreset` Z-Image preset, local
`/home/alex/.serenity/models/zimage_base`, batch 2, LR `3e-4`, logit-normal
timestep sampling, BFLOAT_16 train/weight/output dtype, and the Klein/Alina
cache. At the 100-step save:

```
loss=0.541
smooth_loss=0.457
warm_speed=2.0-2.2s/it
checkpoint=/home/alex/OneTrainer/workspace/alina_zimage_OTpreset_100_baseline/save/2026-06-02_01-06-16-save-99-3-24.safetensors
```

The process was manually terminated after OneTrainer continued beyond the 100
step save; use only the metrics through the `save-99-3-24` checkpoint as the
baseline.

### Full-depth Mojo result after tensor-resident main-stack fix

The full-depth Z-Image LoRA trainer now runs all 30 main layers in the real
Alina 512 path. The speed fix was not a recipe change; it removed hot host
boundaries:

- LoRA adapters upload once per step as device tensors.
- Main-stack forward/backward recompute stays tensor-resident.
- Per-block `to_host()` / `Tensor.from_host()` round trips are gone from the
  production path.
- Frozen final norm uses dx-only backward instead of computing discarded norm
  weight grads.
- Observed cache buckets `72x56/cap224`, `72x56/cap256`, `88x48/cap224`, and
  `88x48/cap256` are all dispatched.

100-step verification:

```
log=output/logs/zimage_train_100_speed2_tensor_main_2026-06-02.log
loss=0.47321588 -> 0.35350168
nonfinite=0
speed=~1.96-2.00s/step warm, final step 1.993s
checkpoint=output/alina_zimage/zimage_lora_step100.safetensors
```

A 2000-step convergence run was started from the same code path on 2026-06-02.
Append its final loss/speed/sample metrics here after it finishes.

1024 sampling hook: `serenitymojo/pipeline/zimage_generate.mojo` now accepts
`[lora_path|base] [out_png] [seed]`, merges a PEFT LoRA into resident `NextDiT`
with `LoraSet.merge_into_indexed`, and preserves the existing 1024 denoise/VAE
path.
Compile-only gate passed:

```
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_generate.mojo -o /tmp/zimage_generate_lora_check
```

Run three samples after the 2000-step trainer frees the GPU:

```
/tmp/zimage_generate_lora_check output/alina_zimage/zimage_lora_step2000.safetensors output/alina_zimage/sample_step2000_seed42_1024.png 42
/tmp/zimage_generate_lora_check output/alina_zimage/zimage_lora_step2000.safetensors output/alina_zimage/sample_step2000_seed31415_1024.png 31415
/tmp/zimage_generate_lora_check output/alina_zimage/zimage_lora_step2000.safetensors output/alina_zimage/sample_step2000_seed27182_1024.png 27182
```

### Full-F32 blocker status

MEASURED: full-model resident all-F32 base = 24.62 GB > 24 GB GPU. That remains
unsupported and must stay documented. The working path is BF16/BP16-style base
residency with F32 limited to LoRA/Adam, reductions, and short transients.

### NEXT

1. Finish the 2000-step convergence run and sample through the Mojo Z-Image
   sampler with the saved PEFT LoRA.
2. Add/verify validation sampling cadence for Z-Image so the trainer can save
   and sample like Klein.
3. Keep expanding tensor-resident and offloaded APIs rather than reintroducing
   host-list block boundaries in production code.
