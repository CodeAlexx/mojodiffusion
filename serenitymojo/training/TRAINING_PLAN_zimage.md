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

## STATUS (2026-06-02): reduced-depth real run done; full-depth BF16/offload work pending.

### Files
- `models/zimage/weights.mojo` — +load_zimage_block_weights_prefixed (nr/cr/main)
  for parity plus `load_zimage_block_weights_prefixed_mixed` for training.
- `models/zimage/real_weights.mojo` — NEW. ZImageRealAux + embedder/modvec/
  rope/f_scale builders translating zimage_dit.mojo host math.
- `pipeline/zimage_prepare.mojo` — NEW. cache reader/inspector (VAE port skipped;
  cache exists). Verified: 51 samples, latent [1,16,64,64], text_emb [1,512,2560].
- `training/train_zimage_real.mojo` — NEW. The real loop.

### REAL run result (real base weights + real cache, MAIN_DEPTH=4)
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

### BLOCKER for full-depth (MAIN=30) real run

MEASURED: full-model resident all-F32 base = 24.62 GB > 24 GB GPU (computed from
real shapes). That is not a supported training mode. The full-depth Mojo path
must preserve BF16/BP16 base projection weights and add BF16 residency and/or a
turbo/offload block loader like Klein's before flipping `MAIN_DEPTH=30`.

The existing reduced-depth run is only a correctness probe. Loss MAGNITUDE
(~445) and the slow per-step delta are partly the MAIN=4 truncation artifact
(26 missing layers => final output uncalibrated to the latent scale); the
DIRECTION (monotonic down) is the correctness signal.

### NEXT (to land full-depth)
1. BF16/BP16-preserving zimage block fwd/bwd (or reuse an offload loader) so
   full-depth training never expands the base model to all-F32. Then flip
   MAIN_DEPTH=30.
2. Per-layer parity of build_x_seq/build_adaln/build_block_modvecs/build_f_scale
   vs zimage_dit.mojo on real weights (cos>=0.999) before the long run, to
   confirm the loss magnitude is purely the truncation artifact.
