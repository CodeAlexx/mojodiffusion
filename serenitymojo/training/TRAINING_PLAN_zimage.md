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
baseline. This is training-speed evidence only. It is not a sampler-speed
parity claim because it has no matched prompt, seed, guidance, denoise timing,
VAE decode timing, peak VRAM, or image artifact pair.

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

2000-step convergence verification:

```
log=output/logs/zimage_train_2000_speed2_tensor_main_2026-06-02.log
loss=0.47321588 -> 0.5490076
mean_loss=0.459294
mean_speed=2.0215s/step
last_speed=2.0117514s/step
nonfinite=0
loraB_sum=267434.25
loraB_nonzero=210/210
checkpoint=output/alina_zimage/zimage_lora_step2000.safetensors
```

The run hit the production Alina buckets instead of dropping singleton or
long-caption cases. This is the current convergence baseline: finite throughout,
LoRA-B nonzero on every main adapter. Treat the speed lines above as historical
run notes unless the named log/checkpoint artifacts are present in the current
workspace and a strict speed manifest records matched OneTrainer/Mojo identity,
stage timings, and peak VRAM.

1024 sampling hook: `serenitymojo/pipeline/zimage_generate.mojo` now accepts
`[lora_path|base] [out_png] [seed] [prompt]`. It loads the BF16/BP16-preserving
Mojo Z-Image stack and applies main-only LoRA with AI Toolkit-style forward
overlay:

```
base_forward(x) + lora_up(lora_down(x)) * multiplier * alpha/rank
```

Do not change this back to `LoraSet.merge_into_indexed`; Z-Image production
sampling must exercise the same overlay path as training. The sampler also uses
`zimage_block_lora_predict_device_tensor`, a no-save inference main-block
forward, because the training saved-activation forward can OOM at 1024.

Compile-only gate passed:

```
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_generate.mojo -o /tmp/zimage_generate_lora_check
```

Caption-based 1024 samples completed from the 2000-step LoRA:

```
output/alina_zimage/sample_step2000_alina000_seed42_1024.png    # 224 tokens, 364.42s
output/alina_zimage/sample_step2000_alina003_seed31415_1024.png # 193 tokens, 367.26s
output/alina_zimage/sample_step2000_alina007_seed27182_1024.png # 208 tokens, 364.02s
```

All three logs show `overlay loaded 210 main-layer adapters; scale alpha/rank =
0.0625`; all three PNGs are valid 1024x1024 RGB and visually align with their
staged captions. These wall times are not accepted sampler speed parity: the
current no-CUDA gate requires prompt, seed, resolution, steps, guidance, dtype,
OneTrainer/Mojo denoise seconds per step, VAE decode seconds, peak VRAM, and
artifact paths before a strict speed claim can pass.

Strict sampler-speed readiness is guarded by
`scripts/check_zimage_sampler_contract.py --strict-speed`. That gate is
manifest-only and no-CUDA: it reads the existing sampler/forward JSON plus an
optional `zimage_sampler_speed.json` or `zi_sampler_speed.json`. It accepts
either a shared `run_identity` or side-specific OneTrainer/Mojo identity fields,
but the effective prompt, seed, resolution, steps, guidance, and dtype must
match exactly between the two runs. It also requires paired OneTrainer and Mojo
denoise seconds per step, VAE decode seconds, peak VRAM MiB, and non-empty
artifact paths that exist locally. It still does not compare pixels.

Use `scripts/check_zimage_sampler_contract.py --write-speed-readiness <path>`
to write the same field audit as JSON. As of 2026-06-06,
`scripts/check_zimage_sampler_contract.py --strict-speed` passes as comparable
speed evidence against `/home/alex/onetrainer-mojo/parity/zimage_sampler_speed.json`.
That is not a speed-parity claim: the paired 1024x1024, 28-step, seed 42,
CFG 3.5 record has OneTrainer denoise `2.144007647s/step`, VAE
`1.0628398s`, peak `14340 MiB`, while the paired Mojo record has denoise
`5.206695175s/step`, text encode `33.874484788s`, VAE `1.809631493s`,
peak `22238 MiB`, and supervisor wall `225.13s`.
Use `scripts/check_zimage_sampler_contract.py --write-strict-speed-template /tmp/zimage_sampler_speed_template.json`
to write the expected speed-manifest shape. The template contains placeholder
values and does not satisfy `--strict-speed`; it is only a checklist for the
matched OneTrainer/Mojo timing and VRAM run.

Product-control smoke now covers the OneTrainer-style entrypoint for
`zimage_lora_16gb`: it resolves to `train_zimage_real`, binds Z-Image cache
preflight fields, requires the sample prompt file, and proves step-500
save-before-sample cadence. At sample cadence, `train_zimage_real` now saves
LoRA/state and writes a `serenity.zimage.sample_request.v1` manifest under
`output/alina_zimage/sample_requests/` with the standalone `zimage_generate.mojo`
build/run command plus a `result_manifest` path. The generator now accepts
`--request <manifest.json>`, validates the LoRA/state/sample paths before CUDA
setup, runs the existing LoRA overlay sampler, saves the PNG, and writes a
`serenity.zimage.sample_result.v1` JSON with Mojo-side text-encode, denoise,
and VAE-decode timings. The result manifest intentionally records
`accepted_sampler_parity=false` and `accepted_speed_parity=false`; it does not
measure peak VRAM. This is the safe 24GB direction: run the sampler after
trainer memory is released. It is not sampled-output parity until a supervisor
or manual standalone run consumes a real request and records the matched
OneTrainer/Mojo artifact, timing, and VRAM bundle.

`serenitymojo/training/zimage_sample_supervisor.mojo` now provides the
Mojo-native process-separation runner for queued requests:

```
pixi run mojo build -I . serenitymojo/training/zimage_sample_supervisor.mojo -o /tmp/zimage_sample_supervisor
/tmp/zimage_sample_supervisor output/alina_zimage/sample_requests/step500_request.json /tmp/zimage_generate_prod dryrun
```

For measurement runs, `scripts/run_zimage_sample_requests.py` is the support
wrapper that can launch the Mojo sampler, poll `nvidia-smi`, validate the
emitted result manifest, and write a Mojo-side `zimage_sampler_speed.json`
shape. Dry-run metadata and generator-only `peak_vram_mib=0` do not satisfy
`--strict-speed`; the strict gate still needs positive measured VRAM plus
paired OneTrainer fields.

### 2026-06-06 sampler speed/profile pass

Current code-path changes for the standalone 1024 sampler:

- base mode builds a zero-LoRA device set instead of uploading full host zero
  adapters;
- `zimage_lora_apply_device` skips zero-scale adapters;
- the noise-refiner image sequence is computed once per step and reused for
  cond/uncond CFG;
- `cfg <= 1.0` skips the uncond main-stack pass;
- final patch rows are unpatchified on device in Z-Image channel-minor order;
- main-block inference uses `vec_modulate` and `vec_swiglu`;
- `--trace-denoise` is available on `zimage_generate.mojo` and is off by
  default. Trace mode synchronizes the first denoise step to print real stage
  timings.

Measured after the fused/device cleanup, but before accepted speed parity:

```
artifact=/tmp/zimage_speed_28step_fused_speed.json
identity=1024x1024 seed=42 steps=28 cfg=3.5 dtype=bf16
text_encode_seconds=10.36549498
denoise_seconds=118.562242613
denoise_seconds_per_step=4.234365807607142
vae_decode_seconds=1.499334517
peak_vram_mib=22137
```

Current one-step supervisor evidence after the device mod-vector cleanup:

```
artifact=/tmp/zimage_speed_1step_moddev_speed.json
text_encode_seconds=10.645762139
denoise_seconds_per_step=4.302052192
vae_decode_seconds=1.487801541
peak_vram_mib=22111
output=/tmp/zimage_speed_1step_moddev.png
```

The device mod-vector cleanup built and ran, but did not materially improve
speed. A traced one-step run printed:

```
adaln_mods 0.017444382s
noise_refiner 0.396803831s
main_cond 1.975725455s
main_uncond 1.905335169s
cfg_combine 0.002771174s
scheduler_update 0.001488217s
```

The remaining speed blocker is the CFG denoise structure. OneTrainer batches
cond/uncond when `cfg_scale > 1.0`; the current Mojo path runs the two main
stacks serially. A previous naive batch-2 tiled-attention attempt was rejected
because it was much slower and near the 24 GB ceiling. The next real speed
target is a fast BF16 `Dh=128` attention path and/or a proven batched-CFG path,
not another small host-boundary cleanup.

The focused Z-Image train-readiness guard now clears the latent/text cache
host-F32 step-math staging blocker. Cached BF16/F16 latent/text tensors are read
through dtype-specific host lists, with F32 limited to local scalar math,
targets, reductions, and transient upload lists.

Z-Image full-finetune is still not production-supported. The
`zimage_full_finetune_inventory_smoke` validates the current transformer
full-weight inventory at `521` keys, and
`zimage_full_finetune_checkpoint_smoke` saves the live-struct payload order,
writes the tensor-name manifest, and loads the flat payload back with BF16
storage preserved. `zimage_full_finetune_state_smoke` now binds that `521`
tensor manifest to the TrainState optimizer sidecar order
`param.N`/`adam_m.N`/`adam_v.N`/`__meta__`. `scripts/check_full_finetune_inventory_keys.py --strict`
also validates the `521` inventory keys against the local sharded index and
confirms context refiners have no AdaLN keys. The remaining full-finetune
blockers are the LoRA-only train loop, product-loop full-finetune parity, and
parity artifact.

### Full-F32 blocker status

MEASURED: full-model resident all-F32 base = 24.62 GB > 24 GB GPU. That remains
unsupported and must stay documented. The working path is BF16/BP16-style base
residency with F32 limited to LoRA/Adam, reductions, and short transients.

### NEXT

1. Add the external sampler/supervisor step that consumes queued
   `serenity.zimage.sample_request.v1` manifests, verifies the emitted
   `serenity.zimage.sample_result.v1`, records peak VRAM externally, and writes
   the matched OneTrainer/Mojo artifact bundle.
2. Keep validation sampling split-process for Z-Image; do not treat a queued
   request as sampled-output, image, or speed parity.
3. Keep expanding tensor-resident and offloaded APIs rather than reintroducing
   host-list block boundaries in production code.
