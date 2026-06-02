# Klein 9B Training Handoff - 2026-06-02

This is the restart handoff for the Mojo trainer work. The active user goal is a
valid Klein 9B LoRA training run with normal sampling and convergence, then the
same training/runtime foundation applied to Z-Image and Anima.

For future trainer/runtime edits, also read
`docs/MOJO_TRAINER_RUNTIME_API_GUIDE.md`. It captures the Z-Image speed lessons,
offloader loop contract, scratch ring allocator lifetimes, dtype rules, and
PEFT/ai-toolkit save-format rule in one API-facing document.

## User Priorities

- Klein 9B LoRA must train for a real convergence run, normally 2000 steps.
- Expected mature loss range is roughly `0.200` to `0.900`; user has seen useful
  convergence around 1400-1600 steps, often by 2000.
- Klein learning rate must be `0.0004`.
- Speed target is Rust-like `2.xx s/step` for 9B. Current Mojo smoke is closer
  than before but still not there.
- Sampling must be 1024x1024 by train-time config/prompt JSON, not separate
  fixed-resolution sampler files.
- Samples should happen at step 0 and every 500 steps for real runs; short
  smoke runs under `sample_every` intentionally skip samples.
- Sample prompts belong in a shared JSON file reusable across models.
- Trainer display must be Mojo, shared across trainers, and Rust-like:
  loss, grad_norm, step speed, elapsed, ETA, and noise/noising speed.
- SerenityBoard wiring is required. It is in `/home/alex/serenityboard`.
- Python/PyTorch is allowed for parity tests and reference stats only. Runtime
  trainer UI, sampler, prepare/cache generation, and final implementation must
  be Mojo. Rust side stays pure Rust.
- Trainer runtimes must be self-contained Mojo. Do not depend on Rust/EriDiffusion
  caches or Python/OneTrainer caches for production prepare/train/sample paths.
- LoRA saves should stay in the generic PEFT/ai-toolkit-compatible safetensors
  format. It works with our inference side and `/home/alex/ai-toolkit`; do not
  switch Z-Image/Klein saves to a model-private format unless the inference
  loader is updated too.
- Prefer shared/core speed work that benefits all models, not Klein-only hacks.

## Reference Trees

- Klein, SDXL, Z-Image, Ernie training reference: `/home/alex/OneTrainer`
- Anima training reference: `/home/alex/OneTrainer-anima-ref`
- Rust/EriDiffusion and Flame Core are bug-fix/speed cheat sheets, especially
  for offload, allocator, optimizer, sampler, and convergence behavior.

Do not treat Rust as the preset authority when OneTrainer has the model preset;
use Rust as proof of solved implementation issues only. OneTrainer is read-only:
reference formulas, presets, and baseline numbers there, but do not modify it
and do not make our trainer depend on its cache files.

## Current Repo State

Workspace: `/home/alex/mojodiffusion`

There is a large dirty worktree with many user and generated changes. Do not
revert unrelated files. Important touched/added areas include:

- `serenitymojo/training/train_klein_real.mojo`
- `serenitymojo/models/klein/klein_stack_lora.mojo`
- `serenitymojo/training/progress_display.mojo`
- `serenitymojo/training/serenityboard.mojo`
- `serenitymojo/training/sample_prompt_config.mojo`
- `serenitymojo/training/lora_save.mojo`
- `serenitymojo/configs/klein9b.json`
- `serenitymojo/configs/klein9b_alina_samples.json`
- `serenitymojo/docs/TRAINER_DISPLAY_CONTRACT_2026-05-31.md`
- `serenitymojo/docs/TRAINER_SAMPLE_PROMPTS_AND_BOARD_2026-05-31.md`
- `serenitymojo/docs/TRAINER_RUNTIME_PARITY_AUDIT_2026-06-01.md`

The last build session completed successfully with warnings only:

```bash
pixi run mojo build -I . -Xlinker -lcuda -Xlinker -lm \
  -Xlinker /lib/x86_64-linux-gnu/libsqlite3.so.0 \
  serenitymojo/training/train_klein_real.mojo \
  -o /tmp/train_klein_real_save4
```

Warnings were around unreachable code from compile-time constants and an unused
`clip_scale`; no compile failure.

## Klein Config State

`serenitymojo/configs/klein9b.json` currently has:

- `validation_prompts_file`: `/home/alex/mojodiffusion/serenitymojo/configs/klein9b_alina_samples.json`
- `learning_rate`: `4e-4`
- `timestep_shift`: `1.0`
- `sample_every`: `500`

OneTrainer Klein presets checked so far also use `timestep_shift: 1.0` with
dynamic shift disabled.

Z-Image correction: the active OneTrainer Z-Image 512 baseline was verified on
2026-06-02 with `timestep_shift=1.0`, `dynamic_timestep_shifting=false`, and
`timestep_distribution=LOGIT_NORMAL`. Earlier notes mentioning `1.8` were for
other Serenity/Klein paths, not the OneTrainer Z-Image baseline. Use OneTrainer
as source of truth unless a newer Z-Image preset says otherwise.

Z-Image VAE scaling is `shift_factor=0.1159`, `scaling_factor=0.3611`; it is not
the BN/Flux2/Klein latent normalization path.

Z-Image dtype rule: do not try full-F32 Z-Image training. OneTrainer baseline is
BF16/BP16-style for train/base/output dtypes, and full-F32 Z-Image model loads
will OOM on the 24 GB 3090 Ti. F32 LoRA masters/small scalar reductions are fine;
the large base model must stay BF16/BP16/offloaded.

Z-Image OneTrainer 100-step baseline:

- preset: `/home/alex/OneTrainer/configs/alina_zimage_OTpreset_100_baseline.json`
- resolution/batch/lr: `512`, batch `2`, learning rate `3e-4`
- LoRA filter: `^(?=.*attention)(?!.*refiner).*,^(?=.*feed_forward)(?!.*refiner).*`
- target set: main `layers.*` attention + feed-forward only; exclude noise/context
  refiners.
- final observed line near save: loss `0.541`, smooth loss `0.457`
- warm speed: about `2.0-2.2s/it` for batch 2

## 2026-06-02 Klein Result

The 2000-step Klein 9B LoRA run completed successfully with normal checkpoint
and sample cadence:

```bash
timeout 14400s /tmp/train_klein_real_shape_kernel \
  /home/alex/mojodiffusion/serenitymojo/configs/klein9b.json \
  2000 0
```

Final line:

- Step `2000/2000`
- Loss `0.8455`
- Grad norm `0.1116`
- Speed `2.1s/step`
- Offload fallbacks `0`

Warm training speed held around `2.0-2.1s/step`. The only slower lines were
immediately after sampling restarts, around `3.1-3.2s/step`.

Saved final artifacts:

- `output/alina_train/alina_lora_final.safetensors`
- `output/alina_train/alina_lora_final.safetensors.state.safetensors`
- `output/alina_train/sample_step2000_alina_garden.png`
- `output/alina_train/sample_step2000_alina_evening.png`

Checkpoints and samples at steps `500`, `1000`, and `1500` also completed.
The final 512 PNGs were visually checked and are nonblank.

The in-process run used `512x512` validation prompts because `1024x1024`
samples in the live trainer context OOM on the 24 GB 3090 Ti. Use the
process-separated sampler/cadence path for 1024 validation. A standalone 1024
garden sample from the final LoRA succeeded:

```bash
/tmp/klein_sample_cli_1024_current \
  /home/alex/mojodiffusion/serenitymojo/configs/klein9b.json \
  /home/alex/mojodiffusion/output/alina_train/alina_lora_final.safetensors \
  /home/alex/mojodiffusion/serenitymojo/configs/klein9b_alina_samples_1024.json \
  alina_garden \
  /home/alex/mojodiffusion/output/alina_train/sample_step2000_alina_garden_1024.png
```

That run used `N_IMG=4096`, denoised at about `4.8-4.9s/step`, reported
`fallbacks 0`, and saved
`output/alina_train/sample_step2000_alina_garden_1024.png`.

## Speed Fixes Proven

`DBL_SAVE_TAIL=4` OOMed during step-1 forward; keep
`DBL_SAVE_TAIL = 0` until a real memory-planned activation strategy exists.

The effective speed improvement came from removing hot-path rank-2 shape-copy
storms in `serenitymojo/ops/tensor_algebra.mojo`:

- F32/BF16/F16 rank-2 concat fast paths for arity 2 and 3, `dim=1`.
- F32/BF16/F16 rank-2 slice fast paths, `dim=1`.
- Klein q/k/v and gate split/join paths no longer perform thousands of per-row
  D2D sub-buffer copies.

Nsight Systems profile with local `nsys` 2024.1.1:

- Old profile: `321,626` D2D copies, `424.10 ms` GPU D2D, about `1.05s`
  `cuMemcpyDtoDAsync` API time.
- New profile: `1,013` D2D copies, `30.49 ms` GPU D2D.

The tiny 8/16/24/48 KiB D2D storm is gone. Remaining large H2D traffic is block
streaming, not the shape-copy bug.

## Completed Fixes In This Pass

### Shared Training Math

`serenitymojo/training/schedule.mojo`

- Added fail-loud dtype checks for F32-only tensor helpers:
  `flow_match_noise_target`, `ema_update`, and `grad_accumulate`.
- This avoids silent raw bitcasts of non-F32 buffers.

`serenitymojo/training/noise_modifiers.mojo`

- Bernoulli helper now uses shared `sample_timestep_uniform(seed)` instead of a
  separate PCG-style fraction.
- Token-space multires noise now raises instead of silently no-oping when
  enabled. Multires requires a real 4D NCHW path.

`serenitymojo/training/noise_modifiers_smoke.mojo`

- Added smoke coverage for the multires fail-loud path.

### Ernie Trainer

`serenitymojo/training/train_ernie_real.mojo`

- Fixed RNG advancement in `_sample_sigma_idx`.
- Old code sampled from a copied state and did not advance the caller state.

### Klein Trainer

`serenitymojo/training/train_klein_real.mojo`

- SerenityBoard/machine LR logging now uses scheduled `step_lr`, not static
  `cfg.lr`.
- Mid grad-accum branch now computes and passes pending scheduled LR.
- Sampling is controlled by shared prompt JSON and `sample_every`.
- Short runs below `sample_every` skip samples so a 50-step smoke does not waste
  time judging a useless LoRA sample.

### Docs

`serenitymojo/docs/TRAINER_RUNTIME_PARITY_AUDIT_2026-06-01.md`

- Added authoritative training references.
- Added Flame Core additions audit.
- Captured green gates and caveats.

This handoff doc is the continuation entry point.

## Gates That Passed

Training/parity smokes reported green:

- `lr_schedule_parity`
- `loss_weight_parity`
- `timestep_bias_parity`
- `noise_modifiers_smoke`
- `train_config_reader_smoke`
- `reader_levers_reachability_smoke`
- `loop_levers_integration_smoke`
- `grad_accum_smoke`
- `opt_lion_parity`
- `opt_stableadamw_parity`
- `opt_prodigy_parity`
- `opt_adafactor_parity`
- `opt_schedulefree_parity`
- `disk_check_smoke`
- `transfer_benchmark_smoke`
- `dpmpp_2m_parity`
- `unipc_parity`
- `dpmpp_2m_tensor_smoke`
- `unipc_tensor_smoke`
- `schedule_parity`
- `board_roundtrip_smoke`

Compile gates passed after the fixes:

- `train_ernie_real.mojo`
- `train_zimage_real.mojo`
- `train_sdxl_real.mojo`
- `train_klein_real.mojo`

`train_klein_real.mojo` compiled again after the `DBL_SAVE_TAIL=4` change, but
that runtime has not been profiled yet.

## Bottlenecks And Best Next Fixes

### 1. Activation/Recompute Strategy

Backward is currently about `1.63s` of a `3.1s` step. The current
`DBL_SAVE_TAIL=4` attempt may not help and may OOM. If it fails, do not keep
pushing fixed constants. Better shared direction:

- make activation retention configurable by memory budget;
- profile per-block save/recompute costs;
- prefer a ring/slab activation allocator or planner that can benefit every
  transformer trainer;
- avoid saving huge activation tails blindly.

### 2. Device-Resident LoRA Optimizer Path

Klein currently uploads the LoRA set to device around each step and pulls grads
host-side for AdamW. This is likely a shared speed ceiling for every LoRA model.

Look around:

- `serenitymojo/training/train_klein_real.mojo`
- `serenitymojo/models/klein/klein_stack_lora.mojo`
- `serenitymojo/training/fused_adamw_multitensor.mojo`
- `serenitymojo/training/on_device_global_norm.mojo`

Best shared target: keep trainable tensors and optimizer state device-resident
where possible, compute global grad norm on device, and reduce per-step
host/device copies.

### 3. Grad Accum Default Path

Subagent found Klein still allocates/copies accumulation buffers even when
`accum_steps == 1`. Patch the default path to skip accumulation structures when
unused. This is smaller than the optimizer rewrite and should help all trainers
once shared.

Careful area in `train_klein_real.mojo`: the training loop around grad groups,
`accumulate_grad_group`, grad norm, clipping, and optimizer application.

### 4. Caption Dropout Default Path

Klein still builds a zero uncond text tensor even when caption dropout is off.
With `caption_dropout_prob == 0.0`, avoid the allocation entirely. Keep a
fail-loud/error path if dropout is enabled without usable uncond/cached embeds.

### 5. Turbo Loader Startup Memory

`serenitymojo/offload/turbo_loader.mojo` currently preloads all block bytes into
a pinned host `block_store` during open. This makes startup slow and system RAM
heavy. In the measured Klein run it used about 54% system memory while loading.

This is a shared infrastructure issue. Better direction:

- streaming pinned host window;
- optional persistent block-store policy;
- memory-budgeted loader planner;
- keep the fast async path but avoid pinning the entire model payload.

Raw transfer benchmark is already about `26 GB/s`, so the issue is policy and
runtime structure, not PCIe copy speed alone.

## Sampler Status

The user rejected fixed-resolution CLI file naming. Keep one adjustable sampler,
not `klein_sample_512_cli`, `klein_sample_1024_cli`, etc.

Current relevant files:

- `serenitymojo/sampling/klein_sample_cli.mojo`
- `serenitymojo/sampling/klein_sampler.mojo`
- `serenitymojo/sampling/base_sampler.mojo`
- `serenitymojo/sampling/dpmpp_2m.mojo`
- `serenitymojo/sampling/unipc.mojo`
- `serenitymojo/training/validation_sampler.mojo`
- `serenitymojo/training/sample_prompt_config.mojo`
- `serenitymojo/training/lora_save.mojo`

Sampling should read size, seed, steps, guidance, negative prompt, and cached cap
paths from shared sample prompt JSON. The trainer should save LoRA, sample, then
resume at cadence for full 2000-step runs.

LoRA checkpoint format should remain the generic PEFT/ai-toolkit safetensors
format emitted by `save_lora_peft`; the inference side and `/home/alex/ai-toolkit`
can use that shape/key convention.

Short smoke runs below `sample_every=500` should skip samples.

## Prompt JSON Contract

Docs:

- `serenitymojo/docs/TRAINER_SAMPLE_PROMPTS_AND_BOARD_2026-05-31.md`
- `serenitymojo/configs/sample_prompts.example.json`
- `serenitymojo/configs/klein9b_alina_samples.json`

Contract is `serenity.sample_prompts.v1`. Use shared `defaults` plus individual
`prompts`. This is meant to support image and video models, not just Klein.

Prompt precache for Klein:

```bash
pixi run mojo build -I . -Xlinker -lcuda -Xlinker -lm \
  serenitymojo/pipeline/klein9b_precache_sample_prompts.mojo \
  -o /tmp/klein9b_precache_sample_prompts

/tmp/klein9b_precache_sample_prompts \
  /home/alex/mojodiffusion/serenitymojo/configs/klein9b_alina_samples.json
```

## SerenityBoard And Display

Shared Mojo display file exists:

- `serenitymojo/training/progress_display.mojo`

Board wrapper exists:

- `serenitymojo/training/serenityboard.mojo`

Docs:

- `serenitymojo/docs/TRAINER_DISPLAY_CONTRACT_2026-05-31.md`
- `serenitymojo/docs/TRAINER_SAMPLE_PROMPTS_AND_BOARD_2026-05-31.md`

Expected terminal format should look like Rust:

```text
[Klein-lora] step 1613/2000 | epoch 14/17 | loss 0.5909 | grad_norm 0.1527 | 2.1s/step | elapsed 0:55:37 | ETA 0:13:20
```

Also include noising/noise speed in output so the user can see what the trainer
is doing.

Do not use `scripts/train_progress.py` as runtime UI. Python is allowed only for
parity/dev diagnostics.

## Launching And Tail

For a measured short profile after the current build:

```bash
timeout 240s /tmp/train_klein_real_save4 \
  /home/alex/mojodiffusion/serenitymojo/configs/klein9b.json \
  2 0 - nosample_profile
```

For a 50-step no-sample convergence/speed smoke after verifying no OOM:

```bash
timeout 3600s /tmp/train_klein_real_save4 \
  /home/alex/mojodiffusion/serenitymojo/configs/klein9b.json \
  50 0 - nosample
```

For a real run with a tail log, use `tee` so the user can watch:

```bash
mkdir -p /home/alex/mojodiffusion/output/alina_train/logs
/tmp/train_klein_real_save4 \
  /home/alex/mojodiffusion/serenitymojo/configs/klein9b.json \
  2000 0 - train \
  2>&1 | tee /home/alex/mojodiffusion/output/alina_train/logs/klein9b_2000_$(date +%Y%m%d_%H%M%S).log
```

Then tell the user:

```bash
tail -f /home/alex/mojodiffusion/output/alina_train/logs/<log-name>.log
```

GPU temp watch:

```bash
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5
```

## Subagent Audit Results

Three audit agents completed before this handoff:

- Locke audited standalone Flame Core parity modules.
- Descartes audited integration wiring.
- Mill audited the new trainers.

Main findings:

- Schedule dtype misuse was real and fixed.
- Noise Bernoulli mismatch was real and fixed.
- Multires token-space path needed fail-loud behavior and was fixed.
- Ernie RNG advancement bug was real and fixed.
- Klein LR logging used static LR and was fixed.
- Klein grad-accum default path still needs optimization.
- Klein caption-dropout default path still needs optimization.
- Prodigy parity is single-param scope; do not wire it as a default optimizer
  without broader tests.
- Anima trainer is not production: currently smoke/reduced, with hard-coded
  small latent dimensions and simplified pieces.
- Z-Image LoRA trainer is now full-depth for the 512 Alina path, self-contained
  Mojo for staging/prepare/train, and speed-fixed. Remaining gap is validation
  sampling wired through the Mojo Z-Image generator with trained LoRA.
- SDXL trainer is not production: current crop/latent path is reduced.

## Model Readiness

### Klein

Closest to real training. Current smoke got `3.1s/step` with no offload
fallbacks. Need:

1. verify/revert the `DBL_SAVE_TAIL=4` experiment;
2. reduce default grad/copy overhead;
3. run a 50-step smoke;
4. run a real 2000-step sample/save/resume cadence;
5. confirm loss in target range and real sample change after enough steps.

### Z-Image

Use OneTrainer as the parity reference only. Do not modify OneTrainer, do not
use MGDS, and do not use Rust/Python/EDv2/OneTrainer caches in production. The
trainer, stager, text encoder, VAE encoder, cache writer, and LoRA saver are
Mojo-owned.

Current production LoRA path:

- `serenitymojo/image/decode.mojo` adds pure-Mojo PNG/JPEG loading through
  libpng/libturbojpeg FFI.
- `serenitymojo/pipeline/zimage_stage_alina.mojo` stages raw
  `/home/alex/datasets/AlinaAignatova` images/captions into
  `output/alina_zimage_stage` with OneTrainer-style 512 buckets quantized to 64.
  Current buckets are `576x448` (`72x56` latents) and `704x384` (`88x48`
  latents).
- `serenitymojo/pipeline/zimage_prepare.mojo` prepares
  `output/alina_zimage_cache` from the staged Mojo image tensors using the Mojo
  Qwen3 text encoder and Mojo Z-Image VAE encoder. It dispatches all current
  production cases: `72x56/cap224`, `72x56/cap256`, `88x48/cap224`, and
  `88x48/cap256`.
- `serenitymojo/training/train_zimage_real.mojo` uses full-depth
  `MAIN_DEPTH=30`, frozen noise/context refiners, BF16/BP16 base weights, and
  main-layer attention/feed-forward LoRA only, matching the OneTrainer filter
  that excludes refiners.
- LoRA saving stays PEFT/ai-toolkit-compatible for `/home/alex/ai-toolkit` and
  Serenity inference; do not switch Z-Image LoRA output to a private format.

Do not try full-F32 Z-Image. OneTrainer does not train this model in full F32,
and a full-F32 base/model load will OOM on the local 24 GB GPU. Keep large base
weights BF16/BP16/offloaded. F32 is allowed only for small reductions,
transients, LoRA/Adam masters, and compatibility carriers.

Important parity fixes already landed:

- Z-Image VAE scaling is `(latent - 0.1159) * 0.3611`, not the Klein/BN path.
- Logit-normal timestep policy uses OneTrainer's Z-Image baseline settings:
  `timestep_shift=1.0`, dynamic shift disabled.
- Noise generation now uses the full 53-bit uniform path. The old mask biased
  the target mean and caused unstable loss.
- Final-layer modulation now passes raw scale into `modulate()`. Do not apply
  `1 + scale` twice.
- Learned caption/image pad tokens and position rows are used for padded rows;
  padded image rows are excluded from loss.

Speed status after the `nsys` pass:

- Old full-depth diagnostic: about `100s/step`; `nsys` showed about `35s` in
  full RMSNorm backward weight-gradient reductions for frozen norms.
- First production speed fix kept LoRA on device and used `rms_norm_backward_dx`
  for frozen norms, dropping warm cadence to about `4.0-4.15s/step`.
- Second production speed fix keeps the 30-layer main stack tensor-resident
  across forward and backward recompute. Do not reintroduce per-main-block
  `to_host()`/`Tensor.from_host()` boundaries.
- 100-step run: `output/logs/zimage_train_100_speed2_tensor_main_2026-06-02.log`
- Result: loss `0.47321588 -> 0.35350168`, `nonfinite=0`, final step
  `1.993s`, warm cadence about `1.96-2.00s/step` batch 1.
- Bucket proof on the speed path: cap256 steps `27`, `30`, `44`, `78`, `81`,
  and `95` stayed finite; the `88x48/cap224` singleton at step `51` ran at
  `1.981s` with loss `0.46069825`.
- Saved LoRA: `output/alina_zimage/zimage_lora_step100.safetensors`

Remaining Z-Image gap: validation sampling is not yet wired into
`train_zimage_real.mojo`. Add it through the Mojo Z-Image generator/LoRA path,
not Python or Rust, before calling the trainer sample cadence complete.

### Anima

Use `/home/alex/OneTrainer-anima-ref`. Current Mojo trainer is smoke-level only.
Do not present it as production.

### SDXL / Ernie

Have compile/smoke progress but are not the active immediate goal.

## Do Not Forget

- User is frustrated by Python UI and fixed sampler files. Keep runtime UX Mojo
  and config-driven.
- Do not claim Klein convergence until a real longer run proves it.
- Do not judge samples from 50-step LoRA; user explicitly called that useless.
- If the trainer OOMs during sampling, fix block swapping/sampler offload rather
  than lowering sample resolution behind the user's back.
- Work on core pieces where possible: offload planner, activation memory,
  ring/slab allocator, fused optimizer/global norm, shared display, shared
  prompt/sampler contracts.
- Document new trainer behavior in `serenitymojo/docs/`.
