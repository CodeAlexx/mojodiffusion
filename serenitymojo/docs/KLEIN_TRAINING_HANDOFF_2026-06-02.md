# Klein 9B Training Handoff - 2026-06-02

This is the restart handoff for the Mojo trainer work. The active user goal is a
valid Klein 9B LoRA training run with normal sampling and convergence, then the
same training/runtime foundation applied to Z-Image and Anima.

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
  trainer UI, sampler, and final implementation must be Mojo. Rust side stays
  pure Rust.
- Prefer shared/core speed work that benefits all models, not Klein-only hacks.

## Reference Trees

- Klein, SDXL, Z-Image, Ernie training reference: `/home/alex/OneTrainer`
- Anima training reference: `/home/alex/OneTrainer-anima-ref`
- Rust/EriDiffusion and Flame Core are bug-fix/speed cheat sheets, especially
  for offload, allocator, optimizer, sampler, and convergence behavior.

Do not treat Rust as the preset authority when OneTrainer has the model preset;
use Rust as proof of solved implementation issues.

## Current Repo State

Workspace: `/home/alex/mojodiffusion`

There is a large dirty worktree with many user and generated changes. Do not
revert unrelated files. Important touched/added areas include:

- `serenitymojo/training/train_klein_real.mojo`
- `serenitymojo/models/klein/klein_stack_lora.mojo`
- `serenitymojo/training/progress_display.mojo`
- `serenitymojo/training/serenityboard.mojo`
- `serenitymojo/training/sample_prompt_config.mojo`
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

Z-Image note from user: use `timestep_shift=1.8` for 512 training and `3.0` for
1024 or above. Verify against OneTrainer before final wiring, but the user
explicitly expects this policy.

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

Sampling should read size, seed, steps, guidance, negative prompt, and cached cap
paths from shared sample prompt JSON. The trainer should save LoRA, sample, then
resume at cadence for full 2000-step runs.

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
- Z-Image trainer is not production: current depth/memory path is reduced and
  needs BF16/offload/full-depth work.
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

Next after Klein unless user chooses Anima first. Use OneTrainer as training
reference. Respect user shift policy: `1.8` at 512, `3.0` at 1024+ unless
verified otherwise. Needs memory/offload and full-depth readiness.

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
