# OneTrainer Core Port Status - 2026-06-06

This is the short core/runtime companion to the longer port audit. It records
what changed in mojodiffusion shared code for the OneTrainer-style trainer and
what those changes do not prove yet.

## Shared Runtime Changes

- `serenitymojo/io/train_config_reader.mojo` and
  `serenitymojo/training/train_config.mojo` parse typed OneTrainer-style fields
  for paths, model type, training method, dtype, optimizer, cache/output,
  workspace, sample/save/backup, validation, EMA, masked/prior flags, and
  checkpoint/offload policy. Legacy bool-shaped checkpoint/offload fields are
  migrated into typed policy values.
- `serenitymojo/training/onetrainer_train_loop_policy.mojo` is the common real
  loop validator for Qwen, Ernie, Anima, SD3.5, SDXL, Flux.1, Klein/Flux2
  class, Chroma, and Z-Image. It keeps config validation before CUDA setup,
  makes `only_cache` exit before `DeviceContext()`, applies save/sample cadence,
  resolves output/state paths, and fails loudly for unsupported product modes.
- `serenitymojo/training/optimizer_dispatch.mojo`,
  `serenitymojo/training/optimizer_fixed_update_smoke.mojo`, and
  `serenitymojo/training/lr_schedule.mojo` pin the current OneTrainer optimizer
  and LR surface: ADAMW/default, ADAFACTOR dispatch coverage, unsupported
  optimizer fail-loud mapping, cosine/linear/REX schedules, AdamW state/resume,
  BF16 parameter storage with F32 Adam moments, and PyTorch/OneTrainer AdamW
  order where decoupled weight decay is applied before the adaptive Adam
  subtraction.
- `serenitymojo/training/onetrainer_product_run.mojo`,
  `onetrainer_train_dry_run.mojo`, `onetrainer_lifecycle.mojo`,
  `onetrainer_resume_contract.mojo`, and `onetrainer_cache_preflight.mojo`
  provide the no-CUDA product runner, lifecycle, resume, and cache preflight
  contracts. They are product wiring gates, not real model-loop parity.
- `serenitymojo/training/full_finetune_contract.mojo` and
  `full_finetune_save.mojo` provide full-weight checkpoint/save sidecar
  scaffolding. The product/lifecycle path now creates a per-target blocker
  preflight for full finetune requests, but real model-specific full finetune
  loops are still unsupported. `scripts/check_full_finetune_contracts.py
  --target-model klein --write-readiness <path>` and
  `--target-model zimage --write-readiness <path>` each report `3` blockers:
  one LoRA-only loop blocker plus real product-loop full-finetune parity and
  parity artifacts. Klein and Z-Image now have inventory plus payload
  save/load/manifest smokes:
  `klein_full_finetune_inventory_smoke` validates `201` keys,
  `klein_full_finetune_checkpoint_smoke` saves/loads a synthetic `201`-tensor
  BF16 payload, `zimage_full_finetune_inventory_smoke` validates `521` keys,
  and `zimage_full_finetune_checkpoint_smoke` saves/loads a synthetic
  `521`-tensor BF16 payload. `klein_full_finetune_state_smoke` binds the
  `201`-tensor Klein manifest to `604` TrainState sidecar tensors, and
  `zimage_full_finetune_state_smoke` binds the `521`-tensor Z-Image manifest
  to the same optimizer key order. `scripts/check_full_finetune_inventory_keys.py
  --strict` verifies the inventories against the local checkpoint header/index
  without CUDA.
  `--write-template-dir` can write implementation skeletons for selected
  targets, but those files are not readiness evidence.

## Upstream OneTrainer PR Watch

- `Nerogar/OneTrainer#1509` was checked on 2026-06-06 and is still open/draft.
  It changes the offload/checkpoint direction from global
  `gradient_checkpointing` / `layer_offload_fraction` fields to per-model-part
  `gradient_checkpointing`, `activation_offloading`, `offload_fraction`, and
  `load_on_demand`. It also changes component parking toward `release()`
  lifecycle semantics. The local scalar conductor policy is a legacy bridge,
  not the final upstream parity shape.
- `Nerogar/OneTrainer#1344` was checked on 2026-06-06 and is still open. It
  adds advanced optimizer behavior outside the current AdamW/Adafactor gates:
  scaled/spectral optimizer behavior, centered weight-decay anchors, factored
  second moment, state precision selection, SinkSGD, signed optimizer changes,
  and OrthoGrad modes. It also highlights BF16 optimizer-state resume risk; Mojo
  resume must preserve expected state dtype and must not silently cast BF16
  state to F32 at a storage boundary.

## Dtype And Loader Changes

- The safetensors reader path remains mmap/header based and is not double
  loading checkpoint payloads.
- The official trainer path now prefers `Tensor.from_view` at cache/checkpoint
  boundaries so BF16/F16 checkpoint and cache tensors stay BF16/F16 on device.
- Mixed-dtype core ops were widened where needed for frozen checkpoint weights:
  `linear`/`linear_backward`, `conv2d`/`conv2d_backward`,
  `rms_norm`/`group_norm_backward`, RoPE forward/backward, and gate residual
  backward support F32 compute or activation grads with BF16/F16 storage where
  the model path requires it.
- The train-loop cache/preflight guard now separates unsafe F32 cache-boundary
  markers from broader host-F32 carriers. The narrow cache-boundary scan now
  passes for Qwen, Ernie, Anima, Klein, Z-Image, Chroma, Flux.1, SD3.5, and
  SDXL. All nine loops now stage selected cache readbacks through the cached
  storage dtype before host step math; SDXL reuploads cropped/noisy latents
  using the cached latent dtype. Remaining host
  `Float32` carriers in training/offload/sample surfaces are still blockers for
  strict dtype-boundary parity. F32 is allowed for compute internals, schedules,
  AdamW moments, and parity/debug readback, not as a production tensor boundary.
  `scripts/check_klein_chroma_flux_sd35_train_readiness.py` is the no-GPU
  readiness gate for the remaining Klein/Chroma/Flux.1/SD3.5 sampler
  evidence/runtime-wiring and full-finetune blockers.

## Model-Surface Changes

- Raw OneTrainer LoRA key save/resume gates exist for Qwen, Ernie, Anima,
  SDXL, Flux.1, Klein/Flux2 class, and Chroma. Unsupported selected LoRA
  surfaces are explicit fail-loud gaps.
- Qwen, Ernie, and Anima now have Mojo artifact gates that open the real
  OneTrainer step/adapters/meta files and validate zero-lr optimizer
  state-initialization. These do not prove nonzero AdamW update parity.
- Chroma and SDXL now have Mojo artifact gates that open the real update-bearing
  OneTrainer step/adapters/meta files and compare sampled
  `adapter_post -> adapter_after` deltas. These prove update-delta artifact
  consumption only.
- Klein/Flux2 class has split-slot LoRA carrier alignment, gradient/state
  evidence, and a bounded synthetic positive-lr AdamW math gate from the
  captured gradients. `scripts/check_klein_loss_replay.py --strict` also
  replays the dumped `output.predicted`/`output.target` loss bridge on CPU:
  default MSE loss recomputes as `0.12243739306158122` against dumped
  `0.12243738770484924` and produces finite `d_loss = (2/N) * diff` stats.
  `scripts/check_klein_adamw_state_init_replay.py --strict` streams all six
  adapter phases, validates the zero-lr optimizer metadata (`288` entries /
  `864` state tensors / `87032096` state elements), confirms `27262171`
  nonzero post-clip gradient elements with zero adapter delta, and projects the
  Mojo BF16 moment state (`exp_avg_l2=0.0005975040211676537`,
  `exp_avg_sq_l2=8.403631552140676e-11`).
  `serenitymojo/models/klein/parity/klein_lora_adamw_state_init_smoke.mojo`
  exercises the real model-level `klein_lora_adamw_step` helper on a bounded
  synthetic `1` double / `1` single LoRA set: zero-lr params stay unchanged,
  moment error is `0.0`, and a bounded positive-lr follow-up changes `644` BF16
  adapter values.
  `serenitymojo/models/klein/parity/klein_train_ref_adamw_state_init_replay.mojo`
  opens the real OneTrainer adapter dump, loads all `144` LoRA modules / `288`
  A/B tensors plus post-clip grads, calls `klein_lora_adamw_step` at `lr=0`,
  confirms unchanged params, reports `max_moment_err=0.0`,
  `nonzero_moments=27262171`, and a bounded positive-lr follow-up changing
  `27262275` BF16 adapter values.
  `scripts/check_klein_adamw_positive_lr_oracle.py --strict` is only a CPU
  host-list optimizer support oracle for the same captured gradients at
  synthetic positive lr with BF16 moment projection and deterministic
  stochastic BF16 param rounding (`changed=27262275`,
  `l2=0.014876430049126985`, `max_abs=6.103515625e-05`); it is not CUDA/GPU
  parity.
  This is accepted loss/state-init support evidence only; Klein still needs
  accepted full predict/backward/AdamW replay and a real later update-bearing
  OneTrainer dump.
- Klein weight streaming is distinct from OneTrainer CPU_OFFLOADED activation
  and layer offload. `TurboPlannedLoader` block streaming is present in the
  train loop, but it is not activation/layer CPU offload and not
  bounded-CUDA offload/checkpoint backward replay parity.
- `check_klein_offload_checkpoint_contract.py` reads OneTrainer's
  `GradientCheckpointingMethod`, `OffloadCheckpointLayer`,
  `LayerOffloadConductor`, `offload_quantized`, and Flux2 low-VRAM presets to
  prove the OT target: `CPU_OFFLOADED` is checkpointing plus activation/layer
  movement, not file-level weight streaming.
- Safe offload scaffolding smokes pass without full-model CUDA:
  `offload_checkpoint_config_smoke`, `conductor_policy_smoke`, `plan_smoke`,
  `planned_loader_smoke`, `residency_smoke`, and `turbo_slots_smoke`. They
  prove typed config derivation, OneTrainer scalar conductor gates with
  `Float64` offload fractions, block-plan topology, plan-aware loader API
  typechecking, residency metadata, and slot-planning logic only; they do not
  prove model replay, activation offload, or GPU parity.
- `klein_activation_tape_plan_smoke` pins the current Klein 9B LoRA boundary
  activation target at `432,013,312` BF16 bytes. The minimal live backward tape
  excludes unused `img_in_act` / `txt_in_act` and is `419,430,400` BF16 bytes.
  F32 would double those targets, so the CPU_OFFLOADED branch uses
  raw/dtype-preserving activation offload instead of `Tensor.to_host()`.
- `serenitymojo/models/klein/activation_tape.mojo` now provides the narrow
  LoRA backward offload bridge: block inputs plus `img_out` / `ln_img_out` are
  carried as `HostOffload` raw bytes and restored without an F32 host-list
  boundary. `klein_activation_tape_offload_smoke` passes on a tiny BF16 CUDA
  roundtrip and rejects an F32 storage-dtype claim. This is bridge evidence for
  the later bounded backward replay, not accepted `CPU_OFFLOADED` parity.
- `klein_stack_lora_offloaded_tape_parity` now executes that bounded backward
  replay on the small Klein LoRA stack. It uses the current
  `build_klein_lora_set` production slot layout, offloads `2560` tiny-case raw
  activation bytes, restores the tape, and matches resident backward for
  input-token grads, modulation grads, and all LoRA adapter grads with
  `max_abs=0.0`. This proves the tape bridge can feed the existing recompute
  backward, but it is not the full train-ref dump replay or product
  `CPU_OFFLOADED` acceptance.
- On 2026-06-06 the bounded Klein GPU gates passed on the local RTX 3090 Ti:
  `/tmp/klein_activation_tape_offload_smoke`,
  `/tmp/klein_stack_lora_offloaded_tape_parity`,
  `/tmp/klein_lora_adamw_state_init_smoke`, and
  `/tmp/klein_train_ref_adamw_state_init_replay`. The resident
  `/tmp/klein_train_ref_forward_replay` opened the real train-ref dump plus the
  9B checkpoint and then OOMed, so it must not be treated as a usable 24GB
  acceptance path. The remaining target is a bounded train-ref
  CPU_OFFLOADED/checkpoint replay.
- `train_klein_real.mojo` now accepts OneTrainer `CPU_OFFLOADED` policy and
  selects the direct offloaded-tape forward/backward branch when
  `enable_activation_offloading=true`. The cache preload path preserves latent
  and text-token storage dtype instead of forcing F32, and the caption-dropout
  zero text embedding follows the cached text-token dtype. This is required
  product plumbing, not accepted train-ref/runtime parity.
- Flame autograd v2 reinforces the target shape for Mojodiffusion: saved tensor
  records should be releasable, block backward should recompute from saved
  block inputs, activation saving should not use `Tensor.clone(ctx)`, and async
  host-device movement should come after the synchronous correctness branch.
- Klein cap-cache sampler/train-loop handoffs now preserve cached cap/text
  embedding storage dtype instead of forcing an F32 boundary cast. This fixes
  that narrow boundary only; full Klein dtype parity still requires the broader
  train/offload/sample dtype guards.
- Klein sampler bring-up exposed three valid mixed-dtype helper boundaries:
  `layer_norm`, `modulate`, and `residual_gate` now cast small affine/modulation
  tensors to the activation storage dtype internally instead of rejecting BF16
  activations with F32 modulation vectors or forcing an F32 activation boundary.
  The output dtype remains the activation dtype; F32 is still limited to the
  reduction/elementwise compute inside the kernels.
- Klein's real trainer validates sample cap-cache headers before
  `DeviceContext()` when runtime sampling is enabled. The preflight checks
  magic, BF16 dtype, 9B text shape, and exact byte size, and
  `serenitymojo/io/cap_cache_header_smoke.mojo` covers the no-CUDA valid/reject
  cases. This is a cache/sampler readiness guard, not sampler parity.
- Klein sampler wrappers now dispatch both 9B `H=32` and 4B `H=24`
  specializations from config. This is dispatch coverage, not accepted denoise,
  VAE/image, speed, or full training parity.
- The standalone Klein sampler now preflights checkpoint, VAE, LoRA,
  initial-noise sidecar, and cap-cache headers before creating `DeviceContext`.
  CUDA runs must execute the sampler binary outside the sandbox; otherwise
  `DeviceContext()` fails at NVML initialization even for identical binaries.
  On 2026-06-06 `/tmp/klein_sample_cli` loaded
  `/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors` and wrote real
  512x512 RGB PNG smoke artifacts. Guided `cfg=4.0` one-step denoise measured
  `24.5s/step`; the fast validation preset uses `cfg=1.0` and measured
  `3.1s/step` denoise with about `7s` total command wall time including load
  and VAE decode. These are product wiring/speed smokes, not accepted sampler
  parity.
- A bounded resume20 continuation also completed from
  `/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors` using
  `serenitymojo/configs/klein9b_cpu_offloaded_resume20_smoke.json`. Steps
  `11-20` reported losses `0.84224933`, `1.310655`, `0.7975194`,
  `1.3193466`, `0.6252446`, `0.70147544`, `0.712633`, `0.24021716`,
  `0.6145748`, and `0.89166784`; step time stayed around `13.5-14.8s`.
  Warmed timing was forward `2.52-2.68s`, backward `3.92-4.09s`, and optimizer
  `6.89-7.16s`, making the current Klein product path optimizer-bound. The
  resume20 LoRA passed the product artifact guard with `432/432` BF16 tensors
  and a state sidecar of `288` BF16 adapter tensors plus `576` F32 AdamW
  moments. The fast sampler loaded the resume20 LoRA and wrote
  `output/alina_train/klein_lora_resume20_fast512_cfg1.png`, a 512x512 RGB
  smoke PNG at `3.1s/step`.
- The default Klein sample cap-cache is now complete for `alina_garden` and
  `alina_evening`, positive and negative. `check_klein_cap_cache_contract.py`
  validates all four generated files as BF16 `[1,512,12288]`, and the sampler
  parity contract now removes cap-cache readiness from the known blocker list.
- Klein validation sampling now honors `lora_multiplier`. The sampler scales
  the live adapter contribution scalar before device upload and leaves BF16 A/B
  storage plus AdamW state unchanged. A post-change fast resume20 sampler run
  still measured `3.1s/step` and wrote a valid 512x512 RGB PNG.
- Klein sampler parity wiring now has an explicit
  `klein_sample_with_initial_noise` entry and CLI sidecar argument for
  OT-equivalent post-patch/post-pack initial-noise tensors. The sidecar dtype is
  preserved, and the default product path still uses BF16 Mojo randn. This is
  trajectory-replay wiring only; it does not accept sampler parity without
  matched OneTrainer noise/latent trajectory artifacts.
- `scripts/check_klein_conditioning_template_contract.py` passes for the
  configured Klein sample prompts: it compares OneTrainer's Qwen3
  `apply_chat_template(add_generation_prompt=True, enable_thinking=False)` path,
  the local precache template, and padded token ids from the local tokenizer
  files without loading model weights or creating CUDA work.
- Z-Image now has a no-CUDA strict-speed readiness report path:
  `scripts/check_zimage_sampler_contract.py --write-speed-readiness <path>`
  records the exact missing paired OneTrainer/Mojo sampler identity, timing,
  VRAM, and artifact-path fields. `--write-strict-speed-template <path>` writes
  the corresponding skeleton speed manifest with placeholder values.
  `--strict-speed` now passes as comparable evidence against
  `/home/alex/onetrainer-mojo/parity/zimage_sampler_speed.json`, but speed
  parity is not accepted: OneTrainer records `2.144007647s/step` denoise and
  `14340 MiB` peak VRAM, while the paired Mojo record is `5.206695175s/step`
  denoise and `22238 MiB` peak VRAM. The current Mojo-only post-cleanup
  supervisor artifact `/tmp/zimage_speed_1step_moddev_speed.json` records
  `4.302052192s/step`, text encode `10.645762139s`, VAE `1.487801541s`, and
  peak `22111 MiB`; it is not paired OT/Mojo parity evidence.
- `serenitymojo/training/onetrainer_product_run_smoke.mojo` now checks the
  Z-Image named preset control plane. `zimage_lora_16gb` resolves to
  `train_zimage_real`, binds the Z-Image cache preflight, requires the sample
  file, and enforces step-500 save-before-sample cadence. The train loop now
  writes a `serenity.zimage.sample_request.v1` manifest after saving LoRA/state
  at sample cadence. This deliberately queues standalone `zimage_generate.mojo`
  work for after trainer memory is released instead of co-residing the trainer,
  text encoder, DiT, and VAE. The request includes a `result_manifest` path.
  `zimage_generate.mojo` now accepts that manifest with `--request`, validates
  LoRA/state/sample paths before CUDA setup, saves the PNG, and writes a
  `serenity.zimage.sample_result.v1` JSON with Mojo-side text/denoise/VAE
  timings plus explicit non-parity flags. `zimage_sample_supervisor.mojo` is the
  process-separated Mojo runner for queued requests, while
  `scripts/run_zimage_sample_requests.py` is support tooling for dry-run
  validation, external VRAM polling, and Mojo-side speed metadata output. It is
  product entrypoint/cadence evidence only, not sampled-output parity, and
  `train_zimage_real` is still LoRA-only.
  `zimage_generate.mojo --trace-denoise` is now an opt-in first-step profiler:
  it is disabled by default and syncs only for trace timing. The latest trace
  shows Z-Image CFG is still dominated by two serial main-stack passes
  (`main_cond` about `1.98s`, `main_uncond` about `1.91s`); OneTrainer batches
  those branches when CFG is enabled.

## Current Required Gates

- `python3 scripts/check_onetrainer_train_math_contract.py --strict-product`
  must pass before any product train-math claim; it includes the static AdamW
  order guard for `optim.mojo`, fused AdamW, and `train_step.mojo`.
- `python3 scripts/check_zero_lr_mojo_state_init_consumers.py --require-mojo-state-init`
- `python3 scripts/check_qwen_adapter_update_replay.py --require-update-bearing`,
  `python3 scripts/check_ernie_adapter_update_replay.py --require-update-bearing`,
  and `python3 scripts/check_anima_adapter_update_replay.py --require-update-bearing`
  must still fail with exit `2` until step001 positive-lr dumps exist. Use each
  wrapper's `--write-readiness /tmp/<model>_update_bearing_readiness.json` to
  emit a template-only checklist for the required later dump metadata.
- `python3 scripts/check_chroma_sdxl_mojo_update_consumers.py`
- `python3 scripts/check_chroma_sdxl_mojo_update_consumers.py --require-mojo-parity`
  must still fail with exit `2` until full backward/AdamW parity exists.
- `python3 scripts/check_klein_adapter_grad_update_replay.py --require-synthetic-positive-lr`
  must pass as bounded optimizer-math support evidence.
- `python3 scripts/check_klein_adapter_grad_update_replay.py --require-update-bearing`
  must still fail with exit `2` until a later Klein OneTrainer dump has
  `lr_before > 0` and nonzero adapter deltas.
- `python3 scripts/check_klein_adapter_grad_update_replay.py --require-mojo-parity`
  must still fail with exit `2` until an accepted Mojo Klein
  predict/backward/AdamW replay consumes the train-ref artifacts.
- `python3 scripts/check_klein_loss_replay.py --strict` must pass before using
  the captured Klein loss bridge as support evidence. It is CPU replay of dumped
  tensors, not transformer forward, backward, AdamW, or sampler parity.
- `python3 scripts/check_klein_adamw_state_init_replay.py --strict` must pass
  before using the captured Klein zero-lr optimizer state-init projection as
  support evidence. It does not execute Mojo backward/AdamW or prove nonzero
  update parity.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/models/klein/parity/klein_lora_adamw_state_init_smoke.mojo -o /tmp/klein_lora_adamw_state_init_smoke`
  plus `/tmp/klein_lora_adamw_state_init_smoke` must pass before using the
  bounded Mojo AdamW helper path as support evidence. It must not be used to
  satisfy `--require-mojo-parity`, `--require-update-bearing`, or
  `check_klein_offload_checkpoint_contract.py --strict`.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/models/klein/parity/klein_train_ref_adamw_state_init_replay.mojo -o /tmp/klein_train_ref_adamw_state_init_replay`
  plus `/tmp/klein_train_ref_adamw_state_init_replay` must pass before using
  the all-train-ref-adapter Mojo AdamW state-init path as support evidence. It
  must not be used to satisfy `--require-mojo-parity`, `--require-update-bearing`,
  or `check_klein_offload_checkpoint_contract.py --strict`.
- `python3 scripts/check_klein_adamw_positive_lr_oracle.py --strict` must pass
  before using the synthetic positive-lr CPU host-list optimizer oracle as
  support evidence. It must not be used to satisfy CUDA/GPU parity,
  `--require-mojo-parity`,
  `--require-update-bearing`, or
  `check_klein_offload_checkpoint_contract.py --strict`.
- `python3 scripts/check_klein_offload_checkpoint_contract.py --strict`
  must still fail with exit `2` until accepted train-ref low-memory
  offload/checkpoint-backward replay exists for Klein.
- `python3 scripts/check_klein_sampler_parity_contract.py --strict`
  must still fail with exit `2` until accepted Klein sampler parity and
  speed/VRAM evidence exists. Current trajectory evidence still needs matched
  OneTrainer post-patch/post-pack initial-noise and latent artifacts.
- `python3 scripts/check_klein_sampler_artifact_manifest.py --strict`
  must still fail with exit `2` until `/home/alex/onetrainer-mojo/parity/klein_sampler_parity_manifest.json`
  records paired OneTrainer/Mojo raw/post-patch/post-pack noise, scheduler
  timesteps, latent trajectory, final packed and VAE pre-decode latents, PNG,
  denoise/VAE timing, and peak VRAM evidence.
- `python3 scripts/check_flux1_sampler_artifact_manifest.py --strict`
  must still fail with exit `2` until `/home/alex/onetrainer-mojo/parity/flux1_sampler_parity_manifest.json`
  records paired OneTrainer/Mojo text conditioning, raw/packed noise,
  FlowMatch sigmas/timesteps/step trace, packed latent trajectory, final
  latent, VAE tensor/PNG, per-stage timing, peak VRAM, and numeric comparisons.
- `python3 scripts/check_chroma_sampler_artifact_manifest.py --strict`
  must still fail with exit `2` until `/home/alex/onetrainer-mojo/parity/chroma_sampler_parity_manifest.json`
  records paired OneTrainer/Mojo prompt/negative CFG branches, attention masks,
  raw/packed noise, FlowMatch schedule, CFG prediction trajectory, final latent,
  VAE tensor/PNG, timing, VRAM, and numeric comparisons.
- `python3 scripts/check_klein_initial_noise_sidecar_contract.py --strict`
  must pass before using the Klein sampler sidecar replay path; it is a static
  hook/dtype guard, not trajectory or image parity.
- `python3 scripts/check_klein_cap_cache_contract.py --strict`
  now passes for the configured Klein validation/sample cap files as BF16
  cap-cache tensors with the expected 9B shape and byte size. This remains a
  pre-CUDA header/file-size guard, not denoise, VAE/image, speed, or GPU parity.
- `pixi run mojo build -I . serenitymojo/io/cap_cache_header_smoke.mojo -o /tmp/cap_cache_header_smoke`
  plus `/tmp/cap_cache_header_smoke` must pass before relying on the Klein
  train-loop sample cap preflight. This gate is no-CUDA and does not accept
  denoise, VAE/image, speed, or GPU parity.
- Model artifact smokes under `serenitymojo/models/*/parity/*train_ref*_smoke.mojo`
  prove artifact consumption, not production training readiness.

## Still Not Production Ready

- No target model has complete OneTrainer parity across loss, gradients, AdamW
  update, sampler image output, speed/VRAM, resume, and dtype boundaries.
- Chroma and SDXL need full backward replay from the matching step dump or new
  step-with-grads oracles before their update-bearing dumps can become full
  AdamW parity.
- Qwen, Ernie, Anima, and Klein need later OneTrainer dumps with
  `lr_before > 0` for nonzero update parity. Klein's synthetic positive-lr
  AdamW gate is not a replacement for that dump.
- Full finetune, product resume continuation, and sampler speed/image parity
  remain per-model work.
