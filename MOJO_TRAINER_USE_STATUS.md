# Mojo Trainer Use Status

Date: 2026-06-06

## Bottom Line

This is not ready for real training yet.

It is ready as a structured OneTrainer-to-Mojo port harness: configs, model
surfaces, LoRA key contracts, save/resume scaffolds, sampler contracts, and
many OneTrainer artifact gates exist. The missing work is still the part that
makes the trainer trustworthy for use: per-model numeric replay of loss,
gradients, optimizer update, sampler output/speed, resume behavior, and strict
dtype-boundary cleanup.

Core/runtime change summary:
`serenitymojo/docs/ONETRAINER_CORE_PORT_STATUS_2026-06-06.md`.

Upstream OneTrainer PRs checked on 2026-06-06:

- `Nerogar/OneTrainer#1509` is open/draft. It moves offload/checkpoint policy
  toward per-model-part `gradient_checkpointing`, `activation_offloading`,
  `offload_fraction`, and `load_on_demand`, plus `release()` lifecycle
  behavior. The current global `layer_offload_fraction` policy is only a
  legacy bridge for the checked OneTrainer checkout and low-VRAM preset math.
- `Nerogar/OneTrainer#1344` is open. It expands advanced optimizers and exposes
  state-precision/resume dtype risk. Current AdamW/Adafactor gates do not cover
  scaled optimizers, centered weight decay, factored second moment, SinkSGD,
  OrthoGrad modes, or BF16 optimizer-state resume parity.

## What Looks Ready

- OneTrainer-style config and preset plumbing exists for the target model set.
- Shared train-loop policy is wired into Qwen, Ernie, Anima, SD3.5, SDXL,
  Flux.1 dev, Klein/Flux2 class, Chroma, and Z-Image.
- LoRA key/save/resume contracts exist for the main model families and fail
  loud where a selected OneTrainer surface is not implemented.
- Qwen, Ernie, and Anima consume real OneTrainer step/adapters/meta artifacts
  for zero-lr optimizer state initialization.
- Chroma and SDXL consume real update-bearing OneTrainer artifacts and compare
  sampled `adapter_post -> adapter_after` update deltas.
- The shared train-math guard now pins PyTorch/OneTrainer AdamW order:
  decoupled weight decay is applied to the parameter before the adaptive Adam
  subtraction in `optim.mojo`, fused AdamW, and the host-list train-step helper.
- Sampler contracts and helper smokes exist for the model families, but most
  are scalar/helper gates rather than full denoise/decode/image parity.
- Klein 9B has bounded CUDA `CPU_OFFLOADED` product-smoke coverage through a
  resume20 LoRA artifact, with real losses, BF16 LoRA/state save checks, and a
  standalone 512px sampler smoke. This is usable evidence for the low-memory
  branch shape, not a full OneTrainer train/sampler parity claim.
- Z-Image now has product-control wiring for the `zimage_lora_16gb` named
  preset: cache preflight, sample-file policy, save-before-sample, and step
  cadence are checked from the OneTrainer-style entrypoint. The real train loop
  now queues a split-process sampler request manifest after saving LoRA/state at
  sample cadence; it does not run the 1024 sampler in-process while training
  memory is resident. The standalone generator now accepts the manifest through
  `--request`, validates LoRA/state/sample paths before CUDA setup, and then
  uses the existing LoRA overlay sampler. The request now carries
  `result_manifest`, and the generator writes a
  `serenity.zimage.sample_result.v1` JSON with Mojo-side text/denoise/VAE
  timings plus explicit `accepted_sampler_parity=false` and
  `accepted_speed_parity=false`. `zimage_sample_supervisor.mojo` is the
  process-separated runner for those requests, and
  `scripts/run_zimage_sample_requests.py` can dry-run or execute the request
  while polling external VRAM and writing Mojo-side speed metadata.
  `scripts/check_zimage_sample_request_contract.py --strict` verifies this
  source/request/supervisor contract without CUDA.

## What Is Smoke-Only

- Qwen, Ernie, and Anima prove artifact/state-init consumption only. Their
  captured dumps have `lr_before=0`, so they do not prove nonzero AdamW update
  parity. Their adapter readiness wrappers can write template-only JSON for the
  required later positive-lr dump with `--write-readiness /tmp/<model>_update_bearing_readiness.json`;
  that file is a capture checklist, not parity evidence.
- Chroma and SDXL prove update-delta artifact consumption only. Their current
  adapter dumps do not include per-adapter gradient phases, so full Mojo AdamW
  parity still needs backward replay from the matching step dump or a new
  step-with-grads oracle.
- Klein has gradient evidence, state-init evidence, and bounded synthetic
  positive-lr AdamW math from the captured gradients, but its current captured
  step still has unchanged weights because `lr_before=0`.
- Full finetune is not runnable in product loops yet. Product loops are still
  LoRA-only for real full-weight training. Klein and Z-Image now have
  model-specific full-weight inventory plus payload save/load/manifest smokes:
  Klein saves and reloads a synthetic `201`-tensor BF16 payload, and Z-Image
  saves and reloads a synthetic `521`-tensor BF16 payload. They also have
  bounded TrainState sidecar smokes: Klein binds `201` manifest tensors to
  `604` optimizer sidecar keys, and Z-Image binds `521` manifest tensors to
  the same `param.N` / `adam_m.N` / `adam_v.N` / `__meta__` order. These are
  optimizer/master sidecar scaffolds only; they do not provide product-loop
  full-weight update, runtime rebind, resume parity, or OneTrainer parity.
  `scripts/check_full_finetune_inventory_keys.py --strict` also verifies the
  current inventory keys against the local Klein safetensors header and Z-Image
  sharded index without CUDA.
- Resume has sidecar/scaffold coverage, but not full product continuation
  parity for every model.
- Z-Image strict sampler speed now has comparable evidence, but speed parity is
  not accepted. `scripts/check_zimage_sampler_contract.py --strict-speed`
  passes against `/home/alex/onetrainer-mojo/parity/zimage_sampler_speed.json`.
  The paired 1024x1024, seed 42, 28-step, CFG 3.5 BF16 record shows
  OneTrainer denoise `2.144007647s/step` and peak `14340 MiB`, while the paired
  Mojo record shows denoise `5.206695175s/step`, text encode `33.874484788s`,
  VAE `1.809631493s`, peak `22238 MiB`, and supervisor wall `225.13s`. The
  current Mojo-only post-cleanup one-step supervisor artifact
  `/tmp/zimage_speed_1step_moddev_speed.json` records denoise
  `4.302052192s/step`, text encode `10.645762139s`, VAE `1.487801541s`, and
  peak `22111 MiB`; it is evidence only until paired with OneTrainer.

## Klein Train-Replay / Offload Blockers

Accepted Klein evidence is artifact/oracle evidence only: the current
`klein_train_ref_step000` dump validates the train-ref tensors, all `288`
adapter gradient tensors per phase, zero clipping delta, optimizer state
initialization, unchanged adapter weights at `lr_before=0.0`, and bounded
synthetic positive-lr AdamW math from the same gradients. `scripts/check_klein_loss_replay.py`
also replays the dumped `output.predicted` and `output.target` tensors on CPU,
proving the default MSE loss and `d_loss = (2/N) * diff` bridge for the captured
step. `scripts/check_klein_adamw_state_init_replay.py --strict` now streams all
six adapter phases, validates the zero-lr state-init contract (`288` entries /
`864` state tensors / `87032096` state elements), confirms `27262171` nonzero
post-clip gradient elements with zero adapter delta, and projects the Mojo BF16
moment state (`exp_avg_l2=0.0005975040211676537`,
`exp_avg_sq_l2=8.403631552140676e-11`). The bounded Mojo smoke
`serenitymojo/models/klein/parity/klein_lora_adamw_state_init_smoke.mojo`
also executes the real model-level `klein_lora_adamw_step` helper on a
synthetic `1` double / `1` single LoRA set: zero-lr params stay unchanged,
moment error is `0.0`, and a bounded positive-lr follow-up changes `644` BF16
adapter values. `serenitymojo/models/klein/parity/klein_train_ref_adamw_state_init_replay.mojo`
then opens the real OneTrainer adapter dump, loads all `144` LoRA modules /
`288` A/B tensors plus post-clip grads, calls `klein_lora_adamw_step` at
`lr=0`, confirms unchanged params, reports `max_moment_err=0.0`,
`nonzero_moments=27262171`, and a bounded positive-lr follow-up changing
`27262275` BF16 adapter values. `scripts/check_klein_adamw_positive_lr_oracle.py --strict`
is only a CPU host-list optimizer support oracle for the same captured gradients
at synthetic positive lr with BF16 moment projection and deterministic
stochastic BF16 param rounding (`changed=27262275`,
`l2=0.014876430049126985`, `max_abs=6.103515625e-05`). Those gates are support
evidence only; they are not CUDA/GPU parity, a real later OneTrainer model step,
or full Mojo backward parity.

Exact remaining blockers:

- No accepted Mojo replay reruns Klein `predict -> backward_lora` from
  `/home/alex/onetrainer-mojo/parity/klein_train_ref_step000.safetensors` and
  compares all adapter gradients against OneTrainer; the standalone CPU
  loss/d_loss bridge is covered by `scripts/check_klein_loss_replay.py`.
- No accepted full Mojo `predict -> backward_lora -> AdamW` replay compares all
  gradients, optimizer state payloads, and adapter deltas against OneTrainer;
  `scripts/check_klein_adamw_state_init_replay.py`,
  `serenitymojo/models/klein/parity/klein_lora_adamw_state_init_smoke.mojo`,
  and `serenitymojo/models/klein/parity/klein_train_ref_adamw_state_init_replay.mojo`
  cover zero-lr optimizer support evidence; `scripts/check_klein_adamw_positive_lr_oracle.py`
  covers only a CPU host-list synthetic positive-lr optimizer oracle from the
  same captured gradients and is not CUDA/GPU parity.
- No later OneTrainer Klein dump exists yet with `lr_before > 0` and nonzero
  `adapter_after - adapter_post_clip`; bounded synthetic positive-lr AdamW math
  is support evidence only, not a real later-step oracle or model parity.
- No accepted train-ref bounded-CUDA/offload/checkpoint backward replay proves
  the production low-memory Klein path against the train-ref dump. A small
  offloaded-tape backward parity gate now exists, and the real product trainer
  now passes a one-step `CPU_OFFLOADED` smoke, but neither one is the train-ref
  replay.
- The prebuilt resident `/tmp/klein_train_ref_forward_replay` was run on
  2026-06-06 on the local RTX 3090 Ti 24GB and OOMed after opening the real
  Klein train-ref dump and 9B checkpoint. That is evidence that the resident
  replay is not a production path for this machine; acceptance requires the
  bounded CPU_OFFLOADED/checkpoint replay, not a larger resident retry.
- Klein weight streaming is distinct from OneTrainer CPU_OFFLOADED activation
  and layer offload. The train loop uses `TurboPlannedLoader` for block weight
  streaming, but that is not activation/layer CPU offload and not
  bounded-CUDA offload/checkpoint backward replay parity.
- Klein product plumbing now accepts `CPU_OFFLOADED` policy and, when
  `enable_activation_offloading=true`, routes through a direct raw-byte
  offloaded-tape forward/backward path:
  `klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_offloaded_tape`
  and `klein_stack_lora_backward_offloaded_tape_turbo_moddev_rope_scratch`.
  The cache preload path also preserves latent and text-token storage dtype
  instead of forcing F32, and the caption-dropout zero text embedding follows
  the cached text-token dtype. This still needs the bounded CUDA train-ref
  replay before it is accepted as product parity.
- On 2026-06-06 `/tmp/train_klein_real serenitymojo/configs/klein9b_cpu_offloaded_smoke.json 1 0 - nosample_profile`
  completed one real product step on the local RTX 3090 Ti using that
  `CPU_OFFLOADED` branch. It loaded `40` compatible cached samples, used a
  host activation tape of `838,860,800` bytes, reported forward `4.038s`,
  backward `4.086s`, optimizer `4.404s`, and total `12.8s/step`, then saved
  `144` LoRA pairs and `144` optimizer-state pairs. The product LoRA file
  `/tmp/klein9b_cpu_offloaded_smoke.safetensors` has `432/432` BF16 tensors;
  the state file has `288` BF16 adapter tensors plus `576` F32 AdamW moment
  tensors. This is a product smoke/speed datapoint, not train-ref parity.
- A follow-up 5-step run with
  `serenitymojo/configs/klein9b_cpu_offloaded_5step_smoke.json` also completed
  on 2026-06-06. Per-step loss was `2.274788`, `1.652754`, `1.513462`,
  `1.6152831`, `1.3873278`; grad norm was `0.4500`, `0.7378`, `0.1915`,
  `0.2995`, `0.6938`. Step times were `12.8s`, `13.2s`, `13.6s`, `13.7s`,
  and `13.6s`. Warmed forward stayed around `2.63-2.78s`, backward around
  `3.97-4.02s`, optimizer around `6.32-6.87s`, and the activation tape stayed
  `838,860,800` host bytes. Mid-run `nvidia-smi` showed the trainer at
  `10,394 MiB` VRAM and total GPU memory at `11,502 MiB`; after exit the GPU
  returned to desktop-only usage. The saved 5-step LoRA file has `432/432`
  BF16 tensors; its state file has `288` BF16 adapter tensors plus `576` F32
  AdamW moment tensors.
- Resume from that 5-step state also completed on 2026-06-06 with
  `serenitymojo/configs/klein9b_cpu_offloaded_resume10_smoke.json`, starting at
  global step `5` from `/tmp/klein9b_cpu_offloaded_5step_smoke.safetensors` and
  continuing through step `10`. Losses for steps `6-10` were `1.3712093`,
  `0.8650592`, `0.7500965`, `1.0972869`, and `1.0009087`; step times were
  `15.2s`, `14.1s`, `14.1s`, `14.0s`, and `14.2s`. The saved resume LoRA file
  `/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors` has `432/432` BF16
  tensors, and its state sidecar has `288` BF16 adapter tensors plus `576` F32
  AdamW moment tensors.
- Resume from that 10-step state through step `20` also completed with
  `serenitymojo/configs/klein9b_cpu_offloaded_resume20_smoke.json`, starting
  from `/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors`. Losses for
  steps `11-20` were `0.84224933`, `1.310655`, `0.7975194`, `1.3193466`,
  `0.6252446`, `0.70147544`, `0.712633`, `0.24021716`, `0.6145748`, and
  `0.89166784`; step times stayed around `13.5-14.8s/step` after resume.
  Warmed phase timing remained forward `2.52-2.68s`, backward `3.92-4.09s`,
  and optimizer `6.89-7.16s`, so the current product path is optimizer-bound.
  `/tmp/klein9b_cpu_offloaded_resume20_smoke.safetensors` passed the product
  artifact guard with `432/432` BF16 LoRA tensors; its state sidecar has
  `288` BF16 adapter tensors plus `576` F32 AdamW moment tensors.
- The standalone Klein sampler now runs as a separate process from that resume
  LoRA and writes real PNG artifacts. The fast validation preset
  `serenitymojo/configs/klein9b_alina_samples_fast512.json` uses 512x512,
  one denoise step, and `cfg=1.0`; it produced
  `output/alina_train/klein_lora_resume10_fast512_cfg1.png`, a `512 x 512`
  RGB PNG, with denoise speed `3.1s/step` and about `7s` command wall time
  including model load and VAE decode. The guided `cfg=4.0` one-step run also
  produced `output/alina_train/klein_lora_resume10_fast512.png`, but denoise
  was `24.5s/step`, so guided sampling is not the fast validation path yet.
  The same fast preset also loaded the resume20 LoRA and wrote
  `output/alina_train/klein_lora_resume20_fast512_cfg1.png`, a `512 x 512`
  RGB PNG, again at `3.1s/step`. The resume20 image is smoke-quality because
  it is one denoise step from a 20-step LoRA, not an accepted quality or parity
  sample.
  These are product smoke artifacts, not accepted OneTrainer sampler parity;
  `check_klein_sampler_parity_contract.py` still reports missing paired
  OT/Mojo sampler manifest, trajectory, VAE/PNG numeric parity, and speed/VRAM
  parity evidence.
- The default sample config cap-cache preflight is now clear. Running
  `/tmp/klein9b_precache_sample_prompts serenitymojo/configs/klein9b_alina_samples.json`
  wrote both `alina_garden` and `alina_evening` positive/negative caps, and
  `check_klein_cap_cache_contract.py --samples serenitymojo/configs/klein9b_alina_samples.json`
  validates all four as BF16 `[1,512,12288]` files. The sampler contract report
  now removes cap-cache readiness from the known blocker list.
- The validation sampler now threads `lora_multiplier` through to the Klein
  sampler. The implementation scales each adapter's runtime `scale` field before
  device upload and does not mutate BF16 LoRA A/B storage or AdamW moments.
  After the change, `/tmp/klein_sample_cli` still runs the fast resume20 sample
  at `3.1s/step` and writes a valid 512x512 RGB PNG.
- `check_klein_offload_checkpoint_contract.py` reads OneTrainer's
  `GradientCheckpointingMethod`, `OffloadCheckpointLayer`,
  `LayerOffloadConductor`, `offload_quantized`, and Flux2 low-VRAM presets to
  prove the OT target: `CPU_OFFLOADED` is checkpointing plus activation/layer
  movement, not file-level weight streaming.
- Safe offload scaffolding smokes pass without CUDA: `offload_checkpoint_config_smoke`
  validates typed config derivation, `conductor_policy_smoke` validates
  OneTrainer activation/layer/async gates and keeps `layer_offload_fraction`
  as `Float64`, `plan_smoke` validates shared block-plan topology,
  `planned_loader_smoke` typechecks the plan-aware loader API, `residency_smoke`
  validates state/budget/refcount/eviction logic and the Klein9B plan, and
  `turbo_slots_smoke` validates metadata slot planning. These are
  control/runtime scaffolding gates, not model replay or GPU parity.
- `klein_activation_tape_plan_smoke` pins the current Klein 9B LoRA boundary
  activation target at `432,013,312` BF16 bytes for currently retained
  boundaries. The minimal live backward tape excludes unused `img_in_act` /
  `txt_in_act` and is `419,430,400` BF16 bytes. F32 would double those targets,
  so the CPU_OFFLOADED branch must use raw/dtype-preserving activation offload,
  not `Tensor.to_host()`.
- `klein_activation_tape_offload_smoke` now covers the bounded bridge for that
  first slice: `serenitymojo/models/klein/activation_tape.mojo` offloads only
  `dbl_img_in`, `dbl_txt_in`, `sgl_x_in`, `img_out`, and `ln_img_out` through
  `HostOffload` raw bytes, restores BF16 tensors byte-exactly on a tiny CUDA
  smoke, and rejects an F32 storage-dtype claim. This unblocks building the
  bounded Klein backward replay, but it is not that replay and does not accept
  `CPU_OFFLOADED` product parity.
- `klein_stack_lora_offloaded_tape_parity` now runs the bounded backward replay
  through the offloaded tape on the small Klein LoRA stack. It uses the current
  production `build_klein_lora_set` slot layout (`12` double-block slots plus
  `2` single-block slots), offloads `2560` raw activation bytes in the tiny
  case, restores the tape, reruns backward, and matches resident-tape backward
  for input-token, modulation, and every LoRA gradient with `max_abs=0.0`. This
  is the reusable unblocker pattern; it is still not the full train-ref or
  product `CPU_OFFLOADED` acceptance gate.
- 2026-06-06 bounded GPU checks passed:
  `/tmp/klein_activation_tape_offload_smoke`,
  `/tmp/klein_stack_lora_offloaded_tape_parity`,
  `/tmp/klein_lora_adamw_state_init_smoke`, and
  `/tmp/klein_train_ref_adamw_state_init_replay`. The train-ref AdamW replay
  loaded all `144` LoRA modules / `288` A/B tensors, kept all `43,515,904`
  adapter elements unchanged at zero lr, reported `max_moment_err=0.0`, and a
  synthetic positive-lr branch changed `27,262,275` BF16 adapter values. These
  are still support gates, not full predict/backward/AdamW parity.
- 2026-06-06 production-loop smoke passed after rebuilding
  `/tmp/train_klein_real` from `train_klein_real.mojo`; the build needed the
  timer type fix `UInt(0)` for the backward profiler and an explicit Pixi
  SQLite link path for SerenityBoard logging.
- Flame autograd v2 points at the same production shape: block-level recompute,
  saved/offloaded tensor records with release state, no `Tensor.clone(ctx)` for
  activation saving, and async/event host-device movement later. The first Klein
  slice should keep CPU_OFFLOADED fail-loud while adding a LoRA-specific host
  tape for block inputs plus `img_out` / `ln_img_out`.
- `python3 scripts/check_klein_adapter_grad_update_replay.py --require-mojo-parity`
  is the strict promotion guard and must return exit `2` until the full Mojo
  `predict -> backward_lora -> AdamW` replay exists.
- `python3 scripts/check_klein_loss_replay.py --strict` must pass before using
  the train-ref loss bridge as support evidence; it does not prove transformer
  forward, backward, AdamW, or sampler parity.
- `python3 scripts/check_klein_adamw_state_init_replay.py --strict` must pass
  before using the captured Klein zero-lr optimizer state-init projection as
  support evidence; it does not execute Mojo backward/AdamW or prove nonzero
  update parity.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/models/klein/parity/klein_lora_adamw_state_init_smoke.mojo -o /tmp/klein_lora_adamw_state_init_smoke`
  plus `/tmp/klein_lora_adamw_state_init_smoke` must pass before using the
  bounded Mojo AdamW helper path as support evidence. This smoke does not
  satisfy `--require-mojo-parity`, `--require-update-bearing`, or
  `check_klein_offload_checkpoint_contract.py --strict`.
- `pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/models/klein/parity/klein_train_ref_adamw_state_init_replay.mojo -o /tmp/klein_train_ref_adamw_state_init_replay`
  plus `/tmp/klein_train_ref_adamw_state_init_replay` must pass before using
  the all-train-ref-adapter Mojo AdamW state-init path as support evidence.
  This replay still does not satisfy `--require-mojo-parity`,
  `--require-update-bearing`, or `check_klein_offload_checkpoint_contract.py --strict`.
- `python3 scripts/check_klein_adamw_positive_lr_oracle.py --strict` must pass
  before using the synthetic positive-lr CPU host-list optimizer oracle as
  support evidence. It does not satisfy CUDA/GPU parity, `--require-mojo-parity`,
  `--require-update-bearing`, or `check_klein_offload_checkpoint_contract.py --strict`.
- `python3 scripts/check_klein_offload_checkpoint_contract.py --strict` must
  return exit `2` until accepted train-ref low-memory offload/checkpoint
  backward replay exists for Klein.
- `python3 scripts/check_klein_sampler_parity_contract.py --strict` must return
  exit `2` until paired post-patch/post-pack OT/Mojo initial-noise trajectory
  artifacts, VAE/image, and speed/VRAM sampler parity evidence exists.
- `python3 scripts/check_klein_sampler_artifact_manifest.py --strict` must
  return exit `2` until `/home/alex/onetrainer-mojo/parity/klein_sampler_parity_manifest.json`
  records paired OneTrainer/Mojo run identity, raw/post-patch/post-pack noise,
  scheduler timesteps, latent trajectory, final packed and VAE pre-decode
  latents, PNG, denoise/VAE timing, and peak VRAM evidence.
  No-CUDA planning artifacts can be written without accepting parity:
  `python3 scripts/check_klein_sampler_artifact_manifest.py --write-template /tmp/klein_sampler_manifest_template.json --template-artifact-root /tmp/klein_sampler_parity_artifacts --template-width 512 --template-height 512 --template-steps 1 --write-readiness /tmp/klein_sampler_manifest_readiness.json`.
  The OneTrainer producer also supports
  `python3 scripts/run_klein_onetrainer_sampler_parity.py --no-run --write-checker-template /tmp/klein_sampler_manifest_template.json --write-checker-readiness /tmp/klein_sampler_manifest_readiness.json`;
  this does not load OneTrainer/CUDA and remains template/readiness evidence only.
- `python3 scripts/check_flux1_sampler_artifact_manifest.py --strict` must
  return exit `2` until `/home/alex/onetrainer-mojo/parity/flux1_sampler_parity_manifest.json`
  records paired OneTrainer/Mojo Flux.1 text conditioning, raw/packed noise,
  FlowMatch sigmas/timesteps/step trace, packed latent trajectory, final
  latent, VAE tensor/PNG, per-stage timing, peak VRAM, and numeric comparisons.
- `python3 scripts/check_chroma_sampler_artifact_manifest.py --strict` must
  return exit `2` until `/home/alex/onetrainer-mojo/parity/chroma_sampler_parity_manifest.json`
  records paired OneTrainer/Mojo Chroma prompt/negative CFG branches, attention
  masks, raw/packed noise, FlowMatch schedule, CFG prediction trajectory, final
  latent, VAE tensor/PNG, timing, VRAM, and numeric comparisons.
- The Flux.1 and Chroma manifest guards can now write no-CUDA planning outputs
  without creating evidence: run
  `python3 scripts/check_flux1_sampler_artifact_manifest.py --write-template /tmp/flux1_sampler_parity_manifest.template.json --write-readiness /tmp/flux1_sampler_manifest_readiness.json`
  and
  `python3 scripts/check_chroma_sampler_artifact_manifest.py --write-template /tmp/chroma_sampler_parity_manifest.template.json --write-readiness /tmp/chroma_sampler_manifest_readiness.json`.
  These generated files are TODO/readiness scaffolds only; strict mode rejects
  `template=true`, nonnumeric schedule traces, missing real artifacts, zero
  timings/VRAM, and false or over-tolerance comparisons.
- `python3 scripts/check_klein_cap_cache_contract.py --strict` now passes for
  the configured Klein sample caps. This proves BF16 9B cap-cache headers and
  exact byte sizes only; it does not accept denoise, VAE/image, speed, or GPU
  parity.
- Klein's real trainer now validates sample cap-cache headers before
  `DeviceContext()` when runtime sampling is enabled. This catches missing,
  F32, 4B-width, or truncated sample caps before CUDA setup; it is a preflight
  guard, not sampler parity.
- `python3 scripts/check_klein_conditioning_template_contract.py` must pass
  before accepting Klein Qwen3 conditioning; it compares OneTrainer's Qwen3
  chat-template path and padded token ids against the local tokenizer files and
  the local precache template.
- `python3 scripts/check_klein_initial_noise_sidecar_contract.py --strict` must
  pass before using the Klein sampler sidecar replay path; it proves only that
  the post-patch/post-pack initial-noise sidecar hook exists and preserves dtype.

## Not Ready For Use

- No target model has complete OneTrainer parity across loss, gradients,
  optimizer update, sampler output, speed, resume, and dtype boundaries.
- Advanced optimizer parity is not covered beyond the current AdamW/Adafactor
  surfaces. PR #1344 adds centered weight-decay anchors, scaled optimizers,
  factored/state-precision modes, SinkSGD/OrthoGrad options, and BF16 resume
  dtype behavior that need explicit future gates.
- Speed parity is not accepted yet. Some OneTrainer timings exist, but matching
  Mojo train/sampler timings with VRAM and identical settings are still needed.
  `scripts/check_flux_family_sampler_contracts.py` now rejects accepted
  Flux2/Klein sampler speed claims unless paired OneTrainer/Mojo timing, VRAM,
  prompt, seed, resolution, steps, CFG, dtype, and trajectory evidence exist;
  it also reports the missing Flux.1 and Chroma sampler artifact manifests without
  accepting sampler parity.
  `scripts/check_klein_sampler_parity_contract.py` tracks the matching
  Klein-specific sampler blockers, and
  `scripts/check_klein_cap_cache_contract.py` catches missing or wrong-shape
  precomputed Qwen cap files before a sampler run.
- `serenitymojo/io/cap_cache_header_smoke.mojo` is the no-CUDA Mojo smoke for
  the same runtime preflight: it accepts BF16 `[1,512,12288]` and
  `[512,12288]` caps, and rejects F32 or 4B-width caps for the 9B sampler.
- Klein now has an explicit sampler parity entry that can consume an
  OT-equivalent post-patch/post-pack initial-noise tensor sidecar without a dtype
  cast. This is wiring only; it does not accept trajectory parity until matched
  OneTrainer noise/latent artifacts exist.
- Z-Image generation is still too slow for production speed parity until the
  denoise path is optimized against the OneTrainer baseline.
  `scripts/check_klein_chroma_flux_sd35_train_readiness.py --models zimage`
  now reports split-process sampler request/control wiring, request contract,
  and strict sampler speed/VRAM evidence as passing, but keeps full-finetune
  blocked because the product loop is still LoRA-only. This is queued-output
  plumbing and comparable timing evidence, not accepted sampled-output parity or
  speed parity. The split-process sample request source contract passes,
  including `zimage_generate --request` and the request-side `result_manifest`.
  Generator-only result manifests still record `peak_vram_mib=0`; use the
  supervisor wrapper for positive VRAM evidence. The latest trace shows the
  speed blocker is two serial CFG main-stack passes, while OneTrainer batches
  CFG cond/uncond.
- Strict dtype cleanup is incomplete. The sharpened no-GPU cache/preflight
  guard now passes the narrow unsafe F32 cache-boundary scan for Qwen, Ernie,
  Anima, Klein, Z-Image, Chroma, Flux.1, SD3.5, and SDXL, but all real loops
  still have broader host `Float32` carriers in training/offload/sample
  surfaces. All nine loops now stage selected cache readbacks through their
  stored dtype before host step math. For Z-Image specifically,
  `scripts/check_klein_chroma_flux_sd35_train_readiness.py --models zimage`
  now reports split-process sample-request/control plumbing and strict
  speed/VRAM evidence as passing; its remaining reported blocker is
  full-finetune product dispatch. These passes do not accept sampled-output or
  speed parity. Other target models still have their own sampler artifact/speed
  and full-finetune blockers.
- Full finetune is not production-supported yet, even for models where
  OneTrainer supports it. `scripts/check_full_finetune_contracts.py
  --target-model klein --write-readiness /tmp/full_ft_klein_readiness.json`
  and the same command with `--target-model zimage` both report 3 blockers
  each: 1 LoRA-only loop blocker plus product-loop full-finetune parity and
  parity artifacts. The Klein and Z-Image payload save/load/manifest smokes
  compile and run, their TrainState sidecar smokes bind optimizer masters to
  manifest order, and `scripts/check_full_finetune_inventory_keys.py --strict`
  passes for Klein `201` keys and Z-Image `521` keys. The all-target report
  now shows 62 blockers (`loop=9`, `evidence=53`). Use
  `scripts/check_full_finetune_contracts.py --target-model klein --target-model zimage --write-template-dir /tmp`
  to write no-CUDA implementation templates; those templates are not readiness
  evidence.

## Short Next List

1. For Qwen, Ernie, Anima, and Klein, capture later OneTrainer dumps where
   `lr_before > 0`, then replay nonzero adapter updates in Mojo.
2. For Chroma and SDXL, rerun Mojo backward from the matching step dump or
   capture step-with-grads artifacts, then compare full AdamW updates.
3. Add matching Mojo train speed and VRAM gates beside each OneTrainer baseline.
4. Finish full sampler denoise/decode/image parity for each target model.
5. Finish product resume and full-finetune runtime per model.
6. Remove or justify remaining production F32 tensor-boundary carriers.
7. Add per-part offload policy and advanced optimizer fail-loud/numeric gates
   for the two tracked upstream OneTrainer PRs.

SD3.5 note: `python3 scripts/check_sd35_contract.py --write-readiness <path>`
now writes the no-CUDA blocker report. Current status is still blocked with
`18` blockers: `onetrainer_config=4`, `onetrainer_baseline=7`,
`train_ref_artifact=3`, and `mojo_dtype_or_memory=4`.

## Practical Answer

Use this repo now for continuing the port and running parity gates. Do not use
it yet for a real training job that you expect to trust or ship from.
