# Checkpoint Dtype Loading Audit - 2026-06-05

Status: handoff for the OneTrainer-parity Mojo trainer port.

## Trainer Direction

The OneTrainer-parity Mojo port is the supported trainer path. Older
mojodiffusion trainer experiments are reference or compatibility surfaces unless
they use the same `serenitymojo` model loaders, offload runtime, sampler, save
and resume, progress display, and dtype contract.

Inference and training must share the same model/runtime primitives. A fix in a
loader or core op is expected to benefit both train and sample paths, not create
a train-only fork.

Current shared OneTrainer-port core changes are summarized in
`serenitymojo/docs/ONETRAINER_CORE_PORT_STATUS_2026-06-06.md`.

## Safetensors Result

The core safetensors reader is not double-loading checkpoint payloads:

- `serenitymojo/io/safetensors.mojo` reads the header eagerly, mmaps the data
  segment, and exposes `tensor_bytes` spans.
- `serenitymojo/io/sharded.mojo` opens each unique shard once and maps tensor
  names to shard handles.
- `Tensor.from_view` preserves the view dtype and copies raw bytes to device.

The F32 expansion bug was in model loaders and offload adapters that converted
checkpoint tensors through host-F32 lists or F32 device tensors.

## Fixed In This Pass

- Core ops: `rms_norm`, `rms_norm_backward`, and `group_norm_backward` now
  support F32 activations/grads with BF16/F16 norm weights while returning
  activation grads in activation dtype and norm grads in norm-weight dtype. RoPE
  forward/backward supports mixed grad and table storage dtypes without
  materializing full F32 table tensors. Gate residual backward supports F32
  gradient/gate tensors with BF16/F16 activation storage and returns gradients
  in the gradient dtype.
- Flux, Chroma, Klein, Qwen-Image, ZImage, L2P, Anima, and Ernie loaders now
  preserve checkpoint dtype for block weights, biases, norm scales, stack-base
  projections, and final projections touched by the official trainer path.
- Chroma row-stacking now happens on device with `concat`, so separated q/k/v
  checkpoint tensors are fused without host-F32 roundtrips.
- Flux, Klein, and Qwen stack/block structs gained direct `ArcPointer[Tensor]`
  constructors so loaders can pass device tensors without rebuilding host lists.
- ZImage final projection and Ernie patch/final stack-base tensors now preserve
  checkpoint dtype instead of using F32 helper loaders.
- Qwen-Image FP8 checkpoint tensors now dequantize to BF16 through the Qwen
  loader path instead of relying on the generic F32/BF16 cast helper. BF16
  streamed block tensors are reused by Arc instead of cloned through a cast.
- Shared full-finetune model tensor save/load scaffolding now round-trips named
  safetensors entries without widening BF16 storage to host F32.
- Shared OneTrainer text-conditioning contract metadata now pins the target
  model prompt/token/cache/mask/shape requirements without introducing tensor
  storage casts. This is a contract/static gate, not full tokenizer or text
  encoder numeric parity.
- Shared OneTrainer VAE encoder/cache contract metadata now pins raw cache
  layout, prepared latent layout, patch/token handoff, and raw encode readiness
  without introducing tensor storage casts. Flux2/Klein remains prepared-only in
  the current Mojo encoder because it emits the 128-channel patchified+BN latent,
  not the raw 32-channel cached mean that OneTrainer stores before setup.
- Ernie and Anima raw OneTrainer LoRA key guards now validate exact BF16
  `.alpha`, `.lora_down.weight`, and `.lora_up.weight` inventories for their
  selected local reference surfaces. The reduced Mojo smokes also save and
  resume through those raw keys without using the generic PEFT key family.
- Z-Image now has a sampler/setup source contract against local OneTrainer that
  pins FlowMatch scheduler use, latent scale/unscale, resolution defaults,
  rank-5 transformer input, textbook CFG, and training flow target. This is a
  sampler scaffold guard, not image or speed parity.
- The product-run dry-run now carries a shared cache preflight that binds the
  text-conditioning and VAE encode/cache contracts. `only_cache` fails loudly
  for models whose current Mojo encoder cannot emit OneTrainer raw VAE cache.
- Shared OneTrainer train-loop policy now lives in
  `serenitymojo/training/onetrainer_train_loop_policy.mojo`. Qwen, Ernie,
  Anima, SD3.5, SDXL, Flux.1, Klein, Chroma, and Z-Image use it for
  LoRA/AdamW validation, checkpoint/offload policy, sample cadence, save
  cadence, output paths, and LoRA state sidecar naming.
- The real train-loop cache readers for Qwen-Image, Ernie, Anima, Z-Image,
  Chroma, Flux.1, SD3.5, and SDXL no longer materialize cache tensors as
  device F32 just to cross the loader boundary. They now use `Tensor.from_view`
  first, preserving the safetensors dtype on device before any legacy host
  readback required by the current train stack.
- SD3.5 stack-base embedder and final-layer weights now load with
  `Tensor.from_view` into checkpoint-dtype resident tensors instead of host F32
  lists. The streamed block weights remain a separate host-F32 blocker.
- The Chroma and SDXL update-bearing OneTrainer dump consumers now read the
  exact step/adapters/meta artifacts and compare sampled `adapter_after -
  adapter_post` values without claiming full AdamW parity. The full parity gate
  still blocks because those adapter dumps have no per-adapter gradient phase
  and no Mojo consumer reruns backward against the matching step dump yet.
- `scripts/check_onetrainer_train_math_contract.py --strict-product` now also
  statically guards PyTorch/OneTrainer AdamW order in `optim.mojo`,
  `fused_adamw_multitensor.mojo`, and `train_step.mojo`: decoupled weight decay
  must be applied to the parameter before the adaptive Adam subtraction. This
  catches stale post-Adam decay formulas, but it is still a shared math/source
  guard, not full per-model backward/update parity.
- `linear`/`linear_backward` and `conv2d`/`conv2d_backward` now support the
  narrow frozen-weight path of F32 activations/grads with BF16/F16 checkpoint
  weights and biases while returning F32 activations/grads and parameter grads in
  the checkpoint weight dtype. SDXL real-weight loading now uses these paths plus
  mixed GroupNorm backward to keep embedding, SpatialTransformer, final-GroupNorm,
  ResBlock GroupNorm, ResBlock embedding-linear, ResBlock conv, conv-in/out,
  downsample, and upsample conv tensors in checkpoint dtype. BF16/F16 conv RSCF
  remapping uses dtype-preserving host lists; F32 lists are only used for
  explicitly F32 checkpoint tensors and placeholder tensors.
- Qwen-Image now has two separate dtype gates: the loader/streamed-block gate
  passes in default mode, while `--strict-train-boundaries` reports the remaining
  host-F32 activation/gradient/mod-vector carriers in the train/offload path.
- LTX2 production offload remains BF16-preserving. Its legacy host-list extractor
  is not the supported production trainer path.

Allowed F32 boundaries remain: GEMM accumulation, reductions, schedules,
timestep/noise scalars, RoPE tables, generated modulation vectors, host AdamW
master state, host grad readback, debug/parity oracles, and file formats that
require F32.

## Remaining Non-Official F32 Surfaces

The wide scan still finds host-F32 or F32-native loaders in older or unfinished
families:

- `serenitymojo/models/sd35/sd35_stack_lora.mojo`
- `serenitymojo/models/wan22/wan22_stack_lora.mojo`
- legacy `load_ltx2_block_weights_from_block`

Do not treat those as official production trainer paths until they are ported to
the same checkpoint-dtype storage contract.

SD3.5 has additional non-dtype parity blockers:

- The local OneTrainer SD3.5 configs still use `STABLE_DIFFUSION_3` instead of
  the SD3.5 `STABLE_DIFFUSION_35` path. Plain SD3 is not a port target.
- The current Mojo SD35 train/cache path now uses OneTrainer-style
  `latent_image` plus separate CLIP/T5 token, mask, hidden, and pooled
  text-cache entries, with VAE scale/shift applied once in the train loop.
- `sd35/weights.mojo` now preserves checkpoint dtype for stack-base
  x/context/timestep/pooled/final weights.
- `sd35_block.mojo` / `sd35_stack_lora.mojo` are host `List[Float32]` hot paths
  that save full activations instead of CPU-offloaded checkpoint inputs.
- Current SD35 stack LoRA math now wires all eight per-block targets through
  forward and backward (`ctx/x` qkv, proj, fc1, fc2). The all-slot smoke verifies
  shaped, finite, nonzero grads for every target, but only the representative
  x-stream QKV path has a torch numerical parity gate so far.

Guard:

- `python3 scripts/check_sd35_contract.py` reports the current SD35 blockers.
- Current report mode passes while naming `10` blockers. Strict mode must pass
  before SD3.5 train speed/loss numbers can be treated as official
  OneTrainer parity evidence.
- `python3 scripts/check_train_loop_cache_contract_bindings.py --marker-limit 1`
  should show all nine real loops bound to text/VAE cache preflight. Remaining
  F32 reports are host compute carriers, debug/readback paths, RoPE tables, or a
  documented model-compute cast; they are not evidence of full dtype parity.

## Klein Split-Slot Status

The dtype fixes and split-slot carrier fix make mojodiffusion core structurally
traceable to OneTrainer's Flux2/Klein LoRA target set. They do not yet make
Klein accepted train parity.

OneTrainer's Flux2/Klein LoRA setup wraps `12` double-block Linear targets plus
`2` single-block targets, so Klein 9B has `144` LoRA adapters. Current status:

- `/home/alex/onetrainer-mojo/src/onetrainer_mojo/model/klein/klein_stack_lora.mojo`
  has the split `DBL_SLOTS=12` port mirror.
- `serenitymojo/models/klein/klein_stack_lora.mojo` now also uses split
  `DBL_SLOTS=12` / `SGL_SLOTS=2`.
- `serenitymojo/models/klein/double_block.mojo` now carries separate image/text
  q/k/v/out/ff_in/ff_out LoRA slots, with matching scatter fields.
- `serenitymojo/models/klein/single_block.mojo` now uses the OneTrainer
  `to_qkv_mlp_proj` and `to_out` shapes: `D -> 3D+2F` and `D+F -> D`.
- `serenitymojo/models/klein/lora_adapter.mojo` and
  `serenitymojo/util/bf16_stochastic_rounding.mojo` provide OneTrainer-style
  BF16 LoRA storage and stochastic-BF16 AdamW helper behavior in core.
- `train_klein_real.mojo` builds after the split-12 carrier swap.
- `klein_stack_lora_real_smoke.mojo` builds after the split-12 carrier swap.
- `TurboBlockLoader` now avoids a full-model pinned host block store by default.
  It fills the active pinned slot from mmap on prefetch and then DMA-copies that
  slot to the device slab. This is required for Klein 9B because the old
  full-checkpoint pinned store OOMed at loader open.
- `train_klein_real.mojo` now has a training-specific stack-base loader that
  skips the startup seed final-mod GEMM. The trainer overwrites final
  shift/scale from per-step timestep modulation before every forward.
- `train_klein_real.mojo` now reads and validates `TrainConfig` before CUDA
  context construction, applies config cache/output/sample cadence controls, and
  returns for `only_cache=true` before checkpoint, cache, board, sample, or CUDA
  work.
- `klein_train_ref_artifact_smoke.mojo` now opens the OneTrainer metadata JSON,
  step dump, and adapter/gradient dump, validating shapes/dtypes/scalars,
  optimizer state-init metadata, representative gradient payloads, and bounded
  synthetic positive-lr AdamW math from the captured gradients.
- `scripts/check_klein_adapter_grad_update_replay.py --require-synthetic-positive-lr`
  passes as bounded optimizer-math support evidence: it reuses the captured
  gradients with `lr_after=2.9999999999999997e-06` and proves finite nonzero
  AdamW deltas. This is not a real later OneTrainer model step.
- A one-step split-12 offload trainer run completed on the local RTX 3090 Ti:
  loss `2.274788`, grad norm `0.4500`, forward `2.699163s`, backward
  `2.9471564s`, optimizer `4.5635543s`, total `10.5s/step`, save pairs `144`.
  The saved LoRA file now uses raw OneTrainer keys and its header validates as
  `144` adapters / `432` BF16 tensors.

Still not accepted:

- The real OneTrainer dumped train replay has not been rerun through
  `predict -> backward_lora -> AdamW` after the split-12 core swap.
- A full two-step Klein OneTrainer dump attempt for a positive-lr update OOMed
  locally. The current accepted dump remains `lr_before=0.0`, so
  `adapter_after - adapter_post_clip` is zero and real nonzero update parity is
  still missing.
- Speed, phase timing, and VRAM have not been measured on the accepted
  byte-identical OneTrainer replay path.
- The full resident split-12 stack smoke was run outside the sandbox and OOMed;
  the offload trainer is the production path.
- Low-memory/offload/checkpoint backward still needs real replay validation.

Guard:

- `python3 scripts/check_klein_lora_contract.py` reports the current split.
- `python3 scripts/check_klein_lora_contract.py --strict-core` must pass before
  mojodiffusion Klein train speed/loss numbers can be treated as official
  OneTrainer parity evidence.
- `python3 scripts/check_klein_adapter_grad_update_replay.py --require-synthetic-positive-lr`
  must pass; `--require-update-bearing` must continue to fail until a later
  positive-lr OneTrainer dump exists.

## Verification

Target accelerator used here: `sm_86` for the local RTX 3090 Ti.

Passed:

- `python3 scripts/check_ltx2_dtype_contract.py`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_train_loop_policy_smoke.mojo`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_product_run_smoke.mojo`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/qwen_train_control_wiring_smoke.mojo`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/ernie_train_control_wiring_smoke.mojo`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/anima_train_control_wiring_smoke.mojo`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/sd35_train_control_wiring_smoke.mojo`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/sdxl_train_control_wiring_smoke.mojo`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/flux_train_control_wiring_smoke.mojo`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/klein_train_control_wiring_smoke.mojo`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/chroma_train_control_wiring_smoke.mojo`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/zimage_train_control_wiring_smoke.mojo`
- `python3 scripts/check_klein_lora_contract.py`
- `python3 scripts/check_klein_lora_contract.py --strict-core`
- `python3 scripts/check_klein_lora_contract.py --strict-core --safetensors output/alina_train/alina_lora_step1.safetensors --num-double 8 --num-single 24`
- `python3 scripts/check_sd35_contract.py`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/models/sd35/parity/sd35_lora_all_slots_smoke.mojo`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/models/klein/parity/klein_lora_adapter_compile_check.mojo -o /tmp/klein_lora_adapter_compile_check`
- `/tmp/klein_lora_adapter_compile_check` outside sandbox: core Klein
  OneTrainer-style LoRA helper forward/backward/AdamW PASS.
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib -Xlinker -lsqlite3 serenitymojo/training/train_klein_real.mojo -o /tmp/train_klein_real_core_lora_adapter`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib -Xlinker -lsqlite3 serenitymojo/training/train_klein_real.mojo -o /tmp/train_klein_real_split12_core`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/models/klein/parity/klein_stack_lora_real_smoke.mojo -o /tmp/klein_stack_lora_real_smoke_split12`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/klein_train_control_wiring_smoke.mojo`:
  Klein train-control wiring PASS. It covers config-driven cache/output/sample
  cadence, pre-CUDA text/VAE cache preflight validation, disabled sampling,
  save-before-sample decisions, and fail-loud unsupported full-finetune,
  non-AdamW, CPU/offloaded or disabled activation checkpointing, non-step
  cadence, sharded checkpoints, non-9B compiled shapes, and raw VAE `only_cache`
  until the Mojo encoder can emit OneTrainer's raw 32-channel cached mean.
- `pixi run mojo run -I . serenitymojo/training/sample_prompt_config_smoke.mojo`
- `pixi run mojo build -I . serenitymojo/io/offload_checkpoint_config_smoke.mojo -o /tmp/offload_checkpoint_config_smoke`
- `/tmp/offload_checkpoint_config_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/training/loop_levers_integration_smoke.mojo -o /tmp/loop_levers_integration_smoke`
- `/tmp/loop_levers_integration_smoke`
- `timeout 180 prlimit --as=48000000000 /tmp/klein_stack_lora_real_smoke_split12`
  outside sandbox: resident full-stack smoke reached CUDA and OOMed, so it is
  not the Klein 9B production path on this GPU.
- `timeout 180 prlimit --as=48000000000 /tmp/train_klein_real_ot_save_keys serenitymojo/configs/klein9b.json 1 0 - nosample_profile`
  outside sandbox: split-12 offload train step PASS; forward/backward/AdamW/save
  completed with `144` saved adapter pairs and raw OneTrainer key header PASS.
- `python3 scripts/check_qwenimage_contract.py`
- `python3 scripts/check_qwenimage_contract.py --strict-train-boundaries` fails
  as expected with `58` named Qwen train-boundary blockers, grouped as
  cache-inputs, noise-target-loss, timestep-rope, modulation, activation-tape,
  and gradient-tape.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/models/qwenimage/parity/qwen_train_ref_artifact_smoke.mojo`
  Qwen OneTrainer one-step artifact gate PASS over
  `/home/alex/onetrainer-mojo/parity/qwen_train_ref_step000.safetensors` and
  `/home/alex/onetrainer-mojo/parity/qwen_train_ref_step000_adapters.safetensors`,
  including header contract plus scalar payload checks for loss and representative
  adapter F32 values.
- `python3 scripts/check_qwen_replay_readiness.py`
  Qwen replay-readiness PASS now requires the artifact-consuming Mojo gate; the
  remaining replay gaps are numeric predict/loss, LoRA backward, AdamW update,
  and speed replay over the dump values.
- `python3 scripts/check_qwenimage_lora_keys.py --strict-port`
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/qwen_train_control_wiring_smoke.mojo`
  Qwen train-control wiring PASS after moving `DeviceContext()` behind config
  parsing and `only_cache`, with LoRA state-sidecar naming covered by the smoke.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/models/qwenimage/parity/qwen_lora_ot_save_key_smoke.mojo`
  outside sandbox: Qwen OneTrainer save-key smoke PASS.
- `timeout 180 prlimit --as=12000000000 --cpu=180 pixi run mojo run -I . serenitymojo/models/qwenimage/parity/qwen_lora_resume_state_smoke.mojo`:
  Qwen LoRA resume-state smoke PASS. Raw OneTrainer weight resume reloads A/B
  with zeroed AdamW moments; trainer-state resume reloads A/B plus AdamW
  moments.
- `python3 scripts/check_ernie_contract.py`
- `python3 scripts/check_ernie_contract.py --strict-train-boundaries` fails as
  expected with `48` named Ernie train-boundary blockers.
- `python3 scripts/check_ernie_lora_keys.py --strict-port`
- `python3 scripts/check_ernie_lora_keys.py --strict-port --safetensors /tmp/ernie_lora_ot_save_key_smoke.safetensors --smoke-dims`:
  Ernie reduced raw-key header PASS with `1` layer / `7` adapters / `21` BF16
  tensors.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/ernie_train_control_wiring_smoke.mojo`:
  ERNIE train-control wiring PASS after the config-first entrypoint update,
  including config-driven cache/output paths and `only_cache` pre-CUDA exit.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/models/ernie/parity/ernie_lora_ot_save_key_smoke.mojo`
  outside sandbox: Ernie OneTrainer save-key smoke PASS.
- `python3 scripts/check_anima_contract.py`: Anima loader/storage report gate
  passes with `0` loader blockers while naming `67` train-boundary blockers.
  `--strict-train-boundaries` fails as expected until the Anima train/sample
  paths stop using production host-F32 cache, activation, LoRA-grad, optimizer,
  prepare, and denoise carriers.
- `timeout 120 prlimit --as=34359738368 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/training/anima_train_control_wiring_smoke.mojo`:
  Anima train-control wiring PASS. It covers config-driven rank/alpha/lr,
  checkpoint/cache/output/save/sample cadence, validation prompt fallback,
  disabled sampling, save-before-sample decisions, LoRA state-sidecar naming,
  and fail-loud unsupported CPU/offload controls.
- `python3 scripts/check_anima_lora_keys.py --strict-port`: Anima raw
  OneTrainer LoRA save/resume source contract PASS.
- `python3 scripts/check_anima_lora_keys.py --strict-port --use-baseline`:
  Anima production raw-key header PASS against `/home/alex/OneTrainer-anima-ref`
  with `28` blocks / `280` adapters / `840` BF16 tensors.
- `python3 scripts/check_anima_lora_keys.py --strict-port --profile smoke --safetensors /tmp/anima_lora_ot_save_key_smoke.safetensors`:
  Anima reduced raw-key header PASS with `1` block / `10` adapters / `30` BF16
  tensors.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/models/anima/parity/anima_lora_ot_save_key_smoke.mojo`:
  Anima raw OneTrainer LoRA key smoke PASS, including
  `load_anima_lora_resume`.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/training/full_finetune_save_smoke.mojo`
  outside sandbox: shared full-finetune tensor save/load smoke PASS, including
  BF16 raw-byte preservation after load and a U8 safetensors tensor-name
  manifest for binding full-weight names to `param.N` optimizer sidecar order.
  This is shared scaffolding only; per-model reload mapping remains separate
  work.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/full_finetune_contract_smoke.mojo`:
  full-finetune no-CUDA contract PASS. It maps Qwen, Flux.1, Flux2/Klein,
  Z-Image, Chroma, SDXL, SD3.5, Ernie, and Anima full-finetune targets,
  keeps Anima on `/home/alex/OneTrainer-anima-ref`, pins optimizer sidecar keys,
  and keeps all real product loops marked unsupported until full-weight math is
  wired.
- `python3 scripts/check_full_finetune_contracts.py`:
  full-finetune source/readiness audit report-only PASS. Shared blockers are
  `0`; real-loop full-finetune blockers remain for the target families.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/training/parity/loop_parity.mojo`:
  generic `TrainState` optimizer/master sidecar PASS. It round-trips F32
  masters and AdamW moments byte-exact, restores optimizer step `50`, and
  resumes with matching loss then continued descent.
- `python3 scripts/check_sdxl_lora_keys.py --strict-port`: OneTrainer SDXL
  baseline header validates as `2382` BF16 tensors / `794` adapters. Strict
  mode now passes for the current contract because the Mojo surface marks the
  implemented SpatialTransformer linears separately from unsupported targets
  that must fail loud until full math/save coverage is wired.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/models/sdxl/parity/sdxl_lora_ot_save_key_smoke.mojo`:
  SDXL raw OneTrainer save-key smoke PASS for the implemented
  SpatialTransformer adapter surface, including `load_sdxl_lora_resume`.
- `timeout 180 prlimit --as=16000000000 --cpu=180 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/models/sdxl/parity/sdxl_lora_resume_state_smoke.mojo`:
  SDXL LoRA state-resume smoke PASS. Raw OneTrainer weight resume reloads A/B
  with zeroed AdamW moments; trainer-state resume reloads A/B plus AdamW
  moments.
- `python3 scripts/check_flux_lora_keys.py --strict-port`: OneTrainer
  Flux.1-dev baseline header validates as `1512` BF16 tensors / `504`
  adapters with exact groups `double=228`, `double_norm=38`, `single=190`,
  `single_norm=38`, and `stack=10`. Strict mode now passes for the current
  contract: `418` transformer adapters are saveable, `86` missing OT
  transformer targets are explicit fail-loud gaps, and text-encoder LoRA is also
  fail-loud until implemented. The missing-target report lists `10` stack
  targets, `38` double-block norm-modulation targets, and `38` single-block
  norm-modulation targets by OneTrainer module path.
- `python3 scripts/check_flux_onestep_dump_contract.py`: Flux.1-dev one-step
  dump source contract PASS. It verifies the local OneTrainer baseline config,
  scheduler config, LoRA artifact, Flux predict/loss/noise/flow setup anchors,
  and GenericTrainer first-step order. It reports the required but still missing
  numeric replay artifacts `/tmp/ot_flux1_step1_inputs.safetensors` and
  `/tmp/ot_flux1_step1_inputs_manifest.json`.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/flux_train_control_wiring_smoke.mojo`:
  Flux.1-dev train-control wiring PASS with the local OneTrainer baseline LoRA
  scale (`rank=16`, `alpha=16`) and accepted `CPU_OFFLOADED` checkpoint policy
  for the block-swap/offload trainer path. This is policy/wiring evidence only;
  it is not OneTrainer activation/layer `CPU_OFFLOADED` parity, full
  predict/backward/AdamW replay, speed parity, or GPU parity.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/models/flux/parity/flux_lora_ot_save_key_smoke.mojo`:
  Flux raw OneTrainer save-key smoke PASS for the implemented block-projection
  adapter surface, including exact reduced-smoke prefix inventory and
  `load_flux_lora_resume`.
- `python3 scripts/check_flux_lora_keys.py --strict-port --safetensors /tmp/flux_lora_ot_save_key_smoke.safetensors --expected-adapters 17 --expected-rank 2`:
  Flux reduced smoke header PASS with `51` BF16 tensors / `17` adapters
  (`12` double, `5` single), all using raw OneTrainer alpha/down/up suffixes.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/models/flux/parity/flux_lora_resume_state_smoke.mojo`:
  Flux LoRA state-resume smoke PASS. Raw OneTrainer weight resume reloads A/B
  with zeroed AdamW moments; trainer-state resume reloads A/B plus AdamW
  moments.
- `timeout 180 prlimit --as=24000000000 pixi run mojo build -I . -Xlinker -lm serenitymojo/models/flux/parity/lora_step_smoke.mojo -o /tmp/flux_lora_step`
  and `timeout 180 prlimit --as=24000000000 /tmp/flux_lora_step`: Flux reduced
  LoRA training-step smoke PASS. It runs forward/backward/clip/AdamW/save/load,
  reports finite grads, updates all `34/34` implemented adapters from zero B,
  and round-trips A/B byte-exact through raw OneTrainer save/load.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/ops/parity/rope_struct_bwd_parity.mojo`:
  RoPE/QKV/gate backward parity PASS against PyTorch after the mixed-dtype
  RoPE table and gate-residual backward fixes.
- `python3 scripts/check_chroma_lora_keys.py --strict-port`: OneTrainer Chroma
  baseline header validates as `912` BF16 tensors / `304` adapters for the
  local `attn,ff.net` layer filter. Strict mode now passes for that selected
  baseline while still warning that `lora_te`, stack-level, and
  distilled-guidance targets are not implemented for configs that select them.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/sampling/onetrainer_sampler_contract_smoke.mojo`:
  shared OneTrainer sampler contract PASS. It gates sample defaults, resolution
  quantization, scheduler mode, timestep mode, CFG mode, and staged
  text/transformer/VAE expectations for the target image models.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_lifecycle_smoke.mojo`:
  shared OneTrainer lifecycle contract PASS. It pins `TrainProgress`,
  `TimedActionMixin` step/epoch/time semantics, product runner selection, and
  fail-loud behavior for unsupported full-finetune and optimizer product paths.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_product_run_smoke.mojo`:
  shared OneTrainer `scripts/train.py` entrypoint contract PASS. It checks
  config/concept/sample file presence, product runner dispatch for Qwen, Ernie,
  Anima, Klein, Z-Image, Chroma, Flux.1-dev, SD3.5, and SDXL, workspace
  config/save/backup/sample dirs, validation/sampler creation flags, cache
  preflight binding, `continue_last_backup` latest-backup discovery, cold-start
  behavior when no backup exists, fail-loud missing LoRA optimizer state,
  fail-loud unsupported full-finetune or optimizer product paths, fail-loud raw
  VAE `only_cache` gaps, plus the resolved Mojo runner command. This is not a
  model-loop execution gate.
- `timeout 180 prlimit --as=12000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_cache_preflight_smoke.mojo`:
  product cache preflight PASS. It validates all current target model
  text/VAE cache contracts in no-CUDA mode, accepts raw-cache-ready Qwen and
  SD3 `only_cache`, and blocks raw-cache-missing Klein, Flux2-dev, Flux.1-dev,
  and Chroma `only_cache`.
- `python3 scripts/check_onetrainer_cache_preflight_contract.py`: product cache
  preflight static guard PASS. It checks shared contract imports, the fail-loud
  `only_cache` raw VAE guard, product dry-run/product-smoke exposure, local-only
  reference scope, and forbidden F32 storage-boundary markers.
- `python3 scripts/check_train_loop_cache_contract_bindings.py --marker-limit 1`:
  train-loop binding inventory PASS in report-only mode. It confirms the nine
  current real loops import/read/validate `TrainConfig`, return for `only_cache`
  before `DeviceContext()`, wire step sample cadence, and validate the shared
  text/VAE cache preflight before CUDA setup. All nine loops still report
  host-F32 cache/activation markers. This is a gap report, not a parity pass.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_train_dry_run.mojo /tmp/ot_product_qwen_config.json`:
  no-CUDA product entrypoint dry-run PASS. It validates a config and prints the
  resolved `train_qwenimage_real.mojo <config> <max_steps>` command plus
  text/VAE cache readiness without spawning the model loop or creating CUDA
  context.
- `python3 scripts/check_onetrainer_resume_contract.py` and
  `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_resume_contract_smoke.mojo`:
  shared OneTrainer checkpoint/resume contract PASS. This is metadata/control
  only: it pins save/backup/sample naming, cadence, internal LoRA sidecar
  expectations, optimizer-state-required resume, and fail-loud unsupported
  full-finetune/incompatible model or training-method surfaces. The static guard
  also checks that the product entrypoint exposes the `continue_last_backup`
  preflight without adding F32 tensor-boundary markers.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/zimage_train_control_wiring_smoke.mojo`:
  Z-Image train-control wiring PASS. The real train loop reads `TrainConfig`,
  applies checkpoint/cache/output/sample cadence policy where supported, returns
  early for `only_cache`, and fails loudly for unsupported full-finetune,
  ADAFACTOR, CPU-offload, non-step cadence, or mismatched compiled recipe
  constants.
- `python3 scripts/check_zimage_sampler_contract.py` and
  `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/sampling/zimage_sampler_contract_smoke.mojo`:
  Z-Image sampler/setup source contract PASS. It pins the local OneTrainer
  FlowMatch scheduler, default sample dimensions/steps/CFG, latent scale/unscale,
  no external latent pack/unpack, transformer input rank, CFG formula, and
  training flow target. It explicitly remains a contract scaffold, not denoise,
  image, speed, or VRAM parity.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/chroma_train_control_wiring_smoke.mojo`:
  Chroma train-control wiring PASS. The real train loop reads `TrainConfig`,
  applies checkpoint/cache/output/sample cadence policy for its supported
  single-file checkpoint path, returns early for `only_cache` before CUDA init,
  and fails loudly for sharded checkpoint dirs, unsupported full-finetune,
  ADAFACTOR, CPU-offload, non-step cadence, or mismatched compiled recipe
  constants.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/flux_train_control_wiring_smoke.mojo`:
  Flux.1-dev train-control wiring PASS. The real train loop reads
  `TrainConfig`, applies checkpoint/cache/output/sample cadence policy, returns
  early for `only_cache` before CUDA init, and fails loudly for unsupported
  full-finetune, ADAFACTOR, CPU-offload, non-step cadence, or mismatched
  compiled recipe constants. The update path now threads parsed AdamW `beta1`,
  `beta2`, `eps`, `weight_decay`, `lr`, and clip values.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/sd35_train_control_wiring_smoke.mojo`:
  SD3.5 train-control wiring PASS. The bounded train loop reads
  `TrainConfig`, applies single-file checkpoint/cache/output/sample cadence
  policy, returns early for `only_cache` before CUDA init, emits LoRA state
  sidecar calls at save/sample cadence, and fails loudly for sharded
  checkpoints, unsupported full-finetune, ADAFACTOR, CPU-offload, non-step
  cadence, resume args, or mismatched compiled recipe constants.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/sdxl_train_control_wiring_smoke.mojo`:
  SDXL train-control wiring PASS. The real train loop reads `TrainConfig`,
  applies checkpoint/cache/output/sample cadence policy, derives
  per-SpatialTransformer LoRA output paths, threads parsed AdamW `beta1`,
  `beta2`, `eps`, `weight_decay`, `lr`, and clip into the update path, returns
  early for `only_cache` before CUDA init, and fails loudly for unsupported
  full-finetune, ADAFACTOR, CPU-offload, non-step cadence, or mismatched
  compiled recipe constants.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/ops/parity/linear_mixed_bias_smoke.mojo`:
  mixed F32-activation plus BF16/F16 checkpoint weight+bias forward/backward
  smoke PASS.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/ops/parity/group_norm_mixed_bwd_smoke.mojo`:
  mixed F32 activation/upstream-grad plus BF16/F16 GroupNorm weight backward
  smoke PASS. It returns F32 activation grads and BF16/F16 parameter grads.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/ops/parity/norm_bwd_parity.mojo`:
  norm backward parity PASS against PyTorch after the mixed GroupNorm branch.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/ops/parity/conv2d_mixed_dtype_smoke.mojo`:
  mixed F32 activation/upstream-grad plus BF16/F16 conv weight/bias forward and
  backward smoke PASS. It returns F32 outputs/d_x and BF16/F16 d_w/d_b.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/ops/parity/conv2d_bwd_parity.mojo`
  and `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/ops/parity/conv2d_bwd_s2_parity.mojo`:
  same-dtype conv backward parity PASS against PyTorch for stride 1 and stride 2
  after the mixed conv branch.
- `timeout 180 prlimit --as=16000000000 pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_sdxl_real.mojo -o /tmp/train_sdxl_real_compile_check`:
  SDXL real trainer build PASS after the config-driven AdamW threading and SDXL
  embedding/ST/final-GN/ResBlock-GN/ResBlock-embedding-linear/conv
  checkpoint-dtype loader slice.
- `python3 scripts/check_sampler_product_harness_contract.py` and
  `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/sampling/product_sampler_harness_smoke.mojo`:
  product sampler harness contract PASS. This requires OneTrainer seconds/step,
  peak VRAM, Mojo per-stage timing, Mojo seconds/step, and Mojo peak VRAM before
  sampler speed parity can be accepted. It is a measurement scaffold, not image
  or speed parity by itself.
- `python3 scripts/audit_onetrainer_optimizers.py`:
  local OneTrainer optimizer audit PASS. The target presets/configs use explicit
  ADAMW/ADAFACTOR or missing/default ADAMW.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/optimizer_dispatch_smoke.mojo`:
  OneTrainer optimizer identifier dispatch PASS for ADAMW/default, ADAFACTOR,
  and fail-closed unsupported/aliased identifiers.
- `python3 scripts/check_onetrainer_train_math_contract.py --strict-product`:
  shared train-math contract PASS. It validates the local OneTrainer AdamW
  fixed-input ref, LR refs, masked-loss scaling, optimizer dispatch/defaults,
  fixed update smoke presence, product-loop binding, adapter-consumer gates, and
  static AdamW order: decoupled weight decay before adaptive subtraction.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/optimizer_fixed_update_smoke.mojo`:
  fixed optimizer update gate PASS. It pins AdamW and Adafactor update numbers,
  AdamW resume equivalence, cosine LR scalar behavior, and BF16 param/grad
  storage preservation while keeping AdamW moments F32.
- `python3 scripts/check_flux_family_sampler_contracts.py`: Flux.1-dev,
  Flux2/Klein, and Chroma sampler source/static dtype-boundary contract PASS
  against local OneTrainer sampler/setup files and scheduler configs.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/sampling/flux1_dev_smoke.mojo`:
  FLUX.1-dev scheduler/pack smoke PASS. It validates the local OneTrainer
  FlowMatch scheduler config and pins sigma, timestep, guidance-embed,
  single-batch CFG, and Euler scalar contracts. This is not denoise/image/speed
  parity.
- `python3 scripts/check_onetrainer_conditioning_contracts.py`: text
  conditioning source/static contract PASS against OneTrainer plus
  OneTrainer-anima-ref markers. This checks tokenizer lengths, prompt templates,
  mask/crop modes, selected layer conventions, cache fields, and forbidden F32
  storage-boundary markers in the contract helpers.
- `timeout 180 prlimit --as=12000000000 pixi run mojo run -I . serenitymojo/models/text_encoder/onetrainer_conditioning_contract_smoke.mojo`:
  shared OneTrainer text-conditioning contract smoke PASS for SDXL, SD3.5,
  Qwen, Ernie, Anima, Flux.1-dev, Flux2 dev, Flux2/Klein, Chroma, and Z-Image.
- `python3 scripts/check_vae_sampler_contracts.py`: VAE/postprocess source
  contract PASS against local OneTrainer and OneTrainer-anima-ref for SDXL, SD3,
  Qwen, Ernie, Anima, Flux.1-dev, Flux2/Klein, Chroma, and Z-Image. This pins
  VAE encode/precache source markers, raw cache versus prepared latent shapes,
  VAE scale/unscale formulas, latent pack/patch/unpack layout, frame squeezing,
  resolution quantization, and image postprocess mode, but not real encode/decode
  pixels.
- `timeout 180 prlimit --as=12000000000 pixi run mojo run -I . serenitymojo/sampling/vae_encoder_contract_smoke.mojo`:
  shared VAE encoder/cache contract PASS, including fail-loud Flux2/Klein raw
  cache blocking for the current prepared-only Mojo encoder.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/sampling/vae_postprocess_contract_smoke.mojo`:
  shared VAE/postprocess contract PASS.
- Sampler helper smokes PASS:
  `flux1_dev_smoke`, `flux2_klein_smoke`, `chroma1_hd_smoke`,
  `qwenimage_sampling_smoke`, `sdxl_euler_smoke`, `sd3_flow_match_smoke`,
  `ernie_sampling_smoke`, `anima_sampling_smoke`, and `flow_match_smoke`.
  These prove scheduler/CFG/update contracts only; full text conditioning,
  denoise, VAE decode, image output, speed, and VRAM parity remain separate.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/models/chroma/parity/chroma_lora_ot_save_key_smoke.mojo`:
  Chroma raw OneTrainer save-key smoke PASS for the current block-adapter
  surface, including `load_chroma_lora_resume`.
- `timeout 180 prlimit --as=16000000000 pixi run mojo run --target-accelerator sm_86 -I . serenitymojo/models/chroma/parity/chroma_lora_resume_state_smoke.mojo`:
  Chroma LoRA state-resume smoke PASS. Raw OneTrainer weight resume reloads A/B
  with zeroed AdamW moments; trainer-state resume reloads A/B plus AdamW
  moments.
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm serenitymojo/models/flux/parity/load_flux_weights_smoke.mojo -o /tmp/load_flux_weights_smoke_dtype`
- `/tmp/load_flux_weights_smoke_dtype` outside sandbox: Flux real checkpoint shape smoke PASS.
- `/tmp/norm_bwd_parity_dtype_sm86`: all norm backward gates PASS at cosine >= 0.999.
- `/tmp/qwenimage_real_smoke_dtype`: Qwen-Image real-dim forward/backward finite, LoRA grads nonzero, AdamW applied.
- `pixi run mojo build --target-accelerator sm_86 -I . serenitymojo/models/qwenimage/parity/qwen_fp8_loader_smoke.mojo -o /tmp/qwen_fp8_loader_smoke`
- `/tmp/qwen_fp8_loader_smoke` outside sandbox: real Qwen `F8_E4M3` checkpoint tensor dequantized to BF16 PASS.
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_qwenimage_real.mojo -o /tmp/train_qwenimage_real_current`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_chroma_real.mojo -o /tmp/train_chroma_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib -Xlinker -lsqlite3 serenitymojo/training/train_klein_real.mojo -o /tmp/train_klein_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_zimage_real.mojo -o /tmp/train_zimage_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_l2p_real.mojo -o /tmp/train_l2p_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_anima_real.mojo -o /tmp/train_anima_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_ernie_real.mojo -o /tmp/train_ernie_real_dtype`
- `pixi run mojo build --target-accelerator sm_86 -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/training/train_sd35_real.mojo -o /tmp/train_sd35_real_current`

The Flux smoke had to run outside the sandbox because CUDA `DeviceContext`
initialization failed under sandbox NVML.
