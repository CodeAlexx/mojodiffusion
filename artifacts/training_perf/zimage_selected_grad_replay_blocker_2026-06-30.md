# ZImage Selected Grad Replay Blocker

Date: 2026-06-30

Evidence level: next-gate blocker/design note plus full-depth all-trainable
replay evidence. This is not product-loop parity and not strict BF16 activation
storage. It records the smallest honest replay target after the real device
loss-root replay, the adapter oracle metadata smoke, and the selected gradient
preflight.

Target:

- Step dump:
  `/home/alex/serenity-trainer/parity/zimage_train_ref_step000.safetensors`
- Adapter dump:
  `/home/alex/serenity-trainer/parity/zimage_train_ref_step000_adapters.safetensors`
- Proposed Mojo gate:
  `serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo`
- Proposed command:
  `pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo`
- Opt-in full-depth command:
  `pixi run mojo run -D ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH=30 -I . serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo`
- External observed VRAM command:
  `python3 scripts/run_zimage_selected_grad_replay_vram.py --no-echo`
- Current preflight artifact:
  `artifacts/training_perf/zimage_selected_grad_replay_preflight_2026-06-30.md`
- Current external VRAM artifact:
  `artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json`

Required replay shape:

- Full batch-2 forward/backward is required before selected adapter gradients are
  comparable. A layer-only replay is not comparable because the OneTrainer dump
  does not include intermediate activations or upstream block gradients.
- Use the non-StepIO streamed B2 path:
  `zimage_stack_lora_forward_main_device_b2_masked_streamed` and
  `zimage_stack_lora_backward_main_device_b2_masked_streamed`.
- Selected replay must use streamed masked B2 APIs; resident masked B2 APIs are not accepted for 24GB because they require resident main base block lists.
- Do not use the B=1 StepIO path for adapter grad comparison. It can replay a
  single sample loss root, but the dumped adapter grads are batch-2 mean grads.
- A real-input bounded streamed smoke now runs through the masked B=2 path with
  the actual OneTrainer step0 BF16 latents/text tensors, real aux/embedder
  weights, four streamed frozen refiner blocks, per-sample image RoPE, and
  zero streamed main blocks.
- The opt-in full-depth gate now extends that streamed/offload integration to
  all `30` main blocks, builds real loss-root gradients, avoids a resident main
  base block list, and compares all `420` adapter-gradient tensors against the
  adapter dump.
- The external wrapper now records selected-replay VRAM with `nvidia-smi`
  sampling: `streamed_b2_selected_replay_peak_vram_bytes=22567452672`,
  `external_peak_vram_delta_mib=21522`, and `sample_count=203`. It still does
  not cover the product loop.
- The training pad mask contract is still part of the acceptance surface:
  sample1's shorter caption/unified rows must remain masked with OneTrainer's
  semantics throughout the streamed replay.

Step-0 tensor geometry:

- `zimage_train_ref_step000.safetensors` has `42` tensors.
- `batch.latent_image`: BF16 `[2,16,64,64]`
- `latent_input`: BF16 `[2,16,1,64,64]`
- `flow_target` / `predicted_flow`: BF16 `[2,16,64,64]`
- `batch.text_encoder_hidden_state`: BF16 `[2,512,2560]`
- exact caption rows: sample0 `145`, sample1 `127`
- OneTrainer pad-to-32 caption rows: sample0 `160`, sample1 `128`
- exact unified sequence rows before final batch padding: sample0 `1184`,
  sample1 `1152`
- batch max sequence rows: `1184`; sample1 has `32` masked unified rows
- product trainer bucket `CAP=224` is not strict step0 OneTrainer geometry; it
  would add masked rows `160..223` for sample0 and `128..223` for sample1
- `text_encoder_output_0`: BF16 `[145,2560]`
- `text_encoder_output_1`: BF16 `[127,2560]`
- `timestep`: F32 `[2]`
- `sigma`: F32 `[2,1,1,1]`
- per-sample image rows: `1024`
- output channels: `64`
- batch rows: `2048`
- MSE elements: `131072`

Grad comparison target:

- Compare adapter-gradient tensors against `adapter_post_clip_grad.*` from
  `zimage_train_ref_step000_adapters.safetensors`.
- The adapter dump has `3360` F32 tensors: `8` phases x `420` trainable
  tensors.
- The `420` trainable tensors are `30` layers x `7` LoRA sites x down/up.
- One selected layer has `14` tensors and `1167360` elements:
  `attention.to_q`, `attention.to_k`, `attention.to_v`,
  `attention.to_out.0`, `feed_forward.w1`, `feed_forward.w2`, and
  `feed_forward.w3` down/up tensors.
- The full-depth replay now compares all `420` trainable-gradient tensors and
  `35020800` elements. The selected layer-0 subset remains printed as a stable
  comparison anchor.

Dtype boundary:

- OneTrainer's ZImage base/train/output path is BF16 for this config. The train
  reference step tensors confirm BF16 latents, text embeddings, predicted flow,
  and flow target.
- OneTrainer's live LoRA params are `FLOAT_32` for this config because
  `lora_weight_dtype` is absent and defaults to `FLOAT_32`; the adapter dump
  stores those live trainable params and gradient targets for comparison.
- Normal OneTrainer final LoRA export uses `output_dtype=BFLOAT_16`.
- Mojo replay must keep adapter/device storage boundaries BF16 in/out. F32 is
  allowed only inside compute kernels/reductions or for host-side comparison
  after readback.
- The previous F32 device-carrier interpretation is invalid for this runtime and
  must not be used to claim selected-gradient replay progress.
- The oracle metadata smoke is now covered by
  `serenitymojo/models/zimage/parity/zimage_train_ref_f32_adapter_carrier_smoke.mojo`.
- Evidence is recorded at
  `artifacts/training_perf/zimage_f32_adapter_carrier_smoke_2026-06-30.md`.
- The metadata smoke checks layer `0` `adapter_before.*` tensors with replay
  scale `0.0625`, verifies matching `adapter_post_clip_grad.*` target shapes,
  and checks `14` F32 adapter-dump tensors / `1167360` selected elements without
  creating a device context or uploading F32 LoRA carriers.
- The executable selected replay now performs the next boundary step: it uploads
  selected step tensors as BF16 device tensors, converts selected layer-0
  `adapter_before.*` F32 host dump matrices through the existing `LoraAdapter`
  BF16 storage conversion and `zimage_lora_adapter_to_device`, and asserts all
  `14` selected adapter device tensors remain BF16.
- It also opens the real sharded transformer directory and loads only
  `layers.0` through the preserve-dtype mixed block loader, asserting the `13`
  selected base block device tensors remain BF16. This is a
  `stream_prereq=single_block_load` gate, not a resident full-stack replay.
- Streamed masked B2 currently has
  `activation_carrier_dtype=F32 not_strict_BF16_step_storage` and
  `adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only`.
  The hard boundaries remain `checkpoint_boundary=BF16 step_input_boundary=BF16`.

Selected-gradient replay preflight:

- Executable preflight is now covered by
  `serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo`.
  This selected-gradient replay preflight is a blocker gate, not gradient parity.
- Evidence is recorded at
  `artifacts/training_perf/zimage_selected_grad_replay_preflight_2026-06-30.md`.
- The preflight verifies step0 tensors, the `3360` F32 adapter phase tensors,
  selected layer-0 `adapter_before.*` and `adapter_post_clip_grad.*` surfaces,
  `14` selected tensors / `1167360` selected elements, and the exact padded OT
  geometry: `145 -> 160`, `127 -> 128`, `seq=(1184,1152)`, `max_seq=1184`.
- It verifies real checkpoint access with
  `base_block_ingest PASS checkpoint_boundary=BF16`, `transformer_tensors=521`,
  and `base_block_device_tensors=13`.
- It now also runs a bounded real-input streamed smoke with
  `evidence=real-input-bounded-smoke`,
  `streamed_refiner_blocks=4 streamed_main_blocks=0`,
  `prepared_main_mod_b2=30`, `cap_attn_len=(160,128)`,
  `main_attn_len=(1184,1152)`, `x_rope_per_sample=true`, and
  `observed_vram_mib_lower_bound=372.00146484375`.
- The opt-in full-depth define provides
  `streamed_b2_selected_replay_blocks=30`,
  `streamed_b2_selected_replay_no_resident_main_blocks=true`, and all-trainable
  comparison against `adapter_post_clip_grad.*`.
  `all_trainable_grad_tensors=420`, `all_trainable_grad_numel=35020800`, and
  `all_trainable_grad_max_abs=3.6748774618899915e-06` is below
  `all_trainable_grad_tol=1e-05`. The retained selected layer-0 subset is
  `selected_layer0_grad_max_abs=8.392975701099203e-07`.
- It still prints
  `streamed_b2_selected_replay_peak_vram_bytes_missing=true`; the lower-bound
  memory sample is not a true peak monitor. The external wrapper fills that
  evidence gap for the full-depth all-trainable replay with
  `streamed_b2_selected_replay_peak_vram_bytes=22567452672` in
  `zimage_selected_grad_replay_vram_2026-06-30.json`.

Shared SDPA mask progress:

- The shared backward prerequisite is partially covered by
  `serenitymojo/ops/parity/sdpa_bwd_batched_mask_parity.mojo`.
- Evidence is recorded at
  `artifacts/training_perf/zimage_batched_mask_sdpa_backward_2026-06-30.md`.
- `sdpa_backward_masked` now preserves legacy F32 `[H*S,S]` broadcast masks and
  also accepts full F32 `[B,H,S,S]` / `[B*H*S,S]` masks.
- `training_sdpa_backward_masked_batched_strict` is the training-wrapper entry
  point for per-sample additive masks.
- This is replay evidence only. ZImage still needs product-loop integration and
  the shared device ABI path before it can claim product training parity.
  Graph/slab remains no-mask and excluded in this slice.

Masked B2 stack progress:

- Non-graph masked B2 stack wiring is now covered by
  `artifacts/training_perf/zimage_masked_b2_stack_wiring_2026-06-30.md`.
- Implemented `zimage_key_tail_mask_f32`, `zimage_refiner_forward_masked`,
  `zimage_block_lora_forward_device_tensor_batch_masked`,
  `zimage_block_lora_backward_device_tensors_batch_masked`,
  `zimage_stack_lora_forward_main_device_b2_masked`, and
  `zimage_stack_lora_backward_main_device_b2_masked`.
- Focused smokes verify device mask construction, all-valid masked B2 LoRA
  block forward/backward equivalence to no-mask, and zero-block masked stack API
  runtime wiring.
- This is still not product training parity. The opt-in selected replay now
  loads the real step0 tensors, keeps Mojo adapter/device tensors BF16 in/out,
  runs full masked B2 forward/backward, and compares all `420`
  adapter-gradient tensors against the F32 host-oracle `adapter_post_clip_grad.*`
  tensors.

Remaining blocker:

- Add masked graph/slab support or fail-loud exclusion for graph B2, which is
  currently no-mask.
- graph/slab B2 remains no-mask; masked selected replay must not use
  `zimage_stack_lora_backward_main_device_b2_graph`.
- Wire the streamed replay into the shared device train-step/product loop and
  collect product-loop steady-state speed/VRAM evidence.
- Preserve the exact OT batch geometry and caption padding. Do not use `CAP=145`
  as an unpadded shortcut, and do not use product `CAP=224` rows unless their
  extra rows are masked with the same semantics as OneTrainer.

Acceptance boundary:

- Accepted replay now runs masked full B=2 forward/backward on step0 tensors,
  handles exact OneTrainer caption padding and batch masking, loads
  `adapter_before.*` as host oracle comparison data, keeps Mojo adapter/device
  storage BF16, and compares all `420` adapter-gradient tensors.
