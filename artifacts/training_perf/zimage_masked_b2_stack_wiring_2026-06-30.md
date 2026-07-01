# ZImage Masked B2 Stack Wiring

Date: 2026-06-30

Evidence level: masked non-graph B2 wiring smoke. This is not OneTrainer
selected adapter-gradient parity and not product speed evidence.

Implemented:

- `zimage_key_tail_mask_f32[B,H,S]`
  - builds F32 additive `[B,H,S,S]` key-tail masks on device
  - supports per-sample valid lengths such as ZImage step0 unified `1184/1152`
- `zimage_refiner_forward_masked`
  - frozen context-refiner forward with additive attention mask
- `zimage_block_lora_forward_device_tensor_batch_masked`
  - B2 LoRA block forward with additive `[B,H,S,S]` mask
- `zimage_block_lora_backward_device_tensors_batch_masked`
  - B2 LoRA block backward with F32 batched mask via
    `training_sdpa_backward_masked_batched_strict`
- `zimage_stack_lora_forward_main_device_b2_masked`
  - non-graph B2 stack forward with caption-refiner masks and unified main mask
- `zimage_stack_lora_backward_main_device_b2_masked`
  - non-graph B2 stack backward/recompute using the same unified main mask
- `zimage_stack_lora_forward_main_device_b2_masked_streamed`
  - non-graph B2 stack forward that streams frozen base block weights one block
    at a time from `ShardedSafeTensors` instead of taking resident block lists
  - accepts per-sample image RoPE tables so ragged caption padding can use the
    correct OneTrainer image axis-0 offsets (`161` for sample0, `129` for
    sample1 in the step0 dump)
- `zimage_stack_lora_backward_main_device_b2_masked_streamed`
  - non-graph B2 stack backward/recompute that reloads one frozen main block at
    a time in reverse using `saved.main_x_in`; it avoids a resident full-stack
    base-weight list and keeps checkpoint tensors in their stored dtype
  - selected replay must use streamed masked B2 APIs; resident masked B2 APIs
    are not accepted for 24GB
- `zimage_stack_lora_backward_main_device_b2_masked_streamed_adamw_device_grads`
  - sibling of the streamed masked B2 backward that copies transient per-block
    device dA/dB tensors into `LoraAdamWPlainDeviceState.dev_g` with
    `lora_adamw_plain_device_state_copy_device_grad_pair`
  - converts only at the optimizer-gradient boundary with
    `_zimage_device_grad_f32`; LoRA/checkpoint storage stays BF16 and the flat
    AdamW grad buffer stays F32
  - fails before returning if `grad_count != opt_state.end - opt_state.start`
    or `streaming_sync_count != num_main`

Commands:

```bash
pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_b2_attention_mask_smoke.mojo
pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_b2_masked_lora_block_smoke.mojo
pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_b2_masked_stack_compile_smoke.mojo
pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_b2_masked_streamed_stack_compile_smoke.mojo
```

Observed output:

```text
PASS: zimage B2 key-tail attention masks
PASS: ZImage B2 masked LoRA block all-valid mask matches no-mask
PASS: ZImage masked B2 stack APIs compile and run zero-block smoke
PASS: ZImage streamed masked B2 stack APIs compile and run zero-block smoke  transformer_tensors= 521  streamed_blocks=0 evidence=compile-runtime-smoke  checkpoint_boundary=BF16 step_input_boundary=BF16  activation_carrier_dtype=F32 not_strict_BF16_step_storage  adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only  b2_device_grad_sibling=zero_block_static_runtime
```

The selected-gradient replay preflight also runs a real-input bounded streamed
smoke through this API:

```text
[zimage-selected-grad-replay] real_streamed_input_smoke PASS  evidence=real-input-bounded-smoke  step_boundary=BF16 checkpoint_boundary=BF16  streamed_refiner_blocks=4 streamed_main_blocks= 0  prepared_main_mod_b2= 30  depth_define=ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH  cap_attn_len=( 160 , 128 ) main_attn_len=( 1184 , 1152 ) x_rope_per_sample=true  activation_carrier_dtype=F32 not_strict_BF16_step_storage  observed_vram_mib_lower_bound= 372.00146484375  selected_layer0_grad_max_abs= -1.0  peak_vram_bytes_missing=true
```

The opt-in full-depth all-trainable replay also runs through this streamed
API:

```text
[zimage-selected-grad-replay] full_selected_grad_replay PASS  evidence=full-depth-all-trainable-grad-replay  step_boundary=BF16 checkpoint_boundary=BF16  streamed_refiner_blocks=4 streamed_main_blocks= 30  streamed_b2_selected_replay_blocks= 30  streamed_b2_selected_replay_no_resident_main_blocks=true  prepared_main_mod_b2= 30  depth_define=ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH  cap_attn_len=( 160 , 128 ) main_attn_len=( 1184 , 1152 ) x_rope_per_sample=true  all_trainable_grad_tensors= 420  all_trainable_grad_numel= 35020800  all_trainable_grad_max_abs= 3.6748774618899915e-06  selected_layer0_grad_max_abs= 8.392975701099203e-07  all_trainable_grad_tol= 1e-05  activation_carrier_dtype=F32 not_strict_BF16_step_storage  observed_vram_mib_lower_bound= 1479.42333984375  streamed_b2_selected_replay_peak_vram_bytes_missing=true
```

The external VRAM wrapper records observed peak-delta bytes for that full-depth
selected replay:

```text
[zimage-selected-grad-vram] PASS evidence=external-observed-vram-full-depth-all-trainable-grad-replay streamed_b2_selected_replay_peak_vram_bytes= 22567452672 external_peak_vram_delta_mib= 21522 sample_count= 203 all_trainable_grad_tensors= 420 all_trainable_grad_numel= 35020800 all_trainable_grad_max_abs= 3.6748774618899915e-06 all_trainable_grad_tol= 1e-05 selected_layer0_grad_max_abs= 8.392975701099203e-07 artifact= /home/alex/mojodiffusion/artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json
```

Dtype and memory boundary:

- `checkpoint_boundary=BF16 step_input_boundary=BF16`
- `activation_carrier_dtype=F32 not_strict_BF16_step_storage`
- `adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only`
- `b2_device_grad_sibling=zero_block_static_runtime` proves the streamed B2
  AdamW-device-grad sibling compiles and executes the zero-block range checks;
  it is not a nonzero gradient-value parity test.
- the streamed compile smoke proves API/runtime wiring only; the real-input
  bounded smoke proves step0 input construction and streamed refiner execution
  only.
- the opt-in full-depth all-trainable replay proves
  `streamed_b2_selected_replay_blocks=30`,
  `streamed_b2_selected_replay_no_resident_main_blocks=true`, and all `420`
  adapter-gradient tensor comparisons against `adapter_post_clip_grad.*`. The inline Mojo
  output still has only `observed_vram_mib_lower_bound=1479.42333984375`, while
  `scripts/run_zimage_selected_grad_replay_vram.py` adds external observed
  `streamed_b2_selected_replay_peak_vram_bytes=22567452672` for this selected
  replay. This is not product-loop steady-state VRAM evidence.

Graph/slab boundary:

- `zimage_stack_lora_backward_main_device_b2_graph` remains no-mask only.
- Autograd-v2 `OPK_SDPA` records `sdpa_nomask` and does not carry arbitrary
  `[B,H,S,S]` masks.
- Masked B2 selected-gradient replay must use the non-graph streamed masked B2 path until `record_sdpa_masked(_slab)`, masked engine dispatch, and
  fixed-address mask plumbing are implemented and parity-gated.
- graph/slab B2 remains no-mask; masked selected replay must not use
  `zimage_stack_lora_backward_main_device_b2_graph`.

Remaining selected-gradient work:

- wire the streamed replay into the shared device ABI/product train-step path
- replace the current host-origin B2 loss/root in the non-StepIO B2 path with a
  device-resident loss/root before claiming full B2 product fast path
