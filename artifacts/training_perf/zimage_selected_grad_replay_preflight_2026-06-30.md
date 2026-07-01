# ZImage Selected Grad Replay Preflight

Date: 2026-06-30

Evidence level: executable preflight, real-input bounded streamed smoke, and
opt-in full-depth all-trainable grad replay with external observed VRAM
sampling. This is not product-loop parity and not strict BF16 activation
storage.

Command:

```bash
pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo
pixi run mojo run -D ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH=30 -I . serenitymojo/models/zimage/parity/zimage_train_ref_selected_grad_replay.mojo
python3 scripts/run_zimage_selected_grad_replay_vram.py --no-echo
```

Observed default output:

```text
[zimage-selected-grad-replay] preflight PASS step_tensors= 42  adapter_tensors= 3360  selected_layer= 0  selected_tensors= 14  selected_numel= 1167360
[zimage-selected-grad-replay] exact_ot_geometry img_rows= 1024  cap_valid=( 145 , 127 ) cap_padded=( 160 , 128 ) max_cap= 160  seq=( 1184 , 1152 ) max_seq= 1184  sample1_masked_cap_rows= 32  sample1_masked_unified_rows= 32
[zimage-selected-grad-replay] bf16_ingest PASS step_boundary=BF16  adapter_dump_dtype=F32 adapter_device_boundary=BF16  selected_layer= 0  adapter_device_tensors= 14  replay_scale= 0.0625
[zimage-selected-grad-replay] base_block_ingest PASS checkpoint_boundary=BF16  transformer_tensors= 521  selected_layer= 0  base_block_device_tensors=13  stream_prereq=single_block_load
[zimage-selected-grad-replay] real_streamed_input_smoke PASS  evidence=real-input-bounded-smoke  step_boundary=BF16 checkpoint_boundary=BF16  streamed_refiner_blocks=4 streamed_main_blocks= 0  prepared_main_mod_b2= 30  depth_define=ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH  cap_attn_len=( 160 , 128 ) main_attn_len=( 1184 , 1152 ) x_rope_per_sample=true  activation_carrier_dtype=F32 not_strict_BF16_step_storage  observed_vram_mib_lower_bound= 372.00146484375  selected_layer0_grad_max_abs= -1.0  peak_vram_bytes_missing=true
[zimage-selected-grad-replay] streamed_bridge_required  forward=zimage_stack_lora_forward_main_device_b2_masked_streamed  backward=zimage_stack_lora_backward_main_device_b2_masked_streamed  resident_masked_b2_accepted=false  activation_carrier_dtype=F32 not_strict_BF16_step_storage  adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only
[zimage-selected-grad-replay] BLOCKED missing=masked_b2_streamed_forward_backward_replay_integration; non-graph masked B2 stack wiring exists, and selected step/adapters/base block now ingest with BF16 device boundaries, but the full streamed replay/comparison against adapter dump host tensors is not yet wired, so strict adapter gradient comparison is intentionally not run
```

Observed opt-in full-depth output:

```text
[zimage-selected-grad-replay] full_selected_grad_replay PASS  evidence=full-depth-all-trainable-grad-replay  step_boundary=BF16 checkpoint_boundary=BF16  streamed_refiner_blocks=4 streamed_main_blocks= 30  streamed_b2_selected_replay_blocks= 30  streamed_b2_selected_replay_no_resident_main_blocks=true  prepared_main_mod_b2= 30  depth_define=ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH  cap_attn_len=( 160 , 128 ) main_attn_len=( 1184 , 1152 ) x_rope_per_sample=true  all_trainable_grad_tensors= 420  all_trainable_grad_numel= 35020800  all_trainable_grad_max_abs= 3.6748774618899915e-06  selected_layer0_grad_max_abs= 8.392975701099203e-07  all_trainable_grad_tol= 1e-05  activation_carrier_dtype=F32 not_strict_BF16_step_storage  observed_vram_mib_lower_bound= 1479.42333984375  streamed_b2_selected_replay_peak_vram_bytes_missing=true
[zimage-selected-grad-replay] streamed_bridge_required  forward=zimage_stack_lora_forward_main_device_b2_masked_streamed  backward=zimage_stack_lora_backward_main_device_b2_masked_streamed  resident_masked_b2_accepted=false  activation_carrier_dtype=F32 not_strict_BF16_step_storage  adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only
[zimage-selected-grad-replay] BLOCKED missing=streamed_b2_selected_replay_peak_vram_bytes; full streamed all-trainable grad comparison passed, but VRAM evidence is only an in-process lower-bound sample, not a true peak monitor
```

Observed external VRAM wrapper output:

```text
[zimage-selected-grad-vram] PASS evidence=external-observed-vram-full-depth-all-trainable-grad-replay streamed_b2_selected_replay_peak_vram_bytes= 22567452672 external_peak_vram_delta_mib= 21522 sample_count= 203 all_trainable_grad_tensors= 420 all_trainable_grad_numel= 35020800 all_trainable_grad_max_abs= 3.6748774618899915e-06 all_trainable_grad_tol= 1e-05 selected_layer0_grad_max_abs= 8.392975701099203e-07 artifact= /home/alex/mojodiffusion/artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json
```

What it proves:

- The real OneTrainer step0 dump is present and has the expected `42` tensors.
- The real step0 adapter dump is present and has the expected `3360` F32 phase tensors.
- The selected layer-0 target surface is present: `14` tensors and `1167360` elements.
- The exact OneTrainer batch geometry is not the earlier `CAP=145` shortcut and not the product `CAP=224` bucket: caption rows pad to `160` and `128`, then batch padding masks the shorter sample.
- The selected replay gate now uploads step tensors as BF16 device tensors and
  converts the selected layer-0 `adapter_before.*` host F32 dump matrices through
  `LoraAdapter`/`zimage_lora_adapter_to_device`, asserting BF16 adapter device
  tensors. The F32 adapter dump remains host comparison data only.
- The selected replay gate now opens the real sharded transformer, loads only
  `layers.0` through the preserve-dtype block loader, and asserts all `13` base
  block device tensors are BF16. This is a streamed-replay prerequisite, not a
  resident full-stack run.
- The selected replay gate now names the accepted streamed B2 bridge explicitly:
  `zimage_stack_lora_forward_main_device_b2_masked_streamed` and
  `zimage_stack_lora_backward_main_device_b2_masked_streamed`.
  `resident_masked_b2_accepted=false` for the 24 GB target. The bridge still
  records `activation_carrier_dtype=F32 not_strict_BF16_step_storage` and
  `adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only`.
- The executable preflight now also builds the real step0 inputs through the
  streamed bridge with a bounded zero-main-block smoke: BF16 latents/text
  tensors, real aux/embedder weights, four streamed frozen refiner blocks,
  per-sample image RoPE (`x_rope_per_sample=true`), cap/main mask lengths
  `160/128` and `1184/1152`, and
  `observed_vram_mib_lower_bound=372.00146484375`. This proves real-input
  wiring only; it deliberately uses `streamed_main_blocks=0`.
- The opt-in full-depth define runs all `30` streamed main blocks without a
  resident full-stack block list, builds real loss-root gradients from
  `flow_target` and `predicted_flow`, and compares all `420` adapter-gradient
  tensors (`35020800` elements) against `adapter_post_clip_grad.*`. The
  observed all-trainable max abs error was `3.6748774618899915e-06` with
  tolerance `1e-05`; the retained selected layer-0 subset was
  `8.392975701099203e-07`.
- `scripts/run_zimage_selected_grad_replay_vram.py` runs the same full-depth
  replay under an external `nvidia-smi` poller. The recorded artifact
  `artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json`
  has `streamed_b2_selected_replay_peak_vram_bytes=22567452672`,
  `external_peak_vram_delta_mib=21522`, `sample_count=203`, and preserves the
  all-trainable grad result. This is not product-loop steady-state VRAM; it is
  external observed peak-delta evidence and not CUDA profiler high-water
  evidence.

Current blocker:

- Product-loop steady-state speed/VRAM evidence is still missing. The inline
  Mojo output still reports only `observed_vram_mib_lower_bound`; the wrapper
  supplies external observed replay VRAM for this replay gate only.
- The opt-in full-depth run is not yet a product-loop train step, save/resume
  gate, or strict BF16 activation-storage path.
- The non-graph masked B=2 caption-refiner and unified main attention path is
  the only accepted path for this gate until graph/slab masking exists.
- The graph/slab B2 path still records no-mask SDPA only, so masked selected-gradient replay must stay on the non-graph masked B2 path until masked graph/slab recording and dispatch are added.
