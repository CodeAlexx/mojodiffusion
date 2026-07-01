# ZImage Adapter Oracle Metadata Smoke

Date: 2026-06-30

Evidence level: OneTrainer dtype-boundary and adapter-dump metadata smoke. This
replaces the earlier F32 device-carrier interpretation, which is invalid for
ZImage on the 24 GB BF16-boundary runtime. OneTrainer's ZImage base/train/output
path is BF16; the `FLOAT_32` adapter dump reflects live OneTrainer LoRA params
for comparison, not Mojo adapter/device storage.

Command:

```bash
pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_train_ref_f32_adapter_carrier_smoke.mojo
```

Output:

```text
[zimage-adapter-oracle-metadata] PASS layer= 0  forward_phase=adapter_before grad_target=adapter_post_clip_grad  tensors= 14  selected_numel= 1167360  runtime_boundary=BF16 adapter_dump_dtype=F32 mojo_storage_boundary=BF16 rank= 16  replay_scale= 0.0625
[zimage-adapter-oracle-metadata] scope=OneTrainer BF16 runtime/step boundary plus live LoRA dump metadata; no device upload; adapter dump dtype is not Mojo storage authority
```

What this covers:

- Verifies OneTrainer step0 runtime tensors are BF16 at boundaries:
  `scaled_noisy_latent_image`, `latent_input`, `predicted_flow`, `flow_target`,
  and text encoder outputs.
- Consumes `/home/alex/serenity-trainer/parity/zimage_train_ref_step000_adapters.safetensors`.
- Uses the OneTrainer `adapter_before.*` phase for forward LoRA weights.
- Verifies matching `adapter_post_clip_grad.*` selected-gradient target shapes.
- Checks layer `0` in Mojo ZImage slot order:
  `to_q`, `to_k`, `to_v`, `to_out.0`, `w1`, `w3`, `w2`.
- Verifies `14` F32 adapter-dump comparison tensors and `1167360` selected
  elements.
- Records strict LoRA replay scale `0.0625` (`lora_alpha / rank = 1 / 16`).
- Does not create a device context and does not upload F32 LoRA carriers.
- OneTrainer source context: the config uses BF16 train/transformer/output
  dtype; its missing `lora_weight_dtype` key defaults live LoRA params to
  `FLOAT_32`, while final LoRA export goes through `output_dtype=BFLOAT_16`.
- Source anchors:
  - `/home/alex/OneTrainer/configs/eri2_zimage_base_2500.json`: transformer
    weight dtype, text encoder weight dtype, train dtype, and output dtype are
    `BFLOAT_16`.
  - `/home/alex/OneTrainer/modules/modelSetup/BaseZImageSetup.py`: `predict`
    casts latent input to `model.train_dtype`.
  - `/home/alex/OneTrainer/modules/modelSetup/ZImageLoRASetup.py`: the LoRA
    wrapper is cast to `config.lora_weight_dtype`.
  - `/home/alex/OneTrainer/modules/util/config/TrainConfig.py`: absent
    `lora_weight_dtype` defaults to `FLOAT_32`.
  - `/home/alex/OneTrainer/modules/trainer/GenericTrainer.py` plus
    `modules/modelSaver/mixin/LoRASaverMixin.py`: normal final LoRA save is cast
    through `config.output_dtype`.

Current boundary:

- This metadata smoke remains dtype-only: it proves the adapter dump is a host
  comparison oracle and must not become Mojo storage authority.
- The full streamed masked B=2 all-trainable replay is tracked in
  `artifacts/training_perf/zimage_selected_grad_replay_preflight_2026-06-30.md`
  and `artifacts/training_perf/zimage_selected_grad_replay_vram_2026-06-30.json`.
  It runs through
  `zimage_stack_lora_forward_main_device_b2_masked_streamed` and
  `zimage_stack_lora_backward_main_device_b2_masked_streamed`, keeps
  adapter/device storage BF16, and compares all `420` adapter-gradient tensors
  against the adapter dump's F32 host-oracle `adapter_post_clip_grad.*` tensors.
  Resident masked B2 APIs are not accepted for the 24 GB selected replay target.
