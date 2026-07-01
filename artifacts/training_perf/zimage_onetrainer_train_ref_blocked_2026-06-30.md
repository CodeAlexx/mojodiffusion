# ZImage OneTrainer Train-Ref State-Init And AdamW Update Evidence

Date: 2026-06-30

Evidence level: OneTrainer state-init train-ref artifact consumer, Mojo loss
bridge, OneTrainer update-bearing adapter oracle, full CPU AdamW update replay
oracle, sampled Mojo scalar AdamW replay, and full all-420 Mojo shared device ABI AdamW update replay,
plus real step-0 Mojo device loss-root replay. This
is a real ZImage batch-2 step-0/step-1 loss/backward/adapter tensor dump; Mojo
recomputes the step-0 flow-MSE loss from the dumped `predicted_flow` and
`flow_target` tensors; the v5 device loss root consumes the same dump after
product-layout patchification and writes matching `d_patches`; the step-1
OneTrainer adapter dump has nonzero AdamW update deltas; and the dumped
step-0/step-1 grads reproduce the step-1 AdamW update through both CPU and
shared device optimizer replay. It is not transformer forward replay, Mojo
gradient parity, save/resume equivalence, or product speed evidence.

Evidence anchor: real OneTrainer step0 device loss-root replay and full all-420
Mojo shared device ABI AdamW update replay.

Source config:

- `/home/alex/OneTrainer/configs/eri2_zimage_base_2500.json`
- `model_type`: `Z_IMAGE`
- `training_method`: `LORA`
- `batch_size`: `2`
- `resolution`: `512`
- `train_dtype`: `BFLOAT_16`
- `weight_dtype`: `BFLOAT_16`
- `output_dtype`: `BFLOAT_16`
- `cache_dir`: `/home/alex/OneTrainer/workspace-cache/eri2_zimage_base_lora`
- `base_model_name`: `/home/alex/.serenity/models/zimage_base`

Producer:

- `/home/alex/serenity-trainer/scripts/zimage_dump_train_ref.py`

Run command:

```bash
timeout 480s /home/alex/OneTrainer/venv/bin/python \
  /home/alex/serenity-trainer/scripts/zimage_dump_train_ref.py \
  --adapter-dump step-with-grads \
  --max-steps 2 \
  2>&1 | tee /home/alex/serenity-trainer/parity/zimage_train_ref_dump.log
```

Produced train-ref artifacts:

- `/home/alex/serenity-trainer/parity/zimage_train_ref_meta.json`
- `/home/alex/serenity-trainer/parity/zimage_train_ref_step000.safetensors`
- `/home/alex/serenity-trainer/parity/zimage_train_ref_step000_adapters.safetensors`
- `/home/alex/serenity-trainer/parity/zimage_train_ref_step001.safetensors`
- `/home/alex/serenity-trainer/parity/zimage_train_ref_step001_adapters.safetensors`
- `/home/alex/serenity-trainer/parity/zimage_train_ref_dump.log`

Observed step-0 metadata:

- loss before scaling: `0.40854018926620483`
- loss for backward: `0.40854018926620483`
- pre-clip grad norm: `0.0005584948230534792`
- no-clip grad norm: `0.0005584948230534792`
- trainable LoRA tensors: `420`
- trainable LoRA numel: `35020800`
- `lr_before`: `[0.0]`
- `lr_after`: `[1.4999999999999998e-06]`
- adapter delta: `0.0`

Observed step-1 update-bearing metadata:

- loss for backward: `0.419974148273468`
- pre-clip grad norm: `9.306404535891488e-05`
- `lr_before`: `[1.4999999999999998e-06]`
- `lr_after`: `[2.9999999999999997e-06]`
- optimizer entries before/after: `420 -> 420`
- adapter delta max abs: `1.4883222547723562e-06`
- adapter delta L2: `0.002961230231449008`
- adapter tensors with nonzero update: `210 / 420`

Validation command:

```bash
python3 scripts/check_zimage_train_ref_contract.py --require-input-dump
pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_train_ref_loss_replay.mojo
pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_train_ref_device_loss_replay.mojo
python3 scripts/check_zimage_adapter_update_replay.py --step-index 1 --expect-update yes --require-update-bearing
python3 scripts/check_adapter_update_replay.py zimage --step-index 1 --expect-update yes --require-update-bearing
python3 scripts/check_zimage_adamw_update_replay.py
pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_train_ref_adamw_update_replay.mojo
pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_train_ref_fused_adamw_update_replay.mojo
```

Current result:

```text
[zimage-train-ref-contract] input dump PASS: /home/alex/serenity-trainer/parity/zimage_train_ref_step000_inputs.safetensors
[zimage-train-ref-contract] PASS: /home/alex/OneTrainer/configs/eri2_zimage_base_2500.json has required train-ref artifacts in /home/alex/serenity-trainer/parity (evidence=state-init)
[zimage-train-ref-loss] loss_bridge PASS stored= 0.4085402  replayed= 0.40854487
[zimage-train-ref-device-loss] PASS loss= 0.40854034  host_loss= 0.4085402  rows= 2048  out_ch= 64  numel= 131072  nonzero_grad= 130494  nonzero_error= 0  grad_max_abs= 0.0  grad_l2= 0.0  backend= device-mse-block-reduce-into-scratch  full_readbacks= 0  scalar_readbacks= 1  syncs= 1
[zimage-train-ref-device-loss] scope=real OneTrainer step0 dump through v5 device flow-MSE d_patches root; not transformer forward/backward parity
[zimage-adapter-update] PASS zimage
[adapter-update-replay] PASS zimage
[zimage-adamw-update-replay] PASS zimage
[zimage-adamw-update-mojo] sampled_replay PASS tensors= 8  numel= 696320  nonzero_update= 696320  nonzero_error= 630515  max_abs= 4.4337867e-12  l2= 6.233504967178367e-10
[zimage-fused-adamw-update-mojo] full_device_abi_replay PASS tensors= 420  numel= 35020800  nonzero_update= 19046400  nonzero_param_error= 18890164  max_param_abs= 5.2295945e-12  param_l2= 7.972661741797729e-09  nonzero_state_error= 9848638  max_state_abs= 2.2737368e-13  state_l2= 2.1890955386202706e-12  grad_norm= 9.3064045e-05  clip_scale= 1.0  syncs= 2
```

Claim boundary:

- The strict train-ref artifact triplet now exists and validates tensor shapes,
  adapter phases, OneTrainer adapter names, and nonzero sampled LoRA gradients.
- The Mojo loss bridge at
  `serenitymojo/models/zimage/parity/zimage_train_ref_loss_replay.mojo`
  recomputes the step-0 flow-MSE loss from `predicted_flow`, `flow_target`, and
  `batch.loss_weight`. The small difference from the stored scalar is within
  the explicit BF16/Float32 accumulation tolerance.
- The real OneTrainer step0 device loss-root replay at
  `serenitymojo/models/zimage/parity/zimage_train_ref_device_loss_replay.mojo`
  patchifies dumped BF16 `predicted_flow` and `flow_target` tensors into the
  product `[rows, OUT_CH]` layout, runs
  `zimage_step_io_write_flow_mse_d_patches`, and verifies `d_patches` against a
  host reference with `rows=2048`, `out_ch=64`, `numel=131072`,
  `grad_max_abs=0.0`, `full_readbacks=0`, `scalar_readbacks=1`, and `syncs=1`.
  The post-root `d_patches` readback is parity inspection, not part of the
  device loss root fast path.
- The first AdamW step is state-init only because `lr_before` is `0.0`, so
  `adapter_after - adapter_before` is zero.
- Step 1 is update-bearing OneTrainer oracle evidence: `lr_before` is positive
  and `adapter_after - adapter_post` has nonzero deltas. This proves the local
  OneTrainer reference has a nonzero adapter update target.
- `scripts/check_zimage_adamw_update_replay.py` fully replays the step-1 AdamW
  update across `420` tensors and `35020800` elements from dumped step-0/step-1
  gradients. It reproduces `adapter_after` with max abs
  `4.547473508864641e-13`, L2 `2.36219986530941e-10`, and
  `19046400` nonzero update elements.
- `serenitymojo/models/zimage/parity/zimage_train_ref_adamw_update_replay.mojo`
  opens the same real adapter safetensors in Mojo and replays a representative
  scalar AdamW subset of `696320` elements with max abs `4.4337867e-12`.
  This proves Mojo can consume the update oracle and run the scalar math; it
  does not prove fused device-optimizer parity.
- `serenitymojo/models/zimage/parity/zimage_train_ref_fused_adamw_update_replay.mojo`
  opens the same real adapter safetensors in Mojo, uploads all `420` step-1
  adapter params/grads plus inferred step-0 Adam moments as device tensors, and
  replays the step-1 update through `DeviceTrainableSet`, `DeviceGradSet`,
  `DeviceAdamWState`, and `device_adamw_train_step_update`. It covers
  `35020800` elements, `19046400` nonzero update elements, max param abs
  `5.2295945e-12`, and clip scale `1.0`. This proves optimizer-only shared
  device ABI replay of the OneTrainer update oracle; it does not prove
  transformer forward/backward parity.
- A full Mojo ZImage replay still needs to run transformer forward/backward on
  byte-identical inputs and compare produced grads against the step-1 adapter
  update oracle.
- `/home/alex/OneTrainer/configs/alina_zimage_OTpreset_100_baseline.json`
  remains structurally useful, but its local dataset path
  `/home/alex/datasets/AlinaAignatova` is absent, so the current collected
  batch-2 reference uses the available Eri2 cache.

Related precursor:

- `artifacts/training_perf/zimage_onetrainer_input_dump_2026-06-30.md`
  records the older input-only hook artifact. It is retained as continuity
  evidence but is superseded by the state-init train-ref triplet above.

Product-readiness side blocker:

- `/home/alex/mojodiffusion/serenitymojo/configs/zimage_eri_1024smoke.json`
  targets batch-2/1024-ish product cadence.
- Existing logs `/tmp/zimage_1024smoke.log` and `/tmp/zimage_1024res.log`
  indicate training runs, but in-process 1024 sample render OOMs after step 2/4
  and step 4/4, then falls back to 512.
