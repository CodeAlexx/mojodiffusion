# ZImage OneTrainer Input Dump

Date: 2026-06-30

Evidence level: input-dump artifact consumer. This validates the older
OneTrainer hook surface for a ZImage batch-2 step-0 input bundle. It is not loss
replay, gradient parity, optimizer update parity, save/resume equivalence, or
product speed evidence. It is now superseded by the state-init train-ref triplet
documented in
`artifacts/training_perf/zimage_onetrainer_train_ref_blocked_2026-06-30.md`.

Source config:

- `/home/alex/OneTrainer/configs/eri2_zimage_base_2500.json`
- temporary run config: `/tmp/eri2_zimage_step0_dump_config.json`
- dataset path: `/home/alex/eri2`
- batch: `2`
- resolution: `512`
- dtype: `BFLOAT_16`

Run command:

```bash
cd /home/alex/OneTrainer
OT_DUMP_STEP1_INPUTS=1 \
OT_DUMP_STEP1_PATH=/home/alex/serenity-trainer/parity/zimage_train_ref_step000_inputs.safetensors \
OT_DUMP_ITER0_GRADS=1 \
OT_DEBUG_STATS=1 \
timeout 180s /home/alex/OneTrainer/venv/bin/python scripts/train.py \
  --config-path /tmp/eri2_zimage_step0_dump_config.json \
  2>&1 | tee /home/alex/OneTrainer/output/eri2_zimage_step0_dump/run.log
```

Produced artifact:

- `/home/alex/serenity-trainer/parity/zimage_train_ref_step000_inputs.safetensors`

Validated tensors:

- `scaled_latent_image`: `[2,16,64,64]` BF16
- `latent_noise`: `[2,16,64,64]` BF16
- `scaled_noisy_latent_image`: `[2,16,64,64]` BF16
- `latent_input`: `[2,16,1,64,64]` BF16
- `timestep`: `[2]` F32
- `sigma`: `[2,1,1,1]` F32
- `flow_target`: `[2,16,64,64]` BF16
- `predicted_flow`: `[2,16,64,64]` BF16
- `text_encoder_output_0`: `[145,2560]` BF16
- `text_encoder_output_1`: `[127,2560]` BF16
- `text_encoder_output_batch_size`: scalar I64

Observed log evidence:

- step-0 loss before scaling: `0.4086`
- step-0 timestep print: `980.00`
- first-backward LoRA grad log count: `420` params
- step-0 pre-clip grad norm: `5.5936e-04`

Validation command:

```bash
python3 scripts/check_zimage_train_ref_contract.py --require-input-dump
```

Current result:

```text
[zimage-train-ref-contract] input dump PASS: /home/alex/serenity-trainer/parity/zimage_train_ref_step000_inputs.safetensors
[zimage-train-ref-contract] PASS: /home/alex/OneTrainer/configs/eri2_zimage_base_2500.json has required train-ref artifacts in /home/alex/serenity-trainer/parity (evidence=state-init)
```

Superseded by the state-init train-ref triplet:

The current strict triplet contains formal metadata, step tensors, adapter
before/pre/post/after phases, and tensor-level LoRA gradients. It is still not
nonzero-LR optimizer update parity because the first AdamW step has
`lr_before=0.0` and zero adapter delta.
