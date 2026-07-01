# SDXL Scorecard Blocker

Evidence level: blocked-not-collected. This is not a performance result, not numeric Mojo grad/update evidence, not shared device-ABI evidence, and not production parity.

The SDXL row remains in the benchmark matrix as the non-transformer architecture
target, but it must not emit a Mojo-current `TrainingPerfRecord` yet. The current
SDXL trainer surface is a small-latent smoke harness with host prediction
readback, host loss construction, host-list adapter gradients, and incomplete
full-UNet adapter parity evidence.

Current blockers:

- No SDXL perf JSONL is accepted under `artifacts/training_perf/`.
- The sibling product trainer
  `/home/alex/serenity-trainer/src/serenity_trainer/trainer/train_sdxl_real.mojo`
  must not emit `[training-perf-json]` until the blocker is cleared.
- `serenitymojo/models/sdxl/sdxl_real_train.mojo` returns host-list LoRA grads,
  not a `DeviceGradSet`.
- `serenitymojo/models/sdxl/sdxl_unet_stack_lora.mojo` still exposes host-list
  `SdxlStLoraGrads` and host readback/rehydration in the LoRA backward path.
- `serenitymojo/models/sdxl/lora_block.mojo` uses host-list `to_host` /
  `from_host` compatibility plumbing for SDXL LoRA math.
- Local replay artifacts are missing:
  `/home/alex/OneTrainer/output/sdxl_100step_baseline/step000_replay.safetensors`,
  `/home/alex/OneTrainer/output/sdxl_100step_baseline/step000_replay_manifest.json`,
  `/home/alex/onetrainer-mojo/parity/sdxl_train_ref_step000.safetensors`, and
  `/home/alex/onetrainer-mojo/parity/sdxl_train_ref_step000_adapters.safetensors`.
- The existing SDXL train-ref Mojo consumer is artifact-consumption only; it
  does not prove UNet forward, backward, AdamW update, save/resume, sampler, or
  product speed parity.

Current positive evidence that remains useful:

- OneTrainer SDXL 100-step baseline metadata is present and validated by
  `scripts/check_sdxl_lora_keys.py`: 100 steps, mean step
  `0.973008s`, final loss `0.018004747`, final grad norm `0.002959009`,
  max allocated CUDA memory `6616MiB`, max reserved `6898MiB`.
- The in-repo SDXL source scaffold and artifact-consumer gate are visible, but
  they stop before full Mojo replay or device-fast evidence.

Verification status:

- `python3 scripts/check_sdxl_lora_keys.py --strict-port --require-replay-dump`:
  expected FAIL on missing replay/train-ref artifacts after passing source,
  baseline, and save-key checks.
- `python3 scripts/check_sdxl_adapter_update_replay.py --require-update-bearing`:
  expected FAIL because `/home/alex/onetrainer-mojo/parity/sdxl_train_ref_meta.json`
  is missing.
- `python3 scripts/check_sdxl_training_perf_blocker.py`: required PASS while the
  blocker is active.

Clearance condition:

Replace this blocker with an SDXL Mojo-current scorecard only after the run is
honestly labeled and backed by OneTrainer replay artifacts, comparable backward
or update evidence, and an explicit decision about the host-grad compatibility
surface versus the shared `DeviceTrainableSet` / `DeviceGradSet` /
`TrainStepDeviceResult` path.
