# Klein Mojo-Current Scorecard Smoke

Evidence level: one-step product-worker smoke at 512px; not production parity,
not a OneTrainer replay, and not a device-fast claim.

Command:

```bash
timeout 1800 /home/alex/serenity-trainer/target/serenity_klein_live_trainer serenitymojo/configs/klein9b_scorecard_smoke.json 1 0 - nosample
```

Captured record:

- JSONL: `artifacts/training_perf/klein_mojo_current_2026-06-30.jsonl`
- Output LoRA: `/tmp/klein9b_scorecard_smoke.safetensors`
- Output state: `/tmp/klein9b_scorecard_smoke.safetensors.state.safetensors`
- Log: `/tmp/klein_scorecard_run.log`

Observed values:

- loss: `0.5272`
- grad norm: `0.0100`
- measured steps: `1`
- dtype label: `BF16_BASE_BF16_LORA_F32_OPT` (BF16 base/LoRA storage, F32 grads/moments/reductions)
- seconds per step: `10.184290664`
- forward seconds: `2.269932525`
- backward seconds: `2.68129066`
- save seconds: `0.885887693`
- peak VRAM bytes: `18906398720`
- visible host-device transfers: `2`
- visible syncs: `1`
- conservative full tensor readbacks: `2`
- fast-path label: `host-grad-compat-slow`
- attention backend label: `klein-stack-direct`

Limits:

- The target roadmap row remains 1024px; this smoke used the existing 512px
  cache at
  `/home/alex/flame-diffusion-archive/klein-trainer/cache/eri2_klein9b_512`.
- Counters are labeled `visible-counter-lower-bound`, not complete profiler
  counters.
- The run still uses host loss and host-list grad compatibility plumbing.
- OneTrainer parity is not proven by this artifact.
- Device-fast acceptance still requires full readbacks at zero and a
  `DeviceTrainableSet` / `DeviceGradSet` / `TrainStepDeviceResult` path.

Verification:

- Product worker run: PASS, emitted `[training-perf-json]` and reached
  `DONE: worker reached step 1 of 1 target`.
- `python3 scripts/check_klein_loss_replay.py --strict`: BLOCKED before parity
  evaluation because `/home/alex/onetrainer-mojo/parity/klein_train_ref_meta.json`
  is missing.
