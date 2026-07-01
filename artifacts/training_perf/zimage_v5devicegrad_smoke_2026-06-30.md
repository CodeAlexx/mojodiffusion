# ZImage v5devicegrad Smoke Perf Evidence

Date: 2026-06-30

Command:

```bash
cd /home/alex/serenity-trainer
./target/serenity_zimage_live_trainer \
  /home/alex/mojodiffusion/serenitymojo/configs/zimage_v5devicegrad_smoke.json \
  3 0 - v5devicegrad
```

Evidence level: three-step product-trainer smoke, not production parity.

Inputs:

- Config: `/home/alex/mojodiffusion/serenitymojo/configs/zimage_v5devicegrad_smoke.json`
- Cache: `/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_zimage_512_smoke`
- Checkpoint: `/home/alex/.serenity/models/zimage_base/transformer`

Output artifacts:

- `/tmp/zimage_v5devicegrad_smoke/lora.safetensors`
- `/tmp/zimage_v5devicegrad_smoke/lora.safetensors.state.safetensors`

Observed run summary:

- Step 1: loss `0.34357935`, `2.0s/step`.
- Step 2: loss `0.65501225`, `1.2s/step`.
- Step 3: loss `0.433318`, `1.2s/step`.
- Scorecard seconds/step: `1.6521998196666667`.
- Phase totals: forward `1.090750497s`, loss `0.174322235s`,
  backward `2.764714736s`, optimizer `0.022807304s`, save `4.855654281s`.
- Peak VRAM: `19789279232` bytes
- Host-device transfers: `16`
- Full tensor readbacks: `1`
- Syncs: `7`
- Dtype label: `BF16_BASE_BF16_LORA_F32_OPT`
- Optimizer backend: `fused_adamw_multitensor-arena-grad-stats-adamw-descriptors`
- Fast-path label: `host-grad-compat-slow`

Known limits:

- This is a bounded three-step smoke.
- Counters are visible lower-bound accounting, not complete profiler accounting.
- Phase timings come from product-loop timestamp boundaries; grad norm and clip
  are folded into the device AdamW optimizer timing for this smoke.
- The v5 optimizer branch now reports `TrainStepDeviceResult` from the shared
  `DeviceTrainableSet`/`DeviceGradSet` ABI. This is still not a product
  device-fast claim because the smoke retains lower-bound counters and final
  save/inspection readback.
- Full tensor readback count is still nonzero because params sync once after the loop for final save/inspection.
