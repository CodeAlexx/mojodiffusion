# Krea2 Txtfusion Devicegrad Real-Cache Smoke

Date: 2026-06-30

Evidence level: one-step real-cache product-loop smoke for the opt-in
`KREA2_TXTFUSION_LORA` path. This proves the full 256-adapter key surface,
device-grad preload count, final BF16 save surface, and 24 GB fit for one
512px real-cache step. It is not convergence parity, selected-gradient parity,
optimizer replay parity, or sampling support. Full-surface resume is covered
separately as bounded Mojo product-path evidence at
`artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_resume_smoke_2026-06-30.md`.

Build command:

```bash
cd /home/alex/serenity-trainer
mkdir -p target
pixi run mojo build -I . -I src -I /home/alex/mojodiffusion \
  -DKREA2_TXTFUSION_LORA=1 -DKREA2_LTMAX=896 \
  -Xlinker -lm -Xlinker -lcuda \
  -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib \
  -Xlinker -L/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
  -Xlinker -lserenity_cudnn_sdpa \
  -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
  -Xlinker -rpath -Xlinker /home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
  -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/.pixi/envs/default/lib \
  /home/alex/mojodiffusion/serenitymojo/models/krea2/train_krea2.mojo \
  -o target/serenity_krea2_live_trainer_txtfusion_lt896
```

Run command:

```bash
cd /home/alex/serenity-trainer
target/serenity_krea2_live_trainer_txtfusion_lt896 \
  /home/alex/trainings/krea2_giger_cache_512.safetensors \
  1 \
  /home/alex/mojodiffusion/serenitymojo/configs/krea2_devicegrad_realcache_smoke.json \
  krea2devicegrad
```

Observed product run:

- Rank/alpha: `32` / `32`.
- LTMAX/LFULL: `896` / `1920`.
- Selected first sample: sample `34`, text length `803`.
- Device-grad proof:
  `grad_pairs= 256`, `streaming_syncs= 29`.
- Step 1: loss `0.4813`, grad norm `0.0025`, `67.6s/step`.
- Scorecard seconds/step: `69.81506392`.
- Peak VRAM: `2926937088` bytes.
- Host-device transfers: `13`.
- Full tensor readbacks: `1`.
- Syncs: `33`.
- Final save:
  `/tmp/krea2_devicegrad_realcache_smoke/krea2_devicegrad_realcache_smoke_1.safetensors`.

Surface comparison:

```bash
cd /home/alex/mojodiffusion
python3 scripts/check_krea2_trainable_surface.py \
  --mojo /tmp/krea2_devicegrad_realcache_smoke/krea2_devicegrad_realcache_smoke_1.safetensors \
  --expect-match
```

Result:

```text
[krea2-surface] ai_toolkit: path=/home/alex/ai-toolkit/output/my_first_lora_v1/my_first_lora_v1_000002994.safetensors total=512 blocks=448 txtfusion=64 non_block=64 target_prefixes=256 dtypes=['torch.bfloat16']
[krea2-surface] mojo: path=/tmp/krea2_devicegrad_realcache_smoke/krea2_devicegrad_realcache_smoke_1.safetensors total=512 blocks=448 txtfusion=64 non_block=64 target_prefixes=256 dtypes=['torch.bfloat16']
[krea2-surface] delta: common_keys=512 missing_from_mojo=0 missing_txtfusion=0 missing_non_txtfusion=0 extra_in_mojo=0 block_key_delta=0 shape_mismatch=0 dtype_mismatch=0
[krea2-surface] PASS exact_match
```

Known limits:

- `KREA2_TXTFUSION_LORA` is opt-in.
- Full-surface resume is no longer fail-loud for this opt-in path, but the
  current gate is bounded Mojo product-path continuation evidence, not
  byte-equivalent or ai-toolkit resume parity.
- Sampling is blocked until txtfusion LoRA conditioning is wired into the inline
  sampler.
- The txtmlp/txtfusion backward path is covered by smoke/build/product surface
  evidence, not strict ai-toolkit selected-gradient replay yet.
- Counters are visible accounting, not profiler-complete transfer/sync
  accounting.
