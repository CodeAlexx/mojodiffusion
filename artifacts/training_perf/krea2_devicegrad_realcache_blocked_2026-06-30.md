# Krea2 Real-Cache Device-Grad Smoke Compile-Bucket Blocker

Date: 2026-06-30

Evidence level: historical compile-bucket blocker plus resolution note. The
real-cache Mojo smoke result is recorded separately in
`artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.md`.

Target cache:

- `/home/alex/trainings/krea2_giger_cache_512.safetensors`

Preflight command that passes:

```bash
cd /home/alex/mojodiffusion
python3 scripts/check_krea2_real_cache_contract.py \
  /home/alex/trainings/krea2_giger_cache_512.safetensors \
  --lh 64 --lw 64 --ltmax 896 --min-samples 2 --require-real
```

Observed passing preflight:

- samples: `70`
- latent shape: `[1,16,64,64]`
- context dtype: `BF16`
- clean dtype: `F32`
- text length min/max: `398` / `803`
- required bucket: `LTMAX=896`

Original blocker:

The default Krea2 trainer build arm uses `KREA2_RES_512=True` and
`KREA2_LTMAX=384`. That arm cannot consume this real cache:

```bash
python3 scripts/check_krea2_real_cache_contract.py \
  /home/alex/trainings/krea2_giger_cache_512.safetensors \
  --lh 64 --lw 64 --ltmax 384 --min-samples 2 --require-real
```

Observed failure:

```text
[krea2-cache-contract] FAIL: only 0/70 samples fit LTMAX=384; min_lt=398 max_lt=803
```

Resolution:

- `serenitymojo/models/krea2/train_krea2.mojo` now defaults to
  `KREA2_LTMAX=384` but accepts `mojo build -DKREA2_LTMAX=896`.
- The synthetic fixture scorecard stays reproducible because the default build
  arm remains 384.
- The real-cache smoke was collected with the 896 arm and stored as
  `artifacts/training_perf/krea2_devicegrad_realcache_smoke_2026-06-30.jsonl`
  plus `.md`.

Next required work:

- ai-toolkit real-cache parity
- profiler-complete transfer/sync accounting beyond the counted streaming fences
- phase timings
- longer measured run

Claim boundary:

Resolving this blocker produced a real-cache smoke, not ai-toolkit parity, loss
replay, gradient parity, optimizer update parity, save/resume equivalence, or
device-fast product evidence.
