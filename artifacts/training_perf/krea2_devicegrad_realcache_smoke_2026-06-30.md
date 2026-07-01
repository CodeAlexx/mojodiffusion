# Krea2 Real-Cache Device-Grad Smoke

Date: 2026-06-30

Evidence level: Mojo current product smoke on a real Krea2 cache. This is not ai-toolkit parity, loss replay, gradient parity, optimizer parity, save/resume equivalence, or a device-fast product claim.

Cache:

- `/home/alex/trainings/krea2_giger_cache_512.safetensors`
- samples: `70`
- latent shape: `[1,16,64,64]`
- context dtype: `BF16`
- clean dtype: `F32`
- text length min/max: `398` / `803`

Preflight:

```bash
python3 scripts/check_krea2_real_cache_contract.py \
  /home/alex/trainings/krea2_giger_cache_512.safetensors \
  --lh 64 --lw 64 --ltmax 896 --min-samples 2 --require-real
```

Build:

```bash
rm -f serenitymojo.mojopkg
timeout 900 pixi run mojo build --optimization-level 2 -DKREA2_LTMAX=896 -I . \
  -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib \
  -Xlinker -lserenity_cudnn_sdpa \
  serenitymojo/models/krea2/train_krea2.mojo \
  -o /tmp/krea2_train_lt896_check
```

Run:

```bash
LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
  timeout 1200 /tmp/krea2_train_lt896_check \
  /home/alex/trainings/krea2_giger_cache_512.safetensors 2 \
  serenitymojo/configs/krea2_devicegrad_realcache_smoke.json krea2devicegrad
```

Observed:

- compiled bucket: `LTMAX=896`, `LFULL=1920`
- config rank/alpha: `32` / `32`, matching the inspected ai-toolkit output
- step 1: loss `0.4814`, grad norm `0.0019`, `140.0s/step`
- step 2: loss `0.1370`, grad norm `0.0009`, `71.0s/step`
- perf record seconds/step: `106.176643333`
- dtype label: `BF16_BASE_BF16_LORA_F32_OPT` (BF16 base/LoRA storage, F32 grads/moments/reductions)
- peak VRAM: `2830140416`
- counted host-device transfers: `22`
- counted syncs: `63`
- full tensor readbacks: `2`
- fast-path label: `host-grad-compat-slow`
- final LoRA: `/tmp/krea2_devicegrad_realcache_smoke/krea2_devicegrad_realcache_smoke_2.safetensors`
- saved tensor check: `448` tensors, dtype set `['torch.bfloat16']`

The compile-bucket blocker is resolved for this smoke by the
`KREA2_LTMAX=896` build arm. The main-block LoRA rank/shape blocker is also
closed for this smoke: the saved `448` block tensors are BF16 and shape-match
the inspected ai-toolkit rank-32 output. Remaining work is the missing
`diffusion_model.txtfusion.*` LoRA surface, ai-toolkit real-cache parity,
profiler-complete transfer/sync accounting beyond counted fences, phase timing
breakdown, save/resume equivalence, and longer measured runs before any speed
or convergence claim.
