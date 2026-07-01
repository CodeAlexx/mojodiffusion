# Krea2 Txtfusion Devicegrad Real-Cache Resume Smoke

Date: 2026-06-30

Evidence level: bounded Mojo product-path save/resume smoke for the opt-in
`KREA2_TXTFUSION_LORA` path. This validates full-surface `.state` save/load,
BF16 PEFT surface save, F32 AdamW moment round-trip, and a resumed real-cache
step within this bounded Mojo smoke. It is not
ai-toolkit full-surface loss/gradient/update parity, not byte-equivalent resume, not convergence
evidence, and not sampling support.

Oracle boundary:

- Existing Krea2 oracle floor remains the reduced-depth ai-toolkit
  `SingleStreamDiT` stack and AdamW replay at
  `artifacts/training_perf/krea2_stack_adamw_update_replay_2026-06-30.md`.
- There is no current full-surface txtfusion ai-toolkit resume oracle. This
  artifact compares Mojo resumed output against an uninterrupted Mojo run and
  records the tolerance explicitly.

Build command:

```bash
cd /home/alex/mojodiffusion
mkdir -p target
pixi run mojo build -I . \
  -DKREA2_TXTFUSION_LORA=1 -DKREA2_LTMAX=896 \
  -Xlinker -lm \
  -Xlinker -Lserenitymojo/ops/cshim/lib \
  -Xlinker -lserenity_cudnn_sdpa \
  -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
  serenitymojo/models/krea2/train_krea2.mojo \
  -o target/serenity_krea2_live_trainer_txtfusion_lt896
```

Resume-leg first step:

```bash
LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
timeout 1200 target/serenity_krea2_live_trainer_txtfusion_lt896 \
  /home/alex/trainings/krea2_giger_cache_512.safetensors \
  1 \
  serenitymojo/configs/krea2_devicegrad_realcache_txtfusion_resume_smoke.json \
  krea2devicegrad
```

Observed:

- `grad_pairs= 256`, `streaming_syncs= 29`
- step 1 loss `0.4813`, grad norm `0.0025`, progress `75.6s/step`
- wrote `/tmp/krea2_txtfusion_resume_smoke/krea2_txtfusion_resume_smoke_1.safetensors`
- wrote `/tmp/krea2_txtfusion_resume_smoke/krea2_txtfusion_resume_smoke_1.safetensors.state`
- perf record: `79.473964471` seconds/step, peak VRAM `2926937088`, transfers `15`, syncs `36`

Resume step:

```bash
LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
timeout 1200 target/serenity_krea2_live_trainer_txtfusion_lt896 \
  /home/alex/trainings/krea2_giger_cache_512.safetensors \
  2 \
  serenitymojo/configs/krea2_devicegrad_realcache_txtfusion_resume_smoke.json \
  /tmp/krea2_txtfusion_resume_smoke/krea2_txtfusion_resume_smoke_1.safetensors.state \
  1 \
  krea2devicegrad
```

Observed:

- `[krea2-resume] FULL full-surface resume (A/B + AdamW moments) from ...`
- `[krea2-resume] reloaded 256 adapters; resuming at step 1 / 2`
- `grad_pairs= 256`, `streaming_syncs= 29`
- step 2 loss `0.1370`, grad norm `0.0009`, progress `72.7s/step`
- wrote `/tmp/krea2_txtfusion_resume_smoke/krea2_txtfusion_resume_smoke_2.safetensors`
- wrote `/tmp/krea2_txtfusion_resume_smoke/krea2_txtfusion_resume_smoke_2.safetensors.state`
- perf record: `75.218699162` seconds/step, peak VRAM `2926937088`, transfers `15`, syncs `36`

Uninterrupted comparison run:

```bash
LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
timeout 1800 target/serenity_krea2_live_trainer_txtfusion_lt896 \
  /home/alex/trainings/krea2_giger_cache_512.safetensors \
  2 \
  serenitymojo/configs/krea2_devicegrad_realcache_txtfusion_continuous_smoke.json \
  krea2devicegrad
```

Observed:

- step 1 loss `0.4813`, grad norm `0.0025`
- step 2 loss `0.1370`, grad norm `0.0009`
- wrote `/tmp/krea2_txtfusion_continuous_smoke/krea2_txtfusion_continuous_smoke_2.safetensors`
- wrote `/tmp/krea2_txtfusion_continuous_smoke/krea2_txtfusion_continuous_smoke_2.safetensors.state`
- perf record: `68.7869964305` seconds/step over two measured steps, peak VRAM `2926937088`, transfers `24`, syncs `68`

Surface checks:

```bash
python3 scripts/check_krea2_trainable_surface.py \
  --mojo /tmp/krea2_txtfusion_resume_smoke/krea2_txtfusion_resume_smoke_2.safetensors \
  --expect-match
python3 scripts/check_krea2_trainable_surface.py \
  --mojo /tmp/krea2_txtfusion_continuous_smoke/krea2_txtfusion_continuous_smoke_2.safetensors \
  --expect-match
```

Both passed against the local ai-toolkit LoRA output with `512` common keys,
`missing_txtfusion=0`, `shape_mismatch=0`, and `dtype_mismatch=0`.

Resume equivalence check:

```bash
python3 scripts/check_krea2_resume_equivalence.py \
  --resumed-peft /tmp/krea2_txtfusion_resume_smoke/krea2_txtfusion_resume_smoke_2.safetensors \
  --continuous-peft /tmp/krea2_txtfusion_continuous_smoke/krea2_txtfusion_continuous_smoke_2.safetensors \
  --resumed-state /tmp/krea2_txtfusion_resume_smoke/krea2_txtfusion_resume_smoke_2.safetensors.state \
  --continuous-state /tmp/krea2_txtfusion_continuous_smoke/krea2_txtfusion_continuous_smoke_2.safetensors.state \
  --atol 0.0005
```

Result:

```text
[krea2-resume-equivalence] PASS peft tensors=512 dtypes=['torch.bfloat16'] max_abs=0.0003681182861328125 max_key=diffusion_model.txtfusion.layerwise_blocks.1.attn.gate.lora_B.weight atol=0.0005
[krea2-resume-equivalence] PASS state tensors=1536 dtypes=['torch.bfloat16', 'torch.float32'] max_abs=0.0003681182861328125 max_key=diffusion_model.txtfusion.layerwise_blocks.1.attn.gate.lora_B.weight atol=0.0005
[krea2-resume-equivalence] PASS save_resume_equivalence scope=Mojo product-path bounded resume equivalence; not byte parity or ai-toolkit parity
```

Strict byte-equivalence result:

- Exact `--atol 0` comparison failed for resumed step 2 vs uninterrupted step 2:
  PEFT max abs `0.0003681182861328125`; state max abs `0.0003681182861328125`.
- A fresh one-step vs fresh one-step comparison also failed exact equality with
  PEFT/state max abs `0.0001983642578125`.
- The accepted tolerance `0.0005` is above the observed fresh-run nondeterminism
  envelope for this smoke and below one BF16 percent-scale training step drift.

Known limits:

- `KREA2_TXTFUSION_LORA` remains opt-in.
- Sampling is still blocked until txtfusion LoRA conditioning is wired into the
  inline sampler.
- Full Krea2 parity still requires ai-toolkit full-surface loss, selected
  gradients, optimizer update, and resume oracle evidence.
- Counters are visible accounting, not profiler-complete transfer/sync
  accounting.
