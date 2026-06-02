# Trainer Display Contract - 2026-05-31

All trainer runtime display is pure Mojo.

## Rule

- Use `serenitymojo/training/progress_display.mojo` for on-screen trainer and sampler progress.
- Do not require Python for trainer UI.
- Python scripts may exist only for development, parity replay, or old-log filtering.
- The trainer itself must print the operator-facing line.
- Other trainers are valid reference sources. For every new Mojo trainer, capture
  baseline stats from OneTrainer/Rust/PyTorch as needed: loss curve, grad_norm,
  noising speed, step speed, save format, validation sample settings, and any
  parity tensors needed to reproduce the run. Those baselines are development
  gates; the final Mojo trainer must still run without Python.
- Treat the Rust trainer/inference stack as a bug-history cheat sheet. Many
  model-specific failures were already found and fixed there, often using
  Python/PyTorch parity to isolate the problem. When Mojo hits a strange loss,
  sampler, LoRA-save, timestep, RoPE, CFG, dtype, or offload issue, first check
  the Rust implementation and its parity notes before inventing a new theory.

## Required Trainer Line

Format:

```text
[Klein-lora] step 18/50 | epoch 1/2 | loss 0.2721 | grad_norm 0.6860 | 8.4s/step | noise 56.6M/s | elapsed 0:05:51 | ETA 0:04:27
```

Required fields:

- `step current/total`
- `epoch current/total`
- `loss`
- `grad_norm`
- seconds per train step
- noising speed
- elapsed time
- ETA

Model trainers should pass their own display name, for example `Klein-lora`,
`ZImage-lora`, or `Anima-lora`.

## Sampling Display

Sampling should use the same module and keep output compact:

```text
[Klein-sample] setup model=klein steps=20 cfg=4.0 N_IMG=1024 blocks=32
[Klein-sample] denoise step 5/20 | sigma 0.9645 | 7.3s/step | 0.1366 steps/s
[Klein-sample] saved /home/alex/mojodiffusion/output/alina_train/sample_step10.png
```

Raw per-phase spam such as every `denoise_begin`, `PROG_STAGE`, or backend debug
line should stay behind an explicit debug flag, not in the normal trainer UI.

## Resolution Dispatch

Do not create sampler files per resolution, such as `*_512_cli.mojo`,
`*_1024_cli.mojo`, or `*_1248_cli.mojo`. Mojo model shapes often need comptime
specialization, but the entry point should stay clean: one CLI/trainer hook with
runtime dispatch to supported comptime cases inside the same file/module.

Short smoke runs should not produce quality samples from under-trained LoRAs.
Use a short run to verify loss, grad_norm, saving, resume, and loader behavior.
Run visual validation at meaningful save points in a real convergence run, or
explicitly label a sample as a sampler/LoRA-load smoke.

## Current Klein State

- `train_klein_real.mojo` calls `print_trainer_progress` directly.
- `klein_sampler.mojo` calls shared sample display helpers directly.
- The old `scripts/train_progress.py` is not the runtime UI. It is retained only
  as a replay/filter helper for historical raw logs.
- Normal operator tail is:

```bash
tail -f /home/alex/mojodiffusion/output/train_klein_real.log
```

- Latest stable speed smoke on 2026-06-01 is still not accepted: about
  `4.7s/step` for Klein 9B LoRA at 512 latent training size. Keep the display
  contract, but do not call this performance parity with Rust.
