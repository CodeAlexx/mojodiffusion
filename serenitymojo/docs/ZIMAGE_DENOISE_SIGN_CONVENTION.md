# Z-Image Denoise Sign Convention

Date: 2026-05-26

This note exists so a future Codex or Claude Code does not repeat the denoise
divergence investigation. The Z-Image DiT, tokenizer, text encoder, VAE, and
Euler update were parity-clean. The failure was at the boundary between the raw
transformer output and the diffusers scheduler.

## Rule

For the diffusers `ZImagePipeline`, the transformer returns a raw velocity-like
tensor. The pipeline then applies CFG and negates the result before calling
`FlowMatchEulerDiscreteScheduler.step`.

The Mojo pipeline must match this sequence:

```text
pred_raw = v_cond + cfg * (v_cond - v_uncond)
model_output = -pred_raw
x_next = x + (sigma_next - sigma) * model_output
```

Do not move this sign flip into `NextDiT.forward`. The DiT parity files compare
the raw transformer output against diffusers forward-hook captures, and those
captures happen before `noise_pred = -noise_pred`.

## What Was Changed

- `serenitymojo/pipeline/zimage_pipeline.mojo`
  - Keeps latent state in F32 and feeds BF16 to DiT.
  - Computes raw CFG as `vc + cfg * (vc - vu)`.
  - Applies `pred = -pred` before the Euler update.
  - Uses full float32 diffusers sigma constants instead of 5-decimal rounded
    constants.
- `serenitymojo/pipeline/parity/parity_denoise.mojo`
  - Uses the same post-CFG negate.
  - Uses `CFG=4.0`, matching the saved diffusers `lat_step_NN.bin` oracle
    dumps.
  - Uses the same full float32 sigma constants.
- `STATUS_ZIMAGE_DENOISE_DIVERGENCE.md`
  - Updated from "unresolved precision/radial-bias" to "likely resolved, GPU
    verify pending".

## CPU-Only Proof From Existing Dumps

These files already exist under `serenitymojo/pipeline/parity/`:

- `noise.bin`
- `lat_step_00.bin`
- `velc_00.bin`
- `velu_00.bin`

Using raw CFG output gives the old wrong direction:

```text
plus_raw:  std=1.00903082
lat_step_00 oracle: std=0.99900973
plus_raw vs lat_step_00:  rmse=0.016076218
```

Using the diffusers post-CFG negate matches step 0:

```text
minus_raw: std=0.99898678
lat_step_00 oracle: std=0.99900973
minus_raw vs lat_step_00: rmse=0.001667794
```

Reproduce without touching the GPU:

```bash
cd /home/alex/mojodiffusion
python3 - <<'PY'
import os
import numpy as np

PD = "serenitymojo/pipeline/parity"

def load(name):
    shape = tuple(map(int, open(os.path.join(PD, name + ".shape")).read().split(",")))
    return np.fromfile(os.path.join(PD, name + ".bin"), dtype="<f4").reshape(shape)

noise = load("noise")
lat0 = load("lat_step_00")
vc = load("velc_00").reshape(noise.shape)
vu = load("velu_00").reshape(noise.shape)
pred = vc + 4.0 * (vc - vu)

s = np.linspace(1.0, 0.0, 30, dtype=np.float32)
s = (6.0 * s) / (1.0 + 5.0 * s)
dt = float(s[1] - s[0])

for label, arr in [("plus_raw", noise + dt * pred), ("minus_raw", noise + dt * (-pred)), ("lat_step_00", lat0)]:
    print(f"{label}: mean={arr.mean():+.8f} std={arr.std():.8f}")
PY
```

## Verification When GPU Is Free

Run the denoise parity first:

```bash
cd /home/alex/mojodiffusion
pixi run mojo run -I . serenitymojo/pipeline/parity/parity_denoise.mojo
```

Then run the image pipeline:

```bash
pixi run mojo run -I . serenitymojo/pipeline/zimage_pipeline.mojo
```

Expected pipeline behavior after the fix: final latent std should move from the
old broken value around `3.36` toward the diffusers reference around `0.75`.

For AOT compile checks without running the GPU-heavy path, pass libm explicitly:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_pipeline.mojo -o /tmp/mojo_zimage_pipeline_check
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/parity/parity_denoise.mojo -o /tmp/mojo_parity_denoise_check
```

