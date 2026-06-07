# Sampler Product Harness - 2026-06-05

Reference is OneTrainer only. The product sampler path mirrors:

1. Build `SampleConfig` from the train/sample config.
2. Stage text conditioning.
3. Run transformer denoise and call `on_update_progress(i + 1, len(timesteps))`.
4. Stage VAE decode.
5. Postprocess/save the output and call `on_sample`.

## Current Scope

`serenitymojo/sampling/product_sampler_harness.mojo` is a Measurement scaffold, not accepted speed parity and not an image sampler. It records the required stage flags and measurements, then fails loud while any product stage is missing.

Flux2 dev is represented as a distinct sampler contract (`FLUX_2_DEV` /
`flux2_dev`) because OneTrainer distinguishes it from Flux2/Klein inside
`Flux2Model.is_dev()` (`num_attention_heads == 48`). This harness may build a
Flux2 dev sample-run contract, but the product lifecycle still rejects Flux2 dev
until a real Mojo runner exists; it must not be dispatched to `train_klein_real`.

The required stage flags are:

- `sample_config_ready`
- `text_conditioning_ready`
- `transformer_denoise_ready`
- `vae_decode_ready`
- `postprocess_save_ready`
- `progress_callbacks_ready`
- `output_callback_ready`
- `timing_ready`
- `vram_ready`

The required measurement fields are:

- OneTrainer baseline `seconds/step`
- OneTrainer peak VRAM MiB
- Mojo total wall seconds
- Mojo text-stage seconds
- Mojo denoise-stage seconds
- Mojo VAE decode seconds
- Mojo postprocess/save seconds
- Mojo `seconds/step`
- Mojo peak VRAM MiB
- progress callback count
- measured step count
- explicit `speed_parity_accepted`

## Flux.1-dev Speed-Parity Evidence Gate

Flux.1-dev must not be marked as accepted sampler speed parity from a generic
harness smoke. Any accepted Flux.1-dev sampler speed-parity record must carry the
matching OneTrainer and Mojo run evidence in one place:

- OneTrainer `seconds/step`
- Mojo `seconds/step`
- OneTrainer peak VRAM
- Mojo peak VRAM
- prompt
- seed
- resolution
- steps
- cfg
- dtype
- denoise trajectory evidence

`scripts/check_flux_family_sampler_contracts.py` scans the bounded Flux.1-dev
sampler/product files for accepted speed-parity claims and fails if any claim is
missing those evidence markers.

## Qwen-Image Speed-Parity Evidence Gate

Qwen-Image must not be marked as accepted sampler speed parity from a generic
harness smoke or metadata smoke. Any accepted Qwen-Image sampler speed-parity
record must carry the matching OneTrainer and Mojo run evidence in one place:

- OneTrainer `seconds/step`
- Mojo `seconds/step`
- OneTrainer peak VRAM
- Mojo peak VRAM
- prompt
- seed
- resolution
- steps
- cfg/guidance
- dtype
- denoise trajectory evidence

`scripts/check_qwen_sampler_speed_contract.py` scans the bounded Qwen-Image
sampler/product files for accepted speed-parity claims and fails if any claim is
missing those evidence markers.

## Z-Image Speed-Parity Evidence Gate

Z-Image has comparable sampler-speed evidence, but speed parity is not accepted.
The current paired record is
`/home/alex/onetrainer-mojo/parity/zimage_sampler_speed.json`:

- identity: 1024x1024, seed 42, 28 steps, CFG 3.5, BF16
- OneTrainer: denoise `2.144007647s/step`, VAE `1.0628398s`, peak `14340 MiB`
- Mojo paired record: denoise `5.206695175s/step`, text encode
  `33.874484788s`, VAE `1.809631493s`, peak `22238 MiB`
- current Mojo-only post-cleanup record: `/tmp/zimage_speed_1step_moddev_speed.json`
  with denoise `4.302052192s/step`, text encode `10.645762139s`, VAE
  `1.487801541s`, peak `22111 MiB`

`scripts/check_zimage_sampler_contract.py --strict-speed` verifies that the
paired fields and artifacts exist. Passing that guard means comparable evidence
is present, not that speed parity is accepted. Mojo remains slower and higher
VRAM than OneTrainer.

Explicit UI/request `sample_result.v1` manifests are now treated more strictly
under `--strict-speed`: a generator-only result with `mojo.peak_vram_mib=0` is
valid split-process plumbing, but it cannot be consumed as speed/VRAM readiness
evidence. Strict speed requires positive Mojo VRAM inside any explicit result
manifest, in addition to paired OneTrainer/Mojo identity, timing, artifact, and
claim fields.

`zimage_generate.mojo --trace-denoise` is an opt-in profiler mode. It syncs the
first denoise step and currently shows the main blocker: serial CFG main-stack
passes (`main_cond` around `1.98s`, `main_uncond` around `1.91s`). OneTrainer
batches cond/uncond for CFG; the Mojo path needs a proven fast BF16 `Dh=128`
attention/batched-CFG path before this can be accepted as production sampler
speed.

## How To Run

Compile/run the Mojo harness smoke:

```bash
pixi run mojo run -I . serenitymojo/sampling/product_sampler_harness_smoke.mojo
```

Run the static OneTrainer-source guard:

```bash
python3 scripts/check_sampler_product_harness_contract.py
```

## What Still Blocks Product Sampling

The scaffold intentionally reports missing pieces until the model-specific
samplers wire real text conditioning, transformer denoise, VAE decode,
postprocess/save, progress callback counts, per-stage timing, and peak VRAM.

VAE/postprocess math and text/conditioning math are separate implementation
tasks. This harness only defines how those pieces must plug into the product
path and what numbers must be present before sampler speed parity can be
accepted.
