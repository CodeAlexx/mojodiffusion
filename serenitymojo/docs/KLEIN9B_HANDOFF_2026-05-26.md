# Klein 9B Mojo Handoff - 2026-05-26

This is the cold-start note for the current FLUX.2 Klein 9B Mojo port.

## User constraints

- Production inference tensor work must stay on GPU.
- CPU/host is acceptable for porting, tests, scalar setup, file I/O, stats, and
  final PNG serialization.
- Do not add CPU model math or host activation round-trips in the inference
  path.

## References

- Rust source of truth:
  `/home/alex/EriDiffusion/inference-flame`
  - `src/bin/klein9b_infer.rs`
  - `src/sampling/klein_sampling.rs`
  - `src/models/klein.rs`
  - `src/vae/klein_vae.rs`
- Modular source:
  `/home/alex/modular`
  - `max/python/max/pipelines/architectures/flux2/`
  - `max/python/max/pipelines/diffusion/schedulers/scheduling_flow_match_euler_discrete.py`

## Current working result

The Mojo path produces a valid native 1024x1024 PNG:

`/home/alex/mojodiffusion/output/klein9b_first_1024.png`

Confirmed output:

```text
PNG image data, 1024 x 1024, 8-bit/color RGB, non-interlaced
```

Latest verified 1024 one-step stats:

```text
init_tokens mean=-0.00066099525 std=1.0012506 absmax=4.9166226 n=524288
tokens      mean=0.0022830823  std=1.69384   absmax=8.619747  n=524288
image       mean=-0.14633068   std=0.46503106 absmax=1.4440922 n=3145728
```

This is still a one-step smoke. It proves wiring, memory, RNG, offload, and VAE
decode. It is not a quality target.

## Important semantics

- Klein uses direct velocity Euler:
  `x = x + (sigma_next - sigma_current) * pred`.
- CFG is:
  `pred_neg + guidance_scale * (pred_pos - pred_neg)`.
- Do not apply the Z-Image post-CFG sign flip to Klein.
- Timestep fed to the Klein DiT is the raw sigma as F32. The model handles its
  internal `*1000` behavior.
- Latent noise must be drawn in Rust reference layout:
  `[1,128,LH,LW]` NCHW, then packed to `[1,N_IMG,128]` NHWC token order.

## Main files added or changed

- `serenitymojo/models/dit/klein_dit.mojo`
  - `Klein9BOffloaded` keeps shared weights resident and streams
    `double_blocks.i` / `single_blocks.i` one block at a time.
  - Offload cleared the native 1024 OOM from the all-resident DiT path.
- `serenitymojo/models/vae/klein_decoder.mojo`
  - Decodes packed FLUX.2/Klein latents `[1,128,LH,LW]` to RGB.
  - Current VAE path expects F32 packed latents because `flux2-vae.safetensors`
    weights are F32.
- `serenitymojo/ops/cast.mojo`
  - GPU materialized F32/BF16/F16 casts.
- `serenitymojo/ops/random.mojo`
  - GPU deterministic standard-normal fill matching Rust `rand 0.8`
    `StdRng::seed_from_u64`: PCG32 seed expansion, ChaCha12, `Standard<f32>`,
    then Box-Muller.
- `serenitymojo/ops/random_smoke.mojo`
  - Checks distribution, same-seed determinism, changed-seed difference, and
    first 16 seed-42 samples against a Rust reference.
- `serenitymojo/pipeline/klein9b_pipeline_64_smoke.mojo`
  - Fast end-to-end Klein image smoke.
- `serenitymojo/pipeline/klein9b_pipeline_1024_smoke.mojo`
  - Native 1024 one-step end-to-end smoke.

## Weight paths

- Qwen3-8B dense HF cache:
  `/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`
- Klein 9B checkpoint:
  `/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors`
- FLUX.2/Klein VAE:
  `/home/alex/.serenity/models/vaes/flux2-vae.safetensors`

Note: the `.serenity` Qwen file is Comfy-quantized and is not usable with the
current dense Qwen3 loader. Use the dense HF cache above until a dequant loader
exists.

## Verified commands

Run from `/home/alex/mojodiffusion`.

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/ops/random_smoke.mojo -o /tmp/random_smoke
/tmp/random_smoke
```

Expected key line:

```text
rust_std_rng_first16_max_abs= 8.6426735e-07
GPU randn smoke PASS
```

Fast image smoke:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_64_smoke.mojo -o /tmp/klein9b_pipeline_64_smoke
/tmp/klein9b_pipeline_64_smoke
```

Native image smoke:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_1024_smoke.mojo -o /tmp/klein9b_pipeline_1024_smoke
/tmp/klein9b_pipeline_1024_smoke
```

The 1024 VAE decode is slow and compute-bound. A healthy run can sit quiet with
GPU utilization near 100 percent and memory around 22 GB before it prints the
final image stats.

## Known pitfalls

- The all-resident Klein 9B real 1024 DiT path OOMs; use
  `Klein9BOffloaded`.
- Do not compare this one-step output to a quality image. It is a memory and
  wiring smoke.
- Do not reuse Z-Image scheduler sign conventions for Klein.
- Do not host-fill the production latent. Use `ops/random.randn`.
- Do not load the Comfy-quantized `.serenity` Qwen file with the dense loader.
- Current long attention is math-mode for DiT Dh128 and VAE Dh512 on sm_86.
  This works, but it is slow.

## Next work

1. Turn the 1024 smoke into a practical multi-step path. Try small step counts
   first before attempting the Rust default 50 steps.
2. Optimize performance: reduce block streaming overhead and replace/tile the
   long-sequence math-mode attentions.
3. Add a non-smoke production entry point, likely `pipeline/klein_pipeline.mojo`,
   once multi-step behavior is usable.
4. Add broader RNG parity if needed: the first 16 samples are checked against
   Rust, but a full-latent parity fixture would make future regressions easier
   to catch.
5. Continue SDXL after Klein quality/perf is in better shape.
