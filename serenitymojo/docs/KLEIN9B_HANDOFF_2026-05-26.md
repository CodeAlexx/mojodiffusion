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

The Mojo path now produces coherent native 1024x1024 PNGs.

Additional 20-step native 1024 outputs generated on 2026-05-26:

- `output/klein9b_fairy_fire_ice_1024.png`
- `output/klein9b_neon_portrait_20step_1024.png`
- `output/klein9b_honeycomb_eye_bee_20step_1024.png`

Validated post-reboot, after the Klein timestep `*1000` fix:

`/home/alex/mojodiffusion/output/klein9b_multistep_1024.png`

Run shape/stat summary:

```text
=== Klein 9B multistep smoke - 4 steps, grid 64 x 64 ===
init_tokens mean=-0.00066099525 std=1.0012506 absmax=4.9166226 n=524288
step 1      mean=-0.0037878805  std=0.9695445 absmax=4.762716  n=524288
step 2      mean=-0.008100661   std=0.9131945 absmax=4.4850283 n=524288
step 3      mean=-0.016325992   std=0.79481745 absmax=3.8067355 n=524288
step 4      mean=-0.058099616   std=0.7715113 absmax=4.7160587 n=524288
image       mean=-0.59562224    std=0.59540033 absmax=1.6061474 n=3145728
```

The older one-step smoke also produces a valid native 1024x1024 PNG:

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

The one-step output proves wiring, memory, RNG, offload, and VAE decode, but it
is under-denoised and not a quality target. Use the multistep smoke for image
coherence.

## Important semantics

- Klein uses direct velocity Euler:
  `x = x + (sigma_next - sigma_current) * pred`.
- CFG is:
  `pred_neg + guidance_scale * (pred_pos - pred_neg)`.
- Do not apply the Z-Image post-CFG sign flip to Klein.
- The Mojo shared `t_embedder` does not apply Klein's BFL `time_factor`, so the
  pipeline must feed the Klein DiT `sigma * 1000.0`.
- Latent noise must be drawn in Rust reference layout:
  `[1,128,LH,LW]` NCHW, then packed to `[1,N_IMG,128]` NHWC token order.

## Main files added or changed

- `serenitymojo/models/dit/klein_dit.mojo`
  - `Klein9BOffloaded` keeps shared weights resident and streams
    `double_blocks.i` / `single_blocks.i` one block at a time.
  - Offload cleared the native 1024 OOM from the all-resident DiT path.
  - `Klein9BOffloaded.forward_full_cfg` now runs positive and negative CFG
    branches through each loaded block before unloading it. This preserves the
    image output and roughly halves block H2D traffic versus calling
    `forward_full` twice per step.
- `serenitymojo/models/vae/klein_decoder.mojo`
  - Decodes packed FLUX.2/Klein latents `[1,128,LH,LW]` to RGB.
  - Current VAE path expects F32 packed latents because `flux2-vae.safetensors`
    weights are F32.
- `serenitymojo/models/vae/decoder2d.mojo`
  - NCHW/NHWC entry/exit conversions now use GPU `ops.tensor_algebra.permute`
    instead of host `to_host` loops.
- `serenitymojo/ops/embeddings.mojo`
  - `t_embedder` now casts timestep embeddings with GPU `cast_tensor` instead
    of round-tripping the tiny embedding through host.
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
- `serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo`
  - Native 1024 multistep smoke by default (`N_IMG=4096`, `LH=LW=64`).
  - Uses cached caption embeddings from `klein9b_encode_smoke.mojo`.
  - Uses `Klein9BOffloaded` so Qwen3-8B and Klein 9B never co-reside.

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

Native one-step image smoke:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_1024_smoke.mojo -o /tmp/klein9b_pipeline_1024_smoke
/tmp/klein9b_pipeline_1024_smoke
```

The 1024 VAE decode is slow and compute-bound. A healthy run can sit quiet with
GPU utilization near 100 percent and memory around 22 GB before it prints the
final image stats.

Native multistep smoke:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_encode_smoke.mojo -o /tmp/klein9b_encode
/tmp/klein9b_encode
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo -o /tmp/klein9b_ms
/tmp/klein9b_ms
```

Expected output:

```text
[done] saved /home/alex/mojodiffusion/output/klein9b_multistep_1024.png
```

Latest timed fused-CFG run, current honeycomb prompt/cache:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo -o /tmp/k_ms_cfgfused
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/k_ms_cfgfused
```

Observed result:

```text
elapsed=14:24.76 user=863.08 sys=2.67 maxrss=17806536KB
sha256(output/klein9b_honeycomb_eye_bee_20step_1024.png)
  97d8215ae1ad2fd61f5fbb5e0bd38a61b85ff75414ddf3c4d522d138f5cf7d03
```

The regenerated fused-CFG PNG was byte-identical to the pre-fused backup:
`output/klein9b_honeycomb_eye_bee_20step_1024_prefused_backup.png`.

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

## Performance references

Correctness currently uses Mojo `BlockLoader` / `Klein9BOffloaded`. For speed,
reuse the existing offload/turbo work rather than inventing a new scheduler:

- EDV2 root: `/home/alex/EriDiffusion/EriDiffusion-v2`.
- EDV2 block-swap/offload notes:
  `docs/INVESTIGATION_2026-05-12_OFFLOAD_AND_CHECKPOINTING_VS_ONETRAINER.md`.
- Flame-core FlexTensor port pieces:
  `flame-core/src/offload/{telemetry,strategy,manager,state}.rs`.
- FlexTensor upstream checkout: `/home/alex/flextensor`.
- Inference-flame turbo Klein refs:
  - `inference-flame/src/turbo/{loader,arena,block,api}.rs`
  - `inference-flame/src/bin/klein9b_infer_turbo.rs`
  - `inference-flame/src/models/klein.rs::forward_with_turbo`

The key turbo difference: VMM-backed double-buffered block slots publish
BF16-view weights in on-disk `[out,in]` layout, so the per-block transpose work
used by the regular Rust `BlockOffloader` path is avoided. The current Mojo
`Klein9BOffloaded` is correct and now avoids duplicate CFG block loads, but it
is still synchronous/block-at-a-time. The next speed path is a Mojo equivalent
of the turbo loader or the EDV2/FlexTensor lookahead/resident-set strategy.

Explorer audit findings to keep:

- Current unfused CFG path streamed about 16.25 GiB of block weights per Klein
  forward. The old 20-step CFG path did 40 forwards, or about 650 GiB of H2D
  block traffic. `forward_full_cfg` reduces this to roughly 325 GiB by loading
  each block once and running both CFG branches before unload.
- Current Dh128 attention falls back to math-mode SDPA and allocates a large
  F32 score buffer. Klein now uses `sdpa_nomask` in `_attn_rope_only`, avoiding
  materialized all-zero additive masks; further speed work needs fused/tiled
  attention and native-layout linear paths.
- VAE 1024 decode remains slow. A native 1024 VAE smoke after GPU permute
  cleanup took `elapsed=4:29.13` including build and decode. VAE is important,
  but DiT/block streaming is still the larger repeated cost.

## Next work

1. Port the Rust turbo/native-layout linear path (`linear3d_nt`) and QKV
   split/permute fusions for Klein/Lance/SenseNova reuse.
2. Add an 8-step or Rust-default-step quality target after the 4-step smoke.
3. Add a non-smoke production entry point, likely `pipeline/klein_pipeline.mojo`,
   once multi-step behavior is usable.
5. Add broader RNG parity if needed: the first 16 samples are checked against
   Rust, but a full-latent parity fixture would make future regressions easier
   to catch.
6. Continue Lance T2V first; Sensenova and HiDream are queued after Lance.
