# SDXL, FLUX, and Klein Port Status

Date: 2026-05-26

This is the current handoff note for porting SDXL plus FLUX.2/Klein 4B and
9B from `/home/alex/EriDiffusion/inference-flame` into `serenitymojo`.
The Modular reference lives under `/home/alex/modular/max/python/max/pipelines`,
especially `architectures/flux2` and `diffusion/schedulers`.

## Hard Constraint

Production inference tensor work must stay on GPU. CPU is allowed for porting,
compile checks, scalar schedule setup, parity oracles, tokenizer/control-plane
work, file I/O, and final image serialization. Do not add production paths that
read model activations to host for math.

Existing host readbacks that must be replaced before calling a path GPU-only:

- `pipeline/zimage_pipeline.mojo`: `_cast` and seeded Gaussian noise are
  host round trips today.
- `models/vae/decoder2d.mojo`: `nchw_to_nhwc` and `nhwc_to_nchw` are documented
  as host-side transpose round trips.
- `ops/linear.mojo`, `ops/conv.mojo`, `models/vae/conv3d.mojo`: bias staging
  currently reads bias tensors to host before uploading an F32 bias copy.
- `sampling/flow_match.mojo`: `cfg_qwen` computes the last-dim L2 ratio on host.
- `models/text_encoder/qwen3_encoder.mojo` and `models/dit/zimage_dit.mojo` have
  host-built RoPE tables, and the Qwen3 text encoder still has a host-built
  causal mask; those are setup scalars/tables, but they should be reviewed for
  strict production rules if the user wants zero host setup. Z-Image DiT/VAE
  all-zero attention masks are now device allocations plus GPU memset.

PNG writeback in `image/png.mojo` is output serialization and is not model
inference.

## What Changed In This Pass

- Removed the out-of-scope Qwen-Image helper files that were started before the
  target was corrected:
  - `serenitymojo/ops/qwenimage.mojo`
  - `serenitymojo/ops/qwenimage_smoke.mojo`
- Added `serenitymojo/sampling/flux2_klein.mojo`.
  - `compute_empirical_mu(image_seq_len, num_steps)`
  - `build_flux2_sigma_schedule(num_steps, image_seq_len)`
  - `build_flux2_fixed_shift_schedule(num_steps, shift)`
  - `build_flux2_img2img_sigmas(num_steps, shift, denoise)`
  - `flux2_cfg(pred_pos, pred_neg, guidance_scale, ctx)`
  - `flux2_euler_step(latents, noise_pred, dt, ctx)`
  - `Flux2KleinScheduler`
- Added `serenitymojo/sampling/flux2_klein_smoke.mojo`.
  - This is schedule-only and does not create a `DeviceContext`.
  - Verified with `/tmp/flux2_klein_smoke`: `FLUX2/Klein schedule smoke PASS`.
- Added `serenitymojo/sampling/sdxl_euler.mojo`.
  - scaled-linear SDXL Euler sigmas/timesteps
  - SDXL initial noise multiplier and UNet input scale helpers
  - textbook SDXL CFG and eps-prediction Euler step
  - `SDXLEulerScheduler`
- Added `serenitymojo/sampling/sdxl_euler_smoke.mojo`.
  - This is scalar schedule-only and does not create a `DeviceContext`.
  - Verified with `/tmp/sdxl_euler_smoke`: `SDXL Euler schedule smoke PASS`.
- Extended `serenitymojo/models/text_encoder/qwen3_encoder.mojo` for Klein:
  - `Qwen3Config.klein_4b()`
  - `Qwen3Config.klein_9b()`
  - `klein_extract_layers()` -> `[8, 17, 26]`
  - `encode_klein(...)` -> stacked `[1, seq, 3 * hidden]`
- Added `serenitymojo/pipeline/klein9b_text_smoke.mojo`.
  - Reuses the same Qwen3 path as Z-Image with a modified Klein chat template.
  - Uses dense BF16 Qwen3-8B shards from the HF cache:
    `/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`
  - The `.serenity/models/text_encoders/qwen_3_8b.safetensors` file is
    Comfy-quantized (`.comfy_quant`, `.weight_scale*`) and is not usable by the
    current dense BF16/F16/F32 Mojo Qwen3 loader without a dequant path.
  - Verified output shapes: positive and negative `[1,512,12288]`.
- Added real-weight Klein 9B DiT smoke targets:
  - `serenitymojo/models/dit/klein_dit.mojo`
  - `serenitymojo/pipeline/klein9b_dit_smoke.mojo`
  - `serenitymojo/pipeline/klein9b_dit_full_smoke.mojo`
  - The fast smoke loads a subset of
    `/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors`
    and runs one real double-stream block plus one real single-stream block on
    GPU, then final AdaLN/projection.
  - Verified output shape: `[1,4,128]`; stats were finite
    (`mean=-0.16727221`, `std=0.56651324`, `absmax=1.4609375`).
  - The full tiny smoke loads all 201 BF16 transformer tensors and runs all
    8 double blocks + 24 single blocks on the same tiny token grid. Verified
    output shape: `[1,4,128]`; stats were finite (`mean=0.034185875`,
    `std=0.2429931`, `absmax=0.69921875`). GPU memory during the full load
    stayed under the 24GB card limit.
- Ran the Z-Image GPU pipeline after the denoise sign fix:
  - output: `/home/alex/mojodiffusion/output/zimage_first_1024.png`
  - PNG size: 1024x1024
  - final latent std: `0.77510124`
- Updated Z-Image to default to its native 1024x1024 target:
  - `pipeline/zimage_pipeline.mojo`: `HL = WL = 128`
  - `models/dit/zimage_dit.mojo`: `_zero_mask` now uses device memset.
  - `models/vae/decoder2d.mojo`: VAE mid-attn zero mask now uses device memset.

## Source Map

SDXL source in `inference-flame`:

- `src/bin/sdxl_infer.rs`: end-to-end cached-embedding text-to-image path.
- `src/bin/sdxl_encode.rs`: embedding cache path.
- `src/bin/sdxl_inpaint.rs`: inpaint path.
- `src/bin/sdxl_lora_infer.rs`: LoRA path.
- `src/bin/sdxl_vae_decode_test.rs`: VAE decode test.
- `src/models/sdxl_unet.rs`: LDM-format SDXL UNet.
- `src/models/clip_encoder.rs`: CLIP-L and CLIP-G text encoder implementation.
- `src/vae/ldm_decoder.rs`: generic LDM AutoencoderKL decoder used by SDXL.

FLUX and Klein source in `inference-flame`:

- `src/models/flux1_dit.rs`: FLUX.1 DiT. This is useful background, but the
  corrected scope is FLUX.2/Klein unless the user explicitly asks for FLUX.1.
- `src/bin/flux1_infer.rs`: FLUX.1 end-to-end path.
- `src/models/klein.rs`: FLUX.2 Klein transformer, 4B and 9B.
- `src/bin/klein_infer.rs`: Klein 4B text-to-image.
- `src/bin/klein9b_infer.rs`: Klein 9B text-to-image.
- `src/bin/klein_edit_infer.rs`, `src/bin/klein9b_edit_infer.rs`: edit paths.
- `src/bin/klein_inpaint.rs`: inpaint path.
- `src/vae/klein_vae.rs`: FLUX.2/Klein VAE encoder and decoder.
- `src/sampling/klein_sampling.rs`: dynamic-mu schedule, direct velocity Euler,
  reference IDs, and img2img schedule helpers.

Modular reference paths:

- `max/python/max/pipelines/architectures/flux2/`
  - `model_config.py`
  - `flux2_executor.py`
  - `flux2_klein_executor.py`
  - `tokenizer.py`
  - `components/cfg_combine.py`
  - `components/denoise_predict.py`
  - `components/denoise_compute.py`
  - `components/denoiser.py`
  - `components/vae_decoder.py`
  - `components/image_encoder.py`
- `max/python/max/pipelines/diffusion/schedulers/scheduling_flow_match_euler_discrete.py`
- `max/python/max/pipelines/architectures/autoencoders_modulev3/autoencoder_kl_flux2.py`
- `max/python/max/pipelines/architectures/autoencoders_modulev3/model_config.py`

## SDXL Port Notes

The Rust SDXL path expects cached text embeddings:

- `context`: `[B, 77, 2048]`, CLIP-L plus CLIP-G hidden states.
- `y`: `[B, 2816]`, pooled conditioning.
- `context_uncond` and `y_uncond` for CFG.

Core constants from `src/models/sdxl_unet.rs`:

- latent channels: 4
- image downscale: 8
- `model_channels = 320`
- `channel_mult = (1, 2, 4)`
- `num_res_blocks = 2`
- `context_dim = 2048`
- `adm_in_channels = 2816`
- `head_dim = 64`
- input transformer depths: `[0, 0, 2, 2, 10, 10]`
- middle transformer depth: `10`
- output transformer depths: `[10, 10, 10, 2, 2, 2, 0, 0, 0]`

Scheduler constants from `src/bin/sdxl_infer.rs`:

- `beta_start = 0.00085`
- `beta_end = 0.012`
- scaled-linear beta schedule
- `num_train_steps = 1000`
- leading timestep spacing with `steps_offset = 1`
- initial latent multiplier `sqrt(sigma_max^2 + 1)`
- UNet input scale `1 / sqrt(sigma^2 + 1)`
- Euler update `x += (sigma_next - sigma) * eps_pred`

Mojo reuse candidates:

- `ops/linear.mojo`, `ops/conv.mojo`, `ops/norm.mojo`, `ops/attention.mojo`
- `ops/tensor_algebra.mojo`, `ops/layout.mojo`, `ops/activations.mojo`
- `models/vae/decoder2d.mojo` and `models/vae/zimage_decoder.mojo` patterns
- `offload/block_loader.mojo` for block-level UNet loading

Missing Mojo pieces:

- `models/dit/sdxl_unet.mojo`
- `models/text_encoder/clip_encoder.mojo`
- CLIP tokenizer support, or an explicit cached-embedding-only SDXL entry point
- `models/vae/ldm_decoder.mojo` or a generalized `decoder2d` wrapper for SDXL
- full SDXL pipeline wrapper around `sampling/sdxl_euler.mojo`
- production GPU noise generation and GPU dtype cast helpers
- rectangular cross-attention support in `ops/attention.mojo` or an SDXL-local
  attention wrapper (`q` sequence length often differs from text length)
- inpaint and LoRA wiring after the base path is correct

## FLUX.2/Klein 4B and 9B Port Notes

Klein transformer constants from `src/models/klein.rs`:

| Variant | inner_dim | joint_attention_dim | double blocks | single blocks | heads | head dim |
|---|---:|---:|---:|---:|---:|---:|
| Klein 4B | 3072 | 7680 | 5 | 20 | 24 | 128 |
| Klein 9B | 4096 | 12288 | 8 | 24 | 32 | 128 |

Shared Klein constants:

- latent token channels: 128
- timestep embedding dim: 256
- SwiGLU hidden dim: `inner_dim * 3`
- RoPE axes dims: `[32, 32, 32, 32]`
- RoPE theta: `2000.0`
- no linear biases in the transformer
- no `guidance_in` and no `vector_in`
- shared modulation at model level

Pipeline shape from `klein_infer.rs` and `klein9b_infer.rs`:

- Qwen3 text encoder output, padded to 512 tokens.
- `encode_klein(...)` now provides the Rust convention of concatenating hidden
  states after layers `[8, 17, 26]` into `[1,512,7680]` for 4B or
  `[1,512,12288]` for 9B. Modular may describe the same positions as
  `[9, 18, 27]` because HF hidden-state lists include the embedding output.
- Klein chat template:
  `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
- negative prompt uses the same template.
- output 1024x1024 -> latent H/W `64x64` -> `4096` image tokens.
- `img_ids`: `[N_img, 4]` with `[0, row, col, 0]`.
- `txt_ids`: `[512, 4]`, all zeros in inference-flame. Modular's tokenizer uses
  `[0, 0, 0, arange(seq_len)]`; this should be parity-checked before locking.
- initial noise shape before packing: `[1, 128, latent_h, latent_w]`.
- packed noise shape for the transformer: `[1, latent_h * latent_w, 128]`.
- default steps: 50.
- default CFG guidance: 4.0.

VAE notes from `src/vae/klein_vae.rs` and Modular `AutoencoderKLFlux2Config`:

- packed latent channels: 128
- decoder latent channels after unpatchify: 32
- patch size: `(2, 2)`
- `batch_norm_eps = 1e-4`
- decoder path: inverse BN -> unpatchify `[B,128,H,W]` to `[B,32,2H,2W]`
  -> `post_quant_conv(32,32,1)` -> LDM decoder.
- VAE output scale factor is 8 after the unpatchified 32-channel latent, which
  means original packed 128-channel latents map to pixels by an effective factor
  of 16.

Mojo reuse candidates:

- `models/text_encoder/qwen3_encoder.mojo`
- `models/dit/klein_dit.mojo` for the current Klein 9B DiT wiring
- `ops/rope.mojo` interleaved RoPE
- `ops/activations.mojo` `swiglu`
- `ops/attention.mojo` math-mode SDPA for head dim 128
- `offload/block_loader.mojo` for 9B and any memory-limited 4B path
- `sampling/flux2_klein.mojo` added in this pass

Implemented Mojo pieces:

- `models/dit/klein_dit.mojo` has 9B config constants, single-file safetensors
  loading, Klein RoPE table construction, input projections, timestep MLP,
  shared modulation, generic double-stream blocks, generic single-stream
  blocks, final projection, and both truncated and full all-block forwards.
- `pipeline/klein9b_text_smoke.mojo` proves Qwen3-8B -> Klein conditioning
  `[1,512,12288]`.
- `pipeline/klein9b_dit_smoke.mojo` proves real 9B DiT block math on GPU for a
  fast one-double/one-single slice.
- `pipeline/klein9b_dit_full_smoke.mojo` proves all 201 DiT tensors and the
  complete 8+24 block sequence run on GPU at tiny token count.

Missing Mojo pieces:

- Add a memory-safe 1024x1024 denoise path around `models/dit/klein_dit.mojo`.
  The all-resident tiny smoke fits; production 4096 image tokens will likely
  need offloading and/or attention memory work.
- `models/vae/klein_vae.mojo`
- `pipeline/klein_pipeline.mojo`
- `pipeline/klein_edit_pipeline.mojo`
- production tokenizer/chat-template wrapper for Klein prompts around the
  existing Qwen3 tokenizer
- GPU `patchify_latents` / `unpatchify_latents` variants matching the packed
  FLUX.2 VAE layout
- GPU attention-bias or attention-mask construction if strict zero-host setup
  is required
- production GPU RNG
- LoRA support after base 4B/9B parity

## Kernel Work Likely Needed

Check `/home/alex/modular/mojo` and `/home/alex/modular/max/kernels` before
writing new kernels. When no callable Mojo/SDK primitive exists, keep kernels
small, typed, and cataloged in `docs/SERENITYMOJO_KERNELS.md`.

Likely needed kernels:

- GPU dtype cast/copy for `Tensor` to replace host `_cast`.
- GPU Gaussian RNG/noise fill.
- NCHW <-> NHWC materialized permutes for VAE and SDXL UNet.
- OIHW -> RSCF weight conversion or loader-side layout adapter for conv weights.
- GPU bias cast/staging so `linear`/`conv` do not read bias to host.
- SDXL UNet skip concat and NCHW-oriented block helpers if we do not rewrite the
  UNet to NHWC end-to-end.
- Klein/FLUX.2 packed latent patchify/unpatchify kernels.
- Last-dim reductions for norm-ratio CFG and other per-token statistics.
- Device construction of CLIP/Qwen attention masks if host setup is disallowed.

## Suggested Next Order

1. Finish strict GPU-only foundation gaps: device cast, device noise, VAE layout
   permutes, bias staging.
2. Port Klein VAE decode (`flux2-vae.safetensors`), including inverse BN and
   packed 128-channel latent unpatchify.
3. Compose `pipeline/klein_pipeline.mojo`: Qwen3 conditioning -> packed noise ->
   full Klein DiT -> direct-velocity Euler -> Klein VAE -> PNG.
4. Add/offload the production 1024x1024 Klein DiT denoise path if full resident
   weights plus long-sequence attention does not fit comfortably.
5. Port SDXL cached-embedding path: scheduler, UNet skeleton, LDM VAE decode.
   Keep CLIP tokenizer/encoder separate until the denoise path is stable.
6. Add edit/inpaint and LoRA paths after base text-to-image parity.

## Verification Done

- `pixi run mojo build -I . -Xlinker -lm serenitymojo/sampling/flux2_klein_smoke.mojo -o /tmp/flux2_klein_smoke`
- `/tmp/flux2_klein_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/sampling/sdxl_euler_smoke.mojo -o /tmp/sdxl_euler_smoke`
- `/tmp/sdxl_euler_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/models/text_encoder/smoke_compile.mojo -o /tmp/qwen3_text_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_text_smoke.mojo -o /tmp/klein9b_text_smoke`
- `/tmp/klein9b_text_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_dit_smoke.mojo -o /tmp/klein9b_dit_smoke`
- `/tmp/klein9b_dit_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_dit_full_smoke.mojo -o /tmp/klein9b_dit_full_smoke`
- `/tmp/klein9b_dit_full_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_pipeline.mojo -o /tmp/zimage_pipeline_1024`
- `/tmp/zimage_pipeline_1024`

GPU-heavy commands in this pass were the Z-Image 1024 pipeline rerun, the
Klein Qwen3-8B text smoke, the truncated Klein 9B DiT smoke, and the full
all-block Klein 9B DiT tiny-token smoke. Klein still is not a complete image
pipeline; the text conditioner and full transformer wiring now run.
