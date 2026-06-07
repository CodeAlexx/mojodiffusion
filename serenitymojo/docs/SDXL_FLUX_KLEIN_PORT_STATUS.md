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
- `models/vae/decoder2d.mojo`: NCHW/NHWC conversion has been moved to GPU
  `permute`; older notes that call this a host transpose are superseded.
- `ops/linear.mojo`: bias add has been moved to device kernels for
  F32/BF16/F16; older notes that call this a host bias staging path are
  superseded.
- `ops/conv.mojo`, `models/vae/conv3d.mojo`: bias staging currently reads bias
  tensors to host before uploading an F32 bias copy.
- `sampling/flow_match.mojo`: `cfg_qwen` computes the last-dim L2 ratio on host.
- `sampling/hidream_o1_scheduler.mojo`: Dev Flash mode clips per-step noise by
  computing std on host. The current HiDream smoke avoids this with
  deterministic `full_n_step(1, 3.0)`, but production Dev mode needs a GPU
  reduction/clamp kernel.
- `models/text_encoder/qwen3_encoder.mojo` and `models/dit/zimage_dit.mojo` have
  host-built RoPE tables, and the Qwen3 text encoder still has a host-built
  causal mask; those are setup scalars/tables, but they should be reviewed for
  strict production rules if the user wants zero host setup. Z-Image DiT/VAE
  all-zero attention masks are now device allocations plus GPU memset.

PNG writeback in `image/png.mojo` is output serialization and is not model
inference.

## What Changed In This Pass

### 2026-06-06 Klein Trainer/Sampler Update

- Klein 9B LoRA product training now has bounded `CPU_OFFLOADED` smoke evidence
  on the local RTX 3090 Ti. The 5-step run completed with BF16 LoRA saves,
  BF16 adapter state plus F32 AdamW moments, `838,860,800` host activation-tape
  bytes, and warmed step time around `13-14s`.
- Resume from the 5-step state through step `10` completed from
  `/tmp/klein9b_cpu_offloaded_5step_smoke.safetensors`; the resume save
  `/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors` preserves the same
  BF16/F32 state split.
- Resume from the 10-step state through step `20` also completed from
  `/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors`; losses for steps
  `11-20` included several `.2-.8` range values (`0.7975194`, `0.6252446`,
  `0.70147544`, `0.712633`, `0.24021716`, `0.6145748`) and the resume20 save
  `/tmp/klein9b_cpu_offloaded_resume20_smoke.safetensors` passed the same
  BF16/F32 product artifact split. Warmed timing shows the current path is
  optimizer-bound: forward `2.52-2.68s`, backward `3.92-4.09s`, optimizer
  `6.89-7.16s`.
- The standalone sampler `serenitymojo/sampling/klein_sample_cli.mojo` now runs
  separately from the trainer, preflights checkpoint/VAE/LoRA/cap files before
  CUDA setup, and writes real 512x512 PNG smoke artifacts from the resumed
  LoRA. Guided `cfg=4.0` one-step denoise measured `24.5s/step`; the fast
  validation preset uses `cfg=1.0` and measured `3.1s/step`, including from
  the resume20 LoRA at
  `output/alina_train/klein_lora_resume20_fast512_cfg1.png`.
- The default Klein sample caps for `alina_garden` and `alina_evening`
  positive/negative prompts are now generated and validate as BF16
  `[1,512,12288]`; cap-cache readiness is no longer a sampler-contract blocker.
- Validation sampling now honors `lora_multiplier` by scaling the live adapter
  contribution scalar before upload, without changing BF16 LoRA A/B storage.
- `layer_norm`, `modulate`, and `residual_gate` now handle the Klein BF16
  activation plus F32 modulation-vector boundary by casting the small
  affine/modulation tensors internally and returning BF16 activations.
- `klein_full_finetune_inventory_smoke` validates the current Klein 9B
  transformer full-weight inventory at `201` keys, and
  `klein_full_finetune_checkpoint_smoke` saves/loads the synthetic BF16
  full-weight payload plus name manifest in that order. The new
  `klein_full_finetune_state_smoke` also binds those `201` manifest tensors to
  `604` TrainState sidecar tensors in `param.N`, `adam_m.N`, `adam_v.N`,
  `__meta__` order. This is scaffolding for the later OneTrainer
  full-finetune port; product full-finetune remains unsupported until
  product-loop dispatch/parity, runtime rebind, and parity artifacts exist.

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
- Added `serenitymojo/models/dit/sdxl_contract.mojo`.
  - Header-only SDXL contract checks for the 1024 cached-embedding path.
  - Validates the SDXL manifest profile, standalone UNet tensor count/key
    shapes, standalone LDM VAE tensor count/key shapes, and BF16 cached
    embedding schema.
  - The static checkpoint contract rejects UNet/VAE paths that drift from the
    registered SDXL manifest.
  - `pipeline/sdxl_pipeline_smoke.mojo` now calls this strict contract before
    device-loading tensors.
- Added `serenitymojo/pipeline/sdxl_contract_smoke.mojo`.
  - This is metadata-only and does not create a `DeviceContext`.
  - Verified with `/tmp/sdxl_contract_smoke`: UNet/VAE headers PASS and the
    local BF16 cached embedding artifact passes dtype/count/shape validation.
- Added `serenitymojo/models/dit/flux1_contract.mojo`.
  - Header-only FLUX.1-dev 1024 contract over the registered manifest,
    CLIP-L/T5 tokenizer and safetensors assets, the 780-tensor
    `flux1-dev.safetensors` DiT, and the 244-tensor FLUX LDM VAE.
  - The contract also has an optional captured-input header gate for
    `/home/alex/EriDiffusion/inference-flame/output/flux1_inputs.safetensors`.
  - Added `serenitymojo/pipeline/flux1_contract_smoke.mojo`.
  - Verified with `/tmp/flux1_contract_smoke`: `FLUX.1-dev pipeline contract PASS`.
  - `pipeline/flux1_pipeline_smoke.mojo` now validates that contract and derives
    CLIP/T5/DiT/VAE paths from the registered manifest instead of duplicating
    hardcoded checkpoint paths.
- Added `serenitymojo/pipeline/flux1_pipeline_cached_smoke.mojo`.
  - It consumes Rust-captured `img_packed`, `t5_hidden`, and `clip_pooled`
    tensors, bypassing the current placeholder Mojo token IDs.
  - The 20-step 1024 smoke ran through FLUX DiT, VAE decode, and PNG output in
    `17:37.42` after the no-mask SDPA patch, writing
    `output/flux1_cached_inputs.png` with coherent astronaut/horse imagery and
    identical denoise stats to the earlier cached-input run.
  - The placeholder-token `pipeline/flux1_pipeline_smoke.mojo` runtime artifact
    `output/flux1_first.png` is all-white, so prompt assembly remains the
    quality blocker.
- Added `sdpa_nomask` in `serenitymojo/ops/attention.mojo` and wired FLUX.1,
  Klein, and Z-Image Dh128 attention to it. This avoids allocating full zero
  additive masks such as `[1,24,4608,4608]` for FLUX.1 1024 full attention;
  timing is still dominated by math-mode attention and other hot-path costs.
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
- Added the first Klein VAE and image pipeline path:
  - `serenitymojo/models/vae/klein_decoder.mojo`
  - `serenitymojo/ops/cast.mojo`
  - `serenitymojo/ops/random.mojo`
  - `serenitymojo/pipeline/klein_vae_smoke.mojo`
  - `serenitymojo/pipeline/klein_vae_1024_smoke.mojo`
  - `serenitymojo/pipeline/klein9b_pipeline_64_smoke.mojo`
  - `serenitymojo/pipeline/klein9b_pipeline_1024_smoke.mojo`
  - The 1024 image smoke uses Qwen3-8B conditioning, offloaded Klein 9B DiT,
    GPU Gaussian latent noise in the Rust NCHW draw/pack order, one denoise
    step, and the FLUX.2 VAE. Output:
    `/home/alex/mojodiffusion/output/klein9b_first_1024.png`.
  - Rust-compatible GPU-RNG one-step rerun stats: initial token mean
    `-0.00066099525`, std `1.0012506`; image shape `[1,3,1024,1024]`,
    mean `-0.14633068`, std `0.46503106`, absmax `1.4440922`.
- Ran the Z-Image GPU pipeline after the denoise sign fix:
  - output: `/home/alex/mojodiffusion/output/zimage_first_1024.png`
  - PNG size: 1024x1024
  - final latent std: `0.77510124`
- Updated Z-Image to default to its native 1024x1024 target:
  - `pipeline/zimage_pipeline.mojo`: `HL = WL = 128`
  - `models/dit/zimage_dit.mojo`: `_zero_mask` now uses device memset.
  - `models/vae/decoder2d.mojo`: VAE mid-attn zero mask now uses device memset.
- Post-Lance SenseNova/HiDream runtime smokes:
  - `tokenizer/tokenizer.mojo` now supports split SenseNova tokenizer assets:
    `vocab.json` + `merges.txt` + `added_tokens.json`.
  - Added `tokenizer/sensenova_tok_check.mojo`; SenseNova tokenizer gate is
    `4/4`.
  - `models/dit/sensenova_u1.mojo` now loads only T2I-required resident shared
    weights, skipping `language_model.lm_head` and understanding-side
    `vision_model.embeddings.*`.
  - Added `models/dit/sensenova_u1_load_probe.mojo`; load probe reports
    `resident shared tensors=19`.
  - `pipeline/sensenova_u1_gen_smoke.mojo` now has a verified real-weight,
    real-token, full-layer, `64x64`, 2-step output:
    `output/sensenova_u1_smoke_64.png`.
  - `tokenizer/tokenizer.mojo` also supports both array-pair and string-pair
    BPE merge serialization, keeping Z-Image/Qwen-Image parity and fixing
    HiDream's tokenizer JSON.
  - Added `tokenizer/hidream_tok_check.mojo`; HiDream tokenizer gate is `3/3`.
  - Added `Tensor.from_view_as_bf16` and
    `BlockLoader.load_block_as_bf16` for F32-on-disk 8B checkpoints.
  - Added `HiDreamO1Offloaded[S]` and converted
    `pipeline/hidream_o1_smoke.mojo` from run-gated skeleton to real offloaded
    one-step smoke: `output/hidream_o1_smoke.png`.
  - Detailed status:
    `serenitymojo/docs/SENSENOVA_HIDREAM_HANDOFF_2026-05-26.md`.

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

- `src/models/flux1_dit.rs`: FLUX.1 DiT. FLUX.1-dev is active again as of the
  2026-05-28 user request; keep it in the P1 image bring-up lane alongside
  SDXL and SD3.
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

Current Mojo pieces:

- `models/dit/sdxl_unet.mojo` contains the LDM-format SDXL UNet scaffold and
  SDXL-local rectangular attention wiring.
- `models/dit/sdxl_attention.mojo` provides the SDXL rectangular SDPA helper.
- `models/vae/ldm_decoder.mojo` provides the SDXL LDM VAE decode path.
- `sampling/sdxl_euler.mojo` provides the scaled-linear Euler scheduler, CFG,
  input scaling, and eps-prediction Euler update helpers.
- `models/dit/sdxl_contract.mojo` provides the metadata contract for SDXL
  manifest, UNet/VAE safetensors headers, cached embedding shape/dtype, and
  the exact Rust `sdxl_encode` handoff command for the default cache artifact.
- `pipeline/sdxl_pipeline_smoke.mojo` is the cached-embedding SDXL text-to-image
  wrapper around scheduler -> UNet -> VAE -> PNG. It now runs the strict
  contract before device loads, pulls UNet/VAE paths from the registered
  manifest, keeps denoise loop state in F32, casts the UNet input to BF16, and
  casts eps predictions back to F32 before CFG/Euler. For normal development it
  runs a one-step 1024 smoke and writes `output/sdxl_one_step_1024.png`; the
  full Rust-quality schedule remains 30 steps.
- `pipeline/sdxl_pipeline_full_smoke.mojo` is the long 30-step 1024
  cached-embedding target. It writes `output/sdxl_30step_1024.png`; the first
  run completed in `53:38.90` with nonblank RGB output.
- `pipeline/sdxl_contract_smoke.mojo` validates UNet/VAE headers and validates
  cached embedding dtype/count/shape when the cache file is present. If it is
  absent, the smoke prints the expected artifact path and generator handoff.
- `pipeline/sdxl_pipeline_contract_smoke.mojo` keeps the manifest, scheduler,
  and cached-embedding pipeline contract compile-checked without opening the
  full UNet/VAE headers. If the cached embedding file is present, it now uses
  the same strict cached-embedding validator as the runnable pipeline; if absent,
  it reports static-contract pass and prints the generator handoff.
- The default cached embedding artifact now exists at
  `/home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.safetensors`
  and validates as BF16. `src/bin/sdxl_encode.rs` in inference-flame now uses a
  local BF16 safetensors writer because the shared flame-core `save_tensors`
  path writes F32 safetensors.

Default cached-embedding handoff:

```bash
cd /home/alex/EriDiffusion/inference-flame && cargo run --release --bin sdxl_encode -- --prompt '<prompt>' --negative '' --output /home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.safetensors
```

Remaining SDXL blockers:

- CLIP tokenizer plus prompt-embedding assembly if SDXL should accept raw
  prompts inside Mojo instead of relying on cached embeddings.
- Rust/diffusers parity assessment for the full 30-step artifact.
- Production GPU-only cleanup for any remaining layout conversions, bias staging,
  and conv-weight layout adaptation.
- Inpaint and LoRA wiring after the base cached-embedding path is correct.

## FLUX.1-dev Port Notes

Current FLUX.1-dev status:

- `models/dit/flux1_contract.mojo` validates the manifest, CLIP-L/T5 assets,
  FLUX DiT checkpoint, FLUX LDM VAE checkpoint, and optional cached-input
  sidecar.
- `pipeline/flux1_pipeline_smoke.mojo` builds/runs the end-to-end path from
  placeholder CLIP/T5 token IDs. A runtime run produced
  `output/flux1_first.png`, but that image is all-white and should not be used
  as a quality claim.
- `pipeline/flux1_pipeline_cached_smoke.mojo` uses the Rust-captured
  `flux1_inputs.safetensors` sidecar and produced a coherent 1024 PNG at
  `output/flux1_cached_inputs.png`. The current path uses `sdpa_nomask` instead
  of materializing a zero additive mask.

Remaining FLUX.1-dev blockers:

- tokenizer-backed CLIP/T5 prompt assembly in Mojo,
- F32 RoPE table/math parity decision versus the current BF16 table path,
- memory lifetime cleanup around encoder/DiT/VAE scopes,
- borrowed-bias/per-call allocation cleanup in the FLUX DiT hot path,
- Rust/Modular parity once the raw-prompt path feeds real conditioning.

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
- `models/vae/klein_decoder.mojo`
- `ops/cast.mojo`
- `ops/rope.mojo` interleaved RoPE
- `ops/activations.mojo` `swiglu`
- `ops/attention.mojo` math-mode SDPA for head dim 128
- `offload/block_loader.mojo` for 9B and any memory-limited 4B path
- `sampling/flux2_klein.mojo` added in this pass

Implemented Mojo pieces:

- `models/dit/klein_dit.mojo` has 9B config constants, single-file safetensors
  loading, Klein RoPE table construction, input projections, timestep MLP,
  shared modulation, generic double-stream blocks, generic single-stream
  blocks, final projection, truncated and full all-block forwards, and an
  offloaded full forward that streams one block at a time through
  `BlockLoader`.
- `models/vae/klein_decoder.mojo` decodes packed FLUX.2/Klein latents
  `[1,128,LH,LW]` to `[1,3,16*LH,16*LW]`; it applies inverse BN, packed
  128->32 unpatchify, `post_quant_conv`, the shared 2D decoder stack, and PNG
  output through the existing image writer.
- `ops/cast.mojo` provides GPU dtype casts needed for BF16 DiT output -> F32
  VAE decode.
- `pipeline/klein9b_text_smoke.mojo` proves Qwen3-8B -> Klein conditioning
  `[1,512,12288]`.
- `pipeline/klein9b_dit_smoke.mojo` proves real 9B DiT block math on GPU for a
  fast one-double/one-single slice.
- `pipeline/klein9b_dit_full_smoke.mojo` proves all 201 DiT tensors and the
  complete 8+24 block sequence run on GPU at tiny token count.
- `pipeline/klein9b_pipeline_1024_smoke.mojo` exercises the current end-to-end
  Klein 9B 1024 image path. It is intentionally one denoise step with GPU
  Gaussian noise, so it is a wiring/memory proof, not a quality target. It is
  not OneTrainer sampler/trajectory/VAE-PNG parity or speed/VRAM parity. The
  separate bounded trainer path now has `CPU_OFFLOADED` activation-tape product
  smoke evidence, but full train-ref backward/AdamW replay parity is still
  required before calling it production training parity.

Missing Mojo pieces:

- Turn the one-step 1024 smoke into a quality path: more denoise steps and
  parity checks against `inference-flame`/Modular.
- Optimize performance. The current 1024 path is slow because it streams every
  9B block from safetensors and uses math-mode SDPA for long Dh128 DiT attention
  and Dh512 VAE attention.
- `pipeline/klein_pipeline.mojo` as the non-smoke production entry point.
- `pipeline/klein_edit_pipeline.mojo`
- production tokenizer/chat-template wrapper for Klein prompts around the
  existing Qwen3 tokenizer
- Broader RNG parity coverage if needed. `ops/random_smoke.mojo` checks the
  first 16 seed-42 samples against Rust rand 0.8 `StdRng`/ChaCha12; add longer
  vector checks if future work depends on bit-identical full-latent comparisons.
- GPU attention-bias or attention-mask construction if strict zero-host setup
  is required
- LoRA support after base 4B/9B parity

## Kernel Work Likely Needed

Check `/home/alex/modular/mojo` and `/home/alex/modular/max/kernels` before
writing new kernels. When no callable Mojo/SDK primitive exists, keep kernels
small, typed, and cataloged in `docs/SERENITYMOJO_KERNELS.md`.

Likely needed kernels:

- GPU dtype cast/copy for `Tensor` to replace host `_cast` in remaining paths
  (`ops/cast.mojo` now covers F32<->BF16/F16).
- GPU Gaussian RNG/noise fill.
- NCHW <-> NHWC materialized permutes for VAE and SDXL UNet.
- OIHW -> RSCF weight conversion or loader-side layout adapter for conv weights.
- GPU bias cast/staging so `linear`/`conv` do not read bias to host.
- SDXL UNet skip concat and NCHW-oriented block helpers if we do not rewrite the
  UNet to NHWC end-to-end.
- Klein/FLUX.2 packed latent patchify/unpatchify kernels for encoder/edit paths
  (decoder-side 128->32 unpatchify exists in `klein_decoder.mojo`).
- Last-dim reductions for norm-ratio CFG and other per-token statistics.
- Device construction of CLIP/Qwen attention masks if host setup is disallowed.

## Suggested Next Order

1. Make the Klein 1024 smoke a quality path: run a practical multi-step schedule
   toward the 50-step reference, then add parity/visual checks against Rust and
   Modular.
2. Optimize Klein performance: reduce block-stream overhead and replace/tile the
   long-sequence math-mode attentions where needed.
3. Finish strict GPU-only foundation gaps: VAE layout permutes, bias staging,
   GPU text/ID/mask setup if required.
4. Move SDXL past the full cached-embedding artifact by comparing against Rust
   or diffusers, then wire raw-prompt CLIP assembly for non-cached prompts.
5. Move FLUX.1-dev past cached-input runtime smoke: tokenizer-backed CLIP/T5
   assembly, F32/BF16 RoPE decision, memory/per-call allocation cleanup, and
   Rust/Modular parity.
6. Start SD3.5 Large implementation behind its manifest/header contract: triple
   encoder assembly, MMDiT, embedded VAE decode, and shifted-flow CFG.
7. Add edit/inpaint and LoRA paths after base text-to-image parity.

## Verification Done

- `pixi run mojo build -I . -Xlinker -lm serenitymojo/sampling/flux2_klein_smoke.mojo -o /tmp/flux2_klein_smoke`
- `/tmp/flux2_klein_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/sampling/sdxl_euler_smoke.mojo -o /tmp/sdxl_euler_smoke`
- `/tmp/sdxl_euler_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sdxl_contract_smoke.mojo -o /tmp/sdxl_contract_smoke`
- `/tmp/sdxl_contract_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sdxl_pipeline_contract_smoke.mojo -o /tmp/sdxl_pipeline_contract_smoke`
- `/tmp/sdxl_pipeline_contract_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/flux1_contract_smoke.mojo -o /tmp/flux1_contract_smoke`
- `/tmp/flux1_contract_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/ops_smoke2.mojo -o /tmp/ops_smoke2`
- `/tmp/ops_smoke2`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/flux1_pipeline_cached_smoke.mojo -o /tmp/flux1_pipeline_cached_smoke`
- `/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/flux1_pipeline_cached_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sd3_pipeline_contract_smoke.mojo -o /tmp/sd3_pipeline_contract_smoke`
- `/tmp/sd3_pipeline_contract_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sdxl_pipeline_smoke.mojo -o /tmp/sdxl_pipeline_smoke`
- `/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/sdxl_pipeline_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sdxl_pipeline_full_smoke.mojo -o /tmp/sdxl_pipeline_full_smoke`
- `/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/sdxl_pipeline_full_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/models/text_encoder/smoke_compile.mojo -o /tmp/qwen3_text_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_text_smoke.mojo -o /tmp/klein9b_text_smoke`
- `/tmp/klein9b_text_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_dit_smoke.mojo -o /tmp/klein9b_dit_smoke`
- `/tmp/klein9b_dit_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_dit_full_smoke.mojo -o /tmp/klein9b_dit_full_smoke`
- `/tmp/klein9b_dit_full_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein_vae_smoke.mojo -o /tmp/klein_vae_smoke`
- `/tmp/klein_vae_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein_vae_1024_smoke.mojo -o /tmp/klein_vae_1024_smoke`
- `/tmp/klein_vae_1024_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/ops/random_smoke.mojo -o /tmp/random_smoke`
- `/tmp/random_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_64_smoke.mojo -o /tmp/klein9b_pipeline_64_smoke`
- `/tmp/klein9b_pipeline_64_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_1024_smoke.mojo -o /tmp/klein9b_pipeline_1024_smoke`
- `/tmp/klein9b_pipeline_1024_smoke`
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_pipeline.mojo -o /tmp/zimage_pipeline_1024`
- `/tmp/zimage_pipeline_1024`

GPU-heavy commands in this pass were the Z-Image 1024 pipeline rerun, the
Klein Qwen3-8B text smoke, the truncated/full Klein 9B DiT smokes, Klein VAE
64/1024 smokes, the one-step end-to-end Klein 9B 1024 image smoke, the SDXL
full 30-step cached run, and the FLUX.1 cached-input run. These paths now
produce image artifacts; quality parity and speed are the remaining work.
