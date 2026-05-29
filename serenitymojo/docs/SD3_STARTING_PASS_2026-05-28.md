# SD3 Starting Pass - 2026-05-28

Scope: SD3.5 Large plus local "small" mapping to SD3.5 Medium metadata,
scheduler, and embedded VAE runtime gates. Helios, Nucleus, and Stable Cascade
remain removed from active scope; no Lance, offload, HiDream, SenseNova, Lens,
ERNIE, Chroma, or Z-Image L2P internals were touched.

## Rust Reference Read

- `/home/alex/EriDiffusion/inference-flame/src/bin/sd3_medium_infer.rs`,
  `/home/alex/EriDiffusion/inference-flame/src/bin/sd3_lora_infer.rs`, and
  `/home/alex/EriDiffusion/inference-flame/src/bin/sd3_inpaint.rs`
  - Medium variants share the triple text encoder, shifted flow schedule, and
    SD3 VAE constants, but include dual-attention/model-path differences.
  - The local "small" SD3 lane maps to SD3.5 Medium, specifically the Rust
    reference path
    `/home/alex/.serenity/models/checkpoints/stablediffusion35_medium.safetensors`.
- `/home/alex/EriDiffusion/inference-flame/src/bin/sd3_infer.rs`
  - Lines 21-30: CLIP-L, CLIP-G, T5-XXL, and SD3.5 Large checkpoint paths.
  - Lines 41-54: `steps=28`, `cfg=4.5`, `shift=3.0`, `1024x1024`,
    CLIP length 77, T5 length 256, VAE latent channels 16, scale 1.5305,
    shift 0.0609.
  - Lines 60-71: shifted rectified-flow schedule.
  - Lines 184-193: CLIP-L and CLIP-G hidden states are zero-padded to 4096
    channels, concatenated with T5 along sequence, and pooled CLIP projections
    are concatenated to 2048.
  - Lines 247-249: Large uses `load_sd3_all_chunked` then `SD3MMDiT::new`.
  - Lines 300-310: two-pass CFG; model timestep is `t * 1000.0`.
  - Lines 353-354: VAE decode is loaded from the same SD3 checkpoint blob.
- `/home/alex/EriDiffusion/inference-flame/src/models/sd3_mmdit.rs`
  - Lines 6-14: MMDiT structure, pre-only last context block, patch embed,
    learned position embedding, no RoPE.
  - Lines 49-66: detected config fields; Medium can use dual attention.
  - Lines 72-103: hidden, channels, patch size, context dim, pooled dim,
    timestep dim, and position grid are inferred from checkpoint shapes.
  - Lines 718-779: forward contract is latent `[B,C,H,W]`, timestep `[B]`,
    context `[B,N,4096]`, pooled `[B,2048]`, returns velocity `[B,C,H,W]`.
  - Lines 795-880: loader strips `model.diffusion_model.` and loads resident
    weights plus joint blocks.

## Local Checkpoint Findings

- `sd3.5_large.safetensors`: 16G, 1167 tensors, hidden `2432`, depth `38`,
  heads `38`, no dual-attention blocks, learned position grid `192x192`.
- `stablediffusion35_medium.safetensors`: 4.8G, 909 tensors, hidden `1536`,
  depth `24`, heads `24`, first `13` image blocks have `attn2`, learned
  position grid `384x384`.
- `sd3.5_medium.safetensors`: also present and header-equivalent to
  `stablediffusion35_medium.safetensors`, but not the path used by the Rust
  `sd3_medium_infer.rs` reference.
- Hugging Face `models--stabilityai--stable-diffusion-3-medium` is only a
  partial cache marker locally, not a usable diffusers snapshot.

## Mojo Starting Step

- Added `sd3_5_large_default_manifest()` and `sd3_5_medium_default_manifest()` in
  `serenitymojo/runtime/model_manifest.mojo`.
- Registered `sd3_5_large` and `sd3_5_medium` in
  `serenitymojo/registry/checkpoints.mojo`.
- Extended `serenitymojo/runtime/manifest_smoke.mojo` to require both SD3
  entries.
- Added `serenitymojo/models/dit/sd3_contract.mojo`.
  - Shared SD3.5 Large/Medium manifest, checkpoint header, shape/byte-size
    anchor, and shifted-flow scalar schedule helpers for the 1024 lanes.
  - Captures `steps=28`, `shift=3.0`, CFG `4.5`, model timestep `t*1000`, and
    SD3 VAE scale/shift constants without importing model math.
  - Adds explicit 1024-token geometry checks: image `1024x1024`, latent
    `128x128`, patch grid `64x64`, image tokens `4096`, patch vector dim `64`,
    latent elements `262144`, and learned position grid capacity `192x192`.
- Refactored `serenitymojo/pipeline/sd3_pipeline_contract_smoke.mojo` to call
  the shared SD3 contract helpers.
  - Header-only safetensors checks; it opens the checkpoint metadata but does
    not create a `DeviceContext`, import model math, or load tensors.
  - Checks Large checkpoint tensor count `1167`, MMDiT resident shape anchors,
    `38` block Large last-context pre-only modulation shape, and embedded VAE
    `first_stage_model.decoder.conv_in.weight [512,16,3,3]`.
  - Also checks the latent-to-token plan used by the Rust forward path before
    any MMDiT math is ported.
- Added `serenitymojo/pipeline/sd3_medium_pipeline_contract_smoke.mojo`.
  - Header-only safetensors checks for the local Medium checkpoint: `909`
    tensors, hidden `1536`, depth `24`, heads `24`, first `13` image blocks
    with `attn2`, single-attention block `13`, pre-only final context block,
    and embedded VAE anchor.
- Added `serenitymojo/pipeline/sd3_schedule_smoke.mojo`.
  - Scalar-only smoke for the Large and Medium shifted-flow schedules; checks
    first/mid/final sigmas, model timestep scaling, descending order, and
    negative Euler deltas.
- Added `serenitymojo/sampling/sd3_flow_match.mojo` and
  `serenitymojo/sampling/sd3_flow_match_smoke.mojo`.
  - Tiny tensor smoke for the production scheduler surface without MMDiT/VAE
    weights: Large/Medium defaults, textbook SD3 CFG, `sigma * 1000` model
    timestep, negative Euler deltas, and
    `latent + velocity * (sigma_next - sigma)` updates.
- Extended `serenitymojo/models/vae/ldm_decoder.mojo` with
  `load_sd3_embedded_ldm_decoder[LH,LW]`.
  - Loads the local checkpoint's `first_stage_model.decoder.*` LDM-format keys,
    uses SD3 scale/shift `(1.5305, 0.0609)`, and disables post-quant conv
    because the checkpoint does not carry `first_stage_model.post_quant_conv`.
- Added `serenitymojo/pipeline/sd3_vae_smoke.mojo`.
  - Decodes deterministic BF16 latent noise `[1,16,128,128]` through the
    embedded VAE and writes `output/sd3_vae_noise_1024.png`.
  - Latest AOT run passed in `2:33.52`, max RSS `788140KB`.
- Added `serenitymojo/pipeline/sd3_medium_vae_smoke.mojo`.
  - Decodes deterministic BF16 latent noise `[1,16,128,128]` through the
    Medium embedded VAE and writes `output/sd3_medium_vae_noise_1024.png`.
  - Latest AOT run passed in `2:40.65`, max RSS `788296KB`.
- Added `serenitymojo/models/dit/sd3_mmdit.mojo`.
  - Real-weight resident pre/post-block gate for Large and local "small" Medium:
    NCHW latent patchify -> `x_embedder.proj` -> centered learned `pos_embed`
    crop, `sigma*1000` timestep MLP, pooled `y_embedder` MLP, and
    `context_embedder` projection.
  - Also gates the post-block resident tail: final no-affine LayerNorm, final
    AdaLN modulation, final linear projection, and unpatchify back to
    `[1,16,128,128]`.
- Added `serenitymojo/pipeline/sd3_mmdit_preblock_smoke.mojo`.
  - Runs Large and Medium pre/post-block MMDiT math from the local combined
    checkpoints. Latest source run passed with Large x/c/context/final/latent
    max_abs `1.4453125 / 9.25 / 0.40234375 / 10.25 / 10.25` and Medium
    `2.359375 / 10.3125 / 1.265625 / 19.5 / 19.5`.

## Residual Blockers

- No full Mojo SD3 MMDiT forward yet for Large or Medium: joint attention
  concat/split, QK norm, Medium `attn2`, and pre-only last context block
  remain. Patch embed, no-RoPE learned position crop, final layer, and
  unpatchify are now runtime-gated.
- No SD3 prompt encoder assembly yet: CLIP-L, CLIP-G, T5-XXL staged loading,
  zero-pad-to-4096, concatenation, pooled projection, and negative branch.
- No production SD3 pipeline: shifted flow schedule, `t*1000` timestep, and
  tensor CFG/Euler helpers plus VAE decode are captured and smoke-tested, but
  are not wired to SD3 MMDiT model math.
- Medium is registered and covered by contract/scheduler/VAE plus MMDiT
  pre-block smokes, but has no joint-block MMDiT forward or prompt assembly yet.
