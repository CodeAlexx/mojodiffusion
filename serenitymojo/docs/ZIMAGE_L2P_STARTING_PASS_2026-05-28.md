# Z-Image L2P Starting Pass - 2026-05-28

Scope: bounded SerenityMojo sidecar pass for **Z-Image L2P** only. Treat this
as a VAE-less / pixel-space Z-Image variant, not a separate generic L2P family.
Stable Cascade, Helios, and Nucleus remain out of active scope.

The initial pass did not port full DiT or production MicroDiffusionModel math.
The follow-up L2P gates now load real DiT pre-block and `local_decoder.*`
checkpoint weights, execute patch/time/caption embeddings, and run a full tiny
MicroDiffusionModel path through all encoder/decoder stages. The broader pass added the
scheduler contract, a real cached-conditioning sidecar, Rust pixel oracles for
future Mojo parity, and a 1024 BF16 pixel patchify16 roundtrip runtime gate for
the VAE-less data path.

## Source Facts

- Rust reference entrypoint:
  `/home/alex/EriDiffusion/inference-flame/src/bin/l2p_infer.rs`.
- Rust model references:
  `/home/alex/EriDiffusion/inference-flame/src/models/l2p/{dit.rs,local_decoder.rs,rope.rs,weight_loader.rs}`
  plus `/home/alex/EriDiffusion/inference-flame/src/sampling/l2p_sampling.rs`.
- Rust docs:
  `/home/alex/EriDiffusion/inference-flame/src/models/l2p/PORT_SPEC.md`,
  `BUILD_PLAN.md`, `PORT_STATE.md`, `MATH_AUDIT_2026-05-22.md`,
  and parity/skeptic notes in that directory.
- Local checkpoint:
  `/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors`.

Z-Image L2P is explicitly built from **Z-Image-Turbo**, not base Z-Image. It
keeps the Z-Image-style DiT body shape: hidden 3840, 30 main layers, 30 heads,
head dim 128, QK-norm, SwiGLU, adaLN, and 3-axis RoPE with axes `[32,48,48]`.

The important differences from base Z-Image are:

- **Pixel-space input/output:** denoise tensor is `[B,3,H,W]`, not
  `[B,16,H/8,W/8]` latent space.
- **No VAE:** output is already pixel-space BF16, then converted/clamped to PNG.
- **Patchify16:** pixel patch embedder is `16*16*3=768 -> 3840`; base Z-Image
  uses `2*2*16=64 -> 3840` over VAE latents.
- **Micro U-Net pixel head:** Z-Image `FinalLayer + unpatchify` is replaced by
  `MicroDiffusionModel`, a 4-stage local decoder that combines noisy pixels with
  the DiT feature map.
- **Timestep/sign convention:** the model consumes normalized flow sigma, maps
  internally to `(1 - sigma) * 1000`, and negates the U-Net output before the
  sampler. The sampler must not apply a second negate.
- **Schedule/CFG defaults:** Rust CLI defaults to 1024x1024, 30 steps, CFG 2.0,
  shift 3.0, seed 42.
- **Checkpoint dtype footgun:** the merged safetensors file is mixed precision
  on disk. Header inspection found 545 tensors total: 245 BF16 and 300 F32.
  Rust `weight_loader.rs` force-casts every weight to BF16 before runtime use.

## SerenityMojo Status

Existing SerenityMojo state before this pass had no Z-Image L2P Mojo model math.
The inventory already lists **Z-Image L2P** as a Rust-sourced variant.

This pass added metadata/schedule contracts and a pixel runtime gate:

- `serenitymojo/models/dit/zimage_l2p_contract.mojo`
- `serenitymojo/models/dit/zimage_l2p_dit.mojo`
- `serenitymojo/models/dit/zimage_l2p_local_decoder.mojo`
- `serenitymojo/pipeline/zimage_l2p_contract_smoke.mojo`
- `serenitymojo/pipeline/zimage_l2p_dit_preblock_smoke.mojo`
- `serenitymojo/pipeline/zimage_l2p_local_decoder_smoke.mojo`
- `serenitymojo/pipeline/zimage_l2p_schedule_smoke.mojo`
- `serenitymojo/pipeline/zimage_l2p_pixel_smoke.mojo`

The contract validates only static geometry and safetensors headers:

- 1024x1024 pixel-space plan,
- patch grid `64x64` and 4096 image tokens,
- patch vector dim 768,
- Z-Image L2P hidden/head/timestep constants,
- FlowMatch shifted sigma endpoints and `(1 - sigma) * 1000` timestep mapping,
- local checkpoint existence and representative tensor count/shape/dtype
  anchors, including one F32-on-disk layer anchor to preserve the BF16-coercion
  warning,
- cached-conditioning sidecar schema when present:
  `cap_feats [1, seq, 2560]` plus optional
  `cap_feats_uncond [1, seq, 2560]`, accepting BF16 or F32 because the Rust
  CLI force-casts to BF16 before the DiT.
- the FLUX-shift 30-step schedule helpers, negative Euler deltas, and
  `(1 - sigma) * 1000` model timestep mapping.
- the full 1024 VAE-less pixel pack/unpack path:
  `[1,3,1024,1024] <-> [1,4096,768]`, exact after BF16 storage.
- all 28 `local_decoder.*` tensors with Rust/PyTorch checkpoint names:
  encoder and decoder convs use `.0`, upsample convs use `.1`, bottleneck uses
  `.0`, and `out_conv` is direct.

The DiT pre-block smoke gates:

- channel-minor L2P/Z-Image patchify16,
- `all_x_embedder.16-1` pixel patch embedding,
- `(1 - sigma) * 1000` timestep embedding through `t_embedder.mlp`,
- `cap_embedder.0` RMSNorm and `cap_embedder.1` caption projection.
- the cached conditioning runtime boundary: default `cap_feats [1,32,2560]`
  and `cap_feats_uncond [1,8,2560]` are loaded from the sidecar, coerced to
  BF16, and embedded through the same caption path.

The local decoder real-weight smoke currently gates:

- `enc1.0`: BF16 NCHW noisy pixels -> NHWC conv3x3 + SiLU + L2P-local
  maxpool2x2.
- `bottleneck.0`: BF16 `cat([p4, feat_map], channels)` -> conv1x1 + SiLU,
  including the 512 + 3840 channel contract.
- `full_tiny_forward`: full 32x32 MicroDiffusionModel path through enc1-4,
  bottleneck, up4-1, dec4-1, and `out_conv`, returning `[1,3,32,32]`.

The smoke uses small spatial tensors to keep runtime bounded, but the weights
are loaded from `/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors`
and converted from checkpoint OIHW to the SerenityMojo NHWC/RSCF conv layout.

The shared manifest registry now includes the Z-Image L2P checkpoint. The
default conditioning sidecar now exists at
`/home/alex/EriDiffusion/inference-flame/output/l2p_embeddings.safetensors`
and validates as:

- `cap_feats`: BF16 `[1,32,2560]`
- `cap_feats_uncond`: BF16 `[1,8,2560]`

Rust pixel-space oracles were generated from that sidecar:

- `/home/alex/EriDiffusion/inference-flame/output/l2p_512_oracle.png`
- `/home/alex/EriDiffusion/inference-flame/output/l2p_1024_oracle.png`

Both are coherent nonblank mountain/lake images. No Mojo production runner,
text encoder wrapper, transformer block stack, native 1024 MicroDiffusionModel,
or full tensor sampler math was added.

## Next Work

1. Port only when explicitly requested: copy Z-Image patterns into
   Z-Image-L2P-specific Mojo files, keeping base `zimage_dit.mojo` and
   `zimage_pipeline.mojo` untouched.
2. Extend from pre-block DiT and full tiny MicroDiffusionModel gates to the
   first L2P transformer block. Patchify16 is already runtime-gated at 1024.
   Do not attempt full model quality claims before source parity fixtures are
   pinned.
3. Preserve Rust's BF16 coercion behavior for the mixed-precision merged
   checkpoint.
4. Carry over Rust's sign/timestep contract exactly: no extra sampler-level
   negate.
5. Use GPU parity captures from the inference-flame Z-Image L2P path before any
   native-size quality claim.

## Blockers / Unknowns

- No Z-Image L2P-specific transformer block stack exists yet.
- The native 1024 local decoder is not yet a smoke: current SerenityMojo conv2d
  is a naive NHWC kernel with compile-time shapes, and the 1024 path includes
  full-resolution decoder activations plus a `[1,1024,1024,128]` skip concat.
  That is too expensive for a routine smoke without a pinned full-decoder
  parity fixture and a better runtime/offload plan.
- The full 19.6 GB checkpoint exists, but loading every transformer tensor is
  still not the default smoke path; the new gates load only the real weights
  they execute.
  `Tensor.concat` materializes contiguous output, covering the Rust
  contiguous-after-cat requirement for the gated bottleneck.
