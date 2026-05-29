# ERNIE-Image Mojo Handoff — 2026-05-28

## Current Status

ERNIE-Image is now visible in the modular Mojo registry and has metadata,
scheduler/tensor, resident DiT pre-block runtime gates, and a bounded real-weight
layer-0 DiT smoke. The block0 smoke now validates the loaded resident/layer0
weight dtypes and shapes, reports raw resident conditioning stats, scales the
synthetic latent/text inputs, and uses a bounded AdaLN modulation for the actual
layer-0 runtime gate. The full Mistral3B encoder and full 36-layer ERNIE block
stack are still open.

- Manifest: `ernie_image` / `ernie_image_1024`
- Model root: `/home/alex/models/ERNIE-Image`
- Rust reference:
  - `/home/alex/EriDiffusion/inference-flame/src/bin/ernie_image_infer.rs`
  - `/home/alex/EriDiffusion/inference-flame/src/models/ernie_image.rs`
  - `/home/alex/EriDiffusion/inference-flame/src/models/mistral3b_encoder.rs`
  - `/home/alex/EriDiffusion/inference-flame/src/sampling/ernie_sampling.rs`
- Mojo gates:
  - `serenitymojo/models/dit/ernie_contract.mojo`
  - `serenitymojo/models/dit/ernie_image.mojo`
  - `serenitymojo/pipeline/ernie_contract_smoke.mojo`
  - `serenitymojo/pipeline/ernie_resident_smoke.mojo`
  - `serenitymojo/pipeline/ernie_block0_smoke.mojo`
  - `serenitymojo/sampling/ernie_sampling.mojo`
  - `serenitymojo/sampling/ernie_sampling_smoke.mojo`

Verified:

```bash
pixi run mojo -I . serenitymojo/pipeline/ernie_contract_smoke.mojo
pixi run mojo -I . serenitymojo/pipeline/ernie_resident_smoke.mojo
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/ernie_block0_smoke.mojo -o /tmp/ernie_block0_smoke_bound
/tmp/ernie_block0_smoke_bound
pixi run mojo -I . serenitymojo/sampling/ernie_sampling_smoke.mojo
pixi run mojo -I . serenitymojo/runtime/manifest_smoke.mojo
```

Current outputs:

```text
ERNIE-Image metadata contract PASS
ERNIE-Image resident DiT math smoke PASS
ERNIE-Image block0 real-weight smoke PASS
ERNIE FlowMatch scheduler/tensor smoke PASS
[manifest] registered paths checked/missing: 176 0
```

Latest bounded block0 stats:

```text
patch_tokens absmax: 26.625
text_tokens absmax: 24.5
temb_raw absmax: 14208.0
adaln_raw absmax: 215040.0
adaln_bounded absmax: 2.15625
seq_bounded absmax: 18.5
block0_out absmax: 30848.0
```

## Contract Facts

- Image profile: `1024x1024`, one frame.
- Latent: `[1,128,64,64]`.
- Image tokens: `4096`.
- Max text tokens: `256`.
- Total sequence: `4352`.
- DiT: hidden `4096`, layers `36`, heads `32`, head dim `128`, FFN `12288`.
- Text encoder: Mistral3B hidden `3072`, layers `26`, extract layer `24`.
- Scheduler: fixed-shift FlowMatch, `shift=3.0`, `steps=50`, `cfg=4.0`.
- Model timestep: `sigma * 1000`.
- Euler update: `latent + velocity * (sigma_next - sigma)`.
- CFG: `uncond + scale * (cond - uncond)`.
- Uncond branch: encoded empty string, not zero embeddings.
- VAE dependency: Klein VAE file under ERNIE root, 251 tensors.

## Remaining Work

1. Port Mistral3B text encoder:
   tokenizer handoff, YaRN RoPE, causal/pad mask, 26 layers, return layer 24.
2. Port ERNIE DiT:
   the resident pre-block patch projection, text projection, timestep MLP,
   shared AdaLN modulation, image-first/text-second sequence concat, full
   doubled 3-axis RoPE table builder, layer-0 QK RMSNorm, attention, and
   GELU-gated MLP now have a bounded real-weight smoke with explicit loaded-key
   dtype/shape checks. Still missing full 36-layer iteration, final
   norm/projection/unpatchify, full-size
   memory/offload policy, and an active attention-mask path if padded text is
   used instead of trimmed text.
3. Wire denoise:
   sequential CFG, fixed-shift FlowMatch scheduler, F32 latent with BF16 model
   input, periodic pool trim before VAE load.
4. Reuse/adjust Klein VAE decode:
   ERNIE DiT emits 128-channel latent, while the ERNIE VAE file is the Klein
   VAE layout that decodes through the 32-channel post-quant path.
5. Add Rust or diffusers parity fixtures before full quality tuning.
