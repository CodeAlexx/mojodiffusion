# Anima Sidecar Handoff - 2026-05-28

Scope: Anima remains in the full Mojo port campaign. Stable Cascade, Helios,
and Nucleus are out of active scope and were not touched in this pass.

This is a bounded starting pass. It captures local source facts and adds a
metadata contract smoke, cached-conditioning tensor gate, and VAE latent decode
gate. No MiniTrainDIT, adapter, or Qwen3 encoder model math was implemented.

## Local Sources Inspected

- `/home/alex/EriDiffusion/inference-flame/docs/binaries/anima.md`
- `/home/alex/EriDiffusion/inference-flame/src/models/anima.rs`
- `/home/alex/EriDiffusion/inference-flame/src/bin/anima_infer.rs`
- `/home/alex/EriDiffusion/inference-flame/src/bin/anima_lora_infer.rs`
- `serenitymojo/docs/FULL_PORT_MODEL_INVENTORY_2026-05-27.md`
- `serenitymojo/docs/FULL_PORT_STATUS_2026-05-27.md`
- `serenitymojo/docs/FULL_PORT_RUNNER_CONTRACTS_2026-05-27.md`

## Source Facts

- Anima is a Cosmos Predict2 image model, but the project rule is to keep it
  independent: do not cross-import a Cosmos Mojo body when the port starts.
- Rust `AnimaConfig::default()` defines a MiniTrainDIT backbone with hidden
  size 2048, 28 blocks, 16 heads, head dim 128, GELU MLP 8192, latent channels
  16, patch size 2x2x1, and AdaLN LoRA dim 256.
- Patchify appends a zero padding-mask channel before patching, so the DiT
  input projection is 68 wide: `(16 latent + 1 mask) * 2 * 2`.
- The final DiT projection emits 64 values per patch: `16 * 2 * 2`.
- The LLM adapter is resident inside the Anima checkpoint: 6 blocks, dim 1024,
  16 heads, head dim 64, MLP 4096, vocab 32128, 1D RoPE.
- Text conditioning in the end-to-end Rust path is Qwen3 0.6B final hidden
  state with the final `model.norm` applied, masked by Qwen3 padding, plus T5
  token IDs passed as F32 into the Anima adapter, then masked by T5 padding.
- Token contract from `anima_lora_infer.rs`: `MAX_SEQ_LEN=256`,
  `QWEN3_PAD_ID=151643`, `T5_PAD_ID=0`.
- Denoise is rectified-flow Euler over a linear sigma schedule from 1.0 to 0.0,
  no timestep x1000 scaling, default 30 steps, CFG 4.5, seed 42, 1024x1024.
  The Euler update uses `dt = sigma_next - sigma` and `x = x + dt * pred`.
- Latent layout in Rust is `[1, T=1, H/8, W/8, 16]`; VAE input is permuted to
  `[1, 16, 1, H/8, W/8]`.
- `anima_infer` consumes cached `context_cond` and `context_uncond` tensors and
  saves the final latent for external decode. `anima_lora_infer` runs full
  Qwen3/T5/adapter encode, attaches runtime LoRA via `Anima::set_lora`, and
  decodes with the Rust `Wan21VaeDecoder::decode_image` over the
  Qwen-Image/Wan2.1 VAE file.
- Runtime LoRA is applied at `linear_no_bias` after the base matmul. External
  kohya Anima LoRAs use `lora_unet_blocks_{i}_<sub>` naming; the Rust LoRA
  registry also accepts the `lora_unet_net_blocks_*` variant.
- Local Anima safetensors are present under both:
  `/home/alex/EriDiffusion/Models/anima` and
  `/home/alex/.serenity/models/anima`.

## Mojo Sidecar Added

- `serenitymojo/models/dit/anima_contract.mojo`
  - Header-only contract over the local Anima safetensors.
  - Checks the local DiT, Qwen3 0.6B, and VAE paths.
  - Checks representative BF16 tensor counts and shapes:
    - DiT: 685 tensors, including x embedder, timestep embedder, final layer,
      LLM adapter, first block, and last block anchors.
    - Qwen3: 310 tensors, including embeddings, first attention projection,
      last-layer MLP, and final norm.
    - VAE: 194 tensors, including encoder input and decoder image anchors.
  - Checks static Anima 1024 geometry, patch dimensions, text length, default
    CFG/step schedule, and padding IDs.
- `serenitymojo/pipeline/anima_contract_smoke.mojo`
  - Builds and runs without GPU context or tensor allocation.
  - Anima is now registered in the shared manifest; no production runner
    boundary or model math exists yet.
  - Validates cached conditioning at
    `/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors`
    when present. The fixture was generated with the Rust encode-only path and
    contains `context_cond` and `context_uncond`, both `[1, 256, 1024]`.
  - Validates the Rust cached-context latent oracle at
    `/home/alex/EriDiffusion/inference-flame/output/anima_rust_latent.safetensors`
    when present. Expected tensor is `latent [1, 16, 1, 128, 128]`, F32.
- `serenitymojo/sampling/anima_sampling.mojo`
  - Linear 30-step FlowMatch schedule, no timestep scaling, textbook CFG, and
    `latent + velocity*(sigma_next - sigma)` Euler helpers.
- `serenitymojo/pipeline/anima_cached_context_smoke.mojo`
  - Loads `context_cond`/`context_uncond` and the Rust latent oracle into Mojo
    tensors, validates full-sidecar CFG with guidance 1.0, verifies
    zero-velocity Euler over the full latent oracle tensor, and checks tiny
    known CFG/Euler values. Latest output reports context mean_abs
    `0.0026288519998161064`, latent mean_abs `0.5931320527924981`, and
    `Anima cached-conditioning tensor smoke PASS`.
- `serenitymojo/pipeline/anima_vae_latent_smoke.mojo`
  - Decodes the Rust cached-context latent oracle through the local
    Wan/Qwen-style image VAE using `QwenImageVaeDecoder.load_wan21_keys`.
    Latest run wrote
    `/home/alex/mojodiffusion/output/anima_vae_from_rust_latent_1024.png`,
    passed in `1:23.95`, and reported `Anima VAE latent runtime smoke PASS`.

## Current Blockers

- No Anima MiniTrainDIT/adapter body exists yet. The contract smoke deliberately
  stops at metadata/header validation; runtime coverage currently comes from
  the cached-context tensor and VAE latent smokes.
- The explicit Qwen3 and T5 tokenizer paths used for the cached fixture command
  exist locally. End-to-end Mojo prompt encoding still needs tokenizer path
  resolution and wrapper code before it can run without the cached sidecar.
- The VAE file is present, header-checked, and now decoded from the Rust latent
  oracle. Full prompt-to-latent Anima model math is still unported.
- Full parity still needs a Mojo denoise wrapper, adapter/MiniTrainDIT model
  body port, and source parity captures against the Rust fixture.

## Suggested Next Step

The cached Anima conditioning sidecar is now the first Mojo tensor input gate.
Next, port a cached-context MiniTrainDIT block slice against this fixture before
attempting full prompt/tokenizer assembly. The Rust helper command used for the
current fixture is:

```bash
cd /home/alex/EriDiffusion/inference-flame
cargo run --release --bin anima_lora_infer -- \
  --prompt "A scenic landscape with a serene lake" \
  --negative "" \
  --base /home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors \
  --qwen3 /home/alex/.serenity/models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --vae /home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors \
  --qwen3-tokenizer /home/alex/Anima-Standalone-Trainer/configs/qwen3_06b/tokenizer.json \
  --t5-tokenizer /home/alex/Anima-Standalone-Trainer/configs/t5_old/tokenizer.json \
  --save-embeddings /home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors \
  --encode-only
```

The cached-context Rust denoise oracle now also exists:

```bash
cd /home/alex/EriDiffusion/inference-flame
cargo run --release --bin anima_infer -- \
  /home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors
```

Observed output:

- `context_cond/context_uncond`: `[1, 256, 1024]`
- Context mean_abs: `0.0026`
- 30-step denoise elapsed: `22.2s` (`0.74s/step`)
- Saved latent:
  `/home/alex/EriDiffusion/inference-flame/output/anima_rust_latent.safetensors`
- Latent tensor: `latent [1, 16, 1, 128, 128]`, F32, mean_abs
  `0.5931320786476135`
- The Mojo Anima contract, cached-context tensor, and VAE latent smokes validate
  these sidecar/latent/decode facts.
