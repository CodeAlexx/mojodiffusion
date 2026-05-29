# Microsoft Lens Sidecar Handoff - 2026-05-28

Scope: Microsoft Lens only. Stable Cascade, Helios, and Nucleus remain outside
this pass. This is a bounded sidecar start for serenitymojo metadata/docs; it
does not port MXFP4 dequant, GPT-OSS execution, full Lens DiT forward, or VAE
decode.

## Source Facts

- Reference Rust lives under `/home/alex/EriDiffusion/inference-flame`.
- Main Lens files inspected:
  - `src/models/lens_dit.rs`
  - `src/models/gpt_oss_encoder.rs`
  - `src/sampling/lens_flowmatch.rs`
  - `src/vae/lens_vae_wrapper.rs`
  - `src/bin/lens_infer.rs`
  - `lens/PORT_STATE.md`, `lens/BUILD_PLAN.md`,
    `lens/M2_D7_INTEGRATION_2026-05-23.md`
  - parity metadata under `lens/parity/captures*`
- Inference-flame says Lens is pure-Rust end-to-end: sequential
  GPT-OSS encoder -> drop encoder -> DiT -> FLUX.2 VAE. 512 5/20-step and
  1024 5-step runs were green; `PORT_STATE.md` later records a shipped 1024
  20-step run at about 38.7 minutes.
- Before this sidecar started, serenitymojo had no Lens-specific Mojo.

## Local Checkpoint Inventory

Local root: `/home/alex/.serenity/models/microsoft_lens`.

- `model_index.json`: `LensPipeline` with `FlowMatchEulerDiscreteScheduler`,
  `AutoencoderKLFlux2`, `LensGptOssEncoder`, GPT-OSS tokenizer, and
  `LensTransformer2DModel`.
- `transformer/`: two F32 safetensors shards plus index, 1264 tensors.
  Static anchors checked by the new smoke include `img_in.weight [1536,128]`,
  `txt_in.weight [1536,11520]`, block-0 `img_qkv.weight [4608,1536]`, and
  `proj_out.weight [128,1536]`.
- `text_encoder/`: three safetensors shards plus index, 459 tensors. GPT-OSS
  config is 24 layers, hidden 2880, 64 Q heads, 8 KV heads, head_dim 64,
  32 experts, top-4 token-choice routing, sliding window 128, YaRN theta
  150000/factor 32. Expert blocks/scales are U8 MXFP4 sidecar tensors.
- `tokenizer/`: GPT-OSS Harmony tokenizer plus `chat_template.jinja`.
- `vae/`: single F32 `AutoencoderKLFlux2` safetensors file, 251 tensors,
  `latent_channels=32`, `patch_size=[2,2]`, `batch_norm_eps=1e-4`.

## Contract Added

Added a Lens-only metadata/header gate:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lens_contract_smoke.mojo -o /tmp/lens_contract_smoke
/tmp/lens_contract_smoke
```

The smoke validates local Lens paths, static token geometry, and representative
safetensors headers only. It does not allocate a GPU context, does not H2D-load
weights, and does not dequantize MXFP4.

Static geometry recorded:

- 1024 image tokens: `64 x 64 = 4096` DiT tokens.
- DiT patch vector dim: `32 * 2 * 2 = 128`.
- Real encoder default: `max_text_len=512`, `txt_offset=97`, post-offset text
  tokens `415`, total DiT sequence `4511`.
- Zero-feature parity captures: text tokens `256`, total sequence `4352`.
- Default denoise: 20 steps, CFG 5.0.
- Scheduler config anchors: `FlowMatchEulerDiscreteScheduler`, 1000 train
  steps, shift `3.0`, dynamic shifting enabled, exponential time shift, and no
  Karras/exponential/beta sigma variants.
- Text smoke capture gate:
  `captures_text_smoke/{input_ids,attention_mask,hidden_layer_05,hidden_layer_11,hidden_layer_17,hidden_layer_23}.safetensors`.
  The IDs/mask are `F32 [1, 64]`; selected hidden layers are
  `BF16 [1, 64, 2880]` for layers `5, 11, 17, 23`.

Added Lens-only FlowMatch scalar and BF16 tensor scheduler gates:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/sampling/lens_flowmatch_smoke.mojo -o /tmp/lens_flowmatch_smoke
/tmp/lens_flowmatch_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/sampling/lens_flowmatch_tensor_smoke.mojo -o /tmp/lens_flowmatch_tensor_smoke
/tmp/lens_flowmatch_tensor_smoke
```

The scalar scheduler smoke validates the Rust/Diffusers scalar contract:

- `mu` uses image tokens only, not full text+image DiT sequence:
  `1024 / 16 * 1024 / 16 = 4096`.
- Default `mu(4096, 20) = 2.1980220725551165`.
- Raw sigmas are exactly `N` values, `linspace(1.0, 1.0 / N, N)`;
  default raw endpoints are `1.0` and `0.05`.
- Shifted sigmas use exponential FlowMatch shifting; default shifted midpoint
  and tail are `0.90007174` and `0.32160255`.
- The final Euler step uses `sigma_next = 0.0`; default final `dt` is
  `-0.32160255`.
- The model timestep tensor is the shifted sigma directly, not `sigma * 1000`
  and not `1 - sigma`.

The tensor smoke validates the runtime dtype contract: `dt * model_output`
rounds in BF16, the add runs in F32, and the result returns to the latent dtype.
The latest gate passed with manual step `[-0.5, 1.75, 0.75, -1.5]` and terminal
scheduler first value `-0.14453125`.

Added a small Lens-only real-weight DiT math gate:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lens_dit_qkv_smoke.mojo -o /tmp/lens_dit_qkv_smoke
/tmp/lens_dit_qkv_smoke
```

The QKV smoke reads the real transformer weights under
`/home/alex/.serenity/models/microsoft_lens/transformer` and the 1024 capture
fixtures under `/home/alex/EriDiffusion/inference-flame/lens/parity/captures`.
It runs the block-0 image QKV pre-attention path on both CFG batch rows and
four image tokens (`0, 1, 137, 4095`) across the full `4608` QKV width:

`hs -> img_in -> RMSNorm(img_norm1) -> modulate(img_mod(silu(temb))) -> img_qkv`

Latest sampled result:

- samples/QKV values: `8 / 36864`
- finite values: `36864`
- computed mean/std/absmax: `0.05466018 / 8.0732477 / 92.58603`
- captured mean/std/absmax: `0.05445449 / 8.0754855 / 92.5`
- mean/max absolute drift: `0.01656939 / 0.31386185`

This is intentionally not a full block parity test. It compares CPU F32
accumulation against captured BF16 CUDA output, so it is a finite-stats and
sampled parity guard rather than a bit-exact kernel assertion.

Added the next Lens-only DiT math gate for image Q/K RMSNorm plus Lens RoPE:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lens_dit_qk_rope_smoke.mojo -o /tmp/lens_dit_qk_rope_smoke
/tmp/lens_dit_qk_rope_smoke
```

The Q/K RoPE smoke reuses the same real transformer weights and 1024 capture
fixtures, but advances past QKV into:

`img_qkv -> split image q/k -> per-head Q/K RMSNorm -> Lens 3-axis interleaved-pair RoPE`

It builds the Lens image RoPE angles locally from the frame/H/W axes
`(8, 28, 28)`, `theta=10000`, and `scale_rope=True` centered 64x64 image-token
grid, then compares both `img_q` and `img_k` against
`block_00_step0_qk_after_rope.safetensors`.

Latest sampled result:

- samples/QK values: `8 / 24576`
- finite values: `24576`
- computed mean/std/absmax: `-0.00388171 / 0.80079766 / 7.36691799`
- captured mean/std/absmax: `-0.00391445 / 0.80080436 / 7.375`
- mean/max absolute drift: `0.00157795 / 0.05229756`

This is still a sampled CPU-side parity guard, not the production DiT runtime.
It removes the previous numeric RoPE blocker for block-0 image Q/K, but does
not yet cover text-stream Q/K, joint SDPA, output projections, or the MLP path.

Added a Lens-only text-stream block-0 pre-attention gate:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lens_dit_text_qk_rope_smoke.mojo -o /tmp/lens_dit_text_qk_rope_smoke
/tmp/lens_dit_text_qk_rope_smoke
```

The text Q/K RoPE smoke reads the real GPT-OSS text-smoke captures
(`hidden_layer_05/11/17/23`) plus the real transformer weights. It runs the
selected text layers through:

`txt_norm.{0..3} -> concat -> txt_in -> RMSNorm(txt_norm1) -> modulate(txt_mod(silu(temb))) -> txt_qkv -> split text q/k -> per-head added Q/K RMSNorm -> Lens text RoPE`

The text RoPE rows use `max(h/2, w/2) + token_index`, matching the Lens
`scale_rope=true` slice for the 1024 capture (`32 + token_index`). It samples
four text tokens (`0, 1, 17, 63`) across all 24 heads and all 64 head channels.

Latest sampled result:

- samples/QK values: `4 / 12288`
- finite values: `12288`
- computed mean/std/absmax: `-0.01903617 / 1.00893581 / 5.79719321`

This advances the text-side block0 math, but it is not yet a parity comparison
against a dedicated text Q/K capture. It also does not cover joint SDPA,
attention output projections, residual gates, or the SwiGLU MLP path.

## Blockers Before Real Mojo Port

- GPT-OSS MXFP4 expert execution is not ported in serenitymojo. The reference
  uses token-choice top-k routing and on-the-fly MXFP4 dequant; do not replace
  this with expert-choice routing.
- GPT-OSS attention needs YaRN RoPE, alternating sliding/full causal masks,
  attention sinks, GQA, and GPT-OSS-specific clamped gated activation.
- Full Lens DiT math is MM-DiT with dual streams, interleaved-pair 3-axis RoPE,
  Q/K RMSNorm, modulation, and joint SDPA. The current gates cover sampled
  block-0 image QKV, sampled image Q/K RMSNorm + RoPE, and sampled text Q/K
  RMSNorm + RoPE.
- Joint image+text SDPA, attention output projections, residual gates, and
  SwiGLU MLP are still not ported in serenitymojo.
- The production memory strategy is sequential encoder -> DiT -> VAE. Do not
  assume encoder and DiT can be co-resident on the 24 GB dev GPU.

## Next Safe Steps

1. Keep `lens_contract_smoke.mojo` as the starting gate for any future Lens
   work.
2. Keep `pipeline/lens_dit_qkv_smoke.mojo` as the first real-weight Lens DiT
   math gate.
3. Keep `pipeline/lens_dit_qk_rope_smoke.mojo` as the sampled Lens-owned image
   Q/K RMSNorm + 3-axis RoPE parity gate.
4. Keep `sampling/lens_flowmatch_smoke.mojo` and
   `sampling/lens_flowmatch_tensor_smoke.mojo` as the scalar/tensor scheduler
   gates.
5. The next Lens DiT increment should capture/reference text Q/K RoPE for
   parity or move to sampled block-0 joint attention/output if a small
   intermediate capture is available.
6. Only after those pass should MXFP4/token-choice MoE be designed. That is
   a shared text-encoder feature, not a quick Lens DiT edit.
