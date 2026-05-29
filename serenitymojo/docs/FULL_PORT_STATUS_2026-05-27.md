# Full Port Status - 2026-05-27

Purpose: quick handoff for the modular Mojo port before switching tasks.

## Current State

The shared modular/offload substrate is now in place and compile-tested.

- `serenitymojo/runtime/`
  - Manifest/request/config/shape/production-guard scaffold exists.
  - Registry smoke covers 16 model entries: Z-Image, Klein 9B, Qwen-Image,
    Qwen-Image-Edit, Chroma, SD1.5, Lance T2V, Flux.1 Dev, SDXL,
    SenseNova U1, HiDream O1, SD3.5 Large, Anima, Microsoft Lens,
    Z-Image L2P, and ERNIE-Image.
- `serenitymojo/registry/checkpoints.mojo`
  - Validates registered checkpoint/tokenizer/sidecar paths.
  - Last smoke result: `registered paths checked/missing: 166 0`.
- `serenitymojo/offload/plan.mojo`
  - Block plans:
    - Klein 9B: 8 double blocks + 24 single blocks.
    - Qwen-Image: 60 `transformer_blocks.{i}` double-stream blocks for
      paired-CFG streaming.
    - Lance T2V: 36 `language_model.model.layers.{i}` blocks.
    - HiDream O1: 36 `model.language_model.layers.{i}` blocks.
    - SenseNova U1: 42 `language_model.model.layers.{i}` blocks.
- `serenitymojo/offload/planned_loader.mojo`
  - Plan-indexed wrapper over current synchronous `BlockLoader`.
  - Provides `prefetch`, `prefetch_next`, `await_block`, `block_count`,
    `pinned_bytes`, `PlannedBlockHandle`, and stats.
  - `pinned_bytes()` is currently `0`; real pinned/turbo storage is not built.
- `serenitymojo/offload/turbo_slots.mojo`
  - Metadata-only two-slot backend contract over `BlockPlan`.
  - Verifies staging, prepared promotion, slot reuse, metadata eviction, stale
    handle detection, prefetch hits, per-block byte/tensor hints, and planned
    pinned-slot byte pressure.
  - Still reports `pinned_bytes=0`, `async_enabled=False`, and
    `vmm_enabled=False`; `planned_pinned_bytes()` reports the bytes a real
    slot backend would reserve.
- `serenitymojo/ops/attention.mojo`
  - Added `sdpa_nomask` for known-full diffusion attention. It reuses the
    math-mode SDPA path without materializing an all-zero `[B,H,S,S]` additive
    mask and is verified against the zero-mask path in both `ops_smoke2` and
    `ops/parity/sdpa_math_parity.mojo`.
  - FLUX.1, Klein, and Z-Image DiT attention now use this no-mask path.
- `serenitymojo/sampling/lance_t2v.mojo`
  - Shared Lance shifted-flow schedule, timestep tensor helper, textbook CFG,
    GPU-only global CFG renorm, and Euler denoise step.
- `serenitymojo/sampling/flux1_dev.mojo`
  - Shared FLUX.1-dev BFL time-shift schedule, Euler delta, and packed latent
    shape contract. `flux1_contract` and `flux1_pipeline_smoke` now validate
    against this helper instead of duplicating schedule/shape constants.
- `serenitymojo/models/lance/cfg_kv_cache.mojo`
  - Metadata gate for Lance production variable-length CFG/KV-cache row
    planning: conditional text-prefix cache, text-uncond empty cache, visual
    query span, packed-index shift, per-layer branch call metadata, and
    256x256/9-frame production latent validation.
- `serenitymojo/components/artifacts.mojo`
  - Shared video frame PNG extraction/writers for `[1,3,T,H,W]` tensors,
    including first/last frame, full frame-sequence output, and ffmpeg-backed
    MP4 muxing over that sequence.
- `serenitymojo/runtime/static_dispatch.mojo`
  - Compile-only registry for SenseNova and HiDream static specialization
    profiles.
- `serenitymojo/runtime/static_entrypoints.mojo`
  - Compile-only family entrypoint contracts over that registry. SenseNova
    exposes `(L_TOKENS,TEXT_LEN)` profiles; HiDream exposes static `S` profiles
    and marks CFG as requiring a common sequence length.
- `serenitymojo/models/dit/flux1_contract.mojo`
  - Metadata-only FLUX.1-dev 1024 gate for the manifest, CLIP-L/T5 tokenizer
    and safetensors headers, the 780-tensor DiT checkpoint, and the 244-tensor
    FLUX LDM VAE. No GPU context or H2D load.
  - Also validates the captured Rust input sidecar when present:
    `/home/alex/EriDiffusion/inference-flame/output/flux1_inputs.safetensors`
    with `noise_nchw`, `img_packed`, `img_ids`, `txt_ids`, `t5_hidden`, and
    `clip_pooled`.
  - `pipeline/flux1_pipeline_smoke.mojo` now validates the same manifest
    contract and derives CLIP/T5/DiT/VAE paths from the registry before the
    placeholder-token runtime pipeline body.
  - `pipeline/flux1_pipeline_cached_smoke.mojo` bypasses the current placeholder
    Mojo tokenization path by consuming the captured Rust `img_packed`,
    `t5_hidden`, and `clip_pooled` tensors. The 20-step 1024 runtime smoke ran
    through FLUX DiT, VAE decode, and PNG output in `17:37.42` after the
    no-mask SDPA patch, writing `output/flux1_cached_inputs.png` with
    nonblank/coherent image stats. The denoise statistics match the pre-patch
    cached-input run, so the optimization preserves the trajectory. The
    older placeholder-token smoke output `output/flux1_first.png` is all-white,
    so the remaining quality blocker is prompt assembly, not DiT/VAE collapse.
- `serenitymojo/models/dit/sdxl_contract.mojo`
  - Metadata-only SDXL cached-embedding 1024 gate over the registered manifest,
    standalone UNet header, standalone SDXL LDM VAE header, and BF16 cached
    embedding schema. The local embedding cache now exists at
    `/home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.safetensors`
    and both SDXL contract smokes validate it as BF16.
  - Missing-cache messages still include the exact Rust `sdxl_encode` handoff
    command for regenerating that artifact.
  - `src/bin/sdxl_encode.rs` in inference-flame now writes BF16 safetensors for
    this sidecar because the shared flame-core `save_tensors` path hardcodes F32
    safetensors output.
  - `pipeline/sdxl_pipeline_smoke.mojo` now takes UNet/VAE paths from the
    registered manifest and keeps denoise loop state in F32, casting only the
    UNet input to BF16 and casting eps predictions back to F32 before CFG/Euler.
    The 1024 one-step smoke now runs through cached embeddings, UNet, VAE, and
    PNG output, producing `output/sdxl_one_step_1024.png`.
  - `pipeline/sdxl_pipeline_full_smoke.mojo` is a separate long GPU target for
    the full 30-step cached-embedding loop. It writes
    `output/sdxl_30step_1024.png` and keeps the quick one-step smoke cheap.
    The first run completed in `53:38.90` and produced a nonblank 1024 RGB PNG
    with per-channel means `[143.235, 134.776, 120.884]` and stddev
    `[65.828, 63.423, 57.835]`.
- `serenitymojo/pipeline/sd3_pipeline_contract_smoke.mojo`
  - Metadata/header-only SD3.5 Large gate over shared
    `models/dit/sd3_contract.mojo`, the registered manifest, local checkpoint
    tensor count, representative MMDiT anchors, shifted-flow schedule constants,
    and embedded VAE shape anchors.
- `serenitymojo/pipeline/sd3_medium_pipeline_contract_smoke.mojo`
  - Metadata/header-only SD3.5 Medium gate for the local "small" SD3 lane:
    registered `sd3_5_medium` manifest, Rust-reference checkpoint path
    `stablediffusion35_medium.safetensors`, `909` tensors, hidden `1536`,
    depth `24`, heads `24`, first `13` image blocks with `attn2`, and embedded
    VAE anchors.
- `serenitymojo/pipeline/sd3_mmdit_preblock_smoke.mojo`
  - Real-weight SD3.5 Large + Medium MMDiT pre/post-block gate. It loads
    `models/dit/sd3_mmdit.mojo` resident weights from each combined
    checkpoint, runs BF16 latent patch embedding plus centered learned
    `pos_embed` crop over the full 1024 latent grid, runs `sigma*1000`
    timestep MLP plus pooled projection MLP, and projects bounded text tokens
    through `context_embedder`. It also runs final AdaLN, final linear
    projection, and unpatchify back to `[1,16,128,128]`. Latest run reports
    Large x/c/context/final/latent max_abs
    `1.4453125 / 9.25 / 0.40234375 / 10.25 / 10.25` and Medium
    `2.359375 / 10.3125 / 1.265625 / 19.5 / 19.5`.
- `serenitymojo/sampling/sd3_flow_match_smoke.mojo`
  - SD3 tensor scheduler gate over the production runtime surface:
    Large/Medium defaults, textbook CFG, `sigma*1000` model timestep scaling,
    negative Euler deltas, and `latent + velocity*(sigma_next - sigma)` updates
    on tiny tensors.
- `serenitymojo/pipeline/sd3_vae_smoke.mojo`
  - SD3.5 Large embedded VAE runtime smoke over deterministic BF16 latent
    noise. It loads the local checkpoint's `first_stage_model.decoder.*` keys
    through the generic LDM decoder, applies SD3 scale/shift
    `(1.5305, 0.0609)`, and writes `output/sd3_vae_noise_1024.png`.
- `serenitymojo/pipeline/sd3_medium_vae_smoke.mojo`
  - SD3.5 Medium embedded VAE runtime smoke over deterministic BF16 latent
    noise. It uses the same embedded LDM decoder path and writes
    `output/sd3_medium_vae_noise_1024.png`.
- `serenitymojo/pipeline/qwenimage_contract_smoke.mojo`
  - Qwen-Image base metadata/header gate over
    `/home/alex/.serenity/models/checkpoints/qwen-image-2512`: 9-shard
    1933-tensor DiT, 4-shard 729-tensor Qwen2.5-VL text encoder, tokenizer,
    scheduler, and 194-tensor Qwen image VAE. It validates the 1024 profile
    (`latent [1,16,128,128]`, packed image tokens `4096`, max text tokens
    `1024`, total sequence `5120`) plus `DROP_IDX=34`, pad token `151643`,
    dynamic exponential FlowMatch endpoints, and representative BF16 tensor
    anchors.
- `serenitymojo/pipeline/qwenimage_pipeline_smoke.mojo`
  - Qwen-Image 512 runtime smoke now runs tokenizer -> Qwen2.5-VL text encoder
    -> 60-block streamed DiT with paired CFG -> Qwen image VAE -> PNG. It
    writes `output/qwenimage_first_512.png`. The latest AOT run completed in
    `5:16.07` wall time; max RSS was `53142404KB`. The all-resident DiT path
    remains intentionally avoided because the DiT payload exceeds 24GB VRAM.
- `serenitymojo/pipeline/qwenimage_vae_smoke.mojo`
  - Qwen-Image VAE-only runtime smoke over deterministic BF16 latent noise. It
    validates the real VAE decoder key/dtype path and writes
    `output/qwenimage_vae_noise_512.png`.
- `serenitymojo/pipeline/qwenimage_edit_contract_smoke.mojo`
  - Qwen-Image-Edit metadata/header gate over the local 2511 HF snapshot:
    5-shard 1933-tensor edit DiT, 4-shard Qwen2.5-VL text encoder, processor
    tokenizer/template, scheduler, and Qwen image VAE. It records the
    edit-specific `zero_cond_t=True` contract and target+reference geometry:
    target tokens `4096`, reference tokens `4096`, image tokens `8192`, max
    text tokens `1024`, total sequence `9216`.
- `serenitymojo/pipeline/qwenimage_edit_synthetic_512_smoke.mojo`
  - Qwen-Image-Edit synthetic runtime gate: deterministic target/reference
    latents, synthetic text states, streamed edit DiT paired CFG with
    reference `t_ref=0`, target slice extraction, Qwen VAE decode, and PNG
    output at `output/qwenimage_edit_synth_512.png`.
- `serenitymojo/pipeline/chroma_contract_smoke.mojo`
  - Chroma1-HD metadata/header gate over the single merged DiT checkpoint,
    diffusers 2-shard transformer snapshot, 2-shard T5 text encoder, tokenizer,
    scheduler, and FLUX-style VAE. It validates 1024 image geometry, T5
    `[1,512,4096]` conditioning, 19 double + 38 single blocks, the
    distilled-guidance modulation index `344`, and header counts
    `1023/219/244` for DiT/text/VAE.
- `serenitymojo/pipeline/chroma_dit_smoke.mojo`
  - Real-weight Chroma runtime slice. It loads the BF16
    `distilled_guidance_layer` tensors from `chroma1_hd_bf16.safetensors`,
    builds the approximator input for a static `N_IMG=4`, `N_TXT=8` grid, runs
    the Chroma-specific guidance MLP/residual stack, and builds FLUX/Chroma
    3-axis RoPE. It now also runs real-weight double blocks 0-1, single blocks
    0-1, and final image projection to `[1,N_IMG,64]` with finite output
    stats. Full Chroma denoise, attention-mask/CFG wrapper, VAE decode, and PNG
    pipeline are still open.
- `serenitymojo/pipeline/sd15_contract_smoke.mojo`
  - SD1.5 metadata/header gate over the local diffusers snapshot. It validates
    CLIP-L, UNet, VAE, tokenizer, and scheduler paths plus representative F32
    tensor anchors. The registered 512 profile is `latent [1,4,64,64]`,
    image tokens `4096`, text tokens `77`, total sequence `4173`.
- `serenitymojo/pipeline/sd15_vae_smoke.mojo`
  - SD1.5 VAE runtime smoke over deterministic latent noise. It handles the
    local snapshot's legacy VAE attention key spelling and writes
    `output/sd15_vae_noise_512.png`. No SD1.5 UNet denoise wrapper is ported
    in Mojo yet.
- `serenitymojo/sampling/sd15_euler_smoke.mojo`
  - SD1.5 EulerDiscreteScheduler wrapper over the same scaled-linear
    eps-prediction schedule used by the SDXL path. The scalar smoke checks
    timesteps, sigmas, initial noise sigma, and input scaling.
- `serenitymojo/pipeline/ernie_resident_smoke.mojo`
  - ERNIE-Image resident DiT math gate. It loads real ERNIE transformer weights
    for `x_embedder`, `time_embedding`, shared AdaLN, and `text_proj`, then
    produces nonzero patch tokens `[1,4096,4096]`, timestep embedding
    `[1,4096]`, AdaLN modulation `[1,24576]`, and text projection
    `[1,256,4096]` from synthetic latent/text inputs.
- `serenitymojo/pipeline/ernie_block0_smoke.mojo`
  - ERNIE-Image bounded layer-0 DiT gate. It feeds the resident projection path
    into an image-first/text-second slice, builds full doubled ERNIE 3-axis
    RoPE tables, and runs real layer-0 QK RMSNorm, half-split RoPE, SDPA,
    attention output projection, residual gates, and GELU-gated MLP. Loaded
    resident/block0 tensors are shape/dtype checked, synthetic inputs are
    scaled, and the smoke enforces bounded block0 output (`absmax <= 65536`;
    latest `30848.0`).
- `serenitymojo/pipeline/anima_contract_smoke.mojo`
  - Metadata/header-only Anima sidecar over local Anima DiT, Qwen3 0.6B, and
    Qwen-Image/Wan2.1 VAE safetensors. It checks static 1024 shape geometry,
    adapter/token constants, linear sigma/Euler sign, representative BF16
    tensor anchors, local path presence, and validates the generated cached
    conditioning fixture `context_cond`/`context_uncond` with shape
    `[1, 256, 1024]`. No Anima adapter, MiniTrainDIT, prompt encoder, or full
    denoise math is ported yet.
  - The Rust cached-context Anima oracle now runs with that fixture and writes
    `/home/alex/EriDiffusion/inference-flame/output/anima_rust_latent.safetensors`
    (`latent [1, 16, 1, 128, 128]`, F32, mean_abs `0.5931320786476135`);
    the Mojo smoke validates this latent header.
- `serenitymojo/pipeline/anima_cached_context_smoke.mojo`
  - Anima cached-context runtime tensor gate. It loads the Rust
    `context_cond`/`context_uncond` sidecar to GPU tensors, validates
    guidance=1 CFG across the full `[1,256,1024]` context tensor, loads the
    Rust latent oracle to GPU, verifies zero-velocity Euler on the full
    `[1,16,1,128,128]` latent, and checks tiny known CFG/Euler values with the
    new `sampling/anima_sampling.mojo` linear FlowMatch helper. Latest run
    passed with context mean_abs `0.0026288519998161064` and latent mean_abs
    `0.5931320527924981`.
- `serenitymojo/pipeline/anima_vae_latent_smoke.mojo`
  - Anima VAE runtime artifact gate. It loads the Rust cached-context latent
    oracle, reshapes `[1,16,1,128,128]` to image-mode `[1,16,128,128]`, casts
    to BF16, decodes it through the local Wan/Qwen-style VAE using
    `QwenImageVaeDecoder.load_wan21_keys`, and writes
    `output/anima_vae_from_rust_latent_1024.png`. Latest run passed in
    `1:23.95` wall time, max RSS `1952216KB`.
- `serenitymojo/pipeline/lens_contract_smoke.mojo`
  - Microsoft Lens sidecar metadata/header gate over the local
    `/home/alex/.serenity/models/microsoft_lens` snapshot, transformer/text/VAE
    safetensors headers, parity metadata paths, text-smoke capture headers, and
    Lens token geometry. The text smoke fixture checks `input_ids` and
    `attention_mask` as `F32 [1,64]` plus GPT-OSS selected hidden layers
    `5/11/17/23` as `BF16 [1,64,2880]`. No DeviceContext, H2D weight load,
    MXFP4 dequant, GPT-OSS execution, Lens DiT, or VAE math is ported here.
- `serenitymojo/sampling/lens_flowmatch_smoke.mojo`
  - Microsoft Lens FlowMatch scalar scheduler gate. It validates image-token
    `mu` input (`4096` for 1024), the exact N-value raw sigma list, exponential
    shift endpoints, and the final `sigma_next=0.0` Euler delta.
- `serenitymojo/sampling/lens_flowmatch_tensor_smoke.mojo`
  - Microsoft Lens GPU tensor Euler gate. It verifies the diffusers/Rust dtype
    contract where `dt * model_output` rounds in BF16, the add happens in F32,
    and the result returns to the latent dtype. Latest run passed with manual
    step `[-0.5, 1.75, 0.75, -1.5]` and terminal scheduler step first value
    `-0.14453125`.
- `serenitymojo/pipeline/lens_dit_qkv_smoke.mojo`
  - Microsoft Lens real-weight block0 sampled QKV gate. It checks existing
    hidden-state/temb/qkv captures, runs the sampled CPU-side path
    `hs -> img_in -> RMSNorm(img_norm1) -> img_mod(silu(temb)) -> img_qkv`
    using real transformer weights, and compares 36,864 QKV values to the
    captured BF16 sidecar.
- `serenitymojo/pipeline/lens_dit_qk_rope_smoke.mojo`
  - Microsoft Lens real-weight block0 image Q/K RMSNorm + RoPE gate. It splits
    the sampled QKV path, applies per-head Q/K RMSNorm and Lens-owned 3-axis
    interleaved RoPE, and compares 24,576 Q/K values to
    `block_00_step0_qk_after_rope.safetensors`.
- `serenitymojo/pipeline/lens_dit_text_qk_rope_smoke.mojo`
  - Microsoft Lens real-weight block0 text-stream Q/K RMSNorm + RoPE gate. It
    loads the captured text hidden-state sidecar, applies the text QKV path,
    per-head Q/K RMSNorm, and Lens text-position RoPE, and validates all
    12,288 sampled Q/K values are finite with latest mean/std/absmax
    `-0.01903616584596017 / 1.0089358054611002 / 5.797193213048453`.
- `serenitymojo/pipeline/zimage_l2p_contract_smoke.mojo`
  - Z-Image L2P sidecar metadata/header gate over the local
    `/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors`
    checkpoint. Treats L2P as a VAE-less pixel-space Z-Image-Turbo variant,
    not a separate generic family. It validates the cached-conditioning fixture
    at `/home/alex/EriDiffusion/inference-flame/output/l2p_embeddings.safetensors`
    (`cap_feats [1,32,2560]`, `cap_feats_uncond [1,8,2560]`, BF16). Full Mojo
    DiT, MicroDiffusionModel, sampler, and conditioning producer math are not
    ported yet.
- `serenitymojo/pipeline/zimage_l2p_schedule_smoke.mojo`
  - Z-Image L2P FlowMatch scalar scheduler gate. It validates the FLUX-shift
    30-step sigma curve, negative Euler deltas, and `(1 - sigma) * 1000`
    model timestep mapping for the pixel-space no-VAE path.
  - Rust pixel oracles now exist for the same sidecar:
    `/home/alex/EriDiffusion/inference-flame/output/l2p_512_oracle.png`
    and `/home/alex/EriDiffusion/inference-flame/output/l2p_1024_oracle.png`;
    both are coherent nonblank mountain/lake images.
- `serenitymojo/pipeline/zimage_l2p_pixel_smoke.mojo`
  - VAE-less pixel-space runtime gate for L2P. It builds a full BF16
    `[1,3,1024,1024]` RGB tensor on GPU, patchifies with `patch_size=16` to
    `[1,4096,768]`, unpatchifies back, and verifies exact roundtrip after BF16
    storage. Latest AOT run passed in `0:00.50`, max RSS `514008KB`, with
    schedule endpoints `1.0/0.75/0.0`.
- `serenitymojo/pipeline/zimage_l2p_dit_preblock_smoke.mojo`
  - Z-Image L2P real-weight DiT pre-block gate. It loads BF16 checkpoint
    weights for `all_x_embedder.16-1`, `t_embedder.mlp`, and `cap_embedder`,
    then runs channel-minor patchify16, pixel patch embedding, `(1-sigma)*1000`
    timestep embedding, and caption RMSNorm+Linear on bounded tensors plus the
    real cached `cap_feats [1,32,2560]` and `cap_feats_uncond [1,8,2560]`
    conditioning sidecar.
- `serenitymojo/pipeline/zimage_l2p_local_decoder_smoke.mojo`
  - Z-Image L2P real-weight local decoder gate. It validates all
    `local_decoder.*` checkpoint headers, loads BF16 local decoder weights,
    still reports the old `enc1+SiLU+pool` and bottleneck subgate stats, and
    now also runs a full tiny 32x32 MicroDiffusionModel path through all four
    encoder stages, bottleneck, all four decoder stages, and `out_conv`.
    Native 1024 local decoder remains open because full-resolution skip concat
    is too heavy for the current naive NHWC conv backend.
- `serenitymojo/pipeline/ernie_contract_smoke.mojo`
  - ERNIE-Image metadata/header gate over the local
    `/home/alex/models/ERNIE-Image` snapshot, 2-shard 409-tensor ERNIE DiT,
    458-tensor Mistral3B text encoder file, tokenizer/scheduler assets, and
    251-tensor Klein VAE file. It checks the 1024 profile
    (`latent [1,128,64,64]`, image tokens `4096`, max text tokens `256`,
    total sequence `4352`), Mistral hidden-state extraction layer `24`, and
    representative DiT/text/VAE tensor anchors. No full ERNIE block stack,
    Mistral encoder, or denoise/decode wrapper is ported yet.
- `serenitymojo/sampling/ernie_sampling_smoke.mojo`
  - ERNIE fixed-shift FlowMatch scheduler/tensor gate. It validates the
    50-step `shift=3.0` sigma curve, textbook CFG on tiny tensors, negative
    Euler deltas, `latent + velocity*(sigma_next - sigma)` tensor updates, and
    `sigma * 1000` model timestep mapping.

## Model Loop Status

These offloaded paths now use `PlannedBlockLoader`:

- `serenitymojo/models/dit/klein_dit.mojo`
  - `Klein9BOffloaded.forward_full`
  - `Klein9BOffloaded.forward_full_cfg`
- `serenitymojo/models/lance/lance_t2v.mojo`
  - `LanceT2VOffloaded.forward_velocity`
  - Same-static-length padded-uncond input builder for dense CFG smokes.
  - Variable-length CFG/KV-cache metadata is now explicit, but cached attention
    execution is not implemented yet.
- `serenitymojo/models/dit/hidream_o1.mojo`
  - `HiDreamO1Offloaded.forward`
  - Uses BF16 load policy for F32-on-disk layer tensors.
- `serenitymojo/models/dit/sensenova_u1.mojo`
  - `SenseNovaU1.forward_und`
  - `SenseNovaU1.forward_gen`
- Lance T2V smokes now use shared `sampling/lance_t2v.mojo` helpers, including
  padded-uncond CFG and GPU-only CFG renorm.
- Lance image/video smokes stage the run as Lance denoise first, then Wan2.2
  VAE load/decode after the Lance model leaves scope.
- Lance video smokes now use shared artifact frame extraction helpers and mux
  the saved frame sequence to MP4.
- `pipeline/lance_t2v_pipeline.mojo` now provides the compile-tested
  production-entry contract for the `256x256`, 9-frame Lance static profile.
  It points at the full dense 256x9 target and validates the decoded artifact
  set when present.
- `pipeline/lance_t2v_256_9f_dense_probe.mojo` now runs the production-shaped
  256x256/9-frame dense path: `T_lat=3,H=W=16`, `768` latent tokens,
  full 36 streamed layers, one denoise step, Wan2.2 temporal decode, 9 PNG
  frames, and MP4 mux. Verified artifacts:
  `output/lance_t2v_256_9f_dense_frame0_256.png` through frame 8 and
  `output/lance_t2v_256_9f_dense.mp4`
  (`sha256=11eee7ba7e6da88529d11d05e0274ea057dabbb206f1982f58c031bb9c29296b`).

Production invariant remains: inference model math must stay on GPU. CPU is
only acceptable for setup, tokenization, scalar schedules, file I/O, metadata,
serialization, and testing/debug scaffolding.

## Verification Run

All of these passed after the latest edits:

```bash
pixi run mojo build -I . -Xlinker -lm serenitymojo/offload/plan_smoke.mojo -o /tmp/offload_plan_smoke && /tmp/offload_plan_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/offload/planned_loader_smoke.mojo -o /tmp/planned_loader_smoke && /tmp/planned_loader_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/offload/turbo_slots_smoke.mojo -o /tmp/turbo_slots_smoke && /tmp/turbo_slots_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/runtime/manifest_smoke.mojo -o /tmp/manifest_smoke && /tmp/manifest_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/runtime/static_dispatch_smoke.mojo -o /tmp/static_dispatch_smoke && /tmp/static_dispatch_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/runtime/static_entrypoints_smoke.mojo -o /tmp/static_entrypoints_smoke && /tmp/static_entrypoints_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/ops_smoke2.mojo -o /tmp/ops_smoke2 && /tmp/ops_smoke2
pixi run mojo build -I . -Xlinker -lm serenitymojo/sampling/flux1_dev_smoke.mojo -o /tmp/flux1_dev_smoke && /tmp/flux1_dev_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/flux1_contract_smoke.mojo -o /tmp/flux1_contract_smoke && /tmp/flux1_contract_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/flux1_pipeline_smoke.mojo -o /tmp/flux1_pipeline_smoke_compile
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/flux1_pipeline_cached_smoke.mojo -o /tmp/flux1_pipeline_cached_smoke
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/flux1_pipeline_cached_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sdxl_contract_smoke.mojo -o /tmp/sdxl_contract_smoke && /tmp/sdxl_contract_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sdxl_pipeline_contract_smoke.mojo -o /tmp/sdxl_pipeline_contract_smoke && /tmp/sdxl_pipeline_contract_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sdxl_pipeline_smoke.mojo -o /tmp/sdxl_pipeline_smoke
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/sdxl_pipeline_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sd3_schedule_smoke.mojo -o /tmp/sd3_schedule_smoke && /tmp/sd3_schedule_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sd3_pipeline_contract_smoke.mojo -o /tmp/sd3_pipeline_contract_smoke && /tmp/sd3_pipeline_contract_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/sampling/sd3_flow_match_smoke.mojo -o /tmp/sd3_flow_match_smoke && /tmp/sd3_flow_match_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/anima_contract_smoke.mojo -o /tmp/anima_contract_smoke && /tmp/anima_contract_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/anima_cached_context_smoke.mojo -o /tmp/anima_cached_context_smoke && /tmp/anima_cached_context_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/anima_vae_latent_smoke.mojo -o /tmp/anima_vae_latent_smoke && /usr/bin/time -v /tmp/anima_vae_latent_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lens_contract_smoke.mojo -o /tmp/lens_contract_smoke && /tmp/lens_contract_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_l2p_contract_smoke.mojo -o /tmp/zimage_l2p_contract_smoke && /tmp/zimage_l2p_contract_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_l2p_schedule_smoke.mojo -o /tmp/zimage_l2p_schedule_smoke && /tmp/zimage_l2p_schedule_smoke
pixi run mojo -I . serenitymojo/pipeline/ernie_contract_smoke.mojo
pixi run mojo -I . serenitymojo/sampling/ernie_sampling_smoke.mojo
pixi run mojo build -I . -Xlinker -lm serenitymojo/sampling/lance_t2v_smoke.mojo -o /tmp/lance_t2v_sampling_smoke && /tmp/lance_t2v_sampling_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/models/lance/cfg_kv_cache_smoke.mojo -o /tmp/lance_cfg_kv_cache_smoke && /tmp/lance_cfg_kv_cache_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/components/artifacts_smoke.mojo -o /tmp/artifacts_smoke && /tmp/artifacts_smoke
pixi run mojo -I . serenitymojo/pipeline/lance_t2v_pipeline.mojo
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo -o /tmp/klein9b_pipeline_multistep_smoke_compile
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lance_t2v_smoke.mojo -o /tmp/lance_t2v_smoke_compile
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lance_t2v_image_smoke.mojo -o /tmp/lance_t2v_image_smoke_compile
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lance_t2v_video_smoke.mojo -o /tmp/lance_t2v_video_smoke_compile
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lance_wan22_vae_video_smoke.mojo -o /tmp/lance_wan22_vae_video_smoke_compile
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lance_t2v_pipeline.mojo -o /tmp/lance_t2v_pipeline && /tmp/lance_t2v_pipeline
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lance_t2v_smoke.mojo -o /tmp/lance_t2v_smoke_compile
/usr/bin/time -f 'elapsed=%E user=%U sys=%S maxrss=%MKB' /tmp/lance_t2v_smoke_compile
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/hidream_o1_smoke.mojo -o /tmp/hidream_o1_smoke_compile
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo -o /tmp/sensenova_u1_gen_smoke_compile
git diff --check
```

The full 30-step SDXL cached-embedding target is now present and has run
through UNet, VAE, and PNG output. The FLUX.1 cached-input target also ran
through DiT, VAE, and PNG output; the placeholder-token FLUX path produced a
white image and is intentionally not a quality signal. Full production video
inference was not run in this slice. The SDXL one-step 1024 cached-embedding
path also remains as the cheap runtime smoke; the contract/helper smokes were
executed.

## Detailed Docs

- `serenitymojo/docs/FULL_PORT_MODULAR_PLAN_2026-05-27.md`
- `serenitymojo/docs/FULL_PORT_MODEL_INVENTORY_2026-05-27.md`
- `serenitymojo/docs/FULL_PORT_RUNTIME_ARCH_2026-05-27.md`
- `serenitymojo/docs/FULL_PORT_OFFLOAD_TURBO_2026-05-27.md`
- `serenitymojo/docs/FULL_PORT_RUNNER_CONTRACTS_2026-05-27.md`
- `serenitymojo/docs/FULL_PORT_RUST_OFFLOAD_PARITY_2026-05-27.md`
- `serenitymojo/docs/LENS_SIDECAR_HANDOFF_2026-05-28.md`
- `serenitymojo/docs/SERENITYMOJO_MODULES.md`

## Remaining Work

Highest-value next port work:

1. Turn the `turbo_slots` metadata backend into a real backend:
   pinned BF16 block packing, two GPU slots, copy/compute stream events, and
   handle-gated slot reuse.
2. Implement Lance cached attention/forward using the new variable-length
   CFG/KV-cache metadata, then replace padded-uncond dense CFG and add
   sparse/flex attention for larger sequence lengths.
3. Move SDXL beyond the full 30-step artifact: assess output against
   Rust/diffusers parity, assemble prompt encoders, and clean up any remaining
   loader/layout staging.
4. Continue the SD3.5 Large/Medium Mojo bodies beyond the new resident
   pre/post-block gate: triple encoder assembly, MMDiT joint blocks,
   shifted-flow CFG, and Large-first then Medium dual-attention parity.
5. Implement the production wrapper bodies behind the SenseNova/HiDream static
   entrypoint contracts: prompt padding/dispatch for SenseNova and common-S CFG
   dispatch for HiDream.
6. Move FLUX.1-dev from cached-input runtime smoke toward production wrapper
   parity: tokenizer-backed prompt assembly, real wrapper boundaries,
   mixed-dtype F32 RoPE or accepted BF16-quality caveat, borrowed-bias/perf
   cleanup, and Rust/Modular parity checks.
7. Continue Anima from the cached-conditioning and VAE runtime gates: schema,
   fixture, Rust latent oracle, CFG, Euler runtime surface, and VAE decode to
   PNG are covered; next port a cached-context MiniTrainDIT block slice, then
   the independent adapter/body.
8. Keep Microsoft Lens visible as a sidecar: FlowMatch scalar/BF16 tensor
   Euler gates, sampled block0 QKV, sampled image Q/K RoPE, and sampled
   text-stream Q/K RoPE math are now covered; next safe work is sampled joint
   SDPA before tackling MXFP4 or GPT-OSS/full Lens model execution.
9. Continue Z-Image L2P as a Z-Image variant: cached `cap_feats` sidecar and
   Rust 512/1024 pixel oracles exist; patchify16 is now runtime-gated at 1024
   and DiT pre-block plus full tiny local-decoder gates now run real weights,
   so next port transformer blocks only behind those fixtures, preserving the
   no-VAE pixel-space sign/timestep contract.
10. Run targeted GPU inference only when asked/free: start with tiny smokes,
   then Klein 1024, then Lance T2V, then SenseNova/HiDream.

## Notes

- `/home/alex/modular/skills` was requested but is absent locally.
- Lance creator source remains `/home/alex/Lance`.
- Rust reference remains `/home/alex/EriDiffusion/inference-flame`.
- EDv2 reference remains `/home/alex/EriDiffusion/EriDiffusion-v2`.
- Anima source facts and blockers are captured in
  `serenitymojo/docs/ANIMA_HANDOFF_2026-05-28.md`.
- Lens source facts and blockers are captured in
  `serenitymojo/docs/LENS_SIDECAR_HANDOFF_2026-05-28.md`.
- Z-Image L2P source facts and blockers are captured in
  `serenitymojo/docs/ZIMAGE_L2P_STARTING_PASS_2026-05-28.md`.
