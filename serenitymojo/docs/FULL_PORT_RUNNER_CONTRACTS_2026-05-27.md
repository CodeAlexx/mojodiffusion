# Full Port Runner Contracts - 2026-05-27

Scope: runner contracts for the modular Mojo port across the active image/video
families. This document started as a contract note, but later passes added
runtime smoke and oracle artifact status where those gates now exist.

## Why This Exists

The runner layer should not know model internals. It should bind a
`GenerationRequest` plus a model manifest to the minimum model-specific hooks
needed to produce an image or video:

1. Prompt conditioning.
2. Latent or pixel-state shape and initialization.
3. Scheduler/timestep loop.
4. Transformer/denoiser call.
5. Decode or final image assembly.
6. Offload lifecycle.

The current code already exposes most of these as smoke-specific entry points.
The full port should lift those entry points into explicit family contracts
without changing the model math.

## Local Source Paths

Existing Mojo runner/runtime code:

- `/home/alex/mojodiffusion/serenitymojo/runtime/request.mojo`
- `/home/alex/mojodiffusion/serenitymojo/runtime/model_manifest.mojo`
- `/home/alex/mojodiffusion/serenitymojo/offload/block_loader.mojo`
- `/home/alex/mojodiffusion/serenitymojo/offload/plan.mojo`
- `/home/alex/mojodiffusion/serenitymojo/sampling/flux2_klein.mojo`
- `/home/alex/mojodiffusion/serenitymojo/sampling/hidream_o1_scheduler.mojo`

Existing Mojo model/pipeline code:

- `/home/alex/mojodiffusion/serenitymojo/models/dit/sdxl_unet.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/dit/sdxl_contract.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/sdxl_pipeline_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/sdxl_pipeline_full_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/dit/flux1_dit.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/dit/flux1_contract.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/flux1_pipeline_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/flux1_pipeline_cached_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/lance/lance_t2v.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/vae/wan22_decoder.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/lance_t2v_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/lance_t2v_image_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/lance_t2v_video_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/lance_wan22_vae_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/lance_wan22_vae_video_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/dit/klein_dit.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/vae/klein_decoder.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/klein9b_encode_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/dit/sensenova_u1.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/dit/hidream_o1.mojo`
- `/home/alex/mojodiffusion/serenitymojo/pipeline/hidream_o1_smoke.mojo`

Lance creator/reference code:

- `/home/alex/Lance/inference_lance.py`
- `/home/alex/Lance/inference_lance.sh`
- `/home/alex/Lance/config/examples/t2v_example.json`
- `/home/alex/Lance/modeling/lance/lance.py`
- `/home/alex/Lance/modeling/lance/qwen2_navit.py`
- `/home/alex/Lance/modeling/lance/modeling_utils.py`
- `/home/alex/Lance/data/datasets_custom/validation_dataset.py`
- `/home/alex/Lance/data/common.py`
- `/home/alex/Lance/modeling/vae/wan/model.py`
- `/home/alex/Lance/modeling/vae/wan/vae2_2.py`

Rust reference/creator code already mirrored by Mojo:

- `/home/alex/EriDiffusion/inference-flame/src/bin/lance_t2v.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/lance.rs`
- `/home/alex/EriDiffusion/inference-flame/src/vae/wan22_vae.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/klein.rs`
- `/home/alex/EriDiffusion/inference-flame/src/sampling/klein_sampling.rs`
- `/home/alex/EriDiffusion/inference-flame/src/bin/klein9b_encode.rs`
- `/home/alex/EriDiffusion/inference-flame/src/bin/klein9b_infer.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/sensenova_u1.rs`
- `/home/alex/EriDiffusion/inference-flame/src/bin/sensenova_u1_gen.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/model.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/pipeline.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/scheduler.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/mrope.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/decoder.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/bottleneck_patch_embed.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/timestep_embedder.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/final_layer.rs`
- `/home/alex/EriDiffusion/inference-flame/src/offload.rs`
- `/home/alex/EriDiffusion/inference-flame/src/offload_api.rs`

The requested `/home/alex/modular/skills` path is absent on this machine.

## Shared Request Contract

The current request surface is enough for first production dispatch:

- `model_id`
- `family`
- `prompt`
- `negative_prompt`
- `width`
- `height`
- `frames`
- `steps`
- `seed`
- `guidance_scale`
- `output_path`

Source: `/home/alex/mojodiffusion/serenitymojo/runtime/request.mojo`.

The manifest should remain the first shape resolver:

- `default_width`, `default_height`, `default_frames`
- `latent_channels`
- `latent_downsample_t`
- `latent_downsample_s`
- `latent_width()`
- `latent_height()`
- `latent_frames()`
- `production_entry`

Source: `/home/alex/mojodiffusion/serenitymojo/runtime/model_manifest.mojo`.

Contract rule: a runner may override width/height/frames from the request, but
must derive any latent grid through the model-specific shape adapter below, not
by assuming Stable Diffusion-style `[B,C,H/8,W/8]`.

## Minimum Runner Interfaces

### Prompt Encode

The prompt stage returns model-specific conditioning, not a universal embedding
type.

- Text-token contract: returns token ids and per-token position metadata.
  Lance and HiDream use this path.
- Text-embedding contract: returns resident or cached embeddings. Klein uses
  this path via separate Qwen3-8B process output.
- Prefix-cache contract: returns cond/uncond KV caches. SenseNova uses this path
  because its Qwen3 backbone is both text path and generation transformer.

Minimum fields the runner needs after prompt encode:

- `cond`: model-specific conditioning value.
- `uncond`: optional model-specific conditioning value.
- `do_cfg`: `guidance_scale > model_cfg_threshold`.
- `text_len` or equivalent prefix length when image/video positions depend on
  the prompt.
- `static_sequence_len`: required when the model's SDPA path is comptime-shaped.

### Latent Shape

The latent-plan hook must return both the denoiser input shape and the final
decode/finalization interpretation.

- For latent-space models, return the initial latent tensor shape and decode
  target.
- For pixel-space flow models, return the initial image/noise shape and patch
  shape. These models have no VAE.

Minimum fields:

- `width`, `height`, `frames`
- `latent_t`, `latent_h`, `latent_w`
- `tokens`
- `channels_or_patch_dim`
- `state_layout`: token, NCHW image, or packed patches.
- `decode_kind`: VAE image, VAE video, unpatchify image, or none.

### Scheduler Loop

The scheduler hook owns host-side scalar timesteps and one GPU update per step.
The runner must not bake one sign convention into all models.

Minimum operations:

- Build `steps + 1` schedule or model-specific per-step timesteps.
- For each step, expose model timestep input, velocity/prediction conversion,
  CFG combine, and state update.
- Keep state on GPU. Host scalar schedules are okay; activation readback is not.

### Transformer Call

The denoiser call must be model-specific but shaped uniformly:

- Input: current state, step scalar/tensor, cond/uncond context, position tables
  or caches, and offload handle.
- Output: model prediction in that model's native convention.

The runner should treat the output as opaque until the scheduler adapter turns
it into a state update. Examples: Lance returns velocity toward noise and
subtracts it; Klein returns direct velocity and adds `dt * pred`; HiDream flips
the post-CFG velocity before scheduler step.

### VAE Decode / Finalization

Decode is optional and model-specific:

- Lance T2V: Wan2.2 VAE video decode from latent tokens.
- Lance image smoke: Wan2.2 first-frame decode from latent tokens.
- Klein: Flux2/Klein VAE decode from packed NCHW latent.
- SenseNova: no VAE; unpatchify RGB patches and denorm.
- HiDream: no VAE; unpatchify RGB patches in signed range.

The runner should name this stage `finalize`, not `vae_decode`, so pixel-space
models fit without fake VAE stubs.

### Offload Hooks

The shared offload hook is now the plan-indexed `PlannedBlockLoader` surface on
top of the existing `BlockLoader` backend:

- `PlannedBlockLoader.open(path_or_dir, plan, config)`
- `prefetch(i)`
- `prefetch_next(i)`
- `await_block(i, ctx)`
- `PlannedBlockHandle.block`
- `PlannedBlockHandle.prefix`

Sources:

- `/home/alex/mojodiffusion/serenitymojo/offload/block_loader.mojo`
- `/home/alex/mojodiffusion/serenitymojo/offload/plan.mojo`
- `/home/alex/mojodiffusion/serenitymojo/offload/planned_loader.mojo`

Contract rule: runners call prefetch/await around each transformer block or each
paired CFG block visit. They do not hold per-layer blocks across steps unless a
future turbo/VMM loader explicitly changes the policy. The handle owns a
GPU-resident block; dropping it releases the block tensors.

## Model Contracts

### Lance T2V

Priority: first after Klein.

Current Mojo surface:

- `LanceT2VConfig.lance_3b_video()`
- `build_lance_t2v_input(tok, prompt, latent_t, latent_h, latent_w)`
- `LanceT2VOffloaded[S].load(model_dir, ctx)`
- `LanceT2VOffloaded[S].forward_velocity(input, x_t, timestep, max_layers, ctx)`
- `Wan22VaeImageDecoder[LH,LW].decode_tokens(x_t, ctx)`
- `Wan22VaeImageDecoder[LH,LW].decode_video_tokens(x_t, latent_t, ctx)`

Prompt contract:

- Current Mojo T2V smoke uses `text_template=false`: `[BOS, prompt ids, EOS,
  start_of_image, video_token * L, end_of_image]`.
- Full reference has a templated T2V system prompt path in
  `/home/alex/EriDiffusion/inference-flame/src/bin/lance_t2v.rs` and creator
  dataset code in `/home/alex/Lance/data/datasets_custom/validation_dataset.py`.
- Runner should support both `text_template=false` smoke parity and templated
  T2V production parity. Do not hardwire one into the denoiser.

Latent contract:

- True latent-space T2V.
- State is token layout `[L, 48]`, where `L = latent_t * latent_h * latent_w`.
- Manifest default has spatial downsample 16 and temporal downsample 4.
- Pixel-frame relation from the Rust reference: `T_pixel = (T_lat - 1) * 4 + 1`,
  so `latent_t = 1 + (frames - 1) // 4`.
- Current smoke examples:
  - `T=1,H=2,W=2,L=4` first-frame image artifact.
  - `T=3,H=1,W=1,L=3` tiny video artifact producing 9 frames.

Scheduler contract:

- Shifted flow schedule: `t' = shift * t / (1 + (shift - 1) * t)`.
- Current Mojo smoke uses shift 3.5, descending `t`, positive `dt = t0 - t1`,
  and update `x = x - dt * v`.
- Full CFG path should match Rust/Python: cond/uncond denoise with text CFG,
  then the same Lance sign convention.

Transformer contract:

- `forward_velocity` embeds full token stream, replaces video-token rows with
  `vae2llm(x_t) + time + latent_pos`, builds Qwen2.5-VL-style mRoPE and a T2V
  mask, streams `language_model.model.layers.{i}`, then projects gen rows
  through `llm2vae`.
- Current output is gen rows only, projected to patch latent dim 48.
- Production-size video still needs the block-sparse/flex attention path before
  dense full-resolution runs are practical.

Decode contract:

- First-frame/image path: `decode_tokens([LH*LW,48]) -> [1,3,16*LH,16*LW]`.
- Video path: `decode_video_tokens([T*LH*LW,48], latent_t) -> [1,3,T_pixel,H,W]`.
- Wan2.2 decode owns causal temporal cache semantics; do not fake video decode
  by looping independent first-frame decodes.

Offload contract:

- Shared tensors stay resident through `LanceWeights.load_shared`.
- Each decoder layer streams through `PlannedBlockLoader` using
  `build_lance_t2v_block_plan` and unloads when the handle drops.
- For CFG, prefer paired cond/uncond per loaded block once the full Lance CFG
  Mojo path exists.

### Klein9B

Status: current main path. The runner contract should preserve it while making
room for Lance next.

Current Mojo surface:

- `Qwen3Tokenizer` + `Qwen3Encoder.encode_klein` in
  `/home/alex/mojodiffusion/serenitymojo/pipeline/klein9b_encode_smoke.mojo`
- `Klein9BOffloaded.load(path, ctx)`
- `Klein9BOffloaded.forward_full_cfg[N_IMG,N_TXT,S](...)`
- `build_klein_rope_tables[N_IMG,N_TXT,32,128](ctx, dtype)`
- `build_flux2_sigma_schedule(num_steps, N_IMG)`
- `flux2_cfg(pred_pos, pred_neg, guidance_scale, ctx)`
- `KleinVaeDecoder[LH,LW].decode(packed, ctx)`

Prompt contract:

- Separate process by design. Encode Qwen3-8B positive and negative prompts,
  save `[1,512,12288]` BF16 embeddings, exit to release encoder VRAM.
- Denoise process loads cached `klein9b_caps_pos.bin` and
  `klein9b_caps_neg.bin`; it must not import or load `Qwen3Encoder`.

Latent contract:

- Image latent tokens: `[1, N_IMG, 128]`.
- `N_IMG = LH * LW`, with `LH = height // 16`, `LW = width // 16`.
- Initial draw is NCHW `[1,128,LH,LW]`, then packed to tokens in NHWC order.
- Before VAE decode, tokens are reshaped back to packed NCHW `[1,128,LH,LW]`.

Scheduler contract:

- FLUX.2/Klein schedule from `build_flux2_sigma_schedule(num_steps, N_IMG)`.
- Per step:
  - `timestep = sigma * 1000` as `[1]` F32.
  - Cast latent to BF16 only for the DiT call; keep loop state F32.
  - CFG is `pred_neg + scale * (pred_pos - pred_neg)`.
  - Euler update is `x = x + (sigma_next - sigma) * pred`.
- No post-CFG sign flip.

Transformer contract:

- `forward_full_cfg` streams each double block once, runs pos and neg branches,
  unloads, then does the same for single blocks.
- Static `S == N_IMG + N_TXT`.
- Current full path uses 8 double blocks and 24 single blocks from
  `double_blocks.{i}` and `single_blocks.{i}`.

Decode contract:

- `KleinVaeDecoder[LH,LW].decode([1,128,LH,LW]) -> [1,3,16*LH,16*LW]`.

Offload contract:

- Shared DiT tensors resident in `Klein9BDiT.load_shared`.
- Block prefixes:
  - `double_blocks.{i}`
  - `single_blocks.{i}`
- CFG block pairing is already implemented; runner should call that path, not
  two independent full forwards.

### FLUX.1-dev

Status: P1 bring-up path. Current Mojo has a DiT/pipeline smoke and a
metadata-only checkpoint contract gate; no quality-parity run yet.

Current Mojo surface:

- `Flux1Offloaded.load(path, Flux1Config.dev(), ctx)`
- `Flux1Offloaded.forward[N_IMG,N_TXT,S](img, txt, timestep, guidance, vector, cos, sin, ctx)`
- `build_flux1_rope_tables[N_IMG,N_TXT,24,128](img_h2, img_w2, ctx, dtype)`
- `load_flux1_ldm_decoder[LATENT_H,LATENT_W](vae_path, ctx)`
- `validate_flux1_pipeline_contract(manifest)` in
  `models/dit/flux1_contract.mojo`

Prompt contract:

- CLIP-L pooled vector `[1,768]` plus T5-XXL hidden states `[1,512,4096]`.
- Dev is guidance-distilled: no negative prompt and no classifier-free
  guidance. The guidance scalar is fed as a model input.
- Current smoke has placeholder token ids; production wrapper must use the
  CLIP and T5 tokenizer JSON files validated by the contract smoke.

Latent contract:

- Initial latent is NCHW `[1,16,2*ceil(H/16),2*ceil(W/16)]`.
- Pack to image tokens `[1,ceil(H/16)*ceil(W/16),64]`.
- For the 1024 profile: latent NCHW `[1,16,128,128]`, image tokens `4096`,
  text tokens `512`, total sequence `4608`.

Scheduler contract:

- FLUX.1 schedule: `linspace(1,0,steps+1)` then BFL time shift with linear
  `mu` from `(256,0.5)` to `(4096,1.15)`.
- Per step uses one forward: `x = x + (t_next - t_current) * pred`.
- The Mojo pipeline pre-scales timestep and guidance by `1000` before
  `t_embedder`, matching Rust's internal FLUX.1 time factor.

Transformer contract:

- Static `S == N_IMG + N_TXT`.
- 19 double-stream blocks and 38 single-stream blocks, with per-block
  modulation and biases everywhere.
- Remaining parity risk: RoPE tables are still BF16 in the smoke because the
  shared RoPE op currently requires matching q/k and PE dtypes; Rust keeps
  FLUX.1 PE in F32.

Decode contract:

- FLUX uses the LDM/BFL-format `ae.safetensors`, not the diffusers VAE layout.
- Decode via `load_flux1_ldm_decoder`; scale/shift are `(0.3611, 0.1159)`.

Offload contract:

- Shared DiT tensors stay resident; block prefixes are `double_blocks.{i}` and
  `single_blocks.{i}` via the current synchronous `BlockLoader`.
- The future production wrapper should move this to planned/turbo offload after
  the generic slot backend owns real pinned/GPU storage.

### SD3.5 Large/Medium

Status: P1 scheduler + MMDiT pre/post-block + embedded VAE runtime smoke. Current Mojo has a
registered Large manifest, registered local "small" Medium manifest, header-only
`pipeline/sd3_pipeline_contract_smoke.mojo` and
`pipeline/sd3_medium_pipeline_contract_smoke.mojo`, tensor scheduler smoke, and
Large/Medium MMDiT pre/post-block and VAE decode smokes. Joint MMDiT blocks and
triple-encoder assembly are not ported yet.

Current contract surface:

- `sd3_5_large_default_manifest()` in `runtime/model_manifest.mojo`
- `sd3_5_medium_default_manifest()` in `runtime/model_manifest.mojo`
- `default_manifest_by_id("sd3_5_large")`
- `default_manifest_by_id("sd3_5_medium")`
- `validate_manifest_paths(manifest)` returns 10 checked paths for the Large
  or Medium checkpoint and text encoder sidecars.
- `pipeline/sd3_pipeline_contract_smoke.mojo` validates the local checkpoint
  header without creating a `DeviceContext`.
- `pipeline/sd3_medium_pipeline_contract_smoke.mojo` validates the local
  Medium checkpoint header, including its first `13` dual-attention image
  blocks.
- `pipeline/sd3_mmdit_preblock_smoke.mojo` loads real Large/Medium resident
  MMDiT tensors and runs latent patch embedding plus centered learned
  `pos_embed` crop, `sigma*1000` timestep MLP, pooled text MLP, and bounded
  `context_embedder` projection, then final AdaLN/projection and unpatchify
  back to `[1,16,128,128]`.
- `pipeline/sd3_vae_smoke.mojo` loads `first_stage_model.decoder.*` from the
  same checkpoint through `load_sd3_embedded_ldm_decoder[128,128]` and writes
  `output/sd3_vae_noise_1024.png`.
- `pipeline/sd3_medium_vae_smoke.mojo` uses the same embedded VAE path and
  writes `output/sd3_medium_vae_noise_1024.png`.

Prompt contract from Rust:

- CLIP-L hidden `[1,77,768]` and CLIP-G hidden `[1,77,1280]` are zero-padded to
  context width `4096`.
- T5 hidden is `[1,256,4096]`.
- Concatenated context length is `77 + 77 + 256 = 410`.
- Pooled CLIP projections concatenate to `[1,2048]`.

Latent/model contract:

- 1024 profile uses latent `[1,16,128,128]`, patch size `2`, image tokens
  `4096`, text tokens `410`, total sequence `4506`.
- MMDiT Large anchors: hidden `2432`, depth `38`, heads `38`, head dim `64`.
- MMDiT Medium anchors: hidden `1536`, depth `24`, heads `24`, head dim `64`,
  with `13` initial image blocks carrying `attn2`.
- The checkpoint embeds the SD3 VAE under `first_stage_model.*`; decode scale
  and shift are `1.5305` and `0.0609`.
- The local checkpoint does not include `post_quant_conv`, so the SD3 loader
  uses the generic LDM no-PQC path with prefix `first_stage_model.decoder`.

Scheduler contract:

- 28 shifted-flow steps, CFG `4.5`, schedule shift `3.0`.
- Model timestep is `t * 1000.0`.

### SenseNova-U1

Status: later after Lance; current Mojo has an end-to-end small T2I smoke.

Current Mojo surface:

- `SenseNovaU1[L_TOKENS,TEXT_LEN].load(weights_dir, ctx)`
- `forward_und(token_ids, ctx) -> KvCache`
- `extract_feature_gen(pixel_values, grid_h, grid_w, ctx)`
- `time_or_scale_embed(t, "timestep" | "noise", ctx)`
- `forward_gen(image_embeds, text_len, token_h, token_w, cache, ctx)`
- `fm_head_forward(hidden, ctx)`
- Local patchify/unpatchify in
  `/home/alex/mojodiffusion/serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo`

Prompt contract:

- No separate text encoder. The base Qwen3 path in `SenseNovaU1` is the prompt
  encoder.
- Run `forward_und` once for cond and once for uncond; the result is a per-layer
  KV cache.
- Gen path reads the cache and does not update it.
- Current smoke uses split-file tokenizer files under
  `/home/alex/.serenity/models/sensenova_u1/{vocab.json,merges.txt,added_tokens.json}`.

Latent contract:

- Pixel-space flow model; no VAE and no latent-space channels.
- Current state is image tensor `[1,3,H,W]`.
- Per step:
  - `z = patchify(img, patch * merge, channel_first=false)` -> `[1,L,3072]`.
  - `pixel_values = patchify(img, patch, channel_first=true)` -> `[1,grid_h*grid_w,768]`.
  - `patch=16`, `merge=2`, so `L = (H/32) * (W/32)`.
- `compute_noise_scale(grid_h, grid_w)` scales initial image noise.

Scheduler contract:

- Build uniform `[0,1]` schedule with `num_steps + 1` entries.
- Apply SenseNova time shift. Current smoke uses standard shifted sigma with
  `TIMESTEP_SHIFT = 3.0`.
- Per step:
  - `denom = max(1 - t, t_eps)`.
  - `v = (x_pred - z) / denom`.
  - CFG is `v_uncond + scale * (v_cond - v_uncond)`.
  - `z_next = z + (t_next - t) * v`.
  - Unpatchify `z_next` back to `[1,3,H,W]` before the next step.

Transformer contract:

- Text prefix uses BASE weights and causal attention.
- Image generation uses `_mot_gen` weights, full 3D RoPE, cached prefix K/V, and
  full attention over prefix + image tokens.
- MoT means two dense transformer paths by modality, not top-k MoE routing.
- `L_TOKENS` and `TEXT_LEN` are comptime parameters. Production runner needs a
  dispatch table for supported `(L_TOKENS, TEXT_LEN)` pairs.

Decode/finalize contract:

- No VAE.
- Final image is `img * 0.5 + 0.5`, clamp/save in unit range.

Offload contract:

- Shared resident tensors include embeddings, final norms, vision gen embedder,
  timestep/noise embedders, and `fm_head`.
- Stream `language_model.model.layers.{i}` for both prefix and gen paths through
  `PlannedBlockLoader` and `build_sensenova_u1_block_plan`.
- For cond/uncond CFG, cache prefix once per branch, then sequence gen forwards
  through the same model/offloader.

### HiDream-O1

Status: later after Lance/SenseNova; current Mojo has a smoke skeleton with
known compile/runtime structure notes and a planned-loader offloaded path.

Current Mojo surface:

- `HiDreamO1Config.dev_8b()`
- `build_t2i_input(tok, cfg, prompt, h_patches, w_patches)`
- `HiDreamO1Offloaded[S].load(model_dir, cfg, ctx)`
- `HiDreamO1Offloaded[S].forward(input_ids, noise_patches, t_pos, h_pos, w_pos, ar_len, timestep, ctx)`
- `HiDreamO1Scheduler.dev_28step()`
- `HiDreamO1Scheduler.full_n_step(n, shift)`
- `HiDreamO1Scheduler.step(...)`
- `patchify` / `unpatchify` from `/home/alex/mojodiffusion/serenitymojo/ops/layout.mojo`

Prompt contract:

- No separate text encoder. Qwen3-VL embed tokens live inside the HiDream model.
- Prompt template appends BOI + one TMS token.
- `build_t2i_input` returns text ids, mRoPE positions, text length, and `ar_len`.
- Uncond prompt is literal single space when CFG is enabled.

Latent contract:

- Pixel-space image model; no VAE.
- Initial state is RGB noise `[1,3,H,W]` scaled by scheduler/model mode
  (`7.5` Dev flash, `8.0` Full default in Rust reference).
- Patchify with `patch_size=32`: `[1,L,3072]`, `L = (H/32) * (W/32)`.
- DiT full sequence length is `S = text_len + L` and is a comptime parameter in
  current Mojo.

Scheduler contract:

- Pipeline-side timestep input to model is `t_pixeldit = 1 - step_t / 1000`.
- Velocity is `v = (x_pred - z) / max(step_t / 1000, 0.001)`.
- CFG is `v_uncond + scale * (v_cond - v_uncond)`.
- Before scheduler step, flip sign: `model_output = -v_guided`.
- Dev/Flash scheduler may inject per-step noise; Full/default is deterministic
  Euler with shift 3.0.

Transformer contract:

- Forward pass embeds text, scatters timestep embedding into TMS row, patch-embeds
  image patches, concatenates, builds interleaved mRoPE and prefix-causal mask,
  streams 36 language-model layers, final norm, final linear to `[1,S,3072]`.
- Caller gathers image rows from the tail or by mask, then converts to velocity.
- Current smoke works around a Mojo 1.0.0b1 monomorphization issue by keeping
  `S` module-level and putting denoise in its own top-level `def`.

Decode/finalize contract:

- No VAE.
- Unpatchify `[1,L,3072]` to `[1,3,H,W]` in signed `[-1,1]` range.

Offload contract:

- Resident keys are loaded as BF16 from F32-on-disk weights.
- Stream `model.language_model.layers.{i}` through `PlannedBlockLoader` and
  `build_hidream_o1_block_plan` with BF16 load policy.
- Use paired cond/uncond only if a future path can do so without violating the
  current compile workaround.

## Dispatch Shape

The production runner should dispatch by `model_id` to a family adapter:

- `klein9b`: cached text embeddings -> latent tokens -> Flux2 scheduler ->
  `Klein9BOffloaded.forward_full_cfg` -> Klein VAE.
- `lance_t2v`: tokenizer/template -> latent token plan -> Lance shifted-flow
  scheduler -> `LanceT2VOffloaded.forward_velocity` -> Wan2.2 VAE video decode.
  The current runner-visible target is the dense 256x256/9-frame artifact path,
  guarded by `pipeline/lance_t2v_pipeline.mojo`.
- `sensenova_u1`: tokenizer -> cond/uncond `forward_und` caches -> pixel
  patch loop -> `forward_gen` + `fm_head` -> unpatchify.
- `hidream_o1`: tokenizer/template -> pixel patch loop -> `HiDreamO1Offloaded`
  forward -> HiDream scheduler -> unpatchify.
- `sdxl`: cached CLIP embeddings -> SDXL Euler -> `SDXLUNet` -> SDXL LDM VAE;
  one-step and full 30-step cached-embedding smokes exist; Rust/diffusers
  parity and raw-prompt CLIP assembly remain open.
- `flux1_dev`: cached Rust inputs -> FLUX.1 schedule -> `Flux1Offloaded` ->
  FLUX LDM VAE; cached-input runtime smoke exists, while tokenizer-backed
  CLIP/T5 prompt assembly remains open.
- `sd3_5_large` / `sd3_5_medium`: manifest/header contracts, tensor FlowMatch
  CFG/update gates, and embedded VAE decode smokes exist; MMDiT and triple
  encoder remain open.
- `anima`: manifest/header contract plus cached-conditioning tensor gate,
  Rust latent oracle, and VAE latent PNG runtime smoke; adapter, MiniTrainDIT,
  and cached-context denoise wrapper are not ported yet.
- `microsoft_lens`: manifest/header contract plus scalar/BF16 tensor FlowMatch
  gates, text-smoke header gates, and a sampled real-weight block0 image-QKV
  gate; GPT-OSS MXFP4 text encoder, full Lens DiT forward/RoPE parity, and
  FLUX.2 VAE path are not ported yet.
- `zimage_l2p`: manifest/header/schedule contract plus cached-conditioning
  sidecar, Rust 512/1024 pixel oracles, and a 1024 BF16 pixel patchify16
  roundtrip runtime gate; this is the VAE-less pixel-space Z-Image variant,
  with a bounded real-weight local-decoder subgate. Full MicroDiffusionModel,
  DiT body, and full-resolution local-decoder path remain unported.
- `ernie_image`: manifest/header contract plus scheduler/tensor CFG and Euler
  gate over Mistral3B -> ERNIE DiT -> Klein VAE assets, plus resident
  pre-block DiT projection/AdaLN slices; the full Mistral3B text encoder,
  ERNIE block stack, 3-axis RoPE builder, and denoise/decode wrapper are not
  ported yet.

Do not force these through one universal `TextEncoder`, `Latent`, or `VAE`
trait. The stable abstraction is the runner phase boundary, not the tensor type
behind it.

## Open Contract Gaps

- Lance dense padded CFG smokes and the CFG/KV-cache metadata gate are in place.
  Remaining production work is variable-length cached CFG/KV execution and
  sparse/flex attention for larger sequence lengths.
- Lance production-size video needs sparse/block attention; current Mojo smoke
  intentionally uses tiny dense grids.
- SenseNova now has a compile-only static entrypoint contract for
  `(L_TOKENS, TEXT_LEN)` profiles. The remaining production work is prompt
  padding/dispatch and the real `sensenova_u1_pipeline.mojo` body.
- HiDream now has a compile-only static entrypoint contract for static `S`
  profiles. The remaining production work is common-S CFG dispatch and the real
  `hidream_o1_pipeline.mojo` body.
- `/home/alex/modular/skills` is absent, so no Modular skill references were
  inspected from that path.
