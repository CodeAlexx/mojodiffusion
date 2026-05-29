# LTX2 Architecture Inventory (Planner B, 2026-05-28)

Read-only inventory of the **Lightricks LTX-Video 2** / **LTX-2.3** model family. Pairs with `LTX2_RUST_STATE_2026-05-28.md` (Planner A, Rust port state). This document covers the *model itself*: components, shapes, weight keys, dtypes, and Mojo port leverage points.

## Inputs surveyed

| Path | What it gave |
|---|---|
| `~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/f90a633.../` | `model_index.json`, all sub-configs (transformer dir **absent**), full video VAE safetensors |
| `~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5a9c1c6.../` | LTX-2.3 single-file checkpoints (22B dev / distilled / upscalers) — primary weights source |
| `~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/76730e6.../` | LTX-2.3 v1.1 (distilled, spatial-x2) |
| `/home/alex/LTX-2/packages/ltx-core/src/ltx_core/model/` | Upstream Python source: `transformer/`, `video_vae/`, `audio_vae/`, `upsampler/` |
| `/home/alex/EriDiffusion/inference-flame/cached_ltx2_*.safetensors` | Conditioning sidecar contract |
| `/home/alex/.serenity/models/ltx2/` | Only contains `ic-loras/` (empty); not a primary source |

Note: the official HF "LTX-2" snapshot has `model_index.json` + sub-component configs (`vae/`, `audio_vae/`, `text_encoder/`, `connectors/`, `scheduler/`) but `transformer/` is **not** present in that repo — the actual DiT weights live in the **LTX-2.3 22B single-file checkpoints**, which bundle DiT + video VAE + audio VAE + vocoder + text-projection into one safetensors file. Configs come from LTX-2; weights come from LTX-2.3.

---

## 1. Pipeline components (top level)

From `…/LTX-2/snapshots/f90a633…/model_index.json`:

| Component | Class | Provider | Notes |
|---|---|---|---|
| `transformer` | `LTX2VideoTransformer3DModel` | diffusers | DiT (audio+video joint, see §2a) |
| `vae` | `AutoencoderKLLTX2Video` | diffusers | 3D causal video VAE, 32×32×8 compression |
| `audio_vae` | `AutoencoderKLLTX2Audio` | diffusers | 2D mel-spectrogram VAE |
| `vocoder` | `LTX2Vocoder` | ltx2 | BigVGAN-v2-style mel→waveform + BWE branch |
| `text_encoder` | `Gemma3ForConditionalGeneration` | transformers | Gemma3 ~3.84K hidden, 48 layers (+SigLIP vision tower) |
| `tokenizer` | `GemmaTokenizerFast` | transformers | vocab 262208 |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | diffusers | dynamic shift, exponential time shift |
| `connectors` | `LTX2TextConnectors` | ltx2 | Per-modality projection of Gemma hidden → DiT context (audio+video) |

The single-file LTX-2.3 22B checkpoint contains the following top-level prefixes (from `safetensors.safe_open` on `ltx-2.3-22b-dev.safetensors`, 5947 tensors total):

| Prefix | Tensor count | Param count | Section |
|---|---|---|---|
| `model.diffusion_model.*` | 4444 | 21.005 B | DiT (audio+video joint) |
| `vae.*` | 170 | 0.726 B | Video VAE (3D causal) |
| `vocoder.*` | 1227 | 0.129 B | Vocoder (BigVGAN-v2 + BWE + mel-stft buffers) |
| `audio_vae.*` | 102 | 0.053 B | Audio VAE (2D, mel input) |
| `text_embedding_projection.*` | 4 | 1.156 B | Two big Linear: Gemma-MM → DiT context (audio + video) |
| **TOTAL** | **5947** | **23.070 B** | |

Storage dtype throughout: **BF16** (except a small number of F32 scale_shift tables in the DiT).

Other LTX-2.3 files in the same snapshot:
| File | Size | Notes |
|---|---|---|
| `ltx-2.3-22b-distilled.safetensors` | 46.1 GB | Distilled all-in-one DiT+VAE+vocoder, same key layout |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.6 GB | LoRA delta over the dev checkpoint, rank-384 |
| `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` | 1.09 GB | 73 tensors, 2D latent upscaler |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | 996 MB | 72 tensors, 3D latent upscaler |
| `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` | 262 MB | 72 tensors, 3D latent upscaler |

---

## 2. Per-component spec

### 2a. DiT (`model.diffusion_model.*`) — joint audio+video Audio-Video transformer

From `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py:18-71` (the `LTXModelConfigurator.from_config` defaults):

| Setting | Video | Audio |
|---|---|---|
| `num_attention_heads` | 32 | 32 |
| `attention_head_dim` | 128 | 64 |
| `inner_dim` (hidden) | **4096** | **2048** |
| `in_channels` / `out_channels` (DiT I/O) | 128 / 128 | 128 / 128 |
| `num_layers` | **48** | 48 |
| `cross_attention_dim` (context) | 4096 | 2048 |
| `norm_eps` | 1e-6 | 1e-6 |
| `attention_type` | "default" (resolves to SDPA / xformers / FA3) | same |
| `positional_embedding_theta` | 10000.0 | 10000.0 |
| `positional_embedding_max_pos` | `[20, 2048, 2048]` (T, H, W) | `[20]` (T only) |
| `timestep_scale_multiplier` | 1000 | 1000 |
| `rope_type` | `SPLIT` (default; `INTERLEAVED` legacy) | same |
| `apply_gated_attention` | True (checkpoint has `to_gate_logits.*`) | True |
| `qk_norm` | `rms_norm` | `rms_norm` |
| `standardization_norm` | `rms_norm` | `rms_norm` |
| `activation_fn` | `gelu-approximate` | same |
| `cross_attention_norm` | True | True |
| `use_middle_indices_grid` | True | True |
| `frequencies_precision` | `float64` (config "rope_double_precision: true" in `connectors/config.json`) | same |

**Total DiT params: 21.005 B** (matches the "22B" naming after counting VAE+vocoder).

Per-block layout (`model.diffusion_model.transformer_blocks.0.*`, 86 keys, 48 identical blocks):

| Submodule | Shape (W) | Notes |
|---|---|---|
| `attn1.to_{q,k,v,out.0}` | `[4096,4096]` BF16 + bias | Video self-attn |
| `attn1.{q_norm,k_norm}.weight` | `[4096]` BF16 | RMSNorm on Q,K |
| `attn1.to_gate_logits.{weight,bias}` | `[32,4096]` / `[32]` BF16 | Per-head sigmoid gate (32 heads); `2*sigmoid(...)` |
| `attn2.*` (video cross-attn → text) | `[4096,4096]` BF16 | Cross-attn from text context (4096-dim) |
| `audio_attn1.*` (audio self-attn) | `[2048,2048]` BF16 | Audio self-attn (32 heads × 64 dim) |
| `audio_attn2.*` (audio cross-attn → text) | `[2048,2048]` BF16 | Cross-attn from audio text context (2048-dim) |
| `audio_to_video_attn.{to_q,out.0}` | `[4096,*]` BF16 | Q in 4096 (video), KV in 2048 (audio) — cross-modal attention from video → audio |
| `audio_to_video_attn.{to_k,to_v}` | `[2048,2048]` BF16 | Audio key/value |
| `video_to_audio_attn.{to_q,out.0}` | `[2048,*]` BF16 | Q in 2048 (audio), KV from 4096 (video) — audio → video |
| `video_to_audio_attn.{to_k,to_v}` | `[2048,4096]` BF16 | Video key/value, projected to audio inner dim |
| `ff.net.0.proj` / `ff.net.2` | `[16384,4096]` / `[4096,16384]` BF16 | Video FFN, 4× expansion, GELU-approx |
| `audio_ff.net.0.proj` / `audio_ff.net.2` | `[8192,2048]` / `[2048,8192]` BF16 | Audio FFN, 4× expansion |
| `scale_shift_table` | `[9,4096]` F32 | AdaLN-single shift/scale table per block (video) |
| `audio_scale_shift_table` | `[9,2048]` F32 | Audio version |
| `prompt_scale_shift_table` | `[2,4096]` F32 | Cross-attn AdaLN for text context (video) |
| `audio_prompt_scale_shift_table` | `[2,2048]` F32 | Same, audio |
| `scale_shift_table_a2v_ca_audio` | `[5,2048]` F32 | Cross-modal AV cross-attn AdaLN (audio side) |
| `scale_shift_table_a2v_ca_video` | `[5,4096]` F32 | Same, video side |

Notable global DiT modules (under `model.diffusion_model.*`):

| Key | Shape | Notes |
|---|---|---|
| `patchify_proj.{weight,bias}` | `[4096,128]`/`[4096]` | Video latent in_ch=128 → DiT 4096 |
| `audio_patchify_proj.{weight,bias}` | `[2048,128]`/`[2048]` | Audio latent in_ch=128 → DiT 2048 |
| `proj_out.{weight,bias}` | `[128,4096]`/`[128]` | DiT 4096 → video out_ch=128 |
| `audio_proj_out.{weight,bias}` | `[128,2048]`/`[128]` | DiT 2048 → audio out_ch=128 |
| `adaln_single.emb.timestep_embedder.linear_{1,2}` | `[4096,256]`/`[4096,4096]` | Sinusoidal-256 → 4096 |
| `audio_adaln_single.emb.timestep_embedder.linear_{1,2}` | `[2048,256]`/`[2048,2048]` | Audio variant |
| `adaln_single.linear` | `[36864,4096]` | 4096 → 9×4096 (the 9 AdaLN coefficients) |
| `audio_adaln_single.linear` | `[18432,2048]` | 2048 → 9×2048 |
| `prompt_adaln_single.linear` | `[8192,4096]` | 4096 → 2×4096 (cross-attn AdaLN) |
| `audio_prompt_adaln_single.linear` | `[4096,2048]` | 2048 → 2×2048 |
| `av_ca_{video,audio}_scale_shift_adaln_single.linear` | `[20480,4096]`/`[10240,2048]` | 4×inner_dim each |
| `av_ca_{a2v,v2a}_gate_adaln_single.linear` | `[4096,4096]`/`[2048,2048]` | Gate AdaLN |
| `scale_shift_table` (top-level) | `[2,4096]` F32 | Final video output shift/scale |
| `audio_scale_shift_table` (top-level) | `[2,2048]` F32 | Final audio output shift/scale |
| `video_embeddings_connector.learnable_registers` | `[128,4096]` | Q-Former-style learned tokens |
| `audio_embeddings_connector.learnable_registers` | `[128,2048]` | Same for audio |
| `{video,audio}_embeddings_connector.transformer_1d_blocks.{0..7}.*` | 8 blocks each | Self-attn + FFN, 8 layers per modality |

Each connector block has just `attn1.*` (no cross-attn) + `ff.*`, with `to_gate_logits` (gated attention), `q_norm`/`k_norm` RMSNorm, and 4× FFN expansion.

**Key cross-modal pattern**: each of the 48 main DiT blocks has *all six* attention paths active:
- video self-attn (`attn1`)
- video↔text cross-attn (`attn2`)
- audio self-attn (`audio_attn1`)
- audio↔text cross-attn (`audio_attn2`)
- video→audio cross-modal (`video_to_audio_attn`)
- audio→video cross-modal (`audio_to_video_attn`)

This is fundamentally *not* a video-only or audio-only model at the architectural level: the joint AV path is woven into every block.

#### RoPE details

From `transformer/rope.py`:
- **SPLIT** RoPE (default): freqs are computed per dim along 3D position grid `(T,H,W)` with `max_pos=[20,2048,2048]`, then `cos`/`sin` are computed in fp64 and cast to BF16 (set by `rope_double_precision: true` in `connectors/config.json`).
- For the audio path the position grid is 1D with `max_pos=[20]`.
- Rotation form: `apply_split_rotary_emb` splits each head dim into two halves and rotates the pair (`x[..0,:], x[..1,:]`) by `(cos,sin)` — equivalent to a complex-multiply but on contiguous chunks (different from interleaved-RoPE).
- "Use middle indices grid" → positions are averaged from `(start,end)` bounds (frame patch granularity).

### 2b. Video VAE (`vae.*`) — 3D causal, 32×32×8 compression

From `…/LTX-2/audio_vae/config.json` and `video_vae/model_configurator.py` defaults + comment in `video_vae.py:155-165`:

| Setting | Value |
|---|---|
| `in_channels` (pixels) | 3 (RGB) |
| `out_channels` (decoder) | 3 |
| `latent_channels` | 128 |
| `patch_size` (pre-patchify, pixel shuffle) | 4 |
| Convolution dims | 3 (Conv3d throughout) |
| Norm layer | `pixel_norm` (PixelNorm; not GroupNorm) |
| `latent_log_var` | `uniform` (latent emits `latent_channels+1` channels → 129 → drop logvar at sample time) |
| Encoder block recipe | `1× compress_space_res, 1× compress_time_res, 2× compress_all_res` |
| Total compression | **H/32, W/32, F/8** (input requires `F = 1 + 8k`) |
| Stored dtype | BF16 |
| Total weights | 726 M params, 170 tensors |
| Encoder down_blocks | indices 0..8 (9 down stages including res blocks between compressors) |
| Encoder conv_in | `[128, 48, 3,3,3]` (in = 3 × 4² = 48 after pixel-shuffle) |
| Encoder conv_out | `[129, 1024, 3,3,3]` (129 = 128 + 1 logvar channel) |
| Decoder conv_in | `[1024, 128, 3,3,3]` |
| Decoder conv_out | `[48, 128, 3,3,3]` (48 = 3 × 4² pre pixel-unshuffle) |
| Per-channel statistics | `per_channel_statistics.{mean-of-means,std-of-means}: [128]` BF16 (latent normalization) |
| Up/down depth at widest | `[1024, 1024, 3,3,3]` (channels) |
| Padding | `causal=True` for temporal axis; encoder uses `zeros` spatial pad, decoder uses `reflect` |

Example shape pipeline from `video_vae.py` docstring: `(B,3,33,512,512) → (B,128,5,16,16)`.

### 2c. Audio VAE (`audio_vae.*`) — 2D mel VAE

From `…/LTX-2/audio_vae/config.json`:

| Setting | Value |
|---|---|
| `in_channels` (waveform stereo) | 2 |
| `output_channels` | 2 |
| `latent_channels` | 8 |
| `base_channels` | 128 |
| `ch_mult` | `[1, 2, 4]` (3 stages) |
| `num_res_blocks` per stage | 2 |
| `mel_bins` | 64 |
| `mel_hop_length` | 160 |
| `sample_rate` | 16000 |
| `resolution` | 256 |
| `causality_axis` | `"height"` (causal along mel-time) |
| `is_causal` | true |
| `norm_type` | `pixel` |
| `double_z` | true |
| `mid_block_add_attention` | false |
| Storage dtype | BF16, 53 M params, 102 tensors |

Encoder/decoder are 2D Conv2d on mel-spectrogram (not waveform-direct). Encoder: `[128, 2, 3,3]` (mel in), decoder out `[2, 128, 3,3]` (mel out, then vocoder maps mel→waveform).

### 2d. Vocoder (`vocoder.*`) — BigVGAN-v2 + BWE

From `model/audio_vae/vocoder.py`:

| Module | Notes |
|---|---|
| `vocoder.vocoder.*` (667 tensors) | Main BigVGAN-v2 generator |
| `vocoder.bwe_generator.*` (557 tensors) | Bandwidth-extension generator (16k → 44.1k/48k) |
| `vocoder.mel_stft.*` (3 tensors) | mel basis `[64,257]`, STFT forward/inverse basis `[514,1,512]` (precomputed) |

Vocoder structure (from source):
- `conv_pre`: `[1536, 128, 7]` (mel ch=128 → 1536) — note: takes the **audio_vae latent** as mel-equivalent input, NOT mel bins directly; channels=128 matches audio_vae latent_channels=8 × something? Actually `conv_pre.weight = [1536,128,7]` (the 128 is audio DiT out_channels passed through audio_vae decoder); check: audio_vae.decoder.conv_out outputs 2 ch (stereo mel), not 128. **Open question** — see §8.
- Upsample rates: `[6, 5, 2, 2, 2]` → 240× upsampling (Hop 200ms → ~48 kHz?). Default kernel sizes form anti-aliased BigVGAN-v2 blocks (kaiser-sinc low/high-pass filters).
- ResBlocks use snake-1d activation (`alpha`, `beta` per-channel learnable).
- `act_post.act.{alpha,beta}: [24]` — final activation, 24 channels = stereo × something; `conv_post.weight: [2, 24, 7]` confirms stereo 2-ch waveform output.
- BWE generator: same structure at smaller scale (`conv_pre.weight: [512, 128, 7]`, `conv_post.weight: [2, 16, 7]`) — applied after main vocoder to upsample bandwidth.
- 129 M params total.

### 2e. Text encoder (`text_encoder/`) — Gemma3 + SigLIP

From `…/LTX-2/text_encoder/config.json` (`Gemma3ForConditionalGeneration`):

| Setting | Value |
|---|---|
| `model_type` | `gemma3` |
| Text `hidden_size` | **3840** |
| Text `num_hidden_layers` | 48 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 (GQA, kv=8 vs q=16) |
| `head_dim` | 256 |
| `intermediate_size` | 15360 |
| `vocab_size` | 262208 |
| Layer types | Hybrid `sliding_attention` + `full_attention` every 6 layers (8 full-attn layers total) |
| `sliding_window` | 1024 |
| `max_position_embeddings` | 131072 |
| `rope_theta` | 1e6 (global) / 10000 (local) |
| `rope_scaling` | `linear`, factor 8.0 |
| `attention_bias` | false |
| `final_logit_softcapping` | null (off) |
| `query_pre_attn_scalar` | 256 |
| `rms_norm_eps` | 1e-6 |
| `use_bidirectional_attention` | **false** (causal) |
| `hidden_activation` | `gelu_pytorch_tanh` |
| Vision tower (SigLIP) | 1152-dim, 27 layers, 16 heads, patch 14, image 896 |
| Storage dtype | F32 (declared in config; checkpoint not in HF cache locally) |

The Gemma3 text encoder is **NOT** present as weights in any local file — it must be downloaded separately. The LTX-2.3 22B checkpoint instead provides a **post-encoder text projection** (`text_embedding_projection.*`) which projects the multi-modal Gemma hidden into the DiT context dim:

| Key | Shape |
|---|---|
| `text_embedding_projection.video_aggregate_embed.weight` | `[4096, 188160]` BF16 |
| `text_embedding_projection.video_aggregate_embed.bias` | `[4096]` BF16 |
| `text_embedding_projection.audio_aggregate_embed.weight` | `[2048, 188160]` BF16 |
| `text_embedding_projection.audio_aggregate_embed.bias` | `[2048]` BF16 |

Note `188160 = 49 × 3840`. The `49` matches `text_proj_in_factor: 49` in `connectors/config.json` — i.e. the projection concatenates the last 49 layers (or 49 patches) of the Gemma3 multi-modal hidden, each of width 3840, into a single `[B,1,188160]` token-aggregated representation, then linearly projects to 4096 (video context) or 2048 (audio context). This is *not* a per-token cross-attn context but a **per-batch aggregate** for the connector input.

### 2f. Connectors (`LTX2TextConnectors`)

From `…/LTX-2/connectors/config.json`:

| Setting | Value |
|---|---|
| `caption_channels` | 3840 (Gemma hidden) |
| `text_proj_in_factor` | 49 |
| `video_connector_attention_head_dim` | 128 |
| `video_connector_num_attention_heads` | 30 |
| `video_connector_num_layers` | **2** (config) — but checkpoint shows 8 `transformer_1d_blocks.{0..7}` per side |
| `video_connector_num_learnable_registers` | 128 |
| `audio_connector_*` (mirrors above) | head_dim=128, heads=30, layers=2, registers=128 |
| `connector_rope_base_seq_len` | 4096 |
| `rope_double_precision` | true |
| `rope_theta` | 10000.0 |
| `rope_type` | `split` |
| `causal_temporal_positioning` | false |

**Discrepancy**: config says 2 layers, checkpoint has 8 (`transformer_1d_blocks.{0..7}`) per modality. Possibly the config in `LTX-2` is an older variant and the 22B (LTX-2.3) connector was upgraded; see §8. Each connector block is video-only or audio-only with gated self-attn + FFN, no cross-attn, plus 128 learnable registers prepended.

Connectors produce the **per-modality text context** that feeds `attn2` (text cross-attn) inside the main 48 DiT blocks: video connector emits 4096-dim context, audio connector emits 2048-dim.

### 2g. Scheduler

From `scheduler/scheduler_config.json`: `FlowMatchEulerDiscreteScheduler`
- `num_train_timesteps`: 1000
- `shift`: 1.0
- `base_shift`: 0.95
- `max_shift`: 2.05
- `shift_terminal`: 0.1
- `time_shift_type`: `exponential`
- `use_dynamic_shifting`: true
- `base_image_seq_len`: 1024
- `max_image_seq_len`: 4096
- `invert_sigmas`: false
- `stochastic_sampling`: false

Flow-matching Euler with sequence-length-dependent shift; the shift function reshapes the schedule by `exp(α·log(seq_len))`.

### 2h. Upsamplers (LTX-2.3 only)

Spatial-x2 (`ltx-2.3-spatial-upscaler-x2-1.0.safetensors`, 996 MB, 72 tensors):
- `initial_conv.weight: [1024, 128, 3,3,3]` — 3D Conv3d on latents (in_ch=128 matches DiT/VAE latent space)
- `initial_norm.weight: [1024]` BF16
- 16 `res_blocks` + 16 `post_upsample_res_blocks`
- `upsampler.*` (2 tensors)
- `final_conv.weight: [128, 1024, 3,3,3]`

Spatial-x1.5 (`...x1.5-1.0.safetensors`, 1.09 GB, 73 tensors): adds a `blur_down.kernel: [1,1,5,5]` antialias kernel; `conv.weight: [9216, 1024, 3,3]` (note 2D 3×3, not 3D — this is the upsample-conv that produces pixel-shuffle by 9216 = 1024×9 → 3×3 spatial upscale by 1.5).

Temporal-x2 (`ltx-2.3-temporal-upscaler-x2-1.0.safetensors`, 262 MB, 72 tensors): smaller (`[512,128,3,3,3]` initial), same overall ResBlock+Upsample structure but applied along the temporal latent axis only.

All three upsamplers operate in **latent space** (input/output 128-channel latents that the video VAE then decodes), not pixel space.

---

## 3. Conditioning sidecar contract

From `safe_open` on `/home/alex/EriDiffusion/inference-flame/cached_ltx2_embeddings.safetensors` and `cached_ltx2_negative.safetensors`:

| File | Key | Shape | Dtype | Meaning |
|---|---|---|---|---|
| `cached_ltx2_embeddings.safetensors` | `text_hidden` | `[1, 1024, 4096]` | BF16 | Positive prompt — already **post-projection** (4096 = DiT video context dim) |
| `cached_ltx2_negative.safetensors` | `text_hidden` | `[1, 1024, 4096]` | BF16 | Negative prompt, same layout |

**Contract**:
- Single tensor per file, single key `text_hidden`
- Batch=1, sequence length=1024 tokens, hidden=4096 (= `cross_attention_dim` for the video DiT)
- Per-batch, **NOT** per-frame — same context replays across all video frames; the temporal axis is implicit in the DiT's 3D-positional Q grid, not in the context tensor.
- This is *already* the output of the **video** connector + text-embedding-projection pipeline — i.e. the Rust path bypasses Gemma3 + connectors entirely and feeds pre-baked context straight into the DiT's `attn2`. **The Mojo port should match this contract** so that the Rust-side encode pipeline can be reused (or any equivalent Mojo-side Gemma3 encoder must produce the same `[B,1024,4096]` tensor).
- No audio context cached — implies the current Rust path is video-only or that audio context is generated at run-time from the same Gemma hidden via the 2048-dim projection.
- No CLIP pooled embedding, no per-frame conditioning signal, no negative-positive concat — those happen at the sampler level (CFG-Star is referenced by `ltx2_cfg_star_ref.py`).

---

## 4. Two-stage video → audio data flow (latent shapes)

Default resolution example: 512×512 spatial, 33 frames, 16 kHz stereo audio.

```
PIXELS (video):            (B, 3, F=33, H=512, W=512)         BF16/F32
  └── VAE encode (×32 sp / ×8 t):
LATENT (video):            (B, 128, F'=5, H/32=16, W/32=16)   BF16
  └── patchify_proj (Linear 128→4096):
DIT tokens (video):        (B, T_video = 5·16·16 = 1280, 4096) BF16
                            + 3D RoPE on (T,H,W) positions

TEXT (prompt):             Gemma3 multi-modal hidden          F32
                            ↓ text_embedding_projection (188160 → 4096 video / 2048 audio)
TEXT context (video):      (B, 1, 4096)  →  expanded to (B, 1024, 4096) by tiling-of-registers
  └── video connector (8 layers self-attn + 128 learnable registers):
DiT text context (video):  (B, 1024 or 128+1024, 4096)         BF16  (this is the sidecar)

AUDIO mel:                 (B, 64, F_mel) at 16kHz/hop=160
  └── audio_vae encode:
LATENT (audio):            (B, 8, F_mel'=…)                   BF16
  └── audio_patchify_proj (128→2048): (matches audio in_channels=128, so latent flattens differently)
DIT tokens (audio):        (B, T_audio, 2048)                  BF16
                            + 1D RoPE on temporal positions only (max_pos=[20])

  ┌──────────────────────────────────────────────────────┐
  │  48× LTXTransformerBlock (each block does all 6 paths)│
  │   video self-attn → video↔text → video→audio          │
  │   audio self-attn → audio↔text → audio→video          │
  │   + AdaLN-single time conditioning (9-vec scale_shift)│
  │   + cross-modal AdaLN gates (a2v / v2a)               │
  └──────────────────────────────────────────────────────┘

DIT out (video) → proj_out (4096→128) → LATENT' (video)
DIT out (audio) → audio_proj_out (2048→128) → LATENT' (audio)

LATENT' (video)  → [optional spatial-x1.5 / x2 / temporal-x2 upscaler in latent space]
                 → VAE decode (×32 sp / ×8 t) → PIXELS

LATENT' (audio)  → audio_vae decode → mel  → vocoder (BigVGAN-v2 + BWE) → STEREO WAV @ ~48 kHz
```

Where do upsamplers sit? **All three upsamplers (`x1.5`, `x2`, `temporal-x2`) operate on latents**, NOT pixels. Their `initial_conv` is `[1024, 128, ...]` (in_ch=128 = latent channels) and `final_conv` returns to 128 channels. They are wedged between DiT output and VAE decode.

---

## 5. Dtype + memory budget

| Component | Params | BF16 size | Dtype storage |
|---|---|---|---|
| DiT (video+audio joint) | 21.005 B | 42.01 GB | BF16 (a few F32 scale_shift tables, total ~few MB) |
| Video VAE | 0.726 B | 1.45 GB | BF16 |
| Audio VAE | 0.053 B | 0.11 GB | BF16 |
| Vocoder | 0.129 B | 0.26 GB | BF16 |
| Text projection | 1.156 B | 2.31 GB | BF16 |
| Gemma3 text encoder | ~4 B (estimate) | ~16 GB at F32, ~8 GB at BF16 | F32 declared (external download) |
| Spatial upscaler ×2 | ~0.5 B (est. from 996 MB) | 0.99 GB | BF16 |
| Spatial upscaler ×1.5 | ~0.55 B | 1.09 GB | BF16 |
| Temporal upscaler ×2 | ~0.13 B | 0.26 GB | BF16 |
| **Single-file 22B BF16 checkpoint** | **23.07 B** | **46.15 GB** | BF16 (matches actual file size 46.149 GB) |

**MXFP4**: not present in headers — the LTX-2.3 22B is **pure BF16**, no MXFP4 quantization in the official checkpoint. However, the LTX-2 repo has `ltx_core/quantization/{fp8_cast,fp8_scaled_mm,trtllm_scaled_usable}.py` and there is a Rust binary `ltx2_fp8_stream_parity` — so an **FP8 streaming variant** is run in EriDiffusion (likely cast on the fly from BF16). No FP8 weights file on disk.

**Distilled LoRA** (`ltx-2.3-22b-distilled-lora-384.safetensors`, 7.6 GB): rank-384 LoRA delta that, when fused, yields the distilled-1.1 22B model. Not a separate full DiT.

**VRAM forward pass budget at 512×512×33 frames** (back-of-envelope, single sample, batch=1):
- Weights resident: 23 B × 2 B = 46 GB BF16. Will not fit on a single 24/32/48 GB GPU → requires block-streaming (`block_streaming/` in upstream, FP8 stream parity in Rust) or multi-GPU.
- Activations at DiT (per block): video tokens 1280 × 4096 × 2 B ≈ 10 MB, audio similar ≈ 5 MB, plus KV-cache and AdaLN intermediates. Across 48 blocks with recompute disabled ≈ 1–2 GB for video+audio.
- Patched VAE encode: 33 × 512² × 3 × 2 B ≈ 50 MB input, then 5 × 16² × 1024 × 2 B ≈ 2.6 MB latent at widest channel — VAE is cheap compared to DiT.
- Total realistic memory at full-resolution FP8-stream: ~12–16 GB activations + ~12 GB FP8 weights resident in transit = fits on a single 24 GB card.

---

## 6. Mojo port leverage points (vs serenitymojo ops)

Existing Mojo ops at `/home/alex/mojodiffusion/serenitymojo/ops/`:

| Op | LTX-2 use | Reusability |
|---|---|---|
| `linear.mojo` | All Q/K/V/out, FFN, patchify, adaln | DIRECT — every BF16 linear in DiT is reusable |
| `attention.mojo` | self-attn, cross-attn, cross-modal | DIRECT (SDPA + bf16) — but **needs to be extended for the per-head sigmoid gate** (`2*sigmoid(gate_logits)` multiplied with attn output) |
| `norm.mojo` (RMSNorm) | `q_norm`, `k_norm` | DIRECT |
| `rope.mojo` | RoPE on Q,K | **NEW MODE required**: LTX-2 uses 3D RoPE with `[20,2048,2048]` max-pos grid and *split* layout (not interleaved). Existing serenitymojo RoPE is 1D/2D in most models. |
| `embeddings.mojo` | Sinusoidal timestep emb (256-dim) | DIRECT |
| `softmax.mojo` | Attn softmax | DIRECT |
| `activations.mojo` | GELU-tanh approx for FFN, SiLU for VAE | Check `gelu_pytorch_tanh` formula matches |
| `conv.mojo` + `models/vae/conv3d.mojo` | VAE Conv3d (causal) | **Conv3d exists but causal padding (left-only on T) needs to be wired** — file already notes "causal temporal conv the caller pads D manually (left-only)" |
| `moe.mojo`, `mxfp4.mojo` | Not used by LTX-2 (no MoE, no MXFP4) | N/A |

**NEW Mojo ops needed for LTX-2** (in priority order):

1. **3D causal Conv3d wrapper** (small): existing `conv3d.mojo` does Conv3d but the LTX-2 video VAE needs *causal* temporal padding (`(k-1, 0)` only on T axis) plus *zeros* spatial in encoder and *reflect* spatial in decoder. Wrapper is mostly existing-op + padding helper.
2. **3D split-RoPE** (medium): split-mode (not interleaved) rotary on a 3D position grid with per-axis `max_pos = [20, 2048, 2048]`, fractional positions, double-precision freqs. Doesn't exist in serenitymojo. Block-cacheable.
3. **Gated SDPA** (small wrapper around existing attention): per-head sigmoid gate multiplied with attention output. Adds ~3 lines after the existing attention call.
4. **Cross-modal attention** (medium): video↔audio attention with mismatched Q (4096) and KV (2048) dims — already supported by `attn.to_q/to_k/to_v` with different in/out widths via existing Linear; the only twist is that `to_out` projects from one inner-dim to the *other* modality's inner-dim (e.g. audio→video out=4096, video→audio out=2048).
5. **AdaLN-single with 9-vec scale_shift_table** (medium): existing serenitymojo DiT models use 6-vec AdaLN; LTX-2 uses **9 coefficients per block** (scale1, shift1, gate1, scale2, shift2, gate2, scale_cross, shift_cross, gate_cross — likely). Just a wider broadcast.
6. **Snake-1d activation** (small): vocoder uses `x + (1/(β+ε)) * sin²(α x)` (BigVGAN). Not in `activations.mojo`. Per-channel `alpha,beta` learnables.
7. **STFT / iSTFT** (medium): vocoder has precomputed `forward_basis` and `inverse_basis` `[514,1,512]` — they're stored as conv-style weights so the iSTFT can be a Conv1d-Transpose. mel_basis is a `[64,257]` linear projection. Implementable as Linear + Conv1d-Transpose.
8. **Kaiser-Sinc low/high-pass filter convolution** (small): BigVGAN-v2 anti-aliased upsample uses 12-tap filters; stored in the checkpoint (`lowpass.filter`, `upsample.filter` etc.) so just a regular Conv1d — but with `groups=channels` (depthwise).
9. **Pixel-shuffle / pixel-unshuffle for 4×4×1 patch** (small): not in serenitymojo as a dedicated op. Just a reshape+transpose.
10. **PixelNorm** (small): per-pixel-position L2-normalize over channels — different from RMSNorm. Used in VAE and connectors.
11. **STG (Spatio-Temporal Guidance) mask** (small): from `ltx2_stg_mask_ref.py` — this is a sampler-side op (not DiT-internal); applied to perturbed CFG. Skip-layer style masking on a subset of frames.
12. **AdaIN tone-mapping** (small): from `ltx2_adain_tone_map_ref.py` — per-channel mean/std matching between reference and generated latent. Pure ops, no learnables.

**FP8 streaming**: the Rust path uses an FP8 streamer (`ltx2_fp8_stream_parity`); for Mojo we already have `offload/block_loader.mojo`. The LTX-2 22B fits cleanly into the existing offload pattern — load N blocks at a time, run forward, evict — with FP8 quantization on the fly. Existing `serenitymojo/offload/` is the natural anchor.

**No MoE, no MXFP4, no expert routing** — LTX-2 is dense. The Mojo `moe.mojo` / `mxfp4.mojo` ops are not in scope here.

---

## 7. Recommended chunking for a 3-team Mojo port

Each team gets a self-contained sub-model with a small parity check that can be wired up before full E2E generation works. Files map cleanly to existing `serenitymojo/models/{dit,vae,text_encoder}/` directories.

### Team A — Joint Audio-Video DiT (the 21 B core)

**Scope**: `serenitymojo/models/dit/ltx2_av_dit.mojo` + new `ops/rope3d_split.mojo`, gated-attention extension in `ops/attention.mojo`, AdaLN-single 9-vec wrapper. Weight prefix: `model.diffusion_model.*` (4444 tensors).

**Parity gates**:
1. RoPE3D-split single-position numerical match vs Python reference (`ltx2_dit_parity_inputs.py` already exists).
2. **Single block 0 forward** parity: ingest `block_0_input.safetensors` from `/home/alex/ltx2-refs/` → produce `block_0_output.safetensors` within cos≥0.999. (Block I/O tensors already exist on disk.)
3. Full 48-block stack forward parity using the cached sidecar `cached_ltx2_embeddings.safetensors` as text context.

**Rationale**: this is 91% of the params and the entire novel architecture (cross-modal joint attention is unprecedented in serenitymojo). The team owns the whole DiT directory, no overlap with VAE/vocoder files. Existing `/home/alex/ltx2-refs/block_N_{input,output}.safetensors` give a free per-block ladder — they can ship block-by-block parity.

### Team B — Video VAE + latent upsamplers (~0.7 B + ~1.1 B)

**Scope**: `serenitymojo/models/vae/ltx2_video_vae.mojo` (encoder + decoder), `serenitymojo/models/upsampler/ltx2_spatial_x2.mojo`, `serenitymojo/models/upsampler/ltx2_temporal_x2.mojo`, `serenitymojo/models/upsampler/ltx2_spatial_x1_5.mojo`. Weight prefixes: `vae.*` (170 tensors), and each upsampler's safetensors. New op work: causal Conv3d wrapper, PixelNorm, pixel-shuffle helpers.

**Parity gates**:
1. VAE encode parity: `(B,3,33,512,512) → (B,128,5,16,16)` within cos≥0.999 vs `ltx2_vae_encode_parity.py` reference.
2. VAE decode parity: same shape inverted.
3. Spatial-x2 upsampler in-latent parity using `ltx2_latent_upsampler_parity.rs` reference outputs.

**Rationale**: clean module boundary (`vae/` and `upsampler/` are entirely new files), self-contained ops (no AV cross-attn), and there's an existing diffusers reference + a Rust parity binary to bench against. Upsamplers reuse 90% of VAE ResBlock+Conv3d code so co-locating them in one team avoids duplicate ops.

### Team C — Audio path + connectors + text projection + sampler glue (~1.34 B + sampler)

**Scope**:
- `serenitymojo/models/vae/ltx2_audio_vae.mojo` (`audio_vae.*`, 102 tensors)
- `serenitymojo/models/vocoder/ltx2_vocoder.mojo` (`vocoder.*`, 1227 tensors) — needs STFT, snake-1d, kaiser-sinc filter ops
- `serenitymojo/models/text_encoder/ltx2_connectors.mojo` — the per-modality 8-layer connector transformer (`{video,audio}_embeddings_connector.*`)
- `serenitymojo/models/text_encoder/ltx2_text_projection.mojo` — the two `text_embedding_projection.{video,audio}_aggregate_embed` Linears
- `serenitymojo/sampling/ltx2_flowmatch.mojo` — `FlowMatchEulerDiscreteScheduler` w/ exponential time shift
- Sampler-side guidance ops: STG mask, AdaIN tone map, CFG-Star (reference scripts already exist)

**Parity gates**:
1. Audio VAE encode/decode parity vs `ltx2_audio_vae_decode_ref.py`.
2. Vocoder mel→waveform parity vs `ltx2_vocoder_ref.py`.
3. Connector forward parity (8-block transformer w/ learnable registers) on a synthetic input.
4. Scheduler sigma schedule parity vs `ltx2_sigma_schedule_ref.py` + `python_sigmas.safetensors` golden.

**Rationale**: Team C owns all the "thin" components — none of them is more than 1.3 B params individually, but they include several new ops (snake-1d, STFT, kaiser filter, PixelNorm shared with Team B). Pairing the audio decode chain with the sampler+guidance code keeps the sampler-side perturbation logic close to where audio is materialized. Crucially, **Team C does not need Team A's DiT done** — the connector and text-projection produce the sidecar that Team A consumes, so the two can develop in parallel and gate-check against the existing `cached_ltx2_embeddings.safetensors`.

**Cross-team coordination points** (just 3):
- `ops/rope3d_split.mojo` — owned by Team A but exported, Team B's upsamplers don't need RoPE but the connectors in Team C also use split-RoPE, so Team A delivers this op first.
- `ops/conv3d_causal.mojo` — owned by Team B, used by Team A's `patchify_proj` only conceptually (the DiT-side patchify is a plain Linear, not a Conv3d, so no real shared dependency).
- Sidecar contract (`text_hidden: [B,1024,4096] BF16`) — frozen on day 1, Team C produces, Team A consumes, no further coordination.

---

## 8. Open architectural questions

1. **Connector depth mismatch**: `connectors/config.json` declares `video_connector_num_layers: 2` and `audio_connector_num_layers: 2`, but the LTX-2.3 22B checkpoint has `transformer_1d_blocks.{0..7}` (8 blocks) for each. Resolution: the config in the `LTX-2` HF repo is likely the **LTX-2 19B** schema, while the actual 22B/2.3 weights use 8-layer connectors. Confirm by either: (a) loading with the actual diffusers `LTX2TextConnectors` class and counting; (b) finding an `LTX-2.3` config snapshot (no such config file present in the LTX-2.3 snapshots — they ship single-file weights only). Treat 8 layers as ground truth from the safetensors header.

2. **Vocoder `conv_pre.weight = [1536, 128, 7]`** input channels are 128, but `audio_vae.decoder.conv_out.conv.weight` outputs only 2 channels (stereo mel). So either (a) the vocoder consumes the **audio DiT latent directly** (128-ch) rather than passing through the audio VAE decoder, or (b) there is an unsurfaced 2→128 expansion. The mel_stft `mel_basis: [64, 257]` and `stft_fn.{forward,inverse}_basis: [514, 1, 512]` suggest the path may be: audio DiT → audio_vae.decode → 2-ch *waveform-like mel* → STFT → mel basis → 128-ch mel → vocoder. Cross-check against `vocoder.py:271-360` runtime call signature when Team C wires it.

3. **`text_proj_in_factor = 49`**: the text projection input is `188160 = 49 × 3840`. 49 = 7×7, but Gemma3's `mm_tokens_per_image = 256` and `vision_use_head = false`. Unclear whether 49 corresponds to a fixed-grid spatial pooling of vision tokens (7×7), to a 49-token chunk of the text sequence, or to a frame-rate-dependent count. Plausibly: a 7×7 grid average-pool of the 256 vision-tokens for keyframe conditioning. Confirm via running the upstream pipeline once.

4. **Audio positional grid `max_pos=[20]`**: only 20 positions for audio temporal — implies the audio is *very* coarsely tokenized (a few seconds per latent token). Verify by computing audio VAE temporal compression from `mel_hop_length=160` × `audio_vae` downsample factor.

5. **Bidirectional vs causal text encoding**: Gemma3 config sets `use_bidirectional_attention: false`. LTX-2 wraps the encoder output via the connectors that *can* use bidirectional self-attn (the connector blocks have no causal mask referenced). Confirm at Team A start.

6. **`apply_gated_attention` default**: the configurator defaults to `False` in `model_configurator.py:67`, but the 22B checkpoint clearly has `to_gate_logits.{weight,bias}` everywhere. The 22B Lightricks runtime must set it to True via a config we don't have locally. Treat as True.

7. **`av_ca_timestep_scale_multiplier` value**: configurator default is 1, but the LTX-2.3 weights have `av_ca_a2v_gate_adaln_single` and `av_ca_v2a_gate_adaln_single` — meaning the cross-modal gate AdaLN uses *separate* timestep scaling. The actual scalar (1, 1000, something else) determines whether audio and video share the same denoising step or run on different sigma schedules. Resolve by running `ltx2_two_stage.rs` once and dumping the scale.

8. **Pixel resolution constraint**: `F = 1 + 8k` per the VAE docstring. The DiT positional max for T is 20, so max latent F' = 20 → max pixel F = 1 + 8·19 = 153 frames. At 24 fps that's ~6.4 s. Confirm — this caps single-pass generation length without temporal upsampler.
