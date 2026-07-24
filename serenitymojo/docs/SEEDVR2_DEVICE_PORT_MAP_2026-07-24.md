# SeedVR2-3B — pure-Mojo device port (source map)

Port of the full **SeedVR2-3B** video super-resolution model (ByteDance-Seed,
ICLR'26, Apache-2.0) to pure Mojo + MAX, GPU, inference-only. Every computational
stage is parity-gated against a torch reference; the whole model runs end-to-end at
**arbitrary resolution**. Net-new GPU kernels across the entire model: **one**
(`apply_rope3d`); everything else reused the existing stack.

Reference source (cloned, code only): `/home/alex/SeedVR` (github ByteDance-Seed/SeedVR,
config `configs_3b/main.yaml`). Weights + oracles + shims: `/home/alex/models/seedvr2-3b`.

## Architecture (config `configs_3b/main.yaml`)
- **VAE** `video_vae_v3` `s8_c16_t4_inflation_sd3`: causal-3D AutoencoderKL, block
  channels [128,256,512,512], latent 16ch, spatial ÷8, temporal ÷4, GroupNorm-32,
  SiLU, mid-block attention. `scaling_factor` 0.9152.
- **DiT** `dit_v2.NaDiT`: vid_dim 2560, 32 layers (mm_layers 0-9 dual-stream separate
  weights, 10-31 shared `.all`, 31 vid-only MLP), heads 20 × head_dim 128, patch
  [1,2,2], window (4,3,3) (num-windows; even layers 720pwin, odd 720pswin shifted),
  mmrope3d rope_dim 128, AdaSingle, swiglu MLP, fusedrms qk-norm. vid_in 33ch → 16.
- **Sampler**: euler, v_lerp (flow-matching), lerp schedule T=1000, 50 steps
  uniform-trailing + SD3 shift-transform (resolution-dependent), CFG 7.5 rescale 0.
  Text-free at inference (ships `pos_emb.pt`[58,5120] / `neg_emb.pt`[64,5120]).
- SR conditioning: DiT input = cat(z_t[16], latent_blur[16], flag[1]) = 33ch.

## Mojo module layout
- `serenitymojo/models/vae/seedvr2_vae.mojo` — causal-3D VAE. `encode_seedvr2_vae`,
  `decode_seedvr2_vae`; blocks `causal_conv3d[KD,KH,KW]`, `resnet_block3d`,
  `upsample3d[SR,TR]`, `downsample3d[TEMPORAL_DOWN]`, `mid_attention`,
  `_group_norm_per_frame`. Resolution-agnostic (only kernel/ratio comptime; H/W
  runtime via `conv3d_fcqrs_cudnn`).
- `serenitymojo/models/dit/seedvr2_dit.mojo` — the DiT. `vid_in`, `emb_in`, `txt_in`;
  attention `attn_qkv_norm`, `apply_rope3d` (the one net-new kernel), fixed-grid
  `window_attn_projout`, general `window_attn_projout_general`; `mmdit_block(_g/_general)`,
  `dit_stack(_g/_general)`, `dit_out_tail(_general)`, `compute_dit_freqs(_general)`,
  `compute_windows` / `compute_windows_shifted` (720p window sizing), `DiTWeights`
  + `load_dit_weights` + `load_block`, `full_dit_forward` / `full_dit_forward_pre` /
  `full_dit_forward_general`.
- `serenitymojo/models/dit/seedvr2_sampler.mojo` — `cfg`, `euler_vlerp_step`,
  `sampler_timesteps`.
- `serenitymojo/sampling/seedvr2_upscale_cli.mojo` — fixed 128×128×13 upscale CLI.
- `serenitymojo/sampling/seedvr2_upscale_general_cli.mojo` — arbitrary-resolution CLI
  (computes latent grid + SD3 sampler shift from image size).
- Parity gates: `serenitymojo/models/{vae,dit}/tests/seedvr2_*.mojo`.

## Reused stack ops (measure-first — no reimplementation)
`conv3d_fcqrs_cudnn`, `group_norm`, `silu`, `swiglu`, `linear`/`linear_bias`,
`rms_norm`, `softmax_lastdim`, `sdpa_nomask`, `gather_rows`, `pixel/depth_to_space`,
`upsample_nearest2x_nhwc`, `concat`/`reshape`/`permute`/`slice`/`add`/`mul`/`mul_scalar`.

## Parity gates (all re-run by the orchestrator; bf16 unless noted)
| stage | grid | cos |
|---|---|---|
| VAE decode | latent[1,16,4,16,16]→[1,3,13,128,128] | 1.0 (F32) |
| VAE encode | video→latent | 0.9999999 (F32) |
| vid_in / emb_in | 128²×13 | 1.0 |
| attn qkv+qk-norm / rope / window+projout | 128²×13 | 0.99999 / 1.0 / 0.99999 |
| full MMDiT block | 128²×13 | 0.9999963 |
| 32-block stack | 128²×13 | 0.9927 (deep bf16 accum) |
| output tail | 128²×13 | 0.99999 |
| self-contained full DiT fwd (computed freqs) | 128²×13 | 0.9995 |
| sampler math (CFG/euler/schedule) | — | 1.0 |
| sampler driver → final_latent | 128²×13 | 0.9816 |
| **e2e decoded vs torch pipeline** | 128²×13 | **0.9987** |
| general window attention | 8 uneven windows (2,24,24) | 0.99999 |
| grid-general full DiT fwd (shifted+non) | (2,24,24) | 0.9995 |
| general CLI verify (reproduces fixed) | 128²×13 | 0.9987 |

## Oracles / shims (`/home/alex/models/seedvr2-3b`, torch = OneTrainer venv)
- Shims (on `sys.path` before SeedVR for the DiT only — they break diffusers, so the
  VAE runs without them): `flash_attn`→torch SDPA, `apex.normalization`→pure-torch
  RMSNorm/LayerNorm, plus `pip install rotary_embedding_torch`. RoPE `get_freqs`
  monkeypatched to a small max-grid table (position-identical, avoids ~8GB OOM).
- Oracle scripts: `vae_oracle.py`, `vae_encode_oracle.py`, `dit_oracle.py`,
  `dit_tail_oracle.py`, `sampler_oracle.py`, `pipeline_oracle.py` (3-process
  encode|sample|decode), `window_gen_oracle.py`, `dit_fullfwd_2x24x24.py`.
- Weights (bf16 safetensors): `seedvr2_dit.safetensors`, `seedvr2_vae.safetensors`,
  `seedvr2_text_emb.safetensors`. Source `.pth` (fp32) + `pos/neg_emb.pt` alongside.

## Running (5080 16GB, single process)
Fixed 128×128×13 or general CLI — AOT build with the PNG/JPEG link flags (mirror
`realesrgan_cli`):
```
pixi run mojo build -I . -I vendor/mojo-libs -Xlinker -lm -Xlinker -lcuda \
  -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 -Xlinker -lpng16 \
  -Xlinker -lturbojpeg serenitymojo/sampling/seedvr2_upscale_general_cli.mojo -o /tmp/gup
env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/gup
```
Peak ~10.7 GB. Constraints: image H,W divisible by 16; frames = 4k+1.

## Known upstream issue — `vid_out_ada` (NOT a port defect)
The shipped SeedVR2-3B `vid_out_ada` **crashes** as released: `AdaSingle` asserts
`emb_dim == 6*dim` (2-layer design) but the output ada uses `layers=["out"]` (1
layer), so `l=len(self.layers)` factorizes the emb to 5120 vs the 2560
`out_scale`/`out_shift` params → RuntimeError. Verified byte-identical on the github
repo AND the live HF Space demo; the 7B config has no `vid_out_norm` so only the 3B
hits it. No released artifact can produce ground truth, so the port uses the
deterministic consistent reconstruction (the `6*dim`/l=2 factorization,
`idx=layers.index("out")=0` → first emb slot: shift=col0, scale=col1). This is the
only DiT op not gatable against the unmodified model; the full pipeline is gated
against a torch reference using this identical reconstruction. See `dit_out_tail`.

## Follow-on (not done)
- Higher step counts / the full 50-step schedule (validated at 8 steps).
- Int8 / FP8 for larger batches or the 7B.
- A general-CLI torch full-pipeline oracle at a non-128² resolution (components are
  each gated; only the fixed-grid full pipeline has an e2e torch reference).
