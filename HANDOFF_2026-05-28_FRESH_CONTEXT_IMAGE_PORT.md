> **SUPERSEDED 2026-05-29** — the model-reality table below is STALE. All image models
> listed here as noise/incomplete (Qwen, L2P, Anima, ERNIE, Lens, SD3, Chroma) are now
> COHERENT, plus SenseNova-U1 and the full LTX2 T2V+audio pipeline. See the authoritative
> `serenitymojo/docs/PORT_STATUS_2026-05-29.md`. Kept for historical context only.

# Fresh Handoff - Image Model Port Reality Check - 2026-05-28

Timestamp: 2026-05-28 08:09 PDT

This handoff is for the next Codex/agent resuming `/home/alex/mojodiffusion`.
It supersedes any optimistic wording in older status docs that implies the
image-model port is done. It is not done. Several gates compile or smoke, but
the user checked the outputs and correctly called out that Qwen and others are
noise or missing.

## User Direction To Preserve

- Finish the image models before moving full attention to video.
- Do not include Helios or Nucleus; the user removed them because they are bad.
- Do not add Stable Cascade; user explicitly rejected it as old/not used.
- Do not forget ERNIE, Microsoft Lens, Anima, Qwen, SD3 large/small, Chroma,
  and **ZImage L2P**.
- ZImage L2P is one model: a VAE-less ZImage/L2P pixel-space model.
- Do not call a model working based on contract/preblock/tensor smokes.
- Image models are only "working" after an actual generated image is visually
  coherent, not just finite.

## Current Process State

No active Mojo generation jobs were running at handoff time. A stale
`qwenimage_edit_synthetic_512_smoke.mojo` process was killed after it stuck in a
D/Z state.

Command used to check:

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'pixi run mojo|/bin/mojo|qwenimage|zimage_l2p|ernie|lens|anima|sd3|chroma' || true
```

Only the `rg` command itself appeared.

## Worktree State

The repo is very dirty. Do not revert user or previous-agent changes.

`git status --short` shows many modified tracked files plus many untracked port
files under `serenitymojo/`. Important untracked image-model files include:

- `serenitymojo/models/dit/anima_contract.mojo`
- `serenitymojo/models/dit/chroma_contract.mojo`
- `serenitymojo/models/dit/chroma_dit.mojo`
- `serenitymojo/models/dit/ernie_contract.mojo`
- `serenitymojo/models/dit/ernie_image.mojo`
- `serenitymojo/models/dit/qwenimage_contract.mojo`
- `serenitymojo/models/dit/qwenimage_edit_contract.mojo`
- `serenitymojo/models/dit/sd3_contract.mojo`
- `serenitymojo/models/dit/sd3_mmdit.mojo`
- `serenitymojo/models/dit/zimage_l2p_contract.mojo`
- `serenitymojo/models/dit/zimage_l2p_dit.mojo`
- `serenitymojo/models/dit/zimage_l2p_local_decoder.mojo`
- `serenitymojo/models/lens/`
- `serenitymojo/pipeline/*qwenimage*`
- `serenitymojo/pipeline/*zimage_l2p*`
- `serenitymojo/pipeline/*ernie*`
- `serenitymojo/pipeline/*lens*`
- `serenitymojo/pipeline/*anima*`
- `serenitymojo/pipeline/*sd3*`
- `serenitymojo/pipeline/*chroma*`

Use `git status --short` again before editing. Do not run destructive git
commands.

## Current Output Artifacts

Fresh `find output` result for the called-out models:

```text
2026-05-28 04:26     787072 output/qwenimage_vae_noise_512.png
2026-05-28 05:00     787072 output/qwenimage_edit_synth_512.png
2026-05-28 05:03    3147060 output/sd3_vae_noise_1024.png
2026-05-28 05:12     787072 output/qwenimage_first_512.png
2026-05-28 05:43    3147060 output/sd3_medium_vae_noise_1024.png
2026-05-28 08:05    3147060 output/anima_vae_from_rust_latent_1024.png
```

Important: there is no Mojo `output/*zimage_l2p*`, no `output/*ernie*`, no
`output/*lens*`, and no `output/*chroma*` image artifact. SD3 images are VAE
noise decodes, not generated images.

Viewed artifacts:

- `output/qwenimage_first_512.png`: visibly noisy/under-denoised one-step
  artifact. It is not a working Qwen image.
- `output/qwenimage_edit_synth_512.png`: synthetic/random edit artifact. It is
  not a working edit pipeline.
- `output/anima_vae_from_rust_latent_1024.png`: coherent decode, but it is
  Mojo VAE decode of a Rust latent oracle, not a Mojo Anima denoise pipeline.

## Model Reality Table

| Model | Real current state | Next required work |
| --- | --- | --- |
| Z-Image | Previously working end-to-end native 1024. Do not confuse this with L2P. | Leave alone unless regression appears. |
| ZImage L2P | Contract/preblock/local-decoder/pixel-space gates only. No full transformer block stack, no full native L2P sampler, no Mojo image output. Rust oracles exist and are coherent. | Port first L2P transformer block and parity against Rust. Then full 30-layer/30-step pixel-space runner. |
| Qwen-Image | Has a full-looking pipeline file but `STEPS = 1` and it keeps full padded text instead of applying Qwen drop_idx/template parity. Output is noise/one-step artifact. | Fix prompt hidden-state handling/drop_idx, make true multistep 20/30/50-step runner, then inspect PNG. |
| Qwen-Image-Edit | Synthetic target/reference only, not a real edit pipeline. Last run stuck and was killed. | Do not spend time here before base Qwen is coherent. Need real image/reference encode path. |
| ERNIE-Image | Metadata/scheduler/resident preblock plus bounded real-weight block0 gate. No Mistral3B encoder, no 36-layer DiT, no final unpatchify/denoise/VAE image. | Port encoder or cached text sidecar, full ERNIE block loop, final projection, then denoise + Klein/ERNIE VAE. |
| Microsoft Lens | Header/scheduler and sampled block0 QKV/QK RoPE gates only. No GPT-OSS MXFP4 execution, no full Lens DiT, no VAE decode. | Keep as sidecar until GPT-OSS/MXFP4 and full DiT are ported; use Rust parity captures. |
| Anima | Metadata/cached-context tensor gate and Mojo VAE decode of Rust latent oracle. No MiniTrainDIT/adapter/Qwen3/T5 prompt path. | Start cached-context MiniTrainDIT block slice against existing Rust sidecar. |
| SD3 Large/Medium | Contracts, schedules, VAE noise decodes, and MMDiT pre/post-block gate. No full joint MMDiT blocks or prompt encoder assembly. | Port full joint-block stack and triple encoder/cached prompt path. |
| Chroma | Real-weight staged DiT smoke through early double/single blocks and final projection, not full denoise. No image output. | Extend to all 19 double + 38 single blocks and cached T5/VAE pipeline. |

## ZImage L2P Details

This is the user’s “where is L2P?” answer.

Primary handoff:

- `serenitymojo/docs/ZIMAGE_L2P_STARTING_PASS_2026-05-28.md`

Mojo files:

- `serenitymojo/models/dit/zimage_l2p_contract.mojo`
- `serenitymojo/models/dit/zimage_l2p_dit.mojo`
- `serenitymojo/models/dit/zimage_l2p_local_decoder.mojo`
- `serenitymojo/pipeline/zimage_l2p_contract_smoke.mojo`
- `serenitymojo/pipeline/zimage_l2p_dit_preblock_smoke.mojo`
- `serenitymojo/pipeline/zimage_l2p_local_decoder_smoke.mojo`
- `serenitymojo/pipeline/zimage_l2p_schedule_smoke.mojo`
- `serenitymojo/pipeline/zimage_l2p_pixel_smoke.mojo`

Rust source:

- `/home/alex/EriDiffusion/inference-flame/src/bin/l2p_infer.rs`
- `/home/alex/EriDiffusion/inference-flame/src/models/l2p/{dit.rs,local_decoder.rs,rope.rs,weight_loader.rs}`
- `/home/alex/EriDiffusion/inference-flame/src/sampling/l2p_sampling.rs`

Checkpoint and sidecars:

- Checkpoint:
  `/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors`
- Conditioning sidecar:
  `/home/alex/EriDiffusion/inference-flame/output/l2p_embeddings.safetensors`
- Rust coherent image oracles:
  `/home/alex/EriDiffusion/inference-flame/output/l2p_512_oracle.png`
  and `/home/alex/EriDiffusion/inference-flame/output/l2p_1024_oracle.png`

Critical facts:

- Pixel-space input/output: `[B,3,H,W]`, no VAE.
- Patchify16: `[1,3,1024,1024] <-> [1,4096,768]`.
- Hidden 3840, 30 main layers, 30 heads, head dim 128.
- Consumes normalized flow sigma and maps internally to `(1 - sigma) * 1000`.
- Model output is negated before sampler in Rust; do not double-negate.
- Defaults: 1024, 30 steps, CFG 2.0, shift 3.0, seed 42.
- Current Mojo local decoder smoke only does native pixel roundtrip plus tiny
  32x32 MicroDiffusionModel path. It is not a full 1024 image runner.

Run current L2P gates, but do not interpret these as image success:

```bash
pixi run mojo -I . serenitymojo/pipeline/zimage_l2p_contract_smoke.mojo
pixi run mojo -I . serenitymojo/pipeline/zimage_l2p_schedule_smoke.mojo
pixi run mojo -I . serenitymojo/pipeline/zimage_l2p_pixel_smoke.mojo
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_l2p_dit_preblock_smoke.mojo -o /tmp/zimage_l2p_dit_preblock_smoke
/tmp/zimage_l2p_dit_preblock_smoke
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/zimage_l2p_local_decoder_smoke.mojo -o /tmp/zimage_l2p_local_decoder_smoke
/tmp/zimage_l2p_local_decoder_smoke
```

## Qwen-Image Immediate Root Cause

File: `serenitymojo/pipeline/qwenimage_pipeline_smoke.mojo`

Current constants:

```mojo
comptime N_TXT = 256
comptime STEPS = 1
comptime CFG = Float32(4.0)
```

The file itself says the diffusers pipeline drops chat-template prefix tokens
from hidden states (`drop_idx`) but this smoke keeps the full padded sequence.
That plus one denoise step explains the noise output. Do not waste time tuning
the VAE before fixing this.

Next Qwen steps:

1. Read `/home/alex/EriDiffusion/inference-flame/src/bin/qwenimage_gen.rs` and
   `/home/alex/EriDiffusion/inference-flame/src/models/qwenimage_dit.rs`
   line-by-line for prompt template, drop_idx, schedule, CFG, timestep, and VAE
   scaling.
2. Create a separate multistep runner rather than mutating a smoke into a
   quality claim. Suggested name:
   `serenitymojo/pipeline/qwenimage_pipeline_512_multistep.mojo`.
3. Keep heavy jobs sequential. Do not launch Qwen, Anima, and L2P together.
4. Save a fresh artifact with a name that includes steps, e.g.
   `output/qwenimage_512_30step.png`.
5. Use `view_image`/visual inspection before saying it works.

## ERNIE Details

Primary handoff:

- `serenitymojo/docs/ERNIE_HANDOFF_2026-05-28.md`

Current verified gates claimed by that doc:

```bash
pixi run mojo -I . serenitymojo/pipeline/ernie_contract_smoke.mojo
pixi run mojo -I . serenitymojo/pipeline/ernie_resident_smoke.mojo
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/ernie_block0_smoke.mojo -o /tmp/ernie_block0_smoke_bound
/tmp/ernie_block0_smoke_bound
pixi run mojo -I . serenitymojo/sampling/ernie_sampling_smoke.mojo
```

Current ERNIE state is not an image model. Missing full Mistral3B encoder, full
36-layer ERNIE block stack, final projection/unpatchify, denoise loop, and VAE
decode. There is no `output/*ernie*` image.

## Lens Details

Primary handoff:

- `serenitymojo/docs/LENS_SIDECAR_HANDOFF_2026-05-28.md`

Current Lens state is sidecar-only. It validates headers, schedule math, and
sampled block0 QKV/QK RoPE from real weights/captures. It does not port MXFP4
GPT-OSS, the full Lens DiT forward, or the Lens VAE. There is no
`output/*lens*` image.

## Anima Details

Primary handoff:

- `serenitymojo/docs/ANIMA_HANDOFF_2026-05-28.md`

Current Anima state:

- `anima_cached_context_smoke.mojo` validates cached sidecar tensors and Euler
  helpers.
- `anima_vae_latent_smoke.mojo` decodes a Rust latent oracle through Mojo VAE
  and writes `output/anima_vae_from_rust_latent_1024.png`.
- No MiniTrainDIT, adapter, Qwen3, T5, or full Mojo denoise path exists yet.

Do not claim the coherent Anima PNG proves Anima port completion; Rust produced
the latent.

## SD3 Details

Primary handoff:

- `serenitymojo/docs/SD3_STARTING_PASS_2026-05-28.md`

Current SD3 Large/Medium state:

- Metadata/header/checkpoint contracts.
- SD3 shifted-flow schedule smoke.
- VAE noise decodes:
  `output/sd3_vae_noise_1024.png`,
  `output/sd3_medium_vae_noise_1024.png`.
- `sd3_mmdit_preblock_smoke.mojo` performs patch embed, pos embed, timestep,
  pooled conditioning, context projection, final projection, and unpatchify for
  Large and Medium.

Still missing: full joint attention block stack, CLIP-L/CLIP-G/T5 prompt
assembly, CFG denoise loop, and generated image.

## Chroma Details

Current Chroma files:

- `serenitymojo/models/dit/chroma_contract.mojo`
- `serenitymojo/models/dit/chroma_dit.mojo`
- `serenitymojo/pipeline/chroma_contract_smoke.mojo`
- `serenitymojo/pipeline/chroma_dit_smoke.mojo`

The staged smoke says explicitly that it is not a full image generator. It
exercises a real-weight subset and final projection. No full denoise, no cached
T5/VAE image pipeline, and no `output/*chroma*` image exist.

## Recommended Resume Order

1. Qwen-Image base: most complete path, currently bad for clear reasons
   (`STEPS=1`, missing drop_idx/text parity). Fix this first and generate a
   visually coherent PNG.
2. ZImage L2P: user specifically asked for it. Build from Rust parity and do
   not confuse it with base Z-Image. First transformer block parity before full
   runner.
3. Anima: use existing cached-context sidecar and Rust latent oracle to port
   MiniTrainDIT slices without fighting tokenizer/prompt plumbing first.
4. ERNIE: port cached/sidecar encoder boundary or Mistral3B, then full DiT.
5. SD3 and Chroma: extend existing preblock/staged gates into full blocks.
6. Lens: leave until GPT-OSS MXFP4/full DiT can be addressed with parity.

## Hard Rules For The Next Session

- Do not mark image models done from compile or smoke output.
- Use real source parity before quality claims.
- Generate one heavy image at a time.
- Inspect the saved PNG before reporting success.
- Keep all old handoff docs, but treat this file as the current truth.
