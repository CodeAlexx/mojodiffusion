# serenitymojo Port Status — 2026-05-29 (authoritative)

This supersedes the model-reality tables in `HANDOFF_2026-05-28_FRESH_CONTEXT_IMAGE_PORT.md`,
`FULL_PORT_STATUS_2026-05-27.md`, and `FULL_PORT_MODEL_INVENTORY_2026-05-27.md` (all stale —
they predate the 2026-05-28/29 completions below).

Gate doctrine (unchanged, hard rule): a model is "working" only when a generated artifact is
**visually coherent** (image) / **coherent + audible** (video) — never on compile or finite-stats.
Per-op/block parity = cos≥0.999 vs the Rust/Python reference; deep multi-forward chains use 0.99.

## Image models — ALL COHERENT (verified by direct view)

| Model | State | Gate | Commit |
| --- | --- | --- | --- |
| Z-Image (base) | working (pre-existing) | native 1024 | (prior) |
| Qwen-Image 512 & 1024 | ✅ coherent | fixed by adding sdpa `(546,28,128)` comptime case | 840eda8 |
| Klein9B (FLUX.2) | ✅ coherent | honeycomb 1024; also turbo-loader byte-exact | 840eda8 |
| ZImage L2P | ✅ coherent | 30-layer pixel-space; matches Rust 512 oracle | 840eda8 |
| Anima (MiniTrainDIT) | ✅ coherent | 28-block; latent cos 0.9948 vs Rust | 840eda8 |
| ERNIE-Image | ✅ coherent | 36-layer streamed + Mistral3B sidecar + Klein VAE | 840eda8 |
| SD3.5 Medium | ✅ coherent | 24-block joint MMDiT + triple-encoder sidecar | 840eda8 |
| SD3.5 Large | ✅ coherent | 38-block, block-streamed | 840eda8 |
| Microsoft Lens | ✅ coherent + **DiT math parity-verified** (cos 0.99996) | block + GPT-OSS captures; real-prompt img blocked on GPT-OSS encoder OOM | 840eda8 |
| Chroma | ✅ coherent | 19+38 block + T5 sidecar + FLUX VAE (early "blob" was a VAE-dtype bug) | 840eda8 |
| SenseNova-U1 | ✅ coherent | 512 ginger-tabby; RoPE blocker fixed + SYSTEM_MESSAGE conditioning. Pixel-parity vs Rust = follow-on | 1fad1f6 |

Per-model caveats live in `serenitymojo/parity/SKEPTIC_FINDINGS_*.md`.

## LTX2 — text-to-video WITH AUDIO + LoRA (full pipeline)

First end-to-end T2V+audio clip generated in pure Mojo+MAX. Plan: `docs/LTX2_PORT_PLAN_2026-05-28.md`.
- De-risk foundation (ops + STFT-as-conv + connector + FP8 streaming, 23B @ ~10GB): 840eda8
- P3 joint-AV block (6 attention paths, fail-closed v2a gate): 3d90c57
- P5 video VAE (cos 0.99998) + 48-block DiT stack: bfa2836
- P1/P4 BigVGAN vocoder (cos 0.99996, audible) + audio VAE (cos 0.99999): cd92b0d
- P7 MVP — first coherent T2V+audio mp4: e74f187
- P6 LoRA (rank-384, **added at dequanted linear, NEVER fused**, all 1660 keys): d8f9e4f
- Proper audio: real `feature_extract_and_project` context (slice deleted; "Gemma broken in Rust" was stale): 3492580
- **Quality pass (IN PROGRESS):** res_2s 2nd-order RK sampler + HQ recipe (distilled + res_2s + 15 steps + CFG + distilled-LoRA@0.25 + 768×512 + full context). MVP at 256² was soft; HQ targets visibly sharp. The desktop HQ workflow (`ltx2-app/archive_pre_lightricks_*`) was NOT previously ported.
- Known follow-ons: stage-x2/AdaIN upsampler unbuilt; video-velocity 0.9947 deep-chain bf16 reduction-order drift (no visual impact); audio not yet full-DSP-parity-gated end-to-end.

## PiD — Pixel Diffusion Decoder (NVIDIA), in progress

Drop-in pixel-space diffusion decoder (replaces VAE + upsamples 2k/4k) for SD3/Z-Image/FLUX backbones.
Plan: `docs/PID_PORT_PLAN_2026-05-29.md`.
- Ops foundation DONE (all fail-closed, no checkpoint): patchify/unpatchify, NTK 2D RoPE,
  TimestepConditioner(mp=10), SigmaAwareGate, LQProjection2D, PiTBlock, PixelDiT MMDiT block — `serenitymojo/models/pid/`: 35a7a62
- Parity phase STARTING: assemble PixelDiT net → 4-step sampler → full-net bf16 parity vs Python PiD ref (risk sink) → SD3 PiD-decode 2048². Checkpoint (`nvidia/PiD` SD3 decoder 2.72GB + sd3 VAE) downloading.

## Async turbo offload (stagehand → Mojo)

`serenitymojo/offload/`: `TurboBlockLoader` (create_stream + DeviceEvent double-buffer; fence
proven load-bearing — 91% corruption without it), `residency.mojo` (eviction/budget),
`TurboPlannedLoader` + Klein wiring (byte-exact parity). Committed 840eda8. Note: no wall-clock
speedup on compute-bound 1024 Klein; payoff is large block-streamed models (LTX2 23B uses the
streaming path). The real stagehand reference is `serenityflow-v2/.../memory/stagehand/`.

## New op / module inventory (this campaign)

- `ops/`: `unary` (sin/exp/sqrt/rsqrt/tanh/reciprocal), `reduce` (axis-reduction/AdaIN), `conv1d`
  (+conv_transpose1d, depthwise/dilated), `snake` (snake-beta), `pixelshuffle` (depth_to_space_3d +
  pixel_shuffle/unshuffle), `fp8` (E4M3 dequant), `activation1d`, `cast` (+from_view_as_f32 in tensor.mojo),
  `activations.sigmoid`.
- `models/vocoder/`: `ltx2_stft` (forward-STFT-as-conv), `ltx2_vocoder` (BigVGAN+BWE).
- `models/vae/`: `ltx2_vae_decoder` (video), `ltx2_audio_vae`.
- `models/dit/`: `ltx2_dit` (joint-AV block + FP8 from_fp8_block), `ltx2_connector`, `ltx2_rope`,
  `anima_dit`, `sd3_mmdit`, `sensenova_u1`, `ernie_image`, `chroma_dit`, `zimage_l2p_*`.
- `models/pid/`: `pid_ops`, `lq_projection`, `pit_block`, `pixeldit_block`.
- `offload/`: `turbo_loader`, `turbo_planned_loader`, `residency`, `ltx2_block_stream`, `turbo_slots`.
- `sampling/`: `ltx2_sampling` (distilled + res_2s WIP), `ltx2_guidance` (CFG-star + STG).

## Durable rules (see also memory)
- **LoRA is NEVER fused into a saved model** — always ADDED (W += scale·B@A in-memory, or runtime
  overlay; for FP8-streamed models, applied at the dequanted linear per stream).
- **MAX 26.3 async**: `DeviceContext` is a singleton; for stream overlap use `ctx.create_stream()`
  + `DeviceStream.enqueue_function` + `DeviceEvent`; `enqueue_copy` is default-stream-only.

## Not done / parked / queued
- **PiD**: parity + decode phases (need the downloaded checkpoint).
- **LTX2**: quality pass finishing; Gemma audio-context encoder (audio not yet prompt-faithful at full fidelity); stage upsampler; 4K.
- **Lens**: real-prompt oracle image (GPT-OSS encoder OOM).
- **SenseNova-U1**: pixel-parity vs Rust oracle (GPU contention); perf (re-streams layers/step).
- **Qwen-Image-Edit**: deferred (real reference-image encode path).
- Excluded by user: Helios, Nucleus, Stable Cascade.

## Commit trail (mojodiffusion, branch master, NO remote — local only)
840eda8 (session base: 8 image + turbo + LTX2 foundation) → 3d90c57 (P3) → bfa2836 (P5) →
cd92b0d (P1/P4) → e74f187 (P7 MVP) → d8f9e4f (P6 LoRA) → 3492580 (proper audio) →
1fad1f6 (SenseNova) → 35a7a62 (PiD foundation). (Plus probe f5a78b0.)
LTX2 Desktop app refactor committed separately in `ltx2-app` (5e21699).
