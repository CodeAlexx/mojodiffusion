# serenitymojo — Full inference-flame → Mojo Port Roadmap

**Goal:** port the entire `inference-flame` Rust inference stack (~40 models + shared VAEs/encoders/samplers) to **pure Mojo + MAX**, inference-only, GPU-only, as embeddable `prompt→image/video/audio` pipelines. Reference = `inference-flame/` (architecture) + diffusers/upstream (numerics oracle). Method = the `mojo-port` skill loop (builder→skeptic→bugfix→parity, cos≥0.999 vs GPU-bf16 oracle).

**Date:** 2026-05-26. **Companion:** memory `project_mojodiffusion_max_phase0`, `STATUS_ZIMAGE_DENOISE_DIVERGENCE.md`, `serenitymojo/MAP.md`.

---

## 0. Current state (the baseline this builds on)

**DONE / verified:** foundation — `tensor`, `parity` (ParityHarness), all `ops/` (linear, norm{rms/layer/group}, rope{interleaved/halfsplit}, attention/sdpa[math-mode Dh=128], conv2d, **conv3d**, **moe**{router/grouped-FFN/scatter}, embeddings, layout{patchify/unpatchify/deinterleave}, tensor_algebra, activations, softmax); `io/` loader (byte-parity); `offload/BlockLoader`; `tokenizer` (Qwen BPE); `image/png`; samplers `flow_match` + `flux2_klein` + `sdxl_euler`; VAE kit `decoder2d`+`conv3d`+`upsample`+`vae_ops`.
- **Z-Image** — WORKING end-to-end (1024², coherent images). Encoder `qwen3`, `zimage_nextdit`, VAE `zimage_decoder`, flow-match.
- **Klein 4B/9B** — IN FLIGHT (Codex): `klein_dit.mojo` + `flux2_klein` sampler + klein9b smokes.
- **SDXL** — NEXT (Codex): `sdxl_euler` sampler started.
- `qwen25vl_encoder` started.

**Hard-won conventions to carry to EVERY model** (see §6): GPU-bf16 oracle only (never fp32-CPU/host-load); read the diffusers/repo **pipeline denoise loop line-by-line for sign/CFG/timestep** (the Z-Image bug was a missed post-CFG `noise_pred = -noise_pred`); **trust the numerical fingerprint over the source-read** when they disagree; comptime sequence lengths for sdpa; F32 latent + bf16 model input; all file I/O through `io/ffi`.

## 1. Reuse map (why families port together)

Most cost is shared infra, not the DiT. Build each shared piece ONCE, reuse across the family:

| Shared piece | Built? | Reused by |
|---|---|---|
| qwen3 encoder | ✅ | Z-Image, Klein, (Qwen text path) |
| qwen2.5-VL encoder | ⏳ started | Qwen-Image (+edit) |
| **clip encoder** | ❌ | SDXL, SD1.5, SD3, Cascade, Kandinsky5 |
| **t5 / t5gemma2** | ❌ | SD3, Chroma, FLUX.1, ERNIE, LTX-2 |
| gemma3 / mistral / gpt_oss | ❌ | HiDream-O1, Lumina-ish, misc |
| 2D-VAE kit (decoder2d) | ✅ | Z-Image, Klein, SDXL/SD (ldm), Chroma |
| **ldm VAE (enc+dec)** | ❌ (kit exists) | SDXL, SD1.5, SD3 |
| qwenimage VAE | ❌ | Qwen-Image |
| wan21/wan22 3D VAE | ⏳ conv3d done | Wan2.2, Cosmos2.5, video |
| MoE ops | ✅ | Nucleus (17B MoE), SenseNova (MoT) |
| flow-match euler | ✅ | most DiTs |
| UniPC sampler | ❌ | MagiHuman, Cosmos |
| DMD / pyramid | ❌ | Helios |
| audio VAE + STFT/vocoder | ❌ | AceStep, LTX-2 audio |
| **UNet blocks (ResBlock+cross-attn+up/down)** | ❌ | SD1.5/SDXL/SD3-partial, Cascade |

## 2. Phased roadmap (order = reuse leverage × oracle availability × value)

### Phase A — headline image DiTs, MAX-oracle-backed *(in progress)*
Gradeable against `max serve` AND diffusers. Heavy reuse of the Z-Image foundation.
1. **Klein 4B/9B** (AsymFLUX.2) — *in flight*. New: `klein_vae` config + Klein DiT + oklab color. Oracle: MAX `Flux2KleinPipeline`.
2. **SDXL** — *next*. New: **UNet** (ResBlock, down/mid/up, cross-attention) + 2× **clip** encoders + **ldm VAE**. First UNet (not DiT) — biggest new structural work; unlocks all SD-family. Oracle: diffusers `StableDiffusionXLPipeline`.
3. **Qwen-Image (+edit)** — MMDiT (60 double-stream blocks, 3-axis RoPE) + **qwen2.5-VL** encoder (⏳) + **qwenimage 3D VAE**. Oracle: MAX `QwenImagePipeline` (+ the OOM lesson: bf16-GPU oracle only).
4. **FLUX.1-dev** — `flux1_dit` + **t5** + clip. Oracle: MAX FLUX.1. (Shares flux arch w/ Klein → fast after Klein.)
5. **SD1.5 + SD3** — SD1.5 reuses SDXL UNet kit + clip; SD3 = MMDiT + t5+clip. Oracle: diffusers.

### Phase B — MAX-GAP image models (the real product value)
MAX will never ship these; oracle = the upstream repo (read-line-by-line) or diffusers if present.
6. **Chroma** (`chroma_dit` + t5) — flux-like, reuses Phase-A t5/VAE.
7. **HiDream-O1** (model + decoder + mRoPE + scheduler + prompt_agent) — complex, multi-encoder; the Rust port is done end-to-end so the arch is mapped.
8. **ERNIE-Image** (+ its sampler).
9. **Anima 2B** (= Cosmos-Predict2 image-only; keep independent, no cross-imports per project rule).
10. **Nucleus-Image 17B MoE** (3 dense + 29 MoE × 64 experts) — reuses MoE ops; memory/offload-heavy (BlockLoader).
11. **SenseNova-U1 8B MoT** — MoT routing layer over dense ops.

### Phase C — video (3D VAE, temporal attention, multi-step samplers)
New axis: temporal. conv3d done; need temporal attention + 3D patchify + video VAEs + UniPC/DMD samplers + `mux` (ffmpeg).
12. **Wan2.2 T2V/I2V** — wan VAE + wan22_dit + vace; **quant-allowed exception** (fp8/14B). Oracle: MAX Wan2.1/2.2.
13. **Cosmos-Predict2.5** (t2v/i2v/v2v) + **MagiHuman + SR** (UniPC) + **Helios 14B** (DMD/pyramid).
14. **LTX-2** — largest: model + latent/temporal upsamplers + conditioning + **audio VAE + vocoder** (video+audio). Multi-phase on its own.
15. **Lance i2v**, **Kandinsky5**.

### Phase D — audio + long-tail
16. **AceStep** (audio: dit + condition + vae + oobleck + sampler + STFT/mel). 
17. **Stable Cascade** (wuerstchen UNet + paella VQ + ddpm). 
18. **Hunyuan15**, **Motif**, **Lens**, **L2P** (≈ Z-Image variant, near-done).

## 3. New ops/kernels likely required (front-load per phase)
- **Phase A:** UNet primitives (ResBlock, GroupNorm-SiLU, cross-attention, up/downsample conv) — for SDXL. Most attention/conv exist; UNet is *composition* + a few up/down conv variants.
- **Phase C:** temporal attention (attn over frame axis), 3D patchify/unpatchify, video VAE decode (conv3d resblocks — conv3d done), `mux` via ffmpeg subprocess.
- **Phase D:** STFT/iSTFT + mel, vocoder, VQ codebook lookup (paella).
- Cross-cutting: **fp8 quant** (Wan exception only — not for Z-Image-class), GGUF reader (if any model needs it).

## 4. Per-model build template (use the `mojo-port` skill)
For each model, one chunked loop: (1) intake — read the diffusers/repo pipeline + the inference-flame `*.rs` line-by-line, write the conventions (sign/CFG/timestep/extract-layer/VAE-scale); (2) build encoder→DiT/UNet→VAE→sampler→pipeline glue in chunks; (3) skeptic each chunk; (4) bugfix; (5) **parity-gate** each component cos≥0.999 vs GPU-bf16 oracle, fresh single-sample refs; (6) end-to-end image/video/audio + a coherence eyeball. The skill (`/home/alex/mojodiffusion/.claude/skills/mojo-port/`) bakes in the gotchas.

## 5. Effort shape (rough, not commitments)
- A model that reuses an existing encoder+VAE+sampler ≈ **DiT/UNet + glue** (days). 
- A model needing a new encoder OR new VAE adds that component (encoder ~1-2 days, VAE ~1-2 days). 
- Video/audio models are multi-week (new VAE + temporal + sampler + mux/vocoder).
- Phase A: weeks. B: weeks. C: 1-2 months. D: weeks. Horizon matches the project's ~mid-2027.

## 6. Non-negotiable per-model checklist (lessons paid for in blood)
1. **Read the pipeline denoise loop for the sign convention.** Z-Image hid a post-CFG `noise_pred = -noise_pred`. Every framework/model may differ. Check: post-CFG negate? timestep `t` vs `1-t` vs `*1000`? CFG code-form vs textbook? scheduler step formula?
2. **Trust the numerical fingerprint over the source-read.** A per-step std/parity anomaly means a real bug even if your source-read says otherwise (the Z-Image 1% std gap was the negate, ignored for too long).
3. **Oracle = GPU bf16, single big model at a time.** Never fp32-CPU (cos 0.5/layer). Never fp32-host-load a full model (60GB OOM). Use `serenityflow-v2/.venv` or `max serve`.
4. **cos is magnitude-blind** — also check magnitude ratio + per-step trajectory for accumulating loops.
5. **Fresh single-sample refs** — a batched-CFG-hook velocity ≠ single-sample (false-bug trap).
6. **comptime seq lengths** for sdpa; pad to a supported length, slice back.
7. **F32 latent, bf16 model input** (diffusers convention); all file I/O via `io/ffi`.
8. **AOT works** via `mojo build -I . -Xlinker -lm` (libm sinf blocker resolved) — ship embeddable binaries; device-allocate big zero masks (`ctx.enqueue_memset`, not host List) for ≥1024².
9. **Reuse foundation ops + shared encoders/VAEs** — don't reimplement; check `serenitymojo/MAP.md` + `docs/SERENITYMOJO_MODULES.md`.
10. **MAX coverage:** Z-Image/Klein/Qwen/FLUX/Wan have MAX oracles (gradeable). Chroma/HiDream/MagiHuman/Helios/Nucleus/SenseNova/Anima/ERNIE are MAX-gap (= the real port value, oracle from upstream repo).
