# serenitymojo — Full inference-flame → Mojo Port Roadmap

**Goal:** port the entire `inference-flame` Rust inference stack (~40 models + shared VAEs/encoders/samplers) to **pure Mojo + MAX**, inference-only, GPU-only, as embeddable `prompt→image/video/audio` pipelines. Reference = `inference-flame/` (architecture) + diffusers/upstream (numerics oracle). Method = the `mojo-port` skill loop (builder→skeptic→bugfix→parity, cos≥0.999 vs GPU-bf16 oracle).

**Date:** 2026-05-26. **Companion:** memory `project_mojodiffusion_max_phase0`, `STATUS_ZIMAGE_DENOISE_DIVERGENCE.md`, `serenitymojo/MAP.md`.

**Scope update 2026-05-28:** Helios, Nucleus, and Stable Cascade are
intentionally removed from the active port scope. Older references below are
superseded by this note if they imply checkpoint audits, downloads, or bring-up
work for those families.

---

## 0. Current state (the baseline this builds on)

**DONE / verified:** foundation — `tensor`, `parity` (ParityHarness), all `ops/` (linear, norm{rms/layer/group}, rope{interleaved/halfsplit}, attention/sdpa[math-mode Dh=128], conv2d, **conv3d**, **moe**{router/grouped-FFN/scatter}, embeddings, layout{patchify/unpatchify/deinterleave}, tensor_algebra, activations, softmax); `io/` loader (byte-parity); `offload/BlockLoader`; `tokenizer` (Qwen BPE); `image/png`; samplers `flow_match` + `flux2_klein` + `sdxl_euler`; VAE kit `decoder2d`+`conv3d`+`upsample`+`vae_ops`.
- **Z-Image** — WORKING end-to-end (1024², coherent images). Encoder `qwen3`, `zimage_nextdit`, VAE `zimage_decoder`, flow-match.
- **Klein 4B/9B** — coherent block-streamed inference/image smokes exist.
  These are not OneTrainer `CPU_OFFLOADED` activation/layer parity, train
  backward replay, sampler trajectory parity, speed parity, or GPU parity.
  Remaining work is production wrapper/parity/perf polish.
- **SDXL** — cached-embedding one-step and full 30-step 1024 runtime smokes
  exist: `output/sdxl_one_step_1024.png` and `output/sdxl_30step_1024.png`
  from cache -> UNet -> VAE -> PNG. Remaining work is Rust/diffusers parity
  and raw prompt encoder assembly.
- **Lance 3B + Wan2.2** — 256x256/9-frame dense decoded runtime artifact
  exists: `output/lance_t2v_256_9f_dense.mp4` plus 9 PNG frames. Remaining
  production work is variable-length cached CFG/KV execution, sparse attention,
  and multi-step/larger-frame quality targets.
- **FLUX.1-dev** — cached Rust-input path now runs 20-step DiT -> VAE -> PNG
  at `output/flux1_cached_inputs.png`; raw CLIP/T5 prompt assembly remains.
- **ERNIE-Image** — manifest/checkpoint contract plus fixed-shift FlowMatch
  scheduler/tensor smoke are registered against `/home/alex/models/ERNIE-Image`;
  the tensor gate covers CFG and Euler update math. A resident DiT math smoke
  now runs real patch projection, timestep MLP, and text projection weights;
  block0 now has a bounded real-weight RoPE/attention/MLP smoke. Full Mistral3B
  encoder and the full ERNIE block stack remain.
- **Qwen-Image + Qwen-Image-Edit** — base checkpoint now has a 512 runtime
  smoke: tokenizer -> Qwen2.5-VL -> streamed 60-block DiT paired-CFG -> Qwen
  VAE -> PNG at `output/qwenimage_first_512.png`. Edit now has a synthetic
  512 target+reference streamed DiT/VAE smoke at
  `output/qwenimage_edit_synth_512.png`; real edit prompt/VAE-encode parity
  remains.
- **Chroma + SD1.5 + SD3** — all are registered with metadata/header
  contracts. SD1.5 has a real VAE runtime smoke at
  `output/sd15_vae_noise_512.png` plus an SD1.5 Euler scheduler smoke. SD3
  now has Large plus local "small" SD3.5 Medium manifests, tensor FlowMatch,
  header contracts, a real-weight MMDiT resident pre/post-block gate, and
  embedded-checkpoint VAE runtime smokes, writing
  `output/sd3_vae_noise_1024.png` and
  `output/sd3_medium_vae_noise_1024.png`;
  Chroma now has a real-weight distilled-guidance step-cache plus a staged
  DiT smoke (`pipeline/chroma_dit_smoke.mojo`) that builds `[1,344,3072]`
  pooled modulation, RoPE, two double blocks, first two single blocks, and final
  image projection from the BF16 checkpoint. Chroma still needs full
  denoise/pipeline math, SD1.5 still needs
  the runtime UNet
  wrapper/prompt path, and SD3 still needs joint-block MMDiT/triple-encoder
  assembly.
- `qwen25vl_encoder` is exercised by the Qwen-Image base 512 runtime smoke;
  text parity and edit conditioning remain open.

**Hard-won conventions to carry to EVERY model** (see §6): GPU-bf16 oracle only (never fp32-CPU/host-load); read the diffusers/repo **pipeline denoise loop line-by-line for sign/CFG/timestep** (the Z-Image bug was a missed post-CFG `noise_pred = -noise_pred`); **trust the numerical fingerprint over the source-read** when they disagree; comptime sequence lengths for sdpa; F32 latent + bf16 model input; all file I/O through `io/ffi`.

## 1. Reuse map (why families port together)

Most cost is shared infra, not the DiT. Build each shared piece ONCE, reuse across the family:

| Shared piece | Built? | Reused by |
|---|---|---|
| qwen3 encoder | ✅ | Z-Image, Klein, (Qwen text path) |
| qwen2.5-VL encoder | ✅ base runtime smoke / parity pending | Qwen-Image (+edit) |
| **clip encoder** | ⏳ module/cached paths | SDXL, FLUX.1, SD1.5, SD3, Kandinsky5 |
| **t5 / t5gemma2** | ⏳ modules/cached paths | SD3, Chroma, FLUX.1, ERNIE, LTX-2 |
| gemma3 / mistral / gpt_oss | ❌ | HiDream-O1, Lumina-ish, misc |
| 2D-VAE kit (decoder2d) | ✅ | Z-Image, Klein, SDXL/SD (ldm), Chroma |
| **ldm VAE (dec)** | ✅ SDXL/SD1.5/SD3 decode paths | SDXL, SD1.5, SD3 |
| qwenimage VAE | ✅ 512 decode/runtime smoke | Qwen-Image |
| wan21/wan22 3D VAE | ⏳ conv3d done | Wan2.2, Cosmos2.5, video |
| MoE/MoT ops | ✅ | SenseNova (MoT), future active MoE family if re-added |
| flow-match euler | ✅ | most DiTs |
| UniPC sampler | ❌ | MagiHuman, Cosmos |
| DMD / pyramid | ❌ | none in active scope |
| audio VAE + STFT/vocoder | ❌ | AceStep, LTX-2 audio |
| **UNet blocks (ResBlock+cross-attn+up/down)** | ✅ SDXL one-step/full cached path / generalized kit partial | SD1.5/SDXL/SD3-partial |

## 2. Phased roadmap (order = reuse leverage × oracle availability × value)

### Phase A — headline image DiTs, MAX-oracle-backed *(in progress)*
Gradeable against `max serve` AND diffusers. Heavy reuse of the Z-Image foundation.
1. **Klein 4B/9B** (AsymFLUX.2) — *in flight*. New: `klein_vae` config + Klein DiT + oklab color. Oracle: MAX `Flux2KleinPipeline`.
2. **SDXL** — *one-step and full 30-step runtime smokes landed*. New: **UNet**
   (ResBlock, down/mid/up, cross-attention) + **ldm VAE** cached-embedding path.
   Still missing raw-prompt 2x **clip** encoder assembly and Rust/diffusers
   parity. Oracle: diffusers `StableDiffusionXLPipeline`.
3. **Qwen-Image (+edit)** — MMDiT (60 double-stream blocks, 3-axis RoPE) + **qwen2.5-VL** encoder runtime smoke + **qwenimage 3D VAE**. Oracle: MAX `QwenImagePipeline` (+ the OOM lesson: bf16-GPU oracle only).
4. **FLUX.1-dev** — `flux1_dit` + **t5** + clip. Cached-input runtime smoke
   now proves DiT -> VAE -> PNG with Rust-captured conditioning; raw-prompt
   CLIP/T5 assembly and parity remain. Oracle: MAX FLUX.1.
5. **SD1.5 + SD3** — SD1.5 has scheduler + VAE gates and reuses the SDXL
   UNet kit + clip; SD3 has Large/Medium scheduler + embedded VAE gates plus
   resident MMDiT pre/post-block math, and still needs joint blocks + t5+clip.
   Oracle: diffusers.

### Phase B — MAX-GAP image models (the real product value)
MAX will never ship these; oracle = the upstream repo (read-line-by-line) or diffusers if present.
6. **Chroma** (`chroma_dit` + t5) — flux-like, reuses Phase-A t5/VAE.
   Real-weight distilled-guidance step-cache plus staged double0-1/single0/proj
   smoke landed; full 19+38 denoise and staged image pipeline remain.
7. **HiDream-O1** (model + decoder + mRoPE + scheduler + prompt_agent) — complex, multi-encoder; the Rust port is done end-to-end so the arch is mapped.
8. **ERNIE-Image** (+ its sampler) — manifest/header plus scheduler/tensor
   CFG/Euler contract started; resident DiT patch/text/time projection and
   bounded block0 attention/MLP smokes landed. Full Mistral3B encoder and
   full ERNIE block stack still open.
9. **Anima 2B** (= Cosmos-Predict2 image-only; keep independent, no
   cross-imports per project rule). Cached-conditioning tensor smoke now loads
   the Rust sidecar and latent oracle into Mojo tensors, validates Anima's
   linear CFG/Euler runtime surface, and the Wan/Qwen VAE path decodes the Rust
   latent oracle to `output/anima_vae_from_rust_latent_1024.png`;
   adapter/MiniTrainDIT math remains.
10. **SenseNova-U1 8B MoT** — MoT routing layer over dense ops.

### Phase C — video (3D VAE, temporal attention, multi-step samplers)
New axis: temporal. conv3d done; need temporal attention + 3D patchify + video VAEs + UniPC/DMD samplers + `mux` (ffmpeg).
12. **Wan2.2 T2V/I2V** — wan VAE + wan22_dit + vace; **quant-allowed exception** (fp8/14B). Oracle: MAX Wan2.1/2.2.
13. **Cosmos-Predict2.5** (t2v/i2v/v2v) + **MagiHuman + SR** (UniPC).
14. **LTX-2** — largest: model + latent/temporal upsamplers + conditioning + **audio VAE + vocoder** (video+audio). Multi-phase on its own.
15. **Lance i2v**, **Kandinsky5**.

### Phase D — audio + long-tail
16. **AceStep** (audio: dit + condition + vae + oobleck + sampler + STFT/mel). 
17. **Hunyuan15**, **Motif**.
18. **Microsoft Lens** and **Z-Image L2P** stay visible as active P2 sidecar
    lanes; Lens now has scalar/BF16 FlowMatch gates plus a real-weight sampled
    block0 image-QKV gate and sampled image Q/K RoPE parity, and Z-Image L2P
    now has a 1024 BF16 pixel patchify16 roundtrip smoke, real-weight DiT
    pre-block embeddings, and a full tiny local-decoder gate for its VAE-less
    data path. See the 2026-05-28 handoff docs before full transformer-block
    model math.

## 3. New ops/kernels likely required (front-load per phase)
- **Phase A:** UNet primitives (ResBlock, GroupNorm-SiLU, cross-attention, up/downsample conv) — for SDXL. Most attention/conv exist; UNet is *composition* + a few up/down conv variants.
- **Phase C:** temporal attention (attn over frame axis), 3D patchify/unpatchify, video VAE decode (conv3d resblocks — conv3d done), `mux` via ffmpeg subprocess.
- **Phase D:** STFT/iSTFT + mel and vocoder. No VQ lookup work is active unless
  explicitly requested.
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
10. **MAX coverage:** Z-Image/Klein/Qwen/FLUX/Wan have MAX oracles (gradeable). Chroma/HiDream/MagiHuman/SenseNova/Anima/ERNIE are MAX-gap (= the real port value, oracle from upstream repo).
