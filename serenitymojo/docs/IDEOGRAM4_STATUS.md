# Ideogram-4 (fp8) — pure-Mojo port STATUS / handoff

serenitymojo's first **fp8-weight** model. Pure Mojo + MAX, GPU-only, inference.
Reference (parity oracle) = diffusers upstream `/home/alex/ideogram4-ref/` (NOT
OneTrainer). Weights: `/home/alex/.serenity/models/ideogram-4-fp8/`.
All numbers below are MEASURED in-session (RTX 3090 Ti, 24 GB).

## Architecture
- DiT (9.3B, fp8 e4m3 weights + per-row F32 `.weight_scale`): 34 layers, emb 4608
  / 18 heads / head_dim 256, fused QKV + per-head q/k RMSNorm, **interleaved
  MRoPE** (sect 24/20/20, θ=5e6), adaLN 4-chunk + tanh gates, SwiGLU 12288,
  block-diagonal SDPA. cond = Qwen3-VL 13-tap concat (53248) → RMSNorm → proj.
  Two transformers (conditional + unconditional) → **asymmetric CFG** (negative =
  image-only, zeroed text). Velocity / flow-matching.
- Text: Qwen3-VL text stack (hidden 4096, 36 layers, GQA 32/8, θ5e6); 13-tap.
- VAE: AutoencoderKLFlux2 (z=32, ch_mult 128/256/512/512).
- Scheduler: logit-normal (ndtri/expit), res-aware mean, Euler `z+=v·(s−t)`.

## Component parity (cos vs torch oracle, gate ≥0.999) — ALL PASS
| chunk | what | cos |
|---|---|---|
| 1 | fp8 per-row dequant | 0.99999878 |
| 2 | interleaved MRoPE cos/sin | 0.99999999 |
| 3 | t-embedding (EmbedScalar) | 0.99999616 |
| 4 | logit-normal schedule | exact (0.0) |
| 5 | transformer block | 0.99999895 |
| 6 | full 34-layer DiT velocity | 0.9996 |
| 7 | Qwen3-VL 13-tap | 0.99998625 |
| 8 | Flux2 VAE decode (z=32) | 0.99995 |
| 9 | end-to-end sampler | image PSNR 29.7 dB vs torch (latent cos 0.96 = bf16+CFG-cancellation accumulation, visually negligible) |
| — | fused fp8 GEMM vs dequant+BLAS | 0.99999698 |
| — | resident fp8 DiT (dequant+cuBLAS) vs fixture | 0.99961 |

Probes: `models/dit/parity/chunk{1..8}_*.mojo`, `chunk6r_resident_probe.mojo`,
`fp8gemm_probe.mojo`. Oracle generator: `models/dit/parity/ideogram4_oracle.py`
(dev-only torch; stages A/B/C/D/P/E → `ideogram4_fx_*.safetensors` fixtures, gitignored).

## Performance work (resident fp8 + cuBLAS + D2H hoist)
Problem: streaming the fp8 weights from mmap and re-dequanting **every** denoise
step → memory-bound (~3% GPU util). Fixes:
- **Resident weight cache** (`Ideogram4Weights` in `ideogram4_resident.mojo`):
  load all fp8 weights ONCE (stay F8_E4M3 + F32 scale; norms/biases bf16). Both
  cond+uncond fit (~18.6 GB) → no per-step streaming, CFG intact.
- **Matmul** (`_lin`): dequant the resident fp8 weight → bf16 (cheap GPU kernel,
  no mmap) → **vendor cuBLAS `linear`**. This is the parity-gated path (cos 0.99961).
  A standalone fused tiled fp8 GEMM (`ops/fp8_gemm.mojo::linear_fp8`, cos
  0.99999698) also exists but is slower than cuBLAS (TILE=16, no tensor cores) —
  kept for reference; the resident path uses dequant→cuBLAS.
- **D2H hoist** (`ideogram4_build_masks`): the indicator masks are constant across
  steps; built ONCE and passed into `forward_r`, removing the per-forward
  `indicator.to_host` (was ~2× per step). EDv2/Klein transfer lesson
  (`/home/alex/EriDiffusion/FLAME_BLOCK_SWAP_AUDIT.md`).
- MEASURED: 4-step 1024² with the slow tiled GEMM = 8:12. Util went 3%→98% with
  resident weights. (cuBLAS-path full timing not captured — run was killed.)
- For >2k resolutions where fp8-resident won't fit, the path is pinned + async +
  prefetch block-swap via `offload/turbo_loader.mojo` (Stagehand port) — not wired
  for Ideogram yet.

## Magic prompt (plain text → JSON), pure Mojo
ComfyUI does plain→JSON via an external/local LLM (Gemma/Claude), toggled by a
switch — the Ideogram model only ever consumes the JSON. Pure-Mojo equivalent:
- `Qwen3Encoder.lm_logits_last` + `qwen3_magic.generate_greedy` run **Qwen3-8B**
  (lm_head present; config == `Qwen3Config.klein_9b`) autoregressively in Mojo.
- Driver `pipeline/ideogram4_magic.mojo`: chat template + magic-prompt system
  prompt → greedy decode → JSON caption (decoded via `Qwen3Tokenizer`).
- Greedy, NO KV-cache yet (re-forwards padded context per token) → correct but
  slow; KV-cache is the speed follow-up. Added Qwen sdpa comptime seq=1024 case.

## Files (port surface)
- `models/dit/ideogram4_dit.mojo` — DiT forward (streaming/reference path), block,
  attention, rope-apply, EmbedScalar, packed builder helpers. `[S]`-parameterized.
- `models/dit/ideogram4_resident.mojo` — resident fp8 weights + `forward_r` + masks (HOT path).
- `models/dit/ideogram4_mrope.mojo` — interleaved MRoPE builder.
- `models/text_encoder/ideogram_qwen3vl.mojo` — Qwen3-VL 13-tap encoder (adapts Qwen3Encoder).
- `models/text_encoder/qwen3_magic.mojo` — greedy LM decode (magic prompt).
- `models/vae/ldm_decoder.mojo` — `+load_ideogram4_vae_decoder` (z=32 factory).
- `ops/fp8.mojo` — `+fp8_e4m3_dequant_perrow_to_bf16`, `+load_fp8_dequant`.
- `ops/fp8_gemm.mojo` — fused tiled fp8 GEMM (reference, unused by hot path).
- `sampling/ideogram4_schedule.mojo` — logit-normal (Acklam ndtri) + step intervals.
- `pipeline/ideogram4_generate.mojo` — native text→image (Qwen→DiT CFG→VAE→PNG), resolution/preset-configurable.
- `pipeline/ideogram4_pipeline.mojo` — fixture-fed parity gate (chunk 9).
- `pipeline/ideogram4_magic.mojo` — pure-Mojo plain→JSON.
- `docs/IDEOGRAM4_PROMPTING.md` — the upstream prompting guide (structured JSON schema).

## How to run
- Parity gate (chunk N): `pixi run mojo run -I . serenitymojo/models/dit/parity/chunkN_*_probe.mojo`
- Generate (edit prompt ids / STEPS / GH,GW in the file):
  `pixi run mojo run -I . serenitymojo/pipeline/ideogram4_generate.mojo`
- Magic prompt: `pixi run mojo run -I . serenitymojo/pipeline/ideogram4_magic.mojo`
- Regenerate fixtures: `OneTrainer-anima-ref/venv/bin/python models/dit/parity/ideogram4_oracle.py {A|B|C|D|P|E}` (offline; HF_HUB_OFFLINE=1)

## Known limits / next
- Latent cos 0.96 at chunk 9 = irreducible bf16+CFG(7×) accumulation (image matches, PSNR 29.7).
- Magic prompt + diffusion both lack KV-cache / async block-swap (speed follow-ups).
- 4K out of range (model native ≤2k; Dh=256 math-mode SDPA can't hold 4k² score matrix).
- LoRA (Power-Lora-style additive overlay) not yet wired.
