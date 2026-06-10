# FLUX.1-dev pure-Mojo — numeric parity verification (2026-06-09)

**Repo:** `/home/alex/mojodiffusion` · **Branch:** `training-port-5models-lora`
**Hardware:** RTX 3090 Ti 24GB · **Follows:** `HANDOFF_FLUX_2026-06-09.md` (item #1: quality parity)

---

## TL;DR

The handoff delivered a real coherent FLUX image but flagged that it was **NOT**
verified for **numeric parity** vs a reference. This pass closes that gap: every
stage of the pure-Mojo FLUX.1-dev pipeline is now measured against its reference
implementation on byte-identical inputs. **All six gates pass.**

| Gate | Stage | Reference | Metric | Result | Bar |
|------|-------|-----------|--------|--------|-----|
| A  | VAE decoder (per-tile) | torch BFL-AE, real `ae.safetensors` | PSNR | **88.7 dB** | >40 |
| B  | DiT forward (full 19+38 block stack, real weights) | BFL `Flux`, real `flux1-dev.safetensors` | cosine | **0.99942** | ≥0.99 |
| B2 | Sigma schedule (20 steps) | BFL `get_schedule` | max abs diff | **0.0 (bit-exact)** | <1e-6 |
| D-T5 | T5-XXL encoder | HF `T5EncoderModel`, real `t5xxl_fp16` | cosine | **0.99920** (no NaN) | ≥0.99 |
| D-CLIP | CLIP-L pooled | HF `CLIPTextModel`, real `clip_l` | cosine | **0.99999** | ≥0.99 |
| C  | Full 20-step denoise integration | BFL `Flux` × 20 Euler | cosine (final latent) | **0.99969** | ≥0.99 |
| Seam | Tiled VAE decode (3×3 overlap+blend) vs seamless full-1024² | torch BFL-AE | PSNR / worst-row | **58.6 dB**, worst-row MAD 0.0045 (no seam spike) | report |
| B-LoRA | DiT forward + Kohya-BFL LoRA overlay | BFL `Flux` + applied delta | cosine | **0.99937** | ≥0.99 |

Conclusion: the Mojo FLUX pipeline is **numerically faithful** to BFL+HF at every
stage. The DiT and VAE match at exact-level (>0.999 / 88 dB); the T5 fix is
confirmed (no NaN, cos 0.999); the schedule is bit-identical; and the composed
20-step denoise matches at cos 0.9997 (per-step error damps, not compounds —
flow-match Euler is contractive toward the data manifold).

---

## Method (per the numeric-parity-testing discipline)

For each stage: a torch/HF oracle runs the **reference's own runtime** on FIXED
inputs and dumps inputs + output as raw-F32 (and int32 ids) `.bin`. The Mojo
probe reads the **byte-identical** inputs, runs the Mojo code, and compares with
the metric appropriate to the stage (PSNR for decoded pixels, cosine for
forwards, exact for the deterministic schedule). Numbers were measured in this
session — not asserted.

**Dtype faithfulness.** The Mojo stores bf16 weights with F32 accumulation, so
each oracle loads the real weights and casts **fp16 → bf16 → fp32** (bf16 ⊂ fp32,
lossless) — i.e. the *Mojo's exact weights* with clean F32 accumulation. The DiT
oracle runs on CPU (the 24 GB bf16 weights won't fit a 24 GB card alongside a
CUDA context); a **tiny 4×4 token grid** (N_IMG=16, N_TXT=16) exercises the FULL
57-block stack with REAL weights while staying fast and OOM-free — the DiT is
sequence-length agnostic.

**Convention match.** Timestep/guidance are passed RAW to BFL (whose
`timestep_embedding` applies the ×1000 `time_factor` internally) and pre-scaled
×1000 by the Mojo caller (whose embedder uses factor 1). RoPE is built
independently on each side from the identical BFL `EmbedND` convention
(axes [16,56,56], θ=10000, txt-ids=(0,0,0), img-ids=(0,row,col), txt-first
concat). T5 runs unmasked on both sides (BFL passes `attention_mask=None`). CLIP
pooled = post-LN hidden at the first-EOS position, no projection (= HF
`pooler_output`); causal attention makes it invariant to post-EOS padding.

**RNG caveat (why there is no single end-to-end image diff).** A real "same
seed" image comparison is impossible: the Mojo custom `randn` and torch
generators produce different noise streams. Gate C therefore **pins** the initial
noise + embeds and feeds byte-identical bytes to both sides — it measures the
denoise math (DiT × 20 + schedule + Euler), not RNG. This is the correct
treatment for an RNG-dependent pipeline.

---

## Files (all under `serenitymojo/`)

VAE decode: `vae/parity/flux_vae_decode_oracle.py` + `flux_vae_decode_parity.mojo`
DiT / schedule / encoders / denoise: `models/flux/parity/`
  `flux_dit_oracle.py` + `flux_dit_parity.mojo`
  `flux_sched_oracle.py` + `flux_sched_parity.mojo`
  `flux_t5_oracle.py` + `flux_t5_parity.mojo`
  `flux_clip_oracle.py` + `flux_clip_parity.mojo`
  `flux_denoise_oracle.py` + `flux_denoise_parity.mojo`

## Reproduce (oracle FIRST, then build+run the Mojo gate — never chained)

```bash
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
# Gate A
python3 serenitymojo/vae/parity/flux_vae_decode_oracle.py 64 64
pixi run mojo run -I . serenitymojo/vae/parity/flux_vae_decode_parity.mojo
# Gate B (CPU oracle ~1-2 min: 48GB fp32 load)
python3 serenitymojo/models/flux/parity/flux_dit_oracle.py 4 4 16
pixi run mojo run -I . -Xlinker -lcuda serenitymojo/models/flux/parity/flux_dit_parity.mojo
# Gate D-T5 / D-CLIP
python3 serenitymojo/models/flux/parity/flux_t5_oracle.py
pixi run mojo run -I . -Xlinker -lcuda serenitymojo/models/flux/parity/flux_t5_parity.mojo
python3 serenitymojo/models/flux/parity/flux_clip_oracle.py
pixi run mojo run -I . -Xlinker -lcuda serenitymojo/models/flux/parity/flux_clip_parity.mojo
# Gate C (CPU oracle ~1-2 min)
python3 serenitymojo/models/flux/parity/flux_denoise_oracle.py 4 4 16 20
pixi run mojo run -I . -Xlinker -lcuda serenitymojo/models/flux/parity/flux_denoise_parity.mojo
```

---

## Follow-up handoff items — DONE (2026-06-09, second pass)

- **Tile-seam A/B** ✅ — `flux_tiled_decode` extracted to a shared module (CLI +
  gate run ONE code path); tiled 1024² vs seamless full-1024² torch decode =
  **58.6 dB**, worst-row MAD 0.0045 (~1.5× global, not a seam). Effectively
  seamless. Gate: `serenitymojo/vae/parity/flux_tiled_decode_parity.mojo`.
- **JSON-driven sampler params** ✅ — `flux_sample_cli` now honors steps /
  guidance / seed from the prompt JSON at runtime (defaults when unset);
  width/height stay comptime and a mismatched size is rejected fail-loud.
  Verified: a JSON with steps=3/cfg=2.0/seed=123 ran 3 steps with those values.
- **FLUX LoRA at inference** ✅ — runtime ADDITIVE overlay (never fused):
  `flux_lora_overlay.mojo` + `Flux1Offloaded.load_with_lora`. Kohya/sd-scripts
  BFL format (`lora_unet_double_blocks_*`); 304 targets loaded; A/B image diff
  14.9/255 mean (90% of pixels changed); and **numerically verified**: DiT
  forward + overlay vs torch BFL+LoRA = **cos 0.99937** (gate B-LoRA). Diffusers-
  format LoRAs are rejected fail-loud (not yet mapped).

## NOT covered / honest scope

1. **Tokenizer parity** — the gates feed identical *ids* to isolate the
   encoders; CLIP/T5 tokenizers are claimed bit-exact elsewhere, not re-checked
   here.
2. **Full-resolution (4096-token) DiT** — verified at a tiny grid (the DiT is
   resolution-agnostic, so this is sound) but not at the production 1024² grid.
3. **Diffusers-format FLUX LoRA** — the overlay supports the Kohya BFL key
   scheme; diffusers `transformer.*.lora_A/lora_B` (q/k/v split + remap) is
   rejected fail-loud, not yet supported.
4. **LoRA multiplier** — fixed at 1.0 in the CLI (the overlay takes a multiplier
   arg; not yet surfaced as a CLI/JSON field).
