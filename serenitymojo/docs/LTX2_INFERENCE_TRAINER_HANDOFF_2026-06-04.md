# LTX-2.3 22B AV — Inference + Trainer Handoff (2026-06-04)

Authoritative session handoff. **Inference is anchored to LTX-Desktop / ltx_core
(the official app + its pip libs); training is anchored to musubi-tuner.** Every
"DONE/works" claim below is tagged MEASURED (a tool result in-session) or
HYPOTHESIS (reasoned, not yet verified). Read §0 first.

Supersedes nothing; complements `LTX2_AV_HANDOFF_2026-06-04.md` (that doc is the
block-0 parity story for the NEW block #3; this doc is the end-to-end
inference + trainer story and corrects which forward is the spine).

---

## 0. TL;DR (the state in 8 lines)

1. **Inference RUNS end-to-end** (MEASURED): `pipeline/ltx2_t2v_av_mvp.mojo` and
   `…_hq.mojo` produce MP4+WAV from the FP8 distilled 22B checkpoint.
2. **Output is NOT yet good** (MEASURED by eye): the *scene* is coherent but the
   **face is distorted/melted** at both 256² (MVP) and 768×512 (HQ, LoRA OFF).
3. **Speed is bad** (MEASURED): MVP ~6 min / HQ ~14–16 min on a 24 GB 3090 Ti.
   LTX should be seconds. **Root cause is the streamer design, NOT disk** (§5).
4. **Spine = `models/dit/ltx2_dit.mojo` (#2)** — the forward the working
   inference pipelines already use. NOT the handoff's `ltx2_av_block.mojo` (#3).
5. **Two independent problems**: (A) SPEED = FP8 re-streaming design; (B)
   QUALITY = distorted face. Do not conflate.
6. **Trainer**: exists but wired to the WRONG (video-only) arch and has no AV
   backward. Real work remaining. Anchor to musubi (§7).
7. **References on disk**: inference → `/home/alex/LTX-Desktop` +
   `/home/alex/LTX-2-official`; training → `/home/alex/musubi-tuner` (§2).
8. **An HQ+LoRA run was in flight at handoff time** — view its frames to decide
   whether the missing LoRA was the face cause (§4, §6 step 1).

---

## 1. Hardware / environment (MEASURED this session)

- GPU: **NVIDIA RTX 3090 Ti, 24564 MiB (24 GB)**. Desktop apps (Xorg/gnome/totem)
  hold ~1.2 GB; ~23 GB usable. Driver 580.126.09, CUDA 13.0.
- Host RAM: **62 GB** total. During HQ run: process RSS ~17.8 GB, `buff/cache`
  **40 GB** (→ the 28 GB FP8 ckpt is fully page-cache-resident; see §5).
- Toolchain: `pixi run mojo …` (Mojo 1.0.0b1 dialect). Build/run LTX pipelines
  with `-Xlinker -lm -Xlinker -lcuda` (without `-lm` the link fails on `cos`).
- musubi venv: `/home/alex/musubi-tuner/venv/bin/python` — torch 2.9.1+cu128,
  CUDA available. Used for oracle generation.

---

## 2. Reference sources (LOCAL paths — use these, not the web)

### Inference reference = LTX-Desktop + ltx_core/ltx_pipelines
- **App (Electron orchestration, cloned this session):** `/home/alex/LTX-Desktop`
  (from `github.com/Lightricks/LTX-Desktop`). The real math is NOT vendored here;
  it imports the pip libs `ltx_core` / `ltx_pipelines`.
  - A2V orchestration (our exact use case): `backend/services/a2v_pipeline/`
    → `ltx_a2v_pipeline.py`, `distilled_a2v_pipeline.py`, `a2v_pipeline.py`.
  - Shared recipe glue: `backend/services/ltx_pipeline_common.py` (imports
    `DISTILLED_SIGMA_VALUES`, `SimpleDenoiser`, `GaussianNoiser`,
    `MultiModalGuiderParams`, VAE `TilingConfig`).
- **Actual model + pipeline source (the ground truth for inference math):**
  `/home/alex/LTX-2-official/packages/`
  - Pipelines: `ltx-pipelines/src/ltx_pipelines/`
    - `distilled.py` — the distilled denoise pipeline (our path).
    - `ti2vid_two_stages.py` — the **2-stage** (base → spatial upsampler) render.
    - `ti2vid_one_stage.py` — single-stage.
    - `utils/constants.py` — `DISTILLED_SIGMA_VALUES`, default resolutions.
    - `utils/denoisers.py` (`SimpleDenoiser`), `utils/args.py`, `utils/helpers.py`,
      `utils/blocks.py`.
  - Core model: `ltx-core/src/ltx_core/`
    - `model/transformer/` — transformer_args.py, the DiT (matches musubi).
    - `model/video_vae/`, `model/audio_vae/` (audio_vae.py, vocoder.py),
      `model/upsampler/` (model.py, pixel_shuffle.py, spatial_rational_resampler.py)
      ← **the stage-2 spatial upsampler serenitymojo has NOT built**.
    - `components/noisers.py` (GaussianNoiser), `components/guiders`.
  - Other copies exist (`/home/alex/LTX-2/packages/…`,
    `/home/alex/Wan2GP/models/ltx2/{ltx_core,ltx_pipelines}`); LTX-2-official is
    the one to trust.

### Training reference = musubi-tuner
- `/home/alex/musubi-tuner/src/musubi_tuner/`
  - Entrypoints: `ltx2_train.py`, `ltx2_train_network.py` (LoRA), `ltx2_train_slider.py`,
    `ltx2_merge_lora.py`.
  - LoRA network + targets: `networks/lora_ltx2.py` (LTX2 include patterns; audio
    keys are optional/filterable — `_filter_audio_keys`).
  - Model: `ltx_2/model/transformer/transformer.py` (`BasicAVTransformerBlock`,
    `_forward` ~466, `get_ada_values` ~195, `_apply_text_cross_attention` ~267,
    `apply_cross_attention_adaln` ~853), `…/attention.py`, `…/rope.py`
    (`LTXRopeType.INTERLEAVED`), `…/modality.py` (`sigma: Tensor # Shape: (B,)`),
    `…/transformer_args.py` (preprocessor that builds AdaLN/RoPE/prompt inputs).
  - LoRA→Comfy conversion: `ltx_2/convert_lora_to_comfy.py`.

### Checkpoints / data on disk (all MEASURED present)
- FP8 distilled DiT (streamed by inference): `/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors` (**28 GB**).
- BF16 distilled DiT (full precision, untested here): `…/ltx-2.3-22b-distilled.safetensors` (**44 GB**).
- DEV (non-distilled) DiT: `…/ltx-2.3-22b-dev.safetensors` (46 GB; oracle weights, block-0 only).
- Distilled LoRA: `…/ltx-2.3-22b-distilled-lora-384.safetensors` (**7 GB**), 1660 mappings, format `LTX2Distilled`.
- Genuine Gemma-3 context dump (text/video/audio context, the inference conditioning):
  `/home/alex/EriDiffusion/inference-flame/output/audio_context_dump/ltx2_audio_context.safetensors` (25 MB; keys `video_context [1,1024,4096]`, `audio_context [1,1024,2048]`).
- Cached embeddings: `/home/alex/EriDiffusion/inference-flame/cached_ltx2_embeddings.safetensors` (9 MB).
- Video VAE: `/home/alex/.serenity/models/vaes/LTX2/LTX2_video_vae_old_bf16.safetensors`.

---

## 3. The forward landscape — which file is the spine (MEASURED)

There are **four** LTX2 forward/transformer code paths in serenitymojo:

| # | File | Arch | Status (MEASURED) |
|---|------|------|--------|
| 1 | `models/ltx2/ltx2_block.mojo` | video-only attn1+ff | OLD/dead. The 2026-06-04 AV handoff's "old block". DELETE. |
| 2 | `models/dit/ltx2_dit.mojo` `ltx2_block_forward_av` | **full AV** | **INFERENCE SPINE.** mvp/hq pipelines run on it and produce coherent scenes (face still bad). Block-0 cos never measured. |
| 3 | `models/ltx2/ltx2_av_block.mojo` | full AV | Handoff's parity block. Reproduced cos: **video 0.9993907 / audio 0.9961671** (max_abs 1.72 / 2.70) vs adapted oracle. Wired to NOTHING. Built unaware #2 existed. |
| 4 | `models/ltx2/ltx2_stack_lora.mojo` (`ltx2_stack_lora_forward/backward_offload`) | **video-only attn1+ff** | The TRAINER's fwd+bwd. Has LoRA(q,k,v,out ×48)+adamw+offload+data loop, but WRONG arch (no cross-attn/audio/cross-modal). |

**Decision (maintainer: "I want the one that works; the diff is your problem"):**
spine = **#2**. Inference already works on it. Build the trainer's AV backward on
#2's forward (or a forward numerically identical to it) so training math ==
inference math. Retire #1 and #3.

Note `ltx2_dit.ltx2_block_forward_av` signature (the seam, MEASURED):
`[S_V,S_A,N_TXT,S_VPAD,S_APAD](weights:LTX2AVBlockWeights, hidden, ahs, enc, aenc,
v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, v_prompt_ts, a_prompt_ts,
v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, eps, ctx)
-> Tuple[Tensor,Tensor,Tensor]`. It has **zero backward** (grep `backward` = 0).

### Settled correctness question (MEASURED — resolves AV-handoff §5a.3)
musubi `…/modality.py:22` documents `sigma: torch.Tensor # Shape: (B,)`.
`prompt_timestep = _prepare_timestep(sigma, …)` → `[B,1,dim]` → broadcast constant
across text tokens in `apply_cross_attention_adaln`. So the **real model's
cross-attn prompt modulation is PER-BATCH**, not per-token. The Mojo per-batch
`v_prompt_ts` is exact; the oracle script's `rnd(B,N_TXT,…)` was an
over-generalization. No missing feature in #3 (or #2).

---

## 4. What was MEASURED this session

### Inference (the runs)
- **MVP** (`/tmp/ltx2_mvp` ← `pipeline/ltx2_t2v_av_mvp.mojo`): 256×256, 16f,
  S_V=128 S_A=16 N_TXT=128, 48 blocks, **8-step distilled** (sigmas
  `1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0`), **LoRA OFF**.
  Ran end-to-end in **~6 min** (~28 s/step). All per-step v_vel/a_vel/x finite,
  std ~1.0–1.4. Output: `output/ltx2_mvp/{ltx2_t2v_av_256_16f.mp4, mvp_audio.wav (rms 0.20, nonsilent), mvp_frame00..08.png}`.
  **Face distorted.** (Maintainer: "what the hell is that".)
- **HQ** (`/tmp/ltx2_hq` ← `pipeline/ltx2_t2v_av_hq.mojo`): 768×512, 16f, S_V=768
  N_TXT=1024, **15-step res_2s** (2 evals/step = 30 evals), **LoRA OFF** (ran the
  binary bare; LoRA needs the `lora` argv — see §6). Ran in **~16 min** (~55 s/step).
  Output: `output/ltx2_hq/{ltx2_t2v_av_hq.mp4, hq_audio.wav, hq_frame00..08.png (1.18 MB each)}`.
  **Background sharper, face STILL distorted.** → resolution alone is NOT the cause.
- **HQ+LoRA** (in flight at handoff): `/tmp/ltx2_hq lora output/ltx2_hq_lora`.
  Confirmed `LoRA: ON @0.25`, 1660 `LTX2Distilled` mappings, 28 global deltas
  applied additively. **VIEW `output/ltx2_hq_lora/hq_frame0*.png` when done** — this
  is the recipe's actual intended config; it decides whether missing-LoRA was the face cause.

### Block-0 parity (#3) — reproduced (MEASURED)
`pixi run mojo run -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_mojo_parity.mojo`
→ `video cos 0.9993907 (max_abs 1.72) PASS`, `audio cos 0.9961671 (max_abs 2.70) PASS`.
Also prints `WARNING: ref vs intermediates mismatch — oracle may be inconsistent`
(the intermediates file is out of sync with the ref; audio < 0.999 is the known §5b gap).

### Canonical recipe (MEASURED from `ltx_pipelines/utils/constants.py`)
- `DISTILLED_SIGMA_VALUES = [1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0]`
  — **EXACTLY** the schedule the MVP printed → serenitymojo's distilled schedule is correct.
- `STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375,0.725,0.421875,0.0]`.
- `DEFAULT_1_STAGE_HEIGHT=512, DEFAULT_1_STAGE_WIDTH=768`; stage-2 = ×2 → 1024×1536.
- → **256×256 (MVP) is well below the trained default**; the real render is **2-stage**
  (512×768 base → spatial upsampler → 1024×1536). serenitymojo HQ runs single-stage
  only ("stage-2 upsampler + AdaIN not built", per its own header).

---

## 5. PROBLEM A — SPEED (MEASURED root cause; fix is HYPOTHESIS)

**It is NOT disk-bound.** `free -g` showed `buff/cache = 40 GB` while running → the
28 GB FP8 ckpt is fully resident in the OS page cache (RAM). Steps read from RAM.

**Real cause = the streamer design.** `offload/ltx2_block_stream.mojo` header +
code (MEASURED):
- It uses a **single-resident-window** discipline: load block i (read FP8 from the
  mmap, **dequant FP8→BF16 on GPU** via `ops/fp8.fp8_e4m3_dequant_to_bf16`, copy to
  GPU), run forward, **drop it**, load block i+1. Only ONE block on GPU at a time.
- So HQ does ≈ 48 blocks × 30 evals = **~1,440 full block loads**, each re-read +
  re-dequanted from host every eval.
- Consequence (MEASURED): **GPU util 7%**, ~11 GB used. The GPU idles waiting on
  host→GPU PCIe copy + dequant; it barely computes.
- This module was built (LTX2_PORT_PLAN P2) to *prove streaming fits a VRAM
  ceiling*, not for speed.

**Why it can't just stay resident as-is:** it dequants to BF16 (2 B/param). The
BF16 DiT (~40–56 GB) can't fit 24 GB. **But the FP8 bytes (~1 B/param, ~20 GB for
the DiT blocks) WOULD fit 24 GB.**

**Fix (HYPOTHESIS — not built/measured):** an **FP8-resident** path mirroring the
Rust `fp8_resident.rs`: keep FP8 weights in VRAM for the whole denoise, dequant
on-GPU per use (or run FP8 GEMM kernels), and overlap prefetch — stop re-reading
the model from host every eval. That's the 14 min → seconds change. Reference the
official resident behavior in `ltx_core` and the Rust `fp8_resident.rs`.
**This is orthogonal to face quality.**

---

## 6. PROBLEM B — FACE QUALITY (open; suspects ranked)

The scene is coherent (composition, lighting, background, hands) but the **face is
melted/uncanny** at both 256² and 768×512 with LoRA OFF (MEASURED by eye). Suspects,
none confirmed (all HYPOTHESIS), cheapest test first:

1. **Missing distilled LoRA.** Both runs were LoRA OFF; the HQ recipe header says
   the proven-sharp path is *distilled + res_2s + LoRA@0.25*. **Test in flight** —
   view `output/ltx2_hq_lora/hq_frame0*.png`. Enable via the `lora` argv (the binary
   defaults `apply_lora=False`; `main()` sets it only if `argv[1]=="lora"`).
2. **FP8 weight quantization.** Compute is already BF16 (FP8 is storage; dequant→BF16
   per `ltx2_block_stream` header), but the *weights* are FP8-quantized. **Test with
   the 44 GB BF16 distilled ckpt** (`ltx-2.3-22b-distilled.safetensors`) — removes
   FP8 error. (Will stream slower: 44 GB > 24 GB.)
3. **Missing 2-stage spatial upsampler** (`ltx_core/model/upsampler/`). Real render
   is base→×2 upsample→1024×1536; serenitymojo is single-stage. Affects sharpness/detail.
4. **Real bug in `ltx2_dit` forward or VAE decode.** #2 was never block-0 gated.
   Localize by diffing serenitymojo against `ltx_core/model/transformer` (forward)
   and `ltx_core/model/video_vae` (decode). The #3 parity block + the block-0 oracle
   (`output/ltx2_av/block0_ref.safetensors`) is the available numeric gate.
5. **Conditioning mismatch.** The text/video/audio context is a *cached dump* from
   inference-flame, not produced by this pipeline's own encoders. Confirm the dump
   matches what `ltx_core` expects (caption projection split video|audio, ordering).

**Recommended order:** (1) read LoRA result → if clean, done. Else (2) BF16 weights →
isolates FP8. Else (3/4) diff VAE decode + forward against `ltx_core`, then (5).

---

## 7. TRAINER — status + plan (anchor: musubi)

**Status (MEASURED / from TRAINER_STATUS_2026-06-04.md):** ltx2 trainer = "CODED,
NEVER TESTED". `training/train_ltx2_real.mojo` is wired to
`models/ltx2/ltx2_stack_lora.mojo` (#4), which is the **video-only attn1+ff** arch
(no cross-attn/audio/cross-modal). It has the scaffolding (LoRA set q,k,v,out ×48
blocks; `ltx2_lora_adamw_step`; offload via `ltx2_plan` + `TurboPlannedLoader`;
logit-normal timestep; progress display) but trains the WRONG model. `ltx2_dit`
(#2) has no backward at all.

**The real work (build, in order):**
1. Give the trainer a **full-AV forward == inference #2** (`ltx2_dit.ltx2_block_forward_av`),
   so loss is computed on the correct model output. Audio computed forward (frozen).
2. **AV backward for the video path** (mirror wan22 block bwd; native BF16, grads F32,
   local F32 casts only at rope_backward/cat_backward/gate_residual_backward — the
   established repo pattern; see memory `project_bf16_carrier_fix`). For a **video
   T2V LoRA** (musubi `networks/lora_ltx2.py` targets), trainable params live only on
   video Linears, so backward must reach: video FFN → A2V(video-Q branch) → video
   cross-attn → video self-attn → video LoRA. Audio self/cross/V2A/audio-FFN need
   only forward (no trainable audio params unless audio LoRA is desired). Confirm
   audio-LoRA scope with maintainer before widening.
3. LoRA targets + init/scale + Comfy save format: follow musubi
   `networks/lora_ltx2.py` + `ltx_2/convert_lora_to_comfy.py`.
4. Reuse existing scaffolding (data loop, adamw, offload, progress).
5. **Data cache**: none exists (AV handoff §5d). A "TESTED WORKING" verdict
   (loss drop + base-vs-LoRA sample shift) is blocked until a real
   latent+text+audio cache exists. Trainer mechanism can be unit-gated before that
   (per-sublayer vs torch autograd on synthetic non-degenerate inputs).

**Gate (do not skip):** per-sublayer backward parity vs a torch autograd oracle
built from musubi `BasicAVTransformerBlock` (same approach as the forward oracle:
real block-0 weights, non-degenerate seeded inputs, compare grads cos≥0.999).

---

## 8. Exact commands

```bash
cd /home/alex/mojodiffusion

# --- INFERENCE (build once, then run; solo on GPU — see §9 OOM note) ---
pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/pipeline/ltx2_t2v_av_mvp.mojo -o /tmp/ltx2_mvp
/tmp/ltx2_mvp                                   # 256² 8-step, LoRA OFF
pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/pipeline/ltx2_t2v_av_hq.mojo  -o /tmp/ltx2_hq
/tmp/ltx2_hq                                    # 768×512 res_2s, LoRA OFF
/tmp/ltx2_hq lora output/ltx2_hq_lora           # 768×512 res_2s, LoRA@0.25  <-- recipe's real config
/tmp/ltx2_hq staged lora output/ltx2_hq_staged  # (staged path; run_staged)

# --- BLOCK-0 PARITY (#3, light, block-0 only) ---
pixi run mojo run -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_mojo_parity.mojo

# --- ORACLE regen (musubi venv has torch + musubi) ---
/home/alex/musubi-tuner/venv/bin/python scripts/ltx2_av_oracle.py
/home/alex/musubi-tuner/venv/bin/python scripts/ltx2_av_oracle_intermediates.py

# --- speed/diag while a run is live ---
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # expect ~7% util (streaming-bound)
free -g                                                                    # buff/cache ~40G => ckpt in page cache
```

Outputs land in `output/ltx2_mvp/`, `output/ltx2_hq/`, `output/ltx2_hq_lora/`
(`*.mp4`, `*.wav`, `*_frame0*.png`). View frames with an image viewer / Read tool.

---

## 9. Gotchas / discipline

- **Build needs `-Xlinker -lm -Xlinker -lcuda`** or linking fails on `cos`.
- **Run heavy inference SOLO** (AV handoff §8): a prior host OOM killed agents when
  a full-model job ran concurrently with agent builds on the 62 GB host. Don't
  compile/spawn agents during a streamed 22B run.
- **Block-0 parity is light** (block-0 only, GPU < 1 GB) — safe anytime.
- **NEVER load the full 46 GB dev ckpt** for the oracle — block-0 keys only
  (`model.diffusion_model.transformer_blocks.0.`, ~86 tensors).
- **MEASUREMENT BEATS ASSERTION**: a green cos against a self-made/adapted oracle
  proves little; cos>1.0 or max_abs==0 ⇒ self-comparison, reject. Inference "good"
  = look at the pixels, not just finite stats (this session's lesson — finite,
  sane-std latents still produced a melted face).

---

## 10. Open decisions for the maintainer

1. **Speed vs quality first?** Speed (FP8-resident streamer) is a contained,
   high-impact build; quality is a diagnosis funnel (§6). Likely do the LoRA-result
   read (cheap) then start the FP8-resident streamer in parallel with quality diff.
2. **Audio LoRA?** Video-only T2V LoRA bounds the backward to the video path.
   Audio LoRA roughly doubles backward scope. musubi makes audio keys optional.
3. **Retire #1/#3 now or after trainer lands on #2?** (Recommend: after, so #3's
   parity harness + oracle stay available as the numeric gate for #2.)
