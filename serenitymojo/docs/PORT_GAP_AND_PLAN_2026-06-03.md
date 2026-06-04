# Port Gap Audit + Phased Plan — 2026-06-03

Scope: gap between the Rust stacks (`inference-flame` = inference, `EriDiffusion-v2` =
training, `flame-core` = engine) and `serenitymojo` (pure Mojo+MAX). Grounded in a
direct file inventory of all four trees plus `PORT_STATUS_2026-05-29.md` and
`VAE_PORT_GAP_AUDIT_2026-06-03.md`.

Gate doctrine (unchanged): a model is "working" only when the artifact is visually
coherent (image) / coherent+audible (video/audio). Per-op/block parity = cos≥0.999 vs
Rust/Python oracle; deep multi-forward chains use 0.99. LoRA is ADDED, never fused.

## Codebase roles
- **flame-core** — Rust tensor/autograd/CUDA engine. NOT a per-model port target. Mojo
  equivalent = MAX + `serenitymojo/ops` + `runtime`. Only missing *ops/kernels* port here.
- **edv2** — Rust trainers for: acestep, anima, chroma, ernie, flux, klein, l2p, ltx2,
  qwenimage, sd35, sdxl, sensenova_u1, wan22, zimage.
- **inference-flame** — the comprehensive inference set (the real gap source).
- **serenitymojo** — target.

## A. Image/video DiT + UNet coverage

DONE & coherence-verified in Mojo (PORT_STATUS_2026-05-29): Z-Image, Qwen-Image
512/1024, Klein9B(FLUX.2), ZImage-L2P, Anima, ERNIE-Image, SD3.5 M+L, Lens(DiT-math
parity; real-image blocked on GPT-OSS), Chroma, SenseNova-U1, LTX2 T2V+audio+LoRA,
HiDream-O1(dit+smoke). PiD = in progress.

BUILT but inference-coherence UNVERIFIED (files exist, not in the coherent table):
SDXL, SD1.5, Flux.1-dev, Lance(full t2v).

MISSING ENTIRELY from serenitymojo (present in inference-flame):
| Model | Files in Rust | Class | Notes |
|---|---|---|---|
| asymflux2 | asymflux2.rs | image | used by edv2 `asymflux2_klein9b` training |
| hidream_i1 | hidream_i1/ | image | have O1, not I1 |
| wan22_dit / wan_vace_dit | wan22_dit.rs, wan_vace_dit.rs | video | Lance uses wan22 VAE decode only |
| hunyuan15_dit | hunyuan15_dit.rs | video | + hunyuan_vae |
| kandinsky5_dit | kandinsky5_dit.rs | video | + kandinsky5 sampler |
| cosmos_predict25_dit | cosmos_predict25_dit.rs, cosmos_reason1.rs | video | i2v/t2v/v2v |
| magihuman_dit / sr_dit | magihuman_dit.rs, magihuman_sr_dit.rs | human video | + magihuman_unipc |
| acestep_dit | acestep_dit.rs, acestep_condition.rs | audio | + acestep vae+sampler |
| nava_av | nava_av.rs, nava_blocks.rs | audio-video | + nava audio vae/sampler |
| EXCLUDED by user | helios_dit, nucleus_dit, wuerstchen/paella (Stable Cascade) | — | do not port |

## B. VAE coverage (from VAE_PORT_GAP_AUDIT_2026-06-03)

Decoders are broadly present. The gap is ENCODERS (needed for prepare/training/img2img/
roundtrip):
1. **SDXL/LDM AutoencoderKL encoder** — highest priority (SDXL/SD1.5/SD3 training & img2img)
2. **LTX2 video VAE encoder**
3. **Wan2.2/Lance VAE encoder**
4. SD3 embedded encoder (only when SD3 training/img2img active)
5. Exotic: acestep_vae, hunyuan_vae, nava_audio, sa_audio_vae (audio/video)

## C. Sampler coverage
Have: anima, dpmpp_2m, ernie, flow_match, flux1, flux2_klein, hidream_o1, klein,
lance_t2v, lens_flowmatch, ltx2(guidance/sampling), sd15_euler, sd3_flow_match,
sdxl_euler, unipc, pid_distill.
MISSING: acestep, cosmos_rf, cosmos_unipc, kandinsky5, magihuman_unipc, motif, nava,
exponential_multistep, l2p_sampling. EXCLUDED: ddpm_wuerstchen, helios(dmd/pyramid).

## D. Text encoder coverage
Have: clip, mistral3b, qwen25vl, qwen3, t5.
MISSING: gpt_oss (Lens real-image blocker — high value), gemma3/gemma4, t5gemma2,
umt5 (wan), llama3.

## E. Training (edv2 → serenitymojo training/)
Have trainers: anima, ernie, flux, klein, sdxl, zimage.
MISSING trainers: chroma, l2p, ltx2, qwenimage, sd35, sensenova_u1, wan22, acestep.

---

## Phased plan — 3 teams × (builder + skeptic + bugfixer), mojo-port loop

Each component runs the proven `mojo-port` loop: builder ports → skeptic refutes →
bugfix → gate at cos≥0.999 vs a diffusers/Rust GPU-bf16 oracle. 3 teams run in
parallel; each takes one component at a time. Phases are dependency-ordered.

**Phase 0 — Unblockers & shared ops** (1 sprint)
- T1: GPT-OSS encoder (unblocks Lens real-image; OOM-aware streamed).
- T2: missing shared ops/kernels parity sweep vs flame-core (audit which ops new
  models need that serenitymojo lacks).
- T3: SDXL/LDM AutoencoderKL **encoder** (VAE gap #1).

**Phase 1 — VAE encoders + close coherence gaps** 
- T1: LTX2 video VAE encoder.  T2: Wan2.2/Lance VAE encoder.
- T3: verify/finish inference coherence for SDXL, Flux.1-dev, HiDream-O1, Lance.

**Phase 2 — New image DiTs**
- T1: asymflux2.  T2: hidream_i1.  T3: wan22_dit + wan_vace_dit (+ wan VAE encoder if not done).

**Phase 3 — Video DiTs** (heavy)
- T1: hunyuan15 (dit+vae+sampler).  T2: kandinsky5 (dit+sampler).
- T3: cosmos_predict25 (dit+reason1+rf/unipc samplers).

**Phase 4 — Human video + audio**
- T1: magihuman (dit+sr_dit+unipc).  T2: acestep (dit+condition+vae+sampler).
- T3: nava_av (av+blocks+audio vae+sampler) + sa_audio_vae.

**Phase 5 — Training parity for new families** (edv2 → training/)
- Per family, per-block torch-autograd parity + non-degenerate data
  (per `mojo-train-port`): chroma, l2p, ltx2, qwenimage, sd35, sensenova_u1, wan22, acestep.

EXCLUDED throughout (user directive in PORT_STATUS): Helios, Nucleus, Stable Cascade.

## Progress log

**Phase 0 — COMPLETE (2026-06-03).** build→skeptic→bugfix→gate loop, all measured:
- GPT-OSS encoder: gate PASS, all Lens extract layers cos≥0.9993 (l23 0.99986) vs Lens
  text_encoder (GptOssForCausalLM) oracle. Unblocks Lens real-prompt image. Loop caught 3
  real bugs end-to-end (22 device-path compile sites, MoE F32 dtype contract, oracle
  capture-point post-norm) that compile-probe+skeptic missed.
- SDXL/LDM VAE encoder: gate PASS bf16 (weights cast to BF16 to match Rust), moments
  cos 0.99955 / mag 0.99976 @256². Decoder pre-existed.
- ops gap sweep: build_multiaxis_rope_tables (max_err 1.27e-7) + audit OPS_GAP_AUDIT_2026-06-03.md.
- Out-of-band fused kernels: bias_gelu (tanh, cos 0.9999955), layernorm_linear + residual_layernorm (cos≈1.0).

**Phase 1 — COMPLETE (2026-06-03).**
- LTX2 video VAE encoder: gate PASS bf16, cos 0.99994 (temporal verified T=9→2; oracle = torch
  transcription of Rust forward, independence established structurally).
- Wan2.2 VAE encoder: gate PASS bf16, cos 0.99998 (image-mode T=1; temporal time_conv UNVERIFIED).
- patchify3d op: cos 0.9999963, conv3d-patchembed≡unfold+linear PROVEN; + unpatchify3d.

**Carried follow-ups:** (1) ~~video VAE temporal (T2V) paths — Wan2.2 time_conv + feat-cache loop~~
DONE 2026-06-03: Wan2.2 temporal T2V cos 0.99998 @T=17 + 256² gate self-runs + LTX2 pad→comptime;
(2) coherence-closing for SDXL/Flux.1/Lance — visual gate, not yet run; (3) HiDream-O1 — DONE 2026-06-03:
DiT math was verified clean but the pipeline was a skeleton (dit.forward unwired, x_pred=z no-op).
Wired dit.forward into a real 20-step denoise (pipeline/hidream_o1_generate.mojo); generated a
COHERENT 256² image (output/hidream_o1_256_20step.png — red apple on wood table, prompt-matched,
proving the forward is wired not a no-op). Moves O1 into the coherent-image table. Follow-ups:
CFG common-S (guidance>1), higher res, Dev Flash GPU std/clamp kernel;
(4) SD3 embedded VAE encoder (deferred until SD3 training active).

**Phase 2 (in progress 2026-06-03):** wan22_dit (building); asymflux2 = AsymFlow velocity wrapper
on UNCHANGED Klein backbone — algebra gated cos≈1.0 + skeptic-verified vs LakonLab common.py (truth
source). Weighted-E2E DEFERRABLE (not blocked): the Rust ref asymflux2_klein9b_infer.rs IS runnable
(the "prints 30 keys and bails" was a STALE COMMENT; real body has full key translator + wired
main); E2E gate needs a Mojo asymflux2 pipeline driver (Klein+adapter+AsymFlow→Oklab→PNG) + a golden
capture from the Rust ref. HiDream-I1 DROPPED by user (weights not on disk).

**Phase 3 (2026-06-03):** video DiTs, all reuse patchify3d + rope_tables + wan22_dit pattern.
- cosmos_predict25_dit: block-0 cos 0.99999605 (real 4.1GB ckpt), skeptic-clean. Needed PER-AXIS NTK
  theta (ratios h=w=3/t=1 → θ 10000/31694) — added ops/rope_tables.build_multiaxis_rope_tables_per_axis
  (the gap the ops audit predicted; scalar version no-regression). Uses rope_halfsplit (not interleaved).
  FULL-RES BLOCKED on foundation: Dh=128 math-SDPA OOMs (68.7GB [H,S,S] at S=32760, no flash on sm_86).
- kandinsky5_dit: block cos 0.9999959 — but ONLY after a major skeptic catch: the builder ported the
  decoder self-attn as HEAD-AXIS attention to match a BUGGY oracle (rank-3 input); the real model
  (fractal_flatten→rank-2) does STANDARD SPATIAL attention (confirmed vs nn.py AND Rust). Bugfix
  corrected oracle→rank-2 + HEAD_AXIS=False; re-gated vs corrected ref = 0.9999959 (spatial/head-axis
  inverted to 0.99999/0.033). LESSON: a green cos vs a self-made oracle can be a wrong-oracle false pass.
- hunyuan15 DROPPED by user (no weights). Both kandinsky5/cosmos full-forward deferred (heavier oracle / OOM).

**Phase 4 (2026-06-03):** human video + audio.
- magihuman_dit: block cos 0.999964 (real distill 30.6GB ckpt), skeptic-clean. Uses ElementWiseFourier
  rope (NOT rope_tables — 16 bands × 3 axes → partial halfsplit on 96/128) + a new swiglu7 kernel +
  (w+1) RMSNorm. Skeptic caught oracle FABRICATES rope bands vs checkpoint-LOADED (value-equal, FRAGILE).
  Full-res deferred on Dh=128 OOM. SR-DiT + unipc deferred.
- acestep_dit: block 0.99998 + FULL-forward 0.99942 (real turbo ckpt) — first video/audio model to gate
  full-forward (1D audio ~128 tok dodges the Dh=128 OOM). Oracle = REAL canonical AceStepDiTModel (best
  independence). FRAGILE: full stack uses sdpa_nomask all 24 layers → production audio >128 tok UNTESTED
  (12 sliding-attn layers wrong at long seq, TODO in header). VAE + condition encoder + RF sampler deferred.
- nava SKIPPED by user (4K empty stub, no weights). hunyuan15 dropped.

**KEY FOUNDATION — RESOLVED 2026-06-03:** tiled/online-softmax SDPA for Dh=128 (any Dh) at large S.
Added sdpa_tiled + sdpa_nomask_tiled to ops/attention.mojo (ADDITIVE, 0 deletions — existing flash/math
paths byte-identical, ~20 models unaffected). Online-softmax recurrence (running max/denom/acc rescale,
KV blocks of 512); cos=1.0 vs math-mode multi-block (incl causal + fully-masked-row + Dh=64, all
skeptic-re-run); S=8192 Dh=128 completes on 24GB where math-mode OOMs. Callers swap sdpa->sdpa_tiled /
sdpa_nomask->sdpa_nomask_tiled when S is large. UNBLOCKS full-res forward for cosmos/magihuman/wan22/
kandinsky5 + 10+ Dh=128 DiTs. (Caveat: literal IEEE -inf masked rows NaN in BOTH tiled and math
identically — pre-existing, no caller uses literal -inf.)

**FULL-FORWARD SWEEP (2026-06-03, via sdpa_tiled) — DONE for 3 video DiTs:**
- wan22_dit: full 30-block forward cos 0.999518 @ S=4096 vs canonical WanModel (independent oracle).
  Dispatch on comptime S (>512 → tiled); small-grid gate byte-unchanged (0.99963833, no regression).
- cosmos_predict25_dit: full 28-block forward cos 0.99999453 @ S=8192, no-OOM (12.5GB/24GB). Oracle is a
  hand-transcription BUT skeptic re-derived composition vs canonical minimal_v4_dit.py = CANONICAL-FAITHFUL;
  tiled-vs-torch-flash agreement at S=8192 is a genuine independent attention cross-check. FRAGILE: image-mode
  (video non-zero-mask + fps-modulation untested).
- magihuman_dit: CHUNK B built (adapter embed + Fourier rope w/ REAL bands + MM layers gelu7/swiglu7 +
  final heads) → full 40-layer forward cos 0.99908 @ L=128, no-OOM (streamed 30.6GB). Oracle transcription,
  skeptic-verified vs canonical dit_module.py = transcription-FAITHFUL. UNMEASURED: multi-axis t/w rope
  coords, large-S tiled path (L=128; proven in wan22/cosmos), bf16-flash prod SDPA.
RESULT: tiled SDPA foundation fix validated END-TO-END in real models at large S (S=4096, S=8192) against
independent/canonical oracles. kandinsky5 oracle-provenance discipline applied to all 3 (2 transcription
oracles checked + confirmed faithful).

NEXT: Phase 5 trainers (mojo-train-port: chroma/l2p/ltx2/qwenimage/sd35/sensenova_u1/wan22/acestep); then
per-model follow-ons (acestep long-audio sliding mask, VAEs/samplers/SR, wan_vace, asymflux2 E2E,
SDXL/Flux/Lance visual pass, SD3 VAE enc, kandinsky5 full-forward w/ text-encoder oracle).

## Campaign state after Phases 0-4 + O1 (2026-06-03)
Gated this campaign (cos vs GPU-bf16 oracle, all skeptic-verified): GPT-OSS enc, SDXL/LDM VAE enc,
bias_gelu, layernorm_linear, residual_layernorm, build_multiaxis_rope_tables(+per_axis), patchify3d,
LTX2 VAE enc, Wan2.2 VAE enc(+temporal), wan22_dit, asymflux2(AsymFlow algebra), kandinsky5_dit(block,
after attn-axis bugfix), cosmos_predict25_dit(block), magihuman_dit(block), acestep_dit(block+full).
Coherent image: HiDream-O1 (finished from skeleton). DROPPED: HiDream-I1, hunyuan15, nava (no weights).
Excluded (user): Helios, Nucleus, Stable Cascade.
NEXT: (1) Dh=128 tiled attention foundation; (2) full-forward gates for the video DiTs; (3) Phase 5
training parity (mojo-train-port): chroma/l2p/ltx2/qwenimage/sd35/sensenova_u1/wan22/acestep; (4) per-model
follow-ons (VAEs/samplers/SR, wan_vace, asymflux2 E2E, SDXL/Flux/Lance visual pass, SD3 VAE enc).

DTYPE RULE (locked): match Rust dtype exactly — Rust casts VAE/encoder weights to bf16, so Mojo
runs bf16 and the parity oracle runs bf16. No F32 detours.

## Open scope decisions (need user) — RESOLVED: full scope, exclusions held (Helios/Nucleus/Cascade,
## + later kandinsky5/sensenova porting and weights-less models).

## FINAL STATE (2026-06-03) — "all except kandinsky/sensenova, skip weights-less" mandate CLOSED

FOUNDATION: Dh=128 tiled/online-softmax SDPA (sdpa_tiled/sdpa_nomask_tiled, additive, cos=1.0 vs
math-mode, S=8192 no-OOM). Unblocked full-res video.

FULL-FORWARD (via tiled SDPA): wan22_dit cos 0.999518@S=4096 (canonical WanModel), cosmos
0.99999453@S=8192 (canonical-faithful), magihuman 0.99908@L=128 (CHUNK B built).

INFERENCE COMPLETIONS: cosmos rf/unipc samplers (0.9999999); acestep long-audio sliding mask
(0.99976, via tiled full-mask) + RF sampler (1.0) + Oobleck audio VAE (0.99992); magihuman SR-DiT
(0.9999591)+unipc; SD3 VAE encoder (0.9999992); LTX2 upsampler (already done, cos 1.0); asymflux2 E2E
(coherent 512² image, adapter added-not-baked); O1 CFG+512² (coherent). wan_vace = architecture-
faithful, no weights → left alone.

TRAINER BACKWARD SURFACES (per-block torch-autograd parity, cos≥0.99999999, scope=backward only):
qwenimage, chroma, l2p, sd35, wan22, ltx2, acestep — all PASS. 3 new arms total across 7 models
(wan per-token AdaLN modulate/gate, GQA repeat_kv); rest reused the ~68-arm ops/*_backward library.
chroma block==Flux, l2p==Z-Image, sd35==Klein-joint pattern.

GENUINELY OPEN (not done): (a) shared block-offload-into-training-loop → needed for full L2P real-run
(loss-drops+sample-shifts) on 24GB; (b) zimage_stack_lora.mojo broken mid-refactor (pre-existing,
blocks full-stack LoRA gate); (c) SDXL/Flux.1/Lance inference visual-coherence pass (never run);
(d) weights-less left alone: HiDream-I1, hunyuan15, nava, wan_vace, sd35-via-HF(have .serenity);
(e) EXCLUDED by user: kandinsky5 (full-forward/sampler), sensenova_u1 (trainer), Helios/Nucleus/Cascade.
</content>
