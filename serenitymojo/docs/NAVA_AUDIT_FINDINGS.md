# NAVA Mojo vs torch — end-to-end DISTORTION audit (read-only)

Date: 2026-06-07. Auditor pass over the FULL output + sampler path (denoise loop →
unpatchify → VAE decode → PNG), Mojo (`/home/alex/mojodiffusion`) vs torch ref
(`/home/alex/EriDiffusion/inference-flame/ports/nava`).

Symptom: 832×480 frame is COHERENT (recognizable man-in-penthouse) but the FACE is
melted and motion smeared; real NAVA is crisp.

## BOTTOM LINE

I did **NOT** find a spatial-scramble / H-W-swap / patch-order bug. Every geometric
stage of the assembly is provably correct (verified line-by-line below). The
divergence is **numeric + feature-completeness in the denoise loop**, not layout.

**Single most-likely root cause:** bf16 magnitude/scale drift in the DiT forward that
the single-forward **cosine** gate (0.9998) is mathematically blind to, compounding
over 25 steps × 3-forward CFG (guidance 3) into the already-measured **+12 % latent
std** inflation; the Wan2.2 VAE decodes off-distribution latents as smeared/melted
fine detail while preserving coarse structure. Secondary: the Mojo pipeline omits two
production-config features (`add_spk_emb`, `timbre_cfg`) that the crisp reference uses.

A cosine gate cannot see a consistent ~1–2 %/forward magnitude error. The proper
confirmation is an **L2-norm ratio** ‖vel_mojo‖/‖vel_torch‖ per forward (NOT cosine) —
could not be run here (GPU busy / read-only).

---

## VERIFIED-CORRECT (MATCHES) — do not chase these

### S1. PNG value mapping + channel order — MATCHES
- Mojo `image/png.mojo:194-197` SIGNED → `(v+1)*127.5`, clamp[0,255], round; RGB planes
  R/G/B at `png.mojo:251-253`.
- Torch `inference_nava.py:277` `_to01 = clamp((x+1)/2,0,1)` then `*255` (`:623`/`:664`);
  `write_video` consumes RGB.
- `(x+1)/2*255 ≡ (x+1)*127.5`. Identical. RGB order identical.

### S2. H/W orientation (no transpose) — MATCHES
- Latent grid (F=5, Hlat=30, Wlat=52) → decode `wan22_decoder.mojo:1163`
  `permute(0,4,1,2,3)` → `[1,3,17,480,832]` (H=480, W=832).
- Pipeline `nava_hires_pipeline.mojo:240-247` slices frame → `[1,3,480,832]` → `save_png`
  (expects `[1,3,H,W]`, `png.mojo:222`). H=480, W=832 throughout. No swap.

### S3. Unpatchify token / within-patch order — MATCHES (exactly)
- Torch `model_mm.py:1714-1716`: `view(*v,*patch_size,c)` then `einsum('fhwpqrc->cfphqwr')`
  → within-token layout `(pf,ph,pw,c)` channel-innermost; patch index `(f,h,w)` row-major.
- Mojo kernel `ops/patchify3d.mojo:264-266`:
  `patch=(fi*HO+hi)*WO+wi`, `src_ch=((pfi*ph+phi)*pw+pwi)*C+ci`, `seq_off=patch*PD+src_ch`.
  This is the bit-exact algebraic inverse of the torch einsum. `_unpatchify_video_hires`
  (`nava_dit.mojo:411-425`) then permutes CFHW→FHWC and flattens to `[7800,48]`.
- Latent token order `(f,h,w,c)` is consistent across fixture, DiT in/out, and VAE in
  (see S4). No scramble.

### S4. Latent→VAE reshape + denorm — MATCHES
- Mojo `wan22_decoder.mojo:1147` `reshape(lat, [1,5,30,52,48])` (NDHWC) — channels-last
  equivalent of torch's `[1,48,5,30,52]` NCDHW (adapter permute `local_video_vae.py:117`).
  Same `(t,h,w,c)` token order torch uses (`model_nava.py:408-412` builds `[L,C]` as
  `(t,h,w)`×C; `local_video_vae.py:98` encode permute confirms).
- Denorm `wan22_decoder.mojo:1148` `z = z/inv_std + mean` (= z·std+mean). Torch
  `vae2_2.py:815` `z = z/scale[1]+scale[0]`, `scale=[mean,1/std]` (`vae2_2.py:1014`).
  mean/std tensors hardcoded `vae2_2.py:906-1013` — match the checkpoint stats the Mojo
  loads. (VAE decode already gated cos 0.99999992.)
- Global `scaling_factor/shift_factor` at torch `pipeline_nava.py:580` is a **no-op**:
  `LocalVideoVAEAdapter.config = (scaling_factor=1.0, shift_factor=0.0)`
  (`local_video_vae.py:56`). Correctly absent in Mojo.

### S5. Temporal causal decode (5 latent → 17 frames) — MATCHES
- Mojo `wan22_decoder.mojo:1153-1159`: frame0 `first=True`, frames 1..4 `first=False`,
  concat dim 1. Torch `vae2_2.py:821-838`: `first_chunk=True` at i==0 else False, cat dim 2.
  1 + 4×4 = 17. (Already gated.)

### S6. CFG formula, guidance scales, align_3d_cfg — MATCHES
- Active config is `configs/nava.yaml` (NOT `inference_fusion.yaml`):
  `video_guidance_scale=3.0`, `audio_guidance_scale=2.0`,
  `video_align_guidance_scale=3.0`, `audio_align_guidance_scale=2.0`, `align_3d_cfg=true`,
  `shift=5`.
- Torch `pipeline_nava.py:539`
  `eps_vision = eps_cond + g·(eps_cond-eps_uncond) + g_align·(eps_cond-eps_mmask)`.
  Mojo `nava_hires_pipeline.mojo:167-171` `eps_v = cond + 3·(cond-unco) + 3·(cond-mmask)`.
  g=g_align=3 ✓. CFG combine is done in F32 in Mojo (`:159-164`) — correct.

### S7. UniPC scheduler config — MATCHES
- Torch `FlowUniPCMultistepScheduler` (`fm_solvers_unipc.py`) defaults: `solver_order=2`,
  `prediction_type=flow_prediction`, `predict_x0=True`, `solver_type=bh2`,
  `lower_order_final=True`, `final_sigmas_type=zero`, `disable_corrector=[]`
  (`:80-94`); instantiated `pipeline_nava.py:85-92` `num_train_timesteps=1000, shift=5,
  use_dynamic_shifting=False`.
- Mojo `sampling/unipc.mojo:293-337` is the same variant (bh2/order-2/predict_x0/flow/
  lower_order_final/zero). predict_x0 conversion `unipc.mojo:354-357` `x0 = sample -
  sigma_t·v` matches alpha=1-sigma flow. Timestep `t = sigma·1000`
  (`nava_hires_pipeline.mojo:138`) matches torch `timesteps = sigmas*num_train_timesteps`
  (`fm_solvers_unipc.py:205`).

### S8. SLG (skip-layer guidance, layer 11) — MATCHES (it's a NO-OP in both)
- Torch passes `slg_layer=11` ONLY on the uncond forward (`pipeline_nava.py:496`), but
  `use_mmdit_model=true` ⇒ backbone is `WanAVModel` (`model_nava.py:190`,
  `model_mm.py:1173`). `WanAVModel.forward` (`model_mm.py:1574-1691`) takes `**kwargs` and
  **never references slg_layer** — no block is skipped. (SLG is only wired in the unused
  `fusion.py:459` non-mmdit path.) So both Mojo and torch skip nothing. Not a divergence.

---

## DIVERGENCES (ranked by how well each explains "coherent + melted/smeared")

### D1 — RANK 1 — DIVERGENCE (numeric, gate-blind): bf16 magnitude drift in DiT forward
- The single-forward gate is **cosine** (vel_vid 0.99980). Cosine is invariant to a global
  magnitude scale — a consistent ~1–2 % over-/under-magnitude per forward yields cos≈1.0
  yet compounds.
- Mojo runs the **DiT internals in bf16** (`nava_dit.mojo:506-565`, all linear/attention/
  head in BF16; F32 cast happens only AFTER the DiT, at the CFG combine
  `nava_hires_pipeline.mojo:159-164`). 25 steps × 3 forwards × guidance 3 amplifies the
  bf16 error (each `(cond-uncond)` term is a bf16 difference ×3).
- Already-measured effect: +12 % latent std (256²: 1.708 vs torch 1.526) — matches an
  energy-inflation signature. The Wan2.2 VAE is highly sensitive to latent statistics;
  off-distribution latents decode to **smeared/melted fine texture with intact coarse
  layout** = the exact symptom.
- Why it best explains the symptom: coarse structure (low-freq, dominated by cond
  direction → cos-correct) survives; high-freq detail (sensitive to magnitude/accumulated
  rounding) melts.
- CONFIRM (not yet done): per-forward **L2 norm ratio** ‖vel_mojo‖/‖vel_torch‖ on the
  hi-res fixture; and an A/B with **F32 DiT compute** (heads/attention/FFN accumulate F32).
  Mojo `nava_dit.mojo` head/attention/FFN.  Torch `model_mm.py` forward (bf16 autocast,
  `inference_nava.py:540-541`).

### D2 — RANK 2 — DIVERGENCE (feature-completeness): spk_emb + timbre 4th forward omitted
- `configs/nava.yaml`: `add_spk_emb=true`, `spk_emb_prob=0.9`, `timbre_cfg=true`,
  `timbre_align_guidance_scale=3.0`. The first-video target has a voice reference ⇒
  `spk_embs` present ⇒ `effective_timbre=True`.
- Torch injects speaker embeddings into the context (`model_mm.py:1515-1527`) and — when
  `spk_embed is not None` — uses **context_audio for BOTH streams** in the cond & mmask
  forwards (`model_mm.py:1644`). Torch also runs a **4th forward** (`eps_timbre_uncond`,
  `pipeline_nava.py:515-529`) and adds `timbre_align·(eps_cond-eps_timbre)` to the AUDIO
  eps (`:548/:550`).
- Mojo `nava_hires_pipeline.mojo:144/150/156` runs a bare **3-forward, text-only** CFG;
  `NavaDiTHires.forward` (`nava_dit.mojo:492-500`) has **no spk_embed/spk_pos** parameter
  and no 4th forward.
- Effect on VIDEO is real but second-order: (a) the cond/mmask video cross-attention sees
  caption-with-spk vs Mojo's caption-only (a few perturbed context tokens); (b) more
  importantly, the audio latent is denoised with the WRONG CFG (no timbre term) and
  **joint cross-modal attention** in every DiT block couples the drifting audio tokens
  back into the video velocity — degrading video too.
- Why RANK 2 not RANK 1: the gate's reference (`nava_fx_hires.safetensors`) cos-matched the
  Mojo's no-spk forward at 0.9998, so that oracle was itself a no-spk SUBSET — i.e. the gate
  never validated the production (spk+timbre) path, but spk's direct video effect is small.

### D3 — RANK 3 — DIVERGENCE (minor): timestep int64 truncation not replicated
- Torch `fm_solvers_unipc.py:210-211`: `self.timesteps = ....to(dtype=torch.int64)` — the
  per-step timestep is **truncated to integer** before the sinusoidal time-embedding.
- Mojo `nava_hires_pipeline.mojo:138` `t_val = Float32(sigmas[i]*1000.0)` — exact float,
  consumed by `timestep_embedding(in_t,...)` (`nava_dit.mojo:511`).
- Magnitude: up to <1 in ~1000 (<0.1 %) per step → small per-step modulation mismatch.
  Real but unlikely to "melt." Fix: `floor(sigmas[i]*1000)`.

### D4 — RANK 4 — DIVERGENCE (negligible): sigma-schedule double-shift endpoint
- Torch `set_timesteps` linspaces from `self.sigma_max` which is the **already-shifted**
  0.99980 (`fm_solvers_unipc.py:107-132` then `:183-193` reshifts). Mojo linspaces from the
  **unshifted** 0.999 then shifts once (`unipc.mojo:89-97`).
- Net per-sigma difference ≈ 0.0001–0.0002 (<0.02 %). Negligible; not the cause.

---

## Recommended verification order (when GPU frees)
1. Measure ‖vel_mojo‖/‖vel_torch‖ (L2, NOT cosine) for cond/uncond/mmask on the hi-res
   fixture at several timesteps → confirms/eliminates D1.
2. Re-render with F32 DiT compute (D1 fix) and compare per-step lv_std to a torch trace.
3. If D1 insufficient: add spk_emb injection + the timbre 4th forward (D2) and a full
   torch end-to-end latent-trajectory gate (NOT a single forward).

---

# SKEPTIC REBUTTAL (2026-06-07, adversarial read-only pass)

Verdict header per finding: **UPHELD / OVERTURNED / NEEDS-MEASUREMENT**. The auditor's
two headline hypotheses (RANK 1 bf16-magnitude, RANK 2 spk/timbre) are **both built on
premises that the gated ground-truth facts contradict**. The auditor also **never looked
at the temporal frame count**, which is the single biggest off-distribution lever in this
run. Details below with file:line.

## Per-finding verdicts

### BOTTOM LINE (auditor: "numeric + feature-completeness, single most-likely = bf16 magnitude drift")
**OVERTURNED as stated.** The auditor's own decisive evidence — "+12 % latent std (256²:
1.708 vs 1.526)" — is from the **256² square run**, an explicitly-superseded resolution.
At HI-RES the loop std MATCHES torch (Mojo final lv_std **1.152 ≈ torch 1.143**, gated;
handoff `NAVA_HANDOFF_2026-06-07.md:138`). Citing the 256² number as the hi-res root cause
is the central error of this audit. A magnitude/energy-inflation mechanism that predicts
+12 % std is falsified by a matching std. (See "Reconciling RANK 1" below.)

### S1 PNG mapping / channel order — **UPHELD.** `png.mojo:194-197` vs `inference_nava.py:277`. Algebra checks; not the issue.
### S2 H/W orientation — **UPHELD.** No transpose. Verified independently.
### S3 unpatchify token order — **UPHELD.** `patchify3d.mojo:264-266` is the algebraic inverse of the torch einsum. Not the issue.
### S4 latent→VAE reshape + denorm — **UPHELD.** denorm identity (LocalVideoVAEAdapter scaling=1/shift=0); Wan mean/std inside decoder. Gated 0.99999992.
### S5 temporal causal decode 5→17 — **UPHELD as mechanics, but the auditor MISSED the load-bearing fact.** The 5-latent→17-video decode is *mechanically* correct (`wan22_decoder.mojo:1153-1159`). What the auditor never asked: **is a 5-latent-frame / 17-video-frame clip in-distribution for this model?** It is NOT (see NEW-1). A mechanically-correct decode of an off-distribution latent still melts. "MATCHES" here is a true-but-irrelevant verification.
### S6 CFG formula / guidance — **UPHELD.** `nava_hires_pipeline.mojo:166-177` mirrors `pipeline_nava.py:539/546`. But note this confirms the **timbre term is absent from eps_vision in torch too** — see RANK 2 overturn.
### S7 UniPC config — **UPHELD.** bh2/order-2/predict_x0/zero, shift=5. Gated ≥0.9999999 (handoff:112).
### S8 SLG no-op — **UPHELD.** `slg_layer` never consumed by `WanAVModel.forward`. Confirmed `grep slg model_mm.py` = passed, never read.

### D1 — RANK 1 (bf16 magnitude drift) — **OVERTURNED as stated; residual NEEDS-MEASUREMENT.**
- The mechanism the auditor describes is **global magnitude inflation** ("+12 % latent std",
  "energy-inflation signature"). This is **directly killed** by the gated hi-res fact:
  Mojo lv_std 1.152 ≈ torch 1.143 (`NAVA_HANDOFF:138`). If Mojo's hi-res latent has torch's
  std, there is no global magnitude drift to blame.
- **Reconciling the skeptic's question** ("could a per-element error leave global std
  unchanged but still smear?"): yes in principle — but that is a *per-element rounding*
  hypothesis, **not** the auditor's *magnitude* hypothesis. The auditor conflated the two.
  The per-element thread is the handoff's own RNE story (`NAVA_HANDOFF:138-139`), and that
  story is **already implemented**: `nava_hires_pipeline.mojo:121,123,141,142,147,148,153,154`
  all use `torch_f32_to_bf16_rne` at every f32→bf16 boundary cast. **The handoff's claim that
  RNE is "NOT YET APPLIED" is STALE** — it IS applied in the current file. So if the melted
  frame was produced by the current build, boundary-RNE did *not* fix it, which *also*
  undercuts the handoff's RNE root-cause, not just the auditor's.
- The only surviving numeric thread is **DiT-INTERNAL bf16 rounding** (the 30 blocks' native
  store-cast, which gives per-forward cos 0.99980, `NAVA_HANDOFF:141`) vs torch's bf16
  autocast. But torch's own reference *also* runs the DiT in bf16 autocast
  (`nava_diag_hires_cmp.py:46` `torch.autocast(dtype=bf16)`), so this is a sub-quantum
  rounding-mode delta, not a dtype-class delta. **NEEDS-MEASUREMENT** via per-step latent
  cos (below), but I predict it stays ≥0.999 (matching std is strong prior evidence the
  latents nearly coincide).
- Call-out: the auditor's "CONFIRM via L2-norm ratio (could not be run, GPU busy)" is a
  hand-wave that **dodges the gated std fact already in the handoff**. The std *was* measured;
  it matches; that is the L2-energy answer at the latent level and it refutes RANK 1.

### D2 — RANK 2 (missing add_spk_emb + timbre 4th forward) — **OVERTURNED. Dead for this run.**
Traced the full data flow. For **this run, spk_embs=None** (t2av, `<S>...<E>` speech span,
no speaker WAV):
1. `pipeline_nava.py:461` `effective_timbre = timbre_cfg and spk_embs is not None` ⇒ **False**.
2. The 4th forward is guarded `pipeline_nava.py:515 if effective_timbre:` ⇒ **does not run**.
   Torch runs exactly 3 forwards — identical to Mojo (`nava_hires_pipeline.mojo:144,150,156`).
3. spk injection is guarded `model_mm.py:1515 if self.add_spk_emb and spk_embed is not None:`
   ⇒ **skipped** when spk_embed is None.
4. `model_mm.py:1644` `context = context_vid if (... and spk_embed is None) else context_audio`
   ⇒ with spk_embed=None, **context = context_vid (caption-only)** — exactly Mojo. The
   auditor's claim (D2: "torch uses context_audio for BOTH streams") is **only true WITH a
   speaker** and is therefore inapplicable here.
5. **Even with a speaker, the timbre term never enters `eps_vision`**: `pipeline_nava.py:535`
   shows the timbre-in-vision line is **commented out**; the live path `:539` has no timbre
   term. Timbre is audio-eps-only (`:548/:550`). So RANK 2 cannot touch the video latent at
   first order under any config.
- The auditor's D2 rests on the **false premise** at its own line 128 ("The first-video
  target has a voice reference ⇒ spk_embs present ⇒ effective_timbre=True"). The run has
  spk_embs=None. This premise was asserted, never checked against the input fixture
  (`nava_first_video_inputs.py` supplies no spk_embs).

### D3 — int64 timestep truncation — **NEEDS-MEASUREMENT, but agree low-priority.**
Real torch `fm_solvers_unipc.py:210` truncates timesteps to int64; the diag uses those exact
`sv.timesteps` (`nava_diag_hires_cmp.py:46`), while Mojo uses `sigma*1000` float
(`nava_hires_pipeline.mojo:138`). Trivial to make exact (`floor`). Sub-0.1 %; not a melt cause.

### D4 — sigma double-shift endpoint — **UPHELD (negligible).**

## NEW findings the auditor AND prior debugging missed

### NEW-1 (STRONG) — the clip is BELOW the model's minimum trained temporal length.
`nava.yaml:82-85`: `video_fps: 24`, `video_min_frames: 33`, `video_tgt_frames: 121`. The run
uses **t_h_w=(5,30,52) → 5 latent frames → (5-1)*4+1 = 17 video frames** (`nava_hires_pipeline.mojo:6,232`).
**17 < 33 = below the trained minimum.** The 5s target the project actually wants
(`MEMORY: NAVA 5s target`, 121 frames @ 24fps) is **~31 latent frames** [(121-1)/4+1], a
[48360,48] latent — not [7800,48]. Running the temporal-RoPE DiT and the Wan2.2 causal-cache
VAE at a frame count below `video_min_frames` is **off-distribution in the temporal axis** →
the exact "motion smeared + melted detail, coarse layout intact" symptom. Neither the auditor
nor the pipeline note this. This is decoupled from any port-numeric question.

### NEW-2 — no prompt-engineering rewrite.
NAVA ships a Qwen3-4B PE rewriter (`NAVA_HANDOFF:128`) that expands the brief prompt into a
dense cinematic caption; the showcase ("crisp real NAVA") almost certainly uses it. The Mojo
run feeds the raw hand-written caption (`nava_first_video_inputs.py:18-21`). Caption density
is a documented quality lever. Plausible secondary contributor to "less crisp."

### NEW-3 — the decisive end-to-end gate ALREADY EXISTS and is ALREADY FIXED.
`nava_diag_hires_cmp.py` is the clean torch e2e the project needs and it is **better than the
auditor implies**: it loads init **bit-identically from `nava_fx_hires.safetensors`**
(`:38` `hr=load_file(...); lv=hr["in_lat_vid"]`), loads text from the **same
`nava_first_inputs.safetensors`** Mojo reads (`:37`) — so `<extra_id_2>` is present and
identical by construction — runs the **faithful 3-forward/25-step/shift-5** config matching
Mojo line-for-line (`:46-54`), and **saves both the final latent and frames** (`:55` `save_file`).
The handoff's "errored on a missing save_file import" is **STALE** — the import is present
(`nava_diag_hires_cmp.py:11 from safetensors.torch import load_file, save_file`). It just has
not been RUN (GPU busy). The "torch used the WRONG text (missing extra_id_2)" caveat does NOT
apply to this script (both sides read the identical encoded tensor).

## THE ONE decisive measurement (exact command + predictions)

```
# GPU must be free. Produces torch final latent + frames at Mojo's EXACT config/init/text.
cd /home/alex/EriDiffusion/inference-flame/ports/nava && \
HF_HUB_OFFLINE=1 /home/alex/serenityflow-v2/.venv/bin/python nava_diag_hires_cmp.py
# Then two comparisons:
#  (a) PIXEL: diff output/nava_torch_hires/frame_*.png  vs  output/nava_hires/frame_*.png
#  (b) LATENT: cos( Mojo final lv , pipeline/parity/nava_torch_hires_final.safetensors:final_lv )
```
(Mojo side already wrote `output/nava_hires/`; to also dump Mojo's final latent for (b), add a
one-line `save_file` of `lv` before VAE decode in `nava_hires_pipeline.mojo:228`.)

**Predicted outcomes:**
- **CONFIG/usage (my hypothesis):** torch frames are **ALSO melted/smeared**, and
  cos(Mojo_lv, torch_final_lv) **≥ 0.999**. ⇒ the distortion is NOT a port bug; it is the
  17-frame-below-min temporal off-distribution (NEW-1) ± raw prompt (NEW-2). RANK 1 and
  RANK 2 are both wrong. Fix = generate ≥9 (ideally 31) latent frames and/or run the PE
  rewriter; re-decode. No DiT change.
- **Port-numeric (auditor RANK 1, revised to INTERNAL bf16):** torch frames are **CRISP**,
  Mojo melted, and the latent cos **drops below ~0.99**. ⇒ DiT-internal bf16 rounding is the
  culprit; fix = F32 DiT compute (heads/attention/FFN accumulate F32). Only this outcome
  justifies the auditor's direction, and even then it is internal-rounding, not the global
  magnitude drift the audit argued.

## My single best hypothesis
**CONFIG/usage, not a port bug — dominated by NEW-1 (temporal frame count below trained min).**
Evidence: (1) hi-res Mojo lv_std 1.152 ≈ torch 1.143 (gated) means Mojo's latent is
statistically what torch itself produces at this config — decode them and both melt equally;
(2) RNE is already applied yet (per the symptom) output is still distorted, so the per-element
numeric story is spent; (3) RANK 2 is provably inert for spk=None; (4) 17 video frames sits
below `video_min_frames=33`, and the real 5s target is 121 frames / 31 latent — the run is
generating a clip shorter than anything the model/VAE/temporal-RoPE was trained to produce.
The melt lives in the latent CONTENT both implementations generate at an off-distribution
frame count, not in a Mojo↔torch divergence. `nava_diag_hires_cmp.py` settles it in one run.

## ORCHESTRATOR FOLLOW-UP (measured)
- **NEW-1 (too-few-frames) FALSIFIED.** `inference_nava.py:321` default `--frames=5` (latent) → (5-1)*4+1 = **17 video frames** = EXACTLY what the Mojo pipeline generates. `t2av.py:202/321` default `frames=5`. `video_min_frames`/`min_frames` (`pipeline_nava.py:583`, `qwen_vl_utils.py:145`) is i2v INPUT-video frame selection, NOT a generation floor. So 17 frames is the real default, not off-distribution.
- **All 3 hypotheses now falsified by measurement** (magnitude drift / spk-timbre / frame-count). Remaining sole ground truth = the decisive torch e2e at 17 frames (correct text), running as `nava_diag_hires_cmp.py`. Likely outcomes: torch ALSO melts (port faithful; quality gap = Qwen PE rewriter + curation + maybe i2v init) OR torch crisp (real port bug → F32 DiT / internal-op RNE). The demo reel is rewritten+curated+1280×704; a raw-prompt 832×480 single shot is the honest comparison.
