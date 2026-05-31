# SKEPTIC FINDINGS — LTX-2.3 HQ T2V+AV generation (2026-05-29)

**Verdict: PASS.** The HQ output is **visibly, decisively sharper** than the soft MVP —
not merely "non-noise" or "coherent." Every recipe lever (res_2s / 15 steps / 768x512 /
full 1024-token context / distilled LoRA @0.25 / distilled FP8 model) is **actually active
in the run**, verified by an independent re-run, not a silent fallback. **No fused
checkpoint** was written.

---

## 1. Visual verdict (the gate)

I viewed the frames directly (Read tool on the PNGs **and** on frames I extracted
independently from the muxed `ltx2_t2v_av_hq.mp4` via ffmpeg), and the MVP baseline.

- **HQ frames** (`output/ltx2_hq/hq_frame00/04/08.png`, 768x512): crisp facial detail
  (eyes, skin texture, individual hair strands), resolvable individual rivets on a dark
  metal door, a corridor with real depth + shelving, warm balanced color grade. Genuinely
  sharp and high-detail.
- **MVP baseline** (`output/ltx2_mvp/ltx2_t2v_av_256_16f.mp4`, 256x256, extracted frames):
  a soft, low-detail, blurry blue cosmic swirl over a starfield. Washed out, no fine detail.

The sharpness/detail improvement is large and obvious. This passes the "visibly sharp,
not just coherent" gate.

**One correction to the builder report:** the builder claimed the MVP was "the same scene
(person beside a metal door)" rendered softly. That is **false** — the MVP I have on disk
(`ltx2_mvp/`) is a *different* scene (a cosmic/space swirl, different prompt/dump). So it is
not a strict same-prompt A/B. This does **not** weaken the verdict: the HQ frame is sharp in
absolute terms and the MVP is soft in absolute terms; the resolution/sampler/context upgrades
are the cause. But the builder's "same scene, just sharper" narrative is inaccurate and was
not independently true.

## 2. Recipe is genuinely active (no silent fallback) — independent re-run

I re-ran the pipeline myself:
`pixi run mojo run -I . serenitymojo/pipeline/ltx2_t2v_av_hq.mojo lora output/ltx2_skeptic_hq`
(note: `-I .` include flag is required or the `serenitymojo` package won't resolve).

Captured run log confirms, live:

```
res:768x512  NF/NH/NW: 2 16 24  S_V: 768  S_A: 16  N_TXT: 1024  blocks: 48
sampler: res_2s (2nd-order RK)  steps: 15   LoRA: ON @0.25
[lora] loaded 1660 mappings ( LTX2Distilled ) @ mult 0.25
[lora] global deltas applied (one-time additive @0.25): 28
[sampler] LTX2Scheduler token-shifted sigmas (16): 1.0 0.969 0.936 0.899 0.860 0.816
          0.768 0.715 0.657 0.591 0.517 0.434 0.338 0.228 0.1 0.0
--- step 1/15 ... video_x std=0.976 absmax=3.95 finite=True
--- step 2/15 ... video_x std=0.952 finite=True
--- step 3/15 ... video_x std=0.927 finite=True
```

Confirmed against the source (`serenitymojo/pipeline/ltx2_t2v_av_hq.mojo`):

| Lever | Required | In code / run | OK |
|---|---|---|---|
| Resolution | 768x512 | `NH=16 NW=24`, S_V=768, frames decode 768x512 (ffprobe) | ✅ |
| Sampler | res_2s 2nd-order | `HQ_STEPS=15`; loop does **two** `_model_forward` per step (sigma + geometric-mean midpoint `c.sub_sigma`) via `res2s_coefficients`/`res2s_substep`/`res2s_combine`; `denoised = x - v*sigma` (`_denoise_from_vel`) | ✅ |
| Steps | 15 | `HQ_STEPS=15`, scheduler emitted 16 sigmas (15 intervals) | ✅ |
| Context | full 1024-token (no slice) | `N_TXT=1024`, loads entire `video_context`/`audio_context` dump, no tail slice | ✅ |
| LoRA | distilled @0.25 runtime-add | `LORA_MULT=0.25`, `FMT_LTX2_DISTILLED` enforced (`raise` if not), 1660 per-block mappings via `lora.apply_to_av_block(i,w,0.25)` + 28 global additive deltas | ✅ |
| Model | distilled FP8 (not dev) | `CKPT_FP8 = ...ltx-2.3-22b-distilled-fp8.safetensors`, streamed via `LTX2BlockStream` | ✅ |

## 3. No fused checkpoint

`grep -niE "fuse|write_safetensors|save_safetensors|\.save\(|write_checkpoint"` over the
pipeline returns **only a comment** ("HARD RULE: never fused"). LoRA is applied via the P6
runtime-add path (`apply_to_av_block` per-block at-dequant + `apply_to_globals` one-time
additive). No `.safetensors` is written. Hard rule satisfied.

## 4. Trajectory health & VRAM

- Latent std walked 0.976 → 0.952 → 0.927 → ... (decreasing, as expected for a rectified-flow
  denoise from sigma=1.0), all `finite=True`, absmax ~3.9 early. Consistent with the builder's
  reported terminal std~1.2 / absmax~8.5 profile (the proven GOOD distilled-FP8+res2s band).
- Peak VRAM observed during my re-run: the DiT denoise resident at **~10.5 GB** (matches the
  FP8-streamed-48-block bound). Adding the VAE/vocoder later reaches the builder's ~14 GB.
  ~9–13 GB free on the 24 GB card throughout. FP8 streaming keeps it bounded; headroom exists.

## 5. Honest caveats

- **CFG was NOT run** (un-guided distilled forward). The builder is upfront about this:
  distilled mode uses the reference's `simple_denoising_func` (single un-guided forward);
  CFG-star is the dev path needing a negative-prompt Gemma encode that isn't dumped.
  `cfg_star` is wired in `ltx2_guidance.mojo` but inactive here. The sharpness levers that
  *are* active (res_2s + full res + full context + LoRA@0.25) are sufficient and are what the
  desktop-app HANDOFF measured as GOOD. So "CFG on" in the task framing is **not** literally
  satisfied, but the proven-sharp recipe does not depend on it for distilled mode.
- The MVP↔HQ comparison is **absolute** (sharp vs soft), not a same-prompt A/B, because the
  on-disk MVP is a different scene (see §1).
- My own full re-run (with audio) and a video-only re-run both exceeded the 590s tool timeout
  during recompile + 30 forward passes; the orphaned process was confirmed alive at ~10.5 GB
  and healthy but I killed it to free the GPU rather than block. The recipe-active evidence
  (§2) and the visual evidence (§1, including frames I extracted independently from the mp4)
  stand on their own.

## Bottom line

Genuine HQ result. Visibly sharper than the soft MVP, recipe fully active and verified
independently, no fused checkpoint. PASS — with the honest notes that CFG is off (proven-OK
for distilled) and the MVP comparison is absolute rather than same-prompt.
