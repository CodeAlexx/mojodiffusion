# SKEPTIC FINDINGS — LTX2 P7 MVP capstone (2026-05-28/29)

**Verdict: PASS — independently confirmed. The builder did NOT over-claim.**

The P7 hard rule — a coherent decoded artifact (visually coherent video frames AND a
finite, non-silent audio track, muxed into a playable mp4) — is met. Verified by reading
the builder's artifacts, then re-running the pipeline independently on GPU and viewing
MY OWN decoded frames. The pipeline is fully deterministic: my frame00.png is
**byte-identical (md5 88a1cc32…)** to the builder's, so the result is reproducible, not
a hand-picked or doctored output.

---

## 1. Wiring vs `ltx2_generate_av.rs` — FAITHFUL

The reference binary's main loop (`ltx2_generate_av.rs:586-676`) is a **single-stage**
joint AV Euler denoise over `LTX2_DISTILLED_SIGMAS`. There is NO spatial-x2 upscaler /
AdaIN / stage boundary in the Rust binary itself — the "two-stage" is the sigma schedule
(the 8-step distilled table descends through 0.909375→0.725→0.421875→0, which ARE the
"stage-2" sigmas). The builder's "MVP simplification" (no dedicated upsampler stage) is
therefore **not a deviation from the reference binary** — `ltx2_generate_av.rs` doesn't
do a separate upscale stage either. (The separate `LTX2_STAGE2_DISTILLED_SIGMAS[4]` const
exists in the Rust sampler module but is NOT used by this generation binary.)

| Aspect | Rust ref | Mojo MVP | Match |
|---|---|---|---|
| Latent shapes | video `[1,C,lf,lh,lw]`, audio `[1,AC,af,mel]` | video `[1,128,2,8,8]`, audio `[1,8,16,16]` | ✓ |
| Noise seeds | `SEED` / `SEED+1` | 42 / 43 | ✓ (RNG differs — see §5) |
| Sigma table | `LTX2_DISTILLED_SIGMAS` (9 vals, 8 steps) | bit-exact in `ltx2_sampling.mojo` | ✓ |
| Forwards/step | 1 (distilled: do_cfg=false, do_stg=false) | 1 | ✓ |
| Euler step | `σ_next==0 → x−v·σ` else `x+v·(σ_next−σ)` | identical (`LTX2Scheduler.step`) | ✓ |
| Forward order | patchify→adaln→48 blocks→proj_out→euler | identical | ✓ |
| CFG-star / STG | gated off in distilled path (cfg=1, stg=0) | not wired (correct — distilled needs neither) | ✓ |
| Decode order | video VAE + audio VAE→vocoder, then mux | identical | ✓ |

CFG-star rescale and STG skip-layer are correctly ABSENT from the MVP: the distilled
path runs `do_cfg=false, do_stg=false` (cfg_scale=1, stg_scale=0), so the bare 1-forward
Euler loop is exactly what `ltx2_generate_av.rs` executes in distilled mode. Not a gap.

Connector: the MVP feeds `ltx2_connector_forward` output directly as context with no
separate `caption_projection`. This is **correct for this checkpoint** — the verified
connector module documents that the ComfyUI LTX-2.3 distilled ckpt has 0 `caption_projection`
keys; the connector IS the projection. (The plan's P2.5 prose mentioning a separate
caption_projection is superseded by the session's checkpoint audit.)

Verified modules untouched: `git status` shows NO modifications to `ltx2_vocoder.mojo`,
`ltx2_sampling.mojo`, or `ltx2_dit.mojo`. Only the new MVP pipeline file is untracked.
The builder's claim of "verified module left untouched, debug prints reverted" holds.

---

## 2. Video coherence — COHERENT (independent eyeball verdict)

Viewed frames 0,2,4,6,8 from MY OWN re-run. This is a **real scene, not a plausible blob**:
- A glowing translucent disc/saucer top-right with **concentric ring/striation detail** —
  real internal structure, not a smeared gradient.
- A bright diagonal **purple light beam** crossing the frame with directional falloff.
- A **colored starfield** (discrete blue and yellow points) over a black background.
- Real depth, lighting, and a consistent palette. No NaN, no gray collapse, no checkerboard,
  no per-frame scene change.

Temporal stability (quantified, not just asserted): frames 0,4,8 have **distinct md5 hashes**
(NOT frozen duplicates) and **SSIM(frame0,frame8)=0.799** — high enough that it is the SAME
scene throughout (a per-frame regeneration would give SSIM≈0), low enough that there is
**genuine subtle motion**. This is exactly what a coherent, temporally-stable clip looks like.

Frame stats from my run: mean −0.43, std 0.62, absmax 1.36, finite. Matches builder report.

---

## 3. Audio — FINITE, NON-SILENT (independent verdict)

`ffmpeg astats` on MY wav (independent of the in-Mojo rms print):
- Number of samples: 29280/ch, 48 kHz stereo, 0.61 s.
- **Flat factor: 0.000** → no silent/constant runs (not a DC track, not zeros).
- RMS level −13.7 dB, Peak −2.9 dB, 2177 zero-crossings → real oscillating waveform.
- DC offset ≈ 0.00015, finite, no NaN. In-Mojo: rms=0.206, absmax=0.716.

Audio passes the hard rule (finite + non-silent). HONEST CAVEAT, correctly documented by
the builder in-file: there is no cached audio context and no in-Mojo Gemma encoder, so the
2048-dim audio pre-connector context is DERIVED by slicing the first 2048 dims of the cached
video `text_hidden`. Audio is therefore AV-coupled through the joint denoise and finite/
non-silent, but **NOT prompt-faithful**. The P7 hard rule asks for finite/non-silent, which
is met; prompt-faithful audio requires the deferred Gemma encoder (out of MVP scope).

---

## 4. mp4 playability — VALID CONTAINER (ffprobe-verified)

Both streams present and well-formed:
- **Video:** H.264 (High profile), 256×256, yuv420p, 9 frames, 25 fps, dur 0.36 s.
- **Audio:** AAC-LC, 48 kHz, stereo, 17 frames, dur 0.341 s.
- Container: mov/mp4, 2 streams, probe_score 100.

Playable, correct dimensions/fps, has BOTH a video and an audio stream. The ~0.36 s
duration is correct for 9 frames at 25 fps.

---

## 5. Chain-drift (0.9947 video velocity) — does NOT manifest visually

Per-step velocity + latent stats across all 8 distilled steps (my run) are finite and
stable: video_x std contracts 0.99→0.985→0.98→0.974→0.92→0.837→0.90→1.28; v_vel std
1.21→1.39 (bounded, no blowup); audio_x std stable 1.0→0.84→1.0→1.5; a_vel std ~1.13–1.37.
No NaN, no monotone divergence, no exploding absmax (v_vel absmax peaks 7.3, bounded).

Visually: the frames are clean and temporally stable (SSIM 0.799 across the full 9-frame
chain, distinct frames with smooth motion). The known 0.9947 video-velocity chain-drift
**does NOT produce visible degradation** — no accumulating noise, banding, or collapse
across the 8-step chain. Confirmed.

---

## 6. VRAM — confirmed independently

Sampled `nvidia-smi` during my run: peak **~10.0 GB** resident (10006 MiB), 24 GB card,
no OOM, exit code 0. Matches the builder's ~10.0 GB claim. FP8 block streaming keeps the
48-block stack within budget (boundary blocks BF16, inner FP8 dequant-on-use).

---

## 7. The one bug the builder reported — confirmed benign + correctly handled

The builder cast the audio-VAE mel to F32 before `voc.forward` to dodge a BF16
`activation1d`/`snake_beta` dtype mismatch in the vocoder's AMP path, noting the vocoder's
gated/verified path runs F32 (its smoke loads mel via `from_view_as_f32`). My run used the
same F32 path and produced finite, non-silent audio (Flat factor 0, rms 0.206). The fix is
in the MVP file only; `ltx2_vocoder.mojo` is unmodified per git status. Correct and honest.

---

## Bottom line

Builder report is ACCURATE, not over-claimed. The pipeline wiring matches
`ltx2_generate_av.rs` (single-stage distilled-8 joint AV Euler, correct shapes/sigmas/step/
decode order; CFG-star+STG correctly omitted for the distilled path). I re-ran it
independently on GPU: deterministic, byte-identical frame output, coherent video (real
sci-fi scene, temporally stable with genuine motion), finite non-silent audio, playable
2-stream mp4, ~10 GB peak. The 0.9947 chain-drift does not degrade the result visually.

P7 MVP capstone: **PASS.**

Caveats (all documented by the builder, none invalidate the gate):
- Audio is AV-coupled + finite but NOT prompt-faithful (derived audio context; Gemma encoder deferred).
- No dedicated spatial-x2 upscale stage — but the reference binary doesn't run one either, so this is faithful, not a shortcut.
- Latents are not bit-comparable to a Rust run (Mojo `randn` seed vs Rust Box-Muller `make_noise`); the P7 gate is the coherent artifact, not Rust latent cos, so this is acceptable for the MVP.
