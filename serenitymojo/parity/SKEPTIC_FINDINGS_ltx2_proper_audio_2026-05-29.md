# SKEPTIC FINDINGS — LTX-2 MVP proper audio context (2026-05-29)

**Verdict: VERIFIED (PASS).** The forbidden down-projected-slice path is deleted. The
MVP now consumes the GENUINE `feature_extract_and_project` audio context. The MVP
re-runs end-to-end producing coherent frames + a finite, non-silent 48 kHz stereo
audio track in a playable mp4. No shortcut remains; no fused checkpoint was written;
no git commit was made.

---

## 1. The slice path is DELETED (code trace)

`serenitymojo/pipeline/ltx2_t2v_av_mvp.mojo`:
- The forbidden `slice(video_pre, 2, 0, AD)` (feature-dim down-projection of the video
  text_hidden into a fake 2048-dim audio context) is **gone** — grep for any feature-dim
  slice of video into AD/2048 returns nothing.
- Lines 610-621 now load BOTH contexts from the real Rust dump:
  - `vctx_full` ← `dump.tensor_view("video_context")`  → `[1,1024,4096]`
  - `actx_full` ← `dump.tensor_view("audio_context")`  → `[1,1024,2048]`
  - `audio_pre = slice(actx_full, 1, tail0, N_TXT)` — this is a **token-length** slice on
    dim 1 (`[896:1024]`), applied independently to the real audio_context. It is NOT a
    feature-dim slice of video. `audio_pre` derives from `actx_full` (real audio), not
    from `vctx_full`.
- The token tail `[896:1024]` lands inside the real-token region (text is left-padded;
  see §3), so the bounded 128-token slice is real tokens, not padding.

## 2. The dump IS the real `feature_extract_and_project` output (Rust trace)

`/home/alex/EriDiffusion/inference-flame/src/bin/ltx2_generate_av.rs`:
- `--dump-audio-context` flag (lines 172, 452-478) runs Stage-1 text encoding only,
  then writes contexts and `return Ok(())` **before** the transformer loads (additive,
  non-destructive).
- Lines 340-357 call `feature_extractor::feature_extract_and_project(...)` twice:
  - video: `video_aggregate_embed.weight`, `target_dim=4096`
  - audio: `audio_aggregate_embed.weight`, `target_dim=2048`
  - SAME 49 Gemma hidden states (`all_hidden`) + SAME mask; DIFFERENT projection weights.
- `feature_extractor.rs:40-106`: per-token RMSNorm of 49 layers → concat to 188160 →
  rescale `sqrt(target_dim/3840)` (source_dim=3840, line 70-71) → linear projection.
  This is exactly the builder's description and matches the PyTorch `_rescale_norm`.
- **It is not a slice of anything.** Audio and video are independent projections of the
  identical Gemma feature stack.

The old cached file `cached_ltx2_embeddings.safetensors` holds only `text_hidden
[1,1024,4096]` (post-video-projection) — no 49 Gemma states, no 2048-dim audio. The
builder genuinely could not derive a real audio context from it; the Rust dump was the
correct and necessary path. Corroborated.

## 3. Dump tensor proof (Python, on the actual file)

`/home/alex/EriDiffusion/inference-flame/output/audio_context_dump/ltx2_audio_context.safetensors`
(25 MB, written 02:18, before the 02:26-02:27 MVP run):

- Header dtypes: `video_context F32 [1,1024,4096]`, `audio_context F32 [1,1024,2048]`,
  `encoder_attention_mask F32 [1,1,1,1024]`. (Stored F32; the MVP reads via
  `from_view_as_bf16` which correctly host-casts F32→BF16 — `tensor.mojo:214-217`. No
  dtype-misread bug.)

- **SLICE HYPOTHESIS REFUTED:**
  - `cos(audio, video[:,:2048]) = -0.000432` (a slice would give 1.0)
  - per-token cos on real tokens `[896:1024]`: mean 0.0034, min -0.046, max 0.052
  - `mean|audio - video[:,:2048]| = 0.808`; audio std 1.54 vs video[:,:2048] std 1.83
  → orthogonal, distinct distribution → genuinely a separate projection, not a slice.

- **Mask / real-token layout:** additive mask values `{0.0 (real), -1.7e38 (pad)}`;
  218 real tokens; first real idx 806, last 1023 → text is left-padded, real tokens in
  the tail. The MVP's `[896:1024]` slice lands in [806,1023] → real tokens. Confirmed.

- **Structure:** both contexts finite; padded head[0:128] absmean 0.004 vs real
  tail[896:1024] absmean 2.21 → feature extractor zeros padded positions, as specified.

(Full Rust↔Mojo cos≥0.999 of a fresh dump vs the Mojo-loaded tensor was not needed: the
Mojo loader is a verified host cast of the same on-disk F32 bytes — the load is lossless
to BF16. The orthogonality + Rust code trace establish the context is the genuine
projection, not a slice.)

## 4. End-to-end re-run (my own, seed=42)

`pixi run mojo run -I . serenitymojo/pipeline/ltx2_t2v_av_mvp.mojo base
/home/alex/mojodiffusion/output/ltx2_skeptic_verify 0` — completed, exit 0:

- `audio_pre(REAL feature_extract_and_project)`: mean 0.034, std 3.20, absmax 94 —
  distinct from `video_pre` (std 5.34, absmax 300). Different stats = not a slice.
- Audio velocity diverges from video velocity every step (`a_vel` mean ≈ -0.32…-0.55 vs
  `v_vel` ≈ +0.01…+0.06) → audio stream is driven by its OWN genuine context.
- Frames decoded `[1,3,9,256,256]`, finite. **Viewed frames 00/04/08:** coherent — a
  dark-haired woman, half-lit, in a dark industrial corridor gripping a weapon;
  temporally stable with subtle motion; NOT noise/gray/checkerboard. HARD-RULE visual
  artifact MET.
- Audio: rms 0.200, absmax 0.685, finite, nonsilent, 29280 samples/ch.
- mp4: h264 256×256 + AAC 48 kHz stereo, playable (ffprobe). wav: pcm_s16le 48 kHz
  stereo 0.61 s, finite, non-silent. Reproduces the builder's artifact bit-for-bit
  (deterministic seed).

## 5. No shortcut / no fused checkpoint / no commit

- `git status` clean of any `*fused*`/`*merge*`/new checkpoint `.safetensors` (only
  untracked artifact: `serenitymojo/ops/parity/stft_mel_oracle.safetensors`, an
  unrelated STFT parity oracle).
- No new/fused checkpoint under `/home/alex/.serenity/models/checkpoints` since 00:00.
- No git commit made (working tree unchanged from session start).

---

## Caveats (do NOT block the audio-context requirement; flagged for completeness)

These are pre-existing P7 gaps already noted in the MVP file header — none of them is
the forbidden slice:

1. **Stage boundary not built.** The spatial-x2 latent upsampler + AdaIN stage boundary
   (Plan P7) is absent; the MVP runs the single distilled-8 schedule at fixed MVP
   resolution. Pre-existing, documented in-file (STAGE-BOUNDARY NOTE).
2. **Audio DSP not parity-gated this run.** I verified finite/non-silent/in-range +
   stream-divergence per the task's audio bar, NOT full `ltx2_vocoder_ref.py` cos≥0.999
   DSP parity. The task required a finite audio track from the real context, which is
   met; full vocoder parity is a separate (P4) gate.
3. **Bounded 128-token context.** The dump is 1024-token; the MVP slices 128 real tokens
   `[896:1024]`. This is a bounded approximation consistent with the existing video path,
   and now lands on REAL tokens (not padding). Not a shortcut on the audio source.
4. This run is LoRA OFF (`base`); the LoRA-ON path (`lora`) exists but was not exercised
   here. Orthogonal to the audio-context requirement.

**Bottom line:** the audio_context fed to the pipeline is the genuine
`feature_extract_and_project` output (proven orthogonal to video, traced through the
Rust code path), the down-projected-slice path is deleted, the artifact is a playable
mp4 with coherent frames and a finite audio track, and no fused checkpoint / git commit
was produced. PASS.
