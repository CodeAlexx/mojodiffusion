# LTX2 Parity Matrix - 2026-06-04

Purpose: make the LTX2 Mojo render engine and trainer prove themselves against
reference behavior before we call anything production.

Runtime rule: Mojo only. Python is allowed here only as dev/oracle tooling.

## Runner

```bash
python3 scripts/ltx2_parity_gate.py --mode fast
python3 scripts/ltx2_parity_gate.py --mode full
python3 scripts/ltx2_parity_gate.py --only attention_tiled_cross,audiosync_profile_contract
python3 scripts/ltx2_parity_gate.py --only trainer_masked_av_loss,trainer_audio_ref_ic
python3 scripts/ltx2_parity_gate.py --only dit_forward_48_block
python3 scripts/ltx2_parity_gate.py --mode full --area inference
python3 scripts/ltx2_parity_gate.py --mode full --area training
python3 scripts/ltx2_parity_gate.py --only render_staged_hq_smoke
```

Use `--refresh-oracles` only when intentionally regenerating Python reference
dumps. Missing oracle dumps fail by default.

## Current Proven Gates

- Core attention: online-softmax tiled SDPA covers square nomask, masked, and
  rectangular cross-attention; the rectangular path matches the old padded-mask
  oracle and avoids square padding for LTX2 text/cross-modal attention.
- Sampler/guidance: distilled schedules, CFG-star, STG masks/rescale, NAG
  combine, deterministic res2s, and HQ res2s SDE+bong step.
- Workflow fixtures: local Comfy AudioSync and DEV-HQ workflow JSON settings
  are parsed and asserted as dev/oracle northstars. The Mojo AudioSync profile
  contract asserts `97` frames, `24fps`, `768x512` stage-1, `1536x1024`
  stage-2, `6.5s` audio, `20` scheduler steps, and staged token pressure. The
  HQ runner now has a first-class `audiosync` mode for the 97-frame single-stage
  geometry.
- LoRA: official distilled LoRA mapping/apply/add math, HQ stack block-0 apply
  for distilled, camera-static, detailer, and local Musubi LoRA, factorized
  runtime attachment for the production HQ stack, and synthetic factorized math
  parity against materialized `W + scale*(B@A)`.
- DiT: AV block-0 parity and full 48-block AV velocity parity.
- Decoders/audio: video VAE, audio VAE, BigVGAN+BWE vocoder parity.
- Offload: 48 blocks streamed with bounded single-resident memory; resident
  raw-FP8 blocks materialize BF16 through a resident-only no-sync dequant path
  so repeated denoise evals avoid per-FP8-tensor host/device fences.
- Render structure: staged HQ smoke builds and runs stage-1 res2s, spatial x2
  latent upsample, bounded stage-2 refine, and high-resolution video decode.
- Guidance integration: video-side NAG loads cached null Gemma context, projects
  it through the video connector, and runs the NAG-aware AV block path. Audio
  NAG remains blocked until a proper audio null context exists.
- Trainer: AV trainer foundation contracts pass, and production training remains
  fail-closed until full AV backward and train-time AV LoRA runtime exist.
  Musubi LTX2 audio bucket policy, masked AV loss semantics, and
  audio-reference IC conditioning layout are Mojo-owned and gated.

## Latest Evidence

- `python3 scripts/ltx2_parity_gate.py --mode fast --fail-fast`: 12/12 pass,
  including tiled rectangular attention, workflow fixture, AudioSync profile,
  NAG combine, factorized LoRA math, trainer fail-closed foundation gates, and
  Musubi masked video/audio loss plus audio-reference IC parity.
- `output/bin/fp8_dequant_smoke`: 5/5 pass after splitting the per-tensor FP8
  wrapper; the synchronized public API remains bit-exact for all 256 E4M3 byte
  values at several scales and for the real block-4 `attn1.to_q` torch
  reference slice.
- `output/bin/ltx2_fp8_resident_smoke`: pass; block 4 preloads
  `386924928` resident bytes (`369 MiB`), exposes `34` FP8 tensors, and
  materializes representative video/audio weights as BF16 through the
  resident-only no-sync dequant path before a final synchronization.
- `pixi run mojo run -I . serenitymojo/ops/sdpa_tiled_probe.mojo`: pass;
  square nomask and masked tiled SDPA both match math-mode at cos `1.0`,
  rectangular cross-attention `Sq=1536, Skv=1024, Dh=128` matches padded-mask
  reference at cos `1.0`, and large `S=8192, H=2, Dh=128` completes finite.
- `pixi run mojo run -I . serenitymojo/pipeline/ltx2_audiosync_profile_smoke.mojo`:
  pass; AudioSync contract is `97` frames at `24fps`, stage-1 video tokens
  `4992`, stage-2 video tokens `19968`, fixture audio tokens `162`, and naive
  stage-2 F32 attention scores would be `48672 MiB`.
- `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/ltx2_t2v_av_hq.mojo -o /tmp/ltx2_hq_build_check`:
  pass after routing LTX2 AV attention through rectangular online-softmax
  cross-attention, switching the HQ runner/mux profile to `24fps`, adding the
  97-frame `audiosync` runner shape, and forcing incomplete denoise runs to use
  `dev_frame*` / `dev_smoke` artifact names.
- `python3 scripts/ltx2_parity_gate.py --mode full --area inference/lora --fail-fast`:
  4/4 pass, including distilled surface, HQ stack surface, factorized surface,
  and factorized math.
- `python3 scripts/ltx2_parity_gate.py --only trainer_masked_av_loss --fail-fast`:
  pass; covers Musubi broadcast masks for video `[B,F]`/5D and audio `[B,T]`/4D,
  all-false mask fallback to unmasked mean, and `mse`/`mae`/`huber`.
- `python3 scripts/ltx2_parity_gate.py --only trainer_audio_ref_ic --fail-fast`:
  pass; covers Musubi `audio_ref_only_ic` reference concat, zero reference
  timesteps, zero reference targets, false reference loss mask, separate target
  positions, optional negative reference positions, A2V reference masking, and
  reference-from-text masking.
- `python3 scripts/ltx2_parity_gate.py --only trainer_parity_audit --fail-fast`:
  pass after importing the masked-loss contract into the audit.
- `python3 scripts/ltx2_parity_gate.py --mode full --area training --fail-fast`:
  6/6 pass, including readiness, acceptance, standalone masked AV loss parity,
  audio-reference IC parity, trainer parity audit, and legacy/real trainer build.
- `python3 scripts/ltx2_parity_gate.py --mode full --fail-fast`: previous 14/14
  pass before the new workflow/math gates were added; rerun full sweep after
  full Comfy workflow oracle lands.
- `python3 scripts/ltx2_parity_gate.py --only dit_forward_48_block --fail-fast`:
  pass after the rectangular attention rewrite; video velocity cosine
  `0.99478066`, audio velocity cosine `0.9998851`.
- `python3 scripts/ltx2_parity_gate.py --only av_block0 --fail-fast`: pass
  after the rectangular attention rewrite; block-0 video cosine `0.9999943`,
  audio cosine `0.99999666`, and V2A-delta cosine `0.9999951`.
- `python3 scripts/ltx2_parity_gate.py --only lora_hq_stack_surface --fail-fast`:
  pass; block-0 stack applied `82` LoRA deltas.
- `python3 scripts/ltx2_parity_gate.py --mode full --area inference/lora --fail-fast`:
  pass; includes `lora_factorized_surface` with 70 HQ block-0 A/B factors
  attached without materializing full-rank deltas.
- `python3 scripts/ltx2_parity_gate.py --only comfy_workflow_fixture --fail-fast`:
  pass; asserts AudioSync `97` frames at `768x512`, `24fps`, scheduler `[20, 2.05,
  0.95, terminal=0.1]`, LoRA strengths `distilled@1.0`,
  `camera-static@0.3`, `detailer@0.6`, `6.5s` audio, audio VAE encode,
  MelBandRoFormer, and spatial upscaler.
- `python3 scripts/ltx2_parity_gate.py --only lora_factorized_math --fail-fast`:
  pass; bias max_abs `2.9802322e-08`, no-bias max_abs `5.9604645e-08`.
- `pixi run mojo run -I . serenitymojo/training/parity/ltx2_audio_bucket_parity.mojo`:
  pass; covers Musubi `pad`/`truncate`, 25fps bucket quantization, audio/non-audio
  bucket split, quota sampler, and probability sampler policy.
- Obsolete NAG one-step smoke artifact:
  `/home/alex/mojodiffusion/output/ltx2_hq_nag_single_smoke/ltx2_t2v_hq_fast.mp4`
  is 9 frames, `768x512`, `25fps`, `0.36s`, video-only, and visually denoised
  noise. It is not a quality sample. The runner now prevents this class of
  incomplete-denoise output from being named `hq_fast`/`hq_frame*`.
- Patched NAG one-step smoke naming check:
  `/tmp/ltx2_hq_build_check single lora stream noaudio nag output/ltx2_nag_dev_name_check 1`
  completed finite and wrote only `dev_frame00.png` ... `dev_frame08.png` plus
  `ltx2_t2v_dev_smoke.mp4`. MP4 metadata is `768x512`, `24fps`, `0.375s`,
  9 frames, video-only. This is still deliberately not quality evidence.
- Staged HQ smoke:
  `/tmp/ltx2_hq_stack_safe2 staged lora stream noaudio output/ltx2_hq_factorized_safe_staged_smoke 1`
  completed in `8:57.37`, peak RSS `19325700KB`, wrote 9 frames at `1536x1024`
  and an old `ltx2_t2v_hq2_fast.mp4` name. This proves staged mechanics and
  memory, not final quality, because it runs one denoise step per stage.
  Current builds write `dev_frame*` and `ltx2_t2v_stage2_dev_smoke.mp4` for
  incomplete staged runs.
- Full safe 25-frame A/V render:
  `/tmp/ltx2_hq_stack_safe2 long lora stream audio output/ltx2_hq_factorized_safe_long_audio 0`
  completed in `25:07.54`, peak RSS `18659784KB`, wrote `25` frames at
  `768x512`, `ltx2_t2v_av_hq.mp4`, and `hq_audio.wav`. MP4 metadata:
  video `25fps`, duration `1.000s`, AAC stereo audio duration `0.981s`.
  Audio check: WAV mean volume `-25.4 dB`, max `-7.5 dB`, nonsilent. Visual
  check: frame 0 is coherent and much cleaner with detailer disabled; mid/end
  frames still blur/warp under motion, so this is progress but not final
  production acceptance.
- Video NAG smoke:
  `/tmp/ltx2_hq_nag_check single lora stream noaudio nag output/ltx2_hq_nag_single_smoke 1`
  completed in `2:29.55`, peak RSS `18663752KB`, loaded
  `cached_ltx2_negative.safetensors::text_hidden`, projected it through the
  video connector, ran NAG-aware AV blocks, and wrote 9 finite frames plus a
  video-only MP4. This proves video-side NAG integration, not final quality.
- Full one-step HQ stack render smoke:
  `/tmp/ltx2_hq_stack_check single lora stream audio output/ltx2_hq_factorized_memcpy_single_smoke 1`
  completed in `2:26.26`, wrote 9 PNG frames and a video-only MP4 under the old
  naming. This is a runtime smoke, not a quality sample; current builds name
  incomplete one-stage runs `ltx2_t2v_dev_smoke.mp4`.
- Core tensor load path now stages safetensors views with libc `memcpy` instead
  of a byte-by-byte Mojo host loop; this is required for multi-GB LoRA factors
  and benefits all model weight loading.

## Northstar Gaps

- Full ComfyUI workflow parity: prompt/negative context, NAG wiring, HQ sampler,
  LoRA stack strengths, IC/detailer conditioning, VAE/vocoder, and mux in one
  oracle.
- Full numeric multi-LoRA forward parity across all 48 blocks, not just
  coverage/apply.
- Audio-reference IC conditioning is contract-gated; image/control IC
  conditioning and full runtime integration remain open.
- Final render acceptance against the reference class of clips: duration,
  resolution, temporal stability, face/detail quality, audio presence, loudness,
  and A/V coherence.
- Trainer production parity: AV backward, train-time AV LoRA/DoRA/control
  runtime, GPU/backward masked-loss integration, checkpoint/resume trajectory
  parity, validation sampling, and GPU-heavy hot-loop proof.
