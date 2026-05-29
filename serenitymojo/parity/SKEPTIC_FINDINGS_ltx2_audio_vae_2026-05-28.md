# SKEPTIC FINDINGS — LTX-2.3 Audio VAE decoder (P4)

Date: 2026-05-28
Skeptic target: `audio_vae` (full decoder)
Verdict: **PASS CONFIRMED — gate is real and fail-closed.**

## Scope audited
- `serenitymojo/models/vae/ltx2_audio_vae.mojo` (pure-Mojo decoder, `decode`)
- `serenitymojo/pipeline/ltx2_audio_vae_smoke.mojo` (parity gate)
- `scripts/ltx2_audio_vae_ref.py` (PyTorch oracle)
- vs Rust spec `inference-flame/src/vae/ltx2_audio_vae.rs`
- vs OFFICIAL Python `ltx_core/model/audio_vae/{audio_vae,causal_conv_2d,upsample,resnet}.py`
  and `common/normalization.py` (found at /home/alex/ltx2-official-ref/...)

## Baseline gate (reproduced on RTX 3090 Ti this session)
- regenerated oracle, rebuilt smoke clean (no warnings), ran:
  - latent `[1,8,8,16]` -> decoded NCHW `[1,2,29,64]` (T_out=4*8-3=29, F_out=64)
  - **cosine = 0.99999595** (gate >= 0.999): PASS
  - max_abs_diff = 0.03125 (one BF16 ULP at this magnitude)
  - Mojo decoded mean=-4.442 std=1.172 vs ref mean=-4.441 std=1.170 — matched INDEPENDENTLY
    (not a degenerate constant/self-compare; log-mel range, non-trivial std).

## Fail-closed verification (MUTATE + confirm gate FAILS) — 3 independent paths
All mutations were applied to the Mojo decoder, rebuilt, run, then REVERTED.
Baseline restored to cos=0.99999595 after reverts.

1. **Causal padding direction** — top-pad (causal) -> bottom-pad (non-causal) on
   the time axis in `_causal_conv2d_named`.
   -> cos = 0.9514, gate FAILED (raised). Confirms causal-time padding is exercised.

2. **un_normalize statistics** — dropped the `+mean` term (kept only `*std`).
   -> cos = 0.9762, gate FAILED. Confirms the 128-dim per-channel stats path is exercised.

3. **Upsample frame-drop** — drop LAST time frame instead of FIRST
   (`slice(y,1,1,..)` -> `slice(y,1,0,..)`). Output shape stayed `[1,2,29,64]`.
   -> cos = 0.9887, gate FAILED. Confirms the gate catches correct-shape/wrong-content
   errors, and the drop-first semantic is the load-bearing one.

Conclusion: the gate is genuinely numeric, not trivially passing. A wrong decoder
fails it.

## Structural audit vs OFFICIAL ltx_core (anchored, not co-invented)
Checkpoint `audio_vae.decoder.*` = 56 tensors + 2 stats = 58 (matches builder claim).
Confirmed against the checkpoint header AND the official Python:

- conv_in [512,8,3,3] (8->512); conv_out [2,128,3,3] (128->2; stereo). ✓
- mid: block_1, block_2 both 512->512; attn_1 = Identity (NO mid-attn weights in ckpt). ✓
- up stages, forward iterates `reversed(range(3))` -> up[2]->up[1]->up[0]; upsample
  only when `level != 0` (so up[0] has NO upsample). ✓
    - up[2]: 3x [512,512]; upsample [512,512]
    - up[1]: block.0 [256,512] + nin_shortcut [256,512,1,1], then [256,256]x2; upsample [256,256]
    - up[0]: block.0 [128,256] + nin_shortcut [128,256,1,1], then [128,128]x2; NO upsample
  (3 blocks/stage = num_res_blocks+1; nin_shortcut only on the channel-changing block.0
   of up.0/up.1 — matches the Mojo probe candidates). ✓
- CausalConv2d (causality_axis=HEIGHT): `F.pad = (pad_w//2, pad_w-pad_w//2, pad_h, 0)` =
  symmetric on freq(W), all on TOP for time(H), ZERO pad. Matches Mojo (manual zero
  top-pad on conv3d D axis + symmetric conv pad on H/freq, singleton W). ✓
- Upsample (HEIGHT): nearest x2 -> conv -> `x[:, :, 1:, :]` (drop FIRST height/time
  frame; official docstring derives this explicitly). Matches Mojo. ✓
- ResnetBlock: norm1->silu->conv1->norm2->silu->conv2; skip = nin_shortcut(x) if in!=out
  else x; temb=None; dropout=identity in eval. Matches Mojo `_resnet_block`. ✓
- PixelNorm: `x / sqrt(mean(x^2, dim=1, keepdim) + 1e-6)`, no affine weights
  (NormType.PIXEL, eps overridden to 1e-6). The checkpoint has ZERO norm weight/bias
  tensors -> confirms PixelNorm (weightless), not GroupNorm. Matches Mojo rms_norm
  over last/channel dim with ones-gamma, eps inside sqrt. ✓
- T_out: official `target_frames = frames*4 - (4-1) = 29` (causal); `_adjust_output_shape`
  crop/pad is a no-op for this shape since the native decode already yields 29. F_out=64.
  Matches. ✓

## Notes / residual caveats (non-blocking)
- The oracle stores BF16 activations between major ops (matches the Mojo BF16 storage
  path); convs accumulate F32. The 0.03125 max-abs-diff is BF16 quantization, expected.
- Decoder only (encoder + vocoder are separate gates). `_adjust_output_shape` final
  crop/pad is exercised as a no-op here; a variable-length input that triggers a real
  crop/pad is NOT covered by this single-shape gate (out of scope for P4 decoder parity).

## Verdict
PASS. cos = 0.99999595 >= 0.999, fail-closed verified via 3 mutations, structure and
numerics confirmed against the official ltx_core source. No git commit made.
Working tree left clean (all mutations reverted; clean baseline rebuilt + rerun).
