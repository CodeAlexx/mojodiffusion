# SKEPTIC findings — LTX-2.3 BigVGAN-v2 vocoder + BWE (P1/P4)

Date: 2026-05-28
Auditor role: SKEPTIC for "vocoder" (audit vs Rust spec + re-run gate on GPU + verify fail-closed).
Reference spec: `/home/alex/EriDiffusion/inference-flame/src/vae/ltx2_vocoder.rs`
Mojo under audit:
- `serenitymojo/models/vocoder/ltx2_vocoder.mojo`
- `serenitymojo/ops/activation1d.mojo`
- `serenitymojo/ops/snake.mojo`
Gate: `serenitymojo/pipeline/ltx2_vocoder_smoke.mojo`
Oracle: `scripts/ltx2_vocoder_ref.py` → `output/ltx2_vocoder/vocoder_ref.safetensors`

## VERDICT: PASS — builder claim upheld. Gate is real, fail-closed verified.

The Mojo `LTX2VocoderWithBWE.forward` is structurally faithful to the Rust spec
and numerically matches a freshly-regenerated PyTorch oracle on the real
distilled-checkpoint weights. The gate fails closed under structural and
numeric mutation. One documentation discrepancy noted (non-blocking).

## What I independently re-verified (not trusting prior artifacts)

1. Regenerated the oracle from scratch (`python3 scripts/ltx2_vocoder_ref.py`).
   - Deterministic: byte-identical to the prior dump (`cmp` clean).
   - wav48 rms 0.23280, range [-0.7941, 0.8082].
2. Rebuilt the Mojo smoke binary from source (clean compile, warnings only).
3. Ran the gate on GPU (RTX 3090 Ti, 23 GB free throughout, ~1 GB used):
   - **cos = 0.99996371** (gate cos ≥ 0.999 — PASS)
   - **max_abs = 0.011909** (gate < 0.02 — PASS)
   - length match (7680 == 7680), finite, non-silent (rms 0.23287).
4. .wav artifact verified via Python `wave`: 2ch / 48000 Hz / 3840 frames /
   16-bit, finite, rms 0.2328, range ±0.81 (matches oracle).

## Structural audit vs Rust spec (all confirmed against the real checkpoint)

Checkpoint inspection (`vocoder.*` = 1227 keys, matches audit):
- `vocoder.vocoder`: 6 ups, 18 resblocks.
  - ups shapes: (1536,768,11),(768,384,4),(768→…,4)×4 → channel flow
    1536→768→384→192→96→48→24, kernels [11,4,4,4,4,4]. Matches
    `BASE_UPSAMPLE_RATES=[5,2,2,2,2,2]` and the spec's `ups[0..6]` header.
  - conv_pre (1536,128,7), conv_post (2,24,7) → stereo out. Matches.
  - act_post alpha shape (24,) = final_channels. Matches.
- `vocoder.bwe_generator`: 5 ups, 15 resblocks, kernels [12,11,4,4,4],
  matches `BWE_UPSAMPLE_RATES=[6,5,2,2,2]`.
- mel_basis (64,257), forward_basis (514,1,512). Matches BWE compute_mel.

Code-path checks (Mojo vs Rust):
- **6 upsample stages**: `_upsample_rate(6,i)` = [5,2,2,2,2,2]; BWE
  `_upsample_rate(5,i)` = [6,5,2,2,2]. ✓
- **ResBlock snake counts**: each AMPBlock1 runs acts1[0..3] + acts2[0..3]
  (6 activation1d/block); 3 AMPBlock1 per stage averaged. `_ampblock_forward`
  loops i∈0..3 calling acts1[i].apply → conv1 → acts2[i].apply → conv2 → add.
  ✓ matches rs:598-653.
- **Anti-alias kaiser-sinc FIR depthwise**: folds [B,C,L]→[B*C,1,L], applies
  the [1,1,12] filter via conv1d (groups=1 after fold), with replicate-pad
  5/5 (up), zero-insert ×2, side-pad 11/11, slice 15:-15, ×2 scale; then
  snake; then replicate-pad 5/6 (down), conv1d stride=2. ✓ matches the Rust
  REFERENCE tensor path (rs:247-285). NOTE: Rust ships a FUSED CUDA kernel
  (`activation1d_fused`, rs:423) for production; the Mojo and the oracle both
  use the *reference* (non-fused) path — same DSP, so this is a doc detail.
- **Snake alpha/beta log-scale precompute**: `exp(alpha)`,
  `1/(exp(beta)+1e-9)` with the +1e-9 INSIDE the reciprocal denominator. ✓
  matches rs:522-524 (`beta_exp.add_scalar(1e-9).reciprocal()`).
- **get_padding(k,d)** = (k·d − d)//2. ✓
- **48 kHz stereo output**: conv_post→2ch, BWE ratio 3 → 48000 Hz, finite. ✓
- **kaiser up==down filter**: confirmed the checkpoint stores BYTE-IDENTICAL
  upsample.filter and downsample.lowpass.filter (max diff 0.0). This is a
  genuine BigVGAN-v2 property (shared symmetric lowpass), not a bug — see
  mutation 3 below.

## Fail-closed verification (mutate, confirm gate fails)

| # | Mutation | Result | Gate |
|---|----------|--------|------|
| 1 | Drop last upsample stage (6→5) | Crash: "broadcast: incompatible dims 48 vs 24" | FAIL-CLOSED (cannot produce output) |
| 2 | Scale snake-beta output ×0.5 in activation1d | cos = -0.0152, max_abs = 0.808, rms 0.0002 | FAIL (raises) |
| 3 | Swap up/down kaiser FIRs in ActParams.apply | cos = 0.99996 (NO-OP) | PASS — filters are byte-identical in ckpt (verified), so swap is a true no-op, not a gate hole |
| 4 | Use only resblock[0], skip 3-AMPBlock averaging | cos = 0.0214, max_abs = 0.808, rms 0.0001 | FAIL (raises) |

Two distinct shape-preserving numeric mutations (2 and 4) drive cos far below
the 0.999 gate and trip the raise; the structural mutation (1) fails closed by
crashing. The gate is sensitive to both the snake activation and the per-stage
resblock averaging. Reverted all mutations; `git diff` clean; clean rebuild
re-confirms the baseline PASS.

## Discrepancies (non-blocking)

1. **Builder prose vs actual gate constant.** The builder report says
   "max_abs = 0.0119 (gate < 0.02)" in one place but the handoff/Plan text and
   an earlier summary say "max_abs < 0.01". The binding gate constant in the
   smoke is `_MAXABS_GATE = 0.02` (ltx2_vocoder_smoke.mojo:52), and measured
   max_abs is 0.0119 — so it would FAIL a literal `<0.01` gate. The 0.02
   tolerance is justified in-code as F32 conv-reduction-order + libdevice
   sin/exp jitter over the ~110-conv chain (mean_abs ~0.0014, p99 ~0.007). The
   binding Plan HARD RULE is cos ≥ 0.999, which passes at 0.99996 with wide
   margin. Recommend the prose be corrected to "max_abs 0.0119 < 0.02" to avoid
   confusion. This does not affect correctness.

2. **Reference vs fused path** (above): Mojo+oracle use the non-fused
   activation1d; Rust production uses a fused CUDA kernel. Equivalent DSP;
   the localization micro-gate (builder's standalone activation1d, cos=1.0)
   already proved the reference path. No action needed.

## Files (unchanged by this audit)
- `serenitymojo/models/vocoder/ltx2_vocoder.mojo`
- `serenitymojo/ops/activation1d.mojo`, `serenitymojo/ops/snake.mojo`
- `serenitymojo/pipeline/ltx2_vocoder_smoke.mojo`
- `scripts/ltx2_vocoder_ref.py`
- Artifact: `output/ltx2_vocoder/mojo_vocoder.wav` (48 kHz stereo, finite, audible)

No git commit made. GPU stayed within budget.
