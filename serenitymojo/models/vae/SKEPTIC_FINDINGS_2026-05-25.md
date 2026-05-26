# SKEPTIC FINDINGS — Z-Image VAE decoder (Serenitymojo) — 2026-05-25

Adversarial review of `serenitymojo/models/vae/{decoder2d,zimage_decoder,vae_ops,upsample}.mojo`
+ parity drivers. Assumption going in: the build lies and was only proven on a toy
8×8 latent. Scope code was NOT edited; only oracle/probe scripts were added under
`parity/` and `models/vae/` (parity drivers).

## Verdict (TL;DR)

**No blockers found.** The decoder is real-resolution correct. The headline gap —
"only tested at 8×8" — was closed: it compiles, runs, and matches the diffusers
oracle at **512×512 and 1024×1024**, with no OOM and no thermal issue on the
3090 Ti. The max_abs is benign BF16 accumulation with **no systematic bias**.
Weight mapping is **100% complete** (all 138 decoder keys consumed). The unusual
single-head Dh=512 attention was isolated and proven correct at S=4096 against a
CPU softmax reference. Non-square shapes work. Upsample is bit-exact.

---

## Real-resolution parity (the big gap — claim #1)

All vs the diffusers F32 oracle (`gen_oracle_real.py`, seed 1234), Mojo compute BF16:

| Latent → image            | cos          | max_abs | result |
|---------------------------|--------------|---------|--------|
| [1,16,8,8]   → [1,3,64,64]   (baseline, reproduced) | 0.9999846 | 0.0263 | PASS |
| [1,16,8,16]  → [1,3,64,128]  (non-square)            | 0.9999800 | 0.0490 | PASS |
| **[1,16,64,64] → [1,3,512,512]**                     | **0.9999840** | **0.0530** | **PASS** |
| **[1,16,128,128] → [1,3,1024,1024]**                 | **0.9999842** | **0.0531** | **PASS** |

- Comptime (LH,LW) parameterization **scales correctly** from 8×8 to 128×128. No
  recompile-shape mismatch, no index error.
- **No OOM at 1024².** Peak GPU temp 70 °C, ~580 MiB resident reported at end
  (buffers freed as they go; transient peak higher but well under 24 GB).
- max_abs is essentially **flat** across resolutions (0.049–0.053) once past the
  toy size — it does NOT blow up with token count. Drivers:
  `parity_decode_real.mojo`, `parity_decode_1024.mojo`, `parity_nonsquare.mojo`.

---

## max_abs severity verdict (claim #2) — ACCEPTABLE (benign BF16)

At 512×512 I histogrammed the per-pixel abs-diff and probed for a hidden
offset/scale bias (`parity_decode_real.mojo`):

- abs-diff distribution over 786,432 pixels: **92.6 % < 0.005, 99.96 % < 0.02**,
  only **2 pixels ≥ 0.05** (max 0.0530).
- **mean SIGNED diff = −1.3e-4 ≈ 0** → no constant offset.
- **per-channel signed-mean diff: R −3.5e-5, G −4.4e-4, B +7.2e-5** → no
  per-channel scale/bias. If there were a systematic offset or a wrong
  scale/shift fold, these would be O(1e-2+), not O(1e-4).
- Sample pixels (Mojo vs oracle, channel R): (0,0) −0.1641/−0.1606,
  (256,256) 0.2285/0.2277, (511,511) 0.3965/0.3930 — agree to ~3 decimals,
  errors both signs, no spatial structure (corners not worse than center).

**Why it's BF16, not a bug:** the foundation conv2d is F32-accumulate but
**BF16-store** at every op output (verified `ops/conv.mojo:116`), and all 244 VAE
weights are **BF16 on disk** (so BF16 is the native dtype, not a downgrade). Each
of the ~80 conv/norm/add ops rounds its output to BF16 (~3 decimal digits). A
0.05 worst-case absolute error on outputs that range to ±1.7 is the expected
floor of that rounding chain, accumulated over depth — not a missing/incorrect
layer. The up_block_3 per-block max_abs=4.74 in the build report is the SAME
phenomenon scaled by that block's activation magnitude (oracle std≈92, so 4.74 is
~5 % relative on huge pre-norm activations that GroupNorm immediately divides out;
the post-head image max_abs is back to 0.008).

**Severity: ACCEPTABLE.** For a VAE decoder feeding a display image this is
invisible. If a future consumer needs tighter parity, the only lever is F32
compute (upcast weights), which the on-disk BF16 format does not warrant.

---

## Weight-mapping completeness (claim #3) — COMPLETE

Cross-checked `ZImageDecoder.load` consumption against the safetensors keys
programmatically:

- decoder keys on disk: **138**
- consumed by ZImageDecoder: **138**
- **UNCONSUMED (missing layer): NONE**
- **CONSUMED-BUT-ABSENT (phantom key): NONE**

Shortcut handling is correct: `up_blocks.2.resnets.0` and `up_blocks.3.resnets.0`
are the only two resnets with `conv_shortcut.{weight,bias}` (Cin≠Cout: 512→256 and
256→128), and the `has_shortcut = Cin != Cout` gate matches exactly. conv_in
(16→512), conv_norm_out, conv_out (128→3), mid res0/attn/res1, and all 12 up-block
resnets + 3 upsamplers map 1:1. `use_post_quant_conv=false` confirmed in config →
correctly no post_quant_conv. scaling=0.3611 / shift=0.1159 match config.

---

## NHWC-native vs NCHW/NHWC (claim #4) — CORRECT at non-square

- `parity_nonsquare.mojo` decodes 8×16 → 64×128 with cos=0.9999800 and the
  **correct output shape [1,3,64,128]** (not transposed to 128×64). The entry
  `nchw_to_nhwc` and exit `nhwc_to_nchw` host permutes (decoder2d.mojo:114-167)
  use distinct H/W loop bounds and index formulas; a transposed index would have
  produced a shape mismatch or garbage cos here. Clean.
- The single NCHW→NHWC at entry and NHWC→NCHW at exit (no per-op ping-pong) is a
  legitimate optimization since the foundation conv2d AND group_norm are both
  NHWC-native — unlike the Rust reference which ping-pongs for cuDNN NCHW conv.

---

## Kit-glue + upsample (claim #5) — CORRECT

- **upsample**: `upsample_probe.mojo` on a non-square 3×5×2 NHWC tensor →
  6×10×2, **max_abs=0.0 vs the host nearest reference**, 0 bad elements, correct
  output shape. Matches diffusers `F.interpolate(scale_factor=2, mode="nearest")`
  exactly: `out[oh,ow,c]=in[oh//2,ow//2,c]` (2×2 replication, NOT bilinear, NOT
  align_corners/nearest-exact). The 3×3 conv after it is the foundation conv2d.
- **add**: F32-accumulate elementwise, dtype+numel guarded (vae_ops.mojo:105-156).
  Result takes `a`'s shape; in all call sites `a` (residual) and `b` (h) are the
  same shape by construction, so this is safe. Residual ordering is correct:
  ResnetBlock returns `add(residual, h)`, AttnBlock clones x as residual BEFORE
  group_norm and returns `add(residual, out)` — matches diffusers
  `residual_connection=True`.
- **reshape**: device-copy + metadata, numel-mismatch guarded. Used only for the
  NHWC↔token flatten around mid-attn; shapes are comptime-derived. Correct.

---

## Mid-attn Dh=512 single-head (claim #6) — CORRECT, including at real S

- diffusers config confirmed: heads=1, scale=1/√512=0.04419417, residual_connection
  =True, **rescale_output_factor=1** (no hidden output scaling), group_norm 32/1e-6,
  spatial_norm=None, norm_q/norm_k=None, to_out = [Linear(512,512), Dropout(0,
  identity at inference)]. The Mojo AttnBlock reproduces all of this.
- The diffusers attn weights are **native Linear [C,C]** (to_q/to_k/to_v/to_out.0),
  NOT Conv2d-1×1 — the build's "use foundation `linear` directly, no squeeze"
  claim is correct for this checkpoint (verified key shapes [512,512]).
- **Flash_attention at the REAL shape**: `sdpa_probe_4096.mojo` runs
  S=4096 (=64×64), Dh=512, H=1 and checks 16 query rows (every 256th, full S
  coverage) against an O(S²·Dh) CPU softmax reference → **max_abs_diff = 7.2e-7**.
  So the SDK flash_attention is near-bit-exact at this shape, and S=16384
  (1024² decode) ran to completion with PASS. The high full-decode cos is NOT
  masking an attention error.

---

## Clean checks (positive evidence)

- 8×8 per-block parity reproduced: conv_in/mid/up0..3/head all cos ≥ 0.99999.
- 8×8 full-decode reproduced: cos 0.9999846, max_abs 0.0263 (matches build report).
- conv2d RSCF (OIHW→[Kh,Kw,Cin,Cout]) transpose validated by `conv_probe.mojo`
  against a CPU conv reference; the same index formula is used in
  `_load_conv_weight_rscf`. max_abs<1e-3.
- All 244 VAE tensors BF16 on disk → BF16 compute is the native choice; F32
  oracle parity at 0.99998 confirms adequacy.

---

## Couldn't verify / out of scope

- **Encoder**: not in scope and not implemented (encoder.* keys exist on disk but
  the task is the decoder). Not a finding.
- **End-to-end visual decode of a real model latent**: parity used seeded
  `torch.randn` latents, not an actual Z-Image DiT output. Numerically this is
  strictly harder (randn has full dynamic range), so it's a conservative test, but
  a real-latent visual spot-check was not performed (no DiT in this scope).
- **F32 compute path**: not exercised (weights are BF16; no upcast path in the
  decoder). If ever needed for tighter parity, untested.
- **Batch N>1**: all paths hard-code/were tested at N=1 (VAE decode is always N=1
  in this pipeline). The kit is comptime-parameterized on N but only N=1 ran.

## Artifacts added (under parity/ and models/vae/, no scope edits)
- `parity/gen_oracle_real.py` — parametrized oracle (LH LW args), suffixed dumps.
- `parity_decode_real.mojo` — 512² decode + abs-diff histogram + bias probe.
- `parity_decode_1024.mojo` — 1024² decode (S=16384 attn stress).
- `parity_nonsquare.mojo` — 8×16 non-square decode.
- `sdpa_probe_4096.mojo` — flash_attention vs CPU softmax at S=4096, Dh=512.
- `upsample_probe.mojo` — nearest-2× exactness on non-square NHWC.
