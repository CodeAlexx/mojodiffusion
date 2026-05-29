# SKEPTIC findings — LTX-2.3 P5 Video VAE full decoder

**Date:** 2026-05-28
**Subject:** `serenitymojo/models/vae/ltx2_vae_decoder.mojo` (`LTX2VaeDecoderWeights` + `decode[...]`)
**Rust spec audited:** `/home/alex/EriDiffusion/inference-flame/src/vae/ltx2_vae.rs` (`LTX2VaeDecoder::decode_with_dump`)
**Oracle:** `scripts/ltx2_video_vae_decode_ref.py` (fresh PyTorch port of the Rust, not derived from Mojo)

## VERDICT: GREEN — builder's claim CONFIRMED

Gate reproduced on GPU (RTX 3090 Ti, 23 GB free): **cosine 0.99998194 (>= 0.999), max_abs_diff 0.0138**, decoded frame coherent. Fail-closed verified.

## What was audited and verified line-for-line vs ltx2_vae.rs

1. **Block schedule (9 up_blocks).** Mojo `MID_CH=[1024,_,512,_,512,_,256,_,128]`, `MID_NRES=[2,_,2,_,4,_,6,_,4]`, DepthToSpace at indices 1,3,5,7 with strides (2,2,2)r2 / (2,2,2)r1 / (2,1,1)r2 / (1,2,2)r2 — **exact match** to Rust `DECODER_BLOCKS` (ltx2_vae.rs:59-69).
2. **Checkpoint key coverage (independent audit of the .safetensors header).** Confirmed all 5 Mid blocks present with the correct res_block counts (up_block.0:2, .2:2, .4:4, .6:6, .8:4) and all 4 DepthToSpace upsamplers (up_block.1,3,5,7) + conv_in + conv_out + 2 per_channel_statistics. **84 decoder + 2 stats = 86 keys**, matching the oracle's load count. The Mojo `wanted` list requests exactly this set; `_w()` raises on any missing key and load succeeded → **no silent gaps**.
3. **un_normalize** = x*std+mean per-channel (ltx2_vae.rs:359-363). Mojo applies in NDHWC with std/mean reshaped to [1,1,1,1,128] broadcast over channel — equivalent.
4. **PixelNorm** = x·rsqrt(mean(x²,channel)+1e-6), no weights. Mojo uses `rms_norm` with ones-gamma (eps INSIDE sqrt) — byte-identical formula, eps=1e-6 (ltx2_vae.rs:40,209-215).
5. **ResnetBlock3D** order: PixelNorm→SiLU→conv1→PixelNorm→SiLU→conv2 + skip (ltx2_vae.rs:233-241) — exact match.
6. **CausalConv3d** non-causal replicate-pad on TIME, (k-1)/2=1 each side, zero pad H/W=1 (cfg causal_decoder:False, ltx2_vae.rs:107-127). Mojo replicates first/last frame via slice+concat then conv3d with pad_d=0 — match.
7. **DepthToSpace.** Channel split c-major `ct=((c·p1+i1)·p2+i2)·p3+i3`; temporal stride-2 drops the FIRST output frame (ltx2_vae.rs:301-322). Verified the Mojo `depth_to_space_3d` kernel index math and the `drop=(p1==2)` flag.
8. **unpatchify (patch=4).** Rust reshape `[b,c,4,4,f,h,w]` (dim2=a, dim3=b2, c-major `ct=((c·4+a)·4+b2)`) + permute `[0,1,4,5,3,6,2]` → `ho=h·4+b2, wo=w·4+a`. Mojo folds (B,F)→batch and does the equivalent rank-6 permute `[0,1,4,3,5,2]` on `[BF,3,a,b2,H,W]` → `ho=h·4+b2, wo=w·4+a`. **Traced and confirmed identical mapping** (the rank-6 fold is a faithful workaround for the rank-7 einops; the permute op caps at rank 6).

## Shape / temporal-stride / pixel-range checks

- Latent `[1,128,2,2,2]` → decoded `[1,3,9,64,64]`. Frames: 2 →(d2s ×2 drop) 3 →5 →9 →9 = `1+(2-1)*8` ✓. Spatial: 2 →4 →8 →8 →16 →(unpatchify ×4) 64 ✓. Both Mojo and oracle agree on shape.
- Pixel range sane: Mojo absmax 0.898 (in ~[-1,1]); mean -0.2887/std 0.1903 vs oracle -0.2881/0.1902; oracle min/max -0.898/0.750. Finite=True (no NaN/Inf).

## GPU gate (reproduced this session)

```
cosine: 0.99998194   max_abs_diff: 0.013793945
[gate] decode cos >= 0.999: PASS
```
Oracle regenerated fresh from the checkpoint (86 VAE tensors) before the run, so the gate is not run against a stale ref.

## FAIL-CLOSED verification (the binding skeptic test)

Mutation: changed the Mid res_block loop from `range(MID_NRES[i])` to `range(1, MID_NRES[i])` (drop the first res_block of each Mid block — preserves output shape, changes values). Rebuilt and re-ran:

```
[stat] decoded (mojo) mean=-1.5501 std=1.8041 absmax=9.6875   <-- diverged
cosine: 0.6214216   max_abs_diff: 9.530273
Unhandled exception: video VAE decode cosine 0.6214216 < 0.999   <-- gate RAISED (non-zero exit)
```

The gate **collapsed to 0.621 and raised**, confirming it is a binding parity check, not a finite-stats rubber stamp. Mutation reverted; clean rebuild reproduces 0.99998194 PASS.

## Eyeball

Frame-0 PNG (`output/ltx2_video_vae/ltx2_video_frame00.png`): smooth low-frequency color gradients, no per-pixel static, no checkerboard, no gray NaN-out. Coherent decode of the random-normal·0.5 latent (correctly a smooth texture, not passed-through noise).

## Caveats / notes (non-blocking)

- The gate uses a synthetic deterministic latent (randn·0.5), not a real DiT-produced latent. This is appropriate for an isolated VAE parity gate; the real-latent end-to-end check is P7's job.
- BF16 compute path; max_abs_diff 0.0138 is consistent with BF16 rounding across the depth of the network. Cosine 0.99998 is well clear of the 0.999 bar.
- Oracle is a fresh PyTorch reimplementation of the Rust, structurally independent of the Mojo (both unpatchify/d2s implementations were cross-checked against the Rust einops, not against each other).

**No git commit performed (hard rule).**
