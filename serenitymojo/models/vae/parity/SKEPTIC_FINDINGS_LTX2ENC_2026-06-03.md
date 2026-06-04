# SKEPTIC FINDINGS — LTX-2.3 Video VAE ENCODER (2026-06-03)

Component: `ltx2_vae_encoder`. Reviewer ran adversarially, assuming the port lies.

## §0 receipts
- `serenitymojo/MAP.md` — wayfinding for the pure-Mojo+MAX inference-only GPU diffusion lib; "where does X live".
- `serenitymojo/docs/SERENITYMOJO_MODULES.md` — per-module public API (structs, comptime params, method signatures), ✅=parity-verified.
- `inference-flame/src/vae/ltx2_encoder.rs` — the Rust ground truth: LTX2VaeEncoder forward (patchify→conv_in→9 down_blocks→pixel_norm→silu→conv_out→expand→mean→normalize), the SPEC for the 2.3 checkpoint.
- `/home/alex/.claude/skills/mojo-syntax/SKILL.md` — current Mojo syntax correction layer (fn→def, alias→comptime, inout→mut, etc.).

## Method
The #1 risk flagged: the oracle is a torch transcription of the SAME Rust the Mojo was
written from, so cos≈1.0 could be a shared misread. I did NOT trust cos. I independently
re-derived the novel layout ops from the Rust **permute strings** (not the builder's kernels)
in fresh numpy, hand-computed the temporal/channel arithmetic, and verified the real
checkpoint shapes/keys. Then I re-ran the parity myself.

---

## INDEPENDENCE VERDICT: oracleIsIndependent = TRUE (for the load-bearing ops)

The shared-source risk is REAL in principle but I established independence for every place a
shared misread could hide:

1. **space_to_depth_3d / patchify_3d layout (the highest-risk novel gather indices).**
   I implemented the Rust `space_to_depth` (permute `[0,1,3,5,7,2,4,6]`) and `patchify`
   (permute `[0,1,3,7,5,2,4,6]`) directly from the .rs reshape/permute strings in numpy,
   and separately transcribed the Mojo kernel index formulas. On a NON-symmetric shape
   `[2,3,4,6,8]` with distinct per-position values, for strides (1,2,2),(2,1,1),(2,2,2):
   **Mojo kernel == Rust permute, bit-exact**, AND `depth_to_space(space_to_depth(x))==x`
   (the s2d IS the exact inverse of the decoder-trusted d2s). patchify p=4 also bit-exact.
   This is a genuinely independent oracle (Rust permute math, not the builder's code).
   → spaceToDepthCorrect = TRUE.

2. **pixel_norm axis.** Mojo runs NDHWC and calls `rms_norm` which reduces the LAST dim
   (= channel in NDHWC). Oracle/Rust reduce dim=1 on NCDHW (= channel). Same physical axis
   reached through DIFFERENT layouts and DIFFERENT code (hand-rolled rms kernel vs torch
   `.mean(dim=1)`). A shared transcription bug here is essentially impossible.

3. **Causal pad direction.** Independently confirmed encoder = LEFT-ONLY 2 frames
   (`concat first,first,x`) vs decoder = SYMMETRIC 1+1 (`concat first,x,last`), matching
   Rust `time_pad = k-1 = 2` left-only (encoder, causal=True) vs decoder symmetric.
   The two siblings genuinely differ as claimed.

4. **Real checkpoint.** Verified against the actual .safetensors header: conv_out is
   `[129,1024,3,3,3]` (the 2.3 arch, NOT diffusers' 2048), per_channel_statistics are real
   non-trivial `[128]` BF16 (std range 0.074..0.914), all s2d conv out-channels match
   `out_ch/prod` (64,256,128,128). The oracle loads and applies these, not a random init.

5. **Different compute stacks agree.** Oracle is torch `F.conv3d`; Mojo is the hand-rolled
   `conv3d.mojo` F32-accumulate kernel. They agree to cos 0.99998 on raw latents. A shared
   bug would have to manifest identically in two unrelated conv implementations — only
   possible for pure layout/ordering bugs, which I independently ruled out in (1).

Caveat (FRAGILE, see below): independence is established for the *structural* ops. The
oracle and Mojo share the same block-schedule constants and the same Rust-derived
expand/normalize logic; those are simple enough to audit by eye and I did, but they were not
cross-checked against an unrelated implementation (no independent LTX2-2.3 forward exists —
diffusers is a different arch). I rate the gate STRONG but not absolute.

---

## PARITY (re-ran myself)
`pixi run mojo run -I . serenitymojo/models/vae/parity/ltx2_encoder_parity.mojo` → **exit 0**
- RAW MEAN latent:   cos=**0.99997664**, max_abs=0.00391, magRatio=1.00020, shape [1,128,2,2,2]
- NORMALIZED moments: cos=**0.99994266**, max_abs=0.04395, magRatio=1.00073
- video std 0.5774 (matches oracle 0.5774), norm std 0.7509 vs ref 0.7505.
Matches the builder's reported 0.99998 / 0.99994.

## temporalTested = TRUE
T=9 → T'=2. The 3 stride-2 (temporal) s2d blocks (indices 3,5,7) each prepend (st-1)=1
causal frame then halve: 9→5→3→2. Output shape [1,128,2,2,2] confirms the left-only
temporal causal pad is genuinely exercised end-to-end (not collapsed to T=1).
Hand-arithmetic of every stage's T matches the dumped lat_shape.

## moments expand/mean — matches Rust exactly
Rust: n_ch=129, repeat channel 128 ×127 → 256, mu=first 128. The first 128 channels are
untouched by the expansion, so mu == conv_out[:,0:128]. Mojo skips the dead expansion and
slices the first 128 directly (line 447) — mathematically identical, verified by eye.
Oracle does the full expansion then slices — same result. Consistent.

## per-channel normalize — correct
`(mu - mean_of_means)/std_of_means`, stats `[128]→[1,128,1,1,1]` on NCDHW, computed in F32
(Mojo `_normalize` casts to F32, sub, div, cast back — matches Rust `PerChannelStatistics::
normalize`). Loaded from `vae.per_channel_statistics.{mean,std}-of-means` (correct keys,
real values). norm_std 0.751 (≠1.0) confirms real stats applied on the right axis.

## group-average residual — correct
n_groups == out_ch and group_size contiguous for all 4 s2d blocks; reshape
`[B,n_groups,group_size,...]` + mean(dim=2,keepdim=False). Identical in Rust/oracle/Mojo.
block.3 (group_size=1) passes residual through unchanged in all three. Verified arithmetic.

## hygiene — clean
- No banned syntax (`fn`/`alias`/`inout`/`owned`/`let`), no torch/autograd/backward leak in
  shipped `.mojo`.
- Reuses `conv3d.mojo` (no reimplementation) and the sibling `depth_to_space_3d` from the
  same ops file (encoder s2d is its verified inverse).
- `List[ArcPointer[Tensor]]` weight store, `def ... raises`, comptime block schedule — all
  per project convention.

---

## TAGGED ISSUES

**STYLE — hardcoded TIME_PAD in `_causal_conv3d`.** `ltx2_vae_encoder.mojo:273-274` hardcodes
`concat(1, ctx, first, first, x)` (2 left frames) and ignores the `comptime TIME_PAD = 2`. For
k=3 (every conv in this VAE) it is correct, but if this helper is ever reused for a non-k=3
conv it silently pads wrong. Cosmetic only; no correctness impact on the shipped path.

**FRAGILE — oracle independence is structural, not end-to-end.** No second independent
implementation of the LTX-2.3 (non-diffusers) encoder exists, so the FULL forward parity gate
is Mojo-vs-(torch transcription of the same Rust). I closed the layout/axis/pad/checkpoint
gaps independently (above), which removes the classes of shared-misread bug that matter most,
but the gate is fundamentally one-source for the high-level block wiring. Rated STRONG, not
absolute. Mitigation already present: two different conv stacks (torch vs hand-rolled) agree.

**FRAGILE — bf16 max_abs on normalized moments is 0.044.** cos is high (0.99994) but the
per-element max abs deviation on normalized latents is ~0.044 (because dividing by small
per-channel std up to ~0.074 amplifies bf16 conv noise). This is expected for bf16 and the
cos/magRatio gates pass, but downstream consumers sensitive to absolute latent values (not
just direction) should be aware the encoder is bf16-noisy, not bit-exact.

---

{component:"ltx2_vae_encoder", reRanParity:true, cos:0.9999426607339947, oracleIsIndependent:true, spaceToDepthCorrect:true, temporalTested:true, blockers:[], verdict:"clean"}
