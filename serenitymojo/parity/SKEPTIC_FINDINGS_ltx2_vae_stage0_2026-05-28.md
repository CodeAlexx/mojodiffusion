# SKEPTIC FINDINGS — LTX-2.3 Video VAE Decoder, STAGE 0 (2026-05-28)

**Team:** LTX2 Team 2 — video VAE decoder stage 0
**Skeptic verdict:** **PASS** — all builder claims verified against Rust + checkpoint; smoke
runs on GPU producing finite output of exact expected shape. Zero FAILs.

## Files audited (read-only)
- NEW `serenitymojo/models/vae/ltx2_vae_decoder.mojo` (304 lines)
- NEW `serenitymojo/pipeline/ltx2_vae_decode_stage0_smoke.mojo` (103 lines)
- Reused `serenitymojo/models/vae/conv3d.mojo`, `serenitymojo/ops/norm.mojo`,
  `serenitymojo/ops/tensor_algebra.mojo`, `serenitymojo/tensor.mojo`

## Rust ground-truth reference
- `/home/alex/EriDiffusion/inference-flame/src/vae/ltx2_vae.rs`

## Checkpoint / config reference
- `/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors`
- Config: inline `__metadata__["config"]["vae"]` in the safetensors header.

---

## A. Rust parity

### A1 — Decoder stage-0 prefix & order — **PASS**
`ltx2_vae.rs:453-457` decode forward:
```
let x = self.stats.un_normalize(x)?;     // un_normalize FIRST
let mut h = self.conv_in.forward(&x)?;   // conv_in
for block in &self.blocks { h = block.forward(&h)? } // up_blocks.0 = Mid{1024,2} is blocks[0]
```
`DECODER_BLOCKS[0] = Mid{channels:1024, n_res:2}` (rs:60). Mojo `decode_stage0`
(mojo:290-302) does `permute → _un_normalize → conv_in → 2× _resnet_block`. Order
and channel/n_res match exactly. The key prefix `vae.decoder.*` is correct
(Rust strips the `vae.` outer prefix, rs:513-519; Mojo loads with `vae.decoder.*`
literal keys).

### A2 — PixelNorm: RMS-over-channel, NO learnable gamma, eps=1e-6 — **PASS**
`ltx2_vae.rs:209-215`:
```
let mean_sq = x_sq.mean_along_dims(&[1], true)?;          // dim=1 = channel (NCDHW)
let denom = mean_sq.add_scalar(PIXEL_NORM_EPS)?.rsqrt()?; // eps INSIDE sqrt
x_f32.mul(&denom)?...                                      // NO gamma multiply
```
`PIXEL_NORM_EPS = 1e-6` (rs:40). No weights in PixelNorm. Mojo uses
`rms_norm(x, ones_c, 1e-6)` (mojo:250) with `PIXEL_NORM_EPS = Float32(1e-6)`
(mojo:73) and a ones gamma `[1024]` (mojo:153-158) → mathematically identical
(gamma=1 is a no-op). Channel-last NDHWC means rms_norm's last-dim reduction ==
PyTorch dim=1 reduction. Eps confirmed inside sqrt — see Q2.

### A3 — ResnetBlock order + identity skip — **PASS**
`ltx2_vae.rs:233-241`:
```
let h = pixel_norm(x)?; let h = h.silu()?; let h = self.conv1.forward(&h)?;
let h = pixel_norm(&h)?; let h = h.silu()?; let h = self.conv2.forward(&h)?;
x.add(&h)   // identity skip, NO shortcut conv
```
Mojo `_resnet_block` (mojo:257-263): `pixel_norm → silu → conv1 → pixel_norm →
silu → conv2 → add(x,h)`. Identical. Skip is **identity** (no shortcut conv) —
correct because conv_in outputs 1024 and both res blocks are 1024→1024, so no
channel change. Checkpoint dump (B1) shows NO `conv_shortcut`/`nin_shortcut`
keys under `up_blocks.0.res_blocks.*` — only conv1/conv2. Confirmed identity.

### A4 — un_normalize formula + stat application — **PASS**
`ltx2_vae.rs:359-363`:
```
let std = self.std_of_means.reshape(&[1, 128, 1, 1, 1])?;
let mean = self.mean_of_means.reshape(&[1, 128, 1, 1, 1])?;
x.mul(&std)?.add(&mean)        // x * std + mean
```
Tensors are `per_channel_statistics.std-of-means` / `mean-of-means` (rs:354-355),
broadcast over the 128 latent channels. Mojo `_un_normalize` (mojo:267):
`add(mul(x, stat_std), stat_mean)` = `x*std + mean`, with stats reshaped to
NDHWC-broadcast `[1,1,1,1,128]` (mojo:147-151). Formula, direction, and channel
broadcast all match. scaling_factor=1.0 (config) so no divide — correct.

### A5 — patch_size=4 / unpatchify NOT in stage 0 — **PASS**
`unpatchify` (rs:370-385) and `conv_out` run only at the very end of `decode`
(rs:474-479), after all 9 blocks. Stage 0 stops after `up_blocks.0`, so
patch_size never touches the stage-0 output. Mojo `decode_stage0` returns right
after the 2 res blocks (mojo:304). Output channels stay 1024, F/H/W unchanged —
no unpatchify interaction. Confirmed.

---

## B. Checkpoint reality

### B1 — Stage-0 keys & shapes — **PASS**
Live dump from the checkpoint:
```
vae.decoder.conv_in.conv.bias                         [1024]            BF16
vae.decoder.conv_in.conv.weight                       [1024,128,3,3,3]  BF16
vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight[1024,1024,3,3,3] BF16
vae.decoder.up_blocks.0.res_blocks.0.conv2.conv.weight[1024,1024,3,3,3] BF16
vae.decoder.up_blocks.0.res_blocks.1.conv1.conv.weight[1024,1024,3,3,3] BF16
vae.decoder.up_blocks.0.res_blocks.1.conv2.conv.weight[1024,1024,3,3,3] BF16
(+ matching .bias [1024] for each conv)
vae.per_channel_statistics.mean-of-means              [128]             BF16
vae.per_channel_statistics.std-of-means               [128]             BF16
```
All builder-listed shapes match exactly. conv weights are OIDHW `[Cout,Cin,kD,kH,kW]`.
NO shortcut-conv keys present (confirms A3 identity skip). All BF16.

### B2 — Config values — **PASS**
From inline `__metadata__["config"]["vae"]`:
`causal_decoder: false`, `norm_layer: "pixel_norm"`, `latent_channels: 128`,
`patch_size: 4`, `scaling_factor: 1.0`, `spatial_padding_mode: "zeros"`,
`timestep_conditioning: false`, `use_quant_conv: false`. Every builder config
claim verified.

---

## C. conv3d reuse correctness

### C1 — conv3d layout / caller-owns-temporal-pad — **PASS**
`conv3d.mojo` header (lines 16-23) + signature (110-132): input NDHWC
`[N,D,H,W,Cin]`, filter QRSCF `[Q,R,S,Cin,Cout]`, F32-accumulated, **symmetric**
padding on all axes; caller pads D manually and passes `pad_d=0`. Builder's
`_causal_conv3d` (mojo:218-239): replicate-pad the time axis by slicing first
frame `slice(x,1,0,1)` and last frame `slice(x,1,d-1,1)`, `concat(1,first,x,last)`,
then `conv3d(..., pad_d=0, pad_h=1, pad_w=1)`. half_pad=(3-1)/2=1 frame each side.
`spatial_padding_mode: zeros` → symmetric zero pad of 1 for k=3, handled by the
kernel. `slice`/`concat` semantics verified (tensor_algebra.mojo:743, 675):
slice on dim 1 yields `[N,1,H,W,C]`, concat on dim 1 yields `[N,d+2,H,W,C]`.
The conv then consumes the 2 added frames (kT=3, pad_d=0) back to D=d — confirmed
by smoke output D=4 from input F=4.

### C2 — OIDHW → QRSCF weight remap — **PASS**
`_conv3d_w` (mojo:171-199): reads OIDHW flat index
`(((o*cin+ci)*kd+d)*kh+r)*kw+c`, writes QRSCF flat index
`(((d*kh+r)*kw+c)*cin+ci)*cout+o`, output shape `[kd,kh,kw,cin,cout]`. For
row-major `[Q,R,S,Cin,Cout]` the canonical flat index is exactly
`((((d*R+r)*S+c)*Cin+ci)*Cout+o)` — **matches**. No transpose error.
Dtype path is correct: `to_host` upcasts BF16→F32 (tensor.mojo:220-223), remap
is a pure index permutation on the F32 list, `from_host(...,w.dtype()=BF16)`
casts back to BF16 (tensor.mojo:108-111). No spurious precision change beyond
the original BF16 storage.

---

## D. Smoke run — **PASS**

GPU free: 22911 MiB (ample). Binary at `/tmp/ltx2_vae_decode_stage0_smoke`.

```
LTX-2.3 Video VAE decoder STAGE 0 smoke
  loading stage-0 weights from: .../ltx-2.3-22b-distilled.safetensors
  weights loaded.
  [stat] input latent (NCDHW) mean= -5.4977834e-05 std= 0.24512202 absmax= 0.40039063 n= 32768 finite= True
  output shape (NDHWC): [ 1 , 4 , 8 , 8 , 1024 ]
  [stat] stage0 output mean= -0.0039609275 std= 0.14607155 absmax= 2.421875 n= 262144 finite= True
  PASS: stage-0 forward produced finite output of expected shape.
run_exit=0
```
Finite, no NaN/Inf. Output shape `[1,4,8,8,1024]` (NDHWC) == expected (F/H/W
preserved, channels lifted 128→1024). Stats sane: absmax 2.42, std 0.146.

---

## E. No regressions

### E1 — git status — **PASS (with note)**
`?? serenitymojo/models/vae/ltx2_vae_decoder.mojo`
`?? serenitymojo/pipeline/ltx2_vae_decode_stage0_smoke.mojo`
Both new files present as untracked. **NOTE:** the working tree also shows
` M conv3d.mojo`, ` M decoder2d.mojo`, ` M ldm_decoder.mojo`,
` M qwenimage_decoder.mojo`, and untracked `wan22_decoder*.mojo` from OTHER
teams' work — NOT introduced by LTX2 Team 2. Not this team's regression; flagged
for awareness only. The audited `conv3d.mojo` content is correct and unchanged in
the dimensions stage 0 relies on.

### E2 — Idempotent rebuild — **PASS**
```
pixi run mojo build -I . -Xlinker -lm \
  serenitymojo/pipeline/ltx2_vae_decode_stage0_smoke.mojo \
  -o /tmp/ltx2_vae_decode_stage0_smoke
build_exit=0
```

---

## Builder question resolutions

**Q1 — Non-causal (symmetric-replicate) vs causal-left pad → SYMMETRIC-REPLICATE. Builder is CORRECT.**
`ltx2_vae.rs:117-124`:
```
let half_pad = (self.kernel.0 - 1) / 2;           // = 1 for k=3
let first = x.narrow(2, 0, 1)?;
let last  = x.narrow(2, d - 1, 1)?;
let first_rep = first.repeat_axis_device(2, half_pad)?;
let last_rep  = last.repeat_axis_device(2, half_pad)?;
Tensor::cat(&[&first_rep, x, &last_rep], 2)?       // [first, x, last]
```
Despite the `CausalConv3d` type name, with `causal_decoder: false` (verified
B2) it replicates the FIRST **and** LAST frame 1× each side — symmetric, NOT
left-only causal. Mojo `_causal_conv3d` (mojo:232-235) does
`concat(1, first, x, last)` with single first/last frames — exact match. Builder
did NOT get it backwards.

**Q2 — PixelNorm eps placement → INSIDE the sqrt. Builder is CORRECT.**
Rust `ltx2_vae.rs:213`: `mean_sq.add_scalar(PIXEL_NORM_EPS)?.rsqrt()` → eps added
before rsqrt = inside sqrt. Mojo `ops/norm.mojo:68` (and bf16 line 103):
`var inv = 1.0 / sqrt(shared[0] / Float32(cols) + eps)` → `1/sqrt(mean(x²)+eps)`,
eps INSIDE the sqrt. Byte-for-byte the same formula. Both eps = 1e-6.

**Q3 — NDHWC layout self-consistency → CONFIRMED. Builder is CORRECT.**
`conv3d.mojo` natively expects NDHWC input / QRSCF filter (header lines 16-23,
verified against SDK kernel `conv3d_gpu_naive_ndhwc_qrscf`). Builder keeps the
whole stage channel-last `[B,F,H,W,C]`, so PixelNorm-over-channel == rms_norm
last-dim reduction, and stats broadcast as `[1,1,1,1,C]`. The permute
(0,2,3,4,1) from input NCDHW to NDHWC (mojo:288-290) is correct. Output is
NDHWC `[B,F,H,W,1024]` — confirmed by smoke. Fully self-consistent.

---

## Smoke run output
See section D above (captured to `/tmp/ltx2_vae_run.log`). Finite, shape
`[1,4,8,8,1024]`, exit 0.

---

## Bugfix Worklist
No correctness bugs found in LTX2 Team 2's stage-0 code. Optional / non-blocking:

1. (INFO, not this team) Working tree has uncommitted modifications to
   `conv3d.mojo`, `decoder2d.mojo`, `ldm_decoder.mojo`, `qwenimage_decoder.mojo`
   and untracked `wan22_decoder*.mojo` from other teams. Coordinate before commit
   so the LTX2 commit does not accidentally include unrelated changes.
2. (NIT) `decode_stage0` comptime params `[B,C,F,H,W]` are documentation-only
   (conv3d reads dims at runtime); fine as-is. Consider asserting C==128 at
   comptime for earlier error surfacing — already raised at runtime (mojo:284).
3. (FUTURE) Stage 0 is BF16 end-to-end; downstream stage builders should keep
   the same channel-last NDHWC contract so `up_blocks.1` DepthToSpace can consume
   `[B,F,H,W,1024]` without a re-permute.
