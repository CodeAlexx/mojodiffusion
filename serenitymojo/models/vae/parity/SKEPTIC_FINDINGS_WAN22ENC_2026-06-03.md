# SKEPTIC FINDINGS — Wan2.2 VAE ENCODER (image-mode T=1) — 2026-06-03

Adversarial review of `serenitymojo/models/vae/wan22_vae_encoder.mojo` +
`parity/wan22_vae_encoder_oracle.py` + `parity/wan22_vae_encoder_parity.mojo`.
Reference read line-by-line: `/home/alex/Lance/modeling/vae/wan/vae2_2.py`
(Encoder3d / WanVAE_.encode / patchify / AvgDown3D / CausalConv3d / RMS_norm /
Down_ResidualBlock / Resample / AttentionBlock). Rust encoder ref present at
`EriDiffusion-v2/.../encoders/wan22_vae.rs` (1397 ln) + inference-flame copy.

## §0 receipts
- `serenitymojo/MAP.md` — "where does X live" wayfinding; pure-Mojo+MAX inference-only GPU lib, BF16 storage / F32 accum.
- `docs/SERENITYMOJO_MODULES.md` — per-module API catalog (not re-read in full; MAP.md sufficient for this gate).
- `wan22_vae.rs` encoder path + `wan22_decoder.mojo` — decoder is the trusted block-library (CausalConv3d, RMS_norm, AvgDown3D↔DupUp3D mirror); encoder reuses its conv3d + `_wan22_mean/std` + helpers.
- `mojo-syntax/SKILL.md` — `comptime` not `alias`, `def` needs explicit `raises`, no `let`, List `^` transfer.

## VERDICT: CLEAN — gate reproduced, architecture faithful to Lance reference.

cos=0.9999837662 @64², mag_ratio=0.9995360, max_abs=0.015625, GATE PASS (≥0.999).
I re-ran the parity myself (read actual stdout, exit 0).

## Reference-vs-Mojo verification (each checked against vae2_2.py line numbers)

### CLEAN — latent scaling (the headline hunt)
- `WanVAE_.encode` (ln 779-783): `mu,log_var = conv1(out).chunk(2,dim=1)` →
  mu = first 48 channels. Then `mu = (mu - scale[0].view(1,z_dim,1,1,1)) *
  scale[1].view(1,z_dim,1,1,1)`, z_dim=48, scale=[mean, 1/std]. Per-channel over
  the CHANNEL axis (dim=1).
- Mojo `encode_image` ln 657-661: `_causal_conv3d("conv1",0,0,0)` →
  `slice(x,4,0,48)` = first 48 of NDHWC channel axis = mu ✓; then
  `mul(sub(mu, self.mean), self.inv_std)` with mean/inv_std `[1,1,1,1,48]`
  broadcasting over the LAST (channel) axis ✓. Applied AFTER the mu chunk ✓.
- Constants: `_wan22_mean()/_wan22_std()` in the trusted decoder are
  BYTE-IDENTICAL to the oracle's MEAN/STD arrays (diffed all 48 each) and are the
  SAME constants the decoder un-normalizes with (inverse: decode does
  `z/scale[1]+scale[0]`). inv_std built as `1/std[i]` per element ✓.
- Per-channel sanity (NOT just global cos): oracle mu per-channel means span
  −1.75…+1.79 (structured, non-degenerate); a wrong axis or swapped mean/std
  would scramble these and collapse cos far below 0.999. mag_ratio 0.99954 ⇒
  magnitude correct, not merely direction. **latentScalingCorrect: true.**

### CLEAN — patchify(2) channel interleave
- Lance ln 281-286: `'b c f (h q)(w r) -> b (c r q) f h w'` ⇒
  out_c = (c·2 + r)·2 + q = 4c + 2r + q.
- Mojo kernel ln 187-190: `q=oc%2; r=(oc//2)%2; c=oc//4`; `hi=oho*2+q;
  wi=owo*2+r` ⇒ exactly inverts 4c+2r+q ✓ (q↔h-subpos, r↔w-subpos, matching the
  `(h q)(w r)` order). Same bug-class as patchify3d but layout is correct.

### CLEAN — AvgDown3D shortcut (parameter-free)
- Lance ln 328-359: left-pad T by `(ft - T%ft)%ft`; view
  `[B,C,T/ft,ft,H/fs,fs,W/fs,fs]`; permute `(0,1,3,5,7,2,4,6)`; view
  `[B, C·factor, ...]` (C-major flatten e=((ci·ft+ti)·fs+si)·fs+sj); view
  `[B, out_c, group_size, ...]`; mean over group_size.
- Mojo ln 304-321: `e=co*group_size+gm`; decode `sj=e%fs; si=(e//fs)%fs;
  ti=(e//fs//fs)%ft; ci=e//fs//fs//ft` ✓; `di=odo*ft+ti, hi=oho*fs+si,
  wi=owo*fs+sj`; left-pad via `if di>=pad_t` (zeros below) ✓; `/group_size` ✓.
- Per-group factors match Down_ResidualBlock(ln 414-419): g0(ft1,fs2),
  g1(ft2,fs2), g2(ft2,fs2), g3(ft1,fs1) — matches encode_image ln 620/628/636/643.
  group_size = Cin·factor/Cout integral-checked ✓. avg_shortcut on ORIGINAL
  input (`x_copy`, ln 435/439) — Mojo clones g0..g3 BEFORE the resblocks ✓.

### CLEAN — RMS_norm eps 1e-12 + √dim fold (the "1e-12 is unusual" hunt)
- Lance ln 73-74: `F.normalize(x, dim=1) * scale * gamma`, scale=`dim**0.5`.
  `F.normalize` = L2 (x/√Σx²) with DEFAULT eps **1e-12**; ·√dim folds it to
  x/√(mean x²)·gamma = RMS-norm.
- Mojo `_VAE_EPS=1e-12` (ln 83), `rms_norm` divides by √(mean x²+eps) — RMS, so
  √dim is ALREADY implicit (RMS=√mean, not √sum); no separate √dim scale needed.
  This is the trusted decoder pattern. eps is correctly 1e-12, NOT 1e-6. ✓

### CLEAN — CausalConv3d temporal LEFT-pad
- Lance CausalConv3d (ln 40-58): `_padding=(pw,pw,ph,ph, 2·pad[0], 0)` — temporal
  pad = `2·pad_d` on the LEFT only (causal), spatial symmetric.
- Mojo `_causal_conv3d` ln 500-512: `time_pad=2*pad_d`; concat zeros on dim=1
  (D/temporal) on the LEFT (`concat(1, ctx, zpad, x)`); spatial pad_h/pad_w passed
  symmetric to conv3d; pad_d=0 to the kernel ✓. conv3d.mojo header confirms the
  SDK kernel pads symmetric, so temporal is handled manually — correct.

### CLEAN — downsample2d ZeroPad2d(0,1,0,1)
- Lance ln 114/116: `nn.ZeroPad2d((0,1,0,1))` = pad (left0,right1,top0,bottom1)
  then Conv2d(dim,dim,3,stride2,pad0). Mojo `_zero_pad_rb` ln 537-545 appends a
  right column then a bottom row (concat dim2 then dim1 in NHWC) ✓; conv2d static
  K3 s2 pad0 ✓.

### CLEAN — middle Attn (single head)
- Lance AttentionBlock (ln 253-272): to_qkv Conv2d1x1, single-head SDPA (default
  is_causal=False ⇒ full attn, default scale 1/√head_dim, head_dim=c), proj
  Conv2d1x1, `+identity`. Mojo `_attn_block` ln 571-593: linear qkv, zero additive
  mask (=full attn), `scale=1/√DIM`, single head [1,SEQ,1,DIM], proj linear,
  `add(identity,out5d)` ✓. out5d reshape uses H3·W3 (middle runs at final spatial) ✓.

### CLEAN — group/middle/head structure + weight names
- Encoder3d (ln 506-537): conv1 CausalConv3d(12→160,pad1); 4× Down_ResidualBlock
  (down_flag = i≠3); middle [ResBlock, Attn, ResBlock]; head [RMS_norm, SiLU,
  CausalConv3d(640→96=z_dim·2)]. WanVAE_.conv1 (ln 742) CausalConv3d(96→96,1x1x1).
  Mojo prefixes `encoder.downsamples.{g}.downsamples.{0,1,2}`,
  `encoder.middle.{0,1,2}`, `encoder.head.{0,2}`, `conv1` — all match the
  nn.Sequential indices. `_is_wan22_encoder_tensor` filters `encoder.*` + top
  `conv1.*` (encoder owns the moments conv1) ✓.

### CLEAN — dtype / oracle bf16-GPU
- Oracle (ln 62-101): `dt=torch.bfloat16`, `dev="cuda"`,
  `model.eval().to(dev,dtype=dt)`, image cast to bf16, `model.encode(x5,scale)` in
  bf16 → returns mu.float() for the dump. This is bf16-on-GPU WanVAE_, NOT
  fp32-CPU. Matches the Mojo BF16 storage / F32-accum path ✓.

### CLEAN — reuse, no reimplementation
- Imports `conv3d` from `models/vae/conv3d`, and `_wan22_mean/_wan22_std,
  _load_conv3d_qrscf_bf16, _clone, _zeros_device, _shape*, _perm5` from the
  trusted `wan22_decoder` ✓. Only genuinely-new ops: `_patchify2_*` and
  `_avgdown3d_*` kernels + a conv2d-RSCF loader (encoder-only resample). No
  duplicate block library.

### CLEAN — Mojo syntax / hygiene
- `comptime` (no `alias`), `def … raises`, List `^` transfers (ln 446-479), no
  `var ref`, io/ffi-only file reads in the parity harness, no torch/numpy/autograd/
  backward leak in the .mojo (only a comment cites the .rs reference path).

## FRAGILE (not blockers)

1. **temporalPathTested = false (T=1 scope).** This gate exercises ONLY the
   single-frame image path. At T=1: (a) CausalConv3d's temporal left-pad IS run
   (2·pad_d zero-frames prepended) but the kernel just sees 1 real + 2 zero
   frames — a SYMMETRIC-pad bug would be INVISIBLE here because there is no
   second real frame to leak across; (b) Resample.downsample3d's `time_conv` is
   NEVER applied for a single chunk (Lance ln 159-169 only runs it when
   `feat_cache[idx] is not None`, i.e. on chunk ≥2) — Mojo `_downsample3d` ≡
   `_downsample2d`, faithful for T=1 but the temporal stride-2 conv is UNTESTED;
   (c) AvgDown3D's temporal averaging at ft=2 collapses 1-real+1-pad-zero → the
   left-pad branch is touched but cross-frame temporal mixing is not.
   ⇒ The temporal-causal path is UNVERIFIED by this gate. Acceptable IF the
   image/Lance single-frame encode is the only current use-case (it is, per the
   file header + decoder image-mode slice). Flag, not blocker. A full T2V encode
   would need the feat-cache chunk loop + downsample3d.time_conv ported and a
   multi-frame parity before trusting video.

2. **256² parity not shipped-runnable.** Builder claims cos 0.99998 @256² too;
   the oracle DID dump `wan22enc_mu_256x256.bin` (meta present, mu_std 0.844), but
   `wan22_vae_encoder_parity.mojo` hardcodes `IMG=64` and only tests 64². The
   256² claim is plausible (same code path, only comptime H/W differ) but is NOT
   reproduced by the shipped gate. Minor: parametrize IMG or add a 256² parity
   file to make the second-resolution claim self-verifying.

## STYLE
- None material. The two new kernels carry derivation comments inline; layout
  inversions are documented at the call site.

```json
{"component":"wan22_vae_encoder","reRanParity":true,"cos":0.9999837662018228,"magRatio":0.9995360300878464,"latentScalingCorrect":true,"temporalPathTested":false,"blockers":[],"verdict":"clean"}
```
