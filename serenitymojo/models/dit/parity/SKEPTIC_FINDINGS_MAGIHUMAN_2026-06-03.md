# SKEPTIC FINDINGS — MagiHuman DiT (CHUNK A: SharedTransformerLayer, layers 4..35)
Date: 2026-06-03 · Reviewer: adversarial skeptic · Verdict: **CLEAN (0 blockers)**

Re-ran parity MYSELF: `pixi run mojo run -I . .../magihuman_block4_gate.mojo`
→ exit 0, **cos = 0.999964454341149**, max_abs = 16.004, n = 655360. Matches builder claim.
(Output std = 91.9, range ±1151 → max_abs 16 ≈ 1.4% worst-element, consistent w/ bf16
matmul rounding across qkv·proj·up·down, not a hidden bug.)

## Oracle-reflects-Rust verification (the Kandinsky5 trap)
I re-derived every unusual piece from `magihuman_dit.rs` + `dit_module.py`, NOT from the harness.

### 1. ElementWise Fourier RoPE bands — VERIFIED (one FRAGILE note)
- Rust loads `adapter.rope.bands` FROM THE CHECKPOINT (line 702); it does NOT compute them.
- Oracle FABRICATES bands = `1/(10000^(arange(0,16)/16))` instead of reading the tensor.
- I dumped the real `adapter.rope.bands` [16] BF16 from the checkpoint and compared:
  ckpt = [1.0, 0.5625, 0.3164, 0.1777, 0.1001, …]  vs  formula = [1.0, 0.5623, 0.3162, 0.1778, 0.1000, …].
  → The checkpoint bands ARE the bf16-rounded formula. The formula is correct in value.
- 16 bands (head_dim/8), 3 axes, sin-then-cos cat → [L,96], split [L,48]/[L,48]: ALL match Rust
  `rope_from_coords` (lines 736–746) and `ROPE_DIM=(128/8)*2*3=96`.
- **FRAGILE**: oracle uses f32 formula bands, Rust uses bf16 checkpoint bands. At the synthetic
  1D coords (h-axis 0..127, t=w=0) the proj delta is sub-ulp; harmless here. If the full Fourier
  rope is ever gated, READ the checkpoint tensor — do not reuse the formula.
- **FRAGILE**: oracle rope is a synthetic 1D instance (t=w=0). Faithfully exercises the
  partial-halfsplit path the block consumes, but does NOT test real multi-axis coords / scales /
  centers. Acceptable for a block gate (block only ingests cos/sin); noted as unmeasured surface.

### 2. Partial halfsplit RoPE — VERIFIED correct (NOT interleaved)
- Rust `apply_rope_to_heads`: `out = x_rot*cos_full + rotate_half(x_rot)*sin_full`,
  `cos_full = cat([cos,cos])`, `rotate_half = cat([-x2, x1])`, x1=x[:48] x2=x[48:].
- Mojo `ops/rope.rope_halfsplit` kernel: out[i]=x[i]·cos[i]−x[i+h]·sin[i];
  out[i+h]=x[i+h]·cos[i]+x[i]·sin[i]. Algebraically IDENTICAL to the Rust halfsplit (verified
  both index halves). It is HALFSPLIT, not interleaved. ✅
- Partial split: rotate first 96 of 128, concat passthrough tail [96:128]. Mojo `_rope_partial`
  slices [0:96] → rope_halfsplit → concat x[96:128]. Matches Rust `rope_partial_halfsplit`
  (lines 175–178). ✅
- sin/cos assignment: Rust forward `sin=rope[:,0:48]`, `cos=rope[:,48:96]`; oracle build_rope
  `sin_emb=rope[:,:48]`, `cos_emb=rope[:,48:]`; both call `apply(x, cos, sin)`. ✅

### 3. swiglu7 kernel (hand-rolled) — VERIFIED line-by-line vs Rust 377–382
- INTERLEAVED split: glu=x[0::2] (even), linear=x[1::2] (odd). Mojo: x[2c]=glu, x[2c+1]=lin. ✅
- clamp: glu = clamp_max(7) one-sided; linear = clamp(−7,7) two-sided. Mojo matches exactly. ✅
- gate: sigmoid(1.702·glu)·glu · (linear+1). Mojo: 1/(1+exp(−1.702·glu)) · glu · (lin+1). ✅
- F32 math, [..,D]→[..,D/2]. ✅. The 1.702 GELU-approx const, the (lin+1) factor, and the
  asymmetric clamp are all correct. (gelu7 for layers 0..3 omits (lin+1) — out of scope, noted.)

### 4. (weight+1) RMSNorm gain — VERIFIED, no double-apply
- Rust `mm_rms_norm_single`: normed·(weight+1) (line 62, 72). Pre-adds via `precompute_w_plus_1`.
- Mojo passes a pre-built `.p1 = add_scalar(w,1.0)` into reused `rms_norm` (out=normed·weight).
  Net = normed·(weight+1). Applied ONCE per norm (pre_norm, q_norm, k_norm, mlp.pre_norm). No
  double +1. Gate builds .p1 in the harness; block reads `*.p1`. ✅

### 5. Fused QKV + GQA grouping — VERIFIED grouped (not interleaved-wrong)
- qkv split 5120/1024/1024/40 = 7208; checkpoint `linear_qkv [7208,5120]` confirms. ✅
- GQA n_rep=5: Rust `repeat_interleave_dim1` → out head j ← kv head j//5 (consecutive copies).
  Mojo `_gqa_expand` [1,Hkv,1,L,D]→[1,Hkv,5,L,D]→[1,40,L,D], out head j ← kv j//5. Same map. ✅
- per-head gating: att *= sigmoid(g), g:[L,40] broadcast over Dh. Mojo `_mul_bcast_lastdim`
  with g3=[L,H,1]. Rust line 594–595 identical. ✅

### 6. Weights / naming / single-stream — VERIFIED against real checkpoint
- All 8 keys present at `block.layers.4.*` AND `block.layers.35.*`, no renames:
  pre_norm[5120], q/k_norm[128], linear_qkv[7208,5120], linear_proj[5120,5120],
  mlp.pre_norm[5120], up_gate_proj[27304,5120], down_proj[5120,13652]. ✅
- Layer 4 has ZERO `cross`/`w_cross_attention` keys → README "self-attention-only" CONFIRMED. ✅

### 7. Op reuse / Mojo hygiene — VERIFIED
- Reused: linear, rms_norm, sdpa_nomask, rope_halfsplit, sigmoid, cast_tensor, tensor_algebra.
  Only NEW: swiglu7 kernel (one enqueue_function) + `_weight_plus_1` — both justified. ✅
- Mojo: NO `var ref`; uses `comptime` (not alias); all `def` (raising); List+ArcPointer Dict;
  NO torch/numpy/autograd leak (only a Python-formula comment). ✅

## Honest gaps (NOT failures)
- **FRAGILE (oracle vs production sdpa)**: oracle does F32 SDPA to match the Mojo math-mode path;
  Rust production uses flame BF16 SDPA. Reference python (`dit_module.py`) is f32, so the oracle is
  faithful to canonical math, but parity vs the bf16-flash production kernel is UNMEASURED.
- **UNMEASURED (full forward)**: `sdpa_nomask` math-mode materializes [B*H,S,S] F32; Dh=128 at
  full sequence length OOMs. Confirmed this is the SHARED foundation SDPA gap (no flash for Dh≠64,
  used by 10+ DiTs), NOT a MagiHuman model bug. Block gate L=128 is genuine; full forward + the 40-
  layer stack + adapter/Fourier-rope kernel + MM layers 0..3,36..39 + heads + SR DiT + unipc are
  all explicitly DEFERRED, not claimed.

## Tags
- BLOCKER: none.
- FRAGILE: (a) oracle fabricates rope bands by formula instead of reading `adapter.rope.bands`
  (value-correct only because ckpt==bf16(formula); fix: read the tensor if full rope is gated);
  (b) synthetic 1D rope coords leave multi-axis path untested; (c) oracle f32-sdpa ≠ Rust bf16-sdpa.
- STYLE: `_weight_plus_1` defined but unused in block (the gate harness does the +1 inline) — dead
  helper, harmless.

{component:"magihuman_dit", reRanParity:true, blockCos:0.999964454341149, oracleReflectsRust:true, fourierRopeCorrect:true, swiglu7Correct:true, blockers:[], verdict:"clean"}
