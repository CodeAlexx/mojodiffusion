# SKEPTIC FINDINGS — Cosmos-Predict2.5-2B DiT FULL-FORWARD gate (2026-06-03)

Adversarial review of the COMPOSITION the full-forward oracle adds on top of the
already-canonical-verified per-block math (see SKEPTIC_FINDINGS_COSMOS_2026-06-03.md).
Question posed: is the N=8192 full-forward gate (cos 0.99999453 via tiled SDPA)
CANONICAL-FAITHFUL, or merely TRANSCRIPTION-SELF-CONSISTENT (Kandinsky5 trap —
oracle + Mojo sharing a wrong-but-consistent composition)?

**VERDICT UP FRONT: CANONICAL-FAITHFUL. 0 blockers.** Every composition step the
oracle adds (channel concat, patchify order, x_embedder, timestep MLP routing,
per-block adaLN routing, F32 residual accumulation, rope position decomposition,
FinalLayer 2-chunk, unpatchify) was re-derived directly from the canonical source
`minimal_v4_dit.py` + `minimal_v1_lvg_dit.py` — NOT from the oracle and NOT from
the Mojo. The tiled-vs-flash agreement is a genuine cross-impl check. Gate reran
clean.

---

## RE-RAN (executed myself, read the real output)

| gate | result | exit |
|---|---|---|
| `pixi run mojo run -I . pipeline/cosmos_dit_full_smoke.mojo` | **cos=0.999994529935775**, max_abs=0.0625, n=524288, RMS=1.3665321, out[16,2,128,128], **FULL-FORWARD GATE: PASS** | 0 |

No OOM at N=8192 (tiled SDPA streams K/V, O(N·Dh) peak). 569 weights loaded. The
cos reproduces the claimed 0.99999453 to the digit.

---

## CANONICAL SOURCE OF TRUTH (re-derived each, line-cited)

Canonical = `cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py` (MiniTrainDIT)
wrapped by `minimal_v1_lvg_dit.py` (MinimalV1LVGDiT). Top-level call order:
`MinimalV1LVGDiT.forward` → LVG concat → `MiniTrainDIT.forward` →
`prepare_embedded_sequence` (padding concat + x_embedder + rope) → crossattn_proj →
t_embedder → t_embedding_norm → 28× block → final_layer → unpatchify.

### 1. Channel composition [VERIFIED CANONICAL]
- `MinimalV1LVGDiT.forward:47-52`: concat 1 LVG condition-mask channel (image/non-VIDEO
  mode = zeros[1,T,H,W]) → 16+1=17.
- `MiniTrainDIT.prepare_embedded_sequence` (concat_padding_mask, :~1686): concat 1
  padding-mask channel → 17+1=18. x_embedder patch_in = 18·2·2·1 = 72 (matches ckpt
  `x_embedder.proj.1.weight[2048,72]`).
- Oracle: `cat([x_lat, zeros1, zeros1])` = 16+1(lvg)+1(pad) = 18, both zeros (image
  mode). Mojo (:577-582): `x_lat → +lvg_zeros → +pad_zeros` = 18. Order
  `[latent, lvg, pad]` matches canonical; both masks zero so inter-mask order is
  numerically moot but structurally faithful. ✓

### 2. crossattn_proj [VERIFIED]
- Canonical `MiniTrainDIT.forward` (:~1740 `if use_crossattn_projection`): Linear
  100352→1024 **with bias**, applied before blocks. Oracle line 125 (Linear+bias),
  Mojo :585 (`Optional(bias)`). ✓

### 3. patchify order [VERIFIED CANONICAL]
- Canonical `PatchEmbed` (:1004): `Rearrange("b c (t r)(h m)(w n) -> b t h w (c r m n)")`.
  Token order t-major→h→w; within-patch `(c,r,m,n)` c-SLOWEST, w-patch fastest.
  x_embedder Linear **bias=False**.
- Oracle lines 133-136: reshape `[C,Tp,PT,Hp,PS,Wp,PS]` → permute `[1,3,5,0,2,4,6]`
  → `[Tp,Hp,Wp,C,PT,PS,PS]`, c-slowest within patch; x_embedder via `_lin_nobias`.
  Mojo `patchify3d` + `_lin_nobias_t` (:590-591). ✓

### 4. timestep MLP routing [VERIFIED — the non-obvious one, NOT a shared misread]
- Canonical `TimestepEmbedding` (:884-920, use_adaln_lora=True): `linear_1`
  **bias=False** (because `not use_adaln_lora`); forward = linear_1→SiLU→linear_2(→3D);
  **returns `(emb_B_T_D = sample, adaln_lora = mlp_output)`** — the modulation `emb`
  fed to blocks is the RAW sinusoidal `sample`, NOT the MLP output. Then
  `t_embedding_norm` RMSNorm is applied to it (`MiniTrainDIT.forward:~1755`).
- This is the single easiest place to silently copy a Mojo bug. Both sides route it
  correctly: Oracle (lines 152-154) `adaln_lora = lin(silu(lin(sample,l1)),l2);
  emb = rms(sample, t_embedding_norm.weight)`. Mojo (:604-607) identical, linear_1
  bias=None. Verified against SOURCE, not via the oracle. ✓
- Timestep embedding cos-first: canonical `Timesteps.forward:878` `cat([cos_emb,sin_emb])`;
  exponent `-log(10000)*arange/half`. Oracle `timestep_embedding` `cat([cos,sin])`
  same exponent; Mojo `timestep_embedding(...,10000.0)` header-asserts cos-first. ✓

### 5. per-block adaLN routing + F32 residual [VERIFIED CANONICAL]
- Canonical `Block.forward:1271-1288`: per sub-block `(adaln_modulation_X(emb) +
  adaln_lora).chunk(3)` → **shift, scale, gate** (that order); F32 autocast.
  `_fn = norm*(1+scale)+shift` (:1306). Residual:
  self `x = x + gate*result` (:1339), cross `x = result*gate + x` (:1372),
  mlp `x = x + gate*result` (:1381). Modulation broadcast `b t d -> b t 1 1 d` over
  H,W (= per-T → per-token repeat hpwp).
- Oracle (lines 206-212 `adaln_chunk` returns [:D],[D:2D],[2D:3D] unpacked as sh,sc,ga;
  235/255/270 apply per sub-block; 252/267/276 gated F32 add; `repeat_interleave(hpwp)`).
- Mojo (`cosmos_block_forward` :373-401): `_adaln_chunk3` → sa[0]=shift, sa[1]=scale,
  sa[2]=gate; `_ln_modulate(x, scale, shift)`; `_gated_add(x_f32, out, gate)` F32;
  `_expand_t_to_tokens` = per-T→per-token. Chunk order, modulation formula, residual
  base, and gate placement ALL independently match canonical. A shared misread would
  require both to err identically on chunk(3) order AND the modulate sign — they do
  not. ✓

### 6. rope position decomposition over (Tp,Hp,Wp) [VERIFIED CANONICAL]
- Canonical `VideoRopePosition3DEmb.generate_embeddings:785`: `cat([half_emb_t,
  half_emb_h, half_emb_w]*2, dim=-1)` then `rearrange(... "t h w d -> (t h w) ...")`
  → positions token-major t→h→w, axis order [t,h,w], halves [dim_t/2, dim_h/2,
  dim_w/2] = [22,21,21], NTK per-axis theta. (Per-axis theta/halfsplit already
  canonical-verified in the block findings.)
- Oracle `cosmos_rope` (lines 158-191): pos loops `for ti: for hi: for wi` token-major
  t,h,w; `halves=[dim_t//2,dim_h//2,dim_w//2]=[22,21,21]`; thetas `[t,h,w]`; cat in
  axis order [t,h,w]. Mojo `cosmos_rope_positions`/`cosmos_rope_axes` return [t,h,w]
  / [44,42,42]. Axes NOT swapped, position iteration matches the `(t h w)` flatten. ✓

### 7. FinalLayer 2-chunk [VERIFIED CANONICAL]
- Canonical `FinalLayer.forward:1107-1123`: `(adaln_modulation(emb) +
  adaln_lora[:,:,:2*hidden]).chunk(2)` → **shift, scale**; SiLU on emb only inside
  adaln_modulation; `_fn = norm*(1+scale)+shift`; then Linear D→64 (bias-free).
- Oracle (lines 282-290): `silu(emb)→lin→lin → +adaln_lora[:,:2D]; f_sh=[:D],
  f_sc=[D:2D]; ln_mod; lin → [N,64]`. Mojo (:624-635): `silu(emb)→lin→lin; +adaln_lora
  first 2D; shift=[:D], scale=[D:2D]; _ln_modulate; final_layer.linear`. ✓

### 8. unpatchify [VERIFIED CANONICAL]
- Canonical `unpatchify:1702`: `rearrange("B T H W (p1 p2 t C) -> B C (T t)(H p1)(W p2)")`
  — within-patch `(p1,p2,t,C)` **C innermost/fastest**.
- Oracle (lines 295-298): reshape `[Tp,Hp,Wp,PS,PS,OUT_CH]` c-fastest → permute
  `[5,0,1,3,2,4]` → `[c,Tp,Hp,p1,Wp,p2]`. Mojo `cosmos_unpatchify`. C-fastest matches.
  (pt=1 drops the size-1 t; guard correct for V2_2B.) ✓

---

## TILED vs FLASH — the REAL independent signal [CONFIRMED GENUINE CROSS-CHECK]

- Oracle self-attn (lines 247-248): `with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
  F.scaled_dot_product_attention(...)` — torch fused flash.
- Mojo self-attn (`cosmos_self_attention` :304): `sdpa_nomask_tiled[1,N,H,DH]` →
  `ops/attention.mojo:502+` a hand-written ONLINE-SOFTMAX tiled kernel (running max m,
  running sum l, acc rescale by `exp(m-m_new)` — comments :513-519). Structurally a
  DIFFERENT algorithm from torch flash, not a re-import.
- Inputs byte-identical both sides: x_lat seed=99 scale=0.2, text seed=7 scale=0.05,
  same LCG `s=(s*1103515245+12345)%2147483648`, `(s/2^31 - 0.5)*scale`. Oracle
  `rand4`/`rand2` (lines 96-117) ≡ Mojo `_rand4`/`_rand2` (smoke :83-108). ✓
- ∴ cos 0.99999453 at S=8192 = torch-flash output ≈ Mojo-tiled output on identical
  inputs through 28 self-attn layers = a REAL cross-implementation validation of the
  tiled SDPA, not transcription self-agreement. The 5.5e-6 residual is bf16/f32
  accumulation-order noise across 28 blocks (expected; max_abs 0.0625 on RMS≈1.37).
- Cross-attn note: oracle uses **F32** SDPA (line 263) to mirror Mojo's f32
  linear+softmax cross-attn path (:332-334) — there it is impl-matched, not a flash
  cross-check, but cross-attn math is already block-verified canonical.

---

## KANDINSKY5-TRAP SPECIFIC SPOT-CHECKS (things a transcription would copy from Mojo)
- rope axis order (t,h,w): canonical cat `[t,h,w]` + `(t h w)` flatten → oracle/Mojo
  [t,h,w]. NOT swapped. ✓
- per-axis theta application point: applied as `1/(theta_a ** dim_range)` per axis;
  oracle/Mojo identical, already block-verified. ✓
- modulation chunk order: canonical chunk(3)=shift,scale,gate / chunk(2)=shift,scale;
  oracle AND Mojo independently match — verified vs source, not vs each other. ✓
- head/unpatchify layout: canonical `(p1 p2 t C)` C-fastest input-patch `(c r m n)`
  c-slowest; oracle/Mojo match both. ✓
- t_embedder `emb=sample` (raw, not MLP out): the highest-risk shared-misread point —
  both correct vs canonical. ✓

No shared-misread found. Each composition primitive was checked against the canonical
python, and oracle vs Mojo route the modulation/residual independently (different code,
same canonical contract).

---

## RESIDUAL / FRAGILE (not blockers)
- The gate runs at T=2,H=128,W=128 → N=8192 (a SYNTHETIC LCG grid, image-mode zeros
  for both mask channels). It does NOT exercise: (a) a non-zero LVG condition-mask
  or padding-mask channel (VIDEO data_type), (b) fps modulation in rope (canonical
  `enable_fps_modulation` temporal-scaling branch — gate is image-mode T-as-1-per-frame
  via the cos-first sinusoid; the canonical image path `half_emb_t = outer(seq[:T],
  temporal_freqs)` IS what's reproduced, but the fps!=None scaling branch is untested).
  Both are inference-config branches unused by the text2world/image image-mode path
  this checkpoint+pipeline targets. FRAGILE, not a blocker — if a future VIDEO/fps
  path is added, the mask-channel content and temporal fps scaling need their own gate.
- Cross-attn is f32-matched (not flash-cross-checked); acceptable since block-verified.

---

{component:"cosmos_full_forward", reRanGate:true, fullForwardCos:0.999994529935775, oracleIsCanonicalFaithful:true, tiledVsFlashIsRealCrosscheck:true, compositionVerifiedVsCanonical:true, blockers:[], verdict:"CANONICAL-FAITHFUL — every composition step (channel concat, patchify (c r m n) c-slowest, x_embedder no-bias, t_embedder emb=raw-sample routing, per-block shift/scale/gate adaLN + F32 gated residual, rope (t,h,w) position decomposition, FinalLayer shift/scale 2-chunk, (p1 p2 t C) C-fastest unpatchify) re-derived from canonical minimal_v4_dit.py + minimal_v1_lvg_dit.py, NOT from oracle/Mojo. Oracle self-attn is torch FLASH vs Mojo online-softmax tiled SDPA on byte-identical LCG inputs, so cos 0.99999453 at S=8192 is a genuine cross-impl check of the tiled SDPA, not transcription self-consistency. Untested branches (non-zero VIDEO masks, fps rope scaling) are FRAGILE not blockers — unused by this image-mode pipeline."}
