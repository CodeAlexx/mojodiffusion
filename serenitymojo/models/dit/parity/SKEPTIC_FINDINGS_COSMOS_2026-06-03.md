# SKEPTIC FINDINGS — Cosmos-Predict2.5-2B DiT (2026-06-03)

Adversarial review of the pure-Mojo+MAX port. Reran every available gate/probe;
re-derived every claimed formula from the CANONICAL python source
(`cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py`) and the Rust ref,
NOT just the harness. Verdict up front: **CLEAN (0 blockers).** The block-0
parity reproduces, the new per-axis NTK op is numerically faithful to source, the
scalar builder is not regressed, half-split is correct, the config matches the
689-tensor checkpoint, and the full-res OOM is a genuine foundation kernel gap.

---

## RE-RAN (I executed these myself, read the real output)

| gate/probe | result | exit |
|---|---|---|
| `ops/rope_tables_probe.mojo` (scalar builder regression) | max_err 1.27e-07, **PASS** | 0 |
| `models/dit/parity/cosmos_block0_gate.mojo` | **cos=0.99999604976347**, max_abs=6.92, n=262144, **PASS** | 0 |
| `pipeline/cosmos_dit_probe.mojo` (per-axis rope op runs for real) | axes 44/42/42, thetas 10000.0/31694.02/31694.02, cos[4,64], block ran, **PASS** | 0 |
| `pipeline/cosmos_dit_full_smoke.mojo` (real 4.1GB ckpt, small grid) | out [16,2,16,16], finite ssq=21.06, 569 weights loaded, **PASS** | 0 |

Block cos reproduces the builder's claimed 0.99999605 to the digit. max_abs=6.92 is
expected (cosmos residual magnitudes are huge; cos is the meaningful metric).

---

## PER-AXIS NTK THETA — the new op [VERIFIED CORRECT, against SOURCE]

Re-derived from `minimal_v4_dit.py:695-715`:
- `dim_h = dim//6*2 = 42`, `dim_w = 42`, `dim_t = dim - 2*dim_h = 44` (sum 128). ✓
- halves 22/21/21 sum to 64 = head_dim/2. Confirmed by ckpt buffers
  `pos_embedder.dim_spatial_range [21]`, `dim_temporal_range [22]`. ✓
- `h_ntk = ratio**(dim_h/(dim_h-2))` (line 713) → exponent **42/40 = 1.05** is right.
- `h_theta = 10000 * 3^1.05 = 31694.02`; `t_theta = 10000 * 1^... = 10000`. ✓
- Axis APPLICATION: ratio 3.0 on h,w; 1.0 on t (lines 755-761 + cat order
  `[t,h,w]` line 785). The Mojo `cosmos_rope_axes`/`cosmos_rope_thetas` return
  `[t,h,w]` = `[44,42,42]` / `[10000, 31694, 31694]`. Correct axes, NOT swapped.
- Exponent FORM: python `freq = 1/(theta**(2i/dim)) = theta^(-i/half)`; Mojo
  kernel `inv_freq = theta_a^(-local_i/ha)`. Identical (verified `2i/dim ≡ i/half`).

**Independent numeric cross-check (NOT via the harness):** reproduced the Mojo
per-axis builder's angle table in numpy and compared to the canonical
`VideoRopePosition3DEmb.generate_embeddings` (image-mode, no fps): cos/sin table
cossim = 0.9999999999999, max abs diff 2.4e-7 (= F32 epsilon). The op is faithful
to the SOURCE, not just self-consistent with the oracle.

### The ratio (3.0 vs 2.0) was a real trap — I resolved it empirically
The V1 default `net.py` would give the 2B h=w ratio **1.0** (inherits 7B base);
the predict2.5 experiment configs split between **2.0** (stage3_2B:131) and
**3.0** (stage3_2B:278, sparse_2B). The rope ratio is a RUNTIME hyperparam, NOT
stored in weights — so the checkpoint alone cannot disambiguate. I used the
captured `rope_freqs` (the ground truth the shipped checkpoint was run with) and
fit each axis block: ratio=3.0 → maxerr 0.003; ratio=2.0 → 0.077; ratio=1.0 →
0.22. **The capture unambiguously selects ratio 3.0.** Builder's h=w=3.0 is
CORRECT for the shipped post-trained checkpoint. [STYLE] The header should cite
that the ratio is pinned by the capture, not by a config file, since the config
tree contains 1.0/2.0/3.0 variants for "2B".

---

## SCALAR BUILDER — NOT REGRESSED [VERIFIED]
- `build_multiaxis_rope_tables` is untouched; the per-axis variant is an ADDITIVE
  new function (`build_multiaxis_rope_tables_per_axis`) sharing nothing mutable.
- `rope_tables_probe.mojo` (used by wan22/kandinsky5) reran → max_err 1.27e-07 PASS.
- Per-axis kernel is line-for-line identical to the scalar kernel except it reads
  `theta_a` from a per-axis F32 device buffer instead of a scalar. When all thetas
  are equal it reduces EXACTLY to the scalar kernel (same exponent, same per-column
  `ha` walk). Confirmed by inspection + the equal-theta reduction argument.

---

## HALF-SPLIT vs INTERLEAVED [VERIFIED HALF-SPLIT]
- Python builds `em = cat([t,h,w] * 2, dim=-1)` (`minimal_v4_dit.py:785-791`) →
  angle[d] == angle[d+64]. Confirmed in the capture: `rope[:, :64] == rope[:, 64:]`
  exactly (max diff 0.0). This is GPT-NeoX half-split, NOT interleaved.
- Apply path uses TE `apply_rotary_pos_emb(..., fused=True)` with NO
  `interleaved=True` → TE default (half-split/NeoX).
- Rust ref documents it explicitly (lines 20/29-35): `cat([t,h,w]*2)`, index `d`
  pairs `d+head_dim/2`, fed to `rope_halfsplit_bf16`.
- Mojo uses `ops/rope.rope_halfsplit`. ✓ (wan22 was interleaved — different model,
  correctly distinguished here.)

---

## CONFIG vs CHECKPOINT [VERIFIED — all 689 tensors accounted]
Inspected `cosmos_predict25_2b_dit.safetensors` header directly:
- **689 tensors** = 28 blocks × 24 keys (20 real + 4 `_extra_state`) + 17 non-block.
  Loader loads 569 = 689 − 3 (`pos_embedder.*`) − 4 (`accum_*`) − 113 (`_extra_state`:
  28×4 + `t_embedding_norm._extra_state`). Full-smoke printed `loaded weights: 569`. ✓
- `x_embedder.proj.1.weight [2048,72]` → patch_in 72 = (16 + lvg 1 + pad 1)·2·2·1. ✓
- `crossattn_proj.0.weight [1024,100352]` + `.bias [1024]` → use_crossattn_projection. ✓
- NO `extra_pos_embedder`/abs-pos keys → `extra_per_block_abs_pos_emb=false`. ✓
- NO `k_img`/`v_img` keys → text-only cross-attn (no image branch for V2_2B). ✓
- `cross_attn.k_proj/v_proj [2048,1024]` → text k/v in-dim 1024. ✓
- `pos_embedder.{seq,dim_spatial_range,dim_temporal_range}` correctly SKIPPED
  (rope buffers, ratio-independent, recomputed by the op). ✓
- No silent renames: `_block_weights` lists 20 explicit suffixes that exactly match
  the 20 real per-block keys; `_extra_state` correctly excluded.
- [STYLE] Rust ref comment at `:1622` says "patch_dim=68"; that is a stale Rust
  author comment (its line 78 correctly says 18→72). Mojo uses 72 (ckpt ground
  truth). Not a Mojo bug.

---

## THE OOM CLAIM — FOUNDATION GAP, NOT A MODEL BUG [VERIFIED]
- `ops/attention.mojo` math fallback (Dh==128 path, since SDK flash fails to
  compile for Dh=128 on sm_86) materializes a `[B*H*S, S]` F32 scores buffer.
- Full-res N (capture) = 21·30·52 = **32760**. `[H,S,S]` F32 = 16·32760²·4 =
  **68.7 GB** (even a single head's [S,S] = 4.3 GB). Vastly over 24 GB → genuine OOM.
- This is the inherent O(S²) of attention (correct), exposed because there is no
  tiled/flash kernel for Dh=128 on this GPU — a FOUNDATION/kernel gap, not an
  accidental O(S²) the model shouldn't have. The block math is independent of the
  OOM: the block-0 gate and full smoke both run the SAME block code at small N and
  pass (cos 0.99999605, finite output, correct shape).

---

## unpatchify patch_temporal==1 [VERIFIED CORRECT for V2_2B]
- V2_2B nets all set `patch_temporal=1` (text2world/video2world `defaults/net.py:32/65`;
  ckpt confirms via x_embedder in-dim 72 = 18·2·2·1). The `pt!=1 → raise` guard is
  correct and unreachable for this variant.
- Layout `(p1,p2,t',c)` with c slowest, target `b c (t t') (h p1) (w p2)`
  (`minimal_v4_dit.py` rearrange; Rust `:1580`). Mojo rank-6 reduction (pt=1 drops
  the size-1 t') permute `[5,0,1,3,2,4]` is the exact reduction of the Rust rank-8
  `[0,7,1,6,2,4,3,5]`. Verified dim-by-dim: both yield `[c,Tp,Hp,p1,Wp,p2]`. ✓

---

## OTHER CHECKS
- FinalLayer: 2-chunk (shift,scale), reads first 2D of adaln_lora; SiLU on emb
  only, lora added after the two linears (Rust `:1646-1673`, Python `:1097-1127`).
  Mojo matches (file lines 620-631). ✓
- Timestep embedding cos-first: `cat([cos,sin])` (`minimal_v4_dit.py:876-878`).
  Mojo header asserts cos-first via `timestep_embedding`. ✓
- Residual stream F32, sub-blocks BF16, gated residual in F32, per-head RMSNorm on
  Q,K (not V) eps 1e-6, no-affine LayerNorm modulation — all match the oracle and
  reproduce in the gate.
- MOJO style: `def ... raises`, `comptime`, `List[ArcPointer[Tensor]]` for the
  cross-attn per-head accumulation, `var` everywhere, no obvious leaks. One unused
  `for i` warning in the probe's `_rand` (cosmetic).

---

## HONEST GAP (not a pass, not a blocker)
- **Full-forward cos parity is UNMEASURED at full resolution** (OOM-blocked at
  N=32760). The numeric evidence is: (a) block-0 cos 0.99999605 on real-checkpoint
  weights at a cropped real-activation sub-grid (N=128), and (b) shape-correct +
  finite full forward at a tiny synthetic grid. End-to-end full-res cos vs the
  reference is a GAP awaiting a Dh=128 tiled-attention kernel. This is disclosed
  honestly in the file headers.
- **The block-0 gate does NOT exercise the per-axis rope OP** — it loads cos/sin
  from the fixture (split from the checkpoint's captured `rope_freqs`). The per-axis
  op's correctness is instead established by (i) `cosmos_dit_probe.mojo` running it
  for real and (ii) my independent numpy cross-check vs the canonical
  `generate_embeddings` (cossim ~1.0). Acceptable, but a dedicated per-axis-op
  numeric gate (Mojo op output vs captured `rope_freqs[:, :64]`) would close the
  loop inside the harness. [STYLE / FRAGILE-adjacent, not a blocker since proven
  externally.]

---

{component:"cosmos_predict25_dit", reRanParity:true, blockCos:0.99999604976347, perAxisThetaCorrect:true, halfsplitCorrect:true, scalarRopeNoRegression:true, oomIsFoundationGap:true, blockers:[], verdict:"clean"}
