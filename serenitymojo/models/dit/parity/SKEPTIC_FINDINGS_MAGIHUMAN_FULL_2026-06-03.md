# SKEPTIC FINDINGS — MagiHuman DiT FULL FORWARD (CHUNK B)
Date: 2026-06-03 · Reviewer: adversarial skeptic · Verdict: **CLEAN (0 blockers)**

## Re-ran the gate MYSELF
`pixi run mojo run -I . serenitymojo/models/dit/parity/magihuman_full_gate.mojo`
→ exit 0, **cos = 0.9990820196469521**, max_abs = 0.2408, n = 24576 (=128×192). No OOM
(streams the 30.6 GB ckpt layer-by-layer; ran in ~minutes). Matches builder's 0.99908.
CHUNK A block-4 gate re-run: **cos = 0.999964443743454** (still PASS, matches prior).

## KANDINSKY5 RULE — oracle is a TRANSCRIPTION, so I re-derived COMPOSITION vs canonical
`gen_magihuman_full_oracle.py` does NOT instantiate the canonical `DiTModel`/`load_state_dict`.
It HAND-TRANSCRIBES the forward (numpy/torch ops) — exactly the cosmos/kandinsky5 trap. I therefore
re-derived EVERY composition piece against BOTH the Rust truth source (`magihuman_dit.rs`) AND the
canonical Python (`daVinci-MagiHuman/inference/model/dit/dit_module.py` + `common/config.py`).
**Mitigating fact:** the oracle reads the REAL checkpoint for `adapter.rope.bands` (line 250) and
ALL weights; only the forward *wiring* is transcribed. So bands fragility from CHUNK A is RESOLVED
(both oracle and Mojo gate read the same checkpoint bands tensor).

### LAYER MEMBERSHIP — VERIFIED canonical (config.py defaults)
- `common/config.py`: `num_layers=40`, `mm_layers=[0,1,2,3,36,37,38,39]`, `gelu7_layers=[0,1,2,3]`,
  `post_norm_layers=[]`, `local_attn_layers=[]`, `enable_attn_gating=True`, head_dim=128, hidden=5120.
- `dit_module.py:744` `num_modality = 3 if layer_idx in mm_layers else 1`;
  `:761` `GELU7 if layer_idx in gelu7_layers else SWIGLU7`.
- Oracle `MM_LAYERS={0,1,2,3,36,37,38,39}`, `GELU7_LAYERS={0,1,2,3}`, `use_swiglu=i not in GELU7`.
  → layers 36-39 are MM **but SwiGLU7** (correct, the easy-to-miss case). ✅
- Mojo `_is_mm_layer(i)= i<4 or i>=36`; `_is_gelu7_layer(i)= i<4`; stack uses
  `use_swiglu = not _is_gelu7_layer(i)`. EXACT match to canonical. ✅
- post_norm=[] and local_attn=[] confirm NO post-norm / NO local-attn branch fires in the main DiT
  (oracle/Mojo omit both — correct).

### MM BLOCK — VERIFIED vs canonical NativeMoELinear + MultiModalityRMSNorm
- `MultiModalityRMSNorm.forward_multi_experts` (dit_module.py:272): `rms(x)` over last dim, then
  `weight.chunk(num_modality, dim=0)`, `t_list[i]*(weight_chunked[i]+1)`. **chunk AXIS 0.** ✅
  Oracle `mm_rms_norm_multi` chunks `weight_full[i*last:(i+1)*last]` (axis 0) · `(+1)`. ✅
  Mojo `_mm_rms_norm_p1` slices `weight_p1[i*last_dim, last_dim]` + reuses `rms_norm` with pre-`.p1`
  gain. ✅ (single +1 — gate builds `.p1=add_scalar(w,1.0)`.)
- `NativeMoELinear.forward` (dit_module.py:353): `weight.chunk(num_experts, dim=0)`, weight shape
  `[out*num_experts, in]`, per-chunk `input_i @ weight_chunked[i].t()`. **chunk AXIS 0** (weight rows).
  ✅ Oracle `mm_linear` chunks `weight_full[i*out_per:(i+1)*out_per]` (axis 0) `@ chunk_w.t()`. ✅
  Mojo `_mm_linear` slices weight rows `[i*out_per, out_per]` then `linear(x @ wᵀ)`. ✅
  bias=False confirmed for linear_qkv/proj/up/down (dit_module.py:586,593,678,687) — oracle/Mojo
  correctly ignore bias for these. ✅
- gelu7 vs swiglu7 selector + intermediate sizes: SwiGLU7 `gated_act=True` →
  up_gate out=2·13652=27304, down in=13652 (swiglu7 halves). GELU7 `gated_act=False` →
  up_gate out=20480, down in=20480 (NO halving). Oracle/Mojo derive `up_per=up_w.shape[0]//3` and
  `down_per=down.shape[0]//3` FROM THE REAL CHECKPOINT, so both widths are correct automatically. ✅

### gelu7 / swiglu7 FORMULAS — VERIFIED line-by-line vs canonical (dit_module.py:50-70)
- swiglu7: `x_glu=x[...,::2].clamp(max=7)`, `x_linear=x[...,1::2].clamp(-7,7)`,
  `out_glu*sigmoid(1.702·x_glu)` `*(x_linear+1)`. F32. Oracle + Mojo `swiglu7` IDENTICAL (CHUNK A
  already verified the Mojo kernel; same kernel reused here). ✅
- gelu7: `x.clamp(max=7)`, `clamped·sigmoid(1.702·clamped)` — **NO interleave split, NO (linear+1)**.
  Oracle `gelu7` and Mojo `_gelu7_kernel_f32` both clamp_max(7) then `x·sigmoid(1.702x)`, same width
  out. EXACT. ✅ (the distinguishing gelu7-vs-swiglu7 details are all correct.)

### ADAPTER EMBEDDERS + rope_from_coords — VERIFIED vs canonical Adapter/ElementWiseFourierEmbed
- Embedders: `nn.Linear(in, hidden, bias=True)` per modality (dit_module.py:720-722). Oracle/Mojo:
  `x @ w.t() + b`. ✅ Tokens sorted V,A,T; masks disjoint so write-order irrelevant. ✅
- `ElementWiseFourierEmbed.forward` (dit_module.py:198-230): xyz=c[:,:3], sizes=c[:,3:6],
  refs=c[:,6:9]; `scales=(refs-1)/(sizes-1)` with (refs==1&sizes==1)→1; `centers=(sizes-1)/2`,
  `centers[:,0]=0`; `proj=(xyz-centers)[...,None]·scales[...,None]·bands` `[L,3,16]`;
  `cat((sin,cos),dim=1).flatten(1)` → `[L,96]`. Oracle `rope_from_coords` and Mojo
  `magihuman_rope_from_coords` reproduce this EXACTLY (eps 1e-30 in the denom is sub-ulp at test
  coords where sizes=refs=L=128>1, special-case never fires). ✅
- `freq_bands(dim//8=16, temp=10000, step=1)` = `1/(10000^(arange(0,16)/16))`. Both oracle AND the
  Mojo gate read the REAL `adapter.rope.bands` tensor from the checkpoint (gate line 95) instead of
  the formula — the CHUNK-A fragility is FIXED here. ✅
- rope arg order: canonical `apply_rotary_emb_torch(q, cos, sin)` → `x·cos + rotate_half·sin`,
  `rotate_half=cat(-x2,x1)`, partial first ro_dim=96 then passthrough. Oracle `apply_rope_partial`
  and Mojo `_rope_partial` (halfsplit) match (CHUNK A verified halfsplit≡this). sin=rope[:,:48],
  cos=rope[:,48:] via `tensor_split(2,-1)`. ✅
- **GAP (honest, builder-flagged):** test coords sweep ONLY the h-axis (col1), with t=col0=0 and
  w varying only trivially; sizes=refs=L. So the multi-axis t/w scale+center math is exercised
  algebraically (all 3 axes are computed) but NOT swept over non-trivial t/w values. The per-axis
  code path is identical across axes, so this is low-risk, but real multi-axis coords are UNMEASURED.

### FINAL HEADS — VERIFIED vs canonical DiTModel.forward (dit_module.py:893-948)
- `final_norm_video/audio = MultiModalityRMSNorm(hidden, num_modality=1)` → `(weight+1)` single gain.
  `final_linear_video=Linear(hidden,192,bias=False)`, `final_linear_audio=Linear(hidden,64,bias=False)`.
- `x_out=zeros(L, max(192,64)=192)`; video rows←[:192], audio rows←[:64] (zero-pad 64→192), text
  rows stay zero. Oracle and Mojo `magihuman_final_heads` reproduce: rms_norm_single(+1) → linear →
  audio cat zero-pad [a,128] → text zero block. ✅

### THE _concat_arc PAIRWISE WORKAROUND — VERIFIED numerically identical
- `_concat_arc(dim, pieces)`: `acc=concat(p0,p1); for i in 2..n: acc=concat(acc,pi)`. For the 3-way
  V,A,T case: `concat(concat(pv,pa),pt)` = `[pv;pa;pt]` along the SAME dim. Concatenation is
  associative and order-preserving → byte-identical to a single 3-way cat. Right order (V,A,T),
  correct axis (dim=0 for token rows in embed/mm_norm/mm_linear/heads; dim=1 only for the audio
  192-pad). No axis error. ✅ The "pairwise" exists only to dodge a cross-module variadic mis-bind,
  not a semantic change.

## SCOPE HONESTY
- Gate is **L=128** (V=64,A=32,T=32), NOT production ~16K tokens. The large-S TILED SDPA path of
  MagiHuman is therefore **NOT exercised here** (`sdpa_nomask_tiled` runs but at small S). The tiled
  online-softmax op itself is already skeptic-clean (proven at full res in wan22/cosmos, cos=1.0 vs
  math-mode). So large-S correctness is INHERITED, not re-proven by this gate. `largeSExercised:false`.
- Oracle F32 reference vs Mojo BF16-storage path: cos 0.99908 / max_abs 0.24 across a 40-layer chain
  is consistent with bf16 matmul accumulation over 40×(qkv·proj·up·down), not a hidden bug. The
  canonical Python is itself a mixed bf16/f32 path; oracle uses bf16 round-trips at every matmul to
  mirror the Mojo dtype flow, so the oracle is faithful to canonical *math*. Parity vs the flame
  BF16-flash production SDPA kernel remains UNMEASURED (inherited from CHUNK A note).
- Weight provenance: the Mojo gate loads ALL weights (adapter, 40 layers, heads, bands) from the
  REAL 30.6 GB `magihuman_distill_bf16.safetensors`. The Python fixture supplies ONLY inputs
  (xv/xa/xt/coords) + `expected`. Same checkpoint feeds both sides → not a self-shaped oracle on
  weights; only the forward composition was transcribed, and that composition I re-derived above.

## Tags
- BLOCKER: none.
- FRAGILE: (a) multi-axis (t/w) rope coords UNTESTED — only h-axis swept; per-axis code is uniform so
  low-risk, but real coords unmeasured. (b) large-S tiled SDPA path not exercised at L=128 (inherited,
  proven elsewhere). (c) oracle f32-ref ≠ flame bf16-flash production SDPA (inherited from CHUNK A).
- STYLE: none material.

{component:"magihuman_full_forward", reRanGate:true, fullForwardCos:0.9990820196469521, oracleIsCanonical:"transcription-faithful", layerMembershipCorrect:true, mmBlockCorrect:true, largeSExercised:false, blockers:[], verdict:"clean — oracle is a hand-transcription (kandinsky5 risk) but I re-derived layer membership, MM rms/linear chunk-axis-0, gelu7/swiglu7 selector+formulas, adapter+Fourier rope, final heads, and the pairwise-concat workaround against BOTH magihuman_dit.rs AND canonical dit_module.py/config.py; all match. Gate re-ran by me: cos 0.99908, no OOM. CHUNK A still 0.999964. Open surfaces (multi-axis coords, large-S tiled SDPA, bf16-flash prod SDPA) are honestly UNMEASURED, not falsely claimed."}
