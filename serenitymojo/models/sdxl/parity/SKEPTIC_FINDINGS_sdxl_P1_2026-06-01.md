# SKEPTIC FINDINGS — SDXL training-port Phase 1 (ResBlock + conv2d-bwd)
Date: 2026-06-01 · Skeptic pass on builder's "conv2d-bwd PASS, ResBlock 15/15 cos≥0.99999" claim.
Verdict header: **builder's Phase-1 claims hold. No REFUTED items. Two UNVERIFIED gaps — both already on the plan's risk register, neither blocks the parity-gated ResBlock unit, both block the full UNet.**

All numbers below are reproduced by me on this 3090, not taken from the builder's report.

---

## ATTACK 1 — conv2d-bwd parity reproduction + stride coverage

**Reproduced (stride-1):** ran `conv2d_bwd_oracle.py` → rebuilt `conv2d_bwd_parity.mojo` → ran.
```
conv2d d_x vs torch: cos=0.99999999999999  max_abs=4.47e-08  n=384  PASS
conv2d d_w vs torch: cos=0.99999999999996  max_abs=5.96e-08  n=108  PASS
conv2d d_b vs torch: cos=0.99999999999999  max_abs=2.86e-06  n=4   PASS
```
**Oracle independence — CLEAN.** `conv2d_bwd_oracle.py:99-111` computes grads via `F.conv2d` + `y.backward(gy)` (torch autograd), NOT a hand-rolled adjoint tuned to the Mojo kernel. Data is built in Mojo layout, permuted to NCHW for torch, grads permuted back. Independent ground truth. cos AND max_abs both at F32-floor → not a direction-only pass.

**Stride coverage — UNVERIFIED (the real finding here).** The oracle/gate test **stride-1 ONLY** (`conv2d_bwd_parity.mojo:129` `SH=SW=1`, `conv2d_bwd_oracle.py:46`). SDXL downsample (`.op`, `TRAINING_PLAN_sdxl.md:37,81-83`) is a **stride-2 Conv3×3**. The kernel's stride handling looks correct by inspection — the d_x kernel gates on `num_h % sh != 0` and `num_w % sw != 0` divisibility (`conv2d_backward.mojo:97,106`), and d_w uses `oh*sh-ph+kh` (`:148`) — i.e. general stride is *implemented*. But per Tenet 4 (measurement beats assertion) an un-gated code path does not count as verified. Plan acknowledges this explicitly (`TRAINING_PLAN_sdxl.md:82-83,150`).

→ **VERIFIED-CLEAN** for the Phase-1 stride-1 claim. **UNVERIFIED** for stride-2 (Phase-3 gate, already planned). **Impact: MED** (no stride-2 conv exists in the parity-gated ResBlock; it only appears in Downsample, which is Phase-3 assembly).

---

## ATTACK 2 — ResBlock oracle independence + eps (the #1 lie vector)

**Reproduced:** ran `resblock_oracle.py` → rebuilt `resblock_parity.mojo` → ran. Forward + all 15 grads PASS:
```
out       cos=0.99999999999909  max_abs=2.96e-05   n=16384
d_x       cos=0.99999999999817  max_abs=2.23e-05   n=8192
d_emb_in  cos=0.99999999999185  max_abs=9.82e-07   n=512
d_gn1_w   cos=0.99999999999616  d_gn1_b cos=0.99999999999648
d_conv1_w cos=0.99999999999868  d_conv1_b cos=0.99999999999669
d_emb_w   cos=0.99999999999684  d_emb_b cos=0.99999999999669
d_gn2_w   cos=0.99999999999771  d_gn2_b cos=0.99999999999474
d_conv2_w cos=0.99999999999973  d_conv2_b cos=0.99999999999997
d_skip_w  cos=0.99999999999988  d_skip_b cos=0.99999999999997
```
max_abs is tiny on every arm → magnitude is right, not just direction.

**Oracle independence — CLEAN.** `resblock_oracle.py:103-119` builds the ResBlock from torch primitives (`F.group_norm`, `F.silu`, `F.conv2d`, `F.linear`) and gets the backward from a single `out.backward(go)` autograd call. It is NOT transcribed from `block.mojo`'s hand-chained backward — the Mojo backward manually composes `conv2d_backward → silu_backward → group_norm_backward → linear_backward` (`block.mojo:216-268`), an entirely different code path from torch autograd. A bug in the Mojo hand-chaining would diverge from torch. It does not. This is a genuine independent oracle.

**GroupNorm eps split — VERIFIED against the real model, not just self-consistency.** Cross-checked the Rust reference `inference-flame/src/models/sdxl_unet.rs`:
- ResBlock GN: `group_norm_nchw(..., 1e-5)` at `:597` and `:615` — matches `GN_EPS_RES = 1e-5` (`config.mojo:30`), used in `block.mojo:156,205`, and the oracle `EPS = 1e-5` (`resblock_oracle.py:46`). ✅
- SpatialTransformer GN: `1e-6` at `:772` — matches `GN_EPS_ST = 1e-6` (`config.mojo:31`). ✅ (not exercised in Phase 1; flagged for the Phase-2 ST port.)

The eps is not a self-consistent tautology — it is the same value the real SDXL model uses. A wrong eps would still pass this self-consistent gate but fail the real checkpoint; that trap is closed.

→ **VERIFIED-CLEAN. Impact: n/a (clean).**

---

## ATTACK 3 — OIHW→RSCF weight remap on the real checkpoint

**Reproduced:** rebuilt + ran `weights_load_smoke.mojo` against real `sdxl_unet_bf16.safetensors` (5.1 GB, on disk).
```
input_blocks.4.0 (320->640, skip): conv1_w [3,3,320,640]  skip_w [1,1,320,640]  conv2_w [3,3,640,640]  has_skip True
input_blocks.1.0 (320->320, no skip): conv1_w [3,3,320,320]  conv2_w [3,3,320,320]  has_skip False
```
**Remap is a correct axis permutation, not a scrambling reshape — VERIFIED.** `weights.mojo:66-72` reads OIHW with `src = ((o*cin+ci)*kh+r)*kw+s` and writes RSCF with `dst = ((r*kw+s)*cin+ci)*cout+o` — a true index permutation moving each element to its transposed slot. Cross-checked against the Rust loader: `sdxl_unet.rs:547` uses `weight_ocickhkw_to_khwkicoc` (Oc,Ic,Kh,Kw → Kh,Kw,Ic,Oc = RSCF), `:535` confirms "OIHW->HWIO permutation done once at load time". HWIO ≡ RSCF [Kh,Kw,Cin,Cout]. The Mojo `ops/conv.mojo` and `conv2d_backward.mojo` both expect RSCF (`conv2d_backward.mojo:11-15`). Layouts are consistent end to end.

**Caveat (LOW):** the smoke checks *shapes* only; an OIHW↔RSCF mismatch is shape-preserving, so shapes alone wouldn't catch a value scramble. BUT the *value* correctness of this exact remap is transitively gated: `resblock_oracle.py:85-86` builds conv weights in RSCF then permutes `(3,2,0,1)` to OIHW for torch, the inverse of the loader's remap, and the forward+backward match at cos≈1 (Attack 2). So the permutation's value-correctness is covered. Still, a dedicated value-level remap gate on real `input_blocks.4.0` would be cheap insurance.

→ **VERIFIED-CLEAN (shapes measured, value-correctness transitively gated). Impact: LOW residual.**

---

## ATTACK 4 — skip-path backward (1×1 conv, active at Cin≠Cout)

**VERIFIED-CLEAN.** The 15/15 ran at Cin=64≠Cout=128, so `has_skip=True` and the skip conv is live. `block.mojo:254-260`:
- `g_skip = conv2d_backward[...1,1...0,0](acts.x, w.skip_w, go)` — backprops the 1×1 skip conv, producing `d_skip_w`, `d_skip_b`, and `g_skip.d_x`.
- `d_x = add(d_x_main, g_skip.d_x)` — the skip-branch input grad is **added into** the total d_x, not dropped. Both branches reach d_x.

The oracle backprops the skip independently: `r = F.conv2d(x, skip, skip_b, ...)`, `out = r + c2`, single `out.backward` — torch accumulates the skip's contribution into `x.grad` automatically. Measured: `d_skip_w cos=0.99999999999988`, `d_skip_b cos=0.99999999999997`, `d_x cos=0.99999999999817` (which can only be right if BOTH the GN-branch and skip-branch contributions to d_x are present and correctly summed). A dropped skip-branch in d_x would tank d_x's cos; it doesn't.

→ **VERIFIED-CLEAN. Impact: n/a.**

---

## ATTACK 5 — temb path (SiLU(Linear(temb)) broadcast-add, spatial reduction in bwd)

**VERIFIED-CLEAN.** Forward: `e=SiLU(emb)`, `el=Linear(e)` → `[N,Cout]`, reshape to `[N,1,1,Cout]`, `h2 = add(c1, el4)` broadcasting over H,W (`block.mojo:166-169`). Backward: `_spatial_sum_to_nc` (`block.mojo:132-144`) sums `d_h2[n,h,w,co]` over the HW spatial positions to `d_el[n,co]` — the correct adjoint of the broadcast. Then `linear_backward` → `silu_backward` → `d_emb_in` (`:232-235`).

- `d_emb_in` is the grad w.r.t. the **pre-SiLU** time-embedding input (`[N,Eemb=256]`), matching the oracle's `emb.grad` where `emb` is the pre-SiLU tensor (`resblock_oracle.py:79,131`). Measured `d_emb_in cos=0.99999999999185`. ✅
- The spatial reduction is real, not a coincidence-at-8×8: the oracle does the same reduction independently via autograd over `el[:,:,None,None]` (`resblock_oracle.py:108`). A missing/incorrect HW sum on the Mojo side would diverge from torch's autograd reduction. `d_emb_w cos=0.99999999999684` (n=32768) and `d_emb_b cos=0.99999999999669` (n=128) both pass, which they could not if the spatial sum feeding the linear bwd were wrong.

→ **VERIFIED-CLEAN. Impact: n/a.**

---

## Ranked summary

| # | Attack | Verdict | Impact |
|---|--------|---------|--------|
| 1 | conv2d-bwd stride-1 parity + oracle independence | VERIFIED-CLEAN | — |
| 1b | conv2d-bwd **stride-2** (downsample `.op`) | **UNVERIFIED** (no gate) | **MED** |
| 2 | ResBlock oracle independence + eps 1e-5/1e-6 split | VERIFIED-CLEAN | — |
| 3 | OIHW→RSCF remap (shapes measured, value transitively gated) | VERIFIED-CLEAN | LOW |
| 4 | skip-path backward (d_skip_w/b + d_x accumulation) | VERIFIED-CLEAN | — |
| 5 | temb broadcast-add + spatial-reduction backward | VERIFIED-CLEAN | — |

No REFUTED findings. The builder's "15/15 cos≥0.99999" and "conv2d-bwd PASS" are **honest and reproduced**. The oracle does not lie: it is torch-autograd, structurally independent of `block.mojo`'s hand-chained backward, and the eps split is validated against the real model source.

---

## Prioritized fix list for the bugfixer (before Phase 2)

1. **[MED] Add a stride-2 conv2d-backward parity case** to `ops/parity/conv2d_bwd_parity.mojo` + oracle (e.g. N=2, Cin=4, Cout=8, Hi=Wi=8, K=3, **stride=2**, pad=1 → Ho=Wo=4). The kernel *looks* correct for general stride but is un-gated; SDXL Downsample is stride-2. Tenet 4: do not trust the path until a measurement names it. **Cheapest, highest-value next gate.** (Phase-3 per plan, but worth pulling forward since the kernel is already written.)

2. **[LOW] Add a value-level OIHW→RSCF remap gate** on a real `input_blocks.4.0` conv weight (load via Mojo, load the same tensor via torch+numpy in OIHW, permute to RSCF in numpy, compare element-wise). Closes the "shapes-only smoke" gap directly rather than relying on transitive coverage.

3. **[INFO, no action] Confirm `GN_EPS_ST = 1e-6` is wired into the actual SpatialTransformer GN call when that block is ported** (Phase 2). The constant is correct; verify it is *used*, not just defined.

## Phase-2 dependency (shared with the Anima track)
- **Rectangular SDPA backward (Sq≠Skv)** is required before the SpatialTransformer cross-attention can train: cross-attn has Sq=H·W ≠ Skv=77. Existing `sdpa_backward` is square-only (`TRAINING_PLAN_sdxl.md:148`). This is the same primitive gap the Anima cross-attention track needs — build once in `ops/attention_backward.mojo` (Tenet 1), gate with `ops/parity/sdpa_rect_bwd_*`, and both tracks inherit it. **This is the gating item for Phase 2, not anything in the ResBlock.**
