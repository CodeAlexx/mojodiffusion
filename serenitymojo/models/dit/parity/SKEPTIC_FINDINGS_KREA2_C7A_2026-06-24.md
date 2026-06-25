# Skeptic findings — krea2 chunk 7a (`krea2_forward` = SingleStreamDiT.forward WIRING)

Date: 2026-06-24. Reviewer: adversarial skeptic (fresh eyes). Assumed the port lies.
Scope: `serenitymojo/models/dit/krea2_dit.mojo` `krea2_forward[LFULL,LPAD,LT,NBLOCKS]`
(lines 1200-1361) + the NEW masked branches in `krea2_attention` (line 698-701) /
`krea2_single_stream_block` (mask Optional param) + the pad/mask/rope helpers
(`_pad_seq_zeros`, `build_krea2_text_mask`, `build_krea2_rope`).
Reference: `/home/alex/ai-toolkit/.../krea2/src/mmdit.py` `SingleStreamDiT.forward`
(413-461), `_mask` (66-68), `attention` (51-63).

## Lead verdict on the masked main-block path (what 7b depends on)

**The masked main-block path is CORRECT.** Re-ran the live probe (4-block resident,
seeded random weights, ~5 GB, display-safe):

```
krea2_forward (4-block wiring) parity: ParityResult(cos=0.9996006656197588, max_abs=0.078125, n=2560, PASS)
krea2_forward WIRING PASS   EXIT=0
```

The pad-to-256 main blocks run through the additive `[1,HEADS,LPAD,LPAD]` bf16 mask
via `sdpa[...] -> _sdpa_math` (cuBLAS QKᵀ + F32 `_scale_mask` add + F32 softmax +
cuBLAS P@V). The end-to-end velocity matches the reference's natural-path output at
cos 0.99960 ≥ 0.999. This is the FIRST live exercise of the masked block path
(chunk-4's own probe gates `mask=None` only — see FRAGILE-1).

## Re-gated dependency probes (ran them myself)

GPU precheck before every run: `nvidia-smi memory.free = 23532 MiB`, no compute apps.

| chunk | probe | result | EXIT |
|-------|-------|--------|------|
| 3 attention | `krea2_attention_parity_probe.mojo` | cos=0.9999951093589294 (builder claimed 0.9999951) | 0 |
| 4 block (mask=None) | `krea2_block_parity_probe.mojo` | cos=0.9999939499442451 (builder claimed 0.9999939) | 0 |
| 7a forward | `krea2_forward_small_parity_probe.mojo` | cos=0.9996006656197588 | 0 |

Both chunk-3 and chunk-4 claims reproduce to the digit. The probes are genuinely
fail-loud (`raise Error` / non-zero exit on cos < 0.999; verified in source).

## Hunt-list results (file:line)

### 1. Pad-to-256 — CORRECT
- `krea2_dit.mojo:1288-1292` `combined = concat(1, ctx, ctx_proj, img_e)` then
  `_pad_seq_zeros(combined, LFULL, LPAD, FEATURES, ctx)`. `_pad_seq_zeros`
  (1188-1197) appends `zeros[LPAD-LFULL]` AFTER x via `concat(1, ctx, x, pad)` —
  matches `F.pad(combined, (0,0,0,_padlen))` (mmdit.py:437, pad at seq END).
- pos pad with ZEROS: `1310-1318` appends `(LPAD-LFULL)*3` zeros to the host pos
  list → padded positions get pos=0 → `omega*0=0` → cos=1/sin=0 → identity rope.
  Matches `F.pad(pos, (0,0,0,_padlen))` (mmdit.py:439).
- mask pad with FALSE: `1296-1298` `keep = ones[0:LFULL], zeros[LFULL:LPAD]` →
  `build_krea2_text_mask` outer-products to the additive mask. Matches
  `F.pad(mask, value=False)` then `_mask(mask)` (mmdit.py:438+441).
- LPAD is the multiple-of-256: probe sets `LPAD=256` for `LFULL=60`
  (`ceil(60/256)*256=256`). Correct. (LPAD is a comptime PARAM, not derived in the
  function — caller responsibility; the probe sets it right. See FRAGILE-2.)

### 2. The additive mask (the subtle part) — CORRECT
- `build_krea2_text_mask` kernel (918-933): `out_m[i,j] = keep[i]*keep[j]` as a
  **0/1 FLOAT** (`ki*kj`), bf16, shape `[1,H,LPAD,LPAD]`. NOT a -inf hard mask.
  Matches `_mask` (mmdit.py:66-68) = `keep.unsqueeze*keep.unsqueeze` outer product.
- It is ADDED to the scores before softmax: `krea2_attention:698-699` routes
  `mask` to `sdpa[1,L,HEADS,HEADDIM]` → `_sdpa_math(...,apply_mask=True)` →
  `_scale_mask` (attention.mojo:91-104) does `scores*scale + mask` then
  `_softmax_rows_f32`. This is exactly `F.sdpa(attn_mask=mask)` float-mask
  additive semantics (NOT cuDNN's float-mask kernel, which the chunk-6 gen notes
  deviates — the faithful target is the MATH/additive formula, which this is).
- pad keys get +0 vs real keys +1 → a SOFT down-weight (ratio exp(1)≈2.72), not a
  hard mask. Pad QUERY rows are all-0 (uniform softmax) but discarded by the final
  `slice(final,1,LT,imglen)` (real image tokens only). Verified the output region
  `[LT:LT+imglen]` is fully within the real (unpadded) span.

### 3. cat order + slice — CORRECT
- cat: `concat(1, ctx, ctx_proj, img_e)` (1289). `concat` (tensor_algebra.mojo:1442)
  lays operands out IN ORDER along dim (col_off advances per input), so combined =
  [context THEN img]. Matches `torch.cat((context, img), dim=1)` (mmdit.py:431).
- slice: `slice(final, 1, LT, imglen)` (1361) = `final[:, LT:LT+imglen, :]`.
  `slice` (tensor_algebra.mojo:1528) narrows dim to `[start, start+length)`.
  `LT == txtlen == context.shape[1]`. Matches `final[:, txtlen:txtlen+imglen]`
  (mmdit.py:459).

### 4. NEW masked branch didn't break no-mask path — CORRECT
- `krea2_attention:696-701`: `if mask: sdpa(...mask.value()...) else: sdpa_nomask(...)`.
  The `else` is the unchanged chunk-3 path. Chunk-3 probe re-ran clean
  (cos 0.9999951). Block-level no-mask path (chunk-4) re-ran clean (cos 0.9999939).
  `q_rot` (rotated, line 699) is correctly passed to the masked sdpa (not raw `q`).

### 5. tvec broadcast — CORRECT
- block modulation vec = `tproj(t)` reshaped `[1,6F]` (`blk_vec2`, line 1263); fed
  to `DoubleSharedModulation` add+chunk → 6 raw `[features]` vecs → `modulate`
  applies `(1+scale)*x+shift` per-channel over all L rows (AdaLN-over-tokens). The
  +1 reparam lives ONLY in `modulate` (elementwise.mojo:108); chunks are raw → no
  double-add.
- LastLayer tvec = `t3` (the **tmlp** output `[1,1,F]`, line 1252), NOT the tproj
  `blk_vec`. `krea2_last_layer(x, t3, ...)` (1349-1350). Matches `last(combined, t)`
  with `t = tmlp(...)` (mmdit.py:458). `t3` is reused for BOTH `tproj(t3)` and
  `last(..., t3)` — exactly the reference's `t`/`tvec` split.

### 6. Weight load (118 by name) — CORRECT, ZERO discrepancy
Verified against the oracle's safetensors header (no GPU load): the forward
requests **118 distinct `w.*` names** for `NBLOCKS=4,LT=20`; the oracle has
**118 `w.*` keys**; **0 missing, 0 unrequested**. No silent rename or miss. (Blocks
4..27 absent because the small oracle is LAYERS=4 — expected.) Names cross-checked:
`first`, `tmlp.{0,2}`, `tproj.1`, `txtfusion.{layerwise,refiner}_blocks.{0,1}.*`,
`txtfusion.projector`, `txtmlp.{0,1,3}`, `blocks.{0..3}.*`, `last.*` — all the
`.weight`/`.bias`/`.scale`/`.mod.lin` suffixes resolve.

### 7. Mojo correctness + fail-loud probe — CORRECT
- F32-internal RMSNorm (weight=scale+1), GQA `repeat_kv_f32` (dst head h reads kv
  head `h//n_rep` = torch repeat_kv / `enable_gqa` order), interleaved RoPE with F64
  range reduction, txtfusion projector via `transpose(1,2)` + `Linear(12→1)` — all
  verified against the reference and all gated by chunks 1-6 + 7a.
- Probe raises non-zero on cos<0.999. Confirmed.

## Non-blocking findings

- **FRAGILE-1** (test gap, not a 7a defect): `krea2_block_parity_probe.mojo:82`
  passes `Optional[Tensor](None)` — the chunk-4 gate covers ONLY the no-mask block
  path. The masked block path's FIRST and ONLY live test is chunk 7a (passed). The
  masked SDPA primitive itself IS independently gated by chunk-6 CASE-B
  (`krea2_txtfusion_parity_probe.mojo:100-111`, real 0/1 bf16 mask vs the math
  backend) — BUT see FRAGILE-3.

- **FRAGILE-2** (caller contract): `LPAD`/`LFULL`/`LT` are comptime PARAMS, not
  derived inside `krea2_forward`. A caller that passes an `LPAD` that is NOT
  `ceil(LFULL/256)*256`, or an `LT` ≠ real txtlen, will silently produce wrong
  output (wrong pad span / wrong output slice) with no guard. The probe sets them
  correctly; 7b must too. Consider a `comptime`/runtime assert
  `LPAD == ((LFULL+255)//256)*256 and LPAD>=LFULL` and `LT < LFULL`.

- **FRAGILE-3** (test infra): `krea2_txtfusion_oracle.safetensors` is MISSING — the
  chunk-6 probe (which holds the only standalone masked-sdpa-path gate, CASE-B)
  cannot be run (`EXIT=1`, "no known single-file safetensors"). Regenerate
  `gen_krea2_txtfusion.py` so the masked-path CASE-B is re-runnable. The masked
  path is still validated transitively by 7a, so this is not a 7a blocker.

- **STYLE**: `_attn_pv_kernel`/`_cast_buf_to_f32` use deprecated `fn` (compiler
  warning, harmless).

## One-line verdict

BLOCKERS (0) / clean — masked main-block forward path is correct (7a PASS cos
0.99960; chunks 3/4 re-gated; 118/118 weights load exact); only non-blocking test
gaps (missing txtfusion oracle, masked-block tested only e2e, no LPAD guard).
