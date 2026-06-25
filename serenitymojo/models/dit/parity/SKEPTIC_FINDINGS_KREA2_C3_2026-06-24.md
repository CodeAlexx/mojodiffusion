# Skeptic findings — krea2 chunk 3: `krea2_attention` (GQA + QKNorm + RoPE + σ-gate)

Reviewer: fresh-eyes adversarial pass (did NOT write the port). Assumed the port lies.
Date: 2026-06-24.

Files reviewed line-by-line:
- `serenitymojo/models/dit/krea2_dit.mojo` — `krea2_attention[L,HEADS,KVHEADS,HEADDIM]` (616-695),
  `_tile_rope_table_bshd` / `_tile_rope_table` (546-613).
- Dependencies: `ops/gqa_backward.repeat_kv_f32` (44), `ops/rope.rope_interleaved` (373),
  `ops/attention.sdpa_nomask`→`_sdpa_math` (1690, 466, 332), `ops/activations.sigmoid` (167),
  `ops/tensor_algebra.mul` (446), `ops/linear.linear`.
- Reference: ai-toolkit `krea2/src/mmdit.py` — `Attention` (197-228), `QKNorm` (153-160),
  `attention()` (51-63), `ropeapply` (42-48), `RMSNorm` (163-177).
- Oracle generator `gen_krea2_attention.py` + probe `krea2_attention_parity_probe.mojo`.

Probe re-run by me (real exit/numbers, not trusted from a doc):
```
rope cos vs oracle:  cos=0.99999999  max_abs=4.2e-07  PASS
rope sin vs oracle:  cos=0.99999999  max_abs=3.0e-07  PASS
krea2_attention parity: cos=0.9999951093589294  max_abs=0.0009765625  n=196608  PASS
EXIT=0
```

---

## #1 GQA mapping — the #1 risk. **CORRECT (interleave). Test is discriminating.**

The reference attends with `F.scaled_dot_product_attention(..., enable_gqa=True)`
(mmdit.py:60-61, `gqa=self.heads!=self.kvheads`). The Mojo path applies RoPE to the
12-kv-head `k` then `repeat_kv_f32(k_rot, L, kvheads, n_rep, headdim, ctx)` (krea2_dit.mojo:679),
whose kernel maps dst head `head` → src kv head `head // n_rep` (gqa_backward.mojo:44).

I independently verified, NOT trusting the docstring:
- `enable_gqa=True` == `repeat_interleave(k, n_rep, dim=1)` (dst head h → kv head `h//n_rep`),
  max_abs **0.0**; and != `k.repeat(1,n_rep,1,1)` tile (dst head h → kv head `h%KVHEADS`),
  max_abs **3.06**. Explicit per-head table confirmed `h//4` is the interleave order.
- Discrimination on the ACTUAL oracle tensors (full attention recomputed in torch):
  `enable_gqa` cos **0.99999998**, `interleave` cos **0.99999998**, `tile` cos **0.13091621**
  (max_abs 0.296). So a tile-vs-interleave regression would crater cos to 0.13 — the
  0.999 gate WOULD catch it. `_repeat_kv_fwd_kernel` uses `head//n_rep` = interleave. ✅

Severity: clean. (The single highest-risk item is correct and the gate is not blind to it.)

## #2 Per-head RoPE tiling for BSHD — **CORRECT, no transpose.**

`rope_interleaved` flattens BSHD `[1,L,H,Dh]` leading dims row-major to rows = `l*H+h`
(rope.mojo:383-385 → row index `((0*L+l)*H+h)`), reading cos/sin row == data row.
`_tile_rope_table_bshd` (krea2_dit.mojo:553-561) writes `out_t[l*H+h, col] = table[l, col]`.
I re-derived the index decomposition for output element `(R=l*H+h, c)`:
`idx=(l*H+h)*half+c` ⇒ `col=c`, `rest=l*H+h`, `h=rest%H` (h<H ✓), `l=rest//H` ✓ ⇒ writes
`table[l]` to every head of token l. This is the `(l*H+h)` order rope_interleaved needs,
NOT a wrong `(h*L+l)` order. q tiles to 48 heads, k to 12 (lines 671-674) — both from the
SAME per-token table[l], matching the reference's `freqs[:,None,...]` broadcast over the head
axis. End-to-end the built table matches the oracle cos/sin at cos≈1.0 and the full op passes
with real multi-token freqs (incl. global pos 2000+, the F64-reduction stress case). ✅

Severity: clean.

## #3 σ-gate order & target — **CORRECT.**

Reference (mmdit.py:215, 226): `gate = self.gate(qkv)`; `out = self.wo(attention(...) * F.sigmoid(gate))`.
Mojo (krea2_dit.mojo:651, 692-694): `gate = linear(x, gate_w)`; `g = sigmoid(gate)`;
`gated = mul(merged, g)`; `linear(gated, wo)`. So (a) gate IS a projection of the INPUT x,
not of the attention output, and (b) `sigmoid(gate)` multiplies the MERGED `[1,L,6144]`
attention output BEFORE `wo`. Both requirements met. ✅

Minor dtype note (STYLE, not a defect): reference does the `attn * sigmoid(gate)` product in
bf16; Mojo `sigmoid`/`mul` do F32-internal then bf16 store. This is the standard serenity
bf16 boundary, well within the 0.999 bar (op passes at cos 0.99999). No action.

## #4 QKNorm — **CORRECT order and target.**

Reference order (mmdit.py:215-226): wqkv → rearrange to heads → `qknorm(q,k,v)` → `ropeapply(q,k)`
→ SDPA. `QKNorm(self.headdim)` (208) so RMSNorm is over headdim=128; `QKNorm.forward` norms q,k
and returns v UNTOUCHED (160). Mojo (krea2_dit.mojo:648-666): linear → reshape to
`[1,L,H,128]` → `krea2_rmsnorm(q)`, `krea2_rmsnorm(k)` (v not normed) → rope → repeat_kv → sdpa.
`krea2_rmsnorm` normalizes over the last dim (=128). Layout differs (BSHD vs ref BHLD) but
RMSNorm is per-row over the last 128 channels, so immaterial. The `scale+1` F32 reparam was
gated in chunk 2 (0 blockers). v is reshaped (660-662) but never normed/roped — straight to
`repeat_kv_f32(v,...)` (680). ✅

## #5 SDPA scale — **CORRECT (1/sqrt(128), once).**

Reference passes `scale=None` → `F.sdpa` default `1/sqrt(headdim)=1/sqrt(128)`. Mojo
`scale = 1.0/sqrt(Float32(headdim))` (krea2_dit.mojo:683) → `sdpa_nomask` → `_sdpa_math(...,
apply_mask=False)` → `_scale_f32` multiplies QKᵀ scores by `scale` exactly once
(attention.mojo:401-402, 107-119). Not 1/sqrt(6144), not unscaled, not double-applied. ✅

## #6 BSHD vs ref BHLD merge — **CORRECT head order.**

Reference final `rearrange("B H L D -> B L (H D)")` ⇒ channel `c = h*Dh + d` (head high-order).
Mojo SDPA returns BSHD `[1,L,H,Dh]`; `reshape` to `[1,L,features]` (krea2_dit.mojo:689-691)
collapses last two dims row-major ⇒ `c = h*Dh + d`. I verified the head channel-split is
identical between `rearrange("B L (H D)->B H L D")` and `x.reshape(1,L,H,D)`: head h owns
contiguous channels `[h*D, h*D+D)` in both. The wq/wk/wv reshapes (652-662) and the wo merge
all use the same `h*Dh+d` convention, consistent with the reference. SDPA is per-head so the
intermediate BSHD-vs-BHLD axis order is numerically immaterial. ✅

## #7 Mojo correctness — **clean.**

- comptime params bound to `[L=32, HEADS=48, KVHEADS=12, HEADDIM=128]` in the probe (37-40);
  `features=HEADS*HEADDIM=6144`, `half=64`, `n_rep=HEADS//KVHEADS=4` all derived correctly
  (640-645).
- No `List[ArcPointer[Tensor]]` / `var ref` / opaque-handle hazards — this is a plain functional
  op over by-value `Tensor`s. `^` transfers on the shape `List[Int]`s are correct (656/659/662/691).
- `repeat_kv_f32(k_rot, L, kvheads, n_rep, headdim, ctx)` matches sig `(x, s, h_kv, n_rep, dh, ctx)`.
- `raises` propagates; no io/ffi gotchas; compiled + ran (EXIT=0).
- One pre-existing deprecation warning (`fn _cast_buf_to_f32` in ops/attention.mojo:260) is in a
  dependency, not this chunk — out of scope, cosmetic.

---

## Carried-in flags from earlier chunks (checked, none apply to chunk 3)

- C1 F2 (F32 `log` wasting the F64 rationale): RESOLVED here — `build_krea2_rope` computes
  `log(Float64(theta))` (krea2_dit.mojo:246), so the exponent path is F64. rope tables match the
  oracle at cos≈1.0.
- C2 SimpleModulation b>1 broadcast (FRAGILE): not used by attention — no SimpleModulation in
  chunk 3. N/A.

## Independent torch cross-checks I ran (evidence, not assumption)

1. `enable_gqa` ≡ `repeat_interleave` (h//n_rep), ≠ tile (h%kv): max_abs 0.0 vs 3.06.
2. Full-attention recompute on the oracle's own tensors: interleave cos 0.99999998 vs tile cos
   0.131 — the gate discriminates the #1 risk.
3. `rearrange("B L (H D)->B H L D")` channel split == `reshape(1,L,H,D)`: head h owns `[h*D,h*D+D)`.

---

## Verdict

**BLOCKERS: 0 / clean.** All seven hunt targets verified faithful to mmdit.py at the tested
operating point (b=1, bf16 weights, production width 6144 / heads 48 / kvheads 12 / Dh 128).
The two highest-risk items — GQA mapping (#1, interleave `h//n_rep`) and per-head RoPE tiling
(#2, `l*H+h` row order) — are correct AND I confirmed the 0.999 gate is discriminating (a tile
GQA regression would drop cos to 0.13). The probe genuinely passes at cos 0.9999951. No fixes
required. (Report-only; no changes made.)
