# SKEPTIC FINDINGS — krea2 chunk 4: `krea2_single_stream_block` (AdaLN composition)

Date: 2026-06-24
Reviewer: fresh-eyes skeptic (did NOT write this code)
Scope: `serenitymojo/models/dit/krea2_dit.mojo` lines 698–771 (CHUNK 4) + its parity
harness (`gen_krea2_block.py`, `krea2_block_parity_probe.mojo`).
Reference: `/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src/mmdit.py`
`SingleStreamBlock.forward` (328–337), `DoubleSharedModulation` (122–133),
`RMSNorm` (163–177).

## Compile / parity honesty (re-run by me)

```
cd /home/alex/mojodiffusion && pixi run mojo run -I . \
  serenitymojo/models/dit/parity/krea2_block_parity_probe.mojo
→ EXIT 0
→ krea2_single_stream_block parity:
    ParityResult(cos=0.9999939499442451, max_abs=0.03125, n=196608, PASS)
```

- `cos = 0.99999395` — high; matches the predicted ≈0.99999.
- `max_abs = 0.03125` = exactly `2^-5`, ONE bf16 ULP at magnitude ~1.0. The block
  output `y` has values ~O(1); two gated residual branches each end in a bf16
  cast (`residual_gate` stores bf16), plus two bf16 matmul chains (attn + SwiGLU)
  and an SDPA. A single-ULP worst-case element over 196 608 elements at cos
  0.99999 is **consistent with pure bf16 roundoff, not a structural error.** The
  max_abs does not hide a real bug.
- The test is NOT degenerate: gen uses full-magnitude random `vec` (randn) and
  random `mod.lin` (randn·0.1), so all 6 modulation chunks are distinct nonzero
  vectors and the gates pregate/postgate are ~N(0,1) — the residual branches
  contribute materially, so a swapped chunk / swapped norm / wrong residual base
  WOULD drop cos well below 0.999. It passing at 0.99999 actively exercises and
  clears those failure modes.

## Hunt results (line-by-line vs reference)

### 1. Forward structure (mmdit.py 331–335) — CLEAN
`krea2_dit.mojo:758-771`:
- 759-760: `xn = prenorm(x); xm = (1+prescale)*xn + preshift` — prenorm feeds the
  ATTN branch. ✓ (ref 333)
- 761-763: `a = attn(xm, …)`. ✓
- 764: `x1 = residual_gate(x, pregate, a)` = `x + pregate·a`. Residual adds onto
  the PRE-modulation `x` (arg is `x`, not `xn`/`xm`). ✓ (ref: `x = x + pregate*attn(...)`)
- 767-768: `xn2 = postnorm(x1); xm2 = (1+postscale)*xn2 + postshift` — postnorm
  feeds the MLP branch (NOT swapped with prenorm). ✓ (ref 335)
- 769: `m = mlp(xm2)` (SwiGLU). ✓
- 770: `x2 = residual_gate(x1, postgate, m)` = `x1 + postgate·m`. Second line's
  base is `x1` = OUTPUT of the first line, NOT the original `x` — genuinely
  sequential. ✓ (ref: the two `x = x + …` lines chain).

### 2. The two distinct +1 reparam sites — CLEAN, no double/missing +1
- RMSNorm weight = `scale + 1.0` lives inside `krea2_rmsnorm`'s kernel
  (`krea2_dit.mojo:367`, F32), the chunk-2 site. It adds 1 to the WEIGHT, applied
  once per norm. ✓ (ref 175: `weight=self.scale.float()+1.0`).
- AdaLN `(1+prescale)` lives inside `ops/elementwise.modulate` (`elementwise.mojo:56`
  `(1.0 + sv)*xv + shv`), applied once per modulate call.
- `krea2_double_shared_modulation` returns RAW chunks (no +1 — `tensor_algebra.mojo`
  `add(vec, lin)` then `slice`, no constant added). The block passes those raw
  chunks straight into `modulate`, which supplies the single +1. So scale gets
  exactly ONE +1, and the norm-weight +1 is a SEPARATE quantity on a separate
  tensor. No double-+1, no missing-+1. ✓

### 3. Broadcast over the L token axis at b==1 — CLEAN
- `_reshape_chunk_to_vec` reshapes each chunk `[1, features]`→`[features]`
  (`krea2_dit.mojo:719-726`). `reshape` is a metadata/clone reshape preserving
  row-major bytes; `[1,6144]`→`[6144]` is contiguity-safe. ✓
- In `modulate`/`residual_gate` for x `[1, L, features]`: `rows = 1·L = 32`,
  param is `[D]` so `nvec=1`, `rows_per_vec = rows//nvec = 32`. Kernel computes
  `vo = (r // 32)*cols = 0` for every `r ∈ [0,32)`, so `out[r,c]` reads
  `param[c]` for all token rows — true per-channel broadcast over L.
  (`elementwise.mojo:52-56`, `:327-331`). ✓
- pregate/postgate go through `residual_gate` (`x + g·y`, g=[6144] broadcasting)
  — confirmed at `:764`, `:770`. ✓

### 4. 6-tuple order (prescale,preshift,pregate,postscale,postshift,postgate) — CLEAN
`krea2_double_shared_modulation` slices c0..c5 in increasing offset order
(`krea2_dit.mojo:523-529`) and returns them in that order; the block binds
`mods[0]=prescale … mods[5]=postgate` (`:751-756`) — identical to the reference
`out.chunk(6, dim=-1)` unpacking order (mmdit.py 130-133). With random distinct
chunks the 0.99999 cos confirms no silent swap. ✓

### 5. Mask / RoPE — CLEAN (and in scope only at the call seam)
- Block runs `mask=None` via chunk-3's `sdpa_nomask` (b==1). ✓
- `cos/sin` are built once from `pos` via `build_krea2_rope` in the probe and
  threaded unchanged into the block→attention. comptime dims `[L=32, HEADS=48,
  KVHEADS=12, HEADDIM=128]` match the gen. ✓
- The `_padlen`/256-pad lives in `SingleStreamDiT.forward`, NOT in
  `SingleStreamBlock.forward`; correctly OUT of chunk-4 scope (the block oracle
  feeds raw L=32). ✓

### 6. Mojo correctness — CLEAN
- `mods` is a 6-tuple from a `raises def`; element access `mods[i]` then
  `reshape(... ^)`; final `return x2^`. No `var ref`, no `fn`-value misuse,
  raises propagate, `comptime features` (not `alias`). ✓
- Loader dtype trap CHECKED and is a FALSE alarm: gen saves `mod_lin` as F32 on
  disk (`gen_krea2_block.py:86`) but the probe loads it with `from_view_as_bf16`
  (`krea2_block_parity_probe.mojo:50`). `Tensor.from_view_as_bf16`
  (`tensor.mojo:246-249`) CASTS F32→bf16 on the host (does NOT byte-reinterpret),
  which exactly reproduces the reference's bf16 `mod.lin` (the gen's `.float()`
  round-trips losslessly for bf16-origin values). Correct and intentional
  (documented at probe :64-68). ✓

## Numerics-vs-reference notes (NOT bugs, recorded for completeness)
- `modulate` computes `(1+scale)*x+shift` in F32 then casts bf16; the reference
  does it in bf16 (`(1+prescale)*prenorm(x)+preshift`, all bf16). Mojo is *more*
  precise. Likewise `residual_gate` accumulates `x+g·y` in F32. Both are the
  expected "F32-internal, bf16-store" pattern and are inside the 0.99999 cos.
- The +1 ordering differs by dtype (`1.0_f32 + scale_f32` vs `bf16(1+bf16scale)`)
  — sub-ULP, absorbed by the bar.

## Verdict

**BLOCKERS: 0 / clean.** Chunk 4 (`krea2_single_stream_block`) faithfully
reproduces `SingleStreamBlock.forward`: correct residual bases (pre-mod x, then
x1), correct norm→branch assignment, correct 6-chunk order, single well-placed
+1, genuine per-channel broadcast over L, and a non-degenerate gate that passes
cos=0.99999 / max_abs=2^-5 (one bf16 ULP) — roundoff, not a hidden error.
