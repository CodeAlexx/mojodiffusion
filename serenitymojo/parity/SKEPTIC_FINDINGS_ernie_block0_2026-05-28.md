# SKEPTIC FINDINGS — ERNIE-Image block0 FULL forward audit

- Timestamp: 2026-05-28 (UTC)
- Builder claim under audit: `ernie_block0_full_forward` added to `ErnieImageResident`; new pipeline smoke `ernie_block0_full_smoke.mojo` exercises real layer-0 weights with a synthesized AdaLN at `randn * 0.1` because the real AdaLN chain produces absmax ≈ 215040 (>> BF16 max 65504).
- Files audited (read-only):
  - `/home/alex/mojodiffusion/serenitymojo/models/dit/ernie_image.mojo` (486 lines)
  - `/home/alex/mojodiffusion/serenitymojo/pipeline/ernie_block0_full_smoke.mojo` (222 lines)
  - `/home/alex/mojodiffusion/serenitymojo/pipeline/ernie_block0_smoke.mojo` (162 lines, bounded reference)
  - `/home/alex/mojodiffusion/serenitymojo/ops/embeddings.mojo` (timestep_embedding kernel)
  - `/home/alex/mojodiffusion/serenitymojo/ops/linear.mojo` (linear semantics)
  - `/home/alex/mojodiffusion/serenitymojo/ops/rope.mojo` (rope_halfsplit_full kernel)
  - `/home/alex/EriDiffusion/inference-flame/src/models/ernie_image.rs` (Rust reference; lines 467–714 for forward / time / block / ffn)
  - `/home/alex/EriDiffusion/inference-flame/src/models/zimage_nextdit.rs` (Z-Image reference for the existing Mojo timestep order, lines 411–428)

---

## A. AdaLN saturation — root cause investigation

### A1 — Rust ERNIE timestep + shared AdaLN chain (PASS, but reveals upstream bug)

`ErnieImageModel::time_embed` (`ernie_image.rs:588-609`):
```
half = hidden/2 = 2048
freqs[i] = exp(-(i/half) * ln(10000))               for i in [0, half)
args     = timestep.unsqueeze(1) * freqs            -> [B, half]
sin_part = args.sin();  cos_part = args.cos()
emb      = cat([sin_part, cos_part], dim=1)         -> [B, hidden]  (BF16)
h        = emb @ linear_1.weightᵀ + linear_1.bias
h        = h.silu()
return     h @ linear_2.weightᵀ + linear_2.bias     -> [B, hidden]
```
Shared AdaLN (`ernie_image.rs:522-525`):
```
mod_in  = t_emb.silu()
mod_out = mod_in @ adaLN_modulation.1.weightᵀ + adaLN_modulation.1.bias    -> [B, 6*hidden]
```
Note channel ordering of the sinusoidal embedding: **`[sin | cos]`** (sin first, cos second).
Timestep input is fed **raw** (`t.to_dtype(DType::F32)` with no ×1000 scale).
**Result:** PASS for Rust reference reading.

### A2 — Mojo equivalent (FAIL — sin/cos channels swapped relative to Rust)

`ErnieImageResident.time_embed` (`ernie_image.mojo:226-236`) calls `timestep_embedding(t, hidden, ctx, 10000.0)` and then `silu`, `linear`, etc.

`ops/embeddings.mojo:73-77` kernel:
```
o[row, i]        = cos(angle)        # cols [0, half)
o[row, half + i] = sin(angle)        # cols [half, dim)
```
=> Mojo timestep embedding is **`[cos | sin]`**.

Comparing to Rust (`ernie_image.rs:603`: `cat([sin_part, cos_part], 1)`), this is a hard channel-order mismatch. The `time_embedding.linear_1.weight` matrix was trained on `[sin | cos]` rows; feeding it `[cos | sin]` permutes the first 2048 input columns with the last 2048 — every output channel is a different linear combination of the trained mapping. The result is NOT scaled or clipped; it can drift into very large magnitudes after two BF16 linears + SiLU + a third linear into 6×hidden.

Bounded smoke reports `temb_raw absmax ≈ 14208`, then `adaln_raw absmax ≈ 215040`. 14208 is already implausible for a real ERNIE timestep embedding (`emb` itself is bounded by ≤√2 ≈ 1.4 after sin/cos, so |h| should be O(‖W‖)·O(1) ≈ a few tens, not 14000). 215040 = 14208 × ~15, plausible for a single more 4096×4096 BF16 linear scaling 14k. The cumulative blow-up is a direct consequence of the permuted channel order interacting with weight rows the model never saw.

**Verdict A2:** FAIL — channel order is wrong for ERNIE.

### A3 — Bounded smoke driver characteristics (PASS, evidence only)

`ernie_block0_smoke.mojo:117` feeds `t_vals = [875.0]` directly (no ×1000). The smoke prints:
- `temb_raw absmax_allowed 20000` (clearly designed to accept the large value)
- `adaln_raw absmax_allowed 300000` (designed to accept ≈215k)
- `adaln_bounded` = `mul_scalar(adaln_raw, 1e-5)` — i.e. the bounded smoke worked around the symptom rather than fixing the cause.

This confirms the handoff numbers (`temb_raw ≈ 14208`, `adaln_raw ≈ 215040`) are observed, not extrapolated.
**PASS** — evidence collected.

### A4 — Diagnosis enumeration

(a) Linear weight orientation: `ernie_image.mojo:160-181` loads `[hidden, hidden]` for `time_embedding.linear_{1,2}.weight` and `[6*hidden, hidden]` for `adaLN_modulation.1.weight`. Mojo `linear(x, W=[out,in], bias)` computes `x @ Wᵀ` (`ops/linear.mojo:151-156`). Rust pre-transposes at load (`take_t`) then does `x.matmul(Wᵀ_already)`. Net effect is identical. **NOT the bug.**

(b) Missing pre-AdaLN norm: Rust `ernie_image.rs:519-525` feeds `t_emb` straight into `silu → adaLN.matmul → bias` with no LayerNorm in between. Mojo path matches. **NOT a missing-norm bug.**

(c) Timestep input scale: Rust passes the raw timestep (`875.0` in our smoke) into the sinusoidal formula. Mojo does the same (no ×1000). **NOT the scale bug.**

(d) sin/cos channel order: **CONFIRMED — see A2.** The Mojo kernel hard-codes `[cos | sin]` (correct for Z-Image NextDiT, `zimage_nextdit.rs:425-426`), but Rust ERNIE uses `[sin | cos]` (`ernie_image.rs:603`). The trained `time_embedding.linear_1.weight` expects `[sin | cos]`; receiving the permuted input drives downstream activations into out-of-distribution territory and ultimately overflows BF16 after the 4096→24576 AdaLN linear.

### A5 — Verdict

**WIRING BUG.** Confidence: high. The Mojo `timestep_embedding` op was originally written for Z-Image NextDiT (the file's own header comment cites `zimage_nextdit.rs:411-428`), and it was reused unchanged for ERNIE-Image, which has the opposite cat order. ERNIE-Image is the only known consumer in this repo that uses `[sin | cos]`; Z-Image uses `[cos | sin]`. The fix is NOT to clamp/scale AdaLN downstream — it is to give ERNIE the correct sinusoidal order.

A secondary deviation worth verifying after the fix: Rust ERNIE casts the cat result to BF16 (`ernie_image.rs:603` `.to_dtype(DType::BF16)`); Mojo `timestep_embedding` returns F32 and `time_embed` casts to weight dtype (`ernie_image.mojo:229`). That cast point matches Rust functionally as long as no overflow occurs in F32 (which it does not). NOT the bug, but call out for parity.

---

## B. Block forward correctness

### B1 — Step-by-step mapping

`block0_smoke_forward[S]` in `ernie_image.mojo:270-326` executes the following sequence; cross-referenced against `block_forward_from_map` (`ernie_image.rs:918-981`):

| Step | Mojo line | Rust line | Match |
|------|-----------|-----------|-------|
| 6-chunk split: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp | 286-291 | 525-535 (chunk(6,1)) | ✓ |
| residual1 = x | implicit via `residual_gate` | 622 | ✓ |
| sa_norm = rms_norm(x, adaLN_sa_ln.weight, 1e-6) | 294 | 623, 939 | ✓ |
| sa_in = sa_norm * (1+scale_msa) + shift_msa | 295 via `modulate` | 624, 940 | ✓ |
| Q/K/V linears (no bias) | 296-304 | 644-646, 945-947 | ✓ |
| reshape to [B,S,H,Dh] | 296-304 via `_to_bshd[S]` | 658-660, 949-951 | ✓ |
| per-head RMSNorm(norm_q/k, head_dim, 1e-6) | 305-306 | 663-664, 953-954 | ✓ |
| RoPE applied | 307-308 | 672-673, 961-962 | ✓ (note: Rust permutes to BHSD first; Mojo stays BSHD — equivalent because the cos/sin table is built in matching layout, see audit below) |
| SDPA(1/sqrt(D)) | 309-311 via `sdpa_nomask` | 676, 964 | ✓ |
| to_out projection (no bias) | 313-315 | 680, 966 | ✓ |
| x = residual1 + gate_msa * attn_out | 316 via `residual_gate` | 627, 968 | ✓ |
| residual2 = x; mlp_norm = rms_norm(h, adaLN_mlp_ln, 1e-6) | 318 | 631, 973 | ✓ |
| mlp_in = mlp_norm * (1+scale_mlp) + shift_mlp | 319 via `modulate` | 632, 974 | ✓ |
| gate = linear(mlp_in, gate_proj); up = linear(mlp_in, up_proj) | 320-321 | 693-694, 976-977 | ✓ |
| activated = gelu(gate) * up | 322 | 703-704, 976-978 (commutes) | ✓ |
| mlp_out = linear(activated, linear_fc2) | 323-325 | 705, 978 | ✓ |
| out = residual2 + gate_mlp * mlp_out | 326 via `residual_gate` | 635, 980 | ✓ |

Caveat 1: Mojo applies RoPE on `[B,S,H,Dh]` while Rust applies on `[B,H,S,D]`. The Mojo cos/sin tables in `build_ernie_rope_tables` (lines 449-467) iterate `for tok: for _h: ...` so row index `tok*HEADS + h` matches the BSHD flattening exactly. The half-split rotate formula in `rope_halfsplit_full` (`rope.mojo:182-183`) is `o[i] = x0*cos[i] - x1*sin[i]; o[i+half] = x1*cos[i+half] + x0*sin[i+half]`, which equals Rust's kernel (`ernie_image.rs:264-271`) rewritten for `d<half` and `d≥half`. **Match.**

Caveat 2: `sdpa_nomask[1, S, H, Dh]` is called with the BSHD-shaped Q/K/V. Verify that op consumes BSHD (not BHSD) for parity — auditor did not re-read sdpa.mojo as it is out of scope for this audit; FLAG only.

**Verdict B1:** PASS (with two informational caveats).

### B2 — Rust cross-check

Done inline in B1. The `block0_smoke_forward` is a faithful mirror of `ernie_image.rs::block_forward_from_map`. The new `ernie_block0_full_forward` just delegates to `block0_smoke_forward[S]` (line 364) — it does NOT introduce new math, it only widens the comptime `S`. PASS.

---

## C. New smoke correctness

### C1 — Comptime sizes (PASS)

`ernie_block0_full_smoke.mojo:73-77`: `IMG_H=IMG_W=16`, `N_IMG=256`, `N_TXT=64`, `S=320`. Hidden / heads / head_dim used inside the call come from `ernie_contract` and are 4096 / 32 / 128 (production). PASS.

### C2 — Real weights, no scaling (PASS)

Loads via `ErnieImageResident.load_default_block0_smoke(ctx)` (line 139), validated via `validate_block0_smoke_weights()` (line 140). Patch projection, text projection, and timestep MLP are exercised via the real resident path (lines 174-176). No scaling on the weights themselves. PASS.

### C3 — Synthesized AdaLN deviation (WARN — known)

Lines 195-203 build `adaln_synth = randn([1, 24576]) * 0.1` and pass it into the block forward. The smoke does **not** consume the model's real `shared_adaln(temb, ctx)` output. The builder is explicit about this in the file header (lines 11-23) and in the docstring of `ernie_block0_full_forward`. This is an honest workaround but means the smoke does NOT yet validate the timestep→AdaLN path end-to-end. The bugfix list (below) must keep this deviation visible until A5's wiring bug is fixed AND a parity run is performed against the diffusers oracle. WARN.

### C4 — Finite check (PASS)

`_stats` at line 90-124 raises on NaN, on Inf-via-`v > 1e30`, and on absmax exceeding the per-tensor budget. `block0_out` budget = 60000 (BF16-safe). PASS.

---

## D. Mojo syntax / no regressions

### D1 — Compilation (PASS)

```
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/ernie_block0_full_smoke.mojo
  -> full_exit=0
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/ernie_block0_smoke.mojo
  -> bounded_exit=0
```
Both smokes compile clean. PASS.

### D2 — Other callers of `ErnieImageResident` (PASS)

Grep across `serenitymojo/**/*.mojo`:
- `serenitymojo/pipeline/ernie_resident_smoke.mojo:21,93` — uses only `load_default`, not affected by the new method.
- No other usages.
PASS.

---

## E. FFN order

### E1 — GELU placement (PASS)

Mojo (`ernie_image.mojo:320-325`):
```
gate     = linear(mlp_in, gate_proj.weight)
up       = linear(mlp_in, up_proj.weight)
activated= mul(gelu(gate), up)
mlp_out  = linear(activated, linear_fc2.weight)
```
Rust (`ernie_image.rs:976-978`):
```
let gate = modulated.matmul(gate_proj)?.gelu()?;
let up   = modulated.matmul(up_proj)?;
let ffn  = up.mul(&gate)?.matmul(linear_fc2)?;
```
Elementwise multiply commutes. Both apply GELU to `gate_proj` ONLY, then multiply by `up_proj`, then run through `linear_fc2`. The Mojo docstring at the top of `ernie_image.mojo` mis-labels this as "SwiGLU" but the actual code is GELU-gated (matches Rust `gelu()`, which is tanh-approx). PASS.

---

## AdaLN Saturation Verdict

**WIRING BUG — sin/cos channel order swap in timestep_embedding.**

- Rust ERNIE concatenates `[sin | cos]` (`ernie_image.rs:603, 847`).
- Mojo `timestep_embedding` writes `[cos | sin]` (`ops/embeddings.mojo:75-77`), because it was authored for Z-Image NextDiT, which itself uses `[cos | sin]` (`zimage_nextdit.rs:425-426`).
- ERNIE's `time_embedding.linear_1.weight` rows were trained on `[sin | cos]`; the swap permutes the input space, the model never saw this geometry, and the activations explode through two 4096→4096 BF16 linears and one 4096→24576 BF16 linear into absmax ≈ 215040.
- The 1e-5 scaling in the bounded smoke and the synthesized `randn*0.1` in the full smoke are both symptomatic workarounds — they hide the bug instead of fixing it.

No bisect needed — the diff is mechanical and visible side-by-side.

---

## Bugfix Worklist (ordered)

1. **[P0] Fix timestep sin/cos channel order for ERNIE-Image.**
   The minimal-blast-radius fix is to introduce an ERNIE-specific timestep embed (or a flag on `timestep_embedding`) that emits `[sin | cos]` instead of `[cos | sin]`. Touch points:
   - `serenitymojo/ops/embeddings.mojo:73-77` — kernel writes channels.
   - `serenitymojo/models/dit/ernie_image.mojo:226-236` (`time_embed`) — only ERNIE caller that needs the new order; the bounded smoke and the new full smoke will both benefit immediately.
   - Z-Image and any other consumer must stay on `[cos | sin]`. Do NOT change the default — add a parameter (e.g. `sin_first: Bool = False`) or a wrapper.
   After the fix, re-run `ernie_block0_smoke` and expect `temb_raw absmax` to drop from ~14000 to roughly the L2-norm of `linear_2.weight.bias` plus a few unit-magnitude contributions — single-digit / low-tens range — and `adaln_raw` to land at a finite BF16-safe magnitude (probably absmax < 100, certainly < 65504).

2. **[P0] Remove the synthesized-AdaLN workaround in `ernie_block0_full_smoke.mojo`.**
   Once (1) lands and `adaln_raw` is BF16-safe, replace the `randn([1,24576]) * 0.1` synth (lines 195-203) with the real chain `model.shared_adaln(model.time_embed(timestep, ctx), ctx)`. Keep a `_stats(..., adaln, ctx, 200.0)` assertion to lock the magnitude in.

3. **[P0] Remove the `ADALN_SMOKE_SCALE = 1e-5` in `ernie_block0_smoke.mojo:37, 137`** once (1) lands. The bounded smoke should consume `adaln_raw` directly. Tighten `_stats(adaln_raw, ..., 50.0)`.

4. **[P1] Document the sin/cos convention difference between Z-Image and ERNIE-Image** in the head comment of `ops/embeddings.mojo`. Right now the file's own docstring only mentions the Z-Image convention, which is what caused the bug.

5. **[P1] Verify `sdpa_nomask[B,S,H,Dh]` consumes BSHD layout (not BHSD).**
   The Mojo block applies RoPE on `[B,S,H,Dh]` and then calls `sdpa_nomask[1, S, ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM](q, k, v, ...)`. Auditor did not re-read `ops/attention.mojo` for this audit; please confirm the layout contract matches before declaring layer-0 parity.

6. **[P2] After (1)–(3), run a cos-sim parity gate** against a single-block diffusers oracle (BF16 GPU) for the real-AdaLN path with the bounded `N_IMG=4, N_TXT=8, S=12` smoke. Target ≥0.999. If parity is below threshold even after the timestep fix, the next suspect is the BSHD-vs-BHSD RoPE permute caveat from B1.

7. **[P2] Rename or remove the "SwiGLU" terminology** in `ernie_image.mojo` (top docstring) and any handoff documents — the FFN is GELU-gated, not SwiGLU. Cosmetic but the diffusers source and the Rust ref both clearly use `gelu()`.

---

## Summary

PASS: A1, A3, B1, B2, C1, C2, C4, D1, D2, E1
WARN: C3
FAIL: A2 (and by extension A4(d), A5)
