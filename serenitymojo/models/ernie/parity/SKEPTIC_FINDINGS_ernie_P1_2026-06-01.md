# SKEPTIC FINDINGS — ERNIE training-port Phase 1 (2026-06-01)

Skeptic attack on the builder's claim: "all 19 grad tensors pass cos≥0.9999999999
at real dims". The block parity gate **reproduces** (I re-ran it — see below), so
the claim is not fabricated. But the pass is built on a **degenerate RoPE table the
real model never uses**, so the RoPE arms of the pass are UNVERIFIED-against-reference
and in fact **REFUTED** when measured against the diffusers ground truth.

Reference ground truth used: diffusers
`transformer_ernie_image.py` (`/home/alex/.local/lib/python3.12/site-packages/diffusers/models/transformers/transformer_ernie_image.py`)
and the real weights at `/home/alex/models/ERNIE-Image/transformer` (409 tensors, config 4096/32/128/36, ffn 12288, theta 256, axes (32,48,48), eps 1e-6).

Gate reproduced (oracle + `/tmp/ernie_block_parity`): all 19 tensors cos ≥ 0.99999999995. The pass is REAL — on the oracle's table.

---

## Attack 1 — Oracle independence  →  PARTIAL: math CLEAN, but oracle is a TAUTOLOGY for RoPE  [MED]

The oracle's own docstring (`block_oracle.py:4-7`) states it "Replicates the EXACT
math of … `block.mojo`'s `ernie_block_forward`" — i.e. it was written to mirror the
Mojo, not derived independently from the reference. That is the classic
tautology-risk. I therefore validated the oracle's math **against the real diffusers
`ErnieImageSharedAdaLNBlock` module** directly (not against the Mojo):

- On shared weights, the REAL diffusers block vs the oracle's reproduced math with
  the **interleaved (real) RoPE table** and **exact-erf GELU**:
  `cos = 1.0000000000000004`, max abs `1.97e-7`. **The non-RoPE graph is correct**:
  pre-norm `modulate(RMSNorm(x))` (diffusers line 266 `x*(1+scale)+shift`), QK-RMSNorm
  (lines 104-107, 192-194), SDPA non-causal scale `1/√Dh`, `gate_msa`-scaled residual
  (line 270), GELU-gated MLP `gelu(gate_proj)*up_proj→linear_fc2` (line 234),
  `gate_mlp`-scaled residual (line 274). All structurally match. **VERIFIED-CLEAN.**

- Two deviations from the reference:
  1. **GELU**: oracle + Mojo use the **tanh approximation** (`block_oracle.py:80-83`,
     `block.mojo:295` `gelu()`); diffusers uses `F.gelu(x)` = **exact erf** (default
     `approximate='none'`, line 234). Measured cos(exact, tanh) = 0.99999999, max abs
     4.7e-4. Tolerable at the BF16 floor, but it is a real reference deviation that the
     tautological oracle hides (the gate compares Mojo-tanh vs torch-tanh → it can never
     catch this). **MED** — confirm flame-core/Klein `.gelu()` is also tanh (likely), else
     a per-block bias accumulates over 36 layers.
  2. **RoPE table** — see Attack 2. This is the load-bearing failure.

Verdict: oracle math independence **VERIFIED-CLEAN for everything except RoPE+GELU**;
the RoPE portion of the oracle is a tautology (it was constructed to be self-consistent
with the Mojo on a table neither matches to the real model).

---

## Attack 2 — RoPE doubling convention  →  REFUTED. FORWARD **and** BACKWARD are wrong vs the real table.  [HIGH — #1 blocker]

This is the decisive finding. The builder flagged the table as a "blocker"; it is worse
than flagged: **both the forward op and the backward op are wrong against the real model.**

### What the real model does (diffusers, decisive)
`ErnieImageEmbedND3.forward` (line 63): `torch.stack([emb,emb],dim=-1).reshape(...)` →
**interleaved doubling** `[θ0, θ0, θ1, θ1, …]`. `apply_rotary_emb` (lines 111-119):
pairing is **half-split** `x1,x2 = x.chunk(2)`, `cat(-x2,x1)`, `out = x*cos + x_rot*sin`.
So channel `i` (i<half) uses `cos[i]` and channel `i+half` uses `cos[i+half]`, and because
the table is interleaved-doubled, **`cos[i] ≠ cos[i+half]`** in general.

Measured (`cos[0:half]` vs `cos[half:D]` on a 1-axis sample): `[-.839,-.839,.408,.408]`
vs `[.154,.154,-.667,-.667]` — **NOT equal** → `cos[i] != cos[i+half]`.

The Mojo real-table builder agrees with diffusers: `build_ernie_rope_tables`
(`models/dit/ernie_image.mojo:455-473`) appends `cos(ang)` **twice consecutively** per
axis → interleaved `[c0,c0,c1,c1,…]`. So the real Mojo forward will receive an
interleaved-doubled table.

### What the gate used
The oracle (`block_oracle.py:153-158`) builds `cos = cat([cos_half, cos_half])` →
**half-split doubling** where `cos[i] == cos[i+half]`. This is a DIFFERENT table that the
real model never produces.

### Forward op is wrong on the real table
`ops/rope.mojo` `_rope_halfsplit_kernel_f32` (lines 100-118, comment lines 12-13) reads
`cos[r,i]`/`sin[r,i]` for `i∈[0,half)` and applies the **same** `cos[i]` to BOTH `o[i]`
and `o[i+half]`. It never reads `cos[i+half]`. Against the diffusers RoPE on the real
interleaved table: **cos = 0.9109, max abs 1.57**. (Note: this Mojo kernel does NOT even
match the oracle's OWN python `rope_halfsplit_full`, which reads `c1=cos[half:]`
separately — they coincide only on the degenerate table, which is why the gate is green.)

### Backward op is wrong on the real table
`rope_backward(interleaved=False)` (`rope_struct_backward.mojo:92-111`) ingests only a
half-width `[rows, D/2]` table and applies one angle per pair. Against torch autograd of
the half-split-full forward on the real interleaved table: **d_x cos = 0.8863, max abs 1.55**.

### Bottom line
- The block gate PASSES only because forward-op-bug and backward-op-bug **cancel** on the
  degenerate `cos[i]==cos[i+half]` table, and the oracle was built on that same table.
- On the real interleaved table the forward is ~0.91 and the backward ~0.89 → the
  real-weight training forward and its gradients will be **wrong**.
- **The "19/19 cos≥0.9999999999" pass does NOT cover the RoPE convention the real model
  uses.** Downgrade the RoPE-touching results (out, d_x, d_wq/k/v, d_q_norm/d_k_norm, and
  everything downstream of attention) to **UNVERIFIED** for the real path.

Required fix path (primitive selection per the plan's own E3 note, but BOTH ops):
- The real ERNIE RoPE = half-split **pairing** with an interleaved-doubled **angle table**
  where the two halves carry different angles. Neither current Mojo op implements this:
  the halfsplit forward assumes one angle per pair; the interleaved arm assumes (2i,2i+1)
  pairing. **A primitive that pairs (i, i+half) but reads cos[i] for o[i] and cos[i+half]
  for o[i+half]** is needed (this is exactly the oracle's python `rope_halfsplit_full`,
  which is correct — the Mojo kernel must be brought up to it). Its backward must consume
  the FULL-width table (both `cos[i]` and `cos[i+half]`), not a half-width table.
- This is **not a one-line swap to the existing interleaved arm** (the plan's E3
  "either way it's a primitive-selection fix" is wrong — the interleaved arm uses (2i,2i+1)
  pairing, which is a third, also-wrong convention). It needs a corrected halfsplit
  forward kernel + a matching full-width-table halfsplit backward kernel, then a new
  oracle that builds the real interleaved table and re-gates.

---

## Attack 3 — Small-S 3-axis masking  →  REFUTED (degenerate; covers nothing positional)  [HIGH, folds into Attack 2]

The gate ran at S=8 with an **arbitrary** cos/sin table (`block_oracle.py:155-156`:
`t2(S*H, half, 0.030, 0.20, 1.0)*0.6` — pure `sin(0.03*idx+…)` fills). It does **NOT**
build positions from the 3-axis (image row/col grid + text) structure at all. The real
positional assignment (`build_ernie_rope_tables:441-473` / diffusers `forward:367-390`:
image axis-0 = `text_len`, axis-1 = row, axis-2 = col; text axis-0 = token idx) is never
exercised. So:
- The gate cannot catch the axis-concat ordering (32|48|48), the image vs text axis
  assignment, the `text_len` offset on image axis-0, or any per-token angle.
- Combined with Attack 2, the gate proves nothing about the real RoPE — neither the
  doubling convention NOR the position math. **The full real positional structure is
  uncovered.** The E2 full-stack gate (per-block cos ≥ 0.999 with the REAL table builder)
  is the first thing that will actually test this, and it will fail until Attack 2 is fixed.

---

## Attack 4 — Weight key map  →  VERIFIED-CLEAN  [LOW]

Dumped the real `transformer/*.safetensors` keys (409 tensors). All 11 per-block keys
load by exact name in `weights.mojo:104-115`:
`layers.{i}.adaLN_sa_ln.weight`, `self_attention.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}.weight`,
`adaLN_mlp_ln.weight`, `mlp.{gate_proj,up_proj,linear_fc2}.weight`. These match the real
model keys, the diffusers module names, and the Rust loader (`ernie_image.rs:414-424`)
exactly. Orientation note (`weights.mojo:27-30`): stored `[out,in]`, Mojo `linear` does
`x@Wᵀ` internally → consumed directly, no pre-transpose; consistent with Klein. No key is
silently defaulted or missing. `tensor_info` would raise on a missing key (no silent default).
- Minor (LOW): `configs/ernie_image.json` does not carry `rope_axes_dim`; it lives in
  comptime constants (`ERNIE_DIT_ROPE_AXIS_{0,1,2}`). Verify those constants equal (32,48,48)
  before the real run (config-driven-binding tenet) — not a Phase-1 gate blocker.

---

## Attack 5 — Backward chain shortcuts (mod-vec + gate grads)  →  VERIFIED-CLEAN (chained, not stubbed)  [LOW]

- The oracle backprops into the 6 mod vectors **independently**: they are torch leaves
  with `requires_grad_(True)` (`make_mod` + lines 132-133), and `loss.backward()` fills
  `m[k].grad` via autograd — NOT copied from the Mojo. So the 6 mod-vec grads are a genuine
  independent reference. The Mojo chains them through `modulate_backward` (`block.mojo:360,419`)
  and `gate_residual_backward` (`block.mojo:333,376`) — d_scale/d_shift from modulate, d_gate
  from the gate-residual reduction. All six matched cos ≥ 0.99999999995 on the gate.
- The hand-chained backward is structurally faithful to the Klein single-block pattern:
  dual-use accumulation is correct — `d_h = grg2.d_x + rb_mlp.d_x` (line 370), `d_sa_in =
  lb_q.d_x + lb_k.d_x + lb_v.d_x` (line 416, split q/k/v), `d_x = grg1.d_x + rb_sa.d_x`
  (line 430). No arm is stubbed or zeroed.
- **Caveat**: this is CLEAN **only on the degenerate table**. d_scale_msa/d_shift_msa/d_gate_msa
  and d_q_norm/d_k_norm sit downstream of the broken RoPE, so on the real table their values
  will move (they are correct *given* the attention output, but the attention output is wrong).
  The MLP-side mod-vec grads (d_*_mlp) are RoPE-independent and remain trustworthy.

---

## Prioritized fix list for the bugfixer (before Phase 2)

1. **[HIGH / BLOCKER] Fix the ERNIE RoPE primitive — forward AND backward.**
   The real convention = half-split pairing (i, i+half) with an interleaved-doubled angle
   table where `cos[i] ≠ cos[i+half]`.
   - Forward: `ops/rope.mojo` `_rope_halfsplit_kernel_*` must read `cos[r,i]` for `o[i]`
     and **`cos[r,i+half]`** for `o[i+half]` (i.e. consume the full-width table), matching
     `block_oracle.py:86-101` (the oracle's python is already correct) and diffusers
     `apply_rotary_emb`. Current kernel applies one angle per pair → cos 0.91 vs real.
   - Backward: add/repair a halfsplit backward that consumes the **full-width** table
     (`cos[i]`, `cos[i+half]` distinct), not the half-width `[rows,D/2]` table. Current
     `rope_backward(False)` → d_x cos 0.89 vs real.
   - This is a flame-core/`ops` primitive fix (Tenet 1), NOT a model-file workaround, NOT
     the existing interleaved (2i,2i+1) arm.
2. **[HIGH] Re-gate against the REAL table.** Rewrite the oracle to build cos/sin via the
   diffusers `ErnieImageEmbedND3` interleaved doubling on real 3-axis ids (image row/col
   grid + text), exercising S that spans both image and text tokens. The current
   degenerate `sin(0.03*i)` table must go. Only a green gate on the interleaved table
   counts. (Bonus: gate the full `build_ernie_rope_tables` Mojo builder against
   `ErnieImageEmbedND3` directly.)
3. **[MED] GELU convention.** Decide tanh vs exact-erf. diffusers uses exact erf
   (`F.gelu`, approximate='none'). If flame-core/Klein and the real path standardize on
   tanh, document the BF16-floor deviation; otherwise switch the Mojo `gelu` to erf for
   ERNIE. Either way the oracle must match the real model's GELU, not the Mojo's.
4. **[LOW] Verify `ERNIE_DIT_ROPE_AXIS_{0,1,2}` constants == (32,48,48)** and surface
   `rope_axes_dim` into the JSON config (config-driven binding).

### What IS safe to carry into Phase 2 unchanged
- Non-RoPE block graph (modulate, residual gates, FFN gating, QK-RMSNorm, SDPA): matches
  the real diffusers block at cos 1.0 (exact GELU). VERIFIED-CLEAN.
- Weight key map + loader: VERIFIED-CLEAN.
- Backward chaining structure + mod-vec grad independence: VERIFIED-CLEAN (subject to the
  RoPE fix re-flowing the attention-side grads).
