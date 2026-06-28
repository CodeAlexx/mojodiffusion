# LyCORIS carrier-dispatch + OneTrainer re-target — 2026-06-27

Authoritative detail doc for the LyCORIS "make all families trainable" workstream
(TODO.md ACTIVE row 12). Extends T2.F/F-2/T2.G (row 7): those verified the 7
family PRIMITIVES vs pip lycoris 3.4.0 and shipped LoKr-on-klein; this session
verified vs the version ai-toolkit ACTUALLY ships, generalized the trainer
dispatch to a second family, and re-targeted DoRA/OFT to OneTrainer.

Ledger: `recall lycoris carrier` → MJ-1020/1021/1022/1023/1024 (eng-knowledge).

## What shipped (all gates run THIS session, built `--optimization-level 2`)

### 1. ai-toolkit (lycoris 1.8.3) verification — MJ-1020
- ai-toolkit ships **lycoris_lora 1.8.3**, NOT the 3.4.0 the T2.F gates used
  (1.8.3 API: `get_weight(orig_weight)`/`make_weight`, NO `get_diff_weight`).
- MEASURED: 1.8.3 reconstruction == the Mojo formula EXACT (cos=1.0, norm_rel=0.0)
  for LoKr (both-full + both-factored), LoHa, LoCon-linear. The reconstruction
  math is version-stable 1.8.3↔3.4.0.
- `lycoris_family_parity.mojo` re-run this session: all families PASS (LoHa exact;
  LoKr×3 cos≥0.99999; LoCon cos=0.99999; saves byte-exact).
- The 1.8.3 `scalar` gate (per-module learnable, init 0 LoKr / 1 LoHa-LoCon, folded
  into the first factor on save) is fold-on-save → reconstruction + loadability
  UNAFFECTED; matters only for bit-exact training-TRAJECTORY parity (TODO row 6 / opt).

### 2. Shared (a,b) carrier dispatch — additive families, e2e on REAL Klein-9B
The model LoRA stack consumes plain-LoRA `(a,b)` adapters. An additive low-rank
LyCORIS delta factors into a SMALL carrier (no stack/kernel change):
- **LoKr** (MJ-1021): Kronecker → `r_eff` per L1/L2/L3 (`lokr_stack.mojo`, pre-existing
  T2.G). `klein_stack_lokr_real_smoke.mojo` PASS on real Klein-9B: 144 carriers (83 MB),
  fwd+all d_A/d_B FINITE, master AdamW moved the W2 zero-leg 0→4.58, `save_klein_lokr`.
- **LoHa** (MJ-1022, NEW): Hadamard of two rank-r products has rank ≤ r² ⇒ carrier
  `r_eff=r²` (NOT a VRAM-prohibitive full delta). Derivation:
  `a[(k,l),i]=w1a[i,k]·w2a[i,l]`, `b[o,(k,l)]=scale·w1b[k,o]·w2b[l,o]`, `b@a=ΔW^T`.
  New `loha_stack.mojo` (carrier + chain + klein orchestration mirroring lokr_stack).
  Gates: `loha_carrier_parity.mojo` PASS (fwd cos=0.99999986; all 4 factor grads
  cos≥0.99999 vs `loha_backward`); `klein_stack_loha_real_smoke.mojo` PASS on real
  Klein-9B (w2a zero-leg 0→403, save 144 modules).
- LoCon-linear ≡ plain LoRA on klein (all-linear DiT) → already covered.
- **Limit (measured):** OFT/BOFT (multiplicative block-rotation, full-rank `(R−I)W`)
  and DoRA (per-column renorm) are NOT (a,b)-carrier compatible → need new stack hooks.

### 3. OneTrainer re-target (DoRA/OFT/BOFT oracle = OneTrainer, per user)
- **DoRA** (MJ-1023, FIXED fwd): OneTrainer's default DoRA normalizes per-INPUT axis
  `[1,in]`; Mojo/lycoris used per-OUTPUT `[out]` (measured cos 0.9988 apart, faithful
  OneTrainer forward). Added `wd_on_out` flag to `dora_adapter.mojo` (default True=
  lycoris; False=OneTrainer per-input via `_col_l2_norm` + axis-aware eff-weight/
  backward). `dora_onetrainer_parity.mojo` PASS: per-input Mojo == OneTrainer's OWN
  `DoRAModule.forward` cos=0.99999998 nrel=2.0e-4. lycoris path UNCHANGED (no regression).
- **OFT** (MJ-1024, SCOPED): OneTrainer default = 5-term NEUMANN Cayley (non-orthogonal
  truncation), NOT exact Cayley. Measured (OneTrainer's own `_cayley_batch`):
  cos(Neumann,exact)=0.974@skew0.05 → 0.31@0.50; Neumann ‖RᵀR−I‖ 1e-4→60.9. The Mojo
  exact-Cayley does NOT match OneTrainer. Re-target = port `R=I+2Q+2Q²+2Q³+Q⁴`
  (Q=skew(weight), no 0.5, input-side) + a matching backward. Not yet built.

## Files (this session)
- `serenitymojo/training/loha_stack.mojo` — LoHa carrier + klein orchestration
- `serenitymojo/training/dora_adapter.mojo` — `wd_on_out` axis flag (+`_col_l2_norm`)
- `serenitymojo/training/tests/loha_carrier_parity.mojo` — LoHa carrier gate
- `serenitymojo/training/tests/dora_onetrainer_parity.mojo` + `gen_dora_onetrainer_oracle.py`
- `serenitymojo/models/klein/parity/klein_stack_{lokr,loha}_real_smoke.mojo`

## OPEN (next, in order)
1. **OFT/BOFT primitive rewrite** to OneTrainer 5-term Neumann + input-side + new
   backward (MJ-1024). DoRA per-input BACKWARD FD-gate.
2. **Multiplicative/renorm stack-hook mechanism** — the shared integration for
   OFT/BOFT (pre-projection block rotation) + DoRA (per-column eff-weight rescale)
   into the klein stack. The big architectural piece; design deliberately.
3. **Production dispatch wiring** — `serenity-trainer` (its own lora_block, autograd_v2
   path); `adapter_algo` dispatch is wired into NO live trainer yet (the
   `adapter_algo_policy` guard correctly fails loud). Wire LoKr/LoHa carrier + loosen
   the guard once wired. Then replicate to other trainers (zimage…).
4. (opt) trainable `scalar` surface for ai-toolkit training-trajectory parity.
