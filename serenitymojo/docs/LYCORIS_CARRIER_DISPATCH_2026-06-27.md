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
- **OFT** (MJ-1024, FIXED fwd+bwd): OneTrainer default = 5-term NEUMANN Cayley
  (non-orthogonal truncation), NOT exact Cayley (measured cos(Neumann,exact)=0.974
  @skew0.05 → 0.31@0.50). New `training/oft_onetrainer.mojo` (distinct from the lycoris
  exact-Cayley `oft_adapter.mojo`): `R=I+2Q+2Q²+2Q³+Q⁴`, Q=skew(vec) no-0.5, input-side
  block rotation. `oft_onetrainer_parity.mojo` PASS (fwd cos=0.99999999999999 vs
  OneTrainer's own OFTModule.forward); `oft_onetrainer_backward_fd.mojo` PASS (analytic
  d_vec/d_x cos≥0.999999999 vs central-FD).
- **BOFT** (MJ-1025, RESOLVED): OneTrainer has NO BOFT, and ai-toolkit's lycoris 1.8.3
  has no oft/boft/dora either — BOFT exists ONLY in lycoris 3.4.0. So BOFT stays on its
  only oracle (lycoris 3.4.0, already gated by `lycoris_family_parity.mojo`); no
  re-target possible/needed.
- **DoRA backward** also gated vs OneTrainer's autograd (detached norm): d_A cos=0.99998,
  d_B cos=0.99999988, d_m cos=0.99999. DoRA primitive COMPLETE fwd+bwd vs OneTrainer.

## Files (this session)
- `serenitymojo/training/loha_stack.mojo` — LoHa carrier + klein orchestration
- `serenitymojo/training/dora_adapter.mojo` — `wd_on_out` axis flag (+`_col_l2_norm`)
- `serenitymojo/training/tests/loha_carrier_parity.mojo` — LoHa carrier gate
- `serenitymojo/training/tests/dora_onetrainer_parity.mojo` + `gen_dora_onetrainer_oracle.py`
- `serenitymojo/models/klein/parity/klein_stack_{lokr,loha}_real_smoke.mojo`

## PRIMITIVES — ALL DONE
- Additive (LoKr/LoHa/LoCon): verified vs ai-toolkit 1.8.3, carrier dispatch e2e on klein.
- DoRA: re-targeted to OneTrainer, fwd+bwd verified (MJ-1023).
- OFT: re-targeted to OneTrainer (5-term Neumann), fwd+bwd verified (MJ-1024).
- BOFT: lycoris-3.4.0-only (no OneTrainer/ai-toolkit equivalent), already gated (MJ-1025).

## OPEN (next, in order)
1. **Multiplicative/renorm stack-hook mechanism** — the shared integration into the
   model stack: pre-projection block rotation (OFT/BOFT) + per-column effective-weight
   rescale (DoRA). NOT (a,b)-carrier compatible. The big architectural piece; design
   deliberately. (LoKr/LoHa already train via the carrier.)
2. **Production dispatch wiring** — `serenity-trainer` (its own lora_block, autograd_v2
   path); `adapter_algo` dispatch is wired into NO live trainer yet (the
   `adapter_algo_policy` guard correctly fails loud). Wire LoKr/LoHa carrier + loosen
   the guard once wired. Then replicate to other trainers (zimage…).
3. (opt) trainable `scalar` surface for ai-toolkit training-trajectory parity.
