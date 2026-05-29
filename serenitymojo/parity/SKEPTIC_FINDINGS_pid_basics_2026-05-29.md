# SKEPTIC findings — PiD scalar primitives ("basics")

Date: 2026-05-29
Subject: serenitymojo/models/pid/pid_ops.mojo (patchify/unpatchify, NTK 2D RoPE,
TimestepConditioner, SigmaAwareGate) + parity smoke pid_ops_smoke.mojo.
Reviewer: skeptic agent (audit vs /tmp/PiD_repo source + mutation/fail-closed re-run).

## Verdict: PASS — all four primitives faithful, every gate fail-closed.

The builder's claim ("ALL GATES PASS") reproduced on the RTX 3090 Ti (9.7 GB free),
real numeric GPU gates (not compile-only). I audited each oracle against the PiD
repo source, then mutated each primitive in the spec-flagged way and confirmed the
gate FAILS and the smoke `raise`s (fail-closed). Clean state restored and re-run green.

## Source audit (oracle vs /tmp/PiD_repo)

1. patchify == `F.unfold(x, ks=ps, stride=ps).transpose(1,2)`  — pid_net.py:303. ✓
   Token channel order `c*ps*ps + kh*ps + kw` matches unfold. Bit-exact (cos=1.0, max_abs=0).

2. unpatchify == `F.fold`  — pid_net.py:464. ✓
   AUDITED the actual network fold path (pid_net.py:454-464), which is NOT a naive
   fold of unfold-ordered tokens:
     final_layer -> x_pixels [B*L, P², C_out]
     .view(B,L,P2,C).permute(0,3,2,1).view(B, C*P2, L) -> fold
   The permute reduces the fold-input channel index to `c*P2 + (kh*ps+kw)` ==
   `c*ps*ps + kh*ps + kw` == the SAME ordering as unfold. So the builder's
   unpatchify (inverse scatter with C-outer, spatial-inner) is correct for PiD's
   fold path. Initial concern about `(P2,C)` vs `(C,P2)` ordering RESOLVED — the
   `view(B,L,P2,C)` is on the final-linear output last dim and is undone by the
   permute. Round-trip unpatchify(patchify(x))==x bit-exact (cos=1.0, max_abs=0).

3. NTK 2D RoPE == `precompute_freqs_cis_2d_ntk` — pixeldit_official.py:154-193. ✓
   - dim_axis = dim//2; ntk_factor = (cur/ref)**(dim_axis/(dim_axis-2)), dim_axis>2.
   - h_theta/w_theta per-axis; freqs_w uses w_theta, freqs_h uses h_theta.
   - meshgrid(y,x, indexing="ij") row-major; reshape interleaves col 2k=x, col 2k+1=y.
   - Mojo `is_y` (odd col) -> h_theta + y_pos; even col -> w_theta + x_pos. Matches.
   - ref grid 1024, tested at non-ref 4×6 (NTK h=2.7e-3 / w=4.2e-3) so per-axis
     scaling is genuinely exercised, not the ref-grid identity.
   cos: cos=0.99999999999997, max_abs=9.39e-7. sin: cos=0.99999999999991, max_abs=1.56e-6.

4. TimestepConditioner — pixeldit_official.py:80-108. ✓
   - max_period=10 (NOT 10000) confirmed in source (line 91 default).
   - freqs=exp(-ln(mp)*arange(half)/half); emb=cat([cos,sin]) COS-FIRST. Matches.
   - forward: Linear(freq->hidden) -> SiLU -> Linear(hidden->hidden). Matches.
   embedding gate cos=0.99999999999999 max_abs=3.13e-7; full forward cos=0.99999999999999 max_abs=1.49e-8.

5. SigmaAwareGatePerTokenPerDim — lq_projection_2d.py:28-56. ✓
   content_proj(cat([x,lq],-1)); offset=-exp(log_alpha)*sigma.view(-1,1,1);
   gate=sigmoid(logit+offset); out=x+gate*lq. Matches exactly (init bias=2.0,
   log_alpha=log(5)). Tested sigma={0.0, 0.4} -> gate ~0.88/~0.5 (per repo docstring).
   cos=0.99999999999999 max_abs=2.38e-7.

## Mutation / fail-closed verification (each mutated, gate re-run, confirmed FAIL)

| # | Mutation                                              | Gate | Result (cos / max_abs)        |
|---|-------------------------------------------------------|------|-------------------------------|
| a | patchify token order `c*ps*ps+..` -> `krem*C+c`       | 1    | FAIL cos=0.011  max_abs=5.97  |
| b | NTK RoPE is_y branch uses w_theta instead of h_theta  | 3a   | FAIL cos=0.978  max_abs=0.625 |
| c | timestep max_period 10 -> 10000                       | 4    | FAIL cos=0.978  max_abs=0.435 |
| d | sigma offset sign `-exp(log_alpha)` -> `+exp(...)`    | 6    | FAIL cos=0.976  max_abs=1.15  |

In every case the smoke raised `Error("pid_ops smoke: gate failure")` and exited
non-zero (fail-closed). All four mutations reverted; clean state re-run = ALL GATES PASS.

## Notes / caveats for the next phase
- Mutation (b) only drops cos to 0.978 because at the tiny 4×6 grid the h-axis vs
  w-axis NTK factors differ modestly (2.7e-3 vs 4.2e-3). The 0.999 threshold and the
  max_abs (0.625) still catch it cleanly, but a sharper RoPE regression test at a
  grid with larger H/W asymmetry would tighten the margin.
- The gate uses random SEEDED weights (seed=7777), tiny inputs, F32 — no checkpoint
  this phase, per plan. These primitives are verified as ops, not against real PiD
  weights; weight-loading parity is a separate downstream gate.
- The NTK reference theta default is 10000.0 (pixeldit_official.py:160); the docstring
  shorthand "theta=1e4" is consistent.

No git commit made.
