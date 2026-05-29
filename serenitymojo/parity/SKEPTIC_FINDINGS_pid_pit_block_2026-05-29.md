# SKEPTIC FINDINGS — PiD PiTBlock parity gate

Date: 2026-05-29
Auditor: SKEPTIC (re-audit of builder's `pit_block` claim)
Subject: `serenitymojo/models/pid/pit_block.mojo` :: `pit_block_forward`
Gate: `serenitymojo/models/pid/pit_block_smoke.mojo`
Reference: `serenitymojo/models/pid/parity/gen_pit_block_reference.py` (system python3, torch 2.10, seed 4242)
Repo source: `/tmp/PiD_repo/pid/_src/networks/pixeldit_official.py` lines 416–509 (PiTBlock), plus RotaryAttention (174), MLP (128), RMSNorm (122), precompute_freqs_cis_2d_ntk (131), apply_rotary_emb (90).

## Verdict

PASS — with one documented caveat about gate sensitivity (not a correctness defect).

The implementation is structurally faithful to the repo and numerically correct on
the fixture (cos = 1.0, max_abs = 1.19e-07 = F32 rounding floor, n = 192). The gate
is real (GPU, not compile-only) and fails-closed on the dominant path. It is, however,
INSENSITIVE to errors in the gated inner branches (attention / RoPE / adaLN-scale)
because the reference fixture scales those branch weights to 0.1–0.2x, leaving the
residual passthrough `x` dominant. This is a test-strength limitation, not an
implementation bug — the correct code still produces F32-exact agreement everywhere.

## Structural audit vs repo (all confirmed correct)

- adaLN: `Linear(context_dim -> 6*pixel_dim*P2)`, view `[BL,P2,6C]`, chunk6 ->
  shift/scale/gate {msa,mlp} each `[BL,P2,C]`. apply_adaln = `x*(1+scale)+shift`,
  per-token broadcasting (not per-channel `modulate`). MATCHES repo lines 467, 503.
- compress_to_attn: `Linear(P2*pixel_dim -> attn_dim)` on flattened x_norm,
  reshaped to `[B, L, attn_dim]`. Attention runs at attn_dim across L tokens
  (NOT at pixel_dim across P2). MATCHES repo lines 440, 497.
- RotaryAttention: qkv_bias=False, qk_norm=True (head-dim RMSNorm eps=1e-6),
  qkv reshape `(B,N,3,H,Hc).permute` -> split q/k/v; q_norm/k_norm before RoPE;
  apply_rotary_emb (interleaved view_as_complex); SDPA scale = head_dim^-0.5;
  proj has bias. MATCHES repo lines 174-220.
- RoPE: `rope_interleaved` computes out[2i]=x0*cos-x1*sin, out[2i+1]=x0*sin+x1*cos
  == complex mult (x0+i*x1)*(cos+i*sin), identical to torch view_as_complex path.
  NTK 2D table interleaves (x_freq,y_freq) per pos, pos ordered row-major (H then W),
  matching `cat([x_cis,y_cis],-1).reshape(H*W,-1)` with meshgrid indexing="ij".
  MATCHES repo lines 90-96, 131-176.
- expand_from_attn: `Linear(attn_dim -> P2*pixel_dim)`, reshaped `[BL,P2,C]`. MATCHES.
- MLP: fc1 -> nn.GELU (EXACT erf, builder wrote `gelu_erf` 0.5*x*(1+erf(x/sqrt2)),
  NOT the tanh approx) -> fc2. MATCHES repo lines 128-143.
- residuals: `x = x + gate_msa*attn_exp`; `x = x + gate_mlp*mlp(...)`. MATCHES repo lines 502, 505.

Reference generator (`gen_pit_block_reference.py`) mirrors all the above sub-modules
verbatim (verified line-by-line) in the cp_size=1, mask=None path. Tiny config exercises
multi-patch (Hs=2,Ws=3,L=6), multi-head (H=2,Dh=8), NTK (2,3 vs ref 1024).

## Gate re-run (verbatim, unmutated)

    PiTBlock.forward   cos= 1.0   max_abs= 1.1920928955078125e-07   n= 192   [ PASS ]
    ALL GATES PASS

GPU run, ~20GB free. Real numeric gate, not compile-only.

## Fail-closed verification (mutate -> confirm fail -> revert)

ParityHarness gates on COS ONLY (`passed = cos >= 0.999`); max_abs is reported but
not thresholded (`serenitymojo/parity.mojo:92`).

1. adaLN `+1` dropped (`add_scalar(scale_msa,1.0)` -> `0.0`):
   cos = 0.99999999981, max_abs = 8.6e-05 -> STILL PASS (caveat below).
2. RoPE cos/sin SWAPPED (`rope_interleaved(q_n, sin, cos)`):
   cos = 0.99999999998, max_abs = 2.8e-05 -> STILL PASS (caveat below).
3. First residual `x` passthrough replaced by `shift_msa`
   (`add(x, gate*attn)` -> `add(shift_msa, gate*attn)`):
   cos = 0.0758, max_abs = 2.88 -> FAIL (gate fails-closed on dominant path). ✅

All three mutations reverted; post-revert re-run restores cos = 1.0, max_abs = 1.19e-07,
`git diff` clean.

## Caveat (gate sensitivity, NOT a code defect)

PiTBlock is residual-dominated: output ≈ input passthrough `x` plus two GATED branches
(attn, mlp). The reference scales attn/mlp/adaLN weights by 0.1–0.2x (gen script lines
232-238), so the inner branches contribute << the passthrough. Cosine similarity, being
magnitude-normalized and dominated by the largest component, therefore cannot detect
errors confined to those branches: both the adaLN-scale break and a full RoPE cos/sin
swap left cos > 0.999 (though max_abs rose 700x and 240x respectively — proving the
branches ARE live and the unmutated 1.19e-7 is genuine all-path agreement, not a dead
branch). The correct implementation produces F32-exact numbers on every path; the gate
simply lacks the resolution to FAIL a wrong inner branch.

Recommended hardening (optional, for a future tightening pass — does not block this phase):
- Add a max_abs threshold (e.g. < 1e-4) to the pass criterion, OR
- In the reference, do NOT down-scale the branch weights (or down-scale the residual
  input instead) so attn/RoPE/adaLN carry comparable magnitude, OR
- Add a per-branch sub-gate (compare x_comp, attn_out, mlp_out individually).

## Conclusion

The PiTBlock port matches the PiD repo verbatim and is numerically correct
(cos=1.0, max_abs at F32 floor). The gate runs on real GPU and fails-closed on the
dominant path. The one limitation — cos-only thresholding being blind to small gated
branches under the down-scaled fixture — is a test-strength issue, not a defect in
`pit_block.mojo`. No correctness problem found. No code changes made (all mutations reverted).
