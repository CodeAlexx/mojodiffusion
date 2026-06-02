# SKEPTIC FINDINGS — Anima block fwd+bwd parity gate (P1b, round 2)

Date: 2026-06-01
Auditor stance: assume the gate LIES until proven otherwise (Tenet 4 — reproduce/refute every claim with a tool result).

Artifacts attacked:
- `serenitymojo/models/anima/block.mojo` (`anima_block_forward` / `anima_block_backward`, new `_cross_attn` path)
- `serenitymojo/models/anima/parity/block_oracle.py`, `block_parity.mojo`
- Reference: `/home/alex/EriDiffusion/inference-flame/src/models/anima.rs` (self-attn @378, cross-attn @433, `build_3d_rope_cossin` @1044)
- flame-core `rope_halfsplit_bf16` (`flame-core/src/bf16_ops.rs:1185`)

## Reproduction: gate runs GREEN on the REAL table

Oracle dumped, gate rebuilt clean (`BUILD_EXIT=0`), ran clean (`RUN_EXIT=0`).
All 23 compared tensors cos ≥ 0.99999999 (max observed deviation a BF16-floor
1e-13 from 1.0). Key lines:

```
out        cos 0.9999999999999266   d_x  0.9999999999998102   d_t_silu 0.999999999999883
d_sa_q 0.99999999999953  d_sa_k 0.99999999999942  d_sa_qn 0.99999999999976
d_ca_q 0.99999999999957  d_ca_k 0.99999999999963  d_ca_v 0.99999999999963
d_mlp1 0.9999999999988   d_sa_mod1 0.99999999999978  d_mlp_mod2 0.99999999999866
VERDICT: PASS
```

---

## ATTACK 1 — RoPE NON-DEGENERACY (the Ernie trap) → **VERIFIED-CLEAN** (HIGH)

The Ernie gate went green on a degenerate small-S table where the half-split bug
was invisible. The Anima gate's table is **NOT** degenerate.

Evidence (recomputed the oracle table independently, `block_oracle.py:131-139`):
```
oracle cos shape (6, 64)            # [S_IMG, half_d] — half-width, matches anima.rs
pos1 num distinct cos bins: 64 of 64
pos5 num distinct cos bins: 64 of 64
pos5 sin min/max: -0.9589 .. 0.9999
frac bins with |sin|>0.01: 0.508    # half the bins carry a real rotation
```
- Every position has 64 distinct cos bins; sin reaches ±0.96; ~51% of bins have
  non-trivial sin. A wrong pairing/sign in the half-split rope would change the
  output — the op is genuinely exercised. This is NOT the Ernie all-zero/aliased
  table.
- The real anima.rs table (`build_3d_rope_cossin`, anima.rs:1044-1161) is also a
  **half-width** `[1,1,S,half_d=64]` table (it takes `chunk[..half_d]`, the first
  half of the doubled data, line 1141-1146). The oracle's table is the same
  *shape and width*; only the *frequency content* differs (oracle: single monotonic
  theta ramp; anima: 3-axis 22/21/21 split with per-axis NTK theta). For testing
  fwd/bwd **math correctness**, the specific frequency values are irrelevant — the
  half-split arithmetic `o[i]=x0·c−x1·s, o[i+half]=x1·c+x0·s` is identical regardless
  of how the 64 angles were generated. Non-degeneracy (proven above) is what matters,
  and it holds.

Teeth proof: flipping the block's two `rope_backward(..., False)` calls to `True`
(interleaved) and rebuilding collapses the rope-touched grads while V (no rope)
stays clean — the gate FAILS exactly where it should:
```
d_sa_q 0.823 FAIL   d_sa_k 0.806 FAIL   d_sa_qn 0.855 FAIL
d_sa_v 0.99999 PASS  (V correctly bypasses rope)
d_x 0.946 FAIL       d_sa_mod1 0.988 FAIL
VERDICT: FAIL
```
Mutation reverted; block.mojo:621-622 restored to `False`.

## ATTACK 2 — RoPE CONVENTION correctness (mirror of Ernie's fwd/bwd width mismatch) → **VERIFIED-CLEAN** (HIGH)

Ernie's bug was a fwd/bwd table-width mismatch: forward used a FULL-width table
(`cos[i] != cos[i+half]`) but backward read only `cos[i]`. The flame-core source
documents this exact hazard.

- anima.rs:378-380 self-attn uses `rope_halfsplit_bf16(q, cos, sin)` where cos/sin
  are `[1,1,S,D/2]` (half-width). flame-core kernel (`bf16_ops.rs:1185`,
  `rope_halfsplit_bf16_impl`) flattens cos to `[cos_bh, N, half]` and pairs
  `(i, i+half)` with the single angle `cos[i]`. Confirmed half-split single-angle.
- Mojo forward `_rope_halfsplit_kernel_f32` (rope.mojo:113-118): reads `cos[r,i]`,
  `sin[r,i]`; `o[i]=x0·c−x1·s`, `o[i+half]=x1·c+x0·s`. Matches.
- Mojo backward `rope_backward(interleaved=False)` → `_rope_bwd_halfsplit_kernel_f32`
  (rope_struct_backward.mojo:92-111): reads the SAME `cos[r,i]`, `sin[r,i]` (half-width),
  `dx[i]=g0·c+g1·s`, `dx[i+half]=−g0·s+g1·c`. This is the exact Jacobian-transpose
  of the forward with one shared angle. **Matched fwd/bwd pair.**
- The codebase even ships a `rope_halfsplit_full_backward` (rope_struct_backward.mojo:125-145,
  254+) for the FULL-width case and warns (line 268-271): "Use this (NOT
  rope_backward(..., interleaved=False)) whenever the forward used a full-width table…
  it silently aliases the wrong angle otherwise." Anima does NOT use a full-width
  table (anima.rs:1148 emits `[1,1,S,half_d]`), so `interleaved=False` is the
  CORRECT choice. No mirror bug.

Independent forward cross-check: reimplemented the whole block forward using the
`rotate_half` formulation (`x·cos + rotate_half(x)·sin` with doubled `[cos,cos]`
tables) — a different algebraic spelling from the oracle's explicit o1/o2 — and it
reproduces the oracle's `ref_out` at **cos 0.9999999999999976, max_abs 0.0050**.
The half-split convention is the genuine Cosmos/anima `apply_rotary_emb`.

## ATTACK 3 — ORACLE INDEPENDENCE → **VERIFIED-CLEAN** (HIGH)

- `block_oracle.py` builds the forward in standard torch ops (lines 75-203) and
  obtains ALL grads via `torch.autograd` `.backward()` (line 207). The Mojo block
  **hand-chains** its backward (block.mojo:515-660). These are structurally
  independent: autograd differentiation vs a hand-written reverse pass. A
  transcription tautology would require the oracle to also hand-write the backward;
  it does not.
- The oracle's forward math matches anima.rs op-for-op: LayerNorm-no-affine AdaLN-pre
  (oracle:76-79,166-168 ↔ anima.rs `apply_adaln`/`modulate_pre_fused`), per-head
  RMSNorm eps=1e-6 (oracle:82-85 ↔ anima.rs:375), half-split RoPE on self-attn only
  (oracle:87-99,178-179 ↔ anima.rs:378-380), maskless rectangular cross-attn
  (oracle:184-194 ↔ anima.rs:397-441), tanh-GELU MLP (oracle:113-114,199 ↔
  anima.rs:447-452), AdaLN-LoRA-256 W1→W2 chain + base_adaln (oracle:158-164 ↔
  anima.rs `adaln_modulation`), F32 gated residuals (oracle:182,194,201 ↔
  anima.rs:483,496,508).
- Verified independently above that a SECOND torch forward (rotate_half spelling)
  matches `ref_out`, so the oracle's forward is not self-referential to the Mojo.

## ATTACK 4 — CROSS-ATTN asymmetry actually exercised → **VERIFIED-CLEAN** w/ one LOW note

- Forward routes through the genuine rectangular path: `sdxl_sdpa[1, S_IMG=6, S_TXT=8, H, Dh]`
  (block.mojo:399). `sdxl_sdpa` (sdxl_attention.mojo:264-276) hard-asserts q `[B,Sq,H,Dh]`
  and k `[B,Skv,H,Dh]` with Sq=6, Skv=8 — a square path (Sq==Skv) would have failed the
  shape assert. The gate ran with S_q=6 ≠ S_kv=8 and PASSED, so the asymmetric path is
  load-bearing and correct (d_ca_q/k/v all cos 0.9999).
- Backward uses `sdpa_backward_rect[1, S_IMG, S_TXT, H, Dh]` (block.mojo:563), the
  rectangular SIBLING (attention_backward.mojo:497-529), not the square `sdpa_backward`.
- d_context gating: block.mojo:593-594 intentionally discards the k/v-side d_x (text
  context is frozen, not a block leaf). This is correct — cross-attn LoRA lives on the
  projection weights `ca_k`/`ca_v` (both tested: d_ca_k 0.99999, d_ca_v 0.99999), NOT
  on the context. The oracle sets `context.requires_grad_(True)` but never emits
  `ref_d_context`, so the gate consistently does not test that grad. Not needed.
- NO-mask claim: anima.rs:433 is `sdpa(&q,&k,&v,None)` — the reference itself is
  maskless. The Mojo matches the reference. **LOW note:** the gate runs S_txt=8 with no
  padding, so it cannot catch a wrongly-omitted *text-padding* mask. But the reference
  has none, so this is not a block-math defect — it is a trainer-level question (variable
  text length) the reference does not answer. Flag for Phase B if real captions are
  padded.

## ATTACK 5 — AdaLN-LoRA-256 mod-grad independence → **VERIFIED-CLEAN** (HIGH)

- All 6 mod weights are independent torch leaves: confirmed `is_leaf=True,
  requires_grad=True` with correct shapes — mod1 `[256,2048]` (W1 down 2048→256),
  mod2 `[6144,256]` (W2 up 256→6144). Their grads come from torch autograd, not copied
  from the Mojo.
- The oracle backprops the FULL W1→W2 chain (oracle:158-164: `h=t_silu@mod1.T;
  mod_out=h@mod2.T+base`), and the Mojo gates the full chain (block.mojo:548-552 /
  599-603 / 645-649: `linear_backward(...W2)` → `linear_backward(...W1)`, accumulating
  d_t_silu), not merely the final shift/scale/gate. All 6 mod-weight grads + d_t_silu
  pass at cos ≥ 0.99999999.
- The interleaved-mutation teeth test (Attack 1) also drove `d_sa_mod1` to 0.988 FAIL,
  confirming the mod-chain grads are downstream-sensitive (real signal, not constants).

---

## Verdict

| Attack | Result | Severity |
|--------|--------|----------|
| 1 RoPE non-degeneracy | VERIFIED-CLEAN | HIGH |
| 2 RoPE convention (mirror bug) | VERIFIED-CLEAN | HIGH |
| 3 Oracle independence | VERIFIED-CLEAN | HIGH |
| 4 Cross-attn asymmetry/routing/d_context/no-mask | VERIFIED-CLEAN (1 LOW note) | HIGH |
| 5 AdaLN-LoRA-256 mod-grad independence | VERIFIED-CLEAN | HIGH |

This gate does NOT lie. It is materially stronger than the Ernie gate that fooled us:
its rope table is non-degenerate (proven), it uses the matched half-split fwd/bwd pair
that anima.rs actually uses (the full-width sibling exists and is correctly NOT used),
the oracle is autograd-derived (not a transcription of the hand-chained Mojo backward),
the cross-attn asymmetric rectangular path is genuinely exercised, and all mod-LoRA-256
chain grads are independent leaves tested end-to-end. Teeth confirmed by a deliberate
convention-flip that the gate catches.

## Prioritized fix list before Phase B (stack)

No correctness fixes required. Two non-blocking items to carry forward:

1. **LOW — text-padding mask.** The block (and the anima.rs reference) are maskless.
   Before stacking into a real trainer with variable-length captions, confirm whether
   Anima/Cosmos applies a text key-padding mask in cross-attn; if so, that is a
   trainer/stack-level addition (the per-block gate cannot surface it at S_txt=8
   no-padding). Verify against OT `AnimaModel.py` / diffusers `transformer_cosmos.py`
   cross-attn mask handling.

2. **LOW — gate coverage of frequency content.** The gate's rope table uses a single
   monotonic theta ramp, not the 3-axis 22/21/21 NTK split of `build_3d_rope_cossin`.
   This does not affect fwd/bwd math correctness (proven), but if Phase B wants the
   stacked-model gate to also validate the *table builder*, feed a `build_3d_rope_cossin`-
   shaped table into the block gate. Optional; the math arm is already covered.
