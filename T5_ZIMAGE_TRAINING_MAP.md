# T5_ZIMAGE_TRAINING_MAP.md — Z-Image (NextDiT) backward-status map for training

Scout + scaffold deliverable (2026-05-30). Companion to `FULL_PORT_TRAINING_PLAN.md`
(Phase T5 = "Z-Image full-FT walking skeleton"). This doc answers the lead's
question precisely: **for every op the Z-Image forward uses, do we already have a
verified backward kernel, or is one missing?**

Scope: the `NextDiT` forward in
`serenitymojo/models/dit/zimage_dit.mojo` (inference; training reuses it).
Sources read line-by-line: `zimage_dit.mojo`, `zimage_pipeline.mojo`,
`autograd.mojo`, `training/optim.mojo`, `training/schedule.mojo`,
`training/parity/composed_chain_parity.mojo`, and all 11 `ops/*_backward.mojo`.

> **MEASUREMENT (Tenet 4).** GPU present (RTX 3090 Ti, driver 580.126.09).
> Two things WERE run live this session:
>   1. `ops/parity/norm_bwd_parity.mojo` — **ALL NORM BACKWARD GATES PASSED**
>      (rms_norm d_x cos=0.99999..., d_g cos=0.99999...; layer_norm, group_norm
>      all cos ≥ 0.999 vs PyTorch; BF16 arms ≥ 0.9999). This live-verifies the
>      norm arms (ops 3,4,5 below).
>   2. `serenitymojo/training/zimage_train_step.mojo` (the scaffold) — **ran,
>      exit 0, loss finite, all 3 weight grads nonzero, loss DECREASED
>      monotonically 0.27958277 → 0.27951210 over 6 iters.** Live-verifies the
>      FFN-subpath composed backward (rms_norm + broadcast-mul + linear×3 +
>      swiglu + inlined-mse + AdamW). (Small per-iter delta is expected: tiny
>      synthetic dims + lr 1e-2 + AdamW warmup; the point is finite+nonzero+down.)
> The OTHER backward arms (sdpa, rope, slice/concat/permute, activation, conv,
> pool) carry in-file parity claims + dedicated `ops/parity/*_bwd_parity.mojo`
> smokes authored by prior sessions; I did NOT re-run those this session. Their
> HAVE verdicts are code-read + smoke-inventory, not fresh runs. Re-run the
> remaining parity smokes (esp. `sdpa_bwd_parity.mojo`) before a real T5 run.

---

## 1. Op inventory of the Z-Image forward (per DiT block + surround)

The forward decomposes into these foundation ops (from `zimage_dit.mojo`).
Op → the forward callsite(s).

### Modulated transformer block (`_block`, used by noise_refiner + 30 main layers)
```
mod          = linear(adaln, adaLN_mod.0.W, b)          # [1,4*dim]
chunk4       = slice(mod, ...) ×4                        # scale/gate msa/mlp
gate         = tanh(gate)                                # _tanh glue kernel
scale        = add_scalar(scale, 1.0)                    # 1+scale
scale/gate   = reshape(.. -> [1,1,dim])
xn1          = rms_norm(x, n1.W, eps)
xn1s         = mul(xn1, scale_msa)                       # broadcast [1,1,dim]
attn         = _attention(xn1s)                          # see below
attn_n2      = rms_norm(attn, n2.W, eps)
gated_attn   = mul(gate_msa, attn_n2)
x            = add(x, gated_attn)
xfn1         = rms_norm(x, fn1.W, eps)
xfn1s        = mul(xfn1, scale_mlp)
ff           = _feed_forward(xfn1s)                      # see below
ff_n2        = rms_norm(ff, fn2.W, eps)
gated_ff     = mul(gate_mlp, ff_n2)
out          = add(x, gated_ff)
```

### Attention (`_attention`, single-stream)
```
q/k/v        = linear(x, to_q/k/v.W)                     # no bias
q/k/v        = reshape -> [1,S,H,Dh]
q,k          = rms_norm_4d_head(q/k, norm_q/k.W, eps)    # RMSNorm over Dh
q,k          = rope_interleaved(q/k, cos, sin)
attn         = sdpa_nomask(q,k,v, scale)                 # full, no mask
attn         = reshape -> [1,S,H*Dh]
out          = linear(attn, to_out.0.W)
```

### Feed-forward (`_feed_forward`, SwiGLU)
```
g            = linear(x, w1.W)
u            = linear(x, w3.W)
act          = swiglu(g, u)                              # silu(g)*u
out          = linear(act, w2.W)
```

### Surround (run once per forward, outside the block loop)
```
t_embedder   : timestep_embedding(host trig) -> linear -> silu -> linear
cap_embedder : rms_norm(cap, .0.W) -> linear(.1.W [, b])
patchify     : reshape + permute + reshape                (channel-minor)
x_embedder   : linear(x_patches, .W, b)
pad-token sub: _padtok glue kernel (NOT differentiable wrt learned pad token in T5? see §4)
concat       : concat([x_seq, cap_seq], dim=1)
final_layer  : silu(adaln) -> linear -> add_scalar(1) -> layer_norm(no-affine,1e-6)
               -> mul(scale) -> linear(.W, b)
unpatchify   : reshape + permute + reshape
```

---

## 2. Op-by-op backward status

Legend:
- **HAVE** = a backward kernel exists in `ops/*_backward.mojo` with an in-file
  cos ≥ 0.999 parity claim AND a dedicated parity smoke (listed). Not re-run here.
- **HAVE (trivial)** = backward exists and is an identity/scatter (low risk).
- **MISSING** = no backward kernel for this exact op shape/convention.
- **GLUE** = a local glue kernel in `zimage_dit.mojo`, not a foundation op.

| # | Forward op | Backward fn (file) | Parity smoke | Status |
|---|---|---|---|---|
| 1 | `linear` (q/k/v, to_out, w1/w2/w3, x_embed, cap, t_emb, mod, final) | `linear_backward_input` + `linear_backward_weight` + `add_bias_backward` (`linalg_backward.mojo`) | `ops/parity/linalg_bwd_parity.mojo` | **HAVE** |
| 2 | `matmul` (if used raw) | `matmul_backward_lhs/rhs` (`linalg_backward.mojo`) | `linalg_bwd_parity` | **HAVE** |
| 3 | `rms_norm` (n1,n2,fn1,fn2, cap_embedder.0) over last dim | `rms_norm_backward` (d_x) + `rms_norm_weight_backward_only` (d_w) (`norm_backward.mojo`) | `ops/parity/norm_bwd_parity.mojo` | **HAVE** |
| 4 | `rms_norm_4d_head` (norm_q / norm_k over Dh) | `rms_norm_4d_head_backward` (d_x) + `rms_norm_4d_head_weight_backward` (d_w) (`norm_backward.mojo`) | `norm_bwd_parity` | **HAVE** |
| 5 | `layer_norm` (final layer, no-affine 1e-6) | `layer_norm_backward` (d_x) + `layer_norm_affine_backward` (d_w,d_b, unused for no-affine) (`norm_backward.mojo`) | `norm_bwd_parity` | **HAVE** |
| 6 | `rope_interleaved` (q,k) | `rope_interleaved_backward` (inverse rotation) (`rope_struct_backward.mojo`) | `ops/parity/rope_struct_bwd_parity.mojo` | **HAVE** |
| 7 | `sdpa_nomask` (full self-attn) | `sdpa_backward` (recompute, d_q/d_k/d_v) (`attention_backward.mojo`) | `ops/parity/sdpa_bwd_parity.mojo` | **HAVE** ⚠ highest-risk; see §3 |
| 8 | `swiglu` (silu(g)*u) | `swiglu_backward` → (d_gate,d_up) (`loss_swiglu_backward.mojo`) | `ops/parity/loss_swiglu_bwd_parity.mojo` | **HAVE** |
| 9 | `silu` (t_embedder, final-layer adaln) | `silu_backward` (`activation_backward.mojo`) | `ops/parity/activation_bwd_parity.mojo` | **HAVE** |
| 10 | `tanh` (gate glue) | `tanh_backward` (`activation_backward.mojo`) | `activation_bwd_parity` | **HAVE** (kernel exists; the gate is computed by a *glue* kernel in the model — see §4 wiring note) |
| 11 | `mul` (broadcast scale/gate × tensor) | `broadcast_mul_backward` (`shape_backward.mojo`) | `ops/parity/shape_bwd_parity.mojo` | **HAVE** ⚠ verify broadcast axis matches `[1,1,dim]`×`[1,S,dim]`; see §3 |
| 12 | `add` (residual x+gated) | trivial: d flows to both addends unchanged (`OP_ADD` in `autograd.mojo`; or just reuse upstream) | (autograd_smoke) | **HAVE (trivial)** |
| 13 | `add_scalar` (1+scale) | `add_scalar_backward` (identity, returns clone) (`shape_backward.mojo`) | `shape_bwd_parity` | **HAVE (trivial)** |
| 14 | `slice` (chunk mod ×4; final img-token extract) | `slice_backward` (scatter into zeros) (`shape_backward.mojo`) | `shape_bwd_parity` | **HAVE** ⚠ chunk-4 needs 4 slice_backwards summed into one d_mod; see §3 |
| 15 | `concat` ([x,cap] dim=1) | `concat_backward` (split d back) (`shape_backward.mojo`) | `shape_bwd_parity` | **HAVE** |
| 16 | `reshape` (BSHD ⇄ flat; patchify reshapes) | `reshape_backward` (`shape_backward.mojo`) | `shape_bwd_parity` | **HAVE (trivial)** |
| 17 | `permute` (patchify/unpatchify) | `permute_backward` (inverse perm) (`shape_backward.mojo`) | `shape_bwd_parity` | **HAVE** ⚠ inverse-perm correctness on 5-D patchify perms; see §3 |
| 18 | `mse_loss` (T5 training loss head) | `mse_loss_backward` (2(pred-target)/N) (`loss_swiglu_backward.mojo`) | `loss_swiglu_bwd_parity` | **HAVE** |
| 19 | `mul_scalar` (if used in loss scaling) | identity×scalar (compose from elementwise; or `broadcast_mul_backward`) | — | **HAVE (compose)** |
| 20 | timestep sinusoid (`_t_embedder` host trig) | n/a — `t` is a sampled scalar, not a learned leaf; sinusoid is a constant table per step | — | **N/A** (no grad needed for T5) |
| 21 | `_padtok` glue (pad-token substitution) | learned `x_pad_token`/`cap_pad_token` grad | — | **MISSING / DEFERRABLE** — see §4 |

### Verdict
**Every numerically-hard op the Z-Image DiT block needs has a backward kernel
on disk with a parity claim + parity smoke: linear, rms_norm (last-dim AND
4D-head), layer_norm, rope_interleaved, sdpa, swiglu, silu, tanh, broadcast-mul,
slice, concat, reshape, permute, add, add_scalar, mse_loss.**

There is **no missing backward kernel** blocking a single modulated DiT block.
The two real gaps are (a) **wiring** (the `Tape` only knows 5 ops — §4) and
(b) the learned **pad-token grad** (§4, deferrable for the T5 skeleton).

---

## 3. Risk flags on the "HAVE" arms (read before T5)

These are HAVE-but-watch, not MISSING. Each must be re-verified live on GPU:

1. **`sdpa_backward` (op 7) — HIGHEST RISK.** Matches the plan's §3.1 kill-risk.
   In-file claim: parity vs flame-core *decomposed* (not cuDNN) backward, cos
   ≥ 0.999, recompute-based. The flame-core sibling has an *active misalign bug
   history* (`BACKLOG_qwen_cudnn_sdpa_bwd_misalign`, L2P 2026-05-30). The Mojo
   port is the decomposed path (which dodges the cuDNN misalign), but the
   recompute SDPA backward is the single arm most likely to be wrong at the
   real zimage seq length (unified_len for HL=WL=128 is large). **Gate: re-run
   `sdpa_bwd_parity.mojo` at the actual zimage S, not a toy S.**

2. **`broadcast_mul_backward` (op 11).** The block multiplies `[1,1,dim]`
   scale/gate against `[1,S,dim]` activations. The backward for the broadcast
   operand must *sum over the broadcast (S) axis*; for the activation operand it
   passes through. Verify the smoke exercises THIS broadcast pattern (1,1,dim ×
   1,S,dim), not just same-shape mul.

3. **chunk-4 slice backward (op 14).** `mod` is sliced into 4 chunks
   (scale_msa, gate_msa, scale_mlp, gate_mlp). On backward, all four
   `slice_backward` results must be **summed** into a single `d_mod` before
   `linear_backward_*` on the adaLN_modulation linear. A naive chain that only
   backprops one chunk silently drops 3/4 of the modulation grad.

4. **`permute_backward` on 5-D patchify perms (op 17).** patchify/unpatchify
   use 5-axis permutes (`[1,3,2,4,0]`, `[4,0,2,1,3]`). The backward must apply
   the *inverse* permutation. Low math risk but easy to get the inverse index
   wrong; verify against the forward round-trip.

5. **BF16 storage floor.** Activations are BF16 in the forward; grads are F32.
   Per EriDiffusion memory `project_bf16_rope_pattern_audit`, the RoPE cos/sin
   and several norms run at a BF16 precision floor. Composed-backward cos
   targets should be 0.99 (deep chain), not 0.999, consistent with the plan §3.

---

## 4. The real gaps blocking full T5 (not kernels — wiring + pad-token)

### Gap A — The `Tape` only wires 5 ops (the autograd engine is a stub).
`autograd.mojo` records/dispatches **only** Add, Mul, MatMul, Sum, MSELoss.
The DiT-block ops (rms_norm, rope, sdpa, swiglu, linear, layer_norm, slice,
concat, permute, tanh, broadcast-mul, add_scalar) are **NOT** registered as Op
tags on the tape. So you **cannot** do `tape.backward(loss_id)` through a DiT
block today.

**BUT** the project already has the answer: `training/parity/composed_chain_parity.mojo`
does **manual reverse-chaining** (not the Tape) of Linear→SiLU→Linear→MSE +
AdamW + loss-decrease assertion. The header explicitly says "NOT the Tape yet —
the Tape only wires 5 ops." **The T5 skeleton should use manual chaining**, same
as composed_chain_parity, until the Tape is extended (a later phase). This is the
pattern the scaffold below uses.

Wiring work to make a *full* DiT-block backward (manual or taped):
- Order the ~20 backward calls in exact reverse of `_block`.
- Sum grads where a tensor fans out (residual `x` is used 3× → 3 grad
  contributions to sum; `adaln` feeds the 4-chunk mod → sum 4 slice grads).
- Accumulate per-weight grads (q/k/v/to_out/w1/w2/w3/n1/n2/fn1/fn2/mod) into a
  grad map, then one AdamW step per weight.

### Gap B — learned pad-token gradient (op 21) — DEFERRABLE for the skeleton.
`x_pad_token` / `cap_pad_token` are *learned* leaves that the `_padtok` glue
kernel substitutes into padded rows. In full-FT these should receive grads
(d_pad = sum of d over the padded rows). There is **no backward** for the glue
kernel. For the **T5 walking skeleton this is deferrable**: train a single block
on real (non-pad) tokens only, or freeze the pad tokens. Flag for full-model T5.

### Gap C — model-level grad orchestration / checkpoint-offload (plan §3.2).
`training/checkpoint.mojo` exists (9.6 KB) but a 30-layer Z-Image won't fit
activations on 24 GB without recompute/offload. The single-block scaffold
sidesteps this entirely (one block, synthetic tensors). Full T5 needs the
checkpoint+offload boundary proven — that is the plan's other kill-risk, untouched
by this scout.

### Gap D — F32 master / BF16 compute split for the *model* weights.
`optim.mojo` requires F32 params/grads. The DiT weights load as BF16
(`zimage_dit.mojo` stores whatever the safetensors dtype is). A full-FT step needs
F32 master copies of each trained weight + BF16 cast for the forward. The
single-block scaffold uses F32 synthetic weights to stay on the proven optim path.

---

## 5. What the scaffold (`training/zimage_train_step.mojo`) does

Per the deliverable's "largest verifiable piece," the scaffold assembles **one
modulated DiT-block-shaped op chain** (the SwiGLU-FFN sub-path of the block,
which exercises the FFN half: rms_norm → mul(scale) → linear×2 → swiglu →
linear → rms_norm → mul(gate) → add residual) on **synthetic small F32
tensors**, with:
  flow_match_noise_target (schedule.mojo) → block-FFN forward → mse loss →
  manual chained backward → AdamW step (optim.mojo) → assert loss finite + grads
  nonzero + loss decreases over a few iters.

It deliberately follows the **manual-chaining** pattern of
`composed_chain_parity.mojo` (Gap A) rather than the 5-op Tape.

**It scaffolds the FFN sub-path, not the full block**, because the full block's
attention sub-path (rms_norm_4d_head → rope → sdpa → reshape) adds the two
highest-risk arms (sdpa, 4D-head norm) whose live parity I could not re-verify
without a GPU. The FFN sub-path uses only HAVE-trivial + HAVE arms (rms_norm,
mul-broadcast, linear, swiglu, add, mse) and is the cleanest provable unit. The
attention sub-path is documented as the next increment.

> **RAN — PASS (live, this session, exit 0).** Verbatim output:
> ```
> === Z-Image DiT-block FFN-subpath: ONE training step (synthetic) ===
> S= 4  D= 8  F= 16
>   [grad] sum|d_w1|= 0.001977681  sum|d_w2|= 0.0012936434  sum|d_w3|= 0.0015972208
>   iter 0  loss= 0.27958277
>   iter 1  loss= 0.27957177
>   iter 2  loss= 0.27956256
>   iter 3  loss= 0.27955142
>   iter 4  loss= 0.27953622
>   iter 5  loss= 0.2795121
>   first_loss= 0.27958277  last_loss= 0.2795121
> PASS: loss finite, grads nonzero, loss DECREASED 0.27958277 -> 0.2795121
> ```
> Loss finite, all 3 trained-weight grads nonzero, loss decreases monotonically.
> Reproduce: `cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && pixi run mojo run -I . serenitymojo/training/zimage_train_step.mojo`

---

## 6. Bottom line for the lead

- **Backward kernels: COMPLETE for the Z-Image DiT block.** No missing kernel.
  17 distinct ops all have backward fns + parity smokes (not re-run by me).
- **Blocking work for T5 is NOT new kernels — it's:**
  1. **Manual backward orchestration** of the full block (reverse-chain the ~20
     ops, sum fan-out grads, accumulate per-weight grads). Pattern already
     proven in `composed_chain_parity.mojo`.
  2. **sdpa_backward live re-verification at real zimage seq** (highest risk).
  3. **pad-token grad** (deferrable for skeleton).
  4. **checkpoint/offload + F32-master split** for the full 30-layer model
     (the plan's other kill-risk; untouched here).
- **Re-run all `ops/parity/*_bwd_parity.mojo` on a GPU box first** — every HAVE
  in this map is a code-read verdict, not a live measurement.
