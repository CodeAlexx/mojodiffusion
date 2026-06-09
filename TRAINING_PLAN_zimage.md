# Z-Image (NextDiT) — pure-Mojo training port plan

Status: **Phase 1 (block) + Phase 2 (refiners + full-stack composition) + Phase 3
(parity-gated LoRA training step) ALL GREEN — Z-Image per-model MILESTONE CLOSED,
MEASURED 2026-06-01.**
Scope of this doc: the training port living in `serenitymojo/models/zimage/`.
Forward parity oracle: `serenitymojo/models/dit/zimage_dit.mojo` (matches diffusers
`transformer_z_image.py`). Backward arms reused from `serenitymojo/ops/` (Tenet 1 —
no new primitives added). Reference Rust: `inference-flame/src/models/zimage_nextdit.rs`.

---

## 1. Architecture (base Z-Image, NOT the L2P variant)

From `zimage_nextdit.rs` `NextDiTConfig::default()` + `zimage_dit.mojo`:

| field | value |
|---|---|
| dim (hidden) | 3840 (= 30 heads × 128 head_dim) |
| num_heads / head_dim | 30 / 128 |
| main layers | 30 |
| noise refiners / context refiners | 2 / 2 |
| cap_feat_dim (text, Qwen3-4B) | 2560 |
| mlp_hidden (SwiGLU per-gate) | 10240 |
| patch_size | 2 (patchify 2×2 → Linear(64, 3840)) |
| t_embedder_hidden / min_mod | 1024 / 256 |
| rope axes / theta | [32, 48, 48] / 256 |
| block norm eps | **1e-5** (block RMSNorm + qk RMSNorm) |
| final-layer LayerNorm eps | 1e-6 |
| time_scale / pad multiple | 1000 / 32 |
| model output | negated velocity (`-img`) |

### Main-layer block (the Phase-1 unit) — sandwich-norm, adaLN-tanh, SwiGLU
```
mod = adaLN_modulation.0(t_emb) -> chunk4: scale_msa, gate_msa, scale_mlp, gate_mlp
scale = 1 + raw_scale ;  gate = tanh(raw_gate)         # block owns these grads
# attention sub-block (sandwich norm)
xn1   = rms_norm(x, attention_norm1, eps)
xn1s  = (1 + scale_msa) * xn1                          # scale modulation, NO shift
q,k,v = to_q/to_k/to_v(xn1s) -> [1,S,H,Dh]
q,k   = rms_norm(q/k, norm_q/norm_k, eps)              # per-head QK RMSNorm
q,k   = rope_interleaved(q/k, cos, sin)                # INTERLEAVED, half-width table
att   = sdpa(q,k,v, 1/sqrt(Dh))
att_o = to_out.0(att)
h     = x + gate_msa * rms_norm(att_o, attention_norm2, eps)
# MLP sub-block (SwiGLU, sandwich norm)
xfn1  = rms_norm(h, ffn_norm1, eps)
xfn1s = (1 + scale_mlp) * xfn1
ff    = w2( silu(w1(xfn1s)) * w3(xfn1s) )              # SwiGLU
out   = h + gate_mlp * rms_norm(ff, ffn_norm2, eps)
```

### RoPE convention — INTERLEAVED (corrects the build-prompt's "half-split" claim)
The build prompt stated Z-Image shares Ernie's **half-split** RoPE. **That is wrong
for the base Z-Image.** Evidence (Tenet 4, measured at source):
- diffusers `transformer_z_image.py:114-119` `apply_rotary_emb`:
  `view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))` → pairs **adjacent**
  elements `(x[2i], x[2i+1])` into complex, multiplies by `freqs_cis` →
  `view_as_real(...).flatten` → this is **interleaved** RoPE, NOT half-split.
- Mojo forward oracle `zimage_dit.mojo:35-38, 380-381` uses `rope_interleaved`
  (header: "Applied as INTERLEAVED RoPE … pair (x[2i],x[2i+1])").
So the backward arm is `rope_backward(..., interleaved=True)` on a **half-width**
`[S*H, Dh/2]` cos/sin table (one angle per pair). Ernie's `rope_halfsplit_full*`
machinery is **not** used here. (Half-split is the Ernie family; Z-Image is the
FLUX/Klein interleaved family. The shared lever was the *backward arm existing*,
not the convention.)

Also corrects the stale `models/zimage/config.mojo` (was hardcoded dim/eps with
`mlp_hidden=2560` placeholder and `eps=1e-5` mislabeled): now JSON-driven from
`configs/zimage.json` with `mlp_hidden=10240`, block eps `1e-5`, final eps `1e-6`.

---

## 2. Ernie-reuse assessment

| Component | Ernie | Z-Image | Reuse verdict |
|---|---|---|---|
| Block forward style (save acts → hand-chain bwd) | ✅ | ✅ | Pattern copied verbatim |
| Weight loader (device-resident TArc, [out,in] linear) | ✅ | ✅ | `weights.mojo` mirrors ernie |
| Config (JSON-driven `read_model_config`) | ✅ | ✅ | `config.mojo` mirrors ernie |
| Parity oracle + gate skeleton | ✅ | ✅ | `parity/` mirrors ernie |
| RMSNorm fwd/bwd (`rms_norm`, `rms_norm_backward`) | ✅ | ✅ | reused as-is |
| Linear fwd/bwd (`linear`, `linear_backward`) | ✅ | ✅ | reused as-is |
| Separate q/k/v projections (3× linear, sum d_x) | ✅ | ✅ | reused (oracle uses to_q/k/v, like ernie) |
| SDPA fwd/bwd (`sdpa_nomask`, `sdpa_backward`) | ✅ | ✅ | reused as-is |
| `gate_residual_backward` | ✅ | ✅ | reused; Z-Image gates norm2(out) not raw out |
| `modulate_backward` | ✅ (scale+shift) | ✅ (scale only, shift=0) | reused; feed zeros shift, drop d_shift |
| RoPE backward | `rope_halfsplit_full_backward` | `rope_backward(interleaved=True)` | **different arm**, both pre-built+gated |
| MLP activation | GELU-gated (`gelu`/`gelu_backward`) | **SwiGLU** (`swiglu`/`swiglu_backward`) | different arm, both pre-built+gated |
| Gate nonlinearity | none (shared AdaLN passed raw) | **tanh** (`tanh_op`/`tanh_backward`) | extra arm, pre-built+gated |
| Modulation source | shared across blocks | per-block adaLN_modulation.0 | structural note (Phase 2) |

Net: the *scaffold* (forward-save / backward-chain / weights / config / parity)
is a near-exact Ernie copy. The *math arms* differ in 4 places (rope convention,
swiglu vs gelu, tanh gates, scale-only modulation), all satisfied by existing
gated `ops/` primitives.

---

## 3. SwiGLU-backward verdict

**EXISTS and is the correct MLP arm — no ops/ gap.** `ops/loss_swiglu_backward.mojo`
contains `swiglu_backward(grad_out, gate, up) -> SwigluGrads{d_gate, d_up}` for
`y = silu(gate) * up` (`d_up = grad_out*silu(gate)`, `d_gate = grad_out*up*silu'(gate)`),
ported verbatim from flame-core `kernels/swiglu_backward.cu` and parity-gated
(`ops/parity/loss_swiglu_bwd_parity.mojo`). The file name is misleading — it bundles
MSE/Huber loss backward AND the FFN SwiGLU backward; the SwiGLU half is exactly the
MLP activation backward Z-Image needs. Confirmed against the Z-Image forward
`feed_forward = w2(silu(w1(x)) * w3(x))` (`zimage_dit.mojo:396-403`, `swiglu(g,u)=silu(g)*u`).

---

## 4. ops/ gap report

**NONE.** Every backward arm the Z-Image main block needs already exists in
`serenitymojo/ops/` and is gated:
`rms_norm_backward`, `linear_backward`, `sdpa_backward`, `swiglu_backward`,
`tanh_backward`, `modulate_backward`, `gate_residual_backward`,
`rope_backward(interleaved=True)`. No edits to `ops/` were made (Tenet 1 + the
concurrent-agent constraint respected).

---

## 5. Phase-1 result (MEASURED 2026-06-01)

Oracle: `serenitymojo/models/zimage/parity/block_oracle.py` (independent torch
F64 autograd from the diffusers/Rust math, NOT from Mojo). Gate:
`serenitymojo/models/zimage/parity/block_parity.mojo`. REAL dims D=3840 / H=30 /
Dh=128, S=8 (2 caption + 2×3 image tokens), F=96 (small FFN to bound the 3090),
non-degenerate 3-axis interleaved RoPE (cos std 0.32). Build clean (no errors),
fresh oracle-regen + build + run:

```
==== zimage block_parity (Z-Image main-layer block vs torch) ====
H= 30  Dh= 128  D= 3840  S= 8  F= 96
---- forward output vs torch ----
  cos( out ) = 0.99999999999  max_abs = 8.2e-06  n = 30720   PASS
---- input grad vs torch ----
  cos( d_x ) = 0.99999999999  max_abs = 9.3e-07  n = 30720   PASS
---- trainable weight grads vs torch ----
  d_n1, d_wq, d_wk, d_wv, d_wo, d_q_norm, d_k_norm, d_n2, d_fn1,
  d_w1, d_w3, d_w2, d_fn2  -> all cos >= 0.99999999  PASS
---- RAW adaLN modulation-vector grads vs torch ----
  d_scale_msa, d_gate_msa, d_scale_mlp, d_gate_mlp -> all cos >= 0.99999999  PASS
VERDICT: PASS — Z-Image main-layer block fwd+bwd matches torch (cos>=0.999)
```

19/19 arms PASS, 0 nonfinite. Files delivered:
- `serenitymojo/configs/zimage.json` (single source of arch+recipe)
- `serenitymojo/models/zimage/config.mojo` (JSON-driven, mirrors ernie)
- `serenitymojo/models/zimage/weights.mojo` (`ZImageBlockWeights` loader + shape verifier)
- `serenitymojo/models/zimage/block.mojo` (`zimage_block_forward` + `zimage_block_backward`)
- `serenitymojo/models/zimage/parity/block_oracle.py` (torch F64 oracle)
- `serenitymojo/models/zimage/parity/block_parity.mojo` (gate, output above)

---

## 6b. Phase 2 result — refiners + full-stack composition (MEASURED 2026-06-01)

### Refiner-layer verification (refiner structure cited at source)
The Z-Image refiners are NOT a new block kind — they are the SAME sandwich-norm
attention+SwiGLU block, in two conditioning modes (verified at
`zimage_nextdit.rs:591-609` + `zimage_dit.mojo:_block`):
- **noise_refiner** (2 blocks, image tokens): `transformer_block(x, rope, Some(t_cond))`
  → `has_adaln == true` branch → IDENTICAL to a main layer. Reuses
  `zimage_block_forward/_backward` verbatim (just a different weight prefix). NO
  new code.
- **context_refiner** (2 blocks, text tokens): `transformer_block(c, rope, None)`
  → `has_adaln == false` branch (`zimage_nextdit.rs:349-355, 365-370, 377-383,
  393-398`) → UNMODULATED: norm1 NOT scaled, residual is a PLAIN add (no
  tanh-gate), norm1-ffn NOT scaled, ffn residual PLAIN add. New
  `zimage_refiner_forward/_backward` in `block.mojo` — same arms minus
  modulate/tanh/gate_residual; the residual becomes `x + norm2(...)` so backward
  routes the seed grad to BOTH the residual and the norm2 branch (gate=1).

Refiner gate (`parity/refiner_parity.mojo`, oracle `parity/refiner_oracle.py`,
REAL D=3840/H=30/Dh=128, S=5, interleaved RoPE cos std 0.30):
```
==== zimage refiner_parity (UNMODULATED context-refiner vs torch) ====
  cos( out ) = 0.99999999999  max_abs = 2.6e-06   PASS
  cos( d_x ) = 0.99999999999  max_abs = 8.4e-07   PASS
  d_n1,d_wq,d_wk,d_wv,d_wo,d_q_norm,d_k_norm,d_n2,d_fn1,d_w1,d_w3,d_w2,d_fn2
    -> all cos >= 0.99999999  PASS
VERDICT: PASS — Z-Image unmodulated refiner fwd+bwd matches torch (cos>=0.999)
```
15/15 arms PASS.

### Full-stack composition (`models/zimage/zimage_stack.mojo`)
`zimage_stack_forward`/`_backward` compose: noise refiners (modulated) on x_seq +
context refiners (unmodulated) on cap_seq → concat **[x, cap]** (matches
`zimage_dit.mojo:730`, NOT the .rs `[cap, image]` — equivalent under unmasked SDPA,
and the Mojo order is the VERIFIED forward) → main layers (modulated) → final layer
(LayerNorm-no-affine 1e-6 → scale-ONLY modulate `(1+f_scale)*ln`, no shift → Linear)
→ image rows. Mirrors `ernie/ernie_stack.mojo` (per-block recompute backward,
host-List stream carrier, reverse d_x→d_y handoff). Z-Image differs from Ernie:
TWO streams through the refiners then concat (backward splits at the seam and
routes each half back through its stream); PER-BLOCK modulation (each block's RAW
mod-vec grads returned separately, NO cross-block summation — unlike Ernie's
shared AdaLN); final modulation is scale-only.

SCOPE (mirrors Ernie): embedders / t_embedder / per-block adaLN_modulation.0 MLP /
final adaLN MLP backprop links are the TRAIN-LOOP phase (deferred). The stack is
passed post-embedder tokens + per-block RAW mod vectors + final f_scale PRECOMPUTED,
and RETURNS the grads into them (the embedder-output token grads = full-chain proof;
the per-block mod-vec grads ready for one linear_backward into each adaLN at step end).

Composition gate (`parity/stack_parity.mojo`, oracle `parity/stack_oracle.py`,
reduced depth 1 noise + 1 context + 2 main, REAL D=3840/H=30/Dh=128, S=10):
```
==== zimage stack_parity (Z-Image FULL STACK composition vs torch) ====
  cos( out )        = 0.99999999999  PASS
  cos( d_x_seq )    = 0.99999999998  PASS   (full-chain into noise-refiner input)
  cos( d_cap_seq )  = 0.99999999999  PASS   (full-chain into context-refiner input)
  cos( d_f_scale )  = 0.99999999999  PASS
  cos( d_final_lin )= 0.99999999999  PASS
  d_main_deep_wq/w2, d_main_shallow_wq -> all cos >= 0.99999999  PASS
  d_nr0_wq/w2 (noise refiner), d_cr0_wq/w2 (context refiner) -> all PASS
  d_nr0_mod[4D], d_main_deep_mod[4D] -> all cos >= 0.99999999  PASS
VERDICT: PASS — Z-Image full-stack composition fwd+bwd matches torch (cos>=0.999)
```
16/16 arms PASS.

### #1 RISK GATE — finite-difference self-consistency (NO torch)
`parity/stack_finitediff.mojo`: central-difference numerical grad from the Mojo
stack's OWN forward vs the analytic backward (the Klein composition-defect check —
per-block-correct does NOT prove composed-bwd-correct, and a torch oracle can mask
a shared-assumption bug). VECTOR-relative metric `||ana-num||/||ana||` over 7
x_seq probes spread across rows/channels (per-element ratio is unreliable where a
single component's gradient is near zero — F32 storage + truncation dominate there):
```
==== zimage stack_finitediff (x-path self-consistency, NO torch) ====
  worst |ana-num| = 1.4e-04  (gradients are ~1e-2)
  VECTOR-relative err  ||ana-num|| / ||ana|| = 0.0053
VERDICT: PASS — composed backward = grad of composed forward (ratio≈1.0)
```
0.53% — at the F32 truncation floor, far below the ~0.33 Klein composition-defect
signature. The composed backward IS the gradient of the composed forward.

### Files delivered (Phase 2)
- `models/zimage/block.mojo` (EXTENDED): `zimage_refiner_forward`/`_backward`
  (+ `ZImageRefinerSaved`/`ZImageRefinerForward`/`ZImageRefinerGrads`) — the
  UNMODULATED context-refiner path. Main block (Phase 1) UNCHANGED + re-verified.
- `models/zimage/zimage_stack.mojo` (NEW): `zimage_stack_forward`/`_backward`
  (+ `ZImageStackForward`/`ZImageStackGrads`).
- `models/zimage/parity/refiner_oracle.py` + `refiner_parity.mojo` (refiner gate).
- `models/zimage/parity/stack_oracle.py` + `stack_parity.mojo` (composition gate).
- `models/zimage/parity/stack_finitediff.mojo` (#1 risk gate).

### Residency verdict (real config: 30 main + 2 noise + 2 context = 34 blocks)
Per-block F32 weights ≈ 177M params → ~708 MB/block. 34 blocks F32-resident ≈
**24.1 GB weights + ~5.3 GB adaLN ≈ 29 GB — does NOT fit a 24 GB 3090.** The
per-block-recompute backward already bounds the ACTIVATION footprint to ~one
block, but the WEIGHT residency is the constraint. Phase 3 needs either the
STREAMED/block-offload weight path (the Ernie `*_streamed` variant pattern) or
BF16 base residency (~12 GB weights + 2.7 GB adaLN ≈ 15 GB, headroom for LoRA
activations). BF16 base matches `feedback_zimage_base_not_turbo`. NO streamed
variant built here (Phase 3 territory).

NO ops/ gap (Tenet 1): every backward arm already existed and is gated. NO edits
to `ops/` or other shared dirs.

---

## 6. Phase 3 — LoRA training step — **GREEN + VERIFIED 2026-06-01 (MILESTONE CLOSED)**

The parity-gated LoRA training step is wired and proven. Z-Image per-model
milestone is **CLOSED** (Phase 1 block + Phase 2 stack + Phase 3 LoRA all green).

### What shipped (models/zimage/, parity/ only — no shared dirs touched)
- `models/zimage/lora_block.mojo` (NEW): LoRA-on-projection wrappers for BOTH block
  flavors — `zimage_block_lora_forward/_backward` (MODULATED: noise refiners + main
  layers) and `zimage_refiner_lora_forward/_backward` (UNMODULATED: context
  refiners). 7 optional adapters per block (slots Q,K,V,O,w1,w3,w2). Sums LoRA d_x
  into the projection-input grad; reduces bit-for-bit to base when adapters absent.
  REUSES `LoraAdapter` + `linear`/`linear_backward` by CALLING (no ops/ edit).
- `models/zimage/zimage_stack_lora.mojo` (NEW): `ZImageLoraSet` (flat carrier, 3
  segments nr|cr|main × 7 slots) + `build_zimage_lora_set` + resident
  `zimage_stack_lora_forward/_backward` + per-block d_A/d_B scatter +
  `zimage_lora_adamw_step` (reuses `_lora_adamw`) +
  `save_zimage_lora`/`load_zimage_lora_resume` (reuse `save_lora_peft`/
  `load_lora_for_resume`). Per-block RAW mod-vec grads returned per-block (NOT
  summed — Z-Image mod is per-block).
- `models/zimage/parity/lora_stack_oracle.py` + `lora_stack_parity.mojo` +
  `lora_step_smoke.mojo` (NEW gates).

### OneTrainer LoRA target map (source-fidelity, cited)
- `OneTrainer/modules/modelSetup/ZImageLoRASetup.py:57` →
  `LoRAModuleWrapper(model.transformer, "transformer", config,
  config.layer_filter.split(","))`. With the default empty filter,
  `OneTrainer/modules/module/LoRAModule.py:638-656` adapts EVERY `nn.Linear` child
  of the transformer.
- diffusers `ZImageTransformerBlock` (`transformer_z_image.py:184-224`): each block
  exposes `attention` (diffusers `Attention` → `to_q/to_k/to_v/to_out.0`) +
  `feed_forward` (`w1/w3/w2`, lines 172-174). → 7 LoRA targets per block, across
  `noise_refiner.<i>`, `context_refiner.<i>`, `layers.<i>` (key paths confirmed in
  `inference-flame/src/models/zimage_nextdit.rs:593-619` + `zimage/weights.mojo`).
- **rank/alpha defaults**: `TrainConfig.py:1143-1144` → `lora_rank=16`,
  `lora_alpha=1.0` (scale = alpha/rank). The parity gate uses rank=8/alpha=16 (live
  perturbation for non-degenerate grads, mirroring the Ernie gate); the trainer
  recipe uses OT defaults from `configs/zimage.json`.
- **save key naming**: `ZImageLoRASaver.py:24-25` saves `transformer_lora
  .state_dict()` (kohya `lora_down/lora_up` under `transformer.<module>`). We REUSE
  `save_lora_peft` (PEFT `.lora_A.weight`/`.lora_B.weight`, the train_klein /
  lora.mojo inference convention) keyed by the diffusers module path
  (`noise_refiner.<i>.attention.to_q` etc.) — the Ernie precedent (omit OT's
  `transformer.` wrapper prefix; lora.mojo detects DiffusionModel by the
  `.lora_A.weight` suffix). A = lora_down [rank,in], B = lora_up [out,rank].

### MEASURED gate results (Tenet 4)
- Base stack gate: `VERDICT: PASS — Z-Image full-stack composition fwd+bwd matches
  torch (cos>=0.999)`.
- LoRA composition gate (`lora_stack_parity`): forward cos 0.99999999999, both
  token grads + d_f_scale + all 3 per-block mod grads + **all 56 A/B grads (28
  adapters × dA,dB)** cos ≥ 0.99999999, `nonfinite_lora_grads = 0` →
  `VERDICT: PASS`.
- Step smoke (`lora_step_smoke`): B 0→nonzero, **28/28 nonzero ratio = 1.0**,
  global-norm clip 1.343→1.0, `nonfinite = 0`, save/load `max_abs_diff = 0.0
  BYTE-EXACT` → `VERDICT: PASS`. (`|dA|_1 = 0` at step 0 is correct PEFT behavior:
  with B=0, d_t = d_dy@B = 0 ⇒ d_A = 0 on the first step; B's grad path is live.)

### Post-milestone (NOT this gate)
The streamed/BF16-resident real-depth train loop (34 blocks, 24 GB) is the
follow-on — the composition gate uses reduced depth (NR=1/CR=1/MAIN=2, real
dim/H/Dh) per the residency verdict above. The Ernie `*_streamed` pattern is the
template; `zimage/weights.mojo` already loads real block weights by key.

### Notes / watch items
- Modulation source is PER-BLOCK (`layers.{i}.adaLN_modulation.0`), unlike Ernie's
  shared AdaLN. Phase 2 wires each block's mod from the same t_emb but distinct
  weights; the block already returns RAW d_scale/d_gate ready to backprop into
  each block's `adaLN_modulation.0` Linear.
- L2P (`zimage_l2p_*`) is a SEPARATE Z-Image variant — explicitly out of scope.
- BF16 RoPE precision floor (memory `project_bf16_rope_pattern_audit`): the F32
  parity interior is the contract; the real bf16 run will sit at the bf16 floor.
```

---

## UPDATE 2026-06-09 — block LoRA-grad gate + oracle fidelity + overfit convergence (MEASURED)

Three new gates run live this session, raising Z-Image to the Klein "TRAINS" bar
**and one rung higher** (oracle fidelity to the real diffusers block now
independently confirmed, which Klein's campaign never did):

1. **Per-block LoRA d_A/d_B gate** — `models/zimage/parity/zimage_block_lora_parity.mojo`
   (+ oracle `zimage_block_lora_oracle.py`). NextDiT main block + all 7 LoRA slots
   (to_q/k/v/out + SwiGLU w1/w3/w2), rank 8 scale 2.0. Forward cos 0.99999, d_x
   0.99999, trainable norm-scale + adaLN-mod grads ≥0.99999, **14 LoRA d_A/d_B all
   cos ≥ 0.99997** (bf16 adapters). Frozen base projection grads are empty *by
   design* (LoRA freezes base weights) — the gate checks only the still-trainable
   norm/adaLN params + the adapters. → `VERDICT: PASS`.

2. **Oracle fidelity vs REAL diffusers** — `models/zimage/parity/zimage_block_vs_diffusers.py`.
   Drives the actual diffusers `ZImageTransformerBlock(modulation=True)` — the exact
   class OneTrainer `ZImageModel` instantiates — at real dims (D3840/H30/Dh128/
   F10240) in float64 and matches the oracle's hand math at **cos = 1.0
   (max_abs ~1e-7)** on forward + d_x + all 15 weight grads (incl qk-norm/rope path
   + adaLN modulation). LoRA delta == OneTrainer `LoRAModule.forward` (L328-329:
   `orig + up(down(x))·alpha/rank`, A=[r,in] B=[out,r]) — code-identical. So the
   Mojo→oracle→diffusers/OneTrainer chain is closed for block math; the per-block
   torch oracle is now *measured*-authoritative, not asserted.

3. **Multi-block overfit convergence** — `models/zimage/parity/zimage_overfit_convergence.mojo`.
   Real stack (1 noise-refiner + 1 context-refiner + 2 main + final, S=10) through
   fwd→MSE→bwd→clip→AdamW, 50 steps, one fixed batch. MSE driven **0.0444 → 4.7e-5
   (945× reduction)**, no divergence (max@step≥3 = 0.007 ≪ start). → bf16 LoRA grads
   ARE a correct descent direction. (Strict per-step monotonicity NOT required —
   sub-1e-4 jitter is the bf16 noise floor; criterion is running-min ≥20× drop +
   no divergence.)

**Still unproven** (same caveat as the whole campaign): full-scale training at real
dims (F=10240, all 30+4 blocks, real data/VAE/text-encoder wiring) and end-to-end
image quality. The gates above use reduced dims (F=96, 4 blocks).
