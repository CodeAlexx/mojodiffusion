# SKEPTIC FINDINGS — Anima training-port Phase 1 (2026-06-01)

Adversarial verification of the builder's Phase-1 artifacts under
`serenitymojo/models/anima/`. Rule applied: no claim accepted without a tool
result reproduced this session. "finite but not parity-gated" = UNVERIFIED.

Artifacts reviewed: `config.mojo`, `weights.mojo`, `weights_verify.mojo`,
`block.mojo`, `block_smoke.mojo`, `__init__.mojo`, `configs/anima.json`,
`/home/alex/mojodiffusion/TRAINING_PLAN_anima.md`.

Sources cross-checked: flame-core `bf16_ops.rs`, inference `anima.rs`,
inference `anima_dit.mojo`, OT `BaseAnimaSetup.py` / `AnimaLoRASetup.py` /
`#anima LoRA.json`, diffusers `transformer_cosmos.py`, mojo `ops/attention*.mojo`,
`ops/rope.mojo`, `ops/linear.mojo`, and the real checkpoint header.

---

## CLAIM 1 — AdaLN-pre is LayerNorm-no-affine, NOT RMSNorm → **VERIFIED** (HIGH)

Evidence (three independent sources agree):

- flame-core `bf16_ops.rs:1665-1673` (the needs_grad / training path of
  `modulate_pre_fused_bf16`): `let normed = crate::layer_norm::layer_norm(x,
  &[dim], None, None, eps)?;` then `(1+scale)*normed + shift`. `None, None`
  = LayerNorm with **no affine**. This is the path training takes.
- inference `anima.rs:336-344` `apply_adaln`: doc "(1 + scale) * LayerNorm(x)
  + shift", body calls `modulate_pre_fused_bf16`.
- diffusers `transformer_cosmos.py:118` `CosmosAdaLayerNormZero.norm =
  nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)`. norm1/norm2/
  norm3 (the AdaLN-pre on x for self/cross/ff) ALL use this. CANONICAL.
- The inference Mojo kernel `_adaln_modulate_kernel_bf16`
  (`anima_dit.mojo:181-248`) BODY computes mean (pass 1, lines 202-217),
  variance from `(v-mean)` (pass 2, 220-236), normalizes `(v-mean)*inv`
  (line 243) = **LayerNorm**, despite the `_apply_adaln_modulate` docstring
  (line 257) lying "RMSNorm". The builder's read of the kernel is correct;
  the inference docstring is the liar.

QK-norm is the separate one that IS RMSNorm: `transformer_cosmos.py:323-324`
`q_img_norm/k_img_norm = RMSNorm(..., eps=1e-6, elementwise_affine=True)`;
`anima.rs:209-225` `rms_norm_per_head`. And `t_cond` IS RMSNorm
(`anima.rs:280`). The builder correctly split these: AdaLN-pre = LayerNorm,
QK + t_cond = RMSNorm.

Builder's `block.mojo::_adaln_pre` (lines 69-97) uses `layer_norm(x_f32, ones,
zeros, eps)` = LayerNorm-no-affine. **CONSISTENT with the true answer.**
`_rms_per_head` (186-193) uses `rms_norm` for QK. Correct.

Impact: this gates all parity. The builder got it right. No fix needed here.

---

## CLAIM 2 — Weight loader reads 20 tensors w/ correct block-0 shapes → **VERIFIED** (HIGH)

Reproduced this session:
- Built `weights_verify.mojo` clean (BUILD_RC=0).
- Ran `/tmp/anima_weights_verify` against the REAL checkpoint
  `/home/alex/.serenity/.../anima-base-v1.0.safetensors` (RUN_RC=0):
  **"ALL 20 block-0 shapes verified."** Every one of the 20 printed OK.
- Independently dumped the real header in Python: `net.blocks.0.*` has
  **EXACTLY 20 tensors**, names+shapes identical to what the loader reads.

No block tensors are missing from the 20:
- q_norm/k_norm present for BOTH self_attn and cross_attn ([128] each). ✅
- both attn1 (self) and attn2 (cross) qkv + output_proj present. ✅
- cross_attn k/v are [2048,1024] (joint_dim 1024), correctly expected. ✅
- The "AdaLN base bias" is NOT per-block — `base_adaln` is global
  (`net.t_embedder.1.linear_2.weight [6144,2048]`), added once. anima.rs
  treats it the same. So nothing is missing; the builder's 20-key set is
  complete for one block.

Config dims also confirmed against header: in_ch 68 (`x_embedder.proj.1
[2048,68]`), out_ch 64 (`final_layer.linear [64,2048]`), 28 blocks (max idx
27), joint_dim 1024, mlp 8192, base_adaln 6144. `anima.json` matches.

`linear` convention double-checked (`ops/linear.mojo:1-7,147-182`): weight is
`[out,in]`, `y = x@Wᵀ`, requires x/weight same dtype. block.mojo's linear
calls are convention-correct.

Note (MED): the loader loads only ONE block. The full transformer also has a
6-block frozen `llm_adapter` (T5-style text encoder, dims 1024/64-head),
`t_embedder`, `t_embedding_norm`, `x_embedder`, `final_layer`. Those are out
of scope for the block primitive but MUST be wired (frozen) in the full
trainer before any end-to-end run.

---

## CLAIM 3 — Both sdpa_backward variants symmetric-only; asymmetric cross
##            backward must be built → **VERIFIED** (HIGH)

Evidence (`ops/attention_backward.mojo`):
- `sdpa_backward[B,S,H,Dh]` (line 369): single `S` param. Hard assert
  line 393: `qshape[1] != S → raise`. All buffers built with one shared `S`:
  `head_qk_rl = [S,Dh]` used for q, k, AND v (lines 402, 427-428);
  scores `[S,S]` (line 403); the QKᵀ matmul (430) assumes square. Feeding
  S_q=4096, S_k=256 is impossible.
- `sdpa_backward_scratch[B,S,H,Dh]` (line 497): identical single-`S` assert
  (line 518). Same symmetric structure.
- `grep "sdpa_cross\|cross_backward"` across the whole tree: **no asymmetric
  variant exists.** Confirmed absent.

Worse than flagged — the FORWARD is also symmetric-only:
- `sdpa_nomask[B,S,H,Dh]` (`ops/attention.mojo:643`) takes ONE `S` and
  asserts `qshape[1] != S → raise` (lines 663-669). There is no asymmetric
  forward sdpa either.
- block.mojo has **NO `_cross_attn` function at all** (grep: only
  `_adaln_pre/_adaln_mod/_self_attn/_rms_per_head/_mlp`). The cross-attn the
  docstring describes (lines 23-25, `cross_sdpa(q[S_img], k/v[S_txt])`) is
  unimplemented in both forward and backward.

Cross-attn IS on the LoRA critical path: the Anima LoRA preset targets
`attn1,attn2,ff` (see Claim 4), and `attn2` = cross-attention. So
asymmetric SDPA (forward + backward) is mandatory before LoRA training can
run. The existing symmetric op CANNOT be reused (hard equal-S assert).

---

## CLAIM 4 — Recipe faithfulness → **VERIFIED** (HIGH for sign/timestep, with corrections)

Verified against OT source `BaseAnimaSetup.py`:
- **Target sign CORRECT**: line 143 `flow = latent_noise - scaled_latent_image`
  (= noise − latent), and `model_output_data['target'] = flow` (line 148),
  `loss_type='target'` (145) = MSE(predicted, flow). Builder's "target =
  noise − latent" is exactly right.
- **Noise convention** (`ModelSetupFlowMatchingMixin.py:36-37`):
  `noisy = noise*sigma + latent*(1-sigma)` ⇒ velocity = noise − latent.
  Consistent. Rectified-flow / flow-matching, NOT epsilon or v-pred-DDPM.
- **timestep/1000 CORRECT**: line 137 `timestep=timestep/1000`. Builder right.
- **Frozen TE+base+VAE CORRECT**: `AnimaLoRASetup.py:50-52`
  `text_encoder/transformer/vae.requires_grad_(False)`.
- **LoRA target modules** — builder said "attn1/attn2/ff". This is NOT
  hardcoded in setup (line 62 reads `config.layer_filter.split(",")`), BUT
  the shipped Anima LoRA preset `training_presets/#anima LoRA.json` has
  `layer_filter: "attn1,attn2,ff"` (preset "attn-mlp"). So the builder's
  claim is correct **for the default preset**. VERIFIED.
- **Optimizer** (AdamW eps1e-8/wd0.01/β0.9-0.999): the preset has
  `optimizer: null` — these specific values are NOT sourced from the OT
  preset. They live only in `anima.json`. Treat as AGENT-DEFAULT, not
  source-verified. LOW risk (standard AdamW), but flag it.

Correction to flag: rank/alpha are `null` in the preset too; `anima.json`
hardcodes rank=16/alpha=16. AGENT-DEFAULT, not OT-sourced. (Klein uses
shift 1.8 per memory; anima.json uses shift 1.0 — also a default, and the OT
default `timestep_shift` should be confirmed before a real run.)

---

## CLAIM 5 — No single-block forward parity vs torch (only "finite") → **VERIFIED gap** (HIGH)

`block_smoke.mojo` is explicitly NOT a parity gate (its own header lines 1-5
say so). It only checks `_adaln_pre` and `_adaln_mod` produce finite outputs
of the right shape at B=1,S=4. It does NOT exercise `_self_attn`, RoPE,
attention, or the residual/gate composition, and compares to NO oracle.

The plan (`TRAINING_PLAN_anima.md:17`) marks forward as "RAN, RC=0, finite"
and the actual parity gate (block_oracle.py + block_parity.mojo, cos≥0.999)
as PENDING (plan items A2, lines 113-116). I could not run a forward smoke at
all (see Claim 6 — `_self_attn` won't even execute).

HIGH finding: no backward parity is trustworthy until the forward is
cos-gated at real dims (H16, Dh128, S_img, S_txt=256) against an independent
torch oracle derived from anima.rs / diffusers math. This gate does not yet
exist.

---

## CLAIM 6 — block.mojo `_self_attn` RoPE shaping breaks at runtime → **VERIFIED** (HIGH)

`block.mojo::_self_attn` (lines 147-183) calls
`rope_halfsplit(q, rope_cos, rope_sin, ctx)` with q `[B,S,H,Dh]=[1,S,16,128]`
and rope_cos/sin declared `[1,1,S_img,64]` F32 (lines 152-153, 175-176).

`ops/rope.mojo` `rope_halfsplit` contract (`_rope_common_validate`, the
flat entry):
- flattens ALL leading dims of x to `rows`: for `[1,S,16,128]`,
  `rows = 1*S*16 = 16S`, `half = Dh/2 = 64`.
- requires `cos.numel() == rows*half = 16S*64`, else **raises** "rope: cos
  numel must equal rows*(D/2)" (rope.mojo validate, the `cnum != rows*half`
  check).
- block.mojo's rope_cos numel = `1*1*S*64 = S*64` → **16× too small →
  raises at runtime.**
- ALSO a dtype mismatch: q is BF16 (linear output) but rope_cos/sin are F32;
  validate raises "rope: x/cos/sin dtype mismatch".

The inference path uses a 4D wrapper `_rope_halfsplit_4d`
(`anima_dit.mojo:420`) that broadcasts `[1,1,S,D/2]` over B and H internally;
block.mojo bypassed it and called the flat op directly, so `_self_attn`
**cannot run as written** (two distinct raises). The builder self-flagged the
shaping but understated it — it is a hard runtime failure, not a cosmetic
shape note. The fix is to expand/broadcast cos/sin to q's flattened rows in
q's dtype (copy the `_rope_halfsplit_4d` shaping, per plan item A2).

---

## Summary table

| # | Claim | Verdict | Impact |
|---|-------|---------|--------|
| 1 | AdaLN-pre = LayerNorm-no-affine (not RMSNorm) | VERIFIED (builder correct) | HIGH |
| 2 | Loader reads all 20 block tensors, shapes match | VERIFIED (reproduced RC=0) | HIGH |
| 3 | sdpa backward (+forward) symmetric-only; cross unimpl. | VERIFIED | HIGH |
| 4 | Recipe: target=noise−latent, t/1000, attn1/attn2/ff, frozen | VERIFIED (opt/rank/alpha/shift = AGENT-DEFAULT) | HIGH/LOW |
| 5 | No forward parity gate (only finite-smoke) | VERIFIED gap | HIGH |
| 6 | `_self_attn` RoPE cos/sin shaping breaks at runtime | VERIFIED (2 raises) | HIGH |

What the builder got RIGHT (do not "fix"): the LayerNorm-vs-RMSNorm split,
the 20-tensor loader, the flow-matching target sign + timestep/1000, the
linear `[out,in]` convention, and accurately self-flagging the cross-attn
primitive gap and the RoPE shaping in the plan.

---

## Prioritized fix list for the bugfixer (before Phase 2)

1. **[HIGH] Fix `_self_attn` RoPE call** (Claim 6). cos/sin must be expanded
   to q's flattened rows (`B*S*H` for `[B,S,H,Dh]`) and cast to q's dtype
   (BF16), per the `_rope_halfsplit_4d` broadcast in `anima_dit.mojo:420`.
   Until this is fixed `_self_attn` raises and no forward can run.

2. **[HIGH] Implement cross-attention forward + asymmetric SDPA** (Claim 3).
   Build `_cross_attn` in block.mojo (currently absent) AND an asymmetric
   `sdpa_nomask`-equivalent for S_q≠S_k (q[S_img], k/v[S_txt=256]). The
   existing symmetric op cannot be reused. Cross-attn carries `attn2`, a LoRA
   target, so it is on the training critical path.

3. **[HIGH] Build asymmetric `sdpa_cross_backward[B,S_q,S_k,H,Dh]`** in
   `ops/attention_backward.mojo` with a torch-oracle parity gate in
   `ops/parity/` (cos≥0.999 at e.g. S_q16/S_k8). Both existing backward
   variants hard-assert equal S.

4. **[HIGH] Stand up the single-block FORWARD parity gate** (Claim 5):
   `parity/block_oracle.py` (torch, from anima.rs/diffusers math) +
   `block_parity.mojo`, cos≥0.999 at real dims (H16, Dh128, S_img, S_txt=256).
   This must pass before ANY backward parity is trusted.

5. **[MED] Wire the full-model frozen context** the block needs: global
   `base_adaln` (`net.t_embedder`), `t_embedding_norm` (RMSNorm), the 6-block
   frozen `llm_adapter`, `x_embedder`, `final_layer`. The loader currently
   covers one block only.

6. **[LOW] Confirm AGENT-DEFAULT recipe values** against OT defaults, not just
   `anima.json`: optimizer eps/wd/betas, lora rank/alpha, and `timestep_shift`
   (anima.json=1.0; the OT preset leaves these null). Document owner as
   AGENT-DEFAULT until a real run validates.

Forward parity (item 4) is the gate; items 1-3 are prerequisites to even
reach it. Do NOT trust any backward or smoke until 1, 6, and the forward
cos-gate are green.
