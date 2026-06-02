# TRAINING_PLAN_anima.md — Anima (Cosmos-Predict2 MiniTrainDIT) training port

> Phase-1 builder output (2026-06-01). Maps the remaining Anima training port to
> the proven Klein template. Verified facts carry a source; hypotheses are
> labelled. Tenet 4: nothing here is asserted as measured unless a tool result in
> the session backed it.

## ✅ PHASES D + E DONE — train_anima_real RUNS A REAL RUN (2026-06-01)

`serenitymojo/training/train_anima_real.mojo` (built RC=0) — a line-faithful
TRANSLATION of `train_anima.rs` onto the parity-verified LoRA stack. It loads the
REAL `anima-base-v1.0.safetensors` (28 blocks streamed per-step via
`anima_stack_lora_forward_streamed/_backward_streamed`), the cached latent
(`anima_synth_smoke/sample0.safetensors`), and the captured frozen LLM-adapter
context (`anima_embeddings.safetensors::context_cond [1,256,1024]`), computes
`t_cond`/`base_adaln` from the REAL t_embedder weights (sinusoidal→linear→silu→
linear + RMSNorm, faithful to anima_dit `_prepare_timestep`), patchifies INPUT
([Cp,pH,pW]) and the flow TARGET in OUTPUT layout ([pH,pW,C], C-fastest — the
inverse of `_unpatchify`, VERIFIED anima_dit.mojo:715), then per step:
sigmoid→shift→clamp sigma → flow noisy/target → forward → MSE loss → d_out=2/N·(pred−target)
→ backward → global-norm clip(1.0) → AdamW over all 280 adapters → save (kohya keys).

**MEASURED (fixed-sigma smoke, σ=0.5, 6 steps, real weights, S_IMG=64 crop):**
loss **1.868 → 1.823 → 1.731 → 1.603 → 1.473 → 1.374** (Δ = −0.495, monotone↓);
LoRA-B |.|₁ **0 → 577 → 1130 → 1624 → 2053 → 2416 → 2724** (280/280 adapters
nonzero, ratio 1.0); nonfinite=0 every step; 280 LoRA pairs saved; ~22 s/step,
peak well under 24 GB, 64 °C. **VERDICT: PASS, RUN_RC=0.** The random-sigma
schedule (FIXED_SIGMA_SMOKE=False, the real recipe) also runs RC=0 with finite
grads + LoRA-B growth; loss there bounces per-step because each step draws a new
σ (different target scale) — fixed-σ isolates the learning signal for the
loss-decrease gate. To run full-resolution / full schedule: set
FIXED_SIGMA_SMOKE=False, raise LATENT_HW + RUN_STEPS (S_IMG is comptime).

`serenitymojo/pipeline/anima_prepare.mojo` (built RC=0): TRANSLATION of the
prepare CONTRACT — validates a cache dir against the 5-key schema + reports the
adapter-context sidecar, and REUSES existing caches (the FAST PATH). It does NOT
fabricate latents/embeddings: the real-image encode needs the Qwen-Image VAE
ENCODER (Mojo has only the DECODER) + a Qwen3/T5 text path (not ported) + the
6-block LLM adapter. Those three ports are the remaining real-image-prep work;
for LoRA training the adapter output (context) is a FROZEN input, so its backward
is not needed. Run: `/tmp/anima_prepare <cache_dir>`.

### Remaining (out of THIS milestone's scope — trainer runs without them)
- Qwen-Image VAE encoder port (decoder exists; encoder is the 8× down stack).
- Qwen3-0.6B + T5 tokenizer + 6-block LLM adapter port (for real-image prep +
  per-caption context; currently consumes captured sidecar context).
- Full-resolution (S_IMG=1024) + random-σ multi-step convergence run on a free GPU.

## What is DONE + VERIFIED this session (RC=0, ran on GPU)

| Deliverable | File | Status |
|---|---|---|
| Config (JSON source of truth) | `serenitymojo/configs/anima.json` | reads OK |
| Config accessor (mirror klein/config) | `serenitymojo/models/anima/config.mojo` | RC=0 |
| Per-block weight loader (20 tensors) | `serenitymojo/models/anima/weights.mojo` | RC=0 |
| Block-0 weights-load + shape verify | `serenitymojo/models/anima/weights_verify.mojo` | RAN, all 20 shapes OK |
| Block forward primitives (AdaLN-pre + AdaLN-LoRA mod) | `serenitymojo/models/anima/block.mojo` | compiles clean |
| Forward-primitive smoke (finite, shapes) | `serenitymojo/models/anima/block_smoke.mojo` | RAN, RC=0, finite |

The inference forward (`serenitymojo/models/dit/anima_dit.mojo`, ~1286 LOC,
struct `AnimaDiT`, `for i in range(ANIMA_DEPTH)` depth loop) ALREADY EXISTS and
mirrors `inference-flame/src/models/anima.rs` faithfully (read line-by-line this
session). This is the parity oracle for the forward.

## Architecture (VERIFIED against anima.rs + checkpoint header)

MiniTrainDIT backbone, 28 uniform blocks. Per block (F32 residual stream):
- 3 sub-blocks (self_attn / cross_attn / mlp), each preceded by AdaLN-LoRA
  modulation and followed by a gated residual `x_f32 += gate * sub_out`.
- **AdaLN-LoRA modulation** per sub-block: `chunk3( W2[6144,256](W1[256,2048](silu(t_cond))) + base_adaln )` → (shift, scale, gate) each [B,2048].
- **AdaLN-pre** (the modulation applied to x): `(1+scale)*LayerNorm(x, no-affine, eps=1e-6) + shift`.
  VERIFIED `modulate_pre_fused_bf16` (flame-core bf16_ops.rs:1637 needs_grad path) =
  `layer_norm(x,[dim],None,None,eps)` then affine — LayerNorm WITHOUT affine, NOT RMSNorm.
  (The inference Mojo kernel `_apply_adaln_modulate` docstring says "RMSNorm" but the
  kernel body computes `(v-mean)*inv` = LayerNorm-no-affine — docstring is wrong, code is right.)
- **self_attn**: q/k/v = no-bias Linear(2048→2048); reshape [B,S,16,128]; per-head
  RMSNorm(q_norm/k_norm,1e-6); 3D-RoPE halfsplit on q,k; SDPA scale 1/√128; out Linear.
- **cross_attn**: Q from image (2048→2048, S_img=4096), K/V from text (1024→2048, S_txt=256);
  per-head RMSNorm; **NO RoPE**; asymmetric SDPA (S_q≠S_k); out Linear.
- **mlp**: Linear(2048→8192) → GELU → Linear(8192→2048). Plain GELU, NOT SwiGLU.
- Timestep: sinusoidal(2048) → Linear(2048→2048) → SiLU → Linear(2048→6144)=base_adaln;
  t_cond = RMSNorm(sinusoidal_emb, t_embedding_norm). `timestep/1000` is NOT applied here —
  OT passes `timestep/1000` to the transformer (BaseAnimaSetup.py:137); our forward takes raw sigma.
- Final layer: AdaLN (2 outputs shift+scale, +base_adaln[:4096]) → Linear(2048→64).
- Patch 2×2×1; in 68 = (16+1 mask)·2·2; out 64 = 16·2·2. 5D latent [B,T,H,W,16].
- **LLM adapter** (resident): 6 blocks dim1024 16h Dh64 MLP4096 (WITH bias) 1D-RoPE +
  embed[32128,1024] + out_proj + norm.

## OT training recipe (AUTHORITATIVE — BaseAnimaSetup.py + AnimaLoRASetup.py)

- **Objective**: flow-matching. `target = latent_noise - scaled_latent_image` (flow),
  `predicted = transformer(...)`, loss = MSE (`_flow_matching_losses(...).mean()`).
- **Timestep**: discrete sample, `timestep/1000` into the model, sigma from scheduler.
  `timestep_shift` (dynamic or static) via `calculate_timestep_shift`.
- **LoRA target**: `LoRAModuleWrapper(transformer, "transformer", config, config.layer_filter.split(","))`.
  Presets (BaseAnimaSetup.LAYER_PRESETS): `attn-mlp = [attn1, attn2, ff]`,
  `attn-only = [attn1, attn2]`, `blocks = [transformer_block]`, `full = []` (all Linear).
  Default `layer_filter_preset = "full"` (TrainConfig.py:339). attn1=self_attn, attn2=cross_attn,
  ff=mlp. → LoRA targets every no-bias Linear in the matched modules; base weights frozen.
- **Kohya save naming** (inference-flame/src/bin/anima_lora_infer.rs:12): `lora_unet_blocks_{i}_<sub>`
  (NOT `lora_unet_net_blocks_*`). The runtime LoRA applies at the `linear_no_bias` chokepoint
  AFTER the base matmul (anima.rs:178-181), base weights never mutated.
- **Frozen**: text_encoder, transformer base, vae all `requires_grad_(False)`; only LoRA trains.
- Backprop does NOT flow into the LLM adapter weights for LoRA on the DiT backbone (adapter is
  part of the frozen transformer; only matched DiT Linear modules get LoRA). The adapter output
  (context) is a FROZEN input to cross-attn — its grad path stops at the cross-attn K/V Linears'
  d_input which is discarded (context is not a trained leaf). So the resident adapter needs NO
  backward for LoRA training. (Full-FT WOULD need it — deferred.)

## Block weight key layout (VERIFIED, checkpoint header) — `net.blocks.{i}.*`, 20 BF16 tensors
adaln_modulation_{self_attn,cross_attn,mlp}.{1[256,2048],2[6144,256]};
{self,cross}_attn.{q_proj,k_proj,v_proj,output_proj}.weight + {q,k}_norm.weight[128];
mlp.{layer1[8192,2048],layer2[2048,8192]}.weight. self k/v_proj=[2048,2048], cross
k/v_proj=[2048,1024]. Resident: x_embedder.proj.1[2048,68], t_embedder.1.linear_{1[2048,2048],2[6144,2048]},
t_embedding_norm[2048], final_layer.{adaln_modulation.1,.2, linear[64,2048]}, llm_adapter.*.

## MISSING ops/ primitive (HARD — Tenet 1: build in ops/, parity-gated)

**`sdpa_cross_backward[B, S_q, S_k, H, Dh]`** — cross-attention SDPA backward with
ASYMMETRIC seq lengths. The existing `ops/attention_backward.sdpa_backward[B,S,H,Dh]` and
`sdpa_backward_scratch` are SYMMETRIC-only (single `S` param; assert q/k/v all `[B,S,H,Dh]`).
Anima cross-attn has S_q=4096 (image) ≠ S_k=256 (text). VERIFIED by reading the signatures
(attention_backward.mojo:369,497). This is the genuine blocker for the cross-attn backward arm.
- The FORWARD already has the asymmetric path: `anima_dit.mojo::_cross_sdpa_nomask[B,S_q,S_k,H,Dh]`
  (math-mode matmuls + softmax). The backward is the decomposed transpose of that same math:
  `d_v = attnᵀ@d_out; grad_attn = d_out@Vᵀ; softmax-bwd → grad_scores; d_q=(grad_scores@K)·scale;
  d_k=(grad_scoresᵀ@Q)·scale` — identical to `sdpa_backward` but with K/V at S_k and Q/d_out at S_q.
- Build it as `ops/attention_backward.sdpa_cross_backward` returning `SdpaGrads{d_q[S_q], d_k[S_k], d_v[S_k]}`.
  Gate it standalone (`ops/parity/` torch oracle, cos≥0.999 at e.g. B1 S_q16 S_k8 H4 Dh32).

## Backward arms — what EXISTS vs what is NEW

EXISTS in ops/ (verified by grep, reusable directly):
- `linalg_backward`: linear_backward / linear_backward_dx / _dw / mm_backward / bmm_backward.
- `norm_backward`: rms_norm_backward(_dx), layer_norm_backward(_dx).
- `activation_backward`: gelu_backward, silu_backward.
- `attention_backward`: sdpa_backward (SYMMETRIC — for self_attn S=4096 only).
- `elementwise_backward`: modulate_backward (the affine (1+scale)*x+shift part).
- `rope_struct_backward`: rope_backward, gate_residual_backward.
- `shape_backward`: reshape/slice/cat/permute/transpose backward.

NEW needed:
1. **`sdpa_cross_backward`** (above) — the only HARD new primitive.
2. **AdaLN-pre backward** — NOT a new primitive: composes `modulate_backward` (d_x,d_scale,d_shift
   for the affine) → `layer_norm_backward_dx` (LN no-affine, discard d_g/d_b). This is EXACTLY the
   Klein single_block chain (single_block.mojo KEY DIFFERENCE #3). The genuinely-Anima piece is that
   d_scale/d_shift must then backprop through the per-block adaln_modulation Linear chain
   (linear_backward W2 → linear_backward W1 → silu_backward), accumulating d_W into the (frozen for
   LoRA / trained for full-FT) modulation Linears. Gate the FULL adaln (mod-chain + pre) backward.

## Ordered remaining phases (Klein-template-mapped)

- **A1 — asymmetric cross SDPA backward** ✅ DONE. The shared primitive
  `ops/attention_backward.sdpa_backward_rect[B,Sq,Skv,H,Dh]` (rectangular, no-mask)
  was provided + consumed as-is. Cross-attn is NO-MASK (see below), so no mask arm
  was needed. Self-attn uses the existing square `sdpa_backward`.
- **A2 — block forward parity gate** ✅ DONE (2026-06-01). `block.mojo::anima_block_forward`
  fixed `_self_attn` RoPE (expand cos/sin to [B·S·H, Dh/2] F32, rope_halfsplit
  NON-full, interleaved=False — matches anima.rs:379) + implemented `_cross_attn`
  (rectangular `sdxl_sdpa`, no mask). `parity/block_oracle.py` (torch, independent
  from anima.rs math) + `block_parity.mojo`. **GATE: forward out cos = 0.99999999**
  at real dims H16/Dh128 (S_img6/S_txt8).
- **A3 — block backward** ✅ DONE (2026-06-01). `block.mojo::anima_block_backward`
  hand-chains the 3 sub-blocks (gated-residual → sub-body → AdaLN-pre →
  AdaLN-LoRA-256 mod-chain), inlined (Mojo Movable structs can't move Tensor fields
  out — borrow-only, like Klein single_block). **GATE: d_x cos 0.99999999, d_t_silu
  0.99999999, all 20 base weight grads ≥0.99999999, all 6 AdaLN-LoRA-256 mod-weight
  grads ≥0.99999999.** Self-attn uses square `sdpa_backward` + rope_backward
  (interleaved=False); cross uses `sdpa_backward_rect`.

### Cross-attn mask determination (VERIFIED 3-source, no mask)
- anima.rs:433 `sdpa(&q,&k,&v, None)` — NO additive mask on the text-token attention.
- OT BaseAnimaSetup.py:95 "Anima encode_text returns a plain Tensor (no mask)".
- The only "padding mask" in the path is the pixel-space +1 patch channel (zeros),
  NOT an attention mask. diffusers CosmosAttention CAN take attn_mask but the
  reference passes None. → consumed `sdxl_sdpa`/`sdpa_backward_rect` as-is; NO mask
  arm added to ops/.
- **B — full 28-block stack fwd+bwd** ✅ DONE + GATE PASS (2026-06-01).
  `anima_stack.mojo::anima_stack_{forward,backward}` (mirror klein_stack_lora.mojo):
  residual carry, block loop, final-layer AdaLN + Linear backward, patch/unpatch
  backward (reshape/permute only, no-op grads). `parity/stack_oracle.py` (torch,
  independent from anima.rs) + `stack_parity.mojo`.
  **ROOT-CAUSE FIX (measured): double-silu.** The stack was passing `t_silu`
  (already silu'd) into `anima_block_forward`'s `t_cond` slot, but the block applies
  `silu` internally (block.mojo:342, matching anima.rs adaln_modulation /
  final_adaln_modulation which both call `t_cond.silu()`). The block thus computed
  `silu(silu(t_cond))`. Measured `cos(t_silu, silu(t_silu))=0.987 max_abs 0.278` —
  a real perturbation that compounded sa_xmod 0.999 → x_after_0 0.63 → out 0.996.
  **Fix (forward-composition wiring, all in `models/anima/`, NOT a primitive):** the
  stack now takes RAW `t_cond` and passes it straight to each block (block silus once);
  the final layer applies `silu(t_cond)` once before `fl_mod1`. Backward mirrors this
  (final-layer `fl_mod1` input activation = `silu(t_cond)`; `fl_lb1.d_x` = grad at the
  silu output = the final-layer contribution to `d_t_silu`, summed with the per-block
  `d_t_silu` — block returns d-w.r.t.-silu-output, NOT d_t_cond; the t_embedder
  silu_backward is the deferred phase-E link). Oracle made RAW `t_cond` the leaf with
  `t_silu.retain_grad()` so `ref_d_t_silu` stays d-w.r.t.-silu-output (matches block).
  **GATE: out cos 0.99999999999; d_patches 0.99999999999; d_t_silu 0.99999999999;
  d_x_embed/d_fl_lin/d_fl_mod1/d_fl_mod2 all ≥0.99999999998; per-block (deepest L-1 +
  shallowest 0) d_sa_q/d_mlp2/d_sa_mod1/d_ca_v all ≥0.99999999996. VERDICT: PASS,
  RUN_RC=0.** Single-block gate (block_parity) re-confirmed PASS (block.mojo untouched).
  **FINITE-DIFF SELF-CONSISTENCY (the Klein composition-defect catch, deliverable 4):**
  `parity/stack_finitediff.mojo` computes the gradient NUMERICALLY from the Mojo
  stack's OWN forward (central diff on L=sum(out·d_out)) and compares to the
  analytic `stack_backward` grad — NO torch in the loop. **mean ratio = 1.0036,
  worst |ratio−1| = 0.045 over 9 well-conditioned probes (patches input + deepest
  sa_q[L-1] + shallowest mlp2[0]); VERDICT PASS.** (Tiny-grad coords |analytic|<5e-3
  excluded — central-diff there is F32-noise-dominated, not composition signal.)
  ⇒ the composed backward IS the gradient of the composed forward.
- **C — weights loader all blocks + resident** ✅ DONE + VERIFIED (2026-06-01).
  `weights.mojo`: `load_anima_block_weights_f32` (per-block F32) + `load_anima_all_blocks_f32`
  (0..28) + `AnimaStackBase` + `load_anima_stack_base` (x_embedder.proj.1 [2048,68],
  t_embedder.1.linear_{1,2}, t_embedding_norm, final_layer.{adaln_modulation.1[256,2048],
  .2[4096,256],linear[64,2048]}) + `verify_anima_stack_shapes`. **REAL tensors load at
  correct shapes (base + block 0 + block 27 all OK, RC=0).** AnimaBlockWeights converted
  to TArc fields (Copyable+Movable, mirrors Ernie) so 28 blocks live in a List;
  block.mojo's 40 weight-access sites updated to `w.<field>[]` (math UNTOUCHED —
  block_parity re-confirmed cos 0.99999999). The LLM adapter (6 frozen blocks) is NOT
  loaded — for LoRA the cached adapter output (context) is a frozen INPUT; its weights
  need no grad and are off the LoRA-DiT path. (Full-FT phase F adds them.)
  **RESIDENCY VERDICT: FITS RESIDENT, no streaming needed.** 28 blocks F32 = 7.46 GiB
  weights; `stack_real_smoke.mojo` (deliverable 5) ran the full 28-block fwd+bwd at
  REAL dims (D2048/H16/Dh128/F8192, S_img64/S_txt16): **ALL FINITE (0 non-finite across
  out + all summed/base grads + all 28 blocks' probe grads), NO OOM, peak GPU 7.30 GiB,
  thermal 49°C.** Comfortable on a shared 24GB 3090 — the Ernie-style streamed path is
  available (weights.mojo has the per-block loader) but is NOT required for Anima.
- **D — text/data path**: Qwen3-0.6B final-hidden (model.norm applied) + T5 ids as F32 into the
  resident LLM adapter → context [B,256,1024] (FROZEN, cache once per caption like Klein). VAE =
  Qwen-Image / Wan2.1 VAE for latent encode (5D [B,1,H/8,W/8,16]). Confirm VAE path in anima_contract
  (ANIMA_VAE_PATH) — flagged: VAE encode backward NOT needed (latents are inputs).
- **E — train_anima_real loop** (mirror training/train_klein_real.mojo + shared train_step):
  flow_target = noise − latent (sign per BaseAnimaSetup), timestep/1000 into model, MSE loss,
  AdamW (reuse training/optim.mojo), global-norm clip, LoRA save (kohya `lora_unet_blocks_{i}_*`,
  reuse training/lora_save.mojo). σ-map: discrete timestep → sigma (scheduler), shift applied.
- **F — full-FT extension** (deferred): unfreeze base + adaln + adapter; adds the resident LLM
  adapter backward (6 blocks: 1D-RoPE bwd, self/cross sdpa bwd, GELU-MLP-with-bias bwd) and
  base-weight grads + offload (24GB). Path exists (block bwd returns base d_W) but out of LoRA scope.

## Flagged missing primitives (consolidated)
| Op | Why | Where |
|---|---|---|
| `sdpa_cross_backward[B,S_q,S_k,H,Dh]` | cross-attn S_q≠S_k; existing sdpa_backward is symmetric-only | `ops/attention_backward.mojo` + `ops/parity/` |
| (none else) AdaLN-pre/mod bwd | composes from existing modulate+layer_norm+linear+silu backward | `models/anima/block.mojo` (chain, not primitive) |

## Decision ownership (EMPOWERMENT)
| Decision | Owner | Rationale |
|---|---|---|
| Reuse FLUX-shaped TrainConfig (num_double=0/num_single=28) | AGENT-DEFAULT | avoids a new config struct; Anima-only dims stay comptime in anima_contract |
| anima.json lr=1e-4, rank/alpha=16, shift=1.0 | AGENT-DEFAULT | placeholder recipe; USER should set the Alina-subject values (cf. Klein shift=1.8 memory) |
| AdaLN-pre = LayerNorm-no-affine (not RMSNorm) | AGENT (source-verified) | flame-core modulate_pre_fused needs_grad path uses layer_norm(None,None) |
| sdpa_cross_backward lives in ops/ | AGENT (Tenet 1) | reusable primitive, not a model-file workaround |
| **Modulation is PER-BLOCK, NOT shared** (unlike Klein/Ernie) | AGENT (source-verified anima.rs:474-503) | each block i has its OWN adaln_modulation_{sa,ca,mlp}.{1,2} (net.blocks.{i}.*), computed from the SHARED t_silu+base_adaln. Composition consequence: 6 mod-weight grads are PER-BLOCK (returned per block, NOT summed); only `d_t_silu` (+ final-layer base_adaln[:4096]) SUM across the 28 blocks + final layer. Verified by the composition gate (per-block deepest≠shallowest grads pass independently) + finite-diff. |
| Stack input = RAW t_cond (block silus internally) | AGENT (source-verified) | block.mojo:342 + anima.rs adaln_modulation/final_adaln_modulation all call t_cond.silu(); passing t_silu double-silus (the measured root-cause bug). Stack returns summed d_t_silu (grad at silu output) → one silu_backward into the deferred t_embedder link. |
| Resident 28-block stack (no streaming) | AGENT (measured) | F32 28 blocks = 7.46 GiB, full fwd+bwd peak 7.30 GiB < 24 GB. Ernie-style streamed loader exists but unneeded for Anima's smaller D=2048. |

## Status after Phase B+C (2026-06-01)

**DONE + GATE-VERIFIED:** block fwd+bwd (A2/A3, cos 0.99999999), 28-block stack
fwd+bwd composition (B, cos 0.99999999999 + finite-diff ratio 1.0036), all-28-block
+ resident weight loader (C, real shapes + 28-block real-depth smoke ALL FINITE,
peak 7.30 GiB). NO new ops/ primitive was needed for the stack (composes existing
linear/norm/activation/attention/rope/elementwise backward arms). Artifacts:
`models/anima/{anima_stack.mojo, weights.mojo}` + `parity/{stack_oracle.py,
stack_parity.mojo, stack_finitediff.mojo, stack_real_smoke.mojo}`.

## LoRA arm DONE + GATE-VERIFIED (2026-06-01)

**E-prep (LoRA arm) is COMPLETE and tri-gated.** Mirrors the proven Ernie/Klein LoRA
template. The 10 target projections (per inference-flame anima.rs `linear_no_bias`
chokepoint sites): self_attn.{q_proj,k_proj,v_proj,output_proj} (anima.rs:365-390),
cross_attn.{q_proj,k_proj,v_proj,output_proj} (anima.rs:411-440), mlp.{layer1,layer2}
(anima.rs:449-451). Cross-attn k/v LoRA input is the FROZEN `context` (anima.rs:413-414),
so their LoRA d_x is discarded (matching the base block discarding d_input there); their
d_A/d_B are still trained. Real inference key prefixes `net.blocks.{i}.<sub>.<proj>`
(FMT_DIFFUSION_MODEL → `.lora_A.weight`/`.lora_B.weight`) so an adapter loads back.

Artifacts:
- `models/anima/lora_block.mojo` — `AnimaBlockLora` (10 optional adapters) +
  `anima_block_lora_forward`/`anima_block_lora_backward` (bit-exact base when absent;
  LoRA d_x summed into trained-stream projection-input grads; ca-k/v d_x discarded).
- `models/anima/anima_stack_lora.mojo` — `AnimaLoraSet` (flat 10×num_blocks) +
  `build_anima_lora_set` + resident & streamed `anima_stack_lora_forward/backward` +
  per-block d_A/d_B scatter + `anima_lora_adamw_step` (reuses `_lora_adamw`) +
  `save_anima_lora`/`load_anima_lora_resume` (reuses `save_lora_peft`/`load_lora_for_resume`).
- `parity/{lora_stack_oracle.py, lora_stack_parity.mojo, lora_step_smoke.mojo}`.

Gate results (Tenet 4, MEASURED):
- composition `lora_stack_parity`: VERDICT PASS — all 60 A/B grads cos≥0.99999999,
  out cos 0.99999999999, d_patches + d_t_silu pass, 0 nonfinite.
- step `lora_step_smoke`: VERDICT PASS — B 0→nonzero (nonzero-slot ratio 1.0/30),
  grads finite, global-norm clip applied (6.35→1.0), save/load byte-exact (max_abs_diff 0.0).
- base `stack_parity` re-run after wiring: still VERDICT PASS (no regression).
NO new `ops/` primitive added (LoRA = two `linear`/`linear_backward` calls). The
double-silu fix is preserved (t_silu computed ONCE in the block fwd, reused). Verified
arch facts unchanged: AdaLN-pre=LayerNorm-no-affine eps1e-6; self_attn RoPE
interleaved=False half-split; cross_attn NO RoPE/no mask; GELU tanh.

**REMAINING (in order):**
1. ~~**LoRA arm** (phase E-prep)~~ — DONE 2026-06-01 (see section above).
   (original note retained below for context:) wrap attn1(self)/attn2(cross)/ff(mlp) no-bias Linears
   with LoRA A/B at the linear chokepoint (OT preset `attn-mlp`, default `full`=all
   Linear). Reuse the Klein LoRA template (lycoris/lora_linear); base weights frozen,
   LoRA-A/B the only trained leaves. The stack already returns base d_W per block —
   the LoRA path projects d_W through B/A. Kohya save naming `lora_unet_blocks_{i}_*`.
2. **`train_anima_real` loop** (phase E): mirror training/train_klein_real.mojo —
   flow_target = noise − latent, timestep/1000 into model, MSE, AdamW (training/optim.mojo),
   global-norm clip, LoRA save (training/lora_save.mojo). The t_embedder silu_backward
   (d_t_silu → d_t_cond) + base_adaln link lands here.
3. **Qwen3/T5 + VAE data path** (phase D): Qwen3-0.6B final-hidden + T5 ids → frozen
   LLM adapter → context [B,256,1024] (cache per caption); Qwen-Image/Wan2.1 VAE latent
   encode (5D). VAE/text backward NOT needed (frozen inputs).
4. **Full-FT** (phase F, deferred): unfreeze base+adaln+adapter; add the resident
   6-block LLM-adapter backward + base-weight grads + offload.

**No blockers for the LoRA phase.** The cross-attn d_context is intentionally discarded
(frozen text); the only open carry-forward is the LOW text-padding-mask note (the block +
reference are maskless; revisit only if real captions are padded in cross-attn).
