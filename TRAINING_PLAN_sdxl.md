# TRAINING_PLAN_sdxl.md — SDXL conv-UNet LoRA training port (pure Mojo)

Status: Phases 1–5 + 5b DONE/VERIFIED. Phase 5b = parity-gated LoRA training step
(attn+ff on the SpatialTransformer) — SDXL per-model milestone CLOSED (2026-06-01).
Parity-gated, phased.
Companion to FULL_PORT_TRAINING_PLAN.md (engine strategy) and the Klein/Z-Image
training port. SDXL is a **convolutional UNet**, not a DiT — its training-block
decomposition differs from Klein (ResBlock + cross-attn SpatialTransformer +
down/up sampling + time/label embedding + skip-concat), so the *methodology* of
the Klein port is adapted, not the literal block files.

Parity oracle (read line-by-line): `inference-flame/src/models/sdxl_unet.rs`
(forward + arch), `src/bin/sdxl_lora_infer.rs` + `src/lora.rs` (LoRA target map),
`src/vae/ldm_decoder.rs` + `src/models/clip_encoder.rs` (VAE/text path).

---

## Phase 1 — DONE (verified this session)

| Item | Result |
|---|---|
| conv2d_backward parity (the Tier-5 risk sink) | d_x cos 0.99999999, d_w 0.99999999, d_b 0.99999999 — **PASS** |
| `models/sdxl/config.mojo` + `configs/sdxl.json` | compiles; reads arch+recipe from JSON (binding rule). |
| `models/sdxl/weights.mojo` (LDM→ResBlock loader, OIHW→RSCF) | loads real `sdxl_unet_bf16.safetensors`: input_blocks.4.0 (320→640, skip) + .1.0 (320→320, no-skip) with correct RSCF shapes — **OK** |
| `models/sdxl/block.mojo` ResBlock fwd (save acts) + hand-chained bwd | compiles RC=0; **15/15 gates PASS** (out + d_x + 13 weight grads, cos ≥ 0.99999, incl. the 1×1 skip-conv path) |
| New `ops/` primitive needed for ResBlock | **NONE** — reused conv2d_backward, group_norm_backward, silu_backward, linear_backward (all pre-existing, parity-gated). Tenet 1 honored. |

Parity artifacts: `models/sdxl/parity/resblock_oracle.py` (+ `resblock_ref.txt`),
`resblock_parity.mojo`, `weights_load_smoke.mojo`.

### SDXL UNet block structure (verified vs the Rust oracle)
- **ResBlock** (`resblock`, rs:593-631): GN(32,eps=1e-5)→SiLU→Conv3×3 →
  [SiLU→Linear(emb)→per-channel add] → GN(32,eps=1e-5)→SiLU→Conv3×3 →
  (+ 1×1 skip conv if Cin≠Cout) → FP32 residual add. **[DONE Phase 1]**
- **SpatialTransformer** (`spatial_transformer`, rs:761-817): GN(32,eps=**1e-6**)
  → linear proj_in → N× BasicTransformerBlock → linear proj_out → residual add.
- **BasicTransformerBlock** (rs:702-755): LN→self-attn(attn1)→res, LN→
  cross-attn(attn2, K/V from text context)→res, LN→GEGLU FF→res.
- **Down/Up**: Downsample = stride-2 Conv3×3 (`.op`); Upsample = nearest-2× →
  Conv3×3 (`.conv`).
- **Embeddings**: time = sinusoidal→Linear(320,1280)→SiLU→Linear(1280,1280);
  label(ADM) = Linear(2816,1280)→SiLU→Linear(1280,1280); summed → emb [B,1280].
- **Stack**: conv_in → 9 input blocks (push skips) → middle (Res+ST+Res) →
  9 output blocks (pop+concat skip on channel axis) → GN→SiLU→conv_out.

### LoRA target map (kohya/sd-scripts, from lora.rs build_kohya_unet_table)
kohya maps any `<module>.weight` → `lora_unet_<module>`; SDXL standard set:
- **Attention (always)**: `*.attn1.to_q/to_k/to_v/to_out.0`,
  `*.attn2.to_q/to_k/to_v/to_out.0` (in every transformer_block).
- **FF (always)**: `*.ff.net.0.proj` (GEGLU in), `*.ff.net.2` (out).
- **LoCon (optional, conv)**: `*.in_layers.2`, `*.out_layers.3`,
  `*.skip_connection`, `*.op` (downsample), `*.conv` (upsample), proj_in/proj_out.
  Conv-LoRA adapters already exist: `training/locon_conv_adapter.mojo`,
  `tucker_conv_adapter.mojo`, save via `training/locon_save.mojo`.
- Text-encoder LoRAs (`lora_te1_/lora_te2_`) are NOT applied to the UNet.

---

## Remaining phases (ordered, each parity-gated cos ≥ 0.999 vs torch autograd)

### Phase 2 — SpatialTransformer cross-attn block fwd+bwd  [DONE — verified 2026-06-01]
`models/sdxl/spatial_transformer.mojo` :: `spatial_transformer_{forward,backward}`
(+ `_basic_block_*`, `_attn_*`). NHWC F32 in/out: GN(eps=1e-6) → reshape
NHWC→tokens (free) → Linear proj_in → BasicTransformerBlock × depth → Linear
proj_out → reshape → FP32 residual. Per-block: LN1→self-attn→res, LN2→cross-attn
→res, LN3→GEGLU-FF→res. Tensor carriers in List-stored structs are TArc
(ArcPointer[Tensor], the klein_stack Copyable-carrier idiom; raw Tensor is move-only).

| Item | Result |
|---|---|
| `spatial_transformer.mojo` fwd+bwd | compiles RC=0 (library; "no main" only) |
| depth=1 parity (out + d_x + d_context + 9 ST-level + 19 per-block weight grads) | **29/29 PASS**, cos ∈ [0.99999982, 1.0] vs torch autograd |
| depth=2 composition parity (b0+b1, per-block + accumulated d_context) | **48/48 PASS**, cos ∈ [0.99998710, 1.0] |
| GN_EPS_ST | **1e-6 confirmed** used in the ST GroupNorm fwd+bwd (config.mojo:31 vs sdxl_unet.rs:775) |

- **Reused (exist, gated):** `group_norm`/`group_norm_backward` (eps=1e-6),
  `layer_norm`/`layer_norm_backward` (eps=1e-5), `linear`/`linear_backward`,
  `sdpa_nomask`+`sdpa_backward` (SQUARE, attn1 self), `sdxl_sdpa`+
  `sdpa_backward_rect` (RECT Sq=H·W≠Skv=77, attn2 cross), `geglu_forward`/
  `geglu_backward` (already-gated FF), `add`/`reshape`.
- **rectangular SDPA backward** (`ops/attention_backward.mojo::sdpa_backward_rect`)
  was the blocking dependency — already built+gated by the dependency builder.
  This phase PROVES it integrates: the attn2 d_q/k/v/out grads + d_context all
  pass through it at cos ≥ 0.999 (depth=1 d_o2 cos 0.999999997; depth=2 b1 d_o2
  cos 0.99999999). **NO new `ops/` primitive built this phase** (Tenet 1 honored).
- **GEGLU backward** was confirmed composable — the pre-gated
  `models/sdxl/geglu.mojo` chain is reused as-is; no inline primitive.
- Parity artifacts: `parity/spatial_transformer_{oracle.py, parity.mojo,
  parity_d2.mojo}` + `spatial_transformer_ref{,_d2}.txt`. Oracle built
  INDEPENDENTLY from sdxl_unet.rs (NHWC leaf x; canonical NHWC flatten for the
  layout-invariant out/d_x compare).

### Phase 3 — Down/Up sampling + skip-concat backward
- **Downsample bwd:** stride-2 Conv3×3 → `conv2d_backward` (already handles
  stride/pad as comptime params — VERIFIED with stride 1; add a stride-2 parity
  case to conv2d_bwd_parity to gate the stride-2 arm explicitly).
- **Upsample bwd:** nearest-2× then Conv3×3. `conv2d_backward` (conv part) +
  `ops/pool_backward.mojo::_upsample_nearest_dx` (**EXISTS** — verify it has a
  public `upsample_nearest2d_backward` wrapper; if only the kernel exists, add
  the wrapper + a parity gate).
- **Skip-concat bwd:** output blocks do `cat([h, skip], channel)`; backward
  splits d_out back into d_h + d_skip. `ops/shape_backward.mojo::cat_backward`
  (2-input case) **EXISTS** — wire it; gate the channel-axis NHWC case.
- GATE per arm cos ≥ 0.999.

### Phase 4 — time/label embedding fwd+bwd
- sinusoidal timestep_embedding (forward exists in `ops/embeddings.mojo`); the
  Linear→SiLU→Linear MLPs use `linear_backward` + `silu_backward` (EXIST).
- The emb is shared across all ResBlocks — backward must SUM the d_emb
  contributions from every ResBlock's `d_emb_in` (the ResBlock bwd already
  returns `d_emb_in`). The stack accumulates these before the embed-MLP bwd.
- No missing primitive. GATE: d_emb_MLP weights + d_timestep cos ≥ 0.999.

### Phase 5 — full UNet stack fwd+bwd (encoder skip stack + composed backward) — **DONE/VERIFIED (2026-06-01)**
- **GATE RESULT (measured, RUN_RC=0):**
  - `parity/unet_stack_parity.mojo` vs PyTorch autograd — **ALL 11 ARMS PASS**,
    cos ≥ 0.99999998 (out, d_x, d_context, d_y, conv_in/out_w, deep mid0 ResBlock,
    in5 ST attn2 to_q, out0 conv2_w, t0_w, l0_w). Prints
    `ALL SDXL UNET-STACK COMPOSITION FWD+BWD GATES PASSED (cos >= 0.999 vs PyTorch)`.
  - `parity/unet_stack_finitediff.mojo` — **X-PATH FINITE-DIFF SELF-CONSISTENCY PASSED**,
    worst |ratio−1| = 0.0056 (composed backward == grad of composed forward).
- Decoder skip-concat backward verified: out0 `resblock_backward[...,128,C3,...]`
  returns `d_x` with Cin=128 channels; `cat_backward(go0.d_x, C3, C3, 3)` splits
  [64|64], matching forward `concat(3, h, skip_in5)` ([h first, skip second]) and
  `sdxl_unet.rs` `cat(&[&h,&skip],1)`. (An earlier session crash — `cat_backward`
  reading channels [64,128) out of a 64-ch tensor — is no longer present; channel
  params at the out0 call site are correct.)

- **Files:** `models/sdxl/sdxl_unet_stack.mojo` (`sdxl_unet_stack_{forward,backward}_reduced`)
  + `parity/unet_stack_{oracle.py,parity.mojo,finitediff.mojo,real_smoke.mojo}`.
  COMPOSES the gated block units (resblock/spatial_transformer/sampling/embed
  fwd+bwd + conv2d_backward + group_norm_backward + silu_backward + cat_backward).
  **NO new `ops/` primitive built** (Tenet 1).
- **Skip-connection ordering** (the #1 risk) verified vs `sdxl_unet.rs`: encoder
  pushes every input block's output (incl. conv_in) in order; decoder pops LIFO so
  output block k consumes input block (N−1−k); concat is `[h | skip]` on the NHWC
  channel axis (== rs:936 `cat([h,skip],dim1)` NCHW). Backward splits
  size0=carry-channels (FIRST)/size1=skip-channels (SECOND); the skip slab is ADDED
  into the matching encoder block's output grad when the reverse walk reaches it.
  A mis-routed skip would tank the d_x finite-diff ratio — it's at ~1.0.
- **REAL-CONFIG finite smoke** (`parity/unet_stack_real_smoke.mojo`): mc=320,
  channel_mult (1,2,4), num_res_blocks=2, depths [0,0,2,2,10,10]/mid 10/
  [10,10,10,2,2,2,0,0,0], NKV=77, CCTX=2048, head_dim 64, **latent 8×8** —
  forward out [1,8,8,4] finite + backward (final→d8→d7→d6→upsample→d5+ST) finite.
  **Peak GPU mem ≈ 7.85 GB** (≈6.98 GB above baseline), all weights resident +
  full forward acts retained at 8×8.
- **Memory / checkpointing verdict for 1024²:** at 1024² (latent 128²) the
  SpatialTransformer self-attention scores are O(N²) in tokens — level-1 STs at
  64×64=4096 tokens give ~671 MB PER score tensor (×2 attns ×2 blocks ×depth +
  softmax + backward), multi-GB on its own; combined with retained per-block acts
  the full fwd+bwd FAR exceeds 24 GB. **Activation checkpointing (recompute-in-
  backward, the klein_stack per-block idiom) IS required before 1024²** — the
  reduced/real stacks currently retain all per-block acts (fine at small dims).
  This is the Tier-5 `Checkpoint` arm in FULL_PORT_TRAINING_PLAN §1; gate at
  256²/512² first.
- **Open (skeptic MED, carried):** stride-2 conv2d-backward (Downsample `.op`) is
  now EXERCISED transitively (the reduced stack has 2 downsamples at scale
  boundaries and the d_x finite-diff passes through them at ratio ~1.0), but a
  DEDICATED stride-2 `conv2d_bwd_parity` case is still worth adding per Tenet 4.

### Phase 5b — LoRA training step (attn + ff on the SpatialTransformer) — **DONE/VERIFIED (2026-06-01)**
- Closes the SDXL per-model milestone: a parity-gated LoRA training step, mirroring
  the just-verified-green Ernie LoRA path (`models/ernie/lora_block.mojo` +
  `ernie_stack_lora.mojo`).
- **Composition unit = the SpatialTransformer** (not the whole conv-UNet): SDXL's
  LoRA targets live ENTIRELY inside the BasicTransformerBlock (attn1/attn2/ff
  linears); the conv-UNet skip/topology that wraps the 5 STs is already gated by
  Phase 5's `unet_stack_parity`. Gating the ST-with-LoRA composition fully exercises
  every adapter's d_A/d_B + base-no-regression. The flat carrier indexes
  `num_blocks × 10 slots`, so it scales directly to all 5 STs in the Phase-7 loop.
- **10 LoRA slots / BasicTransformerBlock:** attn1.{to_q,to_k,to_v,to_out.0},
  attn2.{to_q,to_k,to_v,to_out.0}, ff.net.0.proj, ff.net.2. (Conv-LoCon = optional
  Phase-7 follow-up.)
- **GATE RESULTS (measured this session, RUN_RC=0):**
  - `parity/lora_stack_parity.mojo` (depth=2) vs PyTorch autograd —
    **`VERDICT: PASS`**, all 40 LoRA A/B grads (10 slots × 2 blocks) + forward out +
    d_x + d_context at cos ≥ 0.999 (worst 0.99999997), 0 nonfinite.
  - `parity/lora_step_smoke.mojo` — **`VERDICT: PASS`**: build (B=0 identity) → fwd
    → bwd → global-norm clip → AdamW → LoRA-B 0→nonzero (20/20 slots, ratio 1.0) →
    save → load **byte-exact (max_abs_diff = 0.0)**. (`|dA|_1 = 0` at step 0 is the
    correct PEFT-identity behavior — with B=0, d_A = d_t@x and d_t = d_dy@B = 0; only
    B gets the first-step gradient. Same step-0 expectation as Ernie/Klein.)
  - Base no-regression: `unet_stack_parity` re-run still
    `ALL SDXL UNET-STACK COMPOSITION FWD+BWD GATES PASSED`.
- **Files (all in `models/sdxl/`):** `lora_block.mojo` (10-slot block LoRA carrier +
  `sdxl_lora_apply`/`sdxl_lora_bwd`/`sdxl_proj_lora_into_dx`), `sdxl_unet_stack_lora.mojo`
  (`SdxlLoraSet` + `build_sdxl_lora_set` + LoRA-aware ST fwd/bwd + d_A/d_B scatter +
  `sdxl_lora_adamw_step` + `save_sdxl_lora`/`load_sdxl_lora_resume`),
  `parity/{lora_stack_oracle.py, lora_stack_parity.mojo, lora_step_smoke.mojo}`.
  REUSES `training/train_step.{LoraAdapter,_lora_fwd,_lora_bwd,_lora_adamw}` and
  `training/lora_save.{save_lora_peft,load_lora_for_resume}` by CALLING them — no
  `training/` edits, **NO new `ops/` primitive** (Tenet 1).
- **Save key convention (this milestone):** PEFT `<diffusers-module-path>.lora_A.weight`
  / `.lora_B.weight` (reuses the proven `save_lora_peft` unmodified). Loads byte-exact
  via inference-flame `lora.rs` `map_prefix_diffusion_model` generic fallback
  (`<prefix>.weight` → base key). Phase 7 may add the kohya `lora_unet_*` convention
  (needs a new `training/` save path) — orthogonal to this gate.

### Phase 6 — VAE encode + text path (latent + conditioning for real training)
- **Latents:** LDM VAE *encoder* (forward — only the DECODER is ported in
  `models/vae/ldm_decoder.mojo`). Training needs ENCODE (image→latent). Port the
  VAE encoder forward (no backward — VAE is frozen; we only need latents).
  Reuse `decoder2d.mojo` conv/GN/attn kit. Scale 0.13025, shift 0.0.
- **Text:** CLIP-L + CLIP-G encoders exist (`models/text_encoder/clip_encoder.mojo`,
  forward). Training uses CACHED embeds (offline) OR live encode. context [B,77,2048]
  (L⊕G hidden), y [B,2816] (L_pool 768 ⊕ G_text_embeds 1280 ⊕ zeros 768). The
  context/y ASSEMBLER + CLIP-G text_projection are NOT in Mojo (skeptic S1) — for
  training reuse cached embeds first; port the assembler only if live-encode needed.
- eps-prediction flow: scaled-linear betas (0.00085→0.012, 1000 steps); loss =
  MSE(eps_pred, noise). Reuse `training/schedule.mojo` (add SDXL DDPM eps target).

### Phase 7 — train_sdxl_real loop policy + LoRA save + full-FT extension
- `training/train_sdxl_real.mojo` mirroring `train_klein_real.mojo`: load config
  (sdxl.json) → load UNet weights → attach LoRA (attn + optional LoCon conv) →
  per-step: VAE-encode (or cached latent) → sample timestep+noise → eps target →
  UNet fwd (save acts) → MSE → UNet bwd → grad-clip → AdamW step → log/save/sample.
- Reuse shared `training/`: `train_step.mojo`, `train_config.mojo`, `optim.mojo`,
  `lora_save.mojo` / `locon_save.mojo`, `checkpoint.mojo`, `schedule.mojo`,
  `on_device_global_norm.mojo`, `grad_coverage.mojo`.
- LoRA save key convention = kohya `lora_unet_<module>.lora_down/up.weight` +
  per-module `.alpha` (so the existing `sdxl_lora_infer` can load the output).
- Full-FT (base-weight grads) is the natural extension — the ResBlock bwd already
  returns every base-weight grad; gate it behind a `--full` flag later.

---

## Missing `ops/` backward primitives (consolidated — build in `ops/`, Tenet 1)
| Op | Why missing | Where | Phase |
|---|---|---|---|
| **rectangular SDPA backward** (Sq≠Skv) | existing `sdpa_backward` is square-only; cross-attn Sq=H·W ≠ Skv=77 | `ops/attention_backward.mojo` + `ops/parity/sdpa_rect_bwd_*` | 2 |
| GEGLU backward helper | split·gelu(gate) bwd — *likely composable* from gelu_backward + mul + shape; confirm at build | `ops/` (gate if non-composable) | 2 |
| stride-2 conv2d bwd parity case | conv2d_backward supports stride comptime but only stride-1 is gated | extend `ops/parity/conv2d_bwd_parity.mojo` | 3 |
| upsample_nearest2d_backward wrapper | kernel `_upsample_nearest_dx` exists; verify/add public wrapper + gate | `ops/pool_backward.mojo` | 3 |
| cat_backward channel-axis NHWC gate | `cat_backward` exists; gate the NHWC channel-concat case | `ops/parity/shape_bwd_*` | 3 |
| Checkpoint / activation offload | full-UNet acts > 24 GB at 1024² | flame-core-equiv Tier-5 arm (see FULL_PORT_TRAINING_PLAN §1) | 5 |
| VAE encoder forward (not bwd) | only decoder ported; training needs image→latent | `models/vae/` (encoder) | 6 |

NOT missing (verified to exist + parity-gated): conv2d_backward, group_norm_backward,
layer_norm_backward, silu_backward, gelu_backward, linear_backward, cat_backward,
maxpool2d_backward, _upsample_nearest_dx, reduce/shape backward.

---

## Validation discipline (per FULL_PORT_TRAINING_PLAN §5)
- Per-arm grad parity cos ≥ 0.999 vs torch autograd (F32 interior).
- Composed-backward: per-block dL/dx + dL/dW parity AND full-stack finite-diff
  self-consistency BEFORE any long run (the Klein composition-defect lesson).
- Learning verdict = sample shift, never loss-alone / no-crash.
- Keep parity at SINGLE-BLOCK / small spatial dims on the shared 24 GB 3090.

## Blockers / risks
- **Rectangular SDPA backward** is the one genuinely-new compute unit (Phase 2) —
  the same primitive class flagged as HIGHEST RISK in FULL_PORT_TRAINING_PLAN §1
  (SDPA-bwd misalign history). Mitigated here: the SQUARE decomposed bwd already
  exists+passes (H30/H32 cases), and the rectangular FORWARD already exists, so
  the rect-bwd is a bounded extension of two proven pieces, not greenfield.
- Memory at 1024² (Phase 5) needs checkpoint/offload — gate at 256²/512² first.
