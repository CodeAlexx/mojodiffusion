# TRAINING_PLAN — Flux (flux1-dev) pure-Mojo training port

Status: **Flux.1-dev only, not Flux.2/Klein/dev2. Not production-tested.** The
block, stack, and LoRA-step parity notes below are useful, but the integrated
trainer/sampler/save/resume contract must be revalidated later before Flux is
treated as a supported trainer. Historical sections that mention a "real run" are
kept as prior agent notes, not as current acceptance evidence. TODO: rerun a real
Flux.1-dev smoke with the mandatory progress display, shared prompt JSON sampler,
checkpoint save, resume, and final 1024x1024 sample.

Scope of this doc: the Flux **transformer (DiT)** training port inside
`serenitymojo/models/flux/`. Reuses Klein's `training/` + `ops/` by calling,
never modifying. Text encoders (CLIP-L, T5) and the VAE are out of scope for
this DiT port (their LoRA targets are noted for completeness but the DiT is the
deliverable).

---

## 1. Architecture (flux1-dev)

Source of truth for the FORWARD/arch: `inference-flame/src/models/flux1_dit.rs`
(`Flux1Config::default()`, lines 94-112; block forwards 729-1008).

| field | value |
|---|---|
| num_double_blocks | 19 |
| num_single_blocks | 38 |
| inner_dim (D) | 3072 |
| num_heads (H) | 24 |
| head_dim (Dh) | 128 |
| in_channels | 64 |
| joint_attention_dim (T5) | 4096 |
| mlp_ratio | 4.0 → MLP hidden = 12288 |
| timestep_dim | 256 |
| has_guidance | true (Dev) |
| vector_dim (CLIP pooled) | 768 |
| axes_dims_rope | [16, 56, 56]  (halves 8+28+28 = 64 = Dh/2) |
| rope_theta | 10000.0 |

Config is JSON-driven (binding rule): `serenitymojo/configs/flux.json` read by
`models/flux/config.mojo::flux_dev()` via the shared `read_model_config`. The
three Flux-only stack constants not in the shared `TrainConfig`
(axes_dims_rope, has_guidance, vector_dim, mlp_ratio) are exposed as `comptime`
accessors in `config.mojo` for the stack phase.

### Block structure (both block types)

DOUBLE block (per stream s ∈ {img, txt}):
```
s_norm  = (1+scale1)*LayerNorm(s) + shift1
s_qkv   = linear(s_norm, Wqkv_s, bqkv_s)        # WITH bias
q,k,v   = split(s_qkv) -> [1,N,H,Dh]
q,k     = rms_norm(q, q_norm_s), rms_norm(k, k_norm_s)
JOINT:  q=cat(txt_q, img_q); rope_interleaved; sdpa; slice back (txt FIRST)
s_out      = linear(s_att, Wproj_s, bproj_s)
s_attn_res = s + gate1 * s_out
s_mlp_in   = (1+scale2)*LayerNorm(s_attn_res) + shift2
s_mlp      = linear(gelu(linear(s_mlp_in, Wmlp0_s, b0)), Wmlp2_s, b2)   # GELU MLP
s_final    = s_attn_res + gate2 * s_mlp
```
SINGLE block:
```
x_norm  = (1+scale)*LayerNorm(x) + shift
fused   = linear(x_norm, W1, b1)                # [S, 3D + 4D]
qkv     = fused[:, :3D] ; mlp_in = fused[:, 3D:]
q,k,v   = split(qkv); q,k rms_norm; rope; sdpa -> att_flat [S,D]
out_in  = cat(att_flat, gelu(mlp_in))           # [S, D+4D]
out     = linear(out_in, W2, b2)
result  = x + gate * out
```

---

## 2. Klein-reuse assessment (the directed lever)

**Verdict: Klein's blocks are NOT directly reusable; a Flux-specific block is
required.** Flux IS Klein's family (same DiT topology: double/single streams,
modulate→qkv→rms→rope→sdpa→proj→gated-residual→MLP→gated-residual), and the
backward COMPOSITION pattern + every backward ARM is reused unchanged. But two
block-level differences make Klein's `double_block.mojo`/`single_block.mojo` not
byte-compatible (measured from flux1_dit.rs:5-12, 729-1008):

1. **Biases on every linear** (qkv, proj, mlp.0, mlp.2 / linear1, linear2).
   Klein's blocks are no-bias. → Flux passes `Optional[Tensor](bias)` and the
   backward returns the bias column-sum grad `d_b` (`linear_backward` already
   computes `d_b` — no new primitive needed).
2. **GELU MLP (4× ratio), not SwiGLU.** Klein uses `swiglu(gate, up)` on a 2F
   split. Flux is `linear → gelu → linear` (double) / `gelu(mlp_in)` fused in
   linear1/linear2 (single). → Flux uses `gelu` / `gelu_backward` (both already
   in `ops/`, tanh-approx, matching flame-core `gelu_tanh_derivative.cu`).

Everything else is identical to Klein and reused via the SAME `ops/` arms:
`modulate`/`modulate_backward`, `layer_norm`/`layer_norm_backward`,
`rms_norm`/`rms_norm_backward`, `rope_interleaved`/`rope_backward`,
`sdpa_nomask`/`sdpa_backward`, `gate_residual_backward`, `cat_backward`,
`slice`/`concat`/`reshape`. **No `ops/` primitive was missing or edited.** The
3-axis RoPE table construction is a STACK-level concern (the block consumes
precomputed cos/sin); the rope APPLICATION is the same interleaved-pair op Klein
uses (flame-core kernel comment: "Used by Klein/Flux").

Files written (all under `models/flux/`):
- `serenitymojo/configs/flux.json`
- `serenitymojo/models/flux/config.mojo`
- `serenitymojo/models/flux/block.mojo`  (double + single fwd+bwd)
- `serenitymojo/models/flux/parity/block_oracle.py`  (torch-autograd oracle)
- `serenitymojo/models/flux/parity/double_block_parity.mojo`
- `serenitymojo/models/flux/parity/single_block_parity.mojo`
- `serenitymojo/models/flux/__init__.mojo` + `parity/__init__.mojo`

---

## 3. Phase-1 verified result (MEASURED)

Both gates run at REAL Flux dims (D=3072, H=24, Dh=128) with a NON-DEGENERATE
3-axis Flux RoPE table (oracle asserts cos halves differ). Oracle regen + fresh
build, gate output pasted:

DOUBLE block — VERDICT: PASS (36 arms, all cos ≥ 0.99999, 0 nonfinite):
- img_out 0.99999999999982, txt_out 0.99999999999984
- d_img 0.99999999999941, d_txt 0.99999999999946
- img/txt: d_wqkv, d_bqkv, d_wproj, d_bproj, d_wmlp0, d_bmlp0, d_wmlp2, d_bmlp2,
  d_qnorm, d_knorm all ≥ 0.99999999996
- img/txt: d_shift1/scale1/gate1/shift2/scale2/gate2 all ≥ 0.99999999997

SINGLE block — VERDICT: PASS (15 arms, all cos ≥ 0.99999, 0 nonfinite):
- out 0.99999999999989, d_x 0.99999999999830
- d_w1, d_b1, d_w2, d_b2, d_qnorm, d_knorm all ≥ 0.99999999996
- d_shift, d_scale, d_gate all ≥ 0.99999999999

Reproduce:
```
cd /home/alex/mojodiffusion
/home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/flux/parity/block_oracle.py
rm -f serenitymojo.mojopkg
pixi run mojo run -I . serenitymojo/models/flux/parity/double_block_parity.mojo
pixi run mojo run -I . serenitymojo/models/flux/parity/single_block_parity.mojo
```

---

## 4. Remaining phases

### Phase 2 — Stack composition + finite-diff self-consistency

Build `models/flux/flux_stack.mojo` (mirror `klein/klein_stack.mojo`):
- 19 double blocks then 38 single blocks; img/txt joined for the single stack
  (Flux concatenates txt+img into one stream entering the single blocks — see
  flux1_dit.rs forward, the single blocks run on the cat'd sequence).
- STACK-level embeds (NOT per-block): `img_in` (64→3072), `txt_in` (4096→3072),
  `time_in` (256→3072 silu-MLP), `guidance_in` (Dev), `vector_in` (768→3072
  CLIP-pooled), and the per-block modulation projections `img_mod.lin`/
  `txt_mod.lin`/`modulation.lin` (silu(vec) → linear → 6/6/3 chunks). The
  forward oracle for these is the existing `models/dit/flux1_dit.mojo`.
- 3-axis RoPE table builder (`build_rope_2d`, flux1_dit.rs:411-465): per-axis
  omega = 1/theta^(2i/axis_dim), angles = pos⊗omega, cat halves over the 3 axes
  → [N, 64], tile per head. **Keep F32** (memory project_bf16_rope_pattern_audit:
  Flux RoPE is F32, not BF16-cast).
- `final_layer` (adaLN_modulation.1 + linear, 3072→64).
- GATE: composed-stack finite-difference self-consistency (the Klein-runaway
  lesson — `project_klein_runaway_composition_backward_2026-05-29`): single
  block ratio ≈ 1.0 AND full-stack ratio ≈ 1.0. Refutes "per-block-correct ⇒
  composition-correct". Probe mirrors `parity_klein_single_block.rs`.

#### Phase-2 VERIFIED result (MEASURED 2026-06-01)

**Topology verification** (against flux1_dit.rs + models/dit/flux1_dit.mojo):
- `vec = time_in(t*1000) + guidance_in(g*1000) + vector_in(clip_pooled)`; each
  embed is an MLPEmbedder (in_layer → silu → out_layer, both biased). time/
  guidance are fed by `timestep_embedding` (COS-first, t pre-scaled ×1000);
  vector_in is fed the raw CLIP-pooled vector (NOT a sinusoid).
- PER-block modulation is **not shared/frozen** (the key Flux-vs-Klein stack
  difference): each double block does `silu(vec) → img_mod.lin/txt_mod.lin →
  [1,6D]` chunked (shift1,scale1,gate1,shift2,scale2,gate2); each single block
  `silu(vec) → modulation.lin → [1,3D]` (shift,scale,gate). Final layer
  `silu(vec) → adaLN_modulation.1 → [1,2D]` (shift,scale).
- Backward threads the per-block [6D]/[3D] modvec grads (which the Phase-1 block
  already returns as d_shift1.. etc.) back through mod.lin (linear_backward) →
  silu_backward(vec) → d_vec, **summed across every block AND the final layer**,
  then through the 3 embed MLPs to d_timestep / d_guidance / d_vector.
- double→single seam = `concat(1, txt, img)`; final reads only img rows.

**Files (all under `models/flux/`):**
- `models/flux/flux_stack.mojo` — `flux_stack_forward`/`_backward` + the
  embed/vec chain (`FluxStackBase`, `EmbedMlp`, `ModLin`, `DoubleModLin`).
  Reuses the verified Phase-1 block fwd/bwd unchanged.
- `parity/stack_oracle.py` — independent torch-autograd oracle (real H/Dh/D,
  non-degenerate 3-axis F32 RoPE, NUM_DOUBLE=3 + NUM_SINGLE=3 reduced depth).
- `parity/stack_parity.mojo` — composition gate vs torch.
- `parity/stack_finitediff.mojo` — x-path finite-diff self-consistency (NO
  torch; the #1 risk gate).
- `parity/stack_real_smoke.mojo` — residency arithmetic note.

**Composition gate (vs torch) — VERDICT PASS (15 arms, all cos ≥ 0.99999):**
```
out 0.99999999 ; d_img_tokens 0.99999998 ; d_txt_tokens 0.99999998
d_vec 0.99999999 ; d_timestep 1.0 ; d_guidance 1.0 ; d_vector 0.99999999
d0 img d_wqkv/d_wproj/d_wmlp0 ≥ 0.99999998 ; d0 txt d_wqkv 0.99999998
dL img d_wqkv 0.99999998 ; s0 d_w1/d_w2 ≥ 0.99999998 ; sL d_w1 0.99999998
```

**Finite-diff self-consistency — VERDICT PASS (worst |ratio−1| = 0.0108):**
img/txt/vector top-|grad| coords all within 1.1% of 1.0; high-signal vector
coords exact to ~1e-4. NO Klein-style composition defect (which would offset
ALL coords incl. the high-signal ones by a uniform ~0.67/1.5 ratio — decisively
absent). F64 loss accumulation + top-|grad| coord selection + FD_EPS=0.01 used
to lift the small-gradient signal above the F32 forward-noise floor.

**Residency verdict:** flux1-dev transformer = 8.61G params → 34.4 GB F32 /
17.2 GB BF16 weights ALONE, both > 24 GB before activations/grads. Full-
residency real-depth run does NOT fit a 3090; needs block-swap offload +
per-block recompute-in-backward (Ernie/Klein-9B strategy). Parity stays at
reduced depth; composition is proven. No `ops/` primitive missing or edited.

**Reproduce:**
```
cd /home/alex/mojodiffusion
/home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/flux/parity/stack_oracle.py
rm -f serenitymojo.mojopkg
pixi run mojo run -I . serenitymojo/models/flux/parity/stack_parity.mojo
pixi run mojo run -I . serenitymojo/models/flux/parity/stack_finitediff.mojo
pixi run mojo run -I . serenitymojo/models/flux/parity/stack_real_smoke.mojo
```

### Phase 3 — LoRA step (the OneTrainer-grounded phase)

**Authoritative training reference: OneTrainer (Flux.1, NOT Flux2).** Read
line-by-line (source-fidelity rule), files under `/home/alex/OneTrainer/`.

#### 3a. LoRA target set — OT WINS over my earlier sketch

My Phase-1 sketch was "double/single attn qkv+proj + mlp". **OneTrainer's actual
Flux LoRA target set is BROADER: every `nn.Linear` in the transformer.**
Evidence:
- `modules/module/LoRAModule.py:648-656` — `LoRAModuleWrapper.__create_modules`
  iterates `orig_module.named_modules()` and wraps EVERY `Linear|Conv2d` (line
  650), unless a `module_filter` excludes it.
- `modules/modelSetup/FluxLoRASetup.py:93-94` — the transformer is wrapped with
  `LoRAModuleWrapper(model.transformer, "lora_transformer", config,
  config.layer_filter.split(","))`. With the default empty `layer_filter`, NO
  modules are filtered → all linears get LoRA.

The exact OT module→saved-key map (`modules/util/convert/lora/convert_flux_lora.py`),
which IS the trainable target list:

DOUBLE block (`convert_flux_lora.py:6-27`, per `double_blocks.{i}`):
| BFL module key | trains? |
|---|---|
| img_attn.qkv (split q/k/v) | yes |
| txt_attn.qkv (split q/k/v) | yes |
| img_attn.proj | yes |
| txt_attn.proj | yes |
| img_mlp.0, img_mlp.2 | yes |
| txt_mlp.0, txt_mlp.2 | yes |
| img_mod.lin, txt_mod.lin | yes |

SINGLE block (`convert_flux_lora.py:30-39`, per `single_blocks.{i}`):
| BFL module key | trains? |
|---|---|
| linear1 (split q/k/v + proj_mlp) | yes |
| linear2 | yes |
| modulation.lin | yes |

TRANSFORMER-level shared (`convert_flux_lora.py:44-56`):
| BFL key | trains? |
|---|---|
| img_in.proj, txt_in | yes |
| time_in.{in,out}_layer | yes |
| guidance_in.{in,out}_layer | yes |
| vector_in.{in,out}_layer | yes |
| final_layer.adaLN_modulation.1 (swap_chunks=True), final_layer.linear | yes |

So Phase-1's block-level grads cover the per-block targets; the trainer must ALSO
wire LoRA on the modulation linears and the stack-level embedders/final_layer.
Implementation mirrors `klein_stack_lora.mojo` but with the OT target set above.
NOTE: OT's `final_layer.adaLN_modulation.1` uses `swap_chunks=True` — the
shift/scale chunk order is swapped vs the diffusers `norm_out.linear` layout;
honor this on load/save round-trip.

Text-encoder LoRA (out of DiT scope, recorded for the full recipe):
`lora_te1` = CLIP-L (`convert_flux_lora.py:72`), `lora_te2` = T5
(`convert_flux_lora.py:73`). Trained only if `config.text_encoder.train`.

#### 3b. LoRA key naming for save/load round-trip

- In-memory prefix: `lora_transformer` (FluxLoRASetup.py:93). Per-module key =
  `lora_transformer.<module_path>` with the LoRA tensors named by the wrapper
  (`lora_down.weight`, `lora_up.weight`, `.alpha`) — `LoRAModule.py:31-33,114`.
- Saved format: converted via `convert_flux_lora_key_sets()`
  (`FluxLoRASaver._get_convert_key_sets`) to the **diffusers** target names in
  the table above (e.g. `double_blocks.0.img_attn.qkv` →
  `transformer.transformer_blocks.0.attn.to_q`). The kohya/diffusers key set is
  the round-trip contract — match it exactly so OT-trained and our LoRAs load in
  the same tools.
- rank/alpha defaults (`TrainConfig.py:1143-1144`): **lora_rank=16,
  lora_alpha=1.0**. (Our flux.json currently sets alpha=16 to mirror Klein; the
  trainer should default alpha per OT=1.0 unless the user overrides — flag this
  decision to the user when wiring Phase 3.)

#### 3c. Training recipe (OT BaseFluxSetup.py + mixins)

- **Objective: flow matching.** `flow = latent_noise - scaled_latent_image`
  (`BaseFluxSetup.py:305`), target = flow, predicted = transformer output
  unpacked. Loss via `_flow_matching_losses` (`BaseFluxSetup.py:333`).
- **Timestep sampling: logit-normal.** `normal(bias, scale).sigmoid() *
  num_timestep + min` (`ModelSetupNoiseMixin.py:158-160`), discrete
  (`_get_timestep_discrete`).
- **Dynamic shift (seq-len dependent).** `calculate_timestep_shift(h, w)`
  (`FluxModel.py:344-355`): `image_seq_len = (h//2)*(w//2)`,
  `m=(max_shift-base_shift)/(max_seq_len-base_seq_len)`, `b=base_shift-m*base_seq_len`,
  `mu = image_seq_len*m + b`, `shift = exp(mu)`. base_seq_len/max_seq_len/
  base_shift/max_shift come from the FLUX scheduler config (256 / 4096 / 0.5 /
  1.15 standard). Used only when `config.dynamic_timestep_shifting`; else
  `config.timestep_shift` (`BaseFluxSetup.py:244-251`).
- **Noise add: discrete flow.** `_add_noise_discrete` →
  `scaled_noisy = (1-sigma)*latent + sigma*noise` style (sigma from the
  scheduler at the sampled timestep).
- **Guidance:** Dev feeds `config.transformer.guidance_scale`
  (`BaseFluxSetup.py:269`); timestep passed as `timestep/1000`
  (`BaseFluxSetup.py:289`).
- VAE shift/scale applied to latents (`BaseFluxSetup.py:235`).

GATE for Phase 3: LoRA forward WITH/WITHOUT pixel-diff (the l2p lesson —
`project_l2p_no_subject_convergence`), LoRA-B nonzero ratio, grad-flow assert,
and an OT train-step parity (loss match) on a fixed batch. Mirror
`klein_stack_lora_parity.mojo` + `klein_stack_lora_real_smoke.mojo`.

#### 3d. Phase-3 VERIFIED result (MEASURED 2026-06-01)

**LoRA target set re-verified against OneTrainer (convert_flux_lora.py:6-41).**
OT trains EVERY transformer `nn.Linear`. The per-block target linears (the
load-bearing, dominant set) are wired here; OT splits qkv into 3 separate
adapters (`img_attn.qkv.0/.1/.2` -> to_q/to_k/to_v; `linear1.0/.1/.2/.3` ->
to_q/to_k/to_v/proj_mlp), so this port models 3 SEPARATE rank-r adapters on the
3 D-slices of the fused qkv output — the OT-faithful per-projection family (NOT
Klein's one-fused-qkv-adapter simplification). Round-trip uses the BFL module
keys so OT-trained / ai-toolkit LoRAs interop.

Per-block slot set wired (block-projection LoRA; matches Klein/Ernie scope):
- DOUBLE block, per stream {img,txt}: to_q, to_k, to_v, proj (img/txt_attn.proj),
  mlp0 (img/txt_mlp.0), mlp2 (img/txt_mlp.2) -> 12 adapters/block.
- SINGLE block: to_q, to_k, to_v, proj_mlp (linear1.3), linear2 -> 5 adapters/block.
- NOT wired (next increment, same as Klein/Ernie): the modulation linears
  (img_mod.lin/txt_mod.lin/modulation.lin), the embedder MLPs
  (time/guidance/vector_in), and final_layer.{adaLN_modulation.1,linear}. These
  are STACK-level base linears; their grad path is alive (mod-vec grads thread to
  d_vec) but they are frozen for the optimizer. OT trains them too — wiring LoRA
  on the stack-base linears is the documented follow-up.

**alpha DECISION (was OPEN):** flux.json `lora_alpha` set to **1.0** (OneTrainer
`TrainConfig.py:1144` default), changed from the Phase-1/2 Klein-mirror value 16.
Klein-16 remains a valid alternative for stronger adaptation; the parity gate is
alpha-agnostic for grad correctness (oracle + Mojo share scale = alpha/rank).

**Files (all under models/flux/):**
- `lora_block.mojo` — double + single block LoRA wrappers (per-projection split
  q/k/v + proj/mlp on slices; bit-exact base when adapters absent; LoRA d_x
  summed into each projection-input grad). REUSES the verified base block
  saved-activation contract; NO new ops/ primitive.
- `flux_stack_lora.mojo` — `FluxLoraSet` (flat List[LoraAdapter]) +
  `build_flux_lora_set` + resident `flux_stack_lora_forward`/`_backward` over the
  stack + per-block d_A/d_B scatter + `flux_lora_adamw_step` (reuses _lora_adamw)
  + `save_flux_lora`/`load_flux_lora_resume` (reuse save_lora_peft/
  load_lora_for_resume, OT/BFL key names).
- `parity/lora_stack_oracle.py` — independent torch-autograd oracle (LoRA on the
  split target set; B init NONZERO so every grad arm is exercised).
- `parity/lora_stack_parity.mojo` — composition gate (all A/B cos>=0.999 +
  base-no-regression + 0 nonfinite).
- `parity/lora_step_smoke.mojo` — full step: build (B=0) -> fwd -> bwd ->
  global-norm clip -> AdamW -> B 0->nonzero (ratio=1.0) -> save/load byte-exact.

**Composition gate (vs torch) — VERDICT PASS (75 checks, all cos>=0.999, 0
nonfinite):** out (base-no-regression), d_img_tokens, d_txt_tokens, d_vec,
d_timestep, d_guidance, d_vector, AND every adapter's d_A + d_B across 2 double
(img+txt × 6 slots) + 2 single (5 slots).

**Step smoke — VERDICT PASS:** B |.|_1 0.0 -> 1135.78 after AdamW; nonzero-slot
ratio 34/34 = 1.0; nonfinite_lora_grads = 0; global-norm clip(1.0) applied;
save/load max_abs_diff = 0.0 (BYTE-EXACT). NOTE |dA|_1 = 0.0 at step 1 is correct
PEFT behavior: with B=0, d_A = (d_dy@B)ᵀ@x = 0, so only B moves on step 1 (A
starts once B is nonzero) — same as Klein/Ernie step-1 smoke.

**Reproduce:**
```
cd /home/alex/mojodiffusion
/home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/flux/parity/lora_stack_oracle.py
rm -f serenitymojo.mojopkg
pixi run mojo build -I . -Xlinker -lm serenitymojo/models/flux/parity/lora_stack_parity.mojo -o /tmp/flux_lora_parity && /tmp/flux_lora_parity
pixi run mojo build -I . -Xlinker -lm serenitymojo/models/flux/parity/lora_step_smoke.mojo -o /tmp/flux_lora_step && /tmp/flux_lora_step
```

No ops/ primitive was missing or edited (Tenet 1). No shared training/ helper
was modified (LoraAdapter, _lora_adamw, save_lora_peft/load_lora_for_resume all
CALLED). Only models/flux/ + parity/ + flux.json + this plan changed.

---

## 5. Decisions made

| Decision | Owner | Rationale |
|---|---|---|
| Flux-specific block (not Klein reuse) | AGENT-APPROVED | Measured: biases + GELU differ; coordinator directed the assessment |
| Host-boundary block API (List[Float32]) | AGENT-DEFAULT | Matches Klein block + parity-gate contract; device-resident perf refactor deferred to a later increment (review) |
| TArc weight carriers | AGENT-DEFAULT | Lets weight structs stay Copyable holding move-only Tensors (Klein pattern) |
| flux.json alpha=16 (vs OT default 1.0) | AGENT-DEFAULT | Mirrors Klein for now; Phase 3 should default to OT=1.0 — REVIEW |
| LoRA target set = ALL transformer linears | AGENT (OT-sourced) | OT FluxLoRASetup.py:93 + convert_flux_lora.py — OT wins over my qkv+proj+mlp sketch |
| **flux.json lora_alpha 16 -> 1.0** | AGENT-APPROVED | Phase-3 RESOLVED per USER "default to OT unless overridden" (OT TrainConfig.py:1144 = 1.0). Klein-16 noted as alternative. |
| **q/k/v as 3 SEPARATE adapters (not fused)** | AGENT (OT-sourced) | convert_flux_lora.py emits img_attn.qkv.0/.1/.2 as distinct keys; diffusers to_q/to_k/to_v are separate Linears. 3 independent rank-r adapters ≠ one fused rank-r. Round-trip + recipe fidelity. |
| **Phase-3 LoRA scope = per-block projection linears** | AGENT-APPROVED | Same scope as proven Klein/Ernie LoRA (mod/embedder/final frozen). Stack-base-linear LoRA is the documented next increment. |

---

## 6. Phase 4 — INTEGRATED REAL TRAINING LOOP (BLOCKED, measured 2026-06-01)

The deliverable for this phase is a `train_flux_real` that runs a REAL Alina
LoRA run showing loss↓ + LoRA-B imprint (the Klein blueprint:
`training/train_klein_real.mojo`). It is BLOCKED on four missing components, none
of which is a "wire the loop" task — each is itself a port. Measured evidence
below (Tenet 4: a clear blocker report is the honest deliverable; a fabricated
"one synthetic step = done" is not).

### Blocker A — the verified flux LoRA stack is FULLY RESIDENT; flux1-dev does not fit

- `flux_stack_lora_forward`/`_backward` (the Phase-3-verified entry points) take
  `List[DoubleBlockWeights]` + `List[SingleBlockWeights]` — ALL 19 double + 38
  single blocks materialized at once. Block weight structs upcast to **F32**
  device tensors (`Tensor.from_host(..., STDtype.F32)` in block.mojo:205-214).
- MEASURED param count: flux1-dev transformer = **11.9 B params** (matches the
  23.8 GB BF16 checkpoint). Resident weights: **47.6 GB F32 / 23.8 GB BF16** —
  both exceed the 3090's 24 GB *before* activations, LoRA, scratch, rope, cache,
  and the ~0.86 GB CUDA context. The Phase-3 parity gate ran at REDUCED depth
  (NUM_DOUBLE=3, NUM_SINGLE=3), which fits; full depth does not.
- There is **NO flux offload variant.** Klein has
  `klein_stack_lora_{forward,backward}_offload_turbo_moddev_rope_scratch`
  (~1180 lines combined, klein_stack_lora.mojo:699 + :1457) that stream blocks
  one-at-a-time through `offload.turbo_planned_loader.TurboPlannedLoader`,
  keeping only the active block resident. Flux has nothing equivalent. Porting it
  requires:
  1. ~~A flux block plan (`build_flux_block_plan`)~~ — **DONE (2026-06-01,
     this session).** Added `build_flux_block_plan(num_double, num_single)` +
     `build_flux1_dev_block_plan()` to `offload/plan.mojo` (mirrors
     `build_klein_block_plan`; BFL prefixes verified against the on-disk
     `flux1-dev.safetensors` header: `double_blocks.<i>` ×19 then
     `single_blocks.<i>` ×38). Gate added to `offload/plan_smoke.mojo` and runs
     GREEN: count=57, double→single seam at idx 19, last=single_blocks.37,
     cfg_paired visits=114. The remaining 3 sub-items below are still open.
  2. Flux **device-resident-scratch** block fwd/bwd variants — the Klein offload
     loop calls `double_block_lora_forward_device_resident_scratch` etc.; flux
     `lora_block.mojo` only has the host-List-boundary fwd/bwd (the parity
     contract), NOT the device-resident-scratch variants the streaming loop
     needs.
  3. A flux offload LoRA stack fwd AND bwd (~1200 lines mirroring Klein).
  4. A NEW parity gate proving the offloaded path == the resident path.

### Blocker B — no flux1-dev base-weight loader

- There is no `load_flux_stack_base` / `load_flux_double_block_weights` /
  `load_flux_single_block_weights` reading the real `flux1-dev.safetensors`.
  Klein's `models/klein/weights.mojo` (the blueprint) is Klein-key-specific.
- The checkpoint key layout IS exactly the BFL naming the parity work targeted
  (verified: `double_blocks.{i}.img_attn.qkv.weight [9216,3072]`,
  `single_blocks.{i}.linear1.weight [21504,3072]`, `img_in.weight [3072,64]`,
  `txt_in.weight [3072,4096]`, `time_in/guidance_in/vector_in.{in,out}_layer`,
  `final_layer.adaLN_modulation.1 [6144,3072]`, `final_layer.linear [64,3072]`),
  so the loader is a faithful-but-new port, not a research task. For the
  streaming path it must read **BF16 blocks on demand**, not all at once.

### Blocker C — no flux1-dev VAE ENCODER (prepare half is also blocked)

- flux1-dev uses the original Flux autoencoder: **16-channel latent at /8**
  (confirmed by `img_in.weight [3072, 64]` = 16ch × 2×2 patchify = 64).
- The only VAE *encoder* in the Mojo tree is `models/vae/klein_encoder.py` =
  the **FLUX.2 VAE** (128-channel packed latent at /16) — the WRONG latent space
  for flux1-dev. The weights exist (`/home/alex/.serenity/models/vaes/ae.safetensors`,
  the standard 16-ch Flux AE) but no Mojo encoder loads them. Every other file in
  `models/vae/` is a *decoder*. So `flux_prepare_alina` cannot VAE-encode Alina
  images to the correct 16-ch /8 latent without first porting a flux1 VAE encoder.

### NOT blockers (assets + text path confirmed present)

- DiT checkpoint: `/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors`
  (23.8 GB, BFL keys verified). ✅
- T5: flux1-dev uses **t5-v1_1-xxl** (24-layer, 4096-dim). Present at
  `/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors` +
  `t5xxl_fp16.tokenizer.json`. `models/text_encoder/t5_encoder.mojo`
  `T5Config.t5_xxl()` matches exactly (24 / 4096 / 64 heads / d_ff 10240). ✅
- CLIP-L (pooled → vector_in 768): present at
  `models--openai--clip-vit-large-patch14`; `clip_encoder.mojo ClipConfig.clip_l()`
  matches; pooled = EOS-position row. ✅
- Dataset raw images: `/home/alex/datasets/AlinaAignatova`. ✅
- GPU idle (861 MiB used, 52 °C) — no contention. ✅

### What "done" requires (ordered, each is real port work)

1. **flux1 VAE encoder** (16-ch /8, loads `ae.safetensors`) → `flux_prepare_alina`
   can produce a correct cache (latent [1,16,64,64] for 512px + T5 joint [1,512,4096]
   + CLIP pooled [1,768]). Extend `klein_dataset` for the `clip_pooled` key.
2. **flux base-weight loader** (Blocker B) reading real flux1-dev keys.
3. **flux offload LoRA stack fwd+bwd** + device-resident-scratch block variants +
   flux block plan (Blocker A) + a resident-vs-offload parity gate.
4. **train_flux_real** wiring the verified pieces into the Klein-style loop,
   then a bisected real run (5→20 steps) under the 78 °C / 20-min caps showing
   PROG loss↓ + LoRA-B 0→nonzero growth.

Estimated scope: comparable to the Klein offload + loader + VAE-encoder work
combined (multi-session). The Phase 1-3 parity foundation (block, stack, LoRA
grads) is solid and REUSED unchanged by step 3-4; the gap is the
memory-management + IO + encoder infrastructure, not the math.

### Decisions made this phase

| Decision | Owner | Rationale |
|---|---|---|
| STOP and report rather than fabricate a synthetic/reduced-depth "run" | AGENT-APPROVED | Tenet 4 + EMPOWERMENT: a measured blocker is the honest deliverable; a reduced-depth or synthetic-weight "real run" would be a false "done" (the exact failure the brief forbids). |
| Do NOT hack a partial-depth resident run to fake loss↓ | AGENT-DEFAULT | A 6-block resident flux would "train" but is not flux1-dev; it would mislead. Flagged for review. |

### Session 2026-06-01 (Phase-4 re-confirmation + first prerequisite landed)

Re-verified all four Phase-4 blockers against the current tree (NOT stale):

- **Blocker A** — `flux_stack_lora_forward`/`_backward` still take resident
  `List[DoubleBlockWeights]`/`List[SingleBlockWeights]`, F32-upcast in
  block.mojo:205-214. No `*_offload_*` variant exists in `flux_stack_lora.mojo`
  (Klein's is 1876 lines). STILL BLOCKED — but sub-item 1 (block plan) is now
  DONE + GREEN (see Blocker A above; `offload/plan.mojo` +
  `offload/plan_smoke.mojo`). Sub-items 2-4 (device-resident-scratch flux block
  variants, offload stack fwd/bwd ~1200 lines, resident-vs-offload parity gate)
  remain.
- **Blocker B** — no `models/flux/weights.mojo`; no real flux1-dev loader. Klein's
  blueprint `models/klein/weights.mojo` = 417 lines. STILL BLOCKED.
- **Blocker C** — `vae/vae_encode_general.mojo` now EXISTS but is
  **synthetic-weight only** (`with_synthetic_weights`, no `ae.safetensors`
  loader; CH=32 gate config, not flux1-dev's real AutoencoderKL channels). The
  AutoencoderKL-2D forward math is present and reusable; the missing piece is a
  real-weight loader + the correct 16-ch/128-CH flux1 config. STILL BLOCKED.
- **NOT blockers** (re-checked): flux1-dev.safetensors (23.8 GB, 19 double + 38
  single, BFL keys), ae.safetensors (335 MB), T5xxl/CLIP-L present; GPU idle
  (876 MiB used, 52 °C, 0 %). A real run COULD proceed the moment A+B+C close.

VERDICT: `train_flux_real` real run NOT achievable this session — the deliverable
is gated on A+B+C, each a genuine port (offload stack + base loader + VAE encoder
loader), not loop-wiring. Reporting the exact blocker per the brief
("else STOP and report the exact blocker. No faked/synthetic runs"). One real,
gateable prerequisite landed (flux block plan) to advance Blocker A.

| Decision | Owner | Rationale |
|---|---|---|
| Land `build_flux_block_plan` + smoke gate this session | AGENT-DEFAULT | The one Blocker-A sub-item that is mechanical, GPU-fit-independent, and gateable now; mirrors `build_klein_block_plan`, prefixes verified against the real checkpoint. Advances the critical path without faking a run. |

---

## 7. Phase 4 — Historical Flux.1-dev Trainer Notes, Revalidate Later

Current status: Flux.1-dev trainer plumbing exists, but it is not
production-tested and it is not Flux.2/Klein/dev2. Re-run this section later as a
fresh acceptance gate using the mandatory trainer runtime contract. The notes
below are retained as historical implementation context.

### Blockers closed (all built + gated BEFORE the trainer)
- **A — offload LoRA stack.** `flux_stack_lora.mojo` now has
  `flux_stack_lora_forward_offload` / `_backward_offload` (stream one block at a
  time via `TurboPlannedLoader`, keep only the active block + the host-list
  activation tape resident). Equivalence gate `parity/flux_offload_equiv_parity.mojo`
  PASSES (offload out + every adapter d_A/d_B cos>=0.9999 vs resident, 0
  nonfinite). Real-weight memory smoke `parity/flux_offload_mem_smoke.mojo`
  streams real flux1-dev blocks finite under 24 GB.
- **B — base-weight loader.** Added `load_flux_stack_base` to
  `models/flux/weights.mojo`: loads the NON-streamed FluxStackBase from the real
  checkpoint — img_in/txt_in, the 3 embed MLPs (time/guidance/vector_in), the
  PER-BLOCK modulation linears (`{double,single}_blocks.*.{img,txt_,}mod*.lin`),
  and final_layer.{adaLN_modulation.1,linear}. Dims derived from stored shapes;
  keys verified against the on-disk header. (Block attn/mlp weights are streamed,
  not loaded here.) The base is ~12.3 GB F32 resident (the per-block mod.lin
  weights dominate; they live in FluxStackBase, not in the streamed blocks).
- **C — flux1 VAE encoder.** `vae/flux_vae_encoder.mojo` (`FluxVaeEncoder`) loads
  the real `ae.safetensors` (16-ch /8 Flux.1 AE), gated cos 0.9999985.

### The trainer — `serenitymojo/training/train_flux_real.mojo`
Cache reader (prepare_flux.rs schema: `latent` raw [1,16,64,64], `t5_embed`
[1,seq,4096] padded to N_TXT=512, `clip_pool` [1,768]) → recipe (VAE
shift/scale 0.1159/0.3611, pack_latents channel-major patchify, sigma=(idx+1)/1000,
t_model=idx/1000 pre-scaled ×1000, target=noise−latent, guidance=3.5×1000,
logit-normal shift=1.0) → `flux_stack_lora_forward_offload` (FULL 19+38 depth,
streamed) → MSE loss → `flux_stack_lora_backward_offload` → global-norm clip(1.0)
→ `flux_lora_adamw_step` → PROG log → `save_flux_lora`. Builds with
`-Xlinker -lm -Xlinker -lcuda` (offload uses cuMemcpy/cuMemGetInfo). A
FIXED_SIGMA_SMOKE mode pins sample+timestep+noise so a correct backward MUST
drive loss down monotonically (same correctness probe as zimage/anima).

### Historical smoke log (not current acceptance evidence)
```
=== Flux (flux1-dev) REAL LoRA training loop (block-swap offload) ===
  depth: NUM_DOUBLE=19 NUM_SINGLE=38 (FULL flux1-dev)
  tokens: N_IMG=1024 N_TXT=512 S=1536
[load] base resident                         # ~12.3 GB F32 FluxStackBase
[load] offload loader opened ( 57 blocks)
[lora] adapters: 418  (12 x 19 double + 5 x 38 single)
[lora] LoRA-B |.|_1 at init = 0.0  (expect 0.0)
PROG step= 1  loss= 1.4943842  grad= 0.005886842  lr= 0.0001
     loraB_sum= 2418.2038  loraB_nonzero= 418 / 418  nonfinite= 0  secs= 323.792
```
- LoRA-B grew **0 → 2418.2** with **418/418** adapters nonzero (EVERY trained
  projection across the full 19+38 depth got a gradient through the offload bwd).
- loss finite, **0 nonfinite** LoRA grads. Peak GPU ~20.3 GB (base 17.5 incl.
  mmap + streamed block + activations) under the 24 GB budget; thermals 52→63 °C
  (under the 78 °C cap). cache: cache/eri2_flux_512_smoke (real Rust-encoded flux
  latent+T5+CLIP).
- Per-step cost ~324 s: the offload path streams the WHOLE 24 GB checkpoint from
  disk on the forward AND again on the backward (turbo stat h2d_mib≈24468 per
  pass), at the host-list block boundary. CORRECTNESS is proven; offload
  THROUGHPUT (device-resident-scratch block variants to avoid the host round-trip
  + cut the double-stream) is the documented next increment.

### Prepare — `serenitymojo/pipeline/flux_prepare.mojo`
REAL Mojo Flux-VAE encode of the Alina staged images → RAW latent [1,16,64,64]
(builds GREEN). The T5 text half is the one honest gap: `t5xxl_fp16.tokenizer.json`
is a SentencePiece **Unigram** model, and the in-tree Mojo tokenizer is byte-level
BPE only (Qwen3) — so raw-caption T5 tokenization is not yet possible in pure
Mojo. The prepare therefore sources the REAL T5/CLIP embeddings from the existing
Rust flux cache (the same fast-path anima_prepare/zimage_prepare use for their
un-ported text halves) and pairs them with the real Mojo VAE latents. A fully
self-contained Mojo text encode needs a Unigram-tokenizer port (next increment).

### Files written this session
- `serenitymojo/models/flux/weights.mojo` — +`load_flux_stack_base` (+helpers).
- `serenitymojo/training/train_flux_real.mojo` — the real offload training loop.
- `serenitymojo/pipeline/flux_prepare.mojo` — Mojo VAE encode + real text reuse.

### Decisions made this phase
| Decision | Owner | Rationale |
|---|---|---|
| Use the existing real Rust flux cache (cache/eri2_flux_512_smoke) for the train run | AGENT-DEFAULT | T5 Unigram tokenizer not ported → cannot encode raw captions in pure Mojo. The cache holds REAL VAE+T5+CLIP encodes at the exact prepare_flux schema. The deliverable is "real flux1-dev train run with loss + LoRA-B growth", and the model/training are 100% real on this cache. Flagged for review. |
| Per-block mod.lin lives in FluxStackBase (resident ~12.3 GB), not streamed | AGENT-DEFAULT | Matches the verified offload stack contract (only attn/mlp blocks stream). Fits 24 GB with the streamed block. Streaming the mod.lin too is a memory optimization, not a correctness need. |
| FIXED_SIGMA_SMOKE default True for the first run | AGENT-DEFAULT | The canonical monotone-loss correctness probe (zimage/anima precedent). Production sets it False for per-step sample/timestep variance. |
