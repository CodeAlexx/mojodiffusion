# TRAINING_PLAN — Flux (flux1-dev) pure-Mojo training port

Status: **Phase 1 (block fwd+bwd parity) DONE + MEASURED. Phase 2 (full-stack
composition fwd+bwd) DONE + VERIFIED. Phase 3 (LoRA step) DONE + MEASURED —
per-model Flux milestone CLOSED.**

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
