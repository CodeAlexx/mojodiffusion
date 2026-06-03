# TRAINING_PLAN_ernie.md — ERNIE-Image pure-Mojo training port

Status: Phase 1 DONE + Phase 2 (E1/E2/E3) DONE + E4 (LoRA) DONE + **E5
(train_ernie_real) DONE & REAL-RUN VERIFIED** (2026-06-01) + **E5 speed target
hit** (2026-06-03) + **Mojo Mistral3B prompt path landed** (2026-06-03).
Single-block AND full-36-layer-stack fwd+bwd are parity-clean vs torch at real
hidden dims (H=32, Dh=128, D=4096). The real training loop runs on the real
36-layer checkpoint + real BoxJana cache: loss finite, grads finite & growing,
LoRA-B 0→nonzero (learning), nonfinite=0, PEFT/state save works, and the
built-in sampler writes 1024x1024 validation samples from Mojo-Mistral prompt
sidecars. The production hot path now keeps all ERNIE block matrices
BF16-resident on the 3090 and measures ~2.6-2.7 s/step on local real-cache
smokes. The 2500-step BoxJana convergence run completed with final loss
`0.4701`, grad_norm `0.0549`, `2.6s/step`, elapsed `1:54:53`, final LoRA-B
|.|_1 `509872.78111666883`, and `252/252` LoRA-B slots nonzero. E6 (full-FT)
deferred. Builder→skeptic→bugfix discipline per phase.

## What Phase 2 shipped (VERIFIED this session, 2026-06-01)

| File | Role | Verification |
|---|---|---|
| `models/ernie/weights.mojo` (extended) | `ErnieStackBase` (input proj + final layer, device-resident) + `load_ernie_stack_base` + `load_ernie_all_blocks` + `verify_ernie_stack_shapes` | real shapes loaded RC=0 (base + blocks 0 & 35) |
| `models/ernie/ernie_stack.mojo` | `ernie_stack_forward`/`backward` (IMAGE-FIRST concat, shared-AdaLN broadcast→6 grads SUMMED across blocks, layer_norm→modulate→final_linear→narrow img) + streamed (offload) variants | composition gate + real smoke PASS |
| `models/ernie/parity/stack_oracle.py` | torch autograd reference (L=3, S=8, reduced F/in_ch/text/out) | RC=0 |
| `models/ernie/parity/stack_parity.mojo` | composition cos gate vs torch | VERDICT PASS — out/d_x-tokens/d_final_lin/d_f_scale/d_f_shift/**summed shared-AdaLN [6D]**/deepest+shallowest per-block d_wq+d_wdown ALL cos≥0.99999999 |
| `models/ernie/parity/stack_real_smoke.mojo` | 36-layer REAL-weight finite + mem smoke (streamed blocks) | VERDICT PASS — all FINITE, NO OOM, peak 0.72 GiB, 53°C |

COMPOSITION PROVED (the Klein composition-bug lesson — see memory
project_klein_runaway_composition_backward): the composed backward = grad of the
composed forward at depth L=3, including the genuinely-new SHARED-AdaLN grad that
SUMS across all blocks (cos 0.9999999999999001). Real 36-layer fwd+bwd on the
real checkpoint is finite with no OOM.

### MEMORY FINDING (updated 2026-06-03): full F32 residency of all 36 blocks is
~31 GB and does NOT fit a 24 GB 3090 (measured: `load_ernie_all_blocks` OOMs at
~block 22). BF16/norm-F32 residency DOES fit: `load_ernie_all_blocks_bf16_normf32`
loads all 36 blocks and leaves ~5.2 GB free before the step on the local 3090.
The trainer's production hot path now uses resident BF16 block matrices + F32
norm vectors (`ernie_stack_lora_forward_resident_device` /
`ernie_stack_lora_backward_resident_device`). The older streamed F32 paths remain
as correctness/fallback paths; they are no longer the speed target.

ERNIE-Image = Baidu single-stream DiT: hidden 4096, 36 layers, 32 heads, head
dim 128, FFN 12288, SHARED AdaLN modulation (one mod computed once, broadcast to
all blocks), image-first/text-second concat, half-split 3-axis RoPE (axes
32/48/48, theta 256), QK RMSNorm, GELU-gated MLP. Latent [1,128,64,64] → 4096
img tokens + ≤256 text tokens = seq 4352. Model timestep = sigma*1000.
Weights root `/home/alex/models/ERNIE-Image` (transformer = 409 tensors, 2 shards).

## What Phase 1 shipped (VERIFIED this session)

| File | Role | Verification |
|---|---|---|
| `serenitymojo/configs/ernie_image.json` | dims+recipe (config-driven binding) | `ernie_image()` reads it RC=0 |
| `serenitymojo/models/ernie/config.mojo` | TrainConfig accessor (mirrors klein/config.mojo) | build+run PASS |
| `serenitymojo/models/ernie/weights.mojo` | safetensors→ErnieBlockWeights (one block) | `load_block_smoke` loaded real block-0 (to_q [4096,4096], wgate [12288,4096], q_norm [128]) RC=0 |
| `serenitymojo/models/ernie/block.mojo` | block fwd (saving acts) + hand-chained bwd | block_parity 19/19 cos≥0.99999 |
| `serenitymojo/models/ernie/parity/block_oracle.py` | torch autograd reference | RC=0 |
| `serenitymojo/models/ernie/parity/block_parity.mojo` | cos gate vs torch | VERDICT PASS |

The ERNIE block reuses the EXACT `ops/` backward arms the Klein single block
uses — **no new `ops/` primitive was needed**. ERNIE's deltas from Klein are
all already covered: `gelu`/`gelu_backward` (not swiglu), `rope_halfsplit_full`
+ `rope_backward(interleaved=False)` (not interleaved), and separate
to_q/to_k/to_v (3 `linear_backward` + grad-sum into d_sa_in) instead of fused
qkv. This satisfies Tenet 1 (the backward primitives live in `ops/`, reusable by
every model — nothing inlined in the model file).

## Forward-gate status (Phase 1 prerequisite)

- Mojo block-0 inference forward exists (`models/dit/ernie_image.mojo`
  `block0_smoke_forward`); the prior sin/cos channel-order bug (skeptic A2) was
  FIXED (`time_embed` calls `timestep_embedding_sin_first`).
- BUT before this session the block-0 forward was NEVER parity-verified against a
  torch/diffusers oracle (skeptic worklist items 2/3/6 were left open — only a
  synthesized-AdaLN smoke ran, no cos compare). The new `block_parity.mojo` is
  the first such gate and it PASSES (out cos 0.99999999999992 at real dims).
- The full 36-layer stack, final norm/projection/unpatchify, and full-model
  forward do NOT exist in Mojo yet (see Phase E2).

## Ordered remaining phases (map to Klein template)

### E1 — Full-block weight loader for all 36 layers  [DONE 2026-06-01]
- `weights.mojo` extended: `ErnieStackBase` + `load_ernie_stack_base` (x_embedder
  patch_w[hidden,in_ch]+bias, text_proj, time_embedding.linear_{1,2}+bias,
  adaLN_modulation.1+bias, final_norm.linear+bias, final_linear+bias) +
  `load_ernie_all_blocks` (loop over the Phase-1 `load_ernie_block_weights`) +
  `verify_ernie_stack_shapes` (asserts base + blocks 0 & 35 real shapes, RC=0).
  Both shards spanned by `ShardedSafeTensors` (409 tensors). NOTE patch_proj is
  stored [hidden,in_ch,1,1]; loaded reshaped to [hidden,in_ch] (k=1 conv = linear).

### E2 — Full-stack forward + final layer  [DONE 2026-06-01]
- `models/ernie/ernie_stack.mojo` built (mirrors `klein_stack.mojo`): patch embed
  + text proj + concat (IMAGE-FIRST — note: differs from Klein's txt-first; final
  layer reads narrow(1,0,n_img)) + shared-AdaLN broadcast to all blocks + 36×
  `ernie_block_forward` + final non-learnable `layer_norm` → `modulate(ln_x,
  f_scale, f_shift)` → `final_linear`(+bias) → narrow img rows. The 6 shared
  mod-vec grads SUM across all blocks (`_modvec6` accumulator). f_scale/f_shift
  come from `c@final_norm.linear` chunk(2) — passed in PRECOMPUTED here (the
  final_norm.linear / adaLN / time MLP backprop is the E5 link, deferred exactly
  as klein_stack defers the modulation-MLP link).
- Composition gate (`stack_parity.mojo`, L=3) PASSES: proves composed bwd = grad
  of composed fwd, including the summed shared-AdaLN grad. Real 36-layer smoke
  (`stack_real_smoke.mojo`) PASSES finite + no-OOM via the streamed path.

### E3 — RoPE table layout reconciliation  [RESOLVED 2026-06-01]
- VERDICT: real ERNIE RoPE = half-split pairing (i, i+half) on an
  interleaved-doubled FULL-WIDTH table where cos[i] != cos[i+half] (confirmed
  against diffusers transformer_ernie_image.apply_rotary_emb line-by-line). NOT
  the (2i,2i+1) interleaved arm.
- Forward `rope_halfsplit_full` was ALREADY correct (reads cos[i] AND cos[i+half]).
- The BUG was the backward: the block fed a HALF-WIDTH table to
  rope_backward(interleaved=False), which aliases one angle per pair → correct
  only on a degenerate cos[i]==cos[i+half] table (which is what the old oracle
  built — a tautology). On the real table d_x cos collapsed to ~0.23.
- FIX (Tenet 1, in ops/): added `rope_halfsplit_full_backward` to
  ops/rope_struct_backward.mojo (full-width table, reads both halves) + gate
  ops/parity/rope_halfsplit_full_parity.mojo (FAILS old arm, PASSES new). block.mojo
  now calls it with the same full-width cos/sin as the forward. Oracle rebuilt on
  the REAL interleaved-doubled 3-axis table + real row/col/text positions. Block
  gate: all 19 tensors cos >= 0.99999 on the real table.
- HISTORICAL (pre-fix):
- `build_ernie_rope_tables` (`models/dit/ernie_image.mojo:413`) builds the
  doubled cos/sin with INTERLEAVED doubling `[c0,c0,c1,c1,...]` (appends each
  angle twice consecutively), matching the Rust ref `precompute_rope_cos_sin`
  and the Rust fused kernel (which indexes `COS[d]` per output channel d).
- The Mojo FORWARD op `rope_halfsplit_full` pairs channel i with i+half and reads
  `cos[i]`/`cos[i+half]` SEPARATELY; the Mojo BACKWARD `rope_backward(False)`
  reads only the half-width `[rows, D/2]` table and assumes half-split pairing
  (i, i+D/2). These two are SELF-CONSISTENT only if the doubled table satisfies
  `cos[i] == cos[i+half]` (half-split doubling `[c0..c_{half-1}, c0..c_{half-1}]`).
- The block parity gate (Phase 1) PROVES the fwd+bwd chain is correct under a
  half-split-doubled table (the oracle builds `cat([cos_half, cos_half])`).
- DECISION NEEDED (AGENT-FLAGGED, user/skeptic to confirm): is ERNIE's true RoPE
  pairing (i, i+half) [half-split] or (2i, 2i+1) [interleaved]? Read the diffusers
  `transformer_ernie_image` `apply_rotary_emb` / `rotate_half` line-by-line. If
  half-split: change `build_ernie_rope_tables` to half-split doubling (then fwd
  `_full` + bwd are consistent, and the block gate's table matches the real one).
  If interleaved: the block must use `rope_interleaved` + `rope_backward(True)`
  with a half-width table built as `[θ0,θ1,...,θ_{half-1}]` — a one-line swap in
  `block.mojo` (the interleaved arm is already imported by Klein and gated).
  Either way it is a primitive-selection fix, not new code. This is the #1
  blocker for the skeptic/bugfix loop on the real-weight path.

### E4 — LoRA-on-projection variant  [DONE 2026-06-01]
- `models/ernie/lora_block.mojo`: `ernie_lora_fwd`/`ernie_lora_bwd` (byte-identical
  to train_step._lora_fwd/_lora_bwd + the d_x term the trainer drops), `ErnieBlockLora`
  (7 optional adapters), `ernie_block_lora_forward`/`ernie_block_lora_backward`
  (mirror block.mojo exactly, apply adapters to the 7 projection outputs and SUM the
  LoRA d_x into each projection-input grad). Reduces to base bit-for-bit when an
  adapter is absent (ernie_lora_apply returns base_y unchanged). NO new ops/.
- `models/ernie/ernie_stack_lora.mojo`: `ErnieLoraSet` (flat List[LoraAdapter],
  7×num_layers, slot order Q,K,V,O,gate,up,down) + `build_ernie_lora_set` + RESIDENT
  `ernie_stack_lora_forward`/`backward` (composition gate) + STREAMED
  `ernie_stack_lora_forward_streamed`/`backward_streamed` (fallback/correctness path)
  + DEVICE hot paths (`ernie_lora_set_to_device`,
  `ernie_stack_lora_forward_streamed_device`/`backward_streamed_device`,
  `ernie_stack_lora_forward_resident_device`/`backward_resident_device`) + one bulk
  LoRA-grad D2H gather + scatter of the per-block 7-slot d_A/d_B into flat grads +
  `ernie_lora_adamw_step` (reuses
  _lora_adamw) + `save_ernie_lora`/`load_ernie_lora_resume` (PEFT keys
  `layers.<i>.self_attention.{to_q,to_k,to_v,to_out.0}` / `.mlp.{gate_proj,up_proj,
  linear_fc2}` — inverse of inference-flame ernie_image.rs lora.apply; reuses
  save_lora_peft/load_lora_for_resume).
- ERNIE LoRA targets (confirmed against `ernie_image.rs:649-708` lora.apply sites):
  per layer `self_attention.{to_q,to_k,to_v,to_out.0}` + `mlp.{gate_proj,up_proj,
  linear_fc2}` — SEVEN separate adapters (un-fused q/k/v).
- COMPOSITION LoRA parity (`parity/{lora_stack_oracle.py,lora_stack_parity.mojo}`,
  L=3 real H=32/Dh=128/D=4096, RANK=8 ALPHA=16): VERDICT PASS — out + token grads +
  d_f_scale/d_f_shift + summed shared-AdaLN [6D] + ALL 42 LoRA A/B grads (7 slots ×
  3 layers) cos ≥ 0.99999999999, nonfinite=0. Base no-regression: the base
  `stack_parity` gate still PASSES unchanged; absent adapter = bit-exact base path.
- END-TO-END LoRA STEP smoke (`parity/lora_step_smoke.mojo`): 21 adapters, LoRA-B
  |.|_1 0.0 at init → 494.9 after AdamW, all 21 B slots nonzero (ratio 1.0), grads
  finite, global-norm clip 2.006→1.0, save/load max_abs_diff 0.0 (BYTE-EXACT). PASS.
  (|dA|_1=0 at step 0 is correct PEFT behavior — d_A path = d_t@x where d_t=d_dy@B=0
  while B=0; A starts moving on step ≥2 once B is nonzero.)

### E5 — train_ernie_real loop  [DONE + REAL-RUN VERIFIED 2026-06-01]
- `serenitymojo/training/train_ernie_real.mojo` built (RC=0) and ran a REAL
  multi-step LoRA training run on the real 36-layer ERNIE checkpoint
  (`/home/alex/models/ERNIE-Image/transformer`, 2 shards) + the real cache
  `EriDiffusion-v2/cache/boxjana_ernie_512_FIXED`. TRANSLATION of
  `train_ernie.rs` (not invention): same cache schema (latent[1,128,32,32]
  F32 + text_embedding[1,512,3072] + text_real_len), same logit-normal σ sample
  (shift=1 identity), same flow target `noise-latent`, same integer timestep fed
  to the DiT (`sigma_idx`, NOT σ*1000 — train_ernie.rs:956), same MSE loss +
  `d_loss=(2/N)(pred-target)`, same global-L2 clip @1.0, same AdamW, same PEFT
  save keys.
- COMPTIME real dims: N_IMG=1024 (32×32 latent), N_TXT=256 (fixed trim of the
  PAD-padded cache text; observed real_len ≤ ~203 so all real tokens kept),
  S=1280, D=4096, F=12288, 36 layers, rank=16 α=1.0 for OneTrainer-default
  scale parity. Current hot path uses
  BF16/norm-F32 block residency: all 36 base blocks load once, LoRA A/B upload to
  device each step, frozen base backward uses dX-only helpers, and all LoRA grads
  return through one bulk D2H gather. Fit receipt: after resident block load,
  free VRAM was 5.20-5.25 GB on the local 3090.
- The deferred E2/E5 shared-AdaLN SOURCE is now BUILT in the trainer
  (`_shared_adaln_source`): from the resident `ErnieStackBase` weights +
  `sigma_idx` it computes `c=time_embed(sigma_idx)` (sin-first MLP) →
  `mv=chunk6(silu(c)@adaLN_modulation.1+b)` and
  `[f_scale,f_shift]=chunk2(c@final_norm.linear+b)`. Mirrors ernie_image.rs:519-552
  / ernie_image.mojo time_embed/shared_adaln. All F32.
- REAL-RUN RESULT (tool output, no faking): nonfinite=0 every step; grads finite
  and growing; every adapter slot moves off the PEFT zero-init, i.e. the LoRA is
  LEARNING. Latest 3-step BF16-resident device run (2026-06-03):
  step0 loss=0.6786737 grad=0.0016256 secs=2.7459, step1 loss=1.0965953
  grad=0.0023452 cumulative=5.3837, step2 loss=0.47514582 grad=0.0018576
  cumulative=8.0317. Mean measured loop speed = **2.68 s/step**; all 252/252
  LoRA slots nonzero; final LoRA-B |.|_1=11546.05. Save →
  `/tmp/ernie_probe_lora_resident_bf16_3step.safetensors` with PEFT/ai-toolkit
  keys. Per-step raw loss varies with sampled σ; this matches the expected
  σ-dependent magnitude rather than indicating a regression.
- SPEED LADDER (same real checkpoint/cache, 2026-06-03): legacy host streamed
  path started at ~175 s/step; frozen-base dX-only stripping reduced that to
  ~161 s/step; device LoRA + bulk grad D2H reduced it to ~49 s/step; BF16
  streamed blocks reduced it to ~43.3 s/step; BF16-resident all-block path hit
  2.68 s/step. This is the target class the user wanted: "few seconds per step."
- ONETRAINER BASELINE (2026-06-03): local OneTrainer run on
  `/home/alex/models/ERNIE-Image` + `/home/alex/eri2`, batch=2, LoRA, transformer
  `INT_W8A8`, text encoder `FLOAT_8`, train/output BF16, `LOGIT_NORMAL`,
  rank=16 alpha=1.0 by TrainConfig default, 252 selected LoRA layers.
  `compile=true` hit a torch-inductor ERNIE attention
  backward stride bug at step 3
  (`attn_bias.stride(1)=1500625 should be multiple of 4`), so the baseline target
  is the `compile=false` run:
  `configs/ernie_eri2_100step_baseline.json` +
  `scripts/benchmark_ernie_100.py`. Result: 100/100 steps completed, final LoRA
  saved to `/home/alex/OneTrainer/output/ernie_eri2_100step_baseline/lora.safetensors`.
  Final loss=0.7464916, smooth loss=0.6612294, mean loss=0.6612294,
  median loss=0.6428585, final grad_norm=0.0014650, median grad_norm=0.0010599,
  peak sampled GPU memory=13083 MiB (torch max allocated=10777 MiB,
  reserved=11088 MiB), mean sampled GPU util=92.2%. Timing: warmed median
  2.77 s/step; warmed mean after first 20 intervals 2.76 s/step; warmed p90
  2.90 s/step. Full train-loop wall including epoch/cache overhead was 355.8 s
  (3.56 s/step); all callback intervals excluding only the first averaged
  3.21 s because warmup/epoch-boundary intervals are included. This is the
  practical Mojo ERNIE speed target: first match OneTrainer's stable
  ~2.7-2.9 s/step, then optimize below it.
- MISTRAL3B TEXT ENCODER: LANDED for prompt/cap sidecars. The train loop still
  consumes the historical BoxJana cache for latents/text during this convergence
  run, but validation prompts are encoded by
  `serenitymojo/models/text_encoder/mistral3b_encoder.mojo` through
  `serenitymojo/pipeline/ernie_precache_sample_prompts.mojo`. Tokenizer parity
  passes for empty/studio/beach prompts; encoder smoke loads the local ERNIE
  text encoder and emits `[1,256,3072]`.
- COMPLETED 2500-STEP RUN (launched/completed 2026-06-03):
  `tail -f /home/alex/mojodiffusion/logs/ernie_boxjana_2500_20260603_033140_setsid.log`.
  Output target:
  `/home/alex/mojodiffusion/serenitymojo/output/ernie_boxjana_2500/ernie_lora.safetensors`.
  Step-500 cadence passed: PEFT save and trainer-state save wrote
  `ernie_lora_step500.safetensors*`, both validation prompts sampled, the trainer
  reloaded `.state.safetensors`, and training continued. Step-500 LoRA sample
  shifts: studio `pixel_l1=0.040247522`, beach `pixel_l1=0.01677309`.
  Step-1000 cadence also passed and resumed. Step-1000 LoRA sample shifts:
  studio `pixel_l1=0.07029573`, beach `pixel_l1=0.025967646`.
  Step-1500 cadence passed with PEFT save, trainer-state save, both 1024x1024
  validation samples, and state reload. Step 1500 logged loss `0.1961`,
  grad_norm `0.0080`, `2.6s/step`, elapsed `1:08:26`; sample shifts: studio
  `pixel_l1=0.07712812`, beach `pixel_l1=0.029465288`.
  Step-2000 cadence passed with the same save/sample/reload sequence. Step 2000
  logged loss `0.7755`, grad_norm `0.0170`, `2.6s/step`, elapsed `1:31:40`;
  sample shifts: studio `pixel_l1=0.05731641`, beach
  `pixel_l1=0.043643314`.
  Step-2500/final cadence passed with final PEFT/state save, both 1024x1024
  validation samples, and final output save. Step 2500 logged loss `0.4701`,
  grad_norm `0.0549`, `2.6s/step`, elapsed `1:54:53`; sample shifts: studio
  `pixel_l1=0.05692175`, beach `pixel_l1=0.040390998`; final LoRA-B |.|_1
  `509872.78111666883`, nonzero slots `252/252`.
- FOLLOW-UPS (not blockers): (1) convert this smoke into the production
  config/dataset loop (batch 2, 512 buckets, train shift 1.0, LOGIT_NORMAL);
  (2) add masked attention or same-length buckets before claiming OneTrainer
  batch-2 parity; (3) port Mojo Mistral3/tokenizer and VAE prepare so Rust/Python
  cache files are not required; (4) ring allocator/saved-activation reuse remains
  a later memory-DX project, not needed to hit the current speed target.

#### Historical E5 scope (now satisfied)
- The per-step LoRA mechanics are PROVEN (E4): `ernie_stack_lora_forward_streamed`
  → `ernie_stack_lora_backward_streamed` → host global-norm clip →
  `ernie_lora_adamw_step` → `save_ernie_lora`. E5 = wrap these in the real loop
  (dataset iter, σ-sample, flow target, logging cadence, checkpoint/resume) mirroring
  `training/train_klein_real.mojo`. The smoke (`lora_step_smoke.mojo`) is the
  one-step template; E5 just drives it over a dataset with a real upstream grad.
- DATA-PATH DEPS (block E5, none on the LoRA hot path itself):
  1. **Mistral3B text encoder** (SEPARATE port; FLAGGED below): produces the
     [1,256,3072] text hidden states → text_proj input. Cache to disk like Klein so
     it is off the hot path.
  2. **VAE** (Klein VAE layout, `ERNIE_VAE_FILE`): encodes the training image to the
     128-ch latent → patch_embed input. Reuse the Klein VAE decode path + dataset
     layout (`training/klein_dataset.mojo` template).
  3. **Shared-AdaLN SOURCE** (the deferred E2/E4 link): compute once per step
     `c=time_embed(σ*1000)` → silu(c)@adaLN_modulation.1+bias → chunk6 (the `mv`
     ErnieModVecs) AND c@final_norm.linear+bias → chunk2 = [f_scale,f_shift]. For
     LoRA-only training these mod-source weights are FROZEN (LoRA targets are the 7
     block linears), so their grads (d_shared_mod/d_f_scale/d_f_shift, already
     returned by the backward) are unused; full-FT E6 consumes them. `ErnieStackBase`
     already holds te_w1/w2/adaln/final_norm resident to build c/mv/f_scale/f_shift.
- Re-measure streamed peak at REAL seq (4096 img + 256 txt = 4352): per-block SDPA
  scores [H=32,4352,4352] F32 ≈ 2.4 GB dominate (bounded per block). The smoke used
  S=8 to bound shared-3090 cost.
- Reuse the SHARED pipeline UNCHANGED: `training/train_step.mojo`,
  `train_config.mojo`, `optim.mojo`, `lora_save.mojo`, `checkpoint.mojo`,
  `schedule.mojo`, `on_device_global_norm.mojo`, `grad_coverage.mojo`. Mirror
  `training/train_klein_real.mojo` for the loop policy.
- σ-map: `ernie_model_timestep(sigma) = sigma * 1000` (already in
  `ernie_contract.mojo`); flow target via shared `schedule.build_flow_target`.
- USE THE STREAMED STACK (`ernie_stack_forward_streamed` /
  `ernie_stack_backward_streamed`): 36 blocks do NOT fit 24 GB resident at F32
  (E2 memory finding). Streamed peak is ~0.72 GiB at reduced seq; at the real seq
  (4096 img + 256 txt = 4352) the per-block SDPA scores [H=32, 4352, 4352] F32 ≈
  2.4 GB dominate — bounded per block, but re-measure peak at real seq before the
  real run (the smoke used a reduced seq to bound shared-3090 cost).
- Shared-AdaLN SOURCE (the deferred E2 link, now this phase): compute ONCE per
  step `c = time_embed(σ*1000)` (sin-first timestep MLP — see
  `ernie_image.mojo:226 time_embed`), then `silu(c)@adaLN_modulation.1+bias →
  chunk6` for the 6 mod-vecs (the `mv` passed to the stack), AND
  `c@final_norm.linear+bias → chunk2 = [f_scale, f_shift]` (passed to the stack).
  At step end, backprop the stack's `d_shared_mod [6D]` into adaLN_modulation.1
  with one `linear_backward(d_shared_mod, silu(c), adaln_w)`, and `d_f_scale/
  d_f_shift` (concat→[2D]) into final_norm.linear similarly — IF those weights
  are trained (LoRA targets per E4 are the block linears, not the AdaLN/final
  MLPs, so for LoRA-only training these mod-source grads are unused; full-FT E6
  consumes them). `ErnieStackBase` already holds te_w1/te_w2/adaln_w/adaln_b/
  final_norm_w/final_norm_b resident (loaded by `load_ernie_stack_base`) for
  building `c`, mv, f_scale/f_shift.

### Current Runtime Wiring (2026-06-03)

- `serenitymojo/configs/ernie_image.json` now points at
  `serenitymojo/configs/ernie_image_samples.json`, the shared
  `serenity.sample_prompts.v1` prompt file.
- `train_ernie_real.mojo` reads that prompt JSON, emits the mandatory
  `print_trainer_progress` display line, and supports resume-smoke mode:
  sample at step 0, train 10, save PEFT plus trainer state, reload trainer
  state, train to step 25, save, then sample again.
- `training/ernie_validation_sampler.mojo` applies the active LoRA through the
  resident no-save ERNIE forward path, decodes via the Klein VAE path, and writes
  1024x1024 validation PNGs from the prompt JSON cap paths.

### E6 — Full fine-tune extension  [DEFERRED, path exists]
- The block backward ALREADY returns all base-weight grads (d_wq..d_wdown,
  d_sa_norm, d_mlp_norm, d_q/k_norm). Full-FT = optimize those instead of (or
  with) LoRA — same as the Klein note in FULL_PORT_TRAINING_PLAN.md. 24GB will
  require gradient checkpointing/offload (Klein's checkpoint path is the model).

## Dependencies / flags

- **Mistral3B text encoder is a SEPARATE dependency** (not in this plan). Training
  can cache text embeddings to disk (like Klein caches), so the encoder is not on
  the hot training path — but a real run needs the [1,256,3072] Mistral layer-24
  hidden states. Port is its own intake (handoff §Remaining Work item 1: YaRN
  RoPE, GQA 32q/8kv, causal mask, 26 layers, extract layer 24). FLAGGED.
- **VAE**: ERNIE uses the Klein VAE layout (per handoff + contract:
  `ERNIE_VAE_FILE`, 251 tensors, 32-channel post-quant). The DiT emits 128-channel
  latent; dataset/VAE wiring reuses the Klein VAE decode path. Dataset path =
  Klein dataset layout (`training/klein_dataset.mojo` is the template).

## Missing `ops/` backward primitives

NONE. Every arm the ERNIE block needs already exists in `serenitymojo/ops/` and
is gated:
- `linear_backward` (linalg_backward.mojo)
- `rms_norm_backward` (norm_backward.mojo) — handles per-head [Dh] and [D] scale
- `layer_norm_backward` (norm_backward.mojo) — for the final non-learnable norm (E2)
- `gelu_backward` (activation_backward.mojo) — tanh-approx, takes pre-activation x
- `sdpa_backward` (attention_backward.mojo) — H=32/Dh=128 proven this session
- `rope_backward` (rope_struct_backward.mojo) — both interleaved + halfsplit arms
- `modulate_backward` (elementwise_backward.mojo)
- `gate_residual_backward` (rope_struct_backward.mojo)
- elementwise `mul`/`add` (tensor_algebra.mojo) for the gelu-gate product + grad-sum

If E3 lands on the interleaved convention, no new primitive is needed (swap to
the already-gated interleaved arm). If full-FT (E6) needs activation offload at
24GB, that reuses Klein's checkpoint primitives (`training/checkpoint*.mojo`).
