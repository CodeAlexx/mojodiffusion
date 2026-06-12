# HiDream-O1 training port — plan + recipe (opened 2026-06-11)

Mandate (Alex, 2026-06-11): "do the same for HiDream-O1" as the ideogram4
training fix — wired AND actually training in the UI, **DiffSynth-Studio as
the reference**.

## Reference recipe (DiffSynth, read in full 2026-06-11)

Source: `/tmp/DiffSynth-Studio` clone (re-clone github.com/modelscope/
DiffSynth-Studio if wiped) — `examples/hidream_o1_image/model_training/`
(train.py + lora/HiDream-O1-Image-Dev.sh), `diffsynth/diffusion/loss.py`
(FlowMatchSFTLoss), `diffsynth/diffusion/flow_match.py`
(set_timesteps_hidream_o1_image), `diffsynth/pipelines/hidream_o1_image.py`.

1. **No VAE, no separate text encoder.** input_latents = preprocessed RGB
   image direct (pixel space); prompt tokens embed through the spine's own
   embed_tokens.
2. **t sampling:** timestep_id ~ UNIFORM randint over the 1000-step
   training schedule (flow-match, shift=3.0). NOT logit-normal.
3. **Noise:** `noise = randn_like(latents) * noise_scale` — noise_scale
   7.5 (shell) / 8.0 (train.py default). Scaled BEFORE add_noise.
4. **noisy = (1-σ)·clean + σ·noise; target = noise − clean** (the same
   FlowMatch base zimage/klein use, but with the scaled noise).
5. **loss = MSE(pred, target) × scheduler.training_weight(t)**
   (linear_timesteps_weights — port the base-class formula).
6. **LoRA:** rank 32, lr 1e-4, cfg_scale=1 (no negative at train),
   gradient checkpointing. Target modules:
   spine `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` +
   heads `attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2`.

## What exists (surveyed 2026-06-11)

- `serenitymojo/models/dit/hidream_o1.mojo` (1,074 ln): INFERENCE DiT —
  `HiDreamO1DiT[S]` monolithic struct (36 Qwen3-VL layers, GQA 32 heads /
  Dh=128 with repeat_kv, prefix-causal mask, 3 image heads) + offloaded
  variant. Reference: inference-flame/src/models/hidream_o1/ (end-to-end).
- `serenitymojo/pipeline/hidream_o1_cfg.mojo` — inference pipeline.
- serenity-trainer: `BaseHiDreamSetup/HiDreamLoRASetup`, LoRA loader/saver,
  `HiDreamSampler`, `HiDreamBaseDataLoader` — OneTrainer-port SKELETONS.
- **MISSING:** per-block training fwd/bwd, train step, trainer binary,
  cache prepare, UI backend target ("hidream" not in TrainerConfigModel
  targets).
- Weights ON DISK: `/home/alex/HiDream-O1-Image-Dev-weights`,
  `/home/alex/HiDream-O1-Image-Full-weights`.

## Status (2026-06-11, end of day — P1+P2 SHIPPED, ~1.0 s/step)

- **P1 SHIPPED + GATED** (commits 88d255d, 400a295, 555c078):
  `models/dit/hidream_o1_train_block.mojo` (7-slot LoRA block fwd+bwd) +
  new `sdpa_backward_masked` primitive. Torch-autograd parity ALL PASS
  (out, d_hidden, 14 adapter grads, cos >= 1-4e-13; gates:
  models/dit/tests/hidream_o1_block_parity.mojo +
  ops/tests/sdpa_masked_backward_parity.mojo).
- **P2 SHIPPED** (c509a40, 77b0831): `training/train_hidream_o1_real.mojo`
  — recipe above verbatim, recompute-checkpoint backward, fused resident
  AdamW (zimage kernel), **bf16-RESIDENT weights** (15.2 GB converted
  once; the 30.4 GB figure is the F32 disk form — resident F32 OOMs).
  MEASURED speed arc: 98 s/step (P2 scaffolding) -> 106 (fused AdamW
  alone — loader dominated) -> **~1.0 s/step** (await_block was
  re-reading + re-converting F32 shards EVERY visit, ~60 GB/step).
  30-step gate: B|.|1 2048 -> 21,428 monotonic, LoRA saved
  (DiffSynth-loadable keys). Runner staged:
  serenity-trainer/target/serenity_hidream_live_trainer.
- **P3 COLLAPSED into the trainer**: no VAE + spine-embedded tokens means
  the stage-A images + in-trainer tokenization ARE the data path
  (caption.<i>.txt raw captions added to the stager).
- **P4 REMAINING**: "hidream" UI backend target (insertion points in task
  notes: TrainerRuntimeBridge :20-24/:193/:213/:293 + TrainerConfigModel)
  + sampler hookup for the step-0-vs-step-N visual learning verdict.
- Speed levers beyond ~1.0 s if wanted: grads-pack (504 D2H -> 1),
  per-sample mask/table cache, flash SDPA (S=512 IS 128-aligned but the
  mask is prefix-causal+padding — needs cuDNN bias support, separate
  decision), fp8-resident.

## Phase order (mojo-train-port discipline; gates before advance)

P1 — **Per-block LoRA fwd+bwd** (the Klein single-block pattern: standard
  attn+MLP; complications = GQA repeat_kv backward, prefix-causal-masked
  SDPA backward (sdpa_backward is nomask — need masked variant or the
  decomposed masked bwd), SwiGLU (exists), RMSNorm (exists), mrope
  (rope_backward exists; mrope interleave per mrope.rs)). LoRA slots per
  the DiffSynth target list. GATE: per-block torch-autograd parity vs the
  DiffSynth/torch block on real shapes (numeric-parity-testing).
P2 — **Stack trainer** `training/train_hidream_o1_real.mojo`: recompute
  checkpoint stack loop (Klein conductor shape; 36 blocks ~8B bf16 needs
  the turbo block-swap on 24 GB), recipe per above, fused AdamW reuse.
  GATE: loss class sane + decreasing + all LoRA-B nonzero (ideogram4 gate
  shape), vs a 1-step DiffSynth torch oracle dump if feasible.
P3 — **Cache prepare**: stage A python (jpg+txt → RGB f32 + chat-templated
  prompt; NO VAE pass) + stage B tokenization → ids (the spine embeds at
  train time — cache ids, not features). Template: ideogram4 two-stage
  scripts (serenity-trainer scripts/ideogram4_stage_images.py + smoke/
  ideogram4_prepare_cache.mojo).
P4 — **Runner + UI**: `serenity_hidream_live_trainer` + "hidream" backend
  target in TrainerConfigModel/TrainerRuntimeBridge (the ideogram4 launcher
  pattern).

## Notes
- SDPA: prefix-causal mask means the cuDNN flash path needs the causal
  arg + flame's bwd-causal guard (allow_cudnn_sdpa_bwd_causal) — start
  with math-mode bwd; flash later.
- The 06-11 flash/v2-engine work is orthogonal; do NOT couple this port
  to it (correctness first, Klein-pattern speed levers after).
