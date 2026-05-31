# Capability-Gap Audit — Mojo training port vs flame-core / EriDiffusion-v2

**Date:** 2026-05-30
**Scope:** READ-ONLY. What flame-core (Rust reference framework) + the EDv2 trainers HAVE
that the Mojo port (`serenitymojo`) is MISSING. No compile run (builder holds the lock);
all Mojo claims are source-read citations.

**Method:** Read the JUST-UPDATED port inventory (`docs/MOJO_MODULES.md`, `MOJO_KERNELS.md`),
then grepped the full `serenitymojo/` tree for each EDv2/flame-core feature and confirmed
absence/presence with `file:line` on both sides.

---

## Honest framing (read first)

The Mojo port's PROVEN scope is **correctness through composition**: ~68 backward arms at
cos ≥ 0.999, an AdamW that matches `adam.rs` to 9.8e-15, a flow-match v-target that matches
`train_qwenimage.rs`, byte-exact LoRA save, real VAE encode (std 0.962). A basic Klein/Z-Image
LoRA run is **mechanically possible** from what exists.

The single most important honest finding: **almost everything flame-core/EDv2 has that the
Mojo port lacks is default-OFF in the EDv2 trainers themselves.** Looking at the actual
`train_klein.rs` CLI defaults (`crates/eridiffusion-cli/src/bin/train_klein.rs`):

| EDv2 feature | trainer default | ⇒ needed for a CORRECT basic run? |
|---|---|---|
| `min_snr_gamma` | `None` (`:124`) | NO — opt-in quality |
| `caption_dropout_probability` | `0.0` (`:125`) | NO |
| `multires_noise_iterations` | `0` (`:134`, "no-op, byte-identical") | NO |
| `timestep_bias_strategy` | `"none"` (`:213`) | NO |
| `ema` | `false` (`:166`) | NO |
| `learning_rate_scheduler` | `Constant` (`:1759`, "byte-identical to legacy") | NO |
| `optimizer` | `"adamw"` (`:188`) | **Mojo HAS AdamW** |
| `clip_grad_norm` | `1.0` (`:10`, `:1541`, "ERNIE convergence killer if off") | **YES — see Gap 1** |

So the gap list below is mostly **GOOD/production-quality and PERF**, not basic correctness.
The two things that genuinely bite a *correct basic run* are grad-clip-by-global-norm
(default-on in EDv2, and a known convergence factor) and the gradient-flow assert
(the project's #1 debugging tool, `FLAME_ASSERT_GRAD_FLOW=1`, which has caught dead-LoRA
bugs repeatedly per memory). Everything else is upside.

---

## Group 1 — Training-loop features

| # | Capability | flame-core / EDv2 location | In Mojo port? | Severity |
|---|---|---|---|---|
| 1 | **Grad clip by GLOBAL norm (multi-tensor)** | `flame_core::ops::grad_norm::global_l2_norm` (used `train_klein.rs:1553`); `flame-core/src/gradient_clip.rs:70` `clip_grads_by_norm`, `:160` `compute_grad_norm` | **PARTIAL** — `training/optim.mojo:234` `clip_grad_global_norm` is a hand-rolled **2-tensor** case only (`MOJO_KERNELS.md §12`: "not the multi-tensor reduction"). A Klein LoRA run has ~80 adapters × 2 = 160 grad tensors; 2-tensor clip can't span them. | **IMPORTANT** (default-on in EDv2; ERNIE memory flags missing clip as a convergence killer) |
| 2 | **LR scheduler** (Constant+warmup / Linear / Cosine / CosineWithRestarts / Polynomial / Rex) | `features/lr_schedule.rs:28-161` (`constant_lr`, `cosine_lr`, `rex_lr`, `dispatch_lr`) | **NO** — grep of `training/`+`models/` for `cosine_lr\|warmup\|polynomial` returns nothing. Mojo only has constant `lr` in `TrainConfig`. | NICE-TO-HAVE (EDv2 default is `Constant`, so a basic run matches; cosine/warmup is quality) |
| 3 | **Min-SNR / debiased loss weighting** | `features/loss_weight.rs:37` `min_snr_weight`, `:62` `debiased_weight`, `:89` `apply_loss_weight` | **NO** — no `min_snr`/`snr_weight` anywhere in tree. | NICE-TO-HAVE (EDv2 default `None`; known likeness-quality lever) |
| 4 | **Combined MSE+MAE+Huber loss** | `features/loss_weight.rs:162` `combined_loss` | **PARTIAL** — `ops/loss_swiglu_backward.mojo` HAS `mse_backward` + `huber_backward`, but no combined-weighting driver and no MAE term. | NICE-TO-HAVE |
| 5 | **Caption dropout** (CFG uncond swap) | `features/caption_dropout.rs:30` `maybe_drop_caption`, `:48` `drop_caption` | **NO** | NICE-TO-HAVE (default `0.0`) |
| 6 | **Multires / offset / pyramid / input-perturbation noise** | `features/noise_modifiers.rs:55` offset, `:83` input-perturb, `:120` multires | **NO** | NICE-TO-HAVE (default `0`/off; "byte-identical to no-multires" per CLI doc) |
| 7 | **Timestep bias** (range emphasis) | `features/timestep_bias.rs:15` `Strategy` enum, `:90` `apply_bias` | **NO** | NICE-TO-HAVE (default `"none"`) |
| 8 | **Timestep distributions** (Uniform / Sigmoid / LogitNormal + shift) | `training_features/timestep_dist.rs:45` `TimestepDistribution`, `:158` `sample` | **PARTIAL** — `training/schedule.mojo:253` `sample_timestep_logit_normal` (logit-normal + qwen-shift only). No Uniform/Sigmoid selectable distribution. | NICE-TO-HAVE (logit-normal+shift is the Klein/Z-Image production default — the one that matters IS present) |
| 9 | **Gradient accumulation** (micro-batch) | EDv2 trainer config `gradient_accumulation_steps`; flame-core grad buffers | **PARTIAL** — `training/schedule.mojo:410` `grad_accumulate` primitive EXISTS but is **NOT wired** into any train loop (grep of `models/`+`train_step.mojo`+`loop.mojo` finds no caller). | NICE-TO-HAVE (24 GB fits Klein batch-1; accum is for effective-batch tuning) |
| 10 | **EMA of params** (advanced power-decay schedule) | `training/ema.rs:21` `ParameterEma`, `features/ema_advanced.rs:28` `EmaConfig` (power/warmup/clamp) | **PARTIAL** — `training/schedule.mojo:374` `ema_update` primitive EXISTS (decay-only) but UNWIRED, and no power-decay schedule (`1-(1+t/inv_gamma)^-power`). | NICE-TO-HAVE (default `false`; only affects sample/checkpoint, not loss) |
| 11 | **TREAD / token-routing, asymflow loss, masked loss, slider, caption/image aug** | `features/tread.rs`, `asymflow_loss.rs`, `masked_loss.rs`, `slider.rs`, `caption_aug.rs`, `image_aug.rs` | **NO** (except masked_loss has no analog) | NICE-TO-HAVE / out-of-scope for a first LoRA |

---

## Group 2 — Optimizers

flame-core/EDv2: `training_features/optimizers.rs:117` `enum Optimizer` with **9 variants** —
AdamW (`:118`), Adafactor (`:303`), AdamW8bit (`:561`), Prodigy (`:800`), Lion (`:1064`),
StableAdamW (`:1182`), RAdamScheduleFree (`:1406`), AdamWScheduleFree, StableAdamWScheduleFree.
Plus flame-core fused multi-tensor Adam with stochastic rounding (`adam.rs:870`
`adam_fused_multi_tensor_step`, `:1013` `stochastic_round`).

| # | Optimizer | flame-core/EDv2 | In Mojo? | Severity |
|---|---|---|---|---|
| 12 | **AdamW** | `optimizers.rs:118` / `flame-core/adam.rs` | **YES** — `training/optim.mojo:142` `adamw_step`, gated to 9.8e-15 vs `adam.rs`. The EDv2 default optimizer. | — (covered) |
| 13 | **SGD+momentum** | (flame-core) | **YES** — `optim.mojo:198` `sgd_step` | — |
| 14 | **AdamW8bit (blockwise)** | `optimizers.rs:561` + `flame-core/adam8bit_kernel.rs` | **NO** | NICE-TO-HAVE (Klein 9B LoRA fits F32 moments in 24 GB; 8-bit is for full-model / Wan) |
| 15 | **ScheduleFree (RAdam/AdamW/StableAdamW)** | `optimizers.rs:1406`, `:126`, `:128` | **NO** | NICE-TO-HAVE |
| 16 | **Adafactor / Prodigy / Lion / StableAdamW** | `optimizers.rs:303/800/1064/1182` | **NO** | NICE-TO-HAVE |
| 17 | **Fused multi-tensor Adam + stochastic rounding** | `adam.rs:870`, `:1013` | **NO** — `optim.mojo` is single-tensor F32, in-place, no stochastic round (`MOJO_KERNELS.md §12`) | PERF-ONLY (correctness fine in F32-master) |

---

## Group 3 — Autograd / kernel backward coverage

The Mojo backward surface (`MOJO_KERNELS.md`) is broad: linalg, norm (rms/layer/group),
activations (5), loss (mse/huber/swiglu), reduce (7), rope (interleaved+halfsplit)+qkv+gate,
modulate (AdaLN), conv2d, pool, CE/NLL/BCE/embedding, 18 shape ops, decomposed SDPA. This is
**enough for the Klein/Z-Image DiT path** (the blocks gate 28/28 vs torch per source headers).

| # | flame-core capability | location | In Mojo? | Severity |
|---|---|---|---|---|
| 18 | **cuDNN / flash SDPA backward** | flame-core `flame_cudnn_sdpa_bwd_bf16` (`autograd.rs` dispatch) | **NO** — only decomposed math-mode SDPA bwd (`ops/attention_backward.mojo`). Correct (non-degen gate green at H=30) but slow. | PERF-ONLY |
| 19 | **Tape-dispatched backward for all ops** | `autograd.rs` full op enum | **PARTIAL** — Mojo tape (`autograd.mojo:441`) wires **9 ops** (ADD..MSE); the other ~59 backward arms exist as kernels but are **hand-chained**, not tape-dispatched (`MOJO_MODULES.md:88`). Works for Klein because blocks hand-chain; brittle for new models. | NICE-TO-HAVE (correctness OK via hand-chain; maintainability cost) |
| 20 | **Vectorized norm/permute/transpose kernels** | flame-core `rms_norm_*_bf16_vec`, `permute0213_vec4`, tiled transpose | **NO** — Mojo kernels are scalar/F32 correctness-first (`MOJO_KERNELS.md §12`) | PERF-ONLY |
| 21 | **SDPA backward precision at large S** | — | **WATCH** — Mojo `d_k` cos 0.9975 at S=2304 (F32 accumulation order, `MOJO_KERNELS.md §11`). Not corruption, but watch on the unified-sequence Klein run. | NICE-TO-HAVE (monitor) |

**No correctness blocker in this group.** The H=30 "silent-zero d_q/d_k" scare was a
degenerate-test-data artifact (resolved this session; non-degen gate green at H=30).

---

## Group 4 — Memory / scale

| # | Capability | flame-core/EDv2 | In Mojo? | Severity |
|---|---|---|---|---|
| 22 | **Gradient checkpointing (per-block recompute)** | flame-core `activation_offload.rs`; EDv2 `training/checkpoint.rs` | **YES** — `training/checkpoint_block.rs`→`checkpoint_block.mojo` (full-block dx cos 0.99999999); `klein_stack.mojo` does per-block recompute (fits 8+24 blocks in 24 GB). | — (covered, this is the 24 GB enabler) |
| 23 | **BlockOffloader (weight streaming, CPU↔GPU)** | flame-core `offload/manager.rs`, `offload/planner.rs`, `offload/strategy/adaptive.rs`; EDv2 `training/offload.rs:52` | **PARTIAL** — inference `offload/` block-streaming loaders exist and training borrows them (`MOJO_MODULES.md:323`), but no training-grade adaptive offload **planner/manager** with transfer benchmarking. | NICE-TO-HAVE for Klein 9B LoRA in 24 GB (per-block recompute already fits it); IMPORTANT only for larger / full-finetune |
| 24 | **Activation offload to host** | flame-core `activation_offload.rs`; EDv2 `training_offload.rs` (64 KB) | **PARTIAL** — `training/checkpoint.mojo` does a TOY linear→silu offload round-trip; no general activation-offload manager. | NICE-TO-HAVE |
| 25 | **FP8 / int8 weight quant** | flame-core `int8_weight_only_qt_kernel.rs`, `sage_attention.rs`; Mojo `ops/fp8.mojo`/`mxfp4.mojo` exist (inference) | **PARTIAL** — Mojo HAS `ops/fp8.mojo` + `ops/mxfp4.mojo` as inference forwards, but no training-path quantization (frozen-base FP8 to save VRAM). | PERF-ONLY for Klein/Z-Image (no-quant is the project rule for Z-Image; Wan-only exception) |
| 26 | **CUDA-graph capture/replay + caching allocator** | flame-core `cuda_alloc_pool.rs`, ring_alloc | **NO** — each Mojo op `enqueue_create_buffer`s fresh, no pool (`MOJO_KERNELS.md §12`) | PERF-ONLY |

---

## Group 5 — Observability

| # | Capability | flame-core/EDv2 | In Mojo? | Severity |
|---|---|---|---|---|
| 27 | **BoardWriter (SQLite scalar logging)** | `training/board.rs:31` `BoardWriter`, `:241` `log_scalar`, `:312` `log_image_png` (loss/grad_norm/lr/steps_per_sec → `board.db`) | **NO** — no `sqlite`/`log_scalar`/`board.db` in tree. board.db is the canonical EDv2 metrics store (trainers are silent on stdout w/o RUST_LOG). | **IMPORTANT** — without it you're flying blind on a real run (no loss/grad_norm trajectory to verify convergence). A `println` fallback works but loses the queryable history. |
| 28 | **Grad-flow assert (`FLAME_ASSERT_GRAD_FLOW=1`)** | `flame-core/src/diagnostics.rs:144` `assert_grad_flow`; EDv2 `training/grad_coverage.rs:43` `GradCoverage::measure` | **NO** — no `assert_grad_flow`/`grad_coverage` in tree. This is the project's #1 dead-LoRA detector (caught LoHa/chroma/qkv-collapse bugs per memory). | **IMPORTANT** — for a *new port's first real run*, the inability to assert "all 80 LoRA-B have nonzero grad" is the single biggest correctness risk. A manual nonzero-ratio check is a workable substitute. |
| 29 | **GPU health / thermal monitor (NVML)** | `features/health.rs:18` `GpuHealthMonitor` (temp ≥90 °C sustained → abort) | **NO** | NICE-TO-HAVE (the train-launch skill enforces an external 78 °C cap anyway) |
| 30 | **Progress / webhook / disk-check** | `training/progress.rs`, `features/webhook.rs`, `disk_check.rs` | **NO** | NICE-TO-HAVE |
| 31 | **Parity harness** | flame-core `parity::ParityHarness` | **YES** — `parity.mojo` `ParityHarness` (cos+max-abs in F64, threshold 0.999). The whole port stands on it. | — (covered) |

---

## Group 6 — LoRA variants (LyCORIS)

EDv2: `crates/eridiffusion-core/src/lycoris.rs:260` `enum LycorisAlgo` — LoCon, LoHa, LoKr,
OFT, plus DoRA wrapper and Full (`:36-42` key tables), backed by `lycoris-rs`.

| # | Algo | EDv2 location | In Mojo? | Severity |
|---|---|---|---|---|
| 32 | **Plain LoRA (LoCon)** | `lycoris.rs:263` | **YES** (training) — `training/train_step.mojo:120` `LoraAdapter` + per-block variants in `klein_stack_lora.mojo`; byte-exact PEFT save. | — (covered, this is the target) |
| 33 | **LoHa** | `lycoris.rs:264` | **NO** | NICE-TO-HAVE |
| 34 | **LoKr** | `lycoris.rs:265` | **NO** | NICE-TO-HAVE (note: LyCORIS LoKr adds ~2.1 s/step overhead per memory; plain LoRA is the baseline anyway) |
| 35 | **OFT / BOFT** | `lycoris.rs` OFTModule | **NO** | NICE-TO-HAVE |
| 36 | **DoRA** | `lycoris.rs:21-28` (`.dora_scale`) | **NO** | NICE-TO-HAVE |

Note: Mojo `lora.mojo` is **inference-only merge-at-load** (not training). The *training*
LoRA path is the `LoraAdapter` in `train_step.mojo`/`klein_stack_lora.mojo`, which IS present
and proven. So plain-LoRA training (the actual goal) is covered; only the LyCORIS family is absent.

---

## TOP 10 GAPS — ranked by training impact

### Tier A — needed for a CORRECT basic LoRA run (or you can't trust the result)
1. **Grad-flow assert / coverage** (Gap 28) — IMPORTANT. No way to assert all ~80 LoRA-B
   adapters receive nonzero grad. This project's history is full of dead-LoRA bugs that ONLY
   the grad-flow assert caught. Mitigation: a manual per-adapter nonzero-ratio check is a
   workable substitute for v1.
2. **Grad clip by global norm, multi-tensor** (Gap 1) — IMPORTANT. EDv2 default-ON (`clip=1.0`);
   ERNIE memory explicitly flags missing clip as a convergence killer. Mojo has only a 2-tensor
   clip — cannot span 160 LoRA grad tensors. Needs the multi-tensor L2 reduction
   (`global_l2_norm` analog).
3. **Loss/metric logging (BoardWriter or equivalent)** (Gap 27) — IMPORTANT. Without a loss /
   grad_norm trajectory you cannot verify the run is converging vs the baseline. A plain stdout
   `println` of (step, loss, grad_norm, lora_b_nonzero_ratio) is the minimum acceptable substitute.

### Tier B — needed for a GOOD / production-quality LoRA
4. **Min-SNR loss weighting** (Gap 3) — quality lever for subject likeness; EDv2 opt-in.
5. **LR scheduler (cosine + warmup)** (Gap 2) — EDv2 default is Constant (so basic run matches),
   but cosine+warmup is standard for a polished LoRA.
6. **EMA of params with power-decay schedule** (Gap 10) — improves sample stability at
   checkpoint time; primitive exists but unwired + no schedule.
7. **Caption dropout** (Gap 5) — improves CFG behavior / prompt adherence; default-off.
8. **Gradient accumulation wired into the loop** (Gap 9) — primitive exists, not wired; needed
   for effective-batch tuning beyond batch-1.
9. **Timestep bias + multires noise** (Gaps 6,7) — secondary quality levers, both default-off.

### Tier C — PERF-ONLY (no correctness or quality impact in F32-master)
10. **cuDNN/flash SDPA backward + fused multi-tensor Adam + CUDA-graph/caching allocator**
    (Gaps 18, 17, 26) — throughput only. The decomposed SDPA bwd, single-tensor F32 AdamW, and
    fresh-alloc-per-op are all *correct*, just slow. flame-core's perf advantage here is large
    (30–50× on SDPA bwd) but does not change whether the LoRA learns the subject.

---

## Bottom line

- **A correct basic Klein/Z-Image plain-LoRA run is mechanically possible today** — the DiT
  blocks, AdamW, flow-match v-target, per-block recompute (24 GB fit), and byte-exact PEFT
  save are all PROVEN. No autograd/kernel correctness blocker remains (the H=30 SDPA scare
  was bad test data).
- **The real risk for the port's first run is observability, not math**: you can't currently
  assert grad-flow, can't multi-tensor-clip, and can't log a loss curve — so a silent
  half-learning bug (the exact failure mode that has bitten this project repeatedly) would be
  hard to catch. Those three (Gaps 28, 1, 27) are the priority before a real Alina run.
- **Everything else is upside**: optimizer variety, LyCORIS, min-SNR, schedulers, EMA,
  offload-at-scale, and all the fused/vectorized/graph perf work are quality- or speed-only
  and default-OFF in the EDv2 trainers they were copied from.
