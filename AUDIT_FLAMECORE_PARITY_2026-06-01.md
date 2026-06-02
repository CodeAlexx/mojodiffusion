# AUDIT — serenitymojo (Mojo) vs flame-core / EriDiffusion-v2 parity

**Date:** 2026-06-01
**Scope:** Mojo port at `/home/alex/mojodiffusion/serenitymojo/` vs `flame-core/` + `EriDiffusion-v2/`.
**Method:** 8 domain auditors, cross-checked against TODAY's tree (codex HEAD 28d67d7, uncommitted working tree; offload/ and train_klein_real.mojo modified Jun 1). Supersedes `AUDIT_FLAMECORE_GAPS_2026-05-30.md`.
**Project goal (the bar):** Klein/Z-Image LoRA in 24 GB via block-swap/offload, pure-Mojo runtime (Python = parity oracle only). Memory = block-swap NOT quantization for model weights.

Status legend: PRESENT (full analog exists + gated) · PARTIAL (exists but incomplete/unwired/toy) · ABSENT.
needed_for: BASIC_CORRECT · PRODUCTION_QUALITY · PARITY_COMPLETENESS · PERF_ONLY.

---

## Changes since the 2026-05-30 audit (status flips)

The 05-31 config-driven trainer + staged sampler + A1/A2/dx residency levers moved the tree hard. The following the prior audit flagged as ABSENT/PARTIAL blockers are now **CLOSED** and were re-verified on disk this session:

| Item | Old status (05-30) | New status (06-01) | Evidence (verified) |
|---|---|---|---|
| SQLite board logging | ABSENT ("no board.db in tree") | **PRESENT + wired** | `serenityboard.mojo:180` SerenityBoardWriter, `:237` log_train_step (loss/grad_norm/lr/sps); called `train_klein_real.mojo:720`. libsqlite3 via external_call, pure-Mojo. |
| Multi-tensor global-norm grad clip | PARTIAL ("2-tensor only") | **PRESENT** | `train_klein_real.mojo:657-688` L2 over ALL `dbl/sgl d_A/d_B` (~160 tensors), `_scale_inplace` each when `grad_norm>max_grad_norm`; default `max_grad_norm=1.0`. The old `optim.mojo:234` 2-tensor helper is now dead w.r.t. the real path. |
| Grad-flow / dead-adapter assert | ABSENT | **PARTIAL** (inline warn) | `train_klein_real.mojo:669-677` warns per-adapter when `|d_A|+|d_B|==0` at step≥1. Catches the #1 dead-LoRA failure mode; still WARN-only, no coverage% / no NaN check / no env-gated panic. |
| Caching/scratch-ring allocator | ABSENT | **PARTIAL (now wired)** | `scratch_ring.mojo` ScratchRingAllocator now imported + used in `models/klein/{lora,single,double}_block.mojo` (verified). Not yet extended to all op paths. |
| Block-swap offload (inference) | PARTIAL ("no planner/manager") | **PARTIAL (real async system)** | `offload/{plan,residency,turbo_loader,telemetry,vmm_cuda,vmm_slab}.mojo` — planner + residency state machine + async double-buffered H2D. Inference-only (frozen weights, H2D only). |
| Staged Klein sampler + block-swap | toy/OOM | **PARTIAL (wired, render unproven at 1024)** | `sampling/klein_sampler.mojo` + `klein_sample_cli.mojo` wire TurboPlannedLoader; 512² renders; 1024² branch present but no MEASURED coherent artifact. |
| AdamW parity | (n/a) | **PRESENT** | `optim.mojo:142`, matched to adam.rs ~9.8e-15 via `optim_converge_parity.mojo`. |
| Gradient checkpointing (24 GB enabler) | (n/a) | **PRESENT** | `training/checkpoint_block.mojo` (dx cos≥0.9999); `klein_stack.mojo:432,462` per-block recompute. |

**Net effect:** the three "trust-critical" items the prior audit blocked on are effectively resolved for the Klein path. A correct, observable Klein/Z-Image LoRA run is **not blocked** by anything in this audit. Remaining gaps are production-quality levers, parity-completeness for non-default features, perf, and new-model generalization.

---

## Domain 1 — Training loop

| id | capability | status | feasibility | effort | needed_for |
|---|---|---|---|---|---|
| grad-clip-global-norm-multitensor | global-L2 multi-tensor clip, default-on | PRESENT | CLEAN | S | BASIC_CORRECT |
| board-sqlite-logging | SQLite scalar/artifact logging | PRESENT | CLEAN | S | PRODUCTION_QUALITY |
| grad-flow-dead-adapter-assert | per-adapter dead-branch detect | PARTIAL | CLEAN | S | BASIC_CORRECT |
| min-snr-debiased-loss-weight | min-SNR γ + debiased loss weight | ABSENT | CLEAN | S | PRODUCTION_QUALITY |
| lr-scheduler-enum | const+warmup/linear/cosine/restarts/poly/rex | ABSENT | CLEAN | S | PRODUCTION_QUALITY |
| combined-mse-mae-huber-loss | combined loss driver | PARTIAL | CLEAN | M | PRODUCTION_QUALITY |
| caption-dropout | uncond swap w/ prob p | ABSENT | CLEAN | S | PRODUCTION_QUALITY |
| noise-modifiers | offset / input-perturb / multires | ABSENT | CLEAN | M | PRODUCTION_QUALITY |
| timestep-bias | range emphasis strategies | ABSENT | CLEAN | S | PRODUCTION_QUALITY |
| timestep-distributions-uniform-sigmoid | Uniform/Sigmoid (default LogitNormal present) | PARTIAL | CLEAN | S | PARITY_COMPLETENESS |
| gradient-accumulation-wiring | micro-batch accum (primitive exists, unwired) | PARTIAL | CLEAN | M | PRODUCTION_QUALITY |
| ema-params-wiring | EMA shadow (primitive exists, unwired) | PARTIAL | CLEAN | M | PRODUCTION_QUALITY |
| masked-asymflow-slider-tread-losses | advanced opt-in losses | ABSENT | HARD | L | PARITY_COMPLETENESS |

**Key details.**
- **grad-clip-global-norm-multitensor** — flame: `flame-core/gradient_clip.rs` clip_grads_by_norm + compute_grad_norm. mojo: `train_klein_real.mojo:657-688`; default `train_config.mojo` max_grad_norm=1.0. Residual: host List, not device reduction (PERF only — see on-device-global-norm).
- **board-sqlite-logging** — flame: `EriDiffusion-v2 training/board.rs:31/241/312`. mojo: `serenityboard.mojo:180/229/237/271`; wired `train_klein_real.mojo:439,720`. Schema is a superset.
- **grad-flow-dead-adapter-assert** — flame: `flame-core diagnostics.rs:144 assert_grad_flow`; `EriDiffusion-v2 grad_coverage.rs:43`. mojo: `train_klein_real.mojo:669-677` (verified). To reach parity: coverage ratio + NaN/Inf check + env-gated panic.
- **min-snr-debiased / lr-scheduler / timestep-bias / caption-dropout / noise-modifiers** — ABSENT; grepped (min_snr/snr_weight/debiased, lr_schedul/cosine_lr/warmup/polynomial, caption_drop/drop_caption, offset_noise/input_pert/multires/pyramid, timestep_bias/apply_bias) → only unrelated hits. EDv2 defaults make a basic run match without them. flame refs: `EriDiffusion-v2 features/{loss_weight.rs:37/62/89,lr_schedule.rs:28-65,timestep_bias.rs:15/90,caption_dropout.rs:30,noise_modifiers.rs:46}`.
- **timestep-distributions** — default logit-normal+qwen-shift PRESENT (`schedule.mojo:253`, wired `:571`); missing Uniform/Sigmoid. flame: `training_features/timestep_dist.rs:45`.
- **gradient-accumulation / ema** — primitives exist unwired: `schedule.mojo:410 grad_accumulate`, `:374 ema_update`; no caller (verified the loop steps every iter at `:694`). flame: gradient_accumulation_steps; `ema.rs:21`, `ema_advanced.rs:28`.
- **combined-mse-mae-huber** — mojo has `ops/loss_swiglu_backward.mojo` mse+huber kernels; loop uses MSE only (`train_klein_real.mojo:627-636`). flame: `loss_weight.rs:162`.
- **masked/asymflow/slider/tread** — ABSENT (grep zero). Masked loss is the tractable one; the other three are large. flame: `features/{masked_loss,asymflow_loss,slider,tread}.rs`.

---

## Domain 2 — Optimizers

| id | capability | status | feasibility | effort | needed_for |
|---|---|---|---|---|---|
| adamw-default | AdamW decoupled WD (EDv2 default) | PRESENT | CLEAN | S | BASIC_CORRECT |
| sgd-momentum | SGD+momentum | PRESENT | CLEAN | S | PARITY_COMPLETENESS |
| multi-tensor-grad-clip | global-norm clip over all tensors | PARTIAL→see note | CLEAN | M | BASIC_CORRECT |
| fused-multi-tensor-adam | one-launch packed Adam | ABSENT | HARD | L | PERF_ONLY |
| stochastic-rounding | BF16 stochastic round | ABSENT | HARD | M | PERF_ONLY |
| adamw8bit-blockwise | bnb 8-bit blockwise | ABSENT | HARD | XL | PERF_ONLY |
| adafactor | factored 2nd moment | ABSENT | CLEAN | M | PARITY_COMPLETENESS |
| prodigy | D-adaptive LR | ABSENT | CLEAN | M | PARITY_COMPLETENESS |
| lion | sign-momentum | ABSENT | CLEAN | S | PARITY_COMPLETENESS |
| stableadamw | RMS-normalized update clip | ABSENT | CLEAN | M | PARITY_COMPLETENESS |
| schedulefree-family | RAdam/AdamW/StableAdamW ScheduleFree | ABSENT | HARD | L | PARITY_COMPLETENESS |

**Note on multi-tensor-grad-clip:** the optimizer-domain auditor scored the *generic helper* (`optim.mojo:234`, 2-tensor) as PARTIAL. But the real Klein training path implements the full multi-tensor clip inline (`train_klein_real.mojo:657-688`, verified). So the **capability is PRESENT for Klein**; the gap is (a) lifting it into a reusable N-tensor helper, and (b) moving it on-device. Tracked as one work item (`multitensor-grad-clip-helper`) in the build plan.

**Key refs.** AdamW: `flame-core adam.rs:1145`; `optim.mojo:142`. SGD: `optim.mojo:198` (decoupled-WD, matches torch only at wd=0 — documented). Fused MT Adam: `adam.rs:870`. Stochastic round: `adam.rs:1013` (moot while F32-master). 8bit: `adam8bit_kernel.rs:231/360/561`, dynamic map `:101` (5 python parity refs). Adafactor `optimizers.rs:303/354`; Prodigy `:800/853`; Lion `:1064/1083`; StableAdamW `:1182/1211`; ScheduleFree `:1406/1764`. **All 8 extra optimizers are default-OFF (`optimizer="adamw"`); none block a correct run.**

---

## Domain 3 — Autograd / backward kernels

| id | capability | status | feasibility | effort | needed_for |
|---|---|---|---|---|---|
| tape-dispatch-completeness | unified tape-walked backward (9/66 ops) | PARTIAL | HARD | XL | PARITY_COMPLETENESS |
| tape-bf16-dtype | BF16/F16 grads through tape | PARTIAL | CLEAN | M | PERF_ONLY |
| cudnn-flash-sdpa-backward | fused cuDNN/flash attn backward | ABSENT | NO_MOJO_PATH | XL | PERF_ONLY |
| l1-mae-loss-backward | L1/MAE + combined loss bwd | ABSENT | CLEAN | S | PRODUCTION_QUALITY |
| indexassign-scatteradd-backward | scatter-add / index-assign bwd | PARTIAL | CLEAN | S | PARITY_COMPLETENESS |
| scalar-op-tape-arms | AddScalar/MulScalar/Div tape arms | PARTIAL | CLEAN | S | PARITY_COMPLETENESS |

**The trap (verified):** two backward systems. (1) Reverse-mode TAPE (`autograd.mojo`) — F32-only ("F32 only for the spike", `:16`), exactly 9 ops wired (`OP_ADD..OP_MSE`, verified `:52-60`). (2) ~68 standalone backward KERNELS in `ops/*_backward.mojo` (F32+BF16+F16), NOT tape-dispatched — Klein/Z-Image drive backward by hand-chaining block-level fns in reverse (`klein_stack.mojo:53-57`). flame-core has a 66-variant Op enum (`autograd.rs:191`) with 157 unified match arms. **Nothing blocks a correct Klein/Z-Image run** (hand-chain + per-arm parity gates); tape-dispatch is maintainability + new-model generalization (memory flags this as the source of past Klein composition-backward bugs).

- **tape-dispatch-completeness** — 68 kernels exist + gated; missing is wiring each as a tape arm (OP_ const + record_<op> + elif). Priority ops Klein/Z-Image use: LayerNorm/GroupNorm, RoPePrecomputed, QkvSplitPermute, GateResidual, Permute/Transpose/Reshape/Slice/Cat/Split, Softmax, SDPA composite, Conv2d.
- **l1-mae-loss-backward** — mojo has mse_backward (`loss_swiglu_backward.mojo:69`) + huber (`:132`); no L1. flame: `autograd.rs:6228`.
- **indexassign / scalar-ops** — index_select_backward present (`shape_backward.mojo:849`, the gather dual); scatter primitives in `ops/moe.mojo` unwired to autograd. Not needed for Klein/Z-Image. flame: `autograd.rs:5906/3703/3709`.

---

## Domain 4 — Memory / scale

| id | capability | status | feasibility | effort | needed_for |
|---|---|---|---|---|---|
| grad-checkpointing | per-block recompute (24GB enabler) | PRESENT | CLEAN | S | BASIC_CORRECT |
| block-offloader-h2d-streaming | async planner + double-buffered prefetch | PARTIAL | CLEAN | M | PRODUCTION_QUALITY |
| d2h-weight-writeback | training-grade D2H eviction write-back | ABSENT | HARD | L | PARITY_COMPLETENESS |
| transfer-benchmark | PCIe H2D/D2H bandwidth sweep | ABSENT | CLEAN | M | PARITY_COMPLETENESS |
| adaptive-offload-planner | knapsack/adaptive cost-model planner | PARTIAL | HARD | L | PARITY_COMPLETENESS |
| activation-offload-pool | pinned-host activation pool | PARTIAL | HARD | L | PARITY_COMPLETENESS |
| cuda-graph-capture-replay | backward graph capture/replay | ABSENT | NO_MOJO_PATH | XL | PERF_ONLY |
| caching-allocator | caching/ring device allocator | PARTIAL | HARD | L | PERF_ONLY |

**Bottom line:** for Klein/Z-Image LoRA in 24 GB, **gradient checkpointing already gets you there** (`checkpoint_block.mojo`; `klein_stack.mojo:432,462`). The inference H2D streaming is genuinely built and async (`turbo_loader.mojo:131` cuMemcpyHtoDAsync + DeviceEvent; `residency.mojo:231`). The training-grade adaptive PLANNER/MANAGER + D2H write-back gap is real but only bites >24 GB or full-finetune (LoRA keeps base frozen → H2D-only suffices). CUDA-graph + caching-allocator are PERF_ONLY.

- **d2h-weight-writeback** — `residency.mojo:10-16` explicitly inference-only; grep save_back/write_back/DtoH → only test text. Needs cuMemcpyDtoHAsync + EVICTING-state write-back. flame: `offload/mod.rs`, `offload/state.rs`, `EriDiffusion-v2 training/offload.rs`.
- **transfer-benchmark** — grep transfer_benchmark/bandwidth/peak_h2d → zero. Telemetry exists (`telemetry.mojo`) but no upfront sweep. flame: `offload/transfer_benchmark.rs`.
- **adaptive-offload-planner** — mojo has STATIC scoring (`residency.mojo:464 eviction_order`, `:503 prefetch_targets`); missing Adaptive.plan()/knapsack + discover→profile→activate lifecycle. flame: `offload/strategy/adaptive.rs:38`, `knapsack.rs`, `manager.rs:307/379`.
- **caching-allocator** — `scratch_ring.mojo` now wired into Klein blocks (verified) but most ops still `enqueue_create_buffer` fresh; `vmm_slab.mojo:44` is offload-side. Prior codex ring attempt was correctness-UNVERIFIED (aliasing risk) — any port MUST pass the multi-step no-step-2-divergence gate.

---

## Domain 5 — Observability

| id | capability | status | feasibility | effort | needed_for |
|---|---|---|---|---|---|
| board-writer-sqlite | SQLite scalar+image logging | PRESENT | CLEAN | S | BASIC_CORRECT |
| board-log-trace | per-step phase trace (table exists, no writer) | PARTIAL | CLEAN | S | PRODUCTION_QUALITY |
| grad-flow-report-struct | named coverage report + env-gated panic + NaN check | PARTIAL | CLEAN | S | BASIC_CORRECT |
| multi-tensor-grad-clip | global-norm clip (correctness) | PRESENT | CLEAN | S | BASIC_CORRECT |
| on-device-global-norm | async single-D2H norm | PARTIAL | HARD | M | PERF_ONLY |
| nvml-health-monitor | GPU temp/ECC monitor | ABSENT | HARD | M | PRODUCTION_QUALITY |
| webhook-notify | training webhook POST | ABSENT | HARD | M | PRODUCTION_QUALITY |
| disk-space-check | pre-save disk guard | ABSENT | CLEAN | S | PRODUCTION_QUALITY |
| parity-harness | F64 cos + max-abs, 0.999 gate | PRESENT | CLEAN | S | PARITY_COMPLETENESS |
| progress-display | operator progress + machine PROG line | PRESENT | CLEAN | S | PRODUCTION_QUALITY |

ParityHarness (`parity.mojo:46/54/101`) is the foundation of the whole port — PRESENT, matches `flame_core::parity`. board-log-trace: `trace_events` table exists in `serenityboard.mojo:104` but no log_trace writer (flame `board.rs:373`). grad-flow-report-struct = the same item as Domain-1 grad-flow plus NaN/Inf check + coverage_pct() + FLAME_ASSERT_GRAD_FLOW panic (flame `diagnostics.rs:38/111/144/155`, `grad_coverage.rs:43/62/83`). NVML/webhook/disk-check ABSENT (grep zero); /train-launch already enforces an external 78°C cap so NVML is defense-in-depth.

---

## Domain 6 — LoRA / LyCORIS

| id | capability | status | feasibility | effort | needed_for |
|---|---|---|---|---|---|
| plain-lora-training | LoCon-linear LoRA (the target) | PRESENT | CLEAN | S | BASIC_CORRECT |
| multi-tensor-grad-clip | global-norm over ~160 adapters | PRESENT | CLEAN | S | BASIC_CORRECT |
| loha | Hadamard low-rank | ABSENT | HARD | L | PARITY_COMPLETENESS |
| lokr | Kronecker | ABSENT | HARD | XL | PARITY_COMPLETENESS |
| oft-boft | orthogonal / butterfly | ABSENT | HARD | XL | PARITY_COMPLETENESS |
| full-diff | direct weight delta | ABSENT | CLEAN | S | PARITY_COMPLETENESS |
| dora | weight-decomposed LoRA | ABSENT | HARD | M | PARITY_COMPLETENESS |
| locon-conv-tucker | conv2d LoRA + Tucker | ABSENT | HARD | L | PARITY_COMPLETENESS |

Plain LoRA is PRESENT + gated: `train_step.mojo:119` LoraAdapter, fwd/bwd/adamw `:164/:193/:262`; composed `klein_stack_lora.mojo:154/:1714`; byte-exact PEFT save `lora_save.mojo:83`; parity gates exist (single/double/stack block_lora_parity.mojo). The **entire LyCORIS family is ABSENT** (grep dora/loha/lokr/oft/boft/tucker/hada_w/lokr_w/cayley/kronecker → zero outside vocoder + docs) — all NICE-TO-HAVE; plain LoRA is the documented baseline and LyCORIS carries overhead (LoKr ~2.1 s/step per memory). `serenitymojo/lora.mojo` (826 LOC) is inference merge-at-load only. flame refs: `EriDiffusion-v2 lycoris.rs:263-268`, `eri-lycoris/lycoris-rs/src/algorithms/{locon,loha,lokr,oft,boft,full}.rs`, `dora.rs`.

---

## Domain 7 — Inference / sampler

| id | capability | status | feasibility | effort | needed_for |
|---|---|---|---|---|---|
| klein-1024-blockswap-render-gate | staged 1024² render under offload (unproven) | PARTIAL | CLEAN | M | BASIC_CORRECT |
| dpmpp-2m-exponential-multistep | DPM++2M / res_2m/3m / DEIS | ABSENT | CLEAN | M | PARITY_COMPLETENESS |
| unipc-multistep-scheduler | UniPC bh2 predictor-corrector | ABSENT | CLEAN | M | PARITY_COMPLETENESS |
| inpaint-mask-blend-lanpaint | inpaint mask blend + lanpaint | ABSENT | CLEAN | L | PARITY_COMPLETENESS |
| img2img-reference-latent-pack | edit ref-token packing (sigmas present) | PARTIAL | CLEAN | M | PARITY_COMPLETENESS |
| vae-encode-general | encoders beyond Klein | PARTIAL | CLEAN | L | PARITY_COMPLETENESS |
| negprompt-true-cfg-staged-encode | in-process pos/neg encode (precache works) | PARTIAL | CLEAN | S | PRODUCTION_QUALITY |
| fp8-mxfp4-inference-dequant | FP8 E4M3 + MXFP4 dequant (INFERENCE) | PRESENT | CLEAN | S | PARITY_COMPLETENESS |
| wuerstchen-helios-schedulers | Cascade DDPM / Helios | ABSENT | (out of scope) | M | PARITY_COMPLETENESS |

Flow-matching schedule family the two named minima use is PRESENT: Z-Image (`flow_match.mojo`), Klein/FLUX.2 (`flux2_klein.mojo` build_flux2_sigma_schedule + flux2_cfg). Staged sampler renders 512² and WIRES block-swap (TurboPlannedLoader, `klein_stack_lora.mojo:865`); the 1024² branch exists but has **no measured coherent artifact** — that is the one BASIC_CORRECT gate left here (Tenet 4: working = viewed artifact). FP8/MXFP4 inference dequant PRESENT + GPU-validated (`ops/fp8.mojo:105`, `ops/mxfp4.mojo:88`) — **inference quant is explicitly allowed**, distinct from forbidden training quant. Higher-order samplers (DPM++2M/UniPC) are used only by Qwen/Cosmos, neither a named minimum. Inpaint/img2img/general-encode need a latent mask-blend primitive + non-Klein VAE encoders (only `klein_encoder.mojo` exists; others are decode-only). Wuerstchen + Helios are **user-EXCLUDED** (do not port).

---

## Domain 8 — Perf kernels

| id | capability | status | feasibility | effort | needed_for |
|---|---|---|---|---|---|
| flash-sdpa-dh128 | fused SDPA fwd at Dh=128 | PARTIAL | NO_MOJO_PATH | XL | PERF_ONLY |
| flash-sdpa-bwd-dh128 | fused SDPA bwd | PARTIAL | NO_MOJO_PATH | XL | PERF_ONLY |
| vec-rms-norm | bf16x4 vectorized RMSNorm fwd+bwd | PARTIAL | CLEAN | M | PERF_ONLY |
| vec-permute0213 | bf16x4 permute0213 | PARTIAL | CLEAN | S | PERF_ONLY |
| tiled-transpose | bank-conflict-padded transpose | PARTIAL | CLEAN | M | PERF_ONLY |
| vec-swiglu | vec2 fused SwiGLU | PARTIAL | CLEAN | S | PERF_ONLY |
| vec-modulate | vec2 fused adaLN modulate | PARTIAL | CLEAN | S | PERF_ONLY |
| fused-adamw-multitensor | one-launch packed AdamW | PARTIAL | CLEAN | M | PERF_ONLY |
| multitensor-grad-clip (device) | device reduction grad clip | PARTIAL | CLEAN | S | PRODUCTION_QUALITY |
| bias-epilogue-linear | cuBLASLt bias epilogue | PARTIAL | NO_MOJO_PATH | M | PERF_ONLY |
| cuda-graph-alloc-pool | graph capture + ring adoption | PARTIAL | HARD | L | PERF_ONLY |

Every Mojo norm/permute/transpose/swiglu/activation kernel is **scalar** (grep SIMD/vectorize/simdwidthof/bf16x4 → ZERO). flame-core ships measured-fast vectorized variants (rms_norm 9.5-16×, permute0213 1.5-1.8×, tiled transpose 3.4×, swiglu/modulate vec2). The vectorized items are CLEAN pure-Mojo work (Mojo CAN emit SIMD loads — see mojo-gpu-fundamentals) that does not change correctness. The dominant perf reality — SDPA at Dh=128 — is NO_MOJO_PATH (below).

---

## NOT FEASIBLE / FLAGGED — will NOT be built

### NO_MOJO_PATH (no Mojo/MAX surface on sm_86; would reintroduce the NVIDIA lock the port exists to escape)

| id | domain | reason |
|---|---|---|
| flash-sdpa-dh128 | perf-kernels | Mojo has NO cuDNN; SDK flash does not compile at Dh=128 on sm_86 ("no valid mma"); hand kernel compiles but 17-31× slower (no tensor-core MMA, bench 2026-05-29). Decomposed math path is CORRECT (`sdpa_math_parity.mojo` green) — throughput only. Gated on Modular MMA maturity. Do NOT spend effort on the 3090. |
| flash-sdpa-bwd-dh128 | autograd / perf | Same construction/verdict; decomposed bwd CORRECT (`sdpa_bwd_nondegen_parity.mojo` cos≥0.999 at H=30). |
| cudnn-flash-sdpa-backward | autograd | Mojo has no cuDNN binding; same as above. Do NOT "add flash attention" (project rule: NO flash attention). |
| cuda-graph-capture-replay | memory | Mojo DeviceContext exposes no cuGraph capture API; capture also forbids malloc/free in-region while every op allocs fresh. Would need hand driver FFI + pooled allocator. Backward is correct, just launch-bound. |
| bias-epilogue-linear | perf-kernels | Mojo matmul goes through `linalg.matmul.vendor.blas` which does NOT expose CUBLASLT_EPILOGUE_BIAS; reaching it needs custom cuBLASLt FFI (vendor lock). The cheap partial win (skip cast when F32) is already done (`linear.mojo:248`). |

### CONTRA_RULE — violates a project rule

| id | domain | reason |
|---|---|---|
| (none direct) | — | No gap proposes FP8/int8 quant of Klein/Z-Image *model weights* (the forbidden case; rule = block-swap/offload, NOT quantization). |
| adamw8bit-blockwise (caveat) | optimizers | NOT a hard CONTRA_RULE — the no-quant rule covers model WEIGHTS, and 8-bit here is optimizer STATE. But it is PERF/memory-only and unnecessary: 9B LoRA F32 moments fit in 24 GB with per-block recompute. Flagged so it is not prioritized; build only if a Wan-scale full-model target appears. |

### Out of scope (user exclusion)

| id | domain | reason |
|---|---|---|
| wuerstchen-helios-schedulers | inference | Stable Cascade (Wuerstchen) + Helios are user-EXCLUDED (prior audit A.3). Listed for inventory completeness only. Both are CLEAN scalar-schedule ports IF the exclusion is ever lifted. |

---

## Verdict

A correct, observable, 24 GB Klein/Z-Image LoRA run is **not blocked** by any gap above. The remaining feasible work is: (1) finish the trust trio to flame parity (coverage% + NaN check + env panic; trace logging), (2) prove the 1024² staged render, (3) production-quality training levers (schedulers, loss weighting, EMA, accumulation), (4) parity-completeness (timestep dists, LyCORIS family, higher-order samplers, inpaint/img2img/encoders), (5) PERF vectorization tail. The two NO_MOJO_PATH attention items are the only hard ceiling and are Modular's MMA maturity to fix, not ours.
