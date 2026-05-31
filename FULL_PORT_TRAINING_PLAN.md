# mojodiffusion — TRAINING port plan (autograd + tensors + full fine-tune)

> ## ⟳ PIVOT (2026-05-30, USER-APPROVED) — read before trusting T1/T2/T4 below
> The original USER decisions T2 (full fine-tune, "not LoRA-only") and T4 (engine-
> complete tape FIRST) were **deliberately pivoted this session, with USER approval**:
> - **T2 → LoRA.** The user explicitly chose LoRA (Klein, Alina subject) this session.
>   Full fine-tune is DEFERRED-not-dropped (the Klein stack fwd+bwd already returns
>   base-weight grads, so the path back exists).
> - **Sequencing → Klein-first, model-driven (NOT engine-complete-first).** The tape
>   engine has 9 of ~66 ops wired; the Klein model is HAND-CHAINED through host lists
>   instead — and that hand-chain is PROVEN vs torch (double 28/28, single 9/9, full
>   stack cos≈1.0, 80-adapter LoRA step finite + AdamW-updates + byte-exact save, all
>   lead-verified 2026-05-30). The full tape engine remains a generality/maintainability
>   debt, not a correctness gap for Klein.
> - **Z-Image (the plan's #1 target) is DEFERRED to AFTER the Klein LoRA run** (USER
>   decision 2026-05-30). It reuses all Klein infra; its H=30 attention is proven fine.
> See AUDIT_PLAN_DRIFT_2026-05-30.md + memory project_mojo_trainer_refactor_audit_2026-05-30.
> The T1–T4 table below is the ORIGINAL plan, preserved for history.

Status: DRAFT for review (2026-05-30). Not approved to execute.
Companion to `PLAN.md` (inference, in progress) and `FULL_PORT_ROADMAP.md`.
This doc covers the part `PLAN.md` line 39 explicitly excluded: **training**.

> Decision-ownership (EMPOWERMENT): the four scoping decisions below are USER-made
> this session. Everything tagged AGENT-DEFAULT is mine — override freely.

## USER decisions (2026-05-30)
| # | Decision | Value |
|---|---|---|
| T1 | Autograd approach | **Port flame-core's tape engine** natively to Mojo (no MAX autodiff exists — verified). |
| T2 | Training scope | **Full fine-tune** (backward through the entire DiT, full-weight grads, F32 master weights, offload) — not LoRA-only. |
| T3 | Pre-plan | **Investigate feasibility first** — DONE (see §1). |
| T4 | Sequencing | **Engine-complete first**: build the full tape engine + all backward kernels + optimizers as a parity-tested library BEFORE wiring any model training. |

## §1 — Feasibility findings (MEASURED 2026-05-30, this box)
- **Mojo/MAX provides NO reverse-mode autodiff.** Verified: no `vjp/jvp/autodiff/backward/grad_fn`
  in `max.graph.ops` or `max.experimental` (only `irfft.py`'s `BACKWARD` FFT-direction enum);
  no Mojo `nn` backward kernels on disk. ⇒ the entire backward stack is hand-built. No shortcut.
- **Port surface (flame-core, measured):**
  - `src/autograd.rs` = **7,008 lines** (tape engine) + `autograd_v4/` ~390 lines (graph experiment).
  - **66 distinct `Op::` backward arms** (full list §3).
  - **~119 `.cu` files** containing backward kernels.
  - EDv2 `crates/eridiffusion-core/src/training/` = **12,241 lines** (loop policy, features, offload).
  - Optimizers: `flame-core/src/adam.rs`, `adam8bit_kernel.rs`, `autograd_v2/optim.rs`,
    EDv2 `training/training_features/optimizers.rs`. Grad clip: `gradient_clip.rs` (433 lines).
- **Risk concentration (no SDK backward, active bug history):**
  1. **SDPA backward** (`FlashAttention`/`PrefixCausalFullAttention`/`SageAttention`) — `nn.flash`
     is forward-only. This is the exact primitive that crashed TWICE on 2026-05-30 (qwen step-43 +
     L2P) with `CUDA_ERROR_MISALIGNED_ADDRESS`. Porting to Mojo means owning that bug class
     natively with NO cuDNN-bwd fallback. HIGHEST RISK.
  2. **CheckpointOffloadBoundary** (8 refs — most complex arm) — REQUIRED for full-FT (24 GB can't
     hold full-DiT activations); coupled to the offload/recompute handle-lifetime contract
     (`training_offload.rs`, already flagged in `FULL_PORT_RUST_OFFLOAD_PARITY_2026-05-27.md`).
  3. **Conv2d backward** — MAX has `conv` + `conv_transpose` FORWARD only; data-grad and
     weight-grad are NOT conv_transpose; both hand-written.
- **Verdict:** feasible, but this INVERTS the inference port's economics. Inference = "wiring + 6
  tiny forward kernels" because the SDK supplied the hard forward kernels. Training = the SDK
  supplies ~zero backward, so this is the hardest, most bug-dense ~19k LOC in the stack reproduced
  in Mojo 1.0.0b1. Multi-month. SDPA-bwd + checkpoint-offload are research-grade risk, not volume.

## §2 — What gets ported (TRAINING delta on top of the inference lib)
The inference port already built the FORWARD ops + tensors + safetensors load + parity harness.
Training ADDS:
1. **Autograd tape engine** — Op-tagged tape, topological reverse traversal, gradient accumulation,
   `requires_grad` propagation, no_grad scope, tape clear. Port of `autograd.rs` `compute_gradients`.
2. **Backward kernel per forward op** — 66 arms (§3). Each = analytic grad, parity-tested vs flame-core.
3. **Tensor training extensions** — F32 master weights alongside BF16 compute; grad buffers;
   in-place optimizer updates; dtype-mixed accumulation (the `add_bf16_iter` / F32-accum pattern).
4. **Optimizers** — AdamW (F32 moments), AdamW8bit (quantized moments), SGD.
5. **Loss** — MSE (flow-matching target), Huber, + the combined/weighted-loss + min-SNR γ path.
6. **Grad clip** — global-norm clipping (`gradient_clip.rs`), per-tensor non-finite guard.
7. **Gradient checkpointing + activation offload** — boundary checkpoint + recompute; the
   block-streaming H2D-from-pinned path full-FT needs on 24 GB.
8. **torch-parity RNG** — only if matching reference training runs bit-wise is a goal (decide §6 D3).
9. **Training loop policy** — schedule, timestep sampling, EMA, the per-step orchestration
   (port of EDv2 `training/`), once the engine is proven.

NOT in this plan: the inference forward stack (done/in `PLAN.md`); serving; the cuBLASLt/cuDNN
parity shims (Mojo kernels are native cross-vendor — those CUDA-specific layers disappear, same as
the inference plan noted).

## §3 — The 66 backward arms (the engine-complete checklist)
Grouped by porting difficulty. Each arm GATES at grad-parity cos ≥ 0.999 vs flame-core (single op,
GPU-bf16 ref, F32 accum where flame-core uses it). Deep chains 0.99.

**Tier 0 — trivial elementwise/shape (grad is local):**
Add, Sub, Mul, Div, AddScalar, MulScalar, Abs, Clamp, Maximum, Minimum, Where, Cast, Reshape,
Transpose, Permute, Broadcast, Repeat, Cat, Split, Slice, Sum, SumDim, SumDimKeepdim, SumDims,
Mean, MaxDim.

**Tier 1 — activations (compose from std.math):**
ReLU, GELU, SiLU, Sigmoid, Tanh, Sqrt, Square, Log, Softmax, LogSoftmax.

**Tier 2 — linear algebra (GEMM-family backward):**
MatMul, BatchMatMul, Linear, AddBias. (Backward = transposed GEMMs; lean on `linalg.matmul`.)

**Tier 3 — normalization + positional (analytic, no SDK backward):**
RMSNorm, LayerNorm, GroupNorm, RoPePrecomputed. (Hand-written; watch the BF16-RoPE precision
floor + interleaved-vs-halfsplit convention — see EriDiffusion memory `project_bf16_rope_pattern_audit`.)

**Tier 4 — fused + structural:**
FusedSwiGLU, FusedSwiGLUSplit, GateResidual, QkvSplitPermute, Embedding, IndexSelect, IndexAssign,
UpsampleNearest2D, MaxPool2D.

**Tier 5 — HARD (risk sinks, no SDK backward, active bug history):**
FlashAttention, PrefixCausalFullAttention, SageAttention (SDPA-bwd family),
Conv2d, Conv2dNHWC (conv data-grad + weight-grad),
Checkpoint, CheckpointOffload, CheckpointOffloadBoundary (recompute + offload lifetime).

**Tier 6 — losses (leaf grad sources):**
MSELoss, L1Loss, HuberLoss, BCELoss, NLLLoss.

## §4 — Phases (engine-complete-first per decision T4)
Walking-skeleton discipline is INSIDE each phase (smallest provable unit), but per T4 the full
engine+kernels+optimizers land as a parity-tested library before any model training is wired.

### Phase T0 — De-risk the two hard kernels FIRST  [GATE: grad-parity vs flame-core]
Even under engine-complete-first, build the risk sinks before committing the rest, so failure
surfaces at week 1 not month 3:
- **SDPA backward** in Mojo (start non-causal full-attention, diffusion shapes; the seq lengths
  qwen/zimage actually use). Parity vs flame-core `flame_cudnn_sdpa_bwd_bf16` output (d_q/d_k/d_v).
  Carry the alignment lesson from `BACKLOG_qwen_cudnn_sdpa_bwd_misalign.md` (offset/stride, not just
  seq%128).
- **CheckpointOffloadBoundary** recompute + the offload handle lifetime (forward/backward visits,
  event-marked drops, releasable slots) per `FULL_PORT_RUST_OFFLOAD_PARITY_2026-05-27.md`.
- If either is intractable in Mojo 1.0.0b1 → STOP and re-scope (this is the kill-gate).

### Phase T1 — Tape engine core  [GATE: scalar + small-tensor grad matches finite-diff AND flame-core]
- Op-tagged tape, reverse topological traversal, grad accumulation, requires_grad propagation,
  no_grad scope, tape clear. Mirror `autograd.rs::compute_gradients`.
- Tier 0 + Tier 1 backward arms wired through it. A Mojo `GradParityHarness` (extends the existing
  inference ParityHarness) — finite-diff self-check + flame-core cross-check.

### Phase T2 — Linear-algebra + norm/rope backward  [GATE: per-arm cos ≥ 0.999]
- Tier 2 (MatMul/BatchMatMul/Linear/AddBias) + Tier 3 (RMSNorm/LayerNorm/GroupNorm/RoPE).

### Phase T3 — Fused/structural + loss + the hard tier integrated  [GATE: per-arm cos ≥ 0.999]
- Tier 4 + Tier 6, then integrate the T0-proven Tier 5 (SDPA/Conv/Checkpoint) into the engine.

### Phase T4 — Optimizers + grad-clip + F32 master weights  [GATE: 1-step update == flame-core]
- AdamW (F32 moments), AdamW8bit, SGD; global-norm clip; per-tensor non-finite guard;
  F32 master / BF16 compute split. Parity: one optimizer step on a fixed grad == flame-core.

### Phase T5 — Full-FT walking skeleton on ONE model  [GATE: grad-parity full-model + loss decreases]
- Z-Image full fine-tune end-to-end (forward already exists): forward → loss → backward →
  clip → optimizer step → checkpoint/offload. Per-block grad-parity vs flame-core, then a real
  short run where loss decreases AND a sample shifts (the only real "is it learning" verdict —
  same doctrine as the inference gate and the L2P caveat: never claim learning from loss/no-crash).

### Phase T6 — Training loop policy + remaining models
- Port EDv2 `training/` (schedule, timestep dist, EMA, caption-dropout, multires, etc.) once the
  engine is proven. Fan out full-FT to the other coherent models (Qwen, Klein, SD3, ERNIE, ...).

## §5 — Validation discipline (extends the inference plan's)
- **Grad-parity per arm:** cos ≥ 0.999 vs flame-core single-op GPU-bf16 grad output, F32 accum
  where flame-core uses it. Finite-diff self-check as a second independent oracle (Tier 0/1).
- **Optimizer-step parity:** one AdamW step on a fixed (param, grad, moment) tuple == flame-core
  byte-for-byte (or within BF16 stoch-round noise).
- **Full-model grad-parity:** per-block dL/dx + dL/dW cos vs flame-core BEFORE any long run
  (this session's klein/L2P pain came from skipping composed-backward parity).
- **Learning verdict = sample shift**, never loss-alone or no-crash (EMPOWERMENT + the L2P lesson).
- A regression test per hard kernel (SDPA-bwd, conv-bwd, checkpoint) that fails on misalign/NaN —
  the flame-core `tests/sdpa_bwd_parity.rs` / `narrow_sync_microbench.rs` model (Tenet 4).

## §6 — Open decisions (own before execute)
| # | Decision | AGENT-DEFAULT | Impact |
|---|---|---|---|
| D1 | Doc structure | New companion doc (this file), PLAN.md stays inference | Could merge into PLAN.md as a training section instead. |
| D2 | Tape vs graph engine flavor | Port `autograd.rs` runtime TAPE (not the v4 graph experiment) | Tape matches the proven flame-core path; v4 graph is younger/unproven even in flame-core. |
| D3 | torch-parity RNG | DEFER until a model needs bit-exact reference-run matching | If you want to reproduce a specific reference training trajectory, RNG parity moves up. |
| D4 | FP8 training | OUT (matches "no quant" rule; Wan-style fp8 is a later exception if needed) | Z-Image-class stays BF16/F32; revisit only for 14B-class that won't fit. |
| D5 | First full-FT model | Z-Image (forward already coherent; smallest real path) | Could be Klein/SD3; Z-Image is the lowest-risk skeleton. |
| D6 | Build vs verify SDPA-bwd from scratch | Port flame-core's analytic decomposed SDPA backward (NOT cuDNN shim) | The decomposed path is cross-vendor + is the same path that dodges the misalign crash. |

## §7 — Why engine-complete-first is the chosen risk profile (T4)
Time-to-first-trained-model is LONGER than a walking-skeleton-per-model approach, and risk surfaces
late — EXCEPT that Phase T0 front-loads the two genuine kill-risks (SDPA-bwd, checkpoint-offload)
so they're proven at week 1. With those de-risked, the remaining 60+ arms are volume against a
parity harness, which is exactly what engine-complete-first is good at: one complete, uniformly
parity-tested library rather than per-model partial engines that drift. The kill-gate is end of T0.

## §8 — References (read before executing each phase)
- flame-core `src/autograd.rs` (tape engine — the port source), `adam.rs`, `adam8bit_kernel.rs`,
  `gradient_clip.rs`, `src/cuda/*flash*bwd*.cu`, `kernels/*_backward.cu`, `cuda/conv2d_nhwc_bf16.cu`.
- flame-core `docs/TENETS.md` (Tenet 1 fix-the-primitive, Tenet 4 measurement) + `FLAME_KERNELS.md`.
- EriDiffusion `BACKLOG_qwen_cudnn_sdpa_bwd_misalign.md` (the live SDPA-bwd alignment bug — port
  must not inherit it) + `HANDOFF_2026-05-30_L2P_CUDNN_SDPA_BWD_REAL_CAUSE.md`.
- EDv2 `crates/eridiffusion-core/src/training/` (loop policy — Phase T6 source).
- mojodiffusion `PLAN.md` (inference foundation this builds on), `FULL_PORT_ROADMAP.md`,
  `serenitymojo/docs/PORT_STATUS_2026-05-29.md`, `FULL_PORT_RUST_OFFLOAD_PARITY_2026-05-27.md`,
  `serenitymojo/MAP.md`.
- EriDiffusion memory: `project_mojo_vs_flame_perf_bench_2026-05-29` (attention is 7-31× slower in
  Mojo — relevant to backward perf), `project_bf16_rope_pattern_audit`, `feedback_no_flash_attention`.
