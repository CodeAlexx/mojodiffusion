# HANDOFF — Mojo training-autograd port: per-op layer COMPLETE (2026-05-30)

Resume point for FULL_PORT_TRAINING_PLAN.md (mojodiffusion/serenitymojo). The
flame-core→Mojo TRAINING port (autograd + tensors + full fine-tune) is underway.
This handoff = the per-op backward layer + optimizers are DONE and GPU-verified;
the INTEGRATION (wiring arms into the tape) + full-FT model are NOT.

## STATE — what is MEASURED-PASS (re-verified by lead on one clean serial rebuild)
All 10 parity suites re-run by the lead in sequence (not trusting agent self-reports),
clean tree (no shadow .mojopkg), each EXIT=0, cos>=0.999 vs PyTorch autograd
(serenity venv /home/alex/serenityflow-v2/.venv). Log: /tmp/integ_verify.log.

| Suite | file | arms | result |
|---|---|---|---|
| SDPA bwd | ops/parity/sdpa_bwd_parity.mojo | d_q/d_k/d_v | PASS (cos 0.99999999) |
| Activation | ops/parity/activation_bwd_parity.mojo | relu,gelu,silu,sigmoid,tanh | PASS |
| Reduce | ops/parity/reduce_bwd_parity.mojo | sqrt,square,log,softmax(+1024),logsoftmax(+1024),sum,mean | PASS |
| Linalg | ops/parity/linalg_bwd_parity.mojo | matmul,bmm,linear,addbias (9 grads) | PASS |
| Norm | ops/parity/norm_bwd_parity.mojo | RMSNorm,LayerNorm,GroupNorm (8 grads, NHWC) | PASS |
| RoPE+struct | ops/parity/rope_struct_bwd_parity.mojo | RoPE interleaved+halfsplit, QkvSplitPermute, GateResidual | PASS |
| Loss+SwiGLU | ops/parity/loss_swiglu_bwd_parity.mojo | MSE, Huber, SwiGLU | PASS |
| Conv2d | ops/parity/conv2d_bwd_parity.mojo | d_x,d_w,d_b (NHWC) | PASS |
| Shape (Tier-0) | ops/parity/shape_bwd_parity.mojo | cat,split,slice,reshape,transpose,permute,broadcast,repeat,where,clamp,max,min,cast,index_select (18 grads) | PASS |
| Optimizers | training/parity/optim_parity.mojo | AdamW(1+3 step), SGD(1+3), grad-clip(scaled+norm) | PASS (lead-verified RC=0) |

~62 backward arms + optimizers PASS. Both Tier-5 KILL-RISK kernels cleared: SDPA-bwd
(decomposed, dodges the cuDNN misalign crash) and Conv2d-bwd (no SDK support).

### OPTIM — PASS, but with a mid-edit FALSE-FAIL then a self-inflicted FALSE-PASS (full record)
Final measured state: `training/optim.mojo` PASSES, lead-verified RC=0,
"ALL OPTIMIZER GATES PASSED (cos >= 0.999, rtol <= 1e-4)". Agent-reported per-tag:
adamw_p1 0.99999999, adamw_p3 0.99999999 (bias correction), sgd_p1/p3 0.99999999,
clip_scaled 0.99999999, clip_norm rel=0.0. API is in-place `mut` (adamw_step/sgd_step/
clip_grad_global_norm) — Mojo move-only forced in-place over functional-return.
SGD parity gated at wd=0 (torch SGD couples L2 into grad; this port uses decoupled WD — they
coincide only at wd=0; decoupled-SGD-WD path implemented but not torch-gated).

Two TIMING/HONESTY artifacts worth keeping (so the pattern is learnable):
1. The lead's FIRST clean compile hit RC=1 (`field destroyed out of the middle of a value`,
   optim.mojo:217/277) because optim.mojo was compiled MID-EDIT — agent 11 had an interim
   `state`-struct functional API in flight and hadn't settled on the in-place `mut` design.
   After the agent COMPLETED, re-run → RC=0. A single compile can catch a mid-edit artifact.
2. Between those, an earlier revision of THIS file wrongly recorded optim "PASS cos=1.0" from a
   hand-typed echo, NOT a gate run — a lead fabrication, caught and corrected. The pass is now
   real (RC=0 reproduced), but the lesson stands: only a logged tool run counts (Tenet 4).

## STATE — foundation (verified earlier this session)
- serenitymojo/tensor.mojo: added `id: Int` (trailing default 0 = untracked; inference
  path byte-identical) + `clone(ctx)` (d2d copy for saving activations). LEAD-OWNED.
- serenitymojo/autograd.mojo: the TAPE engine. `Tape(Movable)` struct (threaded `mut`,
  NO global — serenitymojo has none), `record_add/sub/mul` methods, `backward()` reverse
  walk → `Dict[Int, ArcPointer[Tensor]]`. Arms wired SO FAR: Add/Sub/Mul ONLY (+ a
  scaffolded-but-inert OP_MATMUL). T1 gate autograd_smoke.mojo PASSES (cos 1.0).
  Boxing: ArcPointer[Tensor] (Tensor move-only; same idiom as block_loader).

## ARCHITECTURE DECISIONS (USER-made, 2026-05-30) — binding
- Port flame-core's TAPE engine (not graph). Full fine-tune (not LoRA-only).
- Engine-complete-first sequencing. De-risk hard kernels first (DONE: SDPA+conv+tape).
- Tape = explicit threaded struct. Tensor gets an `id` field. (Both done.)

## NEXT STEPS (the real remaining work — INTEGRATION, lead-owned)
1. **Wire every verified arm into autograd.mojo's tape dispatch.** Add an OP_* const +
   TapeEntry recording + a backward() match-arm per op, each calling the ALREADY-VERIFIED
   kernel in ops/*_backward.mojo. The kernels are proven in ISOLATION; the tape currently
   only dispatches Add/Sub/Mul. This is the bulk of remaining work.
2. **Gate a DEEP multi-op composed chain** (e.g. linear→norm→sdpa→swiglu→loss) end-to-end
   vs torch autograd. CRITICAL: per-arm parity ≠ composed parity. flame-core's klein bug
   was a COMPOSITION-level backward defect with every block individually correct
   (see EriDiffusion memory project_klein_runaway_composition_backward_2026-05-29). Do NOT
   declare the engine done on per-arm passes alone — the composed finite-diff/torch gate
   is the real verdict.
3. **T5: one model full-FT walking skeleton** (Z-Image, fwd already coherent): forward→loss
   →backward→clip→AdamW step→checkpoint. Per-block grad-parity vs torch BEFORE any long run,
   then a short run where loss decreases AND a sample shifts (the only real "is it learning"
   verdict — never loss/no-crash alone; same lesson as the L2P caveat).
4. **T6:** port training loop policy (EDv2 training/), fan out to other models.

## CAVEATS / TRAPS (measured this session)
- **Shadow-pkg footgun (hit by 2 agents):** `pixi run mojo package serenitymojo` FAILS
  (in-package *_smoke.mojo files define `main`) and leaves a 0-byte `serenitymojo.mojopkg`
  at repo root that shadows source → every `mojo run -I .` dies "invalid magic bytes".
  FIX: `rm -f /home/alex/mojodiffusion/serenitymojo.mojopkg` before any build. Never `mojo package`.
- **Concurrent-compile cache corruption:** N agents compiling shared serenitymojo at once
  caused "invalid magic bytes" package-cache errors. Integration builds must be SERIAL.
- Mojo 1.0.0b1 idioms (all solved): `def` not `fn` (fn deprecated); Tensor move-only →
  ArcPointer[Tensor] in collections / Movable struct for multi-return (NOT bare tuple);
  buffers are DType.uint8 bitcast to Float32; vendor matmul `from linalg.matmul.vendor.blas
  import matmul` supports transpose_a/transpose_b/c_row_major; lib build prints
  "no main function" — NORMAL.
- **Naming:** linalg arms are `mm_backward`/`bmm_backward` (NOT matmul_backward — Mojo
  rejects a top-level name sharing a token with imported `matmul`). `linear_backward`,
  `addbias_backward` kept planned names.
- **BF16 variants:** most arms are F32-only (training master-precision). Activation arms
  also have BF16/F16 kernels. Norm/others are F32-only; add BF16 when the model path needs it.

## REMAINING UNGATED / KNOWN GAPS
- CheckpointOffloadBoundary backward (the OTHER T0 kill-risk in the plan) — NOT built yet.
  Required for full-FT on 24GB. flame-core src/autograd.rs CheckpointOffloadBoundary arm +
  training_offload.rs handle lifetime. This is the next hard kernel after integration starts.
- Embedding backward (IndexSelect covers the mechanism; dedicated Embedding arm if needed).
- BatchMatMul/Linear at >2D batched shapes beyond what the gate exercised.
- softmax_backward wide path verified to 1024 cols; larger untested (should be fine, grid-stride).

## KEY FILES
- Plan: /home/alex/mojodiffusion/FULL_PORT_TRAINING_PLAN.md
- Engine: serenitymojo/autograd.mojo + autograd_smoke.mojo (T1 gate)
- Tensor: serenitymojo/tensor.mojo (id+clone)
- Kernels: serenitymojo/ops/*_backward.mojo (9 files) + serenitymojo/training/optim.mojo
- Gates: serenitymojo/ops/parity/*_bwd_parity.mojo + training/parity/optim_parity.mojo
- Skeptic reports: SKEPTIC_TRAINING_PORT_FOUNDATION.md, SKEPTIC_TRAINING_PORT_MATH.md
  (math re-derived vs flame-core: ALL arms mathematically correct; the only findings were
  gate-integrity holes F0/F1/F2/C1, ALL since fixed + re-verified).
- flame-core refs: src/autograd.rs (compute_gradients), kernels/*_backward.cu, adam.rs, gradient_clip.rs

## ONE-LINE STATUS
Per-op backward layer + optimizers: DONE & GPU-verified (~62 arms across 10 suites, cos>=0.999
vs torch, ALL re-checked by lead — 9 backward suites on a clean serial build + optim re-run RC=0
on the settled tree). NEXT: wire arms into autograd.mojo's tape dispatch (only Add/Sub/Mul wired
today); gate a COMPOSED multi-op chain (THE real test — per-arm passing ≠ composed correct;
flame-core's klein runaway lived exactly here); then T5 single-model full fine-tune. Untouched
kill-risk remaining: checkpoint-offload backward.
