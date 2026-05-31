# HANDOFF — Mojo training port, Phase T0 SDPA-backward (2026-05-30)

## Where we are
Executing `FULL_PORT_TRAINING_PLAN.md` (just written this session). User approved:
port flame-core tape engine, FULL fine-tune, engine-complete-first, investigate-first.
Phase T0 = de-risk the 2 hard kernels (SDPA bwd + checkpoint-offload) BEFORE the rest.

## What was done this session
1. Wrote `FULL_PORT_TRAINING_PLAN.md` (training scope: 66 backward arms, ~119 bwd
   kernels, 7k-line tape engine, 12k training infra — all measured). Phases T0–T6.
2. Wrote **`serenitymojo/ops/attention_backward.mojo`** — decomposed (math-mode)
   SDPA backward. Mirrors the existing math-mode forward in `ops/attention.mojo`
   (gather BSHD→BHSD F32 → per-head matmuls → scatter). Math ported VERBATIM from
   flame-core `src/autograd.rs:1686 attention_backward_recompute`:
   - recompute attn = softmax(Q@Kᵀ·scale)
   - d_v = attnᵀ@dout ; d_attn = dout@Vᵀ
   - d_logits = attn·(d_attn − rowsum(d_attn·attn))·scale  (fused `_softmax_bwd_rows_f32`)
   - d_q = d_logits@K ; d_k = d_logitsᵀ@Q
   F32 interior (more precise than flame-core's BF16-default recompute; fine for cos≥0.999).
   Decomposed ON PURPOSE — no base-ptr alignment assumption ⇒ does NOT inherit the
   `CUDA_ERROR_MISALIGNED_ADDRESS` crash (BACKLOG_qwen_cudnn_sdpa_bwd_misalign.md).

## COMPILE STATUS — CLEAN (measured 2026-05-30)
`pixi run mojo build -I . serenitymojo/ops/attention_backward.mojo` → **EXIT=0**.
Cleared risks:
- `matmul(..., transpose_a=True, ...)` IS supported by the vendor blas (compiled fine
  in the d_v and d_k paths). No restructure needed.
- Return type must be `Tuple[Tensor, Tensor, Tensor]` (explicit `Tuple[...]`), NOT bare
  `(Tensor, Tensor, Tensor)` — Tensor is move-only; bare-tuple ctor rejects it. Fixed.
  (Pattern confirmed against `ops/layout.mojo:385 deinterleave_pair`.)

COMPILE ≠ CORRECT (Tenet 4). NOT yet parity-verified — that's the real T0 gate below.

## NEXT STEPS (in order)
1. ✅ DONE: compile-check (EXIT=0, transpose_a OK, Tuple return fixed).
2. Write a grad-parity smoke: random Q/K/V/dout (small, e.g. B1 S64 H4 Dh64),
   run sdpa_backward, compare d_q/d_k/d_v vs a flame-core or torch reference
   (cos≥0.999). Use `parity.mojo` ParityHarness. Reference: generate with torch
   `F.scaled_dot_product_attention` + autograd.grad, dump to host List[Float32].
   GATE per FULL_PORT_TRAINING_PLAN §4 Phase T0.
3. Then the 2nd T0 kernel: checkpoint-offload-boundary (read flame-core
   autograd.rs CheckpointOffloadBoundary arm + training_offload.rs handle lifetime).
4. If both T0 kernels parity-pass → T0 kill-gate cleared → proceed to T1 (tape engine).

## Key facts / paths
- Forward SDPA (the template): `serenitymojo/ops/attention.mojo` — math-mode,
  `sdpa[B,S,H,Dh]` + `sdpa_nomask`. BSHD [B,S,H,Dh], mask [B,H,S,S], scale=1/sqrt(Dh).
- Tensor API: `serenitymojo/tensor.mojo` — `Tensor(buf, shape, dtype)`,
  `.from_host(List[Float32], shape, STDtype, ctx)`, `.to_host(ctx)->List[Float32]`,
  `.buf.unsafe_ptr().bitcast[T]()`, `.dtype().to_mojo_dtype()`.
- Parity: `serenitymojo/parity.mojo` — `ParityHarness(cos_threshold).compare(t, ref_list, ctx)`.
- Run: `cd /home/alex/mojodiffusion && pixi run mojo run -I . <file>` (needs `-I .`).
- Mojo 1.0.0b1, RTX 3090 Ti sm_86. matmul: `linalg.matmul.vendor.blas.matmul`.
- flame-core bwd reference: `src/autograd.rs:1686` (recompute) + `:1796` (precise variant).

## Plan docs
- `/home/alex/mojodiffusion/FULL_PORT_TRAINING_PLAN.md` (the training plan, DRAFT).
- `/home/alex/mojodiffusion/PLAN.md` (inference, separate).
- `/home/alex/EriDiffusion/BACKLOG_qwen_cudnn_sdpa_bwd_misalign.md` (the crash this
  decomposed path avoids — qwen 2000-step run died at step 43 on it this session).

## Other live state (not this task)
- qwen 2000-step training run DIED at step 43 (cuDNN SDPA bwd misalign). Backlogged,
  NOT relaunched (user said leave alone). Output: EriDiffusion-v2/output/qwen_alina_2000/.
