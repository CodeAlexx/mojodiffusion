# Claude Code Read-Only Dtype Audit Prompt

Use this prompt with Claude Code from the repository root:

```text
You are auditing /home/alex/mojodiffusion in READ-ONLY mode.

Do not edit files. Do not create files. Do not run formatters. Do not run git
reset, checkout, restore, clean, or any destructive command. Your job is to find
more Runtime Dtype Contract violations and report them with exact evidence.

Project contract:
- Mojo runtime code must preserve model storage dtype at tensor boundaries.
- BF16 activations, latents, checkpoint weights, biases, LoRA factors/deltas,
  connector hidden states, VAE tensors, audio mel tensors, and generated noise
  should remain BF16 unless a checkpoint tensor is explicitly FP8/F16.
- F32 is allowed for compute internals only: GEMM accumulators, reductions,
  norm math, attention score math, scalar schedules/sigmas, host statistics,
  debug inspection, Python/oracle dumps, and file-format conversion where the
  external format requires it.
- If an op needs F32 arithmetic, it should cast internally and return the
  input/storage dtype. Flame/Core style: BF16 in/out, F32 inside compute.
- Any intentional F32 boundary in production code must be justified by a nearby
  comment with the exact reference reason.

Audit targets:
1. Search the entire repo for production Mojo files with suspicious F32 storage:
   - `STDtype.F32`
   - `DType.float32`
   - `Float32`
   - `to_host(`
   - `from_host(`
   - `cast_tensor(... STDtype.F32`
   - `enqueue_create_buffer[DType.uint8](... * 4`
   - `return Tensor(... STDtype.F32`
   - guards like `must be F32`, `dtype() != STDtype.F32`
2. Exclude or clearly separate tests, parity/oracle files, probes, smokes,
   debug/stat collection, file-format conversion, and optimizer master-state
   paths unless they leak into production runtime.
3. For each suspected issue, inspect surrounding code before reporting. Do not
   report legitimate F32 accumulators, scalar schedules, attention score scratch,
   norm reductions, BLAS C buffers that are immediately cast back, or PyTorch
   oracle/debug paths as bugs.
4. Prioritize:
   - public ops returning F32 for BF16/F16 inputs,
   - loaders/stacks that upload checkpoint weights/biases as F32,
   - full tensor casts to F32 before model blocks,
   - F32 device buffers that replace model dtype storage,
   - BF16/F16 production paths rejected by F32-only guards,
   - stale docs/comments that claim F32 storage after implementation preserves
     dtype.

Known context already found:
- `conv1d` had F32 weight/bias staging and was fixed.
- `shape_backward.where_backward` had F32 condition-mask handling and was fixed.
- `reduce_var` / `reduce_std` were found hardcoding F32 output; check current
  tree state before reporting because this may already be under repair.
- `mxfp4.mojo`, `norm.mojo`, `rope.mojo`, `unary.mojo`, `activations.mojo`,
  `patchify3d.mojo`, `layout.mojo`, `activation1d.mojo`, and `pixelshuffle`
  patchify paths were audited and looked dtype-preserving as of the prior pass.

Output format:
Return a concise Markdown report with these sections:

## High Confidence Bugs
For each:
- `file:line`
- exact code pattern
- why it violates the Runtime Dtype Contract
- likely fix pattern
- confidence: high/medium

## Needs Human Decision
For cases that might be intentional F32 boundaries but lack a nearby reference
comment.

## False Positives Checked
List important suspicious-looking F32 uses that are allowed and why.

## Suggested Verification
List the smallest existing smoke/parity commands to run after fixes. Do not run
long training or full generation unless explicitly asked.
```
