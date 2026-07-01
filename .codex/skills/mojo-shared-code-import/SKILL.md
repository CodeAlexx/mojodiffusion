---
name: mojo-shared-code-import
description: Add or import shared Mojo runtime/training code in mojodiffusion. Use when creating new shared modules, moving model code into serenitymojo/ops or serenitymojo/training, wiring imports across product trainers, adding reusable ABI surfaces, or coordinating Codex/Claude Code implementation of new Mojo infrastructure without breaking dtype, parity, or existing trainers.
---

# Mojo Shared Code Import

Use this before adding reusable Mojo code or importing a new shared helper into
Krea2, ZImage, Klein, Ideogram/Qwen-style trainers, or future model ports.

## First Pass

1. Inspect current ownership with `rg` and nearby imports before coding.
2. Classify the change:
   - `serenitymojo/ops/`: model-agnostic math kernels and wrappers.
   - `serenitymojo/training/`: trainer ABI, loss, norm/clip, optimizer, perf, arenas.
   - `serenitymojo/models/<model>/`: model topology, key mapping, weight loading.
   - `/home/alex/serenity-trainer/src/...`: product loop wiring only.
3. Prefer extending an existing shared module over creating another parallel
   helper. Create a new module only when ownership is genuinely shared.
4. Check dirty worktrees in both repos. Do not revert unrelated user/agent work.

## Dtype Contract

- Preserve BF16/F16/FP8 storage dtype at checkpoint, cache, activation boundary,
  LoRA param, save, and inter-stage tensor boundaries.
- Use F32 only for compute internals: GEMM accumulators, reductions, attention
  score math, scalar schedules, grad stats, optimizer grads/moments, host stats,
  debug/oracle dumps, and external file conversion where required.
- If a kernel computes in F32, cast internally and return the input/storage dtype
  unless a checked reference requires otherwise.
- Never add an intentional F32 production boundary without a nearby comment that
  names the exact reference reason.
- For LTX/LTX2-adjacent runtime changes, run `python3 scripts/check_ltx2_dtype_contract.py`
  before claiming dtype safety.

## Import And ABI Rules

- Keep imports explicit and local to the owner module. Avoid import cycles by
  putting shared structs below model-specific code only when the dependency is
  one-way.
- Shared trainer interfaces must be model-agnostic: device trainables, device
  grads, loss/root grads, grad norm/clip, optimizer, perf records, and arena
  lifetimes should not encode Krea2/ZImage/Klein naming.
- Product trainers may adapt model-specific key order into the shared ABI, but
  they must not own optimizer plumbing when a shared path exists.
- Host-list grad extraction is compatibility/debug/save only. Fast-path claims
  require no full per-step grad or prediction readback.
- New surfaces must fail loud for unsupported modes before `DeviceContext()` or
  before large allocation, especially full finetune, resume, EMA, sampling, and
  non-default optimizer/loss combinations.

## Implementation Discipline

- Add the smallest working slice: shared primitive, one model adapter, one smoke
  or replay gate, then product wiring.
- Preserve existing model indices and saved key order. If appending trainables,
  document the old range and new range, and validate counts.
- Do not add save-only fake tensors to pass surface checks. Trainable surface
  parity requires forward, backward, optimizer, save, and resume behavior or an
  explicit blocker.
- Keep device lifetimes explicit. If a D2D cast feeds a later kernel, keep the
  tensor alive until the relevant sync or arena scope ends.
- For GPU code, use Mojo GPU idioms: `DeviceContext`, `TileTensor`, explicit
  kernel binding, `ctx.enqueue_function`, and compile-time layouts. Do not write
  CUDA syntax.
- Python is allowed for reference dumps, static guards, safetensor inspection,
  and scripts. It is not a product runtime substitute.

## Verification

Run the narrowest meaningful checks, then the shared guard:

- Build or run the touched Mojo smoke/replay file.
- Build the affected product trainer through its canonical `pixi` task when it
  exists, for example `krea2-live-trainer-build`, `zimage-live-trainer-build`, or
  `klein-live-trainer-build`.
- Run `python3 scripts/check_training_speed_roadmap_contract.py` after roadmap,
  perf, dtype-label, or shared trainer-speed changes.
- Report exact evidence level: compile, smoke, artifact consumer, loss bridge,
  update-bearing, Mojo replay, or production parity.
- State blockers plainly. Do not turn a smoke/build pass into a parity or speed
  claim.

## Agent Split

When using Codex, Claude Code, or subagents together, split by file ownership:

- Builder: one bounded implementation slice and its compile/smoke.
- Verifier: strict gates and command output; no test weakening.
- Skeptic: dtype boundaries, false claims, key order, save/resume, speed/VRAM.

The lead agent must rerun or independently inspect any result before accepting it.
