# Project Credo

This repo is built to product standards, not demo standards.

- Build quality and functionality first. A compile, smoke test, or tiny artifact is evidence, not a finish line.
- Treat "production ready" as a measurable claim: real outputs, real dimensions, real runtime behavior, and explicit limits.
- Keep runtime work Mojo-native. Python is allowed for development support, parity oracles, inspection, and tooling, not as a substitute for the product path.
- Do not hide weak results behind labels. If an output is a smoke test, call it a smoke test. If it is too short, too small, too slow, or missing a production feature, say so and fix the path.
- Prefer careful engineering over theatrical progress. Read the surrounding code, preserve existing architecture, and make changes that future maintainers can reason about.
- Verify the user-visible artifact whenever the task is about generation. For video, inspect duration, frame count, resolution, muxing, and audio behavior.
- Measure resource reality. VRAM, token count, frame count, and runtime determine whether a path is production, brute force, or only a gate.
- Leave the repo better for the next agent: document hard limits, command lines, outputs, and remaining gaps honestly.

## Runtime Dtype Contract

Mojo runtime code must preserve model storage dtype at tensor boundaries.
For LTX/LTX2 inference this means BF16 activations, latents, checkpoint weights,
biases, LoRA factors/deltas, connector hidden states, VAE tensors, audio mel
tensors, and generated noise unless a checkpoint tensor is explicitly FP8/F16.

- Do not upcast BF16 tensors to F32 just to store them, pass them between model
  stages, or make a parity gate easier.
- F32 is allowed for compute internals only: GEMM accumulators, reductions,
  norm math, attention score math, scalar schedules/sigmas, host statistics,
  debug inspection, Python/oracle dumps, and file-format conversion where the
  external format requires it.
- If an op needs F32 arithmetic, the op/kernel should cast internally and return
  the input/storage dtype. Follow the existing Flame/Core style: BF16 in and out,
  F32 inside compute.
- Random/noise generation must match the reference contract. PyTorch parity must
  use PyTorch-generated oracle noise or a proven matching RNG. Mojo-native RNG is
  not automatically same-seed equivalent to `torch.Generator`/`torch.randn`.
- Any intentional F32 boundary in production code must be justified in a nearby
  comment with the exact reference reason. Otherwise treat it as a bug.
- Before claiming the LTX2 runtime dtype path is fixed, run
  `python3 scripts/check_ltx2_dtype_contract.py`. It is a static guard for the
  known production anti-patterns; passing it is necessary, not sufficient.

The standard is not "we got something to run." The standard is "a person can use this and trust what it claims."

## OneTrainer -> Mojo Skill Pack

For OneTrainer-to-Mojo model work, Codex and Claude Code should use the shared
workflow skills listed in `ONETRAINER_MOJO_SKILLS.md`. The canonical skill
files are mirrored under `.claude/skills/` and `.codex/skills/`. They cover
intake, shared Mojo model runtime/inference, training port, sampler port, parity
gates, production use, and agent handoff.
