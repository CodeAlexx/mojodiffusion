---
name: mojo-port
description: Orchestrates the proven build→skeptic→bugfix→parity loop for porting a model component to PURE MOJO + MAX in serenitymojo (/home/alex/mojodiffusion). Inference-only, GPU-only. Spawns a builder agent, a skeptic agent, and a bugfix agent; gates each chunk on cos≥0.999 parity vs a diffusers GPU-bf16 oracle. Use when building/porting any serenitymojo module (a DiT, encoder, VAE, sampler, op). NOT for Rust/EriDiffusion (that's /port-*) and NOT for Python runtime code — this is pure Mojo.
---

# mojo-port

Orchestrate the multi-agent loop to port a serenitymojo component. **You are the orchestrator — you spawn agents, you do NOT write the Mojo yourself unless the user asks.** Builder writes, skeptic attacks, bugfix repairs, parity gates.

> **SCOPE GUARD (read first).** This is **pure Mojo + MAX, inference-only, GPU-only** (serenitymojo at `/home/alex/mojodiffusion`). There is **NO Rust, NO cargo, NO flame-core, NO autograd/training/LoRA, NO Python in the runtime path.** If a prompt or finding mentions any of those, it leaked from the Rust `/port-*` skills — strip it. Reference impls (diffusers/transformers Python) are **parity oracles only**, run offline under `serenitymojo/.../parity/`, never imported by shipped Mojo.

## §0 — Receipt-of-reading gate (before spawning any agent)

Agents only see what you put in their prompt. State in **one sentence each**:

1. `/home/alex/mojodiffusion/serenitymojo/MAP.md` — receipt: which dir the new module lives in + which foundation ops (`ops/`, `tensor.mojo`, `io/`) it reuses so the builder does NOT reimplement them.
2. `/home/alex/mojodiffusion/serenitymojo/docs/SERENITYMOJO_MODULES.md` — receipt: the exact signatures of the foundation ops/structs this chunk calls.
3. The **reference impl** to mirror (e.g. `diffusers/models/transformers/transformer_*.py`, or `inference-flame/src/models/*.rs` for architecture only) — receipt: name the file + the exact forward the builder must reproduce.
4. `/home/alex/.claude/skills/mojo-syntax/SKILL.md` (the current-Mojo correction layer) — receipt: name the 1-2 syntax traps most relevant to this chunk.

Then proceed.

## The loop

```
plan a chunk → builder agent → skeptic agent → (bugfix agent ⇄ skeptic)* → parity gate (cos≥0.999) → next chunk
```

1. **Plan the chunk.** Smallest independently-parity-testable unit (e.g. "rope + one block", "the t_embedder + adaLN", "patchify+unpatchify"). NOT a whole model at once. Name the foundation ops it reuses and the weight names it loads (no silent renames).

2. **Spawn the builder.** `general-purpose` subagent. Prompt template: `builder-prompt.md` (this dir) — fill in the chunk + reference. The builder writes the Mojo, compiles it with `pixi run mojo run -I .` (a probe `main()`), reports.

3. **Spawn the skeptic** (fresh agent — never the builder). Prompt: `skeptic-prompt.md`. It assumes the port lies and hunts Mojo + numerical bugs. Writes `SKEPTIC_FINDINGS_<date>.md`. Triage each: BLOCKER / FRAGILE / STYLE / DISAGREE.

4. **Spawn the bugfix agent** for BLOCKERs (fresh agent). Prompt: `bugfix-prompt.md`. Minimal targeted fixes, recompile. Re-skeptic until clean.

5. **Parity gate.** Build/run a parity driver: `serenitymojo::parity::ParityHarness.compare(my_tensor, diffusers_ref, ctx)` must hit **cos ≥ 0.999**. The reference comes from a **GPU bf16** diffusers oracle (see Parity rules). Only then mark the chunk done.

## Run/build facts (Mojo, NOT cargo)

- Run anything: `cd /home/alex/mojodiffusion && pixi run mojo run -I . <path/to/file.mojo>`. The `-I .` is REQUIRED for package imports (without it: misleading "module not found").
- Use **`mojo run` (JIT)**, NOT `mojo build` (AOT currently fails on a libm `sinf` link — known blocker, separate task).
- A "compiles" check = a file with a `def main() raises:` probe that imports + calls the new code and runs to exit 0. **A failed compile can print partial output that looks like a pass — always check the actual error, not just that something printed.**

## Mojo gotchas the agents MUST know (bake into prompts)

- Syntax era: **`comptime`** not `alias`; `def` needs explicit **`raises`**; **`ref` is a reserved word** (don't name a var `ref`); `List`/`Dict` literals via brackets.
- `Tensor` is **Movable-not-Copyable** → store as `List[ArcPointer[Tensor]]`, return with `^`, never `List[Tensor]`.
- Lifetime: origin-bound spans (`origin_of(self)`), `from_view` infers origin.
- The stdlib `nn` fused ops have **TileTensor+closure variants that are UNCALLABLE from LayoutTensor** (the `gamma.origin.mut` wall) → hand-roll those (rms_norm, softmax, apply_rope, modulate). Plain-LayoutTensor variants ARE callable: `linalg.matmul`, `conv2d`, `flash_attention`.
- SDK `flash_attention` **fails to compile at Dh=128 on sm_86** → `ops/attention` uses a math-mode SDPA fallback. Sequence length is **comptime** (`sdpa[B,S,H,Dh]`); enumerate supported S or pad to a supported one.
- **ALL file I/O goes through `io/ffi`** (`sys_open`/`sys_pread`/`sys_pwrite`) — NEVER the stdlib builtin `open` or `Path.read_text`, and NEVER `external_call["write"]` (collides with `print`'s write; use `pwrite`). Builtin `open` symbol collides with ffi's at LLVM lowering.
- F32-accumulate / BF16-store is the kernel idiom. For an **accumulating denoise loop, keep the latent in F32**, cast to bf16 only to feed the model (matches diffusers `latents.to(dtype)`).

## Parity rules (the gate — non-negotiable)

- Oracle = the reference framework on **GPU in bf16**. For diffusers: use `/home/alex/serenityflow-v2/.venv/bin/python` (CUDA torch + diffusers). **NEVER fp32-CPU** (CPU-vs-GPU-bf16 diverge at cos≈0.5/layer — a useless reference). **NEVER fp32-host-load a full multi-GB model** (caused a 60GB OOM that killed a session) — load bf16 on GPU, one big model at a time, `torch.no_grad()` + `torch.cuda.empty_cache()`.
- Compare against **fresh single-sample** references. A reference captured from a *batched* (CFG) forward hook can mismatch a single-sample forward and produce a false "bug."
- **cos is magnitude-blind.** Also check the **magnitude ratio** (`|mine|/|ref|`) and, for any loop that accumulates, a **per-step** comparison — a cos-0.9999 op can still hide a magnitude bias or a difference amplified by a guidance/cancellation term.
- Oracle Python lives under `serenitymojo/<area>/parity/`. It is a DEV TOOL, never imported by shipped Mojo (the pure-Mojo rule is about the shipped binary).

## Anti-patterns

- Don't write the whole model in one builder call — chunk it, skeptic between chunks.
- Don't accept builder output without a compile check (and read the ERROR, not just exit).
- Don't let an agent reimplement a foundation op that `MAP.md`/`SERENITYMOJO_MODULES.md` already provides.
- Don't have one agent both build AND skeptic — adversarial review needs fresh eyes.
- Don't let any Rust (`cargo`, `flame_core`, autograd) or Python-runtime concept leak into a Mojo prompt or fix.
- Don't trust a single cos number for an accumulating loop — magnitude + per-step too.
