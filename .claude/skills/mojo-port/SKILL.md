---
name: mojo-port
description: Entry point + orchestrator for porting a model to PURE MOJO + MAX in serenitymojo (/home/alex/mojodiffusion). FIRST asks intent — inference, training, or both — and routes training work to the mojo-train-port skill; itself drives the proven build→skeptic→bugfix→parity INFERENCE loop, spawning builder/skeptic/bugfix agents and gating each chunk on cos≥0.999 vs a GPU-bf16 reference oracle. Use when building/porting any serenitymojo module (a DiT, encoder, VAE, sampler, op) or when the user invokes /port-style work on the Mojo stack. NOT for Rust/EriDiffusion (that's /port-*) and NOT for Python runtime code — this is pure Mojo.
---

# mojo-port

Orchestrate the multi-agent loop to port a serenitymojo component. **You are the orchestrator — you spawn agents, you do NOT write the Mojo yourself unless the user asks.** Builder writes, skeptic attacks, bugfix repairs, parity gates.

## §−1 — Intent gate (ASK FIRST, before §0 — mirrors the Rust `/port` router)

This skill is the entry point for Mojo-stack porting. The Mojo stack splits inference and training into two skills (this one + `mojo-train-port`), so resolve intent before doing anything. **If the user already made it explicit** (named a sampler/VAE/encoder/op for inference, or said "training/LoRA/fine-tune"), skip the question and proceed. **Otherwise** ask ONE `AskUserQuestion` ("Port intent" / "Is this port for inference, training, or both?"):

- **Inference** → stay in THIS skill. The SCOPE GUARD below applies (pure-Mojo, no autograd). Run the §0 receipt-of-reading then the loop.
- **Training (LoRA or full FT)** → **route to the `mojo-train-port` skill** and follow its procedure (shared `training/` pipeline + thin per-model surface; per-block torch-AUTOGRAD parity; the L2P "loss drops AND a sample shifts" gate). Do NOT run the inference loop's scope guard there — training legitimately needs autograd/backward.
- **Both** → do **inference first in this skill** (a verified forward is the oracle + the activation-saving template the backward mirrors), THEN hand to `mojo-train-port` for the backward/LoRA. Say so up front.

Don't pad the interview — one question, route, go. If the user typed a concrete inference target, just start; the question is only for genuine ambiguity.

> **SCOPE GUARD (inference branch — read first).** This is **pure Mojo + MAX, inference-only, GPU-only** (serenitymojo at `/home/alex/mojodiffusion`). There is **NO Rust, NO cargo, NO flame-core, NO autograd/training/LoRA, NO Python in the runtime path.** If a prompt or finding mentions any of those, it leaked from the Rust `/port-*` skills or belongs in `mojo-train-port` — strip it. Reference impls (diffusers/transformers Python) are **parity oracles only**, run offline under `serenitymojo/.../parity/`, never imported by shipped Mojo.

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

## Methodology refinements (NAVA worked example, 2026-06-07 — a full 30-layer audio-video MMDiT + umt5 encoder taken to e2e parity)

Techniques that made a deep, multi-component port go fast and stay honest. Apply them; they generalize.

- **Own the gate — re-run every builder claim yourself.** A builder reporting "cos 0.99999" is a HYPOTHESIS until you re-run it on a clean serial build (`rm -f serenitymojo.mojopkg && pixi run mojo run -I . <probe>`) and read the number. Every chunk this session was reran by the orchestrator before counting. Never bank a builder's printed number.
- **Isolate a deep component with oracle forward-hooks.** To gate block N of a stack without re-deriving its inputs, dump the EXACT inputs from the reference: `module.register_forward_pre_hook(fn, with_kwargs=True)` captures `x` + every kwarg (modulation, context, rope, grid) byte-for-byte. Gate the Mojo component against those + the captured output. This turned "port a 30-layer DiT" into independently-gatable chunks (one double block, one single block) each at cos 0.99999.
- **Gate every piece, THEN wire — the integration run is the wiring test.** Build+gate each sub-unit alone (embeddings, each block kind, encoder), then compose the full forward as a final gate. A wrong unpatchify permute / wrong residual / swapped weight tanks the e2e cos far below 0.999, so the e2e gate IS the test for the POST/wiring you never unit-tested. (NAVA: 4 embedding paths + 2 block kinds gated → full DiT wired → vel_vid/vel_aud cos 0.9996/0.9998.)
- **VERIFY "reuse" claims against the actual file before trusting the reuse map.** "umt5 = reuse t5_encoder" was false (umt5 needs per-LAYER relative bias, not shared-from-block-0); "video VAE exists" was a first-frame/T=1 SLICE, not a 5-frame decoder. Grep the target file's status/scope (header comment, `def` list, "slice" vs "FULL") before scoping a chunk as reuse. A reuse map is a hint, not a guarantee.
- **Follow the oracle's DTYPE — match production, don't bend the oracle to the port.** The reference's production dtype is truth. If the reference runs a component in F32 (e.g. NAVA's `init_ltx_vae` builds the audio VAE `dtype=torch.float32`), the faithful port runs F32 too — do NOT lower the oracle to bf16 to match a bf16 port (that's backwards; it hides the port's error). Measure BOTH dtypes to size the gap: NAVA audio VAE was bf16-Mojo-vs-F32-oracle cos 0.9999966 / max_abs 0.0625, but F32-vs-F32 cos 0.9999999982 / max_abs 0.0016 (~40× tighter). When the serenitymojo module is bf16-by-default (the house idiom), add a backward-compatible `f32: Bool=False` load flag (`Tensor.from_view_as_f32`) so the same code serves a bf16-production sibling AND an F32-production one.
- **"Coded" ≠ "works" — gate every never-run module, and don't trust your own "not ported" claim without grepping.** Two serenitymojo audio modules (the LTX VAE decoder, then the BigVGAN vocoder+STFT) were fully written but never executed; one I wrongly called "not ported" until a `find`/grep surfaced `models/vocoder/ltx2_vocoder.mojo`. A header saying "FULL DECODE" is a HYPOTHESIS; run it on the real checkpoint. (The decoder DID work as-is once pointed at NAVA's keys + run in F32 — but that was proven by a gate, not assumed.)
- **F32→BF16 rounding accumulates in loops — cos is blind to it (NAVA, 2026-06-07).** Mojo's native `Float32.cast[bfloat16]()` (what `cast_tensor` uses) differs from PyTorch's round-to-nearest-even by ~1 BF16 quantum on some values. ONE cast: invisible (cos≈1). But the per-step `cast(latent_f32→bf16)` feeding a model inside an N-step **sampling loop** compounds the bias and **decorrelates from torch** — NAVA's 25-step × 3-forward × CFG-7× loop produced a *visibly distorted* decode while every single-forward gate still read cos 0.9998. The global std even matched torch (it's a per-element bias, not magnitude drift), so std-checks miss it too. **Fix: use `ops/torch_bf16.mojo::torch_f32_to_bf16_rne` for any F32→BF16 feeding a model in a denoise loop** (the latent + text casts), not `cast_tensor`. (OneTrainer hit the same thing — `onetrainer-mojo/.../bf16_stochastic_rounding.mojo`.) Symptom signature: pieces gate fine, the assembled *loop* output is subtly wrong and worsens with step count. When a loop output looks off but every component gates, suspect bf16-rounding accumulation BEFORE re-auditing the math.
- **Make the oracle apples-to-apples with the Mojo math.** Monkeypatch the reference's `flash_attention` → torch SDPA (same math as the Mojo `ops/attention` math-mode fallback). A side effect to EXPLOIT (after verifying per-model): SDPA-via-monkeypatch usually ignores `k_lens`, so there's NO attention masking — the Mojo side uses `sdpa_nomask`/`sdpa_cross_nomask` directly. And for b=1, joint-attention gather/scatter token-reorders are often the identity (all tokens valid) — prove it, then skip.
- **Rectangular attention has its own op.** Self-attn = `sdpa_nomask[B,S,H,Dh]` (square); cross-attn (q_len≠kv_len) = `sdpa_cross_nomask[B,Sq,Skv,H,Dh]`. Don't pad cross to square.
- **Reference weights in a non-safetensors format need a prep conversion.** Mojo `io` reads safetensors only. A pickle `.pth` (e.g. umt5) → convert once with a tiny `load → save_file` python script, run it in the background while other prep proceeds. Watch RAM if the oracle also holds a copy.
- **Isolate an encoder from its tokenizer.** Dump the reference's token IDs (a tokenizer call in the oracle), feed those exact IDs to the Mojo encoder, gate the encoder output alone. Tokenization parity is a separate (later) concern.
- **Deep-stack accumulation: expect lower cos than per-block, and check magnitude.** A 30-layer fp8 stack landed at vel cos 0.9996 vs per-block 0.99999 — normal accumulation, still passes. But report `max_abs` vs the target's std; cos alone can hide a magnitude drift in an accumulating path.
- **Keep a live handoff doc + memory parity table, updated per chunk.** A running table (chunk | probe | cos) in `docs/<MODEL>_HANDOFF_*.md` + the project memory makes cold-start resume trivial and records the non-obvious facts (the monkeypatch trick, the per-layer-bias gotcha, an unfixed fragility like a hardcoded grid) so the next session doesn't re-derive them.
- **Don't sprawl: bank clean gated units, scope the next subsystem before diving.** When a chunk's "reuse" turns into a genuine new build (the video VAE temporal loop), name it, scope it, and treat it as its own focused unit rather than grinding it onto the end of a long session. One verified deliverable beats three half-built ones.

## Anti-patterns

- Don't write the whole model in one builder call — chunk it, skeptic between chunks.
- Don't accept builder output without a compile check (and read the ERROR, not just exit).
- Don't let an agent reimplement a foundation op that `MAP.md`/`SERENITYMOJO_MODULES.md` already provides.
- Don't have one agent both build AND skeptic — adversarial review needs fresh eyes.
- Don't let any Rust (`cargo`, `flame_core`, autograd) or Python-runtime concept leak into a Mojo prompt or fix.
- Don't trust a single cos number for an accumulating loop — magnitude + per-step too.
