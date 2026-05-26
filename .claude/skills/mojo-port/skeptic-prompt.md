# Skeptic agent prompt — serenitymojo (pure Mojo + MAX)

Fresh agent, never the builder. Fill `{{...}}`, send to `general-purpose`. **Your job is to find what's wrong, not to be balanced. Assume the port lies.**

---

Adversarially review **{{CHUNK + the files the builder wrote}}** in serenitymojo (`/home/alex/mojodiffusion`). Pure Mojo + MAX, inference-only. Read the builder's files + the reference impl `{{REFERENCE FILE}}` line by line. Write findings to `serenitymojo/{{area}}/parity/SKEPTIC_FINDINGS_{{date}}.md`, each with severity (BLOCKER / FRAGILE / STYLE).

## Check these (categorized so you don't miss systemic issues)

**Reference fidelity (the #1 source of silent bugs)**
- Does the forward match the reference EXACTLY: norm epsilons, scale/gate form (`1+scale`, `tanh(gate)`?), concat order ([image,cap] vs [cap,cap]), rope layout (interleaved adjacent-pair vs half-split), extract layer, timestep convention (`t` vs `1-t`, and any `*t_scale` inside vs outside)?
- Weight names: every tensor in the checkpoint either has a destination or is explicitly skipped with a reason. No silent renames. Split/fused Q/K/V handled right.

**Mojo correctness**
- `Tensor` stored as `List[ArcPointer[Tensor]]` (not `List[Tensor]`); ownership transferred with `^`; no use-after-move.
- No `alias` (must be `comptime`); every fallible `def` has `raises`; no var named `ref`.
- comptime sequence lengths actually match the data fed to `sdpa[B,S,H,Dh]`.
- File I/O exclusively through `io/ffi` (no builtin `open`, no `Path.read_text`, no `external_call["write"]` — use `pwrite`).
- No hand-rolled reimplementation of a foundation op that already exists in `ops/` (call it instead).

**Numerical hazards**
- rope inv-freq / theta / axis-dim split matches reference.
- A rank/shape convention mismatch (e.g. model expects rank-2 `[seq,dim]` but encoder yields rank-3 `[1,seq,dim]`).
- For an accumulating loop: is the accumulator F32 (not bf16)? Is a guidance/CFG term a difference of near-equal vectors (cancellation-sensitive)?
- BF16 vs F32 at op boundaries matches the reference's dtype flow.

**Parity hygiene**
- No workaround added to mask a foundation-op bug (flag the foundation bug instead).
- No env-gates for default-path code; no new deps; no Python in the shipped path; no Rust/cargo/flame-core/autograd concepts (those don't belong here — flag if present).
- The parity oracle (if the builder made one) is GPU bf16, NOT fp32-CPU, and does NOT fp32-host-load a full model.

**Compile-check honesty**
- Did the builder's probe actually compile + run to exit 0, or did partial output get mistaken for success? Re-run `pixi run mojo run -I . <probe>` yourself and read the error.

## Output
For each finding: file:line, what's wrong, why it'll fail parity (or why it's only fragile/style), and the minimal fix. Be specific. End with a one-line verdict: BLOCKERS (count) / clean.
