# Builder agent prompt — serenitymojo (pure Mojo + MAX, inference-only)

Fill the `{{...}}` and send to a `general-purpose` agent.

---

You are porting **{{CHUNK: e.g. "the Z-Image NextDiT t_embedder + adaLN modulation"}}** to **pure Mojo + MAX** in serenitymojo at `/home/alex/mojodiffusion`. Inference-only, GPU-only. Write idiomatic current-Mojo, reuse the foundation, mirror the reference EXACTLY.

## §0 receipts (state one sentence each before writing)
1. `serenitymojo/MAP.md` — which dir this module goes in + the foundation ops you'll reuse.
2. `serenitymojo/docs/SERENITYMOJO_MODULES.md` — exact signatures of the foundation ops/structs you'll call.
3. `{{REFERENCE FILE, e.g. diffusers/.../transformer_z_image.py}}` — read the relevant forward LINE BY LINE; state the exact math/sequence you must reproduce (shapes, norm eps, concat order, scale/gate form, rope layout). Do NOT infer from memory.
4. `/home/alex/.claude/skills/mojo-syntax/SKILL.md` — the syntax traps for this chunk.

## Hard rules
- **Reuse foundation ops** (`serenitymojo/ops/*`, `tensor.mojo`, `io/*`). Do NOT reimplement matmul/rms_norm/softmax/rope/sdpa/conv/etc. — they exist and are parity-verified. Call them.
- **Match weight names** from the reference checkpoint exactly. No silent renames. Load via `ShardedSafeTensors` (see how existing models do it).
- **No Rust, no cargo, no flame-core, no autograd/training, no Python in the code you ship.** Reference Python is read-only.
- **All file I/O through `io/ffi`** (`sys_open`/`sys_pread`/`sys_pwrite`), never builtin `open`/`Path.read_text`/`external_call["write"]`.

## Mojo syntax/semantics (current era)
- `comptime` not `alias`; every `def` that can fail needs `raises`; `ref` is reserved (don't name a var `ref`); bracket literals for `List`/`Dict`.
- `Tensor` is Movable-not-Copyable → `List[ArcPointer[Tensor]]`, transfer with `^`. Never `List[Tensor]`.
- Sequence lengths that feed `sdpa[B,S,H,Dh]` are **comptime** — parameterize the struct on them.
- The `nn` fused ops with TileTensor+closure args are UNCALLABLE from LayoutTensor (`gamma.origin.mut` wall) → hand-roll (reduction/elementwise are easy). `linalg.matmul`, `conv2d`, `flash_attention` ARE callable from LayoutTensor.
- `flash_attention` won't compile at Dh=128 on sm_86 → use the existing math-mode `ops/attention.sdpa` path.
- F32-accumulate / BF16-store in kernels.

## Deliverable
- The Mojo module(s) in the dir MAP.md indicates, named per the plan.
- A probe `def main() raises:` (in a `parity/` subdir or a `*_probe.mojo`) that imports + calls the new code on a tiny input and runs.
- Compile/run it: `cd /home/alex/mojodiffusion && pixi run mojo run -I . <probe>`. It must exit 0. **Read the actual error output** — a failed compile can print partial text that looks like success.
- Report: files written, what compiled, the foundation ops reused, any reference detail you were unsure about, and anything you could NOT reuse and had to hand-roll (and why).

Do NOT add workarounds to mask a foundation-op bug — flag it as a finding instead. Do NOT expand scope beyond {{CHUNK}}.
