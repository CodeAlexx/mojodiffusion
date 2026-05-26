# Bugfix agent prompt — serenitymojo (pure Mojo + MAX)

Fresh agent. Fill `{{...}}`, send to `general-purpose`. Targeted minimal fixes to the skeptic's BLOCKERs — do not rewrite, do not expand scope.

---

Fix the BLOCKER findings in `serenitymojo/{{area}}/parity/SKEPTIC_FINDINGS_{{date}}.md` for **{{CHUNK}}** in serenitymojo (`/home/alex/mojodiffusion`). Pure Mojo + MAX, inference-only.

## Rules
- **One finding at a time.** Reproduce it (compile/run the probe and see the failure), make the **minimal** fix, re-run, confirm. Then next finding.
- Stay in scope: only the files the finding names. No opportunistic rewrites.
- If the root cause is a **foundation op** (`ops/*`, `tensor.mojo`, `io/*`), STOP and report it as a foundation issue — do NOT add a workaround in the model code to mask it.
- Honor the Mojo invariants: `comptime` not `alias`; `def … raises`; `ref` reserved; `List[ArcPointer[Tensor]]` + `^`; file I/O only via `io/ffi` (`sys_open`/`sys_pread`/`sys_pwrite`, never builtin `open` / `external_call["write"]`); call foundation ops, don't reimplement.
- No Rust/cargo/flame-core/autograd, no Python in the shipped path.

## Reproduce / verify
`cd /home/alex/mojodiffusion && pixi run mojo run -I . <probe.mojo>` — must exit 0 after the fix. **Read the actual error**, not just that something printed (a failed compile can emit partial output that looks like a pass).

## Report
Per finding: what the bug was, the minimal change (file:line), and the before/after compile+run result. If a finding turned out to be a non-bug, say why (skeptic was wrong). List any findings you escalated to a foundation op. Re-run the chunk's probe at the end and confirm exit 0.

After bugfix, control returns to the orchestrator for another skeptic round, then the parity gate (cos≥0.999 vs the GPU-bf16 oracle).
