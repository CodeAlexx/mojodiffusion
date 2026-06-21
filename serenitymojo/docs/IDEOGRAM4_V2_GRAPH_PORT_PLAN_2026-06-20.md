# Ideogram4 → autograd_v2 (graph + capture) port plan (2026-06-20)
>> **STATUS 2026-06-21 — SUPERSEDED on the speed thesis; see IDEOGRAM4_SESSION_HANDOFF_2026-06-21.md.**
>> The premise below ("host-op-construction bound, capture is the only lever") was an
>> UNVERIFIED estimate and was REFUTED by measurement: capture works but yields only ~5%
>> (the block is GPU-BOUND). The real speed lever was KERNEL QUALITY — naive SDPA fwd/bwd
>> kernels → cuBLAS matmul gave 2.1× (5.43→2.56 s/step, commits 6781403/864a884). The
>> autograd_v2 port (Stages 0-2) below IS built + bit-gated and correct (block migrate /
>> engine adapter / slab block), just not the speedup. SEPARATELY there is an OPEN training
>> LOSS bug (≈1.1 vs torch 0.96) localized to the training forward — see the handoff + memory
>> project-ideogram4-loss-too-high. Read the handoff first.


Grounded in: AUTOGRAD_V2_MOJO_DESIGN.md (C1–C15), MOJO_V2_ENGINE_PLAN.md (phase
ledger), and the read-in-full Klein precedent autograd_v2/klein_block_graph.mojo.
This is the **only remaining lever** for the ideogram4 trainer's measured
host-op-construction bottleneck (GPU ~44% busy, ~5,500 launches/step) — see
memory project-ideogram4-trainer-hostbound. Sync/copy removal (committed dbef086)
is its prerequisite (capture cannot contain host syncs). Flash SDPA is ruled out
(Dh=256 > cuDNN flash-bwd's 128, committed 8bf462d).

## Where the speedup is (do NOT skip to engine-only)
Ledger, MEASURED: the graph **engine alone is +2% time**. zimage's win came from
**slab (P4) + CUDA-graph capture (P5)** → 2 cuGraphLaunch/step, host overhead gone,
1.8→1.63 s. Klein's P6 is engine-only (no slab/capture, scope decision) and its
speed came from the resident optimizer, not the graph. **Therefore ideogram4 needs
the FULL zimage-style path (engine → slab → capture), not the Klein-style stop.**

## Oracle (the hand-chain — C14 bit-level gate target)
serenity-trainer Ideogram4LoRABlock.mojo:
- forward: `ideogram4_block_lora_forward[S,Hidden,Heads,Dh,FF,Adaln]` (saves acts)
- backward: `ideogram4_block_lora_backward[...]` (the oracle; returns LoRA d_a/d_b + d_x)
- stack: `ideogram4_stack_lora_forward/backward` (checkpoint block inputs, recompute).
Single-stream block (mirror Klein SINGLE, not the double/joint). 34 layers, 6 LoRA
slots/block (qkv,o,w1,w2,w3,adaln). Heads=18, Dh=256, S=1280 (NT256+NIMG1024).

## Block op sequence → coarse OPK_IDEOGRAM4_* kinds (one apply arm = one section)
From the forward (Ideogram4LoRABlock):
1. adaLN: `_lora_linear(adaln_input)` → mod (scale_msa/gate_msa/scale_mlp/gate_mlp via tanh)
2. attn-in: rms_norm(x)·scale_msa → `_lora_linear_fwd(qkv)` → reshape/slice q/k/v →
   rms_norm_q/k → rope q/k → **sdpa_nomask** → `_lora_linear_fwd(o)`
3. attn-residual: rms_norm2(attn_out)·gate_msa + x → x_mid
4. ffn: rms_norm(x_mid)·scale_mlp → `_lora_linear_fwd(w1)`, `_lora_linear_fwd(w3)` →
   swiglu → `_lora_linear_fwd(w2)`
5. ffn-residual: rms_norm2(ff_out)·gate_mlp + x_mid → out
Coarse kinds (cheaper than fine-grained per the skill, since the oracle has big
helpers with internal ≥3-way folds): e.g. OPK_IDEOGRAM4_ADALN, _ATTN_IN, _SDPA,
_ATTN_RES, _FFN_IN, _SWIGLU(reuse OPK existing), _FFN_OUT. Each apply arm calls the
SAME hand-chain backward sub-sequence the oracle uses (engine never new math, C2/C14).
C15: graph fan-ins are the two residual adds (2-way, commutative → bit-equal).

## Files to write (mirror klein, P7 recipe)
1. `autograd_v2/node.mojo` — add OPK_IDEOGRAM4_* to the kind table.
2. `autograd_v2/ops_record.mojo` — `record_ideogram4_*` wrappers (+ `_slab` variants
   for the capture path); frozen base weights = null edges (C7), LoRA A/B = leaves.
3. `autograd_v2/engine.mojo` — `apply_ideogram4_*` arms (call Ideogram4LoRABlock's
   own backward helpers WHOLE) + `execute_ideogram4` (or generalize execute_klein).
4. `autograd_v2/ideogram4_block_graph.mojo` — `ideogram4_block_graph_backward`
   (mirror klein_single_block_graph_backward: tracked-leaf input + 6 adapter leaves,
   record fwd, seed bwd from out, return the oracle's grad struct).
5. serenity-trainer: `ideogram4_stack_lora_backward_graph` (keep the conductor/
   scratch-ring seam) + trainer flag `IDEOGRAM4_V2_GRAPH` (comptime dispatch at the
   backward call site only, C13 gate-don't-delete).

## Stages + gates (each: Builder → Bug-Fixer → Skeptic; no advance on a BLOCKER)
- **Stage 1 (engine, NO speedup expected, +2%):** record/apply/block-graph + flag.
  GATE: autograd_v2/tests/ideogram4_block_parity.mojo — same-process BIT gate
  (NONZERO LoRA B so d_A non-degenerate; degenerate compares must FAIL); then trainer
  N-step anchor stays byte-identical to the hand-chain (1.12493/1.141433/0.73936844/...).
- **Stage 2 (slab, P4):** route EVERY fwd+bwd allocation through StepSlab; assert ZERO
  enqueue_create_buffer + ZERO cuStreamSynchronize in the step after warmup. Re-gate bit.
- **Stage 3 (capture, P5 — THE WIN):** warmup(0)/capture(1)/replay(≥2) via the existing
  capture.mojo; per-step host values via fixed staging buffers. GATE: bit + MEASURE
  s/step (target: close the ~3 s host gap → toward the ~2.1 s GPU floor; beats Rust 3.9).

## Build/gate commands: see the autograd-v2 skill (`rm -f serenitymojo.mojopkg && pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 <file> -o /tmp/<bin>`).

## Discipline (binding): Tenet 4 — run the gates in-session; sub-agent self-reports are never the gate. Update MOJO_V2_ENGINE_PLAN.md + TODO row 6 on every measured change.
