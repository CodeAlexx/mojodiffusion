---
name: autograd-v2
description: How to use serenitymojo's autograd_v2 graph engine — the dependency-counted backward engine (flame v2 port) that drives zimage and Klein LoRA training via recorded per-block graphs, StepSlab allocation, and CUDA-graph capture/replay. Use when training with the v2 engine, debugging a *_V2_GRAPH path, adding a new op kind or model to the engine (P7 rollout), or writing/verifying engine bit-gates. Read BEFORE touching serenitymojo/autograd_v2/ or the ZIMAGE_V2_*/KLEIN_V2_* trainer paths.
---

# Autograd V2 — using the graph engine (serenitymojo)

## Read these first (order matters)

1. **`serenitymojo/docs/AUTOGRAD_V2_MOJO_DESIGN.md`** — THE CONTRACT
   (C1–C15 are hard constraints; implementation is against that doc, not
   this skill). Read it IN FULL before writing engine code.
2. **`serenitymojo/docs/MOJO_V2_ENGINE_PLAN.md`** — the phase ledger
   (what shipped, with gate evidence; what's next). AUTHORITATIVE status.
3. Companion skills to load for any session touching this code:
   - `mojo-syntax` — always, for any Mojo code.
   - `mojo-gpu-fundamentals` — engine work is GPU work.
   - `numeric-parity-testing` — every change gates against the
     hand-chain oracle; this skill defines the discipline.
   - `mojo-train-port` / `ot-mojo-training-port` — where training code
     lives and the seam contracts, when rolling the engine to a new model.
   - `todo` — row 6 of `/home/alex/mojodiffusion/TODO.md` tracks this
     campaign.
4. flame reference (port pattern, don't re-derive; cite file:line):
   `/home/alex/EriDiffusion/flame-core/src/autograd_v2/`, `src/ring_alloc/`,
   `src/cuda_graph.rs`, `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`.

## What it is (one program, designed-in)

Graph recording + dependency-counted engine + StepSlab static allocator +
resident-set conductor + sync-site elimination + frozen-weight skip +
checkpoint/recompute. Port of flame-core autograd_v2 (which ported
PyTorch). flame's own measurement: the graph engine alone is ~+2% time;
the SPEED comes from the companion pieces (slab, capture, sync-elim,
residency) — never ship the engine without them.

Status (2026-06-11, measured): **complete on zimage B1** (trains via 2
cuGraphLaunch calls, 100-step byte-identical, ~1.63 s/step) and **shipped
on Klein** (per-block graphs, no slab/capture — scope decision; 2.2–2.4
s/step). Next: P7 zimage batch-2 unification, then rest-of-models.

## File map

```
serenitymojo/autograd_v2/
  node.mojo            # Node/Edge structs + OPK_* kind table (enum dispatch, C2)
  graph.mojo           # Graph = node tenure table (NodeId ints, no Arc cycles, C3)
  input_buffer.mojo    # per-slot fan-in accumulation (slot-ordered, C15)
  accumulator.mojo     # leaf grad sinks -> resident device grads
  engine.mojo          # execute / execute_slab / execute_klein + apply* dispatch
  step_slab.mojo       # StepSlab over scratch_ring (C8)
  capture.mojo         # cuda_capture_begin / _end_instantiate / cuda_graph_launch (C9)
  ops_record.mojo      # record_* wrappers: plain + _slab variants + record_klein_*
  zimage_block_graph.mojo  # per-block graph bwd: zimage_block_lora_graph_backward[_slab]
  klein_block_graph.mojo   # per-block graph bwd: klein_{double,single}_block_graph_backward
  tests/toy_gates.mojo         # P1 engine invariants (diamond, dep-count exactness…)
  tests/dit_op_parity.mojo     # P2 per-op bit gates vs ops/*_backward
  tests/capture_smoke.mojo     # P5 capture feasibility (5-node graph, replay)
  tests/klein_block_parity.mojo # P6 same-process bit gate (the Klein gate)
```

Trainer integration:
- `training/train_zimage_real.mojo` — flags `ZIMAGE_V2_ENGINE`,
  `ZIMAGE_V2_GRAPH`, `ZIMAGE_V2_SLAB` (+ capture path when all three on).
- `training/train_klein_real.mojo` — flags `KLEIN_V2_ENGINE`,
  `KLEIN_V2_GRAPH` (graph path requires both; comptime dispatch around the
  backward call, ~line 1195).
- `models/klein/klein_stack_lora.mojo` — `klein_stack_lora_backward_graph`
  (drop-in for `…_offload_turbo_moddev_rope_scratch`, same arg list, same
  conductor loop). models/zimage/zimage_stack_lora.mojo — the v2/v3/v5
  forward + graph backward wiring.

## How training runs through it (don't re-derive — this is the shape)

Per step, per block (recompute-style checkpoint, matching the hand-chain):
1. The stack loop hands the block its saved INPUT(s) (the only saved
   activations — DBL/SGL_SAVE_TAIL == 0 full-recompute discipline).
2. The block-graph module re-runs the block forward THROUGH `record_*`
   wrappers, building a per-block `Graph` whose leaves are the block
   input(s) + adapter A/B tensors.
3. `engine.execute*` runs dependency-counted BFS backward; every `apply`
   arm calls the EXISTING parity-proven hand-chain backward helpers
   (ops/*_backward or the block's own _stream_* helpers, called WHOLE).
4. Grads land in the SAME flat lists feeding grad-norm/clip + the
   resident fused AdamW (`training/lora_adamw_{plain,ot}_fused.mojo` —
   dev_p IS the next step's weights, no per-step set upload).
5. Conductor seam (C10): the stack loop keeps its turbo_loader
   await_block/prefetch/mark_active_block_done calls AROUND the per-block
   graph call — the engine does not own offload.
6. zimage only: all allocations via StepSlab (C8) → step is sync-free and
   alloc-deterministic → CUDA-graph warmup(step0)/capture(step1)/
   replay(step≥2), per-step host values enter via fixed staging buffers.

## Contract clauses that bite daily (full text in the design doc)

- **C14 — the oracle is the hand-chain, bit-level.** zimage gates =
  5-step anchors every digit + 100-step zero-diff. Klein CANNOT be
  bit-gated across runs (~4e-4 documented nondeterminism) → its gate is
  SAME-PROCESS bit equality (tests/klein_block_parity.mojo) + trainer
  anchors within the variance class.
- **C15 — slot-ordered fan-in.** bf16 add is order-sensitive; fan-in
  slots are assigned at RECORDING time = the hand-chain's fold order.
  2-way fan-ins may reverse operand order (commutative, bit-equal);
  ≥3-way must keep the oracle's left-fold — call the oracle's helper
  whole rather than re-deriving the fold.
- **C8 — slab, not enqueue_create_buffer**, anywhere in the steady-state
  step (hard precondition of capture; allocating ops break REPLAY).
- **C12 — engine never casts.** Grads flow in the hand-chain's dtypes
  (bf16-first); no silent .to(F32).
- **C13 — gate-don't-delete.** Every v2 path lands behind a comptime
  flag; the hand-chain stays compiled AND reachable. Never delete files.
- **C2/C3 — enum kind dispatch, integer node ids.** No ArcPointer[Node]
  anywhere (skeptic greps for it), no stored closures (no fn-values).

## Adding a new op kind

1. Add `OPK_*` to node.mojo's table.
2. Add a `record_*` wrapper in ops_record.mojo (+ a `_slab` variant if the
   model will run the slab/capture path). Frozen inputs get null edges
   (C7); tracked leaves get accumulator edges via `_leaf_edge`.
3. Add the `apply` arm in engine.mojo — it MUST call the existing
   parity-proven backward (ops/*_backward), never new math.
4. Gate per-op in tests/dit_op_parity.mojo: record op → engine grad ==
   hand-chain grad BIT-EQUAL on real shapes. Then re-run the model-level
   gates.

## Rolling out to a new model (the P7+ recipe; Klein P6 is the worked example)

1. Identity the model's hand-chain stack backward (the oracle) and its
   per-block forward/backward helper pair.
2. Write `<model>_block_graph.mojo` modeled on klein_block_graph.mojo
   (coarse block-section kinds like OPK_KLEIN_* that call the oracle's
   helpers whole) or zimage_block_graph.mojo (fine-grained DiT op kinds)
   — coarse kinds are the cheaper path when the oracle has big fused
   helpers with internal ≥3-way folds.
3. Write `<model>_stack_lora_backward_graph` keeping the existing
   conductor loop/scratch-ring/offload seam; fail-loud on any surface the
   graph path doesn't carry (Klein precedent: compute_aux_grads, saved
   tails).
4. Trainer flag `<MODEL>_V2_GRAPH` (comptime), dispatching at the
   backward call site only.
5. Gates, in order: same-process per-block bit gate (synthetic-but-real-
   shaped inputs, NONZERO LoRA B so d_A is non-degenerate, degenerate
   compared tensors must FAIL) → trainer N-step anchor gate → speed.
   Deterministic models additionally: cross-run byte-identity.

## Build / run / gate commands (verified 2026-06-11)

```bash
cd /home/alex/mojodiffusion
export L=LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib

# build anything (plain `mojo run` fails JIT symbol resolution):
rm -f serenitymojo.mojopkg && pixi run mojo build -I . -Xlinker -lm \
  -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
  <file.mojo> -o /tmp/<bin>

# Klein P6 same-process bit gate (expect: every line n_mismatch=0):
#   build serenitymojo/autograd_v2/tests/klein_block_parity.mojo, then
env $L /tmp/klein_block_parity

# Klein trainer 3-step anchor gate (variance class ~4e-4 around
# 0.5414262 / 0.2154109 / 0.78077525; ~2.2-2.4 s/step):
env $L /tmp/<klein_bin> serenitymojo/configs/klein9b.json 3 0 - nosample_profile

# zimage gates (anchors every digit, b1match/b2dup byte-identical,
# 100-step zero-diff; see HANDOFF_2026-06-11_OVERNIGHT_OT_PARITY.md §-1
# for the full verify-everything block + OneTrainer oracle commands)
```

## Mojo hazards specific to this code (design doc §4)

- ASAP destruction vs async work: tenure intermediates in Graph/slab
  until step end — never let a Tensor drop right after an enqueue.
- Sub-buffer views never outlive their slab owner (scratch_ring rule).
- Node holds TArc lists, never ArcPointer[Node]; edges are Int indices.
- comptime-bucketed trainers, shape-agnostic engine: only the recording
  wrappers are comptime-specialized; node payloads store runtime dims.
- Capture: no alloc, no sync inside the captured region; per-step host
  values via fixed pre-written device buffers.

## Discipline (binding)

- Measurement beats assertion (Tenet 4): no "bit-exact/parity/faster"
  claims without a tool run in-session. Run the gates yourself; sub-agent
  self-reports are never the gate.
- Builder → Bug-Fixer → Skeptic per phase; no phase advances with a
  BLOCKER. Update MOJO_V2_ENGINE_PLAN.md + BEAT_FLAME_SCOREBOARD +
  TODO.md row 6 on every measured change.
