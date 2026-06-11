# Autograd V2 — Mojo Design Contract & Implementation Plan

Date: 2026-06-11. Status: CONTRACT — phases implement against this doc;
clauses are hard constraints, not suggestions (the flame discipline:
flame-core/docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md).

Mandate (maintainer, 2026-06-11): V2 in Mojo is JOB NUMBER ONE — the
COMPLETE engine flame shipped, as ONE program: graph recording +
dependency-counted engine + static step-slab allocator + resident-set
conductor + sync-site elimination + frozen-weight skip +
checkpoint/recompute prefetch — **designed in from day one**, not bolted
on ("if we come a day when we need it, i don't want to add it, i want it
built in" — 2026-05-13 directive, carried forward).

Reference implementation (port pattern, don't re-derive; cite file:line):
`/home/alex/EriDiffusion/flame-core/src/autograd_v2/` + `src/ring_alloc/`
+ `src/offload/` + `src/cuda_graph.rs`. PyTorch lineage is already encoded
in flame's code; we port flame, flame ported PyTorch.

---

## 0. What exists today (measured / read this session)

| piece | where | state |
|---|---|---|
| T1 tape engine | serenitymojo/autograd.mojo (460 ln) | explicit-threaded Tape, enum ops (OP_ADD…OP_MSE), ArcPointer boxing, reverse-order replay. NOT dependency-counted, NOT graph-shaped. The v2 engine SUPERSEDES it (gate, don't delete). |
| hand-chain trainers | models/zimage/*_stack_lora.mojo, models/klein/klein_stack_lora.mojo | parity-proven fwd/bwd op chains; **bit-stable on zimage** (100-step zero-diff gate 06-11). These are the ORACLE for every v2 phase. |
| two-cursor ring | serenitymojo/scratch_ring.mojo | OT StaticLayerAllocator shape (fwd cursor + reverse cursor, sub-buffer views) — the step-slab primitive ALREADY EXISTS; v2 makes it the engine's only steady-state allocator. |
| conductor seed | serenitymojo/offload/turbo_loader.mojo | two-slot block residency, copy stream + events, persistent pinned store; CUDA driver FFI precedent (external_call, :85-110). |
| resident-set states | training/lora_adamw_{plain,ot}_fused.mojo | persistent device P/M/V (06-11). dev_p is the live param buffer the engine's AccumulateGrad/optimizer integration targets. |
| step anatomy | nsys 06-11 | post-A/C zimage B1: 1.8 s/step, ~kernel-bound; remaining engine residue = NR/CR host fwd, final-layer uploads, grads D2H, clone syncs. |

## 1. Design contract — clauses (C1..C14)

**C1 — One engine, explicit threading.** `Engine` and `Graph` are explicit
structs passed `mut` (repo decision 2026-05-30: no global mutable state in
serenitymojo; Mojo 1.0.0b1 globals are shaky). No thread-locals. Flame's
single-threaded stateless `Engine::execute` (engine.rs:184-600) maps 1:1.

**C2 — Enum dispatch, not dyn traits.** Mojo has no `Arc<dyn GradFn>`.
A graph node is ONE concrete struct:
```
struct Node:           # flame GradFn object ≈ node.rs:94-203
    var kind: Int               # OPK_* comptime table
    var edges: List[Edge]       # next_edges, one per input
    var saved: List[TArc]       # SavedTensor payloads (C6)
    var saved_meta: List[Int]   # shapes/dims/flags packed per kind
    var scalars: List[Float32]  # eps/scale/etc per kind
    var num_inputs: Int         # InputBuffer arity (node.rs num_inputs)
    var node_id: Int            # monotonic (engine-scoped counter)
    var sequence_nr: Int        # recording order
    var topological_nr: Int     # max(child topo)+1 at construction
```
`apply(node, grad_inputs, ctx) -> List[Optional grads]` is one dispatch
function switching on `kind`, each arm calling the EXISTING parity-proven
`ops/*_backward` functions (the same reuse flame v2 did with its kernels).
Precedent: autograd.mojo's OP_* enum (proven pattern in-repo).

**C3 — NodeId table replaces Weak.** flame breaks the
tensor→meta→AccumulateGrad→tensor cycle with `Weak` (meta.rs:54-59,
accumulator.rs:27-105). Mojo port: ALL nodes live in `Graph.nodes:
List[Node]` (the tenure table); tensors carry only integer ids
(`Tensor.id` already exists for this); leaf accumulators are looked up by
param-id in a `Dict[Int, Int]` (param id → accumulator node idx). No
ArcPointer cycles are constructible: Node holds TArc (data) but never
ArcPointer[Node]; edges are integer node indices.

**C4 — Edge = (node_idx: Int, input_nr: Int).** node_idx = -1 means
dropped grad (flame `Edge{function: None}`, node.rs:52-76). `input_nr`
slots into the consumer's InputBuffer (multi-input correctness —
design-review blocking issue #6).

**C5 — Engine algorithm = flame engine.rs verbatim shape.**
dependency-count BFS from roots (engine.rs:230-277); seed InputBuffers
with `ones_like`/caller grads incl. the outputs-as-descendants fix
(engine.rs:335-393); ready queue ordered (topological_nr DESC,
sequence_nr DESC, node_id DESC) (ReadyKey, engine.rs:130-165); per pop:
materialize input grads → apply → arity-check → route via edges →
decrement dep counts → enqueue at zero (engine.rs:437-570); errors
propagate via `raises` (flame's Result clause #3). Single-threaded;
nested execute is legal (fresh local state per call) — that IS the
reentrant/checkpoint surface (checkpoint.rs:121-216).

**C6 — SavedTensor + version counters.** Saved activations are TArc with
a saved version int + a shared version cell. Mojo port of saved_ref:
`Tensor` gains a `version: ArcPointer[Int]`-equivalent cell only where
in-place mutation exists (optimizer dev_p writes, scratch reuse). Phase 0
audits every in-place write site and adds bumps (design-review clause 9).
Steady-state trainers recompute (checkpoint-style), so saved-tensor
pressure is per-block, matching today's ZImageBlockSaved.

**C7 — Frozen-weight skip in the recording surface.** `needs_grad` gate
(recording.rs:87-95): if no input is tracked, record NOTHING (inference
fast path); edges to frozen tensors are null edges → engine never
computes nor routes their grads. The dx-only backward arms (rms_norm
_backward_dx, gate_residual_backward_dxdy — already shipped 06-11) are
the kind-arms for frozen-param ops.

**C8 — Slab-allocated, capture-compatible from day one.** Every engine
allocation in the steady-state step (recompute outputs, grad buffers,
InputBuffer storage) goes through a `StepSlab` handle (scratch_ring
extension), NOT `enqueue_create_buffer`. Slabs are lazy, never freed
mid-step, reset at step boundary (RING_ALLOC_DESIGN.md invariants 1-5).
This is a hard precondition of C9 and is NOT deferrable (mandate:
designed in).

**C9 — CUDA-graph capture surface.** Engine exposes
`warmup/capture/replay` phases (flame cuda_graph.rs:220-240
BackwardGraphCache lifecycle): step N=0 warmup (slabs materialize), N=1
capture via driver FFI `cuStreamBeginCapture/cuStreamEndCapture/
cuGraphInstantiate` (external_call pattern from turbo_loader.mojo:85-110),
N≥2 `cuGraphLaunch`. Invalidation: graph keyed on (bucket shape, node
count). Host-side per-step values (sigma, lr, noise) enter via fixed
device buffers written BEFORE launch (graph reads stable pointers).
Surface lands with the engine; full enablement gates per-trainer.

**C10 — Conductor + prefetch hooks.** Node kinds carry an optional
`block_idx`; the engine fires `pre_block(block_idx)` /
`post_block(block_idx)` callbacks at block-boundary nodes — the hook
surface (hooks.rs:31-81) reduced to what the offload conductor actually
consumes (turbo_loader prefetch/await/mark — same domain flame's
BlockOffloader hooks serve). Klein's offloaded tape plugs in here.

**C11 — Streams/device explicit.** Every engine entry point takes `ctx:
DeviceContext` (+ optional stream handle when the copy-stream overlap
lands). No hardcoded default-stream assumption in the engine structs
(flame clause 16 multi-device surface).

**C12 — dtype discipline.** Engine never casts. Grads flow in the dtype
the backward arm produces (bf16-first per repo contract); F32 transients
only inside kernels. No silent `.to(F32)` (flame clause: BF16 grads
end-to-end; here grads are ALREADY the proven hand-chain dtypes).

**C13 — Gate-don't-delete.** v2 lands behind per-trainer flags
(`*_V2_GRAPH`), hand-chain stays compiled + reachable (zimage/klein flags
already exist for A/C; graph adds a second stage). No file deletions ever
(flame Stage-6a/6b rule).

**C14 — Oracle = the hand-chain, bit-level.** zimage B1 is bit-stable
(measured 06-11: 100-step zero-diff). Every phase that touches the
zimage path must reproduce the 5-step anchors to every printed digit and
the 100-step file byte-for-byte. Klein gates at its documented
variance class (~4e-4) until its nondeterminism is root-caused.

**C15 — Slot-ordered fan-in (deliberate deviation from flame, added after
P1).** flame's InputBuffer accumulates contributions in ARRIVAL order
(ready-queue order). bf16 addition is order-sensitive, and C14 demands
bit-equality with the hand-chain, whose fan-ins are FIXED left-folds
(e.g. zimage d_xn1s = add(add(dq, dk), dv)). Therefore: every fan-in
contribution gets a SLOT assigned at recording time (consumer registration
order = the hand-chain's fold order, which equals forward order);
InputBuffer stores per-slot and reduces in slot index order (left fold,
same `ops.tensor_algebra.add`) at materialization. Memory cost
(fan-out degree × tensor) is slab-bounded (C8). Justification: the oracle
is bit-level; flame's arrival-order semantics cannot reproduce it.

## 2. Package layout (mirrors flame autograd_v2/)

```
serenitymojo/autograd_v2/
  __init__.mojo
  node.mojo          # Node, Edge, OPK_* kind table          ← node.rs
  graph.mojo         # Graph (node tenure table), recording  ← recording.rs + meta.rs
  input_buffer.mojo  # InputBuffer (Optional slots, in-place
                     #   accumulate when unique)             ← input_buffer.rs
  accumulator.mojo   # leaf grad sinks → resident dev grads  ← accumulator.rs
  engine.mojo        # dep-count BFS + ready queue + execute ← engine.rs
  step_slab.mojo     # StepSlab over scratch_ring            ← ring_alloc/ + static_slab_v2.rs
  capture.mojo       # CUDA-graph FFI + 3-phase lifecycle    ← cuda_graph.rs
  ops_record.mojo    # record_* wrappers for the DiT op set  ← ops/ + dispatch.rs
  tests/             # toy-graph gates (below) + parity gates
```

## 3. Phase order (each: Builder → Bug-Fixer → Skeptic, gates before advance)

**P0 — prereqs (small).** In-place-write audit (optimizer dev_p, scratch
reuse, copy_ helpers) + version-cell bumps + tests that a saved tensor
mutated through each path errors on unpack. Flag plumbing.

**P1 — core types + engine, toy gates.** node/graph/input_buffer/
accumulator/engine over OPK_{ADD,MUL,MATMUL,SUM}. Toy gates (flame Phase 2
list, engine.rs tests): single-leaf sum; two branches into one leaf;
diamond accumulation; undefined grad slots; multi-output routing by
input_nr; nested execute returns cleanly; dep-count exactness (every node
fires exactly once — assert via counter).

**P2 — DiT op set recorded.** OPK arms for the zimage block vocabulary:
linear(+LoRA apply), rms_norm(dx), modulate, rope_interleaved,
sdpa_nomask, swiglu, residual_gate(dxdy), add, reshape(no-op),
layer_norm(dx), mse/flow loss seed. Each arm's backward = the existing
ops/*_backward call, verified per-op: record op → engine grad ==
hand-chain grad BIT-EQUAL on real shapes ([S,D]=[1248,3840] etc.).

**P3 — zimage B1 behind `ZIMAGE_V2_GRAPH`.** Forward records per-block
(recompute-style checkpoint nodes, preserving today's memory shape);
engine drives backward; grads land in the SAME flat lists feeding
clip+resident AdamW. GATES: 5-step anchors every digit + b1match/b2dup +
100-step zero-diff (C14). Measure: step time must be ≤ hand-chain 1.8 s
(launch pattern identical → expect ±noise; the win comes at P5/P6).

**P4 — slab steady state. [SHIPPED 2026-06-11, commit a6724cb — 6180 allocs/step deterministic, bwd 1.19→1.11, all C14 gates green]** All P3-path allocations through StepSlab;
assert ZERO `enqueue_create_buffer` and ZERO `cuStreamSynchronize` inside
fwd+bwd after warmup step (countable via the step's own instrumentation;
nsys cross-check when tooling cooperates). Re-run C14 gates.

**P5 — CUDA-graph capture/replay on zimage B1/B2. [SHIPPED for B1 2026-06-11, commit cad34a7 — G_fwd 5,774 + G_bwd 21,036 nodes, replay from step 3, ~1.63 s/step, 100-step zero-diff; B2 = P7]** FEASIBILITY MEASURED
2026-06-11 (tests/capture_smoke.mojo, independently re-run): capture
through MAX DeviceContext WORKS on this box — 5-node graph via
cuStreamBeginCapture_v2 on CUDA(ctx.stream()) (turbo_loader idiom),
correct replay twice with fresh inputs through fixed pointers, enqueue
cost 9.5 → 2.5 µs/iter (3.6×); MAX's allocator does not invalidate
capture but allocating ops break REPLAY (per-call pointers) → P4
fixed-buffer routing is the hard precondition, confirmed. Warmup→capture→replay;
per-step inputs via fixed staging buffers. GATES: C14 bit-gates AND
measured step time (target: close the remaining ~0.3 s host gap; kernels
~1.45 s are then the floor pending SDPA sign-off).

**P6 — klein integration.** Offloaded-tape seam via C10 conductor hooks
(prefetch_block/await_block/mark_compute_done at block boundaries);
OT-resident optimizer already in. Variance-class gates + step time.

**P7 — batch-2 unification + rest-of-models rollout.** One recorded graph
parameterized by B; klein batch-2 rides the graph. Then anima/ernie/
chroma per the parity campaign.

## 4. Hazards (Mojo-specific; each needs a test or a discipline note)

- **ASAP destruction vs async work**: a Tensor dropped right after an
  enqueue must not free under the queue. Discipline: engine tenures all
  intermediates in Graph/slab until step end (C8 makes this structural).
- **Sub-buffer lifetime**: views never outlive slab owners (scratch_ring
  rule, re-stated in C8; tonight's modvec slab follows it).
- **ArcPointer cycles**: forbidden by C3 construction (no ArcPointer[Node]
  anywhere). Skeptic greps for it per phase.
- **Mojo collections need Copyable**: Node holds TArc lists (Copyable);
  Node itself Copyable+Movable; Graph.nodes is List[Node].
- **No fn-values**: dispatch is the kind-switch (C2), never stored
  closures (httpserver gotcha).
- **comptime shapes**: trainers are comptime-bucketed; node payloads store
  runtime dims (List[Int]) — the engine is shape-agnostic; only the
  recording wrappers are comptime-specialized.
- **Capture constraints**: no alloc/no sync inside captured region (P4 is
  the prerequisite gate); cuBLAS/MAX matmul capture-compat must be PROVEN
  by a P5 smoke before relying on it (HYPOTHESIS until measured).

## 5. Workflow

One phase per session-chunk: Builder implements to this contract; Bug-Fixer
hunts the hazard list; Skeptic attacks the gates (esp. C14 bit-gates and
the "engine fired every node exactly once" invariant). No phase advances
with a BLOCKER. All numbers measured, never asserted (Tenet 4).
