# autograd_v2/node.mojo - Node, Edge, OPK_* kind table + the package's raw GPU
# op shims. Port of flame-core src/autograd_v2/node.rs (Edge :52-76, GradFn
# surface :94-203) per design contract C2/C4 (AUTOGRAD_V2_MOJO_DESIGN.md).
#
# Mojo port decisions (contract C2/C3):
#  * No `Arc<dyn GradFn>`: a node is ONE concrete struct dispatched on `kind`
#    (the OP_* comptime-enum precedent from serenitymojo/autograd.mojo).
#  * No ArcPointer[Node] anywhere: edges carry integer node indices into
#    Graph.nodes (C3 NodeId-table replaces flame's Weak cycle-break).
#  * Tensors are boxed as TArc = ArcPointer[Tensor] so Node can live in Mojo
#    collections (List needs Copyable; TArc copy = refcount bump).
#
# Mojo 1.0.0b1, NVIDIA. P1 toy-gate dtype is F32 (from_host F32 tensors).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import (
    add as _ta_add,
    mul as _ta_mul,
    add_scalar as _ta_add_scalar,
    zeros_device as _ta_zeros_device,
    add_slab as _ta_add_slab,
)
from serenitymojo.autograd_v2.step_slab import StepSlab


comptime TArc = ArcPointer[Tensor]

# Node kind table (C2). LEAF = the accumulator node (flame AccumulateGrad,
# accumulator.rs:27-105) - a terminal grad sink for one parameter.
comptime OPK_LEAF = 0
comptime OPK_ADD = 1
comptime OPK_MUL = 2
comptime OPK_MATMUL = 3
comptime OPK_SUM = 4
# ── P2: the zimage DiT block vocabulary (AUTOGRAD_V2_MOJO_DESIGN.md P2).
# Forward op + backward arm per kind; backward arms call ONLY the existing
# parity-proven ops/*_backward functions (engine.mojo apply()).
comptime OPK_PROJ_LORA = 5          # y = linear(x,W) + scale*(x@Aᵀ)@Bᵀ
comptime OPK_RMS_NORM_DX = 6        # frozen-weight rms_norm (dx only)
comptime OPK_MODULATE = 7           # (1+scale)*x + shift (scale frozen or leaf)
comptime OPK_ROPE = 8               # rope_interleaved (frozen cos/sin tables)
comptime OPK_SDPA = 9               # sdpa_nomask[B,S,H,Dh] -> d_q,d_k,d_v
comptime OPK_SWIGLU = 10            # silu(gate)*up -> d_gate,d_up
comptime OPK_RESIDUAL_GATE_DXDY = 11  # o = x + gate_t*y (gate frozen)
comptime OPK_RESHAPE = 12           # metadata-only; grad reshaped back
# ── P6: the Klein-9B per-block vocabulary (AUTOGRAD_V2_MOJO_DESIGN.md P6).
# Klein records at COMPOSITE granularity: each kind's backward arm calls the
# oracle's OWN hand-chain helper / inline op sequence verbatim (models/klein/
# double_block.mojo `_stream_{pre,post}_backward_lora_resident_scratch_tensors`,
# single_block.mojo `single_block_lora_backward_device_resident_scratch_tensors`
# split at its activation seams), so every >=3-way bf16/F32 fan-in fold stays
# INSIDE the oracle code (C15 trivially satisfied at graph level - all
# graph-level fan-ins are 2-way, where IEEE addition is commutative). Arms are
# dispatched by engine.apply_klein (comptime [H,Dh,S] for the sdpa buckets).
comptime OPK_KLEIN_DBL_PRE = 13     # per-stream pre: x -> (q_rms, k_rms, v) + q/k/v LoRA leaves
comptime OPK_KLEIN_DBL_JOINT = 14   # (tq,iq,tk,ik,tv,iv) -> (txt_att, img_att): concat+rope+sdpa+slice
comptime OPK_KLEIN_DBL_POST = 15    # (x, att) -> stream out + out/ff_in/ff_out LoRA leaves
comptime OPK_KLEIN_SGL_IN = 16      # x -> (q_rms,k_rms,v,mlp_gate,mlp_up) + qkv LoRA leaves
comptime OPK_KLEIN_SGL_SDPA = 17    # (q_rms,k_rms,v) -> att_flat (rope+sdpa)
comptime OPK_KLEIN_SGL_OUT = 18     # (x, att_flat, mlp) -> block out + out LoRA leaves (lazy fwd)


struct Edge(Copyable, Movable):
    """(node_idx, input_nr, contrib_slot) - flame node.rs:52-76
    Edge{function, input_nr} + the C15 contribution slot.

    `node_idx == -1` is the null edge (flame `Edge{function: None}`): the
    engine DROPS the gradient instead of forwarding it (frozen/untracked
    input, contract C7). `input_nr` is the slot of the CHILD node's
    InputBuffer this gradient feeds into (= which forward output of the child
    our input tensor was) - multi-input correctness, contract C4.

    `contrib_slot` (C15, deliberate deviation from flame): which CONTRIBUTION
    slot inside the child's (input_nr) fan-in this edge's gradient occupies.
    Assigned at RECORDING time by Graph.record in consumer-registration order
    (= the hand-chain's fold order = forward order); the engine reduces present
    contributions in ascending contrib_slot order (left fold) so bf16 fan-in
    addition is bit-equal to the hand-chain's fixed `add(add(.,.),.)` folds."""

    var node_idx: Int
    var input_nr: Int
    var contrib_slot: Int

    def __init__(out self, node_idx: Int, input_nr: Int, contrib_slot: Int = 0):
        self.node_idx = node_idx
        self.input_nr = input_nr
        self.contrib_slot = contrib_slot

    @staticmethod
    def null() -> Edge:
        # flame node.rs:66-71 Edge::null()
        return Edge(-1, 0)

    def is_valid(self) -> Bool:
        # flame node.rs:73-75
        return self.node_idx >= 0


struct Node(Copyable, Movable):
    """One backward-graph node (contract C2; flame GradFn object node.rs:94-203
    flattened to a concrete enum-dispatched struct).

    * kind: OPK_* arm selector (apply() switches on it - engine.mojo).
    * edges: next_edges, one per FORWARD INPUT in order (node.rs:131-135);
      edges[i] routes the grad for forward-input i.
    * saved: SavedTensor payloads (C6; P1 has no version counters yet - the
      version-cell audit is Phase P0 scope per the design doc).
    * saved_meta: shapes/dims/flags packed per kind (MATMUL: [M, N, K]).
    * scalars: eps/scale/etc per kind (unused by the P1 arms; surface per C2).
    * num_inputs: InputBuffer arity = number of FORWARD OUTPUTS (grads coming
      INTO this backward node; flame node.rs:126-129 num_inputs()).
    * node_id: monotonic engine-scoped id (== index into Graph.nodes).
    * sequence_nr: recording order (node.rs:137-139, stored not recomputed).
    * topological_nr: 1 + max(input-edge target topo); leaves are 0
      (node.rs:141-144, accumulator.rs:69-72).
    * param_id: for OPK_LEAF, which parameter tensor id this node accumulates
      (the Mojo stand-in for flame's Weak<meta> handle, contract C3); -1 for
      every non-leaf kind.
    * contrib_counts (C15): per forward-output slot (len == num_inputs), how
      many consumer edges were registered against that slot so far. Graph.record
      reads-then-increments this on the TARGET node to hand each new consumer
      edge its contrib_slot (registration order = hand-chain fold order)."""

    var kind: Int
    var edges: List[Edge]
    var saved: List[TArc]
    var saved_meta: List[Int]
    var scalars: List[Float32]
    var num_inputs: Int
    var node_id: Int
    var sequence_nr: Int
    var topological_nr: Int
    var param_id: Int
    var contrib_counts: List[Int]

    def __init__(
        out self,
        kind: Int,
        var edges: List[Edge],
        var saved: List[TArc],
        var saved_meta: List[Int],
        var scalars: List[Float32],
        num_inputs: Int,
        node_id: Int,
        sequence_nr: Int,
        topological_nr: Int,
        param_id: Int = -1,
    ):
        self.kind = kind
        self.edges = edges^
        self.saved = saved^
        self.saved_meta = saved_meta^
        self.scalars = scalars^
        self.num_inputs = num_inputs
        self.node_id = node_id
        self.sequence_nr = sequence_nr
        self.topological_nr = topological_nr
        self.param_id = param_id
        self.contrib_counts = List[Int]()
        for _ in range(num_inputs):
            self.contrib_counts.append(0)


# ─────────────────────────────────────────────────────────────────────────────
# Raw GPU op shims for the package (pattern: serenitymojo/autograd.mojo
# _raw_add/_raw_mul/ones_like; that module's helpers are private to it, so
# autograd_v2 carries its own). Contract C12: the engine never casts - dtype
# mismatch raises instead of silently converting.
# ─────────────────────────────────────────────────────────────────────────────


def _raw_add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    if a.dtype() != b.dtype():
        raise Error("autograd_v2 _raw_add: dtype mismatch (C12: engine never casts)")
    return _ta_add(a, b, ctx)


def _raw_add_slab(
    a: Tensor, b: Tensor, ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `_raw_add` (this file :157) — same add kernel via
    ops.tensor_algebra.add_slab (contract C8, Phase P4)."""
    if a.dtype() != b.dtype():
        raise Error("autograd_v2 _raw_add: dtype mismatch (C12: engine never casts)")
    return _ta_add_slab(a, b, ctx, slab)


def _raw_mul(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    if a.dtype() != b.dtype():
        raise Error("autograd_v2 _raw_mul: dtype mismatch (C12: engine never casts)")
    return _ta_mul(a, b, ctx)


def ones_like(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    # same construction as autograd.mojo ones_like: zeros + 1.0
    var z = _ta_zeros_device(t.shape(), t.dtype(), ctx)
    return _ta_add_scalar(z^, Float32(1.0), ctx)


def arc_view(t: Tensor) raises -> TArc:
    """Zero-copy re-box: a fresh Tensor struct SHARING t's device buffer
    (DeviceBuffer copy = refcount bump on the allocation, no d2d transfer).

    Used by the P2 apply arms to lift gradient tensors out of backward result
    structs (SwigluGrads / GateResidualGrads / ModulateBackward / SdpaGrads
    hold plain Tensor fields, and Mojo forbids partially moving a field out of
    a still-live destructible value) without the clone-and-sync the P1 MATMUL
    arm pays. Also the OPK_RESHAPE arm's zero-kernel reshape carrier."""
    return TArc(Tensor(t.buf.copy(), t.shape(), t.dtype()))


def arc_view_reshaped(t: Tensor, var new_shape: List[Int]) raises -> TArc:
    """arc_view with a different (numel-preserving) shape - the OPK_RESHAPE
    forward/backward carrier (metadata-only, zero kernels). Out-of-place by
    construction: the source Tensor's own shape is NEVER mutated (the routed
    grad arc may be shared with other consumers, e.g. OPK_ADD fans the same
    arc to both inputs)."""
    var n = 1
    for i in range(len(new_shape)):
        n *= new_shape[i]
    if n != t.numel():
        raise Error(
            String("arc_view_reshaped: numel mismatch ")
            + String(n)
            + " != "
            + String(t.numel())
        )
    return TArc(Tensor(t.buf.copy(), new_shape^, t.dtype()))
