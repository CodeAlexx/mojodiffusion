# autograd_v2/engine.mojo - the dependency-counted backward driver.
# Port of flame-core src/autograd_v2/engine.rs (contract C5: verbatim shape):
#   1. dep-count BFS over next_edges from the root      (engine.rs:230-277)
#   2. per-node InputBuffers + output-grad seeding with
#      the outputs-as-descendants fix                   (engine.rs:282-393)
#   3. ready queue ordered (topological_nr DESC,
#      sequence_nr DESC, node_id DESC)                  (ReadyKey, engine.rs:130-165)
#   4. drive loop: pop -> materialize input grads ->
#      apply -> arity check -> route via edges ->
#      decrement child deps -> enqueue at zero          (engine.rs:437-570)
#
# Single-threaded; NO state survives a call (flame Engine carries no fields,
# engine.rs:171-177) - every counter/buffer below is a local, so nested
# execute is legal by construction (C5 / checkpoint surface).
# Errors propagate via `raises` (flame's Result clause).
#
# P1 invariant gate: every reachable node fires EXACTLY once - fired counter
# checked against the BFS reachable count at the end (the design doc's
# "dep-count exactness" toy gate).

from std.gpu.host import DeviceContext
from std.collections import Dict
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import mul_scalar as _ta_mul_scalar
from serenitymojo.ops.linalg_backward import mm_backward
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import (
    rope_backward,
    gate_residual_backward_dxdy,
)
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.models.zimage.lora_block import ZImageLoraAdapterDevice
from serenitymojo.autograd_v2.node import (
    Edge,
    Node,
    TArc,
    OPK_LEAF,
    OPK_ADD,
    OPK_MUL,
    OPK_MATMUL,
    OPK_SUM,
    OPK_PROJ_LORA,
    OPK_RMS_NORM_DX,
    OPK_MODULATE,
    OPK_ROPE,
    OPK_SDPA,
    OPK_SWIGLU,
    OPK_RESIDUAL_GATE_DXDY,
    OPK_RESHAPE,
    _raw_add,
    _raw_mul,
    ones_like,
    arc_view,
    arc_view_reshaped,
)
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.input_buffer import InputBuffer
from serenitymojo.autograd_v2.ops_record import (
    proj_lora_backward,
    sdpa_backward_dispatch,
)


def apply(node: Node, grads_in: List[TArc], ctx: DeviceContext) raises -> List[TArc]:
    """Kind-dispatched backward (contract C2: one switch, no dyn traits;
    flame GradFn::apply node.rs:120-124). Returns ONE grad per next_edge
    (engine arity contract, engine.rs:509-520); null edges drop their grad at
    routing time, so every arm produces grads for ALL its input slots.

    P1 arms (toy-gate vocabulary; the DiT op set is P2 ops_record scope):
      ADD    -> route g to both inputs.
      MUL    -> dA = g*B, dB = g*A (saved[0]=A, saved[1]=B; the
                autograd.mojo OP_MUL convention).
      MATMUL -> dA = g @ B^T, dB = A^T @ g via the parity-proven
                ops/linalg_backward.mm_backward (transpose-flag GEMMs).
      SUM    -> broadcast the scalar g: ones_like(saved input) * g.
      LEAF is NOT dispatched here - the engine sinks it inline (the
      AccumulateGrad arm, accumulator.rs:137-206)."""
    var g = grads_in[0].copy()  # every P1 kind is single-forward-output
    if node.kind == OPK_ADD:
        # d/da (a+b) = d/db (a+b) = g; share the arc (accumulation downstream
        # is out-of-place, never mutates the routed tensor).
        var out = List[TArc]()
        out.append(g.copy())
        out.append(g.copy())
        return out^
    elif node.kind == OPK_MUL:
        var a = node.saved[0].copy()
        var b = node.saved[1].copy()
        var d_a = _raw_mul(g[], b[], ctx)
        var d_b = _raw_mul(g[], a[], ctx)
        var out = List[TArc]()
        out.append(TArc(d_a^))
        out.append(TArc(d_b^))
        return out^
    elif node.kind == OPK_MATMUL:
        # C[M,N] = A[M,K] @ B[K,N]; saved_meta = [M, N, K].
        var a = node.saved[0].copy()
        var b = node.saved[1].copy()
        var mg = mm_backward(
            g[], a[], b[],
            node.saved_meta[0], node.saved_meta[1], node.saved_meta[2], ctx)
        # CLONE the struct fields rather than move them out (Mojo forbids
        # partially moving a field out of a still-live destructible value -
        # the autograd.mojo OP_MATMUL arm's exact idiom).
        var out = List[TArc]()
        out.append(TArc(mg.d_a.clone(ctx)))
        out.append(TArc(mg.d_b.clone(ctx)))
        return out^
    elif node.kind == OPK_SUM:
        # y = sum(x) (scalar). dx = ones_like(x) * g. P1 simplification: the
        # scalar g is read back to host for mul_scalar (one D2H sync; later
        # phases keep the broadcast on-device - C8/C9 forbid this sync in the
        # steady-state step).
        var gh = g[].to_host(ctx)
        var ones = ones_like(node.saved[0][], ctx)
        var d_x = _ta_mul_scalar(ones^, gh[0], ctx)
        var out = List[TArc]()
        out.append(TArc(d_x^))
        return out^
    # ── P2 arms: the zimage DiT block vocabulary. Backward = the EXISTING
    # parity-proven ops/*_backward calls (the same functions the hand-chain
    # zimage_block_lora_backward_device_tensors_batch makes,
    # models/zimage/lora_block.mojo:1702-1793). Result-struct Tensor fields are
    # lifted via arc_view (zero-copy buffer share) - Mojo forbids partial moves
    # out of a live struct, and clone would add a d2d copy + sync per op.
    elif node.kind == OPK_PROJ_LORA:
        # saved [x, w, a, b]; meta [M, in_f, out_f, rank]; scalars [scale].
        # edges [x, A_leaf, B_leaf] - W frozen, no edge.
        var lo = ZImageLoraAdapterDevice(
            node.saved[2].copy(), node.saved[3].copy(),
            node.saved_meta[3], node.saved_meta[1], node.saved_meta[2],
            node.scalars[0],
        )
        var pg = proj_lora_backward(
            g[], node.saved[0][], node.saved[1][], lo,
            node.saved_meta[0], node.saved_meta[1], node.saved_meta[2], ctx,
        )
        var out = List[TArc]()
        out.append(pg.d_x.copy())
        out.append(pg.d_a.copy())
        out.append(pg.d_b.copy())
        return out^
    elif node.kind == OPK_RMS_NORM_DX:
        # saved [x, weight]; scalars [eps]; weight frozen -> dx-only arm
        # (ops/norm_backward.mojo:373).
        var d_x = rms_norm_backward_dx(
            g[], node.saved[0][], node.saved[1][], node.scalars[0], ctx
        )
        var out = List[TArc]()
        out.append(TArc(d_x^))
        return out^
    elif node.kind == OPK_MODULATE:
        # saved [x, scale]; meta [param_grads flag]; edges [x, scale]
        # (scale edge null when frozen - the recorder guarantees flag==1 iff
        # the scale edge is a real leaf, so the dummy zero-size d_scale of the
        # frozen path is always dropped at the null edge).
        var want_param = node.saved_meta[0] == 1
        var mb = modulate_backward(
            g[], node.saved[0][], node.saved[1][], ctx,
            compute_param_grads=want_param,
        )
        var out = List[TArc]()
        out.append(arc_view(mb.d_x))
        out.append(arc_view(mb.d_scale))
        return out^
    elif node.kind == OPK_ROPE:
        # saved [cos, sin] (frozen tables); interleaved=True - the hand-chain
        # call (lora_block.mojo:1749-1750).
        var d_x = rope_backward(g[], node.saved[0][], node.saved[1][], True, ctx)
        var out = List[TArc]()
        out.append(TArc(d_x^))
        return out^
    elif node.kind == OPK_SDPA:
        # saved [q_rope, k_rope, v]; meta [B, S, H, Dh]; scalars [scale].
        var sb = sdpa_backward_dispatch(
            node.saved[0][], node.saved[1][], node.saved[2][], g[],
            node.scalars[0],
            node.saved_meta[0], node.saved_meta[1],
            node.saved_meta[2], node.saved_meta[3], ctx,
        )
        var out = List[TArc]()
        out.append(sb.d_q.copy())
        out.append(sb.d_k.copy())
        out.append(sb.d_v.copy())
        return out^
    elif node.kind == OPK_SWIGLU:
        # saved [g_pre, u] -> d_gate, d_up (ops/loss_swiglu_backward.mojo:304).
        var sg = swiglu_backward(g[], node.saved[0][], node.saved[1][], ctx)
        var out = List[TArc]()
        out.append(arc_view(sg.d_gate))
        out.append(arc_view(sg.d_up))
        return out^
    elif node.kind == OPK_RESIDUAL_GATE_DXDY:
        # saved [gate_t] (tanh'd gate vec, frozen); o = x + gate_t*y ->
        # d_x = g, d_y = g*gate_t (ops/rope_struct_backward.mojo:971).
        var grg = gate_residual_backward_dxdy(g[], node.saved[0][], ctx)
        var out = List[TArc]()
        out.append(arc_view(grg.d_x))
        out.append(arc_view(grg.d_y))
        return out^
    elif node.kind == OPK_RESHAPE:
        # saved_meta = the forward INPUT's shape dims; grad reshaped back via
        # a zero-kernel buffer-sharing view (the routed grad arc may be shared
        # with other consumers, so the metadata change must be out-of-place -
        # never reshape_in_place through the arc).
        var back_shape = List[Int]()
        for i in range(len(node.saved_meta)):
            back_shape.append(node.saved_meta[i])
        var out = List[TArc]()
        out.append(arc_view_reshaped(g[], back_shape^))
        return out^
    elif node.kind == OPK_LEAF:
        raise Error("apply: OPK_LEAF is sunk by the engine, never dispatched")
    raise Error(String("apply: unknown node kind ") + String(node.kind))


def _push_ready(mut ready: List[Int], mut in_queue: List[Bool], nid: Int):
    # Dedup against in_queue (flame BUG-FIX 2026-05-13, engine.rs:622-641): a
    # node that is BOTH the seeded root AND a descendant of it must not be
    # pushed twice - the second pop would fire apply() on an empty buffer.
    if in_queue[nid]:
        return
    in_queue[nid] = True
    ready.append(nid)


def _dec_and_maybe_enqueue(
    mut dep: List[Int], mut ready: List[Int], mut in_queue: List[Bool], child: Int
):
    # flame decrement_and_maybe_enqueue (engine.rs:609-643).
    if dep[child] > 0:
        dep[child] -= 1
    if dep[child] == 0:
        _push_ready(ready, in_queue, child)


def _pop_best(mut ready: List[Int], graph: Graph) raises -> Int:
    """Extract the node with max (topological_nr, sequence_nr, node_id) -
    DESC order so the node deepest in the DAG fires first, guaranteeing all
    grad contributions have flowed in (ReadyKey, engine.rs:130-165).
    P1: linear max-extract over a List (toy node counts); the BinaryHeap
    upgrade (flame's BinaryHeap<(ReadyKey, NodeId)>) is a later optimization."""
    if len(ready) == 0:
        raise Error("_pop_best: empty ready queue")
    var best = 0
    for i in range(1, len(ready)):
        var a = ready[i]
        var b = ready[best]
        var at = graph.nodes[a].topological_nr
        var bt = graph.nodes[b].topological_nr
        var as_ = graph.nodes[a].sequence_nr
        var bs = graph.nodes[b].sequence_nr
        if at > bt or (at == bt and (as_ > bs or (as_ == bs and a > b))):
            best = i
    var val = ready[best]
    ready[best] = ready[len(ready) - 1]
    _ = ready.pop()
    return val


def execute(
    mut graph: Graph, root_node: Int, root_grad: TArc, ctx: DeviceContext
) raises -> Dict[Int, TArc]:
    """Drive backward from `root_node`, seeded with `root_grad`. Returns
    param_id -> grad, read off the OPK_LEAF sinks (the standard-backward
    collection path; flame's leaves sink into meta.grad, engine.rs:181-188 -
    here the result Dict IS the leaf grad store, contract C3).

    All engine state (dep counts, buffers, ready queue, fired counter) is
    local to this call - nested execute is legal (engine.rs:171-177)."""
    var n = len(graph.nodes)
    if root_node < 0 or root_node >= n:
        raise Error("execute: root_node out of range")

    # ── Step 1: dep-count BFS over edges from the root (engine.rs:230-277).
    # dep[child] += 1 per incoming edge; duplicate-safe via the seen set.
    var dep = List[Int]()
    var seen = List[Bool]()
    var in_queue = List[Bool]()
    for _ in range(n):
        dep.append(0)
        seen.append(False)
        in_queue.append(False)
    var stack = List[Int]()
    stack.append(root_node)
    var reachable = 0
    while len(stack) > 0:
        var nid = stack.pop()
        if seen[nid]:
            continue
        seen[nid] = True
        reachable += 1
        for i in range(len(graph.nodes[nid].edges)):
            var child = graph.nodes[nid].edges[i].node_idx
            if child >= 0:
                dep[child] += 1  # engine.rs:268-270
                stack.append(child)

    # ── Step 2: per-node InputBuffers + seed the root (engine.rs:282-393).
    # Buffers are per-contribution-slotted (C15): sized off each node's
    # recorded contrib_counts (+1 reserved seed slot per input); cells hold a
    # shared placeholder arc (refcount copy of root_grad, never read while
    # !present).
    var buffers = List[InputBuffer]()
    for i in range(n):
        buffers.append(InputBuffer(graph.nodes[i].contrib_counts, root_grad.copy()))

    # The root grad goes into slot output_nr of the root node's buffer
    # (engine.rs:305-332). P1 execute takes the root NODE directly and its
    # root is single-output, so the input slot is 0; the seed occupies the
    # reserved trailing contribution slot (the caller-supplied grad is one
    # contribution - flame outputs-as-descendants fix).
    buffers[root_node].add(0, buffers[root_node].seed_slot(0), root_grad.copy(), ctx)

    # Output-seeding fix (flame BUG-FIX 2026-05-13, engine.rs:335-393): the
    # caller-supplied root grad IS one dependency contribution. Bump the
    # root's dep count by 1, then decrement-and-maybe-enqueue: a root that is
    # nobody's descendant goes (0+1)->0 and enqueues immediately; a root that
    # IS a descendant of another seeded output would wait for its full count.
    var ready = List[Int]()
    dep[root_node] += 1
    _dec_and_maybe_enqueue(dep, ready, in_queue, root_node)

    # ── Step 3: drive the queue (engine.rs:437-570).
    var result = Dict[Int, TArc]()
    var fired = 0
    while len(ready) > 0:
        var nid = _pop_best(ready, graph)
        fired += 1

        # Materialize this node's input grads from its buffer
        # (engine.rs:448-455): slot-ordered left fold per input (C15).
        # Invariant: every arm emits a grad for every edge it owns, so a
        # firing node has at least one contribution per input; a hole means
        # an engine bug -> raise (flame models holes as None slots; these
        # kinds never produce them).
        var num_in = graph.nodes[nid].num_inputs
        var grads_in = List[TArc]()
        for s in range(num_in):
            if not buffers[nid].any_present(s):
                raise Error(
                    String("execute: node ")
                    + String(nid)
                    + " fired with missing input grad in slot "
                    + String(s)
                )
            grads_in.append(buffers[nid].materialize(s, ctx))

        if graph.nodes[nid].kind == OPK_LEAF:
            # AccumulateGrad sink (accumulator.rs:137-206): accumulate into
            # the result Dict keyed by param_id. First contributor stores;
            # later contributors add out-of-place (the leaf's InputBuffer has
            # already merged same-graph contributions; this merge covers a
            # param shared across leaves only defensively). No next_edges ->
            # arity check trivially holds (engine.rs:509-520 note on
            # AccumulateGrad's empty return).
            var pid = graph.nodes[nid].param_id
            if result.__contains__(pid):
                var old = result[pid]
                var summed = _raw_add(old[], grads_in[0][], ctx)
                result[pid] = TArc(summed^)
            else:
                result[pid] = grads_in[0].copy()
            continue

        var out_grads = apply(graph.nodes[nid], grads_in, ctx)

        # Arity check: one grad per next_edge (engine.rs:509-520).
        var n_edges = len(graph.nodes[nid].edges)
        if len(out_grads) != n_edges:
            raise Error(
                String("execute: apply arity mismatch on node ")
                + String(nid)
                + ": expected "
                + String(n_edges)
                + " got "
                + String(len(out_grads))
            )

        # Route each output grad along its edge into the child's InputBuffer
        # at (input_nr, contrib_slot); decrement child dep; enqueue at zero
        # (engine.rs:529-564). contrib_slot was assigned at recording time
        # (C15) - arrival order is irrelevant to the fan-in fold.
        for s in range(n_edges):
            var child = graph.nodes[nid].edges[s].node_idx
            if child < 0:
                continue  # null edge: drop the grad (engine.rs:532-535)
            var slot = graph.nodes[nid].edges[s].input_nr
            var cslot = graph.nodes[nid].edges[s].contrib_slot
            buffers[child].add(slot, cslot, out_grads[s].copy(), ctx)
            _dec_and_maybe_enqueue(dep, ready, in_queue, child)

    # ── Invariant: every reachable node fired exactly once (the design doc's
    # dep-count exactness gate; flame's queue structure guarantees it, we
    # assert it).
    if fired != reachable:
        raise Error(
            String("execute: fired=")
            + String(fired)
            + " != reachable="
            + String(reachable)
            + " (dep-count exactness violated)"
        )
    return result^
