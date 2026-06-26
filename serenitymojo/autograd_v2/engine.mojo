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
from serenitymojo.ops.norm_backward import (
    rms_norm_backward_dx,
    rms_norm_backward_dx_slab,
)
from serenitymojo.ops.elementwise_backward import (
    modulate_backward,
    modulate_backward_slab,
)
from serenitymojo.ops.rope_struct_backward import (
    rope_backward,
    gate_residual_backward_dxdy,
    rope_backward_slab,
    gate_residual_backward_dxdy_slab,
)
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward, swiglu_backward_slab
from serenitymojo.models.zimage.lora_block import ZImageLoraAdapterDevice

# ── P6 Klein imports: the apply_klein arms call the Klein hand-chain's OWN
# backward helpers / inline op sequence verbatim (file:line cited per arm).
from serenitymojo.ops.norm_backward import layer_norm_backward_dx
from serenitymojo.ops.linalg_backward import (
    linear_backward_dx_scratch,
    linear_backward_dx_split_scratch,
)
from serenitymojo.ops.attention_backward import sdpa_backward_scratch
from serenitymojo.ops.tensor_algebra import (
    add as _ta_add_klein,
    slice as _ta_slice_klein,
    reshape as _ta_reshape_klein,
    reshape_owned as _ta_reshape_owned_klein,
    reshape_in_place as _ta_reshape_in_place_klein,
    add_in_place_f32 as _ta_add_in_place_f32_klein,
)
from serenitymojo.ops.tensor_algebra_scratch import (
    concat2_scratch,
    concat3_scratch,
    slice_scratch,
)
from serenitymojo.ops.tensor_algebra import concat as _ta_concat_klein
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.models.klein.double_block import (
    StreamSaved,
    StreamWeights,
    ModVecsDevice,
    StreamLoraDevice,
    _stream_pre_backward_lora_resident_scratch_tensors,
    _stream_post_backward_lora_resident_scratch_tensors,
)
from serenitymojo.ops.attention_flash import (
    sdpa_flash_backward_f32, sdpa_flash_backward_dispatch,
)
from serenitymojo.ops.cast import cast_tensor as cast_tensor_eng
from serenitymojo.io.dtype import STDtype as STDtypeEng
from serenitymojo.models.klein.single_block import (
    LoraDropout,
    _klein_lora_bwd_dropout_tensors,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from std.collections import Optional
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
    OPK_KLEIN_DBL_PRE,
    OPK_KLEIN_DBL_JOINT,
    OPK_KLEIN_DBL_POST,
    OPK_KLEIN_SGL_IN,
    OPK_KLEIN_SGL_SDPA,
    OPK_KLEIN_SGL_OUT,
    _raw_add,
    _raw_add_slab,
    _raw_mul,
    ones_like,
    arc_view,
    arc_view_reshaped,
)
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.input_buffer import InputBuffer
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.autograd_v2.ops_record import (
    proj_lora_backward,
    sdpa_backward_dispatch,
    proj_lora_backward_slab,
    sdpa_backward_dispatch_slab,
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
        if len(node.saved) >= 8:
            # ZIMAGE_SDPA_FLASH recording (saved 3..7 = padded q/k/v/o+stats)
            var g_bf = cast_tensor_eng(g[], STDtypeEng.BF16, ctx)
            var fb = sdpa_flash_backward_dispatch(
                node.saved[3].copy(), node.saved[4].copy(),
                node.saved[5].copy(), node.saved[6].copy(),
                node.saved[7].copy(), g_bf, node.scalars[0],
                node.saved_meta[0], node.saved_meta[1],
                node.saved_meta[2], node.saved_meta[3], ctx,
            )
            var outf = List[TArc]()
            outf.append(TArc(cast_tensor_eng(fb.d_q, STDtypeEng.F32, ctx)))
            outf.append(TArc(cast_tensor_eng(fb.d_k, STDtypeEng.F32, ctx)))
            outf.append(TArc(cast_tensor_eng(fb.d_v, STDtypeEng.F32, ctx)))
            return outf^
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


# ─────────────────────────────────────────────────────────────────────────────
# StepSlab variants (Phase P4, contract C8): apply_slab/execute_slab are
# copies of apply/execute with every backward arm routed to its _slab sibling
# and every fan-in/leaf accumulation through _raw_add_slab. The non-slab pair
# above stays untouched (C13 gate-don't-delete; the P1 toy gates and the _v3
# path keep using it).
# ─────────────────────────────────────────────────────────────────────────────


def apply_slab(
    node: Node, grads_in: List[TArc], ctx: DeviceContext, mut slab: StepSlab
) raises -> List[TArc]:
    """StepSlab variant of `apply` (this file :65) for the zimage DiT
    vocabulary (the P3/P4 graph path). The P1 toy kinds (MUL/MATMUL/SUM) are
    NOT slab-routed — they only run under the non-slab execute (toy gates);
    dispatching one here is an engine bug -> raise (fail loud)."""
    var g = grads_in[0].copy()  # every kind here is single-forward-output
    if node.kind == OPK_ADD:
        # d/da (a+b) = d/db (a+b) = g; share the arc (no allocation).
        var out = List[TArc]()
        out.append(g.copy())
        out.append(g.copy())
        return out^
    elif node.kind == OPK_PROJ_LORA:
        # saved [x, w, a, b]; meta [M, in_f, out_f, rank]; scalars [scale].
        var lo = ZImageLoraAdapterDevice(
            node.saved[2].copy(), node.saved[3].copy(),
            node.saved_meta[3], node.saved_meta[1], node.saved_meta[2],
            node.scalars[0],
        )
        var pg = proj_lora_backward_slab(
            g[], node.saved[0][], node.saved[1][], lo,
            node.saved_meta[0], node.saved_meta[1], node.saved_meta[2], ctx,
            slab,
        )
        var out = List[TArc]()
        out.append(pg.d_x.copy())
        out.append(pg.d_a.copy())
        out.append(pg.d_b.copy())
        return out^
    elif node.kind == OPK_RMS_NORM_DX:
        var d_x = rms_norm_backward_dx_slab(
            g[], node.saved[0][], node.saved[1][], node.scalars[0], ctx, slab
        )
        var out = List[TArc]()
        out.append(TArc(d_x^))
        return out^
    elif node.kind == OPK_MODULATE:
        var want_param = node.saved_meta[0] == 1
        var mb = modulate_backward_slab(
            g[], node.saved[0][], node.saved[1][], ctx, slab,
            compute_param_grads=want_param,
        )
        var out = List[TArc]()
        out.append(arc_view(mb.d_x))
        out.append(arc_view(mb.d_scale))
        return out^
    elif node.kind == OPK_ROPE:
        var d_x = rope_backward_slab(
            g[], node.saved[0][], node.saved[1][], True, ctx, slab
        )
        var out = List[TArc]()
        out.append(TArc(d_x^))
        return out^
    elif node.kind == OPK_SDPA:
        if len(node.saved) >= 8:
            # ZIMAGE_SDPA_FLASH (pool allocs — capture must be off)
            var g_bf = cast_tensor_eng(g[], STDtypeEng.BF16, ctx)
            var fb = sdpa_flash_backward_dispatch(
                node.saved[3].copy(), node.saved[4].copy(),
                node.saved[5].copy(), node.saved[6].copy(),
                node.saved[7].copy(), g_bf, node.scalars[0],
                node.saved_meta[0], node.saved_meta[1],
                node.saved_meta[2], node.saved_meta[3], ctx,
            )
            var outf = List[TArc]()
            outf.append(TArc(cast_tensor_eng(fb.d_q, STDtypeEng.F32, ctx)))
            outf.append(TArc(cast_tensor_eng(fb.d_k, STDtypeEng.F32, ctx)))
            outf.append(TArc(cast_tensor_eng(fb.d_v, STDtypeEng.F32, ctx)))
            return outf^
        var sb = sdpa_backward_dispatch_slab(
            node.saved[0][], node.saved[1][], node.saved[2][], g[],
            node.scalars[0],
            node.saved_meta[0], node.saved_meta[1],
            node.saved_meta[2], node.saved_meta[3], ctx, slab,
        )
        var out = List[TArc]()
        out.append(sb.d_q.copy())
        out.append(sb.d_k.copy())
        out.append(sb.d_v.copy())
        return out^
    elif node.kind == OPK_SWIGLU:
        var sg = swiglu_backward_slab(g[], node.saved[0][], node.saved[1][], ctx, slab)
        var out = List[TArc]()
        out.append(arc_view(sg.d_gate))
        out.append(arc_view(sg.d_up))
        return out^
    elif node.kind == OPK_RESIDUAL_GATE_DXDY:
        var grg = gate_residual_backward_dxdy_slab(g[], node.saved[0][], ctx, slab)
        var out = List[TArc]()
        out.append(arc_view(grg.d_x))
        out.append(arc_view(grg.d_y))
        return out^
    elif node.kind == OPK_RESHAPE:
        # Zero-kernel, zero-allocation metadata view (same as apply).
        var back_shape = List[Int]()
        for i in range(len(node.saved_meta)):
            back_shape.append(node.saved_meta[i])
        var out = List[TArc]()
        out.append(arc_view_reshaped(g[], back_shape^))
        return out^
    elif node.kind == OPK_LEAF:
        raise Error("apply_slab: OPK_LEAF is sunk by the engine, never dispatched")
    raise Error(
        String("apply_slab: kind ")
        + String(node.kind)
        + " is not slab-routed (P4 covers the zimage DiT vocabulary)"
    )


def execute_slab(
    mut graph: Graph, root_node: Int, root_grad: TArc, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Dict[Int, TArc]:
    """StepSlab variant of `execute` (this file :263) — identical engine
    algorithm (dep-count BFS, slot-ordered buffers, ready-queue order, arity
    checks, fired==reachable invariant); materialize/apply/leaf-merge route
    through the slab. Returned grads are SLAB-RESIDENT views: the caller must
    copy them out of the slab before rewinding past its mark."""
    var n = len(graph.nodes)
    if root_node < 0 or root_node >= n:
        raise Error("execute: root_node out of range")

    # ── Step 1: dep-count BFS over edges from the root (engine.rs:230-277).
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
    var buffers = List[InputBuffer]()
    for i in range(n):
        buffers.append(InputBuffer(graph.nodes[i].contrib_counts, root_grad.copy()))

    buffers[root_node].add(0, buffers[root_node].seed_slot(0), root_grad.copy(), ctx)

    var ready = List[Int]()
    dep[root_node] += 1
    _dec_and_maybe_enqueue(dep, ready, in_queue, root_node)

    # ── Step 3: drive the queue (engine.rs:437-570).
    var result = Dict[Int, TArc]()
    var fired = 0
    while len(ready) > 0:
        var nid = _pop_best(ready, graph)
        fired += 1

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
            grads_in.append(buffers[nid].materialize_slab(s, ctx, slab))

        if graph.nodes[nid].kind == OPK_LEAF:
            var pid = graph.nodes[nid].param_id
            if result.__contains__(pid):
                var old = result[pid]
                var summed = _raw_add_slab(old[], grads_in[0][], ctx, slab)
                result[pid] = TArc(summed^)
            else:
                result[pid] = grads_in[0].copy()
            continue

        var out_grads = apply_slab(graph.nodes[nid], grads_in, ctx, slab)

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

        for s in range(n_edges):
            var child = graph.nodes[nid].edges[s].node_idx
            if child < 0:
                continue  # null edge: drop the grad (engine.rs:532-535)
            var slot = graph.nodes[nid].edges[s].input_nr
            var cslot = graph.nodes[nid].edges[s].contrib_slot
            buffers[child].add(slot, cslot, out_grads[s].copy(), ctx)
            _dec_and_maybe_enqueue(dep, ready, in_queue, child)

    if fired != reachable:
        raise Error(
            String("execute: fired=")
            + String(fired)
            + " != reachable="
            + String(reachable)
            + " (dep-count exactness violated)"
        )
    return result^


# ─────────────────────────────────────────────────────────────────────────────
# P6 Klein variants (AUTOGRAD_V2_MOJO_DESIGN.md P6): apply_klein/execute_klein
# follow the apply/execute_slab precedent above — the IDENTICAL engine
# algorithm with (a) the Klein composite-node arms (each arm = the Klein
# hand-chain's own backward helper / inline op sequence, file:line cited),
# (b) a threaded ScratchRingAllocator (the Klein oracle backward runs through
# the trainer's scratch_bwd ring; "preserve whatever the oracle does"), and
# (c) MULTI-ROOT seeding (the double block has TWO outputs, img_out+txt_out,
# each with its own caller grad — the engine.rs:335-393 outputs-as-descendants
# seeding generalized to a root list: bump ALL root deps first, then
# decrement-and-maybe-enqueue each).
#
# comptime [H, Dh, S]: sdpa_backward_scratch is comptime-[B,S,H,Dh]
# specialized and Klein's S=1536 bucket is fixed by the trainer comptime, so
# the Klein engine entry is comptime-bucketed like the trainer itself
# (the design-doc "trainers are comptime-bucketed" hazard note); runtime
# N_TXT/N_IMG/D/F dims ride in node.saved_meta.
#
# No StepSlab and no CUDA capture for Klein in P6 (scope decision: the
# block-swap copy stream makes capture a separate workstream; the scratch
# rings keep doing what they do).
# ─────────────────────────────────────────────────────────────────────────────


def _req_arc(o: Optional[TArc], name: String) raises -> TArc:
    """Unwrap a grad Optional the Klein helpers ALWAYS populate when the
    adapter is present (the record wrappers require every adapter)."""
    if o:
        return o.value().copy()
    raise Error(String("apply_klein: missing LoRA grad tensor: ") + name)


def apply_klein[
    H: Int, Dh: Int, S: Int
](
    node: Node, grads_in: List[TArc], ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> List[TArc]:
    """Kind-dispatched backward for the Klein P6 vocabulary (+ OPK_SWIGLU,
    which the single block shares with zimage — same swiglu_backward call,
    models/klein/single_block.mojo:1404). Non-Klein kinds raise (fail loud:
    the Klein graphs record only these kinds)."""
    if node.kind == OPK_KLEIN_DBL_PRE:
        # Arm = _stream_pre_backward_lora_resident_scratch_tensors[H,Dh]
        # (models/klein/double_block.mojo:2071-2136), compute_aux_grads=False,
        # default (p==0) dropout — the EXACT call double_block_lora_backward_
        # device_resident_scratch_tensors makes (:2339-2346). Unused struct
        # fields are placeholders (refcount copies, never read by the helper:
        # it reads sv.{x,ln1,norm,q_pre,k_pre}, w.{wqkv,q_norm,k_norm},
        # mv.scale1, lo.{q,k,v}).
        var N = node.saved_meta[0]
        var D = node.saved_meta[1]
        var rank = node.saved_meta[2]
        var eps = node.scalars[0]
        var ph = node.saved[0].copy()
        var sv = StreamSaved(
            node.saved[0].copy(), node.saved[1].copy(), node.saved[2].copy(),
            node.saved[3].copy(), node.saved[4].copy(),
            ph.copy(), ph.copy(), ph.copy(), ph.copy(), ph.copy(), ph.copy(),
            ph.copy(), ph.copy(), ph.copy(), ph.copy(), ph.copy(),
        )
        var ph_w = node.saved[5].copy()
        var w = StreamWeights(
            node.saved[5].copy(), ph_w.copy(), ph_w.copy(), ph_w.copy(),
            node.saved[6].copy(), node.saved[7].copy(),
        )
        var ph_v = node.saved[8].copy()
        var mv = ModVecsDevice(
            ph_v.copy(), node.saved[8].copy(), ph_v.copy(),
            ph_v.copy(), ph_v.copy(), ph_v.copy(),
        )
        var lo = StreamLoraDevice(
            Optional[LoraAdapterDevice](LoraAdapterDevice(
                node.saved[10].copy(), node.saved[11].copy(),
                rank, D, D, node.scalars[1])),
            Optional[LoraAdapterDevice](LoraAdapterDevice(
                node.saved[12].copy(), node.saved[13].copy(),
                rank, D, D, node.scalars[2])),
            Optional[LoraAdapterDevice](LoraAdapterDevice(
                node.saved[14].copy(), node.saved[15].copy(),
                rank, D, D, node.scalars[3])),
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None),
        )
        var r = _stream_pre_backward_lora_resident_scratch_tensors[H, Dh](
            grads_in[0][], grads_in[1][], grads_in[2][],
            w, mv, lo, sv, N, D, eps, node.saved[9][], ctx, scratch,
            compute_aux_grads=False,
        )
        var out = List[TArc]()
        out.append(r.d_x.copy())
        out.append(_req_arc(r.q_d_a, String("dbl q_d_a")))
        out.append(_req_arc(r.q_d_b, String("dbl q_d_b")))
        out.append(_req_arc(r.k_d_a, String("dbl k_d_a")))
        out.append(_req_arc(r.k_d_b, String("dbl k_d_b")))
        out.append(_req_arc(r.v_d_a, String("dbl v_d_a")))
        out.append(_req_arc(r.v_d_b, String("dbl v_d_b")))
        return out^
    elif node.kind == OPK_KLEIN_DBL_JOINT:
        # Arm = the oracle's joint backward block, double_block_lora_backward_
        # device_resident_scratch_tensors (models/klein/double_block.mojo:
        # 2319-2337): reshape per-stream d_att to 4-D, concat2_scratch (txt
        # FIRST), sdpa_backward_scratch, rope_backward x2, slice_scratch x6
        # (txt then img per q/k/v), reshape_in_place on the v slices.
        # saved [q_rope, k_rope, v_joint, cos, sin]; meta [N_TXT, N_IMG, D];
        # scalars [scale]. grads_in: 0=d_txt_att [N_TXT,D], 1=d_img_att.
        var N_TXT = node.saved_meta[0]
        var N_IMG = node.saved_meta[1]
        var D = node.saved_meta[2]
        var scale = node.scalars[0]
        var d_tatt_4d = _ta_reshape_klein(grads_in[0][], [1, N_TXT, H, Dh], ctx)
        var d_iatt_4d = _ta_reshape_klein(grads_in[1][], [1, N_IMG, H, Dh], ctx)
        var d_att_joint = concat2_scratch(1, ctx, scratch, d_tatt_4d, d_iatt_4d)

        var d_q_t: Tensor
        var d_k_t: Tensor
        var d_v_t: Tensor
        if len(node.saved) >= 10:
            # KLEIN_SDPA_FLASH recording (saved 5..9): the SAME flash call
            # the hand-chain double backward makes.
            var fb = sdpa_flash_backward_f32[1, S, H, Dh](
                node.saved[5].copy(), node.saved[6].copy(),
                node.saved[7].copy(), node.saved[8].copy(),
                node.saved[9].copy(), d_att_joint, scale, ctx,
            )
            d_q_t = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
            d_k_t = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
            d_v_t = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
        else:
            var sb = sdpa_backward_scratch[1, S, H, Dh](
                node.saved[0][], node.saved[1][], node.saved[2][],
                d_att_joint, scale, ctx, scratch,
            )
            d_q_t = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
            d_k_t = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
            d_v_t = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

        var d_q_joint = rope_backward(d_q_t, node.saved[3][], node.saved[4][], True, ctx)
        var d_k_joint = rope_backward(d_k_t, node.saved[3][], node.saved[4][], True, ctx)

        var d_txt_q = slice_scratch(d_q_joint, 1, 0, N_TXT, ctx, scratch)
        var d_img_q = slice_scratch(d_q_joint, 1, N_TXT, N_IMG, ctx, scratch)
        var d_txt_k = slice_scratch(d_k_joint, 1, 0, N_TXT, ctx, scratch)
        var d_img_k = slice_scratch(d_k_joint, 1, N_TXT, N_IMG, ctx, scratch)
        var d_txt_v = slice_scratch(d_v_t, 1, 0, N_TXT, ctx, scratch)
        var d_img_v = slice_scratch(d_v_t, 1, N_TXT, N_IMG, ctx, scratch)
        _ta_reshape_in_place_klein(d_img_v, [N_IMG, D])
        _ta_reshape_in_place_klein(d_txt_v, [N_TXT, D])

        # Edge order [tq, iq, tk, ik, tv, iv] — grads routed accordingly. The
        # slices are SCRATCH-RESIDENT views consumed by the PRE arms before the
        # block-level rewind (klein_block_graph rewinds AFTER execute returns),
        # exactly the oracle's lifetime pattern.
        var out = List[TArc]()
        out.append(TArc(d_txt_q^))
        out.append(TArc(d_img_q^))
        out.append(TArc(d_txt_k^))
        out.append(TArc(d_img_k^))
        out.append(TArc(d_txt_v^))
        out.append(TArc(d_img_v^))
        return out^
    elif node.kind == OPK_KLEIN_DBL_POST:
        # Arm = _stream_post_backward_lora_resident_scratch_tensors
        # (models/klein/double_block.mojo:1781-1875), compute_aux_grads=False —
        # the EXACT call the oracle backward makes (:2310-2317). The helper
        # reads sv.{act,gate,up,mlp_in,ln2,attn_res}, w.{wproj,wgu,wd},
        # mv.{gate1,scale2,gate2}, lo.{out,ff_in,ff_out}; x/att ride as args.
        var N = node.saved_meta[0]
        var D = node.saved_meta[1]
        var F = node.saved_meta[2]
        var rank = node.saved_meta[3]
        var eps = node.scalars[0]
        var ph = node.saved[0].copy()
        var sv = StreamSaved(
            node.saved[0].copy(), ph.copy(), ph.copy(), ph.copy(), ph.copy(),
            ph.copy(), ph.copy(), ph.copy(),
            node.saved[1].copy(), node.saved[2].copy(), node.saved[3].copy(),
            node.saved[4].copy(), ph.copy(), node.saved[5].copy(),
            node.saved[6].copy(), node.saved[7].copy(),
        )
        var ph_w = node.saved[8].copy()
        var w = StreamWeights(
            ph_w.copy(), node.saved[8].copy(), node.saved[9].copy(),
            node.saved[10].copy(), ph_w.copy(), ph_w.copy(),
        )
        var ph_v = node.saved[11].copy()
        var mv = ModVecsDevice(
            ph_v.copy(), ph_v.copy(), node.saved[11].copy(),
            ph_v.copy(), node.saved[12].copy(), node.saved[13].copy(),
        )
        var lo = StreamLoraDevice(
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](LoraAdapterDevice(
                node.saved[15].copy(), node.saved[16].copy(),
                rank, D, D, node.scalars[1])),
            Optional[LoraAdapterDevice](LoraAdapterDevice(
                node.saved[17].copy(), node.saved[18].copy(),
                rank, D, 2 * F, node.scalars[2])),
            Optional[LoraAdapterDevice](LoraAdapterDevice(
                node.saved[19].copy(), node.saved[20].copy(),
                rank, F, D, node.scalars[3])),
        )
        var r = _stream_post_backward_lora_resident_scratch_tensors(
            grads_in[0], node.saved[0], node.saved[1],
            w, mv, lo, sv, N, D, F, eps, node.saved[14][], ctx, scratch,
            compute_aux_grads=False,
        )
        var out = List[TArc]()
        out.append(r.d_x.copy())
        out.append(r.d_att.copy())
        out.append(_req_arc(r.out_d_a, String("dbl out_d_a")))
        out.append(_req_arc(r.out_d_b, String("dbl out_d_b")))
        out.append(_req_arc(r.ff_in_d_a, String("dbl ff_in_d_a")))
        out.append(_req_arc(r.ff_in_d_b, String("dbl ff_in_d_b")))
        out.append(_req_arc(r.ff_out_d_a, String("dbl ff_out_d_a")))
        out.append(_req_arc(r.ff_out_d_b, String("dbl ff_out_d_b")))
        return out^
    elif node.kind == OPK_KLEIN_SGL_IN:
        # Arm = the IN segment of single_block_lora_backward_device_resident_
        # scratch_tensors (models/klein/single_block.mojo:1414-1445),
        # compute_aux_grads=False: rms_norm_backward_dx x2, reshape_owned,
        # concat3_scratch(reverse=True) d_qkv, concat2_scratch d_gate_up,
        # linear_backward_dx_split_scratch, concat d_fused,
        # _klein_lora_bwd_dropout_tensors (p==0), add fold, modulate_backward
        # (aux off), layer_norm_backward_dx — same ops, same order, same fold.
        # grads_in: 0=d_q_rms [1,S,H,Dh], 1=d_k_rms, 2=d_v [S,D],
        # 3=d_mlp_gate [S,F], 4=d_mlp_up [S,F].
        var S_rows = node.saved_meta[0]
        var D = node.saved_meta[1]
        var F = node.saved_meta[2]
        var rank = node.saved_meta[3]
        var eps = node.scalars[0]
        var d_q_pre_t = rms_norm_backward_dx(
            grads_in[0][], node.saved[3][], node.saved[6][], eps, ctx
        )
        var d_k_pre_t = rms_norm_backward_dx(
            grads_in[1][], node.saved[4][], node.saved[7][], eps, ctx
        )
        var d_q_pre_flat = _ta_reshape_owned_klein(d_q_pre_t^, [S_rows, D])
        var d_k_pre_flat = _ta_reshape_owned_klein(d_k_pre_t^, [S_rows, D])
        var d_qkv = concat3_scratch(
            1, ctx, scratch, d_q_pre_flat, d_k_pre_flat, grads_in[2][], True
        )
        # d_gate_up = concat of the swiglu grads (the oracle concats
        # sgb.d_gate/sgb.d_up at :1405; here they arrive routed from the
        # OPK_SWIGLU node — same tensors, same concat2_scratch call).
        var d_gate_up = concat2_scratch(1, ctx, scratch, grads_in[3][], grads_in[4][])

        var d_norm_t = linear_backward_dx_split_scratch(
            d_qkv, d_gate_up, node.saved[5][], S_rows, D, 3 * D, 2 * F, ctx, scratch,
        )

        var d_fused = _ta_concat_klein(1, ctx, d_qkv, d_gate_up)
        var lo_qkv = LoraAdapterDevice(
            node.saved[10].copy(), node.saved[11].copy(),
            rank, D, 3 * D + 2 * F, node.scalars[1],
        )
        var lg = _klein_lora_bwd_dropout_tensors(
            d_fused, node.saved[2][], lo_qkv, S_rows, LoraDropout(), ctx
        )
        d_norm_t = _ta_add_klein(d_norm_t, lg.d_x[], ctx)

        var mb = modulate_backward(
            d_norm_t, node.saved[1][], node.saved[8][], ctx,
            compute_param_grads=False,
        )
        var d_x_norm_t = layer_norm_backward_dx(
            mb.d_x, node.saved[0][], node.saved[9][], eps, ctx
        )
        var out = List[TArc]()
        out.append(TArc(d_x_norm_t^))
        out.append(lg.d_a.copy())
        out.append(lg.d_b.copy())
        return out^
    elif node.kind == OPK_KLEIN_SGL_SDPA:
        # Arm = the sdpa segment of the single oracle (models/klein/
        # single_block.mojo:1402-1412): d_att reshaped [1,S,H,Dh] (byte view,
        # the oracle's reshape_in_place), sdpa_backward_scratch,
        # rope_backward x2, d_v reshaped [S,D] (the oracle's :1419
        # reshape_in_place). saved [q_rope, k_rope, v, cos, sin].
        var S_rows = node.saved_meta[0]
        var D = node.saved_meta[1]
        var scale = node.scalars[0]
        var d_att4 = arc_view_reshaped(grads_in[0][], [1, S_rows, H, Dh])
        var d_q_t: Tensor
        var d_k_t: Tensor
        var d_v_t: Tensor
        if len(node.saved) >= 10:
            # KLEIN_SDPA_FLASH recording (saved 5..9 = bf16 q/k/v/o + stats):
            # the flash backward — the SAME call the hand-chain helper makes.
            var fb = sdpa_flash_backward_f32[1, S, H, Dh](
                node.saved[5].copy(), node.saved[6].copy(),
                node.saved[7].copy(), node.saved[8].copy(),
                node.saved[9].copy(), d_att4[], scale, ctx,
            )
            d_q_t = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
            d_k_t = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
            d_v_t = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
        else:
            var sb = sdpa_backward_scratch[1, S, H, Dh](
                node.saved[0][], node.saved[1][], node.saved[2][],
                d_att4[], scale, ctx, scratch,
            )
            d_q_t = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
            d_k_t = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
            d_v_t = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())
        var d_q_rms = rope_backward(d_q_t, node.saved[3][], node.saved[4][], True, ctx)
        var d_k_rms = rope_backward(d_k_t, node.saved[3][], node.saved[4][], True, ctx)
        _ta_reshape_in_place_klein(d_v_t, [S_rows, D])
        var out = List[TArc]()
        out.append(TArc(d_q_rms^))
        out.append(TArc(d_k_rms^))
        out.append(arc_view(d_v_t))
        return out^
    elif node.kind == OPK_KLEIN_SGL_OUT:
        # Arm = the OUT segment of the single oracle (models/klein/
        # single_block.mojo:1364-1400), compute_aux_grads=False:
        # gate_residual_backward_dxdy (gate vec raw — Klein gates are NOT
        # tanh'd), linear_backward_dx_scratch vs w2_att and w2_mlp,
        # _klein_lora_bwd_dropout_tensors on out_in, add_in_place_f32 of the
        # LoRA d_x column slices into d_att/d_mlp. saved [out_in, w2_att,
        # w2_mlp, gate_vec, out_a, out_b].
        var S_rows = node.saved_meta[0]
        var D = node.saved_meta[1]
        var F = node.saved_meta[2]
        var rank = node.saved_meta[3]
        var grg = gate_residual_backward_dxdy(grads_in[0][], node.saved[3][], ctx)
        var d_att = linear_backward_dx_scratch(
            grg.d_y, node.saved[1][], S_rows, D, D, ctx, scratch,
        )
        var d_mlp = linear_backward_dx_scratch(
            grg.d_y, node.saved[2][], S_rows, F, D, ctx, scratch,
        )
        var lo_out = LoraAdapterDevice(
            node.saved[4].copy(), node.saved[5].copy(),
            rank, D + F, D, node.scalars[0],
        )
        var lg2 = _klein_lora_bwd_dropout_tensors(
            grg.d_y, node.saved[0][], lo_out, S_rows, LoraDropout(), ctx
        )
        _ta_add_in_place_f32_klein(d_att, _ta_slice_klein(lg2.d_x[], 1, 0, D, ctx), ctx)
        _ta_add_in_place_f32_klein(d_mlp, _ta_slice_klein(lg2.d_x[], 1, D, F, ctx), ctx)
        var out = List[TArc]()
        out.append(arc_view(grg.d_x))
        out.append(TArc(d_att^))
        out.append(TArc(d_mlp^))
        out.append(lg2.d_a.copy())
        out.append(lg2.d_b.copy())
        return out^
    elif node.kind == OPK_SWIGLU:
        # Shared with zimage: swiglu_backward(g, gate, up) — the single
        # block's exact call (models/klein/single_block.mojo:1404).
        var sg = swiglu_backward(grads_in[0][], node.saved[0][], node.saved[1][], ctx)
        var out = List[TArc]()
        out.append(arc_view(sg.d_gate))
        out.append(arc_view(sg.d_up))
        return out^
    elif node.kind == OPK_LEAF:
        raise Error("apply_klein: OPK_LEAF is sunk by the engine, never dispatched")
    raise Error(
        String("apply_klein: kind ")
        + String(node.kind)
        + " is not in the Klein P6 vocabulary"
    )


def execute_klein[
    H: Int, Dh: Int, S: Int
](
    mut graph: Graph, roots: List[Int], root_grads: List[TArc],
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> Dict[Int, TArc]:
    """Klein variant of `execute` (this file :275) — identical engine
    algorithm (dep-count BFS, slot-ordered buffers, ready-queue order, arity
    checks, fired==reachable invariant) with MULTI-ROOT seeding and
    apply_klein dispatch. Seeding follows the flame outputs-as-descendants
    fix generalized to N roots: every root's buffer takes its caller grad in
    the reserved seed slot, ALL root dep counts are bumped first, then each is
    decremented-and-maybe-enqueued (engine.rs:335-393)."""
    var n = len(graph.nodes)
    if len(roots) == 0 or len(roots) != len(root_grads):
        raise Error("execute_klein: roots/root_grads length mismatch or empty")
    for i in range(len(roots)):
        if roots[i] < 0 or roots[i] >= n:
            raise Error("execute_klein: root node out of range")

    # ── Step 1: dep-count BFS over edges from ALL roots (engine.rs:230-277).
    var dep = List[Int]()
    var seen = List[Bool]()
    var in_queue = List[Bool]()
    for _ in range(n):
        dep.append(0)
        seen.append(False)
        in_queue.append(False)
    var stack = List[Int]()
    for i in range(len(roots)):
        stack.append(roots[i])
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

    # ── Step 2: per-node InputBuffers + seed every root (engine.rs:282-393).
    var buffers = List[InputBuffer]()
    for i in range(n):
        buffers.append(InputBuffer(graph.nodes[i].contrib_counts, root_grads[0].copy()))

    for i in range(len(roots)):
        buffers[roots[i]].add(
            0, buffers[roots[i]].seed_slot(0), root_grads[i].copy(), ctx
        )

    var ready = List[Int]()
    for i in range(len(roots)):
        dep[roots[i]] += 1
    for i in range(len(roots)):
        _dec_and_maybe_enqueue(dep, ready, in_queue, roots[i])

    # ── Step 3: drive the queue (engine.rs:437-570).
    var result = Dict[Int, TArc]()
    var fired = 0
    while len(ready) > 0:
        var nid = _pop_best(ready, graph)
        fired += 1

        var num_in = graph.nodes[nid].num_inputs
        var grads_in = List[TArc]()
        for s in range(num_in):
            if not buffers[nid].any_present(s):
                raise Error(
                    String("execute_klein: node ")
                    + String(nid)
                    + " fired with missing input grad in slot "
                    + String(s)
                )
            grads_in.append(buffers[nid].materialize(s, ctx))

        if graph.nodes[nid].kind == OPK_LEAF:
            var pid = graph.nodes[nid].param_id
            if result.__contains__(pid):
                var old = result[pid]
                var summed = _raw_add(old[], grads_in[0][], ctx)
                result[pid] = TArc(summed^)
            else:
                result[pid] = grads_in[0].copy()
            continue

        var out_grads = apply_klein[H, Dh, S](graph.nodes[nid], grads_in, ctx, scratch)

        # Arity check: one grad per next_edge (engine.rs:509-520).
        var n_edges = len(graph.nodes[nid].edges)
        if len(out_grads) != n_edges:
            raise Error(
                String("execute_klein: apply arity mismatch on node ")
                + String(nid)
                + ": expected "
                + String(n_edges)
                + " got "
                + String(len(out_grads))
            )

        for s in range(n_edges):
            var child = graph.nodes[nid].edges[s].node_idx
            if child < 0:
                continue  # null edge: drop the grad (engine.rs:532-535)
            var slot = graph.nodes[nid].edges[s].input_nr
            var cslot = graph.nodes[nid].edges[s].contrib_slot
            buffers[child].add(slot, cslot, out_grads[s].copy(), ctx)
            _dec_and_maybe_enqueue(dep, ready, in_queue, child)

    if fired != reachable:
        raise Error(
            String("execute_klein: fired=")
            + String(fired)
            + " != reachable="
            + String(reachable)
            + " (dep-count exactness violated)"
        )
    return result^


# ─────────────────────────────────────────────────────────────────────────────
# P7 ideogram4 COARSE block adapter (Stage 1, ideogram4_block_graph.mojo).
# apply_ideogram4 dispatches the single composite kind OPK_IDEOGRAM4_BLOCK by
# reboxing the saved weights/adapters, recomputing the forward, and calling the
# WHOLE ideogram4_block_lora_backward oracle (serenitymojo/models/ideogram4/
# block.mojo) — bit-identical to the hand-chain stack backward by construction.
# execute_ideogram4 is execute_klein's engine VERBATIM with the ideogram4 dispatch
# and comptime block dims. saved=[x,adaln,cos,sin, 13 weights (struct field order),
# 12 lora a/b]; saved_meta=[rank]; scalars=[alpha]; edges(14)=x,adaln,a/b per slot.
from std.memory import ArcPointer as _ArcPtr
from serenitymojo.models.ideogram4.block import (
    ideogram4_block_lora_forward as _i4_fwd,
    ideogram4_block_lora_backward as _i4_bwd,
    Ideogram4BlockWeights as _I4W,
)
from serenitymojo.models.ideogram4.lora_module import LoraAdapter as _I4Lora
from serenitymojo.autograd_v2.node import OPK_IDEOGRAM4_BLOCK, arc_view as _i4_arc_view

comptime _I4LArc = _ArcPtr[_I4Lora]


def _rebox_i4(t: TArc) raises -> Tensor:
    return Tensor(t[].buf.copy(), t[].shape(), t[].dtype())


def apply_ideogram4[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    node: Node, grads_in: List[TArc], ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> List[TArc]:
    if node.kind != OPK_IDEOGRAM4_BLOCK:
        raise Error("apply_ideogram4: unexpected node kind")
    var rank = node.saved_meta[0]
    var alpha = node.scalars[0]
    var w = _I4W(
        _rebox_i4(node.saved[4]), _rebox_i4(node.saved[5]), _rebox_i4(node.saved[6]),
        _rebox_i4(node.saved[7]), _rebox_i4(node.saved[8]), _rebox_i4(node.saved[9]),
        _rebox_i4(node.saved[10]), _rebox_i4(node.saved[11]), _rebox_i4(node.saved[12]),
        _rebox_i4(node.saved[13]), _rebox_i4(node.saved[14]), _rebox_i4(node.saved[15]),
        _rebox_i4(node.saved[16]),
    )
    var loras = List[_I4LArc]()
    for slot in range(6):
        loras.append(_I4LArc(_I4Lora(
            _rebox_i4(node.saved[17 + slot * 2]),
            _rebox_i4(node.saved[17 + slot * 2 + 1]),
            rank, alpha,
        )))
    var rb = _i4_fwd[S, Hidden, Heads, Dh, FF, Adaln](
        node.saved[0][], node.saved[1][], node.saved[2][], node.saved[3][], w, loras, ctx
    )
    var bb = _i4_bwd[S, Hidden, Heads, Dh, FF, Adaln](
        grads_in[0][], rb.acts^, node.saved[2][], node.saved[3][], w, loras, ctx
    )
    var out = List[TArc]()
    out.append(_i4_arc_view(bb.d_x))
    out.append(_i4_arc_view(bb.d_adaln_input))
    for slot in range(6):
        out.append(bb.lora_grads.d_a[slot].copy())
        out.append(bb.lora_grads.d_b[slot].copy())
    return out^


def execute_ideogram4[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    mut graph: Graph, roots: List[Int], root_grads: List[TArc],
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> Dict[Int, TArc]:
    """execute_klein VERBATIM (dep-count BFS, slot-ordered buffers, ready-queue
    order, fired==reachable) with the ideogram4 apply dispatch + comptime dims."""
    var n = len(graph.nodes)
    if len(roots) == 0 or len(roots) != len(root_grads):
        raise Error("execute_ideogram4: roots/root_grads length mismatch or empty")
    for i in range(len(roots)):
        if roots[i] < 0 or roots[i] >= n:
            raise Error("execute_ideogram4: root node out of range")
    var dep = List[Int]()
    var seen = List[Bool]()
    var in_queue = List[Bool]()
    for _ in range(n):
        dep.append(0)
        seen.append(False)
        in_queue.append(False)
    var stack = List[Int]()
    for i in range(len(roots)):
        stack.append(roots[i])
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
                dep[child] += 1
                stack.append(child)
    var buffers = List[InputBuffer]()
    for i in range(n):
        buffers.append(InputBuffer(graph.nodes[i].contrib_counts, root_grads[0].copy()))
    for i in range(len(roots)):
        buffers[roots[i]].add(0, buffers[roots[i]].seed_slot(0), root_grads[i].copy(), ctx)
    var ready = List[Int]()
    for i in range(len(roots)):
        dep[roots[i]] += 1
    for i in range(len(roots)):
        _dec_and_maybe_enqueue(dep, ready, in_queue, roots[i])
    var result = Dict[Int, TArc]()
    var fired = 0
    while len(ready) > 0:
        var nid = _pop_best(ready, graph)
        fired += 1
        var num_in = graph.nodes[nid].num_inputs
        var grads_in = List[TArc]()
        for s in range(num_in):
            if not buffers[nid].any_present(s):
                raise Error(String("execute_ideogram4: node ") + String(nid) + " missing grad slot " + String(s))
            grads_in.append(buffers[nid].materialize(s, ctx))
        if graph.nodes[nid].kind == OPK_LEAF:
            var pid = graph.nodes[nid].param_id
            if result.__contains__(pid):
                var old = result[pid]
                var summed = _raw_add(old[], grads_in[0][], ctx)
                result[pid] = TArc(summed^)
            else:
                result[pid] = grads_in[0].copy()
            continue
        var out_grads = apply_ideogram4[S, Hidden, Heads, Dh, FF, Adaln](
            graph.nodes[nid], grads_in, ctx, scratch
        )
        var n_edges = len(graph.nodes[nid].edges)
        if len(out_grads) != n_edges:
            raise Error(String("execute_ideogram4: apply arity mismatch node ") + String(nid))
        for s in range(n_edges):
            var child = graph.nodes[nid].edges[s].node_idx
            if child < 0:
                continue
            var slot = graph.nodes[nid].edges[s].input_nr
            var cslot = graph.nodes[nid].edges[s].contrib_slot
            buffers[child].add(slot, cslot, out_grads[s].copy(), ctx)
            _dec_and_maybe_enqueue(dep, ready, in_queue, child)
    if fired != reachable:
        raise Error(String("execute_ideogram4: fired=") + String(fired) + " != reachable=" + String(reachable))
    return result^


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4b krea2 COARSE block adapter (krea2_block_graph.mojo).
# apply_krea2 dispatches the single composite kind OPK_KREA2_SINGLE_BLOCK by
# RECOMPUTING the forward from the saved block input x (the conductor's recompute-
# checkpoint discipline) and calling the WHOLE krea2_single_stream_block_lora_backward
# oracle (models/krea2/krea2_block.mojo) — bit-identical to the hand-chain stack
# backward by construction (the gate target; re-derives NO block math). It returns the
# WHOLE Krea2BlockGrads (d_x TArc + 8 HOST-list LoRA pairs), NOT a List[TArc]: the krea2
# LoRA grads are host List[Float32] (the .to_host lives inside the oracle's lora bwd) and
# cannot flow through the engine's TArc edge/Dict machinery, so they ride out-of-band.
#
# execute_krea2_block is execute_ideogram4's engine algorithm (dep-count BFS, InputBuffer
# seed, ready-queue order, fired==reachable) specialized to the krea2 grad shape: the ONE
# composite node feeds its d_x into the x LEAF (the engine genuinely routes the single
# inter-block dependency, so the gate checks the engine path), and the host-list LoRA
# grads are captured from the apply call and returned in the same Krea2BlockGrads struct.
# The bulky weights/adapters/rope/vec structs are threaded as direct params (NOT packed
# in Node.saved) — the same structs the oracle uses, bit-identical.
#
# No StepSlab and no CUDA capture for krea2 in Phase 4b (the slab+capture speed phases
# come after; the conductor keeps the streamed loop's per-block ctx.synchronize seam,
# C10). comptime [HEADS,KVHEADS,HEADDIM] buckets like the trainer; runtime L rides meta.
# ─────────────────────────────────────────────────────────────────────────────
from serenitymojo.autograd_v2.node import OPK_KREA2_SINGLE_BLOCK
from serenitymojo.models.krea2.krea2_block import (
    krea2_single_stream_block_lora as _k2_fwd,
    krea2_single_stream_block_lora_backward as _k2_bwd,
    Krea2BlockWeights as _K2W,
    Krea2BlockLora as _K2L,
    Krea2BlockGrads as _K2Grads,
    Krea2LoraGrad,
)


def apply_krea2[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    node: Node, d_out: TArc,
    vec: Tensor, w: _K2W, lora: _K2L,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
    eps: Float32, real_len: Optional[Int], ctx: DeviceContext,
) raises -> _K2Grads:
    """Recompute the krea2 block forward from the saved input, then call the WHOLE
    krea2_single_stream_block_lora_backward oracle. Returns Krea2BlockGrads (d_x +
    8 host-list LoRA pairs) — bit-identical to the hand-chain conductor's per-block
    (recompute + block backward) pair (krea2_stack.mojo:724-732)."""
    if node.kind != OPK_KREA2_SINGLE_BLOCK:
        raise Error("apply_krea2: unexpected node kind")
    # node.saved[0] is the block input x (the recompute checkpoint).
    var rb = _k2_fwd[L, HEADS, KVHEADS, HEADDIM](
        node.saved[0].copy(), vec, w, lora,
        cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
    )
    var bg = _k2_bwd[L, HEADS, KVHEADS, HEADDIM](
        d_out[], vec, w, lora, rb.saved,
        cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
    )
    return bg^


def execute_krea2_block[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    mut graph: Graph, root: Int, root_grad: TArc,
    vec: Tensor, w: _K2W, lora: _K2L,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
    eps: Float32, real_len: Optional[Int], ctx: DeviceContext,
) raises -> _K2Grads:
    """execute_ideogram4's engine algorithm (dep-count BFS, slot-ordered buffers,
    ready-queue order, fired==reachable invariant) for the krea2 graph: ONE composite
    node + its x leaf. The composite node fires once (the seeded root), produces
    Krea2BlockGrads; its d_x routes along the x edge into the LEAF (engine-driven, the
    gate's d_x path); the host-list LoRA grads are captured and returned. Single root,
    single tracked edge → the InputBuffer/seed/route machinery still runs, but no
    multi-contribution fan-in arises (C15 trivial at graph level)."""
    var n = len(graph.nodes)
    if root < 0 or root >= n:
        raise Error("execute_krea2_block: root out of range")

    # ── Step 1: dep-count BFS over edges from the root (engine.rs:230-277).
    var dep = List[Int]()
    var seen = List[Bool]()
    var in_queue = List[Bool]()
    for _ in range(n):
        dep.append(0)
        seen.append(False)
        in_queue.append(False)
    var stack = List[Int]()
    stack.append(root)
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
                dep[child] += 1
                stack.append(child)

    # ── Step 2: per-node InputBuffers + seed the root (engine.rs:282-393).
    var buffers = List[InputBuffer]()
    for i in range(n):
        buffers.append(InputBuffer(graph.nodes[i].contrib_counts, root_grad.copy()))
    buffers[root].add(0, buffers[root].seed_slot(0), root_grad.copy(), ctx)
    var ready = List[Int]()
    dep[root] += 1
    _dec_and_maybe_enqueue(dep, ready, in_queue, root)

    # ── Step 3: drive the queue (engine.rs:437-570). The composite node's LoRA
    # grads are captured here (the only non-leaf node); the engine routes only its
    # d_x edge.
    var out_grads = _K2Grads(
        root_grad.copy(),
        Krea2LoraGrad(None, None), Krea2LoraGrad(None, None), Krea2LoraGrad(None, None),
        Krea2LoraGrad(None, None), Krea2LoraGrad(None, None), Krea2LoraGrad(None, None),
        Krea2LoraGrad(None, None), Krea2LoraGrad(None, None),
    )
    var captured = False
    var d_x_result = TArc(Tensor(root_grad[].buf.copy(), root_grad[].shape(), root_grad[].dtype()))
    var have_dx = False
    var fired = 0
    while len(ready) > 0:
        var nid = _pop_best(ready, graph)
        fired += 1
        var num_in = graph.nodes[nid].num_inputs
        var grads_in = List[TArc]()
        for s in range(num_in):
            if not buffers[nid].any_present(s):
                raise Error(
                    String("execute_krea2_block: node ") + String(nid)
                    + " missing grad slot " + String(s)
                )
            grads_in.append(buffers[nid].materialize(s, ctx))

        if graph.nodes[nid].kind == OPK_LEAF:
            # The x leaf: the d_x carried here is the engine-routed block-input grad.
            d_x_result = grads_in[0].copy()
            have_dx = True
            continue

        if graph.nodes[nid].kind != OPK_KREA2_SINGLE_BLOCK:
            raise Error(
                String("execute_krea2_block: unexpected node kind ")
                + String(graph.nodes[nid].kind)
            )
        if captured:
            raise Error("execute_krea2_block: more than one composite node fired")
        var bg = apply_krea2[L, HEADS, KVHEADS, HEADDIM](
            graph.nodes[nid], grads_in[0],
            vec, w, lora, cos, sin, cos_q, sin_q, cos_k, sin_k, eps, real_len, ctx,
        )
        # Stash the 8 host-list LoRA grads (out-of-band; not engine leaves).
        out_grads = _K2Grads(
            bg.d_x.copy(),
            bg.wq.copy(), bg.wk.copy(), bg.wv.copy(), bg.gate_w.copy(),
            bg.wo.copy(), bg.mlp_gate_w.copy(), bg.mlp_up_w.copy(), bg.mlp_down_w.copy(),
        )
        captured = True

        # Route the composite node's single d_x edge (one grad per next_edge).
        var n_edges = len(graph.nodes[nid].edges)
        if n_edges != 1:
            raise Error(
                String("execute_krea2_block: composite node must have exactly one edge, got ")
                + String(n_edges)
            )
        var child = graph.nodes[nid].edges[0].node_idx
        if child >= 0:
            var slot = graph.nodes[nid].edges[0].input_nr
            var cslot = graph.nodes[nid].edges[0].contrib_slot
            buffers[child].add(slot, cslot, bg.d_x.copy(), ctx)
            _dec_and_maybe_enqueue(dep, ready, in_queue, child)

    if fired != reachable:
        raise Error(
            String("execute_krea2_block: fired=") + String(fired)
            + " != reachable=" + String(reachable)
            + " (dep-count exactness violated)"
        )
    if not captured:
        raise Error("execute_krea2_block: composite node never fired")
    if not have_dx:
        raise Error("execute_krea2_block: x leaf never sank d_x")
    # The engine-routed d_x (from the leaf) is the authoritative block-input grad.
    return _K2Grads(
        d_x_result.copy(),
        out_grads.wq.copy(), out_grads.wk.copy(), out_grads.wv.copy(), out_grads.gate_w.copy(),
        out_grads.wo.copy(), out_grads.mlp_gate_w.copy(), out_grads.mlp_up_w.copy(),
        out_grads.mlp_down_w.copy(),
    )
