# autograd_v2/ideogram4_block_graph.mojo — ideogram4 ADAPTER onto the v2 engine
# (P7 rollout, klein_block_graph.mojo precedent). Stage 0 (block migrated into
# serenitymojo/models/ideogram4) is DONE, so the engine can call the block backward.
#
# STAGE 1 (this file, COARSE): ONE composite node OPK_IDEOGRAM4_BLOCK per block;
# apply_ideogram4 reboxes the saved weights/adapters, recomputes the forward, and
# calls the WHOLE ideogram4_block_lora_backward oracle — bit-IDENTICAL to the
# hand-chain stack backward by construction (the gate target). No speedup yet (one
# coarse node carries no per-op slab/capture surface; engine alone is +2%). The
# host-overhead win is STAGE 2 (fine-grained kinds + slab + capture); SDPA-agnostic
# so ideogram4's Dh=256 (flash-blocked) does not block it.
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.autograd_v2.node import TArc, arc_view
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute_ideogram4
from serenitymojo.autograd_v2.ops_record import record_ideogram4_block
from serenitymojo.models.ideogram4.block import (
    Ideogram4BlockWeights,
    Ideogram4BlockBwd,
    Ideogram4BlockLoraGrads,
    LArc,
)


def _tracked_leaf(mut g: Graph, x: TArc) raises -> TArc:
    """Tensor as a tracked leaf whose accumulated grad is the returned d_x.
    Zero-copy re-box so the id stamp never mutates the shared saved arc
    (klein_block_graph.mojo:_tracked_leaf_input idiom)."""
    var x_t = Tensor(x[].buf.copy(), x[].shape(), x[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    _ = g.leaf(x_t.id)
    return TArc(x_t^)


def _rebox(t: TArc) raises -> Tensor:
    return Tensor(t[].buf.copy(), t[].shape(), t[].dtype())


def ideogram4_block_graph_backward[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    d_out_t: TArc,
    x_in: TArc,
    adaln_in: TArc,
    cosf: Tensor, sinf: Tensor,
    w: Ideogram4BlockWeights,
    loras: List[LArc],
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> Ideogram4BlockBwd:
    """STAGE-1 graph backward for one ideogram4 block. Returns the SAME struct the
    hand-chain ideogram4_block_lora_backward returns (d_x, d_adaln_input, the 6
    adapter d_a/d_b). Coarse single node => bit-identical to the oracle."""
    var scratch_mark = scratch.mark()
    var g = Graph()

    var x_g = _tracked_leaf(g, x_in)
    var adaln_g = _tracked_leaf(g, adaln_in)
    var x_id = x_g[].id
    var adaln_id = adaln_g[].id

    var a_ids = List[Int]()
    var b_ids = List[Int]()
    for _ in range(6):
        var aid = g.fresh_tensor_id()
        _ = g.leaf(aid)
        a_ids.append(aid)
        var bid = g.fresh_tensor_id()
        _ = g.leaf(bid)
        b_ids.append(bid)

    var cos_arc = arc_view(cosf)
    var sin_arc = arc_view(sinf)

    var out_g = record_ideogram4_block[S, Hidden, Heads, Dh, FF, Adaln](
        g, x_g, adaln_g, cos_arc, sin_arc, w, loras,
        x_id, adaln_id, a_ids, b_ids, ctx,
    )

    var roots = List[Int]()
    roots.append(g.node_of_tensor[out_g[].id])
    var root_grads = List[TArc]()
    root_grads.append(d_out_t.copy())
    var grads = execute_ideogram4[S, Hidden, Heads, Dh, FF, Adaln](
        g, roots, root_grads, ctx, scratch
    )

    var d_x = _rebox(grads[x_id])
    var d_adaln = _rebox(grads[adaln_id])
    var d_a = List[TArc]()
    var d_b = List[TArc]()
    for slot in range(6):
        d_a.append(grads[a_ids[slot]].copy())
        d_b.append(grads[b_ids[slot]].copy())

    scratch.rewind(scratch_mark)
    return Ideogram4BlockBwd(d_x^, d_adaln^, Ideogram4BlockLoraGrads(d_a^, d_b^))
