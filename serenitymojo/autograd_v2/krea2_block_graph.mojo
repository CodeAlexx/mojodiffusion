# autograd_v2/krea2_block_graph.mojo — krea2 ADAPTER onto the v2 engine
# (Phase 4b, ideogram4_block_graph.mojo precedent — the COARSE single-composite-node
# path the autograd-v2 skill recommends when the oracle has big fused helpers with
# internal >=3-way folds).
#
# COARSE: ONE composite node OPK_KREA2_SINGLE_BLOCK per block; apply_krea2 RECOMPUTES
# the forward from the saved block input (krea2_single_stream_block_lora) and calls the
# WHOLE krea2_single_stream_block_lora_backward oracle (models/krea2/krea2_block.mojo)
# — bit-IDENTICAL to the hand-chain stack-backward per-block (recompute + block backward)
# pair (krea2_stack.mojo:724-732) by construction (the gate target). Re-derives NO block
# math; every internal >=3-way fold (e.g. d_xm = add(add(bw_q,bw_k),add(bw_v,bw_g)),
# krea2_block.mojo:677) stays INSIDE the oracle, so C15 is trivially satisfied at graph
# level — the block input x is the ONLY tracked edge, its single contribution needs no
# fan-in fold.
#
# The 8 LoRA grad pairs are HOST List[Float32] by the oracle's construction (the
# .to_host lives inside klein_lora_bwd_device_resident_unfused), so they CANNOT be
# engine TArc leaves; execute_krea2_block returns the WHOLE Krea2BlockGrads (d_x sunk
# through the engine x LEAF; the host LoRA lists captured out-of-band). The wrapper
# returns that struct unchanged — the conductor (krea2_stack_lora_backward_graph)
# scatters the 8 host pairs exactly as the streamed conductor does.
#
# Conductor (C10): NOT here — the stack loop keeps the streamed conductor's per-block
# resident-dequant / H2D stream + the per-block ctx.synchronize() seam around each
# per-block graph call (krea2_stack_lora_backward_graph), the SAME seam the hand-chain
# uses. NO StepSlab and NO CUDA capture for krea2 in Phase 4b (scope decision; the
# slab+capture speed phases come after — the engine alone is ~+2%).
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute_krea2_block
from serenitymojo.autograd_v2.ops_record import record_krea2_single_block
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights,
    Krea2BlockLora,
    Krea2BlockGrads,
)


def _tracked_leaf(mut g: Graph, x: TArc) raises -> TArc:
    """Block input as a tracked leaf whose accumulated grad is the returned d_x.
    Zero-copy re-box so the id stamp never mutates the shared saved arc
    (ideogram4_block_graph.mojo:_tracked_leaf idiom)."""
    var x_t = Tensor(x[].buf.copy(), x[].shape(), x[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    _ = g.leaf(x_t.id)
    return TArc(x_t^)


def krea2_single_stream_block_graph_backward[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out_t: TArc,
    x_in: TArc, vec: Tensor,
    w: Krea2BlockWeights, lora: Krea2BlockLora,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockGrads:
    """Graph-engine replacement for the streamed conductor's per-block pair
    (recompute krea2_single_stream_block_lora + krea2_single_stream_block_lora_backward,
    krea2_stack.mojo:724-732): record the forward from the saved block INPUT (single
    tracked leaf = x), execute the backward from the block output (d_out seed), return
    the SAME Krea2BlockGrads the hand-chain block backward returns (d_x + the 8 host-list
    LoRA pairs). Coarse single node => bit-identical to the oracle.

    `scratch` is threaded for signature parity with the streamed conductor's seam; the
    coarse arm calls the oracle backward whole (the oracle manages its own transient
    device allocations), so the mark/rewind is a no-op safety bracket here."""
    var scratch_mark = scratch.mark()
    var g = Graph()

    var x_g = _tracked_leaf(g, x_in)
    var x_id = x_g[].id

    var out_g = record_krea2_single_block[L, HEADS, KVHEADS, HEADDIM](
        g, x_g, vec, w, lora, cos, sin, cos_q, sin_q, cos_k, sin_k,
        eps, x_id, ctx, real_len,
    )

    var root = g.node_of_tensor[out_g[].id]
    var bg = execute_krea2_block[L, HEADS, KVHEADS, HEADDIM](
        g, root, d_out_t.copy(),
        vec, w, lora, cos, sin, cos_q, sin_q, cos_k, sin_k,
        eps, real_len, ctx,
    )

    scratch.rewind(scratch_mark)
    return bg^
