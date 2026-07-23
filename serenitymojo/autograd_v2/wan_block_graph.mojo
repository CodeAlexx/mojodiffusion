# autograd_v2/wan_block_graph.mojo — Wan2.2-A14B T2V per-block LoRA backward
# driven by the graph engine, COARSE (Phase 1; AUTOGRAD_V2_MOJO_DESIGN.md P7
# rollout recipe, the OPK_KREA2_SINGLE_BLOCK / OPK_LTX2V_VIDEO_BLOCK precedent).
#
# The wan trainer stack keeps FULL saved activations per block (host BF16,
# WanSaved) and calls wan22_block_lora_backward on them directly — NO forward
# recompute (wan22_stack_lora.mojo:1029). So this drop-in mirrors the hand-chain
# backward's arg list EXACTLY (d_out_h, mv, w, lora, saved, cos, sin, dim, ffn,
# eps, ctx) and records ONE composite node (OPK_WAN_T2V_BLOCK) whose only tracked
# leaf is the block input x; execute_wan_t2v_block's apply arm calls the WHOLE
# wan22_block_lora_backward oracle — so every internal >=3-way fold (the sa_in
# base/LoRA folds, wan22_block.mojo:1637-1642) stays INSIDE the oracle (C15
# trivially satisfied at graph level; the ONLY tracked edge is x).
#
# Bit-equality vs the hand-chain (C14; wan is MATH-MODE sdpa — sdpa_backward /
# sdpa_backward_rect, deterministic — so a TRUE bit gate, not a variance class):
# the apply arm calls the SAME oracle backward on the SAME saved WanSaved, so the
# grads are bit-identical by construction. The block-input d_x is a pure F32
# host→device→leaf→host round trip (bit-identical). The gate is
# autograd_v2/tests/wan_block_parity.mojo (same-process, NONZERO LoRA B).
#
# The 20 LoRA d_A/d_B pairs + d_context + mod grads are HOST List[Float32] (the
# oracle's KleinLoraGrads/WanBlockGrads construction) and cannot flow through the
# engine's TArc-only edge/Dict machinery, so execute returns the WHOLE
# WanBlockLoraGrads (d_x sunk through the engine leaf; the host lists captured
# out-of-band) — the krea2 pattern (node.mojo OPK_WAN_T2V_BLOCK note).
#
# Conductor (C10): NOT here — the stack loop keeps its loader.await_block/
# prefetch/scatter calls around each per-block graph call (that is the Phase 2
# stack seam, not built yet). NO StepSlab / NO capture in Phase 1 (correctness
# foundation only).
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.ops_record import record_wan_t2v_block
from serenitymojo.autograd_v2.engine import execute_wan_t2v_block
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanModVecs, WanBlockLora, WanSaved, WanBlockLoraGrads,
)


def wan22_block_lora_graph_backward[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    d_out_h: List[Float32], mv: WanModVecs, w: WanBlockWeights,
    lora: WanBlockLora, saved: WanSaved, cos: Tensor, sin: Tensor,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
) raises -> WanBlockLoraGrads:
    """Graph-engine replacement for the trainer-loop per-block backward
    (wan22_block_lora_backward, models/wan22/wan22_block.mojo:1456): record ONE
    composite node from the block input x (single tracked leaf), execute the
    backward seeded with d_out, return the SAME WanBlockLoraGrads the hand-chain
    backward returns (base d_x/d_context/mod/frozen grads + 20 host-list LoRA
    pairs). Drop-in: identical arg list + identical returned grads."""
    var g = Graph()

    # block input x: the ONE tracked leaf (its accumulated grad = the returned
    # base.d_x). The wan oracle backward takes x only via saved.x, so the leaf
    # needs no real device input — a fresh-id [1] carrier suffices (its buffer is
    # never read; the engine seeds d_out from the caller, and the composite routes
    # the oracle's d_x into this leaf). The klein_block_graph.mojo:86 zero-copy
    # re-box idiom reduced to an id stamp.
    var x_id = g.fresh_tensor_id()
    _ = g.leaf(x_id)
    var carrier = List[Float32]()
    carrier.append(0.0)
    var x_t = Tensor.from_host(carrier^, [1], STDtype.F32, ctx)
    x_t.set_id(x_id)
    var x = TArc(x_t^)

    var out = record_wan_t2v_block(g, x, x_id, ctx)
    var root = g.node_of_tensor[out[].id]

    var d_out_dev = TArc(Tensor.from_host(d_out_h.copy(), [S, dim], STDtype.F32, ctx))
    var grads = execute_wan_t2v_block[H, Dh, S, TXT](
        g, root, d_out_dev, mv, w, lora, saved, cos, sin, dim, ffn, eps, ctx,
    )
    return grads^
