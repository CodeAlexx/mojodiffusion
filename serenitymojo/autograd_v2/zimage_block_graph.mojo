# autograd_v2/zimage_block_graph.mojo - Phase P3 (AUTOGRAD_V2_MOJO_DESIGN.md):
# the zimage B=1 block backward driven by the graph engine.
#
# Recompute-style checkpoint (flame checkpoint.rs inline-mini-execute): the
# stack backward _v3 hands each block its saved INPUT x; this function re-runs
# the block forward THROUGH the P2 record_* wrappers (exact op order of
# zimage_block_lora_forward_device_tensor_batch at B=1 - same kernels, same
# sequence, models/zimage/lora_block.mojo:1619-1699), building a per-block
# Graph whose leaves are the 7 adapters' A/B (14 leaves) plus the block input
# x (its accumulated grad IS the returned d_x); then engine.execute drives the
# backward from the block output seeded with d_out.
#
# Bit-equality vs the hand-chain oracle
# (zimage_block_lora_backward_device_tensors_batch, lora_block.mojo:1702-1793,
# contract C14):
#  * every apply arm calls the SAME ops/*_backward function the oracle calls
#    on the same operands (engine.mojo P2 arms);
#  * the only 3-way fan-in is xn1s <- {q,k,v}: recording q,k,v in forward
#    order assigns C15 contrib slots 0,1,2, and the InputBuffer's slot-ordered
#    left fold reproduces the oracle's d_xn1s = add(add(dq,dk),dv) exactly;
#  * the 2-way fan-ins (h <- {fn1-norm, gate2}, x <- {n1-norm, gate1}) land
#    in slot order norm-first - the REVERSED operand order of the oracle's
#    add(grg.d_x, d_norm) - which is bit-equal because IEEE addition is
#    commutative (associativity is what fails, and 2-way folds have none);
#  * gates' tanh is frozen: tanh_op(mv.gate_*) is computed OUTSIDE the graph
#    (no node) and fed to record_residual_gate as the saved gate_t, whose
#    backward arm is gate_residual_backward_dxdy (gate frozen) - the oracle's
#    exact call.
#
# Frozen base weights (wq/wk/wv/wo/w1/w3/w2, norms, modvecs) are untracked
# (id 0) -> null edges, contract C7. B=1 only in P3 (ROWS == S).
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.ops.unary import tanh_op
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.lora_block import (
    ZImageModVecsDevice,
    ZImageBlockLoraDevice,
    ZImageBlockLoraTensorBackward,
)
from serenitymojo.autograd_v2.node import TArc, arc_view
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute
from serenitymojo.autograd_v2.ops_record import (
    record_proj_lora,
    record_rms_norm_dx,
    record_modulate,
    record_rope,
    record_sdpa,
    record_swiglu,
    record_residual_gate,
    record_reshape,
)


def zimage_block_lora_graph_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: Tensor,
    w: ZImageBlockWeights, mv: ZImageModVecsDevice, lora: ZImageBlockLoraDevice,
    x_in: TArc,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockLoraTensorBackward:
    """Graph-engine replacement for the _v2 per-block recompute+hand-chain
    pair: record the forward, execute the backward, return the SAME struct the
    hand-chain returns (d_x + d_a/d_b in slot order q,k,v,o,w1,w3,w2)."""
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var g = Graph()

    # Block input x: a tracked leaf whose accumulated grad is the returned
    # d_x. Zero-copy re-box so the id stamp never mutates the shared saved arc.
    var x_t = Tensor(x_in[].buf.copy(), x_in[].shape(), x_in[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    var x_id = x_t.id
    _ = g.leaf(x_id)
    var x = TArc(x_t^)

    # 14 adapter leaves, canonical slot order q,k,v,o,w1,w3,w2 (recording in
    # this order IS the C15 slot assignment; ids are graph-local).
    var a_ids = List[Int]()
    var b_ids = List[Int]()
    for _ in range(7):
        a_ids.append(g.fresh_tensor_id())
        b_ids.append(g.fresh_tensor_id())

    # ── forward, recorded (op-for-op zimage_block_lora_forward_device_tensor_
    # batch at B=1, lora_block.mojo:1631-1687) ────────────────────────────────
    var xn1 = record_rms_norm_dx(g, x, w.n1, eps, ctx)
    var xn1s = record_modulate(g, xn1, mv.scale_msa, mv.zeros, 0, ctx)

    var q = record_proj_lora(g, xn1s, w.wq, lora.to_q, a_ids[0], b_ids[0], S, D, D, ctx)
    var k = record_proj_lora(g, xn1s, w.wk, lora.to_k, a_ids[1], b_ids[1], S, D, D, ctx)
    var v_flat = record_proj_lora(g, xn1s, w.wv, lora.to_v, a_ids[2], b_ids[2], S, D, D, ctx)

    var shape4: List[Int] = [1, S, H, Dh]
    var q_pre = record_reshape(g, q, shape4.copy(), ctx)
    var k_pre = record_reshape(g, k, shape4.copy(), ctx)
    var v = record_reshape(g, v_flat, shape4.copy(), ctx)

    # per-head rms_norm ([1,S,H,Dh] with [Dh] weight - same rms_norm /
    # rms_norm_backward_dx calls the oracle makes, lora_block.mojo:1649-1650 /
    # 1752-1753; the wrapper is shape-agnostic).
    var q_rms = record_rms_norm_dx(g, q_pre, w.q_norm, eps, ctx)
    var k_rms = record_rms_norm_dx(g, k_pre, w.k_norm, eps, ctx)

    var cos_arc = arc_view(cos)
    var sin_arc = arc_view(sin)
    var q_rope = record_rope(g, q_rms, cos_arc, sin_arc, ctx)
    var k_rope = record_rope(g, k_rms, cos_arc, sin_arc, ctx)

    var att = record_sdpa[1, S, H, Dh](g, q_rope, k_rope, v, scale, ctx)
    var flat_shape: List[Int] = [S, D]
    var att_flat = record_reshape(g, att, flat_shape.copy(), ctx)
    var att_o = record_proj_lora(
        g, att_flat, w.wo, lora.to_out, a_ids[3], b_ids[3], S, D, D, ctx
    )

    var attn_n2 = record_rms_norm_dx(g, att_o, w.n2, eps, ctx)
    # Frozen gate: tanh computed OUTSIDE the graph (no node); residual_gate's
    # backward uses the saved gate_t with the gate frozen (dxdy arm).
    var gate_msa_t = TArc(tanh_op(mv.gate_msa[], ctx))
    var h = record_residual_gate(g, x, gate_msa_t, attn_n2, ctx)

    var xfn1 = record_rms_norm_dx(g, h, w.fn1, eps, ctx)
    var xfn1s = record_modulate(g, xfn1, mv.scale_mlp, mv.zeros, 0, ctx)

    var g_pre = record_proj_lora(g, xfn1s, w.w1, lora.w1, a_ids[4], b_ids[4], S, D, F, ctx)
    var u = record_proj_lora(g, xfn1s, w.w3, lora.w3, a_ids[5], b_ids[5], S, D, F, ctx)
    var act = record_swiglu(g, g_pre, u, ctx)
    var ff = record_proj_lora(g, act, w.w2, lora.w2, a_ids[6], b_ids[6], S, F, D, ctx)

    var ff_n2 = record_rms_norm_dx(g, ff, w.fn2, eps, ctx)
    var gate_mlp_t = TArc(tanh_op(mv.gate_mlp[], ctx))
    var result = record_residual_gate(g, h, gate_mlp_t, ff_n2, ctx)

    # ── engine backward from the block output, seeded with d_out ────────────
    var root_idx = g.node_of_tensor[result[].id]
    var grads = execute(g, root_idx, arc_view(d_out), ctx)

    var d_a_slots = List[TArc]()
    var d_b_slots = List[TArc]()
    for s in range(7):
        d_a_slots.append(grads[a_ids[s]].copy())
        d_b_slots.append(grads[b_ids[s]].copy())
    return ZImageBlockLoraTensorBackward(grads[x_id].copy(), d_a_slots^, d_b_slots^)
