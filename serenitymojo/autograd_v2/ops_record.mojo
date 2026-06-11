# autograd_v2/ops_record.mojo - record_* wrappers for the zimage DiT op set
# (Phase P2 of AUTOGRAD_V2_MOJO_DESIGN.md; flame ops/ + dispatch.rs).
#
# Each wrapper:
#   1. runs the EXACT forward op the hand-chain forward calls
#      (zimage_block_lora_forward_device_tensor_batch,
#       models/zimage/lora_block.mojo:1619-1699 - same functions, same order);
#   2. saves exactly the tensors the backward arm needs (mirrors what
#      ZImageBlockSaved keeps - TArc refcount copies, never clones);
#   3. records the node with edges in the hand-chain fold order (C15:
#      Graph.record assigns contrib_slots in registration order).
#
# Backward REUSE (contract: never reimplement math): the apply arms in
# engine.mojo call ops/*_backward directly. Two helpers that the arms need
# live HERE:
#   * proj_lora_backward - a REPLICA of the private
#     _proj_bwd_with_lora_device_tensors (models/zimage/lora_block.mojo:
#     568-577; private to that file, so replicated verbatim per the P2
#     instruction instead of editing the source file). Its two callees
#     (linear_backward_dx, zimage_lora_bwd_device_resident_tensors) are the
#     public parity-proven originals.
#   * sdpa_backward_dispatch - sdpa_backward is comptime-[B,S,H,Dh]
#     specialized; the engine is shape-agnostic (design doc hazard list), so
#     this table maps the node's runtime saved_meta dims onto the comptime
#     buckets the trainers use. Unknown bucket -> raise (fail loud).
#
# Wrapper convention: tensor inputs/outputs are TArc (boxed) so saves are
# refcount bumps; autograd ids are stamped on the boxed Tensor before boxing.
# Leaf edges: pass the param tensor id (>0 tracked; the wrapper get-or-creates
# the OPK_LEAF accumulator); id 0 = frozen -> null edge (contract C7).

from std.gpu.host import DeviceContext
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.ops.linear import linear, linear_slab
from serenitymojo.ops.norm import rms_norm, rms_norm_slab
from serenitymojo.ops.elementwise import (
    modulate, residual_gate, modulate_slab, residual_gate_slab,
)
from serenitymojo.ops.rope import rope_interleaved, rope_interleaved_slab
from serenitymojo.ops.attention import sdpa_nomask, sdpa_nomask_slab
from serenitymojo.ops.activations import swiglu, swiglu_slab
from serenitymojo.ops.tensor_algebra import add, add_slab
from serenitymojo.ops.linalg_backward import (
    linear_backward_dx, linear_backward_dx_slab,
)
from serenitymojo.ops.attention_backward import sdpa_backward, sdpa_backward_slab
from serenitymojo.models.zimage.lora_block import (
    ZImageLoraAdapterDevice,
    zimage_lora_apply_device,
    zimage_lora_bwd_device_resident_tensors,
    zimage_lora_apply_device_slab,
    zimage_lora_bwd_device_resident_tensors_slab,
)
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.autograd_v2.node import (
    Edge,
    TArc,
    arc_view,
    OPK_ADD,
    OPK_PROJ_LORA,
    OPK_RMS_NORM_DX,
    OPK_MODULATE,
    OPK_ROPE,
    OPK_SDPA,
    OPK_SWIGLU,
    OPK_RESIDUAL_GATE_DXDY,
    OPK_RESHAPE,
)
from serenitymojo.autograd_v2.graph import Graph


# ─────────────────────────────────────────────────────────────────────────────
# Backward helpers used by the engine's apply arms.
# ─────────────────────────────────────────────────────────────────────────────


struct ProjLoraGrads(Copyable, Movable):
    """proj+LoRA backward outputs: d_x = base d_x + LoRA d_x (summed, the
    hand-chain's lb_*.d_x), d_a [rank,in], d_b [out,rank]."""

    var d_x: TArc
    var d_a: TArc
    var d_b: TArc

    def __init__(out self, var d_x: TArc, var d_a: TArc, var d_b: TArc):
        self.d_x = d_x^
        self.d_a = d_a^
        self.d_b = d_b^


def proj_lora_backward(
    d_y: Tensor, x_in: Tensor, w: Tensor,
    lo: ZImageLoraAdapterDevice,
    M: Int, in_f: Int, out_f: Int,
    ctx: DeviceContext,
) raises -> ProjLoraGrads:
    """REPLICA of _proj_bwd_with_lora_device_tensors
    (models/zimage/lora_block.mojo:568-577) - private to lora_block.mojo, so
    its exact call sequence is reproduced here (P2 instruction: no edits
    outside autograd_v2/). Same callees, same order, same fold."""
    var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    var lg = zimage_lora_bwd_device_resident_tensors(d_y, x_in, lo, M, ctx)
    var summed = add(base_dx^, lg.d_x[], ctx)
    return ProjLoraGrads(TArc(summed^), lg.d_a.copy(), lg.d_b.copy())


struct SdpaGradArcs(Copyable, Movable):
    """sdpa_backward outputs re-boxed as TArc (SdpaGrads holds plain Tensor
    fields, which Mojo cannot partially move out of; the re-box is the
    zero-copy arc_view of the same device buffers)."""

    var d_q: TArc
    var d_k: TArc
    var d_v: TArc

    def __init__(out self, var d_q: TArc, var d_k: TArc, var d_v: TArc):
        self.d_q = d_q^
        self.d_k = d_k^
        self.d_v = d_v^


def sdpa_backward_dispatch(
    q: Tensor, k: Tensor, v: Tensor, d_out: Tensor, scale: Float32,
    B: Int, S: Int, H: Int, Dh: Int,
    ctx: DeviceContext,
) raises -> SdpaGradArcs:
    """Runtime-dims -> comptime-bucket dispatch for sdpa_backward[B,S,H,Dh]
    (ops/attention_backward.mojo:412). The engine stores node dims as runtime
    ints (shape-agnostic per the design-doc hazard list); only this table is
    comptime-specialized. Buckets = the zimage trainer's B1 sequence lengths
    (S=1248 [72x56/224 + 64x64/224], S=1280 [72x56/256, 88x48/224, 64x64/256],
    S=1312 [88x48/256] - P3 covers every _train_one_step_bucket instantiation;
    the sdpa_backward instantiations already exist via the hand-chain) plus
    the reduced S=320 test bucket. Unknown bucket raises (fail loud, add the
    bucket when a trainer needs it)."""
    if B == 1 and S == 1248 and H == 30 and Dh == 128:
        var sb = sdpa_backward[1, 1248, 30, 128](q, k, v, d_out, scale, ctx)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 1 and S == 1280 and H == 30 and Dh == 128:
        var sb = sdpa_backward[1, 1280, 30, 128](q, k, v, d_out, scale, ctx)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 1 and S == 1312 and H == 30 and Dh == 128:
        var sb = sdpa_backward[1, 1312, 30, 128](q, k, v, d_out, scale, ctx)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 1 and S == 320 and H == 30 and Dh == 128:
        var sb = sdpa_backward[1, 320, 30, 128](q, k, v, d_out, scale, ctx)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    raise Error(
        String("sdpa_backward_dispatch: no comptime bucket for (B,S,H,Dh)=(")
        + String(B) + "," + String(S) + "," + String(H) + "," + String(Dh)
        + ")"
    )


# ─────────────────────────────────────────────────────────────────────────────
# record_* wrappers (forward op + node recording).
# ─────────────────────────────────────────────────────────────────────────────


def _leaf_edge(mut g: Graph, param_id: Int) raises -> Edge:
    """Gradient edge for a parameter id: get-or-create the OPK_LEAF
    accumulator when tracked (>0); null edge when frozen (0) - contract C7."""
    if param_id > 0:
        _ = g.leaf(param_id)
    return g.edge_for(param_id)


def record_proj_lora(
    mut g: Graph, x: TArc, w: TArc, lo: ZImageLoraAdapterDevice,
    a_param_id: Int, b_param_id: Int,
    M: Int, in_f: Int, out_f: Int,
    ctx: DeviceContext,
) raises -> TArc:
    """y = linear(x, W_frozen) + scale*(x@Aᵀ)@Bᵀ - the hand-chain projection
    (lora_block.mojo:1634-1643: linear + zimage_lora_apply_device).
    Edges: [x, A_leaf, B_leaf]; W is frozen (no edge - the base d_w is never
    materialized, lora_block.mojo:512-513 note). saved: x (LoRA + base d_x
    input), W (base d_x), A/B (LoRA chain). meta: [M, in_f, out_f, rank];
    scalars: [scale]."""
    var nb = Optional[Tensor](None)
    var base = linear(x[], w[], nb^, ctx)
    var y = zimage_lora_apply_device(base^, x[], lo, M, ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    edges.append(_leaf_edge(g, a_param_id))
    edges.append(_leaf_edge(g, b_param_id))
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(w.copy())
    saved.append(lo.a.copy())
    saved.append(lo.b.copy())
    var meta: List[Int] = [M, in_f, out_f, lo.rank]
    var scalars: List[Float32] = [lo.scale]
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_PROJ_LORA, edges^, saved^, meta^, scalars^, oids)
    return TArc(y^)


def record_rms_norm_dx(
    mut g: Graph, x: TArc, weight: TArc, eps: Float32, ctx: DeviceContext
) raises -> TArc:
    """y = rms_norm(x, weight, eps) with FROZEN weight (dx-only backward arm,
    rms_norm_backward_dx - the hand-chain's frozen-norm call,
    lora_block.mojo:1717/1735/1740/1752-1753/1773)."""
    var y = rms_norm(x[], weight[], eps, ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(weight.copy())
    var scalars: List[Float32] = [eps]
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_RMS_NORM_DX, edges^, saved^, List[Int](), scalars^, oids)
    return TArc(y^)


def record_modulate(
    mut g: Graph, x: TArc, scale: TArc, shift: TArc,
    scale_param_id: Int, ctx: DeviceContext
) raises -> TArc:
    """y = modulate(x, scale, shift) = (1+scale)*x + shift
    (lora_block.mojo:1632/1670). scale_param_id == 0: frozen adaLN vec (the
    block path - modulate_backward(..., compute_param_grads=False),
    lora_block.mojo:1732-1734/1770-1772) -> null scale edge, d_scale dropped.
    scale_param_id > 0: trained scale (final layer) -> leaf edge + real
    d_scale. shift never needs a grad (o is linear in shift; zimage's shift is
    the zeros vec)."""
    var y = modulate(x[], scale[], shift[], ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    edges.append(_leaf_edge(g, scale_param_id))
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(scale.copy())
    var want_param = 0
    if scale_param_id > 0:
        want_param = 1
    var meta: List[Int] = [want_param]
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_MODULATE, edges^, saved^, meta^, List[Float32](), oids)
    return TArc(y^)


def record_rope(
    mut g: Graph, x: TArc, cos: TArc, sin: TArc, ctx: DeviceContext
) raises -> TArc:
    """y = rope_interleaved(x, cos, sin) (lora_block.mojo:1651-1652).
    cos/sin are frozen precomputed tables; backward is
    rope_backward(g, cos, sin, True) (lora_block.mojo:1749-1750)."""
    var y = rope_interleaved(x[], cos[], sin[], ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    var saved = List[TArc]()
    saved.append(cos.copy())
    saved.append(sin.copy())
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_ROPE, edges^, saved^, List[Int](), List[Float32](), oids)
    return TArc(y^)


def record_sdpa[
    B: Int, S: Int, H: Int, Dh: Int
](
    mut g: Graph, q: TArc, k: TArc, v: TArc, scale: Float32, ctx: DeviceContext
) raises -> TArc:
    """att = sdpa_nomask[B,S,H,Dh](q, k, v, scale) (lora_block.mojo:1654).
    saved q_rope/k_rope/v exactly as ZImageBlockSaved keeps them
    (lora_block.mojo:1691-1692); backward = sdpa_backward via the comptime
    bucket dispatch; 3 output grads d_q/d_k/d_v routed by edge order."""
    var y = sdpa_nomask[B, S, H, Dh](q[], k[], v[], scale, ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(q[].id))
    edges.append(g.edge_for(k[].id))
    edges.append(g.edge_for(v[].id))
    var saved = List[TArc]()
    saved.append(q.copy())
    saved.append(k.copy())
    saved.append(v.copy())
    var meta: List[Int] = [B, S, H, Dh]
    var scalars: List[Float32] = [scale]
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_SDPA, edges^, saved^, meta^, scalars^, oids)
    return TArc(y^)


def record_swiglu(
    mut g: Graph, gate: TArc, up: TArc, ctx: DeviceContext
) raises -> TArc:
    """act = swiglu(g_pre, u) (lora_block.mojo:1679); backward =
    swiglu_backward(g, g_pre, u) -> d_gate, d_up (lora_block.mojo:1722)."""
    var y = swiglu(gate[], up[], ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(gate[].id))
    edges.append(g.edge_for(up[].id))
    var saved = List[TArc]()
    saved.append(gate.copy())
    saved.append(up.copy())
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_SWIGLU, edges^, saved^, List[Int](), List[Float32](), oids)
    return TArc(y^)


def record_residual_gate(
    mut g: Graph, x: TArc, gate_t: TArc, y_in: TArc, ctx: DeviceContext
) raises -> TArc:
    """out = x + gate_t*y where gate_t = tanh(gate vec) computed by the caller
    (the hand-chain computes tanh_op separately and feeds residual_gate -
    lora_block.mojo:1666-1667/1686-1687). gate is frozen (null, no d_g) -
    backward = gate_residual_backward_dxdy -> d_x = g, d_y = g*gate_t
    (lora_block.mojo:1715/1738). Edges: [x, y]; saved: [gate_t]."""
    var y = residual_gate(x[], gate_t[], y_in[], ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    edges.append(g.edge_for(y_in[].id))
    var saved = List[TArc]()
    saved.append(gate_t.copy())
    var oids: List[Int] = [y.id]
    _ = g.record(
        OPK_RESIDUAL_GATE_DXDY, edges^, saved^, List[Int](), List[Float32](), oids
    )
    return TArc(y^)


def record_reshape(
    mut g: Graph, x: TArc, var new_shape: List[Int], ctx: DeviceContext
) raises -> TArc:
    """Metadata-only reshape (the hand-chain's reshape_owned/reshape_in_place,
    lora_block.mojo:1645-1647/1655/1745/1755-1757) - ZERO kernels. The forward
    output is an arc_view sharing x's device buffer with the new shape (the
    recorded x arc keeps its own shape). Backward reshapes the grad back to
    x's shape the same way; saved_meta = x's shape dims."""
    var xshape = x[].shape()
    var n = 1
    for i in range(len(new_shape)):
        n *= new_shape[i]
    if n != x[].numel():
        raise Error("record_reshape: numel mismatch")
    # Zero-copy view: fresh Tensor struct sharing x's device buffer (id stamped
    # before boxing - no mutation through an ArcPointer deref).
    var y = Tensor(x[].buf.copy(), new_shape^, x[].dtype())
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    var meta = List[Int]()
    for i in range(len(xshape)):
        meta.append(xshape[i])
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_RESHAPE, edges^, List[TArc](), meta^, List[Float32](), oids)
    return TArc(y^)


def record_add(
    mut g: Graph, a: TArc, b: TArc, ctx: DeviceContext
) raises -> TArc:
    """y = add(a, b) (ops.tensor_algebra.add - the hand-chain's residual /
    fan-in folds). OPK_ADD's arm routes the incoming grad to both inputs;
    saves nothing."""
    var y = add(a[], b[], ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(a[].id))
    edges.append(g.edge_for(b[].id))
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_ADD, edges^, List[TArc](), List[Int](), List[Float32](), oids)
    return TArc(y^)


# ─────────────────────────────────────────────────────────────────────────────
# StepSlab variants (Phase P4, contract C8): byte-identical recording — same
# ops, same edges/saved/meta/scalars, same C15 slot assignment; ONLY the
# forward op's allocation source changes (each runs through its _slab
# sibling). record_reshape needs NO slab variant: it is metadata-only (zero
# kernels, zero allocations).
# ─────────────────────────────────────────────────────────────────────────────


def proj_lora_backward_slab(
    d_y: Tensor, x_in: Tensor, w: Tensor,
    lo: ZImageLoraAdapterDevice,
    M: Int, in_f: Int, out_f: Int,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> ProjLoraGrads:
    """StepSlab variant of `proj_lora_backward` (this file :85) — same callees
    in the same order/fold, routed to their _slab siblings."""
    var base_dx = linear_backward_dx_slab(d_y, w, M, in_f, out_f, ctx, slab)
    var lg = zimage_lora_bwd_device_resident_tensors_slab(d_y, x_in, lo, M, ctx, slab)
    var summed = add_slab(base_dx^, lg.d_x[], ctx, slab)
    return ProjLoraGrads(TArc(summed^), lg.d_a.copy(), lg.d_b.copy())


def sdpa_backward_dispatch_slab(
    q: Tensor, k: Tensor, v: Tensor, d_out: Tensor, scale: Float32,
    B: Int, S: Int, H: Int, Dh: Int,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> SdpaGradArcs:
    """StepSlab variant of `sdpa_backward_dispatch` (this file :116) — same
    comptime buckets, routed to sdpa_backward_slab."""
    if B == 1 and S == 1248 and H == 30 and Dh == 128:
        var sb = sdpa_backward_slab[1, 1248, 30, 128](q, k, v, d_out, scale, ctx, slab)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 1 and S == 1280 and H == 30 and Dh == 128:
        var sb = sdpa_backward_slab[1, 1280, 30, 128](q, k, v, d_out, scale, ctx, slab)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 1 and S == 1312 and H == 30 and Dh == 128:
        var sb = sdpa_backward_slab[1, 1312, 30, 128](q, k, v, d_out, scale, ctx, slab)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 1 and S == 320 and H == 30 and Dh == 128:
        var sb = sdpa_backward_slab[1, 320, 30, 128](q, k, v, d_out, scale, ctx, slab)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    raise Error(
        String("sdpa_backward_dispatch_slab: no comptime bucket for (B,S,H,Dh)=(")
        + String(B) + "," + String(S) + "," + String(H) + "," + String(Dh)
        + ")"
    )


def record_proj_lora_slab(
    mut g: Graph, x: TArc, w: TArc, lo: ZImageLoraAdapterDevice,
    a_param_id: Int, b_param_id: Int,
    M: Int, in_f: Int, out_f: Int,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """StepSlab variant of `record_proj_lora` (this file :170)."""
    var nb = Optional[Tensor](None)
    var base = linear_slab(x[], w[], nb^, ctx, slab)
    var y = zimage_lora_apply_device_slab(base^, x[], lo, M, ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    edges.append(_leaf_edge(g, a_param_id))
    edges.append(_leaf_edge(g, b_param_id))
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(w.copy())
    saved.append(lo.a.copy())
    saved.append(lo.b.copy())
    var meta: List[Int] = [M, in_f, out_f, lo.rank]
    var scalars: List[Float32] = [lo.scale]
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_PROJ_LORA, edges^, saved^, meta^, scalars^, oids)
    return TArc(y^)


def record_rms_norm_dx_slab(
    mut g: Graph, x: TArc, weight: TArc, eps: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """StepSlab variant of `record_rms_norm_dx` (this file :202)."""
    var y = rms_norm_slab(x[], weight[], eps, ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(weight.copy())
    var scalars: List[Float32] = [eps]
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_RMS_NORM_DX, edges^, saved^, List[Int](), scalars^, oids)
    return TArc(y^)


def record_modulate_slab(
    mut g: Graph, x: TArc, scale: TArc, shift: TArc,
    scale_param_id: Int, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """StepSlab variant of `record_modulate` (this file :221)."""
    var y = modulate_slab(x[], scale[], shift[], ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    edges.append(_leaf_edge(g, scale_param_id))
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(scale.copy())
    var want_param = 0
    if scale_param_id > 0:
        want_param = 1
    var meta: List[Int] = [want_param]
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_MODULATE, edges^, saved^, meta^, List[Float32](), oids)
    return TArc(y^)


def record_rope_slab(
    mut g: Graph, x: TArc, cos: TArc, sin: TArc, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """StepSlab variant of `record_rope` (this file :249)."""
    var y = rope_interleaved_slab(x[], cos[], sin[], ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    var saved = List[TArc]()
    saved.append(cos.copy())
    saved.append(sin.copy())
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_ROPE, edges^, saved^, List[Int](), List[Float32](), oids)
    return TArc(y^)


def record_sdpa_slab[
    B: Int, S: Int, H: Int, Dh: Int
](
    mut g: Graph, q: TArc, k: TArc, v: TArc, scale: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """StepSlab variant of `record_sdpa` (this file :267)."""
    var y = sdpa_nomask_slab[B, S, H, Dh](q[], k[], v[], scale, ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(q[].id))
    edges.append(g.edge_for(k[].id))
    edges.append(g.edge_for(v[].id))
    var saved = List[TArc]()
    saved.append(q.copy())
    saved.append(k.copy())
    saved.append(v.copy())
    var meta: List[Int] = [B, S, H, Dh]
    var scalars: List[Float32] = [scale]
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_SDPA, edges^, saved^, meta^, scalars^, oids)
    return TArc(y^)


def record_swiglu_slab(
    mut g: Graph, gate: TArc, up: TArc, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """StepSlab variant of `record_swiglu` (this file :293)."""
    var y = swiglu_slab(gate[], up[], ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(gate[].id))
    edges.append(g.edge_for(up[].id))
    var saved = List[TArc]()
    saved.append(gate.copy())
    saved.append(up.copy())
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_SWIGLU, edges^, saved^, List[Int](), List[Float32](), oids)
    return TArc(y^)


def record_residual_gate_slab(
    mut g: Graph, x: TArc, gate_t: TArc, y_in: TArc, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """StepSlab variant of `record_residual_gate` (this file :311)."""
    var y = residual_gate_slab(x[], gate_t[], y_in[], ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    edges.append(g.edge_for(y_in[].id))
    var saved = List[TArc]()
    saved.append(gate_t.copy())
    var oids: List[Int] = [y.id]
    _ = g.record(
        OPK_RESIDUAL_GATE_DXDY, edges^, saved^, List[Int](), List[Float32](), oids
    )
    return TArc(y^)
