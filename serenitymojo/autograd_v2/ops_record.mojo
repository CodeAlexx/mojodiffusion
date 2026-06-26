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
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_f32, sdpa_flash_train_fwd,
    sdpa_flash_train_fwd_padmask_f32,
)
from serenitymojo.models.zimage.lora_block import ZIMAGE_SDPA_FLASH
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.single_block import KLEIN_SDPA_FLASH
from serenitymojo.ops.cast import cast_tensor
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
# ── P6 Klein vocabulary (record wrappers below call the EXACT forward
# functions the Klein stack-loop recompute calls; see each wrapper's docstring
# for the file:line of the mirrored oracle code).
from serenitymojo.models.klein.double_block import (
    StreamWeights,
    ModVecsDevice,
    StreamLoraDevice,
    _stream_pre_lora_resident,
    _stream_post_lora_resident,
)
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights,
    SingleModVecsDevice,
    SingleBlockLoraDevice,
)
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice,
    klein_lora_fwd_device_resident,
)
from serenitymojo.ops.linear import linear_rows, linear_rows_scratch
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.tensor_algebra import (
    slice as _ta_slice,
    concat as _ta_concat,
    reshape_owned as _ta_reshape_owned,
    add_in_place_f32 as _ta_add_in_place_f32,
)
from serenitymojo.ops.tensor_algebra_scratch import concat2_scratch
from serenitymojo.scratch_ring import ScratchRingAllocator
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
    OPK_KLEIN_DBL_PRE,
    OPK_KLEIN_DBL_JOINT,
    OPK_KLEIN_DBL_POST,
    OPK_KLEIN_SGL_IN,
    OPK_KLEIN_SGL_SDPA,
    OPK_KLEIN_SGL_OUT,
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
    # P7 batch-2 buckets (the _train_one_step_bucket_b2 instantiations:
    # 64x64/224 -> S=1248, 64x64/256 -> S=1280; sdpa_backward[2,...] already
    # instantiated via the b2 hand-chain).
    if B == 2 and S == 1248 and H == 30 and Dh == 128:
        var sb = sdpa_backward[2, 1248, 30, 128](q, k, v, d_out, scale, ctx)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 2 and S == 1280 and H == 30 and Dh == 128:
        var sb = sdpa_backward[2, 1280, 30, 128](q, k, v, d_out, scale, ctx)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    # krea2 fine-grained buckets (H=48, Dh=128): the block gate's L=512 and the
    # trainer's LFULL=4864 (train_krea2.mojo:135). sdpa_backward[1,L,48,128] is
    # already instantiated via the krea2 hand-chain block backward.
    if B == 1 and S == 512 and H == 48 and Dh == 128:
        var sb = sdpa_backward[1, 512, 48, 128](q, k, v, d_out, scale, ctx)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 1 and S == 4864 and H == 48 and Dh == 128:
        var sb = sdpa_backward[1, 4864, 48, 128](q, k, v, d_out, scale, ctx)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 1 and S == 2432 and H == 48 and Dh == 128:
        var sb = sdpa_backward[1, 2432, 48, 128](q, k, v, d_out, scale, ctx)
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
    # P7 batch-2 buckets (see sdpa_backward_dispatch above).
    if B == 2 and S == 1248 and H == 30 and Dh == 128:
        var sb = sdpa_backward_slab[2, 1248, 30, 128](q, k, v, d_out, scale, ctx, slab)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 2 and S == 1280 and H == 30 and Dh == 128:
        var sb = sdpa_backward_slab[2, 1280, 30, 128](q, k, v, d_out, scale, ctx, slab)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    # krea2 fine-grained buckets (H=48, Dh=128): block gate L=512, trainer L=4864.
    if B == 1 and S == 512 and H == 48 and Dh == 128:
        var sb = sdpa_backward_slab[1, 512, 48, 128](q, k, v, d_out, scale, ctx, slab)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 1 and S == 4864 and H == 48 and Dh == 128:
        var sb = sdpa_backward_slab[1, 4864, 48, 128](q, k, v, d_out, scale, ctx, slab)
        return SdpaGradArcs(
            arc_view(sb.d_q), arc_view(sb.d_k), arc_view(sb.d_v)
        )
    if B == 1 and S == 2432 and H == 48 and Dh == 128:
        var sb = sdpa_backward_slab[1, 2432, 48, 128](q, k, v, d_out, scale, ctx, slab)
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
    """StepSlab variant of `record_sdpa` (this file :267).
    ZIMAGE_SDPA_FLASH: forward runs cuDNN flash (bf16-native; pads S=1248
    internally); saved gains [3..7] = padded q/k/v/o + stats and the arm
    dispatches on arity (the Klein pattern). Flash allocs are POOL, not
    slab -> capture must be off (flag doc in lora_block.mojo)."""
    var y: Tensor
    var saved = List[TArc]()
    comptime if ZIMAGE_SDPA_FLASH:
        # zimage graph SDPA runs on F32 activations (measured: the dtype
        # check raised) -> F32<->bf16 boundary casts, the Klein pattern.
        var q_bf = cast_tensor(q[], STDtype.BF16, ctx)
        var k_bf = cast_tensor(k[], STDtype.BF16, ctx)
        var v_bf = cast_tensor(v[], STDtype.BF16, ctx)
        var ff = sdpa_flash_train_fwd[B, S, H, Dh](q_bf, k_bf, v_bf, scale, ctx)
        y = cast_tensor(ff.o, STDtype.F32, ctx)
        saved.append(q.copy())
        saved.append(k.copy())
        saved.append(v.copy())
        saved.append(TArc(Tensor(ff.q_pad.buf.copy(), ff.q_pad.shape(), ff.q_pad.dtype())))
        saved.append(TArc(Tensor(ff.k_pad.buf.copy(), ff.k_pad.shape(), ff.k_pad.dtype())))
        saved.append(TArc(Tensor(ff.v_pad.buf.copy(), ff.v_pad.shape(), ff.v_pad.dtype())))
        saved.append(TArc(Tensor(ff.o_pad.buf.copy(), ff.o_pad.shape(), ff.o_pad.dtype())))
        saved.append(TArc(Tensor(ff.stats.buf.copy(), ff.stats.shape(), ff.stats.dtype())))
    else:
        y = sdpa_nomask_slab[B, S, H, Dh](q[], k[], v[], scale, ctx, slab)
        saved.append(q.copy())
        saved.append(k.copy())
        saved.append(v.copy())
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(q[].id))
    edges.append(g.edge_for(k[].id))
    edges.append(g.edge_for(v[].id))
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


# ─────────────────────────────────────────────────────────────────────────────
# P6: Klein-9B record wrappers (AUTOGRAD_V2_MOJO_DESIGN.md P6).
#
# Klein records at COMPOSITE granularity (one node per oracle hand-chain
# helper / inline activation seam) so that every >=3-way fan-in fold lives
# INSIDE the oracle code the apply arm calls verbatim (engine.apply_klein).
# Each wrapper's forward calls the EXACT functions the Klein stack-loop
# recompute calls:
#   * double block:  double_block_lora_forward_device_resident_scratch
#     (models/klein/double_block.mojo:1389-1434) - _stream_pre_lora_resident,
#     concat2_scratch q/k + concat v + rope (with the oracle's mark/rewind),
#     sdpa_nomask, slice/reshape, _stream_post_lora_resident;
#   * single block:  single_block_lora_recompute_saved_device_resident_scratch
#     (models/klein/single_block.mojo:1037-1087) - layer_norm/modulate,
#     linear_rows bands + linear_rows_scratch gate_up,
#     klein_lora_fwd_device_resident qkv delta, rms_norm, rope, sdpa, swiglu,
#     concat out_in. The single block's final w2/LoRA-out/residual output is
#     NOT computed (the recompute oracle stops at out_in; the aux-off backward
#     never reads the block output value) - the OPK_KLEIN_SGL_OUT node is
#     recorded LAZILY (fresh output id, no forward tensor).
#
# Graph-level fan-ins (C15): ONLY 2-way (block input x <- {pre/in-chain,
# post/out-chain residual}); 2-way folds are bit-equal under operand swap
# (IEEE addition commutativity - the zimage P3 argument). Every >=3-way fold
# (e.g. the pre-stream d_norm <- base + q/k/v LoRA 4-way fold,
# double_block.mojo:2103-2117; the single d_norm <- split-GEMM + qkv LoRA
# fold, single_block.mojo:1422-1433) is INSIDE the oracle function the apply
# arm calls, so its fold order is the oracle's by construction.
#
# Adapter leaves are ALWAYS tracked here (the Klein trainer trains every
# slot); a missing adapter raises (fail loud, contract C7 has no frozen LoRA
# slot in this path).
# ─────────────────────────────────────────────────────────────────────────────


def _require_adapter(
    lo: Optional[LoraAdapterDevice], name: String
) raises -> LoraAdapterDevice:
    if lo:
        return lo.value().copy()
    raise Error(String("klein record: required LoRA adapter missing: ") + name)


def _rebox_with_id(mut g: Graph, t: TArc) raises -> TArc:
    """Zero-copy re-box of a helper-returned arc with a fresh graph tensor id
    (the zimage_block_graph.mojo:86 idiom): a fresh Tensor struct SHARING the
    device buffer, so the id stamp never mutates the shared original."""
    var y = Tensor(t[].buf.copy(), t[].shape(), t[].dtype())
    y.set_id(g.fresh_tensor_id())
    return TArc(y^)


struct KleinPreRecorded(Copyable, Movable):
    """record_klein_dbl_pre outputs (graph-tracked)."""

    var q_rms: TArc
    var k_rms: TArc
    var v: TArc

    def __init__(out self, var q_rms: TArc, var k_rms: TArc, var v: TArc):
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^


struct KleinJointRecorded(Copyable, Movable):
    """record_klein_dbl_joint outputs (graph-tracked)."""

    var txt_att: TArc
    var img_att: TArc

    def __init__(out self, var txt_att: TArc, var img_att: TArc):
        self.txt_att = txt_att^
        self.img_att = img_att^


struct KleinSglInRecorded(Copyable, Movable):
    """record_klein_sgl_in outputs (graph-tracked)."""

    var q_rms: TArc
    var k_rms: TArc
    var v: TArc
    var mlp_gate: TArc
    var mlp_up: TArc

    def __init__(
        out self, var q_rms: TArc, var k_rms: TArc, var v: TArc,
        var mlp_gate: TArc, var mlp_up: TArc,
    ):
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^
        self.mlp_gate = mlp_gate^
        self.mlp_up = mlp_up^


def record_klein_dbl_pre[
    H: Int, Dh: Int
](
    mut g: Graph, x: TArc,
    w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice,
    q_a_id: Int, q_b_id: Int, k_a_id: Int, k_b_id: Int, v_a_id: Int, v_b_id: Int,
    N: Int, D: Int, eps: Float32,
    norm_ones: TArc, norm_zeros: TArc,
    ctx: DeviceContext,
) raises -> KleinPreRecorded:
    """Per-stream PRE: x -> (q_rms, k_rms, v) with separate q/k/v LoRA.
    Forward = _stream_pre_lora_resident[H,Dh] (double_block.mojo:1264-1298),
    the EXACT call double_block_lora_forward_device_resident_scratch makes
    (:1403-1406). Backward arm (engine.apply_klein) =
    _stream_pre_backward_lora_resident_scratch_tensors (double_block.mojo:2071
    -2136) on the saved pieces, compute_aux_grads=False.
    Edges: [x, q_a, q_b, k_a, k_b, v_a, v_b]; outputs 0=q_rms 1=k_rms 2=v."""
    var q_ad = _require_adapter(lo.q, String("dbl pre q"))
    var k_ad = _require_adapter(lo.k, String("dbl pre k"))
    var v_ad = _require_adapter(lo.v, String("dbl pre v"))
    if q_ad.rank != k_ad.rank or q_ad.rank != v_ad.rank:
        raise Error("record_klein_dbl_pre: q/k/v adapter rank mismatch")

    var pre = _stream_pre_lora_resident[H, Dh](
        x, w, mv, lo, N, D, eps, norm_ones[], norm_zeros[], ctx
    )

    var q_rms = _rebox_with_id(g, pre.q_rms)
    var k_rms = _rebox_with_id(g, pre.k_rms)
    var v = _rebox_with_id(g, pre.v)

    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    edges.append(_leaf_edge(g, q_a_id))
    edges.append(_leaf_edge(g, q_b_id))
    edges.append(_leaf_edge(g, k_a_id))
    edges.append(_leaf_edge(g, k_b_id))
    edges.append(_leaf_edge(g, v_a_id))
    edges.append(_leaf_edge(g, v_b_id))
    # saved layout (apply_klein OPK_KLEIN_DBL_PRE arm contract):
    #   0 x, 1 ln1, 2 norm, 3 q_pre, 4 k_pre,
    #   5 wqkv, 6 q_norm, 7 k_norm, 8 scale1, 9 norm_ones,
    #   10 q_a, 11 q_b, 12 k_a, 13 k_b, 14 v_a, 15 v_b
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(pre.ln1.copy())
    saved.append(pre.norm.copy())
    saved.append(pre.q_pre.copy())
    saved.append(pre.k_pre.copy())
    saved.append(w.wqkv.copy())
    saved.append(w.q_norm.copy())
    saved.append(w.k_norm.copy())
    saved.append(mv.scale1.copy())
    saved.append(norm_ones.copy())
    saved.append(q_ad.a.copy())
    saved.append(q_ad.b.copy())
    saved.append(k_ad.a.copy())
    saved.append(k_ad.b.copy())
    saved.append(v_ad.a.copy())
    saved.append(v_ad.b.copy())
    var meta: List[Int] = [N, D, q_ad.rank]
    var scalars: List[Float32] = [eps, q_ad.scale, k_ad.scale, v_ad.scale]
    var oids: List[Int] = [q_rms[].id, k_rms[].id, v[].id]
    _ = g.record(OPK_KLEIN_DBL_PRE, edges^, saved^, meta^, scalars^, oids)
    return KleinPreRecorded(q_rms^, k_rms^, v^)


def record_klein_dbl_joint[
    H: Int, Dh: Int, S: Int
](
    mut g: Graph,
    tq: TArc, iq: TArc, tk: TArc, ik: TArc, tv: TArc, iv: TArc,
    cos: TArc, sin: TArc, scale: Float32,
    N_TXT: Int, N_IMG: Int, D: Int,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> KleinJointRecorded:
    """Joint attention: (txt|img q/k/v) -> (txt_att, img_att). Forward is the
    EXACT oracle sequence double_block.mojo:1408-1421 (concat2_scratch q/k with
    the oracle's mark/rewind, plain concat v, rope_interleaved, sdpa_nomask,
    slice + reshape_owned per stream). Backward arm = the oracle's joint
    backward block (double_block.mojo:2319-2337): reshape, concat2_scratch,
    sdpa_backward_scratch, rope_backward x2, slice_scratch x6,
    reshape_in_place. Edges: [tq, iq, tk, ik, tv, iv] (the concat operand
    order); outputs 0=txt_att 1=img_att."""
    var qk_mark = scratch.mark()
    var q = concat2_scratch(1, ctx, scratch, tq[], iq[])
    var k = concat2_scratch(1, ctx, scratch, tk[], ik[])
    var v_joint = _ta_concat(1, ctx, tv[], iv[])
    var q_rope = rope_interleaved(q, cos[], sin[], ctx)
    var k_rope = rope_interleaved(k, cos[], sin[], ctx)
    scratch.rewind(qk_mark)
    # saved layout: 0 q_rope, 1 k_rope, 2 v_joint, 3 cos, 4 sin
    # (+ flash 5..9 = bf16 q/k/v/o + stats; arm dispatches on arity)
    var saved = List[TArc]()
    var att: Tensor
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v_joint, scale, ctx)
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
        saved.append(TArc(q_rope^))
        saved.append(TArc(k_rope^))
        saved.append(TArc(v_joint^))
        saved.append(cos.copy())
        saved.append(sin.copy())
        saved.append(ff.q_bf.copy())
        saved.append(ff.k_bf.copy())
        saved.append(ff.v_bf.copy())
        saved.append(ff.o_bf.copy())
        saved.append(ff.stats.copy())
    else:
        att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v_joint, scale, ctx)
        saved.append(TArc(q_rope^))
        saved.append(TArc(k_rope^))
        saved.append(TArc(v_joint^))
        saved.append(cos.copy())
        saved.append(sin.copy())

    var txt_att_4d = _ta_slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = _ta_slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att_t = _ta_reshape_owned(txt_att_4d^, [N_TXT, D])
    var img_att_t = _ta_reshape_owned(img_att_4d^, [N_IMG, D])
    txt_att_t.set_id(g.fresh_tensor_id())
    img_att_t.set_id(g.fresh_tensor_id())
    var txt_att = TArc(txt_att_t^)
    var img_att = TArc(img_att_t^)

    var edges = List[Edge]()
    edges.append(g.edge_for(tq[].id))
    edges.append(g.edge_for(iq[].id))
    edges.append(g.edge_for(tk[].id))
    edges.append(g.edge_for(ik[].id))
    edges.append(g.edge_for(tv[].id))
    edges.append(g.edge_for(iv[].id))
    var meta: List[Int] = [N_TXT, N_IMG, D]
    var scalars: List[Float32] = [scale]
    var oids: List[Int] = [txt_att[].id, img_att[].id]
    _ = g.record(OPK_KLEIN_DBL_JOINT, edges^, saved^, meta^, scalars^, oids)
    return KleinJointRecorded(txt_att^, img_att^)


def record_klein_dbl_post(
    mut g: Graph, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice,
    out_a_id: Int, out_b_id: Int,
    ff_in_a_id: Int, ff_in_b_id: Int,
    ff_out_a_id: Int, ff_out_b_id: Int,
    N: Int, D: Int, F: Int, eps: Float32,
    norm_ones: TArc, norm_zeros: TArc,
    ctx: DeviceContext,
) raises -> TArc:
    """Per-stream POST: (x, att) -> stream out with out/ff_in/ff_out LoRA.
    Forward = _stream_post_lora_resident (double_block.mojo:1302-1337), the
    EXACT call double_block_lora_forward_device_resident_scratch makes
    (:1423-1426). Backward arm =
    _stream_post_backward_lora_resident_scratch_tensors (double_block.mojo:
    1781-1875), compute_aux_grads=False.
    Edges: [x, att, out_a, out_b, ff_in_a, ff_in_b, ff_out_a, ff_out_b]."""
    var out_ad = _require_adapter(lo.out, String("dbl post out"))
    var ff_in_ad = _require_adapter(lo.ff_in, String("dbl post ff_in"))
    var ff_out_ad = _require_adapter(lo.ff_out, String("dbl post ff_out"))
    if out_ad.rank != ff_in_ad.rank or out_ad.rank != ff_out_ad.rank:
        raise Error("record_klein_dbl_post: out/ff adapter rank mismatch")

    var post = _stream_post_lora_resident(
        x, att, w, mv, lo, N, D, F, eps, norm_ones[], norm_zeros[], ctx
    )
    var out = _rebox_with_id(g, post.out)

    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    edges.append(g.edge_for(att[].id))
    edges.append(_leaf_edge(g, out_a_id))
    edges.append(_leaf_edge(g, out_b_id))
    edges.append(_leaf_edge(g, ff_in_a_id))
    edges.append(_leaf_edge(g, ff_in_b_id))
    edges.append(_leaf_edge(g, ff_out_a_id))
    edges.append(_leaf_edge(g, ff_out_b_id))
    # saved layout (apply_klein OPK_KLEIN_DBL_POST arm contract):
    #   0 x, 1 att, 2 attn_res, 3 ln2, 4 mlp_in, 5 gate, 6 up, 7 act,
    #   8 wproj, 9 wgu, 10 wd, 11 gate1, 12 scale2, 13 gate2, 14 norm_ones,
    #   15 out_a, 16 out_b, 17 ff_in_a, 18 ff_in_b, 19 ff_out_a, 20 ff_out_b
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(att.copy())
    saved.append(post.attn_res.copy())
    saved.append(post.ln2.copy())
    saved.append(post.mlp_in.copy())
    saved.append(post.gate.copy())
    saved.append(post.up.copy())
    saved.append(post.act.copy())
    saved.append(w.wproj.copy())
    saved.append(w.wgu.copy())
    saved.append(w.wd.copy())
    saved.append(mv.gate1.copy())
    saved.append(mv.scale2.copy())
    saved.append(mv.gate2.copy())
    saved.append(norm_ones.copy())
    saved.append(out_ad.a.copy())
    saved.append(out_ad.b.copy())
    saved.append(ff_in_ad.a.copy())
    saved.append(ff_in_ad.b.copy())
    saved.append(ff_out_ad.a.copy())
    saved.append(ff_out_ad.b.copy())
    var meta: List[Int] = [N, D, F, out_ad.rank]
    var scalars: List[Float32] = [eps, out_ad.scale, ff_in_ad.scale, ff_out_ad.scale]
    var oids: List[Int] = [out[].id]
    _ = g.record(OPK_KLEIN_DBL_POST, edges^, saved^, meta^, scalars^, oids)
    return out^


def record_klein_sgl_in(
    mut g: Graph, x: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lo: SingleBlockLoraDevice,
    qkv_a_id: Int, qkv_b_id: Int,
    S_rows: Int, D: Int, F: Int, eps: Float32, H_: Int, Dh_: Int,
    norm_ones: TArc, norm_zeros: TArc,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> KleinSglInRecorded:
    """Single-block IN: x -> (q_rms, k_rms, v, mlp_gate, mlp_up) with the qkv
    LoRA. Forward mirrors single_block_lora_recompute_saved_device_resident_
    scratch (single_block.mojo:1037-1079) op-for-op: layer_norm, modulate,
    linear_rows q/k/v bands + linear_rows_scratch gate_up,
    klein_lora_fwd_device_resident delta + add_in_place_f32 x4, reshape_owned,
    rms_norm x2, gate/up slices, then the oracle's scratch rewind. Backward
    arm = single_block_lora_backward_device_resident_scratch_tensors's IN
    segment (single_block.mojo:1414-1445), compute_aux_grads=False.
    Edges: [x, qkv_a, qkv_b]; outputs 0=q_rms 1=k_rms 2=v 3=mlp_gate 4=mlp_up."""
    var qkv_ad = _require_adapter(lo.qkv, String("sgl qkv"))

    var ln_t = layer_norm(x[], norm_ones[], norm_zeros[], eps, ctx)
    var norm_t = modulate(ln_t, mv.scale[], mv.shift[], ctx)

    var scratch_mark = scratch.mark()
    var q_pre_flat = linear_rows(norm_t, w.w1[], 0, D, ctx)
    var k_pre_flat = linear_rows(norm_t, w.w1[], D, D, ctx)
    var v_flat = linear_rows(norm_t, w.w1[], 2 * D, D, ctx)
    var gate_up = linear_rows_scratch(norm_t, w.w1[], 3 * D, 2 * F, ctx, scratch)
    # qkv LoRA delta: the SAME dispatcher the recompute oracle calls
    # (single_block.mojo:1059; fused path currently dormant -> unfused chain).
    var dlt = klein_lora_fwd_device_resident(norm_t, qkv_ad, S_rows, ctx)
    _ta_add_in_place_f32(q_pre_flat, _ta_slice(dlt, 1, 0, D, ctx), ctx)
    _ta_add_in_place_f32(k_pre_flat, _ta_slice(dlt, 1, D, D, ctx), ctx)
    _ta_add_in_place_f32(v_flat, _ta_slice(dlt, 1, 2 * D, D, ctx), ctx)
    _ta_add_in_place_f32(gate_up, _ta_slice(dlt, 1, 3 * D, 2 * F, ctx), ctx)
    var q_pre = _ta_reshape_owned(q_pre_flat^, [1, S_rows, H_, Dh_])
    var k_pre = _ta_reshape_owned(k_pre_flat^, [1, S_rows, H_, Dh_])
    var v_t = _ta_reshape_owned(v_flat^, [1, S_rows, H_, Dh_])

    var q_rms_t = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms_t = rms_norm(k_pre, w.k_norm[], eps, ctx)

    # gate/up slices are fresh copies (ops.tensor_algebra.slice) so the
    # gate_up scratch region is dead -> the oracle's rewind (:1079).
    var mlp_gate_t = _ta_slice(gate_up, 1, 0, F, ctx)
    var mlp_up_t = _ta_slice(gate_up, 1, F, F, ctx)
    scratch.rewind(scratch_mark)

    q_rms_t.set_id(g.fresh_tensor_id())
    k_rms_t.set_id(g.fresh_tensor_id())
    v_t.set_id(g.fresh_tensor_id())
    mlp_gate_t.set_id(g.fresh_tensor_id())
    mlp_up_t.set_id(g.fresh_tensor_id())
    var q_rms = TArc(q_rms_t^)
    var k_rms = TArc(k_rms_t^)
    var v = TArc(v_t^)
    var mlp_gate = TArc(mlp_gate_t^)
    var mlp_up = TArc(mlp_up_t^)

    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    edges.append(_leaf_edge(g, qkv_a_id))
    edges.append(_leaf_edge(g, qkv_b_id))
    # saved layout (apply_klein OPK_KLEIN_SGL_IN arm contract):
    #   0 x, 1 ln, 2 norm, 3 q_pre, 4 k_pre,
    #   5 w1, 6 q_norm, 7 k_norm, 8 scale_vec, 9 norm_ones, 10 qkv_a, 11 qkv_b
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(TArc(ln_t^))
    saved.append(TArc(norm_t^))
    saved.append(TArc(q_pre^))
    saved.append(TArc(k_pre^))
    saved.append(w.w1.copy())
    saved.append(w.q_norm.copy())
    saved.append(w.k_norm.copy())
    saved.append(mv.scale.copy())
    saved.append(norm_ones.copy())
    saved.append(qkv_ad.a.copy())
    saved.append(qkv_ad.b.copy())
    var meta: List[Int] = [S_rows, D, F, qkv_ad.rank]
    var scalars: List[Float32] = [eps, qkv_ad.scale]
    var oids: List[Int] = [
        q_rms[].id, k_rms[].id, v[].id, mlp_gate[].id, mlp_up[].id
    ]
    _ = g.record(OPK_KLEIN_SGL_IN, edges^, saved^, meta^, scalars^, oids)
    return KleinSglInRecorded(q_rms^, k_rms^, v^, mlp_gate^, mlp_up^)


def record_klein_sgl_sdpa[
    H: Int, Dh: Int, S: Int
](
    mut g: Graph, q_rms: TArc, k_rms: TArc, v: TArc,
    cos: TArc, sin: TArc, scale: Float32, D: Int,
    ctx: DeviceContext,
) raises -> TArc:
    """Single-block attention core: (q_rms, k_rms, v) -> att_flat. Forward
    mirrors single_block.mojo:1071-1074 (rope_interleaved x2, sdpa_nomask,
    reshape_owned [S,D]). Backward arm = the oracle's sdpa segment
    (single_block.mojo:1402-1412): reshape view, sdpa_backward_scratch,
    rope_backward x2, d_v reshape. Edges: [q_rms, k_rms, v]."""
    var q_rope = rope_interleaved(q_rms[], cos[], sin[], ctx)
    var k_rope = rope_interleaved(k_rms[], cos[], sin[], ctx)
    # saved layout: 0 q_rope, 1 k_rope, 2 v, 3 cos, 4 sin
    # (+ flash: 5 q_bf, 6 k_bf, 7 v_bf, 8 o_bf, 9 stats — KLEIN_SDPA_FLASH,
    # same swap as the hand-chain helper; arm dispatches on saved arity)
    var saved = List[TArc]()
    var att_flat: Tensor
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v[], scale, ctx)
        var af_shape: List[Int] = [S, D]
        att_flat = Tensor(ff.att.buf.copy(), af_shape^, STDtype.F32)
        saved.append(TArc(q_rope^))
        saved.append(TArc(k_rope^))
        saved.append(v.copy())
        saved.append(cos.copy())
        saved.append(sin.copy())
        saved.append(ff.q_bf.copy())
        saved.append(ff.k_bf.copy())
        saved.append(ff.v_bf.copy())
        saved.append(ff.o_bf.copy())
        saved.append(ff.stats.copy())
    else:
        var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v[], scale, ctx)
        att_flat = _ta_reshape_owned(att^, [S, D])
        saved.append(TArc(q_rope^))
        saved.append(TArc(k_rope^))
        saved.append(v.copy())
        saved.append(cos.copy())
        saved.append(sin.copy())
    att_flat.set_id(g.fresh_tensor_id())

    var edges = List[Edge]()
    edges.append(g.edge_for(q_rms[].id))
    edges.append(g.edge_for(k_rms[].id))
    edges.append(g.edge_for(v[].id))
    var meta: List[Int] = [S, D]
    var scalars: List[Float32] = [scale]
    var oids: List[Int] = [att_flat.id]
    _ = g.record(OPK_KLEIN_SGL_SDPA, edges^, saved^, meta^, scalars^, oids)
    return TArc(att_flat^)


def record_klein_sgl_out(
    mut g: Graph, x: TArc, att_flat: TArc, mlp: TArc,
    w: SingleBlockWeights, gate_vec: TArc, lo: SingleBlockLoraDevice,
    out_a_id: Int, out_b_id: Int,
    S_rows: Int, D: Int, F: Int,
    ctx: DeviceContext,
) raises -> Int:
    """Single-block OUT: (x, att_flat, mlp) -> block out, with the to_out
    LoRA. LAZY forward: out_in = concat(att_flat, mlp) (the recompute oracle's
    last computed value, single_block.mojo:1081); the w2 projection + LoRA-out
    delta + residual_gate output are NEVER computed - exactly like the
    recompute oracle, because the aux-off backward
    (gate_residual_backward_dxdy) never reads the output value. Returns the
    block-output TENSOR ID (the engine root); no output tensor exists.
    Backward arm = the oracle's OUT segment (single_block.mojo:1364-1400),
    compute_aux_grads=False: gate_residual_backward_dxdy,
    linear_backward_dx_scratch vs w2_att/w2_mlp, _klein_lora_bwd_dropout_
    tensors on out_in, add_in_place_f32 column folds.
    Edges: [x, att_flat, mlp, out_a, out_b]."""
    var out_ad = _require_adapter(lo.out, String("sgl out"))

    var out_in = _ta_concat(1, ctx, att_flat[], mlp[])

    var out_id = g.fresh_tensor_id()
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    edges.append(g.edge_for(att_flat[].id))
    edges.append(g.edge_for(mlp[].id))
    edges.append(_leaf_edge(g, out_a_id))
    edges.append(_leaf_edge(g, out_b_id))
    # saved layout (apply_klein OPK_KLEIN_SGL_OUT arm contract):
    #   0 out_in, 1 w2_att, 2 w2_mlp, 3 gate_vec, 4 out_a, 5 out_b
    var saved = List[TArc]()
    saved.append(TArc(out_in^))
    saved.append(w.w2_att.copy())
    saved.append(w.w2_mlp.copy())
    saved.append(gate_vec.copy())
    saved.append(out_ad.a.copy())
    saved.append(out_ad.b.copy())
    var meta: List[Int] = [S_rows, D, F, out_ad.rank]
    var scalars: List[Float32] = [out_ad.scale]
    var oids: List[Int] = [out_id]
    _ = g.record(OPK_KLEIN_SGL_OUT, edges^, saved^, meta^, scalars^, oids)
    return out_id


# ─────────────────────────────────────────────────────────────────────────────
# P7 ideogram4 COARSE record (Stage 1). Records the block forward as ONE composite
# node: runs the fwd to produce the block output (acts discarded — apply recomputes,
# the hand-chain stack-backward discipline), saves the recompute inputs
# [x,adaln,cos,sin, 13 weights (Ideogram4BlockWeights field order), 12 lora a/b],
# and edges x,adaln,a/b-per-slot to the 14 leaves. apply_ideogram4 unpacks by index.
# ─────────────────────────────────────────────────────────────────────────────
from std.memory import ArcPointer as _ArcPtrR
from serenitymojo.models.ideogram4.block import (
    ideogram4_block_lora_forward as _i4r_fwd,
    Ideogram4BlockWeights as _I4Wr,
)
from serenitymojo.models.ideogram4.lora_module import LoraAdapter as _I4LoraR
from serenitymojo.autograd_v2.node import (
    OPK_IDEOGRAM4_BLOCK as _OPK_I4R, arc_view as _i4r_arc, Edge as _EdgeI4R,
)

comptime _I4rLArc = _ArcPtrR[_I4LoraR]


def record_ideogram4_block[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    mut g: Graph,
    x: TArc, adaln: TArc, cosf: TArc, sinf: TArc,
    w: _I4Wr, loras: List[_I4rLArc],
    x_id: Int, adaln_id: Int, a_ids: List[Int], b_ids: List[Int],
    ctx: DeviceContext,
) raises -> TArc:
    var rb = _i4r_fwd[S, Hidden, Heads, Dh, FF, Adaln](
        x[], adaln[], cosf[], sinf[], w, loras, ctx
    )
    var out_t = Tensor(rb.out.buf.copy(), rb.out.shape(), rb.out.dtype())
    out_t.set_id(g.fresh_tensor_id())
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(adaln.copy())
    saved.append(cosf.copy())
    saved.append(sinf.copy())
    saved.append(_i4r_arc(w.adaln_w))
    saved.append(_i4r_arc(w.adaln_b))
    saved.append(_i4r_arc(w.attn_norm1))
    saved.append(_i4r_arc(w.attn_norm2))
    saved.append(_i4r_arc(w.ffn_norm1))
    saved.append(_i4r_arc(w.ffn_norm2))
    saved.append(_i4r_arc(w.qkv_w))
    saved.append(_i4r_arc(w.o_w))
    saved.append(_i4r_arc(w.norm_q))
    saved.append(_i4r_arc(w.norm_k))
    saved.append(_i4r_arc(w.w1))
    saved.append(_i4r_arc(w.w2))
    saved.append(_i4r_arc(w.w3))
    for slot in range(6):
        saved.append(_i4r_arc(loras[slot][].a))
        saved.append(_i4r_arc(loras[slot][].b))
    var edges = List[_EdgeI4R]()
    edges.append(g.edge_for(x_id))
    edges.append(g.edge_for(adaln_id))
    for slot in range(6):
        edges.append(g.edge_for(a_ids[slot]))
        edges.append(g.edge_for(b_ids[slot]))
    var meta = List[Int]()
    meta.append(loras[0][].rank)
    var scal = List[Float32]()
    scal.append(loras[0][].alpha)
    var out_ids = List[Int]()
    out_ids.append(out_t.id)
    _ = g.record(_OPK_I4R, edges^, saved^, meta^, scal^, out_ids)
    return TArc(out_t^)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4b krea2 COARSE record (krea2_block_graph.mojo). Records the block forward
# as ONE composite node OPK_KREA2_SINGLE_BLOCK with a SINGLE tracked edge (the block
# input x — the only inter-block dependency the engine routes; the 8 LoRA grads are
# HOST lists captured out-of-band, NOT engine leaves). The bulky weights/adapters/
# rope structs are NOT packed into Node.saved (13 weights + 8 adapters × {a,b,rank,
# in_f,out_f,scale} is error-prone index gymnastics); apply_krea2/execute_krea2_block
# take them as direct params (the execute_ideogram4 comptime-threading precedent,
# generalized to runtime structs — same structs the oracle uses, bit-identical). The
# record runs the forward only to MINT the block-output tensor id (acts discarded —
# apply_krea2 RECOMPUTES from the saved x, the conductor's recompute-checkpoint
# discipline). saved=[x]; saved_meta=[L]; scalars=[eps]; edges(1)=x.
# ─────────────────────────────────────────────────────────────────────────────
from serenitymojo.models.krea2.krea2_block import (
    krea2_single_stream_block_lora as _k2r_fwd,
    Krea2BlockWeights as _K2Wr,
    Krea2BlockLora as _K2Lr,
)
from serenitymojo.autograd_v2.node import (
    OPK_KREA2_SINGLE_BLOCK as _OPK_K2, Edge as _EdgeK2,
)
# ── krea2 FINE-GRAINED record wrappers (this session). The 3 krea2-specific
# kinds + record_mul (krea2 needs the OPK_MUL recorder zimage never recorded).
from serenitymojo.ops.tensor_algebra import mul as _ta_mul_k2, mul_slab as _ta_mul_k2_slab
from serenitymojo.ops.activations import sigmoid as _act_sigmoid, sigmoid_slab as _act_sigmoid_slab
from serenitymojo.ops.gqa_backward import (
    repeat_kv_f32 as _gqa_repeat_kv, repeat_kv_f32_slab as _gqa_repeat_kv_slab,
)
from serenitymojo.models.krea2.krea2_block import (
    _linear_lora as _k2_linear_lora,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice as _K2LoraAdapter
from serenitymojo.autograd_v2.node import (
    OPK_MUL as _OPK_MUL_K2,
    OPK_REPEAT_KV as _OPK_REPEAT_KV,
    OPK_SIGMOID as _OPK_SIGMOID,
    OPK_KREA2_PROJ_LORA as _OPK_K2_PROJ,
)


def record_mul(
    mut g: Graph, a: TArc, b: TArc, ctx: DeviceContext
) raises -> TArc:
    """y = mul(a, b) (ops.tensor_algebra.mul — the krea2 gated = attn_flat * sg,
    krea2_block.mojo:444). OPK_MUL apply: saved[0]=A, saved[1]=B; d_A = g*B,
    d_B = g*A (the oracle's d_attn_flat=mul(d_gated,sg), d_sg=mul(d_gated,attn_flat),
    :599-600). Edges [a, b]; saved [a, b]."""
    var y = _ta_mul_k2(a[], b[], ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(a[].id))
    edges.append(g.edge_for(b[].id))
    var saved = List[TArc]()
    saved.append(a.copy())
    saved.append(b.copy())
    var oids: List[Int] = [y.id]
    _ = g.record(_OPK_MUL_K2, edges^, saved^, List[Int](), List[Float32](), oids)
    return TArc(y^)


def record_repeat_kv(
    mut g: Graph, x: TArc, L: Int, kvheads: Int, n_rep: Int, headdim: Int,
    ctx: DeviceContext,
) raises -> TArc:
    """y = repeat_kv_f32(x, L, kvheads, n_rep, headdim) — GQA head-broadcast
    (krea2_block.mojo:404-405 k_full/v_full). Backward arm (engine.apply) =
    repeat_kv_backward (grouped sum-reduce, :647-648). x [1,L,kvheads,Dh] ->
    [1,L,kvheads*n_rep,Dh]. Edges [x]; saved []; meta [L, kvheads, n_rep, Dh]."""
    var y = _gqa_repeat_kv(x[], L, kvheads, n_rep, headdim, ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    var meta: List[Int] = [L, kvheads, n_rep, headdim]
    var oids: List[Int] = [y.id]
    _ = g.record(_OPK_REPEAT_KV, edges^, List[TArc](), meta^, List[Float32](), oids)
    return TArc(y^)


def record_sigmoid(
    mut g: Graph, x: TArc, ctx: DeviceContext
) raises -> TArc:
    """sg = sigmoid(x) (krea2_block.mojo:443 sg = sigmoid(gate_pre)). Backward
    arm = sigmoid_backward(g, x) (:602 d_gate_pre = sigmoid_backward(d_sg,
    gate_pre)). Edges [x]; saved [x] (sigmoid_backward reads the PRE-activation
    x, not sg — matches the oracle, which passes saved.gate_pre)."""
    var y = _act_sigmoid(x[], ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    var saved = List[TArc]()
    saved.append(x.copy())
    var oids: List[Int] = [y.id]
    _ = g.record(_OPK_SIGMOID, edges^, saved^, List[Int](), List[Float32](), oids)
    return TArc(y^)


def record_krea2_proj_lora(
    mut g: Graph, x: TArc, w: TArc, lo: Optional[_K2LoraAdapter],
    M: Int, in_f: Int, out_f: Int, lora_slot: Int,
    ctx: DeviceContext,
) raises -> TArc:
    """y = linear(x, W_frozen) + scale*(x@Aᵀ)@Bᵀ — the krea2 oracle _linear_lora
    (krea2_block.mojo:112). Backward arm (engine.apply_krea2_fg) = the oracle's
    OWN _linear_bwd_dx (:489): base linear_backward_dx + the unfused LoRA backward
    whose dA/dB are HOST List[Float32]. d_x routes through the single (x) edge;
    the host LoRA pair is captured OUT-OF-BAND by the krea2 driver, keyed by
    `lora_slot` (0..7, or <0 when no adapter). The base weight W and the LoRA A/B
    are FROZEN (no engine leaves — krea2 LoRA grads do not flow as TArc).
    Edges [x]; saved [x, w] (+ [A, B] when an adapter is present, so the apply
    arm can rebuild the LoraAdapterDevice); meta [M, in_f, out_f, rank, lora_slot];
    scalars [scale]."""
    var y = _k2_linear_lora(x[], w[], lo, M, ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))   # ONLY the input is tracked
    var saved = List[TArc]()
    saved.append(x.copy())
    saved.append(w.copy())
    var rank = 0
    var scale = Float32(0.0)
    var slot = -1
    if lo:
        saved.append(lo.value().a.copy())
        saved.append(lo.value().b.copy())
        rank = lo.value().rank
        scale = lo.value().scale
        slot = lora_slot
    var meta: List[Int] = [M, in_f, out_f, rank, slot]
    var scalars: List[Float32] = [scale]
    var oids: List[Int] = [y.id]
    _ = g.record(_OPK_K2_PROJ, edges^, saved^, meta^, scalars^, oids)
    return TArc(y^)


def record_krea2_single_block[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    mut g: Graph,
    x: TArc, vec: Tensor,
    w: _K2Wr, lora: _K2Lr,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
    eps: Float32, x_id: Int, ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> TArc:
    var fb = _k2r_fwd[L, HEADS, KVHEADS, HEADDIM](
        x, vec, w, lora, cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
    )
    var out_t = Tensor(fb.out[].buf.copy(), fb.out[].shape(), fb.out[].dtype())
    out_t.set_id(g.fresh_tensor_id())
    var saved = List[TArc]()
    saved.append(x.copy())
    var edges = List[_EdgeK2]()
    edges.append(g.edge_for(x_id))
    var meta = List[Int]()
    meta.append(L)
    var scal = List[Float32]()
    scal.append(eps)
    var out_ids = List[Int]()
    out_ids.append(out_t.id)
    _ = g.record(_OPK_K2, edges^, saved^, meta^, scal^, out_ids)
    return TArc(out_t^)


# ── krea2 slab record variants (activation-checkpointing slab path, contract C8).
# BYTE-IDENTICAL recording to the non-slab krea2 wrappers (same edges/saved/meta/
# scalars, same C15 slots); only the forward op's allocation source is the slab.
def record_add_slab(mut g: Graph, a: TArc, b: TArc, ctx: DeviceContext, mut slab: StepSlab) raises -> TArc:
    """StepSlab variant of `record_add` — the OPK_ADD forward allocs from slab
    (the backward arm is allocation-free)."""
    var y = add_slab(a[], b[], ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(a[].id))
    edges.append(g.edge_for(b[].id))
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_ADD, edges^, List[TArc](), List[Int](), List[Float32](), oids)
    return TArc(y^)


def record_mul_slab(mut g: Graph, a: TArc, b: TArc, ctx: DeviceContext, mut slab: StepSlab) raises -> TArc:
    """StepSlab variant of `record_mul`."""
    var y = _ta_mul_k2_slab(a[], b[], ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(a[].id))
    edges.append(g.edge_for(b[].id))
    var saved = List[TArc]()
    saved.append(a.copy())
    saved.append(b.copy())
    var oids: List[Int] = [y.id]
    _ = g.record(_OPK_MUL_K2, edges^, saved^, List[Int](), List[Float32](), oids)
    return TArc(y^)


def record_repeat_kv_slab(
    mut g: Graph, x: TArc, L: Int, kvheads: Int, n_rep: Int, headdim: Int,
    ctx: DeviceContext, mut slab: StepSlab,
) raises -> TArc:
    """StepSlab variant of `record_repeat_kv`."""
    var y = _gqa_repeat_kv_slab(x[], L, kvheads, n_rep, headdim, ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    var meta: List[Int] = [L, kvheads, n_rep, headdim]
    var oids: List[Int] = [y.id]
    _ = g.record(_OPK_REPEAT_KV, edges^, List[TArc](), meta^, List[Float32](), oids)
    return TArc(y^)


def record_sigmoid_slab(mut g: Graph, x: TArc, ctx: DeviceContext, mut slab: StepSlab) raises -> TArc:
    """StepSlab variant of `record_sigmoid`."""
    var y = _act_sigmoid_slab(x[], ctx, slab)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(x[].id))
    var saved = List[TArc]()
    saved.append(x.copy())
    var oids: List[Int] = [y.id]
    _ = g.record(_OPK_SIGMOID, edges^, saved^, List[Int](), List[Float32](), oids)
    return TArc(y^)


def record_sdpa_nomask_slab[
    B: Int, S: Int, H: Int, Dh: Int
](
    mut g: Graph, q: TArc, k: TArc, v: TArc, scale: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """ALWAYS-math (sdpa_nomask) StepSlab SDPA recorder — the krea2 no-pad math
    slab arm. Unlike `record_sdpa_slab` it does NOT branch on ZIMAGE_SDPA_FLASH
    (krea2's math gate is deterministic full attention). saved [q,k,v] → the
    OPK_SDPA non-flash arm. Byte-identical recording to `record_sdpa` (non-slab)
    except the forward allocs from slab."""
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


def record_sdpa_flash_nopad_slab[
    B: Int, S: Int, H: Int, Dh: Int
](
    mut g: Graph, q: TArc, k: TArc, v: TArc, scale: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """FLASH (cuDNN, no-pad full attention) StepSlab SDPA recorder — the krea2
    PRODUCTION attn arm (O(L), no [B*H*S,S] scores; the math sdpa_nomask's O(L²)
    scores are why the math attn segment is 13.4GB — flash removes them). Records
    OPK_SDPA with the 8-tensor flash saved set (saved 0..2 = q/k/v placeholders,
    3..7 = q_pad/k_pad/v_pad/o_pad/stats) → the existing OPK_SDPA flash arm
    (len(saved)>=8 → sdpa_flash_backward_dispatch). The flash fwd allocs POOL (not
    slab) — fine: we are NOT capturing (capture OFF); the slab is for the alloc-free
    LINEAR ops. F32 acts→bf16→flash→F32 out (no-op in F32; the krea2 acts boundary).
    flash dQ nondeterministic → grads are value-tolerance class (NOT bit), so this
    arm is the TRAINER path; the bit gate uses record_sdpa_nomask_slab (math)."""
    var q_bf = cast_tensor(q[], STDtype.BF16, ctx)
    var k_bf = cast_tensor(k[], STDtype.BF16, ctx)
    var v_bf = cast_tensor(v[], STDtype.BF16, ctx)
    var ff = sdpa_flash_train_fwd[B, S, H, Dh](q_bf, k_bf, v_bf, scale, ctx)
    var y = cast_tensor(ff.o, q[].dtype(), ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(q[].id))
    edges.append(g.edge_for(k[].id))
    edges.append(g.edge_for(v[].id))
    var saved = List[TArc]()
    saved.append(q.copy())
    saved.append(k.copy())
    saved.append(v.copy())
    saved.append(TArc(Tensor(ff.q_pad.buf.copy(), ff.q_pad.shape(), ff.q_pad.dtype())))
    saved.append(TArc(Tensor(ff.k_pad.buf.copy(), ff.k_pad.shape(), ff.k_pad.dtype())))
    saved.append(TArc(Tensor(ff.v_pad.buf.copy(), ff.v_pad.shape(), ff.v_pad.dtype())))
    saved.append(TArc(Tensor(ff.o_pad.buf.copy(), ff.o_pad.shape(), ff.o_pad.dtype())))
    saved.append(TArc(Tensor(ff.stats.buf.copy(), ff.stats.shape(), ff.stats.dtype())))
    var meta: List[Int] = [B, S, H, Dh]
    var scalars: List[Float32] = [scale]
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_SDPA, edges^, saved^, meta^, scalars^, oids)
    return TArc(y^)


def record_sdpa_flash_padmask_slab[
    B: Int, S: Int, H: Int, Dh: Int
](
    mut g: Graph, q: TArc, k: TArc, v: TArc, real_len: Int, scale: Float32,
    ctx: DeviceContext, mut slab: StepSlab,
) raises -> TArc:
    """cuDNN flash-PADMASK StepSlab SDPA recorder — the krea2 PRODUCTION attn arm
    for the LT-padded trainer (real_len<L; cuDNN masks the [real_len:S] tail). O(L),
    no scores. Records OPK_SDPA with the 8-tensor flash saved set (saved 0..2=q/k/v,
    3..7 = q_bf/k_bf/v_bf/o_bf/stats — the _padmask_f32 variant) + scalars=[scale,
    Float32(real_len)] (len 2 = the PADMASK marker the OPK_SDPA arm dispatches on →
    sdpa_flash_backward_padmask_f32 with real_len). Flash dQ nondeterministic →
    value-tolerance grads (NOT bit). flash allocs POOL (capture OFF)."""
    var q_bf = cast_tensor(q[], STDtype.BF16, ctx)
    var k_bf = cast_tensor(k[], STDtype.BF16, ctx)
    var v_bf = cast_tensor(v[], STDtype.BF16, ctx)
    var ff = sdpa_flash_train_fwd_padmask_f32[B, S, H, Dh](q_bf, k_bf, v_bf, real_len, scale, ctx)
    var y = cast_tensor(ff.att, q[].dtype(), ctx)
    y.set_id(g.fresh_tensor_id())
    var edges = List[Edge]()
    edges.append(g.edge_for(q[].id))
    edges.append(g.edge_for(k[].id))
    edges.append(g.edge_for(v[].id))
    var saved = List[TArc]()
    saved.append(q.copy())
    saved.append(k.copy())
    saved.append(v.copy())
    saved.append(ff.q_bf.copy())
    saved.append(ff.k_bf.copy())
    saved.append(ff.v_bf.copy())
    saved.append(ff.o_bf.copy())
    saved.append(ff.stats.copy())
    var meta: List[Int] = [B, S, H, Dh]
    var scalars: List[Float32] = [scale, Float32(real_len)]   # len 2 = PADMASK
    var oids: List[Int] = [y.id]
    _ = g.record(OPK_SDPA, edges^, saved^, meta^, scalars^, oids)
    return TArc(y^)
