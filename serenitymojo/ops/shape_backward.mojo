# ops/shape_backward.mojo — BACKWARD for the Tier-0 shape/structural arms:
#   Cat, Split, Slice, Reshape, Transpose, Permute, Broadcast, Repeat, Where,
#   Clamp, Maximum, Minimum, Cast, IndexSelect.
#
# Phase T1 of FULL_PORT_TRAINING_PLAN.md (Tier 0 backward arms). The backward of
# the shape ops is almost entirely grad-ROUTING — no learnable parameters, just
# moving / summing / masking the upstream gradient. F32 throughout (the gate
# dtype); loud-fail on shape/dtype mismatch.
#
# Conventions mirror ops/reduce_backward.mojo + ops/linalg_backward.mojo (the
# proven sibling templates):
#   * device buffers are DType.uint8 (Tensor's storage), bitcast to Float32 at
#     the LayoutTensor boundary,
#   * one flat thread per OUTPUT element for the kernels,
#   * F32 interior and single-stream ordering; downstream to_host/sync fences,
#   * multi-output arms return a Movable struct (Tensor is move-only).
#
# Math (verified against flame-core autograd.rs @ 7be76ef):
#   maximum: mask = (a >= b)        (autograd.rs:6114)
#   minimum: mask = (a <= b)        (autograd.rs:6134)
#   where:   grad_t = g*cond ; grad_f = g*(1-cond)  (autograd.rs:6157)
#   repeat:  reshape + sum_dim_keepdim per axis -> sum tiled copies (5749)
#   index_select: scatter_add (repeated indices accumulate)         (5869)
#
# ── Math (grad of each forward) ───────────────────────────────────────────────
#   cat(axis): y = concat(xs, axis)        -> d_x_i = slice of d_y along axis
#   split(axis): xs = split(y, axis)       -> d_y   = concat(d_x_i, axis)
#   slice(dim,start,len): y = x[..,start:start+len,..]
#                                          -> d_x = zeros(x); d_x[slice] = d_y
#   reshape(shape): y = x.reshape(shape)   -> d_x = d_y.reshape(x.shape)
#   transpose(d0,d1)                       -> d_x = d_y.transpose(d0,d1)
#   permute(perm): y = x.permute(perm)     -> d_x = d_y.permute(inverse(perm))
#   broadcast(in->out): y = x expanded     -> d_x = sum over broadcasted dims
#   repeat(reps): y = x tiled              -> d_x = sum the tiled copies
#   where(cond): y = cond ? a : b          -> d_a = cond ? d_y : 0 ;
#                                             d_b = cond ? 0 : d_y
#   clamp(lo,hi): y = clamp(x,lo,hi)       -> d_x = (lo<=x<=hi) ? d_y : 0
#   maximum(a,b): y = max(a,b)             -> d_a = (a>=b)?d_y:0 ; d_b = (a<b)?d_y:0
#   minimum(a,b): y = min(a,b)             -> d_a = (a<=b)?d_y:0 ; d_b = (a>b)?d_y:0
#   cast: y = x.to(dtype)                  -> d_x = d_y (identity in F32 gate)
#   index_select(dim,idx): y = x[idx along dim]
#                                          -> d_x = zeros(x); scatter-ADD d_y rows
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _MAXRANK = 6  # match tensor_algebra.mojo permute/broadcast support


# ── multi-output structs (Tensor is move-only) ────────────────────────────────
struct WhereGrads(Movable):
    """Backward outputs of `where_backward`: gradients wrt the lhs and rhs."""

    var d_a: Tensor
    var d_b: Tensor

    def __init__(out self, var d_a: Tensor, var d_b: Tensor):
        self.d_a = d_a^
        self.d_b = d_b^


struct BinaryGrads(Movable):
    """Backward outputs of `maximum_backward` / `minimum_backward`."""

    var d_a: Tensor
    var d_b: Tensor

    def __init__(out self, var d_a: Tensor, var d_b: Tensor):
        self.d_a = d_a^
        self.d_b = d_b^


struct CatGrads2(Movable):
    """Backward outputs of `cat_backward` for the 2-input case (the diffusion-
    common join: img/txt, gate/up). Tensor is move-only and List requires
    Copyable, so multi-output returns use a fixed-slot struct."""

    var d_0: Tensor
    var d_1: Tensor

    def __init__(out self, var d_0: Tensor, var d_1: Tensor):
        self.d_0 = d_0^
        self.d_1 = d_1^


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _numel(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _require_f32(t: Tensor, who: String) raises:
    if t.dtype() != STDtype.F32:
        raise Error(who + ": inputs must be F32")


# ══════════════════════════════════════════════════════════════════════════════
# RESHAPE / CAST — pure metadata / identity grad (a D2D clone with new/same shape)
# ══════════════════════════════════════════════════════════════════════════════
def reshape_backward(
    grad_out: Tensor, in_shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """d_x = d_y reshaped back to the forward INPUT shape (data unchanged)."""
    _require_f32(grad_out, "reshape_backward")
    var n = _numel(in_shape)
    if n != grad_out.numel():
        raise Error(
            String("reshape_backward: numel mismatch ")
            + String(n) + " != " + String(grad_out.numel()))
    var dev = ctx.enqueue_create_buffer[DType.uint8](grad_out.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=grad_out.buf)
    var sh = List[Int]()
    for i in range(len(in_shape)):
        sh.append(in_shape[i])
    return Tensor(dev^, sh^, STDtype.F32)


def cast_backward(grad_out: Tensor, ctx: DeviceContext) raises -> Tensor:
    """d_x = d_y (identity). In the F32 gate the cast is F32->F32, so backward is
    a straight clone of the upstream grad."""
    _require_f32(grad_out, "cast_backward")
    var dev = ctx.enqueue_create_buffer[DType.uint8](grad_out.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=grad_out.buf)
    return Tensor(dev^, grad_out.shape(), STDtype.F32)


# ══════════════════════════════════════════════════════════════════════════════
# PERMUTE / TRANSPOSE — apply the INVERSE permutation to the upstream grad.
# Kernel = generic gather (mirrors tensor_algebra._permute_kernel): one thread
# per OUTPUT (= d_x) element; map each d_x axis to its source axis in d_y.
# ══════════════════════════════════════════════════════════════════════════════
def _permute_gather_k(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    od0: Int, od1: Int, od2: Int, od3: Int, od4: Int, od5: Int,
    ss0: Int, ss1: Int, ss2: Int, ss3: Int, ss4: Int, ss5: Int,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var rem = idx
        var o5 = rem % od5; rem //= od5
        var o4 = rem % od4; rem //= od4
        var o3 = rem % od3; rem //= od3
        var o2 = rem % od2; rem //= od2
        var o1 = rem % od1; rem //= od1
        var o0 = rem % od0
        var soff = o0*ss0 + o1*ss1 + o2*ss2 + o3*ss3 + o4*ss4 + o5*ss5
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[soff]))


def _permute_materialize(
    src: Tensor, perm: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """Materialize src.permute(perm) contiguous. Identical index math to
    tensor_algebra.permute (output axis k <- input axis perm[k])."""
    var xshape = src.shape()
    var rank = len(xshape)
    if rank > _MAXRANK:
        raise Error(String("permute_backward: rank > ") + String(_MAXRANK))
    if len(perm) != rank:
        raise Error("permute_backward: perm length must equal rank")
    var src_stride = List[Int]()
    for _ in range(rank):
        src_stride.append(0)
    var acc = 1
    for ii in range(rank):
        var i = rank - 1 - ii
        src_stride[i] = acc
        acc *= xshape[i]
    var oshape = List[Int]()
    var od = IndexList[_MAXRANK]()
    var ss = IndexList[_MAXRANK]()
    for i in range(_MAXRANK):
        od[i] = 1
        ss[i] = 0
    for k in range(rank):
        oshape.append(xshape[perm[k]])
    var pad = _MAXRANK - rank
    for k in range(rank):
        od[pad + k] = xshape[perm[k]]
        ss[pad + k] = src_stride[perm[k]]

    var n = src.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        src.buf.unsafe_ptr().bitcast[Float32](), rl)
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_permute_gather_k, _permute_gather_k](
        X, O, od[0], od[1], od[2], od[3], od[4], od[5],
        ss[0], ss[1], ss[2], ss[3], ss[4], ss[5], n,
        grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, oshape^, STDtype.F32)


def permute_backward(
    grad_out: Tensor, perm: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """d_x = permute(d_y, inverse(perm)). `perm` is the FORWARD permutation
    (output axis k <- input axis perm[k]); inverse `inv` satisfies inv[perm[k]]=k."""
    _require_f32(grad_out, "permute_backward")
    var rank = len(perm)
    var inv = List[Int]()
    for _ in range(rank):
        inv.append(0)
    for k in range(rank):
        var p = perm[k]
        if p < 0 or p >= rank:
            raise Error("permute_backward: axis out of range")
        inv[p] = k
    return _permute_materialize(grad_out, inv, ctx)


def transpose_backward(
    grad_out: Tensor, dim0: Int, dim1: Int, ctx: DeviceContext
) raises -> Tensor:
    """d_x = transpose(d_y, dim0, dim1) (transpose is its own inverse)."""
    _require_f32(grad_out, "transpose_backward")
    var rank = len(grad_out.shape())
    if dim0 < 0 or dim0 >= rank or dim1 < 0 or dim1 >= rank:
        raise Error("transpose_backward: axis out of range")
    var perm = List[Int]()
    for i in range(rank):
        perm.append(i)
    perm[dim0] = dim1
    perm[dim1] = dim0
    return _permute_materialize(grad_out, perm, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# CAT / SPLIT — split d_y back into pieces along `dim` (cat bwd) / concat the
# piece-grads (split bwd). Pure D2D block copies (outer/inner block layout).
# ══════════════════════════════════════════════════════════════════════════════
# gather kernel for _slice_along: one thread per OUTPUT element; map the output
# (narrowed) multi-index back to the source offset. outer = prod(dims<dim),
# inner = prod(dims>dim).
def _slice_gather_k(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # src [outer*in_dim*inner]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # dst [outer*length*inner]
    outer: Int, in_dim: Int, inner: Int, length: Int, start: Int,
):
    var t = Int(global_idx.x)
    var total = outer * length * inner
    if t < total:
        var ic = t % inner
        var rem = t // inner
        var ld = rem % length
        var oc = rem // length
        var soff = (oc * in_dim + (start + ld)) * inner + ic
        o[t] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[soff]))


def _slice_along(
    x: Tensor, dim: Int, start: Int, length: Int, ctx: DeviceContext
) raises -> Tensor:
    """Narrow x along `dim` to [start, start+length) -> contiguous F32 copy.
    Kernel-based gather (avoids create_sub_buffer, which fails to compile through
    this module's import chain in Mojo 1.0.0b1)."""
    var xshape = x.shape()
    var rank = len(xshape)
    if dim < 0 or dim >= rank:
        raise Error("slice: dim out of range")
    if start < 0 or length < 0 or start + length > xshape[dim]:
        raise Error("slice: range out of bounds")
    var outer = 1
    for ax in range(dim):
        outer *= xshape[ax]
    var inner = 1
    for ax in range(dim + 1, rank):
        inner *= xshape[ax]
    var in_dim = xshape[dim]
    var oshape = List[Int]()
    for ax in range(rank):
        if ax == dim:
            oshape.append(length)
        else:
            oshape.append(xshape[ax])
    var out_n = outer * length * inner
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_n * 4)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), x_rl)
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_slice_gather_k, _slice_gather_k](
        X, O, outer, in_dim, inner, length, start,
        grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, oshape^, STDtype.F32)


def cat_backward(
    grad_out: Tensor, size0: Int, size1: Int, axis: Int, ctx: DeviceContext
) raises -> CatGrads2:
    """Split d_y back into the two original pieces along `axis`.

    `size0`/`size1` are the per-input extents along `axis` (sum to
    d_y.shape[axis]). Returns CatGrads2{d_0, d_1}."""
    _require_f32(grad_out, "cat_backward")
    var sh = grad_out.shape()
    var rank = len(sh)
    if axis < 0 or axis >= rank:
        raise Error("cat_backward: axis out of range")
    if size0 + size1 != sh[axis]:
        raise Error(
            String("cat_backward: size0+size1 ") + String(size0 + size1)
            + " != grad_out.shape[axis] " + String(sh[axis]))
    var d0 = _slice_along(grad_out, axis, 0, size0, ctx)
    var d1 = _slice_along(grad_out, axis, size0, size1, ctx)
    return CatGrads2(d0^, d1^)


# split bwd = concat the piece-grads. Implemented with a per-output-element
# gather kernel (one kernel per piece, writing its slab into the joined buffer).
# This dodges the variadic + create_sub_buffer borrow combination that fails to
# compile in Mojo 1.0.0b1 (see split_backward3 note). The 2-input split is the
# diffusion-common case (gate/up, img/txt unsplit).
def _concat_slab_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # piece [outer*pdim*inner]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # joined [outer*sum_dim*inner]
    outer: Int, pdim: Int, inner: Int, sum_dim: Int, col_off: Int,
):
    var t = Int(global_idx.x)
    var total = outer * pdim * inner
    if t < total:
        var ic = t % inner
        var rem = t // inner
        var pd = rem % pdim
        var oc = rem // pdim
        var ooff = (oc * sum_dim + (col_off + pd)) * inner + ic
        o[ooff] = rebind[o.element_type](rebind[Scalar[DType.float32]](g[t]))


def _write_slab(
    piece: Tensor, axis: Int, col_off: Int, sum_dim: Int,
    out_buf: DeviceBuffer[DType.uint8], ctx: DeviceContext,
) raises:
    var ps = piece.shape()
    var rank = len(ps)
    var outer = 1
    for ax in range(axis):
        outer *= ps[ax]
    var inner = 1
    for ax in range(axis + 1, rank):
        inner *= ps[ax]
    var pdim = ps[axis]
    var pn = piece.numel()
    var out_n = outer * sum_dim * inner
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](pn))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        piece.buf.unsafe_ptr().bitcast[Float32](), g_rl)
    var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
    var grid = (pn + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_concat_slab_k, _concat_slab_k](
        g, o, outer, pdim, inner, sum_dim, col_off,
        grid_dim=grid, block_dim=_BLOCK)


def split_backward(
    grad_0: Tensor, grad_1: Tensor, axis: Int, ctx: DeviceContext
) raises -> Tensor:
    """Concatenate the two per-piece grads back along `axis` -> d_y.

    The 2-piece case (the diffusion-common split). For N pieces, call
    `_write_slab` per piece into a pre-sized joined buffer with running col_off."""
    _require_f32(grad_0, "split_backward")
    _require_f32(grad_1, "split_backward")
    var s0 = grad_0.shape()
    var s1 = grad_1.shape()
    var rank = len(s0)
    if axis < 0 or axis >= rank:
        raise Error("split_backward: axis out of range")
    if len(s1) != rank:
        raise Error("split_backward: rank mismatch")
    for ax in range(rank):
        if ax != axis and s0[ax] != s1[ax]:
            raise Error("split_backward: dim mismatch")
    var sum_dim = s0[axis] + s1[axis]
    var oshape = List[Int]()
    for ax in range(rank):
        if ax == axis:
            oshape.append(sum_dim)
        else:
            oshape.append(s0[ax])
    var inner = 1
    for ax in range(axis + 1, rank):
        inner *= s0[ax]
    var outer = 1
    for ax in range(axis):
        outer *= s0[ax]
    var out_n = outer * sum_dim * inner
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_n * 4)
    _write_slab(grad_0, axis, 0, sum_dim, out_buf, ctx)
    _write_slab(grad_1, axis, s0[axis], sum_dim, out_buf, ctx)
    return Tensor(out_buf^, oshape^, STDtype.F32)


# ══════════════════════════════════════════════════════════════════════════════
# SLICE — scatter d_y into a zero tensor of the full forward input shape at the
# slice location. One thread per OUTPUT (= d_x, full_shape) element: if the
# element falls inside [start, start+len) along `dim`, copy the corresponding d_y
# value, else write 0. Pure gather kernel (no create_sub_buffer borrow churn).
# outer = prod(dims < dim), inner = prod(dims > dim).
# ══════════════════════════════════════════════════════════════════════════════
def _slice_scatter_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # d_y [outer*length*inner]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # d_x [outer*full_dim*inner]
    outer: Int, full_dim: Int, inner: Int, length: Int, start: Int,
):
    var t = Int(global_idx.x)
    var total = outer * full_dim * inner
    if t < total:
        var ic = t % inner
        var rem = t // inner
        var fd = rem % full_dim
        var oc = rem // full_dim
        if fd >= start and fd < start + length:
            var ld = fd - start
            var goff = (oc * length + ld) * inner + ic
            o[t] = rebind[o.element_type](rebind[Scalar[DType.float32]](g[goff]))
        else:
            o[t] = rebind[o.element_type](Float32(0.0))


def slice_backward(
    grad_out: Tensor, full_shape: List[Int], dim: Int, start: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Scatter d_y into zeros(full_shape) at [start, start+len) along `dim`.

    `full_shape` is the forward INPUT shape; the slice length is taken from
    grad_out.shape[dim]."""
    _require_f32(grad_out, "slice_backward")
    var gsh = grad_out.shape()
    var rank = len(full_shape)
    if len(gsh) != rank:
        raise Error("slice_backward: grad rank != full_shape rank")
    if dim < 0 or dim >= rank:
        raise Error("slice_backward: dim out of range")
    var length = gsh[dim]
    if start < 0 or start + length > full_shape[dim]:
        raise Error("slice_backward: slice range out of bounds")
    for ax in range(rank):
        if ax != dim and gsh[ax] != full_shape[ax]:
            raise Error("slice_backward: non-slice dim mismatch")
    var n = _numel(full_shape)
    var outer = 1
    for ax in range(dim):
        outer *= full_shape[ax]
    var inner = 1
    for ax in range(dim + 1, rank):
        inner *= full_shape[ax]
    var full_dim = full_shape[dim]
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](outer * length * inner))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl)
    var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_slice_scatter_k, _slice_scatter_k](
        g, o, outer, full_dim, inner, length, start,
        grid_dim=grid, block_dim=_BLOCK)
    var sh = List[Int]()
    for i in range(len(full_shape)):
        sh.append(full_shape[i])
    return Tensor(out_buf^, sh^, STDtype.F32)


# ══════════════════════════════════════════════════════════════════════════════
# BROADCAST — sum-reduce d_y over the dims that were broadcast back to in_shape.
# One thread per OUTPUT (= d_x, in_shape) element; the thread loops over the
# broadcast multiplicity and accumulates in F32.
# ══════════════════════════════════════════════════════════════════════════════
def _broadcast_sum_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    id0: Int, id1: Int, id2: Int, id3: Int, id4: Int, id5: Int,   # in (d_x) dims
    od0: Int, od1: Int, od2: Int, od3: Int, od4: Int, od5: Int,   # out (d_y) dims
    os0: Int, os1: Int, os2: Int, os3: Int, os4: Int, os5: Int,   # d_y strides
    n_in: Int,
):
    var idx = Int(global_idx.x)
    if idx < n_in:
        var rem = idx
        var x5 = rem % id5; rem //= id5
        var x4 = rem % id4; rem //= id4
        var x3 = rem % id3; rem //= id3
        var x2 = rem % id2; rem //= id2
        var x1 = rem % id1; rem //= id1
        var x0 = rem % id0
        var acc: Float32 = 0.0
        var r0 = od0 if id0 == 1 else 1
        var r1 = od1 if id1 == 1 else 1
        var r2 = od2 if id2 == 1 else 1
        var r3 = od3 if id3 == 1 else 1
        var r4 = od4 if id4 == 1 else 1
        var r5 = od5 if id5 == 1 else 1
        for b0 in range(r0):
            var c0 = b0 if id0 == 1 else x0
            for b1 in range(r1):
                var c1 = b1 if id1 == 1 else x1
                for b2 in range(r2):
                    var c2 = b2 if id2 == 1 else x2
                    for b3 in range(r3):
                        var c3 = b3 if id3 == 1 else x3
                        for b4 in range(r4):
                            var c4 = b4 if id4 == 1 else x4
                            for b5 in range(r5):
                                var c5 = b5 if id5 == 1 else x5
                                var off = (c0*os0 + c1*os1 + c2*os2
                                           + c3*os3 + c4*os4 + c5*os5)
                                acc += rebind[Scalar[DType.float32]](g[off])
        o[idx] = rebind[o.element_type](acc)


def broadcast_backward(
    grad_out: Tensor, in_shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """d_x = sum d_y over the dims that were broadcast (NumPy right-aligned)."""
    _require_f32(grad_out, "broadcast_backward")
    var osh = grad_out.shape()
    var orank = len(osh)
    var irank = len(in_shape)
    if orank > _MAXRANK:
        raise Error(String("broadcast_backward: out rank > ") + String(_MAXRANK))
    if irank > orank:
        raise Error("broadcast_backward: in rank > out rank")
    var idims = IndexList[_MAXRANK]()
    var odims = IndexList[_MAXRANK]()
    for i in range(_MAXRANK):
        idims[i] = 1
        odims[i] = 1
    for i in range(irank):
        idims[_MAXRANK - irank + i] = in_shape[i]
    for i in range(orank):
        odims[_MAXRANK - orank + i] = osh[i]
    for i in range(_MAXRANK):
        if idims[i] != 1 and idims[i] != odims[i]:
            raise Error("broadcast_backward: incompatible dim (not 1, not equal)")
    var ostr = IndexList[_MAXRANK]()
    var acc = 1
    for ii in range(_MAXRANK):
        var i = _MAXRANK - 1 - ii
        ostr[i] = acc
        acc *= odims[i]
    var n_in = _numel(in_shape)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n_in * 4)
    var n_out = grad_out.numel()
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_out))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_in))
    var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl)
    var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
    var grid = (n_in + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_broadcast_sum_k, _broadcast_sum_k](
        g, o,
        idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
        odims[0], odims[1], odims[2], odims[3], odims[4], odims[5],
        ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
        n_in, grid_dim=grid, block_dim=_BLOCK)
    var sh = List[Int]()
    for i in range(len(in_shape)):
        sh.append(in_shape[i])
    return Tensor(out_buf^, sh^, STDtype.F32)


# ══════════════════════════════════════════════════════════════════════════════
# REPEAT — sum the tiled copies. Torch `Tensor.repeat` tiling: out_dim[i] =
# in_dim[i] * reps[i], copy r of axis i at output index in_idx + r*in_dim.
# ══════════════════════════════════════════════════════════════════════════════
def _repeat_sum_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    id0: Int, id1: Int, id2: Int, id3: Int, id4: Int, id5: Int,   # in dims
    rp0: Int, rp1: Int, rp2: Int, rp3: Int, rp4: Int, rp5: Int,   # reps
    os0: Int, os1: Int, os2: Int, os3: Int, os4: Int, os5: Int,   # out strides
    n_in: Int,
):
    var idx = Int(global_idx.x)
    if idx < n_in:
        var rem = idx
        var x5 = rem % id5; rem //= id5
        var x4 = rem % id4; rem //= id4
        var x3 = rem % id3; rem //= id3
        var x2 = rem % id2; rem //= id2
        var x1 = rem % id1; rem //= id1
        var x0 = rem % id0
        var acc: Float32 = 0.0
        for r0 in range(rp0):
            var c0 = x0 + r0 * id0
            for r1 in range(rp1):
                var c1 = x1 + r1 * id1
                for r2 in range(rp2):
                    var c2 = x2 + r2 * id2
                    for r3 in range(rp3):
                        var c3 = x3 + r3 * id3
                        for r4 in range(rp4):
                            var c4 = x4 + r4 * id4
                            for r5 in range(rp5):
                                var c5 = x5 + r5 * id5
                                var off = (c0*os0 + c1*os1 + c2*os2
                                           + c3*os3 + c4*os4 + c5*os5)
                                acc += rebind[Scalar[DType.float32]](g[off])
        o[idx] = rebind[o.element_type](acc)


def repeat_backward(
    grad_out: Tensor, in_shape: List[Int], repeats: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    """d_x = sum of the `repeats`-tiled copies, back to in_shape."""
    _require_f32(grad_out, "repeat_backward")
    var irank = len(in_shape)
    if len(repeats) != irank:
        raise Error("repeat_backward: repeats length != in_shape length")
    if irank > _MAXRANK:
        raise Error(String("repeat_backward: rank > ") + String(_MAXRANK))
    var osh = grad_out.shape()
    if len(osh) != irank:
        raise Error("repeat_backward: grad rank != in rank")
    for i in range(irank):
        if osh[i] != in_shape[i] * repeats[i]:
            raise Error("repeat_backward: out dim != in*reps")
    var idims = IndexList[_MAXRANK]()
    var reps = IndexList[_MAXRANK]()
    var odims = IndexList[_MAXRANK]()
    for i in range(_MAXRANK):
        idims[i] = 1
        reps[i] = 1
        odims[i] = 1
    var pad = _MAXRANK - irank
    for i in range(irank):
        idims[pad + i] = in_shape[i]
        reps[pad + i] = repeats[i]
        odims[pad + i] = osh[i]
    var ostr = IndexList[_MAXRANK]()
    var acc = 1
    for ii in range(_MAXRANK):
        var i = _MAXRANK - 1 - ii
        ostr[i] = acc
        acc *= odims[i]
    var n_in = _numel(in_shape)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n_in * 4)
    var n_out = grad_out.numel()
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_out))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_in))
    var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl)
    var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
    var grid = (n_in + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_repeat_sum_k, _repeat_sum_k](
        g, o,
        idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
        reps[0], reps[1], reps[2], reps[3], reps[4], reps[5],
        ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
        n_in, grid_dim=grid, block_dim=_BLOCK)
    var sh = List[Int]()
    for i in range(len(in_shape)):
        sh.append(in_shape[i])
    return Tensor(out_buf^, sh^, STDtype.F32)


# ══════════════════════════════════════════════════════════════════════════════
# WHERE — route d_y to lhs where cond!=0, to rhs elsewhere. `cond` is an F32 mask.
# ══════════════════════════════════════════════════════════════════════════════
def _where_bwd_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cond: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    da: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    db: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.float32]](g[i])
        var cv = rebind[Scalar[DType.float32]](cond[i])
        var is_a = cv != Float32(0.0)
        da[i] = rebind[da.element_type](gv if is_a else Float32(0.0))
        db[i] = rebind[db.element_type](Float32(0.0) if is_a else gv)


def where_backward(
    grad_out: Tensor, cond: Tensor, ctx: DeviceContext
) raises -> WhereGrads:
    """d_a = cond ? d_y : 0 ; d_b = cond ? 0 : d_y. `cond` is an F32 mask."""
    _require_f32(grad_out, "where_backward")
    _require_f32(cond, "where_backward")
    var n = grad_out.numel()
    if cond.numel() != n:
        raise Error("where_backward: grad/cond numel mismatch")
    var da_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var db_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
    var c = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        cond.buf.unsafe_ptr().bitcast[Float32](), rl)
    var da = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        da_buf.unsafe_ptr().bitcast[Float32](), rl)
    var db = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        db_buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_where_bwd_k, _where_bwd_k](
        g, c, da, db, n, grid_dim=grid, block_dim=_BLOCK)
    return WhereGrads(
        Tensor(da_buf^, grad_out.shape(), STDtype.F32),
        Tensor(db_buf^, grad_out.shape(), STDtype.F32))


# ══════════════════════════════════════════════════════════════════════════════
# CLAMP — grad passes where lo <= x <= hi, else 0.
# ══════════════════════════════════════════════════════════════════════════════
def _clamp_bwd_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    lo: Float32, hi: Float32, n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var xv = rebind[Scalar[DType.float32]](x[i])
        var gv = rebind[Scalar[DType.float32]](g[i])
        var pass_through = (xv >= lo) and (xv <= hi)
        o[i] = rebind[o.element_type](gv if pass_through else Float32(0.0))


def clamp_backward(
    grad_out: Tensor, x: Tensor, lo: Float32, hi: Float32, ctx: DeviceContext
) raises -> Tensor:
    """d_x = (lo <= x <= hi) ? d_y : 0. `x` is the forward INPUT."""
    _require_f32(grad_out, "clamp_backward")
    _require_f32(x, "clamp_backward")
    var n = x.numel()
    if grad_out.numel() != n:
        raise Error("clamp_backward: grad/x numel mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
    var xt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl)
    var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_clamp_bwd_k, _clamp_bwd_k](
        g, xt, o, lo, hi, n, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, x.shape(), STDtype.F32)


# ══════════════════════════════════════════════════════════════════════════════
# MAXIMUM / MINIMUM — grad to the arg that won. Ties go to `a` (flame-core uses
# a>=b for max, a<=b for min; autograd.rs:6114/6134).
# ══════════════════════════════════════════════════════════════════════════════
def _maxmin_bwd_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    da: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    db: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    is_max: Int, n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.float32]](g[i])
        var av = rebind[Scalar[DType.float32]](a[i])
        var bv = rebind[Scalar[DType.float32]](b[i])
        var a_wins: Bool
        if is_max == 1:
            a_wins = av >= bv
        else:
            a_wins = av <= bv
        da[i] = rebind[da.element_type](gv if a_wins else Float32(0.0))
        db[i] = rebind[db.element_type](Float32(0.0) if a_wins else gv)


def _maxmin_backward(
    grad_out: Tensor, a: Tensor, b: Tensor, is_max: Int, ctx: DeviceContext
) raises -> BinaryGrads:
    _require_f32(grad_out, "maxmin_backward")
    _require_f32(a, "maxmin_backward")
    _require_f32(b, "maxmin_backward")
    var n = a.numel()
    if b.numel() != n or grad_out.numel() != n:
        raise Error("maxmin_backward: numel mismatch")
    var da_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var db_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
    var at = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[Float32](), rl)
    var bt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[Float32](), rl)
    var da = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        da_buf.unsafe_ptr().bitcast[Float32](), rl)
    var db = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        db_buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_maxmin_bwd_k, _maxmin_bwd_k](
        g, at, bt, da, db, is_max, n, grid_dim=grid, block_dim=_BLOCK)
    return BinaryGrads(
        Tensor(da_buf^, a.shape(), STDtype.F32),
        Tensor(db_buf^, b.shape(), STDtype.F32))


def maximum_backward(
    grad_out: Tensor, a: Tensor, b: Tensor, ctx: DeviceContext
) raises -> BinaryGrads:
    """d_a = (a>=b)?d_y:0 ; d_b = (a<b)?d_y:0. a/b are the forward INPUTS."""
    return _maxmin_backward(grad_out, a, b, 1, ctx)


def minimum_backward(
    grad_out: Tensor, a: Tensor, b: Tensor, ctx: DeviceContext
) raises -> BinaryGrads:
    """d_a = (a<=b)?d_y:0 ; d_b = (a>b)?d_y:0. a/b are the forward INPUTS."""
    return _maxmin_backward(grad_out, a, b, 0, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# INDEX_SELECT — scatter-ADD d_y rows back to the selected indices. Repeated
# indices accumulate (flame-core scatter_add; autograd.rs:5869). Gated case:
# dim==0 row-select on a rank-2 table [V, D].
# ══════════════════════════════════════════════════════════════════════════════
def _index_select_bwd_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],      # d_y [N, D]
    idx: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],      # indices [N]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],      # d_x [V, D]
    N: Int, D: Int, V: Int,
):
    # one thread per d_x element (row v, col d): sum d_y[n, d] over all n with
    # idx[n] == v.
    var t = Int(global_idx.x)
    if t < V * D:
        var v = t // D
        var d = t % D
        var acc: Float32 = 0.0
        for n in range(N):
            if Int(rebind[Scalar[DType.int32]](idx[n])) == v:
                acc += rebind[Scalar[DType.float32]](g[n * D + d])
        o[t] = rebind[o.element_type](acc)


def index_select_backward(
    grad_out: Tensor, indices: List[Int], dim: Int, in_shape: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    """Scatter-ADD d_y rows back to the selected indices -> zeros(in_shape).

    Gated case: dim==0 row-select on a rank-2 table [V, D]. grad_out is [N, D]
    with N == len(indices); result is [V, D] (in_shape), F32. Repeated indices
    accumulate."""
    _require_f32(grad_out, "index_select_backward")
    if dim != 0:
        raise Error("index_select_backward: only dim==0 gated")
    if len(in_shape) != 2:
        raise Error("index_select_backward: in_shape must be rank-2 [V, D]")
    var V = in_shape[0]
    var D = in_shape[1]
    var N = len(indices)
    var gsh = grad_out.shape()
    if len(gsh) != 2 or gsh[0] != N or gsh[1] != D:
        raise Error("index_select_backward: grad_out shape != [len(idx), D]")
    var id_host = ctx.enqueue_create_host_buffer[DType.int32](N)
    var ip = id_host.unsafe_ptr()
    for i in range(N):
        var r = indices[i]
        if r < 0 or r >= V:
            raise Error(String("index_select_backward: index ") + String(r)
                        + " out of range [0, " + String(V) + ")")
        ip[i] = Int32(r)
    var id_dev = ctx.enqueue_create_buffer[DType.int32](N)
    ctx.enqueue_copy(dst_buf=id_dev, src_buf=id_host)

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](V * D * 4)
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N * D))
    var id_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](V * D))
    var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl)
    var IDS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        id_dev.unsafe_ptr(), id_rl)
    var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
    var grid = (V * D + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_index_select_bwd_k, _index_select_bwd_k](
        g, IDS, o, N, D, V, grid_dim=grid, block_dim=_BLOCK)
    var sh = List[Int]()
    sh.append(V); sh.append(D)
    return Tensor(out_buf^, sh^, STDtype.F32)
