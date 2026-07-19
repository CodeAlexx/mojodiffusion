# ops/shape_backward.mojo — BACKWARD for the Tier-0 shape/structural arms:
#   Cat, Split, Slice, Reshape, Transpose, Permute, Broadcast, Repeat, Where,
#   Clamp, Maximum, Minimum, Cast, IndexSelect.
#
# Phase T1 of FULL_PORT_TRAINING_PLAN.md (Tier 0 backward arms). The backward of
# the shape ops is almost entirely grad-ROUTING — no learnable parameters, just
# moving / summing / masking the upstream gradient. Storage dtype is preserved;
# reductions and mask comparisons may cast to F32 internally and cast gradients
# back to the incoming storage dtype.
#
# Conventions mirror ops/reduce_backward.mojo + ops/linalg_backward.mojo (the
# proven sibling templates):
#   * device buffers are DType.uint8 (Tensor's storage), bitcast to the tensor
#     storage dtype at the LayoutTensor boundary,
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
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.ops.tensor_algebra import (
    reshape as _ta_reshape,
    permute as _ta_permute,
    transpose as _ta_transpose,
    concat as _ta_concat,
    slice as _ta_slice,
    zeros_device as _ta_zeros_device,
)


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


# ══════════════════════════════════════════════════════════════════════════════
# RESHAPE / CAST — pure metadata / identity grad (a D2D clone with new/same shape)
# ══════════════════════════════════════════════════════════════════════════════
def reshape_backward(
    grad_out: Tensor, in_shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """d_x = d_y reshaped back to the forward INPUT shape (data unchanged)."""
    var n = _numel(in_shape)
    if n != grad_out.numel():
        raise Error(
            String("reshape_backward: numel mismatch ")
            + String(n) + " != " + String(grad_out.numel()))
    var sh = List[Int]()
    for i in range(len(in_shape)):
        sh.append(in_shape[i])
    return _ta_reshape(grad_out, sh^, ctx)


def cast_backward(
    grad_out: Tensor, ctx: DeviceContext, input_dtype: STDtype = STDtype.BOOL
) raises -> Tensor:
    """d_x = d_y (identity), cast back to the forward input storage dtype.

    Default preserves historical behavior (`grad_out.dtype()`). Pass
    `input_dtype` for forward casts that widened BF16/F16 storage to an F32
    compute workspace.
    """
    var out_dtype = input_dtype
    if out_dtype == STDtype.BOOL:
        out_dtype = grad_out.dtype()
    if out_dtype == grad_out.dtype():
        return grad_out.clone(ctx)
    if not (
        out_dtype == STDtype.F32
        or out_dtype == STDtype.BF16
        or out_dtype == STDtype.F16
    ):
        raise Error("cast_backward: unsupported input dtype")
    from serenitymojo.ops.cast import cast_tensor
    return cast_tensor(grad_out, out_dtype, ctx, False)


# ══════════════════════════════════════════════════════════════════════════════
# PERMUTE / TRANSPOSE — apply the INVERSE permutation to the upstream grad.
# ══════════════════════════════════════════════════════════════════════════════
def permute_backward(
    grad_out: Tensor, perm: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """d_x = permute(d_y, inverse(perm)). `perm` is the FORWARD permutation
    (output axis k <- input axis perm[k]); inverse `inv` satisfies inv[perm[k]]=k."""
    var rank = len(perm)
    var inv = List[Int]()
    for _ in range(rank):
        inv.append(0)
    for k in range(rank):
        var p = perm[k]
        if p < 0 or p >= rank:
            raise Error("permute_backward: axis out of range")
        inv[p] = k
    return _ta_permute(grad_out, inv^, ctx)


def transpose_backward(
    grad_out: Tensor, dim0: Int, dim1: Int, ctx: DeviceContext
) raises -> Tensor:
    """d_x = transpose(d_y, dim0, dim1) (transpose is its own inverse)."""
    var rank = len(grad_out.shape())
    if dim0 < 0 or dim0 >= rank or dim1 < 0 or dim1 >= rank:
        raise Error("transpose_backward: axis out of range")
    return _ta_transpose(grad_out, dim0, dim1, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# CAT / SPLIT — split d_y back into pieces along `dim` (cat bwd) / concat the
# piece-grads (split bwd). Pure D2D block copies (outer/inner block layout).
# ══════════════════════════════════════════════════════════════════════════════
def cat_backward(
    grad_out: Tensor, size0: Int, size1: Int, axis: Int, ctx: DeviceContext
) raises -> CatGrads2:
    """Split d_y back into the two original pieces along `axis`.

    `size0`/`size1` are the per-input extents along `axis` (sum to
    d_y.shape[axis]). Returns CatGrads2{d_0, d_1}."""
    var sh = grad_out.shape()
    var rank = len(sh)
    if axis < 0 or axis >= rank:
        raise Error("cat_backward: axis out of range")
    if size0 + size1 != sh[axis]:
        raise Error(
            String("cat_backward: size0+size1 ") + String(size0 + size1)
            + " != grad_out.shape[axis] " + String(sh[axis]))
    var d0 = _ta_slice(grad_out, axis, 0, size0, ctx)
    var d1 = _ta_slice(grad_out, axis, size0, size1, ctx)
    return CatGrads2(d0^, d1^)


# split bwd = concat the piece-grads. Implemented with a per-output-element
# gather kernel (one kernel per piece, writing its slab into the joined buffer).
# This dodges the variadic + create_sub_buffer borrow combination that fails to
# compile in Mojo 1.0.0b1 (see split_backward3 note). The 2-input split is the
# diffusion-common case (gate/up, img/txt unsplit).
def _concat_slab_k[dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # piece [outer*pdim*inner]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # joined [outer*sum_dim*inner]
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
        o[ooff] = rebind[o.element_type](rebind[Scalar[dtype]](g[t]))


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
    var grid = (pn + _BLOCK - 1) // _BLOCK
    var dt = piece.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            piece.buf.unsafe_ptr().bitcast[Float32](), g_rl)
        var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
        ctx.enqueue_function[
            _concat_slab_k[DType.float32], _concat_slab_k[DType.float32]
        ](g, o, outer, pdim, inner, sum_dim, col_off,
          grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var g = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            piece.buf.unsafe_ptr().bitcast[BFloat16](), g_rl)
        var o = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
        ctx.enqueue_function[
            _concat_slab_k[DType.bfloat16], _concat_slab_k[DType.bfloat16]
        ](g, o, outer, pdim, inner, sum_dim, col_off,
          grid_dim=grid, block_dim=_BLOCK)
    else:
        var g = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            piece.buf.unsafe_ptr().bitcast[Float16](), g_rl)
        var o = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl)
        ctx.enqueue_function[
            _concat_slab_k[DType.float16], _concat_slab_k[DType.float16]
        ](g, o, outer, pdim, inner, sum_dim, col_off,
          grid_dim=grid, block_dim=_BLOCK)


def split_backward(
    grad_0: Tensor, grad_1: Tensor, axis: Int, ctx: DeviceContext
) raises -> Tensor:
    """Concatenate the two per-piece grads back along `axis` -> d_y.

    The 2-piece case (the diffusion-common split). For N pieces, call
    `_write_slab` per piece into a pre-sized joined buffer with running col_off."""
    if grad_0.dtype() != grad_1.dtype():
        raise Error("split_backward: grad dtype mismatch")
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
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * grad_0.dtype().byte_size()
    )
    _write_slab(grad_0, axis, 0, sum_dim, out_buf, ctx)
    _write_slab(grad_1, axis, s0[axis], sum_dim, out_buf, ctx)
    return Tensor(out_buf^, oshape^, grad_0.dtype())


# ══════════════════════════════════════════════════════════════════════════════
# SLICE — scatter d_y into a zero tensor of the full forward input shape at the
# slice location. One thread per OUTPUT (= d_x, full_shape) element: if the
# element falls inside [start, start+len) along `dim`, copy the corresponding d_y
# value, else write 0. Pure gather kernel (no create_sub_buffer borrow churn).
# outer = prod(dims < dim), inner = prod(dims > dim).
# ══════════════════════════════════════════════════════════════════════════════
def slice_backward(
    grad_out: Tensor, full_shape: List[Int], dim: Int, start: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Scatter d_y into zeros(full_shape) at [start, start+len) along `dim`.

    `full_shape` is the forward INPUT shape; the slice length is taken from
    grad_out.shape[dim]."""
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
    var full_dim = full_shape[dim]
    var end = start + length
    if start == 0 and end == full_dim:
        return grad_out.clone(ctx)
    var before_shape = List[Int]()
    var after_shape = List[Int]()
    for i in range(len(full_shape)):
        if i == dim:
            before_shape.append(start)
            after_shape.append(full_dim - end)
        else:
            before_shape.append(full_shape[i])
            after_shape.append(full_shape[i])
    if start == 0:
        var after = _ta_zeros_device(after_shape^, grad_out.dtype(), ctx)
        return _ta_concat(dim, ctx, grad_out, after)
    if end == full_dim:
        var before = _ta_zeros_device(before_shape^, grad_out.dtype(), ctx)
        return _ta_concat(dim, ctx, before, grad_out)
    var before = _ta_zeros_device(before_shape^, grad_out.dtype(), ctx)
    var after = _ta_zeros_device(after_shape^, grad_out.dtype(), ctx)
    return _ta_concat(dim, ctx, before, grad_out, after)


# ══════════════════════════════════════════════════════════════════════════════
# BROADCAST — sum-reduce d_y over the dims that were broadcast back to in_shape.
# One thread per OUTPUT (= d_x, in_shape) element; the thread loops over the
# broadcast multiplicity and accumulates in F32.
# ══════════════════════════════════════════════════════════════════════════════
def _broadcast_sum_k[dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
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
                                acc += rebind[Scalar[dtype]](g[off]).cast[
                                    DType.float32
                                ]()
        o[idx] = rebind[o.element_type](acc.cast[dtype]())


def broadcast_backward(
    grad_out: Tensor, in_shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """d_x = sum d_y over the dims that were broadcast (NumPy right-aligned)."""
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
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n_in * grad_out.dtype().byte_size()
    )
    var n_out = grad_out.numel()
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_out))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_in))
    var grid = (n_in + _BLOCK - 1) // _BLOCK
    var dt = grad_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl)
        var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
        ctx.enqueue_function[
            _broadcast_sum_k[DType.float32], _broadcast_sum_k[DType.float32]
        ](
            g, o,
            idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
            odims[0], odims[1], odims[2], odims[3], odims[4], odims[5],
            ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
            n_in, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var g = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), g_rl)
        var o = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
        ctx.enqueue_function[
            _broadcast_sum_k[DType.bfloat16], _broadcast_sum_k[DType.bfloat16]
        ](
            g, o,
            idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
            odims[0], odims[1], odims[2], odims[3], odims[4], odims[5],
            ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
            n_in, grid_dim=grid, block_dim=_BLOCK)
    else:
        var g = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), g_rl)
        var o = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl)
        ctx.enqueue_function[
            _broadcast_sum_k[DType.float16], _broadcast_sum_k[DType.float16]
        ](
            g, o,
            idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
            odims[0], odims[1], odims[2], odims[3], odims[4], odims[5],
            ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
            n_in, grid_dim=grid, block_dim=_BLOCK)
    var sh = List[Int]()
    for i in range(len(in_shape)):
        sh.append(in_shape[i])
    return Tensor(out_buf^, sh^, grad_out.dtype())


# ── StepSlab variant for the autograd_v2 capture path (contract C8) ────────────
# Byte-identical to `broadcast_backward` except the output buffer comes from
# `slab.alloc(nbytes)` instead of `ctx.enqueue_create_buffer[DType.uint8](nbytes)`.
def broadcast_backward_slab(
    grad_out: Tensor, in_shape: List[Int], ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """d_x = sum d_y over the dims that were broadcast (NumPy right-aligned)."""
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
    var out_buf = slab.alloc(
        n_in * grad_out.dtype().byte_size()
    )
    var n_out = grad_out.numel()
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_out))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_in))
    var grid = (n_in + _BLOCK - 1) // _BLOCK
    var dt = grad_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl)
        var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
        ctx.enqueue_function[
            _broadcast_sum_k[DType.float32], _broadcast_sum_k[DType.float32]
        ](
            g, o,
            idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
            odims[0], odims[1], odims[2], odims[3], odims[4], odims[5],
            ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
            n_in, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var g = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), g_rl)
        var o = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
        ctx.enqueue_function[
            _broadcast_sum_k[DType.bfloat16], _broadcast_sum_k[DType.bfloat16]
        ](
            g, o,
            idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
            odims[0], odims[1], odims[2], odims[3], odims[4], odims[5],
            ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
            n_in, grid_dim=grid, block_dim=_BLOCK)
    else:
        var g = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), g_rl)
        var o = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl)
        ctx.enqueue_function[
            _broadcast_sum_k[DType.float16], _broadcast_sum_k[DType.float16]
        ](
            g, o,
            idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
            odims[0], odims[1], odims[2], odims[3], odims[4], odims[5],
            ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
            n_in, grid_dim=grid, block_dim=_BLOCK)
    var sh = List[Int]()
    for i in range(len(in_shape)):
        sh.append(in_shape[i])
    return Tensor(out_buf^, sh^, grad_out.dtype())


# ══════════════════════════════════════════════════════════════════════════════
# REPEAT — sum the tiled copies. Torch `Tensor.repeat` tiling: out_dim[i] =
# in_dim[i] * reps[i], copy r of axis i at output index in_idx + r*in_dim.
# ══════════════════════════════════════════════════════════════════════════════
def _repeat_sum_k[dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
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
                                acc += rebind[Scalar[dtype]](g[off]).cast[
                                    DType.float32
                                ]()
        o[idx] = rebind[o.element_type](acc.cast[dtype]())


def repeat_backward(
    grad_out: Tensor, in_shape: List[Int], repeats: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    """d_x = sum of the `repeats`-tiled copies, back to in_shape."""
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
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n_in * grad_out.dtype().byte_size()
    )
    var n_out = grad_out.numel()
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_out))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_in))
    var grid = (n_in + _BLOCK - 1) // _BLOCK
    var dt = grad_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl)
        var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
        ctx.enqueue_function[
            _repeat_sum_k[DType.float32], _repeat_sum_k[DType.float32]
        ](
            g, o,
            idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
            reps[0], reps[1], reps[2], reps[3], reps[4], reps[5],
            ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
            n_in, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var g = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), g_rl)
        var o = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
        ctx.enqueue_function[
            _repeat_sum_k[DType.bfloat16], _repeat_sum_k[DType.bfloat16]
        ](
            g, o,
            idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
            reps[0], reps[1], reps[2], reps[3], reps[4], reps[5],
            ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
            n_in, grid_dim=grid, block_dim=_BLOCK)
    else:
        var g = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), g_rl)
        var o = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl)
        ctx.enqueue_function[
            _repeat_sum_k[DType.float16], _repeat_sum_k[DType.float16]
        ](
            g, o,
            idims[0], idims[1], idims[2], idims[3], idims[4], idims[5],
            reps[0], reps[1], reps[2], reps[3], reps[4], reps[5],
            ostr[0], ostr[1], ostr[2], ostr[3], ostr[4], ostr[5],
            n_in, grid_dim=grid, block_dim=_BLOCK)
    var sh = List[Int]()
    for i in range(len(in_shape)):
        sh.append(in_shape[i])
    return Tensor(out_buf^, sh^, grad_out.dtype())


# ══════════════════════════════════════════════════════════════════════════════
# WHERE — route d_y to lhs where cond!=0, to rhs elsewhere. `cond` can be stored
# as F32/BF16/F16; comparison casts each scalar to F32 inside the kernel.
# ══════════════════════════════════════════════════════════════════════════════
def _where_bwd_k[dtype: DType, cond_dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    cond: LayoutTensor[cond_dtype, _DYN1, MutAnyOrigin],
    da: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    db: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[dtype]](g[i]).cast[DType.float32]()
        var cv = rebind[Scalar[cond_dtype]](cond[i]).cast[DType.float32]()
        var is_a = cv != Float32(0.0)
        da[i] = rebind[da.element_type](
            (gv if is_a else Float32(0.0)).cast[dtype]()
        )
        db[i] = rebind[db.element_type](
            (Float32(0.0) if is_a else gv).cast[dtype]()
        )


def _where_backward_with_cond[cond_dtype: DType](
    grad_out: Tensor,
    cond: LayoutTensor[cond_dtype, _DYN1, MutAnyOrigin],
    ctx: DeviceContext,
) raises -> WhereGrads:
    var n = grad_out.numel()
    var da_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * grad_out.dtype().byte_size()
    )
    var db_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * grad_out.dtype().byte_size()
    )
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = grad_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
        var da = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            da_buf.unsafe_ptr().bitcast[Float32](), rl)
        var db = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            db_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[
            _where_bwd_k[DType.float32, cond_dtype],
            _where_bwd_k[DType.float32, cond_dtype],
        ](g, cond, da, db, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var g = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var da = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            da_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var db = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            db_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _where_bwd_k[DType.bfloat16, cond_dtype],
            _where_bwd_k[DType.bfloat16, cond_dtype],
        ](g, cond, da, db, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var g = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), rl)
        var da = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            da_buf.unsafe_ptr().bitcast[Float16](), rl)
        var db = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            db_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _where_bwd_k[DType.float16, cond_dtype],
            _where_bwd_k[DType.float16, cond_dtype],
        ](g, cond, da, db, n, grid_dim=grid, block_dim=_BLOCK)
    return WhereGrads(
        Tensor(da_buf^, grad_out.shape(), grad_out.dtype()),
        Tensor(db_buf^, grad_out.shape(), grad_out.dtype()))


def where_backward(
    grad_out: Tensor, cond: Tensor, ctx: DeviceContext
) raises -> WhereGrads:
    """d_a = cond ? d_y : 0 ; d_b = cond ? 0 : d_y."""
    var n = grad_out.numel()
    if cond.numel() != n:
        raise Error("where_backward: grad/cond numel mismatch")
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var cdt = cond.dtype().to_mojo_dtype()
    if cdt == DType.float32:
        var c = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            cond.buf.unsafe_ptr().bitcast[Float32](), rl)
        return _where_backward_with_cond[DType.float32](grad_out, c, ctx)
    elif cdt == DType.bfloat16:
        var c = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            cond.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        return _where_backward_with_cond[DType.bfloat16](grad_out, c, ctx)
    elif cdt == DType.float16:
        var c = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            cond.buf.unsafe_ptr().bitcast[Float16](), rl)
        return _where_backward_with_cond[DType.float16](grad_out, c, ctx)
    raise Error("where_backward: cond mask must be F32/BF16/F16")


# ══════════════════════════════════════════════════════════════════════════════
# CLAMP — grad passes where lo <= x <= hi, else 0.
# ══════════════════════════════════════════════════════════════════════════════
def _clamp_bwd_k[dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    lo: Float32, hi: Float32, n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var xv = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var gv = rebind[Scalar[dtype]](g[i]).cast[DType.float32]()
        var pass_through = (xv >= lo) and (xv <= hi)
        o[i] = rebind[o.element_type](
            (gv if pass_through else Float32(0.0)).cast[dtype]()
        )


def clamp_backward(
    grad_out: Tensor, x: Tensor, lo: Float32, hi: Float32, ctx: DeviceContext
) raises -> Tensor:
    """d_x = (lo <= x <= hi) ? d_y : 0. `x` is the forward INPUT."""
    if grad_out.dtype() != x.dtype():
        raise Error("clamp_backward: grad/x dtype mismatch")
    var n = x.numel()
    if grad_out.numel() != n:
        raise Error("clamp_backward: grad/x numel mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * x.dtype().byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
        var xt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl)
        var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[
            _clamp_bwd_k[DType.float32], _clamp_bwd_k[DType.float32]
        ](g, xt, o, lo, hi, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var g = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var xt = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var o = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _clamp_bwd_k[DType.bfloat16], _clamp_bwd_k[DType.bfloat16]
        ](g, xt, o, lo, hi, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var g = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), rl)
        var xt = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl)
        var o = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _clamp_bwd_k[DType.float16], _clamp_bwd_k[DType.float16]
        ](g, xt, o, lo, hi, n, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, x.shape(), x.dtype())


# ══════════════════════════════════════════════════════════════════════════════
# MAXIMUM / MINIMUM — grad to the arg that won. Ties go to `a` (flame-core uses
# a>=b for max, a<=b for min; autograd.rs:6114/6134).
# ══════════════════════════════════════════════════════════════════════════════
def _maxmin_bwd_k[dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    a: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    b: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    da: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    db: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    is_max: Int, n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[dtype]](g[i]).cast[DType.float32]()
        var av = rebind[Scalar[dtype]](a[i]).cast[DType.float32]()
        var bv = rebind[Scalar[dtype]](b[i]).cast[DType.float32]()
        var a_wins: Bool
        if is_max == 1:
            a_wins = av >= bv
        else:
            a_wins = av <= bv
        da[i] = rebind[da.element_type](
            (gv if a_wins else Float32(0.0)).cast[dtype]()
        )
        db[i] = rebind[db.element_type](
            (Float32(0.0) if a_wins else gv).cast[dtype]()
        )


def _maxmin_backward(
    grad_out: Tensor, a: Tensor, b: Tensor, is_max: Int, ctx: DeviceContext
) raises -> BinaryGrads:
    if grad_out.dtype() != a.dtype() or a.dtype() != b.dtype():
        raise Error("maxmin_backward: grad/a/b dtype mismatch")
    var n = a.numel()
    if b.numel() != n or grad_out.numel() != n:
        raise Error("maxmin_backward: numel mismatch")
    var da_buf = ctx.enqueue_create_buffer[DType.uint8](n * a.dtype().byte_size())
    var db_buf = ctx.enqueue_create_buffer[DType.uint8](n * b.dtype().byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = a.dtype().to_mojo_dtype()
    if dt == DType.float32:
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
        ctx.enqueue_function[
            _maxmin_bwd_k[DType.float32], _maxmin_bwd_k[DType.float32]
        ](g, at, bt, da, db, is_max, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var g = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var at = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var bt = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var da = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            da_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var db = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            db_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _maxmin_bwd_k[DType.bfloat16], _maxmin_bwd_k[DType.bfloat16]
        ](g, at, bt, da, db, is_max, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var g = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), rl)
        var at = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float16](), rl)
        var bt = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float16](), rl)
        var da = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            da_buf.unsafe_ptr().bitcast[Float16](), rl)
        var db = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            db_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _maxmin_bwd_k[DType.float16], _maxmin_bwd_k[DType.float16]
        ](g, at, bt, da, db, is_max, n, grid_dim=grid, block_dim=_BLOCK)
    return BinaryGrads(
        Tensor(da_buf^, a.shape(), a.dtype()),
        Tensor(db_buf^, b.shape(), b.dtype()))


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
def _index_select_bwd_k[dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],      # d_y [N, D]
    idx: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],      # indices [N]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],      # d_x [V, D]
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
                acc += rebind[Scalar[dtype]](g[n * D + d]).cast[DType.float32]()
        o[t] = rebind[o.element_type](acc.cast[dtype]())


def index_select_backward(
    grad_out: Tensor, indices: List[Int], dim: Int, in_shape: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    """Scatter-ADD d_y rows back to the selected indices -> zeros(in_shape).

    Gated case: dim==0 row-select on a rank-2 table [V, D]. grad_out is [N, D]
    with N == len(indices); result is [V, D] (in_shape), same dtype as
    grad_out. Repeated indices accumulate."""
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

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        V * D * grad_out.dtype().byte_size()
    )
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N * D))
    var id_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](V * D))
    var IDS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        id_dev.unsafe_ptr(), id_rl)
    var grid = (V * D + _BLOCK - 1) // _BLOCK
    var dt = grad_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl)
        var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
        ctx.enqueue_function[
            _index_select_bwd_k[DType.float32], _index_select_bwd_k[DType.float32]
        ](g, IDS, o, N, D, V, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var g = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), g_rl)
        var o = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
        ctx.enqueue_function[
            _index_select_bwd_k[DType.bfloat16], _index_select_bwd_k[DType.bfloat16]
        ](g, IDS, o, N, D, V, grid_dim=grid, block_dim=_BLOCK)
    else:
        var g = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), g_rl)
        var o = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl)
        ctx.enqueue_function[
            _index_select_bwd_k[DType.float16], _index_select_bwd_k[DType.float16]
        ](g, IDS, o, N, D, V, grid_dim=grid, block_dim=_BLOCK)
    var sh = List[Int]()
    sh.append(V); sh.append(D)
    return Tensor(out_buf^, sh^, grad_out.dtype())
