# Opt-in scratch-backed tensor algebra helpers.
#
# Keep this out of tensor_algebra.mojo so normal model imports do not compile
# scratch kernels unless a caller explicitly opts into scratch-frame ownership.

from std.builtin.dtype import DType
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import concat, slice


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _concat_dim1_rank2_2_f32_kernel(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    ca: Int,
    cb: Int,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var co = ca + cb
        var r = idx // co
        var c = idx % co
        var rv: Float32
        if c < ca:
            rv = rebind[Scalar[DType.float32]](a[r * ca + c])
        else:
            rv = rebind[Scalar[DType.float32]](b[r * cb + (c - ca)])
        o[idx] = rebind[o.element_type](rv)


def _concat_dim1_rank2_3_f32_kernel(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    c_t: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    ca: Int,
    cb: Int,
    cc: Int,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var co = ca + cb + cc
        var r = idx // co
        var c = idx % co
        var rv: Float32
        if c < ca:
            rv = rebind[Scalar[DType.float32]](a[r * ca + c])
        elif c < ca + cb:
            rv = rebind[Scalar[DType.float32]](b[r * cb + (c - ca)])
        else:
            rv = rebind[Scalar[DType.float32]](c_t[r * cc + (c - ca - cb)])
        o[idx] = rebind[o.element_type](rv)


def _slice_dim1_rank2_f32_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    cols: Int,
    start: Int,
    length: Int,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var r = idx // length
        var c = idx % length
        o[idx] = rebind[o.element_type](
            rebind[Scalar[DType.float32]](x[r * cols + start + c])
        )


def concat2_scratch(
    dim: Int,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    a: Tensor,
    b: Tensor,
) raises -> Tensor:
    """concat(a,b) using scratch storage when the hot F32 rank-2 path applies."""
    if a.dtype() != STDtype.F32 or b.dtype() != STDtype.F32:
        return concat(dim, ctx, a, b)
    var ash = a.shape()
    var bsh = b.shape()
    if len(ash) != 2 or len(bsh) != 2 or dim != 1:
        return concat(dim, ctx, a, b)
    if ash[0] != bsh[0]:
        return concat(dim, ctx, a, b)

    var rows = ash[0]
    var ca = ash[1]
    var cb = bsh[1]
    var out_n = rows * (ca + cb)
    var oshape = List[Int]()
    oshape.append(rows)
    oshape.append(ca + cb)
    var out = scratch.alloc_tensor(oshape^, STDtype.F32)

    var rl_a = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * ca))
    var rl_b = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * cb))
    var rl_o = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[Float32](), rl_a
    )
    var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[Float32](), rl_b
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out.buf.unsafe_ptr().bitcast[Float32](), rl_o
    )
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _concat_dim1_rank2_2_f32_kernel, _concat_dim1_rank2_2_f32_kernel
    ](A, B, O, rows, ca, cb, out_n, grid_dim=grid, block_dim=_BLOCK)
    return out^


def concat3_scratch(
    dim: Int,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    a: Tensor,
    b: Tensor,
    c: Tensor,
) raises -> Tensor:
    """concat(a,b,c) using scratch storage when the hot F32 rank-2 path applies."""
    if a.dtype() != STDtype.F32 or b.dtype() != STDtype.F32 or c.dtype() != STDtype.F32:
        return concat(dim, ctx, a, b, c)
    var ash = a.shape()
    var bsh = b.shape()
    var csh = c.shape()
    if len(ash) != 2 or len(bsh) != 2 or len(csh) != 2 or dim != 1:
        return concat(dim, ctx, a, b, c)
    if ash[0] != bsh[0] or ash[0] != csh[0]:
        return concat(dim, ctx, a, b, c)

    var rows = ash[0]
    var ca = ash[1]
    var cb = bsh[1]
    var cc = csh[1]
    var out_n = rows * (ca + cb + cc)
    var oshape = List[Int]()
    oshape.append(rows)
    oshape.append(ca + cb + cc)
    var out = scratch.alloc_tensor(oshape^, STDtype.F32)

    var rl_a = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * ca))
    var rl_b = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * cb))
    var rl_c = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * cc))
    var rl_o = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[Float32](), rl_a
    )
    var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[Float32](), rl_b
    )
    var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        c.buf.unsafe_ptr().bitcast[Float32](), rl_c
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out.buf.unsafe_ptr().bitcast[Float32](), rl_o
    )
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _concat_dim1_rank2_3_f32_kernel, _concat_dim1_rank2_3_f32_kernel
    ](A, B, C, O, rows, ca, cb, cc, out_n, grid_dim=grid, block_dim=_BLOCK)
    return out^


def slice_scratch(
    x: Tensor,
    dim: Int,
    start: Int,
    length: Int,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> Tensor:
    """slice() using scratch storage when the hot F32 rank-2 path applies."""
    var xshape = x.shape()
    var rank = len(xshape)
    if x.dtype() != STDtype.F32 or rank != 2 or dim != 1:
        return slice(x, dim, start, length, ctx)
    if start < 0 or length < 0 or start + length > xshape[1]:
        return slice(x, dim, start, length, ctx)

    var rows = xshape[0]
    var cols = xshape[1]
    var n = rows * length
    var oshape = List[Int]()
    oshape.append(rows)
    oshape.append(length)
    var out = scratch.alloc_tensor(oshape^, STDtype.F32)

    var rl_x = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * cols))
    var rl_o = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl_x
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out.buf.unsafe_ptr().bitcast[Float32](), rl_o
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _slice_dim1_rank2_f32_kernel, _slice_dim1_rank2_f32_kernel
    ](X, O, rows, cols, start, length, n, grid_dim=grid, block_dim=_BLOCK)
    return out^
