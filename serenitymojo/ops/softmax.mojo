# ops/softmax.mojo — softmax_lastdim(x): numerically-stable softmax over last dim.
#
#   softmax(x)[..., j] = exp(x_j - max_k x_k) / sum_k exp(x_k - max_k x_k)
#
# Subtracting the row max is the standard stability trick (avoids overflow in
# exp). Two reductions per row: (1) max, (2) sum of exp(x - max). One block per
# row, shared-memory tree reductions in F32 (same shape as rms_norm). The
# `softmax_gpu` SDK op is the TileTensor+closure variant (UNCALLABLE per the
# OP-CALLABILITY MAP), so this is hand-rolled. F32 math; store casts back.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import exp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _TPB = 256
# Sentinel "negative infinity" seed for the max reduction.
comptime _NEG_BIG = Float32(-3.0e38)


def _softmax_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cols: Int,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    # Pass 1: row max.
    var lmax: Float32 = _NEG_BIG
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        if v > lmax:
            lmax = v
        c += _TPB
    shared[tid] = lmax
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            var a = shared[tid]
            var b = shared[tid + active]
            shared[tid] = a if a > b else b
        barrier()
        active //= 2
    var rmax = shared[0]
    barrier()
    # Pass 2: sum of exp(x - max).
    var lsum: Float32 = 0.0
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        lsum += exp(v - rmax)
        c += _TPB
    shared[tid] = lsum
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var rsum = shared[0]
    var inv = 1.0 / rsum
    # Write.
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        o[row, c] = rebind[o.element_type](exp(v - rmax) * inv)
        c += _TPB


def _softmax_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    cols: Int,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lmax: Float32 = _NEG_BIG
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        if v > lmax:
            lmax = v
        c += _TPB
    shared[tid] = lmax
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            var a = shared[tid]
            var b = shared[tid + active]
            shared[tid] = a if a > b else b
        barrier()
        active //= 2
    var rmax = shared[0]
    barrier()
    var lsum: Float32 = 0.0
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        lsum += exp(v - rmax)
        c += _TPB
    shared[tid] = lsum
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var rsum = shared[0]
    var inv = 1.0 / rsum
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type]((exp(v - rmax) * inv).cast[DType.bfloat16]())
        c += _TPB


def _softmax_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    cols: Int,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lmax: Float32 = _NEG_BIG
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float16]](x[row, c]).cast[DType.float32]()
        if v > lmax:
            lmax = v
        c += _TPB
    shared[tid] = lmax
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            var a = shared[tid]
            var b = shared[tid + active]
            shared[tid] = a if a > b else b
        barrier()
        active //= 2
    var rmax = shared[0]
    barrier()
    var lsum: Float32 = 0.0
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float16]](x[row, c]).cast[DType.float32]()
        lsum += exp(v - rmax)
        c += _TPB
    shared[tid] = lsum
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var rsum = shared[0]
    var inv = 1.0 / rsum
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float16]](x[row, c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type]((exp(v - rmax) * inv).cast[DType.float16]())
        c += _TPB


def softmax_lastdim(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Numerically-stable softmax over the last dim of x.

    x:      [..., D]   (any compute dtype; leading dims flattened to rows)
    returns [..., D]   (x's dtype; F32-accumulated max/sum).
    """
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("softmax_lastdim: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[_softmax_kernel_f32, _softmax_kernel_f32](
            X, O, d, grid_dim=rows, block_dim=_TPB
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[_softmax_kernel_bf16, _softmax_kernel_bf16](
            X, O, d, grid_dim=rows, block_dim=_TPB
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[_softmax_kernel_f16, _softmax_kernel_f16](
            X, O, d, grid_dim=rows, block_dim=_TPB
        )
    ctx.synchronize()
    return Tensor(out_buf^, xshape.copy(), x.dtype())
