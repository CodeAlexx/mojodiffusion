# ops/reduce_backward.mojo — BACKWARD kernels for Tier-0/Tier-1 reduce + activation
# arms: Sqrt, Square, Log, Softmax, LogSoftmax, Sum, Mean.
#
# Phase T1 of FULL_PORT_TRAINING_PLAN.md (Tier 0 + Tier 1 backward arms).
#
# Elementwise/reduction math uses F32 internally. Tensor outputs preserve storage
# dtype where the API has a tensor input or explicit output dtype.
# Conventions mirror ops/attention_backward.mojo (the proven SDPA-bwd template):
#   * device buffers are DType.uint8 (Tensor's storage), bitcast to the storage
#     dtype at the LayoutTensor boundary,
#   * BF16/F16 storage kernels cast scalar elements to F32 for math/reductions
#     and write the gradient back to storage dtype,
#   * one flat thread per element for the elementwise arms,
#   * one block per row + shared-memory F32 tree-reduction for the softmax family
#     (same row/block approach as attention_backward._softmax_bwd_rows_f32).
#
# Math:
#   sqrt_backward(g, x)      : d_x = g * 0.5 / sqrt(x)          (x > 0)
#   square_backward(g, x)    : d_x = g * 2 * x
#   log_backward(g, x)       : d_x = g / x                      (x > 0)
#   softmax_backward(g, sm)  : d_x = sm * (g - rowsum(g*sm))    over LAST dim;
#                              `sm` is the softmax OUTPUT (probabilities).
#   logsoftmax_backward(g,l) : d_x = g - exp(l) * rowsum(g)     over LAST dim;
#                              `l` is the logsoftmax OUTPUT.
#   sum_backward(g_scalar,n,dtype): d_x[i] = g_scalar           (broadcast)
#   mean_backward(g_scalar,n,dtype): d_x[i] = g_scalar / n      (broadcast)
#
# softmax/logsoftmax operate on a 2D [rows, cols] tensor; the reduction is over
# cols (the last dim).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt, exp
from std.gpu.host import DeviceContext
from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256


# ── elementwise kernels (one flat thread per element) ─────────────────────────
def _sqrt_bwd_k[dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var xv = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var gv = rebind[Scalar[dtype]](g[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](
            (gv * Float32(0.5) / sqrt(xv)).cast[dtype]()
        )


def _square_bwd_k[dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var xv = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var gv = rebind[Scalar[dtype]](g[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](
            (gv * Float32(2.0) * xv).cast[dtype]()
        )


def _log_bwd_k[dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var xv = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var gv = rebind[Scalar[dtype]](g[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv / xv).cast[dtype]())


def _broadcast_scalar_k[dtype: DType](
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    val: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](val.cast[dtype]())


# ── softmax backward, per row ─────────────────────────────────────────────────
# d_x[r,c] = sm[r,c] * (g[r,c] - sum_c sm[r,c]*g[r,c]).  Same row approach as
# attention_backward._softmax_bwd_rows_f32. One block per row, F32 tree-reduce.
def _softmax_bwd_rows_k[dtype: DType](
    sm: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    g: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    cols: Int,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var c = tid
    while c < cols:
        var a = rebind[Scalar[dtype]](sm[row, c]).cast[DType.float32]()
        var gg = rebind[Scalar[dtype]](g[row, c]).cast[DType.float32]()
        lsum += a * gg
        c += _TPB
    shared[tid] = lsum
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var sum_ga = shared[0]
    barrier()
    c = tid
    while c < cols:
        var a = rebind[Scalar[dtype]](sm[row, c]).cast[DType.float32]()
        var gg = rebind[Scalar[dtype]](g[row, c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type](
            (a * (gg - sum_ga)).cast[dtype]()
        )
        c += _TPB


# ── logsoftmax backward, per row ──────────────────────────────────────────────
# d_x[r,c] = g[r,c] - exp(l[r,c]) * sum_c g[r,c].  One block per row.
def _logsoftmax_bwd_rows_k[dtype: DType](
    l: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    g: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    cols: Int,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var c = tid
    while c < cols:
        lsum += rebind[Scalar[dtype]](g[row, c]).cast[DType.float32]()
        c += _TPB
    shared[tid] = lsum
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var sum_g = shared[0]
    barrier()
    c = tid
    while c < cols:
        var lv = rebind[Scalar[dtype]](l[row, c]).cast[DType.float32]()
        var gv = rebind[Scalar[dtype]](g[row, c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type](
            (gv - exp(lv) * sum_g).cast[dtype]()
        )
        c += _TPB


# ── elementwise backward driver (g, x both [N], same storage dtype) ──────────
def _elementwise_bwd[
    kind: Int
](grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    if grad_out.dtype() != x.dtype():
        raise Error("reduce_backward elementwise: grad/x dtype mismatch")
    var n = x.numel()
    if grad_out.numel() != n:
        raise Error("reduce_backward elementwise: grad/x numel mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
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
        comptime if kind == 0:
            ctx.enqueue_function[
                _sqrt_bwd_k[DType.float32], _sqrt_bwd_k[DType.float32]
            ](g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)
        elif kind == 1:
            ctx.enqueue_function[
                _square_bwd_k[DType.float32], _square_bwd_k[DType.float32]
            ](g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _log_bwd_k[DType.float32], _log_bwd_k[DType.float32]
            ](g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var g = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var xt = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var o = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        comptime if kind == 0:
            ctx.enqueue_function[
                _sqrt_bwd_k[DType.bfloat16], _sqrt_bwd_k[DType.bfloat16]
            ](g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)
        elif kind == 1:
            ctx.enqueue_function[
                _square_bwd_k[DType.bfloat16], _square_bwd_k[DType.bfloat16]
            ](g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _log_bwd_k[DType.bfloat16], _log_bwd_k[DType.bfloat16]
            ](g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var g = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), rl)
        var xt = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl)
        var o = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        comptime if kind == 0:
            ctx.enqueue_function[
                _sqrt_bwd_k[DType.float16], _sqrt_bwd_k[DType.float16]
            ](g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)
        elif kind == 1:
            ctx.enqueue_function[
                _square_bwd_k[DType.float16], _square_bwd_k[DType.float16]
            ](g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[
                _log_bwd_k[DType.float16], _log_bwd_k[DType.float16]
            ](g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)

    # sync removed (single-stream ordering; was kernel-trailing host stall)
    var sh = List[Int]()
    sh.append(n)
    return Tensor(out_buf^, sh^, x.dtype())


def sqrt_backward(grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """d_x = grad_out * 0.5 / sqrt(x).  x must be positive."""
    return _elementwise_bwd[0](grad_out, x, ctx)


def square_backward(grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """d_x = grad_out * 2 * x."""
    return _elementwise_bwd[1](grad_out, x, ctx)


def log_backward(grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """d_x = grad_out / x.  x must be positive."""
    return _elementwise_bwd[2](grad_out, x, ctx)


# ── softmax / logsoftmax backward drivers (2D [rows, cols], reduce over cols) ─
def softmax_backward(
    grad_out: Tensor, softmax_out: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """d_x = sm * (grad_out - rowsum(grad_out*sm)) over the last dim.

    `softmax_out` is the softmax OUTPUT (probabilities). Both inputs are 2D
    [rows, cols] with matching storage dtype. Returns d_x in `softmax_out` dtype.
    """
    if grad_out.dtype() != softmax_out.dtype():
        raise Error("softmax_backward: grad_out/softmax_out dtype mismatch")
    var sh = softmax_out.shape()
    if len(sh) != 2:
        raise Error("softmax_backward: expected 2D [rows, cols]")
    var rows = sh[0]
    var cols = sh[1]
    # cols may exceed _TPB (real attention widths, e.g. 1024). The kernel runs
    # one block per row with _TPB threads grid-striding over cols (c=tid; c+=_TPB),
    # then a _TPB-wide shared tree-reduce gives rowsum over ALL cols — correct for
    # any cols. No 256-col cap. (See _softmax_bwd_rows_k.)
    if cols < 1:
        raise Error("softmax_backward: cols must be >= 1")
    var n = rows * cols
    if grad_out.numel() != n:
        raise Error("softmax_backward: grad_out numel mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * softmax_out.dtype().byte_size()
    )
    var rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var dt = softmax_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var sm = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            softmax_out.buf.unsafe_ptr().bitcast[Float32](), rl)
        var g = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
        var o = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[
            _softmax_bwd_rows_k[DType.float32],
            _softmax_bwd_rows_k[DType.float32],
        ](sm, g, o, cols, grid_dim=rows, block_dim=_TPB)
    elif dt == DType.bfloat16:
        var sm = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            softmax_out.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var g = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var o = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _softmax_bwd_rows_k[DType.bfloat16],
            _softmax_bwd_rows_k[DType.bfloat16],
        ](sm, g, o, cols, grid_dim=rows, block_dim=_TPB)
    else:
        var sm = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            softmax_out.buf.unsafe_ptr().bitcast[Float16](), rl)
        var g = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), rl)
        var o = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _softmax_bwd_rows_k[DType.float16],
            _softmax_bwd_rows_k[DType.float16],
        ](sm, g, o, cols, grid_dim=rows, block_dim=_TPB)
    # sync removed (single-stream ordering; was kernel-trailing host stall)
    var os = List[Int]()
    os.append(rows); os.append(cols)
    return Tensor(out_buf^, os^, softmax_out.dtype())


def logsoftmax_backward(
    grad_out: Tensor, logsoftmax_out: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """d_x = grad_out - exp(lsm) * rowsum(grad_out) over the last dim.

    `logsoftmax_out` is the logsoftmax OUTPUT. Both inputs are 2D [rows, cols]
    with matching storage dtype. Returns d_x in `logsoftmax_out` dtype.
    """
    if grad_out.dtype() != logsoftmax_out.dtype():
        raise Error("logsoftmax_backward: grad_out/logsoftmax_out dtype mismatch")
    var sh = logsoftmax_out.shape()
    if len(sh) != 2:
        raise Error("logsoftmax_backward: expected 2D [rows, cols]")
    var rows = sh[0]
    var cols = sh[1]
    # cols may exceed _TPB (real attention widths). Grid-stride row kernel +
    # _TPB-wide shared tree-reduce handles any cols. (See _logsoftmax_bwd_rows_k.)
    if cols < 1:
        raise Error("logsoftmax_backward: cols must be >= 1")
    var n = rows * cols
    if grad_out.numel() != n:
        raise Error("logsoftmax_backward: grad_out numel mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * logsoftmax_out.dtype().byte_size()
    )
    var rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var dt = logsoftmax_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var l = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            logsoftmax_out.buf.unsafe_ptr().bitcast[Float32](), rl)
        var g = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
        var o = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[
            _logsoftmax_bwd_rows_k[DType.float32],
            _logsoftmax_bwd_rows_k[DType.float32],
        ](l, g, o, cols, grid_dim=rows, block_dim=_TPB)
    elif dt == DType.bfloat16:
        var l = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            logsoftmax_out.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var g = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var o = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _logsoftmax_bwd_rows_k[DType.bfloat16],
            _logsoftmax_bwd_rows_k[DType.bfloat16],
        ](l, g, o, cols, grid_dim=rows, block_dim=_TPB)
    else:
        var l = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            logsoftmax_out.buf.unsafe_ptr().bitcast[Float16](), rl)
        var g = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), rl)
        var o = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _logsoftmax_bwd_rows_k[DType.float16],
            _logsoftmax_bwd_rows_k[DType.float16],
        ](l, g, o, cols, grid_dim=rows, block_dim=_TPB)
    # sync removed (single-stream ordering; was kernel-trailing host stall)
    var os = List[Int]()
    os.append(rows); os.append(cols)
    return Tensor(out_buf^, os^, logsoftmax_out.dtype())


# ── sum / mean backward (scalar grad broadcast to in_shape) ──────────────────
def _broadcast_bwd(
    grad_scalar: Float32, in_shape: List[Int], dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var n = 1
    for i in range(len(in_shape)):
        n *= in_shape[i]
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = dtype.to_mojo_dtype()
    if dt == DType.float32:
        var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[
            _broadcast_scalar_k[DType.float32], _broadcast_scalar_k[DType.float32]
        ](o, grad_scalar, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var o = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _broadcast_scalar_k[DType.bfloat16], _broadcast_scalar_k[DType.bfloat16]
        ](o, grad_scalar, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var o = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _broadcast_scalar_k[DType.float16], _broadcast_scalar_k[DType.float16]
        ](o, grad_scalar, n, grid_dim=grid, block_dim=_BLOCK)
    # sync removed (single-stream ordering; was kernel-trailing host stall)
    var sh = List[Int]()
    for i in range(len(in_shape)):
        sh.append(in_shape[i])
    return Tensor(out_buf^, sh^, dtype)


def sum_backward(
    grad_out_scalar: Float32,
    in_shape: List[Int],
    ctx: DeviceContext,
    dtype: STDtype,
) raises -> Tensor:
    """d_x = ones(in_shape) * grad_out_scalar."""
    return _broadcast_bwd(grad_out_scalar, in_shape, dtype, ctx)


def mean_backward(
    grad_out_scalar: Float32,
    in_shape: List[Int],
    ctx: DeviceContext,
    dtype: STDtype,
) raises -> Tensor:
    """d_x = (grad_out_scalar / N) broadcast to in_shape."""
    var n = 1
    for i in range(len(in_shape)):
        n *= in_shape[i]
    return _broadcast_bwd(grad_out_scalar / Float32(n), in_shape, dtype, ctx)
