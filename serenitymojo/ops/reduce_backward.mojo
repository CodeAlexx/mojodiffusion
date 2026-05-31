# ops/reduce_backward.mojo — BACKWARD kernels for Tier-0/Tier-1 reduce + activation
# arms: Sqrt, Square, Log, Softmax, LogSoftmax, Sum, Mean.
#
# Phase T1 of FULL_PORT_TRAINING_PLAN.md (Tier 0 + Tier 1 backward arms).
#
# Each fn returns a fresh F32 Tensor holding d_x. All interior math is F32.
# Conventions mirror ops/attention_backward.mojo (the proven SDPA-bwd template):
#   * device buffers are DType.uint8 (Tensor's storage), bitcast to Float32 at
#     the LayoutTensor boundary,
#   * one flat thread per element for the elementwise arms,
#   * one block per row + shared-memory F32 tree-reduction for the softmax family
#     (reusing the exact _softmax_bwd_rows_f32 row approach from attention_backward).
#
# Math:
#   sqrt_backward(g, x)      : d_x = g * 0.5 / sqrt(x)          (x > 0)
#   square_backward(g, x)    : d_x = g * 2 * x
#   log_backward(g, x)       : d_x = g / x                      (x > 0)
#   softmax_backward(g, sm)  : d_x = sm * (g - rowsum(g*sm))    over LAST dim;
#                              `sm` is the softmax OUTPUT (probabilities).
#   logsoftmax_backward(g,l) : d_x = g - exp(l) * rowsum(g)     over LAST dim;
#                              `l` is the logsoftmax OUTPUT.
#   sum_backward(g_scalar, n): d_x[i] = g_scalar                (broadcast)
#   mean_backward(g_scalar,n): d_x[i] = g_scalar / n            (broadcast)
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
from serenitymojo.ops.cast import cast_tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256


# ── elementwise kernels (one flat thread per element) ─────────────────────────
def _sqrt_bwd_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var xv = rebind[Scalar[DType.float32]](x[i])
        var gv = rebind[Scalar[DType.float32]](g[i])
        o[i] = rebind[o.element_type](gv * Float32(0.5) / sqrt(xv))


def _square_bwd_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var xv = rebind[Scalar[DType.float32]](x[i])
        var gv = rebind[Scalar[DType.float32]](g[i])
        o[i] = rebind[o.element_type](gv * Float32(2.0) * xv)


def _log_bwd_k(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var xv = rebind[Scalar[DType.float32]](x[i])
        var gv = rebind[Scalar[DType.float32]](g[i])
        o[i] = rebind[o.element_type](gv / xv)


def _broadcast_scalar_k(
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    val: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](val)


# ── softmax backward, per row ─────────────────────────────────────────────────
# d_x[r,c] = sm[r,c] * (g[r,c] - sum_c sm[r,c]*g[r,c]).  Same row approach as
# attention_backward._softmax_bwd_rows_f32. One block per row, F32 tree-reduce.
def _softmax_bwd_rows_f32(
    sm: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
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
        var a = rebind[Scalar[DType.float32]](sm[row, c])
        var gg = rebind[Scalar[DType.float32]](g[row, c])
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
        var a = rebind[Scalar[DType.float32]](sm[row, c])
        var gg = rebind[Scalar[DType.float32]](g[row, c])
        o[row, c] = rebind[o.element_type](a * (gg - sum_ga))
        c += _TPB


# ── logsoftmax backward, per row ──────────────────────────────────────────────
# d_x[r,c] = g[r,c] - exp(l[r,c]) * sum_c g[r,c].  One block per row.
def _logsoftmax_bwd_rows_f32(
    l: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
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
        lsum += rebind[Scalar[DType.float32]](g[row, c])
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
        var lv = rebind[Scalar[DType.float32]](l[row, c])
        var gv = rebind[Scalar[DType.float32]](g[row, c])
        o[row, c] = rebind[o.element_type](gv - exp(lv) * sum_g)
        c += _TPB


# ── elementwise backward driver (g, x both [N] F32 Tensors) ──────────────────
def _elementwise_bwd[
    kind: Int
](grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    # BF16/F16 storage path: cast up, run F32 interior, cast grad down.
    # F32 path byte-identical (branch only on non-F32 input). Covers
    # sqrt/square/log backward (all route through this driver).
    if grad_out.dtype() != STDtype.F32 or x.dtype() != STDtype.F32:
        var out_dt = x.dtype()
        var go32 = cast_tensor(grad_out, STDtype.F32, ctx)
        var x32 = cast_tensor(x, STDtype.F32, ctx)
        var dx32 = _elementwise_bwd[kind](go32, x32, ctx)
        return cast_tensor(dx32, out_dt, ctx)
    var n = x.numel()
    if grad_out.numel() != n:
        raise Error("reduce_backward elementwise: grad/x numel mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var g = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
    var xt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl)
    var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK

    comptime if kind == 0:
        ctx.enqueue_function[_sqrt_bwd_k, _sqrt_bwd_k](
            g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)
    elif kind == 1:
        ctx.enqueue_function[_square_bwd_k, _square_bwd_k](
            g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        ctx.enqueue_function[_log_bwd_k, _log_bwd_k](
            g, xt, o, n, grid_dim=grid, block_dim=_BLOCK)

    ctx.synchronize()
    var sh = List[Int]()
    sh.append(n)
    return Tensor(out_buf^, sh^, STDtype.F32)


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
    [rows, cols] F32; softmax over cols. Returns d_x [rows, cols] F32.
    """
    # BF16/F16 storage path: cast up, run F32 interior, cast grad down.
    # F32 path byte-identical (branch only on non-F32 input).
    if grad_out.dtype() != STDtype.F32 or softmax_out.dtype() != STDtype.F32:
        var out_dt = softmax_out.dtype()
        var go32 = cast_tensor(grad_out, STDtype.F32, ctx)
        var sm32 = cast_tensor(softmax_out, STDtype.F32, ctx)
        var dx32 = softmax_backward(go32, sm32, ctx)
        return cast_tensor(dx32, out_dt, ctx)
    var sh = softmax_out.shape()
    if len(sh) != 2:
        raise Error("softmax_backward: expected 2D [rows, cols]")
    var rows = sh[0]
    var cols = sh[1]
    # cols may exceed _TPB (real attention widths, e.g. 1024). The kernel runs
    # one block per row with _TPB threads grid-striding over cols (c=tid; c+=_TPB),
    # then a _TPB-wide shared tree-reduce gives rowsum over ALL cols — correct for
    # any cols. No 256-col cap. (See _softmax_bwd_rows_f32.)
    if cols < 1:
        raise Error("softmax_backward: cols must be >= 1")
    var n = rows * cols
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var sm = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        softmax_out.buf.unsafe_ptr().bitcast[Float32](), rl)
    var g = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
    var o = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl)
    ctx.enqueue_function[_softmax_bwd_rows_f32, _softmax_bwd_rows_f32](
        sm, g, o, cols, grid_dim=rows, block_dim=_TPB)
    ctx.synchronize()
    var os = List[Int]()
    os.append(rows); os.append(cols)
    return Tensor(out_buf^, os^, STDtype.F32)


def logsoftmax_backward(
    grad_out: Tensor, logsoftmax_out: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """d_x = grad_out - exp(lsm) * rowsum(grad_out) over the last dim.

    `logsoftmax_out` is the logsoftmax OUTPUT. Both inputs 2D [rows, cols] F32;
    reduction over cols. Returns d_x [rows, cols] F32.
    """
    # BF16/F16 storage path: cast up, run F32 interior, cast grad down.
    # F32 path byte-identical (branch only on non-F32 input).
    if grad_out.dtype() != STDtype.F32 or logsoftmax_out.dtype() != STDtype.F32:
        var out_dt = logsoftmax_out.dtype()
        var go32 = cast_tensor(grad_out, STDtype.F32, ctx)
        var lsm32 = cast_tensor(logsoftmax_out, STDtype.F32, ctx)
        var dx32 = logsoftmax_backward(go32, lsm32, ctx)
        return cast_tensor(dx32, out_dt, ctx)
    var sh = logsoftmax_out.shape()
    if len(sh) != 2:
        raise Error("logsoftmax_backward: expected 2D [rows, cols]")
    var rows = sh[0]
    var cols = sh[1]
    # cols may exceed _TPB (real attention widths). Grid-stride row kernel +
    # _TPB-wide shared tree-reduce handles any cols. (See _logsoftmax_bwd_rows_f32.)
    if cols < 1:
        raise Error("logsoftmax_backward: cols must be >= 1")
    var n = rows * cols
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var l = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        logsoftmax_out.buf.unsafe_ptr().bitcast[Float32](), rl)
    var g = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
    var o = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl)
    ctx.enqueue_function[_logsoftmax_bwd_rows_f32, _logsoftmax_bwd_rows_f32](
        l, g, o, cols, grid_dim=rows, block_dim=_TPB)
    ctx.synchronize()
    var os = List[Int]()
    os.append(rows); os.append(cols)
    return Tensor(out_buf^, os^, STDtype.F32)


# ── sum / mean backward (scalar grad broadcast to in_shape) ──────────────────
def _broadcast_bwd(
    grad_scalar: Float32, in_shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var n = 1
    for i in range(len(in_shape)):
        n *= in_shape[i]
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_broadcast_scalar_k, _broadcast_scalar_k](
        o, grad_scalar, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var sh = List[Int]()
    for i in range(len(in_shape)):
        sh.append(in_shape[i])
    return Tensor(out_buf^, sh^, STDtype.F32)


def sum_backward(
    grad_out_scalar: Float32, in_shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """d_x = ones(in_shape) * grad_out_scalar."""
    return _broadcast_bwd(grad_out_scalar, in_shape, ctx)


def mean_backward(
    grad_out_scalar: Float32, in_shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """d_x = (grad_out_scalar / N) broadcast to in_shape."""
    var n = 1
    for i in range(len(in_shape)):
        n *= in_shape[i]
    return _broadcast_bwd(grad_out_scalar / Float32(n), in_shape, ctx)
