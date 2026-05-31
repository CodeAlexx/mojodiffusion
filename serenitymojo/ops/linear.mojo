# ops/linear.mojo — linear(x, weight, bias) = x @ weightᵀ + bias.
#
# Weight is PyTorch row-major [out, in] (flame-core's "native" layout). We use
# `linalg.matmul.vendor.blas.matmul` with `transpose_b=True` so the GEMM is
#   C[M, out] = A[M, in] @ B[out, in]ᵀ
# and `c_row_major=True` so C comes back row-major (the default is column-major,
# which produced a transposed result in the probe — verified empirically).
#
# Shapes are RUNTIME (M/K/N vary per call), so all LayoutTensors use a DYNAMIC
# layout (`Layout.row_major(-1, -1)`) plus a `RuntimeLayout.row_major(IndexList)`
# carrying the concrete dims. (Compile-time `Layout.row_major(m, n)` from runtime
# ints is rejected: "cannot use a dynamic value in a parameter list".)
#
# x may be N-D: leading dims flatten to M = prod(leading); trailing dim is the
# contraction `in`. Output is [...leading, out]. Compute path:
#   1. matmul (A,B in stored dtype, C accumulated in F32) -> C_f32[M, out].
#   2. bias-add + cast kernel: out[m, j] = (C_f32[m, j] + bias[j]) -> dtype.
# F32 accumulation throughout. BF16/F16 only at the storage boundary.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _bias_cast_kernel_f32(
    c_f32: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    out_buf: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    m: Int,
    out_dim: Int,
    has_bias: Int,
):
    var idx = Int(global_idx.x)
    var total = m * out_dim
    if idx < total:
        var row = idx // out_dim
        var col = idx % out_dim
        var v = rebind[Scalar[DType.float32]](c_f32[row, col])
        if has_bias != 0:
            v += rebind[Scalar[DType.float32]](bias[col])
        out_buf[row, col] = rebind[out_buf.element_type](v)


def _bias_cast_kernel_bf16(
    c_f32: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    out_buf: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    m: Int,
    out_dim: Int,
    has_bias: Int,
):
    var idx = Int(global_idx.x)
    var total = m * out_dim
    if idx < total:
        var row = idx // out_dim
        var col = idx % out_dim
        var v = rebind[Scalar[DType.float32]](c_f32[row, col])
        if has_bias != 0:
            v += rebind[Scalar[DType.float32]](bias[col])
        out_buf[row, col] = rebind[out_buf.element_type](v.cast[DType.bfloat16]())


def _bias_cast_kernel_f16(
    c_f32: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    out_buf: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    m: Int,
    out_dim: Int,
    has_bias: Int,
):
    var idx = Int(global_idx.x)
    var total = m * out_dim
    if idx < total:
        var row = idx // out_dim
        var col = idx % out_dim
        var v = rebind[Scalar[DType.float32]](c_f32[row, col])
        if has_bias != 0:
            v += rebind[Scalar[DType.float32]](bias[col])
        out_buf[row, col] = rebind[out_buf.element_type](v.cast[DType.float16]())


def _bias_cast_direct_kernel_f32(
    c_f32: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    out_buf: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    m: Int,
    out_dim: Int,
):
    var idx = Int(global_idx.x)
    var total = m * out_dim
    if idx < total:
        var row = idx // out_dim
        var col = idx % out_dim
        var v = rebind[Scalar[DType.float32]](c_f32[row, col])
        v += rebind[Scalar[DType.float32]](bias[col])
        out_buf[row, col] = rebind[out_buf.element_type](v)


def _bias_cast_direct_kernel_bf16(
    c_f32: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    out_buf: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    m: Int,
    out_dim: Int,
):
    var idx = Int(global_idx.x)
    var total = m * out_dim
    if idx < total:
        var row = idx // out_dim
        var col = idx % out_dim
        var v = rebind[Scalar[DType.float32]](c_f32[row, col])
        v += rebind[Scalar[DType.bfloat16]](bias[col]).cast[DType.float32]()
        out_buf[row, col] = rebind[out_buf.element_type](v.cast[DType.bfloat16]())


def _bias_cast_direct_kernel_f16(
    c_f32: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    out_buf: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    m: Int,
    out_dim: Int,
):
    var idx = Int(global_idx.x)
    var total = m * out_dim
    if idx < total:
        var row = idx // out_dim
        var col = idx % out_dim
        var v = rebind[Scalar[DType.float32]](c_f32[row, col])
        v += rebind[Scalar[DType.float16]](bias[col]).cast[DType.float32]()
        out_buf[row, col] = rebind[out_buf.element_type](v.cast[DType.float16]())


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Tensor:
    """y = x @ weightᵀ + bias.

    x:      [..., in]            (any compute dtype; leading dims flattened to M)
    weight: [out, in]            (PyTorch row-major; same dtype as x)
    bias:   [out] or None        (same dtype as x)
    returns [..., out]           (x's dtype; F32-accumulated GEMM + bias add).
    """
    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1:
        raise Error("linear: x must have rank >= 1")
    if len(wshape) != 2:
        raise Error("linear: weight must be rank-2 [out, in]")
    var in_dim = xshape[len(xshape) - 1]
    var out_dim = wshape[0]
    var k = wshape[1]
    if k != in_dim:
        raise Error(
            String("linear: weight in-dim ")
            + String(k)
            + " != x last dim "
            + String(in_dim)
        )
    if x.dtype() != weight.dtype():
        raise Error("linear: x and weight dtype mismatch")
    var m = 1
    for i in range(len(xshape) - 1):
        m *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()

    # GEMM into an F32 device buffer (C is M x out, row-major).
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](m * out_dim * 4)
    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, in_dim))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_dim, k))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, out_dim))
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        c_buf.unsafe_ptr().bitcast[Float32](), c_rl
    )

    if dt == DType.float32:
        var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), a_rl
        )
        var B = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float32](), b_rl
        )
        matmul(ctx, C, A, B, transpose_b=True, c_row_major=True)
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), a_rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
        )
        matmul(ctx, C, A, B, transpose_b=True, c_row_major=True)
    else:  # float16
        var A = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), a_rl
        )
        var B = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float16](), b_rl
        )
        matmul(ctx, C, A, B, transpose_b=True, c_row_major=True)

    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        out_shape.append(xshape[i])
    out_shape.append(out_dim)

    var has_bias = 1 if bias else 0
    if dt == DType.float32 and has_bias == 0:
        return Tensor(c_buf^, out_shape^, x.dtype())

    # Output buffer in x's dtype.
    var bsz = x.dtype().byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](m * out_dim * bsz)

    # Bias stays on device. No bias -> length-1 dummy + has_bias=0 using the
    # older F32-bias kernels. Biases are stored in the same dtype as weights.
    var bias_count = out_dim if bias else 1
    var bias_f32_buf = ctx.enqueue_create_buffer[DType.uint8](bias_count * 4)
    if bias:
        if bias.value().dtype() != x.dtype():
            raise Error("linear: bias dtype mismatch")
        var bshape = bias.value().shape()
        if len(bshape) != 1 or bshape[0] != out_dim:
            raise Error(
                String("linear: bias shape mismatch, expected [")
                + String(out_dim)
                + "]"
            )

    var bias_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](bias_count))
    var c_out_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, out_dim))
    var bias_lt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        bias_f32_buf.unsafe_ptr().bitcast[Float32](), bias_rl
    )
    var c_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        c_buf.unsafe_ptr().bitcast[Float32](), c_out_rl
    )

    var total = m * out_dim
    var grid = (total + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var o_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), c_out_rl
        )
        if bias:
            var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                bias.value().buf.unsafe_ptr().bitcast[Float32](), bias_rl
            )
            ctx.enqueue_function[
                _bias_cast_direct_kernel_f32, _bias_cast_direct_kernel_f32
            ](
                c_lt, B, o_lt, m, out_dim,
                grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            ctx.enqueue_function[_bias_cast_kernel_f32, _bias_cast_kernel_f32](
                c_lt, bias_lt, o_lt, m, out_dim, has_bias,
                grid_dim=grid, block_dim=_BLOCK,
            )
    elif dt == DType.bfloat16:
        var o_lt = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), c_out_rl
        )
        if bias:
            var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                bias.value().buf.unsafe_ptr().bitcast[BFloat16](), bias_rl
            )
            ctx.enqueue_function[
                _bias_cast_direct_kernel_bf16, _bias_cast_direct_kernel_bf16
            ](
                c_lt, B, o_lt, m, out_dim,
                grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            ctx.enqueue_function[_bias_cast_kernel_bf16, _bias_cast_kernel_bf16](
                c_lt, bias_lt, o_lt, m, out_dim, has_bias,
                grid_dim=grid, block_dim=_BLOCK,
            )
    else:  # float16
        var o_lt = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), c_out_rl
        )
        if bias:
            var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                bias.value().buf.unsafe_ptr().bitcast[Float16](), bias_rl
            )
            ctx.enqueue_function[
                _bias_cast_direct_kernel_f16, _bias_cast_direct_kernel_f16
            ](
                c_lt, B, o_lt, m, out_dim,
                grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            ctx.enqueue_function[_bias_cast_kernel_f16, _bias_cast_kernel_f16](
                c_lt, bias_lt, o_lt, m, out_dim, has_bias,
                grid_dim=grid, block_dim=_BLOCK,
            )
    # TIER2-SYNC-REMOVED: single-stream enqueue serializes kernel order; the
    # downstream .to_host()/optimizer barrier is the only required sync. Output is
    # a device buffer with no host-staging buffer to protect here.
    return Tensor(out_buf^, out_shape^, x.dtype())


def linear_scratch(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    reverse: Bool = False,
) raises -> Tensor:
    """Scratch-backed F32 no-bias linear.

    This is an opt-in allocation variant for proven short-lived frame outputs.
    Bias and non-F32 paths fall back to `linear`, preserving the normal dtype
    and cast behavior.
    """
    if bias or x.dtype() != STDtype.F32 or weight.dtype() != STDtype.F32:
        return linear(x, weight, bias, ctx)

    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1:
        raise Error("linear_scratch: x must have rank >= 1")
    if len(wshape) != 2:
        raise Error("linear_scratch: weight must be rank-2 [out, in]")
    var in_dim = xshape[len(xshape) - 1]
    var out_dim = wshape[0]
    var k = wshape[1]
    if k != in_dim:
        raise Error(
            String("linear_scratch: weight in-dim ")
            + String(k)
            + " != x last dim "
            + String(in_dim)
        )

    var m = 1
    for i in range(len(xshape) - 1):
        m *= xshape[i]

    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        out_shape.append(xshape[i])
    out_shape.append(out_dim)
    var out: Tensor
    if reverse:
        out = scratch.alloc_tensor_reverse(out_shape^, STDtype.F32)
    else:
        out = scratch.alloc_tensor(out_shape^, STDtype.F32)

    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, in_dim))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_dim, k))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, out_dim))
    var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), a_rl
    )
    var B = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[Float32](), b_rl
    )
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out.buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    matmul(ctx, C, A, B, transpose_b=True, c_row_major=True)
    return out^


def linear_rows(
    x: Tensor,
    weight: Tensor,
    row_start: Int,
    row_count: Int,
    ctx: DeviceContext,
    alpha: Float32 = 1.0,
) raises -> Tensor:
    """F32 no-bias linear over a contiguous weight-row range."""
    if x.dtype() != STDtype.F32 or weight.dtype() != STDtype.F32:
        raise Error("linear_rows: F32 tensors required")

    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1:
        raise Error("linear_rows: x must have rank >= 1")
    if len(wshape) != 2:
        raise Error("linear_rows: weight must be rank-2 [out, in]")
    var in_dim = xshape[len(xshape) - 1]
    var out_dim = wshape[0]
    var k = wshape[1]
    if k != in_dim:
        raise Error("linear_rows: weight in-dim != x last dim")
    if row_start < 0 or row_count < 0 or row_start + row_count > out_dim:
        raise Error("linear_rows: row range out of bounds")

    var m = 1
    for i in range(len(xshape) - 1):
        m *= xshape[i]

    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        out_shape.append(xshape[i])
    out_shape.append(row_count)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](m * row_count * 4)

    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, in_dim))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](row_count, k))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, row_count))
    var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), a_rl
    )
    var B = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[Float32]() + row_start * k, b_rl
    )
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    matmul(ctx, C, A, B, transpose_b=True, c_row_major=True, alpha=alpha)
    return Tensor(out_buf^, out_shape^, STDtype.F32)


def linear_rows_scratch(
    x: Tensor,
    weight: Tensor,
    row_start: Int,
    row_count: Int,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    reverse: Bool = False,
    alpha: Float32 = 1.0,
) raises -> Tensor:
    """Scratch-backed F32 no-bias linear over a contiguous weight-row range."""
    if x.dtype() != STDtype.F32 or weight.dtype() != STDtype.F32:
        raise Error("linear_rows_scratch: F32 tensors required")

    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1:
        raise Error("linear_rows_scratch: x must have rank >= 1")
    if len(wshape) != 2:
        raise Error("linear_rows_scratch: weight must be rank-2 [out, in]")
    var in_dim = xshape[len(xshape) - 1]
    var out_dim = wshape[0]
    var k = wshape[1]
    if k != in_dim:
        raise Error("linear_rows_scratch: weight in-dim != x last dim")
    if row_start < 0 or row_count < 0 or row_start + row_count > out_dim:
        raise Error("linear_rows_scratch: row range out of bounds")

    var m = 1
    for i in range(len(xshape) - 1):
        m *= xshape[i]

    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        out_shape.append(xshape[i])
    out_shape.append(row_count)
    var out: Tensor
    if reverse:
        out = scratch.alloc_tensor_reverse(out_shape^, STDtype.F32)
    else:
        out = scratch.alloc_tensor(out_shape^, STDtype.F32)

    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, in_dim))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](row_count, k))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, row_count))
    var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), a_rl
    )
    var B = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[Float32]() + row_start * k, b_rl
    )
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out.buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    matmul(ctx, C, A, B, transpose_b=True, c_row_major=True, alpha=alpha)
    return out^


def linear_two_inputs_scratch(
    x0: Tensor,
    x1: Tensor,
    weight0: Tensor,
    weight1: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    reverse: Bool = False,
) raises -> Tensor:
    """Scratch-backed F32 no-bias linear over two input blocks.

    Computes x0 @ weight0.T + x1 @ weight1.T, where both weights have the same
    output dimension and contiguous row-major storage.
    """
    if (
        x0.dtype() != STDtype.F32
        or x1.dtype() != STDtype.F32
        or weight0.dtype() != STDtype.F32
        or weight1.dtype() != STDtype.F32
    ):
        raise Error("linear_two_inputs_scratch: F32 tensors required")

    var x0shape = x0.shape()
    var x1shape = x1.shape()
    var w0shape = weight0.shape()
    var w1shape = weight1.shape()
    if len(x0shape) < 1 or len(x1shape) < 1:
        raise Error("linear_two_inputs_scratch: inputs must have rank >= 1")
    if len(w0shape) != 2 or len(w1shape) != 2:
        raise Error("linear_two_inputs_scratch: weights must be rank-2")
    var in0 = x0shape[len(x0shape) - 1]
    var in1 = x1shape[len(x1shape) - 1]
    var out_dim = w0shape[0]
    if w0shape[1] != in0 or w1shape[1] != in1 or w1shape[0] != out_dim:
        raise Error("linear_two_inputs_scratch: shape mismatch")

    var m = 1
    for i in range(len(x0shape) - 1):
        m *= x0shape[i]
    var m1 = 1
    for i in range(len(x1shape) - 1):
        m1 *= x1shape[i]
    if m1 != m:
        raise Error("linear_two_inputs_scratch: leading dimensions mismatch")

    var out_shape = List[Int]()
    for i in range(len(x0shape) - 1):
        out_shape.append(x0shape[i])
    out_shape.append(out_dim)
    var out: Tensor
    if reverse:
        out = scratch.alloc_tensor_reverse(out_shape^, STDtype.F32)
    else:
        out = scratch.alloc_tensor(out_shape^, STDtype.F32)

    var x0_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, in0))
    var x1_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, in1))
    var w0_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_dim, in0))
    var w1_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_dim, in1))
    var out_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, out_dim))
    var X0 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        x0.buf.unsafe_ptr().bitcast[Float32](), x0_rl
    )
    var X1 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        x1.buf.unsafe_ptr().bitcast[Float32](), x1_rl
    )
    var W0 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        weight0.buf.unsafe_ptr().bitcast[Float32](), w0_rl
    )
    var W1 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        weight1.buf.unsafe_ptr().bitcast[Float32](), w1_rl
    )
    var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out.buf.unsafe_ptr().bitcast[Float32](), out_rl
    )
    matmul(ctx, O, X0, W0, transpose_b=True, c_row_major=True)
    matmul(ctx, O, X1, W1, transpose_b=True, c_row_major=True, beta=1.0)
    return out^
