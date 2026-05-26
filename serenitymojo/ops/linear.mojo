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

    # Output buffer in x's dtype.
    var bsz = x.dtype().byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](m * out_dim * bsz)

    # Bias staged as F32 device buffer (cast up from stored dtype). No bias ->
    # length-1 dummy + has_bias=0.
    var has_bias = 1 if bias else 0
    var bias_count = out_dim if bias else 1
    var bias_f32_buf = ctx.enqueue_create_buffer[DType.uint8](bias_count * 4)
    if bias:
        var bvals = bias.value().to_host(ctx)
        if len(bvals) != out_dim:
            raise Error(
                String("linear: bias length ")
                + String(len(bvals))
                + " != out_dim "
                + String(out_dim)
            )
        var bhost = ctx.enqueue_create_host_buffer[DType.uint8](out_dim * 4)
        var bp = bhost.unsafe_ptr().bitcast[Float32]()
        for i in range(out_dim):
            bp[i] = bvals[i]
        ctx.enqueue_copy(dst_buf=bias_f32_buf, src_buf=bhost)
        ctx.synchronize()

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
        ctx.enqueue_function[_bias_cast_kernel_f32, _bias_cast_kernel_f32](
            c_lt, bias_lt, o_lt, m, out_dim, has_bias,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var o_lt = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), c_out_rl
        )
        ctx.enqueue_function[_bias_cast_kernel_bf16, _bias_cast_kernel_bf16](
            c_lt, bias_lt, o_lt, m, out_dim, has_bias,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:  # float16
        var o_lt = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), c_out_rl
        )
        ctx.enqueue_function[_bias_cast_kernel_f16, _bias_cast_kernel_f16](
            c_lt, bias_lt, o_lt, m, out_dim, has_bias,
            grid_dim=grid, block_dim=_BLOCK,
        )
    ctx.synchronize()

    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        out_shape.append(xshape[i])
    out_shape.append(out_dim)
    return Tensor(out_buf^, out_shape^, x.dtype())
