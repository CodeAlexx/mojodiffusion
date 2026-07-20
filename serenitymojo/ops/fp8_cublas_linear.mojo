# Native Blackwell FP8 E4M3 linear with outer-vector scaling.
#
# Existing Serenity FP8 caches store one F32 scale per output row. Activations
# are quantized per flattened token row at runtime. cuBLASLt consumes both scale
# vectors directly, so the matrix weight never expands to BF16.

from std.collections import List
from std.ffi import external_call
from std.gpu import barrier, block_dim, block_idx, global_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUDA
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _E4M3_MAX = Float32(448.0)


def _quantize_activation_outer_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float8_e4m3fn, _DYN1, MutAnyOrigin],
    cols: Int,
    rows: Int,
):
    """One-block-per-row absmax reduction and E4M3 encoding."""
    var row = Int(block_idx.x)
    if row >= rows:
        return
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var row_base = row * cols
    var row_max: Float32 = 0.0
    var col = tid
    while col < cols:
        var value = Float32(
            rebind[Scalar[DType.bfloat16]](x[row_base + col])
        )
        var magnitude = value if value >= 0 else -value
        if magnitude > row_max:
            row_max = magnitude
        col += _BLOCK
    shared[tid] = row_max
    barrier()
    var active = _BLOCK // 2
    while active > 0:
        if tid < active:
            var other = shared[tid + active]
            if other > shared[tid]:
                shared[tid] = other
        barrier()
        active //= 2
    if tid == 0:
        var scale = (
            Float32(shared[0]) / _E4M3_MAX
            if shared[0] > Float32(0.0)
            else Float32(1.0)
        )
        shared[0] = scale
        s[row] = rebind[s.element_type](scale)
    barrier()
    var scale = Float32(shared[0])
    col = tid
    while col < cols:
        var value = Float32(
            rebind[Scalar[DType.bfloat16]](x[row_base + col])
        )
        o[row_base + col] = rebind[o.element_type](
            (value / scale).cast[DType.float8_e4m3fn]()
        )
        col += _BLOCK


def _pad_rows_bf16_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    actual_numel: Int,
    padded_numel: Int,
):
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    while i < padded_numel:
        if i < actual_numel:
            o[i] = x[i]
        else:
            o[i] = rebind[o.element_type](BFloat16(0.0))
        i += stride


def _quantize_activation_outer(
    x: Tensor, actual_m: Int, padded_m: Int, k: Int, ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var actual_numel = actual_m * k
    var padded_numel = padded_m * k
    var x_in_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](actual_numel))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_in_rl
    )
    var x_grid = (padded_numel + _BLOCK - 1) // _BLOCK
    if x_grid > 65535:
        x_grid = 65535

    var scale_buf = ctx.enqueue_create_buffer[DType.uint8](padded_m * 4)
    var scale_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](padded_m))
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale_buf.unsafe_ptr().bitcast[Float32](), scale_rl
    )
    var fp8_buf = ctx.enqueue_create_buffer[DType.uint8](padded_numel)
    var fp8_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](padded_numel))
    var O = LayoutTensor[DType.float8_e4m3fn, _DYN1, MutAnyOrigin](
        fp8_buf.unsafe_ptr().bitcast[Float8_e4m3fn](), fp8_rl
    )
    if actual_numel == padded_numel:
        ctx.enqueue_function[
            _quantize_activation_outer_kernel,
            _quantize_activation_outer_kernel,
        ](
            X, S, O, k, padded_m,
            grid_dim=padded_m, block_dim=_BLOCK,
        )
    else:
        var x_buf = ctx.enqueue_create_buffer[DType.uint8](padded_numel * 2)
        var x_out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](padded_numel))
        var XP = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x_buf.unsafe_ptr().bitcast[BFloat16](), x_out_rl
        )
        ctx.enqueue_function[_pad_rows_bf16_kernel, _pad_rows_bf16_kernel](
            X, XP, actual_numel, padded_numel,
            grid_dim=x_grid, block_dim=_BLOCK,
        )
        ctx.enqueue_function[
            _quantize_activation_outer_kernel,
            _quantize_activation_outer_kernel,
        ](
            XP, S, O, k, padded_m,
            grid_dim=padded_m, block_dim=_BLOCK,
        )
    var fp8_shape: List[Int] = [padded_m, k]
    var scale_shape: List[Int] = [padded_m]
    return (
        Tensor(fp8_buf^, fp8_shape^, STDtype.F8_E4M3),
        Tensor(scale_buf^, scale_shape^, STDtype.F32),
    )


def _scale_bias_bf16_kernel(
    raw: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    xs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ws: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    bias: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    m: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = m * n
    while i < total:
        var row = i // n
        var col = i - row * n
        var v = rebind[Scalar[DType.float32]](raw[i])
        v *= rebind[Scalar[DType.float32]](xs[row])
        v *= rebind[Scalar[DType.float32]](ws[col])
        v += Float32(rebind[Scalar[DType.bfloat16]](bias[col]))
        o[i] = rebind[o.element_type](v.cast[DType.bfloat16]())
        i += stride


def fp8_cublas_linear_outer(
    x: Tensor,
    weight_fp8: Tensor,
    weight_scale: Tensor,
    bias: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """BF16 linear from E4M3 weight + per-row F32 weight scales."""
    if x.dtype() != STDtype.BF16:
        raise Error("fp8_cublas_linear_outer: activation must be BF16")
    if weight_fp8.dtype() != STDtype.F8_E4M3:
        raise Error("fp8_cublas_linear_outer: weight must be F8_E4M3")
    if weight_scale.dtype() != STDtype.F32:
        raise Error("fp8_cublas_linear_outer: weight scale must be F32")
    if bias.dtype() != STDtype.BF16:
        raise Error("fp8_cublas_linear_outer: bias must be BF16")
    var x_shape = x.shape()
    var w_shape = weight_fp8.shape()
    if len(x_shape) < 2 or len(w_shape) != 2:
        raise Error("fp8_cublas_linear_outer: invalid tensor rank")
    var k = x_shape[len(x_shape) - 1]
    var n = w_shape[0]
    if w_shape[1] != k or weight_scale.numel() != n or bias.numel() != n:
        raise Error("fp8_cublas_linear_outer: shape mismatch")
    var actual_m = x.numel() // k
    var padded_m = ((actual_m + 15) // 16) * 16
    var quantized = _quantize_activation_outer(
        x, actual_m, padded_m, k, ctx
    )
    ref x_fp8 = quantized[0]
    ref x_scale = quantized[1]
    var raw_buf = ctx.enqueue_create_buffer[DType.uint8](padded_m * n * 4)

    var x_ptr = BytePtr(unsafe_from_address=Int(x_fp8.buf.unsafe_ptr()))
    var w_ptr = BytePtr(unsafe_from_address=Int(weight_fp8.buf.unsafe_ptr()))
    var raw_ptr = BytePtr(unsafe_from_address=Int(raw_buf.unsafe_ptr()))
    var stream = CUDA(ctx.stream())
    var rc = Int(external_call[
        "serenity_cublas_gemm_fp8e4m3_f32_rowmajor_nt", Int32
    ](
        x_ptr, w_ptr, raw_ptr,
        Int32(padded_m), Int32(n), Int32(k), stream,
    ))
    if rc != 0:
        raise Error(
            String("fp8_cublas_linear_outer: shim rc=") + String(rc)
            + String(" M=") + String(padded_m)
            + String(" N=") + String(n) + String(" K=") + String(k)
        )

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](actual_m * n * 2)
    var raw_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](padded_m * n))
    var xs_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](padded_m))
    var ws_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var bias_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](actual_m * n))
    var RAW = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        raw_buf.unsafe_ptr().bitcast[Float32](), raw_rl
    )
    var XS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x_scale.buf.unsafe_ptr().bitcast[Float32](), xs_rl
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight_scale.buf.unsafe_ptr().bitcast[Float32](), ws_rl
    )
    var BIAS = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        bias.buf.unsafe_ptr().bitcast[BFloat16](), bias_rl
    )
    var OUT = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )
    var total = actual_m * n
    var grid = (total + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_scale_bias_bf16_kernel, _scale_bias_bf16_kernel](
        RAW, XS, WS, BIAS, OUT, actual_m, n,
        grid_dim=grid, block_dim=_BLOCK,
    )

    var out_shape = List[Int]()
    for i in range(len(x_shape) - 1):
        out_shape.append(x_shape[i])
    out_shape.append(n)
    return Tensor(out_buf^, out_shape^, STDtype.BF16)
