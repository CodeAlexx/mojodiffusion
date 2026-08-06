from std.ffi import external_call
from std.collections import List
from std.gpu import block_dim, global_idx, grid_dim
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUDA
from std.math import min
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.ops.int8_quant import int8_encode_perrow, int8_rowscale
from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _MAX_GEMM_ROWS = 4096


def _h3_cast_f32_chunk_to_bf16(
    c: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    row_start: Int,
    chunk_rows: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = chunk_rows * n
    while i < total:
        var row = i // n
        var col = i - row * n
        var value = rebind[Scalar[DType.float32]](c[i])
        output[(row_start + row) * n + col] = rebind[output.element_type](
            value.cast[DType.bfloat16]()
        )
        i += stride


def minimax_h3_bf16_linear_chunked(
    x: Tensor,
    weight: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """BF16 projection with a bounded F32 accumulator over sequence rows.

    `ops.linear` correctly accumulates BF16 GEMM in F32, but allocates the
    complete F32 `[M,N]` result before casting it back to BF16. At H3's
    960x544/15-second QKV shape that temporary alone is about 4.8 GiB. Rows
    are independent, so reuse one 4096-row F32 accumulator and cast each
    completed chunk into the full BF16 output. The K reduction and rounding
    contract are unchanged.
    """
    if x.dtype() != STDtype.BF16 or weight.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 chunked linear requires BF16 x and weight")
    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1 or len(wshape) != 2:
        raise Error("MiniMax-H3 chunked linear expects x[...,K], weight[N,K]")
    var k = xshape[len(xshape) - 1]
    var n = wshape[0]
    if wshape[1] != k:
        raise Error("MiniMax-H3 chunked linear contraction mismatch")
    var m = x.numel() // k
    var accumulator_rows = min(m, _MAX_GEMM_ROWS)
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](
        accumulator_rows * n * 4
    )
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](m * n * 2)
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n, k))
    var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
    )
    var CFlat = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        c_buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](accumulator_rows * n)),
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](m * n)),
    )
    for row_start in range(0, m, _MAX_GEMM_ROWS):
        var chunk_rows = min(_MAX_GEMM_ROWS, m - row_start)
        var a_rl = RuntimeLayout[_DYN2].row_major(
            IndexList[2](chunk_rows, k)
        )
        var c_rl = RuntimeLayout[_DYN2].row_major(
            IndexList[2](chunk_rows, n)
        )
        var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16]() + row_start * k, a_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            c_buf.unsafe_ptr().bitcast[Float32](), c_rl
        )
        matmul(ctx, C, A, B, transpose_b=True, c_row_major=True)
        var total = chunk_rows * n
        var grid = (total + _BLOCK - 1) // _BLOCK
        if grid > 65535:
            grid = 65535
        ctx.enqueue_function[
            _h3_cast_f32_chunk_to_bf16, _h3_cast_f32_chunk_to_bf16
        ](
            CFlat, O, row_start, chunk_rows, n,
            grid_dim=grid, block_dim=_BLOCK,
        )
    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        out_shape.append(xshape[i])
    out_shape.append(n)
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


def _h3_gemm_s8s8s32_nt(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    m: Int,
    n: Int,
    k: Int,
    row_start: Int,
    ctx: DeviceContext,
) raises:
    var a_ptr = BytePtr(
        unsafe_from_address=Int(a.buf.unsafe_ptr()) + row_start * k
    )
    var b_ptr = BytePtr(unsafe_from_address=Int(b.buf.unsafe_ptr()))
    var c_ptr = BytePtr(unsafe_from_address=Int(c.buf.unsafe_ptr()))
    var stream = CUDA(ctx.stream())
    var rc = Int(
        external_call[
            "serenity_minimax_h3_gemm_s8s8s32_rowmajor_nt", Int32
        ](
            a_ptr, b_ptr, c_ptr,
            Int32(m), Int32(n), Int32(k), stream,
        )
    )
    if rc != 0:
        raise Error(
            String("MiniMax-H3 INT8 GEMM failed rc=") + String(rc)
            + String(" M=") + String(m)
            + String(" N=") + String(n)
            + String(" K=") + String(k)
        )


def _h3_int8_dequant_rowscale(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    x_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    w_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    row_start: Int,
    chunk_rows: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = chunk_rows * n
    while i < total:
        var row = i // n
        var col = i - row * n
        var output_row = row_start + row
        var acc = Float32(rebind[Scalar[DType.int32]](c[i]))
        var value = (
            acc
            * rebind[Scalar[DType.float32]](x_scale[output_row])
            * rebind[Scalar[DType.float32]](w_scale[col])
        )
        output[output_row * n + col] = rebind[output.element_type](
            value.cast[DType.bfloat16]()
        )
        i += stride


def minimax_h3_int8_linear(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Direct H3 W8A8 projection with arbitrary sequence-row count.

    Activations are quantized per token on GPU. Weights are persistent
    per-output-row INT8 with F32 scales. The GEMM accumulates exactly in I32,
    then a GPU kernel rescales and rounds once to BF16.
    """
    if x.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 INT8 linear requires BF16 activations")
    if weight.dtype() != STDtype.I8:
        raise Error("MiniMax-H3 INT8 linear requires I8 weights")
    if weight_scale.dtype() != STDtype.F32:
        raise Error("MiniMax-H3 INT8 linear requires F32 row scales")
    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1 or len(wshape) != 2:
        raise Error("MiniMax-H3 INT8 linear expects x[...,K], weight[N,K]")
    var k = xshape[len(xshape) - 1]
    var n = wshape[0]
    if wshape[1] != k:
        raise Error("MiniMax-H3 INT8 linear contraction mismatch")
    if weight_scale.numel() != n:
        raise Error("MiniMax-H3 INT8 linear scale length mismatch")
    var m = x.numel() // k

    var x_scale = int8_rowscale(x, ctx)
    var x_int8 = int8_encode_perrow(x, x_scale, ctx)
    # The old path allocated I32[M,N] for the entire video sequence. At long
    # H3 shapes the largest MLP projection alone needed more than 2 GiB, even
    # though cuBLAS and the rescale kernel consume independent rows. Reuse one
    # bounded GPU accumulator chunk; stream ordering preserves exact I32 GEMM
    # and BF16 rounding while making resolution and duration independently
    # selectable on a 24-GiB card.
    var accumulator_rows = min(m, _MAX_GEMM_ROWS)
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](
        accumulator_rows * n * 4
    )
    var c_shape: List[Int] = [accumulator_rows, n]
    var c = Tensor(c_buf^, c_shape^, STDtype.I32)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](m * n * 2)
    var flat_c = RuntimeLayout[_DYN1].row_major(
        IndexList[1](accumulator_rows * n)
    )

    var flat_m = RuntimeLayout[_DYN1].row_major(IndexList[1](m))
    var flat_n = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        c.buf.unsafe_ptr().bitcast[Int32](), flat_c
    )
    var XS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x_scale.buf.unsafe_ptr().bitcast[Float32](), flat_m
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight_scale.buf.unsafe_ptr().bitcast[Float32](), flat_n
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](m * n)),
    )
    for row_start in range(0, m, _MAX_GEMM_ROWS):
        var chunk_rows = min(_MAX_GEMM_ROWS, m - row_start)
        _h3_gemm_s8s8s32_nt(
            x_int8, weight, c, chunk_rows, n, k, row_start, ctx
        )
        var total = chunk_rows * n
        var grid = (total + _BLOCK - 1) // _BLOCK
        if grid > 65535:
            grid = 65535
        ctx.enqueue_function[
            _h3_int8_dequant_rowscale, _h3_int8_dequant_rowscale
        ](
            C, XS, WS, O, row_start, chunk_rows, n,
            grid_dim=grid, block_dim=_BLOCK,
        )

    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        out_shape.append(xshape[i])
    out_shape.append(n)
    return Tensor(out_buf^, out_shape^, STDtype.BF16)
