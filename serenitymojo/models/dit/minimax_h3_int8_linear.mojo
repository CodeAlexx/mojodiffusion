from std.ffi import external_call
from std.collections import List
from std.gpu import barrier, block_dim, block_idx, global_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUDA
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.math import exp, min, round
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.ops.int8_quant import (
    int8_dequant_groupwise_to_bf16,
    int8_encode_perrow,
    int8_rowscale,
)
from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _MAX_GEMM_ROWS = 4096
comptime _MLP_GEMM_ROWS = 1024
comptime _GROUPWISE_MLP_ROWS = 256
comptime _I8_MAX = Float32(127.0)
comptime _SCALE_FLOOR = Float32(1.0e-30)


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


def _h3_int8_dequant_swiglu_rowscale(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    x_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    w_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    row_start: Int,
    chunk_rows: Int,
    f: Int,
):
    """Dequantize packed `[value|gate]` FC1 rows and apply SwiGLU.

    This preserves both BF16 boundaries from the unfused path: projection
    outputs round to BF16 first, and SiLU(gate) rounds to BF16 before its
    product with value. Only the otherwise materialized `[M, 2*f]` tensor is
    removed.
    """
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = chunk_rows * f
    var packed_n = 2 * f
    while i < total:
        var row = i // f
        var col = i - row * f
        var output_row = row_start + row
        var value_acc = Float32(
            rebind[Scalar[DType.int32]](c[row * packed_n + col])
        )
        var gate_acc = Float32(
            rebind[Scalar[DType.int32]](c[row * packed_n + f + col])
        )
        var xs = rebind[Scalar[DType.float32]](x_scale[output_row])
        var value = (value_acc * xs * rebind[Scalar[DType.float32]](
            w_scale[col]
        )).cast[DType.bfloat16]().cast[DType.float32]()
        var gate = (gate_acc * xs * rebind[Scalar[DType.float32]](
            w_scale[f + col]
        )).cast[DType.bfloat16]().cast[DType.float32]()
        var silu_gate = gate / (1.0 + exp(-gate))
        silu_gate = silu_gate.cast[DType.bfloat16]().cast[DType.float32]()
        output[output_row * f + col] = rebind[output.element_type](
            (silu_gate * value).cast[DType.bfloat16]()
        )
        i += stride


def _h3_int8_dequant_residual_gate_rowscale(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    x_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    w_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    residual: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    gate: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    row_start: Int,
    chunk_rows: Int,
    n: Int,
):
    """Dequantize FC2 and apply `residual + gate * fc2` directly.

    The projected value is explicitly rounded to BF16 before the residual
    expression, matching `minimax_h3_int8_linear` followed by
    `residual_gate`; only the full intermediate FC2 tensor is removed.
    """
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = chunk_rows * n
    while i < total:
        var row = i // n
        var col = i - row * n
        var output_index = (row_start + row) * n + col
        var acc = Float32(rebind[Scalar[DType.int32]](c[i]))
        var projected = (
            acc
            * rebind[Scalar[DType.float32]](x_scale[row_start + row])
            * rebind[Scalar[DType.float32]](w_scale[col])
        ).cast[DType.bfloat16]().cast[DType.float32]()
        var residual_value = rebind[Scalar[DType.bfloat16]](
            residual[output_index]
        ).cast[DType.float32]()
        var gate_value = rebind[Scalar[DType.bfloat16]](
            gate[output_index]
        ).cast[DType.float32]()
        output[output_index] = rebind[output.element_type](
            (residual_value + gate_value * projected).cast[DType.bfloat16]()
        )
        i += stride


def _h3_int8_rowscale_chunk(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    row_start: Int,
    cols: Int,
    rows: Int,
):
    """Exact `int8_rowscale`, reading a row window from a full tensor."""
    var row = Int(block_idx.x)
    if row >= rows:
        return
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var maximum: Float32 = 0.0
    var base = (row_start + row) * cols
    var col = tid
    while col < cols:
        var value = Float32(
            rebind[Scalar[DType.bfloat16]](x[base + col])
        )
        var magnitude = value if value >= 0 else -value
        if magnitude > maximum:
            maximum = magnitude
        col += _BLOCK
    shared[tid] = maximum
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
        var value = Float32(shared[0]) / _I8_MAX
        if value < _SCALE_FLOOR:
            value = _SCALE_FLOOR
        scale[row] = rebind[scale.element_type](value)


def _h3_int8_encode_chunk(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    row_start: Int,
    rows: Int,
    cols: Int,
):
    """Exact `int8_encode_perrow`, writing one compact row window."""
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = rows * cols
    while i < total:
        var row = i // cols
        var value = Float32(rebind[Scalar[DType.bfloat16]](
            x[row_start * cols + i]
        ))
        var q = Int(round(value / rebind[Scalar[DType.float32]](scale[row])))
        if q > 127:
            q = 127
        elif q < -127:
            q = -127
        output[i] = rebind[output.element_type](Scalar[DType.uint8](q & 0xFF))
        i += stride


def _h3_int8_dequant_residual_gate_chunk(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    x_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    w_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    residual: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    gate: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    row_start: Int,
    chunk_rows: Int,
    n: Int,
):
    """Chunk-scale version that updates the residual buffer in place."""
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = chunk_rows * n
    while i < total:
        var row = i // n
        var col = i - row * n
        var output_index = (row_start + row) * n + col
        var projected = (
            Float32(rebind[Scalar[DType.int32]](c[i]))
            * rebind[Scalar[DType.float32]](x_scale[row])
            * rebind[Scalar[DType.float32]](w_scale[col])
        ).cast[DType.bfloat16]().cast[DType.float32]()
        var residual_value = rebind[Scalar[DType.bfloat16]](
            residual[output_index]
        ).cast[DType.float32]()
        var gate_value = rebind[Scalar[DType.bfloat16]](
            gate[output_index]
        ).cast[DType.float32]()
        residual[output_index] = rebind[residual.element_type](
            (residual_value + gate_value * projected).cast[DType.bfloat16]()
        )
        i += stride


def _h3_int8_dequant_qkv_rowscale(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    x_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    w_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    q: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    k: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    row_start: Int,
    chunk_rows: Int,
    inner: Int,
):
    """Dequantize packed `[Q|K|V]` directly into three BF16 outputs."""
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = chunk_rows * inner
    var packed_n = 3 * inner
    while i < total:
        var row = i // inner
        var col = i - row * inner
        var output_index = (row_start + row) * inner + col
        var xs = rebind[Scalar[DType.float32]](x_scale[row_start + row])
        var qv = (
            Float32(rebind[Scalar[DType.int32]](c[row * packed_n + col]))
            * xs * rebind[Scalar[DType.float32]](w_scale[col])
        )
        var kv = (
            Float32(rebind[Scalar[DType.int32]](
                c[row * packed_n + inner + col]
            )) * xs * rebind[Scalar[DType.float32]](w_scale[inner + col])
        )
        var vv = (
            Float32(rebind[Scalar[DType.int32]](
                c[row * packed_n + 2 * inner + col]
            )) * xs * rebind[Scalar[DType.float32]](w_scale[2 * inner + col])
        )
        q[output_index] = rebind[q.element_type](qv.cast[DType.bfloat16]())
        k[output_index] = rebind[k.element_type](kv.cast[DType.bfloat16]())
        v[output_index] = rebind[v.element_type](vv.cast[DType.bfloat16]())
        i += stride


def _h3_int8_dequant_swiglu_chunk(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    x_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    w_scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    chunk_rows: Int,
    f: Int,
):
    """Local-row FC1 dequant + exact two-boundary SwiGLU."""
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = chunk_rows * f
    var packed_n = 2 * f
    while i < total:
        var row = i // f
        var col = i - row * f
        var value_acc = Float32(
            rebind[Scalar[DType.int32]](c[row * packed_n + col])
        )
        var gate_acc = Float32(
            rebind[Scalar[DType.int32]](c[row * packed_n + f + col])
        )
        var xs = rebind[Scalar[DType.float32]](x_scale[row])
        var value = (value_acc * xs * rebind[Scalar[DType.float32]](
            w_scale[col]
        )).cast[DType.bfloat16]().cast[DType.float32]()
        var gate_value = (gate_acc * xs * rebind[Scalar[DType.float32]](
            w_scale[f + col]
        )).cast[DType.bfloat16]().cast[DType.float32]()
        var silu_gate = gate_value / (1.0 + exp(-gate_value))
        silu_gate = silu_gate.cast[DType.bfloat16]().cast[DType.float32]()
        output[row * f + col] = rebind[output.element_type](
            (silu_gate * value).cast[DType.bfloat16]()
        )
        i += stride


struct MiniMaxH3Int8QKV(Movable):
    var q: Tensor
    var k: Tensor
    var v: Tensor

    def __init__(out self, var q: Tensor, var k: Tensor, var v: Tensor):
        self.q = q^
        self.k = k^
        self.v = v^

    def __del__(deinit self):
        pass


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


def minimax_h3_int8_qkv_linear(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    ctx: DeviceContext,
) raises -> MiniMaxH3Int8QKV:
    """Direct W8A8 packed QKV projection without a packed BF16 output.

    The three outputs preserve `minimax_h3_int8_linear`'s I32 accumulation,
    F32 rescale, and BF16 rounding exactly. Avoiding the additional packed
    `[M,3*inner]` tensor removes the dominant 15-second attention peak.
    """
    if x.dtype() != STDtype.BF16 or weight.dtype() != STDtype.I8 \
            or weight_scale.dtype() != STDtype.F32:
        raise Error("MiniMax-H3 INT8 QKV requires BF16/I8/F32 tensors")
    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1 or len(wshape) != 2 or wshape[0] % 3 != 0:
        raise Error("MiniMax-H3 INT8 QKV shape mismatch")
    var contraction = xshape[len(xshape) - 1]
    var packed_n = wshape[0]
    var inner = packed_n // 3
    if wshape[1] != contraction or weight_scale.numel() != packed_n:
        raise Error("MiniMax-H3 INT8 QKV contraction mismatch")
    var rows = x.numel() // contraction

    var x_scale = int8_rowscale(x, ctx)
    var x_int8 = int8_encode_perrow(x, x_scale, ctx)
    var accumulator_rows = min(rows, _MAX_GEMM_ROWS)
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](
        accumulator_rows * packed_n * 4
    )
    var c_shape: List[Int] = [accumulator_rows, packed_n]
    var c = Tensor(c_buf^, c_shape^, STDtype.I32)
    var q_buf = ctx.enqueue_create_buffer[DType.uint8](rows * inner * 2)
    var k_buf = ctx.enqueue_create_buffer[DType.uint8](rows * inner * 2)
    var v_buf = ctx.enqueue_create_buffer[DType.uint8](rows * inner * 2)
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        c.buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[_DYN1].row_major(
            IndexList[1](accumulator_rows * packed_n)
        ),
    )
    var XS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x_scale.buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](rows)),
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight_scale.buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](packed_n)),
    )
    var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        q_buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](rows * inner)),
    )
    var K = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        k_buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](rows * inner)),
    )
    var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        v_buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](rows * inner)),
    )
    for row_start in range(0, rows, _MAX_GEMM_ROWS):
        var chunk_rows = min(_MAX_GEMM_ROWS, rows - row_start)
        _h3_gemm_s8s8s32_nt(
            x_int8, weight, c, chunk_rows, packed_n, contraction,
            row_start, ctx,
        )
        var total = chunk_rows * inner
        var grid = (total + _BLOCK - 1) // _BLOCK
        if grid > 65535:
            grid = 65535
        ctx.enqueue_function[
            _h3_int8_dequant_qkv_rowscale,
            _h3_int8_dequant_qkv_rowscale,
        ](
            C, XS, WS, Q, K, V, row_start, chunk_rows, inner,
            grid_dim=grid, block_dim=_BLOCK,
        )

    var output_shape = List[Int]()
    for i in range(len(xshape) - 1):
        output_shape.append(xshape[i])
    output_shape.append(inner)
    return MiniMaxH3Int8QKV(
        Tensor(q_buf^, output_shape.copy(), STDtype.BF16),
        Tensor(k_buf^, output_shape.copy(), STDtype.BF16),
        Tensor(v_buf^, output_shape^, STDtype.BF16),
    )


def minimax_h3_groupwise_qkv_linear(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    ctx: DeviceContext,
) raises -> MiniMaxH3Int8QKV:
    """Groupwise INT8 QKV with three bounded BF16 projection outputs.

    Dequantization is identical to the accepted groupwise projection, but
    sub-buffer views of the packed weight feed three independent row-chunked
    GEMMs.  This avoids the otherwise additional `[M,3*inner]` BF16 result.
    """
    if x.dtype() != STDtype.BF16 or weight.dtype() != STDtype.I8 \
            or weight_scale.dtype() != STDtype.F16:
        raise Error("MiniMax-H3 groupwise QKV requires BF16/I8/F16 tensors")
    var xshape = x.shape()
    var wshape = weight.shape()
    var sshape = weight_scale.shape()
    if len(xshape) < 1 or len(wshape) != 2 or wshape[0] % 3 != 0 \
            or len(sshape) != 2 or sshape[0] != wshape[0] \
            or sshape[1] <= 0 or wshape[1] % sshape[1] != 0:
        raise Error("MiniMax-H3 groupwise QKV shape mismatch")
    var contraction = xshape[len(xshape) - 1]
    if wshape[1] != contraction:
        raise Error("MiniMax-H3 groupwise QKV contraction mismatch")
    var group_size = contraction // sshape[1]
    var dequant = int8_dequant_groupwise_to_bf16(
        weight, weight_scale, group_size, ctx
    )
    return minimax_h3_bf16_qkv_linear(x, dequant, ctx)


def minimax_h3_bf16_qkv_linear(
    x: Tensor,
    weight: Tensor,
    ctx: DeviceContext,
) raises -> MiniMaxH3Int8QKV:
    """Split a packed BF16 QKV projection without a packed output tensor."""
    if x.dtype() != STDtype.BF16 or weight.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 BF16 QKV requires BF16 tensors")
    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1 or len(wshape) != 2 or wshape[0] % 3 != 0:
        raise Error("MiniMax-H3 BF16 QKV shape mismatch")
    var contraction = xshape[len(xshape) - 1]
    if wshape[1] != contraction:
        raise Error("MiniMax-H3 BF16 QKV contraction mismatch")
    var inner = wshape[0] // 3
    var section_bytes = inner * contraction * 2
    var q_weight_buf = weight.buf.create_sub_buffer[DType.uint8](
        0, section_bytes
    )
    var k_weight_buf = weight.buf.create_sub_buffer[DType.uint8](
        section_bytes, section_bytes
    )
    var v_weight_buf = weight.buf.create_sub_buffer[DType.uint8](
        2 * section_bytes, section_bytes
    )
    var section_shape: List[Int] = [inner, contraction]
    var q_weight = Tensor(q_weight_buf^, section_shape.copy(), STDtype.BF16)
    var k_weight = Tensor(k_weight_buf^, section_shape.copy(), STDtype.BF16)
    var v_weight = Tensor(v_weight_buf^, section_shape^, STDtype.BF16)
    var q = minimax_h3_bf16_linear_chunked(x, q_weight, ctx)
    var k = minimax_h3_bf16_linear_chunked(x, k_weight, ctx)
    var v = minimax_h3_bf16_linear_chunked(x, v_weight, ctx)
    ctx.synchronize()
    return MiniMaxH3Int8QKV(q^, k^, v^)


def minimax_h3_int8_swiglu_linear(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Direct W8A8 FC1 plus SwiGLU without a full packed FC1 output.

    H3 stores FC1 as `[value|gate]`. Long video sequences cannot afford both
    the complete `[M, 2*ffn]` BF16 projection and its `[M, ffn]` activation.
    Reuse the same bounded I32 accumulator as `minimax_h3_int8_linear`, but
    dequantize and activate each completed row chunk directly into the final
    BF16 `[M, ffn]` buffer.
    """
    if x.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 INT8 SwiGLU linear requires BF16 activations")
    if weight.dtype() != STDtype.I8:
        raise Error("MiniMax-H3 INT8 SwiGLU linear requires I8 weights")
    if weight_scale.dtype() != STDtype.F32:
        raise Error("MiniMax-H3 INT8 SwiGLU linear requires F32 row scales")
    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1 or len(wshape) != 2:
        raise Error("MiniMax-H3 INT8 SwiGLU linear expects x[...,K], weight[N,K]")
    var k = xshape[len(xshape) - 1]
    var n = wshape[0]
    if wshape[1] != k or n % 2 != 0:
        raise Error("MiniMax-H3 INT8 SwiGLU linear contraction/FC1 mismatch")
    if weight_scale.numel() != n:
        raise Error("MiniMax-H3 INT8 SwiGLU linear scale length mismatch")
    var m = x.numel() // k
    var f = n // 2

    var x_scale = int8_rowscale(x, ctx)
    var x_int8 = int8_encode_perrow(x, x_scale, ctx)
    var accumulator_rows = min(m, _MAX_GEMM_ROWS)
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](
        accumulator_rows * n * 4
    )
    var c_shape: List[Int] = [accumulator_rows, n]
    var c = Tensor(c_buf^, c_shape^, STDtype.I32)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](m * f * 2)
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        c.buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](accumulator_rows * n)),
    )
    var XS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x_scale.buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](m)),
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight_scale.buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](n)),
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](m * f)),
    )
    for row_start in range(0, m, _MAX_GEMM_ROWS):
        var chunk_rows = min(_MAX_GEMM_ROWS, m - row_start)
        _h3_gemm_s8s8s32_nt(
            x_int8, weight, c, chunk_rows, n, k, row_start, ctx
        )
        var total = chunk_rows * f
        var grid = (total + _BLOCK - 1) // _BLOCK
        if grid > 65535:
            grid = 65535
        ctx.enqueue_function[
            _h3_int8_dequant_swiglu_rowscale,
            _h3_int8_dequant_swiglu_rowscale,
        ](
            C, XS, WS, O, row_start, chunk_rows, f,
            grid_dim=grid, block_dim=_BLOCK,
        )

    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        out_shape.append(xshape[i])
    out_shape.append(f)
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


def minimax_h3_int8_residual_linear(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    residual: Tensor,
    gate: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Direct W8A8 FC2 fused with the following residual gate.

    Long H3 sequences cannot hold both complete `[M, hidden]` FC2 output and
    the equal-sized block result. This writes the block result directly while
    retaining the original projection-BF16 and residual-BF16 boundaries.
    """
    if x.dtype() != STDtype.BF16 or residual.dtype() != STDtype.BF16 \
            or gate.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 INT8 residual linear requires BF16 tensors")
    if weight.dtype() != STDtype.I8 or weight_scale.dtype() != STDtype.F32:
        raise Error("MiniMax-H3 INT8 residual linear requires I8/F32 weights")
    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1 or len(wshape) != 2:
        raise Error("MiniMax-H3 INT8 residual linear expects x[...,K], weight[N,K]")
    var k = xshape[len(xshape) - 1]
    var n = wshape[0]
    if wshape[1] != k or weight_scale.numel() != n:
        raise Error("MiniMax-H3 INT8 residual linear contraction mismatch")
    var m = x.numel() // k
    if residual.numel() != m * n or gate.numel() != m * n:
        raise Error("MiniMax-H3 INT8 residual linear residual/gate mismatch")

    var x_scale = int8_rowscale(x, ctx)
    var x_int8 = int8_encode_perrow(x, x_scale, ctx)
    var accumulator_rows = min(m, _MAX_GEMM_ROWS)
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](
        accumulator_rows * n * 4
    )
    var c_shape: List[Int] = [accumulator_rows, n]
    var c = Tensor(c_buf^, c_shape^, STDtype.I32)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](m * n * 2)
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        c.buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](accumulator_rows * n)),
    )
    var XS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x_scale.buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](m)),
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight_scale.buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](n)),
    )
    var R = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        residual.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](m * n)),
    )
    var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        gate.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](m * n)),
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
            _h3_int8_dequant_residual_gate_rowscale,
            _h3_int8_dequant_residual_gate_rowscale,
        ](
            C, XS, WS, R, G, O, row_start, chunk_rows, n,
            grid_dim=grid, block_dim=_BLOCK,
        )

    var out_shape = residual.shape()
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


def minimax_h3_int8_residual_linear_inplace(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    var residual: Tensor,
    gate: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """In-place variant for the 24-GiB long-sequence block path.

    Each fused output reads only `residual[row,col]` and writes that same
    element, so the residual buffer can become the block result without a
    separate `[M,N]` allocation. Projection and residual rounding are exactly
    the same as `minimax_h3_int8_residual_linear`.
    """
    if x.dtype() != STDtype.BF16 or residual.dtype() != STDtype.BF16 \
            or gate.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 in-place INT8 residual linear requires BF16 tensors")
    if weight.dtype() != STDtype.I8 or weight_scale.dtype() != STDtype.F32:
        raise Error("MiniMax-H3 in-place INT8 residual linear requires I8/F32 weights")
    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1 or len(wshape) != 2:
        raise Error("MiniMax-H3 in-place INT8 residual linear shape mismatch")
    var k = xshape[len(xshape) - 1]
    var n = wshape[0]
    if wshape[1] != k or weight_scale.numel() != n:
        raise Error("MiniMax-H3 in-place INT8 residual linear contraction mismatch")
    var m = x.numel() // k
    if residual.numel() != m * n or gate.numel() != m * n:
        raise Error("MiniMax-H3 in-place INT8 residual/gate mismatch")

    var accumulator_rows = min(m, _MAX_GEMM_ROWS)
    var x_scale_buf = ctx.enqueue_create_buffer[DType.uint8](
        accumulator_rows * 4
    )
    var x_int8_buf = ctx.enqueue_create_buffer[DType.uint8](
        accumulator_rows * k
    )
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](
        accumulator_rows * n * 4
    )
    var x_int8_shape: List[Int] = [accumulator_rows, k]
    var x_int8 = Tensor(x_int8_buf^, x_int8_shape^, STDtype.I8)
    var c_shape: List[Int] = [accumulator_rows, n]
    var c = Tensor(c_buf^, c_shape^, STDtype.I32)
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](m * k)),
    )
    var XS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x_scale_buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](accumulator_rows)),
    )
    var XI = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        x_int8.buf.unsafe_ptr(),
        RuntimeLayout[_DYN1].row_major(IndexList[1](accumulator_rows * k)),
    )
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        c.buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](accumulator_rows * n)),
    )
    var WS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight_scale.buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](n)),
    )
    var R = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        residual.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](m * n)),
    )
    var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        gate.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](m * n)),
    )
    for row_start in range(0, m, _MAX_GEMM_ROWS):
        var chunk_rows = min(_MAX_GEMM_ROWS, m - row_start)
        ctx.enqueue_function[
            _h3_int8_rowscale_chunk, _h3_int8_rowscale_chunk
        ](
            X, XS, row_start, k, chunk_rows,
            grid_dim=chunk_rows, block_dim=_BLOCK,
        )
        var quant_total = chunk_rows * k
        var quant_grid = (quant_total + _BLOCK - 1) // _BLOCK
        if quant_grid > 65535:
            quant_grid = 65535
        ctx.enqueue_function[
            _h3_int8_encode_chunk, _h3_int8_encode_chunk
        ](
            X, XS, XI, row_start, chunk_rows, k,
            grid_dim=quant_grid, block_dim=_BLOCK,
        )
        _h3_gemm_s8s8s32_nt(
            x_int8, weight, c, chunk_rows, n, k, 0, ctx
        )
        var total = chunk_rows * n
        var grid = (total + _BLOCK - 1) // _BLOCK
        if grid > 65535:
            grid = 65535
        ctx.enqueue_function[
            _h3_int8_dequant_residual_gate_chunk,
            _h3_int8_dequant_residual_gate_chunk,
        ](
            C, XS, WS, R, G, row_start, chunk_rows, n,
            grid_dim=grid, block_dim=_BLOCK,
        )
        # Reuse the same compact activation/accumulator buffers and prevent
        # cuBLAS workspaces from accumulating across the 23 long-row chunks.
        ctx.synchronize()
    return residual^


def minimax_h3_int8_mlp_residual_inplace(
    mlp_in: Tensor,
    fc1_weight: Tensor,
    fc1_scale: Tensor,
    fc2_weight: Tensor,
    fc2_scale: Tensor,
    var residual: Tensor,
    gate: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Chunked W8A8 FC1 → SwiGLU → FC2 → residual for long H3 rows.

    All row-wise quantization, I32 GEMMs, BF16 projection boundaries, SwiGLU
    boundary, and residual F32 expression match the unfused operators. Only
    storage lifetime changes: no full `[M,ffn]` activation is materialized.
    """
    if mlp_in.dtype() != STDtype.BF16 or residual.dtype() != STDtype.BF16 \
            or gate.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 fused MLP requires BF16 activations")
    if fc1_weight.dtype() != STDtype.I8 or fc2_weight.dtype() != STDtype.I8 \
            or fc1_scale.dtype() != STDtype.F32 \
            or fc2_scale.dtype() != STDtype.F32:
        raise Error("MiniMax-H3 fused MLP requires W8A8 row-scale weights")
    var in_shape = mlp_in.shape()
    var fc1_shape = fc1_weight.shape()
    var fc2_shape = fc2_weight.shape()
    if len(in_shape) < 1 or len(fc1_shape) != 2 or len(fc2_shape) != 2 \
            or fc1_shape[0] % 2 != 0:
        raise Error("MiniMax-H3 fused MLP shape mismatch")
    var hidden = in_shape[len(in_shape) - 1]
    var packed_ffn = fc1_shape[0]
    var ffn = packed_ffn // 2
    var out_hidden = fc2_shape[0]
    if fc1_shape[1] != hidden or fc2_shape[1] != ffn \
            or fc1_scale.numel() != packed_ffn \
            or fc2_scale.numel() != out_hidden:
        raise Error("MiniMax-H3 fused MLP contraction mismatch")
    var rows = mlp_in.numel() // hidden
    if residual.numel() != rows * out_hidden \
            or gate.numel() != rows * out_hidden:
        raise Error("MiniMax-H3 fused MLP residual/gate mismatch")

    var chunk_capacity = min(rows, _MLP_GEMM_ROWS)
    var in_scale_buf = ctx.enqueue_create_buffer[DType.uint8](chunk_capacity * 4)
    var in_i8_buf = ctx.enqueue_create_buffer[DType.uint8](
        chunk_capacity * hidden
    )
    var fc1_acc_buf = ctx.enqueue_create_buffer[DType.uint8](
        chunk_capacity * packed_ffn * 4
    )
    var act_buf = ctx.enqueue_create_buffer[DType.uint8](
        chunk_capacity * ffn * 2
    )
    var act_scale_buf = ctx.enqueue_create_buffer[DType.uint8](chunk_capacity * 4)
    var act_i8_buf = ctx.enqueue_create_buffer[DType.uint8](
        chunk_capacity * ffn
    )
    var fc2_acc_buf = ctx.enqueue_create_buffer[DType.uint8](
        chunk_capacity * out_hidden * 4
    )
    var in_i8_shape: List[Int] = [chunk_capacity, hidden]
    var fc1_acc_shape: List[Int] = [chunk_capacity, packed_ffn]
    var act_i8_shape: List[Int] = [chunk_capacity, ffn]
    var fc2_acc_shape: List[Int] = [chunk_capacity, out_hidden]
    var in_i8 = Tensor(in_i8_buf^, in_i8_shape^, STDtype.I8)
    var fc1_acc = Tensor(fc1_acc_buf^, fc1_acc_shape^, STDtype.I32)
    var act_i8 = Tensor(act_i8_buf^, act_i8_shape^, STDtype.I8)
    var fc2_acc = Tensor(fc2_acc_buf^, fc2_acc_shape^, STDtype.I32)
    var In = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        mlp_in.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](rows * hidden)),
    )
    var InScale = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        in_scale_buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](chunk_capacity)),
    )
    var InI8 = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        in_i8.buf.unsafe_ptr(),
        RuntimeLayout[_DYN1].row_major(IndexList[1](chunk_capacity * hidden)),
    )
    var Fc1Acc = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        fc1_acc.buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[_DYN1].row_major(
            IndexList[1](chunk_capacity * packed_ffn)
        ),
    )
    var Fc1Scale = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        fc1_scale.buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](packed_ffn)),
    )
    var Act = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        act_buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](chunk_capacity * ffn)),
    )
    var ActScale = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        act_scale_buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](chunk_capacity)),
    )
    var ActI8 = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        act_i8.buf.unsafe_ptr(),
        RuntimeLayout[_DYN1].row_major(IndexList[1](chunk_capacity * ffn)),
    )
    var Fc2Acc = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        fc2_acc.buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[_DYN1].row_major(
            IndexList[1](chunk_capacity * out_hidden)
        ),
    )
    var Fc2Scale = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        fc2_scale.buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](out_hidden)),
    )
    var Residual = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        residual.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](rows * out_hidden)),
    )
    var Gate = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        gate.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](rows * out_hidden)),
    )

    for row_start in range(0, rows, _MLP_GEMM_ROWS):
        var chunk_rows = min(_MLP_GEMM_ROWS, rows - row_start)
        ctx.enqueue_function[
            _h3_int8_rowscale_chunk, _h3_int8_rowscale_chunk
        ](
            In, InScale, row_start, hidden, chunk_rows,
            grid_dim=chunk_rows, block_dim=_BLOCK,
        )
        var in_total = chunk_rows * hidden
        var in_grid = (in_total + _BLOCK - 1) // _BLOCK
        if in_grid > 65535:
            in_grid = 65535
        ctx.enqueue_function[
            _h3_int8_encode_chunk, _h3_int8_encode_chunk
        ](
            In, InScale, InI8, row_start, chunk_rows, hidden,
            grid_dim=in_grid, block_dim=_BLOCK,
        )
        _h3_gemm_s8s8s32_nt(
            in_i8, fc1_weight, fc1_acc, chunk_rows, packed_ffn, hidden,
            0, ctx,
        )
        var act_total = chunk_rows * ffn
        var act_grid = (act_total + _BLOCK - 1) // _BLOCK
        if act_grid > 65535:
            act_grid = 65535
        ctx.enqueue_function[
            _h3_int8_dequant_swiglu_chunk,
            _h3_int8_dequant_swiglu_chunk,
        ](
            Fc1Acc, InScale, Fc1Scale, Act, chunk_rows, ffn,
            grid_dim=act_grid, block_dim=_BLOCK,
        )
        ctx.enqueue_function[
            _h3_int8_rowscale_chunk, _h3_int8_rowscale_chunk
        ](
            Act, ActScale, 0, ffn, chunk_rows,
            grid_dim=chunk_rows, block_dim=_BLOCK,
        )
        var act_i8_grid = (act_total + _BLOCK - 1) // _BLOCK
        if act_i8_grid > 65535:
            act_i8_grid = 65535
        ctx.enqueue_function[
            _h3_int8_encode_chunk, _h3_int8_encode_chunk
        ](
            Act, ActScale, ActI8, 0, chunk_rows, ffn,
            grid_dim=act_i8_grid, block_dim=_BLOCK,
        )
        _h3_gemm_s8s8s32_nt(
            act_i8, fc2_weight, fc2_acc, chunk_rows, out_hidden, ffn,
            0, ctx,
        )
        var out_total = chunk_rows * out_hidden
        var out_grid = (out_total + _BLOCK - 1) // _BLOCK
        if out_grid > 65535:
            out_grid = 65535
        ctx.enqueue_function[
            _h3_int8_dequant_residual_gate_chunk,
            _h3_int8_dequant_residual_gate_chunk,
        ](
            Fc2Acc, ActScale, Fc2Scale, Residual, Gate,
            row_start, chunk_rows, out_hidden,
            grid_dim=out_grid, block_dim=_BLOCK,
        )
        ctx.synchronize()
    return residual^


def _h3_bf16_acc_swiglu_chunk(
    accumulator: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    output: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    chunk_rows: Int,
    ffn: Int,
):
    """Cast the BF16 projection boundary, then exact packed SwiGLU."""
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = chunk_rows * ffn
    var packed_ffn = 2 * ffn
    while i < total:
        var row = i // ffn
        var col = i - row * ffn
        var value = rebind[Scalar[DType.float32]](
            accumulator[row * packed_ffn + col]
        ).cast[DType.bfloat16]().cast[DType.float32]()
        var gate_value = rebind[Scalar[DType.float32]](
            accumulator[row * packed_ffn + ffn + col]
        ).cast[DType.bfloat16]().cast[DType.float32]()
        var silu_gate = gate_value / (1.0 + exp(-gate_value))
        silu_gate = silu_gate.cast[DType.bfloat16]().cast[DType.float32]()
        output[row * ffn + col] = rebind[output.element_type](
            (silu_gate * value).cast[DType.bfloat16]()
        )
        i += stride


def _h3_bf16_acc_residual_gate_chunk(
    accumulator: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    residual: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    gate: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    row_start: Int,
    chunk_rows: Int,
    hidden: Int,
):
    """Cast FC2 to BF16 before the exact F32 residual expression."""
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = chunk_rows * hidden
    while i < total:
        var row = i // hidden
        var col = i - row * hidden
        var output_index = (row_start + row) * hidden + col
        var projected = rebind[Scalar[DType.float32]](
            accumulator[i]
        ).cast[DType.bfloat16]().cast[DType.float32]()
        var residual_value = rebind[Scalar[DType.bfloat16]](
            residual[output_index]
        ).cast[DType.float32]()
        var gate_value = rebind[Scalar[DType.bfloat16]](
            gate[output_index]
        ).cast[DType.float32]()
        residual[output_index] = rebind[residual.element_type](
            (residual_value + gate_value * projected).cast[DType.bfloat16]()
        )
        i += stride


def minimax_h3_groupwise_mlp_residual_inplace(
    mlp_in: Tensor,
    fc1_weight: Tensor,
    fc1_scale: Tensor,
    fc2_weight: Tensor,
    fc2_scale: Tensor,
    var residual: Tensor,
    gate: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Dequantize groupwise weights, then use the bounded BF16 MLP."""
    if mlp_in.dtype() != STDtype.BF16 or residual.dtype() != STDtype.BF16 \
            or gate.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 groupwise fused MLP requires BF16 activations")
    if fc1_weight.dtype() != STDtype.I8 or fc2_weight.dtype() != STDtype.I8 \
            or fc1_scale.dtype() != STDtype.F16 \
            or fc2_scale.dtype() != STDtype.F16:
        raise Error("MiniMax-H3 groupwise fused MLP requires I8/F16 weights")
    var in_shape = mlp_in.shape()
    var fc1_shape = fc1_weight.shape()
    var fc2_shape = fc2_weight.shape()
    var fc1_scale_shape = fc1_scale.shape()
    var fc2_scale_shape = fc2_scale.shape()
    if len(in_shape) < 1 or len(fc1_shape) != 2 or len(fc2_shape) != 2 \
            or len(fc1_scale_shape) != 2 or len(fc2_scale_shape) != 2 \
            or fc1_shape[0] % 2 != 0:
        raise Error("MiniMax-H3 groupwise fused MLP shape mismatch")
    var hidden = in_shape[len(in_shape) - 1]
    var packed_ffn = fc1_shape[0]
    var ffn = packed_ffn // 2
    var out_hidden = fc2_shape[0]
    if fc1_shape[1] != hidden or fc2_shape[1] != ffn \
            or fc1_scale_shape[0] != packed_ffn \
            or fc2_scale_shape[0] != out_hidden \
            or fc1_scale_shape[1] <= 0 or fc2_scale_shape[1] <= 0 \
            or hidden % fc1_scale_shape[1] != 0 \
            or ffn % fc2_scale_shape[1] != 0:
        raise Error("MiniMax-H3 groupwise fused MLP contraction mismatch")
    var rows = mlp_in.numel() // hidden
    if residual.numel() != rows * out_hidden \
            or gate.numel() != rows * out_hidden:
        raise Error("MiniMax-H3 groupwise fused MLP residual mismatch")
    var fc1_bf16 = int8_dequant_groupwise_to_bf16(
        fc1_weight, fc1_scale, hidden // fc1_scale_shape[1], ctx
    )
    var fc2_bf16 = int8_dequant_groupwise_to_bf16(
        fc2_weight, fc2_scale, ffn // fc2_scale_shape[1], ctx
    )
    return minimax_h3_bf16_mlp_residual_inplace(
        mlp_in, fc1_bf16, fc2_bf16, residual^, gate, ctx
    )


def minimax_h3_bf16_mlp_residual_inplace(
    mlp_in: Tensor,
    fc1_weight: Tensor,
    fc2_weight: Tensor,
    var residual: Tensor,
    gate: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Bounded BF16 FC1 → SwiGLU → FC2 → residual.

    This retains the original F32 GEMM accumulation and BF16 projection,
    activation, and residual boundaries while ensuring neither packed FC1
    nor its activation spans the complete long video sequence.
    """
    if mlp_in.dtype() != STDtype.BF16 or fc1_weight.dtype() != STDtype.BF16 \
            or fc2_weight.dtype() != STDtype.BF16 \
            or residual.dtype() != STDtype.BF16 or gate.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 BF16 fused MLP requires BF16 tensors")
    var in_shape = mlp_in.shape()
    var fc1_shape = fc1_weight.shape()
    var fc2_shape = fc2_weight.shape()
    if len(in_shape) < 1 or len(fc1_shape) != 2 or len(fc2_shape) != 2 \
            or fc1_shape[0] % 2 != 0:
        raise Error("MiniMax-H3 BF16 fused MLP shape mismatch")
    var hidden = in_shape[len(in_shape) - 1]
    var packed_ffn = fc1_shape[0]
    var ffn = packed_ffn // 2
    var out_hidden = fc2_shape[0]
    if fc1_shape[1] != hidden or fc2_shape[1] != ffn:
        raise Error("MiniMax-H3 BF16 fused MLP contraction mismatch")
    var rows = mlp_in.numel() // hidden
    if residual.numel() != rows * out_hidden \
            or gate.numel() != rows * out_hidden:
        raise Error("MiniMax-H3 BF16 fused MLP residual mismatch")

    var chunk_capacity = min(rows, _GROUPWISE_MLP_ROWS)
    var fc1_acc_buf = ctx.enqueue_create_buffer[DType.uint8](
        chunk_capacity * packed_ffn * 4
    )
    var act_buf = ctx.enqueue_create_buffer[DType.uint8](
        chunk_capacity * ffn * 2
    )
    var fc2_acc_buf = ctx.enqueue_create_buffer[DType.uint8](
        chunk_capacity * out_hidden * 4
    )
    var Fc1Weight = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        fc1_weight.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN2].row_major(IndexList[2](packed_ffn, hidden)),
    )
    var Fc2Weight = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        fc2_weight.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN2].row_major(IndexList[2](out_hidden, ffn)),
    )
    var Fc1AccFlat = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        fc1_acc_buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(
            IndexList[1](chunk_capacity * packed_ffn)
        ),
    )
    var ActFlat = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        act_buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](chunk_capacity * ffn)),
    )
    var Fc2AccFlat = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        fc2_acc_buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(
            IndexList[1](chunk_capacity * out_hidden)
        ),
    )
    var Residual = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        residual.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](rows * out_hidden)),
    )
    var Gate = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        gate.buf.unsafe_ptr().bitcast[BFloat16](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](rows * out_hidden)),
    )
    for row_start in range(0, rows, _GROUPWISE_MLP_ROWS):
        var chunk_rows = min(_GROUPWISE_MLP_ROWS, rows - row_start)
        var Input = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            mlp_in.buf.unsafe_ptr().bitcast[BFloat16]() + row_start * hidden,
            RuntimeLayout[_DYN2].row_major(IndexList[2](chunk_rows, hidden)),
        )
        var Fc1Acc = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            fc1_acc_buf.unsafe_ptr().bitcast[Float32](),
            RuntimeLayout[_DYN2].row_major(
                IndexList[2](chunk_rows, packed_ffn)
            ),
        )
        matmul(
            ctx, Fc1Acc, Input, Fc1Weight,
            transpose_b=True, c_row_major=True,
        )
        var act_total = chunk_rows * ffn
        var act_grid = (act_total + _BLOCK - 1) // _BLOCK
        if act_grid > 65535:
            act_grid = 65535
        ctx.enqueue_function[
            _h3_bf16_acc_swiglu_chunk,
            _h3_bf16_acc_swiglu_chunk,
        ](
            Fc1AccFlat, ActFlat, chunk_rows, ffn,
            grid_dim=act_grid, block_dim=_BLOCK,
        )
        var Act = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            act_buf.unsafe_ptr().bitcast[BFloat16](),
            RuntimeLayout[_DYN2].row_major(IndexList[2](chunk_rows, ffn)),
        )
        var Fc2Acc = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            fc2_acc_buf.unsafe_ptr().bitcast[Float32](),
            RuntimeLayout[_DYN2].row_major(
                IndexList[2](chunk_rows, out_hidden)
            ),
        )
        matmul(
            ctx, Fc2Acc, Act, Fc2Weight,
            transpose_b=True, c_row_major=True,
        )
        var out_total = chunk_rows * out_hidden
        var out_grid = (out_total + _BLOCK - 1) // _BLOCK
        if out_grid > 65535:
            out_grid = 65535
        ctx.enqueue_function[
            _h3_bf16_acc_residual_gate_chunk,
            _h3_bf16_acc_residual_gate_chunk,
        ](
            Fc2AccFlat, Residual, Gate, row_start, chunk_rows, out_hidden,
            grid_dim=out_grid, block_dim=_BLOCK,
        )
        ctx.synchronize()
    return residual^
