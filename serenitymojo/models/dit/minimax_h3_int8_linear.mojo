from std.ffi import external_call
from std.collections import List
from std.gpu import block_dim, global_idx, grid_dim
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUDA
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.ops.int8_quant import int8_encode_perrow, int8_rowscale
from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _h3_gemm_s8s8s32_nt(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    var a_ptr = BytePtr(unsafe_from_address=Int(a.buf.unsafe_ptr()))
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
    m: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var total = m * n
    while i < total:
        var row = i // n
        var col = i - row * n
        var acc = Float32(rebind[Scalar[DType.int32]](c[i]))
        var value = (
            acc
            * rebind[Scalar[DType.float32]](x_scale[row])
            * rebind[Scalar[DType.float32]](w_scale[col])
        )
        output[i] = rebind[output.element_type](value.cast[DType.bfloat16]())
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
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](m * n * 4)
    var c_shape: List[Int] = [m, n]
    var c = Tensor(c_buf^, c_shape^, STDtype.I32)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](m * n * 2)
    var flat_c = RuntimeLayout[_DYN1].row_major(IndexList[1](m * n))
    _h3_gemm_s8s8s32_nt(x_int8, weight, c, m, n, k, ctx)

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
        out_buf.unsafe_ptr().bitcast[BFloat16](), flat_c
    )
    var total = m * n
    var grid = (total + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[
        _h3_int8_dequant_rowscale, _h3_int8_dequant_rowscale
    ](
        C, XS, WS, O, m, n,
        grid_dim=grid, block_dim=_BLOCK,
    )

    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        out_shape.append(xshape[i])
    out_shape.append(n)
    return Tensor(out_buf^, out_shape^, STDtype.BF16)
