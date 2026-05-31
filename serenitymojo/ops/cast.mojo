# cast.mojo - GPU dtype cast helper.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _bf16_to_f32(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](
            rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        )


def _f32_to_bf16(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](
            rebind[Scalar[DType.float32]](x[i]).cast[DType.bfloat16]()
        )


def _f16_to_f32(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](
            rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        )


def _f32_to_f16(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](
            rebind[Scalar[DType.float32]](x[i]).cast[DType.float16]()
        )


def cast_tensor(x: Tensor, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    """Materialized GPU dtype cast. Supports F32<->BF16/F16 plus no-op clone."""
    if x.dtype() == dtype:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), dtype)

    var src = x.dtype().to_mojo_dtype()
    var dst = dtype.to_mojo_dtype()
    var n = x.numel()
    # Guard against zero-element tensors (e.g. N_TXT=0 rope tables).
    # Allocate a zero-byte buffer and return the empty tensor with the target dtype.
    if n == 0:
        var empty_buf = ctx.enqueue_create_buffer[DType.uint8](0)
        ctx.synchronize()
        return Tensor(empty_buf^, x.shape(), dtype)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK

    if src == DType.bfloat16 and dst == DType.float32:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_bf16_to_f32, _bf16_to_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif src == DType.float32 and dst == DType.bfloat16:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_f32_to_bf16, _f32_to_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif src == DType.float16 and dst == DType.float32:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_f16_to_f32, _f16_to_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif src == DType.float32 and dst == DType.float16:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_f32_to_f16, _f32_to_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif src == DType.bfloat16 and dst == DType.float16:
        # Route through F32 (no direct BF16->F16 kernel needed)
        var mid_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
        var mid_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            mid_buf.unsafe_ptr().bitcast[Float32](), mid_rl
        )
        ctx.enqueue_function[_bf16_to_f32, _bf16_to_f32](
            X, M, n, grid_dim=grid, block_dim=_BLOCK
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_f32_to_f16, _f32_to_f16](
            M, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif src == DType.float16 and dst == DType.bfloat16:
        # Route through F32
        var mid_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
        var mid_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            mid_buf.unsafe_ptr().bitcast[Float32](), mid_rl
        )
        ctx.enqueue_function[_f16_to_f32, _f16_to_f32](
            X, M, n, grid_dim=grid, block_dim=_BLOCK
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_f32_to_bf16, _f32_to_bf16](
            M, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        raise Error("cast_tensor: unsupported dtype pair")
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), dtype)


def cast_tensor_if_needed(var x: Tensor, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    """Move `x` through unchanged when it already has `dtype`; otherwise cast."""
    if x.dtype() == dtype:
        return x^
    return cast_tensor(x^, dtype, ctx)
