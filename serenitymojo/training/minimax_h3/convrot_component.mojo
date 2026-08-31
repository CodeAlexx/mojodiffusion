# MiniMax-H3 ConvRot numeric components pinned to Musubi b8717864713c9e4e.
#
# Bounded surface only:
#   * regular H4-Kronecker H256 BF16 activation rotation/involution,
#   * ConvRot BF16 activation row quantization with its two BF16 boundaries,
#   * transient BF16-scale INT8 weight dequantization used by BF16 backward.
#
# This is not an INT8 projection, DiT-block, checkpoint loader, or trainer.  In
# particular, it deliberately does not reuse ops/int8_quant.mojo: that generic
# helper clamps to [-127,127] and divides by an F32 scale, while ConvRot's pinned
# Triton BF16 path clamps to [-128,127], casts scale to BF16 before division, and
# casts the quotient to BF16 before RNE.

from max.gpu.host import DeviceContext
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from std.gpu import block_idx, global_idx, grid_dim, thread_idx
from std.math import round
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import reshape, reshape_owned
from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value
from serenitymojo.tensor import Tensor


comptime MINIMAX_H3_CONVROT_GROUP_SIZE = 256
comptime _H4_FACTORS = 4
comptime _BLOCK = 256
comptime _DYN1 = Layout.row_major(-1)
# Triton constant-folds `max_val / 127.0` to multiplication by this F32
# reciprocal (bits 0x3c010204). For some BF16 maxima, a true F32 division is
# one ULP different, so preserve the pinned kernel's compiled arithmetic.
comptime _I8_SCALE_MULTIPLIER = Float32(0.007874015718698502)
comptime _SCALE_FLOOR = Float32(1.0e-30)


@fieldwise_init
struct MiniMaxH3ConvRotActivationI8(Movable):
    """ConvRot activation codes and F32 row scale [rows,1]."""

    var values: Tensor
    var scale: Tensor


def _h4_sign(row: Int, col: Int) -> Float32:
    # Musubi's regular H4 has -1 on the anti-diagonal and +1 elsewhere.
    return Float32(-1.0) if row + col == 3 else Float32(1.0)


def minimax_h3_convrot_regular_h256(ctx: DeviceContext) raises -> Tensor:
    """Build normalized regular H4⊗H4⊗H4⊗H4 as resident BF16 [256,256].

    Every entry is exactly representable ±1/16.  This is the regular ConvRot
    ordering, not the commonly substituted Sylvester/FWHT H256 ordering.
    """
    var values = List[Float32](capacity=(
        MINIMAX_H3_CONVROT_GROUP_SIZE * MINIMAX_H3_CONVROT_GROUP_SIZE
    ))
    for row in range(MINIMAX_H3_CONVROT_GROUP_SIZE):
        for col in range(MINIMAX_H3_CONVROT_GROUP_SIZE):
            var r = row
            var c = col
            var sign = Float32(1.0)
            for _ in range(_H4_FACTORS):
                sign *= _h4_sign(r % 4, c % 4)
                r //= 4
                c //= 4
            values.append(sign * Float32(0.0625))
    return Tensor.from_host(
        values,
        [MINIMAX_H3_CONVROT_GROUP_SIZE, MINIMAX_H3_CONVROT_GROUP_SIZE],
        STDtype.BF16,
        ctx,
    )


def minimax_h3_convrot_rotate_h256(
    x: Tensor, h256: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Apply block-diagonal regular H256 to the last BF16 dimension.

    Mirrors pinned Musubi `_rotate_activation`: reshape all leading dimensions
    and H256 groups to [-1,256], then BF16 x @ H.  Calling this function twice
    applies the same symmetric orthogonal matrix and is the inverse operation.
    """
    if x.dtype() != STDtype.BF16:
        raise Error("minimax_h3_convrot_rotate_h256: x must be BF16")
    if h256.dtype() != STDtype.BF16:
        raise Error("minimax_h3_convrot_rotate_h256: H256 must be BF16")
    var h_shape = h256.shape()
    if len(h_shape) != 2 or h_shape[0] != MINIMAX_H3_CONVROT_GROUP_SIZE \
            or h_shape[1] != MINIMAX_H3_CONVROT_GROUP_SIZE:
        raise Error("minimax_h3_convrot_rotate_h256: H256 must be [256,256]")
    var original_shape = x.shape()
    if len(original_shape) < 1:
        raise Error("minimax_h3_convrot_rotate_h256: x must have rank >= 1")
    var features = original_shape[len(original_shape) - 1]
    if x.numel() <= 0 or features <= 0 \
            or features % MINIMAX_H3_CONVROT_GROUP_SIZE != 0:
        raise Error(
            "minimax_h3_convrot_rotate_h256: nonempty last dimension must be divisible by 256"
        )
    var grouped_rows = x.numel() // MINIMAX_H3_CONVROT_GROUP_SIZE
    var grouped = reshape(
        x, [grouped_rows, MINIMAX_H3_CONVROT_GROUP_SIZE], ctx
    )
    # Explicit x @ H (transpose_b=False). H is symmetric, but the flag records
    # the Musubi boundary rather than relying on symmetry accidentally.
    var rotated = linear(grouped, h256, None, ctx, transpose_b=False)
    return reshape_owned(rotated^, original_shape^)


def _convrot_rowscale_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows_w: Int32,
    cols_w: Int32,
):
    var row = Int(block_idx.x)
    var rows = Int(rows_w)
    if row >= rows:
        return
    var cols = Int(cols_w)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local_max = Float32(0.0)
    var col = tid
    var base = row * cols
    while col < cols:
        var value = rebind[Scalar[DType.bfloat16]](x[base + col]).cast[
            DType.float32
        ]()
        var magnitude = value if value >= Float32(0.0) else -value
        if magnitude > local_max:
            local_max = magnitude
        col += _BLOCK
    shared[tid] = local_max
    barrier()
    var active = _BLOCK // 2
    while active > 0:
        if tid < active and shared[tid + active] > shared[tid]:
            shared[tid] = shared[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        var row_scale = shared[0] * _I8_SCALE_MULTIPLIER
        if row_scale < _SCALE_FLOOR:
            row_scale = _SCALE_FLOOR
        scale[row] = rebind[scale.element_type](row_scale)


def _convrot_encode_bf16_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    cols_w: Int32,
    n_w: Int64,
):
    var cols = Int(cols_w)
    var n = Int(n_w)
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * _BLOCK)
    while i < n:
        var value = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        var scale_f32 = rebind[Scalar[DType.float32]](scale[i // cols])
        # Exact pinned Triton BF16 branch:
        #   q_f = (x / scale.to(tl.bfloat16)).to(tl.bfloat16)
        var denominator = torch_bf16_rne_value(scale_f32).cast[DType.float32]()
        var quotient = torch_bf16_rne_value(value / denominator).cast[
            DType.float32
        ]()
        var code = Int(round(quotient))
        if code > 127:
            code = 127
        elif code < -128:
            code = -128
        dst[i] = rebind[dst.element_type](Scalar[DType.uint8](code & 255))
        i += stride


def minimax_h3_convrot_quantize_activation_bf16(
    x: Tensor, ctx: DeviceContext
) raises -> MiniMaxH3ConvRotActivationI8:
    """Quantize BF16 [rows,cols] exactly as pinned ConvRot Triton row quant."""
    if x.dtype() != STDtype.BF16:
        raise Error("minimax_h3_convrot_quantize_activation_bf16: x must be BF16")
    var shape = x.shape()
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise Error(
            "minimax_h3_convrot_quantize_activation_bf16: x must be nonempty rank-2"
        )
    var rows = shape[0]
    var cols = shape[1]
    var n = rows * cols
    var scale_buf = ctx.enqueue_create_buffer[DType.uint8](rows * 4)
    var values_buf = ctx.enqueue_create_buffer[DType.uint8](n)
    var x_layout = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var scale_layout = RuntimeLayout[_DYN1].row_major(IndexList[1](rows))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_layout,
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=scale_layout,
    )
    var Q = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(values_buf.unsafe_ptr())
        ),
        runtime_layout=x_layout,
    )
    ctx.enqueue_function[_convrot_rowscale_kernel](
        X, S, Int32(rows), Int32(cols), grid_dim=rows, block_dim=_BLOCK
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_convrot_encode_bf16_kernel](
        X, S, Q, Int32(cols), Int64(n), grid_dim=grid, block_dim=_BLOCK
    )
    return MiniMaxH3ConvRotActivationI8(
        Tensor(values_buf^, shape^, STDtype.I8),
        Tensor(scale_buf^, [rows, 1], STDtype.F32),
    )


def _convrot_weight_dequant_bf16_kernel(
    q: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    cols_w: Int32,
    n_w: Int64,
):
    var cols = Int(cols_w)
    var n = Int(n_w)
    var i = Int(global_idx.x)
    if i < n:
        var code = Float32(Int(rebind[Scalar[DType.int8]](q[i])))
        # Pinned Musubi BF16 backward:
        #   wq.to(grad_out.dtype) * w_scale.to(grad_out.dtype)
        var scale_bf16 = torch_bf16_rne_value(
            rebind[Scalar[DType.float32]](scale[i // cols])
        ).cast[DType.float32]()
        dst[i] = rebind[dst.element_type](
            torch_bf16_rne_value(code * scale_bf16)
        )


def minimax_h3_convrot_dequant_weight_bf16(
    q: Tensor, scale: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Transient BF16 W_rot for Musubi's BF16 ConvRot backward branch.

    `q` stays in the rotated basis.  This function does not apply H256; pinned
    backward performs `grad @ W_rot` first and rotates that dX intermediate.
    """
    if q.dtype() != STDtype.I8:
        raise Error("minimax_h3_convrot_dequant_weight_bf16: q must be I8")
    if scale.dtype() != STDtype.F32:
        raise Error("minimax_h3_convrot_dequant_weight_bf16: scale must be F32")
    var q_shape = q.shape()
    var scale_shape = scale.shape()
    if len(q_shape) != 2 or q_shape[0] <= 0 or q_shape[1] <= 0:
        raise Error("minimax_h3_convrot_dequant_weight_bf16: q must be nonempty [rows,cols]")
    if len(scale_shape) != 2 or scale_shape[0] != q_shape[0] \
            or scale_shape[1] != 1:
        raise Error("minimax_h3_convrot_dequant_weight_bf16: scale must be F32 [rows,1]")
    var rows = q_shape[0]
    var cols = q_shape[1]
    var n = rows * cols
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    var values_layout = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var scale_layout = RuntimeLayout[_DYN1].row_major(IndexList[1](rows))
    var Q = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(q.buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=values_layout,
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=scale_layout,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=values_layout,
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_convrot_weight_dequant_bf16_kernel](
        Q, S, O, Int32(cols), Int64(n), grid_dim=grid, block_dim=_BLOCK
    )
    return Tensor(out_buf^, q_shape^, STDtype.BF16)
