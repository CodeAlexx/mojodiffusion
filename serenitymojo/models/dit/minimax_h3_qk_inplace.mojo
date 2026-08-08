# MiniMax-H3 low-headroom Q/K preparation.
#
# The released long-video geometry makes each Q or K tensor more than 1 GiB.
# The ordinary RMSNorm + partial-RoPE composition materializes several complete
# copies of each tensor.  These kernels keep the exact F32 reduction/arithmetic
# and BF16 rounding boundaries, but write the normalized and rotated values back
# into the already-owned split-QKV buffers.

from std.gpu import barrier, block_idx, global_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import sqrt
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _TPB = 256
comptime _ROPE_BLOCK = 256


def _h3_rms_norm_bf16_inplace_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    weight: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    """Bit-identical in-place form of ops.norm._rms_norm_kernel_bf16."""
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local: Float32 = 0.0
    var col = tid
    while col < cols:
        var value = rebind[Scalar[DType.bfloat16]](x[row, col]).cast[
            DType.float32
        ]()
        local += value * value
        col += _TPB
    shared[tid] = local
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(shared[0] / Float32(cols) + eps)
    col = tid
    while col < cols:
        var value = rebind[Scalar[DType.bfloat16]](x[row, col]).cast[
            DType.float32
        ]()
        var scale = rebind[Scalar[DType.bfloat16]](weight[col]).cast[
            DType.float32
        ]()
        x[row, col] = rebind[x.element_type](
            (value * inv * scale).cast[DType.bfloat16]()
        )
        col += _TPB


def _h3_partial_rope_bf16_f32_inplace_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    head_dim: Int,
    rotary_dim: Int,
    heads: Int,
):
    """Exact half-split RoPE over the leading rotary channels, in place."""
    var idx = Int(global_idx.x)
    var half = rotary_dim // 2
    var total = rows * half
    if idx < total:
        var row = idx // half
        var lane = idx - row * half
        var token = row // heads
        var value0 = rebind[Scalar[DType.bfloat16]](x[row, lane]).cast[
            DType.float32
        ]()
        var value1 = rebind[Scalar[DType.bfloat16]](
            x[row, lane + half]
        ).cast[DType.float32]()
        var cos0 = rebind[Scalar[DType.float32]](cos[token, lane])
        var sin0 = rebind[Scalar[DType.float32]](sin[token, lane])
        var cos1 = rebind[Scalar[DType.float32]](cos[token, lane + half])
        var sin1 = rebind[Scalar[DType.float32]](sin[token, lane + half])
        x[row, lane] = rebind[x.element_type](
            (value0 * cos0 - value1 * sin0).cast[DType.bfloat16]()
        )
        x[row, lane + half] = rebind[x.element_type](
            (value1 * cos1 + value0 * sin1).cast[DType.bfloat16]()
        )


def minimax_h3_qk_norm_partial_rope_inplace(
    mut x: Tensor,
    weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    heads: Int,
    rotary_dim: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises:
    """Apply H3 Q/K RMSNorm and partial RoPE without full-tensor copies."""
    if x.dtype() != STDtype.BF16 or weight.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 in-place Q/K norm requires BF16 tensors")
    if cos.dtype() != STDtype.F32 or sin.dtype() != STDtype.F32:
        raise Error("MiniMax-H3 in-place Q/K RoPE requires F32 tables")
    if heads <= 0 or rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise Error("MiniMax-H3 in-place Q/K RoPE geometry is invalid")
    var shape = x.shape()
    if len(shape) < 1:
        raise Error("MiniMax-H3 in-place Q/K input must have rank >= 1")
    var head_dim = shape[len(shape) - 1]
    if rotary_dim > head_dim or weight.numel() != head_dim:
        raise Error("MiniMax-H3 in-place Q/K width mismatch")
    var rows = x.numel() // head_dim
    if rows % heads != 0:
        raise Error("MiniMax-H3 in-place Q/K row/head mismatch")
    var tokens = rows // heads
    if cos.numel() != tokens * rotary_dim \
            or sin.numel() != tokens * rotary_dim:
        raise Error("MiniMax-H3 in-place Q/K table mismatch")

    var x_layout = RuntimeLayout[_DYN2].row_major(
        IndexList[2](rows, head_dim)
    )
    var weight_layout = RuntimeLayout[_DYN1].row_major(
        IndexList[1](head_dim)
    )
    var table_layout = RuntimeLayout[_DYN2].row_major(
        IndexList[2](tokens, rotary_dim)
    )
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_layout
    )
    var Weight = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[BFloat16](), weight_layout
    )
    var Cos = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        cos.buf.unsafe_ptr().bitcast[Float32](), table_layout
    )
    var Sin = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        sin.buf.unsafe_ptr().bitcast[Float32](), table_layout
    )
    ctx.enqueue_function[
        _h3_rms_norm_bf16_inplace_kernel,
        _h3_rms_norm_bf16_inplace_kernel,
    ](X, Weight, head_dim, eps, grid_dim=rows, block_dim=_TPB)
    var total_pairs = rows * (rotary_dim // 2)
    var grid = (total_pairs + _ROPE_BLOCK - 1) // _ROPE_BLOCK
    ctx.enqueue_function[
        _h3_partial_rope_bf16_f32_inplace_kernel,
        _h3_partial_rope_bf16_f32_inplace_kernel,
    ](
        X, Cos, Sin, rows, head_dim, rotary_dim, heads,
        grid_dim=grid, block_dim=_ROPE_BLOCK,
    )
