# ops/vec_modulate.mojo — VECTORIZED fused adaLN modulate, F32 fast path.
#
# NEW STANDALONE kernel. Does NOT replace ops/elementwise.mojo `modulate`; it is
# a faster sibling. Parity gated against the scalar modulate (vec_modulate_parity).
#
# Math (matches ops/elementwise.mojo modulate EXACTLY):
#   o[r,c] = (1 + scale[c]) * x[r,c] + shift[c]      scale/shift per-channel [D]
# The scalar kernel uses one thread per element and recomputes r=idx//cols,
# c=idx%cols per element (a div + mod per element) to index the per-channel
# scale/shift. This kernel uses one thread per vec4 chunk: it loads x as a
# width-4 SIMD vector and the matching scale/shift slice as width-4 vectors
# (scale/shift are contiguous in [D], and a chunk of 4 columns maps to a
# contiguous [c..c+3] slice because D % 4 == 0), then does one fused vector FMA.
# This is the vec2-fused-modulate idea from flame-core's modulate_pre_bf16_kernel
# (FLAME_KERNELS.md bf16_ops.rs:580), at vec4 for F32 storage.
#
# Requirement: D % 4 == 0 (the channel dim is always a multiple of 4 here). Else
# RAISE — caller falls back to the scalar modulate (AGENT-DEFAULT: raise).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.elementwise import modulate as _scalar_modulate
from serenitymojo.ops.elementwise import modulate_slab as _scalar_modulate_slab
from serenitymojo.autograd_v2.step_slab import StepSlab


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _VW = 4


# One thread per vec4 chunk over the flat [rows*cols] buffer. Within a chunk the
# 4 elements are columns [c0..c0+3] of one row (guaranteed by cols % 4 == 0), so
# the scale/shift slice is the contiguous [c0..c0+3] window of the [D] vectors.
def _vec_modulate_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    sh: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cols: Int,
    nchunks: Int,
):
    var chunk = Int(global_idx.x)
    if chunk >= nchunks:
        return
    var base = chunk * _VW
    var c0 = base % cols          # column of the first lane (multiple of 4)
    var xv = x.ptr.load[width=_VW](base)
    var sv = s.ptr.load[width=_VW](c0)
    var shv = sh.ptr.load[width=_VW](c0)
    o.ptr.store[width=_VW](base, (SIMD[DType.float32, _VW](1.0) + sv) * xv + shv)


def vec_modulate(
    x: Tensor, scale: Tensor, shift: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Vectorized modulate(x, scale, shift) = (1+scale)*x + shift for F32.

    BF16/F16 use the dtype-preserving scalar implementation instead of
    materializing F32 fast-path storage."""
    if x.dtype() != STDtype.F32 or scale.dtype() != STDtype.F32 or shift.dtype() != STDtype.F32:
        return _scalar_modulate(x, scale, shift, ctx)
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    var sshape = scale.shape()
    var shshape = shift.shape()
    if len(sshape) != 1 or sshape[0] != d:
        raise Error("vec_modulate: scale must be [D]")
    if len(shshape) != 1 or shshape[0] != d:
        raise Error("vec_modulate: shift must be [D]")
    if d % _VW != 0:
        raise Error(
            String("vec_modulate: D must be a multiple of 4 (got ")
            + String(d) + ") — use the scalar modulate"
        )
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    var n = rows * d
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), v_rl
    )
    var SH = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        shift.buf.unsafe_ptr().bitcast[Float32](), v_rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var nchunks = n // _VW
    var grid = (nchunks + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_vec_modulate_kernel, _vec_modulate_kernel](
        X, S, SH, O, d, nchunks, grid_dim=grid, block_dim=_BLOCK
    )
    return Tensor(out_buf^, xshape.copy(), STDtype.F32)


def vec_modulate_slab(
    x: Tensor, scale: Tensor, shift: Tensor, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `vec_modulate` (above) — byte-identical math (same
    kernel, same launch params); ONLY the allocation source changes
    (autograd_v2 contracts C8/C9, Phase P5). Non-F32 inputs route to
    modulate_slab (the dtype-preserving scalar sibling, same dispatch shape
    as the non-slab pair)."""
    if x.dtype() != STDtype.F32 or scale.dtype() != STDtype.F32 or shift.dtype() != STDtype.F32:
        return _scalar_modulate_slab(x, scale, shift, ctx, slab)
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    var sshape = scale.shape()
    var shshape = shift.shape()
    if len(sshape) != 1 or sshape[0] != d:
        raise Error("vec_modulate: scale must be [D]")
    if len(shshape) != 1 or shshape[0] != d:
        raise Error("vec_modulate: shift must be [D]")
    if d % _VW != 0:
        raise Error(
            String("vec_modulate: D must be a multiple of 4 (got ")
            + String(d) + ") — use the scalar modulate"
        )
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    var n = rows * d
    var out_buf = slab.alloc(x.nbytes())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), v_rl
    )
    var SH = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        shift.buf.unsafe_ptr().bitcast[Float32](), v_rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var nchunks = n // _VW
    var grid = (nchunks + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_vec_modulate_kernel, _vec_modulate_kernel](
        X, S, SH, O, d, nchunks, grid_dim=grid, block_dim=_BLOCK
    )
    return Tensor(out_buf^, xshape.copy(), STDtype.F32)
