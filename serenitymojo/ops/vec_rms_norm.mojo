# ops/vec_rms_norm.mojo — VECTORIZED RMSNorm forward + backward (F32 fast path).
#
# NEW STANDALONE kernel. Does NOT replace ops/norm.mojo / ops/norm_backward.mojo
# — it is a faster sibling whose math is byte-identical to the scalar ones, and
# whose parity is gated AGAINST those scalar kernels (vec_rms_norm_parity.mojo).
#
# The speedup vs the scalar one-thread-per-element-strided loop comes from
# loading/storing the feature dim with width-4 SIMD vectors (float32x4): each
# thread processes 4 contiguous columns per memory transaction instead of 1,
# which is the same trick flame-core's vectorized BF16 kernels use (`__hadd2`,
# `float2`, `__nv_bfloat162` — 2/4 elements per thread, FLAME_KERNELS.md).
#
# Requirement for the vector path: the normalized feature dim D must be a
# multiple of 4. BF16/F16 and non-multiple-D tensors fall back to the general
# dtype-preserving norm kernels instead of materializing F32 storage here.
#
# Math (matches ops/norm.mojo rms_norm + ops/norm_backward.mojo rms_norm_backward
# EXACTLY):
#   inv = 1/sqrt(mean_j(x^2) + eps)
#   y[c]   = x[c]*inv*g[c]
#   d_x[c] = g[c]*go[c]*inv - x[c]*inv^3*(1/D)*sum_j(go[j]*g[j]*x[j])
#   d_g[c] = sum_rows( go[c] * x[c]*inv )
# F32 interior. The vectorized fast path is F32-only, but BF16/F16 storage
# routes to the general dtype-preserving norm/backward kernels.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, global_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.norm import rms_norm as _general_rms_norm
from serenitymojo.ops.norm_backward import rms_norm_backward as _general_rms_norm_backward


comptime _DYN1 = Layout.row_major(-1)
comptime _TPB = 256       # threads per block (one block per row)
comptime _BLOCK = 256
comptime _VW = 4          # SIMD vector width (float32x4)


# ── RMSNorm FORWARD (vec4): one block per row, each thread strides by 4 cols ──
# x/g/o are flat F32 pointers viewed as [rows*cols]; the kernel indexes
# row*cols + c manually so we can issue width-4 SIMD loads/stores.
def _vec_rms_fwd_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = row * cols
    var xp = x.ptr + base
    var op = o.ptr + base
    var sh = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    # sum(x^2) over the row, 4 elements per thread.
    var local: Float32 = 0.0
    var c = tid * _VW
    while c < cols:
        var v = xp.load[width=_VW](c)
        local += (v * v).reduce_add()
        c += _TPB * _VW
    sh[tid] = local
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(sh[0] / Float32(cols) + eps)
    # write y = x*inv*g (vec4)
    c = tid * _VW
    while c < cols:
        var v = xp.load[width=_VW](c)
        var gv = g.ptr.load[width=_VW](c)
        op.store[width=_VW](c, v * inv * gv)
        c += _TPB * _VW


def vec_rms_norm(
    x: Tensor, weight: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Vectorized RMSNorm over the last dim for F32-friendly shapes."""
    if x.dtype() != STDtype.F32 or weight.dtype() != STDtype.F32:
        return _general_rms_norm(x, weight, eps, ctx)
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    if d % _VW != 0:
        return _general_rms_norm(x, weight, eps, ctx)
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * d))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[Float32](), g_rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    ctx.enqueue_function[_vec_rms_fwd_kernel, _vec_rms_fwd_kernel](
        X, G, O, d, eps, grid_dim=rows, block_dim=_TPB
    )
    return Tensor(out_buf^, xshape.copy(), STDtype.F32)


# ── RMSNorm BACKWARD d_x (vec4): one block per row ────────────────────────────
def _vec_rms_bwd_dx_kernel(
    go: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dx: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = row * cols
    var xp = x.ptr + base
    var gop = go.ptr + base
    var dxp = dx.ptr + base
    var sh = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    # sum(x^2)
    var lsq: Float32 = 0.0
    var c = tid * _VW
    while c < cols:
        var v = xp.load[width=_VW](c)
        lsq += (v * v).reduce_add()
        c += _TPB * _VW
    sh[tid] = lsq
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(sh[0] / Float32(cols) + eps)
    barrier()
    # sum(go*g*x)
    var lgwx: Float32 = 0.0
    c = tid * _VW
    while c < cols:
        var gov = gop.load[width=_VW](c)
        var gv = g.ptr.load[width=_VW](c)
        var xv = xp.load[width=_VW](c)
        lgwx += (gov * gv * xv).reduce_add()
        c += _TPB * _VW
    sh[tid] = lgwx
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var sum_gwx = sh[0]
    barrier()
    var inv3 = inv * inv * inv
    var scl = inv3 * (sum_gwx / Float32(cols))
    c = tid * _VW
    while c < cols:
        var xv = xp.load[width=_VW](c)
        var gv = g.ptr.load[width=_VW](c)
        var gov = gop.load[width=_VW](c)
        dxp.store[width=_VW](c, gv * gov * inv - xv * scl)
        c += _TPB * _VW


# ── RMSNorm BACKWARD d_g: one thread per column (vectorized inv recompute) ────
# d_g[c] = sum_rows( go[r,c]*x[r,c]*inv_rms[r] ). Same as the scalar one but the
# per-row inv-rms recompute uses width-4 SIMD loads.
def _vec_rms_bwd_dg_kernel(
    go: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dg: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    cols: Int,
    eps: Float32,
):
    var col = Int(global_idx.x)
    if col >= cols:
        return
    var acc: Float32 = 0.0
    for r in range(rows):
        var base = r * cols
        var xp = x.ptr + base
        var sq: Float32 = 0.0
        var cc = 0
        while cc < cols:
            var v = xp.load[width=_VW](cc)
            sq += (v * v).reduce_add()
            cc += _VW
        var inv = 1.0 / sqrt(sq / Float32(cols) + eps)
        var gov = rebind[Scalar[DType.float32]](go[base + col])
        var xv = rebind[Scalar[DType.float32]](x[base + col])
        acc += gov * xv * inv
    dg[col] = rebind[dg.element_type](acc)


struct VecRmsNormBackward(Movable):
    var d_x: Tensor
    var d_g: Tensor

    def __init__(out self, var d_x: Tensor, var d_g: Tensor):
        self.d_x = d_x^
        self.d_g = d_g^


def vec_rms_norm_backward(
    go: Tensor, x: Tensor, weight: Tensor, eps: Float32, ctx: DeviceContext
) raises -> VecRmsNormBackward:
    """Vectorized backward of rms_norm for F32-friendly shapes."""
    if x.dtype() != STDtype.F32 or go.dtype() != STDtype.F32 or weight.dtype() != STDtype.F32:
        var general = _general_rms_norm_backward(go, x, weight, eps, ctx)
        var dx = general.d_x.clone(ctx)
        var dg = general.d_g.clone(ctx)
        return VecRmsNormBackward(dx^, dg^)
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    if d % _VW != 0:
        var general = _general_rms_norm_backward(go, x, weight, eps, ctx)
        var dx = general.d_x.clone(ctx)
        var dg = general.d_g.clone(ctx)
        return VecRmsNormBackward(dx^, dg^)
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var dg_buf = ctx.enqueue_create_buffer[DType.uint8](weight.nbytes())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * d))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))

    var GO = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        go.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[Float32](), g_rl
    )
    var DX = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dx_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var DG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dg_buf.unsafe_ptr().bitcast[Float32](), g_rl
    )

    ctx.enqueue_function[_vec_rms_bwd_dx_kernel, _vec_rms_bwd_dx_kernel](
        GO, X, G, DX, d, eps, grid_dim=rows, block_dim=_TPB
    )
    var dg_grid = (d + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_vec_rms_bwd_dg_kernel, _vec_rms_bwd_dg_kernel](
        GO, X, DG, rows, d, eps, grid_dim=dg_grid, block_dim=_BLOCK
    )
    return VecRmsNormBackward(
        Tensor(dx_buf^, xshape.copy(), STDtype.F32),
        Tensor(dg_buf^, weight.shape().copy(), STDtype.F32),
    )
