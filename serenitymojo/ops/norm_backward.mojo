# ops/norm_backward.mojo — BACKWARD for rms_norm / layer_norm / group_norm (F32).
#
# Backward partner of ops/norm.mojo. F32-only (the parity gate runs F32; the
# storage-dtype variants can be added later by mirroring the forward's dispatch).
# Same scaffolding as the forward: one block per row (rms/layer) or per (n,group)
# (group), shared-memory F32 tree reductions; the parameter grads (d_g, d_b) use
# a cross-row/spatial reduction kernel (one thread per column/channel) that
# recomputes the per-row/per-group stats — same recompute discipline as
# flame-core's layer_norm/group_norm backward (flame_norm_bf16.cu).
#
# Math (normalize over feature dim D / group; weight g, bias b, eps). LayerNorm
# uses BIASED variance (matches the forward + torch). All interior math F32.
#
#   RMSNorm:   inv = 1/sqrt(mean(x^2)+eps)
#     d_x[c] = g[c]*go[c]*inv - x[c]*inv^3*(1/D)*sum_j(go[j]*g[j]*x[j])
#     d_g[c] = sum_rows( go[c] * x[c]*inv )
#
#   LayerNorm: mu=mean(x), inv=1/sqrt(var+eps), norm[c]=(x[c]-mu)*inv
#     sum_wg  = sum_j(g[j]*go[j]) ;  sum_wgn = sum_j(g[j]*go[j]*norm[j])
#     d_x[c] = inv*g[c]*go[c] - (inv/D)*sum_wg - (norm[c]*inv/D)*sum_wgn
#     d_g[c] = sum_rows( go[c]*norm[c] ) ;  d_b[c] = sum_rows( go[c] )
#
#   GroupNorm (NHWC): per (n,group) over count = (C/G)*H*W; w = g[channel].
#     same d_x form as LayerNorm reduced over the group;
#     d_g[c] = sum_{n,hw}( go*norm ) ;  d_b[c] = sum_{n,hw}( go )   (per channel)
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
from serenitymojo.ops.cast import cast_tensor


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _DYN1 = Layout.row_major(-1)
comptime _TPB = 256  # threads per block (one block per row / group)
comptime _BLOCK = 256


# ════════════════════════════════════════════════════════════════════════════
# RMSNorm backward
# ════════════════════════════════════════════════════════════════════════════

# d_x: one block per row [rows, D]. Reduce sum(x^2) then sum(go*g*x), then write.
def _rms_bwd_dx_kernel(
    go: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dx: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    # sum(x^2)
    var lsq: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        lsq += v * v
        c += _TPB
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
    c = tid
    while c < cols:
        var gov = rebind[Scalar[DType.float32]](go[row, c])
        var gv = rebind[Scalar[DType.float32]](g[c])
        var xv = rebind[Scalar[DType.float32]](x[row, c])
        lgwx += gov * gv * xv
        c += _TPB
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
    c = tid
    while c < cols:
        var xv = rebind[Scalar[DType.float32]](x[row, c])
        var gv = rebind[Scalar[DType.float32]](g[c])
        var gov = rebind[Scalar[DType.float32]](go[row, c])
        var out = gv * gov * inv - xv * inv3 * (sum_gwx / Float32(cols))
        dx[row, c] = rebind[dx.element_type](out)
        c += _TPB


# d_g: one thread per column. d_g[c] = sum_rows( go[r,c]*x[r,c]*inv_rms[r] ).
def _rms_bwd_dg_kernel(
    go: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
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
        var sq: Float32 = 0.0
        for cc in range(cols):
            var v = rebind[Scalar[DType.float32]](x[r, cc])
            sq += v * v
        var inv = 1.0 / sqrt(sq / Float32(cols) + eps)
        var gov = rebind[Scalar[DType.float32]](go[r, col])
        var xv = rebind[Scalar[DType.float32]](x[r, col])
        acc += gov * xv * inv
    dg[col] = rebind[dg.element_type](acc)


struct RmsNormBackward(Movable):
    """Backward outputs of rms_norm: d_x [rows..,D] and d_g [D]."""

    var d_x: Tensor
    var d_g: Tensor

    def __init__(out self, var d_x: Tensor, var d_g: Tensor):
        self.d_x = d_x^
        self.d_g = d_g^


def rms_norm_backward(
    go: Tensor, x: Tensor, weight: Tensor, eps: Float32, ctx: DeviceContext
) raises -> RmsNormBackward:
    """Backward of rms_norm. go/x same shape [..,D] (F32); weight [D] (F32).
    Returns d_x (x's shape) and d_g [D]."""
    # BF16/F16 storage path: cast inputs up to F32, run the F32 interior below
    # verbatim, then cast grads back down to the storage dtype. The F32 path is
    # byte-identical (the branch is only taken when an input is not F32).
    if x.dtype() != STDtype.F32 or go.dtype() != STDtype.F32 or weight.dtype() != STDtype.F32:
        var out_dt = x.dtype()
        var go32 = cast_tensor(go, STDtype.F32, ctx)
        var x32 = cast_tensor(x, STDtype.F32, ctx)
        var w32 = cast_tensor(weight, STDtype.F32, ctx)
        var g32 = rms_norm_backward(go32, x32, w32, eps, ctx)
        var dx_dn = cast_tensor(g32.d_x^, out_dt, ctx)
        var dg_dn = cast_tensor(g32.d_g^, out_dt, ctx)
        return RmsNormBackward(dx_dn^, dg_dn^)
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var dg_buf = ctx.enqueue_create_buffer[DType.uint8](weight.nbytes())

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))

    var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        go.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[Float32](), g_rl
    )
    var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dx_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var DG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dg_buf.unsafe_ptr().bitcast[Float32](), g_rl
    )

    ctx.enqueue_function[_rms_bwd_dx_kernel, _rms_bwd_dx_kernel](
        GO, X, G, DX, d, eps, grid_dim=rows, block_dim=_TPB
    )
    var dg_grid = (d + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_rms_bwd_dg_kernel, _rms_bwd_dg_kernel](
        GO, X, DG, rows, d, eps, grid_dim=dg_grid, block_dim=_BLOCK
    )
    ctx.synchronize()

    return RmsNormBackward(
        Tensor(dx_buf^, xshape.copy(), STDtype.F32),
        Tensor(dg_buf^, weight.shape().copy(), STDtype.F32),
    )


# ════════════════════════════════════════════════════════════════════════════
# LayerNorm backward
# ════════════════════════════════════════════════════════════════════════════

# d_x: one block per row. Reduce mean, var, sum_wg, sum_wgn; then write d_x.
def _ln_bwd_dx_kernel(
    go: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dx: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    # mean
    var ls: Float32 = 0.0
    var c = tid
    while c < cols:
        ls += rebind[Scalar[DType.float32]](x[row, c])
        c += _TPB
    sh[tid] = ls
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var mean = sh[0] / Float32(cols)
    barrier()

    # var (biased)
    var lv: Float32 = 0.0
    c = tid
    while c < cols:
        var dd = rebind[Scalar[DType.float32]](x[row, c]) - mean
        lv += dd * dd
        c += _TPB
    sh[tid] = lv
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(sh[0] / Float32(cols) + eps)
    barrier()

    # sum_wg = sum(g*go)
    var lwg: Float32 = 0.0
    c = tid
    while c < cols:
        lwg += rebind[Scalar[DType.float32]](g[c]) * rebind[Scalar[DType.float32]](go[row, c])
        c += _TPB
    sh[tid] = lwg
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var sum_wg = sh[0]
    barrier()

    # sum_wgn = sum(g*go*norm)
    var lwgn: Float32 = 0.0
    c = tid
    while c < cols:
        var norm = (rebind[Scalar[DType.float32]](x[row, c]) - mean) * inv
        lwgn += rebind[Scalar[DType.float32]](g[c]) * rebind[Scalar[DType.float32]](go[row, c]) * norm
        c += _TPB
    sh[tid] = lwgn
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var sum_wgn = sh[0]
    barrier()

    c = tid
    while c < cols:
        var norm = (rebind[Scalar[DType.float32]](x[row, c]) - mean) * inv
        var wv = rebind[Scalar[DType.float32]](g[c])
        var gov = rebind[Scalar[DType.float32]](go[row, c])
        var out = inv * wv * gov - (inv / Float32(cols)) * sum_wg - (norm * inv / Float32(cols)) * sum_wgn
        dx[row, c] = rebind[dx.element_type](out)
        c += _TPB


# d_g, d_b: one thread per column, recompute per-row mean/inv_std.
def _ln_bwd_param_kernel(
    go: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dg: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    db: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    cols: Int,
    eps: Float32,
):
    var col = Int(global_idx.x)
    if col >= cols:
        return
    var acc_g: Float32 = 0.0
    var acc_b: Float32 = 0.0
    for r in range(rows):
        var s: Float32 = 0.0
        for cc in range(cols):
            s += rebind[Scalar[DType.float32]](x[r, cc])
        var mean = s / Float32(cols)
        var vs: Float32 = 0.0
        for cc in range(cols):
            var dd = rebind[Scalar[DType.float32]](x[r, cc]) - mean
            vs += dd * dd
        var inv = 1.0 / sqrt(vs / Float32(cols) + eps)
        var norm = (rebind[Scalar[DType.float32]](x[r, col]) - mean) * inv
        var gov = rebind[Scalar[DType.float32]](go[r, col])
        acc_g += gov * norm
        acc_b += gov
    dg[col] = rebind[dg.element_type](acc_g)
    db[col] = rebind[db.element_type](acc_b)


struct LayerNormBackward(Movable):
    """Backward outputs of layer_norm: d_x [..,D], d_g [D], d_b [D]."""

    var d_x: Tensor
    var d_g: Tensor
    var d_b: Tensor

    def __init__(out self, var d_x: Tensor, var d_g: Tensor, var d_b: Tensor):
        self.d_x = d_x^
        self.d_g = d_g^
        self.d_b = d_b^


def layer_norm_backward(
    go: Tensor, x: Tensor, weight: Tensor, eps: Float32, ctx: DeviceContext
) raises -> LayerNormBackward:
    """Backward of layer_norm. go/x [..,D] (F32); weight [D] (F32).
    Returns d_x (x's shape), d_g [D], d_b [D]."""
    # BF16/F16 storage path: cast up, run F32 interior, cast grads down.
    # F32 path byte-identical (branch only on non-F32 input).
    if x.dtype() != STDtype.F32 or go.dtype() != STDtype.F32 or weight.dtype() != STDtype.F32:
        var out_dt = x.dtype()
        var go32 = cast_tensor(go, STDtype.F32, ctx)
        var x32 = cast_tensor(x, STDtype.F32, ctx)
        var w32 = cast_tensor(weight, STDtype.F32, ctx)
        var g32 = layer_norm_backward(go32, x32, w32, eps, ctx)
        var dx_dn = cast_tensor(g32.d_x^, out_dt, ctx)
        var dg_dn = cast_tensor(g32.d_g^, out_dt, ctx)
        var db_dn = cast_tensor(g32.d_b^, out_dt, ctx)
        return LayerNormBackward(dx_dn^, dg_dn^, db_dn^)
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var dg_buf = ctx.enqueue_create_buffer[DType.uint8](weight.nbytes())
    var db_buf = ctx.enqueue_create_buffer[DType.uint8](weight.nbytes())

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))

    var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        go.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[Float32](), g_rl
    )
    var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dx_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var DG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dg_buf.unsafe_ptr().bitcast[Float32](), g_rl
    )
    var DB = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        db_buf.unsafe_ptr().bitcast[Float32](), g_rl
    )

    ctx.enqueue_function[_ln_bwd_dx_kernel, _ln_bwd_dx_kernel](
        GO, X, G, DX, d, eps, grid_dim=rows, block_dim=_TPB
    )
    var dg_grid = (d + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_ln_bwd_param_kernel, _ln_bwd_param_kernel](
        GO, X, DG, DB, rows, d, eps, grid_dim=dg_grid, block_dim=_BLOCK
    )
    ctx.synchronize()

    return LayerNormBackward(
        Tensor(dx_buf^, xshape.copy(), STDtype.F32),
        Tensor(dg_buf^, weight.shape().copy(), STDtype.F32),
        Tensor(db_buf^, weight.shape().copy(), STDtype.F32),
    )


# ════════════════════════════════════════════════════════════════════════════
# GroupNorm backward (NHWC, matches forward layout)
# ════════════════════════════════════════════════════════════════════════════
#
# Forward index (norm.mojo): off(n,pix,ch) = (n*hw + pix)*C + ch, where the
# group's channels are [gi*cpg, (gi+1)*cpg). count = cpg*hw per (n,group).

# d_x: one block per (n, group) = grid n*G. Reduce mean, var, sum_wg, sum_wgn
# over the group's cpg*hw elements; then write d_x.
def _gn_bwd_dx_kernel(
    go: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dx: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n_dim: Int,
    hw: Int,
    c_dim: Int,
    num_groups: Int,
    eps: Float32,
):
    var blk = Int(block_idx.x)  # n*num_groups + gi
    var n = blk // num_groups
    var gi = blk % num_groups
    var cpg = c_dim // num_groups
    var c0 = gi * cpg
    var count = hw * cpg
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    # mean
    var ls: Float32 = 0.0
    var i = tid
    while i < count:
        var pix = i // cpg
        var cc = i % cpg
        var off = (n * hw + pix) * c_dim + (c0 + cc)
        ls += rebind[Scalar[DType.float32]](x[off])
        i += _TPB
    sh[tid] = ls
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var mean = sh[0] / Float32(count)
    barrier()

    # var (biased)
    var lv: Float32 = 0.0
    i = tid
    while i < count:
        var pix = i // cpg
        var cc = i % cpg
        var off = (n * hw + pix) * c_dim + (c0 + cc)
        var dd = rebind[Scalar[DType.float32]](x[off]) - mean
        lv += dd * dd
        i += _TPB
    sh[tid] = lv
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(sh[0] / Float32(count) + eps)
    barrier()

    # sum_wg = sum(w*go) over group
    var lwg: Float32 = 0.0
    i = tid
    while i < count:
        var pix = i // cpg
        var cc = i % cpg
        var ch = c0 + cc
        var off = (n * hw + pix) * c_dim + ch
        lwg += rebind[Scalar[DType.float32]](g[ch]) * rebind[Scalar[DType.float32]](go[off])
        i += _TPB
    sh[tid] = lwg
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var sum_wg = sh[0]
    barrier()

    # sum_wgn = sum(w*go*norm) over group
    var lwgn: Float32 = 0.0
    i = tid
    while i < count:
        var pix = i // cpg
        var cc = i % cpg
        var ch = c0 + cc
        var off = (n * hw + pix) * c_dim + ch
        var norm = (rebind[Scalar[DType.float32]](x[off]) - mean) * inv
        lwgn += rebind[Scalar[DType.float32]](g[ch]) * rebind[Scalar[DType.float32]](go[off]) * norm
        i += _TPB
    sh[tid] = lwgn
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    var sum_wgn = sh[0]
    barrier()

    i = tid
    while i < count:
        var pix = i // cpg
        var cc = i % cpg
        var ch = c0 + cc
        var off = (n * hw + pix) * c_dim + ch
        var norm = (rebind[Scalar[DType.float32]](x[off]) - mean) * inv
        var wv = rebind[Scalar[DType.float32]](g[ch])
        var gov = rebind[Scalar[DType.float32]](go[off])
        var out = inv * wv * gov - (inv / Float32(count)) * sum_wg - (norm * inv / Float32(count)) * sum_wgn
        dx[off] = rebind[dx.element_type](out)
        i += _TPB


# d_g, d_b: one thread per channel; recompute (n,group) stats, accumulate over
# all n and spatial positions for that channel.
def _gn_bwd_param_kernel(
    go: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dg: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    db: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n_dim: Int,
    hw: Int,
    c_dim: Int,
    num_groups: Int,
    eps: Float32,
):
    var ch = Int(global_idx.x)
    if ch >= c_dim:
        return
    var cpg = c_dim // num_groups
    var gi = ch // cpg
    var c0 = gi * cpg
    var count = hw * cpg

    var acc_g: Float32 = 0.0
    var acc_b: Float32 = 0.0
    for n in range(n_dim):
        # group mean/inv_std for (n, gi)
        var s: Float32 = 0.0
        for ii in range(count):
            var pix = ii // cpg
            var cc = ii % cpg
            var off = (n * hw + pix) * c_dim + (c0 + cc)
            s += rebind[Scalar[DType.float32]](x[off])
        var mean = s / Float32(count)
        var vs: Float32 = 0.0
        for ii in range(count):
            var pix = ii // cpg
            var cc = ii % cpg
            var off = (n * hw + pix) * c_dim + (c0 + cc)
            var dd = rebind[Scalar[DType.float32]](x[off]) - mean
            vs += dd * dd
        var inv = 1.0 / sqrt(vs / Float32(count) + eps)
        # accumulate over spatial for this channel
        for pix in range(hw):
            var off = (n * hw + pix) * c_dim + ch
            var norm = (rebind[Scalar[DType.float32]](x[off]) - mean) * inv
            var gov = rebind[Scalar[DType.float32]](go[off])
            acc_g += gov * norm
            acc_b += gov
    dg[ch] = rebind[dg.element_type](acc_g)
    db[ch] = rebind[db.element_type](acc_b)


struct GroupNormBackward(Movable):
    """Backward outputs of group_norm: d_x [N,H,W,C], d_g [C], d_b [C]."""

    var d_x: Tensor
    var d_g: Tensor
    var d_b: Tensor

    def __init__(out self, var d_x: Tensor, var d_g: Tensor, var d_b: Tensor):
        self.d_x = d_x^
        self.d_g = d_g^
        self.d_b = d_b^


def group_norm_backward(
    go: Tensor,
    x: Tensor,
    weight: Tensor,
    num_groups: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises -> GroupNormBackward:
    """Backward of group_norm (NHWC). go/x [N,H,W,C] (F32); weight [C] (F32).
    Returns d_x [N,H,W,C], d_g [C], d_b [C]."""
    # BF16/F16 storage path: cast up, run F32 interior, cast grads down.
    # F32 path byte-identical (branch only on non-F32 input).
    if x.dtype() != STDtype.F32 or go.dtype() != STDtype.F32 or weight.dtype() != STDtype.F32:
        var out_dt = x.dtype()
        var go32 = cast_tensor(go, STDtype.F32, ctx)
        var x32 = cast_tensor(x, STDtype.F32, ctx)
        var w32 = cast_tensor(weight, STDtype.F32, ctx)
        var g32 = group_norm_backward(go32, x32, w32, num_groups, eps, ctx)
        var dx_dn = cast_tensor(g32.d_x^, out_dt, ctx)
        var dg_dn = cast_tensor(g32.d_g^, out_dt, ctx)
        var db_dn = cast_tensor(g32.d_b^, out_dt, ctx)
        return GroupNormBackward(dx_dn^, dg_dn^, db_dn^)
    var xshape = x.shape()
    if len(xshape) != 4:
        raise Error("group_norm_backward: x must be NHWC [N,H,W,C]")
    var n = xshape[0]
    var h = xshape[1]
    var w = xshape[2]
    var c = xshape[3]
    var hw = h * w
    if num_groups <= 0 or c % num_groups != 0:
        raise Error("group_norm_backward: num_groups must divide C")

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var dg_buf = ctx.enqueue_create_buffer[DType.uint8](weight.nbytes())
    var db_buf = ctx.enqueue_create_buffer[DType.uint8](weight.nbytes())

    var total = x.numel()
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](c))

    var GO = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        go.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    var DX = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dx_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var DG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dg_buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    var DB = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        db_buf.unsafe_ptr().bitcast[Float32](), c_rl
    )

    var dx_blocks = n * num_groups
    ctx.enqueue_function[_gn_bwd_dx_kernel, _gn_bwd_dx_kernel](
        GO, X, G, DX, n, hw, c, num_groups, eps,
        grid_dim=dx_blocks, block_dim=_TPB,
    )
    var c_grid = (c + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_gn_bwd_param_kernel, _gn_bwd_param_kernel](
        GO, X, DG, DB, n, hw, c, num_groups, eps,
        grid_dim=c_grid, block_dim=_BLOCK,
    )
    ctx.synchronize()

    return GroupNormBackward(
        Tensor(dx_buf^, xshape.copy(), STDtype.F32),
        Tensor(dg_buf^, weight.shape().copy(), STDtype.F32),
        Tensor(db_buf^, weight.shape().copy(), STDtype.F32),
    )
