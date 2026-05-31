# ops/rope_struct_backward.mojo — BACKWARD for three DiT structural primitives:
#   RoPePrecomputed   (rope_backward)
#   QkvSplitPermute   (qkv_split_permute_backward)
#   GateResidual      (gate_residual_backward)
#
# Phase T-struct of FULL_PORT_TRAINING_PLAN.md. All-F32 interior, parity-gated
# vs PyTorch autograd at cos >= 0.999 (parity/rope_struct_bwd_parity.mojo).
#
# These are the structural (non-matmul) ops in a DiT attention/AdaLN block whose
# backward is pure index math + elementwise. Each backward MATCHES the forward
# convention exactly (read alongside ops/rope.mojo, ops/elementwise.mojo, and the
# qkv split/permute used by models/dit/flux1_dit.mojo + sd3_mmdit.mojo).
#
# ── 1) RoPePrecomputed backward ──────────────────────────────────────────────
# Forward (rope.mojo) rotates pairs of channels by a precomputed angle theta;
# cos/sin are NON-learnable tables (no grad), so backward yields only d_x.
# A 2x2 rotation R(theta) = [[c,-s],[s,c]] is orthogonal: R(theta)^T = R(-theta).
# d_x = R(theta)^T @ grad_out = R(-theta) @ grad_out  (rotate grad_out by -theta).
#
#   INTERLEAVED  (FLUX/Klein), pair (2i, 2i+1), angle i:
#     fwd: o0 = x0*c - x1*s ; o1 = x0*s + x1*c
#     bwd: dx0 = g0*c + g1*s ; dx1 = -g0*s + g1*c     (R(-theta) on (g0,g1))
#
#   HALFSPLIT  (Z-Image), pair (i, i+half), angle i:
#     fwd: o0 = x0*c - x1*s ; o1 = x1*c + x0*s   (o1 = x0*s + x1*c, same as above)
#     bwd: dx0 = g0*c + g1*s ; dx1 = -g0*s + g1*c
#
# Both variants share the SAME 2x2 rotation; only the channel pairing differs
# (offset 1 vs offset half). The transpose/inverse derivation is identical.
#
# ── 2) QkvSplitPermute backward ──────────────────────────────────────────────
# Forward: a fused projection [B, N, 3*H*Dh] is split into q/k/v along the last
# dim (q = [:, :, 0:H*Dh], k = [:, :, H*Dh:2*H*Dh], v = [:, :, 2*H*Dh:3*H*Dh]),
# then each is reshaped [B,N,H,Dh] (the BSHD per-head layout SDPA consumes — see
# flux1_dit `_qkv_part` and sd3_mmdit `_qkv_project`). The reshape is a no-op on
# the row-major byte layout (contiguous [B,N,H,Dh] == contiguous [B,N,H*Dh]), so
# the only structural transform is the concat-along-last-dim of three contiguous
# [B,N,H*Dh] blocks back into [B,N,3*H*Dh].
# Backward: scatter grad_q | grad_k | grad_v back into the fused d_qkv at column
# offsets 0 | HD | 2*HD. Returns a single d_qkv Tensor [B, N, 3*H*Dh].
#
# ── 3) GateResidual backward ─────────────────────────────────────────────────
# Forward (elementwise.residual_gate): o[r,c] = x[r,c] + g[c]*y[r,c], g per-chan.
#   d_x[r,c] = grad_out[r,c]
#   d_y[r,c] = grad_out[r,c] * g[c]
#   d_g[c]   = sum_r grad_out[r,c] * y[r,c]      (reduce over all rows)
#
# Mojo 1.0.0b1, NVIDIA GPU. F32 only (parity inputs are F32).

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256


# ─────────────────────────────────────────────────────────────────────────────
# 1) RoPePrecomputed backward
# ─────────────────────────────────────────────────────────────────────────────
def _rope_bwd_interleaved_kernel_f32(
    g: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # grad_out [rows, D]
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin], # [rows, half]
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin], # [rows, half]
    dx: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [rows, D]
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var g0 = rebind[Scalar[DType.float32]](g[r, 2 * i])
        var g1 = rebind[Scalar[DType.float32]](g[r, 2 * i + 1])
        var cv = rebind[Scalar[DType.float32]](cos[r, i])
        var sv = rebind[Scalar[DType.float32]](sin[r, i])
        # R(-theta): dx0 = g0*c + g1*s ; dx1 = -g0*s + g1*c
        dx[r, 2 * i] = rebind[dx.element_type](g0 * cv + g1 * sv)
        dx[r, 2 * i + 1] = rebind[dx.element_type](-g0 * sv + g1 * cv)


def _rope_bwd_halfsplit_kernel_f32(
    g: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dx: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var g0 = rebind[Scalar[DType.float32]](g[r, i])
        var g1 = rebind[Scalar[DType.float32]](g[r, i + half])
        var cv = rebind[Scalar[DType.float32]](cos[r, i])
        var sv = rebind[Scalar[DType.float32]](sin[r, i])
        # R(-theta): dx0 = g0*c + g1*s ; dx1 = -g0*s + g1*c
        dx[r, i] = rebind[dx.element_type](g0 * cv + g1 * sv)
        dx[r, i + half] = rebind[dx.element_type](-g0 * sv + g1 * cv)


def _rope_bwd_validate(
    grad_out: Tensor, cos: Tensor, sin: Tensor
) raises -> List[Int]:
    """Shared shape checks. Returns [rows, half=D/2]. Mirrors rope._rope_common_validate."""
    var gshape = grad_out.shape()
    if len(gshape) < 1:
        raise Error("rope_backward: grad_out must have rank >= 1")
    var d = gshape[len(gshape) - 1]
    if d % 2 != 0:
        raise Error("rope_backward: last dim D must be even")
    var half = d // 2
    var rows = 1
    for i in range(len(gshape) - 1):
        rows *= gshape[i]
    if cos.numel() != rows * half:
        raise Error("rope_backward: cos numel must equal rows*(D/2)")
    if sin.numel() != rows * half:
        raise Error("rope_backward: sin numel must equal rows*(D/2)")
    if grad_out.dtype() != STDtype.F32 or cos.dtype() != STDtype.F32 or sin.dtype() != STDtype.F32:
        raise Error("rope_backward: F32 only (grad_out/cos/sin)")
    var out = List[Int]()
    out.append(rows)
    out.append(half)
    return out^


def rope_backward(
    grad_out: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    """Backward of RoPePrecomputed: apply the inverse rotation R(-theta) to
    grad_out. cos/sin are non-learnable precomputed tables, so this returns only
    d_x (same shape/dtype as grad_out, F32).

    grad_out: [..., D] (D even; leading dims flattened to rows)
    cos/sin:  [rows, D/2]
    interleaved: True = FLUX/Klein pairing (2i, 2i+1); False = Z-Image halfsplit
                 pairing (i, i+D/2). MUST match the forward variant used.
    """
    var dims = _rope_bwd_validate(grad_out, cos, sin)
    var rows = dims[0]
    var half = dims[1]
    var d = half * 2

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](grad_out.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    var G = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        cos.buf.unsafe_ptr().bitcast[Float32](), f_rl
    )
    var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        sin.buf.unsafe_ptr().bitcast[Float32](), f_rl
    )
    var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dx_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    if interleaved:
        ctx.enqueue_function[
            _rope_bwd_interleaved_kernel_f32, _rope_bwd_interleaved_kernel_f32
        ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
    else:
        ctx.enqueue_function[
            _rope_bwd_halfsplit_kernel_f32, _rope_bwd_halfsplit_kernel_f32
        ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(dx_buf^, grad_out.shape(), grad_out.dtype())


# ─────────────────────────────────────────────────────────────────────────────
# 2) QkvSplitPermute backward
# ─────────────────────────────────────────────────────────────────────────────
# Scatter grad_q | grad_k | grad_v (each [rows, HD]) back into d_qkv [rows, 3*HD]
# at column offsets 0 | HD | 2*HD. One thread per element of one source block.
def _qkv_scatter_kernel_f32(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # [rows, HD]
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # [rows, 3*HD]
    rows: Int,
    hd: Int,
    col_off: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * hd
    if idx < total:
        var r = idx // hd
        var c = idx % hd
        dst[r, col_off + c] = rebind[dst.element_type](
            rebind[Scalar[DType.float32]](src[r, c])
        )


def qkv_split_permute_backward(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Backward of QkvSplitPermute. Forward split a fused [B, N, 3*H*Dh] into
    q/k/v (last-dim slices 0|HD|2*HD) reshaped to BSHD [B,N,H,Dh]; the reshape is
    a no-op on the row-major bytes, so backward simply concatenates the three
    per-head grads (each viewed as [rows, H*Dh]) along the last dim back into the
    fused [B, N, 3*H*Dh] layout. Returns d_qkv (F32).

    grad_q/grad_k/grad_v: [B, N, H, Dh] (or any shape with numel = rows*H*Dh);
    treated flat as [rows, H*Dh] where rows = B*N. All three must match.
    """
    if grad_q.dtype() != STDtype.F32 or grad_k.dtype() != STDtype.F32 or grad_v.dtype() != STDtype.F32:
        raise Error("qkv_split_permute_backward: F32 only")
    var qshape = grad_q.shape()
    if len(qshape) < 2:
        raise Error("qkv_split_permute_backward: grad_q must have rank >= 2")
    # last dim = Dh, second-to-last = H -> HD = H*Dh; rows = product of the rest.
    var dh = qshape[len(qshape) - 1]
    var hh = qshape[len(qshape) - 2]
    var hd = hh * dh
    var rows = 1
    for i in range(len(qshape) - 2):
        rows *= qshape[i]
    if grad_k.numel() != rows * hd or grad_v.numel() != rows * hd:
        raise Error("qkv_split_permute_backward: q/k/v numel mismatch")

    var fused = 3 * hd
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](rows * fused * 4)
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, hd))
    var dst_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, fused))
    var total = rows * hd
    var grid = (total + _BLOCK - 1) // _BLOCK

    var DST = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), dst_rl
    )
    var GQ = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        grad_q.buf.unsafe_ptr().bitcast[Float32](), src_rl
    )
    var GK = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        grad_k.buf.unsafe_ptr().bitcast[Float32](), src_rl
    )
    var GV = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        grad_v.buf.unsafe_ptr().bitcast[Float32](), src_rl
    )
    ctx.enqueue_function[_qkv_scatter_kernel_f32, _qkv_scatter_kernel_f32](
        GQ, DST, rows, hd, 0, grid_dim=grid, block_dim=_BLOCK)
    ctx.enqueue_function[_qkv_scatter_kernel_f32, _qkv_scatter_kernel_f32](
        GK, DST, rows, hd, hd, grid_dim=grid, block_dim=_BLOCK)
    ctx.enqueue_function[_qkv_scatter_kernel_f32, _qkv_scatter_kernel_f32](
        GV, DST, rows, hd, 2 * hd, grid_dim=grid, block_dim=_BLOCK)
    # Fused output shape: replace [..., H, Dh] with [..., 3*H*Dh].
    var out_shape = List[Int]()
    for i in range(len(qshape) - 2):
        out_shape.append(qshape[i])
    out_shape.append(fused)
    return Tensor(out_buf^, out_shape^, STDtype.F32)


# ─────────────────────────────────────────────────────────────────────────────
# 3) GateResidual backward
# ─────────────────────────────────────────────────────────────────────────────
struct GateResidualGrads(Movable):
    """Backward outputs of `gate_residual_backward`: d_x, d_g (per-channel), d_y."""

    var d_x: Tensor
    var d_g: Tensor
    var d_y: Tensor

    def __init__(out self, var d_x: Tensor, var d_g: Tensor, var d_y: Tensor):
        self.d_x = d_x^
        self.d_g = d_g^
        self.d_y = d_y^


# d_x[r,c] = grad_out[r,c] ; d_y[r,c] = grad_out[r,c] * g[c]
def _gate_dxdy_kernel_f32(
    g: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # grad_out [rows, cols]
    gate: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], # [cols]
    dx: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [rows, cols]
    dy: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [rows, cols]
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var gv = rebind[Scalar[DType.float32]](g[r, c])
        var gate_v = rebind[Scalar[DType.float32]](gate[c])
        dx[r, c] = rebind[dx.element_type](gv)
        dy[r, c] = rebind[dy.element_type](gv * gate_v)


# d_g[c] = sum_r grad_out[r,c] * y[r,c]. One block per channel c; F32 tree-reduce
# over the rows dimension (mirrors attention_backward _softmax_bwd_rows_f32).
def _gate_dg_kernel_f32(
    g: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # grad_out [rows, cols]
    y: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # [rows, cols]
    dg: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [cols]
    rows: Int,
    cols: Int,
):
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var r = tid
    while r < rows:
        var gv = rebind[Scalar[DType.float32]](g[r, c])
        var yv = rebind[Scalar[DType.float32]](y[r, c])
        lsum += gv * yv
        r += _TPB
    shared[tid] = lsum
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        dg[c] = rebind[dg.element_type](shared[0])


def gate_residual_backward(
    grad_out: Tensor,
    x: Tensor,
    g: Tensor,
    y: Tensor,
    ctx: DeviceContext,
    compute_gate_grad: Bool = True,
) raises -> GateResidualGrads:
    """Backward of GateResidual (forward: o = x + g*y, g per-channel [C]).
      d_x = grad_out
      d_y = grad_out * g (broadcast per channel)
      d_g[c] = sum over all rows of grad_out[r,c]*y[r,c]
    `x` is accepted for signature symmetry / shape validation (the forward keeps
    o linear in x, so d_x does not depend on x's value). Returns F32 grads.

    grad_out, x, y: [..., C] ; g: [C] (per-channel gate).
    """
    if (grad_out.dtype() != STDtype.F32 or x.dtype() != STDtype.F32
            or g.dtype() != STDtype.F32 or y.dtype() != STDtype.F32):
        raise Error("gate_residual_backward: F32 only")
    var gshape = grad_out.shape()
    if len(gshape) < 1:
        raise Error("gate_residual_backward: grad_out must have rank >= 1")
    var cols = gshape[len(gshape) - 1]
    var gateshape = g.shape()
    if len(gateshape) != 1 or gateshape[0] != cols:
        raise Error("gate_residual_backward: g must be [C]")
    if x.numel() != grad_out.numel() or y.numel() != grad_out.numel():
        raise Error("gate_residual_backward: grad_out/x/y numel mismatch")
    var rows = 1
    for i in range(len(gshape) - 1):
        rows *= gshape[i]

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](grad_out.nbytes())
    var dy_buf = ctx.enqueue_create_buffer[DType.uint8](grad_out.nbytes())
    var dg_nbytes = cols * 4
    if not compute_gate_grad:
        dg_nbytes = 0
    var dg_buf = ctx.enqueue_create_buffer[DType.uint8](dg_nbytes)

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](cols))
    var total = rows * cols
    var grid = (total + _BLOCK - 1) // _BLOCK

    var G = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        grad_out.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var GATE = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        g.buf.unsafe_ptr().bitcast[Float32](), v_rl
    )
    var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dx_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var DY = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dy_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )

    ctx.enqueue_function[_gate_dxdy_kernel_f32, _gate_dxdy_kernel_f32](
        G, GATE, DX, DY, rows, cols, grid_dim=grid, block_dim=_BLOCK)
    var dg_shape = List[Int]()
    if compute_gate_grad:
        var Y = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            y.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var DG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            dg_buf.unsafe_ptr().bitcast[Float32](), v_rl
        )
        ctx.enqueue_function[_gate_dg_kernel_f32, _gate_dg_kernel_f32](
            G, Y, DG, rows, cols, grid_dim=cols, block_dim=_TPB)
        dg_shape.append(cols)
    else:
        dg_shape.append(0)
    var dx_t = Tensor(dx_buf^, grad_out.shape(), STDtype.F32)
    var dy_t = Tensor(dy_buf^, grad_out.shape(), STDtype.F32)
    var dg_t = Tensor(dg_buf^, dg_shape^, STDtype.F32)
    return GateResidualGrads(dx_t^, dg_t^, dy_t^)
