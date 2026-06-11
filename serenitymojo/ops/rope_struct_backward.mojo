# ops/rope_struct_backward.mojo — BACKWARD for three DiT structural primitives:
#   RoPePrecomputed   (rope_backward)
#   QkvSplitPermute   (qkv_split_permute_backward)
#   GateResidual      (gate_residual_backward)
#
# Phase T-struct of FULL_PORT_TRAINING_PLAN.md. Storage dtype is preserved at
# tensor boundaries; kernels cast scalar elements to F32 for math and write
# gradients back to the input dtype.
#
# These are the structural (non-matmul) ops in a DiT attention/AdaLN block whose
# backward is pure index math + elementwise. Each backward MATCHES the forward
# convention exactly (read alongside ops/rope.mojo, ops/elementwise.mojo, and the
# qkv split/permute used by models/dit/flux1_dit.mojo + sd3_mmdit.mojo).
#
# ── 1) RoPePrecomputed backward ──────────────────────────────────────────────
# Forward (rope.mojo) rotates pairs of channels by a precomputed angle theta;
# cos/sin are NON-learnable tables (no grad), so backward yields only d_x.
# BF16/F16 activations may carry F32 cos/sin tables: diffusers Flux/Klein uses
# float table math (`x.float() * cos/sin`) and casts the rotated tensor back to
# the activation dtype. Backward mirrors that F32 table compute while preserving
# d_x storage dtype.
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
# Mojo 1.0.0b1, NVIDIA GPU.

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
def _rope_bwd_interleaved_kernel[dtype: DType](
    g: LayoutTensor[dtype, _DYN2, MutAnyOrigin],   # grad_out [rows, D]
    cos: LayoutTensor[dtype, _DYN2, MutAnyOrigin], # [rows, half]
    sin: LayoutTensor[dtype, _DYN2, MutAnyOrigin], # [rows, half]
    dx: LayoutTensor[dtype, _DYN2, MutAnyOrigin],  # [rows, D]
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var g0 = rebind[Scalar[dtype]](g[r, 2 * i]).cast[DType.float32]()
        var g1 = rebind[Scalar[dtype]](g[r, 2 * i + 1]).cast[DType.float32]()
        var cv = rebind[Scalar[dtype]](cos[r, i]).cast[DType.float32]()
        var sv = rebind[Scalar[dtype]](sin[r, i]).cast[DType.float32]()
        # R(-theta): dx0 = g0*c + g1*s ; dx1 = -g0*s + g1*c
        dx[r, 2 * i] = rebind[dx.element_type]((g0 * cv + g1 * sv).cast[dtype]())
        dx[r, 2 * i + 1] = rebind[dx.element_type]((-g0 * sv + g1 * cv).cast[dtype]())


def _rope_bwd_interleaved_kernel_tables[dtype: DType, table_dtype: DType](
    g: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[table_dtype, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[table_dtype, _DYN2, MutAnyOrigin],
    dx: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var g0 = rebind[Scalar[dtype]](g[r, 2 * i]).cast[DType.float32]()
        var g1 = rebind[Scalar[dtype]](g[r, 2 * i + 1]).cast[DType.float32]()
        var cv = rebind[Scalar[table_dtype]](cos[r, i]).cast[DType.float32]()
        var sv = rebind[Scalar[table_dtype]](sin[r, i]).cast[DType.float32]()
        dx[r, 2 * i] = rebind[dx.element_type]((g0 * cv + g1 * sv).cast[dtype]())
        dx[r, 2 * i + 1] = rebind[dx.element_type]((-g0 * sv + g1 * cv).cast[dtype]())


def _rope_bwd_halfsplit_kernel[dtype: DType](
    g: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    dx: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var g0 = rebind[Scalar[dtype]](g[r, i]).cast[DType.float32]()
        var g1 = rebind[Scalar[dtype]](g[r, i + half]).cast[DType.float32]()
        var cv = rebind[Scalar[dtype]](cos[r, i]).cast[DType.float32]()
        var sv = rebind[Scalar[dtype]](sin[r, i]).cast[DType.float32]()
        # R(-theta): dx0 = g0*c + g1*s ; dx1 = -g0*s + g1*c
        dx[r, i] = rebind[dx.element_type]((g0 * cv + g1 * sv).cast[dtype]())
        dx[r, i + half] = rebind[dx.element_type]((-g0 * sv + g1 * cv).cast[dtype]())


def _rope_bwd_halfsplit_kernel_tables[dtype: DType, table_dtype: DType](
    g: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[table_dtype, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[table_dtype, _DYN2, MutAnyOrigin],
    dx: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var g0 = rebind[Scalar[dtype]](g[r, i]).cast[DType.float32]()
        var g1 = rebind[Scalar[dtype]](g[r, i + half]).cast[DType.float32]()
        var cv = rebind[Scalar[table_dtype]](cos[r, i]).cast[DType.float32]()
        var sv = rebind[Scalar[table_dtype]](sin[r, i]).cast[DType.float32]()
        dx[r, i] = rebind[dx.element_type]((g0 * cv + g1 * sv).cast[dtype]())
        dx[r, i + half] = rebind[dx.element_type]((-g0 * sv + g1 * cv).cast[dtype]())


# HALFSPLIT backward with a FULL-WIDTH table (cos[i] may differ from cos[i+half]).
# Forward (rope._rope_halfsplit_full_kernel, = diffusers ERNIE apply_rotary_emb on
# the interleaved-doubled table):
#   o[i]      = x[i]*c0 - x[i+half]*s0     (c0=cos[i],      s0=sin[i])
#   o[i+half] = x[i+half]*c1 + x[i]*s1     (c1=cos[i+half], s1=sin[i+half])
# Jacobian-transpose (g = grad_out):
#   dx[i]      = g[i]*c0 + g[i+half]*s1
#   dx[i+half] = -g[i]*s0 + g[i+half]*c1
# Reduces to the single-angle halfsplit kernel ONLY when c0==c1 and s0==s1
# (the degenerate table). The real ERNIE table has c0!=c1, so the half-width
# kernel is wrong there — this kernel reads BOTH angles (full-width table).
def _rope_bwd_halfsplit_full_kernel[dtype: DType](
    g: LayoutTensor[dtype, _DYN2, MutAnyOrigin],   # grad_out [rows, D]
    cos: LayoutTensor[dtype, _DYN2, MutAnyOrigin], # [rows, D] full-width
    sin: LayoutTensor[dtype, _DYN2, MutAnyOrigin], # [rows, D] full-width
    dx: LayoutTensor[dtype, _DYN2, MutAnyOrigin],  # [rows, D]
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var g0 = rebind[Scalar[dtype]](g[r, i]).cast[DType.float32]()
        var g1 = rebind[Scalar[dtype]](g[r, i + half]).cast[DType.float32]()
        var c0 = rebind[Scalar[dtype]](cos[r, i]).cast[DType.float32]()
        var s0 = rebind[Scalar[dtype]](sin[r, i]).cast[DType.float32]()
        var c1 = rebind[Scalar[dtype]](cos[r, i + half]).cast[DType.float32]()
        var s1 = rebind[Scalar[dtype]](sin[r, i + half]).cast[DType.float32]()
        dx[r, i] = rebind[dx.element_type]((g0 * c0 + g1 * s1).cast[dtype]())
        dx[r, i + half] = rebind[dx.element_type]((-g0 * s0 + g1 * c1).cast[dtype]())


def _rope_bwd_halfsplit_full_kernel_tables[dtype: DType, table_dtype: DType](
    g: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[table_dtype, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[table_dtype, _DYN2, MutAnyOrigin],
    dx: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var g0 = rebind[Scalar[dtype]](g[r, i]).cast[DType.float32]()
        var g1 = rebind[Scalar[dtype]](g[r, i + half]).cast[DType.float32]()
        var c0 = rebind[Scalar[table_dtype]](cos[r, i]).cast[DType.float32]()
        var s0 = rebind[Scalar[table_dtype]](sin[r, i]).cast[DType.float32]()
        var c1 = rebind[Scalar[table_dtype]](cos[r, i + half]).cast[DType.float32]()
        var s1 = rebind[Scalar[table_dtype]](sin[r, i + half]).cast[DType.float32]()
        dx[r, i] = rebind[dx.element_type]((g0 * c0 + g1 * s1).cast[dtype]())
        dx[r, i + half] = rebind[dx.element_type]((-g0 * s0 + g1 * c1).cast[dtype]())


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
    var cos_dt = cos.dtype()
    var sin_dt = sin.dtype()
    if cos_dt != sin_dt:
        raise Error("rope_backward: cos/sin dtype mismatch")
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
    d_x (same shape/dtype as grad_out; F32 scalar math).

    grad_out: [..., D] (D even; leading dims flattened to rows)
    cos/sin:  [rows, D/2] (same dtype as grad_out or F32)
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

    var dt = grad_out.dtype().to_mojo_dtype()
    var table_dt = cos.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var G = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float32](), x_rl)
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            if interleaved:
                ctx.enqueue_function[
                    _rope_bwd_interleaved_kernel[DType.float32],
                    _rope_bwd_interleaved_kernel[DType.float32],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
            else:
                ctx.enqueue_function[
                    _rope_bwd_halfsplit_kernel[DType.float32],
                    _rope_bwd_halfsplit_kernel[DType.float32],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        elif table_dt == DType.bfloat16:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            if interleaved:
                ctx.enqueue_function[
                    _rope_bwd_interleaved_kernel_tables[DType.float32, DType.bfloat16],
                    _rope_bwd_interleaved_kernel_tables[DType.float32, DType.bfloat16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
            else:
                ctx.enqueue_function[
                    _rope_bwd_halfsplit_kernel_tables[DType.float32, DType.bfloat16],
                    _rope_bwd_halfsplit_kernel_tables[DType.float32, DType.bfloat16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            if interleaved:
                ctx.enqueue_function[
                    _rope_bwd_interleaved_kernel_tables[DType.float32, DType.float16],
                    _rope_bwd_interleaved_kernel_tables[DType.float32, DType.float16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
            else:
                ctx.enqueue_function[
                    _rope_bwd_halfsplit_kernel_tables[DType.float32, DType.float16],
                    _rope_bwd_halfsplit_kernel_tables[DType.float32, DType.float16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var G = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var DX = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            if interleaved:
                ctx.enqueue_function[
                    _rope_bwd_interleaved_kernel_tables[DType.bfloat16, DType.float32],
                    _rope_bwd_interleaved_kernel_tables[DType.bfloat16, DType.float32],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
            else:
                ctx.enqueue_function[
                    _rope_bwd_halfsplit_kernel_tables[DType.bfloat16, DType.float32],
                    _rope_bwd_halfsplit_kernel_tables[DType.bfloat16, DType.float32],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        elif table_dt == DType.bfloat16:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            if interleaved:
                ctx.enqueue_function[
                    _rope_bwd_interleaved_kernel[DType.bfloat16],
                    _rope_bwd_interleaved_kernel[DType.bfloat16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
            else:
                ctx.enqueue_function[
                    _rope_bwd_halfsplit_kernel[DType.bfloat16],
                    _rope_bwd_halfsplit_kernel[DType.bfloat16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            if interleaved:
                ctx.enqueue_function[
                    _rope_bwd_interleaved_kernel_tables[DType.bfloat16, DType.float16],
                    _rope_bwd_interleaved_kernel_tables[DType.bfloat16, DType.float16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
            else:
                ctx.enqueue_function[
                    _rope_bwd_halfsplit_kernel_tables[DType.bfloat16, DType.float16],
                    _rope_bwd_halfsplit_kernel_tables[DType.bfloat16, DType.float16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
    else:
        var G = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var DX = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float16](), x_rl)
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            if interleaved:
                ctx.enqueue_function[
                    _rope_bwd_interleaved_kernel_tables[DType.float16, DType.float32],
                    _rope_bwd_interleaved_kernel_tables[DType.float16, DType.float32],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
            else:
                ctx.enqueue_function[
                    _rope_bwd_halfsplit_kernel_tables[DType.float16, DType.float32],
                    _rope_bwd_halfsplit_kernel_tables[DType.float16, DType.float32],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        elif table_dt == DType.bfloat16:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            if interleaved:
                ctx.enqueue_function[
                    _rope_bwd_interleaved_kernel_tables[DType.float16, DType.bfloat16],
                    _rope_bwd_interleaved_kernel_tables[DType.float16, DType.bfloat16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
            else:
                ctx.enqueue_function[
                    _rope_bwd_halfsplit_kernel_tables[DType.float16, DType.bfloat16],
                    _rope_bwd_halfsplit_kernel_tables[DType.float16, DType.bfloat16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            if interleaved:
                ctx.enqueue_function[
                    _rope_bwd_interleaved_kernel[DType.float16],
                    _rope_bwd_interleaved_kernel[DType.float16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
            else:
                ctx.enqueue_function[
                    _rope_bwd_halfsplit_kernel[DType.float16],
                    _rope_bwd_halfsplit_kernel[DType.float16],
                ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(dx_buf^, grad_out.shape(), grad_out.dtype())


def _rope_bwd_full_validate(
    grad_out: Tensor, cos: Tensor, sin: Tensor
) raises -> List[Int]:
    """Shape checks for the FULL-WIDTH halfsplit backward. Returns [rows, half].

    Mirrors rope._rope_full_validate: cos/sin are [rows, D] (NOT [rows, D/2]),
    so the kernel can read both cos[i] and cos[i+half]."""
    var gshape = grad_out.shape()
    if len(gshape) < 1:
        raise Error("rope_halfsplit_full_backward: grad_out must have rank >= 1")
    var d = gshape[len(gshape) - 1]
    if d % 2 != 0:
        raise Error("rope_halfsplit_full_backward: last dim D must be even")
    var half = d // 2
    var rows = 1
    for i in range(len(gshape) - 1):
        rows *= gshape[i]
    if cos.numel() != rows * d:
        raise Error("rope_halfsplit_full_backward: cos numel must equal rows*D")
    if sin.numel() != rows * d:
        raise Error("rope_halfsplit_full_backward: sin numel must equal rows*D")
    var cos_dt = cos.dtype()
    var sin_dt = sin.dtype()
    if cos_dt != sin_dt:
        raise Error("rope_halfsplit_full_backward: cos/sin dtype mismatch")
    var out = List[Int]()
    out.append(rows)
    out.append(half)
    return out^


def rope_halfsplit_full_backward(
    grad_out: Tensor,
    cos: Tensor,
    sin: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Backward of the half-split RoPE forward `rope.rope_halfsplit_full`, where
    the cos/sin tables are FULL-WIDTH [rows, D] and cos[i] may differ from
    cos[i+D/2] (the real ERNIE interleaved-doubled table). Returns d_x only
    (cos/sin are non-learnable). Matches diffusers ERNIE apply_rotary_emb autograd.

    grad_out: [..., D] (D even; leading dims flattened to rows)
    cos/sin:  [rows, D]  (FULL-WIDTH — both halves carry their own angle;
               same dtype as grad_out or F32)

    Use this (NOT `rope_backward(..., interleaved=False)`) whenever the forward
    used `rope_halfsplit_full` on a table where the two halves differ. The
    half-width `rope_backward` arm is correct only for the degenerate
    cos[i]==cos[i+half] table; it silently aliases the wrong angle otherwise.
    """
    var dims = _rope_bwd_full_validate(grad_out, cos, sin)
    var rows = dims[0]
    var half = dims[1]
    var d = half * 2

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](grad_out.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    var dt = grad_out.dtype().to_mojo_dtype()
    var table_dt = cos.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var G = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float32](), x_rl)
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            ctx.enqueue_function[
                _rope_bwd_halfsplit_full_kernel[DType.float32],
                _rope_bwd_halfsplit_full_kernel[DType.float32],
            ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        elif table_dt == DType.bfloat16:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            ctx.enqueue_function[
                _rope_bwd_halfsplit_full_kernel_tables[DType.float32, DType.bfloat16],
                _rope_bwd_halfsplit_full_kernel_tables[DType.float32, DType.bfloat16],
            ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            ctx.enqueue_function[
                _rope_bwd_halfsplit_full_kernel_tables[DType.float32, DType.float16],
                _rope_bwd_halfsplit_full_kernel_tables[DType.float32, DType.float16],
            ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var G = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var DX = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            ctx.enqueue_function[
                _rope_bwd_halfsplit_full_kernel_tables[DType.bfloat16, DType.float32],
                _rope_bwd_halfsplit_full_kernel_tables[DType.bfloat16, DType.float32],
            ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        elif table_dt == DType.bfloat16:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            ctx.enqueue_function[
                _rope_bwd_halfsplit_full_kernel[DType.bfloat16],
                _rope_bwd_halfsplit_full_kernel[DType.bfloat16],
            ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            ctx.enqueue_function[
                _rope_bwd_halfsplit_full_kernel_tables[DType.bfloat16, DType.float16],
                _rope_bwd_halfsplit_full_kernel_tables[DType.bfloat16, DType.float16],
            ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
    else:
        var G = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var DX = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float16](), x_rl)
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float32](), f_rl)
            ctx.enqueue_function[
                _rope_bwd_halfsplit_full_kernel_tables[DType.float16, DType.float32],
                _rope_bwd_halfsplit_full_kernel_tables[DType.float16, DType.float32],
            ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        elif table_dt == DType.bfloat16:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[BFloat16](), f_rl)
            ctx.enqueue_function[
                _rope_bwd_halfsplit_full_kernel_tables[DType.float16, DType.bfloat16],
                _rope_bwd_halfsplit_full_kernel_tables[DType.float16, DType.bfloat16],
            ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                cos.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                sin.buf.unsafe_ptr().bitcast[Float16](), f_rl)
            ctx.enqueue_function[
                _rope_bwd_halfsplit_full_kernel[DType.float16],
                _rope_bwd_halfsplit_full_kernel[DType.float16],
            ](G, C, S, DX, rows, half, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(dx_buf^, grad_out.shape(), grad_out.dtype())


# ─────────────────────────────────────────────────────────────────────────────
# 2) QkvSplitPermute backward
# ─────────────────────────────────────────────────────────────────────────────
# Scatter grad_q | grad_k | grad_v (each [rows, HD]) back into d_qkv [rows, 3*HD]
# at column offsets 0 | HD | 2*HD. One thread per element of one source block.
def _qkv_scatter_kernel[dtype: DType](
    src: LayoutTensor[dtype, _DYN2, MutAnyOrigin],   # [rows, HD]
    dst: LayoutTensor[dtype, _DYN2, MutAnyOrigin],   # [rows, 3*HD]
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
            rebind[Scalar[dtype]](src[r, c])
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
    fused [B, N, 3*H*Dh] layout. Returns d_qkv in grad_q storage dtype.

    grad_q/grad_k/grad_v: [B, N, H, Dh] (or any shape with numel = rows*H*Dh);
    treated flat as [rows, H*Dh] where rows = B*N. All three must match.
    """
    if grad_q.dtype() != grad_k.dtype() or grad_q.dtype() != grad_v.dtype():
        raise Error("qkv_split_permute_backward: grad_q/k/v dtype mismatch")
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
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * fused * grad_q.dtype().byte_size()
    )
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, hd))
    var dst_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, fused))
    var total = rows * hd
    var grid = (total + _BLOCK - 1) // _BLOCK

    var dt = grad_q.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var DST = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), dst_rl)
        var GQ = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_q.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var GK = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_k.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var GV = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_v.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        ctx.enqueue_function[
            _qkv_scatter_kernel[DType.float32], _qkv_scatter_kernel[DType.float32]
        ](GQ, DST, rows, hd, 0, grid_dim=grid, block_dim=_BLOCK)
        ctx.enqueue_function[
            _qkv_scatter_kernel[DType.float32], _qkv_scatter_kernel[DType.float32]
        ](GK, DST, rows, hd, hd, grid_dim=grid, block_dim=_BLOCK)
        ctx.enqueue_function[
            _qkv_scatter_kernel[DType.float32], _qkv_scatter_kernel[DType.float32]
        ](GV, DST, rows, hd, 2 * hd, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var DST = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), dst_rl)
        var GQ = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            grad_q.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var GK = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            grad_k.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var GV = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            grad_v.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        ctx.enqueue_function[
            _qkv_scatter_kernel[DType.bfloat16], _qkv_scatter_kernel[DType.bfloat16]
        ](GQ, DST, rows, hd, 0, grid_dim=grid, block_dim=_BLOCK)
        ctx.enqueue_function[
            _qkv_scatter_kernel[DType.bfloat16], _qkv_scatter_kernel[DType.bfloat16]
        ](GK, DST, rows, hd, hd, grid_dim=grid, block_dim=_BLOCK)
        ctx.enqueue_function[
            _qkv_scatter_kernel[DType.bfloat16], _qkv_scatter_kernel[DType.bfloat16]
        ](GV, DST, rows, hd, 2 * hd, grid_dim=grid, block_dim=_BLOCK)
    else:
        var DST = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), dst_rl)
        var GQ = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            grad_q.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var GK = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            grad_k.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var GV = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            grad_v.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        ctx.enqueue_function[
            _qkv_scatter_kernel[DType.float16], _qkv_scatter_kernel[DType.float16]
        ](GQ, DST, rows, hd, 0, grid_dim=grid, block_dim=_BLOCK)
        ctx.enqueue_function[
            _qkv_scatter_kernel[DType.float16], _qkv_scatter_kernel[DType.float16]
        ](GK, DST, rows, hd, hd, grid_dim=grid, block_dim=_BLOCK)
        ctx.enqueue_function[
            _qkv_scatter_kernel[DType.float16], _qkv_scatter_kernel[DType.float16]
        ](GV, DST, rows, hd, 2 * hd, grid_dim=grid, block_dim=_BLOCK)
    # Fused output shape: replace [..., H, Dh] with [..., 3*H*Dh].
    var out_shape = List[Int]()
    for i in range(len(qshape) - 2):
        out_shape.append(qshape[i])
    out_shape.append(fused)
    return Tensor(out_buf^, out_shape^, grad_q.dtype())


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
def _gate_dxdy_kernel[dtype: DType](
    g: LayoutTensor[dtype, _DYN2, MutAnyOrigin],   # grad_out [rows, cols]
    gate: LayoutTensor[dtype, _DYN1, MutAnyOrigin], # [cols] or [nvec*cols]
    dx: LayoutTensor[dtype, _DYN2, MutAnyOrigin],  # [rows, cols]
    dy: LayoutTensor[dtype, _DYN2, MutAnyOrigin],  # [rows, cols]
    rows: Int,
    cols: Int,
    rows_per_vec: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var vo = (r // rows_per_vec) * cols
        var gv = rebind[Scalar[dtype]](g[r, c]).cast[DType.float32]()
        var gate_v = rebind[Scalar[dtype]](gate[vo + c]).cast[DType.float32]()
        dx[r, c] = rebind[dx.element_type](gv.cast[dtype]())
        dy[r, c] = rebind[dy.element_type]((gv * gate_v).cast[dtype]())


# d_g[c] = sum_r grad_out[r,c] * y[r,c]. One block per channel c; F32 tree-reduce
# over the rows dimension (mirrors attention_backward _softmax_bwd_rows_f32).
def _gate_dg_kernel[grad_dtype: DType, y_dtype: DType](
    g: LayoutTensor[grad_dtype, _DYN2, MutAnyOrigin],   # grad_out [rows, cols]
    y: LayoutTensor[y_dtype, _DYN2, MutAnyOrigin],      # [rows, cols]
    dg: LayoutTensor[grad_dtype, _DYN1, MutAnyOrigin],  # [cols]
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
        var gv = rebind[Scalar[grad_dtype]](g[r, c]).cast[DType.float32]()
        var yv = rebind[Scalar[y_dtype]](y[r, c]).cast[DType.float32]()
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
        dg[c] = rebind[dg.element_type](shared[0].cast[grad_dtype]())


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
    o linear in x, so d_x does not depend on x's value). `grad_out` and `g`
    share the gradient dtype; `x` and `y` may keep lower-precision activation
    storage. Returns d_x/d_y/d_g in the gradient/gate dtype.

    grad_out, x, y: [..., C] ; g: [C] (per-channel gate).
    """
    if grad_out.dtype() != g.dtype():
        raise Error("gate_residual_backward: grad_out/g dtype mismatch")
    var gshape = grad_out.shape()
    if len(gshape) < 1:
        raise Error("gate_residual_backward: grad_out must have rank >= 1")
    var cols = gshape[len(gshape) - 1]
    var gateshape = g.shape()
    # g: [C] (one vec) or [B, C] (per-sample gates; d_g reduction unsupported
    # for B>1 — LoRA training discards it; pass compute_gate_grad=False).
    var nvec = 1
    if len(gateshape) == 2 and gateshape[1] == cols:
        nvec = gateshape[0]
        if compute_gate_grad:
            raise Error(
                "gate_residual_backward: d_g unsupported for [B, C] gate"
            )
    elif len(gateshape) != 1 or gateshape[0] != cols:
        raise Error("gate_residual_backward: g must be [C] or [B, C]")
    if x.numel() != grad_out.numel() or y.numel() != grad_out.numel():
        raise Error("gate_residual_backward: grad_out/x/y numel mismatch")
    var rows = 1
    for i in range(len(gshape) - 1):
        rows *= gshape[i]
    if rows % nvec != 0:
        raise Error("gate_residual_backward: rows not divisible by vec count")
    var rows_per_vec = rows // nvec

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](grad_out.nbytes())
    var dy_buf = ctx.enqueue_create_buffer[DType.uint8](grad_out.nbytes())
    var dg_nbytes = cols * g.dtype().byte_size()
    if not compute_gate_grad:
        dg_nbytes = 0
    var dg_buf = ctx.enqueue_create_buffer[DType.uint8](dg_nbytes)

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nvec * cols))
    var total = rows * cols
    var grid = (total + _BLOCK - 1) // _BLOCK

    var dg_shape = List[Int]()
    var dt = grad_out.dtype().to_mojo_dtype()
    var ydt = y.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var G = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var GATE = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            g.buf.unsafe_ptr().bitcast[Float32](), v_rl)
        var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var DY = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dy_buf.unsafe_ptr().bitcast[Float32](), x_rl)
        ctx.enqueue_function[
            _gate_dxdy_kernel[DType.float32], _gate_dxdy_kernel[DType.float32]
        ](G, GATE, DX, DY, rows, cols, rows_per_vec, grid_dim=grid, block_dim=_BLOCK)
        if compute_gate_grad:
            var DG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                dg_buf.unsafe_ptr().bitcast[Float32](), v_rl)
            if ydt == DType.float32:
                var Y = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                    y.buf.unsafe_ptr().bitcast[Float32](), x_rl)
                ctx.enqueue_function[
                    _gate_dg_kernel[DType.float32, DType.float32],
                    _gate_dg_kernel[DType.float32, DType.float32],
                ](G, Y, DG, rows, cols, grid_dim=cols, block_dim=_TPB)
            elif ydt == DType.bfloat16:
                var Y = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                    y.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
                ctx.enqueue_function[
                    _gate_dg_kernel[DType.float32, DType.bfloat16],
                    _gate_dg_kernel[DType.float32, DType.bfloat16],
                ](G, Y, DG, rows, cols, grid_dim=cols, block_dim=_TPB)
            else:
                var Y = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                    y.buf.unsafe_ptr().bitcast[Float16](), x_rl)
                ctx.enqueue_function[
                    _gate_dg_kernel[DType.float32, DType.float16],
                    _gate_dg_kernel[DType.float32, DType.float16],
                ](G, Y, DG, rows, cols, grid_dim=cols, block_dim=_TPB)
            dg_shape.append(cols)
        else:
            dg_shape.append(0)
    elif dt == DType.bfloat16:
        var G = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var GATE = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            g.buf.unsafe_ptr().bitcast[BFloat16](), v_rl)
        var DX = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var DY = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            dy_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        ctx.enqueue_function[
            _gate_dxdy_kernel[DType.bfloat16], _gate_dxdy_kernel[DType.bfloat16]
        ](G, GATE, DX, DY, rows, cols, rows_per_vec, grid_dim=grid, block_dim=_BLOCK)
        if compute_gate_grad:
            var DG = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                dg_buf.unsafe_ptr().bitcast[BFloat16](), v_rl)
            if ydt == DType.float32:
                var Y = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                    y.buf.unsafe_ptr().bitcast[Float32](), x_rl)
                ctx.enqueue_function[
                    _gate_dg_kernel[DType.bfloat16, DType.float32],
                    _gate_dg_kernel[DType.bfloat16, DType.float32],
                ](G, Y, DG, rows, cols, grid_dim=cols, block_dim=_TPB)
            elif ydt == DType.bfloat16:
                var Y = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                    y.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
                ctx.enqueue_function[
                    _gate_dg_kernel[DType.bfloat16, DType.bfloat16],
                    _gate_dg_kernel[DType.bfloat16, DType.bfloat16],
                ](G, Y, DG, rows, cols, grid_dim=cols, block_dim=_TPB)
            else:
                var Y = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                    y.buf.unsafe_ptr().bitcast[Float16](), x_rl)
                ctx.enqueue_function[
                    _gate_dg_kernel[DType.bfloat16, DType.float16],
                    _gate_dg_kernel[DType.bfloat16, DType.float16],
                ](G, Y, DG, rows, cols, grid_dim=cols, block_dim=_TPB)
            dg_shape.append(cols)
        else:
            dg_shape.append(0)
    else:
        var G = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var GATE = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            g.buf.unsafe_ptr().bitcast[Float16](), v_rl)
        var DX = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var DY = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            dy_buf.unsafe_ptr().bitcast[Float16](), x_rl)
        ctx.enqueue_function[
            _gate_dxdy_kernel[DType.float16], _gate_dxdy_kernel[DType.float16]
        ](G, GATE, DX, DY, rows, cols, rows_per_vec, grid_dim=grid, block_dim=_BLOCK)
        if compute_gate_grad:
            var DG = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                dg_buf.unsafe_ptr().bitcast[Float16](), v_rl)
            if ydt == DType.float32:
                var Y = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                    y.buf.unsafe_ptr().bitcast[Float32](), x_rl)
                ctx.enqueue_function[
                    _gate_dg_kernel[DType.float16, DType.float32],
                    _gate_dg_kernel[DType.float16, DType.float32],
                ](G, Y, DG, rows, cols, grid_dim=cols, block_dim=_TPB)
            elif ydt == DType.bfloat16:
                var Y = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                    y.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
                ctx.enqueue_function[
                    _gate_dg_kernel[DType.float16, DType.bfloat16],
                    _gate_dg_kernel[DType.float16, DType.bfloat16],
                ](G, Y, DG, rows, cols, grid_dim=cols, block_dim=_TPB)
            else:
                var Y = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                    y.buf.unsafe_ptr().bitcast[Float16](), x_rl)
                ctx.enqueue_function[
                    _gate_dg_kernel[DType.float16, DType.float16],
                    _gate_dg_kernel[DType.float16, DType.float16],
                ](G, Y, DG, rows, cols, grid_dim=cols, block_dim=_TPB)
            dg_shape.append(cols)
        else:
            dg_shape.append(0)
    ctx.synchronize()
    var dx_t = Tensor(dx_buf^, grad_out.shape(), grad_out.dtype())
    var dy_t = Tensor(dy_buf^, grad_out.shape(), grad_out.dtype())
    var dg_t = Tensor(dg_buf^, dg_shape^, g.dtype())
    return GateResidualGrads(dx_t^, dg_t^, dy_t^)


def gate_residual_backward_dxdy(
    grad_out: Tensor,
    g: Tensor,
    ctx: DeviceContext,
) raises -> GateResidualGrads:
    """Backward of `o = x + g*y` when only d_x and d_y are needed.

    This skips the `y` input and d_g reduction. It is for training paths that
    intentionally discard AdaLN/gate-vector grads; d_x and d_y are identical to
    `gate_residual_backward(..., compute_gate_grad=False)`.
    """
    if grad_out.dtype() != g.dtype():
        raise Error("gate_residual_backward_dxdy: grad_out/g dtype mismatch")
    var gshape = grad_out.shape()
    if len(gshape) < 1:
        raise Error("gate_residual_backward_dxdy: grad_out must have rank >= 1")
    var cols = gshape[len(gshape) - 1]
    var gateshape = g.shape()
    # g: [C] (one vec) or [B, C] (per-sample gates, rows split evenly).
    var nvec = 1
    if len(gateshape) == 2 and gateshape[1] == cols:
        nvec = gateshape[0]
    elif len(gateshape) != 1 or gateshape[0] != cols:
        raise Error("gate_residual_backward_dxdy: g must be [C] or [B, C]")
    var rows = 1
    for i in range(len(gshape) - 1):
        rows *= gshape[i]
    if rows % nvec != 0:
        raise Error("gate_residual_backward_dxdy: rows not divisible by vecs")
    var rows_per_vec = rows // nvec

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](grad_out.nbytes())
    var dy_buf = ctx.enqueue_create_buffer[DType.uint8](grad_out.nbytes())
    var dg_buf = ctx.enqueue_create_buffer[DType.uint8](0)

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nvec * cols))
    var total = rows * cols
    var grid = (total + _BLOCK - 1) // _BLOCK

    var dt = grad_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var G = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var GATE = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            g.buf.unsafe_ptr().bitcast[Float32](), v_rl)
        var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var DY = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dy_buf.unsafe_ptr().bitcast[Float32](), x_rl)
        ctx.enqueue_function[
            _gate_dxdy_kernel[DType.float32], _gate_dxdy_kernel[DType.float32]
        ](G, GATE, DX, DY, rows, cols, rows_per_vec, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var G = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var GATE = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            g.buf.unsafe_ptr().bitcast[BFloat16](), v_rl)
        var DX = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var DY = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            dy_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        ctx.enqueue_function[
            _gate_dxdy_kernel[DType.bfloat16], _gate_dxdy_kernel[DType.bfloat16]
        ](G, GATE, DX, DY, rows, cols, rows_per_vec, grid_dim=grid, block_dim=_BLOCK)
    else:
        var G = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var GATE = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            g.buf.unsafe_ptr().bitcast[Float16](), v_rl)
        var DX = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var DY = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            dy_buf.unsafe_ptr().bitcast[Float16](), x_rl)
        ctx.enqueue_function[
            _gate_dxdy_kernel[DType.float16], _gate_dxdy_kernel[DType.float16]
        ](G, GATE, DX, DY, rows, cols, rows_per_vec, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var dg_shape = List[Int]()
    dg_shape.append(0)
    var dx_t = Tensor(dx_buf^, grad_out.shape(), grad_out.dtype())
    var dy_t = Tensor(dy_buf^, grad_out.shape(), grad_out.dtype())
    var dg_t = Tensor(dg_buf^, dg_shape^, g.dtype())
    return GateResidualGrads(dx_t^, dg_t^, dy_t^)
