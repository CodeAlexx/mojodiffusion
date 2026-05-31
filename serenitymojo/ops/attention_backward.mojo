# ops/attention_backward.mojo — decomposed (math-mode) SDPA BACKWARD.
#
# Phase T0 of FULL_PORT_TRAINING_PLAN.md (the SDPA-bwd de-risk kernel).
#
# This is the BACKWARD partner of ops/attention.mojo's math-mode forward. It is
# the DECOMPOSED path (plain matmuls + softmax recompute), NOT a cuDNN/flash
# backward. That is deliberate: the decomposed path has no base-pointer alignment
# assumption, so it does NOT inherit the CUDA_ERROR_MISALIGNED_ADDRESS crash that
# flame-core's cuDNN SDPA backward hits (see EriDiffusion
# BACKLOG_qwen_cudnn_sdpa_bwd_misalign.md + HANDOFF_2026-05-30_L2P_CUDNN_SDPA_BWD).
#
# ── Math (ported verbatim from flame-core attention_backward_recompute,
#    src/autograd.rs:1686) ──────────────────────────────────────────────────
#   recompute: attn = softmax(Q@Kᵀ * scale)                 [BH,S,S]
#   grad_v     = attnᵀ @ d_out                              [BH,S,D]
#   grad_attn  = d_out @ Vᵀ                                 [BH,S,S]
#   sum_ga     = rowsum(grad_attn * attn)                   [BH,S,1]
#   grad_scores= attn * (grad_attn - sum_ga)               [BH,S,S]  (softmax bwd)
#   grad_q     = (grad_scores @ K) * scale                  [BH,S,D]
#   grad_k     = (grad_scoresᵀ @ Q) * scale                 [BH,S,D]
# All interior math is F32 (matmuls accumulate F32; softmax/reduction F32).
# BF16/F16 only at the storage boundary (gather casts up, scatter casts down).
#
# Layout convention MATCHES the forward: q,k,v,d_out are BSHD [B,S,H,Dh]
# (storage dtype); returns (d_q, d_k, d_v) each BSHD [B,S,H,Dh] in q's dtype.
# Non-causal full attention (diffusion). No mask grad (forward mask is additive
# bias / all-zeros for the diffusion paths; mask is not a learnable input here).
#
# Mojo 1.0.0b1, NVIDIA GPU. Mirrors attention.mojo's gather/scatter scaffolding.

from std.math import exp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256
comptime _NEG_BIG = Float32(-3.0e38)


struct SdpaGrads(Movable):
    """Backward outputs of `sdpa_backward`: gradients wrt q, k, v (each BSHD)."""

    var d_q: Tensor
    var d_k: Tensor
    var d_v: Tensor

    def __init__(out self, var d_q: Tensor, var d_k: Tensor, var d_v: Tensor):
        self.d_q = d_q^
        self.d_k = d_k^
        self.d_v = d_v^


def _scratch_f32_flat(
    mut scratch: ScratchRingAllocator,
    n: Int,
    reverse: Bool = False,
) raises -> Tensor:
    var sh = List[Int]()
    sh.append(n)
    if reverse:
        return scratch.alloc_tensor_reverse(sh^, STDtype.F32)
    return scratch.alloc_tensor(sh^, STDtype.F32)


# ── gather BSHD [B,S,H,Dh] -> BHSD-contiguous F32 [B*H*S, Dh] (cast up) ──────
# Identical index math to attention.mojo's forward gather. dst flat index
# (((b*H+h)*S+s)*Dh+d); BSHD source offset (((b*S+s)*H+h)*Dh+d).
def _gather_f32(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    if idx < B * H * S * Dh:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        dst[(b * H + h) * S + s, d] = src[(b * S + s) * H + h, d]


def _gather_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    if idx < B * H * S * Dh:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        var val = rebind[Scalar[DType.bfloat16]](
            src[(b * S + s) * H + h, d]
        ).cast[DType.float32]()
        dst[(b * H + h) * S + s, d] = rebind[dst.element_type](val)


def _gather_f16(
    src: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    if idx < B * H * S * Dh:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        var val = rebind[Scalar[DType.float16]](
            src[(b * S + s) * H + h, d]
        ).cast[DType.float32]()
        dst[(b * H + h) * S + s, d] = rebind[dst.element_type](val)


# ── scatter BHSD-contiguous F32 [B*H*S, Dh] -> BSHD storage dtype ────────────
def _scatter_f32(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    if idx < B * H * S * Dh:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        dst[(b * S + s) * H + h, d] = src[(b * H + h) * S + s, d]


def _scatter_bf16(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    if idx < B * H * S * Dh:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        var val = rebind[Scalar[DType.float32]](src[(b * H + h) * S + s, d])
        dst[(b * S + s) * H + h, d] = rebind[dst.element_type](
            val.cast[DType.bfloat16]()
        )


def _scatter_f16(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    if idx < B * H * S * Dh:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        var val = rebind[Scalar[DType.float32]](src[(b * H + h) * S + s, d])
        dst[(b * S + s) * H + h, d] = rebind[dst.element_type](
            val.cast[DType.float16]()
        )


# ── scale a [rows, cols] F32 buffer in place ─────────────────────────────────
def _scale_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    scale: Float32, rows: Int, cols: Int,
):
    var idx = Int(global_idx.x)
    if idx < rows * cols:
        var r = idx // cols
        var c = idx % cols
        x[r, c] = rebind[x.element_type](
            rebind[Scalar[DType.float32]](x[r, c]) * scale
        )


# ── softmax over last dim, in place on F32 scores [BH*S, S] ──────────────────
# One block per row; shared-memory tree reductions in F32. Identical to the
# forward's _softmax_rows_f32 (attention.mojo) — the recompute must match the
# forward bit-for-bit in F32.
def _softmax_rows_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin], cols: Int,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lmax: Float32 = _NEG_BIG
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        if v > lmax:
            lmax = v
        c += _TPB
    shared[tid] = lmax
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            var a = shared[tid]
            var bb = shared[tid + active]
            shared[tid] = a if a > bb else bb
        barrier()
        active //= 2
    var rmax = shared[0]
    barrier()
    var lsum: Float32 = 0.0
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        lsum += exp(v - rmax)
        c += _TPB
    shared[tid] = lsum
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var rsum = shared[0]
    var inv = 1.0 / rsum
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        x[row, c] = rebind[x.element_type](exp(v - rmax) * inv)
        c += _TPB


# ── softmax backward, fused per row ──────────────────────────────────────────
# In:  attn [BH*S, S] (probabilities, F32), grad_attn [BH*S, S] (F32).
# Out: grad_scores written into grad_attn buffer in place:
#        sum_ga[r]  = sum_c attn[r,c]*grad_attn[r,c]
#        ds[r,c]    = attn[r,c] * (grad_attn[r,c] - sum_ga[r])
# One block per row; F32 tree-reduction for sum_ga (matches flame-core
# sum_dim_keepdim then attn*(grad_attn - sum_ga)).
def _softmax_bwd_rows_f32(
    attn: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    grad: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # in: grad_attn, out: grad_scores
    cols: Int,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var c = tid
    while c < cols:
        var a = rebind[Scalar[DType.float32]](attn[row, c])
        var g = rebind[Scalar[DType.float32]](grad[row, c])
        lsum += a * g
        c += _TPB
    shared[tid] = lsum
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var sum_ga = shared[0]
    barrier()
    c = tid
    while c < cols:
        var a = rebind[Scalar[DType.float32]](attn[row, c])
        var g = rebind[Scalar[DType.float32]](grad[row, c])
        grad[row, c] = rebind[grad.element_type](a * (g - sum_ga))
        c += _TPB


# ── gather a single BSHD tensor to a BHSD F32 buffer (dispatch on dtype) ──────
def _gather_to_f32(
    t: Tensor,
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
    src_rl: RuntimeLayout[_DYN2],
    ctx: DeviceContext,
) raises:
    var dt = t.dtype().to_mojo_dtype()
    var n = B * H * S * Dh
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var s = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            t.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        ctx.enqueue_function[_gather_f32, _gather_f32](
            s, dst, B, S, H, Dh, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var s = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            t.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        ctx.enqueue_function[_gather_bf16, _gather_bf16](
            s, dst, B, S, H, Dh, grid_dim=grid, block_dim=_BLOCK)
    else:
        var s = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            t.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        ctx.enqueue_function[_gather_f16, _gather_f16](
            s, dst, B, S, H, Dh, grid_dim=grid, block_dim=_BLOCK)


# ── scatter a BHSD F32 buffer to a fresh BSHD Tensor in `out_dt` ─────────────
def _scatter_to_tensor(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
    out_dt: STDtype,
    src_rl: RuntimeLayout[_DYN2],
    ctx: DeviceContext,
) raises -> Tensor:
    var dt = out_dt.to_mojo_dtype()
    var bsz = out_dt.byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * S * H * Dh * bsz)
    var n = B * H * S * Dh
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        ctx.enqueue_function[_scatter_f32, _scatter_f32](
            src, od, B, S, H, Dh, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var od = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        ctx.enqueue_function[_scatter_bf16, _scatter_bf16](
            src, od, B, S, H, Dh, grid_dim=grid, block_dim=_BLOCK)
    else:
        var od = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        ctx.enqueue_function[_scatter_f16, _scatter_f16](
            src, od, B, S, H, Dh, grid_dim=grid, block_dim=_BLOCK)
    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(S)
    out_shape.append(H)
    out_shape.append(Dh)
    return Tensor(out_buf^, out_shape^, out_dt)


# ── decomposed SDPA backward (any Dh) ────────────────────────────────────────
def sdpa_backward[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    d_out: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaGrads:
    """Decomposed (math-mode) SDPA backward for non-causal full attention.

    q, k, v, d_out: [B, S, H, Dh] BSHD row-major, same compute dtype.
    scale:          Float32 (the SAME 1/sqrt(Dh) used in the forward).
    returns SdpaGrads{d_q, d_k, d_v}, each [B, S, H, Dh] in q's dtype.

    Ports flame-core attention_backward_recompute (autograd.rs:1686): recompute
    attn = softmax(QKᵀ·scale), then d_v = attnᵀ@d_out, grad_attn = d_out@Vᵀ,
    softmax-bwd → grad_scores, d_q = (grad_scores@K)·scale,
    d_k = (grad_scoresᵀ@Q)·scale. All interior F32.
    """
    if q.dtype() != k.dtype() or q.dtype() != v.dtype() or q.dtype() != d_out.dtype():
        raise Error("sdpa_backward: q/k/v/d_out dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4 or qshape[0] != B or qshape[1] != S or qshape[2] != H or qshape[3] != Dh:
        raise Error("sdpa_backward: q shape != compile-time [B,S,H,Dh]")

    var out_dt = q.dtype()
    comptime BH = B * H
    comptime src_rows = B * S * H
    comptime bhsd_rows = B * H * S
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bhsd_rows, Dh))
    var head_qk_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, S))

    # ── 1) gather q,k,v,d_out BSHD -> BHSD F32 [B*H*S, Dh] ───────────────────
    var qf = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var kf = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var vf = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var gof = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var qd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](qf.unsafe_ptr(), bhsd_rl)
    var kd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](kf.unsafe_ptr(), bhsd_rl)
    var vd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](vf.unsafe_ptr(), bhsd_rl)
    var god = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gof.unsafe_ptr(), bhsd_rl)
    _gather_to_f32(q, qd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(k, kd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(v, vd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(d_out, god, B, S, H, Dh, src_rl, ctx)

    # ── 2) recompute attn = softmax(Q@Kᵀ * scale)  [BH,S,S] ─────────────────
    var attn = ctx.enqueue_create_buffer[DType.float32](BH * S * S)
    var qptr = qf.unsafe_ptr()
    var kptr = kf.unsafe_ptr()
    var vptr = vf.unsafe_ptr()
    var goptr = gof.unsafe_ptr()
    var aptr = attn.unsafe_ptr()
    for bh in range(BH):
        var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](qptr + bh * S * Dh, head_qk_rl)
        var Bt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](kptr + bh * S * Dh, head_qk_rl)
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr + bh * S * S, sc_rl)
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)  # Q@Kᵀ
    # scale + softmax over last dim (one block per [BH*S] row)
    comptime sm_rows = BH * S
    var sc_full_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, S))
    var attn_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr, sc_full_rl)
    var nsm = sm_rows * S
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        attn_full, scale, sm_rows, S, grid_dim=smgrid, block_dim=_BLOCK)
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        attn_full, S, grid_dim=sm_rows, block_dim=_TPB)

    # ── 3) d_v = attnᵀ @ d_out  [BH,S,Dh] ────────────────────────────────────
    var dvf = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var dvptr = dvf.unsafe_ptr()
    for bh in range(BH):
        var P = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr + bh * S * S, sc_rl)
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](goptr + bh * S * Dh, head_qk_rl)
        var DV = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr + bh * S * Dh, head_qk_rl)
        # DV[S,Dh] = Pᵀ[S,S] @ GO[S,Dh]  → transpose_a (attnᵀ)
        matmul(ctx, DV, P, GO, transpose_a=True, c_row_major=True)

    # ── 4) grad_attn = d_out @ Vᵀ  [BH,S,S] (reuse a fresh scores buffer) ─────
    var gscores = ctx.enqueue_create_buffer[DType.float32](BH * S * S)
    var gsptr = gscores.unsafe_ptr()
    for bh in range(BH):
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](goptr + bh * S * Dh, head_qk_rl)
        var Vh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](vptr + bh * S * Dh, head_qk_rl)
        var GA = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gsptr + bh * S * S, sc_rl)
        matmul(ctx, GA, GO, Vh, transpose_b=True, c_row_major=True)  # d_out @ Vᵀ

    # ── 5) softmax backward: grad_scores = attn*(grad_attn - rowsum(attn*grad_attn))
    var gs_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gsptr, sc_full_rl)
    ctx.enqueue_function[_softmax_bwd_rows_f32, _softmax_bwd_rows_f32](
        attn_full, gs_full, S, grid_dim=sm_rows, block_dim=_TPB)

    # ── 6) d_q = (grad_scores @ K) * scale ; d_k = (grad_scoresᵀ @ Q) * scale ─
    var dqf = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var dkf = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var dqptr = dqf.unsafe_ptr()
    var dkptr = dkf.unsafe_ptr()
    for bh in range(BH):
        var DS = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gsptr + bh * S * S, sc_rl)
        var Kh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](kptr + bh * S * Dh, head_qk_rl)
        var Qh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](qptr + bh * S * Dh, head_qk_rl)
        var DQ = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dqptr + bh * S * Dh, head_qk_rl)
        var DK = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dkptr + bh * S * Dh, head_qk_rl)
        matmul(ctx, DQ, DS, Kh, transpose_b=False, c_row_major=True)  # grad_scores @ K
        matmul(ctx, DK, DS, Qh, transpose_a=True, c_row_major=True)   # grad_scoresᵀ @ Q
    # scale d_q and d_k by `scale`
    var dq_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dqptr, bhsd_rl)
    var dk_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dkptr, bhsd_rl)
    var ndqk = bhsd_rows * Dh
    var dqkgrid = (ndqk + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        dq_full, scale, bhsd_rows, Dh, grid_dim=dqkgrid, block_dim=_BLOCK)
    ctx.enqueue_function[_scale_f32, _scale_f32](
        dk_full, scale, bhsd_rows, Dh, grid_dim=dqkgrid, block_dim=_BLOCK)

    # ── 7) scatter d_q,d_k,d_v BHSD F32 -> BSHD storage dtype ─────────────────
    var dq_t = _scatter_to_tensor(dq_full, B, S, H, Dh, out_dt, src_rl, ctx)
    var dk_t = _scatter_to_tensor(dk_full, B, S, H, Dh, out_dt, src_rl, ctx)
    var dv_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr, bhsd_rl)
    var dv_t = _scatter_to_tensor(dv_full, B, S, H, Dh, out_dt, src_rl, ctx)
    return SdpaGrads(dq_t^, dk_t^, dv_t^)


def sdpa_backward_scratch[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    d_out: Tensor,
    scale: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> SdpaGrads:
    """Scratch-backed SDPA backward.

    The returned d_q/d_k/d_v tensors are normal fresh outputs. The large
    recompute/work buffers are allocated from the caller-owned scratch ring and
    rewound before return, following the OneTrainer-style scoped cache pattern.
    """
    var scratch_mark = scratch.mark()
    if q.dtype() != k.dtype() or q.dtype() != v.dtype() or q.dtype() != d_out.dtype():
        raise Error("sdpa_backward_scratch: q/k/v/d_out dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4 or qshape[0] != B or qshape[1] != S or qshape[2] != H or qshape[3] != Dh:
        raise Error("sdpa_backward_scratch: q shape != compile-time [B,S,H,Dh]")

    var out_dt = q.dtype()
    comptime BH = B * H
    comptime src_rows = B * S * H
    comptime bhsd_rows = B * H * S
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bhsd_rows, Dh))
    var head_qk_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, S))

    # Inputs are long-lived through the recompute, so place them at the reverse
    # end of the ring and let forward allocations use the other end.
    var qf = _scratch_f32_flat(scratch, bhsd_rows * Dh, True)
    var kf = _scratch_f32_flat(scratch, bhsd_rows * Dh, True)
    var vf = _scratch_f32_flat(scratch, bhsd_rows * Dh, True)
    var gof = _scratch_f32_flat(scratch, bhsd_rows * Dh, True)
    var qd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        qf.buf.unsafe_ptr().bitcast[Float32](), bhsd_rl
    )
    var kd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        kf.buf.unsafe_ptr().bitcast[Float32](), bhsd_rl
    )
    var vd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        vf.buf.unsafe_ptr().bitcast[Float32](), bhsd_rl
    )
    var god = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        gof.buf.unsafe_ptr().bitcast[Float32](), bhsd_rl
    )
    _gather_to_f32(q, qd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(k, kd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(v, vd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(d_out, god, B, S, H, Dh, src_rl, ctx)

    var attn = _scratch_f32_flat(scratch, BH * S * S)
    var qptr = qf.buf.unsafe_ptr().bitcast[Float32]()
    var kptr = kf.buf.unsafe_ptr().bitcast[Float32]()
    var vptr = vf.buf.unsafe_ptr().bitcast[Float32]()
    var goptr = gof.buf.unsafe_ptr().bitcast[Float32]()
    var aptr = attn.buf.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            qptr + bh * S * Dh, head_qk_rl
        )
        var Bt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            kptr + bh * S * Dh, head_qk_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            aptr + bh * S * S, sc_rl
        )
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)

    comptime sm_rows = BH * S
    var sc_full_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, S))
    var attn_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr, sc_full_rl)
    var nsm = sm_rows * S
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        attn_full, scale, sm_rows, S, grid_dim=smgrid, block_dim=_BLOCK
    )
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        attn_full, S, grid_dim=sm_rows, block_dim=_TPB
    )

    var dvf = _scratch_f32_flat(scratch, bhsd_rows * Dh)
    var dvptr = dvf.buf.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var P = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            aptr + bh * S * S, sc_rl
        )
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            goptr + bh * S * Dh, head_qk_rl
        )
        var DV = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dvptr + bh * S * Dh, head_qk_rl
        )
        matmul(ctx, DV, P, GO, transpose_a=True, c_row_major=True)

    var gscores = _scratch_f32_flat(scratch, BH * S * S)
    var gsptr = gscores.buf.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            goptr + bh * S * Dh, head_qk_rl
        )
        var Vh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            vptr + bh * S * Dh, head_qk_rl
        )
        var GA = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            gsptr + bh * S * S, sc_rl
        )
        matmul(ctx, GA, GO, Vh, transpose_b=True, c_row_major=True)

    var gs_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gsptr, sc_full_rl)
    ctx.enqueue_function[_softmax_bwd_rows_f32, _softmax_bwd_rows_f32](
        attn_full, gs_full, S, grid_dim=sm_rows, block_dim=_TPB
    )

    var dqf = _scratch_f32_flat(scratch, bhsd_rows * Dh)
    var dkf = _scratch_f32_flat(scratch, bhsd_rows * Dh)
    var dqptr = dqf.buf.unsafe_ptr().bitcast[Float32]()
    var dkptr = dkf.buf.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var DS = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            gsptr + bh * S * S, sc_rl
        )
        var Kh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            kptr + bh * S * Dh, head_qk_rl
        )
        var Qh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            qptr + bh * S * Dh, head_qk_rl
        )
        var DQ = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dqptr + bh * S * Dh, head_qk_rl
        )
        var DK = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dkptr + bh * S * Dh, head_qk_rl
        )
        matmul(ctx, DQ, DS, Kh, transpose_b=False, c_row_major=True)
        matmul(ctx, DK, DS, Qh, transpose_a=True, c_row_major=True)

    var dq_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dqptr, bhsd_rl)
    var dk_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dkptr, bhsd_rl)
    var ndqk = bhsd_rows * Dh
    var dqkgrid = (ndqk + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        dq_full, scale, bhsd_rows, Dh, grid_dim=dqkgrid, block_dim=_BLOCK
    )
    ctx.enqueue_function[_scale_f32, _scale_f32](
        dk_full, scale, bhsd_rows, Dh, grid_dim=dqkgrid, block_dim=_BLOCK
    )

    var dq_t = _scatter_to_tensor(dq_full, B, S, H, Dh, out_dt, src_rl, ctx)
    var dk_t = _scatter_to_tensor(dk_full, B, S, H, Dh, out_dt, src_rl, ctx)
    var dv_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr, bhsd_rl)
    var dv_t = _scatter_to_tensor(dv_full, B, S, H, Dh, out_dt, src_rl, ctx)
    scratch.rewind(scratch_mark)
    return SdpaGrads(dq_t^, dk_t^, dv_t^)
