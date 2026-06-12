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
# Scores, softmax, reductions, and GEMM C/accumulator buffers are F32. Q/K/V
# and d_out staging preserve the input storage dtype; BF16/F16 are cast to F32
# only as scalar elements inside kernels or by BLAS accumulation.
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
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.autograd_v2.step_slab import StepSlab


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


# ── gather BSHD [B,S,H,Dh] -> BHSD-contiguous storage [B*H*S, Dh] ────────────
# Identical index math to attention.mojo's forward gather. dst flat index
# (((b*H+h)*S+s)*Dh+d); BSHD source offset (((b*S+s)*H+h)*Dh+d).
def _gather_storage[dtype: DType](
    src: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
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


# ── add an additive attention mask to F32 scores in place ────────────────────
# scores [BH*S, S] (BH = B*H); mask [H*S, S] F32 (per-head rows, broadcast
# over B). Row r of scores maps to mask row (r // S % H) * S + (r % S).
def _add_mask_rows_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    m: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    h: Int, s: Int, rows: Int,
):
    var idx = Int(global_idx.x)
    if idx < rows * s:
        var r = idx // s
        var c = idx % s
        var mr = (r // s % h) * s + r % s
        x[r, c] = rebind[x.element_type](
            rebind[Scalar[DType.float32]](x[r, c])
            + rebind[Scalar[DType.float32]](m[mr, c])
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


def _dv_from_attn_go_kernel[dtype: DType](
    attn: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [B*H*Sq, Skv]
    go: LayoutTensor[dtype, _DYN2, MutAnyOrigin],            # [B*H*Sq, Dh]
    dv: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],    # [B*H*Skv, Dh]
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Skv * Dh
    if idx < total:
        var d = idx % Dh
        var kvrow = idx // Dh
        var j = kvrow % Skv
        var bh = kvrow // Skv
        var qbase = bh * Sq
        var acc: Float32 = 0.0
        for i in range(Sq):
            var p = rebind[Scalar[DType.float32]](attn[qbase + i, j])
            var g = rebind[Scalar[dtype]](go[qbase + i, d]).cast[DType.float32]()
            acc += p * g
        dv[kvrow, d] = rebind[dv.element_type](acc)


def _grad_attn_from_go_v_kernel[dtype: DType](
    go: LayoutTensor[dtype, _DYN2, MutAnyOrigin],            # [B*H*Sq, Dh]
    v: LayoutTensor[dtype, _DYN2, MutAnyOrigin],             # [B*H*Skv, Dh]
    grad: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [B*H*Sq, Skv]
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Sq * Skv
    if idx < total:
        var j = idx % Skv
        var qrow = idx // Skv
        var bh = qrow // Sq
        var krow = bh * Skv + j
        var acc: Float32 = 0.0
        for d in range(Dh):
            var g = rebind[Scalar[dtype]](go[qrow, d]).cast[DType.float32]()
            var vv = rebind[Scalar[dtype]](v[krow, d]).cast[DType.float32]()
            acc += g * vv
        grad[qrow, j] = rebind[grad.element_type](acc)


def _dq_from_scores_k_kernel[dtype: DType](
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin], # [B*H*Sq, Skv]
    k: LayoutTensor[dtype, _DYN2, MutAnyOrigin],              # [B*H*Skv, Dh]
    dq: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],     # [B*H*Sq, Dh]
    scale: Float32,
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Sq * Dh
    if idx < total:
        var d = idx % Dh
        var qrow = idx // Dh
        var bh = qrow // Sq
        var kbase = bh * Skv
        var acc: Float32 = 0.0
        for j in range(Skv):
            var ds = rebind[Scalar[DType.float32]](scores[qrow, j])
            var kv = rebind[Scalar[dtype]](k[kbase + j, d]).cast[DType.float32]()
            acc += ds * kv
        dq[qrow, d] = rebind[dq.element_type](acc * scale)


def _dk_from_scores_q_kernel[dtype: DType](
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin], # [B*H*Sq, Skv]
    q: LayoutTensor[dtype, _DYN2, MutAnyOrigin],              # [B*H*Sq, Dh]
    dk: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],     # [B*H*Skv, Dh]
    scale: Float32,
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Skv * Dh
    if idx < total:
        var d = idx % Dh
        var kvrow = idx // Dh
        var j = kvrow % Skv
        var bh = kvrow // Skv
        var qbase = bh * Sq
        var acc: Float32 = 0.0
        for i in range(Sq):
            var ds = rebind[Scalar[DType.float32]](scores[qbase + i, j])
            var qv = rebind[Scalar[dtype]](q[qbase + i, d]).cast[DType.float32]()
            acc += ds * qv
        dk[kvrow, d] = rebind[dk.element_type](acc * scale)


# ── gather a single BSHD tensor to a BHSD F32 buffer (dispatch on dtype) ──────
def _gather_to_f32(
    t: Tensor,
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
    src_rl: RuntimeLayout[_DYN2],
    ctx: DeviceContext,
) raises:
    var dt = t.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var n = B * H * S * Dh
        var grid = (n + _BLOCK - 1) // _BLOCK
        var s = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            t.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        ctx.enqueue_function[
            _gather_storage[DType.float32], _gather_storage[DType.float32]
        ](
            s, dst, B, S, H, Dh, grid_dim=grid, block_dim=_BLOCK)
    else:
        raise Error("_gather_to_f32 is legacy F32-only; use storage gather")


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
    var dt0 = q.dtype().to_mojo_dtype()
    if dt0 != DType.float32:
        return sdpa_backward_rect[B, S, S, H, Dh](q, k, v, d_out, scale, ctx)

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


def sdpa_backward_masked[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask_f32: Tensor,
    d_out: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaGrads:
    """sdpa_backward SIBLING with an ADDITIVE attention mask (HiDream-O1
    prefix-causal training, 2026-06-11). Identical decomposed recompute with
    the mask added to scores AFTER scale, BEFORE softmax — exactly the
    forward's order (models/dit/hidream_o1.mojo _sdpa_s). mask_f32: F32
    [H*S, S] rows (per-head, broadcast over B; constant — no grad). bf16
    inputs take this same F32 interior path (gathers convert) instead of the
    rect fallback — the mask insert point only exists here."""
    if mask_f32.dtype().to_mojo_dtype() != DType.float32:
        raise Error("sdpa_backward_masked: mask must be F32")
    # bf16/f16 inputs: cast to F32 up front (the gather helpers are F32-only
    # — measured: "_gather_to_f32 is legacy F32-only" raise on the bf16
    # trainer path 2026-06-11). out_dt below keeps the ORIGINAL dtype so
    # grads scatter back to the caller's carrier.
    var orig_dt = q.dtype()
    var q_w: Tensor
    var k_w: Tensor
    var v_w: Tensor
    var do_w: Tensor
    if orig_dt.to_mojo_dtype() != DType.float32:
        q_w = cast_tensor(q, STDtype.F32, ctx)
        k_w = cast_tensor(k, STDtype.F32, ctx)
        v_w = cast_tensor(v, STDtype.F32, ctx)
        do_w = cast_tensor(d_out, STDtype.F32, ctx)
    else:
        # zero-copy re-box (Tensor is not implicitly copyable)
        q_w = Tensor(q.buf.copy(), q.shape(), q.dtype())
        k_w = Tensor(k.buf.copy(), k.shape(), k.dtype())
        v_w = Tensor(v.buf.copy(), v.shape(), v.dtype())
        do_w = Tensor(d_out.buf.copy(), d_out.shape(), d_out.dtype())
    if q.dtype() != k.dtype() or q.dtype() != v.dtype() or q.dtype() != d_out.dtype():
        raise Error("sdpa_backward_masked: q/k/v/d_out dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4 or qshape[0] != B or qshape[1] != S or qshape[2] != H or qshape[3] != Dh:
        raise Error("sdpa_backward_masked: q shape != compile-time [B,S,H,Dh]")
    var out_dt = orig_dt
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
    _gather_to_f32(q_w, qd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(k_w, kd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(v_w, vd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(do_w, god, B, S, H, Dh, src_rl, ctx)

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
    # additive mask (constant): scores += mask[h], the forward's exact order.
    var mask_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](H * S, S))
    var mask_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        mask_f32.buf.unsafe_ptr().bitcast[Float32](), mask_rl
    )
    ctx.enqueue_function[_add_mask_rows_f32, _add_mask_rows_f32](
        attn_full, mask_lt, H, S, sm_rows, grid_dim=smgrid, block_dim=_BLOCK)
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


# ── decomposed RECTANGULAR SDPA backward (S_q != S_kv, any Dh) ───────────────
# Shared asymmetric/cross-attention backward primitive (Tenet 1: build once,
# both Anima cross-attn [Sq=4096,Skv=256,Dh=128] and SDXL cross-attn
# [Sq=H·W,Skv=77,Dh=64] inherit it).
#
# This is a SIBLING of sdpa_backward, NOT an extension of it. The square
# sdpa_backward bakes a single `S` into every matmul/softmax layout via comptime
# params; there is no runtime-shape path to generalize without rewriting that
# comptime API and re-validating Klein/Z-Image/Ernie. A sibling keeps the square
# entry bit-identical (zero regression) while giving cross-attention its own
# correct, easy-to-call entry point (Tenet 2: the main rectangular entry is
# itself correct/fast; callers don't pick a "variant" of the square one).
#
# Math (same decomposition, S_q != S_kv shapes):
#   attn       = softmax_{Skv}(Q@Kᵀ · scale)               [BH, Sq, Skv]
#   d_v        = attnᵀ @ d_out                              [BH, Skv, Dh]
#   grad_attn  = d_out @ Vᵀ                                 [BH, Sq, Skv]
#   grad_scores= softmax_bwd(attn, grad_attn) over Skv      [BH, Sq, Skv]
#   d_q        = (grad_scores @ K)  · scale                 [BH, Sq, Dh]
#   d_k        = (grad_scoresᵀ @ Q) · scale                 [BH, Skv, Dh]
# All interior math F32; BF16/F16 only at the gather/scatter storage boundary.
# Non-causal, no mask grad (mask is additive bias / zero for diffusion paths).
def _sdpa_backward_rect_storage[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int, dtype: DType
](
    qs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    ks: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    vs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    gos: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    out_dt: STDtype,
    ctx: DeviceContext,
    scale: Float32,
) raises -> SdpaGrads:
    comptime BH = B * H
    comptime q_src_rows = B * Sq * H
    comptime kv_src_rows = B * Skv * H
    comptime q_bhsd_rows = B * H * Sq
    comptime kv_bhsd_rows = B * H * Skv

    var q_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_src_rows, Dh))
    var kv_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_src_rows, Dh))
    var q_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_bhsd_rows, Dh))
    var kv_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_bhsd_rows, Dh))
    var q_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Dh))
    var kv_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Skv, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Skv))

    var qbuf = ctx.enqueue_create_buffer[dtype](q_bhsd_rows * Dh)
    var kbuf = ctx.enqueue_create_buffer[dtype](kv_bhsd_rows * Dh)
    var vbuf = ctx.enqueue_create_buffer[dtype](kv_bhsd_rows * Dh)
    var gobuf = ctx.enqueue_create_buffer[dtype](q_bhsd_rows * Dh)
    var qd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](qbuf.unsafe_ptr(), q_bhsd_rl)
    var kd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](kbuf.unsafe_ptr(), kv_bhsd_rl)
    var vd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](vbuf.unsafe_ptr(), kv_bhsd_rl)
    var god = LayoutTensor[dtype, _DYN2, MutAnyOrigin](gobuf.unsafe_ptr(), q_bhsd_rl)
    var nq = q_bhsd_rows * Dh
    var nkv = kv_bhsd_rows * Dh
    var qgrid = (nq + _BLOCK - 1) // _BLOCK
    var kvgrid = (nkv + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _gather_storage[dtype], _gather_storage[dtype]
    ](qs, qd, B, Sq, H, Dh, grid_dim=qgrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_storage[dtype], _gather_storage[dtype]
    ](gos, god, B, Sq, H, Dh, grid_dim=qgrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_storage[dtype], _gather_storage[dtype]
    ](ks, kd, B, Skv, H, Dh, grid_dim=kvgrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_storage[dtype], _gather_storage[dtype]
    ](vs, vd, B, Skv, H, Dh, grid_dim=kvgrid, block_dim=_BLOCK)

    # Recompute attention probabilities into an F32 score/probability slab.
    var attn = ctx.enqueue_create_buffer[DType.float32](BH * Sq * Skv)
    var qptr = qbuf.unsafe_ptr()
    var kptr = kbuf.unsafe_ptr()
    var vptr = vbuf.unsafe_ptr()
    var goptr = gobuf.unsafe_ptr()
    var aptr = attn.unsafe_ptr()
    for bh in range(BH):
        var A = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            qptr + bh * Sq * Dh, q_head_rl
        )
        var Bt = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            kptr + bh * Skv * Dh, kv_head_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            aptr + bh * Sq * Skv, sc_rl
        )
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)

    comptime sm_rows = BH * Sq
    var sc_full_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, Skv))
    var attn_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        aptr, sc_full_rl
    )
    var nsm = sm_rows * Skv
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        attn_full, scale, sm_rows, Skv, grid_dim=smgrid, block_dim=_BLOCK)
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        attn_full, Skv, grid_dim=sm_rows, block_dim=_TPB)

    var dvf = ctx.enqueue_create_buffer[DType.float32](kv_bhsd_rows * Dh)
    var dvptr = dvf.unsafe_ptr()
    var go_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](goptr, q_bhsd_rl)
    var dv_full_k = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dvptr, kv_bhsd_rl
    )
    var dvgrid = (kv_bhsd_rows * Dh + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _dv_from_attn_go_kernel[dtype], _dv_from_attn_go_kernel[dtype]
    ](
        attn_full, go_full, dv_full_k, B, Sq, Skv, H, Dh,
        grid_dim=dvgrid, block_dim=_BLOCK,
    )

    var gscores = ctx.enqueue_create_buffer[DType.float32](BH * Sq * Skv)
    var gsptr = gscores.unsafe_ptr()
    var v_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](vptr, kv_bhsd_rl)
    var ga_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        gsptr, sc_full_rl
    )
    var gagrid = (BH * Sq * Skv + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _grad_attn_from_go_v_kernel[dtype],
        _grad_attn_from_go_v_kernel[dtype],
    ](
        go_full, v_full, ga_full, B, Sq, Skv, H, Dh,
        grid_dim=gagrid, block_dim=_BLOCK,
    )

    var gs_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        gsptr, sc_full_rl
    )
    ctx.enqueue_function[_softmax_bwd_rows_f32, _softmax_bwd_rows_f32](
        attn_full, gs_full, Skv, grid_dim=sm_rows, block_dim=_TPB)

    var dqf = ctx.enqueue_create_buffer[DType.float32](q_bhsd_rows * Dh)
    var dkf = ctx.enqueue_create_buffer[DType.float32](kv_bhsd_rows * Dh)
    var dqptr = dqf.unsafe_ptr()
    var dkptr = dkf.unsafe_ptr()
    var q_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](qptr, q_bhsd_rl)
    var k_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](kptr, kv_bhsd_rl)
    var dq_full_k = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dqptr, q_bhsd_rl
    )
    var dk_full_k = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dkptr, kv_bhsd_rl
    )
    var dqgrid = (q_bhsd_rows * Dh + _BLOCK - 1) // _BLOCK
    var dkgrid = (kv_bhsd_rows * Dh + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _dq_from_scores_k_kernel[dtype], _dq_from_scores_k_kernel[dtype]
    ](
        gs_full, k_full, dq_full_k, scale, B, Sq, Skv, H, Dh,
        grid_dim=dqgrid, block_dim=_BLOCK,
    )
    ctx.enqueue_function[
        _dk_from_scores_q_kernel[dtype], _dk_from_scores_q_kernel[dtype]
    ](
        gs_full, q_full, dk_full_k, scale, B, Sq, Skv, H, Dh,
        grid_dim=dkgrid, block_dim=_BLOCK,
    )

    var dq_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dqptr, q_bhsd_rl)
    var dk_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dkptr, kv_bhsd_rl)
    var dv_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr, kv_bhsd_rl)
    var dq_t = _scatter_to_tensor(dq_full, B, Sq, H, Dh, out_dt, q_src_rl, ctx)
    var dk_t = _scatter_to_tensor(dk_full, B, Skv, H, Dh, out_dt, kv_src_rl, ctx)
    var dv_t = _scatter_to_tensor(dv_full, B, Skv, H, Dh, out_dt, kv_src_rl, ctx)
    return SdpaGrads(dq_t^, dk_t^, dv_t^)


def sdpa_backward_rect[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor,      # [B, Sq,  H, Dh]
    k: Tensor,      # [B, Skv, H, Dh]
    v: Tensor,      # [B, Skv, H, Dh]
    d_out: Tensor,  # [B, Sq,  H, Dh]
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaGrads:
    """Decomposed (math-mode) RECTANGULAR SDPA backward (S_q != S_kv).

    q, d_out: [B, Sq, H, Dh]; k, v: [B, Skv, H, Dh] (BSHD row-major, same dtype).
    scale:    Float32 (the SAME 1/sqrt(Dh) used in the forward).
    returns SdpaGrads{d_q [B,Sq,H,Dh], d_k [B,Skv,H,Dh], d_v [B,Skv,H,Dh]} in
    q's dtype. Self-attention (Sq==Skv) also works through here.
    """
    if q.dtype() != k.dtype() or q.dtype() != v.dtype() or q.dtype() != d_out.dtype():
        raise Error("sdpa_backward_rect: q/k/v/d_out dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4 or qshape[0] != B or qshape[1] != Sq or qshape[2] != H or qshape[3] != Dh:
        raise Error("sdpa_backward_rect: q shape != [B,Sq,H,Dh]")
    var kshape = k.shape()
    if len(kshape) != 4 or kshape[0] != B or kshape[1] != Skv or kshape[2] != H or kshape[3] != Dh:
        raise Error("sdpa_backward_rect: k shape != [B,Skv,H,Dh]")
    var vshape = v.shape()
    if len(vshape) != 4 or vshape[0] != B or vshape[1] != Skv or vshape[2] != H or vshape[3] != Dh:
        raise Error("sdpa_backward_rect: v shape != [B,Skv,H,Dh]")
    var doshape = d_out.shape()
    if len(doshape) != 4 or doshape[0] != B or doshape[1] != Sq or doshape[2] != H or doshape[3] != Dh:
        raise Error("sdpa_backward_rect: d_out shape != [B,Sq,H,Dh]")

    var out_dt = q.dtype()
    comptime BH = B * H
    comptime q_src_rows = B * Sq * H        # BSHD row count for q-side tensors
    comptime kv_src_rows = B * Skv * H      # BSHD row count for kv-side tensors
    comptime q_bhsd_rows = B * H * Sq       # BHSD-contig row count, q side
    comptime kv_bhsd_rows = B * H * Skv     # BHSD-contig row count, kv side

    # The shared gather/scatter helpers take B,S,H,Dh as RUNTIME args, so the
    # same kernels serve both Sq-length (q, d_out, d_q) and Skv-length (k, v,
    # d_k, d_v) tensors; only the RuntimeLayout row counts differ.
    var q_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_src_rows, Dh))
    var kv_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_src_rows, Dh))
    var q_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_bhsd_rows, Dh))
    var kv_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_bhsd_rows, Dh))
    var q_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Dh))   # [Sq,Dh]
    var kv_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Skv, Dh)) # [Skv,Dh]
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Skv))      # [Sq,Skv]
    var dt0 = q.dtype().to_mojo_dtype()
    if dt0 == DType.bfloat16:
        var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), q_src_rl)
        var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), kv_src_rl)
        var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), kv_src_rl)
        var gos = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            d_out.buf.unsafe_ptr().bitcast[BFloat16](), q_src_rl)
        return _sdpa_backward_rect_storage[
            B, Sq, Skv, H, Dh, DType.bfloat16
        ](qs, ks, vs, gos, out_dt, ctx, scale)
    elif dt0 == DType.float16:
        var qs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), q_src_rl)
        var ks = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), kv_src_rl)
        var vs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), kv_src_rl)
        var gos = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            d_out.buf.unsafe_ptr().bitcast[Float16](), q_src_rl)
        return _sdpa_backward_rect_storage[
            B, Sq, Skv, H, Dh, DType.float16
        ](qs, ks, vs, gos, out_dt, ctx, scale)

    # ── 1) gather q,d_out (Sq) and k,v (Skv) BSHD -> BHSD-contig F32 ─────────
    var qf = ctx.enqueue_create_buffer[DType.float32](q_bhsd_rows * Dh)
    var kf = ctx.enqueue_create_buffer[DType.float32](kv_bhsd_rows * Dh)
    var vf = ctx.enqueue_create_buffer[DType.float32](kv_bhsd_rows * Dh)
    var gof = ctx.enqueue_create_buffer[DType.float32](q_bhsd_rows * Dh)
    var qd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](qf.unsafe_ptr(), q_bhsd_rl)
    var kd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](kf.unsafe_ptr(), kv_bhsd_rl)
    var vd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](vf.unsafe_ptr(), kv_bhsd_rl)
    var god = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gof.unsafe_ptr(), q_bhsd_rl)
    _gather_to_f32(q, qd, B, Sq, H, Dh, q_src_rl, ctx)
    _gather_to_f32(k, kd, B, Skv, H, Dh, kv_src_rl, ctx)
    _gather_to_f32(v, vd, B, Skv, H, Dh, kv_src_rl, ctx)
    _gather_to_f32(d_out, god, B, Sq, H, Dh, q_src_rl, ctx)

    # ── 2) recompute attn = softmax_{Skv}(Q@Kᵀ * scale)  [BH,Sq,Skv] ────────
    var attn = ctx.enqueue_create_buffer[DType.float32](BH * Sq * Skv)
    var qptr = qf.unsafe_ptr()
    var kptr = kf.unsafe_ptr()
    var vptr = vf.unsafe_ptr()
    var goptr = gof.unsafe_ptr()
    var aptr = attn.unsafe_ptr()
    for bh in range(BH):
        var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](qptr + bh * Sq * Dh, q_head_rl)
        var Bt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](kptr + bh * Skv * Dh, kv_head_rl)
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr + bh * Sq * Skv, sc_rl)
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)  # Q@Kᵀ -> [Sq,Skv]
    # scale + softmax over last dim Skv (one block per [BH*Sq] row)
    comptime sm_rows = BH * Sq
    var sc_full_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, Skv))
    var attn_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr, sc_full_rl)
    var nsm = sm_rows * Skv
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        attn_full, scale, sm_rows, Skv, grid_dim=smgrid, block_dim=_BLOCK)
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        attn_full, Skv, grid_dim=sm_rows, block_dim=_TPB)

    # ── 3) d_v = attnᵀ @ d_out  [BH,Skv,Dh] ─────────────────────────────────
    var dvf = ctx.enqueue_create_buffer[DType.float32](kv_bhsd_rows * Dh)
    var dvptr = dvf.unsafe_ptr()
    for bh in range(BH):
        var P = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr + bh * Sq * Skv, sc_rl)
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](goptr + bh * Sq * Dh, q_head_rl)
        var DV = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr + bh * Skv * Dh, kv_head_rl)
        # DV[Skv,Dh] = Pᵀ[Skv,Sq] @ GO[Sq,Dh]  → transpose_a (attnᵀ)
        matmul(ctx, DV, P, GO, transpose_a=True, c_row_major=True)

    # ── 4) grad_attn = d_out @ Vᵀ  [BH,Sq,Skv] (fresh scores buffer) ────────
    var gscores = ctx.enqueue_create_buffer[DType.float32](BH * Sq * Skv)
    var gsptr = gscores.unsafe_ptr()
    for bh in range(BH):
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](goptr + bh * Sq * Dh, q_head_rl)
        var Vh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](vptr + bh * Skv * Dh, kv_head_rl)
        var GA = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gsptr + bh * Sq * Skv, sc_rl)
        # GA[Sq,Skv] = GO[Sq,Dh] @ Vh[Skv,Dh]ᵀ
        matmul(ctx, GA, GO, Vh, transpose_b=True, c_row_major=True)

    # ── 5) softmax backward over Skv: grad_scores = attn*(grad_attn - rowsum) ─
    var gs_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gsptr, sc_full_rl)
    ctx.enqueue_function[_softmax_bwd_rows_f32, _softmax_bwd_rows_f32](
        attn_full, gs_full, Skv, grid_dim=sm_rows, block_dim=_TPB)

    # ── 6) d_q = (grad_scores @ K)·scale [BH,Sq,Dh] ;
    #        d_k = (grad_scoresᵀ @ Q)·scale [BH,Skv,Dh] ──────────────────────
    var dqf = ctx.enqueue_create_buffer[DType.float32](q_bhsd_rows * Dh)
    var dkf = ctx.enqueue_create_buffer[DType.float32](kv_bhsd_rows * Dh)
    var dqptr = dqf.unsafe_ptr()
    var dkptr = dkf.unsafe_ptr()
    for bh in range(BH):
        var DS = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gsptr + bh * Sq * Skv, sc_rl)
        var Kh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](kptr + bh * Skv * Dh, kv_head_rl)
        var Qh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](qptr + bh * Sq * Dh, q_head_rl)
        var DQ = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dqptr + bh * Sq * Dh, q_head_rl)
        var DK = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dkptr + bh * Skv * Dh, kv_head_rl)
        matmul(ctx, DQ, DS, Kh, transpose_b=False, c_row_major=True)  # [Sq,Skv]@[Skv,Dh]
        matmul(ctx, DK, DS, Qh, transpose_a=True, c_row_major=True)   # [Skv,Sq]@[Sq,Dh]
    # scale d_q and d_k by `scale`
    var dq_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dqptr, q_bhsd_rl)
    var dk_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dkptr, kv_bhsd_rl)
    var ndq = q_bhsd_rows * Dh
    var ndk = kv_bhsd_rows * Dh
    var dqgrid = (ndq + _BLOCK - 1) // _BLOCK
    var dkgrid = (ndk + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        dq_full, scale, q_bhsd_rows, Dh, grid_dim=dqgrid, block_dim=_BLOCK)
    ctx.enqueue_function[_scale_f32, _scale_f32](
        dk_full, scale, kv_bhsd_rows, Dh, grid_dim=dkgrid, block_dim=_BLOCK)

    # ── 7) scatter BHSD F32 -> BSHD storage dtype (Sq for d_q, Skv for d_k/d_v)
    var dq_t = _scatter_to_tensor(dq_full, B, Sq, H, Dh, out_dt, q_src_rl, ctx)
    var dk_t = _scatter_to_tensor(dk_full, B, Skv, H, Dh, out_dt, kv_src_rl, ctx)
    var dv_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr, kv_bhsd_rl)
    var dv_t = _scatter_to_tensor(dv_full, B, Skv, H, Dh, out_dt, kv_src_rl, ctx)
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
    if q.dtype().to_mojo_dtype() != DType.float32:
        scratch.rewind(scratch_mark)
        return sdpa_backward[B, S, H, Dh](q, k, v, d_out, scale, ctx)

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


# ─────────────────────────────────────────────────────────────────────────────
# StepSlab variants (autograd_v2 contract C8, Phase P4): byte-identical math
# to the originals above — same kernels, same launch params, same per-head
# matmul loops; ONLY the allocation source changes (typed MAX buffers become
# uint8 slab views bitcast to the same element pointers).
# ─────────────────────────────────────────────────────────────────────────────


def _scatter_to_tensor_slab(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
    out_dt: STDtype,
    src_rl: RuntimeLayout[_DYN2],
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `_scatter_to_tensor` (this file :373)."""
    var dt = out_dt.to_mojo_dtype()
    var bsz = out_dt.byte_size()
    var out_buf = slab.alloc(B * S * H * Dh * bsz)
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


def _sdpa_backward_rect_storage_slab[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int, dtype: DType
](
    qs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    ks: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    vs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    gos: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    out_dt: STDtype,
    ctx: DeviceContext,
    scale: Float32,
    mut slab: StepSlab,
) raises -> SdpaGrads:
    """StepSlab variant of `_sdpa_backward_rect_storage` (this file :565)."""
    comptime BH = B * H
    comptime q_src_rows = B * Sq * H
    comptime kv_src_rows = B * Skv * H
    comptime q_bhsd_rows = B * H * Sq
    comptime kv_bhsd_rows = B * H * Skv
    var esz = 4
    comptime if dtype == DType.bfloat16 or dtype == DType.float16:
        esz = 2

    var q_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_src_rows, Dh))
    var kv_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_src_rows, Dh))
    var q_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_bhsd_rows, Dh))
    var kv_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_bhsd_rows, Dh))
    var q_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Dh))
    var kv_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Skv, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Skv))

    var qbuf = slab.alloc(q_bhsd_rows * Dh * esz)
    var kbuf = slab.alloc(kv_bhsd_rows * Dh * esz)
    var vbuf = slab.alloc(kv_bhsd_rows * Dh * esz)
    var gobuf = slab.alloc(q_bhsd_rows * Dh * esz)
    var qd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
        qbuf.unsafe_ptr().bitcast[Scalar[dtype]](), q_bhsd_rl)
    var kd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
        kbuf.unsafe_ptr().bitcast[Scalar[dtype]](), kv_bhsd_rl)
    var vd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
        vbuf.unsafe_ptr().bitcast[Scalar[dtype]](), kv_bhsd_rl)
    var god = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
        gobuf.unsafe_ptr().bitcast[Scalar[dtype]](), q_bhsd_rl)
    var nq = q_bhsd_rows * Dh
    var nkv = kv_bhsd_rows * Dh
    var qgrid = (nq + _BLOCK - 1) // _BLOCK
    var kvgrid = (nkv + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _gather_storage[dtype], _gather_storage[dtype]
    ](qs, qd, B, Sq, H, Dh, grid_dim=qgrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_storage[dtype], _gather_storage[dtype]
    ](gos, god, B, Sq, H, Dh, grid_dim=qgrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_storage[dtype], _gather_storage[dtype]
    ](ks, kd, B, Skv, H, Dh, grid_dim=kvgrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_storage[dtype], _gather_storage[dtype]
    ](vs, vd, B, Skv, H, Dh, grid_dim=kvgrid, block_dim=_BLOCK)

    # Recompute attention probabilities into an F32 score/probability slab.
    var attn = slab.alloc(BH * Sq * Skv * 4)
    var qptr = qbuf.unsafe_ptr().bitcast[Scalar[dtype]]()
    var kptr = kbuf.unsafe_ptr().bitcast[Scalar[dtype]]()
    var vptr = vbuf.unsafe_ptr().bitcast[Scalar[dtype]]()
    var goptr = gobuf.unsafe_ptr().bitcast[Scalar[dtype]]()
    var aptr = attn.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var A = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            qptr + bh * Sq * Dh, q_head_rl
        )
        var Bt = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            kptr + bh * Skv * Dh, kv_head_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            aptr + bh * Sq * Skv, sc_rl
        )
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)

    comptime sm_rows = BH * Sq
    var sc_full_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, Skv))
    var attn_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        aptr, sc_full_rl
    )
    var nsm = sm_rows * Skv
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        attn_full, scale, sm_rows, Skv, grid_dim=smgrid, block_dim=_BLOCK)
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        attn_full, Skv, grid_dim=sm_rows, block_dim=_TPB)

    var dvf = slab.alloc(kv_bhsd_rows * Dh * 4)
    var dvptr = dvf.unsafe_ptr().bitcast[Float32]()
    var go_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](goptr, q_bhsd_rl)
    var dv_full_k = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dvptr, kv_bhsd_rl
    )
    var dvgrid = (kv_bhsd_rows * Dh + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _dv_from_attn_go_kernel[dtype], _dv_from_attn_go_kernel[dtype]
    ](
        attn_full, go_full, dv_full_k, B, Sq, Skv, H, Dh,
        grid_dim=dvgrid, block_dim=_BLOCK,
    )

    var gscores = slab.alloc(BH * Sq * Skv * 4)
    var gsptr = gscores.unsafe_ptr().bitcast[Float32]()
    var v_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](vptr, kv_bhsd_rl)
    var ga_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        gsptr, sc_full_rl
    )
    var gagrid = (BH * Sq * Skv + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _grad_attn_from_go_v_kernel[dtype],
        _grad_attn_from_go_v_kernel[dtype],
    ](
        go_full, v_full, ga_full, B, Sq, Skv, H, Dh,
        grid_dim=gagrid, block_dim=_BLOCK,
    )

    var gs_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        gsptr, sc_full_rl
    )
    ctx.enqueue_function[_softmax_bwd_rows_f32, _softmax_bwd_rows_f32](
        attn_full, gs_full, Skv, grid_dim=sm_rows, block_dim=_TPB)

    var dqf = slab.alloc(q_bhsd_rows * Dh * 4)
    var dkf = slab.alloc(kv_bhsd_rows * Dh * 4)
    var dqptr = dqf.unsafe_ptr().bitcast[Float32]()
    var dkptr = dkf.unsafe_ptr().bitcast[Float32]()
    var q_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](qptr, q_bhsd_rl)
    var k_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](kptr, kv_bhsd_rl)
    var dq_full_k = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dqptr, q_bhsd_rl
    )
    var dk_full_k = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dkptr, kv_bhsd_rl
    )
    var dqgrid = (q_bhsd_rows * Dh + _BLOCK - 1) // _BLOCK
    var dkgrid = (kv_bhsd_rows * Dh + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _dq_from_scores_k_kernel[dtype], _dq_from_scores_k_kernel[dtype]
    ](
        gs_full, k_full, dq_full_k, scale, B, Sq, Skv, H, Dh,
        grid_dim=dqgrid, block_dim=_BLOCK,
    )
    ctx.enqueue_function[
        _dk_from_scores_q_kernel[dtype], _dk_from_scores_q_kernel[dtype]
    ](
        gs_full, q_full, dk_full_k, scale, B, Sq, Skv, H, Dh,
        grid_dim=dkgrid, block_dim=_BLOCK,
    )

    var dq_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dqptr, q_bhsd_rl)
    var dk_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dkptr, kv_bhsd_rl)
    var dv_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr, kv_bhsd_rl)
    var dq_t = _scatter_to_tensor_slab(dq_full, B, Sq, H, Dh, out_dt, q_src_rl, ctx, slab)
    var dk_t = _scatter_to_tensor_slab(dk_full, B, Skv, H, Dh, out_dt, kv_src_rl, ctx, slab)
    var dv_t = _scatter_to_tensor_slab(dv_full, B, Skv, H, Dh, out_dt, kv_src_rl, ctx, slab)
    return SdpaGrads(dq_t^, dk_t^, dv_t^)


def sdpa_backward_rect_slab[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor,      # [B, Sq,  H, Dh]
    k: Tensor,      # [B, Skv, H, Dh]
    v: Tensor,      # [B, Skv, H, Dh]
    d_out: Tensor,  # [B, Sq,  H, Dh]
    scale: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> SdpaGrads:
    """StepSlab variant of `sdpa_backward_rect` (this file :717) —
    byte-identical math; ONLY the allocation source changes."""
    if q.dtype() != k.dtype() or q.dtype() != v.dtype() or q.dtype() != d_out.dtype():
        raise Error("sdpa_backward_rect: q/k/v/d_out dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4 or qshape[0] != B or qshape[1] != Sq or qshape[2] != H or qshape[3] != Dh:
        raise Error("sdpa_backward_rect: q shape != [B,Sq,H,Dh]")
    var kshape = k.shape()
    if len(kshape) != 4 or kshape[0] != B or kshape[1] != Skv or kshape[2] != H or kshape[3] != Dh:
        raise Error("sdpa_backward_rect: k shape != [B,Skv,H,Dh]")
    var vshape = v.shape()
    if len(vshape) != 4 or vshape[0] != B or vshape[1] != Skv or vshape[2] != H or vshape[3] != Dh:
        raise Error("sdpa_backward_rect: v shape != [B,Skv,H,Dh]")
    var doshape = d_out.shape()
    if len(doshape) != 4 or doshape[0] != B or doshape[1] != Sq or doshape[2] != H or doshape[3] != Dh:
        raise Error("sdpa_backward_rect: d_out shape != [B,Sq,H,Dh]")

    var out_dt = q.dtype()
    comptime BH = B * H
    comptime q_src_rows = B * Sq * H
    comptime kv_src_rows = B * Skv * H
    comptime q_bhsd_rows = B * H * Sq
    comptime kv_bhsd_rows = B * H * Skv

    var q_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_src_rows, Dh))
    var kv_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_src_rows, Dh))
    var q_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_bhsd_rows, Dh))
    var kv_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_bhsd_rows, Dh))
    var q_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Dh))
    var kv_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Skv, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Skv))
    var dt0 = q.dtype().to_mojo_dtype()
    if dt0 == DType.bfloat16:
        var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), q_src_rl)
        var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), kv_src_rl)
        var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), kv_src_rl)
        var gos = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            d_out.buf.unsafe_ptr().bitcast[BFloat16](), q_src_rl)
        return _sdpa_backward_rect_storage_slab[
            B, Sq, Skv, H, Dh, DType.bfloat16
        ](qs, ks, vs, gos, out_dt, ctx, scale, slab)
    elif dt0 == DType.float16:
        var qs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), q_src_rl)
        var ks = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), kv_src_rl)
        var vs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), kv_src_rl)
        var gos = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            d_out.buf.unsafe_ptr().bitcast[Float16](), q_src_rl)
        return _sdpa_backward_rect_storage_slab[
            B, Sq, Skv, H, Dh, DType.float16
        ](qs, ks, vs, gos, out_dt, ctx, scale, slab)

    # ── 1) gather q,d_out (Sq) and k,v (Skv) BSHD -> BHSD-contig F32 ─────────
    var qf = slab.alloc(q_bhsd_rows * Dh * 4)
    var kf = slab.alloc(kv_bhsd_rows * Dh * 4)
    var vf = slab.alloc(kv_bhsd_rows * Dh * 4)
    var gof = slab.alloc(q_bhsd_rows * Dh * 4)
    var qd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](qf.unsafe_ptr().bitcast[Float32](), q_bhsd_rl)
    var kd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](kf.unsafe_ptr().bitcast[Float32](), kv_bhsd_rl)
    var vd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](vf.unsafe_ptr().bitcast[Float32](), kv_bhsd_rl)
    var god = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gof.unsafe_ptr().bitcast[Float32](), q_bhsd_rl)
    _gather_to_f32(q, qd, B, Sq, H, Dh, q_src_rl, ctx)
    _gather_to_f32(k, kd, B, Skv, H, Dh, kv_src_rl, ctx)
    _gather_to_f32(v, vd, B, Skv, H, Dh, kv_src_rl, ctx)
    _gather_to_f32(d_out, god, B, Sq, H, Dh, q_src_rl, ctx)

    # ── 2) recompute attn = softmax_{Skv}(Q@Kᵀ * scale)  [BH,Sq,Skv] ────────
    var attn = slab.alloc(BH * Sq * Skv * 4)
    var qptr = qf.unsafe_ptr().bitcast[Float32]()
    var kptr = kf.unsafe_ptr().bitcast[Float32]()
    var vptr = vf.unsafe_ptr().bitcast[Float32]()
    var goptr = gof.unsafe_ptr().bitcast[Float32]()
    var aptr = attn.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](qptr + bh * Sq * Dh, q_head_rl)
        var Bt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](kptr + bh * Skv * Dh, kv_head_rl)
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr + bh * Sq * Skv, sc_rl)
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)  # Q@Kᵀ -> [Sq,Skv]
    # scale + softmax over last dim Skv (one block per [BH*Sq] row)
    comptime sm_rows = BH * Sq
    var sc_full_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, Skv))
    var attn_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr, sc_full_rl)
    var nsm = sm_rows * Skv
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        attn_full, scale, sm_rows, Skv, grid_dim=smgrid, block_dim=_BLOCK)
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        attn_full, Skv, grid_dim=sm_rows, block_dim=_TPB)

    # ── 3) d_v = attnᵀ @ d_out  [BH,Skv,Dh] ─────────────────────────────────
    var dvf = slab.alloc(kv_bhsd_rows * Dh * 4)
    var dvptr = dvf.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var P = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr + bh * Sq * Skv, sc_rl)
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](goptr + bh * Sq * Dh, q_head_rl)
        var DV = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr + bh * Skv * Dh, kv_head_rl)
        # DV[Skv,Dh] = Pᵀ[Skv,Sq] @ GO[Sq,Dh]  → transpose_a (attnᵀ)
        matmul(ctx, DV, P, GO, transpose_a=True, c_row_major=True)

    # ── 4) grad_attn = d_out @ Vᵀ  [BH,Sq,Skv] (fresh scores buffer) ────────
    var gscores = slab.alloc(BH * Sq * Skv * 4)
    var gsptr = gscores.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](goptr + bh * Sq * Dh, q_head_rl)
        var Vh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](vptr + bh * Skv * Dh, kv_head_rl)
        var GA = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gsptr + bh * Sq * Skv, sc_rl)
        # GA[Sq,Skv] = GO[Sq,Dh] @ Vh[Skv,Dh]ᵀ
        matmul(ctx, GA, GO, Vh, transpose_b=True, c_row_major=True)

    # ── 5) softmax backward over Skv: grad_scores = attn*(grad_attn - rowsum) ─
    var gs_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gsptr, sc_full_rl)
    ctx.enqueue_function[_softmax_bwd_rows_f32, _softmax_bwd_rows_f32](
        attn_full, gs_full, Skv, grid_dim=sm_rows, block_dim=_TPB)

    # ── 6) d_q = (grad_scores @ K)·scale [BH,Sq,Dh] ;
    #        d_k = (grad_scoresᵀ @ Q)·scale [BH,Skv,Dh] ──────────────────────
    var dqf = slab.alloc(q_bhsd_rows * Dh * 4)
    var dkf = slab.alloc(kv_bhsd_rows * Dh * 4)
    var dqptr = dqf.unsafe_ptr().bitcast[Float32]()
    var dkptr = dkf.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var DS = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gsptr + bh * Sq * Skv, sc_rl)
        var Kh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](kptr + bh * Skv * Dh, kv_head_rl)
        var Qh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](qptr + bh * Sq * Dh, q_head_rl)
        var DQ = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dqptr + bh * Sq * Dh, q_head_rl)
        var DK = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dkptr + bh * Skv * Dh, kv_head_rl)
        matmul(ctx, DQ, DS, Kh, transpose_b=False, c_row_major=True)  # [Sq,Skv]@[Skv,Dh]
        matmul(ctx, DK, DS, Qh, transpose_a=True, c_row_major=True)   # [Skv,Sq]@[Sq,Dh]
    # scale d_q and d_k by `scale`
    var dq_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dqptr, q_bhsd_rl)
    var dk_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dkptr, kv_bhsd_rl)
    var ndq = q_bhsd_rows * Dh
    var ndk = kv_bhsd_rows * Dh
    var dqgrid = (ndq + _BLOCK - 1) // _BLOCK
    var dkgrid = (ndk + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        dq_full, scale, q_bhsd_rows, Dh, grid_dim=dqgrid, block_dim=_BLOCK)
    ctx.enqueue_function[_scale_f32, _scale_f32](
        dk_full, scale, kv_bhsd_rows, Dh, grid_dim=dkgrid, block_dim=_BLOCK)

    # ── 7) scatter BHSD F32 -> BSHD storage dtype (Sq for d_q, Skv for d_k/d_v)
    var dq_t = _scatter_to_tensor_slab(dq_full, B, Sq, H, Dh, out_dt, q_src_rl, ctx, slab)
    var dk_t = _scatter_to_tensor_slab(dk_full, B, Skv, H, Dh, out_dt, kv_src_rl, ctx, slab)
    var dv_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr, kv_bhsd_rl)
    var dv_t = _scatter_to_tensor_slab(dv_full, B, Skv, H, Dh, out_dt, kv_src_rl, ctx, slab)
    return SdpaGrads(dq_t^, dk_t^, dv_t^)


def sdpa_backward_slab[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    d_out: Tensor,
    scale: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> SdpaGrads:
    """StepSlab variant of `sdpa_backward` (this file :412) — byte-identical
    math; ONLY the allocation source changes (contract C8, Phase P4)."""
    if q.dtype() != k.dtype() or q.dtype() != v.dtype() or q.dtype() != d_out.dtype():
        raise Error("sdpa_backward: q/k/v/d_out dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4 or qshape[0] != B or qshape[1] != S or qshape[2] != H or qshape[3] != Dh:
        raise Error("sdpa_backward: q shape != compile-time [B,S,H,Dh]")
    var dt0 = q.dtype().to_mojo_dtype()
    if dt0 != DType.float32:
        return sdpa_backward_rect_slab[B, S, S, H, Dh](q, k, v, d_out, scale, ctx, slab)

    var out_dt = q.dtype()
    comptime BH = B * H
    comptime src_rows = B * S * H
    comptime bhsd_rows = B * H * S
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bhsd_rows, Dh))
    var head_qk_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, S))

    # ── 1) gather q,k,v,d_out BSHD -> BHSD F32 [B*H*S, Dh] ───────────────────
    var qf = slab.alloc(bhsd_rows * Dh * 4)
    var kf = slab.alloc(bhsd_rows * Dh * 4)
    var vf = slab.alloc(bhsd_rows * Dh * 4)
    var gof = slab.alloc(bhsd_rows * Dh * 4)
    var qd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](qf.unsafe_ptr().bitcast[Float32](), bhsd_rl)
    var kd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](kf.unsafe_ptr().bitcast[Float32](), bhsd_rl)
    var vd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](vf.unsafe_ptr().bitcast[Float32](), bhsd_rl)
    var god = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gof.unsafe_ptr().bitcast[Float32](), bhsd_rl)
    _gather_to_f32(q, qd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(k, kd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(v, vd, B, S, H, Dh, src_rl, ctx)
    _gather_to_f32(d_out, god, B, S, H, Dh, src_rl, ctx)

    # ── 2) recompute attn = softmax(Q@Kᵀ * scale)  [BH,S,S] ─────────────────
    var attn = slab.alloc(BH * S * S * 4)
    var qptr = qf.unsafe_ptr().bitcast[Float32]()
    var kptr = kf.unsafe_ptr().bitcast[Float32]()
    var vptr = vf.unsafe_ptr().bitcast[Float32]()
    var goptr = gof.unsafe_ptr().bitcast[Float32]()
    var aptr = attn.unsafe_ptr().bitcast[Float32]()
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
    var dvf = slab.alloc(bhsd_rows * Dh * 4)
    var dvptr = dvf.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var P = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr + bh * S * S, sc_rl)
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](goptr + bh * S * Dh, head_qk_rl)
        var DV = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr + bh * S * Dh, head_qk_rl)
        # DV[S,Dh] = Pᵀ[S,S] @ GO[S,Dh]  → transpose_a (attnᵀ)
        matmul(ctx, DV, P, GO, transpose_a=True, c_row_major=True)

    # ── 4) grad_attn = d_out @ Vᵀ  [BH,S,S] (reuse a fresh scores buffer) ─────
    var gscores = slab.alloc(BH * S * S * 4)
    var gsptr = gscores.unsafe_ptr().bitcast[Float32]()
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
    var dqf = slab.alloc(bhsd_rows * Dh * 4)
    var dkf = slab.alloc(bhsd_rows * Dh * 4)
    var dqptr = dqf.unsafe_ptr().bitcast[Float32]()
    var dkptr = dkf.unsafe_ptr().bitcast[Float32]()
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
    var dq_t = _scatter_to_tensor_slab(dq_full, B, S, H, Dh, out_dt, src_rl, ctx, slab)
    var dk_t = _scatter_to_tensor_slab(dk_full, B, S, H, Dh, out_dt, src_rl, ctx, slab)
    var dv_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dvptr, bhsd_rl)
    var dv_t = _scatter_to_tensor_slab(dv_full, B, S, H, Dh, out_dt, src_rl, ctx, slab)
    return SdpaGrads(dq_t^, dk_t^, dv_t^)
