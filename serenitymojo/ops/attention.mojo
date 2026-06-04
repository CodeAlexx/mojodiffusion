# ops/attention.mojo — sdpa: SDK flash_attention (Dh==64) + a math-mode fallback
# (any Dh) for the depths the flash MMA tiling can't compile on this GPU.
#
# ── Why two paths ───────────────────────────────────────────────────────────
# SDK-CALLABLE (OP-CALLABILITY MAP): `nn.attention.gpu.mha.flash_attention`, the
# LayoutTensor overload:
#     flash_attention(output, q, k, v, mask, scale, context: DeviceContextPtr)
# q/k/v/output are BSHD = [B, S, H, Dh]; the config reads H and Dh from the
# STATIC q layout at COMPILE TIME (hence B/S/H/Dh are comptime params here, not
# runtime). mask is [B, H, S, S] additive bias added to the QKᵀ scores.
#
# On RTX 3090 Ti (sm_86) the flash kernel FAILS TO COMPILE for `Dh == 128`
# (and 512): its MMA tiling for those depths selects an f16 tensor-core op
# (a=8xf16, b=4xf16) with no implementation on this arch. See
# models/text_encoder/parity/SDPA_DH128_REPRO.md (Dh isolation table: 64 OK,
# 128/512 FAIL, independent of head count / seq).
#
# ── Math-mode fallback (PyTorch "math" backend) ─────────────────────────────
# Plain matmuls + softmax — no flash MMA tiling, so no depth constraint, works
# at ANY Dh. Per (b,h):
#   scores[i,j] = (Q[b,:,h,:] @ K[b,:,h,:]ᵀ)[i,j] * scale + mask[b,h,i,j]
#   P           = softmax_j(scores)                       (last-dim softmax)
#   out[b,:,h,:]= P @ V[b,:,h,:]
# We loop over the B*H heads (correct-first; batched cuBLAS is a later optim).
# All inner math is F32 (cuBLAS accumulates in F32; scale/mask/softmax in F32);
# BF16/F16 only at the storage boundary (gather casts up, scatter casts down).
#
# The BSHD memory layout makes a single (b,h) head a STRIDED [S,Dh] slice
# (row stride = H*Dh), which the vendor 2D matmul (dense row-major) can't take
# directly. So we GATHER each head into a contiguous BHSD F32 buffer
# [B*H, S, Dh] first, run the dense matmuls, then SCATTER the result back to
# BSHD in the storage dtype. The gather/scatter kernels carry the whole
# BSHD↔matmul mapping in one place.
#
# Dispatch is comptime on Dh: Dh==64 keeps the (faster, working) flash path;
# every other Dh (8, 128, 512, ...) uses math-mode.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import exp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.runtime.tracing import DeviceContextPtr
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul
from nn.attention.gpu.mha import flash_attention
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256
comptime _NEG_BIG = Float32(-3.0e38)


# ── gather BSHD [B,S,H,Dh] -> BHSD-contiguous F32 [B*H, S, Dh] ───────────────
# One thread per destination element. dst flat index decomposes as
# (((b*H + h)*S + s)*Dh + d); the matching BSHD source offset is
# (((b*S + s)*H + h)*Dh + d). Casting up to F32 happens here.
def _gather_bshd_to_bhsd_f32(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # [B*S*H, Dh] view
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # [B*H*S, Dh] view
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * S * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh          # ((b*H + h)*S + s)
        var s = t % S
        var t2 = t // S            # (b*H + h)
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * S + s) * H + h
        var dst_row = (b * H + h) * S + s
        dst[dst_row, d] = src[src_row, d]


def _gather_bshd_to_bhsd_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * S * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * S + s) * H + h
        var dst_row = (b * H + h) * S + s
        var v = rebind[Scalar[DType.bfloat16]](src[src_row, d]).cast[DType.float32]()
        dst[dst_row, d] = rebind[dst.element_type](v)


def _gather_bshd_to_bhsd_f16(
    src: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * S * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * S + s) * H + h
        var dst_row = (b * H + h) * S + s
        var v = rebind[Scalar[DType.float16]](src[src_row, d]).cast[DType.float32]()
        dst[dst_row, d] = rebind[dst.element_type](v)


# ── scale + additive mask on the [B*H, S, S] scores buffer (all F32) ─────────
# scores flat row index = (b*H + h)*S + i; col = j. mask is [B,H,S,S] row-major,
# whose flat offset for (b,h,i,j) is (((b*H + h)*S + i)*S + j) — identical row
# layout, so mask_row == scores_row, mask_col == j. mask is cast up to F32.
def _scale_mask_f32(
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [B*H*S, S]
    mask: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],    # [B*H*S, S]
    scale: Float32,
    rows: Int, cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var v = rebind[Scalar[DType.float32]](scores[r, c]) * scale
        v += rebind[Scalar[DType.float32]](mask[r, c])
        scores[r, c] = rebind[scores.element_type](v)


def _scale_mask_bf16(
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    mask: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    scale: Float32,
    rows: Int, cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var v = rebind[Scalar[DType.float32]](scores[r, c]) * scale
        v += rebind[Scalar[DType.bfloat16]](mask[r, c]).cast[DType.float32]()
        scores[r, c] = rebind[scores.element_type](v)


def _scale_mask_f16(
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    mask: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    scale: Float32,
    rows: Int, cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var v = rebind[Scalar[DType.float32]](scores[r, c]) * scale
        v += rebind[Scalar[DType.float16]](mask[r, c]).cast[DType.float32]()
        scores[r, c] = rebind[scores.element_type](v)


def _scale_f32(
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [B*H*S, S]
    scale: Float32,
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var v = rebind[Scalar[DType.float32]](scores[r, c]) * scale
        scores[r, c] = rebind[scores.element_type](v)


# ── softmax over last dim, in place on the F32 scores [B*H*S, S] ─────────────
# One block per row; shared-memory tree reductions in F32 (mirrors softmax.mojo).
def _softmax_rows_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cols: Int,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    # Pass 1: row max.
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
            var b = shared[tid + active]
            shared[tid] = a if a > b else b
        barrier()
        active //= 2
    var rmax = shared[0]
    barrier()
    # Pass 2: sum of exp(x - max).
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
    # Write probabilities back in place.
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        x[row, c] = rebind[x.element_type](exp(v - rmax) * inv)
        c += _TPB


# ── scatter BHSD-contiguous F32 [B*H, S, Dh] -> BSHD storage dtype ───────────
def _scatter_bhsd_to_bshd_f32(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # [B*H*S, Dh]
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # [B*S*H, Dh]
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * S * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh          # ((b*H + h)*S + s)
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * H + h) * S + s
        var dst_row = (b * S + s) * H + h
        dst[dst_row, d] = src[src_row, d]


def _scatter_bhsd_to_bshd_bf16(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * S * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * H + h) * S + s
        var dst_row = (b * S + s) * H + h
        var v = rebind[Scalar[DType.float32]](src[src_row, d])
        dst[dst_row, d] = rebind[dst.element_type](v.cast[DType.bfloat16]())


def _scatter_bhsd_to_bshd_f16(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * S * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % S
        var t2 = t // S
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * H + h) * S + s
        var dst_row = (b * S + s) * H + h
        var v = rebind[Scalar[DType.float32]](src[src_row, d])
        dst[dst_row, d] = rebind[dst.element_type](v.cast[DType.float16]())


# ── math-mode SDPA (any Dh) ─────────────────────────────────────────────────
# q,k,v: BSHD [B,S,H,Dh] (storage dtype). mask: [B,H,S,S] (storage dtype).
# Returns BSHD [B,S,H,Dh] in q's dtype. F32 throughout the interior.
def _sdpa_math[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
    apply_mask: Bool,
) raises -> Tensor:
    var dt = q.dtype().to_mojo_dtype()
    comptime BH = B * H

    # ── 1) gather q/k/v BSHD -> BHSD-contiguous F32 [B*H*S, Dh] ──────────────
    comptime src_rows = B * S * H
    comptime bhsd_rows = B * H * S
    var q_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var k_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var v_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bhsd_rows, Dh))

    var qd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        q_f32.unsafe_ptr(), bhsd_rl
    )
    var kd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        k_f32.unsafe_ptr(), bhsd_rl
    )
    var vd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        v_f32.unsafe_ptr(), bhsd_rl
    )
    var ngather = B * H * S * Dh
    var ggrid = (ngather + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var qs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        var ks = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        var vs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        ctx.enqueue_function[_gather_bshd_to_bhsd_f32, _gather_bshd_to_bhsd_f32](
            qs, qd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_f32, _gather_bshd_to_bhsd_f32](
            ks, kd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_f32, _gather_bshd_to_bhsd_f32](
            vs, vd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        ctx.enqueue_function[_gather_bshd_to_bhsd_bf16, _gather_bshd_to_bhsd_bf16](
            qs, qd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_bf16, _gather_bshd_to_bhsd_bf16](
            ks, kd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_bf16, _gather_bshd_to_bhsd_bf16](
            vs, vd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    else:  # float16
        var qs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        var ks = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        var vs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        ctx.enqueue_function[_gather_bshd_to_bhsd_f16, _gather_bshd_to_bhsd_f16](
            qs, qd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_f16, _gather_bshd_to_bhsd_f16](
            ks, kd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_f16, _gather_bshd_to_bhsd_f16](
            vs, vd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)

    # ── 2) QKᵀ per head -> scores F32 [B*H, S, S] ────────────────────────────
    var scores = ctx.enqueue_create_buffer[DType.float32](BH * S * S)
    var head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, S))
    var qptr = q_f32.unsafe_ptr()
    var kptr = k_f32.unsafe_ptr()
    var scptr = scores.unsafe_ptr()
    for bh in range(BH):
        var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            qptr + bh * S * Dh, head_rl
        )
        var Bt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            kptr + bh * S * Dh, head_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            scptr + bh * S * S, sc_rl
        )
        # C[S,S] = A[S,Dh] @ Bt[S,Dh]ᵀ  (Q @ Kᵀ). c_row_major so C is row-major.
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)

    # ── 3) scale, optionally adding a mask over [B*H*S, S] scores ────────────
    comptime sm_rows = BH * S
    var sc_full_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, S))
    var sc_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        scptr, sc_full_rl
    )
    var nsm = sm_rows * S
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    if apply_mask:
        var mdt = mask.dtype().to_mojo_dtype()
        if mdt == DType.float32:
            var Mf = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                mask.buf.unsafe_ptr().bitcast[Float32](), sc_full_rl
            )
            ctx.enqueue_function[_scale_mask_f32, _scale_mask_f32](
                sc_full, Mf, scale, sm_rows, S, grid_dim=smgrid, block_dim=_BLOCK)
        elif mdt == DType.bfloat16:
            var Mf = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                mask.buf.unsafe_ptr().bitcast[BFloat16](), sc_full_rl
            )
            ctx.enqueue_function[_scale_mask_bf16, _scale_mask_bf16](
                sc_full, Mf, scale, sm_rows, S, grid_dim=smgrid, block_dim=_BLOCK)
        else:  # float16
            var Mf = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                mask.buf.unsafe_ptr().bitcast[Float16](), sc_full_rl
            )
            ctx.enqueue_function[_scale_mask_f16, _scale_mask_f16](
                sc_full, Mf, scale, sm_rows, S, grid_dim=smgrid, block_dim=_BLOCK)
    else:
        ctx.enqueue_function[_scale_f32, _scale_f32](
            sc_full, scale, sm_rows, S, grid_dim=smgrid, block_dim=_BLOCK)

    # ── 4) softmax over last dim (j) in place: one block per [B*H*S] row ─────
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        sc_full, S, grid_dim=sm_rows, block_dim=_TPB)

    # ── 5) P @ V per head -> out F32 BHSD [B*H, S, Dh] ───────────────────────
    var out_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var optr = out_f32.unsafe_ptr()
    var vptr = v_f32.unsafe_ptr()
    for bh in range(BH):
        var P = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            scptr + bh * S * S, sc_rl
        )
        var Vh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            vptr + bh * S * Dh, head_rl
        )
        var Oh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            optr + bh * S * Dh, head_rl
        )
        # Oh[S,Dh] = P[S,S] @ Vh[S,Dh]  (no transpose).
        matmul(ctx, Oh, P, Vh, transpose_b=False, c_row_major=True)

    # ── 6) scatter BHSD F32 -> BSHD output in storage dtype ──────────────────
    var bsz = q.dtype().byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * S * H * Dh * bsz)
    var out_src = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        optr, bhsd_rl
    )
    var nsc = B * H * S * Dh
    var scgrid = (nsc + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var Od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        ctx.enqueue_function[_scatter_bhsd_to_bshd_f32, _scatter_bhsd_to_bshd_f32](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var Od = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        ctx.enqueue_function[_scatter_bhsd_to_bshd_bf16, _scatter_bhsd_to_bshd_bf16](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    else:  # float16
        var Od = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        ctx.enqueue_function[_scatter_bhsd_to_bshd_f16, _scatter_bhsd_to_bshd_f16](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(S)
    out_shape.append(H)
    out_shape.append(Dh)
    return Tensor(out_buf^, out_shape^, q.dtype())


# ── memory-efficient (online-softmax) tiled SDPA ────────────────────────────
# WHY: the math-mode path above materializes the full F32 scores [B*H, S, S].
# At S=32760 that is ~68 GB → OOM, and the SDK flash path won't compile at
# Dh=128 on sm_86 (see SDPA_DH128_REPRO.md). So large-S Dh=128 has no runnable
# path. This streams over K/V in BLOCKS with the standard online-softmax
# recurrence, keeping memory O(S*Dh) (the gathered Q/K/V) + O(Dh) per query —
# never the [S,S] scores. Online softmax is EXACT, so this equals _sdpa_math.
#
# Layout: Q/K/V are gathered to BHSD-contiguous F32 [B*H, S, Dh] by the SAME
# _gather kernels the math path uses. ONE THREAD owns one (bh, query-row) pair
# and streams the whole K/V sequence in registers:
#   m   = running max of scaled (+masked) scores seen so far   (init -inf)
#   l   = running sum of exp(score - m)                         (init 0)
#   acc = running sum of exp(score - m) * V[j]   (Dh-wide)      (init 0)
# For each kv-block, for each kv-row j in the block:
#   s   = scale * dot(Q_i, K_j) [+ mask[b,h,i,j]]
#   m_new = max(m, s)
#   corr  = exp(m - m_new)               # rescale factor for the OLD state
#   l     = l*corr + exp(s - m_new)
#   acc   = acc*corr + exp(s - m_new) * V[j]
#   m     = m_new
# Final: out_i = acc / l. The corr rescaling on EVERY block is the bug surface;
# it is exercised whenever S spans >1 kv-block (the correctness gate uses
# S >> _KV_BLOCK so multiple blocks run).
#
# _DH_MAX caps the per-thread acc register array. 128 covers Dh in {64,128}
# (cosmos / magihuman / qwen3). A Dh above this raises at the driver.
comptime _DH_MAX = 128
comptime _KV_BLOCK = 512


# Online-softmax streaming attention. Grid = B*H*S threads (one per query row);
# blocks of _TPB. apply_mask reads mask[b,h,i,j] on the fly (NO [S,S] buffer).
# q/k/v: F32 BHSD [B*H*S, Dh]. out: F32 BHSD [B*H*S, Dh]. mask (F32): the
# math-path layout [B*H*S, S] so flat (b,h,i) row == query row, col == j.
def _sdpa_online_f32(
    q: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],     # [B*H*S, Dh]
    k: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],     # [B*H*S, Dh]
    v: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],     # [B*H*S, Dh]
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],     # [B*H*S, Dh]
    mask: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [B*H*S, S]
    scale: Float32,
    B: Int, S: Int, H: Int, Dh: Int,
    do_mask: Int,                       # 1 = read mask, 0 = nomask
):
    var qrow = Int(global_idx.x)        # flat (b*H + h)*S + i  == query row
    var total = B * H * S
    if qrow >= total:
        return
    var i = qrow % S                    # query position within head
    var bh = qrow // S                  # (b*H + h)
    var kbase = bh * S                  # first kv row of this head
    # Load this query row into registers.
    var qreg = stack_allocation[_DH_MAX, Scalar[DType.float32]]()
    for d in range(Dh):
        qreg[d] = rebind[Scalar[DType.float32]](q[qrow, d])
    # Running online-softmax state.
    var acc = stack_allocation[_DH_MAX, Scalar[DType.float32]]()
    for d in range(Dh):
        acc[d] = 0.0
    var m: Float32 = _NEG_BIG
    var l: Float32 = 0.0
    # Stream K/V in blocks.
    var jb = 0
    while jb < S:
        var jend = jb + _KV_BLOCK
        if jend > S:
            jend = S
        var j = jb
        while j < jend:
            var krow = kbase + j
            var dot: Float32 = 0.0
            for d in range(Dh):
                dot += qreg[d] * rebind[Scalar[DType.float32]](k[krow, d])
            var s = dot * scale
            if do_mask == 1:
                s += rebind[Scalar[DType.float32]](mask[qrow, j])
            var m_new = m if m > s else s
            var corr = exp(m - m_new)
            var p = exp(s - m_new)
            l = l * corr + p
            for d in range(Dh):
                acc[d] = acc[d] * corr + p * rebind[Scalar[DType.float32]](v[krow, d])
            m = m_new
            j += 1
        jb = jend
    # Normalize and write out (l>0 always: at least one kv row contributes).
    var inv = 1.0 / l
    for d in range(Dh):
        o[qrow, d] = rebind[o.element_type](acc[d] * inv)


# Driver: gather BSHD->BHSD F32 (reusing the math-path kernels), run the
# streaming kernel, scatter BHSD->BSHD storage dtype. mask (if apply_mask) is
# expected F32 in [B,H,S,S] layout (the math path's convention).
def _sdpa_tiled[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
    apply_mask: Bool,
) raises -> Tensor:
    comptime if Dh > _DH_MAX:
        raise Error("sdpa_tiled: Dh exceeds _DH_MAX (128)")
    var dt = q.dtype().to_mojo_dtype()
    if apply_mask and mask.dtype().to_mojo_dtype() != DType.float32:
        raise Error("sdpa_tiled: mask must be F32")

    comptime src_rows = B * S * H
    comptime bhsd_rows = B * H * S
    var q_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var k_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var v_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bhsd_rows, Dh))
    var qd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        q_f32.unsafe_ptr(), bhsd_rl)
    var kd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        k_f32.unsafe_ptr(), bhsd_rl)
    var vd = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        v_f32.unsafe_ptr(), bhsd_rl)
    var ngather = B * H * S * Dh
    var ggrid = (ngather + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var qs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var ks = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var vs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        ctx.enqueue_function[_gather_bshd_to_bhsd_f32, _gather_bshd_to_bhsd_f32](
            qs, qd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_f32, _gather_bshd_to_bhsd_f32](
            ks, kd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_f32, _gather_bshd_to_bhsd_f32](
            vs, vd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        ctx.enqueue_function[_gather_bshd_to_bhsd_bf16, _gather_bshd_to_bhsd_bf16](
            qs, qd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_bf16, _gather_bshd_to_bhsd_bf16](
            ks, kd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_bf16, _gather_bshd_to_bhsd_bf16](
            vs, vd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    else:  # float16
        var qs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var ks = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var vs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        ctx.enqueue_function[_gather_bshd_to_bhsd_f16, _gather_bshd_to_bhsd_f16](
            qs, qd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_f16, _gather_bshd_to_bhsd_f16](
            ks, kd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
        ctx.enqueue_function[_gather_bshd_to_bhsd_f16, _gather_bshd_to_bhsd_f16](
            vs, vd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)

    # Streaming attention -> out F32 BHSD [B*H*S, Dh]. NO [S,S] anywhere.
    var out_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_f32.unsafe_ptr(), bhsd_rl)
    # Mask view: [B*H*S, S] (math-path layout); a 1-elem dummy when no mask.
    comptime mrows = B * H * S
    var mask_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](mrows, S))
    var dummy_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](1, 1))
    var nq = B * H * S
    var qgrid = (nq + _TPB - 1) // _TPB
    if apply_mask:
        var Mf = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[Float32](), mask_rl)
        ctx.enqueue_function[_sdpa_online_f32, _sdpa_online_f32](
            qd, kd, vd, od, Mf, scale, B, S, H, Dh, 1,
            grid_dim=qgrid, block_dim=_TPB)
    else:
        var Mf = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            q_f32.unsafe_ptr(), dummy_rl)   # unused dummy view
        ctx.enqueue_function[_sdpa_online_f32, _sdpa_online_f32](
            qd, kd, vd, od, Mf, scale, B, S, H, Dh, 0,
            grid_dim=qgrid, block_dim=_TPB)

    # Scatter BHSD F32 -> BSHD storage dtype (reusing the math-path kernels).
    var bsz = q.dtype().byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * S * H * Dh * bsz)
    var out_src = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_f32.unsafe_ptr(), bhsd_rl)
    var nsc = B * H * S * Dh
    var scgrid = (nsc + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var Od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), src_rl)
        ctx.enqueue_function[_scatter_bhsd_to_bshd_f32, _scatter_bhsd_to_bshd_f32](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var Od = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        ctx.enqueue_function[_scatter_bhsd_to_bshd_bf16, _scatter_bhsd_to_bshd_bf16](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    else:  # float16
        var Od = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), src_rl)
        ctx.enqueue_function[_scatter_bhsd_to_bshd_f16, _scatter_bhsd_to_bshd_f16](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(S)
    out_shape.append(H)
    out_shape.append(Dh)
    return Tensor(out_buf^, out_shape^, q.dtype())


# ── flash-attention path (Dh==64; SDK MMA tiling supported on sm_86) ────────
def _sdpa_flash[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime qkv_layout = Layout.row_major(B, S, H, Dh)
    comptime mask_layout = Layout.row_major(B, H, S, S)

    var dt = q.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](q.nbytes())
    var dcp = DeviceContextPtr(ctx)

    if dt == DType.float32:
        var Q = LayoutTensor[DType.float32, qkv_layout, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float32]()
        )
        var K = LayoutTensor[DType.float32, qkv_layout, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float32]()
        )
        var V = LayoutTensor[DType.float32, qkv_layout, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float32]()
        )
        var M = LayoutTensor[DType.float32, mask_layout, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[Float32]()
        )
        var O = LayoutTensor[DType.float32, qkv_layout, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32]()
        )
        flash_attention(O, Q, K, V, M, scale, dcp)
    elif dt == DType.bfloat16:
        var Q = LayoutTensor[DType.bfloat16, qkv_layout, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16]()
        )
        var K = LayoutTensor[DType.bfloat16, qkv_layout, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16]()
        )
        var V = LayoutTensor[DType.bfloat16, qkv_layout, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16]()
        )
        var M = LayoutTensor[DType.bfloat16, mask_layout, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[BFloat16]()
        )
        var O = LayoutTensor[DType.bfloat16, qkv_layout, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16]()
        )
        flash_attention(O, Q, K, V, M, scale, dcp)
    else:  # float16
        var Q = LayoutTensor[DType.float16, qkv_layout, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16]()
        )
        var K = LayoutTensor[DType.float16, qkv_layout, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16]()
        )
        var V = LayoutTensor[DType.float16, qkv_layout, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16]()
        )
        var M = LayoutTensor[DType.float16, mask_layout, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[Float16]()
        )
        var O = LayoutTensor[DType.float16, qkv_layout, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16]()
        )
        flash_attention(O, Q, K, V, M, scale, dcp)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(S)
    out_shape.append(H)
    out_shape.append(Dh)
    return Tensor(out_buf^, out_shape^, q.dtype())


# Whether to route Dh==64 attention through the SDK flash_attention kernel.
# FALSE on purpose: benched 2026-05-29 on RTX 3090 Ti (sm_86), the SDK flash
# kernel at Dh=64 is ~2607 us/iter — ~31x SLOWER than flame-core's cuDNN and
# slower than our own math-mode path. So Dh==64 also uses math-mode here.
# Flip to True once Modular's flash kernel is fast on the target arch.
comptime _USE_SDK_FLASH_DH64 = False


def sdpa[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Scaled dot-product attention. Non-causal full attention (diffusion).

    q, k, v: [B, S, H, Dh]   (BSHD row-major; same compute dtype; kv already
                              GQA-expanded to H heads by the caller)
    mask:    [B, H, S, S]    (additive score bias; zeros = full attention)
    scale:   Float32         (typically 1/sqrt(Dh))
    returns  [B, S, H, Dh]   (q's dtype).

    B/S/H/Dh are compile-time params. DISPATCH: math-mode (cuBLAS matmuls + F32
    softmax) is used for ALL Dh by default. The SDK flash_attention kernel is
    gated behind `_USE_SDK_FLASH_DH64` (currently False) because on sm_86 it is
    ~31x slower than cuDNN at Dh=64 and fails to compile at Dh in {128,512,...}
    (the MMA-tiling wall). See SDPA_DH128_REPRO.md and the 2026-05-29 perf bench.
    """
    if q.dtype() != k.dtype() or q.dtype() != v.dtype():
        raise Error("sdpa: q/k/v dtype mismatch")
    if q.dtype() != mask.dtype():
        raise Error("sdpa: q/mask dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4:
        raise Error("sdpa: q must be rank-4 [B,S,H,Dh]")
    if (
        qshape[0] != B
        or qshape[1] != S
        or qshape[2] != H
        or qshape[3] != Dh
    ):
        raise Error("sdpa: q shape does not match compile-time B/S/H/Dh")
    var mshape = mask.shape()
    if (
        len(mshape) != 4
        or mshape[0] != B
        or mshape[1] != H
        or mshape[2] != S
        or mshape[3] != S
    ):
        raise Error("sdpa: mask must be [B,H,S,S]")

    comptime if Dh == 64 and _USE_SDK_FLASH_DH64:
        return _sdpa_flash[B, S, H, Dh](q, k, v, mask, scale, ctx)
    else:
        return _sdpa_math[B, S, H, Dh](q, k, v, mask, scale, ctx, True)


def sdpa_nomask[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Scaled dot-product full attention without an additive mask tensor.

    This is for diffusion paths whose attention mask is known to be all zeros.
    It avoids materializing [B,H,S,S] just to add zero. The implementation uses
    the math-mode SDPA path for all Dh values so it needs no SDK flash mask arg.
    """
    if q.dtype() != k.dtype() or q.dtype() != v.dtype():
        raise Error("sdpa_nomask: q/k/v dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4:
        raise Error("sdpa_nomask: q must be rank-4 [B,S,H,Dh]")
    if (
        qshape[0] != B
        or qshape[1] != S
        or qshape[2] != H
        or qshape[3] != Dh
    ):
        raise Error("sdpa_nomask: q shape does not match compile-time B/S/H/Dh")

    return _sdpa_math[B, S, H, Dh](q, k, v, q, scale, ctx, False)


def sdpa_tiled[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Memory-efficient (online-softmax) SDPA — for LARGE S at Dh in {64,128}.

    Same contract/result as `sdpa` (full non-causal attention, additive
    [B,H,S,S] mask) but NEVER materializes the [S,S] scores: it streams K/V in
    blocks with the online-softmax recurrence, so peak memory is O(S*Dh) (the
    gathered Q/K/V) instead of O(S*S). Use this when S is large enough that the
    default `sdpa` would OOM on the scores buffer (e.g. S >= 8192, Dh=128).
    Online softmax is exact, so the output equals `sdpa` to ~machine precision.

    mask must be F32 [B,H,S,S]. Dh <= 128. Existing sdpa / flash paths are
    untouched — this is a separate, additive entry.
    """
    if q.dtype() != k.dtype() or q.dtype() != v.dtype():
        raise Error("sdpa_tiled: q/k/v dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4:
        raise Error("sdpa_tiled: q must be rank-4 [B,S,H,Dh]")
    if (
        qshape[0] != B or qshape[1] != S or qshape[2] != H or qshape[3] != Dh
    ):
        raise Error("sdpa_tiled: q shape does not match compile-time B/S/H/Dh")
    var mshape = mask.shape()
    if (
        len(mshape) != 4
        or mshape[0] != B or mshape[1] != H
        or mshape[2] != S or mshape[3] != S
    ):
        raise Error("sdpa_tiled: mask must be [B,H,S,S]")
    return _sdpa_tiled[B, S, H, Dh](q, k, v, mask, scale, ctx, True)


def sdpa_nomask_tiled[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Memory-efficient SDPA without an additive mask (mask known all-zero).

    Like `sdpa_nomask` but streams K/V (online softmax) so the [S,S] scores are
    never materialized — runs at large S / Dh=128 where the default path OOMs.
    """
    if q.dtype() != k.dtype() or q.dtype() != v.dtype():
        raise Error("sdpa_nomask_tiled: q/k/v dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4:
        raise Error("sdpa_nomask_tiled: q must be rank-4 [B,S,H,Dh]")
    if (
        qshape[0] != B or qshape[1] != S or qshape[2] != H or qshape[3] != Dh
    ):
        raise Error(
            "sdpa_nomask_tiled: q shape does not match compile-time B/S/H/Dh"
        )
    return _sdpa_tiled[B, S, H, Dh](q, k, v, q, scale, ctx, False)
