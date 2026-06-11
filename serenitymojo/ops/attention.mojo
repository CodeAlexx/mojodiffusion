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
# Scores, softmax, and GEMM C/accumulator buffers are F32. Q/K/V staging
# preserves the input storage dtype; BF16/F16 are cast to F32 only as scalar
# elements inside kernels or by BLAS accumulation.
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
from serenitymojo.autograd_v2.step_slab import StepSlab


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256
comptime _NEG_BIG = Float32(-3.0e38)


# ── gather BSHD [B,S,H,Dh] -> BHSD-contiguous storage [B*H, S, Dh] ───────────
# One thread per destination element. dst flat index decomposes as
# (((b*H + h)*S + s)*Dh + d); the matching BSHD source offset is
# (((b*S + s)*H + h)*Dh + d).
def _gather_bshd_to_bhsd[dtype: DType](
    src: LayoutTensor[dtype, _DYN2, MutAnyOrigin],   # [B*S*H, Dh] view
    dst: LayoutTensor[dtype, _DYN2, MutAnyOrigin],   # [B*H*S, Dh] view
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


# ── scale + additive mask on the [B*H, S, S] scores buffer (all F32) ─────────
# scores flat row index = (b*H + h)*S + i; col = j. mask is [B,H,S,S] row-major,
# whose flat offset for (b,h,i,j) is (((b*H + h)*S + i)*S + j). The mask keeps
# storage dtype and is cast per scalar for the F32 score update.
def _scale_mask[dtype: DType](
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [B*H*S, S]
    mask: LayoutTensor[dtype, _DYN2, MutAnyOrigin],            # [B*H*S, S]
    scale: Float32,
    rows: Int, cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var v = rebind[Scalar[DType.float32]](scores[r, c]) * scale
        v += rebind[Scalar[dtype]](mask[r, c]).cast[DType.float32]()
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


def _attn_pv_kernel[dtype: DType](
    probs: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [B*H*Sq, Skv]
    v: LayoutTensor[dtype, _DYN2, MutAnyOrigin],              # [B*H*Skv, Dh]
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],    # [B*H*Sq, Dh]
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
            var p = rebind[Scalar[DType.float32]](probs[qrow, j])
            var vv = rebind[Scalar[dtype]](v[kbase + j, d]).cast[DType.float32]()
            acc += p * vv
        dst[qrow, d] = rebind[dst.element_type](acc)


# ── math-mode SDPA (any Dh) ─────────────────────────────────────────────────
# q,k,v: BSHD [B,S,H,Dh] (storage dtype). mask: [B,H,S,S] (storage dtype).
# Returns BSHD [B,S,H,Dh] in q's dtype. Scores/softmax/GEMM accumulators are F32.
def _sdpa_math_storage[
    B: Int, S: Int, H: Int, Dh: Int, dtype: DType
](
    qs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    ks: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    vs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    mask: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    scale: Float32,
    ctx: DeviceContext,
    apply_mask: Bool,
    out_dt: STDtype,
) raises -> Tensor:
    comptime BH = B * H

    # ── 1) gather q/k/v BSHD -> BHSD-contiguous storage [B*H*S, Dh] ──────────
    comptime src_rows = B * S * H
    comptime bhsd_rows = B * H * S
    var q_buf = ctx.enqueue_create_buffer[dtype](bhsd_rows * Dh)
    var k_buf = ctx.enqueue_create_buffer[dtype](bhsd_rows * Dh)
    var v_buf = ctx.enqueue_create_buffer[dtype](bhsd_rows * Dh)
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bhsd_rows, Dh))

    var qd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](q_buf.unsafe_ptr(), bhsd_rl)
    var kd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](k_buf.unsafe_ptr(), bhsd_rl)
    var vd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](v_buf.unsafe_ptr(), bhsd_rl)
    var ngather = B * H * S * Dh
    var ggrid = (ngather + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](qs, qd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](ks, kd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](vs, vd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)

    # ── 2) QKᵀ per head -> scores F32 [B*H, S, S] ────────────────────────────
    var scores = ctx.enqueue_create_buffer[DType.float32](BH * S * S)
    var head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, S))
    var qptr = q_buf.unsafe_ptr()
    var kptr = k_buf.unsafe_ptr()
    var scptr = scores.unsafe_ptr()
    for bh in range(BH):
        var A = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            qptr + bh * S * Dh, head_rl
        )
        var Bt = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
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
        ctx.enqueue_function[_scale_mask[dtype], _scale_mask[dtype]](
            sc_full, mask, scale, sm_rows, S, grid_dim=smgrid, block_dim=_BLOCK)
    else:
        ctx.enqueue_function[_scale_f32, _scale_f32](
            sc_full, scale, sm_rows, S, grid_dim=smgrid, block_dim=_BLOCK)

    # ── 4) softmax over last dim (j) in place: one block per [B*H*S] row ─────
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        sc_full, S, grid_dim=sm_rows, block_dim=_TPB)

    # ── 5) P @ V per head -> out F32 BHSD [B*H, S, Dh] ───────────────────────
    var out_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var optr = out_f32.unsafe_ptr()
    comptime if dtype == DType.float32:
        var vptr = v_buf.unsafe_ptr().bitcast[Float32]()
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
            matmul(ctx, Oh, P, Vh, transpose_b=False, c_row_major=True)
    else:
        var v_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            v_buf.unsafe_ptr(), bhsd_rl
        )
        var o_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            optr, bhsd_rl
        )
        var pv_total = B * H * S * Dh
        var pv_grid = (pv_total + _BLOCK - 1) // _BLOCK
        ctx.enqueue_function[
            _attn_pv_kernel[dtype], _attn_pv_kernel[dtype]
        ](sc_full, v_full, o_full, B, S, S, H, Dh, grid_dim=pv_grid, block_dim=_BLOCK)

    # ── 6) scatter BHSD F32 -> BSHD output in storage dtype ──────────────────
    var bsz = out_dt.byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * S * H * Dh * bsz)
    var out_src = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        optr, bhsd_rl
    )
    var nsc = B * H * S * Dh
    var scgrid = (nsc + _BLOCK - 1) // _BLOCK
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    comptime if dtype == DType.float32:
        var Od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        ctx.enqueue_function[_scatter_bhsd_to_bshd_f32, _scatter_bhsd_to_bshd_f32](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    elif dtype == DType.bfloat16:
        var Od = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        ctx.enqueue_function[_scatter_bhsd_to_bshd_bf16, _scatter_bhsd_to_bshd_bf16](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    else:
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
    return Tensor(out_buf^, out_shape^, out_dt)


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
    comptime src_rows = B * S * H
    comptime sm_rows = B * H * S
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var mask_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, S))
    if dt == DType.float32:
        var qs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var ks = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var vs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var ms = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[Float32](), mask_rl)
        return _sdpa_math_storage[B, S, H, Dh, DType.float32](
            qs, ks, vs, ms, scale, ctx, apply_mask, q.dtype())
    elif dt == DType.bfloat16:
        var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var ms = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[BFloat16](), mask_rl)
        return _sdpa_math_storage[B, S, H, Dh, DType.bfloat16](
            qs, ks, vs, ms, scale, ctx, apply_mask, q.dtype())
    else:
        var qs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var ks = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var vs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var ms = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[Float16](), mask_rl)
        return _sdpa_math_storage[B, S, H, Dh, DType.float16](
            qs, ks, vs, ms, scale, ctx, apply_mask, q.dtype())


# ── head-chunked math SDPA (bit-identical to _sdpa_math, smaller peak) ───────
# WHY: _sdpa_math_storage materializes the F32 scores for ALL heads at once:
# [B*H, S, S] F32 = B*H*S*S*4 bytes. At B=1,H=24,S=4297 that is ~1.77 GB — the
# single biggest transient in a Lens 1024 block. Heads are fully independent
# (QKᵀ per head → scale+mask per head's rows → softmax per row → P@V per head),
# so we process ONE head at a time, REUSING a single [S,S] F32 scores buffer.
# Every kernel call is the SAME op on the SAME data as the batched path, just
# interleaved per head; on a single stream the buffer reuse is correctly
# serialized. The result is BIT-IDENTICAL to _sdpa_math_storage; only the scores
# peak drops from O(B*H*S*S) to O(S*S) (~74 MB vs ~1.77 GB at S=4297).
def _sdpa_math_storage_chunked[
    B: Int, S: Int, H: Int, Dh: Int, dtype: DType
](
    qs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    ks: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    vs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    mask: LayoutTensor[dtype, _DYN2, MutAnyOrigin],   # [B*H*S, S]
    scale: Float32,
    ctx: DeviceContext,
    apply_mask: Bool,
    out_dt: STDtype,
) raises -> Tensor:
    comptime BH = B * H
    comptime src_rows = B * S * H
    comptime bhsd_rows = B * H * S

    # ── 1) gather q/k/v BSHD -> BHSD-contiguous storage [B*H*S, Dh] ──────────
    var q_buf = ctx.enqueue_create_buffer[dtype](bhsd_rows * Dh)
    var k_buf = ctx.enqueue_create_buffer[dtype](bhsd_rows * Dh)
    var v_buf = ctx.enqueue_create_buffer[dtype](bhsd_rows * Dh)
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bhsd_rows, Dh))
    var qd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](q_buf.unsafe_ptr(), bhsd_rl)
    var kd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](k_buf.unsafe_ptr(), bhsd_rl)
    var vd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](v_buf.unsafe_ptr(), bhsd_rl)
    var ngather = B * H * S * Dh
    var ggrid = (ngather + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](qs, qd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](ks, kd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](vs, vd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)

    # ── single REUSED per-head F32 scores buffer [S, S] + full F32 output ────
    var scores = ctx.enqueue_create_buffer[DType.float32](S * S)
    var out_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_rows * Dh)
    var head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, S))
    var qptr = q_buf.unsafe_ptr()
    var kptr = k_buf.unsafe_ptr()
    var vptr = v_buf.unsafe_ptr()
    var scptr = scores.unsafe_ptr()
    var optr = out_f32.unsafe_ptr()
    var maskptr = mask.ptr

    var nsm = S * S
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    var pv_total = S * Dh
    var pv_grid = (pv_total + _BLOCK - 1) // _BLOCK

    for bh in range(BH):
        # QKᵀ for this head -> scores[S,S] (row-major).
        var A = LayoutTensor[dtype, _DYN2, MutAnyOrigin](qptr + bh * S * Dh, head_rl)
        var Bt = LayoutTensor[dtype, _DYN2, MutAnyOrigin](kptr + bh * S * Dh, head_rl)
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](scptr, sc_rl)
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)

        # scale (+ optional additive mask for this head's [S,S] block).
        var sc_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](scptr, sc_rl)
        if apply_mask:
            var mh = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
                maskptr + bh * S * S, sc_rl
            )
            ctx.enqueue_function[_scale_mask[dtype], _scale_mask[dtype]](
                sc_full, mh, scale, S, S, grid_dim=smgrid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[_scale_f32, _scale_f32](
                sc_full, scale, S, S, grid_dim=smgrid, block_dim=_BLOCK)

        # softmax over last dim, one block per row.
        ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
            sc_full, S, grid_dim=S, block_dim=_TPB)

        # P @ V for this head -> out_f32[bh*S*Dh : ...].
        comptime if dtype == DType.float32:
            var P = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](scptr, sc_rl)
            var Vh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                vptr.bitcast[Float32]() + bh * S * Dh, head_rl
            )
            var Oh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                optr + bh * S * Dh, head_rl
            )
            matmul(ctx, Oh, P, Vh, transpose_b=False, c_row_major=True)
        else:
            var Vh = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
                vptr + bh * S * Dh, head_rl
            )
            var Oh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                optr + bh * S * Dh, head_rl
            )
            # _attn_pv_kernel with B=1,H=1: probs[S,S], v[S,Dh], dst[S,Dh].
            ctx.enqueue_function[
                _attn_pv_kernel[dtype], _attn_pv_kernel[dtype]
            ](sc_full, Vh, Oh, 1, S, S, 1, Dh, grid_dim=pv_grid, block_dim=_BLOCK)

    # ── scatter BHSD F32 -> BSHD output in storage dtype ─────────────────────
    var bsz = out_dt.byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * S * H * Dh * bsz)
    var out_src = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](optr, bhsd_rl)
    var nsc = B * H * S * Dh
    var scgrid = (nsc + _BLOCK - 1) // _BLOCK
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    comptime if dtype == DType.float32:
        var Od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), src_rl)
        ctx.enqueue_function[_scatter_bhsd_to_bshd_f32, _scatter_bhsd_to_bshd_f32](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    elif dtype == DType.bfloat16:
        var Od = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        ctx.enqueue_function[_scatter_bhsd_to_bshd_bf16, _scatter_bhsd_to_bshd_bf16](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    else:
        var Od = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), src_rl)
        ctx.enqueue_function[_scatter_bhsd_to_bshd_f16, _scatter_bhsd_to_bshd_f16](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(S)
    out_shape.append(H)
    out_shape.append(Dh)
    return Tensor(out_buf^, out_shape^, out_dt)


def _sdpa_math_chunked[
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
    comptime src_rows = B * S * H
    comptime sm_rows = B * H * S
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var mask_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, S))
    if dt == DType.float32:
        var qs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var ks = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var vs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var ms = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[Float32](), mask_rl)
        return _sdpa_math_storage_chunked[B, S, H, Dh, DType.float32](
            qs, ks, vs, ms, scale, ctx, apply_mask, q.dtype())
    elif dt == DType.bfloat16:
        var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var ms = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[BFloat16](), mask_rl)
        return _sdpa_math_storage_chunked[B, S, H, Dh, DType.bfloat16](
            qs, ks, vs, ms, scale, ctx, apply_mask, q.dtype())
    else:
        var qs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var ks = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var vs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var ms = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[Float16](), mask_rl)
        return _sdpa_math_storage_chunked[B, S, H, Dh, DType.float16](
            qs, ks, vs, ms, scale, ctx, apply_mask, q.dtype())


# ── memory-efficient (online-softmax) tiled SDPA ────────────────────────────
# WHY: the math-mode path above materializes the full F32 scores [B*H, S, S].
# At S=32760 that is ~68 GB → OOM, and the SDK flash path won't compile at
# Dh=128 on sm_86 (see SDPA_DH128_REPRO.md). So large-S Dh=128 has no runnable
# path. This streams over K/V in BLOCKS with the standard online-softmax
# recurrence, keeping memory O(S*Dh) (the gathered Q/K/V) + O(Dh) per query —
# never the [S,S] scores. Online softmax is EXACT, so this equals _sdpa_math.
#
# Layout: Q/K/V are gathered to BHSD-contiguous storage [B*H, S, Dh]. ONE THREAD
# owns one (bh, query-row) pair
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
comptime _MATH_SCORE_BUDGET_MIB = 3584


# Online-softmax streaming attention. Grid = B*H*S threads (one per query row);
# blocks of _TPB. apply_mask reads mask[b,h,i,j] on the fly (NO [S,S] buffer).
# q/k/v: storage BHSD [B*H*S, Dh]. out: storage BSHD [B*S*H, Dh].
# mask: [B*H*S, S] so flat (b,h,i) row == query row, col == j. Mask storage
# stays caller dtype; scalar values are cast to F32 for score math.
def _sdpa_online[dtype: DType, mask_dtype: DType](
    q: LayoutTensor[dtype, _DYN2, MutAnyOrigin],             # [B*H*S, Dh]
    k: LayoutTensor[dtype, _DYN2, MutAnyOrigin],             # [B*H*S, Dh]
    v: LayoutTensor[dtype, _DYN2, MutAnyOrigin],             # [B*H*S, Dh]
    o: LayoutTensor[dtype, _DYN2, MutAnyOrigin],             # [B*S*H, Dh]
    mask: LayoutTensor[mask_dtype, _DYN2, MutAnyOrigin],     # [B*H*S, S]
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
        qreg[d] = rebind[Scalar[dtype]](q[qrow, d]).cast[DType.float32]()
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
                dot += qreg[d] * rebind[Scalar[dtype]](k[krow, d]).cast[DType.float32]()
            var s = dot * scale
            if do_mask == 1:
                s += rebind[Scalar[mask_dtype]](mask[qrow, j]).cast[DType.float32]()
            var m_new = m if m > s else s
            var corr = exp(m - m_new)
            var p = exp(s - m_new)
            l = l * corr + p
            for d in range(Dh):
                acc[d] = acc[d] * corr + p * rebind[Scalar[dtype]](v[krow, d]).cast[DType.float32]()
            m = m_new
            j += 1
        jb = jend
    # Normalize and write out (l>0 always: at least one kv row contributes).
    var inv = 1.0 / l
    var h = bh % H
    var b = bh // H
    var dst_row = (b * S + i) * H + h
    for d in range(Dh):
        o[dst_row, d] = rebind[o.element_type]((acc[d] * inv).cast[dtype]())


# Driver: gather BSHD->BHSD storage, run the streaming kernel, and write BSHD
# storage output directly. mask (if apply_mask) is typed storage in [B,H,S,S].
def _sdpa_tiled_storage[
    B: Int, S: Int, H: Int, Dh: Int, dtype: DType, mask_dtype: DType
](
    qs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    ks: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    vs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    mask: LayoutTensor[mask_dtype, _DYN2, MutAnyOrigin],
    scale: Float32,
    ctx: DeviceContext,
    apply_mask: Bool,
    out_dt: STDtype,
) raises -> Tensor:
    comptime if Dh > _DH_MAX:
        raise Error("sdpa_tiled: Dh exceeds _DH_MAX (128)")

    comptime src_rows = B * S * H
    comptime bhsd_rows = B * H * S
    var q_buf = ctx.enqueue_create_buffer[dtype](bhsd_rows * Dh)
    var k_buf = ctx.enqueue_create_buffer[dtype](bhsd_rows * Dh)
    var v_buf = ctx.enqueue_create_buffer[dtype](bhsd_rows * Dh)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        B * S * H * Dh * out_dt.byte_size()
    )
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bhsd_rows, Dh))
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var qd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](q_buf.unsafe_ptr(), bhsd_rl)
    var kd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](k_buf.unsafe_ptr(), bhsd_rl)
    var vd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](v_buf.unsafe_ptr(), bhsd_rl)
    var ngather = B * H * S * Dh
    var ggrid = (ngather + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](qs, qd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](ks, kd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](vs, vd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)

    var nq = B * H * S
    var qgrid = (nq + _TPB - 1) // _TPB
    var do_mask = 1 if apply_mask else 0
    comptime if dtype == DType.float32:
        var od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), src_rl)
        ctx.enqueue_function[
            _sdpa_online[dtype, mask_dtype],
            _sdpa_online[dtype, mask_dtype],
        ](
            qd, kd, vd, od, mask, scale, B, S, H, Dh, do_mask,
            grid_dim=qgrid, block_dim=_TPB)
    elif dtype == DType.bfloat16:
        var od = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        ctx.enqueue_function[
            _sdpa_online[dtype, mask_dtype],
            _sdpa_online[dtype, mask_dtype],
        ](
            qd, kd, vd, od, mask, scale, B, S, H, Dh, do_mask,
            grid_dim=qgrid, block_dim=_TPB)
    else:
        var od = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), src_rl)
        ctx.enqueue_function[
            _sdpa_online[dtype, mask_dtype],
            _sdpa_online[dtype, mask_dtype],
        ](
            qd, kd, vd, od, mask, scale, B, S, H, Dh, do_mask,
            grid_dim=qgrid, block_dim=_TPB)

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(S)
    out_shape.append(H)
    out_shape.append(Dh)
    return Tensor(out_buf^, out_shape^, out_dt)


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
    var dt = q.dtype().to_mojo_dtype()
    if apply_mask and q.dtype() != mask.dtype():
        raise Error("sdpa_tiled: q/mask dtype mismatch")
    comptime src_rows = B * S * H
    comptime mrows = B * H * S
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var mask_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](mrows, S))
    var dummy_buf = ctx.enqueue_create_buffer[DType.float32](1)
    var dummy_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](1, 1))
    var dummy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dummy_buf.unsafe_ptr(), dummy_rl)
    if dt == DType.float32:
        var qs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var ks = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var vs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        if apply_mask:
            var ms = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                mask.buf.unsafe_ptr().bitcast[Float32](), mask_rl)
            return _sdpa_tiled_storage[
                B, S, H, Dh, DType.float32, DType.float32
            ](
                qs, ks, vs, ms, scale, ctx, True, q.dtype())
        return _sdpa_tiled_storage[B, S, H, Dh, DType.float32, DType.float32](
            qs, ks, vs, dummy, scale, ctx, False, q.dtype())
    elif dt == DType.bfloat16:
        var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        if apply_mask:
            var ms = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                mask.buf.unsafe_ptr().bitcast[BFloat16](), mask_rl)
            return _sdpa_tiled_storage[
                B, S, H, Dh, DType.bfloat16, DType.bfloat16
            ](
                qs, ks, vs, ms, scale, ctx, True, q.dtype())
        return _sdpa_tiled_storage[B, S, H, Dh, DType.bfloat16, DType.float32](
            qs, ks, vs, dummy, scale, ctx, False, q.dtype())
    else:
        var qs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var ks = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var vs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        if apply_mask:
            var ms = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                mask.buf.unsafe_ptr().bitcast[Float16](), mask_rl)
            return _sdpa_tiled_storage[
                B, S, H, Dh, DType.float16, DType.float16
            ](
                qs, ks, vs, ms, scale, ctx, True, q.dtype())
        return _sdpa_tiled_storage[B, S, H, Dh, DType.float16, DType.float32](
            qs, ks, vs, dummy, scale, ctx, False, q.dtype())


# Online-softmax rectangular attention. Grid = B*H*Sq threads, one per query row.
# This is for cross-attention and cross-modal attention where Sq != Skv. It
# never pads K/V to Sq and never materializes a [Sq,Skv] score slab. If
# do_mask=1, mask is [B*H*Sq, Skv] and is added to F32 scores on the fly.
def _sdpa_cross_online[dtype: DType, mask_dtype: DType](
    q: LayoutTensor[dtype, _DYN2, MutAnyOrigin],             # [B*H*Sq, Dh]
    k: LayoutTensor[dtype, _DYN2, MutAnyOrigin],             # [B*H*Skv, Dh]
    v: LayoutTensor[dtype, _DYN2, MutAnyOrigin],             # [B*H*Skv, Dh]
    o: LayoutTensor[dtype, _DYN2, MutAnyOrigin],             # [B*Sq*H, Dh]
    mask: LayoutTensor[mask_dtype, _DYN2, MutAnyOrigin],     # [B*H*Sq, Skv]
    scale: Float32,
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int,
    do_mask: Int,
):
    var qrow = Int(global_idx.x)
    var total = B * H * Sq
    if qrow >= total:
        return
    var bh = qrow // Sq
    var kbase = bh * Skv

    var qreg = stack_allocation[_DH_MAX, Scalar[DType.float32]]()
    for d in range(Dh):
        qreg[d] = rebind[Scalar[dtype]](q[qrow, d]).cast[DType.float32]()

    var acc = stack_allocation[_DH_MAX, Scalar[DType.float32]]()
    for d in range(Dh):
        acc[d] = 0.0
    var m: Float32 = _NEG_BIG
    var l: Float32 = 0.0

    var jb = 0
    while jb < Skv:
        var jend = jb + _KV_BLOCK
        if jend > Skv:
            jend = Skv
        var j = jb
        while j < jend:
            var krow = kbase + j
            var dot: Float32 = 0.0
            for d in range(Dh):
                dot += qreg[d] * rebind[Scalar[dtype]](k[krow, d]).cast[DType.float32]()
            var s = dot * scale
            if do_mask == 1:
                s += rebind[Scalar[mask_dtype]](mask[qrow, j]).cast[DType.float32]()
            var m_new = m if m > s else s
            var corr = exp(m - m_new)
            var p = exp(s - m_new)
            l = l * corr + p
            for d in range(Dh):
                acc[d] = acc[d] * corr + p * rebind[Scalar[dtype]](v[krow, d]).cast[DType.float32]()
            m = m_new
            j += 1
        jb = jend

    var inv = 1.0 / l
    var qi = qrow % Sq
    var h = bh % H
    var b = bh // H
    var dst_row = (b * Sq + qi) * H + h
    for d in range(Dh):
        o[dst_row, d] = rebind[o.element_type]((acc[d] * inv).cast[dtype]())


def _sdpa_cross_tiled_storage[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int, dtype: DType, mask_dtype: DType
](
    qs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    ks: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    vs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    mask: LayoutTensor[mask_dtype, _DYN2, MutAnyOrigin],
    scale: Float32,
    ctx: DeviceContext,
    apply_mask: Bool,
    out_dt: STDtype,
) raises -> Tensor:
    comptime if Dh > _DH_MAX:
        raise Error("sdpa_cross: Dh exceeds _DH_MAX (128)")

    comptime q_src_rows = B * Sq * H
    comptime kv_src_rows = B * Skv * H
    comptime q_bhsd_rows = B * H * Sq
    comptime kv_bhsd_rows = B * H * Skv
    var q_buf = ctx.enqueue_create_buffer[dtype](q_bhsd_rows * Dh)
    var k_buf = ctx.enqueue_create_buffer[dtype](kv_bhsd_rows * Dh)
    var v_buf = ctx.enqueue_create_buffer[dtype](kv_bhsd_rows * Dh)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        B * Sq * H * Dh * out_dt.byte_size()
    )

    var q_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_src_rows, Dh))
    var q_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_bhsd_rows, Dh))
    var kv_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_bhsd_rows, Dh))

    var qd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](q_buf.unsafe_ptr(), q_bhsd_rl)
    var kd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](k_buf.unsafe_ptr(), kv_bhsd_rl)
    var vd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](v_buf.unsafe_ptr(), kv_bhsd_rl)

    var nq = B * H * Sq * Dh
    var nkv = B * H * Skv * Dh
    var qgrid = (nq + _BLOCK - 1) // _BLOCK
    var kvgrid = (nkv + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](qs, qd, B, Sq, H, Dh, grid_dim=qgrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](ks, kd, B, Skv, H, Dh, grid_dim=kvgrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](vs, vd, B, Skv, H, Dh, grid_dim=kvgrid, block_dim=_BLOCK)

    var rows = B * H * Sq
    var row_grid = (rows + _TPB - 1) // _TPB
    var do_mask = 1 if apply_mask else 0
    comptime if dtype == DType.float32:
        var od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), q_src_rl)
        ctx.enqueue_function[
            _sdpa_cross_online[dtype, mask_dtype],
            _sdpa_cross_online[dtype, mask_dtype],
        ](qd, kd, vd, od, mask, scale, B, Sq, Skv, H, Dh, do_mask,
          grid_dim=row_grid, block_dim=_TPB)
    elif dtype == DType.bfloat16:
        var od = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), q_src_rl)
        ctx.enqueue_function[
            _sdpa_cross_online[dtype, mask_dtype],
            _sdpa_cross_online[dtype, mask_dtype],
        ](qd, kd, vd, od, mask, scale, B, Sq, Skv, H, Dh, do_mask,
          grid_dim=row_grid, block_dim=_TPB)
    else:
        var od = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), q_src_rl)
        ctx.enqueue_function[
            _sdpa_cross_online[dtype, mask_dtype],
            _sdpa_cross_online[dtype, mask_dtype],
        ](qd, kd, vd, od, mask, scale, B, Sq, Skv, H, Dh, do_mask,
          grid_dim=row_grid, block_dim=_TPB)

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(Sq)
    out_shape.append(H)
    out_shape.append(Dh)
    return Tensor(out_buf^, out_shape^, out_dt)


def _sdpa_cross_tiled[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
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
    if apply_mask and q.dtype() != mask.dtype():
        raise Error("sdpa_cross: q/mask dtype mismatch")
    comptime q_src_rows = B * Sq * H
    comptime kv_src_rows = B * Skv * H
    comptime mask_rows = B * H * Sq
    var q_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_src_rows, Dh))
    var kv_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_src_rows, Dh))
    var mask_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](mask_rows, Skv))
    var dummy_buf = ctx.enqueue_create_buffer[DType.float32](1)
    var dummy_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](1, 1))
    var dummy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dummy_buf.unsafe_ptr(), dummy_rl)
    if dt == DType.float32:
        var qs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float32](), q_src_rl)
        var ks = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float32](), kv_src_rl)
        var vs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float32](), kv_src_rl)
        if apply_mask:
            var ms = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                mask.buf.unsafe_ptr().bitcast[Float32](), mask_rl)
            return _sdpa_cross_tiled_storage[
                B, Sq, Skv, H, Dh, DType.float32, DType.float32
            ](
                qs, ks, vs, ms, scale, ctx, True, q.dtype())
        return _sdpa_cross_tiled_storage[
            B, Sq, Skv, H, Dh, DType.float32, DType.float32
        ](
            qs, ks, vs, dummy, scale, ctx, False, q.dtype())
    elif dt == DType.bfloat16:
        var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), q_src_rl)
        var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), kv_src_rl)
        var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), kv_src_rl)
        if apply_mask:
            var ms = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                mask.buf.unsafe_ptr().bitcast[BFloat16](), mask_rl)
            return _sdpa_cross_tiled_storage[
                B, Sq, Skv, H, Dh, DType.bfloat16, DType.bfloat16
            ](
                qs, ks, vs, ms, scale, ctx, True, q.dtype())
        return _sdpa_cross_tiled_storage[
            B, Sq, Skv, H, Dh, DType.bfloat16, DType.float32
        ](
            qs, ks, vs, dummy, scale, ctx, False, q.dtype())
    else:
        var qs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), q_src_rl)
        var ks = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), kv_src_rl)
        var vs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), kv_src_rl)
        if apply_mask:
            var ms = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                mask.buf.unsafe_ptr().bitcast[Float16](), mask_rl)
            return _sdpa_cross_tiled_storage[
                B, Sq, Skv, H, Dh, DType.float16, DType.float16
            ](
                qs, ks, vs, ms, scale, ctx, True, q.dtype())
        return _sdpa_cross_tiled_storage[
            B, Sq, Skv, H, Dh, DType.float16, DType.float32
        ](
            qs, ks, vs, dummy, scale, ctx, False, q.dtype())


# Matmul-backed rectangular attention. This is the production path when
# [B,H,Sq,Skv] scores fit VRAM: it uses cuBLAS for QK^T and P@V instead of the
# scalar online kernel. Large stage-2 self-attention still uses online tiling.
def _sdpa_cross_math_storage[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int, dtype: DType
](
    qs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    ks: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    vs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    scale: Float32,
    ctx: DeviceContext,
    out_dt: STDtype,
) raises -> Tensor:
    comptime BH = B * H

    comptime q_src_rows = B * Sq * H
    comptime kv_src_rows = B * Skv * H
    comptime q_bhsd_rows = B * H * Sq
    comptime kv_bhsd_rows = B * H * Skv
    var q_buf = ctx.enqueue_create_buffer[dtype](q_bhsd_rows * Dh)
    var k_buf = ctx.enqueue_create_buffer[dtype](kv_bhsd_rows * Dh)
    var v_buf = ctx.enqueue_create_buffer[dtype](kv_bhsd_rows * Dh)

    var q_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_src_rows, Dh))
    var q_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_bhsd_rows, Dh))
    var kv_bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_bhsd_rows, Dh))

    var qd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](q_buf.unsafe_ptr(), q_bhsd_rl)
    var kd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](k_buf.unsafe_ptr(), kv_bhsd_rl)
    var vd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](v_buf.unsafe_ptr(), kv_bhsd_rl)

    var nq = B * H * Sq * Dh
    var nkv = B * H * Skv * Dh
    var qgrid = (nq + _BLOCK - 1) // _BLOCK
    var kvgrid = (nkv + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](qs, qd, B, Sq, H, Dh, grid_dim=qgrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](ks, kd, B, Skv, H, Dh, grid_dim=kvgrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](vs, vd, B, Skv, H, Dh, grid_dim=kvgrid, block_dim=_BLOCK)

    var scores = ctx.enqueue_create_buffer[DType.float32](BH * Sq * Skv)
    var q_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Dh))
    var kv_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Skv, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Skv))
    var qptr = q_buf.unsafe_ptr()
    var kptr = k_buf.unsafe_ptr()
    var scptr = scores.unsafe_ptr()
    for bh in range(BH):
        var Q = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            qptr + bh * Sq * Dh, q_head_rl
        )
        var Kt = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            kptr + bh * Skv * Dh, kv_head_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            scptr + bh * Sq * Skv, sc_rl
        )
        matmul(ctx, C, Q, Kt, transpose_b=True, c_row_major=True)

    comptime sm_rows = BH * Sq
    var sc_full_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, Skv))
    var sc_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        scptr, sc_full_rl
    )
    var nsm = sm_rows * Skv
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_f32, _scale_f32](
        sc_full, scale, sm_rows, Skv, grid_dim=smgrid, block_dim=_BLOCK)
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        sc_full, Skv, grid_dim=sm_rows, block_dim=_TPB)

    var out_f32 = ctx.enqueue_create_buffer[DType.float32](q_bhsd_rows * Dh)
    var optr = out_f32.unsafe_ptr()
    comptime if dtype == DType.float32:
        var vptr = v_buf.unsafe_ptr().bitcast[Float32]()
        for bh in range(BH):
            var P = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                scptr + bh * Sq * Skv, sc_rl
            )
            var Vh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                vptr + bh * Skv * Dh, kv_head_rl
            )
            var Oh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                optr + bh * Sq * Dh, q_head_rl
            )
            matmul(ctx, Oh, P, Vh, transpose_b=False, c_row_major=True)
    else:
        var v_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            v_buf.unsafe_ptr(), kv_bhsd_rl
        )
        var o_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            optr, q_bhsd_rl
        )
        var pv_total = B * H * Sq * Dh
        var pv_grid = (pv_total + _BLOCK - 1) // _BLOCK
        ctx.enqueue_function[
            _attn_pv_kernel[dtype], _attn_pv_kernel[dtype]
        ](sc_full, v_full, o_full, B, Sq, Skv, H, Dh, grid_dim=pv_grid, block_dim=_BLOCK)

    var bsz = out_dt.byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * Sq * H * Dh * bsz)
    var out_src = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        optr, q_bhsd_rl)
    var scgrid = (nq + _BLOCK - 1) // _BLOCK
    comptime if dtype == DType.float32:
        var Od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), q_src_rl)
        ctx.enqueue_function[_scatter_bhsd_to_bshd_f32, _scatter_bhsd_to_bshd_f32](
            out_src, Od, B, Sq, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    elif dtype == DType.bfloat16:
        var Od = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), q_src_rl)
        ctx.enqueue_function[_scatter_bhsd_to_bshd_bf16, _scatter_bhsd_to_bshd_bf16](
            out_src, Od, B, Sq, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    else:
        var Od = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), q_src_rl)
        ctx.enqueue_function[_scatter_bhsd_to_bshd_f16, _scatter_bhsd_to_bshd_f16](
            out_src, Od, B, Sq, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(Sq)
    out_shape.append(H)
    out_shape.append(Dh)
    return Tensor(out_buf^, out_shape^, out_dt)


def _sdpa_cross_math[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var dt = q.dtype().to_mojo_dtype()
    comptime q_src_rows = B * Sq * H
    comptime kv_src_rows = B * Skv * H
    var q_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_src_rows, Dh))
    var kv_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_src_rows, Dh))
    if dt == DType.float32:
        var qs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float32](), q_src_rl)
        var ks = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float32](), kv_src_rl)
        var vs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float32](), kv_src_rl)
        return _sdpa_cross_math_storage[B, Sq, Skv, H, Dh, DType.float32](
            qs, ks, vs, scale, ctx, q.dtype())
    elif dt == DType.bfloat16:
        var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), q_src_rl)
        var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), kv_src_rl)
        var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), kv_src_rl)
        return _sdpa_cross_math_storage[B, Sq, Skv, H, Dh, DType.bfloat16](
            qs, ks, vs, scale, ctx, q.dtype())
    else:
        var qs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), q_src_rl)
        var ks = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), kv_src_rl)
        var vs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), kv_src_rl)
        return _sdpa_cross_math_storage[B, Sq, Skv, H, Dh, DType.float16](
            qs, ks, vs, scale, ctx, q.dtype())


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


def sdpa_chunked[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Head-chunked math SDPA — BIT-IDENTICAL output to `sdpa`, smaller peak.

    Same contract as `sdpa` (full non-causal attention, additive [B,H,S,S] mask),
    but the F32 scores slab is held for ONE head at a time ([S,S]) instead of all
    heads ([B*H,S,S]). Use when the [B*H,S,S] F32 scores would dominate VRAM
    (e.g. Lens 1024: H=24, S=4297 → 1.77 GB scores). Heads are independent so the
    per-head QKᵀ/scale+mask/softmax/P@V is the exact same arithmetic as `sdpa`;
    the result matches to the bit.
    """
    if q.dtype() != k.dtype() or q.dtype() != v.dtype():
        raise Error("sdpa_chunked: q/k/v dtype mismatch")
    if q.dtype() != mask.dtype():
        raise Error("sdpa_chunked: q/mask dtype mismatch")
    var qshape = q.shape()
    if len(qshape) != 4:
        raise Error("sdpa_chunked: q must be rank-4 [B,S,H,Dh]")
    if (
        qshape[0] != B or qshape[1] != S or qshape[2] != H or qshape[3] != Dh
    ):
        raise Error("sdpa_chunked: q shape does not match compile-time B/S/H/Dh")
    var mshape = mask.shape()
    if (
        len(mshape) != 4
        or mshape[0] != B or mshape[1] != H
        or mshape[2] != S or mshape[3] != S
    ):
        raise Error("sdpa_chunked: mask must be [B,H,S,S]")
    return _sdpa_math_chunked[B, S, H, Dh](q, k, v, mask, scale, ctx, True)


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

    mask must match q/k/v storage dtype [B,H,S,S]. Dh <= 128. Existing sdpa / flash paths are
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


def sdpa_cross_nomask[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Rectangular full attention without an additive mask.

    Q: [B,Sq,H,Dh], K/V: [B,Skv,H,Dh], output: [B,Sq,H,Dh].
    This is the production path for LTX2 text cross-attention and AV cross-modal
    attention. When the score slab fits the explicit memory budget, it uses
    matmul-backed attention; otherwise it falls back to online softmax without
    square padding.
    """
    if q.dtype() != k.dtype() or q.dtype() != v.dtype():
        raise Error("sdpa_cross_nomask: q/k/v dtype mismatch")
    var qshape = q.shape()
    var kshape = k.shape()
    var vshape = v.shape()
    if len(qshape) != 4 or len(kshape) != 4 or len(vshape) != 4:
        raise Error("sdpa_cross_nomask: q/k/v must be rank-4 [B,S,H,Dh]")
    if qshape[0] != B or qshape[1] != Sq or qshape[2] != H or qshape[3] != Dh:
        raise Error("sdpa_cross_nomask: q shape mismatch")
    if kshape[0] != B or kshape[1] != Skv or kshape[2] != H or kshape[3] != Dh:
        raise Error("sdpa_cross_nomask: k shape mismatch")
    if vshape[0] != B or vshape[1] != Skv or vshape[2] != H or vshape[3] != Dh:
        raise Error("sdpa_cross_nomask: v shape mismatch")
    comptime score_mib = (B * H * Sq * Skv * 4) // (1024 * 1024)
    comptime if score_mib < _MATH_SCORE_BUDGET_MIB:
        return _sdpa_cross_math[B, Sq, Skv, H, Dh](q, k, v, scale, ctx)
    else:
        return _sdpa_cross_tiled[B, Sq, Skv, H, Dh](q, k, v, q, scale, ctx, False)


def sdpa_cross_masked[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Rectangular full attention with an additive score mask.

    Q: [B,Sq,H,Dh], K/V: [B,Skv,H,Dh], mask: [B,H,Sq,Skv],
    output: [B,Sq,H,Dh]. This is the mask-capable sibling needed by LTX2
    IC/control/audio-reference paths. It streams K/V with online softmax and
    never pads to a square sequence or allocates [Sq,Skv] scores.
    """
    if q.dtype() != k.dtype() or q.dtype() != v.dtype():
        raise Error("sdpa_cross_masked: q/k/v dtype mismatch")
    if q.dtype() != mask.dtype():
        raise Error("sdpa_cross_masked: q/mask dtype mismatch")
    var qshape = q.shape()
    var kshape = k.shape()
    var vshape = v.shape()
    var mshape = mask.shape()
    if len(qshape) != 4 or len(kshape) != 4 or len(vshape) != 4:
        raise Error("sdpa_cross_masked: q/k/v must be rank-4 [B,S,H,Dh]")
    if qshape[0] != B or qshape[1] != Sq or qshape[2] != H or qshape[3] != Dh:
        raise Error("sdpa_cross_masked: q shape mismatch")
    if kshape[0] != B or kshape[1] != Skv or kshape[2] != H or kshape[3] != Dh:
        raise Error("sdpa_cross_masked: k shape mismatch")
    if vshape[0] != B or vshape[1] != Skv or vshape[2] != H or vshape[3] != Dh:
        raise Error("sdpa_cross_masked: v shape mismatch")
    if (
        len(mshape) != 4
        or mshape[0] != B or mshape[1] != H
        or mshape[2] != Sq or mshape[3] != Skv
    ):
        raise Error("sdpa_cross_masked: mask must be [B,H,Sq,Skv]")
    return _sdpa_cross_tiled[B, Sq, Skv, H, Dh](q, k, v, mask, scale, ctx, True)


# ─────────────────────────────────────────────────────────────────────────────
# StepSlab variants (autograd_v2 contract C8, Phase P4): byte-identical math
# to the originals above — same kernels, same launch params, same per-head
# matmul loops; ONLY the allocation source changes (typed MAX buffers become
# uint8 slab views bitcast to the same element pointers).
# ─────────────────────────────────────────────────────────────────────────────


def _sdpa_math_storage_slab[
    B: Int, S: Int, H: Int, Dh: Int, dtype: DType
](
    qs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    ks: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    vs: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    mask: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    scale: Float32,
    ctx: DeviceContext,
    apply_mask: Bool,
    out_dt: STDtype,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `_sdpa_math_storage` (this file :260)."""
    comptime BH = B * H
    var esz = 4
    comptime if dtype == DType.bfloat16 or dtype == DType.float16:
        esz = 2

    # ── 1) gather q/k/v BSHD -> BHSD-contiguous storage [B*H*S, Dh] ──────────
    comptime src_rows = B * S * H
    comptime bhsd_rows = B * H * S
    var q_buf = slab.alloc(bhsd_rows * Dh * esz)
    var k_buf = slab.alloc(bhsd_rows * Dh * esz)
    var v_buf = slab.alloc(bhsd_rows * Dh * esz)
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bhsd_rows, Dh))

    var qd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
        q_buf.unsafe_ptr().bitcast[Scalar[dtype]](), bhsd_rl)
    var kd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
        k_buf.unsafe_ptr().bitcast[Scalar[dtype]](), bhsd_rl)
    var vd = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
        v_buf.unsafe_ptr().bitcast[Scalar[dtype]](), bhsd_rl)
    var ngather = B * H * S * Dh
    var ggrid = (ngather + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](qs, qd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](ks, kd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)
    ctx.enqueue_function[
        _gather_bshd_to_bhsd[dtype], _gather_bshd_to_bhsd[dtype]
    ](vs, vd, B, S, H, Dh, grid_dim=ggrid, block_dim=_BLOCK)

    # ── 2) QKᵀ per head -> scores F32 [B*H, S, S] ────────────────────────────
    var scores = slab.alloc(BH * S * S * 4)
    var head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, S))
    var qptr = q_buf.unsafe_ptr().bitcast[Scalar[dtype]]()
    var kptr = k_buf.unsafe_ptr().bitcast[Scalar[dtype]]()
    var scptr = scores.unsafe_ptr().bitcast[Float32]()
    for bh in range(BH):
        var A = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            qptr + bh * S * Dh, head_rl
        )
        var Bt = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
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
        ctx.enqueue_function[_scale_mask[dtype], _scale_mask[dtype]](
            sc_full, mask, scale, sm_rows, S, grid_dim=smgrid, block_dim=_BLOCK)
    else:
        ctx.enqueue_function[_scale_f32, _scale_f32](
            sc_full, scale, sm_rows, S, grid_dim=smgrid, block_dim=_BLOCK)

    # ── 4) softmax over last dim (j) in place: one block per [B*H*S] row ─────
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        sc_full, S, grid_dim=sm_rows, block_dim=_TPB)

    # ── 5) P @ V per head -> out F32 BHSD [B*H, S, Dh] ───────────────────────
    var out_f32 = slab.alloc(bhsd_rows * Dh * 4)
    var optr = out_f32.unsafe_ptr().bitcast[Float32]()
    comptime if dtype == DType.float32:
        var vptr = v_buf.unsafe_ptr().bitcast[Float32]()
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
            matmul(ctx, Oh, P, Vh, transpose_b=False, c_row_major=True)
    else:
        var v_full = LayoutTensor[dtype, _DYN2, MutAnyOrigin](
            v_buf.unsafe_ptr().bitcast[Scalar[dtype]](), bhsd_rl
        )
        var o_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            optr, bhsd_rl
        )
        var pv_total = B * H * S * Dh
        var pv_grid = (pv_total + _BLOCK - 1) // _BLOCK
        ctx.enqueue_function[
            _attn_pv_kernel[dtype], _attn_pv_kernel[dtype]
        ](sc_full, v_full, o_full, B, S, S, H, Dh, grid_dim=pv_grid, block_dim=_BLOCK)

    # ── 6) scatter BHSD F32 -> BSHD output in storage dtype ──────────────────
    var bsz = out_dt.byte_size()
    var out_buf = slab.alloc(B * S * H * Dh * bsz)
    var out_src = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        optr, bhsd_rl
    )
    var nsc = B * H * S * Dh
    var scgrid = (nsc + _BLOCK - 1) // _BLOCK
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    comptime if dtype == DType.float32:
        var Od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        ctx.enqueue_function[_scatter_bhsd_to_bshd_f32, _scatter_bhsd_to_bshd_f32](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    elif dtype == DType.bfloat16:
        var Od = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        ctx.enqueue_function[_scatter_bhsd_to_bshd_bf16, _scatter_bhsd_to_bshd_bf16](
            out_src, Od, B, S, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    else:
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
    return Tensor(out_buf^, out_shape^, out_dt)


def _sdpa_math_slab[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    ctx: DeviceContext,
    apply_mask: Bool,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `_sdpa_math` (this file :402)."""
    var dt = q.dtype().to_mojo_dtype()
    comptime src_rows = B * S * H
    comptime sm_rows = B * H * S
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var mask_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, S))
    if dt == DType.float32:
        var qs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var ks = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var vs = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var ms = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[Float32](), mask_rl)
        return _sdpa_math_storage_slab[B, S, H, Dh, DType.float32](
            qs, ks, vs, ms, scale, ctx, apply_mask, q.dtype(), slab)
    elif dt == DType.bfloat16:
        var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var ms = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[BFloat16](), mask_rl)
        return _sdpa_math_storage_slab[B, S, H, Dh, DType.bfloat16](
            qs, ks, vs, ms, scale, ctx, apply_mask, q.dtype(), slab)
    else:
        var qs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var ks = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var vs = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var ms = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[Float16](), mask_rl)
        return _sdpa_math_storage_slab[B, S, H, Dh, DType.float16](
            qs, ks, vs, ms, scale, ctx, apply_mask, q.dtype(), slab)


def sdpa_nomask_slab[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `sdpa_nomask` (this file :1479) — same math-mode
    path via _sdpa_math_slab; ONLY the allocation source changes."""
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

    return _sdpa_math_slab[B, S, H, Dh](q, k, v, q, scale, ctx, False, slab)
