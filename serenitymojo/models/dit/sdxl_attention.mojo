# models/dit/sdxl_attention.mojo — SDXL-local rectangular SDPA (no mask).
#
# The foundation ops/attention.sdpa requires q-seq == kv-seq (its mask is a
# SQUARE [B,H,S,S] and the layout params force one S). SDXL cross-attention has
# q-seq = spatial HW != kv-seq = 77 text tokens, so it does not fit. SDXL
# attention is also UNMASKED (full attention, no causal/pad mask). So this is an
# SDXL-LOCAL math-mode SDPA: rectangular scores [Sq, Skv], no mask.
#
# Mirrors the foundation _sdpa_math interior exactly (gather BSHD->BHSD-contig
# F32, per-head QKᵀ matmul + scale, last-dim softmax, P·V matmul, scatter back),
# but with separate Sq (query seq) and Skv (key/value seq) and no mask add.
#
# q: [B, Sq, H, Dh]   k,v: [B, Skv, H, Dh]   (BSHD; same compute dtype)
# scale: Float32 (1/sqrt(Dh))   returns [B, Sq, H, Dh] (q's dtype).
# B/H/Dh and Sq/Skv are compile-time params (the per-head matmul layouts need
# static shapes). F32 throughout the interior; BF16/F16 only at storage edges.
#
# Self-attention (Sq==Skv) routes through here too (uniform path, no mask).
#
# Mojo 1.0.0b1, NVIDIA GPU.

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


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256
comptime _NEG_BIG = Float32(-3.0e38)


# gather BSHD [B,Sx,H,Dh] -> BHSD-contig F32 [B*H, Sx, Dh].
# dst flat = (((b*H + h)*Sx + s)*Dh + d); src BSHD = (((b*Sx + s)*H + h)*Dh + d).
def _gather_f32(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, Sx: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Sx * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % Sx
        var t2 = t // Sx
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * Sx + s) * H + h
        var dst_row = (b * H + h) * Sx + s
        dst[dst_row, d] = src[src_row, d]


def _gather_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, Sx: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Sx * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % Sx
        var t2 = t // Sx
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * Sx + s) * H + h
        var dst_row = (b * H + h) * Sx + s
        var val = rebind[Scalar[DType.bfloat16]](src[src_row, d]).cast[DType.float32]()
        dst[dst_row, d] = rebind[dst.element_type](val)


def _gather_f16(
    src: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, Sx: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Sx * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % Sx
        var t2 = t // Sx
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * Sx + s) * H + h
        var dst_row = (b * H + h) * Sx + s
        var val = rebind[Scalar[DType.float16]](src[src_row, d]).cast[DType.float32]()
        dst[dst_row, d] = rebind[dst.element_type](val)


# scale the [rows, cols] scores buffer in place (no mask). rows = B*H*Sq.
def _scale_only(
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    scale: Float32,
    rows: Int, cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var v = rebind[Scalar[DType.float32]](scores[r, c]) * scale
        scores[r, c] = rebind[scores.element_type](v)


# softmax over last dim, in place on F32 scores [rows, cols]. One block/row.
def _softmax_rows(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cols: Int,
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
            var b = shared[tid + active]
            shared[tid] = a if a > b else b
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


# scatter BHSD F32 [B*H, Sq, Dh] -> BSHD storage dtype [B, Sq, H, Dh].
def _scatter_f32(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    B: Int, Sq: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Sq * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % Sq
        var t2 = t // Sq
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * H + h) * Sq + s
        var dst_row = (b * Sq + s) * H + h
        dst[dst_row, d] = src[src_row, d]


def _scatter_bf16(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    B: Int, Sq: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Sq * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % Sq
        var t2 = t // Sq
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * H + h) * Sq + s
        var dst_row = (b * Sq + s) * H + h
        var v = rebind[Scalar[DType.float32]](src[src_row, d])
        dst[dst_row, d] = rebind[dst.element_type](v.cast[DType.bfloat16]())


def _scatter_f16(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    B: Int, Sq: Int, H: Int, Dh: Int,
):
    var idx = Int(global_idx.x)
    var total = B * H * Sq * Dh
    if idx < total:
        var d = idx % Dh
        var t = idx // Dh
        var s = t % Sq
        var t2 = t // Sq
        var h = t2 % H
        var b = t2 // H
        var src_row = (b * H + h) * Sq + s
        var dst_row = (b * Sq + s) * H + h
        var v = rebind[Scalar[DType.float32]](src[src_row, d])
        dst[dst_row, d] = rebind[dst.element_type](v.cast[DType.float16]())


def _gather_dispatch(
    x: Tensor, B: Int, Sx: Int, H: Int, Dh: Int,
    dst_buf: DeviceBuffer[DType.float32], ctx: DeviceContext,
) raises:
    var dt = x.dtype().to_mojo_dtype()
    var src_rows = B * Sx * H
    var bhsd_rows = B * H * Sx
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](src_rows, Dh))
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bhsd_rows, Dh))
    var dst = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dst_buf.unsafe_ptr(), bhsd_rl
    )
    var n = B * H * Sx * Dh
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var s = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        ctx.enqueue_function[_gather_f32, _gather_f32](
            s, dst, B, Sx, H, Dh, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var s = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        ctx.enqueue_function[_gather_bf16, _gather_bf16](
            s, dst, B, Sx, H, Dh, grid_dim=grid, block_dim=_BLOCK)
    else:
        var s = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        ctx.enqueue_function[_gather_f16, _gather_f16](
            s, dst, B, Sx, H, Dh, grid_dim=grid, block_dim=_BLOCK)


# Rectangular math-mode SDPA. comptime B/H/Dh + Sq/Skv.
def sdxl_sdpa[
    B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int
](
    q: Tensor, k: Tensor, v: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    if q.dtype() != k.dtype() or q.dtype() != v.dtype():
        raise Error("sdxl_sdpa: q/k/v dtype mismatch")
    var qs = q.shape()
    if len(qs) != 4 or qs[0] != B or qs[1] != Sq or qs[2] != H or qs[3] != Dh:
        raise Error("sdxl_sdpa: q must be [B,Sq,H,Dh]")
    var ks = k.shape()
    if len(ks) != 4 or ks[0] != B or ks[1] != Skv or ks[2] != H or ks[3] != Dh:
        raise Error("sdxl_sdpa: k must be [B,Skv,H,Dh]")

    comptime BH = B * H
    var q_f32 = ctx.enqueue_create_buffer[DType.float32](B * H * Sq * Dh)
    var k_f32 = ctx.enqueue_create_buffer[DType.float32](B * H * Skv * Dh)
    var v_f32 = ctx.enqueue_create_buffer[DType.float32](B * H * Skv * Dh)
    _gather_dispatch(q, B, Sq, H, Dh, q_f32, ctx)
    _gather_dispatch(k, B, Skv, H, Dh, k_f32, ctx)
    _gather_dispatch(v, B, Skv, H, Dh, v_f32, ctx)

    # QKᵀ per head -> scores F32 [B*H, Sq, Skv].
    var scores = ctx.enqueue_create_buffer[DType.float32](BH * Sq * Skv)
    var q_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Dh))
    var k_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Skv, Dh))
    var sc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Skv))
    var qptr = q_f32.unsafe_ptr()
    var kptr = k_f32.unsafe_ptr()
    var scptr = scores.unsafe_ptr()
    for bh in range(BH):
        var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            qptr + bh * Sq * Dh, q_head_rl
        )
        var Bt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            kptr + bh * Skv * Dh, k_head_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            scptr + bh * Sq * Skv, sc_rl
        )
        # C[Sq,Skv] = A[Sq,Dh] @ Bt[Skv,Dh]ᵀ.
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)

    # scale (no mask), then softmax over last dim (Skv).
    var sm_rows = BH * Sq
    var sc_full_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, Skv))
    var sc_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        scptr, sc_full_rl
    )
    var nsm = sm_rows * Skv
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scale_only, _scale_only](
        sc_full, scale, sm_rows, Skv, grid_dim=smgrid, block_dim=_BLOCK)
    ctx.enqueue_function[_softmax_rows, _softmax_rows](
        sc_full, Skv, grid_dim=sm_rows, block_dim=_TPB)

    # P @ V per head -> out F32 BHSD [B*H, Sq, Dh].
    var out_f32 = ctx.enqueue_create_buffer[DType.float32](B * H * Sq * Dh)
    var optr = out_f32.unsafe_ptr()
    var vptr = v_f32.unsafe_ptr()
    var v_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Skv, Dh))
    var o_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](Sq, Dh))
    for bh in range(BH):
        var P = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            scptr + bh * Sq * Skv, sc_rl
        )
        var Vh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            vptr + bh * Skv * Dh, v_head_rl
        )
        var Oh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            optr + bh * Sq * Dh, o_head_rl
        )
        matmul(ctx, Oh, P, Vh, transpose_b=False, c_row_major=True)

    # scatter BHSD F32 -> BSHD output (storage dtype).
    var dt = q.dtype().to_mojo_dtype()
    var bsz = q.dtype().byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * Sq * H * Dh * bsz)
    var bhsd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](B * H * Sq, Dh))
    var dst_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](B * Sq * H, Dh))
    var out_src = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        optr, bhsd_rl
    )
    var nsc = B * H * Sq * Dh
    var scgrid = (nsc + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var Od = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), dst_rl
        )
        ctx.enqueue_function[_scatter_f32, _scatter_f32](
            out_src, Od, B, Sq, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var Od = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), dst_rl
        )
        ctx.enqueue_function[_scatter_bf16, _scatter_bf16](
            out_src, Od, B, Sq, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    else:
        var Od = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), dst_rl
        )
        ctx.enqueue_function[_scatter_f16, _scatter_f16](
            out_src, Od, B, Sq, H, Dh, grid_dim=scgrid, block_dim=_BLOCK)
    ctx.synchronize()

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(Sq)
    out_shape.append(H)
    out_shape.append(Dh)
    return Tensor(out_buf^, out_shape^, q.dtype())
