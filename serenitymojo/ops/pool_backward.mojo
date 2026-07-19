# ops/pool_backward.mojo — naive BACKWARD for MaxPool2D + UpsampleNearest2D
# (the VAE / decoder path). Tier-5 de-risk: hand-written naive GPU kernels (one
# thread per INPUT element), F32 interior, no shared memory, no atomics —
# correctness first, parity-gated vs PyTorch (cos >= 0.999).
#
# ── LAYOUT (MUST match the forwards EXACTLY) ─────────────────────────────────
#   NHWC throughout (matches ops/conv.mojo + models/vae/upsample.mojo).
#
# ── MaxPool2D ────────────────────────────────────────────────────────────────
#   Forward (F.max_pool2d on NCHW, == NHWC here per channel):
#     y[n,oh,ow,c] = max_{kh,kw} x[n, oh*sh + kh, ow*sw + kw, c]
#   (no padding — the VAE downsample pools have padding=0; the brief's pooling
#   path is the diffusers VAE MaxPool which is pad=0. Matches F.max_pool2d
#   default padding=0, dilation=1.)
#   Backward: route the upstream grad to the ARGMAX position in each window.
#     d_x[n,ih,iw,c] += grad_y[n,oh,ow,c]   iff (ih,iw) is the argmax of that
#     window. PyTorch routes the FULL grad to the FIRST max (lowest flat
#     window index = row-major (kh,kw) scan); we recompute argmax from x with
#     the SAME first-max tie-break so d_x is byte-position-comparable to torch.
#   One thread per INPUT element (n,ih,iw,c): scan every window that COVERS this
#   input pixel, and for each such window recompute its argmax; if this pixel is
#   that argmax, accumulate grad_y of that window. No atomics needed (each input
#   element is written by exactly one thread).
#
# ── UpsampleNearest2D ────────────────────────────────────────────────────────
#   Forward (models/vae/upsample.mojo, nearest replication, integer scale):
#     out[n, ih*scale + ..., iw*scale + ..., c] = in[n, ih, iw, c]
#     i.e. out[n,oh,ow,c] = in[n, oh//scale, ow//scale, c].
#   Backward (inverse of the "duplicate" forward): each INPUT cell receives the
#   SUM of the `scale*scale` grad_out cells it was broadcast to.
#     d_x[n,ih,iw,c] = sum_{dh in 0..scale, dw in 0..scale}
#                          grad_out[n, ih*scale+dh, iw*scale+dw, c]
#   One thread per INPUT element. d_x shape == input shape [N,in_h,in_w,C].
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16/F16 storage is read directly, F32 is used only
# for scalar routing/sum math inside the kernel, and d_x stores back to the
# activation dtype. Single d_x each, so plain `def` returning a `Tensor` (no
# Movable multi-output struct needed).

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _NEG_BIG = Float32(-3.0e38)


# ── MaxPool2D backward kernel: one thread per INPUT element (n,ih,iw,c) ───────
# Scans every output window that covers (ih,iw); for each window recomputes its
# argmax from x (first-max tie-break, row-major (kh,kw) scan, == PyTorch) and,
# if (ih,iw) is that argmax, accumulates the window's grad_y.
def _maxpool2d_dx_kernel[dtype: DType](
    grad_y: LayoutTensor[dtype, _DYN1, MutAnyOrigin],  # [N*Ho*Wo*C]
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],       # [N*Hi*Wi*C]
    d_x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],     # [N*Hi*Wi*C]
    N: Int, Hi: Int, Wi: Int, C: Int,
    Kh: Int, Kw: Int, Sh: Int, Sw: Int,
    Ho: Int, Wo: Int,
):
    var idx = Int(global_idx.x)
    var total = N * Hi * Wi * C
    if idx >= total:
        return
    # decode NHWC index idx -> (n, ih, iw, c)
    var c = idx % C
    var t = idx // C
    var iw = t % Wi
    var t2 = t // Wi
    var ih = t2 % Hi
    var n = t2 // Hi

    var acc: Float32 = 0.0
    # Output windows that COULD include input row ih: oh*Sh <= ih <= oh*Sh+Kh-1
    #   ⇒ (ih - Kh + 1)/Sh <= oh <= ih/Sh, clamped to [0, Ho).
    var oh_hi = ih // Sh
    if oh_hi >= Ho:
        oh_hi = Ho - 1
    var oh_lo_num = ih - Kh + 1
    var oh_lo = 0
    if oh_lo_num > 0:
        # ceil division by Sh
        oh_lo = (oh_lo_num + Sh - 1) // Sh
    for oh in range(oh_lo, oh_hi + 1):
        var kh = ih - oh * Sh
        if kh < 0 or kh >= Kh:
            continue
        var ow_hi = iw // Sw
        if ow_hi >= Wo:
            ow_hi = Wo - 1
        var ow_lo_num = iw - Kw + 1
        var ow_lo = 0
        if ow_lo_num > 0:
            ow_lo = (ow_lo_num + Sw - 1) // Sw
        for ow in range(ow_lo, ow_hi + 1):
            var kw = iw - ow * Sw
            if kw < 0 or kw >= Kw:
                continue
            # recompute argmax of window (oh,ow) over x, first-max tie-break
            var best: Float32 = _NEG_BIG
            var best_kh = 0
            var best_kw = 0
            for wkh in range(Kh):
                var xih = oh * Sh + wkh
                for wkw in range(Kw):
                    var xiw = ow * Sw + wkw
                    var xoff = (((n * Hi + xih) * Wi + xiw) * C) + c
                    var v = rebind[Scalar[dtype]](x[xoff]).cast[DType.float32]()
                    if v > best:
                        best = v
                        best_kh = wkh
                        best_kw = wkw
            # if THIS input pixel is the window's argmax, take the grad
            if best_kh == kh and best_kw == kw:
                var goff = (((n * Ho + oh) * Wo + ow) * C) + c
                acc += rebind[Scalar[dtype]](grad_y[goff]).cast[DType.float32]()
    d_x[idx] = rebind[d_x.element_type](acc.cast[dtype]())


# ── UpsampleNearest2D backward kernel: one thread per INPUT element ───────────
# d_x[n,ih,iw,c] = sum over the scale*scale grad_out cells it was broadcast to.
def _upsample_nearest_dx_kernel[dtype: DType](
    grad_out: LayoutTensor[dtype, _DYN1, MutAnyOrigin],  # [N*Ho*Wo*C]
    d_x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],       # [N*Hi*Wi*C]
    N: Int, Hi: Int, Wi: Int, C: Int,
    scale: Int, Ho: Int, Wo: Int,
):
    var idx = Int(global_idx.x)
    var total = N * Hi * Wi * C
    if idx >= total:
        return
    var c = idx % C
    var t = idx // C
    var iw = t % Wi
    var t2 = t // Wi
    var ih = t2 % Hi
    var n = t2 // Hi

    var acc: Float32 = 0.0
    for dh in range(scale):
        var oh = ih * scale + dh
        if oh >= Ho:
            continue
        for dw in range(scale):
            var ow = iw * scale + dw
            if ow >= Wo:
                continue
            var goff = (((n * Ho + oh) * Wo + ow) * C) + c
            acc += rebind[Scalar[dtype]](grad_out[goff]).cast[DType.float32]()
    d_x[idx] = rebind[d_x.element_type](acc.cast[dtype]())


def maxpool2d_backward[
    N: Int, Hi: Int, Wi: Int, C: Int,
    Kh: Int, Kw: Int, Sh: Int, Sw: Int,
](
    grad_out: Tensor,
    x: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """MaxPool2D backward (NHWC), padding=0, dilation=1, F32.

    grad_out: [N, Ho, Wo, C]  NHWC  (upstream grad, forward-output layout)
    x:        [N, Hi, Wi, C]  NHWC  (the forward INPUT; used to recompute argmax)
    returns   d_x [N, Hi, Wi, C] NHWC F32.

    Forward = max over (Kh,Kw) windows, stride (Sh,Sw), no padding. Backward
    routes grad to the window argmax (first-max tie-break == PyTorch). Naive
    one-thread-per-input-element; recomputes argmax from x (no saved indices).
    """
    comptime Ho = (Hi - Kh) // Sh + 1
    comptime Wo = (Wi - Kw) // Sw + 1

    var xshape = x.shape()
    if (
        len(xshape) != 4
        or xshape[0] != N or xshape[1] != Hi
        or xshape[2] != Wi or xshape[3] != C
    ):
        raise Error("maxpool2d_backward: x shape must match [N,Hi,Wi,C] params")
    var gshape = grad_out.shape()
    if (
        len(gshape) != 4
        or gshape[0] != N or gshape[1] != Ho
        or gshape[2] != Wo or gshape[3] != C
    ):
        raise Error(
            "maxpool2d_backward: grad_out shape must match [N,Ho,Wo,C]"
        )
    if x.dtype() != grad_out.dtype():
        raise Error("maxpool2d_backward: x/grad_out dtype mismatch")

    comptime nx = N * Hi * Wi * C
    comptime ng = N * Ho * Wo * C

    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nx))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](ng))

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](
        nx * x.dtype().byte_size()
    )

    var grid = (nx + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var xv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var gv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl
        )
        var dxv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[
            _maxpool2d_dx_kernel[DType.float32],
            _maxpool2d_dx_kernel[DType.float32],
        ](
            gv, xv, dxv,
            N, Hi, Wi, C, Kh, Kw, Sh, Sw, Ho, Wo,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var xv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var gv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), g_rl
        )
        var dxv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[
            _maxpool2d_dx_kernel[DType.bfloat16],
            _maxpool2d_dx_kernel[DType.bfloat16],
        ](
            gv, xv, dxv,
            N, Hi, Wi, C, Kh, Kw, Sh, Sw, Ho, Wo,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var xv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var gv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), g_rl
        )
        var dxv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[
            _maxpool2d_dx_kernel[DType.float16],
            _maxpool2d_dx_kernel[DType.float16],
        ](
            gv, xv, dxv,
            N, Hi, Wi, C, Kh, Kw, Sh, Sw, Ho, Wo,
            grid_dim=grid, block_dim=_BLOCK,
        )
    # sync removed (single-stream ordering; was kernel-trailing host stall)

    var dx_shape = List[Int]()
    dx_shape.append(N); dx_shape.append(Hi); dx_shape.append(Wi); dx_shape.append(C)
    return Tensor(dx_buf^, dx_shape^, x.dtype())


def upsample_nearest2d_backward[
    N: Int, in_h: Int, in_w: Int, C: Int, scale: Int,
](
    grad_out: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """UpsampleNearest2D backward (NHWC), integer `scale`, F32.

    grad_out: [N, in_h*scale, in_w*scale, C]  NHWC  (upstream grad).
    returns   d_x [N, in_h, in_w, C] NHWC F32.

    Inverse of nearest-replication forward (models/vae/upsample.mojo, scale=2):
    each input cell sums the scale*scale grad_out cells it was broadcast to.
    Naive one-thread-per-input-element.
    """
    comptime Ho = in_h * scale
    comptime Wo = in_w * scale

    var gshape = grad_out.shape()
    if (
        len(gshape) != 4
        or gshape[0] != N or gshape[1] != Ho
        or gshape[2] != Wo or gshape[3] != C
    ):
        raise Error(
            "upsample_nearest2d_backward: grad_out shape must match"
            " [N,in_h*scale,in_w*scale,C]"
        )
    comptime nx = N * in_h * in_w * C
    comptime ng = N * Ho * Wo * C

    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nx))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](ng))

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](
        nx * grad_out.dtype().byte_size()
    )

    var grid = (nx + _BLOCK - 1) // _BLOCK
    var dt = grad_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var gv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl
        )
        var dxv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[
            _upsample_nearest_dx_kernel[DType.float32],
            _upsample_nearest_dx_kernel[DType.float32],
        ](
            gv, dxv,
            N, in_h, in_w, C, scale, Ho, Wo,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var gv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), g_rl
        )
        var dxv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[
            _upsample_nearest_dx_kernel[DType.bfloat16],
            _upsample_nearest_dx_kernel[DType.bfloat16],
        ](
            gv, dxv,
            N, in_h, in_w, C, scale, Ho, Wo,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var gv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), g_rl
        )
        var dxv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[
            _upsample_nearest_dx_kernel[DType.float16],
            _upsample_nearest_dx_kernel[DType.float16],
        ](
            gv, dxv,
            N, in_h, in_w, C, scale, Ho, Wo,
            grid_dim=grid, block_dim=_BLOCK,
        )
    # sync removed (single-stream ordering; was kernel-trailing host stall)

    var dx_shape = List[Int]()
    dx_shape.append(N); dx_shape.append(in_h); dx_shape.append(in_w); dx_shape.append(C)
    return Tensor(dx_buf^, dx_shape^, grad_out.dtype())
