# ops/conv2d_backward.mojo — naive conv2d BACKWARD (d_x, d_w, d_b).
#
# Tier-5 de-risk: the Mojo SDK packages conv2d FORWARD only
# (nn.conv.conv.conv2d_gpu_naive_nhwc_rscf, wrapped in ops/conv.mojo). There is
# NO SDK conv backward, so this is hand-written: three independent naive GPU
# kernels (one thread per output element), F32 interior, no shared memory, no
# atomics. Correctness first, not speed — this gates parity vs PyTorch.
#
# ── LAYOUT (MUST match ops/conv.mojo forward EXACTLY) ────────────────────────
#   x      NHWC  [N, Hi, Wi, Cin]
#   weight RSCF  [Kh, Kw, Cin, Cout]
#   grad_y NHWC  [N, Ho, Wo, Cout]   (same layout as forward output)
#   d_x    NHWC  [N, Hi, Wi, Cin]
#   d_w    RSCF  [Kh, Kw, Cin, Cout]
#   d_b         [Cout]
# All row-major. dilation=1, num_groups=1 (matches the forward).
#
# ── Forward (cross-correlation, what F.conv2d / the SDK kernel computes) ──────
#   y[n,oh,ow,co] = b[co]
#       + sum_{kh,kw,ci} x[n, oh*sh - ph + kh, ow*sw - pw + kw, ci] * w[kh,kw,ci,co]
#   (out-of-bounds input reads are 0 — implicit zero padding.)
#
# ── Gradients ────────────────────────────────────────────────────────────────
#   d_b[co]            = sum_{n,oh,ow} grad_y[n,oh,ow,co]
#   d_w[kh,kw,ci,co]   = sum_{n,oh,ow} grad_y[n,oh,ow,co]
#                          * x[n, oh*sh-ph+kh, ow*sw-pw+kw, ci]   (skip OOB x = 0)
#   d_x[n,ih,iw,ci]    = sum_{co,kh,kw} grad_y[n,oh,ow,co] * w[kh,kw,ci,co]
#                        for every (oh,ow,kh,kw) s.t. ih = oh*sh-ph+kh,
#                                                      iw = ow*sw-pw+kw  (so oh,ow integral & in-range)
#   (this d_x is the gradient of the cross-correlation w.r.t. input — derived
#    directly from the forward sum, NOT torch conv_transpose-with-bias.)
#
# Mojo 1.0.0b1, NVIDIA GPU. Storage dtype is preserved; each thread accumulates
# in F32 scalar registers and stores the final gradient in the input/weight dtype.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256


struct Conv2dBwd(Movable):
    """Backward outputs of `conv2d_backward`: gradients wrt x, weight, bias.

    d_x: [N, Hi, Wi, Cin]   NHWC
    d_w: [Kh, Kw, Cin, Cout] RSCF
    d_b: [Cout]
    """

    var d_x: Tensor
    var d_w: Tensor
    var d_b: Tensor

    def __init__(out self, var d_x: Tensor, var d_w: Tensor, var d_b: Tensor):
        self.d_x = d_x^
        self.d_w = d_w^
        self.d_b = d_b^


# ── d_x kernel: one thread per input element (n, ih, iw, ci) ─────────────────
# d_x[n,ih,iw,ci] = sum over (co, kh, kw) of grad_y[n,oh,ow,co] * w[kh,kw,ci,co]
# where the forward maps oh*sh - ph + kh = ih  ⇒  oh = (ih + ph - kh) / sh
# (only when divisible by sh and 0 <= oh < Ho; same for ow).
def _conv2d_dx_kernel[dtype: DType](
    grad_y: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [N*Ho*Wo*Cout]
    weight: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [Kh*Kw*Cin*Cout]
    d_x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],      # [N*Hi*Wi*Cin]
    N: Int, Hi: Int, Wi: Int, Cin: Int,
    Kh: Int, Kw: Int, Cout: Int,
    Ho: Int, Wo: Int,
    sh: Int, sw: Int, ph: Int, pw: Int,
):
    var idx = Int(global_idx.x)
    var total = N * Hi * Wi * Cin
    if idx >= total:
        return
    # decode NHWC index idx -> (n, ih, iw, ci)
    var ci = idx % Cin
    var t = idx // Cin
    var iw = t % Wi
    var t2 = t // Wi
    var ih = t2 % Hi
    var n = t2 // Hi

    var acc: Float32 = 0.0
    for kh in range(Kh):
        var num_h = ih + ph - kh
        if num_h < 0:
            continue
        if num_h % sh != 0:
            continue
        var oh = num_h // sh
        if oh < 0 or oh >= Ho:
            continue
        for kw in range(Kw):
            var num_w = iw + pw - kw
            if num_w < 0:
                continue
            if num_w % sw != 0:
                continue
            var ow = num_w // sw
            if ow < 0 or ow >= Wo:
                continue
            # accumulate over output channels
            var gy_base = ((n * Ho + oh) * Wo + ow) * Cout
            var w_base = ((kh * Kw + kw) * Cin + ci) * Cout
            for co in range(Cout):
                var g = rebind[Scalar[dtype]](grad_y[gy_base + co]).cast[
                    DType.float32
                ]()
                var wv = rebind[Scalar[dtype]](weight[w_base + co]).cast[
                    DType.float32
                ]()
                acc += g * wv
    d_x[idx] = rebind[d_x.element_type](acc.cast[dtype]())


# ── d_w kernel: one thread per weight element (kh, kw, ci, co) ────────────────
# d_w[kh,kw,ci,co] = sum over (n, oh, ow) of grad_y[n,oh,ow,co]
#                      * x[n, oh*sh-ph+kh, ow*sw-pw+kw, ci]   (skip OOB x)
def _conv2d_dw_kernel[dtype: DType](
    grad_y: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [N*Ho*Wo*Cout]
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],        # [N*Hi*Wi*Cin]
    d_w: LayoutTensor[dtype, _DYN1, MutAnyOrigin],      # [Kh*Kw*Cin*Cout]
    N: Int, Hi: Int, Wi: Int, Cin: Int,
    Kh: Int, Kw: Int, Cout: Int,
    Ho: Int, Wo: Int,
    sh: Int, sw: Int, ph: Int, pw: Int,
):
    var idx = Int(global_idx.x)
    var total = Kh * Kw * Cin * Cout
    if idx >= total:
        return
    # decode RSCF index idx -> (kh, kw, ci, co)
    var co = idx % Cout
    var t = idx // Cout
    var ci = t % Cin
    var t2 = t // Cin
    var kw = t2 % Kw
    var kh = t2 // Kw

    var acc: Float32 = 0.0
    for n in range(N):
        for oh in range(Ho):
            var ih = oh * sh - ph + kh
            if ih < 0 or ih >= Hi:
                continue
            for ow in range(Wo):
                var iw = ow * sw - pw + kw
                if iw < 0 or iw >= Wi:
                    continue
                var gy_off = (((n * Ho + oh) * Wo + ow) * Cout) + co
                var x_off = (((n * Hi + ih) * Wi + iw) * Cin) + ci
                var g = rebind[Scalar[dtype]](grad_y[gy_off]).cast[
                    DType.float32
                ]()
                var xv = rebind[Scalar[dtype]](x[x_off]).cast[DType.float32]()
                acc += g * xv
    d_w[idx] = rebind[d_w.element_type](acc.cast[dtype]())


# ── d_b kernel: one thread per output channel co ─────────────────────────────
# d_b[co] = sum over (n, oh, ow) of grad_y[n,oh,ow,co]
def _conv2d_db_kernel[dtype: DType](
    grad_y: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [N*Ho*Wo*Cout]
    d_b: LayoutTensor[dtype, _DYN1, MutAnyOrigin],      # [Cout]
    N: Int, Ho: Int, Wo: Int, Cout: Int,
):
    var co = Int(global_idx.x)
    if co >= Cout:
        return
    var acc: Float32 = 0.0
    for n in range(N):
        for oh in range(Ho):
            for ow in range(Wo):
                var off = (((n * Ho + oh) * Wo + ow) * Cout) + co
                acc += rebind[Scalar[dtype]](grad_y[off]).cast[DType.float32]()
    d_b[co] = rebind[d_b.element_type](acc.cast[dtype]())


def conv2d_backward[
    N: Int,
    Hi: Int,
    Wi: Int,
    Cin: Int,
    Kh: Int,
    Kw: Int,
    Cout: Int,
    stride_h: Int,
    stride_w: Int,
    pad_h: Int,
    pad_w: Int,
](
    x: Tensor,
    weight: Tensor,
    grad_y: Tensor,
    ctx: DeviceContext,
) raises -> Conv2dBwd:
    """conv2d backward (NHWC input, RSCF filter), dilation=1, num_groups=1.

    x:      [N, Hi, Wi, Cin]      NHWC  (the forward input)
    weight: [Kh, Kw, Cin, Cout]   RSCF  (the forward filter)
    grad_y: [N, Ho, Wo, Cout]     NHWC  (upstream grad, forward-output layout)
    returns d_x/d_w/d_b in the same storage dtype as x/weight/grad_y.

    Naive one-thread-per-output-element kernels. Matches ops/conv.mojo forward
    layout/stride/pad exactly. Shapes are compile-time params (mirrors forward).
    """
    comptime Ho = (Hi + 2 * pad_h - Kh) // stride_h + 1
    comptime Wo = (Wi + 2 * pad_w - Kw) // stride_w + 1

    # ── loud-fail shape / dtype validation ───────────────────────────────────
    if x.dtype() != grad_y.dtype():
        raise Error("conv2d_backward: x/grad_y dtype mismatch")
    var xshape = x.shape()
    if (
        len(xshape) != 4
        or xshape[0] != N or xshape[1] != Hi
        or xshape[2] != Wi or xshape[3] != Cin
    ):
        raise Error("conv2d_backward: x shape must match [N,Hi,Wi,Cin] params")
    var wshape = weight.shape()
    if (
        len(wshape) != 4
        or wshape[0] != Kh or wshape[1] != Kw
        or wshape[2] != Cin or wshape[3] != Cout
    ):
        raise Error("conv2d_backward: weight must be RSCF [Kh,Kw,Cin,Cout]")
    var gshape = grad_y.shape()
    if (
        len(gshape) != 4
        or gshape[0] != N or gshape[1] != Ho
        or gshape[2] != Wo or gshape[3] != Cout
    ):
        raise Error("conv2d_backward: grad_y shape must match [N,Ho,Wo,Cout]")
    if x.dtype() != weight.dtype():
        if x.dtype() != STDtype.F32:
            raise Error("conv2d_backward: mixed checkpoint weight requires F32 activations")
        if weight.dtype() != STDtype.BF16 and weight.dtype() != STDtype.F16:
            raise Error("conv2d_backward: unsupported mixed checkpoint weight dtype")
        var compute_weight = cast_tensor(weight, x.dtype(), ctx)
        var mixed = conv2d_backward[
            N, Hi, Wi, Cin, Kh, Kw, Cout, stride_h, stride_w, pad_h, pad_w
        ](x, compute_weight^, grad_y, ctx)
        var d_x = mixed.d_x.clone(ctx)
        var d_w_src = mixed.d_w.clone(ctx)
        var d_b_src = mixed.d_b.clone(ctx)
        var d_w = cast_tensor(d_w_src^, weight.dtype(), ctx)
        var d_b = cast_tensor(d_b_src^, weight.dtype(), ctx)
        return Conv2dBwd(d_x^, d_w^, d_b^)

    # ── flat (1-D) views of the device buffers ───────────────────────────────
    comptime nx = N * Hi * Wi * Cin
    comptime nw = Kh * Kw * Cin * Cout
    comptime ng = N * Ho * Wo * Cout

    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nx))
    var w_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nw))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](ng))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](Cout))

    # ── output buffers ───────────────────────────────────────────────────────
    var dtype = x.dtype()
    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](nx * dtype.byte_size())
    var dw_buf = ctx.enqueue_create_buffer[DType.uint8](nw * dtype.byte_size())
    var db_buf = ctx.enqueue_create_buffer[DType.uint8](Cout * dtype.byte_size())

    # ── launch d_x ───────────────────────────────────────────────────────────
    var dx_grid = (nx + _BLOCK - 1) // _BLOCK
    var dw_grid = (nw + _BLOCK - 1) // _BLOCK
    var db_grid = (Cout + _BLOCK - 1) // _BLOCK
    var dt = dtype.to_mojo_dtype()
    if dt == DType.float32:
        var xv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var wv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float32](), w_rl)
        var gv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_y.buf.unsafe_ptr().bitcast[Float32](), g_rl)
        var dxv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var dwv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            dw_buf.unsafe_ptr().bitcast[Float32](), w_rl)
        var dbv = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            db_buf.unsafe_ptr().bitcast[Float32](), b_rl)
        ctx.enqueue_function[
            _conv2d_dx_kernel[DType.float32], _conv2d_dx_kernel[DType.float32]
        ](
            gv, wv, dxv,
            N, Hi, Wi, Cin, Kh, Kw, Cout, Ho, Wo,
            stride_h, stride_w, pad_h, pad_w,
            grid_dim=dx_grid, block_dim=_BLOCK,
        )
        ctx.enqueue_function[
            _conv2d_dw_kernel[DType.float32], _conv2d_dw_kernel[DType.float32]
        ](
            gv, xv, dwv,
            N, Hi, Wi, Cin, Kh, Kw, Cout, Ho, Wo,
            stride_h, stride_w, pad_h, pad_w,
            grid_dim=dw_grid, block_dim=_BLOCK,
        )
        ctx.enqueue_function[
            _conv2d_db_kernel[DType.float32], _conv2d_db_kernel[DType.float32]
        ](gv, dbv, N, Ho, Wo, Cout, grid_dim=db_grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var xv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var wv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[BFloat16](), w_rl)
        var gv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_y.buf.unsafe_ptr().bitcast[BFloat16](), g_rl)
        var dxv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var dwv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            dw_buf.unsafe_ptr().bitcast[BFloat16](), w_rl)
        var dbv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            db_buf.unsafe_ptr().bitcast[BFloat16](), b_rl)
        ctx.enqueue_function[
            _conv2d_dx_kernel[DType.bfloat16], _conv2d_dx_kernel[DType.bfloat16]
        ](
            gv, wv, dxv,
            N, Hi, Wi, Cin, Kh, Kw, Cout, Ho, Wo,
            stride_h, stride_w, pad_h, pad_w,
            grid_dim=dx_grid, block_dim=_BLOCK,
        )
        ctx.enqueue_function[
            _conv2d_dw_kernel[DType.bfloat16], _conv2d_dw_kernel[DType.bfloat16]
        ](
            gv, xv, dwv,
            N, Hi, Wi, Cin, Kh, Kw, Cout, Ho, Wo,
            stride_h, stride_w, pad_h, pad_w,
            grid_dim=dw_grid, block_dim=_BLOCK,
        )
        ctx.enqueue_function[
            _conv2d_db_kernel[DType.bfloat16], _conv2d_db_kernel[DType.bfloat16]
        ](gv, dbv, N, Ho, Wo, Cout, grid_dim=db_grid, block_dim=_BLOCK)
    else:
        var xv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var wv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float16](), w_rl)
        var gv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_y.buf.unsafe_ptr().bitcast[Float16](), g_rl)
        var dxv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var dwv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            dw_buf.unsafe_ptr().bitcast[Float16](), w_rl)
        var dbv = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            db_buf.unsafe_ptr().bitcast[Float16](), b_rl)
        ctx.enqueue_function[
            _conv2d_dx_kernel[DType.float16], _conv2d_dx_kernel[DType.float16]
        ](
            gv, wv, dxv,
            N, Hi, Wi, Cin, Kh, Kw, Cout, Ho, Wo,
            stride_h, stride_w, pad_h, pad_w,
            grid_dim=dx_grid, block_dim=_BLOCK,
        )
        ctx.enqueue_function[
            _conv2d_dw_kernel[DType.float16], _conv2d_dw_kernel[DType.float16]
        ](
            gv, xv, dwv,
            N, Hi, Wi, Cin, Kh, Kw, Cout, Ho, Wo,
            stride_h, stride_w, pad_h, pad_w,
            grid_dim=dw_grid, block_dim=_BLOCK,
        )
        ctx.enqueue_function[
            _conv2d_db_kernel[DType.float16], _conv2d_db_kernel[DType.float16]
        ](gv, dbv, N, Ho, Wo, Cout, grid_dim=db_grid, block_dim=_BLOCK)
    ctx.synchronize()

    # ── wrap outputs ─────────────────────────────────────────────────────────
    var dx_shape = List[Int]()
    dx_shape.append(N); dx_shape.append(Hi); dx_shape.append(Wi); dx_shape.append(Cin)
    var dw_shape = List[Int]()
    dw_shape.append(Kh); dw_shape.append(Kw); dw_shape.append(Cin); dw_shape.append(Cout)
    var db_shape = List[Int]()
    db_shape.append(Cout)

    var dx_t = Tensor(dx_buf^, dx_shape^, dtype)
    var dw_t = Tensor(dw_buf^, dw_shape^, dtype)
    var db_t = Tensor(db_buf^, db_shape^, dtype)
    return Conv2dBwd(dx_t^, dw_t^, db_t^)
