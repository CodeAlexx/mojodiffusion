# conv3d.mojo — conv3d via the SDK naive NDHWC kernel + a bias-add kernel.
#
# This is the ONE genuinely-new op for the Wan2.1 3D VAE decoder (the foundation
# ships conv2d but no conv3d). It is wired EXACTLY like ops/conv.mojo wires
# conv2d_gpu_naive_nhwc_rscf, just one rank up.
#
# === CONV3D-CALLABILITY FINDING ===
# SDK symbol `nn.conv.conv.conv3d_gpu_naive_ndhwc_qrscf` IS callable. Like its 2D
# sibling it is the DEVICE KERNEL BODY (reads block_idx/thread_idx), NOT a host
# launcher — it must be launched via `ctx.enqueue_function`. Verified against the
# source at /home/alex/modular/max/kernels/src/nn/conv/conv.mojo:5208. Seven
# runtime args (same shape as conv2d): (input, filter, output, stride, dilation,
# padding, num_groups). Params: 3 layouts, 3 dtypes, block_size, optional
# epilogue. So a hand-rolled im2col-3D fallback is NOT needed.
#
# LAYOUTS (verified against the kernel source, conv.mojo:5226-5245):
#   input  NDHWC = [N, D, H, W, C_in]      (D = depth = temporal/frame axis)
#   filter QRSCF = [Q, R, S, C_in, C_out]  (Q=Kd, R=Kh, S=Kw)
#   output NDHWC = [N, D_out, H_out, W_out, C_out]
# The kernel computes  d_in = d_out*stride_d + q*dil_d - pad_d  (and h,w alike),
# so PADDING IS SYMMETRIC on every axis. For a CausalConv3d we therefore pad the
# temporal (D) axis MANUALLY (left-only) and pass pad_d=0 here. Spatial pad_h/
# pad_w stay symmetric and are handled by the kernel.
#
# Launch geometry (conv.mojo:5189-5202):
#   grid_dim = (ceildiv(H_out*W_out, bs), ceildiv(D_out, bs), N)
#   block_dim = (bs, bs)
# where block_idx.z = n, the x-thread covers H_out*W_out, y covers D_out.
#
# Bias is added by a follow-up elementwise kernel (broadcast over N,D,H,W),
# F32-accumulated, cast to storage dtype — mirrors ops/conv.mojo exactly.
#
# Shapes/stride/padding are RUNTIME (the VAE spatial size changes per upsample,
# and we want one decoder that works at any latent size), so all LayoutTensors
# use a fully-dynamic 5-D / 2-D layout plus a RuntimeLayout carrying the concrete
# dims. (The SDK kernel itself reads dims via input.dim[i]() at runtime, so a
# dynamic layout is fine — verified: the kernel never indexes the comptime
# Layout for sizes.)
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import ceildiv
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from nn.conv.conv import conv3d_gpu_naive_ndhwc_qrscf
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _DYN5 = Layout.row_major(-1, -1, -1, -1, -1)
comptime _BLOCK = 256
comptime _CONV_BS = 16  # 3D conv block tile (block_size x block_size)


# Bias add: out[r, c] += bias[c], over a flat [rows=N*D*H*W, C_out] view.
def _bias_add_kernel_f32(
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var c = idx % cols
        var v = rebind[Scalar[DType.float32]](o[idx // cols, c])
        v += rebind[Scalar[DType.float32]](bias[c])
        o[idx // cols, c] = rebind[o.element_type](v)


def _bias_add_kernel_bf16(
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var c = idx % cols
        var v = rebind[Scalar[DType.bfloat16]](o[idx // cols, c]).cast[
            DType.float32
        ]()
        v += rebind[Scalar[DType.float32]](bias[c])
        o[idx // cols, c] = rebind[o.element_type](v.cast[DType.bfloat16]())


def _bias_add_kernel_f16(
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var c = idx % cols
        var v = rebind[Scalar[DType.float16]](o[idx // cols, c]).cast[
            DType.float32
        ]()
        v += rebind[Scalar[DType.float32]](bias[c])
        o[idx // cols, c] = rebind[o.element_type](v.cast[DType.float16]())


def conv3d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride_d: Int,
    stride_h: Int,
    stride_w: Int,
    pad_d: Int,
    pad_h: Int,
    pad_w: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """conv3d (NDHWC input, QRSCF filter), dilation=1, num_groups=1.

    x:      [N, D, H, W, Cin]            (NDHWC; compute dtype)
    weight: [Q, R, S, Cin, Cout]         (QRSCF filter; same dtype as x)
    bias:   [Cout] or None               (added per-output-channel)
    returns [N, Do, Ho, Wo, Cout]        (x's dtype; F32-accumulated conv).

    Padding is SYMMETRIC on every axis (the kernel's d_in/h_in/w_in formula).
    For a causal temporal conv the caller pads D manually (left-only) and passes
    pad_d=0. Shapes/stride/pad are runtime; the kernel reads dims at runtime.
    """
    var xshape = x.shape()
    if len(xshape) != 5:
        raise Error("conv3d: x must be rank-5 NDHWC [N,D,H,W,Cin]")
    var wshape = weight.shape()
    if len(wshape) != 5:
        raise Error("conv3d: weight must be rank-5 QRSCF [Q,R,S,Cin,Cout]")
    if x.dtype() != weight.dtype():
        raise Error("conv3d: x/weight dtype mismatch")

    var n = xshape[0]
    var di = xshape[1]
    var hi = xshape[2]
    var wi = xshape[3]
    var cin = xshape[4]

    var q = wshape[0]
    var r = wshape[1]
    var s = wshape[2]
    if wshape[3] != cin:
        raise Error("conv3d: weight Cin != x Cin")
    var cout = wshape[4]

    var do_ = (di + 2 * pad_d - q) // stride_d + 1
    var ho = (hi + 2 * pad_h - r) // stride_h + 1
    var wo = (wi + 2 * pad_w - s) // stride_w + 1

    var dt = x.dtype().to_mojo_dtype()
    var out_n = n * do_ * ho * wo * cout
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )

    var in_rl = RuntimeLayout[_DYN5].row_major(IndexList[5](n, di, hi, wi, cin))
    var filt_rl = RuntimeLayout[_DYN5].row_major(IndexList[5](q, r, s, cin, cout))
    var out_rl = RuntimeLayout[_DYN5].row_major(
        IndexList[5](n, do_, ho, wo, cout)
    )

    var stride = IndexList[3](stride_d, stride_h, stride_w)
    var dilation = IndexList[3](1, 1, 1)
    var padding = IndexList[3](pad_d, pad_h, pad_w)

    var gx = ceildiv(ho * wo, _CONV_BS)
    var gy = ceildiv(do_, _CONV_BS)

    if dt == DType.float32:
        comptime knl = conv3d_gpu_naive_ndhwc_qrscf[
            _DYN5, _DYN5, _DYN5,
            DType.float32, DType.float32, DType.float32,
            _CONV_BS, None,
        ]
        var X = LayoutTensor[DType.float32, _DYN5, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), in_rl
        )
        var F = LayoutTensor[DType.float32, _DYN5, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float32](), filt_rl
        )
        var O = LayoutTensor[DType.float32, _DYN5, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[knl, knl](
            X, F, O, stride, dilation, padding, 1,
            grid_dim=(gx, gy, n), block_dim=(_CONV_BS, _CONV_BS),
        )
    elif dt == DType.bfloat16:
        comptime knl = conv3d_gpu_naive_ndhwc_qrscf[
            _DYN5, _DYN5, _DYN5,
            DType.bfloat16, DType.bfloat16, DType.bfloat16,
            _CONV_BS, None,
        ]
        var X = LayoutTensor[DType.bfloat16, _DYN5, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), in_rl
        )
        var F = LayoutTensor[DType.bfloat16, _DYN5, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[BFloat16](), filt_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN5, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[knl, knl](
            X, F, O, stride, dilation, padding, 1,
            grid_dim=(gx, gy, n), block_dim=(_CONV_BS, _CONV_BS),
        )
    else:  # float16
        comptime knl = conv3d_gpu_naive_ndhwc_qrscf[
            _DYN5, _DYN5, _DYN5,
            DType.float16, DType.float16, DType.float16,
            _CONV_BS, None,
        ]
        var X = LayoutTensor[DType.float16, _DYN5, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), in_rl
        )
        var F = LayoutTensor[DType.float16, _DYN5, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float16](), filt_rl
        )
        var O = LayoutTensor[DType.float16, _DYN5, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[knl, knl](
            X, F, O, stride, dilation, padding, 1,
            grid_dim=(gx, gy, n), block_dim=(_CONV_BS, _CONV_BS),
        )
    ctx.synchronize()

    # Optional bias add (per-output-channel, broadcast over N,Do,Ho,Wo).
    if bias:
        var bvals = bias.value().to_host(ctx)
        if len(bvals) != cout:
            raise Error("conv3d: bias length != Cout")
        var bias_f32_buf = ctx.enqueue_create_buffer[DType.uint8](cout * 4)
        var bhost = ctx.enqueue_create_host_buffer[DType.uint8](cout * 4)
        var bp = bhost.unsafe_ptr().bitcast[Float32]()
        for i in range(cout):
            bp[i] = bvals[i]
        ctx.enqueue_copy(dst_buf=bias_f32_buf, src_buf=bhost)
        ctx.synchronize()

        var rows = n * do_ * ho * wo
        var o_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cout))
        var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](cout))
        var bias_lt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            bias_f32_buf.unsafe_ptr().bitcast[Float32](), b_rl
        )
        var grid = (rows * cout + _BLOCK - 1) // _BLOCK
        if dt == DType.float32:
            var O2 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float32](), o_rl
            )
            ctx.enqueue_function[_bias_add_kernel_f32, _bias_add_kernel_f32](
                O2, bias_lt, rows, cout, grid_dim=grid, block_dim=_BLOCK
            )
        elif dt == DType.bfloat16:
            var O2 = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
            )
            ctx.enqueue_function[_bias_add_kernel_bf16, _bias_add_kernel_bf16](
                O2, bias_lt, rows, cout, grid_dim=grid, block_dim=_BLOCK
            )
        else:
            var O2 = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float16](), o_rl
            )
            ctx.enqueue_function[_bias_add_kernel_f16, _bias_add_kernel_f16](
                O2, bias_lt, rows, cout, grid_dim=grid, block_dim=_BLOCK
            )
        ctx.synchronize()

    var out_shape = List[Int]()
    out_shape.append(n)
    out_shape.append(do_)
    out_shape.append(ho)
    out_shape.append(wo)
    out_shape.append(cout)
    return Tensor(out_buf^, out_shape^, x.dtype())
