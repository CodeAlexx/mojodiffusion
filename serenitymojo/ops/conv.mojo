# ops/conv.mojo — conv2d via the SDK naive NHWC kernel + a bias-add kernel.
#
# SDK-CALLABLE (OP-CALLABILITY MAP): `nn.conv.conv.conv2d_gpu_naive_nhwc_rscf`.
# IMPORTANT FINDING: this symbol is the DEVICE KERNEL BODY (it reads block_idx /
# thread_idx), NOT a host launcher — calling it directly from host fails with
# "target does not support _get_intrinsic_name". It must be launched via
# `ctx.enqueue_function`. The packaged 1.0.0b1 signature has SEVEN runtime args:
#     (input, filter, output, stride, dilation, padding, num_groups)
# (the upstream OSS source has six — no num_groups — so the count is build-
# specific; the packaged build we link against wants num_groups). Grid is 3D
# (W_out, H_out, N); block is 2D (block_size, block_size). The kernel maps
# n=block_idx.z, h=block_idx.y*bs+ty, w=block_idx.x*bs+tx, loops over C_out.
#
# LAYOUTS (verified against the kernel source):
#   input  NHWC  = [N, H, W, C_in]
#   filter RSCF  = [Kh, Kw, C_in, C_out]   (R=Kh, S=Kw)
#   output NHWC  = [N, H_out, W_out, C_out]
# Our row-major Tensor buffers already match these layouts — no transpose.
#
# The kernel has NO bias. We add bias[C_out] (broadcast over N,H_out,W_out)
# with a tiny follow-up elementwise kernel. Bias stays in storage dtype at the
# op boundary; each thread casts scalars to F32 for the add, then stores output
# in x's storage dtype.
# Shapes are compile-time params (static layouts the kernel needs).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import ceildiv
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from nn.conv.conv import conv2d_gpu_naive_nhwc_rscf
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _CONV_BS = 16  # 2D conv block tile (block_size x block_size)


# Bias add: out[r, c] += bias[c], over a flat [rows=N*H_out*W_out, C_out] view.
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
    bias: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
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
        v += rebind[Scalar[DType.bfloat16]](bias[c]).cast[DType.float32]()
        o[idx // cols, c] = rebind[o.element_type](v.cast[DType.bfloat16]())


def _bias_add_kernel_f16(
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
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
        v += rebind[Scalar[DType.float16]](bias[c]).cast[DType.float32]()
        o[idx // cols, c] = rebind[o.element_type](v.cast[DType.float16]())


def conv2d[
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
    bias: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Tensor:
    """conv2d (NHWC input, RSCF filter), dilation=1, num_groups=1.

    x:      [N, Hi, Wi, Cin]          (NHWC; compute dtype)
    weight: [Kh, Kw, Cin, Cout]       (RSCF filter; same dtype as x)
    bias:   [Cout] or None            (added per-output-channel; same dtype)
    returns [N, Ho, Wo, Cout]         (x's dtype; F32-accumulated conv).

    Shapes/stride/padding are compile-time params (the SDK kernel needs static
    layouts). H_out/W_out are derived with dilation=1.
    """
    comptime Ho = (Hi + 2 * pad_h - Kh) // stride_h + 1
    comptime Wo = (Wi + 2 * pad_w - Kw) // stride_w + 1

    var xshape = x.shape()
    if (
        len(xshape) != 4
        or xshape[0] != N
        or xshape[1] != Hi
        or xshape[2] != Wi
        or xshape[3] != Cin
    ):
        raise Error("conv2d: x shape must match [N,Hi,Wi,Cin] params")
    var wshape = weight.shape()
    if (
        len(wshape) != 4
        or wshape[0] != Kh
        or wshape[1] != Kw
        or wshape[2] != Cin
        or wshape[3] != Cout
    ):
        raise Error("conv2d: weight must be RSCF [Kh,Kw,Cin,Cout]")
    if x.dtype() != weight.dtype():
        raise Error("conv2d: x/weight dtype mismatch")

    comptime in_l = Layout.row_major(N, Hi, Wi, Cin)
    comptime filt_l = Layout.row_major(Kh, Kw, Cin, Cout)
    comptime out_l = Layout.row_major(N, Ho, Wo, Cout)

    var dt = x.dtype().to_mojo_dtype()
    var out_n = N * Ho * Wo * Cout
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )

    var gx = ceildiv(Wo, _CONV_BS)
    var gy = ceildiv(Ho, _CONV_BS)
    var stride = IndexList[2](stride_h, stride_w)
    var dilation = IndexList[2](1, 1)
    var padding = IndexList[2](pad_h, pad_w)

    if dt == DType.float32:
        comptime knl = conv2d_gpu_naive_nhwc_rscf[
            in_l, filt_l, out_l,
            DType.float32, DType.float32, DType.float32,
            _CONV_BS, None,
        ]
        var X = LayoutTensor[DType.float32, in_l, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32]()
        )
        var F = LayoutTensor[DType.float32, filt_l, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float32]()
        )
        var O = LayoutTensor[DType.float32, out_l, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32]()
        )
        ctx.enqueue_function[knl, knl](
            X, F, O, stride, dilation, padding, 1,
            grid_dim=(gx, gy, N), block_dim=(_CONV_BS, _CONV_BS),
        )
    elif dt == DType.bfloat16:
        comptime knl = conv2d_gpu_naive_nhwc_rscf[
            in_l, filt_l, out_l,
            DType.bfloat16, DType.bfloat16, DType.bfloat16,
            _CONV_BS, None,
        ]
        var X = LayoutTensor[DType.bfloat16, in_l, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16]()
        )
        var F = LayoutTensor[DType.bfloat16, filt_l, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[BFloat16]()
        )
        var O = LayoutTensor[DType.bfloat16, out_l, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16]()
        )
        ctx.enqueue_function[knl, knl](
            X, F, O, stride, dilation, padding, 1,
            grid_dim=(gx, gy, N), block_dim=(_CONV_BS, _CONV_BS),
        )
    else:  # float16
        comptime knl = conv2d_gpu_naive_nhwc_rscf[
            in_l, filt_l, out_l,
            DType.float16, DType.float16, DType.float16,
            _CONV_BS, None,
        ]
        var X = LayoutTensor[DType.float16, in_l, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16]()
        )
        var F = LayoutTensor[DType.float16, filt_l, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float16]()
        )
        var O = LayoutTensor[DType.float16, out_l, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16]()
        )
        ctx.enqueue_function[knl, knl](
            X, F, O, stride, dilation, padding, 1,
            grid_dim=(gx, gy, N), block_dim=(_CONV_BS, _CONV_BS),
        )
    ctx.synchronize()

    # Optional bias add (per-output-channel, broadcast over N,Ho,Wo).
    if bias:
        if bias.value().dtype() != x.dtype():
            raise Error("conv2d: bias dtype mismatch")
        var bshape = bias.value().shape()
        if len(bshape) != 1 or bshape[0] != Cout:
            raise Error("conv2d: bias must be [Cout]")

        var rows = N * Ho * Wo
        var o_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, Cout))
        var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](Cout))
        var grid = (rows * Cout + _BLOCK - 1) // _BLOCK
        if dt == DType.float32:
            var O2 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float32](), o_rl
            )
            var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                bias.value().buf.unsafe_ptr().bitcast[Float32](), b_rl
            )
            ctx.enqueue_function[_bias_add_kernel_f32, _bias_add_kernel_f32](
                O2, B, rows, Cout, grid_dim=grid, block_dim=_BLOCK
            )
        elif dt == DType.bfloat16:
            var O2 = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
            )
            var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                bias.value().buf.unsafe_ptr().bitcast[BFloat16](), b_rl
            )
            ctx.enqueue_function[_bias_add_kernel_bf16, _bias_add_kernel_bf16](
                O2, B, rows, Cout, grid_dim=grid, block_dim=_BLOCK
            )
        else:
            var O2 = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float16](), o_rl
            )
            var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                bias.value().buf.unsafe_ptr().bitcast[Float16](), b_rl
            )
            ctx.enqueue_function[_bias_add_kernel_f16, _bias_add_kernel_f16](
                O2, B, rows, Cout, grid_dim=grid, block_dim=_BLOCK
            )
        ctx.synchronize()

    var out_shape = List[Int]()
    out_shape.append(N)
    out_shape.append(Ho)
    out_shape.append(Wo)
    out_shape.append(Cout)
    return Tensor(out_buf^, out_shape^, x.dtype())
