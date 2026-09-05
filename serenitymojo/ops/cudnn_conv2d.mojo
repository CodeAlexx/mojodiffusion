# Native cuDNN NCHW BF16 Conv2d wrapper.
#
# LTX Creator's audio VAE uses torch.nn.Conv2d on NCHW activations. Keeping the
# operation 2D is required for waveform parity; a singleton-axis Conv3d changes
# BF16 accumulation order enough for the downstream vocoder to amplify it.

from std.collections import List
from std.ffi import external_call
from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUDA

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.tensor import Tensor


def cudnn_conv2d_bf16_nchw(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride_h: Int,
    stride_w: Int,
    pad_h: Int,
    pad_w: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """NCHW BF16 Conv2d with OIHW BF16 weights and BF16 bias.

    Algorithm 1 is cuDNN IMPLICIT_PRECOMP_GEMM. It was selected by cuDNN's v7
    heuristic and measured bit-exact against Creator LTX's stored BF16 phase on
    the pinned 9.10.2 decode runtime.
    """
    var xs = x.shape()
    var ws = weight.shape()
    if len(xs) != 4 or len(ws) != 4:
        raise Error("cudnn_conv2d_bf16_nchw: expected rank-4 NCHW/OIHW")
    if (
        x.dtype() != STDtype.BF16
        or weight.dtype() != STDtype.BF16
        or bias.dtype() != STDtype.BF16
    ):
        raise Error("cudnn_conv2d_bf16_nchw: BF16 tensors required")

    var n = xs[0]
    var cin = xs[1]
    var hin = xs[2]
    var win = xs[3]
    var cout = ws[0]
    var kh = ws[2]
    var kw = ws[3]
    if ws[1] != cin or bias.numel() != cout:
        raise Error("cudnn_conv2d_bf16_nchw: channel mismatch")

    var hout = (hin + 2 * pad_h - kh) // stride_h + 1
    var wout = (win + 2 * pad_w - kw) // stride_w + 1
    if hout <= 0 or wout <= 0:
        raise Error("cudnn_conv2d_bf16_nchw: invalid output shape")

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * cout * hout * wout * 2
    )
    var x_ptr = BytePtr(unsafe_from_address=Int(x.buf.unsafe_ptr()))
    var w_ptr = BytePtr(unsafe_from_address=Int(weight.buf.unsafe_ptr()))
    var b_ptr = BytePtr(unsafe_from_address=Int(bias.buf.unsafe_ptr()))
    var y_ptr = BytePtr(unsafe_from_address=Int(out_buf.unsafe_ptr()))
    var stream = CUDA(ctx.stream())

    var rc = Int(external_call["serenity_cudnn_conv2d_bf16_nchw", Int32](
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        Int32(n),
        Int32(cin),
        Int32(hin),
        Int32(win),
        Int32(cout),
        Int32(kh),
        Int32(kw),
        Int32(stride_h),
        Int32(stride_w),
        Int32(pad_h),
        Int32(pad_w),
        Int32(1),
        stream,
    ))
    if rc != 0:
        if rc == -91002:
            raise Error(
                "cudnn_conv2d_bf16_nchw: requires cuDNN 9.10.2 or a newer "
                "cuDNN 9 decode runtime"
            )
        raise Error(
            String("cudnn_conv2d_bf16_nchw: shim rc=") + String(rc)
        )
    ctx.synchronize()

    var out_shape = List[Int]()
    out_shape.append(n)
    out_shape.append(cout)
    out_shape.append(hout)
    out_shape.append(wout)
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


def cudnn_conv2d_bf16_nhwc(
    x: Tensor,
    weight_krsc: Tensor,
    bias: Optional[Tensor],
    stride_h: Int,
    stride_w: Int,
    pad_h: Int,
    pad_w: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """NHWC BF16 Conv2d with KRSC BF16 weights and an optional BF16 bias.

    This is the layout every image decoder and the SDXL UNet in this tree
    already use ([N, H, W, C] activations), so it needs no activation
    transposes: only the filter is re-laid from the checkpoint's RSCF
    [Kh, Kw, Cin, Cout] to cuDNN's NHWC filter order KRSC [Cout, Kh, Kw, Cin],
    which the caller does once per weight. The NCHW entry above is the LTX
    audio VAE's and is left exactly as it was.

    Same cuDNN engine as the NCHW wrapper: F32 accumulation, tensor-op math,
    algorithm from the v7 heuristic (-1). Measured before this existed: the
    im2col + GEMM path took ~11 s to decode a 1024x1024 image on three
    different models, because im2col of a full-resolution feature map is
    gigabytes of traffic per convolution.
    """
    var xs = x.shape()
    var ws = weight_krsc.shape()
    if len(xs) != 4 or len(ws) != 4:
        raise Error("cudnn_conv2d_bf16_nhwc: expected rank-4 NHWC/KRSC")
    if x.dtype() != STDtype.BF16 or weight_krsc.dtype() != STDtype.BF16:
        raise Error("cudnn_conv2d_bf16_nhwc: BF16 tensors required")
    var n = xs[0]
    var hin = xs[1]
    var win = xs[2]
    var cin = xs[3]
    var cout = ws[0]
    var kh = ws[1]
    var kw = ws[2]
    if ws[3] != cin:
        raise Error("cudnn_conv2d_bf16_nhwc: channel mismatch")
    var b_addr = 0
    if bias:
        if bias.value().dtype() != STDtype.BF16 or bias.value().numel() != cout:
            raise Error("cudnn_conv2d_bf16_nhwc: bias must be BF16 [Cout]")
        b_addr = Int(bias.value().buf.unsafe_ptr())
    var hout = (hin + 2 * pad_h - kh) // stride_h + 1
    var wout = (win + 2 * pad_w - kw) // stride_w + 1
    if hout <= 0 or wout <= 0:
        raise Error("cudnn_conv2d_bf16_nhwc: invalid output shape")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * hout * wout * cout * 2
    )
    var x_ptr = BytePtr(unsafe_from_address=Int(x.buf.unsafe_ptr()))
    var w_ptr = BytePtr(unsafe_from_address=Int(weight_krsc.buf.unsafe_ptr()))
    var b_ptr = BytePtr(unsafe_from_address=b_addr)
    var y_ptr = BytePtr(unsafe_from_address=Int(out_buf.unsafe_ptr()))
    var stream = CUDA(ctx.stream())
    var rc = Int(external_call["serenity_cudnn_conv2d_bf16_nhwc", Int32](
        x_ptr, w_ptr, b_ptr, y_ptr,
        Int32(n), Int32(cin), Int32(hin), Int32(win),
        Int32(cout), Int32(kh), Int32(kw),
        Int32(stride_h), Int32(stride_w), Int32(pad_h), Int32(pad_w),
        Int32(-1),
        stream,
    ))
    if rc != 0:
        raise Error(
            String("cudnn_conv2d_bf16_nhwc: shim rc=") + String(rc)
        )
    var out_shape: List[Int] = [n, hout, wout, cout]
    return Tensor(out_buf^, out_shape^, STDtype.BF16)
