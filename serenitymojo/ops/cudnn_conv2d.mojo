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
