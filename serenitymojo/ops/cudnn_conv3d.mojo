# Native low-startup cuDNN BF16 Conv3d wrapper for LTX2 video VAE decode.

from std.collections import List
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUDA

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.tensor import Tensor


def cudnn_conv3d_bf16_ndhwc(
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
    """NDHWC BF16 Conv3d with FCQRS weights and optional BF16 bias."""
    var xs = x.shape()
    var ws = weight.shape()
    if len(xs) != 5 or len(ws) != 5:
        raise Error("cudnn_conv3d_bf16_ndhwc: expected rank-5 tensors")
    if x.dtype() != STDtype.BF16 or weight.dtype() != STDtype.BF16:
        raise Error("cudnn_conv3d_bf16_ndhwc: BF16 tensors required")

    var n = xs[0]
    var din = xs[1]
    var hin = xs[2]
    var win = xs[3]
    var cin = xs[4]
    var cout = ws[0]
    var kd = ws[2]
    var kh = ws[3]
    var kw = ws[4]
    if ws[1] != cin:
        raise Error("cudnn_conv3d_bf16_ndhwc: channel mismatch")
    if bias:
        if bias.value().dtype() != STDtype.BF16 or bias.value().numel() != cout:
            raise Error("cudnn_conv3d_bf16_ndhwc: invalid bias")

    var dout = (din + 2 * pad_d - kd) // stride_d + 1
    var hout = (hin + 2 * pad_h - kh) // stride_h + 1
    var wout = (win + 2 * pad_w - kw) // stride_w + 1
    if dout <= 0 or hout <= 0 or wout <= 0:
        raise Error("cudnn_conv3d_bf16_ndhwc: invalid output shape")

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * dout * hout * wout * cout * 2
    )
    var x_ptr = BytePtr(unsafe_from_address=Int(x.buf.unsafe_ptr()))
    var w_ptr = BytePtr(unsafe_from_address=Int(weight.buf.unsafe_ptr()))
    var b_ptr = BytePtr(unsafe_from_address=0)
    if bias:
        b_ptr = BytePtr(
            unsafe_from_address=Int(bias.value().buf.unsafe_ptr())
        )
    var y_ptr = BytePtr(unsafe_from_address=Int(out_buf.unsafe_ptr()))
    var stream = CUDA(ctx.stream())

    var rc = Int(external_call["serenity_cudnn_conv3d_bf16_ndhwc", Int32](
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        Int32(n),
        Int32(din),
        Int32(hin),
        Int32(win),
        Int32(cin),
        Int32(cout),
        Int32(kd),
        Int32(kh),
        Int32(kw),
        Int32(stride_d),
        Int32(stride_h),
        Int32(stride_w),
        Int32(pad_d),
        Int32(pad_h),
        Int32(pad_w),
        stream,
    ))
    if rc != 0:
        raise Error(String("cudnn_conv3d_bf16_ndhwc: shim rc=") + String(rc))

    var out_shape = List[Int]()
    out_shape.append(n)
    out_shape.append(dout)
    out_shape.append(hout)
    out_shape.append(wout)
    out_shape.append(cout)
    return Tensor(out_buf^, out_shape^, STDtype.BF16)
