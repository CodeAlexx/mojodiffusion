# upsample.mojo — nearest-neighbor 2x upsample on an NHWC tensor.
#
# diffusers Upsample2D (used by the VAE decoder up_blocks) is
# F.interpolate(scale_factor=2, mode="nearest") followed by a 3x3 conv. This op
# is JUST the nearest gather; the 3x3 conv is the foundation conv2d, applied by
# the decoder after this. The foundation has no resize/upsample op (it is a
# diffusion-VAE-specific reshape), so the kit owns it.
#
# Nearest 2x, NHWC: out[n, oh, ow, c] = in[n, oh//2, ow//2, c].
# One thread per OUTPUT element over the flat [N, 2H, 2W, C] buffer.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _up_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n_dim: Int, h: Int, w: Int, c: Int,
):
    var idx = Int(global_idx.x)
    var oh = h * 2
    var ow = w * 2
    var total = n_dim * oh * ow * c
    if idx < total:
        var cc = idx % c
        var t = idx // c
        var owi = t % ow
        t = t // ow
        var ohi = t % oh
        var ni = t // oh
        var ih = ohi // 2
        var iw = owi // 2
        var src = ((ni * h + ih) * w + iw) * c + cc
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[src]))


def _up_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n_dim: Int, h: Int, w: Int, c: Int,
):
    var idx = Int(global_idx.x)
    var oh = h * 2
    var ow = w * 2
    var total = n_dim * oh * ow * c
    if idx < total:
        var cc = idx % c
        var t = idx // c
        var owi = t % ow
        t = t // ow
        var ohi = t % oh
        var ni = t // oh
        var ih = ohi // 2
        var iw = owi // 2
        var src = ((ni * h + ih) * w + iw) * c + cc
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[src]))


def _up_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n_dim: Int, h: Int, w: Int, c: Int,
):
    var idx = Int(global_idx.x)
    var oh = h * 2
    var ow = w * 2
    var total = n_dim * oh * ow * c
    if idx < total:
        var cc = idx % c
        var t = idx // c
        var owi = t % ow
        t = t // ow
        var ohi = t % oh
        var ni = t // oh
        var ih = ohi // 2
        var iw = owi // 2
        var src = ((ni * h + ih) * w + iw) * c + cc
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](x[src]))


def upsample_nearest2x_nhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Nearest 2x upsample of an NHWC tensor [N,H,W,C] -> [N,2H,2W,C]."""
    var sh = x.shape()
    if len(sh) != 4:
        raise Error("upsample: need rank-4 NHWC")
    var n = sh[0]
    var h = sh[1]
    var w = sh[2]
    var c = sh[3]
    var oh = h * 2
    var ow = w * 2
    var dt = x.dtype().to_mojo_dtype()
    var out_n = n * oh * ow * c
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )
    var in_n = x.numel()
    var in_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](in_n))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), in_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[_up_kernel_f32, _up_kernel_f32](
            X, O, n, h, w, c, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), in_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[_up_kernel_bf16, _up_kernel_bf16](
            X, O, n, h, w, c, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), in_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[_up_kernel_f16, _up_kernel_f16](
            X, O, n, h, w, c, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()

    var os = List[Int]()
    os.append(n)
    os.append(oh)
    os.append(ow)
    os.append(c)
    return Tensor(out_buf^, os^, x.dtype())
