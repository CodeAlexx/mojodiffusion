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
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import reshape, transpose


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _CONV_BS = 16  # 2D conv block tile (block_size x block_size)


# ── im2col fast path ──────────────────────────────────────────────────────────
# The naive SDK kernel (conv2d_gpu_naive_nhwc_rscf) is one thread per (n,h,w)
# looping over C_out × the K-reduction → it serializes badly (a 1024² VAE decode
# took ~150s). im2col + the foundation `linear` (F32-accumulated gemm) sums the
# IDENTICAL products (reordered), so the result matches to bf16 precision while
# running on the fast matmul. col[row=(n,ho,wo), k=(kh,kw,ci)] = padded input.
# RSCF filter [Kh,Kw,Cin,Cout] is already [K=Kh*Kw*Cin, Cout] row-major, so the
# (kh,kw,ci) column order matches the weight's flattening exactly.
def _im2col_kernel[
    dt: DType
](
    col: LayoutTensor[dt, _DYN1, MutAnyOrigin],
    inp: LayoutTensor[dt, _DYN1, MutAnyOrigin],
    total: Int, K: Int, Hi: Int, Wi: Int, Cin: Int, Ho: Int, Wo: Int,
    Kh: Int, Kw: Int, sh: Int, sw: Int, ph: Int, pw: Int,
):
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var row = idx // K
    var k = idx % K
    var hw = Ho * Wo
    var n = row // hw
    var rem = row % hw
    var ho = rem // Wo
    var wo = rem % Wo
    var ci = k % Cin
    var ks = k // Cin          # = kh*Kw + kw
    var kh = ks // Kw
    var kw = ks % Kw
    var ih = ho * sh - ph + kh
    var iw = wo * sw - pw + kw
    if ih >= 0 and ih < Hi and iw >= 0 and iw < Wi:
        var in_idx = ((n * Hi + ih) * Wi + iw) * Cin + ci
        col[idx] = rebind[col.element_type](inp[in_idx])
    else:
        col[idx] = rebind[col.element_type](SIMD[dt, 1](0))


def conv2d_im2col[
    N: Int, Hi: Int, Wi: Int, Cin: Int, Kh: Int, Kw: Int, Cout: Int,
    stride_h: Int, stride_w: Int, pad_h: Int, pad_w: Int,
](
    x: Tensor, weight: Tensor, bias: Optional[Tensor], ctx: DeviceContext
) raises -> Tensor:
    """conv2d via im2col + `linear` (F32-accumulated gemm). Numerically equal to
    the naive conv (same products, reordered). x NHWC, weight RSCF, bias [Cout]."""
    comptime Ho = (Hi + 2 * pad_h - Kh) // stride_h + 1
    comptime Wo = (Wi + 2 * pad_w - Kw) // stride_w + 1
    comptime M = N * Ho * Wo
    comptime K = Kh * Kw * Cin

    var dt = x.dtype().to_mojo_dtype()
    var col_buf = ctx.enqueue_create_buffer[DType.uint8](
        M * K * x.dtype().byte_size()
    )
    comptime col_l = Layout.row_major(M * K)
    comptime in_l = Layout.row_major(N * Hi * Wi * Cin)
    var grid = ceildiv(M * K, _BLOCK)
    if dt == DType.float32:
        comptime knl = _im2col_kernel[DType.float32]
        var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            col_buf.unsafe_ptr().bitcast[Float32](),
            RuntimeLayout[_DYN1].row_major(IndexList[1](M * K)),
        )
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](),
            RuntimeLayout[_DYN1].row_major(IndexList[1](N * Hi * Wi * Cin)),
        )
        ctx.enqueue_function[knl, knl](
            C, X, M * K, K, Hi, Wi, Cin, Ho, Wo, Kh, Kw,
            stride_h, stride_w, pad_h, pad_w,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        comptime knl = _im2col_kernel[DType.bfloat16]
        var C = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            col_buf.unsafe_ptr().bitcast[BFloat16](),
            RuntimeLayout[_DYN1].row_major(IndexList[1](M * K)),
        )
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](),
            RuntimeLayout[_DYN1].row_major(IndexList[1](N * Hi * Wi * Cin)),
        )
        ctx.enqueue_function[knl, knl](
            C, X, M * K, K, Hi, Wi, Cin, Ho, Wo, Kh, Kw,
            stride_h, stride_w, pad_h, pad_w,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        comptime knl = _im2col_kernel[DType.float16]
        var C = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            col_buf.unsafe_ptr().bitcast[Float16](),
            RuntimeLayout[_DYN1].row_major(IndexList[1](M * K)),
        )
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](),
            RuntimeLayout[_DYN1].row_major(IndexList[1](N * Hi * Wi * Cin)),
        )
        ctx.enqueue_function[knl, knl](
            C, X, M * K, K, Hi, Wi, Cin, Ho, Wo, Kh, Kw,
            stride_h, stride_w, pad_h, pad_w,
            grid_dim=grid, block_dim=_BLOCK,
        )
    # sync removed (single-stream ordering; was kernel-trailing host stall)

    var col = Tensor(col_buf^, [M, K], x.dtype())          # [M, K]
    # Feed the RSCF weight as [K, Cout] directly via transpose_b=False — avoids
    # materializing a [Cout, K] copy that linear would only transpose back (MJ-0910).
    var wk = reshape(weight, [K, Cout], ctx)               # [K, Cout] = [in, out]
    var y = linear(col, wk, bias, ctx, transpose_b=False)  # [M, Cout] = col @ wk + bias
    return reshape(y, [N, Ho, Wo, Cout], ctx)


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
    var mixed_checkpoint_weight = (
        x.dtype() == STDtype.F32
        and (weight.dtype() == STDtype.BF16 or weight.dtype() == STDtype.F16)
    )
    if x.dtype() != weight.dtype() and not mixed_checkpoint_weight:
        raise Error("conv2d: x/weight dtype mismatch")

    # Fast path: im2col + F32-accumulated `linear` gemm (numerically equal to the
    # naive conv, ~50x faster at high resolution). The naive SDK kernel below is
    # kept as reference/fallback.
    return conv2d_im2col[
        N, Hi, Wi, Cin, Kh, Kw, Cout, stride_h, stride_w, pad_h, pad_w
    ](x, weight, bias, ctx)

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
    # sync removed (single-stream ordering; was kernel-trailing host stall)

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
        # sync removed (single-stream ordering; was kernel-trailing host stall)

    var out_shape = List[Int]()
    out_shape.append(N)
    out_shape.append(Ho)
    out_shape.append(Wo)
    out_shape.append(Cout)
    return Tensor(out_buf^, out_shape^, x.dtype())
