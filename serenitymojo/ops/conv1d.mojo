# ops/conv1d.mojo — 1D convolution primitives for the LTX-2 vocoder/BWE chain.
#
# LTX2_PORT_PLAN_2026-05-28 §P-conv. The vocoder critical path needs NCL 1D
# convolution with stride / padding / dilation / groups (incl. depthwise
# groups=C), plus the helpers ConvTranspose1d decomposes into.
#
# Ops (mirrors flame_core / ltx2_vocoder.rs):
#   conv1d(x[B,Cin,L], w[Cout, Cin/g, K], bias[Cout]?, stride, pad, dil, groups)
#       NCL direct convolution, F32 accumulation, cast back to storage dtype.
#       Output length  Lo = (L + 2*pad - dil*(K-1) - 1) // stride + 1.
#       Grouped: output channel oc belongs to group g = oc // (Cout/groups);
#       it reads input channels [g*(Cin/g) : (g+1)*(Cin/g)]. Depthwise = groups=C.
#   zero_insert1d(x, stride)          : insert (stride-1) zeros between samples;
#       Lo = (L-1)*stride + 1                       (ltx2_vocoder.rs:134/182).
#   replicate_pad1d(x, left, right)   : edge-replicate pad on the length axis
#                                                    (ltx2_vocoder.rs:97).
#   conv_transpose1d(x, w[Cin,Cout/g,K], bias?, stride, pad, dil, groups):
#       precompute_conv_transpose_weight (flip last axis + swap Cin/Cout per
#       group, ltx2_vocoder.rs:219) → zero_insert(stride) → side-pad
#       (dil*(K-1)-pad each side) → conv1d(stride=1)   (ltx2_vocoder.rs:816).
#       Output length Lo = (L-1)*stride - 2*pad + dil*(K-1) + 1.
#
# Direct (non-SDK) kernels: the packaged conv2d_gpu_naive_nhwc_rscf kernel is
# NHWC-only and its grouped path is build-specific; a hand-rolled NCL direct
# kernel is bit-reproducible against a host F64 reference (the P-conv gate) and
# avoids the layout round-trips. One thread per output element (B*Cout*Lo).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import ceildiv
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── conv1d direct kernel (NCL), F32 accumulate ────────────────────────────────
# Tensors are passed as flat [-1] LayoutTensors; we index with the explicit
# row-major strides (B,Cin,L) / (Cout,Cin_g,K) / (B,Cout,Lo). One thread per
# output element. Input/weight/bias/output storage stays in the model dtype;
# the kernel casts scalar taps to F32 only for the accumulator.
def _conv1d_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    w: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    B: Int, Cin: Int, L: Int,
    Cout: Int, K: Int, Lo: Int,
    stride: Int, pad: Int, dil: Int, groups: Int, has_bias: Int,
):
    var idx = Int(global_idx.x)
    var total = B * Cout * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var tmp = idx // Lo
    var oc = tmp % Cout
    var b = tmp // Cout

    var cin_g = Cin // groups
    var cout_g = Cout // groups
    var g = oc // cout_g
    var in_base = g * cin_g

    var acc = Float32(0.0)
    var start = lo * stride - pad
    for ic in range(cin_g):
        var x_c = in_base + ic
        var w_row = (oc * cin_g + ic) * K
        var x_row = (b * Cin + x_c) * L
        for k in range(K):
            var li = start + k * dil
            if li >= 0 and li < L:
                var xv = rebind[Scalar[DType.float32]](x[x_row + li])
                var wv = rebind[Scalar[DType.float32]](w[w_row + k])
                acc += xv * wv
    if has_bias != 0:
        acc += rebind[Scalar[DType.float32]](bias[oc])
    o[idx] = rebind[o.element_type](acc)


def _conv1d_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    w: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    bias: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    B: Int, Cin: Int, L: Int,
    Cout: Int, K: Int, Lo: Int,
    stride: Int, pad: Int, dil: Int, groups: Int, has_bias: Int,
):
    var idx = Int(global_idx.x)
    var total = B * Cout * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var tmp = idx // Lo
    var oc = tmp % Cout
    var b = tmp // Cout

    var cin_g = Cin // groups
    var cout_g = Cout // groups
    var g = oc // cout_g
    var in_base = g * cin_g

    var acc = Float32(0.0)
    var start = lo * stride - pad
    for ic in range(cin_g):
        var x_c = in_base + ic
        var w_row = (oc * cin_g + ic) * K
        var x_row = (b * Cin + x_c) * L
        for k in range(K):
            var li = start + k * dil
            if li >= 0 and li < L:
                var xv = rebind[Scalar[DType.bfloat16]](x[x_row + li]).cast[
                    DType.float32
                ]()
                var wv = rebind[Scalar[DType.bfloat16]](w[w_row + k]).cast[
                    DType.float32
                ]()
                acc += xv * wv
    if has_bias != 0:
        acc += rebind[Scalar[DType.bfloat16]](bias[oc]).cast[DType.float32]()
    o[idx] = rebind[o.element_type](acc.cast[DType.bfloat16]())


def _conv1d_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    w: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    bias: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    B: Int, Cin: Int, L: Int,
    Cout: Int, K: Int, Lo: Int,
    stride: Int, pad: Int, dil: Int, groups: Int, has_bias: Int,
):
    var idx = Int(global_idx.x)
    var total = B * Cout * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var tmp = idx // Lo
    var oc = tmp % Cout
    var b = tmp // Cout

    var cin_g = Cin // groups
    var cout_g = Cout // groups
    var g = oc // cout_g
    var in_base = g * cin_g

    var acc = Float32(0.0)
    var start = lo * stride - pad
    for ic in range(cin_g):
        var x_c = in_base + ic
        var w_row = (oc * cin_g + ic) * K
        var x_row = (b * Cin + x_c) * L
        for k in range(K):
            var li = start + k * dil
            if li >= 0 and li < L:
                var xv = rebind[Scalar[DType.float16]](x[x_row + li]).cast[
                    DType.float32
                ]()
                var wv = rebind[Scalar[DType.float16]](w[w_row + k]).cast[
                    DType.float32
                ]()
                acc += xv * wv
    if has_bias != 0:
        acc += rebind[Scalar[DType.float16]](bias[oc]).cast[DType.float32]()
    o[idx] = rebind[o.element_type](acc.cast[DType.float16]())


def conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Int,
    pad: Int,
    dilation: Int,
    groups: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """conv1d (NCL input, [Cout, Cin/groups, K] weight), F32-accumulated.

    x:      [B, Cin, L]                (compute dtype)
    weight: [Cout, Cin/groups, K]      (same dtype as x)
    bias:   [Cout] or None             (same dtype as x)
    returns [B, Cout, Lo]              (x's dtype)
    """
    var xs = x.shape()
    if len(xs) != 3:
        raise Error("conv1d: x must be [B,Cin,L]")
    var B = xs[0]
    var Cin = xs[1]
    var L = xs[2]
    var ws = weight.shape()
    if len(ws) != 3:
        raise Error("conv1d: weight must be [Cout,Cin/groups,K]")
    var Cout = ws[0]
    var cin_g = ws[1]
    var K = ws[2]
    if groups <= 0 or Cin % groups != 0 or Cout % groups != 0:
        raise Error("conv1d: groups must divide Cin and Cout")
    if cin_g != Cin // groups:
        raise Error("conv1d: weight Cin/groups mismatch")
    if x.dtype() != weight.dtype():
        raise Error("conv1d: x/weight dtype mismatch")

    var Lo = (L + 2 * pad - dilation * (K - 1) - 1) // stride + 1
    if Lo <= 0:
        raise Error("conv1d: computed output length <= 0")

    var dt = x.dtype().to_mojo_dtype()
    var total = B * Cout * Lo
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        total * x.dtype().byte_size()
    )

    var has_bias = 0
    var b_n = 1
    if bias:
        ref bt = bias.value()
        if bt.dtype() != x.dtype():
            raise Error("conv1d: bias dtype mismatch")
        if bt.numel() != Cout:
            raise Error("conv1d: bias length != Cout")
        b_n = Cout
        has_bias = 1
    # No-bias launches still pass a valid pointer; has_bias gates all reads.
    var dummy_bias_buf = ctx.enqueue_create_buffer[DType.uint8](
        x.dtype().byte_size()
    )

    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](B * Cin * L))
    var w_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](Cout * cin_g * K))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](b_n))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var W = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float32](), w_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        if bias:
            var Bias = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                bias.value().buf.unsafe_ptr().bitcast[Float32](), b_rl
            )
            ctx.enqueue_function[_conv1d_kernel_f32, _conv1d_kernel_f32](
                X, W, Bias, O, B, Cin, L, Cout, K, Lo,
                stride, pad, dilation, groups, has_bias,
                grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            var Bias = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                dummy_bias_buf.unsafe_ptr().bitcast[Float32](), b_rl
            )
            ctx.enqueue_function[_conv1d_kernel_f32, _conv1d_kernel_f32](
                X, W, Bias, O, B, Cin, L, Cout, K, Lo,
                stride, pad, dilation, groups, has_bias,
                grid_dim=grid, block_dim=_BLOCK,
            )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var W = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[BFloat16](), w_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        if bias:
            var Bias = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                bias.value().buf.unsafe_ptr().bitcast[BFloat16](), b_rl
            )
            ctx.enqueue_function[_conv1d_kernel_bf16, _conv1d_kernel_bf16](
                X, W, Bias, O, B, Cin, L, Cout, K, Lo,
                stride, pad, dilation, groups, has_bias,
                grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            var Bias = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                dummy_bias_buf.unsafe_ptr().bitcast[BFloat16](), b_rl
            )
            ctx.enqueue_function[_conv1d_kernel_bf16, _conv1d_kernel_bf16](
                X, W, Bias, O, B, Cin, L, Cout, K, Lo,
                stride, pad, dilation, groups, has_bias,
                grid_dim=grid, block_dim=_BLOCK,
            )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var W = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float16](), w_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        if bias:
            var Bias = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                bias.value().buf.unsafe_ptr().bitcast[Float16](), b_rl
            )
            ctx.enqueue_function[_conv1d_kernel_f16, _conv1d_kernel_f16](
                X, W, Bias, O, B, Cin, L, Cout, K, Lo,
                stride, pad, dilation, groups, has_bias,
                grid_dim=grid, block_dim=_BLOCK,
            )
        else:
            var Bias = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                dummy_bias_buf.unsafe_ptr().bitcast[Float16](), b_rl
            )
            ctx.enqueue_function[_conv1d_kernel_f16, _conv1d_kernel_f16](
                X, W, Bias, O, B, Cin, L, Cout, K, Lo,
                stride, pad, dilation, groups, has_bias,
                grid_dim=grid, block_dim=_BLOCK,
            )
    ctx.synchronize()

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(Cout)
    out_shape.append(Lo)
    return Tensor(out_buf^, out_shape^, x.dtype())


# ── zero_insert1d ─────────────────────────────────────────────────────────────
# out[b,c, l*stride] = x[b,c,l]; all other positions 0. Lo = (L-1)*stride + 1.
def _zero_insert_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    BC: Int, L: Int, Lo: Int, stride: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var bc = idx // Lo
    var v = Float32(0.0)
    if lo % stride == 0:
        v = rebind[Scalar[DType.float32]](x[bc * L + lo // stride])
    o[idx] = rebind[o.element_type](v)


def _zero_insert_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    BC: Int, L: Int, Lo: Int, stride: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var bc = idx // Lo
    var v = BFloat16(0.0)
    if lo % stride == 0:
        v = rebind[Scalar[DType.bfloat16]](x[bc * L + lo // stride])
    o[idx] = rebind[o.element_type](v)


def _zero_insert_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    BC: Int, L: Int, Lo: Int, stride: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var bc = idx // Lo
    var v = Float16(0.0)
    if lo % stride == 0:
        v = rebind[Scalar[DType.float16]](x[bc * L + lo // stride])
    o[idx] = rebind[o.element_type](v)


def zero_insert1d(x: Tensor, stride: Int, ctx: DeviceContext) raises -> Tensor:
    """Insert (stride-1) zeros between each length-axis sample of [B,C,L].
    Output [B, C, (L-1)*stride + 1]. stride<=1 is a no-op copy."""
    var xs = x.shape()
    if len(xs) != 3:
        raise Error("zero_insert1d: x must be [B,C,L]")
    var B = xs[0]
    var C = xs[1]
    var L = xs[2]
    if stride <= 1:
        # identity copy
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    var BC = B * C
    var Lo = (L - 1) * stride + 1
    var total = BC * Lo
    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        total * x.dtype().byte_size()
    )
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](BC * L))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_zero_insert_kernel_f32, _zero_insert_kernel_f32](
            X, O, BC, L, Lo, stride, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[_zero_insert_kernel_bf16, _zero_insert_kernel_bf16](
            X, O, BC, L, Lo, stride, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[_zero_insert_kernel_f16, _zero_insert_kernel_f16](
            X, O, BC, L, Lo, stride, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(C)
    out_shape.append(Lo)
    return Tensor(out_buf^, out_shape^, x.dtype())


# ── replicate_pad1d ───────────────────────────────────────────────────────────
# Edge-replicate pad on the length axis: out length = L + left + right; the
# first `left` outputs replicate x[...,0], the last `right` replicate x[...,L-1].
def _replicate_pad_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    BC: Int, L: Int, Lo: Int, left: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var bc = idx // Lo
    var li = lo - left
    if li < 0:
        li = 0
    elif li >= L:
        li = L - 1
    o[idx] = rebind[o.element_type](x[bc * L + li])


def _replicate_pad_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    BC: Int, L: Int, Lo: Int, left: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var bc = idx // Lo
    var li = lo - left
    if li < 0:
        li = 0
    elif li >= L:
        li = L - 1
    o[idx] = rebind[o.element_type](x[bc * L + li])


def _replicate_pad_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    BC: Int, L: Int, Lo: Int, left: Int,
):
    var idx = Int(global_idx.x)
    var total = BC * Lo
    if idx >= total:
        return
    var lo = idx % Lo
    var bc = idx // Lo
    var li = lo - left
    if li < 0:
        li = 0
    elif li >= L:
        li = L - 1
    o[idx] = rebind[o.element_type](x[bc * L + li])


def replicate_pad1d(
    x: Tensor, left: Int, right: Int, ctx: DeviceContext
) raises -> Tensor:
    """Edge-replicate pad on the length axis of [B,C,L]. Output [B,C,L+left+right]."""
    var xs = x.shape()
    if len(xs) != 3:
        raise Error("replicate_pad1d: x must be [B,C,L]")
    var B = xs[0]
    var C = xs[1]
    var L = xs[2]
    if left == 0 and right == 0:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    var BC = B * C
    var Lo = L + left + right
    var total = BC * Lo
    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        total * x.dtype().byte_size()
    )
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](BC * L))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_replicate_pad_kernel_f32, _replicate_pad_kernel_f32](
            X, O, BC, L, Lo, left, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[_replicate_pad_kernel_bf16, _replicate_pad_kernel_bf16](
            X, O, BC, L, Lo, left, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[_replicate_pad_kernel_f16, _replicate_pad_kernel_f16](
            X, O, BC, L, Lo, left, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()

    var out_shape = List[Int]()
    out_shape.append(B)
    out_shape.append(C)
    out_shape.append(Lo)
    return Tensor(out_buf^, out_shape^, x.dtype())


# ── precompute_conv_transpose_weight ──────────────────────────────────────────
# ConvTranspose1d weight [Cin, Cout/g, K] -> Conv1d weight [Cout, Cin/g, K].
# Per group g: flip last axis, then swap the Cin and Cout axes
# (ltx2_vocoder.rs:219-232). Done on the host (load-time op; tiny).
def precompute_conv_transpose_weight(
    weight: Tensor, groups: Int, ctx: DeviceContext
) raises -> Tensor:
    var ws = weight.shape()
    if len(ws) != 3:
        raise Error("precompute_conv_transpose_weight: weight must be [Cin,Cout/g,K]")
    var Cin = ws[0]
    var cout_g = ws[1]
    var K = ws[2]
    if Cin % groups != 0:
        raise Error("precompute_conv_transpose_weight: groups must divide Cin")
    var cin_g = Cin // groups
    var Cout = cout_g * groups

    var src = weight.to_host(ctx)  # [Cin, cout_g, K] row-major
    var dst = List[Float32]()
    dst.resize(Cout * cin_g * K, Float32(0.0))

    # grouped reshape of src: [groups, cin_g, cout_g, K]
    # permute -> [groups, cout_g, cin_g, K] then reshape -> [Cout, cin_g, K],
    # with last-axis flip applied.
    for g in range(groups):
        for ocg in range(cout_g):     # within-group out channel
            for icg in range(cin_g):  # within-group in channel
                for k in range(K):
                    # src index in [Cin, cout_g, K]:
                    #   cin = g*cin_g + icg ; cout_idx = ocg ; flipped k
                    var cin = g * cin_g + icg
                    var src_idx = (cin * cout_g + ocg) * K + (K - 1 - k)
                    # dst index in [Cout, cin_g, K]:
                    #   cout = g*cout_g + ocg
                    var cout = g * cout_g + ocg
                    var dst_idx = (cout * cin_g + icg) * K + k
                    dst[dst_idx] = src[src_idx]

    var out_shape = List[Int]()
    out_shape.append(Cout)
    out_shape.append(cin_g)
    out_shape.append(K)
    return Tensor.from_host(dst, out_shape^, weight.dtype(), ctx)


# ── conv_transpose1d ──────────────────────────────────────────────────────────
# Decomposition (ltx2_vocoder.rs:816): precompute weight (flip+swap) ->
# zero_insert(stride) -> side-pad (dil*(K-1)-pad each side) -> conv1d(stride=1).
# Output length Lo = (L-1)*stride - 2*pad + dil*(K-1) + 1.
def conv_transpose1d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Int,
    pad: Int,
    dilation: Int,
    groups: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """ConvTranspose1d via weight-prep + zero-insert + side-pad + conv1d.

    x:      [B, Cin, L]
    weight: [Cin, Cout/groups, K]   (ConvTranspose layout)
    bias:   [Cout] or None
    returns [B, Cout, (L-1)*stride - 2*pad + dilation*(K-1) + 1]
    """
    var ws = weight.shape()
    if len(ws) != 3:
        raise Error("conv_transpose1d: weight must be [Cin,Cout/g,K]")
    var K = ws[2]

    var w_pre = precompute_conv_transpose_weight(weight, groups, ctx)
    var x_zi = zero_insert1d(x, stride, ctx)
    var side = dilation * (K - 1) - pad
    if side < 0:
        raise Error("conv_transpose1d: negative side pad (pad too large)")
    # ConvTranspose side-padding is ZERO padding; conv1d's `pad` arg already
    # zero-pads (out-of-range taps contribute 0), so feed pad=side directly.
    return conv1d(x_zi, w_pre, bias, 1, side, dilation, groups, ctx)
