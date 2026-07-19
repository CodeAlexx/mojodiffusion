# torch_bf16.mojo — PyTorch/CUDA-compatible F32 -> BF16 round-to-nearest-even.
#
# PyTorch on NVIDIA SM80+ uses CUDA's `cvt.rn.bf16.f32`; its header fallback is
# equivalent to keeping the high 16 float bits and rounding the discarded low
# half to nearest-even. Mojo's scalar `Float32.cast[DType.bfloat16]()` currently
# differs by one BF16 quantum on some values, so parity-sensitive kernels should
# route final F32->BF16 stores through this helper.
#
# This file intentionally lives under ops/ so LTX2 is not the only caller.

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import floor, log, pow
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _LN2 = Float64(0.69314718055994530942)


def torch_bf16_rne_value(v: Float32) -> BFloat16:
    """Return the BF16 value PyTorch/CUDA would produce for finite F32 values.

    This is a mathematical RNE implementation because Mojo 1.0.0b1 does not
    expose scalar Float32 bit reinterpret in kernels. It handles normal finite
    model values exactly; NaN/Inf fall back to Mojo's native cast because these
    paths are not valid production latents/weights.
    """
    if not (v == v):
        return v.cast[DType.bfloat16]()
    if v == Float32(0.0):
        return BFloat16(0.0)

    var sign = Float32(1.0)
    var a = v
    if a < Float32(0.0):
        sign = Float32(-1.0)
        a = -a

    # BF16 has 7 explicit mantissa bits. For a normal value in binade 2^e,
    # adjacent BF16 values are spaced by 2^(e - 7).
    var av = Float64(a)
    var e = Int(floor(log(av) / _LN2))
    var step = pow(Float64(2.0), Float64(e - 7))
    var y = av / step
    var kf = floor(y)
    var frac = y - kf
    var k = Int(kf)
    if frac > Float64(0.5) or (frac == Float64(0.5) and (k & 1) != 0):
        k += 1

    var q = Float32(Float64(k) * step)
    if sign < Float32(0.0):
        q = -q
    return q.cast[DType.bfloat16]()


def _torch_f32_to_bf16_rne_kernel(
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](src[i])
        dst[i] = rebind[dst.element_type](torch_bf16_rne_value(v))


def torch_f32_to_bf16_rne(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Cast an F32 tensor to BF16 using PyTorch/CUDA RNE semantics."""
    if x.dtype() != STDtype.F32:
        raise Error("torch_f32_to_bf16_rne: expected F32 tensor")
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * STDtype.BF16.byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_torch_f32_to_bf16_rne_kernel, _torch_f32_to_bf16_rne_kernel](
        X, O, n, grid_dim=grid, block_dim=_BLOCK,
    )
    return Tensor(out_buf^, x.shape(), STDtype.BF16)


def _bf16_f32_broadcast_mul_kernel(
    lhs: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    rhs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    channels: Int,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var a = rebind[Scalar[DType.bfloat16]](lhs[i]).cast[DType.float32]()
        var b = rebind[Scalar[DType.float32]](rhs[i // channels])
        dst[i] = rebind[dst.element_type](a * b)


def _f32_one_minus_kernel(
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](src[i])
        dst[i] = rebind[dst.element_type](Float32(1.0) - v)


def _f32_add_to_bf16_rne_kernel(
    lhs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rhs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var a = rebind[Scalar[DType.float32]](lhs[i])
        var b = rebind[Scalar[DType.float32]](rhs[i])
        dst[i] = rebind[dst.element_type](torch_bf16_rne_value(a + b))


def _bf16_f32_add_to_bf16_rne_kernel(
    lhs: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    rhs: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var a = rebind[Scalar[DType.bfloat16]](lhs[i]).cast[DType.float32]()
        var b = rebind[Scalar[DType.float32]](rhs[i])
        dst[i] = rebind[dst.element_type](torch_bf16_rne_value(a + b))


def _bf16_sub_to_f32_kernel(
    lhs: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    rhs: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var a = rebind[Scalar[DType.bfloat16]](lhs[i]).cast[DType.float32]()
        var b = rebind[Scalar[DType.bfloat16]](rhs[i]).cast[DType.float32]()
        dst[i] = rebind[dst.element_type](a - b)


def _f32_div_to_bf16_rne_kernel(
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    sigma: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](src[i])
        dst[i] = rebind[dst.element_type](torch_bf16_rne_value(v / sigma))


def _bf16_scale_to_f32_kernel(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    scale: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](src[i]).cast[DType.float32]()
        dst[i] = rebind[dst.element_type](v * scale)


def torch_bf16_eager_blend_with_f32_mask(
    noise: Tensor, clean: Tensor, mask: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Match PyTorch eager `(noise * mask + clean * (1 - mask)).to(bfloat16)`.

    The F32 temporaries are intentional. PyTorch materializes the multiply,
    subtraction, second multiply, and addition as F32 tensors before the final
    BF16 cast. Collapsing this into one fused kernel changes tie cases by one
    BF16 quantum, which breaks creator/LTX2 noiser parity and can affect other
    diffusion handoffs.
    """
    if noise.dtype() != STDtype.BF16 or clean.dtype() != STDtype.BF16:
        raise Error("torch_bf16_eager_blend_with_f32_mask: expected BF16 inputs")
    if mask.dtype() != STDtype.F32:
        raise Error("torch_bf16_eager_blend_with_f32_mask: expected F32 mask")
    if noise.numel() != clean.numel():
        raise Error("torch_bf16_eager_blend_with_f32_mask: input size mismatch")

    var n = clean.numel()
    var mask_n = mask.numel()
    if mask_n <= 0 or n % mask_n != 0:
        raise Error("torch_bf16_eager_blend_with_f32_mask: mask does not broadcast over channels")
    var channels = n // mask_n
    var n_f32_bytes = n * STDtype.F32.byte_size()
    var mask_f32_bytes = mask_n * STDtype.F32.byte_size()

    var left_buf = ctx.enqueue_create_buffer[DType.uint8](n_f32_bytes)
    var inv_mask_buf = ctx.enqueue_create_buffer[DType.uint8](mask_f32_bytes)
    var right_buf = ctx.enqueue_create_buffer[DType.uint8](n_f32_bytes)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * STDtype.BF16.byte_size())

    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var mask_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](mask_n))
    var N = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        noise.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var C = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        clean.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        mask.buf.unsafe_ptr().bitcast[Float32](), mask_rl
    )
    var L = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        left_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var IM = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        inv_mask_buf.unsafe_ptr().bitcast[Float32](), mask_rl
    )
    var R = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        right_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl
    )

    var grid = (n + _BLOCK - 1) // _BLOCK
    var mask_grid = (mask_n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _bf16_f32_broadcast_mul_kernel, _bf16_f32_broadcast_mul_kernel
    ](N, M, L, channels, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.enqueue_function[_f32_one_minus_kernel, _f32_one_minus_kernel](
        M, IM, mask_n, grid_dim=mask_grid, block_dim=_BLOCK
    )
    ctx.enqueue_function[
        _bf16_f32_broadcast_mul_kernel, _bf16_f32_broadcast_mul_kernel
    ](C, IM, R, channels, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.enqueue_function[_f32_add_to_bf16_rne_kernel, _f32_add_to_bf16_rne_kernel](
        L, R, O, n, grid_dim=grid, block_dim=_BLOCK
    )

    return Tensor(out_buf^, clean.shape(), STDtype.BF16)


def torch_bf16_eager_add_scaled(
    x: Tensor, velocity: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Match PyTorch eager `(x.float() + velocity.float() * scale).to(bfloat16)`.

    This is the distilled Euler handoff shape used by LTX2. The scaled velocity
    is materialized as F32 before the add, mirroring PyTorch and preventing
    fused multiply-add tie differences.
    """
    if x.dtype() != STDtype.BF16 or velocity.dtype() != STDtype.BF16:
        raise Error("torch_bf16_eager_add_scaled: expected BF16 tensors")
    if x.numel() != velocity.numel():
        raise Error("torch_bf16_eager_add_scaled: input size mismatch")

    var n = x.numel()
    var f32_bytes = n * STDtype.F32.byte_size()
    var prod_buf = ctx.enqueue_create_buffer[DType.uint8](f32_bytes)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * STDtype.BF16.byte_size())

    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        velocity.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        prod_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl
    )

    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_bf16_scale_to_f32_kernel, _bf16_scale_to_f32_kernel](
        V, P, scale, n, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.enqueue_function[
        _bf16_f32_add_to_bf16_rne_kernel, _bf16_f32_add_to_bf16_rne_kernel
    ](
        X, P, O, n, grid_dim=grid, block_dim=_BLOCK
    )

    return Tensor(out_buf^, x.shape(), STDtype.BF16)


def torch_bf16_eager_velocity_from_x0(
    sample: Tensor, denoised: Tensor, sigma: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Match PyTorch eager `((sample.float() - denoised.float()) / sigma).to(bfloat16)`."""
    if sample.dtype() != STDtype.BF16 or denoised.dtype() != STDtype.BF16:
        raise Error("torch_bf16_eager_velocity_from_x0: expected BF16 tensors")
    if sample.numel() != denoised.numel():
        raise Error("torch_bf16_eager_velocity_from_x0: input size mismatch")
    if sigma == Float32(0.0):
        raise Error("torch_bf16_eager_velocity_from_x0: sigma must be nonzero")

    var n = sample.numel()
    var f32_bytes = n * STDtype.F32.byte_size()
    var diff_buf = ctx.enqueue_create_buffer[DType.uint8](f32_bytes)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * STDtype.BF16.byte_size())

    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        sample.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var D = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        denoised.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var Diff = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        diff_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl
    )

    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_bf16_sub_to_f32_kernel, _bf16_sub_to_f32_kernel](
        X, D, Diff, n, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.enqueue_function[_f32_div_to_bf16_rne_kernel, _f32_div_to_bf16_rne_kernel](
        Diff, O, sigma, n, grid_dim=grid, block_dim=_BLOCK
    )

    return Tensor(out_buf^, sample.shape(), STDtype.BF16)
