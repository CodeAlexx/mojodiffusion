# ops/unary.mojo — elementwise unary math: sin, exp, sqrt, rsqrt, tanh, reciprocal.
#
#   sin(x)        = sin(x)
#   exp(x)        = e^x
#   sqrt(x)       = sqrt(x)
#   rsqrt(x)      = 1 / sqrt(x + eps)      (eps clamp avoids Inf/NaN at x=0)
#   tanh(x)       = tanh(x)
#   reciprocal(x) = 1 / (x + sign-aware eps)  (eps clamp avoids Inf/NaN at x=0)
#
# Atoms for snake_beta (sin), STFT magnitude / kaiser filters (exp/sqrt),
# PixelNorm-general / RMS-style (rsqrt), tone-map (tanh/reciprocal). See
# serenitymojo/docs/LTX2_PORT_PLAN_2026-05-28.md §P0.
#
# All are pointwise: one thread per element over the flat buffer (shape-agnostic;
# we treat the Tensor as a 1-D run of numel elements). F32 math; BF16/F16 storage
# upcasts to F32 internally and only the final store casts back. Kernel style
# mirrors the _sigmoid_kernel_{f32,bf16,f16} triplet in ops/activations.mojo.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import exp, tanh, sqrt, sin
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
# Clamp denominators for rsqrt / reciprocal so x≈0 stays finite (no Inf/NaN).
comptime _EPS = Float32(1e-12)


@always_inline
def _sin_f32(v: Float32) -> Float32:
    return sin(v)


@always_inline
def _exp_f32(v: Float32) -> Float32:
    return exp(v)


@always_inline
def _sqrt_f32(v: Float32) -> Float32:
    return sqrt(v)


@always_inline
def _rsqrt_f32(v: Float32) -> Float32:
    # 1/sqrt(x); clamp the radicand at +eps so x<=0 doesn't yield Inf/NaN.
    var r = v
    if r < _EPS:
        r = _EPS
    return Float32(1.0) / sqrt(r)


@always_inline
def _tanh_f32(v: Float32) -> Float32:
    return tanh(v)


@always_inline
def _reciprocal_f32(v: Float32) -> Float32:
    # 1/x; nudge the denominator away from 0 by eps (sign-preserving) so x≈0
    # stays finite.
    var d = v
    if d >= Float32(0.0):
        if d < _EPS:
            d = _EPS
    else:
        if d > -_EPS:
            d = -_EPS
    return Float32(1.0) / d


# ── sin ──────────────────────────────────────────────────────────────────────
def _sin_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_sin_f32(v))


def _sin_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_sin_f32(v).cast[DType.bfloat16]())


def _sin_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_sin_f32(v).cast[DType.float16]())


def sin_op(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """sin(x), elementwise."""
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_sin_kernel_f32, _sin_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_sin_kernel_bf16, _sin_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_sin_kernel_f16, _sin_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── exp ────────────────────────────────────────────────────────────────────────
def _exp_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_exp_f32(v))


def _exp_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_exp_f32(v).cast[DType.bfloat16]())


def _exp_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_exp_f32(v).cast[DType.float16]())


def exp_op(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """exp(x), elementwise."""
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_exp_kernel_f32, _exp_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_exp_kernel_bf16, _exp_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_exp_kernel_f16, _exp_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── sqrt ───────────────────────────────────────────────────────────────────────
def _sqrt_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_sqrt_f32(v))


def _sqrt_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_sqrt_f32(v).cast[DType.bfloat16]())


def _sqrt_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_sqrt_f32(v).cast[DType.float16]())


def sqrt_op(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """sqrt(x), elementwise."""
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_sqrt_kernel_f32, _sqrt_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_sqrt_kernel_bf16, _sqrt_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_sqrt_kernel_f16, _sqrt_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── rsqrt ──────────────────────────────────────────────────────────────────────
def _rsqrt_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_rsqrt_f32(v))


def _rsqrt_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_rsqrt_f32(v).cast[DType.bfloat16]())


def _rsqrt_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_rsqrt_f32(v).cast[DType.float16]())


def rsqrt_op(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """rsqrt(x) = 1/sqrt(x), elementwise. Radicand clamped at +eps (x<=0 finite)."""
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_rsqrt_kernel_f32, _rsqrt_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_rsqrt_kernel_bf16, _rsqrt_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_rsqrt_kernel_f16, _rsqrt_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── tanh ───────────────────────────────────────────────────────────────────────
def _tanh_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_tanh_f32(v))


def _tanh_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_tanh_f32(v).cast[DType.bfloat16]())


def _tanh_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_tanh_f32(v).cast[DType.float16]())


def tanh_op(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """tanh(x), elementwise."""
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_tanh_kernel_f32, _tanh_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_tanh_kernel_bf16, _tanh_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_tanh_kernel_f16, _tanh_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── reciprocal ─────────────────────────────────────────────────────────────────
def _reciprocal_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_reciprocal_f32(v))


def _reciprocal_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_reciprocal_f32(v).cast[DType.bfloat16]())


def _reciprocal_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_reciprocal_f32(v).cast[DType.float16]())


def reciprocal_op(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """reciprocal(x) = 1/x, elementwise. Denominator nudged off 0 by ±eps."""
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_reciprocal_kernel_f32, _reciprocal_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_reciprocal_kernel_bf16, _reciprocal_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_reciprocal_kernel_f16, _reciprocal_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())
