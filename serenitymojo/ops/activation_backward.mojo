# ops/activation_backward.mojo — backward kernels for the Tier-1 activations.
#
# Phase T-act of FULL_PORT_TRAINING_PLAN.md (§3 Tier-1 activation backward arms).
# Backward partners of ops/activations.mojo's elementwise forwards.
#
# Each computes  d_x = grad_out * f'(x)  elementwise (one thread per element
# over the flat buffer). F32 interior; only the final store casts back to the
# storage dtype (mirrors activations.mojo's forward dtype dispatch).
#
# Derivatives (math, ported to match flame-core):
#   relu'(x)    = 1 if x > 0 else 0
#   sigmoid'(x) = s*(1-s)            , s = sigmoid(x)
#   tanh'(x)    = 1 - tanh(x)^2
#   silu'(x)    = s*(1 + x*(1-s))    , s = sigmoid(x)   [silu = x*s]
#   gelu'(x)    : tanh-approx, ported VERBATIM from
#                 flame-core/kernels/gelu_backward.cu::gelu_tanh_derivative —
#                 0.5*(1+t) + 0.5*x*(1-t^2)*c0*(1 + 3*c1*x^2),
#                 t = tanh(c0*(x + c1*x^3)), c0=sqrt(2/pi), c1=0.044715.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import exp, tanh
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
# tanh-approx GELU constants (match activations.mojo forward + gelu_backward.cu).
comptime _GELU_C = Float32(0.7978845608028654)   # sqrt(2/pi)
comptime _GELU_A = Float32(0.044715)


@always_inline
def _sigmoid_f32(v: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-v))


@always_inline
def _relu_deriv(x: Float32) -> Float32:
    return Float32(1.0) if x > Float32(0.0) else Float32(0.0)


@always_inline
def _sigmoid_deriv(x: Float32) -> Float32:
    var s = _sigmoid_f32(x)
    return s * (1.0 - s)


@always_inline
def _tanh_deriv(x: Float32) -> Float32:
    var t = tanh(x)
    return 1.0 - t * t


@always_inline
def _silu_deriv(x: Float32) -> Float32:
    # silu = x*s ; silu' = s + x*s*(1-s) = s*(1 + x*(1-s))
    var s = _sigmoid_f32(x)
    return s * (1.0 + x * (1.0 - s))


@always_inline
def _gelu_deriv(x: Float32) -> Float32:
    # Verbatim from flame-core/kernels/gelu_backward.cu::gelu_tanh_derivative.
    var x2 = x * x
    var x3 = x2 * x
    var arg = _GELU_C * (x + _GELU_A * x3)
    var t = tanh(arg)
    var sech2 = 1.0 - t * t
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * _GELU_C * (1.0 + 3.0 * _GELU_A * x2)


# ── one kernel per dtype per arm (F32 / BF16 / F16 storage; F32 interior) ─────
# Each is d_x = grad_out * f'(x). The forward dtype-dispatch in `_run` selects
# the right kernel and reads x at its storage dtype (cast up to F32 for math).

# --- F32 kernels ---
def _relu_bwd_f32(g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](rebind[Scalar[DType.float32]](g[i]) * _relu_deriv(rebind[Scalar[DType.float32]](x[i])))


def _sigmoid_bwd_f32(g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](rebind[Scalar[DType.float32]](g[i]) * _sigmoid_deriv(rebind[Scalar[DType.float32]](x[i])))


def _tanh_bwd_f32(g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](rebind[Scalar[DType.float32]](g[i]) * _tanh_deriv(rebind[Scalar[DType.float32]](x[i])))


def _silu_bwd_f32(g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](rebind[Scalar[DType.float32]](g[i]) * _silu_deriv(rebind[Scalar[DType.float32]](x[i])))


def _gelu_bwd_f32(g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](rebind[Scalar[DType.float32]](g[i]) * _gelu_deriv(rebind[Scalar[DType.float32]](x[i])))


# --- BF16 kernels ---
def _relu_bwd_bf16(g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.bfloat16]](g[i]).cast[DType.float32]()
        var xv = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv * _relu_deriv(xv)).cast[DType.bfloat16]())


def _sigmoid_bwd_bf16(g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.bfloat16]](g[i]).cast[DType.float32]()
        var xv = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv * _sigmoid_deriv(xv)).cast[DType.bfloat16]())


def _tanh_bwd_bf16(g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.bfloat16]](g[i]).cast[DType.float32]()
        var xv = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv * _tanh_deriv(xv)).cast[DType.bfloat16]())


def _silu_bwd_bf16(g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.bfloat16]](g[i]).cast[DType.float32]()
        var xv = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv * _silu_deriv(xv)).cast[DType.bfloat16]())


def _gelu_bwd_bf16(g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.bfloat16]](g[i]).cast[DType.float32]()
        var xv = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv * _gelu_deriv(xv)).cast[DType.bfloat16]())


# --- F16 kernels ---
def _relu_bwd_f16(g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.float16]](g[i]).cast[DType.float32]()
        var xv = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv * _relu_deriv(xv)).cast[DType.float16]())


def _sigmoid_bwd_f16(g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.float16]](g[i]).cast[DType.float32]()
        var xv = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv * _sigmoid_deriv(xv)).cast[DType.float16]())


def _tanh_bwd_f16(g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.float16]](g[i]).cast[DType.float32]()
        var xv = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv * _tanh_deriv(xv)).cast[DType.float16]())


def _silu_bwd_f16(g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.float16]](g[i]).cast[DType.float32]()
        var xv = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv * _silu_deriv(xv)).cast[DType.float16]())


def _gelu_bwd_f16(g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.float16]](g[i]).cast[DType.float32]()
        var xv = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((gv * _gelu_deriv(xv)).cast[DType.float16]())


# ── shared dtype-dispatch runner ─────────────────────────────────────────────
# `arm` selects which derivative: 0=relu 1=sigmoid 2=tanh 3=silu 4=gelu.
def _run(arm: Int, grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    if grad_out.dtype() != x.dtype():
        raise Error("activation_backward: grad_out/x dtype mismatch")
    if grad_out.numel() != x.numel():
        raise Error("activation_backward: grad_out/x numel mismatch")
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl)
        if arm == 0:
            ctx.enqueue_function[_relu_bwd_f32, _relu_bwd_f32](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        elif arm == 1:
            ctx.enqueue_function[_sigmoid_bwd_f32, _sigmoid_bwd_f32](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        elif arm == 2:
            ctx.enqueue_function[_tanh_bwd_f32, _tanh_bwd_f32](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        elif arm == 3:
            ctx.enqueue_function[_silu_bwd_f32, _silu_bwd_f32](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[_gelu_bwd_f32, _gelu_bwd_f32](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](grad_out.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        if arm == 0:
            ctx.enqueue_function[_relu_bwd_bf16, _relu_bwd_bf16](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        elif arm == 1:
            ctx.enqueue_function[_sigmoid_bwd_bf16, _sigmoid_bwd_bf16](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        elif arm == 2:
            ctx.enqueue_function[_tanh_bwd_bf16, _tanh_bwd_bf16](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        elif arm == 3:
            ctx.enqueue_function[_silu_bwd_bf16, _silu_bwd_bf16](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[_gelu_bwd_bf16, _gelu_bwd_bf16](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](grad_out.buf.unsafe_ptr().bitcast[Float16](), rl)
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float16](), rl)
        if arm == 0:
            ctx.enqueue_function[_relu_bwd_f16, _relu_bwd_f16](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        elif arm == 1:
            ctx.enqueue_function[_sigmoid_bwd_f16, _sigmoid_bwd_f16](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        elif arm == 2:
            ctx.enqueue_function[_tanh_bwd_f16, _tanh_bwd_f16](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        elif arm == 3:
            ctx.enqueue_function[_silu_bwd_f16, _silu_bwd_f16](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
        else:
            ctx.enqueue_function[_gelu_bwd_f16, _gelu_bwd_f16](G, X, O, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── public entry points: d_x = grad_out * f'(x) ──────────────────────────────
def relu_backward(grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """relu backward: d_x = grad_out * (1 if x>0 else 0)."""
    return _run(0, grad_out, x, ctx)


def sigmoid_backward(grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """sigmoid backward: d_x = grad_out * s*(1-s), s=sigmoid(x)."""
    return _run(1, grad_out, x, ctx)


def tanh_backward(grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """tanh backward: d_x = grad_out * (1 - tanh(x)^2)."""
    return _run(2, grad_out, x, ctx)


def silu_backward(grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """silu backward: d_x = grad_out * s*(1 + x*(1-s)), s=sigmoid(x)."""
    return _run(3, grad_out, x, ctx)


def gelu_backward(grad_out: Tensor, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """gelu (tanh-approx) backward: matches flame-core gelu_tanh_derivative."""
    return _run(4, grad_out, x, ctx)
