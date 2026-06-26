# ops/activations.mojo — elementwise activations: silu, gelu(tanh), swiglu.
#
#   silu(x)   = x * sigmoid(x)          = x / (1 + exp(-x))
#   gelu(x)   = 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x³) ))   (tanh-approx)
#   swiglu(g, u) = silu(g) * u           (gate · up; SwiGLU FFN)
#
# All are pointwise: one thread per element over the flat buffer (shape-agnostic;
# we treat the Tensor as a 1-D run of numel elements). F32 math; only the final
# store casts back to the storage dtype. The tanh-approx GELU matches
# torch.nn.functional.gelu(approximate="tanh") (HF diffusion DiTs use this).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import exp, tanh, sqrt, erf
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd_v2.step_slab import StepSlab


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
# sqrt(2/pi) for the tanh GELU approximation.
comptime _GELU_C = Float32(0.7978845608028654)


@always_inline
def _silu_f32(v: Float32) -> Float32:
    return v / (1.0 + exp(-v))


@always_inline
def _sigmoid_f32(v: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-v))


@always_inline
def _gelu_f32(v: Float32) -> Float32:
    var inner = _GELU_C * (v + Float32(0.044715) * v * v * v)
    return Float32(0.5) * v * (1.0 + tanh(inner))


# 1/sqrt(2) for the exact (erf) GELU. Matches CUDA's M_SQRT1_2 used by
# flame-core gelu_exact.cu and torch.nn.GELU(approximate="none").
comptime _INV_SQRT2 = Float32(0.7071067811865476)


@always_inline
def _gelu_exact_f32(v: Float32) -> Float32:
    # y = 0.5 * x * (1 + erf(x / sqrt(2)))  — PyTorch-exact GELU.
    return Float32(0.5) * v * (1.0 + erf(v * _INV_SQRT2))


# ── silu ───────────────────────────────────────────────────────────────────
def _silu_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_silu_f32(v))


def _silu_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_silu_f32(v).cast[DType.bfloat16]())


def _silu_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_silu_f32(v).cast[DType.float16]())


def silu(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """silu(x) = x * sigmoid(x), elementwise."""
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
        ctx.enqueue_function[_silu_kernel_f32, _silu_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_silu_kernel_bf16, _silu_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_silu_kernel_f16, _silu_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── sigmoid ────────────────────────────────────────────────────────────────
def _sigmoid_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_sigmoid_f32(v))


def _sigmoid_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_sigmoid_f32(v).cast[DType.bfloat16]())


def _sigmoid_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_sigmoid_f32(v).cast[DType.float16]())


def sigmoid(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """sigmoid(x) = 1 / (1 + exp(-x)), elementwise."""
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
        ctx.enqueue_function[_sigmoid_kernel_f32, _sigmoid_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_sigmoid_kernel_bf16, _sigmoid_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_sigmoid_kernel_f16, _sigmoid_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


def sigmoid_slab(x: Tensor, ctx: DeviceContext, mut slab: StepSlab) raises -> Tensor:
    """StepSlab variant of `sigmoid` (this file) — byte-identical (same kernel,
    same grid); only the output buffer comes from slab.alloc (autograd_v2 capture
    path, contract C8)."""
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = slab.alloc(x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[_sigmoid_kernel_f32, _sigmoid_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[_sigmoid_kernel_bf16, _sigmoid_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[_sigmoid_kernel_f16, _sigmoid_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── gelu (tanh-approx) ─────────────────────────────────────────────────────
def _gelu_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_gelu_f32(v))


def _gelu_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_gelu_f32(v).cast[DType.bfloat16]())


def _gelu_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_gelu_f32(v).cast[DType.float16]())


def gelu(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """gelu(x), tanh approximation, elementwise."""
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
        ctx.enqueue_function[_gelu_kernel_f32, _gelu_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_gelu_kernel_bf16, _gelu_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_gelu_kernel_f16, _gelu_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── gelu_exact (erf) ───────────────────────────────────────────────────────
#
#   gelu_exact(x) = 0.5*x*(1 + erf(x / sqrt(2)))   (torch GELU approximate="none")
#
# Cosmos-Predict2.5 (and asymflux2) use bare nn.GELU() — the tanh-approx variant
# above diverges ~9e-4 per element from the erf form (flame-core gelu_exact.cu).
# Same one-thread-per-element flat-buffer pattern as `gelu`; F32 math, store cast.
def _gelu_exact_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_gelu_exact_f32(v))


def _gelu_exact_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_gelu_exact_f32(v).cast[DType.bfloat16]())


def _gelu_exact_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_gelu_exact_f32(v).cast[DType.float16]())


def gelu_exact(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """gelu(x), exact (erf) form, elementwise. torch GELU approximate="none"."""
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
        ctx.enqueue_function[_gelu_exact_kernel_f32, _gelu_exact_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_gelu_exact_kernel_bf16, _gelu_exact_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_gelu_exact_kernel_f16, _gelu_exact_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── swiglu ─────────────────────────────────────────────────────────────────
def _swiglu_kernel_f32(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    u: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.float32]](g[i])
        var uv = rebind[Scalar[DType.float32]](u[i])
        o[i] = rebind[o.element_type](_silu_f32(gv) * uv)


def _swiglu_kernel_bf16(
    g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    u: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.bfloat16]](g[i]).cast[DType.float32]()
        var uv = rebind[Scalar[DType.bfloat16]](u[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((_silu_f32(gv) * uv).cast[DType.bfloat16]())


def _swiglu_kernel_f16(
    g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    u: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var gv = rebind[Scalar[DType.float16]](g[i]).cast[DType.float32]()
        var uv = rebind[Scalar[DType.float16]](u[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((_silu_f32(gv) * uv).cast[DType.float16]())


def swiglu(x_gate: Tensor, x_up: Tensor, ctx: DeviceContext) raises -> Tensor:
    """swiglu(gate, up) = silu(gate) * up, elementwise. gate/up same shape."""
    if x_gate.dtype() != x_up.dtype():
        raise Error("swiglu: gate/up dtype mismatch")
    if x_gate.numel() != x_up.numel():
        raise Error("swiglu: gate/up numel mismatch")
    var dt = x_gate.dtype().to_mojo_dtype()
    var n = x_gate.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x_gate.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x_gate.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var U = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x_up.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_swiglu_kernel_f32, _swiglu_kernel_f32](
            G, U, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x_gate.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var U = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x_up.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_swiglu_kernel_bf16, _swiglu_kernel_bf16](
            G, U, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x_gate.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var U = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x_up.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_swiglu_kernel_f16, _swiglu_kernel_f16](
            G, U, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x_gate.shape(), x_gate.dtype())


def swiglu_slab(
    x_gate: Tensor, x_up: Tensor, ctx: DeviceContext, mut slab: StepSlab
) raises -> Tensor:
    """StepSlab variant of `swiglu` (this file :403) — byte-identical math
    (same kernels, same launch params); ONLY the allocation source changes
    (autograd_v2 contract C8, Phase P4)."""
    if x_gate.dtype() != x_up.dtype():
        raise Error("swiglu: gate/up dtype mismatch")
    if x_gate.numel() != x_up.numel():
        raise Error("swiglu: gate/up numel mismatch")
    var dt = x_gate.dtype().to_mojo_dtype()
    var n = x_gate.numel()
    var out_buf = slab.alloc(x_gate.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x_gate.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var U = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x_up.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_swiglu_kernel_f32, _swiglu_kernel_f32](
            G, U, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x_gate.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var U = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x_up.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_swiglu_kernel_bf16, _swiglu_kernel_bf16](
            G, U, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x_gate.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var U = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x_up.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_swiglu_kernel_f16, _swiglu_kernel_f16](
            G, U, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x_gate.shape(), x_gate.dtype())
