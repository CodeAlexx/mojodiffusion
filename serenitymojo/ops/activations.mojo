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
from serenitymojo.scratch_ring import ScratchRingAllocator


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


def gelu_slab(x: Tensor, ctx: DeviceContext, mut slab: StepSlab) raises -> Tensor:
    """StepSlab variant of `gelu` (C8, L5 ltx2 capture path): BYTE-IDENTICAL —
    same kernel, same grid — only the output buffer comes from slab.alloc."""
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
        ctx.enqueue_function[_gelu_kernel_f32, _gelu_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[_gelu_kernel_bf16, _gelu_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[_gelu_kernel_f16, _gelu_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, x.shape(), x.dtype())


def gelu_scratch(
    x: Tensor, ctx: DeviceContext, mut scratch: ScratchRingAllocator,
    reverse: Bool = False,
) raises -> Tensor:
    """gelu(x) (tanh approx) with the output allocated from the scratch ring.

    BYTE-IDENTICAL to `gelu` (this file): same kernels, same launch params; the
    ONLY difference is that the output buffer comes from `scratch.alloc_tensor`
    instead of `ctx.enqueue_create_buffer`, for opt-in frame-scoped storage."""
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out: Tensor
    if reverse:
        out = scratch.alloc_tensor_reverse(x.shape(), x.dtype())
    else:
        out = scratch.alloc_tensor(x.shape(), x.dtype())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_gelu_kernel_f32, _gelu_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_gelu_kernel_bf16, _gelu_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_gelu_kernel_f16, _gelu_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    return out^


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


# ── leaky_relu ───────────────────────────────────────────────────────────────
#   leaky_relu(x) = x if x >= 0 else slope*x   (torch.nn.functional.leaky_relu)
# Real-ESRGAN (RRDBNet/SRVGGNetCompact) uses negative_slope=0.2 throughout.
@always_inline
def _leaky_relu_f32(v: Float32, slope: Float32) -> Float32:
    return v if v >= Float32(0.0) else slope * v


def _leaky_relu_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
    slope: Float32,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](_leaky_relu_f32(v, slope))


def _leaky_relu_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
    slope: Float32,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_leaky_relu_f32(v, slope).cast[DType.bfloat16]())


def _leaky_relu_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
    slope: Float32,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_leaky_relu_f32(v, slope).cast[DType.float16]())


def leaky_relu(
    x: Tensor, ctx: DeviceContext, negative_slope: Float32 = Float32(0.2)
) raises -> Tensor:
    """leaky_relu(x) = x if x>=0 else negative_slope*x, elementwise (default 0.2)."""
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
        ctx.enqueue_function[_leaky_relu_kernel_f32, _leaky_relu_kernel_f32](
            X, O, n, negative_slope, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_leaky_relu_kernel_bf16, _leaky_relu_kernel_bf16](
            X, O, n, negative_slope, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_leaky_relu_kernel_f16, _leaky_relu_kernel_f16](
            X, O, n, negative_slope, grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── prelu ────────────────────────────────────────────────────────────────────
#   prelu(x)[.., c] = x if x >= 0 else alpha[c]*x   (torch.nn.PReLU, per-channel)
# alpha is a length-C F32 vector; C = x's innermost (NHWC channel) dim. Same
# math as leaky_relu but the negative slope is looked up per channel. Scalar
# math in F32; alpha is always F32 (PReLU params are tiny — no dtype branch).
def _prelu_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
    C: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        var s = rebind[Scalar[DType.float32]](a[i % C])
        o[i] = rebind[o.element_type](v if v >= Float32(0.0) else s * v)


def _prelu_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
    C: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        var s = rebind[Scalar[DType.float32]](a[i % C])
        var r = v if v >= Float32(0.0) else s * v
        o[i] = rebind[o.element_type](r.cast[DType.bfloat16]())


def _prelu_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
    C: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        var s = rebind[Scalar[DType.float32]](a[i % C])
        var r = v if v >= Float32(0.0) else s * v
        o[i] = rebind[o.element_type](r.cast[DType.float16]())


def prelu(x: Tensor, alpha: Tensor, ctx: DeviceContext) raises -> Tensor:
    """prelu(x)[..,c] = x if x>=0 else alpha[c]*x (per-channel; alpha is F32 [C])."""
    var sh = x.shape()
    var C = sh[len(sh) - 1]
    if alpha.dtype().to_mojo_dtype() != DType.float32:
        raise Error("prelu: alpha must be F32")
    if alpha.numel() != C:
        raise Error("prelu: alpha length must equal channel count (last NHWC dim)")
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var al = RuntimeLayout[_DYN1].row_major(IndexList[1](C))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        alpha.buf.unsafe_ptr().bitcast[Float32](), al
    )
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_prelu_kernel_f32, _prelu_kernel_f32](
            X, A, O, n, C, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_prelu_kernel_bf16, _prelu_kernel_bf16](
            X, A, O, n, C, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_prelu_kernel_f16, _prelu_kernel_f16](
            X, A, O, n, C, grid_dim=grid, block_dim=_BLOCK
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
        # PyTorch BF16 `F.silu(gate) * up` rounds the SiLU result before the
        # multiply. Match that storage-dtype boundary while keeping scalar math F32.
        var silu_g = _silu_f32(gv).cast[DType.bfloat16]().cast[DType.float32]()
        o[i] = rebind[o.element_type]((silu_g * uv).cast[DType.bfloat16]())


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


# Packed SwiGLU kernels: gate_up is [.., 2F] laid out [gate(F) | up(F)] per row.
# One thread per output element reads both halves directly (gate at base+c,
# up at base+f+c) so callers skip the two channel slices (MJ-1006).
def _swiglu_packed_kernel_f32(
    gu: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n_out: Int,
    f: Int,
):
    var i = Int(global_idx.x)
    if i < n_out:
        var base = (i // f) * (2 * f)
        var c = i % f
        var gv = rebind[Scalar[DType.float32]](gu[base + c])
        var uv = rebind[Scalar[DType.float32]](gu[base + f + c])
        o[i] = rebind[o.element_type](_silu_f32(gv) * uv)


def _swiglu_packed_kernel_bf16(
    gu: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n_out: Int,
    f: Int,
):
    var i = Int(global_idx.x)
    if i < n_out:
        var base = (i // f) * (2 * f)
        var c = i % f
        var gv = rebind[Scalar[DType.bfloat16]](gu[base + c]).cast[DType.float32]()
        var uv = rebind[Scalar[DType.bfloat16]](gu[base + f + c]).cast[DType.float32]()
        # Match PyTorch BF16 `F.silu(gate) * up`: round SiLU before multiply.
        var silu_g = _silu_f32(gv).cast[DType.bfloat16]().cast[DType.float32]()
        o[i] = rebind[o.element_type]((silu_g * uv).cast[DType.bfloat16]())


def _swiglu_packed_kernel_f16(
    gu: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n_out: Int,
    f: Int,
):
    var i = Int(global_idx.x)
    if i < n_out:
        var base = (i // f) * (2 * f)
        var c = i % f
        var gv = rebind[Scalar[DType.float16]](gu[base + c]).cast[DType.float32]()
        var uv = rebind[Scalar[DType.float16]](gu[base + f + c]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((_silu_f32(gv) * uv).cast[DType.float16]())


def swiglu_packed(gate_up: Tensor, ctx: DeviceContext) raises -> Tensor:
    """SwiGLU of a packed [.., 2F] tensor ([gate|up] channel halves) -> [.., F].
    Equivalent to swiglu(gate_up[..,:F], gate_up[..,F:2F]) but reads the packed
    buffer directly, skipping the two channel slices + their buffers (MJ-1006)."""
    var gushape = gate_up.shape()
    var last = gushape[len(gushape) - 1]
    if last % 2 != 0:
        raise Error("swiglu_packed: last dim must be even (2F)")
    var f = last // 2
    var n_out = gate_up.numel() // 2
    var dt = gate_up.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](gate_up.nbytes() // 2)
    var gu_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](gate_up.numel()))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_out))
    var grid = (n_out + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var GU = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            gate_up.buf.unsafe_ptr().bitcast[Float32](), gu_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_swiglu_packed_kernel_f32, _swiglu_packed_kernel_f32](
            GU, O, n_out, f, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var GU = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            gate_up.buf.unsafe_ptr().bitcast[BFloat16](), gu_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[_swiglu_packed_kernel_bf16, _swiglu_packed_kernel_bf16](
            GU, O, n_out, f, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var GU = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            gate_up.buf.unsafe_ptr().bitcast[Float16](), gu_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl
        )
        ctx.enqueue_function[_swiglu_packed_kernel_f16, _swiglu_packed_kernel_f16](
            GU, O, n_out, f, grid_dim=grid, block_dim=_BLOCK
        )
    var out_shape = List[Int]()
    for i in range(len(gushape) - 1):
        out_shape.append(gushape[i])
    out_shape.append(f)
    return Tensor(out_buf^, out_shape^, gate_up.dtype())


# MiniMax-H3 persists its transformed FC1 rows as [value|gate], the reverse of
# `swiglu_packed`'s canonical [gate|up]. This twin reads that representation
# directly so the block does not materialize two 250+ MiB channel slices.
def _swiglu_packed_value_gate_kernel[
    dtype: DType
](
    vg: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n_out: Int,
    f: Int,
):
    var i = Int(global_idx.x)
    if i < n_out:
        var base = (i // f) * (2 * f)
        var c = i % f
        var uv = rebind[Scalar[dtype]](vg[base + c]).cast[DType.float32]()
        var gv = rebind[Scalar[dtype]](vg[base + f + c]).cast[DType.float32]()
        var silu_g = _silu_f32(gv)
        if dtype == DType.bfloat16:
            # Preserve the existing BF16 boundary exactly.
            silu_g = silu_g.cast[DType.bfloat16]().cast[DType.float32]()
        o[i] = rebind[o.element_type]((silu_g * uv).cast[dtype]())


def swiglu_packed_value_gate(
    value_gate: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """SwiGLU from packed `[value|gate]` halves, without slice buffers."""
    var shape = value_gate.shape()
    if len(shape) < 1 or shape[len(shape) - 1] % 2 != 0:
        raise Error("swiglu_packed_value_gate: last dim must be even")
    var f = shape[len(shape) - 1] // 2
    var n_out = value_gate.numel() // 2
    var out_shape = shape.copy()
    out_shape[len(out_shape) - 1] = f
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](value_gate.nbytes() // 2)
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](value_gate.numel()))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_out))
    var grid = (n_out + _BLOCK - 1) // _BLOCK
    var dt = value_gate.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            value_gate.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[
            _swiglu_packed_value_gate_kernel[DType.float32],
            _swiglu_packed_value_gate_kernel[DType.float32],
        ](X, O, n_out, f, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            value_gate.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[
            _swiglu_packed_value_gate_kernel[DType.bfloat16],
            _swiglu_packed_value_gate_kernel[DType.bfloat16],
        ](X, O, n_out, f, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            value_gate.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[
            _swiglu_packed_value_gate_kernel[DType.float16],
            _swiglu_packed_value_gate_kernel[DType.float16],
        ](X, O, n_out, f, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, out_shape^, value_gate.dtype())


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
