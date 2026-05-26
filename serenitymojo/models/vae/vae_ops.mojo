# vae_ops.mojo — VAE-kit-local helpers the foundation does not provide.
#
# The foundation ops/ layer has conv2d / group_norm / silu / linear / sdpa but
# (intentionally, per its chunk scope) NO plain tensor clone and NO plain
# elementwise tensor+tensor add. The VAE decoder needs both:
#   * clone(x)   — a fresh independent device copy (Tensor is Movable-not-
#                  Copyable; we cannot reuse a value after moving it into an op,
#                  and conv2d consumes its Optional[Tensor] bias). One D2D copy.
#   * add(a, b)  — elementwise a + b, same shape/dtype (the resnet residual
#                  `residual + h` and attention `x + out`). F32 math; one
#                  thread per element over the flat buffer (shape-agnostic).
#
# These are genuinely VAE-path glue, not new "ops" — they mirror the foundation
# kernel style (rebind + F32 accumulate + cast on store). Kept here, in the
# team dir, rather than touching ops/.
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


def clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """A fresh, independent device copy of `x` (same bytes/shape/dtype)."""
    var nbytes = x.nbytes()
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def reshape(x: Tensor, var new_shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    """A copy of `x` with a new shape (same numel). Tensor owns its buffer and
    cannot alias, so this is a device clone + metadata change. Used for the
    NHWC flatten/unflatten around the mid-block attention (small at VAE shapes).
    """
    var n = 1
    for i in range(len(new_shape)):
        n *= new_shape[i]
    if n != x.numel():
        raise Error(
            String("reshape: numel mismatch ")
            + String(n)
            + " != "
            + String(x.numel())
        )
    var nbytes = x.nbytes()
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, new_shape^, x.dtype())


# ── elementwise add ───────────────────────────────────────────────────────────


def _add_kernel_f32(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.float32]](a[i])
        var bv = rebind[Scalar[DType.float32]](b[i])
        o[i] = rebind[o.element_type](av + bv)


def _add_kernel_bf16(
    a: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.bfloat16]](a[i]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.bfloat16]](b[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((av + bv).cast[DType.bfloat16]())


def _add_kernel_f16(
    a: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.float16]](a[i]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.float16]](b[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((av + bv).cast[DType.float16]())


def add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Elementwise a + b. Same numel/dtype; result takes a's shape."""
    if a.dtype() != b.dtype():
        raise Error("add: dtype mismatch")
    if a.numel() != b.numel():
        raise Error("add: numel mismatch")
    var dt = a.dtype().to_mojo_dtype()
    var n = a.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](a.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_add_kernel_f32, _add_kernel_f32](
            A, B, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_add_kernel_bf16, _add_kernel_bf16](
            A, B, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_add_kernel_f16, _add_kernel_f16](
            A, B, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, a.shape(), a.dtype())
