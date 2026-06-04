# ops/fused_bias_gelu.mojo — fused (x + bias) -> GELU(tanh-approx) in one pass.
#
# Math (matches flame-core/src/fused_kernels.rs bias_gelu, ~line 26):
#   z = x[idx] + bias[idx % hidden_size]            # per-channel (last-dim) bias
#   o = 0.5*z*(1 + tanh( sqrt(2/pi) * (z + 0.044715*z³) ))   (tanh-approx GELU)
#
# The Rust reference is all-f32 (c0=0.7978845608, c1=0.044715). Here we keep the
# interior in F32 and only cast to the storage dtype at the final store
# (BF16-store / F32-accumulate, the project-wide invariant). The GELU is REUSED
# from ops/activations.mojo `_gelu_f32` — NOT reimplemented; the fusion is only
# the bias-add + that same GELU in a single kernel pass.
#
# Bias broadcast: bias is the per-hidden-channel vector [H] where H is the last
# dim of x; every leading axis (batch, seq, ...) shares the same bias row, so the
# channel index is `idx % H` exactly as the Rust kernel computes `idx % hidden`.
#
# Convention: kernel-triple (f32 / bf16 / f16) + dtype dispatcher (ops/activations.mojo).
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.activations import _gelu_f32


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── kernels: one thread per element; bias indexed by (i % h) ─────────────────
def _bias_gelu_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
    h: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        var bv = rebind[Scalar[DType.float32]](b[i % h])
        o[i] = rebind[o.element_type](_gelu_f32(v + bv))


def _bias_gelu_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
    h: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.bfloat16]](b[i % h]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_gelu_f32(v + bv).cast[DType.bfloat16]())


def _bias_gelu_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
    h: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.float16]](b[i % h]).cast[DType.float32]()
        o[i] = rebind[o.element_type](_gelu_f32(v + bv).cast[DType.float16]())


def bias_gelu(x: Tensor, bias: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Fused (x + bias) -> GELU(tanh-approx), elementwise.

    bias is the per-hidden-channel vector [H] (H == last dim of x); it is
    broadcast over every leading axis. F32 interior, store cast. Matches
    flame-core bias_gelu (tanh-approx GELU, c0=sqrt(2/pi), c1=0.044715)."""
    if x.dtype() != bias.dtype():
        raise Error("bias_gelu: x/bias dtype mismatch")
    var xshape = x.shape()
    if len(xshape) == 0:
        raise Error("bias_gelu: x must have at least 1 dim")
    var h = xshape[len(xshape) - 1]
    var bshape = bias.shape()
    if len(bshape) != 1 or bshape[0] != h:
        raise Error(
            String("bias_gelu: bias must be [H] with H == last dim of x (H=")
            + String(h) + ")"
        )
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](h))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[Float32](), b_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[_bias_gelu_kernel_f32, _bias_gelu_kernel_f32](
            X, B, O, n, h, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[_bias_gelu_kernel_bf16, _bias_gelu_kernel_bf16](
            X, B, O, n, h, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[Float16](), b_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[_bias_gelu_kernel_f16, _bias_gelu_kernel_f16](
            X, B, O, n, h, grid_dim=grid, block_dim=_BLOCK
        )
    # single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, xshape.copy(), x.dtype())
