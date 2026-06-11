# ops/norm.mojo — rms_norm(x, weight, eps).
#
#   rms_norm: y[..., j] = x[..., j] / sqrt(mean_j(x²) + eps) * weight[j]
#
# SDK-OP STATUS (real finding, see chunk A1 report):
#   The intended SDK op is `nn.normalization.rms_norm_gpu`. Its signature is
#   closure-based — input/output are `capturing` lambdas — AND its `gamma`
#   parameter is a `TileTensor` whose `gamma.origin.mut` cannot be inferred when
#   you pass a plain `LayoutTensor` built over a `DeviceBuffer` in Mojo 1.0.0b1.
#   Every attempt (default origin, MutAnyOrigin, get_immutable()) fails with:
#     "value passed to 'gamma' ... depends on an unresolved parameter
#      'gamma.origin.mut'".
#   This is a genuine SDK/compiler ergonomics wall, not a layout mistake. Per
#   PHASE_AB_PLAN ("or compose if the SDK signature is awkward"), rms_norm is
#   COMPOSED from a hand-rolled GPU kernel below. It is NOT a slow placeholder:
#   one block per row, shared-memory tree reduction of sum(x²) in F32, then a
#   normalized + weighted write cast back to the storage dtype. Numerically it
#   matches torch.nn.functional rms_norm (verified in ops_smoke.mojo vs numpy).
#
# F32 accumulation: x² sum and the normalize/scale arithmetic are all F32 even
# for BF16/F16 storage; only the final store casts down. If checkpoint norm
# weights use a different storage dtype than the activation, the tiny [D] weight
# vector is cast inside this op to the activation dtype before dispatch. That is
# an intentional compute boundary so model loaders can preserve checkpoint dtype.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor, cast_tensor_slab
from serenitymojo.autograd_v2.step_slab import StepSlab


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _DYN1 = Layout.row_major(-1)
comptime _TPB = 256  # threads per block (one block per row)


def _rms_norm_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        local += v * v
        c += _TPB
    shared[tid] = local
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(shared[0] / Float32(cols) + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        var gg = rebind[Scalar[DType.float32]](g[c])
        o[row, c] = rebind[o.element_type](v * inv * gg)
        c += _TPB


def _rms_norm_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        local += v * v
        c += _TPB
    shared[tid] = local
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(shared[0] / Float32(cols) + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        var gg = rebind[Scalar[DType.bfloat16]](g[c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type]((v * inv * gg).cast[DType.bfloat16]())
        c += _TPB


def _rms_norm_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float16]](x[row, c]).cast[DType.float32]()
        local += v * v
        c += _TPB
    shared[tid] = local
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(shared[0] / Float32(cols) + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float16]](x[row, c]).cast[DType.float32]()
        var gg = rebind[Scalar[DType.float16]](g[c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type]((v * inv * gg).cast[DType.float16]())
        c += _TPB


def rms_norm(
    x: Tensor, weight: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    """RMS-normalize over the last dim of x, then scale by `weight`.

    x:      [..., D]   (any compute dtype; leading dims flattened to rows)
    weight: [D]        (same dtype as x, or checkpoint storage dtype)
    returns [..., D]   (x's dtype; F32-accumulated reduction).
    """
    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1:
        raise Error("rms_norm: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    if len(wshape) != 1 or wshape[0] != d:
        raise Error(
            String("rms_norm: weight must be [")
            + String(d)
            + "], got rank "
            + String(len(wshape))
        )
    if x.dtype() != weight.dtype():
        var compute_weight = cast_tensor(weight, x.dtype(), ctx)
        return rms_norm(x, compute_weight^, eps, ctx)
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()
    var nbytes = x.nbytes()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](nbytes)

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float32](), g_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[_rms_norm_kernel_f32, _rms_norm_kernel_f32](
            X, G, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[BFloat16](), g_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[_rms_norm_kernel_bf16, _rms_norm_kernel_bf16](
            X, G, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    else:  # float16
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float16](), g_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[_rms_norm_kernel_f16, _rms_norm_kernel_f16](
            X, G, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, xshape.copy(), x.dtype())


def rms_norm_slab(
    x: Tensor, weight: Tensor, eps: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `rms_norm` (this file :151) — byte-identical math
    (same kernels, same launch params); ONLY the allocation source changes
    (autograd_v2 contract C8, Phase P4)."""
    var xshape = x.shape()
    var wshape = weight.shape()
    if len(xshape) < 1:
        raise Error("rms_norm: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    if len(wshape) != 1 or wshape[0] != d:
        raise Error(
            String("rms_norm: weight must be [")
            + String(d)
            + "], got rank "
            + String(len(wshape))
        )
    if x.dtype() != weight.dtype():
        var compute_weight = cast_tensor_slab(weight, x.dtype(), ctx, slab, False)
        return rms_norm_slab(x, compute_weight^, eps, ctx, slab)
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()
    var nbytes = x.nbytes()
    var out_buf = slab.alloc(nbytes)

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float32](), g_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[_rms_norm_kernel_f32, _rms_norm_kernel_f32](
            X, G, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[BFloat16](), g_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[_rms_norm_kernel_bf16, _rms_norm_kernel_bf16](
            X, G, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    else:  # float16
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float16](), g_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[_rms_norm_kernel_f16, _rms_norm_kernel_f16](
            X, G, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, xshape.copy(), x.dtype())


# ── layer_norm ─────────────────────────────────────────────────────────────
#
#   layer_norm: y[..., j] = (x[..., j] - mean) / sqrt(var + eps) * weight[j]
#                           + bias[j]
#   mean / var are over the last dim (D). var is the BIASED (population)
#   variance = mean(x²) - mean(x)² (matches torch.nn.functional.layer_norm).
#
# Same one-block-per-row, shared-memory F32 reduction shape as rms_norm, but we
# reduce BOTH sum(x) and sum(x²) (two shared arrays). F32 accumulation; only the
# final store casts back to the storage dtype.


def _layer_norm_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var s_sum = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_sqr = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var lsqr: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        lsum += v
        lsqr += v * v
        c += _TPB
    s_sum[tid] = lsum
    s_sqr[tid] = lsqr
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            s_sum[tid] = s_sum[tid] + s_sum[tid + active]
            s_sqr[tid] = s_sqr[tid] + s_sqr[tid + active]
        barrier()
        active //= 2
    var mean = s_sum[0] / Float32(cols)
    var var_ = s_sqr[0] / Float32(cols) - mean * mean
    var inv = 1.0 / sqrt(var_ + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        var gg = rebind[Scalar[DType.float32]](g[c])
        var bb = rebind[Scalar[DType.float32]](b[c])
        o[row, c] = rebind[o.element_type]((v - mean) * inv * gg + bb)
        c += _TPB


def _layer_norm_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var s_sum = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_sqr = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var lsqr: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        lsum += v
        lsqr += v * v
        c += _TPB
    s_sum[tid] = lsum
    s_sqr[tid] = lsqr
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            s_sum[tid] = s_sum[tid] + s_sum[tid + active]
            s_sqr[tid] = s_sqr[tid] + s_sqr[tid + active]
        barrier()
        active //= 2
    var mean = s_sum[0] / Float32(cols)
    var var_ = s_sqr[0] / Float32(cols) - mean * mean
    var inv = 1.0 / sqrt(var_ + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        var gg = rebind[Scalar[DType.bfloat16]](g[c]).cast[DType.float32]()
        var bb = rebind[Scalar[DType.bfloat16]](b[c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type](
            ((v - mean) * inv * gg + bb).cast[DType.bfloat16]()
        )
        c += _TPB


def _layer_norm_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var s_sum = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_sqr = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var lsqr: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float16]](x[row, c]).cast[DType.float32]()
        lsum += v
        lsqr += v * v
        c += _TPB
    s_sum[tid] = lsum
    s_sqr[tid] = lsqr
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            s_sum[tid] = s_sum[tid] + s_sum[tid + active]
            s_sqr[tid] = s_sqr[tid] + s_sqr[tid + active]
        barrier()
        active //= 2
    var mean = s_sum[0] / Float32(cols)
    var var_ = s_sqr[0] / Float32(cols) - mean * mean
    var inv = 1.0 / sqrt(var_ + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float16]](x[row, c]).cast[DType.float32]()
        var gg = rebind[Scalar[DType.float16]](g[c]).cast[DType.float32]()
        var bb = rebind[Scalar[DType.float16]](b[c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type](
            ((v - mean) * inv * gg + bb).cast[DType.float16]()
        )
        c += _TPB


def layer_norm(
    x: Tensor, weight: Tensor, bias: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Layer-normalize over the last dim of x, then scale + shift.

    x:      [..., D]   (any compute dtype; leading dims flattened to rows)
    weight: [D]        (gamma; same dtype as x, or checkpoint storage dtype)
    bias:   [D]        (beta; same dtype as x, or checkpoint storage dtype)
    returns [..., D]   (x's dtype; F32-accumulated mean/var).
    """
    var xshape = x.shape()
    var wshape = weight.shape()
    var bshape = bias.shape()
    if len(xshape) < 1:
        raise Error("layer_norm: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    if len(wshape) != 1 or wshape[0] != d:
        raise Error("layer_norm: weight must be [D] matching x last dim")
    if len(bshape) != 1 or bshape[0] != d:
        raise Error("layer_norm: bias must be [D] matching x last dim")
    if x.dtype() != weight.dtype():
        var compute_weight = cast_tensor(weight, x.dtype(), ctx)
        if x.dtype() != bias.dtype():
            var compute_bias = cast_tensor(bias, x.dtype(), ctx)
            return layer_norm(x, compute_weight^, compute_bias^, eps, ctx)
        return layer_norm(x, compute_weight^, bias, eps, ctx)
    if x.dtype() != bias.dtype():
        var compute_bias = cast_tensor(bias, x.dtype(), ctx)
        return layer_norm(x, weight, compute_bias^, eps, ctx)
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float32](), g_rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[Float32](), g_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[_layer_norm_kernel_f32, _layer_norm_kernel_f32](
            X, G, B, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[BFloat16](), g_rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[BFloat16](), g_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[_layer_norm_kernel_bf16, _layer_norm_kernel_bf16](
            X, G, B, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    else:  # float16
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float16](), g_rl
        )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[Float16](), g_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[_layer_norm_kernel_f16, _layer_norm_kernel_f16](
            X, G, B, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, xshape.copy(), x.dtype())


def layer_norm_slab(
    x: Tensor, weight: Tensor, bias: Tensor, eps: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `layer_norm` (above) — byte-identical math (same
    kernels, same launch params); ONLY the allocation source changes
    (autograd_v2 contracts C8/C9, Phase P5). Mixed-dtype gamma/beta cast
    routes through cast_tensor_slab (sync-free)."""
    var xshape = x.shape()
    var wshape = weight.shape()
    var bshape = bias.shape()
    if len(xshape) < 1:
        raise Error("layer_norm: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    if len(wshape) != 1 or wshape[0] != d:
        raise Error("layer_norm: weight must be [D] matching x last dim")
    if len(bshape) != 1 or bshape[0] != d:
        raise Error("layer_norm: bias must be [D] matching x last dim")
    if x.dtype() != weight.dtype():
        var compute_weight = cast_tensor_slab(weight, x.dtype(), ctx, slab, False)
        if x.dtype() != bias.dtype():
            var compute_bias = cast_tensor_slab(bias, x.dtype(), ctx, slab, False)
            return layer_norm_slab(x, compute_weight^, compute_bias^, eps, ctx, slab)
        return layer_norm_slab(x, compute_weight^, bias, eps, ctx, slab)
    if x.dtype() != bias.dtype():
        var compute_bias = cast_tensor_slab(bias, x.dtype(), ctx, slab, False)
        return layer_norm_slab(x, weight, compute_bias^, eps, ctx, slab)
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = slab.alloc(x.nbytes())

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float32](), g_rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[Float32](), g_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[_layer_norm_kernel_f32, _layer_norm_kernel_f32](
            X, G, B, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[BFloat16](), g_rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[BFloat16](), g_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[_layer_norm_kernel_bf16, _layer_norm_kernel_bf16](
            X, G, B, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    else:  # float16
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float16](), g_rl
        )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[Float16](), g_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[_layer_norm_kernel_f16, _layer_norm_kernel_f16](
            X, G, B, O, d, eps, grid_dim=rows, block_dim=_TPB
        )
    # P5-CAPTURE-SYNC-REMOVED (C9): single-stream ordering; no sync inside a
    # captured region.
    return Tensor(out_buf^, xshape.copy(), x.dtype())


# ── layer_norm_no_affine ────────────────────────────────────────────────────
#
#   layer_norm_no_affine: y[...,j] = (x[...,j] - mean) / sqrt(var + eps)
#   over the last dim. NO learned gamma/beta (elementwise_affine=False).
#
# This is the AdaLN(-zero) normalization in every Phase 2-4 DiT (Wan2.2,
# Hunyuan1.5, Kandinsky5, Nava-AV, plus SD3/Qwen-Image/LTX2/FLUX/Chroma): the
# block normalizes with no affine, then `modulate((1+scale)·x + shift)` applies
# the timestep-conditioned affine separately. Mirrors flame-core
# layer_norm_bf16(x, None, None, eps). Biased (population) variance, same F32
# tree-reduce as `layer_norm`; only the store path drops the gamma/beta multiply.
def _layer_norm_noaffine_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var s_sum = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_sqr = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var lsqr: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        lsum += v
        lsqr += v * v
        c += _TPB
    s_sum[tid] = lsum
    s_sqr[tid] = lsqr
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            s_sum[tid] = s_sum[tid] + s_sum[tid + active]
            s_sqr[tid] = s_sqr[tid] + s_sqr[tid + active]
        barrier()
        active //= 2
    var mean = s_sum[0] / Float32(cols)
    var var_ = s_sqr[0] / Float32(cols) - mean * mean
    var inv = 1.0 / sqrt(var_ + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        o[row, c] = rebind[o.element_type]((v - mean) * inv)
        c += _TPB


def _layer_norm_noaffine_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var s_sum = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_sqr = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var lsqr: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        lsum += v
        lsqr += v * v
        c += _TPB
    s_sum[tid] = lsum
    s_sqr[tid] = lsqr
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            s_sum[tid] = s_sum[tid] + s_sum[tid + active]
            s_sqr[tid] = s_sqr[tid] + s_sqr[tid + active]
        barrier()
        active //= 2
    var mean = s_sum[0] / Float32(cols)
    var var_ = s_sqr[0] / Float32(cols) - mean * mean
    var inv = 1.0 / sqrt(var_ + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type](
            ((v - mean) * inv).cast[DType.bfloat16]()
        )
        c += _TPB


def _layer_norm_noaffine_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var s_sum = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_sqr = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var lsqr: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float16]](x[row, c]).cast[DType.float32]()
        lsum += v
        lsqr += v * v
        c += _TPB
    s_sum[tid] = lsum
    s_sqr[tid] = lsqr
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            s_sum[tid] = s_sum[tid] + s_sum[tid + active]
            s_sqr[tid] = s_sqr[tid] + s_sqr[tid + active]
        barrier()
        active //= 2
    var mean = s_sum[0] / Float32(cols)
    var var_ = s_sqr[0] / Float32(cols) - mean * mean
    var inv = 1.0 / sqrt(var_ + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float16]](x[row, c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type](
            ((v - mean) * inv).cast[DType.float16]()
        )
        c += _TPB


def layer_norm_no_affine(
    x: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Layer-normalize over the last dim of x with NO affine (no gamma/beta).

    x:      [..., D]   (any compute dtype; leading dims flattened to rows)
    returns [..., D]   (x's dtype; F32-accumulated mean/biased-var).

    y = (x - mean) / sqrt(var + eps). The DiT block applies the
    timestep-conditioned (1+scale)·y + shift separately via `modulate`.
    """
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("layer_norm_no_affine: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[
            _layer_norm_noaffine_kernel_f32, _layer_norm_noaffine_kernel_f32
        ](X, O, d, eps, grid_dim=rows, block_dim=_TPB)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[
            _layer_norm_noaffine_kernel_bf16, _layer_norm_noaffine_kernel_bf16
        ](X, O, d, eps, grid_dim=rows, block_dim=_TPB)
    else:  # float16
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[
            _layer_norm_noaffine_kernel_f16, _layer_norm_noaffine_kernel_f16
        ](X, O, d, eps, grid_dim=rows, block_dim=_TPB)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, xshape.copy(), x.dtype())


# ── group_norm (NHWC) ──────────────────────────────────────────────────────
#
#   group_norm: split the C channels into `num_groups` groups; normalize each
#   sample's group over (H, W, channels-in-group). Then per-channel affine:
#       y = (x - mean_g) / sqrt(var_g + eps) * weight[c] + bias[c]
#   var is the biased variance over the group's H*W*(C/G) elements.
#
# LAYOUT: input is NHWC (the layout our conv2d/VAE path uses). For a sample n
# and group gi, the group spans channels [gi*cpg, (gi+1)*cpg) across ALL (h, w)
# positions. We launch ONE BLOCK PER (n, group): grid = N * num_groups. Each
# block reduces sum and sum-of-squares over its group's H*W*cpg elements with a
# strided loop + shared-memory tree reduction (same F32 pattern as rms_norm).
# We index the flat NHWC buffer manually: offset(n,h,w,c) = ((n*H + h)*W + w)*C + c.


def _group_norm_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n_dim: Int,
    h_dim: Int,
    w_dim: Int,
    c_dim: Int,
    num_groups: Int,
    eps: Float32,
):
    var blk = Int(block_idx.x)  # = n * num_groups + gi
    var n = blk // num_groups
    var gi = blk % num_groups
    var cpg = c_dim // num_groups
    var c0 = gi * cpg
    var hw = h_dim * w_dim
    var group_elems = hw * cpg
    var tid = Int(thread_idx.x)
    var s_sum = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_sqr = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var lsqr: Float32 = 0.0
    # Flatten the group's (hw, cpg) space into a single strided loop.
    var i = tid
    while i < group_elems:
        var pix = i // cpg  # which (h,w)
        var cc = i % cpg  # channel-in-group
        var off = (n * hw + pix) * c_dim + (c0 + cc)
        var v = rebind[Scalar[DType.float32]](x[off])
        lsum += v
        lsqr += v * v
        i += _TPB
    s_sum[tid] = lsum
    s_sqr[tid] = lsqr
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            s_sum[tid] = s_sum[tid] + s_sum[tid + active]
            s_sqr[tid] = s_sqr[tid] + s_sqr[tid + active]
        barrier()
        active //= 2
    var mean = s_sum[0] / Float32(group_elems)
    var var_ = s_sqr[0] / Float32(group_elems) - mean * mean
    var inv = 1.0 / sqrt(var_ + eps)
    i = tid
    while i < group_elems:
        var pix = i // cpg
        var cc = i % cpg
        var ch = c0 + cc
        var off = (n * hw + pix) * c_dim + ch
        var v = rebind[Scalar[DType.float32]](x[off])
        var gg = rebind[Scalar[DType.float32]](g[ch])
        var bb = rebind[Scalar[DType.float32]](b[ch])
        o[off] = rebind[o.element_type]((v - mean) * inv * gg + bb)
        i += _TPB


def _group_norm_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n_dim: Int,
    h_dim: Int,
    w_dim: Int,
    c_dim: Int,
    num_groups: Int,
    eps: Float32,
):
    var blk = Int(block_idx.x)
    var n = blk // num_groups
    var gi = blk % num_groups
    var cpg = c_dim // num_groups
    var c0 = gi * cpg
    var hw = h_dim * w_dim
    var group_elems = hw * cpg
    var tid = Int(thread_idx.x)
    var s_sum = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_sqr = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var lsqr: Float32 = 0.0
    var i = tid
    while i < group_elems:
        var pix = i // cpg
        var cc = i % cpg
        var off = (n * hw + pix) * c_dim + (c0 + cc)
        var v = rebind[Scalar[DType.bfloat16]](x[off]).cast[DType.float32]()
        lsum += v
        lsqr += v * v
        i += _TPB
    s_sum[tid] = lsum
    s_sqr[tid] = lsqr
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            s_sum[tid] = s_sum[tid] + s_sum[tid + active]
            s_sqr[tid] = s_sqr[tid] + s_sqr[tid + active]
        barrier()
        active //= 2
    var mean = s_sum[0] / Float32(group_elems)
    var var_ = s_sqr[0] / Float32(group_elems) - mean * mean
    var inv = 1.0 / sqrt(var_ + eps)
    i = tid
    while i < group_elems:
        var pix = i // cpg
        var cc = i % cpg
        var ch = c0 + cc
        var off = (n * hw + pix) * c_dim + ch
        var v = rebind[Scalar[DType.bfloat16]](x[off]).cast[DType.float32]()
        var gg = rebind[Scalar[DType.bfloat16]](g[ch]).cast[DType.float32]()
        var bb = rebind[Scalar[DType.bfloat16]](b[ch]).cast[DType.float32]()
        o[off] = rebind[o.element_type](
            ((v - mean) * inv * gg + bb).cast[DType.bfloat16]()
        )
        i += _TPB


def _group_norm_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n_dim: Int,
    h_dim: Int,
    w_dim: Int,
    c_dim: Int,
    num_groups: Int,
    eps: Float32,
):
    var blk = Int(block_idx.x)
    var n = blk // num_groups
    var gi = blk % num_groups
    var cpg = c_dim // num_groups
    var c0 = gi * cpg
    var hw = h_dim * w_dim
    var group_elems = hw * cpg
    var tid = Int(thread_idx.x)
    var s_sum = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_sqr = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lsum: Float32 = 0.0
    var lsqr: Float32 = 0.0
    var i = tid
    while i < group_elems:
        var pix = i // cpg
        var cc = i % cpg
        var off = (n * hw + pix) * c_dim + (c0 + cc)
        var v = rebind[Scalar[DType.float16]](x[off]).cast[DType.float32]()
        lsum += v
        lsqr += v * v
        i += _TPB
    s_sum[tid] = lsum
    s_sqr[tid] = lsqr
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            s_sum[tid] = s_sum[tid] + s_sum[tid + active]
            s_sqr[tid] = s_sqr[tid] + s_sqr[tid + active]
        barrier()
        active //= 2
    var mean = s_sum[0] / Float32(group_elems)
    var var_ = s_sqr[0] / Float32(group_elems) - mean * mean
    var inv = 1.0 / sqrt(var_ + eps)
    i = tid
    while i < group_elems:
        var pix = i // cpg
        var cc = i % cpg
        var ch = c0 + cc
        var off = (n * hw + pix) * c_dim + ch
        var v = rebind[Scalar[DType.float16]](x[off]).cast[DType.float32]()
        var gg = rebind[Scalar[DType.float16]](g[ch]).cast[DType.float32]()
        var bb = rebind[Scalar[DType.float16]](b[ch]).cast[DType.float32]()
        o[off] = rebind[o.element_type](
            ((v - mean) * inv * gg + bb).cast[DType.float16]()
        )
        i += _TPB


def group_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    num_groups: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Group-normalize an NHWC tensor.

    x:      [N, H, W, C]   (NHWC; any compute dtype)
    weight: [C]            (per-channel gamma; same dtype as x)
    bias:   [C]            (per-channel beta; same dtype as x)
    num_groups: divides C
    returns [N, H, W, C]   (x's dtype; F32-accumulated per-group mean/var).
    """
    var xshape = x.shape()
    if len(xshape) != 4:
        raise Error("group_norm: x must be rank-4 NHWC [N,H,W,C]")
    var n = xshape[0]
    var h = xshape[1]
    var w = xshape[2]
    var c = xshape[3]
    if num_groups <= 0 or c % num_groups != 0:
        raise Error("group_norm: num_groups must divide C")
    var wshape = weight.shape()
    var bshape = bias.shape()
    if len(wshape) != 1 or wshape[0] != c:
        raise Error("group_norm: weight must be [C]")
    if len(bshape) != 1 or bshape[0] != c:
        raise Error("group_norm: bias must be [C]")
    # Mixed checkpoint-weight path (F32 activations + BF16/F16 stored weight):
    # locally widen weight/bias to the activation dtype, mirroring layer_norm's
    # existing precedent (norm.mojo layer_norm) and the group_norm_backward mixed
    # contract (norm_backward.mojo). Only the activation may be the wide F32 dtype
    # — never silently downcast an F32 weight to a narrow activation.
    if x.dtype() != weight.dtype() or x.dtype() != bias.dtype():
        if x.dtype() != STDtype.F32:
            raise Error("group_norm: x/weight/bias dtype mismatch")
        var compute_weight = (
            cast_tensor(weight, x.dtype(), ctx) if weight.dtype() != x.dtype()
            else weight.clone(ctx)
        )
        var compute_bias = (
            cast_tensor(bias, x.dtype(), ctx) if bias.dtype() != x.dtype()
            else bias.clone(ctx)
        )
        return group_norm(x, compute_weight^, compute_bias^, num_groups, eps, ctx)

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())

    var total = x.numel()
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](c))
    var blocks = n * num_groups

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float32](), c_rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[Float32](), c_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[_group_norm_kernel_f32, _group_norm_kernel_f32](
            X, G, B, O, n, h, w, c, num_groups, eps,
            grid_dim=blocks, block_dim=_TPB,
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[BFloat16](), c_rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[BFloat16](), c_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[_group_norm_kernel_bf16, _group_norm_kernel_bf16](
            X, G, B, O, n, h, w, c, num_groups, eps,
            grid_dim=blocks, block_dim=_TPB,
        )
    else:  # float16
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            weight.buf.unsafe_ptr().bitcast[Float16](), c_rl
        )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            bias.buf.unsafe_ptr().bitcast[Float16](), c_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[_group_norm_kernel_f16, _group_norm_kernel_f16](
            X, G, B, O, n, h, w, c, num_groups, eps,
            grid_dim=blocks, block_dim=_TPB,
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, xshape.copy(), x.dtype())
