# training/device_loss.mojo — shared device-native training loss leaves.
#
# This module is the fast-path replacement for trainer code that currently
# reads full prediction/target tensors back to host just to compute mean MSE and
# d_pred. Kernels keep pred/target in storage dtype, cast each element to F32 for
# math, reduce the loss sum on device, and write d_pred in the caller-requested
# storage dtype.

from std.atomic import Atomic
from std.gpu import barrier, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.training_arena import (
    TRAINING_ARENA_SYNC_SCALAR_LOG,
    TrainingArena,
)


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _GRID_CAP = 4096
comptime DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE = "device-mse-block-reduce"
comptime DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE_ARENA = "device-mse-block-reduce-arena"
comptime DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE_INTO = "device-mse-block-reduce-into"
comptime DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE_INTO_SCRATCH = "device-mse-block-reduce-into-scratch"


struct DeviceMSELossResult(Movable):
    var loss: Float32
    var d_pred: Tensor
    var scalar_readback_count: Int
    var full_tensor_readback_count: Int
    var sync_count: Int
    var backend: String

    def __init__(
        out self,
        loss: Float32,
        var d_pred: Tensor,
        scalar_readback_count: Int,
        full_tensor_readback_count: Int,
        sync_count: Int,
        backend: String,
    ):
        self.loss = loss
        self.d_pred = d_pred^
        self.scalar_readback_count = scalar_readback_count
        self.full_tensor_readback_count = full_tensor_readback_count
        self.sync_count = sync_count
        self.backend = backend.copy()

    def validate(self) raises:
        if self.scalar_readback_count < 0 or self.full_tensor_readback_count < 0:
            raise Error("DeviceMSELossResult: readback counts must be nonnegative")
        if self.sync_count < 0:
            raise Error("DeviceMSELossResult: sync count must be nonnegative")
        if self.backend == String(""):
            raise Error("DeviceMSELossResult: backend must be labeled")

    def take_d_pred(deinit self) -> Tensor:
        return self.d_pred^


def _supported_loss_dtype(dt: STDtype) -> Bool:
    return dt == STDtype.F32 or dt == STDtype.BF16 or dt == STDtype.F16


def _same_shape(a: Tensor, b: Tensor) -> Bool:
    var ash = a.shape()
    var bsh = b.shape()
    if len(ash) != len(bsh):
        return False
    for i in range(len(ash)):
        if ash[i] != bsh[i]:
            return False
    return True


def _validate_mse_loss_inputs(
    pred: Tensor, target: Tensor, grad_dtype: STDtype
) raises:
    if pred.dtype() != target.dtype():
        raise Error("device_mse_loss_grad: pred/target dtype mismatch")
    if not _same_shape(pred, target):
        raise Error("device_mse_loss_grad: pred/target shape mismatch")
    if not _supported_loss_dtype(pred.dtype()):
        raise Error(
            String("device_mse_loss_grad: unsupported pred dtype ")
            + pred.dtype().name()
        )
    if not _supported_loss_dtype(grad_dtype):
        raise Error(
            String("device_mse_loss_grad: unsupported grad dtype ")
            + grad_dtype.name()
        )
    if pred.numel() <= 0:
        raise Error("device_mse_loss_grad: empty input")


def _validate_mse_loss_output(
    pred: Tensor, target: Tensor, grad_out: Tensor
) raises:
    _validate_mse_loss_inputs(pred, target, grad_out.dtype())
    if not _same_shape(pred, grad_out):
        raise Error("device_mse_loss_grad_into: grad_out shape mismatch")


def _validate_loss_scratch(loss_scratch: Tensor) raises:
    if loss_scratch.dtype() != STDtype.F32:
        raise Error("device_mse_loss_grad_into_scratch: loss_scratch must be F32")
    if loss_scratch.numel() != 1:
        raise Error("device_mse_loss_grad_into_scratch: loss_scratch must have one scalar")


def _mse_loss_grad_kernel[p_dtype: DType, g_dtype: DType](
    pred: LayoutTensor[p_dtype, _DYN1, MutAnyOrigin],
    target: LayoutTensor[p_dtype, _DYN1, MutAnyOrigin],
    grad: LayoutTensor[g_dtype, _DYN1, MutAnyOrigin],
    out_loss_sum: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
    grad_scale: Float32,
):
    var sh = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var stride = Int(block_dim.x) * _GRID_CAP
    var gid = Int(block_idx.x) * Int(block_dim.x) + tid
    var acc: Float32 = 0.0
    while gid < n:
        var p = rebind[Scalar[p_dtype]](pred[gid]).cast[DType.float32]()
        var t = rebind[Scalar[p_dtype]](target[gid]).cast[DType.float32]()
        var diff = p - t
        grad[gid] = rebind[grad.element_type]((diff * grad_scale).cast[g_dtype]())
        acc += diff * diff
        gid += stride
    sh[tid] = acc
    barrier()
    var active = _BLOCK // 2
    while active > 0:
        if tid < active:
            sh[tid] = sh[tid] + sh[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        _ = Atomic[DType.float32].fetch_add(out_loss_sum.ptr, sh[0])


def _device_mse_loss_grad_with_buffers(
    pred: Tensor,
    target: Tensor,
    grad_dtype: STDtype,
    var grad_buf: DeviceBuffer[DType.uint8],
    var loss_dev: DeviceBuffer[DType.uint8],
    ctx: DeviceContext,
    loss_scale: Float32 = Float32(1.0),
    backend: String = String(DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE),
) raises -> DeviceMSELossResult:
    _validate_mse_loss_inputs(pred, target, grad_dtype)
    var n = pred.numel()

    ctx.enqueue_memset(loss_dev, UInt8(0))

    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var loss_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](1))
    var LOSS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        loss_dev.unsafe_ptr().bitcast[Float32](), loss_rl
    )

    var nblocks = (n + _BLOCK - 1) // _BLOCK
    if nblocks > _GRID_CAP:
        nblocks = _GRID_CAP
    var grad_scale = (Float32(2.0) * loss_scale) / Float32(n)
    var pdt = pred.dtype()

    if pdt == STDtype.F32:
        var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        if grad_dtype == STDtype.F32:
            var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                grad_buf.unsafe_ptr().bitcast[Float32](), rl
            )
            ctx.enqueue_function[
                _mse_loss_grad_kernel[DType.float32, DType.float32],
                _mse_loss_grad_kernel[DType.float32, DType.float32],
            ](P, T, G, LOSS, n, grad_scale, grid_dim=nblocks, block_dim=_BLOCK)
        elif grad_dtype == STDtype.BF16:
            var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                grad_buf.unsafe_ptr().bitcast[BFloat16](), rl
            )
            ctx.enqueue_function[
                _mse_loss_grad_kernel[DType.float32, DType.bfloat16],
                _mse_loss_grad_kernel[DType.float32, DType.bfloat16],
            ](P, T, G, LOSS, n, grad_scale, grid_dim=nblocks, block_dim=_BLOCK)
        else:
            var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                grad_buf.unsafe_ptr().bitcast[Float16](), rl
            )
            ctx.enqueue_function[
                _mse_loss_grad_kernel[DType.float32, DType.float16],
                _mse_loss_grad_kernel[DType.float32, DType.float16],
            ](P, T, G, LOSS, n, grad_scale, grid_dim=nblocks, block_dim=_BLOCK)
    elif pdt == STDtype.BF16:
        var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        if grad_dtype == STDtype.F32:
            var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                grad_buf.unsafe_ptr().bitcast[Float32](), rl
            )
            ctx.enqueue_function[
                _mse_loss_grad_kernel[DType.bfloat16, DType.float32],
                _mse_loss_grad_kernel[DType.bfloat16, DType.float32],
            ](P, T, G, LOSS, n, grad_scale, grid_dim=nblocks, block_dim=_BLOCK)
        elif grad_dtype == STDtype.BF16:
            var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                grad_buf.unsafe_ptr().bitcast[BFloat16](), rl
            )
            ctx.enqueue_function[
                _mse_loss_grad_kernel[DType.bfloat16, DType.bfloat16],
                _mse_loss_grad_kernel[DType.bfloat16, DType.bfloat16],
            ](P, T, G, LOSS, n, grad_scale, grid_dim=nblocks, block_dim=_BLOCK)
        else:
            var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                grad_buf.unsafe_ptr().bitcast[Float16](), rl
            )
            ctx.enqueue_function[
                _mse_loss_grad_kernel[DType.bfloat16, DType.float16],
                _mse_loss_grad_kernel[DType.bfloat16, DType.float16],
            ](P, T, G, LOSS, n, grad_scale, grid_dim=nblocks, block_dim=_BLOCK)
    else:
        var P = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var T = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        if grad_dtype == STDtype.F32:
            var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                grad_buf.unsafe_ptr().bitcast[Float32](), rl
            )
            ctx.enqueue_function[
                _mse_loss_grad_kernel[DType.float16, DType.float32],
                _mse_loss_grad_kernel[DType.float16, DType.float32],
            ](P, T, G, LOSS, n, grad_scale, grid_dim=nblocks, block_dim=_BLOCK)
        elif grad_dtype == STDtype.BF16:
            var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                grad_buf.unsafe_ptr().bitcast[BFloat16](), rl
            )
            ctx.enqueue_function[
                _mse_loss_grad_kernel[DType.float16, DType.bfloat16],
                _mse_loss_grad_kernel[DType.float16, DType.bfloat16],
            ](P, T, G, LOSS, n, grad_scale, grid_dim=nblocks, block_dim=_BLOCK)
        else:
            var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                grad_buf.unsafe_ptr().bitcast[Float16](), rl
            )
            ctx.enqueue_function[
                _mse_loss_grad_kernel[DType.float16, DType.float16],
                _mse_loss_grad_kernel[DType.float16, DType.float16],
            ](P, T, G, LOSS, n, grad_scale, grid_dim=nblocks, block_dim=_BLOCK)

    var host = ctx.enqueue_create_host_buffer[DType.uint8](4)
    ctx.enqueue_copy(dst_buf=host, src_buf=loss_dev)
    ctx.synchronize()
    var loss_sum = host.unsafe_ptr().bitcast[Float32]()[0]
    var out = Tensor(grad_buf^, pred.shape(), grad_dtype)
    var result = DeviceMSELossResult(
        (loss_sum / Float32(n)) * loss_scale,
        out^,
        1,
        0,
        1,
        backend,
    )
    result.validate()
    return result^


def device_mse_loss_grad(
    pred: Tensor,
    target: Tensor,
    grad_dtype: STDtype,
    ctx: DeviceContext,
    loss_scale: Float32 = Float32(1.0),
) raises -> DeviceMSELossResult:
    """Mean MSE loss and d_pred on device.

    pred/target must have identical shape and storage dtype. Loss math and the
    reduction are F32. The returned d_pred uses grad_dtype so product trainers
    can keep current F32 upstream-gradient contracts while BF16/F16 model
    tensors remain BF16/F16 at their boundaries.
    """
    _validate_mse_loss_inputs(pred, target, grad_dtype)
    var n = pred.numel()
    var grad_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * grad_dtype.byte_size()
    )
    var loss_dev = ctx.enqueue_create_buffer[DType.uint8](4)
    return _device_mse_loss_grad_with_buffers(
        pred,
        target,
        grad_dtype,
        grad_buf^,
        loss_dev^,
        ctx,
        loss_scale,
        String(DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE),
    )


def device_mse_loss_grad_with_arena(
    pred: Tensor,
    target: Tensor,
    grad_dtype: STDtype,
    mut arena: TrainingArena,
    ctx: DeviceContext,
    loss_scale: Float32 = Float32(1.0),
) raises -> DeviceMSELossResult:
    """Arena-backed MSE loss for product train steps.

    The returned d_pred is a view into arena-owned storage. Use it before
    rewinding past the mark that was active for this call.
    """
    _validate_mse_loss_inputs(pred, target, grad_dtype)
    var n = pred.numel()
    var grad_buf = arena.alloc_bytes(n * grad_dtype.byte_size())
    var loss_dev = arena.alloc_bytes(4)
    var result = _device_mse_loss_grad_with_buffers(
        pred,
        target,
        grad_dtype,
        grad_buf^,
        loss_dev^,
        ctx,
        loss_scale,
        String(DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE_ARENA),
    )
    arena.record_host_device_transfer(result.scalar_readback_count)
    arena.record_sync(TRAINING_ARENA_SYNC_SCALAR_LOG, result.sync_count)
    return result^


def device_mse_loss_grad_into(
    pred: Tensor,
    target: Tensor,
    grad_out: Tensor,
    ctx: DeviceContext,
    loss_scale: Float32 = Float32(1.0),
) raises -> DeviceMSELossResult:
    """Mean MSE loss with d_pred written into caller-owned device storage.

    This is the no-extra-gradient-allocation path for train-step roots that
    already have persistent gradient buffers, such as ZImage's d_patches root.
    The returned d_pred is a view of grad_out's storage.
    """
    _validate_mse_loss_output(pred, target, grad_out)
    var loss_dev = ctx.enqueue_create_buffer[DType.uint8](4)
    var grad_buf = grad_out.buf.create_sub_buffer[DType.uint8](0, grad_out.nbytes())
    return _device_mse_loss_grad_with_buffers(
        pred,
        target,
        grad_out.dtype(),
        grad_buf^,
        loss_dev^,
        ctx,
        loss_scale,
        String(DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE_INTO),
    )


def device_mse_loss_grad_into_scratch(
    pred: Tensor,
    target: Tensor,
    grad_out: Tensor,
    loss_scratch: Tensor,
    ctx: DeviceContext,
    loss_scale: Float32 = Float32(1.0),
) raises -> DeviceMSELossResult:
    """Mean MSE loss with caller-owned d_pred and device loss scratch.

    This is the fixed-root path for capture-friendly train steps: no device
    gradient buffer allocation and no device scalar scratch allocation.
    """
    _validate_mse_loss_output(pred, target, grad_out)
    _validate_loss_scratch(loss_scratch)
    var grad_buf = grad_out.buf.create_sub_buffer[DType.uint8](0, grad_out.nbytes())
    var loss_buf = loss_scratch.buf.create_sub_buffer[DType.uint8](0, 4)
    return _device_mse_loss_grad_with_buffers(
        pred,
        target,
        grad_out.dtype(),
        grad_buf^,
        loss_buf^,
        ctx,
        loss_scale,
        String(DEVICE_LOSS_BACKEND_MSE_BLOCK_REDUCE_INTO_SCRATCH),
    )
