# training/on_device_global_norm.mojo — device 2-stage global L2 grad norm.
#
# NEW STANDALONE kernel. Does NOT replace training/optim.mojo
# `clip_grads_by_global_norm`; it is a faster sibling for the NORM computation.
# Parity is gated against a host F64 sum-of-squares over the SAME grads
# (on_device_global_norm_parity.mojo): the device norm must match cos>=0.999
# (here, scalar-vs-scalar so effectively equal to F32 eps).
#
# Why it's faster: the scalar `clip_grads_by_global_norm` reads EVERY grad back
# to host (N device→host copies) and sums squares on the CPU — N D2H transfers +
# a serial host loop over millions of elements every step. This does the whole
# reduction ON DEVICE in two stages:
#   stage 1: a single kernel over the sum of all grad elements. Each block
#            grid-strides its slice, accumulates sum(g^2) in an F32 register,
#            shared-memory tree-reduces within the block, and atomicAdds the
#            block result into ONE device F32 scalar.
#   stage 2: a single D2H of that 4-byte scalar; sqrt on the host.
# So N grads of any size cost ONE launch + ONE 4-byte D2H instead of N full-grad
# D2H copies. The block-reduce + atomic-into-scalar shape mirrors flame-core's
# sum_bf16_to_f32_scalar_kernel (FLAME_KERNELS.md bf16_reduce.rs:50).
#
# Grads boxed as TArc (Tensor is move-only). F32-only (master grads).
# AGENT-DEFAULT: block size 256, grid capped at 4096 with a grid-stride loop
# (covers arbitrary total element counts — same cap flame-core uses).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt
from std.memory import ArcPointer, stack_allocation
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.atomic import Atomic
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime TArc = ArcPointer[Tensor]
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _GRID_CAP = 4096


# Stage 1: each block grid-strides over the global element range. Per element it
# locates its tensor via the prefix-sum offset table, reads g[j], accumulates
# g^2, tree-reduces in shared memory, atomicAdds the block total into out[0].
def _global_sq_kernel(
    g_addr: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    offs: LayoutTensor[DType.int64, _DYN1, MutAnyOrigin],
    out_scalar: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ntensors: Int,
    total: Int,
):
    var sh = stack_allocation[
        _BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var stride = Int(block_dim.x) * _GRID_CAP
    var gid = Int(block_idx.x) * Int(block_dim.x) + tid
    var acc: Float32 = 0.0
    while gid < total:
        # locate tensor index ti (largest ti with offs[ti] <= gid)
        var ti = 0
        while ti + 1 < ntensors and Int(rebind[Scalar[DType.int64]](offs[ti + 1])) <= gid:
            ti += 1
        var j = gid - Int(rebind[Scalar[DType.int64]](offs[ti]))
        var ga = rebind[Scalar[DType.uint64]](g_addr[ti])
        var gp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(ga))
        var v = gp[j]
        acc += v * v
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
        _ = Atomic[DType.float32].fetch_add(out_scalar.ptr, sh[0])


def on_device_global_norm(
    grads: List[TArc], ctx: DeviceContext
) raises -> Float32:
    """Global L2 norm = sqrt(sum over ALL grads of sum(g*g)), computed fully on
    device with a single 4-byte D2H. F32-only. Matches the host sum-of-squares
    in optim.mojo clip_grads_by_global_norm (to F32 accumulation eps)."""
    var nt = len(grads)
    if nt == 0:
        raise Error("on_device_global_norm: empty grad list")

    # host address + offset tables
    var g_host = ctx.enqueue_create_host_buffer[DType.uint8](nt * 8)
    var off_host = ctx.enqueue_create_host_buffer[DType.uint8]((nt + 1) * 8)
    var gp = g_host.unsafe_ptr().bitcast[UInt64]()
    var op = off_host.unsafe_ptr().bitcast[Int64]()
    var total = 0
    op[0] = Int64(0)
    for i in range(nt):
        if grads[i][].dtype() != STDtype.F32:
            raise Error("on_device_global_norm: all grads must be F32")
        gp[i] = UInt64(Int(grads[i][].buf.unsafe_ptr().bitcast[Float32]()))
        total += grads[i][].numel()
        op[i + 1] = Int64(total)

    var g_dev = ctx.enqueue_create_buffer[DType.uint8](nt * 8)
    var off_dev = ctx.enqueue_create_buffer[DType.uint8]((nt + 1) * 8)
    var scal_dev = ctx.enqueue_create_buffer[DType.uint8](4)
    ctx.enqueue_copy(dst_buf=g_dev, src_buf=g_host)
    ctx.enqueue_copy(dst_buf=off_dev, src_buf=off_host)
    ctx.enqueue_memset(scal_dev, UInt8(0))

    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nt))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nt + 1))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](1))
    var GA = LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin](
        g_dev.unsafe_ptr().bitcast[UInt64](), a_rl)
    var OFF = LayoutTensor[DType.int64, _DYN1, MutAnyOrigin](
        off_dev.unsafe_ptr().bitcast[Int64](), o_rl)
    var OUT = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scal_dev.unsafe_ptr().bitcast[Float32](), s_rl)

    var nblocks = (total + _BLOCK - 1) // _BLOCK
    if nblocks > _GRID_CAP:
        nblocks = _GRID_CAP
    ctx.enqueue_function[_global_sq_kernel, _global_sq_kernel](
        GA, OFF, OUT, nt, total, grid_dim=nblocks, block_dim=_BLOCK,
    )
    # single 4-byte D2H of the summed sum-of-squares
    var host = ctx.enqueue_create_host_buffer[DType.uint8](4)
    ctx.enqueue_copy(dst_buf=host, src_buf=scal_dev)
    ctx.synchronize()
    var ssq = host.unsafe_ptr().bitcast[Float32]()[0]
    return sqrt(ssq)
