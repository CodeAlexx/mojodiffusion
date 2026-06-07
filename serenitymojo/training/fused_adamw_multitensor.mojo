# training/fused_adamw_multitensor.mojo — FUSED multi-tensor AdamW.
#
# NEW STANDALONE kernel. Does NOT replace training/optim.mojo `adamw_step`; it
# is a faster sibling. Parity is gated against running the scalar per-tensor
# adamw_step in a loop (fused_adamw_multitensor_parity.mojo): the fused result
# must be BIT-EQUAL (cos=1.0, max_abs=0.0) — the per-element AdamW math is
# IDENTICAL to optim.mojo `_adamw_kernel`.
#
# Why it's faster: the scalar path launches ONE kernel per parameter tensor AND
# calls ctx.synchronize() once per step. A LoRA model has dozens-to-hundreds of
# adapter tensors; that's dozens-to-hundreds of kernel launches + a host sync
# every step. This packs all N tensors into ONE grid: a single launch covers the
# sum of all elements, each thread locates its (tensor, intra-tensor) index from
# a device prefix-sum offset table, reconstructs the tensor's param/grad/m/v
# device pointers from a u64 address table, and runs the identical AdamW update.
# One launch, one sync. (The pointer-from-address-in-kernel idiom is the same one
# cap_cache.mojo uses host-side: UnsafePointer[T, MutExternalOrigin](
# unsafe_from_address=Int(addr)).)
#
# Bias correction (bc1 = 1-beta1^t, bc2 = 1-beta2^t) is shared across all
# tensors (they step together at the same t) and precomputed host-side, exactly
# as optim.mojo does. DECOUPLED weight decay is applied to p before the adaptive
# Adam subtraction, matching torch.optim.AdamW / OneTrainer AdamW (NOT Adam+L2).
#
# Params/grads may be F32/BF16/F16. Adam math is F32 inside the kernel, then
# params are written back to their original storage dtype. m/v remain F32
# optimizer-state storage, matching optim.mojo and Adam's moment contract; these
# are not model/checkpoint tensor boundaries. Tensors are boxed as TArc for List
# storage (Tensor is move-only — MOJO_CONVENTIONS §2a).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime TArc = ArcPointer[Tensor]
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# Single fused kernel. One thread per GLOBAL element (sum over all tensors).
# Tables (all device-resident, length = N tensors except `offs` which is N+1):
#   p_addr/g_addr/m_addr/v_addr : u64 device addresses of each tensor's buffer
#   offs                        : element-offset prefix sum, offs[N] = total
# A thread finds its tensor `ti` by scanning offs (N is small — dozens), then
# its intra-tensor index `j = gid - offs[ti]`.
def _fused_adamw_kernel[p_dtype: DType, g_dtype: DType](
    p_addr: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    g_addr: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    m_addr: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    v_addr: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    offs: LayoutTensor[DType.int64, _DYN1, MutAnyOrigin],
    ntensors: Int,
    total: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    bc1: Float32,
    bc2: Float32,
):
    var gid = Int(global_idx.x)
    if gid >= total:
        return
    # locate tensor index ti: largest ti with offs[ti] <= gid
    var ti = 0
    while ti + 1 < ntensors and Int(rebind[Scalar[DType.int64]](offs[ti + 1])) <= gid:
        ti += 1
    var j = gid - Int(rebind[Scalar[DType.int64]](offs[ti]))

    var pa = rebind[Scalar[DType.uint64]](p_addr[ti])
    var ga = rebind[Scalar[DType.uint64]](g_addr[ti])
    var ma = rebind[Scalar[DType.uint64]](m_addr[ti])
    var va = rebind[Scalar[DType.uint64]](v_addr[ti])
    var pp = UnsafePointer[Scalar[p_dtype], MutExternalOrigin](unsafe_from_address=Int(pa))
    var gp = UnsafePointer[Scalar[g_dtype], MutExternalOrigin](unsafe_from_address=Int(ga))
    var mp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(ma))
    var vp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(va))

    var gv = gp[j].cast[DType.float32]()
    var pv = pp[j].cast[DType.float32]()
    if weight_decay > 0.0:
        pv = pv * (1.0 - lr * weight_decay)
    var mi = beta1 * mp[j] + (1.0 - beta1) * gv
    var vi = beta2 * vp[j] + (1.0 - beta2) * gv * gv
    mp[j] = mi
    vp[j] = vi
    var m_hat = mi / bc1
    var v_hat = vi / bc2
    pv = pv - lr * m_hat / (sqrt(v_hat) + eps)
    pp[j] = pv.cast[p_dtype]()


def _supported_param_or_grad_dtype(dt: STDtype) -> Bool:
    return dt == STDtype.F32 or dt == STDtype.BF16 or dt == STDtype.F16


def _launch_fused_adamw[p_dtype: DType, g_dtype: DType](
    ctx: DeviceContext,
    PA: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    GA: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    MA: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    VA: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    OFF: LayoutTensor[DType.int64, _DYN1, MutAnyOrigin],
    nt: Int,
    total: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    bc1: Float32,
    bc2: Float32,
) raises:
    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _fused_adamw_kernel[p_dtype, g_dtype],
        _fused_adamw_kernel[p_dtype, g_dtype],
    ](
        PA, GA, MA, VA, OFF, nt, total, lr, beta1, beta2, eps, weight_decay,
        bc1, bc2, grid_dim=grid, block_dim=_BLOCK,
    )


def fused_adamw_step(
    params: List[TArc],
    grads: List[TArc],
    m_states: List[TArc],
    v_states: List[TArc],
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
) raises:
    """One fused AdamW step over N tensors in a SINGLE launch. params/m/v are
    updated IN PLACE; grads read-only. Params and grads preserve F32/BF16/F16
    storage; m/v are F32 optimizer states. Matching numel per (p,g,m,v).
    Per-element math matches optim.mojo adamw_step in F32."""
    var nt = len(params)
    if nt == 0:
        raise Error("fused_adamw_step: empty tensor list")
    if len(grads) != nt or len(m_states) != nt or len(v_states) != nt:
        raise Error("fused_adamw_step: param/grad/m/v list length mismatch")
    if t < 1:
        raise Error("fused_adamw_step: t must be >= 1")

    # bias correction (host integer power, shared across tensors — same t)
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p

    var param_dtype = params[0][].dtype()
    var grad_dtype = grads[0][].dtype()
    if not _supported_param_or_grad_dtype(param_dtype):
        raise Error(
            String("fused_adamw_step: unsupported param dtype ")
            + param_dtype.name()
        )
    if not _supported_param_or_grad_dtype(grad_dtype):
        raise Error(
            String("fused_adamw_step: unsupported grad dtype ")
            + grad_dtype.name()
        )

    # Build host address + offset tables. Tables are metadata only; tensor
    # payloads stay device-resident and keep their storage dtype.
    var p_host = ctx.enqueue_create_host_buffer[DType.uint8](nt * 8)
    var g_host = ctx.enqueue_create_host_buffer[DType.uint8](nt * 8)
    var m_host = ctx.enqueue_create_host_buffer[DType.uint8](nt * 8)
    var v_host = ctx.enqueue_create_host_buffer[DType.uint8](nt * 8)
    var off_host = ctx.enqueue_create_host_buffer[DType.uint8]((nt + 1) * 8)
    var pp = p_host.unsafe_ptr().bitcast[UInt64]()
    var gp = g_host.unsafe_ptr().bitcast[UInt64]()
    var mp = m_host.unsafe_ptr().bitcast[UInt64]()
    var vp = v_host.unsafe_ptr().bitcast[UInt64]()
    var op = off_host.unsafe_ptr().bitcast[Int64]()

    var total = 0
    op[0] = Int64(0)
    for i in range(nt):
        if params[i][].dtype() != param_dtype:
            raise Error("fused_adamw_step: mixed param dtypes in one launch")
        if grads[i][].dtype() != grad_dtype:
            raise Error("fused_adamw_step: mixed grad dtypes in one launch")
        if m_states[i][].dtype() != STDtype.F32 or v_states[i][].dtype() != STDtype.F32:
            raise Error("fused_adamw_step: m/v optimizer states must be F32")
        var n = params[i][].numel()
        if grads[i][].numel() != n or m_states[i][].numel() != n or v_states[i][].numel() != n:
            raise Error("fused_adamw_step: per-tensor numel mismatch at " + String(i))
        pp[i] = UInt64(Int(params[i][].buf.unsafe_ptr()))
        gp[i] = UInt64(Int(grads[i][].buf.unsafe_ptr()))
        mp[i] = UInt64(Int(m_states[i][].buf.unsafe_ptr().bitcast[Float32]()))
        vp[i] = UInt64(Int(v_states[i][].buf.unsafe_ptr().bitcast[Float32]()))
        total += n
        op[i + 1] = Int64(total)

    var p_dev = ctx.enqueue_create_buffer[DType.uint8](nt * 8)
    var g_dev = ctx.enqueue_create_buffer[DType.uint8](nt * 8)
    var m_dev = ctx.enqueue_create_buffer[DType.uint8](nt * 8)
    var v_dev = ctx.enqueue_create_buffer[DType.uint8](nt * 8)
    var off_dev = ctx.enqueue_create_buffer[DType.uint8]((nt + 1) * 8)
    ctx.enqueue_copy(dst_buf=p_dev, src_buf=p_host)
    ctx.enqueue_copy(dst_buf=g_dev, src_buf=g_host)
    ctx.enqueue_copy(dst_buf=m_dev, src_buf=m_host)
    ctx.enqueue_copy(dst_buf=v_dev, src_buf=v_host)
    ctx.enqueue_copy(dst_buf=off_dev, src_buf=off_host)

    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nt))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nt + 1))
    var PA = LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin](
        p_dev.unsafe_ptr().bitcast[UInt64](), a_rl)
    var GA = LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin](
        g_dev.unsafe_ptr().bitcast[UInt64](), a_rl)
    var MA = LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin](
        m_dev.unsafe_ptr().bitcast[UInt64](), a_rl)
    var VA = LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin](
        v_dev.unsafe_ptr().bitcast[UInt64](), a_rl)
    var OFF = LayoutTensor[DType.int64, _DYN1, MutAnyOrigin](
        off_dev.unsafe_ptr().bitcast[Int64](), o_rl)

    if param_dtype == STDtype.F32:
        if grad_dtype == STDtype.F32:
            _launch_fused_adamw[DType.float32, DType.float32](
                ctx, PA, GA, MA, VA, OFF, nt, total, lr, beta1, beta2, eps,
                weight_decay, bc1, bc2,
            )
        elif grad_dtype == STDtype.BF16:
            _launch_fused_adamw[DType.float32, DType.bfloat16](
                ctx, PA, GA, MA, VA, OFF, nt, total, lr, beta1, beta2, eps,
                weight_decay, bc1, bc2,
            )
        else:
            _launch_fused_adamw[DType.float32, DType.float16](
                ctx, PA, GA, MA, VA, OFF, nt, total, lr, beta1, beta2, eps,
                weight_decay, bc1, bc2,
            )
    elif param_dtype == STDtype.BF16:
        if grad_dtype == STDtype.F32:
            _launch_fused_adamw[DType.bfloat16, DType.float32](
                ctx, PA, GA, MA, VA, OFF, nt, total, lr, beta1, beta2, eps,
                weight_decay, bc1, bc2,
            )
        elif grad_dtype == STDtype.BF16:
            _launch_fused_adamw[DType.bfloat16, DType.bfloat16](
                ctx, PA, GA, MA, VA, OFF, nt, total, lr, beta1, beta2, eps,
                weight_decay, bc1, bc2,
            )
        else:
            _launch_fused_adamw[DType.bfloat16, DType.float16](
                ctx, PA, GA, MA, VA, OFF, nt, total, lr, beta1, beta2, eps,
                weight_decay, bc1, bc2,
            )
    else:
        if grad_dtype == STDtype.F32:
            _launch_fused_adamw[DType.float16, DType.float32](
                ctx, PA, GA, MA, VA, OFF, nt, total, lr, beta1, beta2, eps,
                weight_decay, bc1, bc2,
            )
        elif grad_dtype == STDtype.BF16:
            _launch_fused_adamw[DType.float16, DType.bfloat16](
                ctx, PA, GA, MA, VA, OFF, nt, total, lr, beta1, beta2, eps,
                weight_decay, bc1, bc2,
            )
        else:
            _launch_fused_adamw[DType.float16, DType.float16](
                ctx, PA, GA, MA, VA, OFF, nt, total, lr, beta1, beta2, eps,
                weight_decay, bc1, bc2,
            )
    ctx.synchronize()
