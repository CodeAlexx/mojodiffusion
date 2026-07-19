# serenitymojo/llm/sqa.mojo — single-query GQA attention for KV-cache decode.
#
# The square sdpa attends seq x seq. Incremental decode has ONE query row (the
# new token) attending over L cached key/value rows (GQA: each kv head serves
# H/H_kv query heads). This is the new kernel the KV-cache decode needs.
#
# o[hq, :] = softmax( (q[hq] . K[kv, l]) * scale )_l  @ V[kv, l]   (kv = hq // n_rep)
#
# Verified against a CPU reference (sqa_test.mojo) before it is wired into the
# cached decoder.

from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from std.memory import UnsafePointer, stack_allocation
from std.math import exp
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]
comptime BPtr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime _SQA_MAXL = 8192   # max cached-decode length (prompt+gen); scores in
                            # shared = _SQA_MAXL*4B ≈ 32KB, within the 48KB/block cap
comptime _SQA_TPB = 128     # threads/block == dh (thread owns one output dim in pass 3)


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE-RESIDENT single-query GQA — reads q + the K/V cache IN PLACE (no host
# round-trip). Same 3-pass softmax math as _k_sqa, but:
#   - inputs are the bf16 DEVICE tensors (q [1,1,H,dh], cache k/v [1,L,H_kv,dh])
#     in their NATIVE layout — no [L,H_kv,dh]->[H_kv,L,dh] host reorder,
#   - bf16 loads upcast to F32, F32 accumulate (matches the host path's F32 sqa),
#   - one bf16 store per output (F32 accumulator in a per-thread stack array).
# This removes the O(L) host pull of the whole cache per layer per token — the
# O(L²)-total cost that made cached ms/tok grow linearly with L (MEASURED).
# dh is comptime (always 128 here) so the F32 accumulator can be stack-allocated.
# ─────────────────────────────────────────────────────────────────────────────
def _k_sqa_dev[dh: Int](
    q: BPtr,      # [H, dh]         (from q [1,1,H,dh])
    kc: BPtr,     # [L, H_kv, dh]   (cache k [1,L,H_kv,dh])
    vc: BPtr,     # [L, H_kv, dh]   (cache v [1,L,H_kv,dh])
    o: BPtr,      # [H, dh]         (output [1,1,H*dh])
    H: Int,
    H_kv: Int,
    L: Int,
    scale: Float32,
):
    var hq = Int(global_idx.x)
    if hq >= H:
        return
    var n_rep = H // H_kv
    var kv = hq // n_rep
    var qb = hq * dh

    # pass 1: max score
    var m = Float32(-1.0e30)
    for l in range(L):
        var s = Float32(0.0)
        var kb = (l * H_kv + kv) * dh
        for d in range(dh):
            s += q[qb + d].cast[DType.float32]() * kc[kb + d].cast[DType.float32]()
        s *= scale
        if s > m:
            m = s

    # pass 2: denom = sum exp(s - m)
    var denom = Float32(0.0)
    for l in range(L):
        var s = Float32(0.0)
        var kb = (l * H_kv + kv) * dh
        for d in range(dh):
            s += q[qb + d].cast[DType.float32]() * kc[kb + d].cast[DType.float32]()
        s *= scale
        denom += exp(s - m)

    # pass 3: F32 accumulate weighted V, one bf16 store per d
    var acc = stack_allocation[dh, Scalar[DType.float32]]()
    for d in range(dh):
        acc[d] = Float32(0.0)
    for l in range(L):
        var s = Float32(0.0)
        var kb = (l * H_kv + kv) * dh
        for d in range(dh):
            s += q[qb + d].cast[DType.float32]() * kc[kb + d].cast[DType.float32]()
        s *= scale
        var w = exp(s - m) / denom
        var vb = (l * H_kv + kv) * dh
        for d in range(dh):
            acc[d] += w * vc[vb + d].cast[DType.float32]()
    for d in range(dh):
        o[qb + d] = acc[d].cast[DType.bfloat16]()


def sqa_device(
    q: Tensor, kc: Tensor, vc: Tensor, H: Int, H_kv: Int, L: Int, dh: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Device-resident single-query GQA: q [1,1,H,dh] bf16, cache k/v [1,L,H_kv,dh]
    bf16 -> attn [1,1,H*dh] bf16. NO host round-trip. Bit-close (F32 accum) to the
    host `sqa_gpu` path. dh must be 128."""
    var o_buf = ctx.enqueue_create_buffer[DType.uint8](H * dh * 2)  # bf16
    var scale = Float32(1.0) / Float32(Float64(dh) ** 0.5)
    var qp = q.buf.unsafe_ptr().bitcast[BFloat16]()
    var kp = kc.buf.unsafe_ptr().bitcast[BFloat16]()
    var vp = vc.buf.unsafe_ptr().bitcast[BFloat16]()
    var op = o_buf.unsafe_ptr().bitcast[BFloat16]()
    if dh == 128:
        ctx.enqueue_function[_k_sqa_dev[128], _k_sqa_dev[128]](
            qp, kp, vp, op, H, H_kv, L, scale,
            grid_dim=(H + 31) // 32, block_dim=32,
        )
    else:
        raise Error(String("sqa_device: unsupported dh=") + String(dh))
    return Tensor(o_buf^, [1, 1, H * dh], STDtype.BF16)


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL device single-query GQA — one BLOCK per query head (grid=H), _SQA_TPB
# threads split the O(L) reduction so per-token cost stops growing ~linearly with
# L (the serial 1-warp _k_sqa_dev was the O(L)-per-token bottleneck; MEASURED).
#   pass 1  : threads stripe over l, compute score sc[l]=scale·(q·k[l]), local max
#   reduce  : block-tree max → m ; block-tree sum of exp(sc[l]-m) → denom
#   pass 2.5: sc[l] ← exp(sc[l]-m)/denom   (softmax weights, in shared)
#   pass 3  : thread d (d==tid, dh==_SQA_TPB) → o[d]=Σ_l sc[l]·v[l,d]  (coalesced v)
# Same F32-accumulate math as _k_sqa_dev → bit-close; gated by a parity test AND
# the end-to-end decoder cache test. q in shared; scores in shared (≤_SQA_MAXL).
# ─────────────────────────────────────────────────────────────────────────────
def _k_sqa_par[dh: Int, MAXL: Int](
    q: BPtr,      # [H, dh]
    kc: BPtr,     # [L, H_kv, dh]
    vc: BPtr,     # [L, H_kv, dh]
    o: BPtr,      # [H, dh]
    H: Int,
    H_kv: Int,
    L: Int,
    scale: Float32,
):
    var hq = Int(block_idx.x)
    if hq >= H:
        return
    var tid = Int(thread_idx.x)
    var n_rep = H // H_kv
    var kv = hq // n_rep
    var qb = hq * dh

    var sq = stack_allocation[dh, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var sc = stack_allocation[MAXL, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var red = stack_allocation[dh, Scalar[DType.float32], address_space=AddressSpace.SHARED]()

    # load q[hq] → shared (dh == blockDim)
    if tid < dh:
        sq[tid] = q[qb + tid].cast[DType.float32]()
    barrier()

    # pass 1: scores + per-thread local max
    var lmax = Float32(-1.0e30)
    var l = tid
    while l < L:
        var kb = (l * H_kv + kv) * dh
        var s = Float32(0.0)
        for d in range(dh):
            s += sq[d] * kc[kb + d].cast[DType.float32]()
        s *= scale
        sc[l] = s
        if s > lmax:
            lmax = s
        l += dh
    red[tid] = lmax
    barrier()
    # block-tree max
    var stride = dh // 2
    while stride > 0:
        if tid < stride:
            var a = red[tid]
            var b = red[tid + stride]
            red[tid] = a if a > b else b
        barrier()
        stride //= 2
    var m = red[0]
    barrier()

    # pass 2: per-thread local denom
    var lden = Float32(0.0)
    l = tid
    while l < L:
        lden += exp(sc[l] - m)
        l += dh
    red[tid] = lden
    barrier()
    stride = dh // 2
    while stride > 0:
        if tid < stride:
            red[tid] = red[tid] + red[tid + stride]
        barrier()
        stride //= 2
    var denom = red[0]
    barrier()

    # pass 2.5: softmax weights into shared scores
    l = tid
    while l < L:
        sc[l] = exp(sc[l] - m) / denom
        l += dh
    barrier()

    # pass 3: thread d owns output dim d; Σ_l w[l]·v[l,d] (v read coalesced over d)
    if tid < dh:
        var acc = Float32(0.0)
        for ll in range(L):
            var vb = (ll * H_kv + kv) * dh
            acc += sc[ll] * vc[vb + tid].cast[DType.float32]()
        o[qb + tid] = acc.cast[DType.bfloat16]()


def sqa_device_par(
    q: Tensor, kc: Tensor, vc: Tensor, H: Int, H_kv: Int, L: Int, dh: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Parallel device single-query GQA (one block/head). Same contract + numerics
    as `sqa_device`, but O(L) work is split across _SQA_TPB threads so per-token
    cost stays ~flat in L. dh must be 128; L must be ≤ _SQA_MAXL."""
    if dh != 128:
        raise Error(String("sqa_device_par: unsupported dh=") + String(dh))
    if L > _SQA_MAXL:
        raise Error(String("sqa_device_par: L=") + String(L)
                    + " exceeds _SQA_MAXL=" + String(_SQA_MAXL))
    var o_buf = ctx.enqueue_create_buffer[DType.uint8](H * dh * 2)
    var scale = Float32(1.0) / Float32(Float64(dh) ** 0.5)
    var qp = q.buf.unsafe_ptr().bitcast[BFloat16]()
    var kp = kc.buf.unsafe_ptr().bitcast[BFloat16]()
    var vp = vc.buf.unsafe_ptr().bitcast[BFloat16]()
    var op = o_buf.unsafe_ptr().bitcast[BFloat16]()
    ctx.enqueue_function[_k_sqa_par[128, _SQA_MAXL], _k_sqa_par[128, _SQA_MAXL]](
        qp, kp, vp, op, H, H_kv, L, scale,
        grid_dim=H, block_dim=_SQA_TPB,
    )
    return Tensor(o_buf^, [1, 1, H * dh], STDtype.BF16)


def _k_sqa(
    q: FPtr,      # [H, dh]
    kc: FPtr,     # [H_kv, L, dh]
    vc: FPtr,     # [H_kv, L, dh]
    o: FPtr,      # [H, dh]  (output)
    H: Int,
    H_kv: Int,
    L: Int,
    dh: Int,
    scale: Float32,
):
    var hq = Int(global_idx.x)
    if hq >= H:
        return
    var n_rep = H // H_kv
    var kv = hq // n_rep
    var qb = hq * dh
    var kvb = kv * L * dh

    # pass 1: max score (numerical stability)
    var m = Float32(-1.0e30)
    for l in range(L):
        var s = Float32(0.0)
        var kb = kvb + l * dh
        for d in range(dh):
            s += q[qb + d] * kc[kb + d]
        s *= scale
        if s > m:
            m = s

    # pass 2: denom = sum exp(s - m)
    var denom = Float32(0.0)
    for l in range(L):
        var s = Float32(0.0)
        var kb = kvb + l * dh
        for d in range(dh):
            s += q[qb + d] * kc[kb + d]
        s *= scale
        denom += exp(s - m)

    # pass 3: weighted sum of V into o[hq]
    for d in range(dh):
        o[qb + d] = Float32(0.0)
    for l in range(L):
        var s = Float32(0.0)
        var kb = kvb + l * dh
        for d in range(dh):
            s += q[qb + d] * kc[kb + d]
        s *= scale
        var w = exp(s - m) / denom
        var vb = kvb + l * dh
        for d in range(dh):
            o[qb + d] += w * vc[vb + d]


def sqa_gpu(
    ctx: DeviceContext,
    q_host: List[Float32],
    k_host: List[Float32],
    v_host: List[Float32],
    H: Int,
    H_kv: Int,
    L: Int,
    dh: Int,
) raises -> List[Float32]:
    """Run single-query attention on GPU; returns o [H*dh] (F32). scale=1/sqrt(dh)."""
    var nq = H * dh
    var nkv = H_kv * L * dh
    var hq = ctx.enqueue_create_host_buffer[DType.float32](nq)
    for i in range(nq):
        hq[i] = q_host[i]
    var hk = ctx.enqueue_create_host_buffer[DType.float32](nkv)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](nkv)
    for i in range(nkv):
        hk[i] = k_host[i]
        hv[i] = v_host[i]
    var dq = ctx.enqueue_create_buffer[DType.float32](nq)
    var dk = ctx.enqueue_create_buffer[DType.float32](nkv)
    var dv = ctx.enqueue_create_buffer[DType.float32](nkv)
    var do = ctx.enqueue_create_buffer[DType.float32](nq)
    ctx.enqueue_copy(dq, hq)
    ctx.enqueue_copy(dk, hk)
    ctx.enqueue_copy(dv, hv)

    var scale = Float32(1.0) / Float32(Float64(dh) ** 0.5)
    ctx.enqueue_function[_k_sqa, _k_sqa](
        dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(), do.unsafe_ptr(),
        H, H_kv, L, dh, scale,
        grid_dim=(H + 31) // 32, block_dim=32,
    )
    var ho = ctx.enqueue_create_host_buffer[DType.float32](nq)
    ctx.enqueue_copy(ho, do)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(nq):
        out.append(ho[i])
    return out^
