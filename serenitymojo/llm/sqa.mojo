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

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.math import exp

comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]


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
