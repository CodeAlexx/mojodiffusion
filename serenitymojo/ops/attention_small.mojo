# ops/attention_small.mojo — FUSED small-S SDPA (no mask), one block per (b,h).
#
# WHY: `sdpa_nomask` routes through `_sdpa_math_storage`, which loops
# `for bh in range(B*H)` issuing per-(b,h) cuBLAS GEMMs. For krea2's txtfusion
# LAYERWISE blocks the batch is B=LT(=384 bucket) × H=20 heads over a tiny
# S=NLAYERS=12 sequence → 7680 QKᵀ + 7680 P@V LAUNCH-BOUND tiny GEMMs per call
# (nsys 2026-07-07: two ~41ms runs of ~2.3µs kernels per train step = the
# conditioning hotspot). This kernel fuses the WHOLE attention for one (b,h) —
# scores, softmax, P@V — into ONE launch over a B*H grid.
#
# Scope: S <= 16, Dh <= 128, BF16 storage, no mask, no rope, heads==kvheads
# (the krea2 txtfusion layerwise shape). All math F32 (matches the math-path's
# F32 QKᵀ accumulate + F32 softmax; P@V accumulates F32 here where the loop
# path used the storage-dtype GEMM — same-or-better precision). Parity gate:
# ops/tests/sdpa_small_parity.mojo vs sdpa_nomask (cos >= 0.9999).
#
# Layout: q/k/v/o are [B, S, H, Dh] (the sdpa_nomask contract).
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.math import exp
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)


def _sdpa_small_kernel[S: Int, Dh: Int](
    q: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [B*S*H*Dh]
    k: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    H_w: Int32,
    scale: Float32,
):
    var H = Int(H_w)
    # One block per (b,h); Dh threads. Shared: q,k,v tiles [S,Dh] F32 + scores.
    var bh = Int(block_idx.x)
    var b = bh // H
    var h = bh % H
    var tid = Int(thread_idx.x)

    var sq = stack_allocation[
        S * Dh, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var sk = stack_allocation[
        S * Dh, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var sv = stack_allocation[
        S * Dh, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var sp = stack_allocation[
        S * S, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    # ── load q/k/v rows for this (b,h): thread tid covers column tid ─────────
    for i in range(S):
        var src = ((b * S + i) * H + h) * Dh + tid
        sq[i * Dh + tid] = Float32(rebind[Scalar[DType.bfloat16]](q[src]))
        sk[i * Dh + tid] = Float32(rebind[Scalar[DType.bfloat16]](k[src]))
        sv[i * Dh + tid] = Float32(rebind[Scalar[DType.bfloat16]](v[src]))
    barrier()

    # ── scores[i,j] = scale * dot(q_i, k_j): S*S dots spread over Dh threads ─
    var e = tid
    while e < S * S:
        var i = e // S
        var j = e % S
        var acc: Float32 = 0.0
        for d in range(Dh):
            acc += Float32(sq[i * Dh + d]) * Float32(sk[j * Dh + d])
        sp[e] = acc * scale
        e += Dh
    barrier()

    # ── softmax per row (threads 0..S-1 each own a row; S <= 16 << Dh) ───────
    if tid < S:
        var base = tid * S
        var m: Float32 = Float32(sp[base])
        for j in range(1, S):
            var val = Float32(sp[base + j])
            if val > m:
                m = val
        var ssum: Float32 = 0.0
        for j in range(S):
            var ex = exp(Float32(sp[base + j]) - m)
            sp[base + j] = ex
            ssum += ex
        var inv = 1.0 / ssum
        for j in range(S):
            sp[base + j] = Float32(sp[base + j]) * inv
    barrier()

    # ── out[i,d] = sum_j p[i,j] * v[j,d]: thread tid owns column d=tid ───────
    for i in range(S):
        var acc: Float32 = 0.0
        for j in range(S):
            acc += Float32(sp[i * S + j]) * Float32(sv[j * Dh + tid])
        var dst = ((b * S + i) * H + h) * Dh + tid
        o[dst] = rebind[o.element_type](acc.cast[DType.bfloat16]())


def sdpa_nomask_small[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor, k: Tensor, v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Fused no-mask SDPA for SMALL S (<=16), BF16 [B,S,H,Dh] → BF16 [B,S,H,Dh].
    ONE kernel launch over a B*H grid instead of _sdpa_math_storage's 2·B·H
    per-(b,h) GEMM launches. F32 math throughout. Gate: sdpa_small_parity."""
    comptime if S > 16:
        raise Error("sdpa_nomask_small: S must be <= 16")
    comptime if Dh > 128:
        raise Error("sdpa_nomask_small: Dh must be <= 128")
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16 or v.dtype() != STDtype.BF16:
        raise Error("sdpa_nomask_small: q/k/v must be BF16")
    var qshape = q.shape()
    if len(qshape) != 4 or qshape[0] != B or qshape[1] != S or qshape[2] != H or qshape[3] != Dh:
        raise Error("sdpa_nomask_small: q shape != [B,S,H,Dh]")

    var n = B * S * H * Dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(q.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
    var K = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(k.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
    var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(v.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
    ctx.enqueue_function[_sdpa_small_kernel[S, Dh]](
        Q, K, V, O, Int32(H), scale, grid_dim=B * H, block_dim=Dh,
    )
    var oshape: List[Int] = [B, S, H, Dh]
    return Tensor(out_buf^, oshape^, STDtype.BF16)
