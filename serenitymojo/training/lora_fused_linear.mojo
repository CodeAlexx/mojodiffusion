# training/lora_fused_linear.mojo — fused LoRA-linear forward/backward kernels.
#
# SHARED CORE: model-agnostic. Any trainer whose LoRA path is
#     delta = scale · ((x @ Aᵀ) @ Bᵀ)        x [M,in] F32, A [r,in] B [out,r] BF16
# can route here (Klein wires in via models/klein/lora_block.mojo; Qwen/SDXL/
# flux LoRA verticals should reuse these instead of re-deriving the chain).
#
# WHY (MEASURED 2026-06-11, /tmp/kbwd.sqlite): the unfused chain per adapter is
# 2 GEMM + 2 cast launches forward and 1 GEMM + 1 scalar-mul + 4 GEMM + 2 cast
# launches backward — ~60 ms small bf16 GEMMs (s1688gemm, N=rank=16 tiles are
# grossly inefficient at 113-173 µs each) + ~16 ms F32 dw sgemms + 67 ms
# f32→bf16 cast kernels per Klein-9B step. Fused: 1 launch forward, 2 launches
# backward, rank-16 intermediate never touches HBM, no cast kernels.
#
# NUMERICS CONTRACT (must hold or the 8-nines block gates die):
#   Every PRODUCT is bit-identical to the unfused chain in ops/linear.mojo +
#   ops/linalg_backward.mojo with F32 activations and BF16 adapters:
#     t    = Σ bf16(x)·bf16(A)        (bf16 RNE inputs, F32 FMA accumulate —
#                                      same as cast_tensor + vendor bf16 GEMM)
#     delta= scale · Σ bf16(t)·bf16(B)            (scale applied AFTER the sum,
#                                                  in F32, as mul_scalar did)
#     d_dy = scale·dc (F32)  ;  d_t = Σ bf16(d_dy)·bf16(B)
#     d_b  = Σ d_dy·t        (UNROUNDED F32 products — F32 sgemm equivalent)
#     d_x  = Σ bf16(d_t)·bf16(A)
#     d_a  = Σ d_t·x         (UNROUNDED F32 products)
#   Only the ACCUMULATION ORDER differs from cuBLAS tiling — the same accepted
#   class as the per-build pointer-alignment algo-selection shifts (scoreboard
#   2026-06-11): gate at 8-nines cosine vs the torch oracles + 4-dp loss
#   anchors, NOT bit-equality vs the old chain.
#   bf16 rounding is the hardware RNE `.cast[DType.bfloat16]()` — identical to
#   ops/cast.mojo `_f32_to_bf16`.
#
# GATE: training/lora_fused_linear_parity.mojo (fused vs unfused chain on GPU,
# Klein product shapes) + models/klein/parity block LoRA gates + step anchors.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu import thread_idx, block_idx, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.memory import ArcPointer, stack_allocation

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


# PRODUCTION SWITCH — MEASURED 2026-06-11 (3-step Klein-9B, 3090Ti): the v2
# kernels are parity-clean (gate ALL PASS) but SLOWER than the legacy cuBLAS
# chains they replace: bwd stage 2.19-2.26 s fused vs 2.01 s legacy (+180-250
# ms/step). Per-slot ~0.2 TF/s vs cuBLAS small-N chain ~0.9 TF/s — the chain
# was never launch-bound (the "GEMM storm" was SDPA, see scoreboard 06-11
# session-2 entry), so fusion must win on raw throughput and doesn't yet.
# Keep False until a kernel iteration beats legacy on a measured run. The
# kernels remain valuable as the portable path for AMD bring-up (G-X1 — no
# cuBLAS there) and the parity gate keeps them honest.
comptime LORA_FUSED_ENABLED = False

comptime _TPB = 256       # threads per block, all kernels
comptime _KC = 256        # in-dim chunk staged in smem (phase A)
comptime _OC = 256        # out-dim chunk staged in smem (bwd phase A2)
comptime _PAD = 2         # smem row padding (elements) to break bank conflicts


def lora_fused_eligible(x: Tensor, a: Tensor, b: Tensor, rank: Int) -> Bool:
    """True when the fused kernels can handle this slot: F32 activations,
    BF16 adapters, rank 16 (the compiled specialization). Does NOT consult
    the production switch — used by the parity gate and the direct entries."""
    return (
        x.dtype() == STDtype.F32
        and a.dtype() == STDtype.BF16
        and b.dtype() == STDtype.BF16
        and rank == 16
    )


def lora_fused_supported(x: Tensor, a: Tensor, b: Tensor, rank: Int) -> Bool:
    """Production dispatch check: eligibility AND the LORA_FUSED_ENABLED
    switch (currently False — see header note: measured slower than the
    legacy cuBLAS chain)."""
    comptime if not LORA_FUSED_ENABLED:
        return False
    else:
        return lora_fused_eligible(x, a, b, rank)


# ── forward: delta[M,out] = scale · (bf16(x[M,in]) @ Aᵀ → bf16) @ Bᵀ ─────────
# Grid: ceildiv(M, TM) blocks of _TPB threads, TM = _TPB // R rows per block.
# Phase A: cooperative smem staging of x (bf16-rounded on load) and A chunks;
#          thread (tm, tr) accumulates t[tm, tr] in F32.
# Phase B: t tile (rounded to bf16, in smem) drives delta columns.
def _lora_fwd_kernel[R: Int](
    x: UnsafePointer[Float32, MutAnyOrigin],       # [M, in] row-major
    a: UnsafePointer[BFloat16, MutAnyOrigin],      # [R, in]
    b: UnsafePointer[BFloat16, MutAnyOrigin],      # [out, R]
    delta: UnsafePointer[Float32, MutAnyOrigin],   # [M, out]
    m_rows: Int,
    in_f: Int,
    out_f: Int,
    scale: Float32,
):
    comptime TM = _TPB // R
    comptime KCP = _KC + _PAD
    var tid = Int(thread_idx.x)
    var m0 = Int(block_idx.x) * TM

    var x_s = stack_allocation[
        TM * KCP, Scalar[DType.bfloat16], address_space = AddressSpace.SHARED
    ]()
    var a_s = stack_allocation[
        R * KCP, Scalar[DType.bfloat16], address_space = AddressSpace.SHARED
    ]()
    var tb_s = stack_allocation[
        TM * R, Scalar[DType.bfloat16], address_space = AddressSpace.SHARED
    ]()

    var tm = tid // R
    var tr = tid % R

    # Phase A: t[tm, tr] = Σ_k bf16(x[m0+tm, k]) · bf16(A[tr, k])
    # 4 independent accumulators for ILP (v1's single serial FMA chain was
    # latency-bound at ~1 CTA/SM — MEASURED 1.6 ms/launch).
    var ac0 = Float32(0.0)
    var ac1 = Float32(0.0)
    var ac2 = Float32(0.0)
    var ac3 = Float32(0.0)
    var k0 = 0
    while k0 < in_f:
        var kc = in_f - k0
        if kc > _KC:
            kc = _KC
        var li = tid
        while li < TM * _KC:
            var row = li // _KC
            var col = li % _KC
            var gm = m0 + row
            if col < kc and gm < m_rows:
                x_s[row * KCP + col] = x[gm * in_f + k0 + col].cast[
                    DType.bfloat16
                ]()
            else:
                x_s[row * KCP + col] = BFloat16(0.0)
            li += _TPB
        li = tid
        while li < R * _KC:
            var row = li // _KC
            var col = li % _KC
            if col < kc:
                a_s[row * KCP + col] = a[row * in_f + k0 + col]
            else:
                a_s[row * KCP + col] = BFloat16(0.0)
            li += _TPB
        barrier()
        var k = 0
        while k + 4 <= kc:
            ac0 += (
                x_s[tm * KCP + k].cast[DType.float32]()
                * a_s[tr * KCP + k].cast[DType.float32]()
            )
            ac1 += (
                x_s[tm * KCP + k + 1].cast[DType.float32]()
                * a_s[tr * KCP + k + 1].cast[DType.float32]()
            )
            ac2 += (
                x_s[tm * KCP + k + 2].cast[DType.float32]()
                * a_s[tr * KCP + k + 2].cast[DType.float32]()
            )
            ac3 += (
                x_s[tm * KCP + k + 3].cast[DType.float32]()
                * a_s[tr * KCP + k + 3].cast[DType.float32]()
            )
            k += 4
        while k < kc:
            ac0 += (
                x_s[tm * KCP + k].cast[DType.float32]()
                * a_s[tr * KCP + k].cast[DType.float32]()
            )
            k += 1
        barrier()
        k0 += _KC
    tb_s[tm * R + tr] = ((ac0 + ac1) + (ac2 + ac3)).cast[DType.bfloat16]()
    barrier()

    # Phase B: delta[m0+tmi, o] = scale · Σ_r bf16(t[tmi, r]) · bf16(B[o, r])
    var bf = stack_allocation[R, Scalar[DType.float32]]()
    var o = tid
    while o < out_f:
        for r in range(R):
            bf[r] = b[o * R + r].cast[DType.float32]()
        for tmi in range(TM):
            var gm = m0 + tmi
            if gm < m_rows:
                var s = Float32(0.0)
                for r in range(R):
                    s += tb_s[tmi * R + r].cast[DType.float32]() * bf[r]
                delta[gm * out_f + o] = scale * s
        o += _TPB


# ── backward launch 1 (row-space): t, d_t, d_x ───────────────────────────────
# t[M,R]   = bf16(x) @ Aᵀ            (F32, unrounded — feeds d_b in launch 2)
# d_t[M,R] = bf16(scale·dc) @ B      (F32, unrounded — feeds d_a in launch 2)
# d_x[M,in]= bf16(d_t) @ A
def _lora_bwd_rows_kernel[R: Int](
    x: UnsafePointer[Float32, MutAnyOrigin],       # [M, in]
    dc: UnsafePointer[Float32, MutAnyOrigin],      # [M, out] (UNSCALED d_contrib)
    a: UnsafePointer[BFloat16, MutAnyOrigin],      # [R, in]
    b: UnsafePointer[BFloat16, MutAnyOrigin],      # [out, R]
    t_out: UnsafePointer[Float32, MutAnyOrigin],   # [M, R]
    dt_out: UnsafePointer[Float32, MutAnyOrigin],  # [M, R]
    dx: UnsafePointer[Float32, MutAnyOrigin],      # [M, in]
    m_rows: Int,
    in_f: Int,
    out_f: Int,
    scale: Float32,
):
    comptime TM = _TPB // R
    comptime KCP = _KC + _PAD
    comptime OCP = _OC + _PAD
    var tid = Int(thread_idx.x)
    var m0 = Int(block_idx.x) * TM

    var x_s = stack_allocation[
        TM * KCP, Scalar[DType.bfloat16], address_space = AddressSpace.SHARED
    ]()
    var a_s = stack_allocation[
        R * KCP, Scalar[DType.bfloat16], address_space = AddressSpace.SHARED
    ]()
    var dtb_s = stack_allocation[
        TM * R, Scalar[DType.bfloat16], address_space = AddressSpace.SHARED
    ]()

    var tm = tid // R
    var tr = tid % R
    var gm_own = m0 + tm

    # Phase A1: t[tm, tr] = Σ_k bf16(x[m, k]) · bf16(A[tr, k])
    # 4 independent accumulators for ILP (see _lora_fwd_kernel note).
    var ac0 = Float32(0.0)
    var ac1 = Float32(0.0)
    var ac2 = Float32(0.0)
    var ac3 = Float32(0.0)
    var k0 = 0
    while k0 < in_f:
        var kc = in_f - k0
        if kc > _KC:
            kc = _KC
        var li = tid
        while li < TM * _KC:
            var row = li // _KC
            var col = li % _KC
            var gm = m0 + row
            if col < kc and gm < m_rows:
                x_s[row * KCP + col] = x[gm * in_f + k0 + col].cast[
                    DType.bfloat16
                ]()
            else:
                x_s[row * KCP + col] = BFloat16(0.0)
            li += _TPB
        li = tid
        while li < R * _KC:
            var row = li // _KC
            var col = li % _KC
            if col < kc:
                a_s[row * KCP + col] = a[row * in_f + k0 + col]
            else:
                a_s[row * KCP + col] = BFloat16(0.0)
            li += _TPB
        barrier()
        var k = 0
        while k + 4 <= kc:
            ac0 += (
                x_s[tm * KCP + k].cast[DType.float32]()
                * a_s[tr * KCP + k].cast[DType.float32]()
            )
            ac1 += (
                x_s[tm * KCP + k + 1].cast[DType.float32]()
                * a_s[tr * KCP + k + 1].cast[DType.float32]()
            )
            ac2 += (
                x_s[tm * KCP + k + 2].cast[DType.float32]()
                * a_s[tr * KCP + k + 2].cast[DType.float32]()
            )
            ac3 += (
                x_s[tm * KCP + k + 3].cast[DType.float32]()
                * a_s[tr * KCP + k + 3].cast[DType.float32]()
            )
            k += 4
        while k < kc:
            ac0 += (
                x_s[tm * KCP + k].cast[DType.float32]()
                * a_s[tr * KCP + k].cast[DType.float32]()
            )
            k += 1
        barrier()
        k0 += _KC
    if gm_own < m_rows:
        t_out[gm_own * R + tr] = (ac0 + ac1) + (ac2 + ac3)

    # Phase A2: d_t[tm, tr] = Σ_o bf16(scale·dc[m, o]) · bf16(B[o, tr])
    # Reuse x_s as the dc chunk and a_s as the B chunk (both are dead now;
    # smem budgets: dc chunk TM*OCP bf16, B chunk _OC*R bf16 ≤ a_s size R*KCP
    # because _OC == _KC).
    var dc0 = Float32(0.0)
    var dc1 = Float32(0.0)
    var dc2 = Float32(0.0)
    var dc3 = Float32(0.0)
    var o0 = 0
    while o0 < out_f:
        var oc = out_f - o0
        if oc > _OC:
            oc = _OC
        barrier()
        var li = tid
        while li < TM * _OC:
            var row = li // _OC
            var col = li % _OC
            var gm = m0 + row
            if col < oc and gm < m_rows:
                x_s[row * OCP + col] = (
                    scale * dc[gm * out_f + o0 + col]
                ).cast[DType.bfloat16]()
            else:
                x_s[row * OCP + col] = BFloat16(0.0)
            li += _TPB
        li = tid
        while li < _OC * R:
            if li < oc * R:
                a_s[li] = b[(o0 + li // R) * R + li % R]
            else:
                a_s[li] = BFloat16(0.0)
            li += _TPB
        barrier()
        var oi = 0
        while oi + 4 <= oc:
            dc0 += (
                x_s[tm * OCP + oi].cast[DType.float32]()
                * a_s[oi * R + tr].cast[DType.float32]()
            )
            dc1 += (
                x_s[tm * OCP + oi + 1].cast[DType.float32]()
                * a_s[(oi + 1) * R + tr].cast[DType.float32]()
            )
            dc2 += (
                x_s[tm * OCP + oi + 2].cast[DType.float32]()
                * a_s[(oi + 2) * R + tr].cast[DType.float32]()
            )
            dc3 += (
                x_s[tm * OCP + oi + 3].cast[DType.float32]()
                * a_s[(oi + 3) * R + tr].cast[DType.float32]()
            )
            oi += 4
        while oi < oc:
            dc0 += (
                x_s[tm * OCP + oi].cast[DType.float32]()
                * a_s[oi * R + tr].cast[DType.float32]()
            )
            oi += 1
        o0 += _OC
    var dacc = (dc0 + dc1) + (dc2 + dc3)
    if gm_own < m_rows:
        dt_out[gm_own * R + tr] = dacc
    barrier()
    dtb_s[tm * R + tr] = dacc.cast[DType.bfloat16]()
    barrier()

    # Phase B: d_x[m0+tmi, i] = Σ_r bf16(d_t[tmi, r]) · bf16(A[r, i])
    var av = stack_allocation[R, Scalar[DType.float32]]()
    var i = tid
    while i < in_f:
        for r in range(R):
            av[r] = a[r * in_f + i].cast[DType.float32]()
        for tmi in range(TM):
            var gm = m0 + tmi
            if gm < m_rows:
                var s = Float32(0.0)
                for r in range(R):
                    s += dtb_s[tmi * R + r].cast[DType.float32]() * av[r]
                dx[gm * in_f + i] = s
        i += _TPB


# ── backward launch 2 (weight-space): d_b and d_a in one launch ──────────────
# ONE THREAD PER OUTPUT ELEMENT (out·R d_b elements, then R·in d_a elements) —
# v1 used one thread per column with an R-vector accumulator, which produced
# only ~24 CTAs at Klein shapes (MEASURED 855 µs/launch, most SMs idle). This
# layout gives 100k+ threads, coalesced t/x reads, broadcast dc/dt reads.
# Per-output M-loop is sequential ascending — deterministic accumulation.
def _lora_bwd_w_kernel[R: Int](
    x: UnsafePointer[Float32, MutAnyOrigin],       # [M, in]
    dc: UnsafePointer[Float32, MutAnyOrigin],      # [M, out]
    t_in: UnsafePointer[Float32, MutAnyOrigin],    # [M, R]
    dt_in: UnsafePointer[Float32, MutAnyOrigin],   # [M, R]
    d_a: UnsafePointer[Float32, MutAnyOrigin],     # [R, in]
    d_b: UnsafePointer[Float32, MutAnyOrigin],     # [out, R]
    m_rows: Int,
    in_f: Int,
    out_f: Int,
    scale: Float32,
):
    var gid = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    var nb = out_f * R
    if gid < nb:
        # d_b[o, r] = Σ_m (scale·dc[m, o]) · t[m, r]
        var o = gid // R
        var r = gid % R
        var a0 = Float32(0.0)
        var a1 = Float32(0.0)
        var m = 0
        while m + 2 <= m_rows:
            a0 += (scale * dc[m * out_f + o]) * t_in[m * R + r]
            a1 += (scale * dc[(m + 1) * out_f + o]) * t_in[(m + 1) * R + r]
            m += 2
        if m < m_rows:
            a0 += (scale * dc[m * out_f + o]) * t_in[m * R + r]
        d_b[gid] = a0 + a1
    else:
        var gid2 = gid - nb
        if gid2 < R * in_f:
            # d_a[r, i] = Σ_m d_t[m, r] · x[m, i]
            var r = gid2 // in_f
            var i = gid2 % in_f
            var a0 = Float32(0.0)
            var a1 = Float32(0.0)
            var m = 0
            while m + 2 <= m_rows:
                a0 += dt_in[m * R + r] * x[m * in_f + i]
                a1 += dt_in[(m + 1) * R + r] * x[(m + 1) * in_f + i]
                m += 2
            if m < m_rows:
                a0 += dt_in[m * R + r] * x[m * in_f + i]
            d_a[gid2] = a0 + a1


# ── host-side entries ─────────────────────────────────────────────────────────
# ArcPointer fields so callers can hand the tensors to TArc-based grad carriers
# without partial-move-out-of-struct issues (Tensor is move-only).
struct LoraFusedBwdOut(Copyable, Movable):
    var d_a: ArcPointer[Tensor]   # [rank, in] F32
    var d_b: ArcPointer[Tensor]   # [out, rank] F32
    var d_x: ArcPointer[Tensor]   # [M, in] F32

    def __init__(
        out self,
        var d_a: ArcPointer[Tensor],
        var d_b: ArcPointer[Tensor],
        var d_x: ArcPointer[Tensor],
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def _flat_rows(x: Tensor, in_f: Int) raises -> Int:
    var n = x.numel()
    if in_f <= 0 or n % in_f != 0:
        raise Error("lora_fused: x numel not divisible by in_f")
    return n // in_f


def lora_fused_fwd(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    rank: Int,
    in_f: Int,
    out_f: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """delta = scale·((x@Aᵀ)@Bᵀ), one kernel launch. F32 x, BF16 A/B, rank 16.
    Output: x's leading shape + [out_f], F32."""
    if not lora_fused_eligible(x, a, b, rank):
        raise Error("lora_fused_fwd: unsupported dtype/rank (caller must gate)")
    var m = _flat_rows(x, in_f)
    var xshape = x.shape()
    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        out_shape.append(xshape[i])
    out_shape.append(out_f)

    var d_buf = ctx.enqueue_create_buffer[DType.uint8](m * out_f * 4)
    comptime TM = _TPB // 16
    var grid = (m + TM - 1) // TM
    ctx.enqueue_function[_lora_fwd_kernel[16], _lora_fwd_kernel[16]](
        x.buf.unsafe_ptr().bitcast[Float32](),
        a.buf.unsafe_ptr().bitcast[BFloat16](),
        b.buf.unsafe_ptr().bitcast[BFloat16](),
        d_buf.unsafe_ptr().bitcast[Float32](),
        m, in_f, out_f, scale,
        grid_dim=grid, block_dim=_TPB,
    )
    return Tensor(d_buf^, out_shape^, STDtype.F32)


def lora_fused_bwd(
    d_contrib: Tensor,
    x: Tensor,
    a: Tensor,
    b: Tensor,
    rank: Int,
    in_f: Int,
    out_f: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises -> LoraFusedBwdOut:
    """Fused LoRA backward, 2 launches: (t, d_t, d_x) then (d_b, d_a).
    All outputs F32 device tensors; shapes match the unfused chain."""
    if not lora_fused_eligible(x, a, b, rank):
        raise Error("lora_fused_bwd: unsupported dtype/rank (caller must gate)")
    if d_contrib.dtype() != STDtype.F32:
        raise Error("lora_fused_bwd: d_contrib must be F32")
    var m = _flat_rows(x, in_f)
    if d_contrib.numel() != m * out_f:
        raise Error("lora_fused_bwd: d_contrib numel != M*out_f")

    var t_buf = ctx.enqueue_create_buffer[DType.uint8](m * 16 * 4)
    var dt_buf = ctx.enqueue_create_buffer[DType.uint8](m * 16 * 4)
    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](m * in_f * 4)
    var da_buf = ctx.enqueue_create_buffer[DType.uint8](16 * in_f * 4)
    var db_buf = ctx.enqueue_create_buffer[DType.uint8](out_f * 16 * 4)

    var xp = x.buf.unsafe_ptr().bitcast[Float32]()
    var dcp = d_contrib.buf.unsafe_ptr().bitcast[Float32]()
    var ap = a.buf.unsafe_ptr().bitcast[BFloat16]()
    var bp = b.buf.unsafe_ptr().bitcast[BFloat16]()
    var tp = t_buf.unsafe_ptr().bitcast[Float32]()
    var dtp = dt_buf.unsafe_ptr().bitcast[Float32]()

    comptime TM = _TPB // 16
    var grid1 = (m + TM - 1) // TM
    ctx.enqueue_function[_lora_bwd_rows_kernel[16], _lora_bwd_rows_kernel[16]](
        xp, dcp, ap, bp, tp, dtp,
        dx_buf.unsafe_ptr().bitcast[Float32](),
        m, in_f, out_f, scale,
        grid_dim=grid1, block_dim=_TPB,
    )

    var n_w_threads = out_f * 16 + 16 * in_f
    var grid2 = (n_w_threads + _TPB - 1) // _TPB
    ctx.enqueue_function[_lora_bwd_w_kernel[16], _lora_bwd_w_kernel[16]](
        xp, dcp, tp, dtp,
        da_buf.unsafe_ptr().bitcast[Float32](),
        db_buf.unsafe_ptr().bitcast[Float32](),
        m, in_f, out_f, scale,
        grid_dim=grid2, block_dim=_TPB,
    )

    var da_sh = List[Int]()
    da_sh.append(rank)
    da_sh.append(in_f)
    var db_sh = List[Int]()
    db_sh.append(out_f)
    db_sh.append(rank)
    var dx_sh = List[Int]()
    dx_sh.append(m)
    dx_sh.append(in_f)
    return LoraFusedBwdOut(
        ArcPointer[Tensor](Tensor(da_buf^, da_sh^, STDtype.F32)),
        ArcPointer[Tensor](Tensor(db_buf^, db_sh^, STDtype.F32)),
        ArcPointer[Tensor](Tensor(dx_buf^, dx_sh^, STDtype.F32)),
    )
