# serenitymojo/models/klein/lora_block.mojo
#
# LoRA-ON-PROJECTION helpers shared by the Klein double/single block LoRA
# variants (double_block.mojo `double_block_lora_*`, single_block.mojo
# `single_block_lora_*`). This is the SAME LoRA math as
# serenitymojo/training/train_step.mojo (_lora_fwd:164, _lora_bwd:194,
# LoraAdapter:120, LoraGrads:185) — we REUSE the imported LoraAdapter / LoraGrads
# structs and replicate the two tiny linear-path helpers locally, with ONE
# addition the trainer's _lora_bwd discards: the LoRA branch's contribution to
# the projection INPUT grad (d_x).
#
# WHY THE TRAINER MATH IS THE AUTHORITY
#   For a projection y = linear(x, W) (W [out,in]), the LoRA-adapted output is
#       y' = linear(x, W) + scale·((x @ Aᵀ) @ Bᵀ)
#   with A=[rank,in], B=[out,rank], scale = alpha/rank. This MATCHES exactly the
#   inference merge semantics in serenitymojo/lora.mojo::_apply_slot (FULL slot
#   adds scale·B@A into W → merged W' gives y' = x @ (W + scale·B@A)ᵀ ≡ the same).
#
# BACKWARD (given d_y' at the projection output, x the saved projection input):
#       d_dy   = scale · d_y'                          [M,out]
#       d_B    = d_dyᵀ @ t           (t = x @ Aᵀ)       [out,rank]   (linear_backward d_w)
#       d_t    = d_dy  @ B                              [M,rank]     (linear_backward d_x)
#       d_A    = d_tᵀ  @ x                              [rank,in]    (linear_backward d_w)
#       d_x_lo = d_t   @ A                              [M,in]       (linear_backward d_x)
#   The base path (frozen W, base d_W discarded for LoRA) ALSO yields d_x_base =
#   d_y' @ W from the block's existing linear_backward; the caller SUMS d_x_lo
#   into that. d_A / d_B are returned to the optimizer.
#
# Mojo 1.0.0b1: def not fn; Tensor move-only crosses API as host List[Float32];
# no-bias linear = linear(x, w, Optional[Tensor](None), ctx).

from std.gpu.host import DeviceContext, HostBuffer
from std.collections import List, Optional
from std.memory import ArcPointer
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.env import env_int
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear, linear_scratch, linear_rows_scratch
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx, linear_backward_dw,
)
from serenitymojo.ops.tensor_algebra import add, concat, mul_scalar, mul_scalar_bf16out, slice
from serenitymojo.scratch_ring import ScratchRingAllocator

# REUSE the core SerenityTrainer-style LoRA structs (BF16 storage + SR AdamW).
from serenitymojo.models.klein.lora_adapter import LoraAdapter, LoraGrads

# Fused single-launch LoRA kernels (shared training core). Dispatchers below
# route the hot device-resident paths there when dtypes/rank fit; the legacy
# *_unfused chains are kept as the parity-gate reference.
from serenitymojo.training.lora_fused_linear import (
    lora_fused_bwd, lora_fused_fwd, lora_fused_supported,
)


comptime TArc = ArcPointer[Tensor]


struct LoraAdapterDevice(Copyable, Movable):
    var a: TArc
    var b: TArc
    var rank: Int
    var in_f: Int
    var out_f: Int
    var scale: Float32

    def __init__(
        out self, var a: TArc, var b: TArc,
        rank: Int, in_f: Int, out_f: Int, scale: Float32,
    ):
        self.a = a^
        self.b = b^
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f
        self.scale = scale


def lora_adapter_to_device(lo: LoraAdapter, ctx: DeviceContext) raises -> LoraAdapterDevice:
    return LoraAdapterDevice(
        TArc(Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)),
        TArc(Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)),
        lo.rank, lo.in_f, lo.out_f, lo.scale,
    )


struct _HostGradPair(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]

    def __init__(out self, var d_a: List[Float32], var d_b: List[Float32]):
        self.d_a = d_a^
        self.d_b = d_b^


def _host_from_f32_buffer(
    host: HostBuffer[DType.uint8], n: Int
) -> List[Float32]:
    var out = List[Float32]()
    var fp = host.unsafe_ptr().bitcast[Float32]()
    for i in range(n):
        out.append(fp[i])
    return out^


def _tensor_to_host_f32(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    if t.dtype() == STDtype.F32:
        var host = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
        ctx.enqueue_copy(dst_buf=host, src_buf=t.buf)
        ctx.synchronize()
        return _host_from_f32_buffer(host, t.numel())
    # Host AdamW stores master params and moments as F32; device grads may be BF16.
    var t32 = cast_tensor(t, STDtype.F32, ctx)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](t32.nbytes())
    ctx.enqueue_copy(dst_buf=host, src_buf=t32.buf)
    ctx.synchronize()
    return _host_from_f32_buffer(host, t32.numel())


def _to_host_pair_f32(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> _HostGradPair:
    var ah = _tensor_to_host_f32(a, ctx)
    var bh = _tensor_to_host_f32(b, ctx)
    return _HostGradPair(ah^, bh^)


# ── bf16 tensor-core LoRA backward GEMMs (KLEIN_LORA_BF16_GEMM, 2026-07-11) ──
# APPROVED numerics change (Alex, 2026-07-11, reference trainer-autocast-matching class): the
# two weight-grad GEMMs of the resident LoRA backward (d_B = d_dyᵀ@t and
# d_A = d_tᵀ@x) ran F32×F32 on CUDA cores (cutlass simt sgemm + cublasLt
# split-K — the skinny rank-16 shapes force heavy split-K). This routes them to
# bf16 tensor-core INPUTS with F32 accumulate/output (cuBLAS COMPUTE_32F), the
# same dtype recipe SerenityTrainer's autocast uses for these products.
#
# WHAT ROUNDS (and what does not):
#   - t/d_t/d_x GEMMs already ran bf16-input tensor-core (linear/_backward_dx
#     mixed F32-activation × BF16-weight arms cast the F32 operand internally).
#     Here the SAME RNE casts are hoisted so each rounded operand is produced
#     ONCE and reused (x_bf: t-GEMM + dA; d_dy_bf: d_t + dB; d_t_bf: d_x + dA)
#     → t, d_t and d_x stay BIT-IDENTICAL to the =0 chain (same operands, same
#     GEMM calls). The only NEW cast kernel per adapter is t_bf ([M,16], tiny).
#   - NEW rounding exists ONLY at the dA/dB GEMM inputs (t→bf16 for dB; the
#     bf16 views of d_dy/d_t/x as dW operands). dA/dB CARRIERS remain F32 —
#     the fused AdamW + SR byte-flow downstream is unchanged.
# Gate: models/klein/parity/klein_lora_bf16_gemm_parity.mojo (d_x bit-identical,
# dA/dB cos vs the F32 chain) + trainer 10-step and 100-step env-0/1 A/Bs.
comptime _LORA_DYN2 = Layout.row_major(-1, -1)


def klein_lora_bf16_gemm_enabled() -> Bool:
    """Runtime switch KLEIN_LORA_BF16_GEMM (default OFF): bf16 tensor-core
    inputs for the resident LoRA backward dA/dB GEMMs (F32 outputs). Set =1
    for the tensor-core arm; =0/unset keeps the legacy F32 CUDA-core chain
    (byte-identical to pre-change).

    DEFAULT LEFT 0 (2026-07-11 measurement): the change is parity-clean
    (d_x bit-identical; dA/dB cos ~0.999997) but the measured win is only
    ~8 ms/step — the LoRA dW F32 pool is 288 GEMMs/~16 ms/step, NOT the
    ~166 ms 6,912-launch simt pool, which nsys attributes to the NON-FLASH
    per-head SDPA GEMMs (S=1536, 32 heads; separate lever, own signoff)."""
    return env_int("KLEIN_LORA_BF16_GEMM", 0) != 0


def _lora_bf16_gemm_nt_f32(
    x_bf: Tensor, w_bf: Tensor, m: Int, k: Int, n_rows: Int, ctx: DeviceContext
) raises -> Tensor:
    """C[m, n_rows] = x_bf[m, k] @ w_bf[n_rows, k]ᵀ. BF16 inputs, F32 output.
    Same GEMM call as ops/linear.mojo's mixed F32×BF16 arm (t-GEMM)."""
    if x_bf.dtype() != STDtype.BF16 or w_bf.dtype() != STDtype.BF16:
        raise Error("_lora_bf16_gemm_nt_f32: operands must be BF16")
    if x_bf.numel() != m * k or w_bf.numel() != n_rows * k:
        raise Error("_lora_bf16_gemm_nt_f32: shape mismatch")
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](m * n_rows * 4)
    var a_rl = RuntimeLayout[_LORA_DYN2].row_major(IndexList[2](m, k))
    var b_rl = RuntimeLayout[_LORA_DYN2].row_major(IndexList[2](n_rows, k))
    var c_rl = RuntimeLayout[_LORA_DYN2].row_major(IndexList[2](m, n_rows))
    var A = LayoutTensor[DType.bfloat16, _LORA_DYN2, MutAnyOrigin](
        x_bf.buf.unsafe_ptr().bitcast[BFloat16](), a_rl
    )
    var B = LayoutTensor[DType.bfloat16, _LORA_DYN2, MutAnyOrigin](
        w_bf.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
    )
    var C = LayoutTensor[DType.float32, _LORA_DYN2, MutAnyOrigin](
        c_buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    matmul(ctx, C, A, B, transpose_b=True, c_row_major=True)
    return Tensor(c_buf^, [m, n_rows], STDtype.F32)


def _lora_bf16_gemm_nn_f32(
    gy_bf: Tensor, w_bf: Tensor, m: Int, out_f: Int, in_f: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """d_x[m, in_f] = gy_bf[m, out_f] @ w_bf[out_f, in_f]. BF16 inputs, F32
    output. Same GEMM call as linear_backward_dx's BF16-weight arm."""
    if gy_bf.dtype() != STDtype.BF16 or w_bf.dtype() != STDtype.BF16:
        raise Error("_lora_bf16_gemm_nn_f32: operands must be BF16")
    if gy_bf.numel() != m * out_f or w_bf.numel() != out_f * in_f:
        raise Error("_lora_bf16_gemm_nn_f32: shape mismatch")
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](m * in_f * 4)
    var mo_rl = RuntimeLayout[_LORA_DYN2].row_major(IndexList[2](m, out_f))
    var oi_rl = RuntimeLayout[_LORA_DYN2].row_major(IndexList[2](out_f, in_f))
    var mi_rl = RuntimeLayout[_LORA_DYN2].row_major(IndexList[2](m, in_f))
    var GY = LayoutTensor[DType.bfloat16, _LORA_DYN2, MutAnyOrigin](
        gy_bf.buf.unsafe_ptr().bitcast[BFloat16](), mo_rl
    )
    var W = LayoutTensor[DType.bfloat16, _LORA_DYN2, MutAnyOrigin](
        w_bf.buf.unsafe_ptr().bitcast[BFloat16](), oi_rl
    )
    var DX = LayoutTensor[DType.float32, _LORA_DYN2, MutAnyOrigin](
        c_buf.unsafe_ptr().bitcast[Float32](), mi_rl
    )
    matmul(ctx, DX, GY, W, c_row_major=True)
    return Tensor(c_buf^, [m, in_f], STDtype.F32)


def _lora_bf16_gemm_tn_f32(
    gy_bf: Tensor, x2_bf: Tensor, m: Int, out_f: Int, in_f: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """d_w[out_f, in_f] = gy_bf[m, out_f]ᵀ @ x2_bf[m, in_f]. BF16 tensor-core
    inputs, F32 accumulate/output — THE changed dA/dB GEMM (=0 ran F32 simt)."""
    if gy_bf.dtype() != STDtype.BF16 or x2_bf.dtype() != STDtype.BF16:
        raise Error("_lora_bf16_gemm_tn_f32: operands must be BF16")
    if gy_bf.numel() != m * out_f or x2_bf.numel() != m * in_f:
        raise Error("_lora_bf16_gemm_tn_f32: shape mismatch")
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](out_f * in_f * 4)
    var mo_rl = RuntimeLayout[_LORA_DYN2].row_major(IndexList[2](m, out_f))
    var mi_rl = RuntimeLayout[_LORA_DYN2].row_major(IndexList[2](m, in_f))
    var oi_rl = RuntimeLayout[_LORA_DYN2].row_major(IndexList[2](out_f, in_f))
    var GY = LayoutTensor[DType.bfloat16, _LORA_DYN2, MutAnyOrigin](
        gy_bf.buf.unsafe_ptr().bitcast[BFloat16](), mo_rl
    )
    var X2 = LayoutTensor[DType.bfloat16, _LORA_DYN2, MutAnyOrigin](
        x2_bf.buf.unsafe_ptr().bitcast[BFloat16](), mi_rl
    )
    var DW = LayoutTensor[DType.float32, _LORA_DYN2, MutAnyOrigin](
        c_buf.unsafe_ptr().bitcast[Float32](), oi_rl
    )
    matmul(ctx, DW, GY, X2, transpose_a=True, c_row_major=True)
    return Tensor(c_buf^, [out_f, in_f], STDtype.F32)


# Adapter forward contribution on x [M,in] → [M,out] (host list in/out).
# Byte-identical to train_step._lora_fwd.
def klein_lora_fwd(
    x_h: List[Float32], lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var nb1 = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb1^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        nb2^, ctx,
    ).to_host(ctx)                                   # [M,out]
    var out = List[Float32]()
    for i in range(len(dy)):
        out.append(lo.scale * dy[i])
    return out^


# Device-resident sibling of klein_lora_fwd. A/B are borrowed from a resident
# adapter carrier, so hot forward/recompute/backward paths do not re-upload them.
# Hot path: ONE fused kernel (training/lora_fused_linear.mojo) when x is F32
# and A/B are BF16 rank-16 — same products as the unfused chain, accumulation
# order differs (accepted class; gated at 8-nines vs torch oracles).
def klein_lora_fwd_device_resident(
    x: Tensor, lo: LoraAdapterDevice, M: Int, ctx: DeviceContext
) raises -> Tensor:
    if lora_fused_supported(x, lo.a[], lo.b[], lo.rank):
        return lora_fused_fwd(
            x, lo.a[], lo.b[], lo.rank, lo.in_f, lo.out_f, lo.scale, ctx
        )
    return klein_lora_fwd_device_resident_unfused(x, lo, M, ctx)


# Legacy 2-GEMM + scalar-mul chain — parity-gate reference for the fused path.
def klein_lora_fwd_device_resident_unfused(
    x: Tensor, lo: LoraAdapterDevice, M: Int, ctx: DeviceContext
) raises -> Tensor:
    var nb1 = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb1^, ctx)
    var nb2 = Optional[Tensor](None)
    var dy = linear(t, lo.b[], nb2^, ctx)
    return mul_scalar(dy, lo.scale, ctx)


def klein_lora_fwd_rows_device_resident_scratch(
    x: Tensor,
    lo: LoraAdapterDevice,
    M: Int,
    row_start: Int,
    row_count: Int,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> Tensor:
    var nb1 = Optional[Tensor](None)
    var t = linear_scratch(x, lo.a[], nb1^, ctx, scratch)
    return linear_rows_scratch(
        t, lo.b[], row_start, row_count, ctx, scratch, alpha=lo.scale,
    )


struct KleinLoraQKVDeltas(Movable):
    var q: Tensor
    var k: Tensor
    var v: Tensor

    def __init__(out self, var q: Tensor, var k: Tensor, var v: Tensor):
        self.q = q^
        self.k = k^
        self.v = v^


def klein_lora_fwd_qkv_rows_device_resident_scratch(
    x: Tensor,
    lo: LoraAdapterDevice,
    row_dim: Int,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> KleinLoraQKVDeltas:
    var nb1 = Optional[Tensor](None)
    var t = linear_scratch(x, lo.a[], nb1^, ctx, scratch)
    var q = linear_rows_scratch(
        t, lo.b[], 0, row_dim, ctx, scratch, alpha=lo.scale,
    )
    var k = linear_rows_scratch(
        t, lo.b[], row_dim, row_dim, ctx, scratch, alpha=lo.scale,
    )
    var v = linear_rows_scratch(
        t, lo.b[], 2 * row_dim, row_dim, ctx, scratch, alpha=lo.scale,
    )
    return KleinLoraQKVDeltas(q^, k^, v^)


def klein_lora_fwd_device(
    x: Tensor, lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> Tensor:
    var lo_dev = lora_adapter_to_device(lo, ctx)
    return klein_lora_fwd_device_resident(x, lo_dev, M, ctx)


# LoRA backward that ALSO returns the LoRA branch's contribution to d_x.
# d_a/d_b match train_step._lora_bwd exactly; d_x_lo is the term that file drops.
struct KleinLoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]   # LoRA contribution to the projection INPUT grad

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def klein_lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> KleinLoraGrads:
    # t = x @ Aᵀ  (recompute; cheap)
    var nb_t = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb_t^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var d_dy = List[Float32]()
    for i in range(len(d_contrib_h)):
        d_dy.append(lo.scale * d_contrib_h[i])       # [M,out]
    # dy = t @ Bᵀ  → d_B (d_w) and d_t (d_x)
    var lbB = linear_backward(
        Tensor.from_host(d_dy^, [M, lo.out_f], STDtype.BF16, ctx),
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.to_host(ctx)                   # [M,rank]
    var d_b = lbB.d_w.to_host(ctx)                   # [out_f,rank]
    # t = x @ Aᵀ  → d_A (d_w) and d_x_lo (d_x)
    var lbA = linear_backward(
        Tensor.from_host(d_t^, [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_x_lo = lbA.d_x.to_host(ctx)                # [M,in_f]
    var d_a = lbA.d_w.to_host(ctx)                   # [rank,in_f]
    return KleinLoraGrads(d_a^, d_b^, d_x_lo^)


struct KleinLoraDeviceGrads(Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: Tensor

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: Tensor
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


struct KleinLoraDeviceGradTensors(Copyable, Movable):
    var d_a: TArc
    var d_b: TArc
    var d_x: TArc

    def __init__(out self, var d_a: TArc, var d_b: TArc, var d_x: TArc):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


# Device-resident sibling of klein_lora_bwd. d_A/d_B still leave as host lists
# for the existing optimizer, while the large d_x_lo tensor stays on device.
# Hot path: 2 fused launches (training/lora_fused_linear.mojo) replacing the
# 1 GEMM + scalar-mul + 4 GEMM + cast chain when dtypes/rank fit.
def klein_lora_bwd_device_resident(
    d_contrib: Tensor, x: Tensor, lo: LoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> KleinLoraDeviceGrads:
    if (
        lora_fused_supported(x, lo.a[], lo.b[], lo.rank)
        and d_contrib.dtype() == STDtype.F32
    ):
        var g = lora_fused_bwd(
            d_contrib, x, lo.a[], lo.b[], lo.rank, lo.in_f, lo.out_f,
            lo.scale, ctx,
        )
        var pair = _to_host_pair_f32(g.d_a[], g.d_b[], ctx)
        return KleinLoraDeviceGrads(
            pair.d_a.copy(), pair.d_b.copy(), g.d_x[].clone(ctx)
        )
    if (
        d_contrib.dtype() == STDtype.F32
        and x.dtype() == STDtype.F32
        and lo.a[].dtype() == STDtype.BF16
        and lo.b[].dtype() == STDtype.BF16
        and klein_lora_bf16_gemm_enabled()
    ):
        var gb = klein_lora_bwd_device_resident_tensors_bf16gemm(
            d_contrib, x, lo, M, ctx
        )
        var pair_b = _to_host_pair_f32(gb.d_a[], gb.d_b[], ctx)
        return KleinLoraDeviceGrads(
            pair_b.d_a.copy(), pair_b.d_b.copy(), gb.d_x[].clone(ctx)
        )
    return klein_lora_bwd_device_resident_unfused(d_contrib, x, lo, M, ctx)


# Legacy chain — parity-gate reference for the fused path.
def klein_lora_bwd_device_resident_unfused(
    d_contrib: Tensor, x: Tensor, lo: LoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> KleinLoraDeviceGrads:
    # t = x @ Aᵀ  (recompute; cheap)
    var nb_t = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb_t^, ctx)             # [M,rank]
    var d_dy = mul_scalar(d_contrib, lo.scale, ctx)   # [M,out]

    # dy = t @ Bᵀ.
    var d_t = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)
    var d_b_t = linear_backward_dw(d_dy, t, M, lo.rank, lo.out_f, ctx)

    # t = x @ Aᵀ.
    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)
    var d_a_t = linear_backward_dw(d_t, x, M, lo.in_f, lo.rank, ctx)

    var pair = _to_host_pair_f32(d_a_t, d_b_t, ctx)
    return KleinLoraDeviceGrads(pair.d_a.copy(), pair.d_b.copy(), d_x_lo^)


def klein_lora_bwd_device_resident_tensors(
    d_contrib: Tensor, x: Tensor, lo: LoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> KleinLoraDeviceGradTensors:
    """Device-resident LoRA backward.

    This variant keeps d_A/d_B on device so the full Klein stack can enqueue all
    block backward work first and perform one batched D2H fence at the end of the
    step. Hot path = bf16 tensor-core GEMMs (KLEIN_LORA_BF16_GEMM, default ON);
    unfused F32-dW chain kept below as the =0 arm / parity reference.
    """
    if (
        lora_fused_supported(x, lo.a[], lo.b[], lo.rank)
        and d_contrib.dtype() == STDtype.F32
    ):
        var g = lora_fused_bwd(
            d_contrib, x, lo.a[], lo.b[], lo.rank, lo.in_f, lo.out_f,
            lo.scale, ctx,
        )
        return KleinLoraDeviceGradTensors(
            g.d_a.copy(), g.d_b.copy(), g.d_x.copy()
        )
    if (
        d_contrib.dtype() == STDtype.F32
        and x.dtype() == STDtype.F32
        and lo.a[].dtype() == STDtype.BF16
        and lo.b[].dtype() == STDtype.BF16
        and klein_lora_bf16_gemm_enabled()
    ):
        return klein_lora_bwd_device_resident_tensors_bf16gemm(
            d_contrib, x, lo, M, ctx
        )
    return klein_lora_bwd_device_resident_tensors_unfused(
        d_contrib, x, lo, M, ctx
    )


def klein_lora_bwd_device_resident_tensors_bf16gemm(
    d_contrib: Tensor, x: Tensor, lo: LoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> KleinLoraDeviceGradTensors:
    """bf16 tensor-core LoRA backward (KLEIN_LORA_BF16_GEMM=1 arm).

    Same 5-GEMM chain as the unfused reference; the F32 operands are RNE-cast
    to bf16 ONCE each and reused, so t / d_t / d_x are BIT-IDENTICAL to the =0
    chain (identical operands and GEMM calls) and only dA/dB change numerics
    (bf16 tensor-core inputs instead of F32 CUDA-core; F32 accumulate + F32
    output either way — the grad carriers the fused AdamW consumes stay F32).
    """
    if x.numel() != M * lo.in_f:
        raise Error("klein_lora_bwd bf16gemm: x numel != M*in_f")
    if d_contrib.numel() != M * lo.out_f:
        raise Error("klein_lora_bwd bf16gemm: d_contrib numel != M*out_f")
    # t = x @ Aᵀ (recompute) — hoisted x cast; reused below for dA.
    var x_bf = cast_tensor(x, STDtype.BF16, ctx, False)
    var t = _lora_bf16_gemm_nt_f32(x_bf, lo.a[], M, lo.in_f, lo.rank, ctx)
    # d_dy_bf = bf16(scale·d_contrib) in ONE pass (fusion pass 2): the F32
    # mul_scalar value and the RNE bf16 rounding are unchanged — bit-identical
    # to mul_scalar → cast_tensor, minus a full-tensor F32 round trip. The F32
    # d_dy was only ever read by this cast.
    var d_dy_bf = mul_scalar_bf16out(d_contrib, lo.scale, ctx)
    # dy = t @ Bᵀ:  d_t = d_dy @ B  ;  d_B = d_dyᵀ @ t.
    var d_t = _lora_bf16_gemm_nn_f32(d_dy_bf, lo.b[], M, lo.out_f, lo.rank, ctx)
    var t_bf = cast_tensor(t, STDtype.BF16, ctx, False)   # the one NEW cast
    var d_b_t = _lora_bf16_gemm_tn_f32(d_dy_bf, t_bf, M, lo.out_f, lo.rank, ctx)
    # t = x @ Aᵀ:  d_x = d_t @ A  ;  d_A = d_tᵀ @ x.
    var d_t_bf = cast_tensor(d_t, STDtype.BF16, ctx, False)
    var d_x_lo = _lora_bf16_gemm_nn_f32(d_t_bf, lo.a[], M, lo.rank, lo.in_f, ctx)
    var d_a_t = _lora_bf16_gemm_tn_f32(d_t_bf, x_bf, M, lo.rank, lo.in_f, ctx)
    return KleinLoraDeviceGradTensors(TArc(d_a_t^), TArc(d_b_t^), TArc(d_x_lo^))


def klein_lora_bwd_device_resident_tensors_unfused(
    d_contrib: Tensor, x: Tensor, lo: LoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> KleinLoraDeviceGradTensors:
    var nb_t = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb_t^, ctx)
    var d_dy = mul_scalar(d_contrib, lo.scale, ctx)

    var d_t = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)
    var d_b_t = linear_backward_dw(d_dy, t, M, lo.rank, lo.out_f, ctx)

    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)
    var d_a_t = linear_backward_dw(d_t, x, M, lo.in_f, lo.rank, ctx)

    return KleinLoraDeviceGradTensors(TArc(d_a_t^), TArc(d_b_t^), TArc(d_x_lo^))


def klein_lora_bwd_device(
    d_contrib: Tensor, x: Tensor, lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> KleinLoraDeviceGrads:
    var lo_dev = lora_adapter_to_device(lo, ctx)
    return klein_lora_bwd_device_resident(d_contrib, x, lo_dev, M, ctx)


# ── host slice/scatter helpers for partial-projection LoRA (single block) ────
# Take the first `c0` columns of each row of a [rows, total] host buffer.
def klein_take_cols(
    src: List[Float32], rows: Int, total: Int, c0: Int
) -> List[Float32]:
    var o = List[Float32]()
    for r in range(rows):
        var base = r * total
        for c in range(c0):
            o.append(src[base + c])
    return o^


# Add a [rows, c0] delta into the first c0 columns of a [rows, total] buffer,
# returning the merged [rows, total] buffer (columns >= c0 unchanged).
def klein_add_cols(
    dst: List[Float32], delta: List[Float32], rows: Int, total: Int, c0: Int
) -> List[Float32]:
    var o = List[Float32]()
    for r in range(rows):
        var base = r * total
        for c in range(total):
            if c < c0:
                o.append(dst[base + c] + delta[r * c0 + c])
            else:
                o.append(dst[base + c])
    return o^


def klein_take_cols_device(
    src: Tensor, rows: Int, total: Int, c0: Int, ctx: DeviceContext
) raises -> Tensor:
    return slice(src, 1, 0, c0, ctx)


def klein_add_cols_device(
    dst: Tensor, delta: Tensor, rows: Int, total: Int, c0: Int, ctx: DeviceContext
) raises -> Tensor:
    var first = slice(dst, 1, 0, c0, ctx)
    var merged = add(first, delta, ctx)
    if c0 == total:
        return merged^
    var rest = slice(dst, 1, c0, total - c0, ctx)
    return concat(1, ctx, merged, rest)
