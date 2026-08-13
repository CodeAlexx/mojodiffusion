# ops/svdquant_w4a4.mojo — W4A4 runtime (MJ-1099 B.3b).
#
# The activation side of QuaRot W4A4: rotate the activation by a Hadamard, quantize
# per-token to int4, run the CUTLASS int4 GEMM (ops/int4_gemm.mojo) against the
# offline-rotated int4 weight (SquareQ quantize_svdquant_w4a4), rescale int32→bf16,
# add the bf16 low-rank branch + bias.
#
#   y = dequant( (X·H)_int4 @ qweight^T )·(xscale ⊗ wscale)
#       + (X @ lora_down) @ lora_up^T  +  bias
#   since (X·H)@(R·H)^T = X·R^T and R = W − lora_up·lora_down^T (offline residual).
#
# H is a data-free Sylvester Hadamard applied online by a fused fast Walsh–Hadamard
# transform (FWHT, O(K log K)) — NOT a dense X@H GEMM (that would be 4× the main
# GEMM). The FWHT + per-token int4 quant + pack are ONE kernel (one row per block,
# the whole row co-resident in F32 shared memory).
#
# Quality (SquareQ sim + gate): rank-128 per-out ⇒ cos ~0.99/layer, outlier-robust.
# Speed foundation: int4 GEMM = 4.4–7.3× cuBLAS bf16 (MJ-1099 B.1).

from std.gpu import thread_idx, block_idx
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.int4_gemm import int4_gemm_s4_nt
from serenitymojo.ops.linear import linear
from serenitymojo.ops.arch import query_gpu_arch, require_quant_backend, QB_INT4_W4A4

comptime _DYN1 = Layout.row_major(-1)
comptime _TPB = 256


struct SvdquantW4A4(Movable):
    """QuaRot W4A4 weight: per-output int4 of the Hadamard-rotated residual +
    bf16 rank-R low-rank branch (SquareQ quantize_svdquant_w4a4)."""
    var qweight: Tensor    # U8 [out, in/2]  (per-out int4, rotated residual)
    var wscale: Tensor     # BF16 [out]      (per-output scale)
    var lora_down: Tensor  # BF16 [in, R]
    var lora_up: Tensor    # BF16 [out, R]
    var bias: Tensor       # BF16 [out]
    var in_f: Int
    var out_f: Int
    var rank: Int

    def __init__(out self, var qweight: Tensor, var wscale: Tensor,
                var lora_down: Tensor, var lora_up: Tensor, var bias: Tensor,
                in_f: Int, out_f: Int, rank: Int) raises:
        # Fail loud at load time (once) on a GPU without int4 s4 IMMA — the
        # underlying CUTLASS shim is Ampere/Ada-only (removed on sm_90+), so on
        # the 5080 (sm_120) this raises a clear message instead of a later shim
        # load/launch error. On the 3090 (sm_86) it is a no-op. This is the
        # canonical guard point: a loader constructs the weight once.
        require_quant_backend(query_gpu_arch(), QB_INT4_W4A4)
        self.qweight = qweight^
        self.wscale = wscale^
        self.lora_down = lora_down^
        self.lora_up = lora_up^
        self.bias = bias^
        self.in_f = in_f
        self.out_f = out_f
        self.rank = rank


@fieldwise_init
struct FwhtQuantOut(Movable):
    """FWHT+quant output — move-only Tensors can't cross as a tuple."""
    var xq: Tensor      # packed int4 [M, K/2] U8
    var xscale: Tensor  # BF16 [M]


# ─────────────────────────────────────────────────────────────────────────────
# Fused FWHT + per-token int4 quant + pack. One block per row (M blocks).
#   in:  x[m, 0:K] bf16
#   out: xq[m, 0:K/2] packed int4 (lo=even), xscale[m] bf16
# K is comptime (the F32 shared row is stack_allocation[K]); pairs = K/2.
# ─────────────────────────────────────────────────────────────────────────────
def _fwht_quant_kernel[K: Int](
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],      # [M*K] flat
    xq: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],        # [M*K/2] flat
    xscale: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], # [M] flat
):
    var m = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[K, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var red = stack_allocation[_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var row_base = m * K

    # 1) load row → shared F32
    var i = tid
    while i < K:
        sh[i] = rebind[Scalar[DType.bfloat16]](x[row_base + i]).cast[DType.float32]()
        i += _TPB
    barrier()

    # 2) FWHT butterfly: log2(K) stages, K/2 pairs each.
    var half = K // 2
    var length = 1
    while length < K:
        var pp = tid
        while pp < half:
            var blk = pp // length
            var within = pp % length
            var j = blk * (2 * length) + within
            var a = sh[j]
            var b = sh[j + length]
            sh[j] = a + b
            sh[j + length] = a - b
            pp += _TPB
        barrier()
        length *= 2

    # 3) normalize by 1/sqrt(K)
    var inv = 1.0 / sqrt(Float32(K))
    i = tid
    while i < K:
        sh[i] = sh[i] * inv
        i += _TPB
    barrier()

    # 4) per-token absmax reduction → scale = absmax/7
    var lmax: Float32 = 0.0
    i = tid
    while i < K:
        var av = sh[i] if sh[i] >= 0.0 else -sh[i]
        if av > lmax:
            lmax = av
        i += _TPB
    red[tid] = lmax
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            var a = red[tid]
            var bb = red[tid + active]
            red[tid] = a if a > bb else bb
        barrier()
        active //= 2
    var absmax = red[0]
    var scale = absmax / 7.0
    if scale == 0.0:
        scale = 1.0
    var invscale = 1.0 / scale
    if tid == 0:
        xscale[m] = scale.cast[DType.bfloat16]()

    # 5) quantize + pack (lo=even). One byte per (2b, 2b+1).
    var bpair = tid
    var qbase = m * half
    while bpair < half:
        var e = sh[2 * bpair] * invscale
        var o = sh[2 * bpair + 1] * invscale
        var qe = Int(_round_clamp(e))
        var qo = Int(_round_clamp(o))
        var lo = qe & 0xF
        var hi = qo & 0xF
        xq[qbase + bpair] = UInt8(lo | (hi << 4))
        bpair += _TPB


def _round_clamp(v: Float32) -> Float32:
    var r = _rint(v)
    if r > 7.0:
        return 7.0
    if r < -8.0:
        return -8.0
    return r


def _rint(v: Float32) -> Float32:
    # round half to nearest (matches torch.round for our range)
    if v >= 0.0:
        return Float32(Int(v + 0.5))
    return Float32(Int(v - 0.5))


# ─────────────────────────────────────────────────────────────────────────────
# Rescale int32 accumulator → bf16:  out[m,n] = C[m,n] * xscale[m] * wscale[n].
# ─────────────────────────────────────────────────────────────────────────────
def _rescale_kernel(
    c: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],         # [M*N] int32 GEMM acc
    xscale: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], # [M]
    wscale: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], # [N]
    addend: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin], # [M*N] low-rank+bias
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],      # [M*N] out
    M_w: Int32, N_w: Int64,
):
    """out[m,n] = C[m,n]·xscale[m]·wscale[n] + addend[m,n]  (fused epilogue)."""
    var M = Int(M_w)
    var N = Int(N_w)
    var idx = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    var total = M * N
    if idx < total:
        var m = idx // N
        var n = idx % N
        var cv = Float32(rebind[Scalar[DType.int32]](c[idx]))
        var xs = rebind[Scalar[DType.bfloat16]](xscale[m]).cast[DType.float32]()
        var ws = rebind[Scalar[DType.bfloat16]](wscale[n]).cast[DType.float32]()
        var ad = rebind[Scalar[DType.bfloat16]](addend[idx]).cast[DType.float32]()
        o[idx] = (cv * xs * ws + ad).cast[DType.bfloat16]()


# ─────────────────────────────────────────────────────────────────────────────
# Host wrapper: fused FWHT + per-token int4 quant. x BF16 [M,K] → (xq packed int4
# [M,K/2] U8, xscale BF16 [M]). K is comptime — dispatch on the supported sizes.
# ─────────────────────────────────────────────────────────────────────────────
def _fwht_quant_launch[K: Int](
    x: Tensor, M: Int, ctx: DeviceContext
) raises -> FwhtQuantOut:
    var half = K // 2
    var xq_buf = ctx.enqueue_create_buffer[DType.uint8](M * half)
    var xs_buf = ctx.enqueue_create_buffer[DType.uint8](M * 2)  # bf16 [M]

    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * K))
    var xq_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * half))
    var xs_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
    var XQ = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(xq_buf.unsafe_ptr())
        ),
        runtime_layout=xq_rl,
    )
    var XS = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(xs_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=xs_rl,
    )

    comptime kern = _fwht_quant_kernel[K]
    ctx.enqueue_function[kern](X, XQ, XS, grid_dim=M, block_dim=_TPB)

    var xq = Tensor(xq_buf^, [M, half], STDtype.U8)
    var xs = Tensor(xs_buf^, [M], STDtype.BF16)
    return FwhtQuantOut(xq^, xs^)


def fwht_quant(x: Tensor, ctx: DeviceContext) raises -> FwhtQuantOut:
    """x BF16 [M,K] → (xq int4 packed [M,K/2] U8, xscale BF16 [M]). K comptime-
    dispatched (2048/4096/8192; 16384 pending dynamic-shared)."""
    var shp = x.shape()
    var M = shp[0]
    var K = shp[1]
    if K == 2048:
        return _fwht_quant_launch[2048](x, M, ctx)
    elif K == 4096:
        return _fwht_quant_launch[4096](x, M, ctx)
    elif K == 8192:
        return _fwht_quant_launch[8192](x, M, ctx)
    else:
        raise Error(String("fwht_quant: unsupported K=") + String(K)
                    + " (2048/4096/8192; 16384 needs dynamic shared)")


# ─────────────────────────────────────────────────────────────────────────────
# svdquant_linear_w4a4 — the full W4A4 linear:
#   y = rescale( (X·H)_int4 @ qweight^T )·(xscale⊗wscale) + (X@lora_down)@lora_up^T + bias
# The low-rank branch uses the UN-rotated X (bf16). Bias folded into the low-rank
# GEMM; the rescale kernel adds it. K comptime-dispatched via fwht_quant.
# ─────────────────────────────────────────────────────────────────────────────
def svdquant_linear_w4a4(x: Tensor, w: SvdquantW4A4, ctx: DeviceContext) raises -> Tensor:
    var xshape = x.shape()
    var M = xshape[0]
    var inh = w.in_f
    var out_f = w.out_f
    if xshape[len(xshape) - 1] != inh:
        raise Error(String("svdquant_linear_w4a4: x last dim ")
                    + String(xshape[len(xshape) - 1]) + " != in_f " + String(inh))
    if x.dtype() != STDtype.BF16:
        raise Error("svdquant_linear_w4a4: x must be BF16")

    # 1) FWHT rotate + per-token int4 quant of the activation.
    var fq = fwht_quant(x, ctx)

    # 2) int4 GEMM: C[M,out] int32 = (X·H)_int4 @ qweight^T (transpose_b).
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](M * out_f * 4)
    int4_gemm_s4_nt(fq.xq.buf, w.qweight.buf, c_buf, M, out_f, inh, ctx)

    # 3) low-rank (UN-rotated x, bf16) with bias folded in:
    #    down = x @ lora_down  ([in,R], transpose_b=False → [M,R])
    #    up   = down @ lora_up^T + bias  ([out,R], transpose_b=True → [M,out])
    var down = linear(x, w.lora_down, None, ctx, transpose_b=False)
    var up = linear(down, w.lora_up, Optional(w.bias.clone(ctx)), ctx, transpose_b=True)

    # 4) rescale + fused add:  y = C·(xscale⊗wscale) + up
    var o_buf = ctx.enqueue_create_buffer[DType.uint8](M * out_f * 2)
    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * out_f))
    var xs_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M))
    var ws_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_f))
    var up_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](M * out_f))
    var C = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(c_buf.unsafe_ptr().bitcast[Int32]())
        ),
        runtime_layout=c_rl,
    )
    var XS = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(fq.xscale.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=xs_rl,
    )
    var WS = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(w.wscale.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=ws_rl,
    )
    var AD = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(up.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=up_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(o_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=up_rl,
    )
    var total = M * out_f
    var grid = (total + _TPB - 1) // _TPB
    ctx.enqueue_function[_rescale_kernel](
        C, XS, WS, AD, O, Int32(M), Int64(out_f), grid_dim=grid, block_dim=_TPB)

    return Tensor(o_buf^, [M, out_f], STDtype.BF16)
