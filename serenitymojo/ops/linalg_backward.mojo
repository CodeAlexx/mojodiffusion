# ops/linalg_backward.mojo — backward kernels for the GEMM-family ops.
#
# Tier 2 of FULL_PORT_TRAINING_PLAN.md §3 (MatMul, BatchMatMul, Linear, AddBias).
# Backward = transposed GEMMs, leaning on `linalg.matmul.vendor.blas.matmul`
# (the SAME vendor BLAS the forward `linear.mojo` / `attention_backward.mojo`
# use; transpose_a / transpose_b / c_row_major are all proven there).
#
# Pure F32. Inputs/outputs are F32 `Tensor`s. The bias column-sums are not GEMMs,
# so they run as a small one-thread-per-output-column reduction kernel.
#
# ── Math ─────────────────────────────────────────────────────────────────────
#   matmul:   C[M,N] = A[M,K] @ B[K,N]
#             d_a = grad_c @ Bᵀ        [M,K]   (grad_c[M,N], B[K,N] → transpose_b)
#             d_b = Aᵀ @ grad_c        [K,N]   (A[M,K]      → transpose_a)
#
#   bmm:      C[Bt,M,N] = A[Bt,M,K] @ B[Bt,K,N]  (per batch element = matmul)
#             d_a [Bt,M,K], d_b [Bt,K,N]
#
#   linear:   y[M,out] = x[M,in] @ W[out,in]ᵀ + b[out]
#             d_x = grad_y @ W         [M,in]   (W[out,in], contraction over out
#                                                → NO transpose: [M,out]@[out,in])
#             d_W = grad_yᵀ @ x        [out,in] (grad_y[M,out] → transpose_a)
#             d_b = colsum(grad_y)     [out]
#
#   addbias:  y[M,out] = x[M,out] + b[out]
#             d_b = colsum(grad_y)     [out]
#             d_x = grad_y             (passthrough)
#
# NAMING NOTE: the matmul entry points are `mm_backward` / `bmm_backward`, not
# `matmul_backward` / `batchmatmul_backward` as the build plan tentatively named
# them. Mojo 1.0.0b1 rejects defining a top-level function whose name shares the
# base token of an in-scope imported symbol — importing `matmul` from the vendor
# BLAS makes `matmul_*` (and `batchmatmul_*`) un-definable. The `matmul` import
# is mandatory (it is the only working GEMM entry), so the defs are renamed.
# `linear_backward` / `addbias_backward` keep their planned names.
#
# Device views: each F32 `Tensor`'s byte buffer is reinterpreted inline as a
# [rows,cols] row-major LayoutTensor at the call site (a LayoutTensor view cannot
# be returned from a helper def without an explicit origin).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── Result bundles (Tensor is move-only → small owned structs, as SdpaGrads) ──
struct MatmulGrads(Movable):
    """Backward outputs of matmul / bmm: gradients wrt a and b."""

    var d_a: Tensor
    var d_b: Tensor

    def __init__(out self, var d_a: Tensor, var d_b: Tensor):
        self.d_a = d_a^
        self.d_b = d_b^


struct LinearGrads(Movable):
    """Backward outputs of linear: gradients wrt x, W, and bias."""

    var d_x: Tensor
    var d_w: Tensor
    var d_b: Tensor

    def __init__(out self, var d_x: Tensor, var d_w: Tensor, var d_b: Tensor):
        self.d_x = d_x^
        self.d_w = d_w^
        self.d_b = d_b^


struct AddBiasGrads(Movable):
    """Backward outputs of addbias: gradient wrt x (passthrough) and bias."""

    var d_x: Tensor
    var d_b: Tensor

    def __init__(out self, var d_x: Tensor, var d_b: Tensor):
        self.d_x = d_x^
        self.d_b = d_b^


# ── column-sum kernel: d_b[j] = sum_i grad_y[i,j] over the M rows ─────────────
def _colsum_kernel(
    grad_y: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    out_buf: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    m: Int,
    out_dim: Int,
):
    var j = Int(global_idx.x)
    if j < out_dim:
        var acc = Float32(0.0)
        for i in range(m):
            acc += rebind[Scalar[DType.float32]](grad_y[i, j])
        out_buf[j] = rebind[out_buf.element_type](acc)


# ── d2d copy kernel (passthrough d_x for addbias) ────────────────────────────
def _copy_kernel(
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        dst[idx] = src[idx]


# ── helpers ──────────────────────────────────────────────────────────────────
def _new_f32(rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    """Allocate a fresh [rows,cols] F32 Tensor (device buffer)."""
    var buf = ctx.enqueue_create_buffer[DType.uint8](rows * cols * 4)
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor(buf^, sh^, STDtype.F32)


def _colsum(grad_y: Tensor, m: Int, out_dim: Int, ctx: DeviceContext) raises -> Tensor:
    """d_b[out_dim] = column-sum of grad_y[m, out_dim] over the m rows."""
    var gy_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, out_dim))
    var gy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        grad_y.buf.unsafe_ptr().bitcast[Float32](), gy_rl
    )
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_dim * 4)
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_dim))
    var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), o_rl
    )
    var grid = (out_dim + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_colsum_kernel, _colsum_kernel](
        gy, o, m, out_dim, grid_dim=grid, block_dim=_BLOCK
    )
    var sh = List[Int]()
    sh.append(out_dim)
    return Tensor(out_buf^, sh^, STDtype.F32)


# ── matmul backward ──────────────────────────────────────────────────────────
#   C[M,N] = A[M,K] @ B[K,N] ;  d_a = grad_c @ Bᵀ [M,K] ;  d_b = Aᵀ @ grad_c [K,N]
def mm_backward(
    grad_c: Tensor,
    a: Tensor,
    b: Tensor,
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises -> MatmulGrads:
    # BF16/F16 storage path: cast inputs up to F32, run the F32 GEMMs below, then
    # cast grads back down to the storage dtype. F32 path byte-identical (branch
    # only taken when an input is not F32). The vendor BLAS here is F32-only, so
    # the cast-up is mandatory rather than a fast/slow choice.
    if grad_c.dtype() != STDtype.F32 or a.dtype() != STDtype.F32 or b.dtype() != STDtype.F32:
        var out_dt = a.dtype()
        var gc32 = cast_tensor(grad_c, STDtype.F32, ctx)
        var a32 = cast_tensor(a, STDtype.F32, ctx)
        var b32 = cast_tensor(b, STDtype.F32, ctx)
        var g32 = mm_backward(gc32, a32, b32, M, N, K, ctx)
        var da_dn = cast_tensor(g32.d_a^, out_dt, ctx)
        var db_dn = cast_tensor(g32.d_b^, out_dt, ctx)
        return MatmulGrads(da_dn^, db_dn^)
    var d_a = _new_f32(M, K, ctx)
    var d_b = _new_f32(K, N, ctx)

    var mn_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, N))
    var mk_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, K))
    var kn_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](K, N))
    var gc = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](grad_c.buf.unsafe_ptr().bitcast[Float32](), mn_rl)
    var av = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](a.buf.unsafe_ptr().bitcast[Float32](), mk_rl)
    var bv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](b.buf.unsafe_ptr().bitcast[Float32](), kn_rl)
    var da = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_a.buf.unsafe_ptr().bitcast[Float32](), mk_rl)
    var db = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_b.buf.unsafe_ptr().bitcast[Float32](), kn_rl)

    # d_a[M,K] = grad_c[M,N] @ B[K,N]ᵀ  (transpose_b)
    matmul(ctx, da, gc, bv, transpose_b=True, c_row_major=True)
    # d_b[K,N] = A[M,K]ᵀ @ grad_c[M,N]  (transpose_a)
    matmul(ctx, db, av, gc, transpose_a=True, c_row_major=True)
    ctx.synchronize()
    return MatmulGrads(d_a^, d_b^)


# ── batched matmul backward (per batch element = matmul backward) ─────────────
#   a:[Bt*M,K] b:[Bt*K,N] grad_c:[Bt*M,N]  → d_a:[Bt*M,K] d_b:[Bt*K,N]
def bmm_backward(
    grad_c: Tensor,
    a: Tensor,
    b: Tensor,
    Batch: Int,
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises -> MatmulGrads:
    # BF16/F16 storage path: cast up, run F32 GEMMs, cast grads down.
    # F32 path byte-identical (branch only on non-F32 input). Vendor BLAS is
    # F32-only, so the cast-up is mandatory rather than a fast/slow choice.
    if grad_c.dtype() != STDtype.F32 or a.dtype() != STDtype.F32 or b.dtype() != STDtype.F32:
        var out_dt = a.dtype()
        var gc32 = cast_tensor(grad_c, STDtype.F32, ctx)
        var a32 = cast_tensor(a, STDtype.F32, ctx)
        var b32 = cast_tensor(b, STDtype.F32, ctx)
        var g32 = bmm_backward(gc32, a32, b32, Batch, M, N, K, ctx)
        var da_dn = cast_tensor(g32.d_a^, out_dt, ctx)
        var db_dn = cast_tensor(g32.d_b^, out_dt, ctx)
        return MatmulGrads(da_dn^, db_dn^)
    var d_a = _new_f32(Batch * M, K, ctx)
    var d_b = _new_f32(Batch * K, N, ctx)

    var head_mk = RuntimeLayout[_DYN2].row_major(IndexList[2](M, K))
    var head_kn = RuntimeLayout[_DYN2].row_major(IndexList[2](K, N))
    var head_mn = RuntimeLayout[_DYN2].row_major(IndexList[2](M, N))

    var aptr = a.buf.unsafe_ptr().bitcast[Float32]()
    var bptr = b.buf.unsafe_ptr().bitcast[Float32]()
    var gcptr = grad_c.buf.unsafe_ptr().bitcast[Float32]()
    var daptr = d_a.buf.unsafe_ptr().bitcast[Float32]()
    var dbptr = d_b.buf.unsafe_ptr().bitcast[Float32]()

    for bi in range(Batch):
        var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr + bi * M * K, head_mk)
        var B = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](bptr + bi * K * N, head_kn)
        var GC = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gcptr + bi * M * N, head_mn)
        var DA = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](daptr + bi * M * K, head_mk)
        var DB = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dbptr + bi * K * N, head_kn)
        # d_a[M,K] = grad_c[M,N] @ B[K,N]ᵀ
        matmul(ctx, DA, GC, B, transpose_b=True, c_row_major=True)
        # d_b[K,N] = A[M,K]ᵀ @ grad_c[M,N]
        matmul(ctx, DB, A, GC, transpose_a=True, c_row_major=True)
    ctx.synchronize()
    return MatmulGrads(d_a^, d_b^)


# ── linear backward ──────────────────────────────────────────────────────────
#   y[M,out] = x[M,in] @ W[out,in]ᵀ + b[out]
#   d_x = grad_y @ W      [M,in]   ;  d_W = grad_yᵀ @ x [out,in]  ;  d_b = colsum
def linear_backward(
    grad_y: Tensor,
    x: Tensor,
    weight: Tensor,
    M: Int,
    in_features: Int,
    out_features: Int,
    ctx: DeviceContext,
) raises -> LinearGrads:
    # BF16/F16 storage path: cast up, run F32 GEMMs, cast grads down.
    # F32 path byte-identical (branch only on non-F32 input).
    if grad_y.dtype() != STDtype.F32 or x.dtype() != STDtype.F32 or weight.dtype() != STDtype.F32:
        var out_dt = x.dtype()
        var gy32 = cast_tensor(grad_y, STDtype.F32, ctx)
        var x32 = cast_tensor(x, STDtype.F32, ctx)
        var w32 = cast_tensor(weight, STDtype.F32, ctx)
        var g32 = linear_backward(gy32, x32, w32, M, in_features, out_features, ctx)
        var dx_dn = cast_tensor(g32.d_x^, out_dt, ctx)
        var dw_dn = cast_tensor(g32.d_w^, out_dt, ctx)
        var db_dn = cast_tensor(g32.d_b^, out_dt, ctx)
        return LinearGrads(dx_dn^, dw_dn^, db_dn^)
    var d_x = _new_f32(M, in_features, ctx)
    var d_w = _new_f32(out_features, in_features, ctx)

    var mo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, out_features))
    var mi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, in_features))
    var oi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_features, in_features))
    var gy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float32](), mo_rl)
    var xv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), mi_rl)
    var wv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float32](), oi_rl)
    var dx = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_x.buf.unsafe_ptr().bitcast[Float32](), mi_rl)
    var dw = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_w.buf.unsafe_ptr().bitcast[Float32](), oi_rl)

    # d_x[M,in] = grad_y[M,out] @ W[out,in]   (contraction over out → no transpose)
    matmul(ctx, dx, gy, wv, c_row_major=True)
    # d_W[out,in] = grad_y[M,out]ᵀ @ x[M,in]  (transpose_a)
    matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    var d_b = _colsum(grad_y, M, out_features, ctx)
    ctx.synchronize()
    return LinearGrads(d_x^, d_w^, d_b^)


# ── linear backward, d_x ONLY (frozen-weight path) ───────────────────────────
#   For FROZEN base weights (LoRA training): only d_x flows up the chain; d_W and
#   d_b are computed-then-discarded by the full `linear_backward`. This sibling
#   skips the d_W matmul, the d_b colsum, and their allocations/readback entirely.
#   d_x[M,in] = grad_y[M,out] @ W[out,in]   (same GEMM as linear_backward's d_x).
#   x is NOT needed (only used for d_W). The d_x math is byte-identical to
#   linear_backward.d_x — validated by every parity gate that checks d_x.
def linear_backward_dx(
    grad_y: Tensor,
    weight: Tensor,
    M: Int,
    in_features: Int,
    out_features: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    # BF16/F16 storage path: cast up, run the F32 GEMM, cast d_x down.
    # F32 path byte-identical (branch only on non-F32 input). Project is F32-only,
    # so this branch is not exercised by the gates — kept for dtype parity.
    if grad_y.dtype() != STDtype.F32 or weight.dtype() != STDtype.F32:
        var out_dt = grad_y.dtype()
        var gy32 = cast_tensor(grad_y, STDtype.F32, ctx)
        var w32 = cast_tensor(weight, STDtype.F32, ctx)
        var dx32 = linear_backward_dx(gy32, w32, M, in_features, out_features, ctx)
        return cast_tensor(dx32^, out_dt, ctx)
    var d_x = _new_f32(M, in_features, ctx)

    var mo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, out_features))
    var mi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, in_features))
    var oi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_features, in_features))
    var gy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float32](), mo_rl)
    var wv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float32](), oi_rl)
    var dx = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_x.buf.unsafe_ptr().bitcast[Float32](), mi_rl)

    # d_x[M,in] = grad_y[M,out] @ W[out,in]   (contraction over out → no transpose)
    matmul(ctx, dx, gy, wv, c_row_major=True)
    ctx.synchronize()
    return d_x^


# ── addbias backward ─────────────────────────────────────────────────────────
#   y[M,out] = x[M,out] + b[out] ;  d_b = colsum(grad_y) [out] ; d_x = grad_y
def addbias_backward(
    grad_y: Tensor,
    M: Int,
    out_features: Int,
    ctx: DeviceContext,
) raises -> AddBiasGrads:
    var d_b = _colsum(grad_y, M, out_features, ctx)

    # d_x = grad_y (fresh copy so the caller owns an independent buffer).
    var n = M * out_features
    var d_x = _new_f32(M, out_features, ctx)
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var src = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad_y.buf.unsafe_ptr().bitcast[Float32](), src_rl
    )
    var dst = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        d_x.buf.unsafe_ptr().bitcast[Float32](), src_rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_copy_kernel, _copy_kernel](
        src, dst, n, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return AddBiasGrads(d_x^, d_b^)
