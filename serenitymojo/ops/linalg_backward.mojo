# ops/linalg_backward.mojo — backward kernels for the GEMM-family ops.
#
# Tier 2 of FULL_PORT_TRAINING_PLAN.md §3 (MatMul, BatchMatMul, Linear, AddBias).
# Backward = transposed GEMMs, leaning on `linalg.matmul.vendor.blas.matmul`
# (the SAME vendor BLAS the forward `linear.mojo` / `attention_backward.mojo`
# use; transpose_a / transpose_b / c_row_major are all proven there).
#
# GEMM C buffers and bias reductions use F32 math. BF16/F16 public storage paths
# pass BF16/F16 inputs directly to BLAS and cast only the GEMM accumulator output
# back to storage dtype at the boundary.
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
from serenitymojo.ops.cast import cast_tensor, cast_tensor_slab
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.autograd_v2.step_slab import StepSlab


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
def _colsum_kernel[dtype: DType](
    grad_y: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    out_buf: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    m: Int,
    out_dim: Int,
):
    var j = Int(global_idx.x)
    if j < out_dim:
        var acc = Float32(0.0)
        for i in range(m):
            acc += rebind[Scalar[dtype]](grad_y[i, j]).cast[DType.float32]()
        out_buf[j] = rebind[out_buf.element_type](acc.cast[dtype]())


# ── d2d copy kernel (passthrough d_x for addbias) ────────────────────────────
def _copy_kernel[dtype: DType](
    src: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        dst[idx] = src[idx]


def _cast_f32_to_storage_kernel[dtype: DType](
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var v = rebind[Scalar[DType.float32]](src[idx])
        dst[idx] = rebind[dst.element_type](v.cast[dtype]())


# ── helpers ──────────────────────────────────────────────────────────────────
def _new_f32(rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    """Allocate a fresh [rows,cols] F32 Tensor (device buffer)."""
    var buf = ctx.enqueue_create_buffer[DType.uint8](rows * cols * 4)
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor(buf^, sh^, STDtype.F32)


def _new_storage(rows: Int, cols: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    """Allocate a fresh [rows,cols] Tensor in `dtype` storage."""
    var buf = ctx.enqueue_create_buffer[DType.uint8](rows * cols * dtype.byte_size())
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor(buf^, sh^, dtype)


def _new_f32_scratch(
    rows: Int,
    cols: Int,
    mut scratch: ScratchRingAllocator,
    reverse: Bool,
) raises -> Tensor:
    """Allocate a scratch-backed [rows,cols] F32 Tensor."""
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    if reverse:
        return scratch.alloc_tensor_reverse(sh^, STDtype.F32)
    return scratch.alloc_tensor(sh^, STDtype.F32)


def _new_storage_scratch(
    rows: Int,
    cols: Int,
    dtype: STDtype,
    mut scratch: ScratchRingAllocator,
    reverse: Bool,
) raises -> Tensor:
    """Allocate a scratch-backed [rows,cols] tensor in `dtype` storage."""
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    if reverse:
        return scratch.alloc_tensor_reverse(sh^, dtype)
    return scratch.alloc_tensor(sh^, dtype)


def _new_f32_slab(rows: Int, cols: Int, mut slab: StepSlab) raises -> Tensor:
    """StepSlab variant of `_new_f32` (this file :135) — same [rows,cols] F32
    tensor; ONLY the allocation source changes (contract C8, Phase P4)."""
    var buf = slab.alloc(rows * cols * 4)
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor(buf^, sh^, STDtype.F32)


def _clone_to_slab(t: Tensor, ctx: DeviceContext, mut slab: StepSlab) raises -> Tensor:
    """StepSlab variant of Tensor.clone (tensor.mojo:70) — same d2d copy;
    ONLY the allocation source changes (contract C8, Phase P4)."""
    var dev = slab.alloc(t.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=t.buf)
    # P5-CAPTURE-SYNC-REMOVED (C9): single-stream ordering (TIER2 precedent,
    # ops/attention.mojo); no sync allowed inside a captured region. Values
    # unchanged — only host-visibility timing moved (bit-gates protect).
    return Tensor(dev^, t.shape(), t.dtype())


def _cast_f32_to_storage_scratch(
    src: Tensor,
    dtype: STDtype,
    mut scratch: ScratchRingAllocator,
    reverse: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    """Cast an F32 accumulator tensor into scratch-backed `dtype` storage."""
    if src.dtype() != STDtype.F32:
        raise Error("_cast_f32_to_storage_scratch: source must be F32")
    var sh = src.shape()
    if len(sh) != 2:
        raise Error("_cast_f32_to_storage_scratch: expected rank-2 source")
    var out = _new_storage_scratch(sh[0], sh[1], dtype, scratch, reverse)
    if dtype == STDtype.F32:
        ctx.enqueue_copy(dst_buf=out.buf, src_buf=src.buf)
        return out^
    var n = src.numel()
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var src_lt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        src.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    var out_dt = dtype.to_mojo_dtype()
    if out_dt == DType.bfloat16:
        var dst_lt = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[
            _cast_f32_to_storage_kernel[DType.bfloat16],
            _cast_f32_to_storage_kernel[DType.bfloat16],
        ](src_lt, dst_lt, n, grid_dim=grid, block_dim=_BLOCK)
    elif out_dt == DType.float16:
        var dst_lt = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[
            _cast_f32_to_storage_kernel[DType.float16],
            _cast_f32_to_storage_kernel[DType.float16],
        ](src_lt, dst_lt, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        raise Error("_cast_f32_to_storage_scratch: unsupported output dtype")
    return out^


def _colsum(grad_y: Tensor, m: Int, out_dim: Int, ctx: DeviceContext) raises -> Tensor:
    """d_b[out_dim] = column-sum of grad_y[m, out_dim] over the m rows."""
    var gy_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, out_dim))
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_dim * grad_y.dtype().byte_size()
    )
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_dim))
    var grid = (out_dim + _BLOCK - 1) // _BLOCK
    var dt = grad_y.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var gy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_y.buf.unsafe_ptr().bitcast[Float32](), gy_rl)
        var o = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
        ctx.enqueue_function[
            _colsum_kernel[DType.float32], _colsum_kernel[DType.float32]
        ](gy, o, m, out_dim, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var gy = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            grad_y.buf.unsafe_ptr().bitcast[BFloat16](), gy_rl)
        var o = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
        ctx.enqueue_function[
            _colsum_kernel[DType.bfloat16], _colsum_kernel[DType.bfloat16]
        ](gy, o, m, out_dim, grid_dim=grid, block_dim=_BLOCK)
    else:
        var gy = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            grad_y.buf.unsafe_ptr().bitcast[Float16](), gy_rl)
        var o = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl)
        ctx.enqueue_function[
            _colsum_kernel[DType.float16], _colsum_kernel[DType.float16]
        ](gy, o, m, out_dim, grid_dim=grid, block_dim=_BLOCK)
    var sh = List[Int]()
    sh.append(out_dim)
    return Tensor(out_buf^, sh^, grad_y.dtype())


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
    if grad_c.dtype() != a.dtype() or a.dtype() != b.dtype():
        raise Error("mm_backward: grad_c/a/b dtype mismatch")
    var d_a = _new_f32(M, K, ctx)
    var d_b = _new_f32(K, N, ctx)

    var mn_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, N))
    var mk_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, K))
    var kn_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](K, N))
    var da = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_a.buf.unsafe_ptr().bitcast[Float32](), mk_rl)
    var db = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_b.buf.unsafe_ptr().bitcast[Float32](), kn_rl)

    var dt = a.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var gc = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](grad_c.buf.unsafe_ptr().bitcast[Float32](), mn_rl)
        var av = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](a.buf.unsafe_ptr().bitcast[Float32](), mk_rl)
        var bv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](b.buf.unsafe_ptr().bitcast[Float32](), kn_rl)
        matmul(ctx, da, gc, bv, transpose_b=True, c_row_major=True)
        matmul(ctx, db, av, gc, transpose_a=True, c_row_major=True)
    elif dt == DType.bfloat16:
        var gc = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](grad_c.buf.unsafe_ptr().bitcast[BFloat16](), mn_rl)
        var av = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](a.buf.unsafe_ptr().bitcast[BFloat16](), mk_rl)
        var bv = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](b.buf.unsafe_ptr().bitcast[BFloat16](), kn_rl)
        matmul(ctx, da, gc, bv, transpose_b=True, c_row_major=True)
        matmul(ctx, db, av, gc, transpose_a=True, c_row_major=True)
    else:
        var gc = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](grad_c.buf.unsafe_ptr().bitcast[Float16](), mn_rl)
        var av = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](a.buf.unsafe_ptr().bitcast[Float16](), mk_rl)
        var bv = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](b.buf.unsafe_ptr().bitcast[Float16](), kn_rl)
        matmul(ctx, da, gc, bv, transpose_b=True, c_row_major=True)
        matmul(ctx, db, av, gc, transpose_a=True, c_row_major=True)
    if a.dtype() == STDtype.F32:
        return MatmulGrads(d_a^, d_b^)
    var da_dn = cast_tensor(d_a^, a.dtype(), ctx)
    var db_dn = cast_tensor(d_b^, b.dtype(), ctx)
    return MatmulGrads(da_dn^, db_dn^)


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
    if grad_c.dtype() != a.dtype() or a.dtype() != b.dtype():
        raise Error("bmm_backward: grad_c/a/b dtype mismatch")
    var d_a = _new_f32(Batch * M, K, ctx)
    var d_b = _new_f32(Batch * K, N, ctx)

    var head_mk = RuntimeLayout[_DYN2].row_major(IndexList[2](M, K))
    var head_kn = RuntimeLayout[_DYN2].row_major(IndexList[2](K, N))
    var head_mn = RuntimeLayout[_DYN2].row_major(IndexList[2](M, N))

    var daptr = d_a.buf.unsafe_ptr().bitcast[Float32]()
    var dbptr = d_b.buf.unsafe_ptr().bitcast[Float32]()

    var dt = a.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var aptr = a.buf.unsafe_ptr().bitcast[Float32]()
        var bptr = b.buf.unsafe_ptr().bitcast[Float32]()
        var gcptr = grad_c.buf.unsafe_ptr().bitcast[Float32]()
        for bi in range(Batch):
            var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](aptr + bi * M * K, head_mk)
            var B = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](bptr + bi * K * N, head_kn)
            var GC = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](gcptr + bi * M * N, head_mn)
            var DA = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](daptr + bi * M * K, head_mk)
            var DB = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dbptr + bi * K * N, head_kn)
            matmul(ctx, DA, GC, B, transpose_b=True, c_row_major=True)
            matmul(ctx, DB, A, GC, transpose_a=True, c_row_major=True)
    elif dt == DType.bfloat16:
        var aptr = a.buf.unsafe_ptr().bitcast[BFloat16]()
        var bptr = b.buf.unsafe_ptr().bitcast[BFloat16]()
        var gcptr = grad_c.buf.unsafe_ptr().bitcast[BFloat16]()
        for bi in range(Batch):
            var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](aptr + bi * M * K, head_mk)
            var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](bptr + bi * K * N, head_kn)
            var GC = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](gcptr + bi * M * N, head_mn)
            var DA = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](daptr + bi * M * K, head_mk)
            var DB = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dbptr + bi * K * N, head_kn)
            matmul(ctx, DA, GC, B, transpose_b=True, c_row_major=True)
            matmul(ctx, DB, A, GC, transpose_a=True, c_row_major=True)
    else:
        var aptr = a.buf.unsafe_ptr().bitcast[Float16]()
        var bptr = b.buf.unsafe_ptr().bitcast[Float16]()
        var gcptr = grad_c.buf.unsafe_ptr().bitcast[Float16]()
        for bi in range(Batch):
            var A = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](aptr + bi * M * K, head_mk)
            var B = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](bptr + bi * K * N, head_kn)
            var GC = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](gcptr + bi * M * N, head_mn)
            var DA = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](daptr + bi * M * K, head_mk)
            var DB = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dbptr + bi * K * N, head_kn)
            matmul(ctx, DA, GC, B, transpose_b=True, c_row_major=True)
            matmul(ctx, DB, A, GC, transpose_a=True, c_row_major=True)
    if a.dtype() == STDtype.F32:
        return MatmulGrads(d_a^, d_b^)
    var da_dn = cast_tensor(d_a^, a.dtype(), ctx)
    var db_dn = cast_tensor(d_b^, b.dtype(), ctx)
    return MatmulGrads(da_dn^, db_dn^)


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
    var mixed_base = (
        grad_y.dtype() == STDtype.F32
        and x.dtype() == STDtype.F32
        and (weight.dtype() == STDtype.BF16 or weight.dtype() == STDtype.F16)
    )
    if grad_y.dtype() != x.dtype() or (x.dtype() != weight.dtype() and not mixed_base):
        raise Error("linear_backward: grad_y/x/weight dtype mismatch")
    var d_x = _new_f32(M, in_features, ctx)
    var d_w = _new_f32(out_features, in_features, ctx)

    var mo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, out_features))
    var mi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, in_features))
    var oi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_features, in_features))
    var dx = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_x.buf.unsafe_ptr().bitcast[Float32](), mi_rl)
    var dw = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_w.buf.unsafe_ptr().bitcast[Float32](), oi_rl)

    var dt = x.dtype().to_mojo_dtype()
    var wdt = weight.dtype().to_mojo_dtype()
    if dt == DType.float32 and wdt == DType.float32:
        var gy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float32](), mo_rl)
        var xv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), mi_rl)
        var wv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float32](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    elif dt == DType.float32 and wdt == DType.bfloat16:
        var gy_cast = cast_tensor(grad_y, STDtype.BF16, ctx, False)
        var x_cast = cast_tensor(x, STDtype.BF16, ctx, False)
        var gy = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](gy_cast.buf.unsafe_ptr().bitcast[BFloat16](), mo_rl)
        var xv = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](x_cast.buf.unsafe_ptr().bitcast[BFloat16](), mi_rl)
        var wv = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[BFloat16](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    elif dt == DType.float32 and wdt == DType.float16:
        var gy_cast = cast_tensor(grad_y, STDtype.F16, ctx, False)
        var x_cast = cast_tensor(x, STDtype.F16, ctx, False)
        var gy = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](gy_cast.buf.unsafe_ptr().bitcast[Float16](), mo_rl)
        var xv = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](x_cast.buf.unsafe_ptr().bitcast[Float16](), mi_rl)
        var wv = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float16](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    elif dt == DType.bfloat16:
        var gy = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[BFloat16](), mo_rl)
        var xv = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), mi_rl)
        var wv = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[BFloat16](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    else:
        var gy = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float16](), mo_rl)
        var xv = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float16](), mi_rl)
        var wv = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float16](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    var d_b = _colsum(grad_y, M, out_features, ctx)
    if x.dtype() == STDtype.F32:
        return LinearGrads(d_x^, d_w^, d_b^)
    var dx_dn = cast_tensor(d_x^, x.dtype(), ctx)
    var dw_dn = cast_tensor(d_w^, weight.dtype(), ctx)
    return LinearGrads(dx_dn^, dw_dn^, d_b^)


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
    if grad_y.dtype() != weight.dtype() and grad_y.dtype() != STDtype.F32:
        raise Error("linear_backward_dx: grad_y/weight dtype mismatch")
    var d_x = _new_f32(M, in_features, ctx)

    var mo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, out_features))
    var mi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, in_features))
    var oi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_features, in_features))
    var dx = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_x.buf.unsafe_ptr().bitcast[Float32](), mi_rl)

    if weight.dtype() == STDtype.F32:
        if grad_y.dtype() != STDtype.F32:
            raise Error("linear_backward_dx: F32 weight requires F32 grad")
        var gy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float32](), mo_rl)
        var wv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float32](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
    elif weight.dtype() == STDtype.BF16:
        var gy_cast: Tensor
        if grad_y.dtype() == STDtype.BF16:
            gy_cast = grad_y.clone(ctx)
        else:
            gy_cast = cast_tensor(grad_y, STDtype.BF16, ctx, False)
        var gy = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](gy_cast.buf.unsafe_ptr().bitcast[BFloat16](), mo_rl)
        var wv = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[BFloat16](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
    else:
        var gy_cast: Tensor
        if grad_y.dtype() == STDtype.F16:
            gy_cast = grad_y.clone(ctx)
        else:
            gy_cast = cast_tensor(grad_y, STDtype.F16, ctx, False)
        var gy = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](gy_cast.buf.unsafe_ptr().bitcast[Float16](), mo_rl)
        var wv = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float16](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
    if grad_y.dtype() == STDtype.F32:
        return d_x^
    return cast_tensor(d_x^, grad_y.dtype(), ctx)


def linear_backward_dx_slab(
    grad_y: Tensor,
    weight: Tensor,
    M: Int,
    in_features: Int,
    out_features: Int,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `linear_backward_dx` (this file :462) —
    byte-identical math (same GEMMs/flags, same clone d2d + sync, same final
    cast); ONLY the allocation source changes (contract C8, Phase P4)."""
    if grad_y.dtype() != weight.dtype() and grad_y.dtype() != STDtype.F32:
        raise Error("linear_backward_dx: grad_y/weight dtype mismatch")
    var d_x = _new_f32_slab(M, in_features, slab)

    var mo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, out_features))
    var mi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, in_features))
    var oi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_features, in_features))
    var dx = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_x.buf.unsafe_ptr().bitcast[Float32](), mi_rl)

    if weight.dtype() == STDtype.F32:
        if grad_y.dtype() != STDtype.F32:
            raise Error("linear_backward_dx: F32 weight requires F32 grad")
        var gy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float32](), mo_rl)
        var wv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float32](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
    elif weight.dtype() == STDtype.BF16:
        var gy_cast: Tensor
        if grad_y.dtype() == STDtype.BF16:
            gy_cast = _clone_to_slab(grad_y, ctx, slab)
        else:
            gy_cast = cast_tensor_slab(grad_y, STDtype.BF16, ctx, slab, False)
        var gy = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](gy_cast.buf.unsafe_ptr().bitcast[BFloat16](), mo_rl)
        var wv = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[BFloat16](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
    else:
        var gy_cast: Tensor
        if grad_y.dtype() == STDtype.F16:
            gy_cast = _clone_to_slab(grad_y, ctx, slab)
        else:
            gy_cast = cast_tensor_slab(grad_y, STDtype.F16, ctx, slab, False)
        var gy = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](gy_cast.buf.unsafe_ptr().bitcast[Float16](), mo_rl)
        var wv = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float16](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
    if grad_y.dtype() == STDtype.F32:
        return d_x^
    return cast_tensor_slab(d_x^, grad_y.dtype(), ctx, slab, False)


def linear_backward_dx_scratch(
    grad_y: Tensor,
    weight: Tensor,
    M: Int,
    in_features: Int,
    out_features: Int,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    reverse: Bool = False,
    output_dtype: STDtype = STDtype.BOOL,
) raises -> Tensor:
    """d_x-only linear backward with opt-in scratch storage for the output.

    Math and GEMM flags match `linear_backward_dx`; only the output allocation
    source changes. BF16/F16 inputs use BF16/F16 BLAS operands and an F32
    scratch accumulator. By default the public result follows `grad_y.dtype()`;
    pass `output_dtype` when the differentiated activation's storage dtype is
    different from the upstream grad workspace dtype.
    """
    if (
        not (
            weight.dtype() == STDtype.F32
            or weight.dtype() == STDtype.BF16
            or weight.dtype() == STDtype.F16
        )
    ):
        raise Error("linear_backward_dx_scratch: unsupported weight dtype")
    if grad_y.dtype() != weight.dtype() and grad_y.dtype() != STDtype.F32:
        raise Error("linear_backward_dx_scratch: grad/weight dtype mismatch")
    var out_dtype = output_dtype
    if out_dtype == STDtype.BOOL:
        out_dtype = grad_y.dtype()
    if not (
        out_dtype == STDtype.F32
        or out_dtype == STDtype.BF16
        or out_dtype == STDtype.F16
    ):
        raise Error("linear_backward_dx_scratch: unsupported output dtype")

    var d_x = _new_f32_scratch(M, in_features, scratch, reverse)

    var mo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, out_features))
    var mi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, in_features))
    var oi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_features, in_features))
    var dx = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_x.buf.unsafe_ptr().bitcast[Float32](), mi_rl)

    if weight.dtype() == STDtype.F32:
        if grad_y.dtype() != STDtype.F32:
            raise Error("linear_backward_dx_scratch: F32 weight requires F32 grad")
        var gy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float32](), mo_rl)
        var wv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float32](), oi_rl)
        matmul(ctx, dx, gy, wv, c_row_major=True)
    elif weight.dtype() == STDtype.BF16:
        var gy_cast: Tensor
        if grad_y.dtype() == STDtype.BF16:
            gy_cast = grad_y.clone(ctx)
        else:
            gy_cast = cast_tensor(grad_y, STDtype.BF16, ctx, False)
        var gy16 = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            gy_cast.buf.unsafe_ptr().bitcast[BFloat16](), mo_rl)
        var wv = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[BFloat16](), oi_rl)
        matmul(ctx, dx, gy16, wv, c_row_major=True)
    else:
        var gy_cast: Tensor
        if grad_y.dtype() == STDtype.F16:
            gy_cast = grad_y.clone(ctx)
        else:
            gy_cast = cast_tensor(grad_y, STDtype.F16, ctx, False)
        var gy16 = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            gy_cast.buf.unsafe_ptr().bitcast[Float16](), mo_rl)
        var wv = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](weight.buf.unsafe_ptr().bitcast[Float16](), oi_rl)
        matmul(ctx, dx, gy16, wv, c_row_major=True)
    if out_dtype == STDtype.F32:
        return d_x^
    return _cast_f32_to_storage_scratch(d_x^, out_dtype, scratch, reverse, ctx)


def linear_backward_dx_split_scratch(
    grad_y0: Tensor,
    grad_y1: Tensor,
    weight: Tensor,
    M: Int,
    in_features: Int,
    out0_features: Int,
    out1_features: Int,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    reverse: Bool = False,
    output_dtype: STDtype = STDtype.BOOL,
) raises -> Tensor:
    """d_x-only linear backward from two contiguous output-row grad blocks.

    For weight [out0+out1, in], computes:
      d_x = grad_y0 @ weight[0:out0, :] + grad_y1 @ weight[out0:, :]

    This avoids materializing concat(grad_y0, grad_y1) for row-split projections.
    By default the public result follows `grad_y0.dtype()`; pass `output_dtype`
    when the differentiated activation's storage dtype is different from the
    upstream grad workspace dtype.
    """
    if grad_y0.dtype() != grad_y1.dtype():
        raise Error("linear_backward_dx_split_scratch: grad dtype mismatch")
    if not (
        weight.dtype() == STDtype.F32
        or weight.dtype() == STDtype.BF16
        or weight.dtype() == STDtype.F16
    ):
        raise Error("linear_backward_dx_split_scratch: unsupported weight dtype")
    if grad_y0.dtype() != weight.dtype() and grad_y0.dtype() != STDtype.F32:
        raise Error("linear_backward_dx_split_scratch: grad/weight dtype mismatch")
    var out_dtype = output_dtype
    if out_dtype == STDtype.BOOL:
        out_dtype = grad_y0.dtype()
    if not (
        out_dtype == STDtype.F32
        or out_dtype == STDtype.BF16
        or out_dtype == STDtype.F16
    ):
        raise Error("linear_backward_dx_split_scratch: unsupported output dtype")

    var d_x = _new_f32_scratch(M, in_features, scratch, reverse)

    var mo0_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, out0_features))
    var mo1_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, out1_features))
    var mi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, in_features))
    var w0_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out0_features, in_features))
    var w1_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out1_features, in_features))
    var dx = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        d_x.buf.unsafe_ptr().bitcast[Float32](), mi_rl
    )

    if weight.dtype() == STDtype.F32:
        if grad_y0.dtype() != STDtype.F32:
            raise Error("linear_backward_dx_split_scratch: F32 weight requires F32 grads")
        var gy0 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_y0.buf.unsafe_ptr().bitcast[Float32](), mo0_rl
        )
        var gy1 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            grad_y1.buf.unsafe_ptr().bitcast[Float32](), mo1_rl
        )
        var wptr = weight.buf.unsafe_ptr().bitcast[Float32]()
        var w0 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](wptr, w0_rl)
        var w1 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            wptr + out0_features * in_features, w1_rl
        )
        matmul(ctx, dx, gy0, w0, c_row_major=True)
        matmul(ctx, dx, gy1, w1, c_row_major=True, beta=1.0)
    elif weight.dtype() == STDtype.BF16:
        var gy0_cast: Tensor
        var gy1_cast: Tensor
        if grad_y0.dtype() == STDtype.BF16:
            gy0_cast = grad_y0.clone(ctx)
            gy1_cast = grad_y1.clone(ctx)
        else:
            gy0_cast = cast_tensor(grad_y0, STDtype.BF16, ctx, False)
            gy1_cast = cast_tensor(grad_y1, STDtype.BF16, ctx, False)
        var gy0_16 = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            gy0_cast.buf.unsafe_ptr().bitcast[BFloat16](), mo0_rl
        )
        var gy1_16 = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            gy1_cast.buf.unsafe_ptr().bitcast[BFloat16](), mo1_rl
        )
        var wptr = weight.buf.unsafe_ptr().bitcast[BFloat16]()
        var w0 = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](wptr, w0_rl)
        var w1 = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            wptr + out0_features * in_features, w1_rl
        )
        matmul(ctx, dx, gy0_16, w0, c_row_major=True)
        matmul(ctx, dx, gy1_16, w1, c_row_major=True, beta=1.0)
    else:
        var gy0_cast: Tensor
        var gy1_cast: Tensor
        if grad_y0.dtype() == STDtype.F16:
            gy0_cast = grad_y0.clone(ctx)
            gy1_cast = grad_y1.clone(ctx)
        else:
            gy0_cast = cast_tensor(grad_y0, STDtype.F16, ctx, False)
            gy1_cast = cast_tensor(grad_y1, STDtype.F16, ctx, False)
        var gy0_16 = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            gy0_cast.buf.unsafe_ptr().bitcast[Float16](), mo0_rl
        )
        var gy1_16 = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            gy1_cast.buf.unsafe_ptr().bitcast[Float16](), mo1_rl
        )
        var wptr = weight.buf.unsafe_ptr().bitcast[Float16]()
        var w0 = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](wptr, w0_rl)
        var w1 = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            wptr + out0_features * in_features, w1_rl
        )
        matmul(ctx, dx, gy0_16, w0, c_row_major=True)
        matmul(ctx, dx, gy1_16, w1, c_row_major=True, beta=1.0)
    if out_dtype == STDtype.F32:
        return d_x^
    return _cast_f32_to_storage_scratch(d_x^, out_dtype, scratch, reverse, ctx)


# ── linear backward, d_W ONLY (LoRA trainable-weight path) ───────────────────
#   d_W[out,in] = grad_y[M,out]ᵀ @ x[M,in]. This sibling is useful for LoRA
#   adapters where d_A/d_B are needed but the bias gradient is not.
# Default output storage follows x.dtype(); `output_dtype` is explicit for
# trainable weight-grad storage such as OneTrainer LoRA F32 adapter grads.
def linear_backward_dw(
    grad_y: Tensor,
    x: Tensor,
    M: Int,
    in_features: Int,
    out_features: Int,
    ctx: DeviceContext,
    output_dtype: STDtype = STDtype.BOOL,
) raises -> Tensor:
    if grad_y.dtype() != x.dtype():
        raise Error("linear_backward_dw: grad_y/x dtype mismatch")
    var d_w = _new_f32(out_features, in_features, ctx)

    var mo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, out_features))
    var mi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, in_features))
    var oi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_features, in_features))
    var dw = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_w.buf.unsafe_ptr().bitcast[Float32](), oi_rl)

    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var gy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float32](), mo_rl)
        var xv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), mi_rl)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    elif dt == DType.bfloat16:
        var gy = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[BFloat16](), mo_rl)
        var xv = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), mi_rl)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    else:
        var gy = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float16](), mo_rl)
        var xv = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float16](), mi_rl)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    var out_dtype = output_dtype
    if out_dtype == STDtype.BOOL:
        out_dtype = x.dtype()
    if not (
        out_dtype == STDtype.F32
        or out_dtype == STDtype.BF16
        or out_dtype == STDtype.F16
    ):
        raise Error("linear_backward_dw: unsupported output dtype")
    if out_dtype == STDtype.F32:
        return d_w^
    return cast_tensor(d_w^, out_dtype, ctx)


def linear_backward_dw_slab(
    grad_y: Tensor,
    x: Tensor,
    M: Int,
    in_features: Int,
    out_features: Int,
    ctx: DeviceContext,
    mut slab: StepSlab,
    output_dtype: STDtype = STDtype.BOOL,
) raises -> Tensor:
    """StepSlab variant of `linear_backward_dw` (this file :709) —
    byte-identical math (same GEMM/flags, same final cast); ONLY the
    allocation source changes (contract C8, Phase P4)."""
    if grad_y.dtype() != x.dtype():
        raise Error("linear_backward_dw: grad_y/x dtype mismatch")
    var d_w = _new_f32_slab(out_features, in_features, slab)

    var mo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, out_features))
    var mi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, in_features))
    var oi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_features, in_features))
    var dw = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](d_w.buf.unsafe_ptr().bitcast[Float32](), oi_rl)

    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var gy = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float32](), mo_rl)
        var xv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), mi_rl)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    elif dt == DType.bfloat16:
        var gy = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[BFloat16](), mo_rl)
        var xv = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), mi_rl)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    else:
        var gy = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](grad_y.buf.unsafe_ptr().bitcast[Float16](), mo_rl)
        var xv = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float16](), mi_rl)
        matmul(ctx, dw, gy, xv, transpose_a=True, c_row_major=True)
    var out_dtype = output_dtype
    if out_dtype == STDtype.BOOL:
        out_dtype = x.dtype()
    if not (
        out_dtype == STDtype.F32
        or out_dtype == STDtype.BF16
        or out_dtype == STDtype.F16
    ):
        raise Error("linear_backward_dw: unsupported output dtype")
    if out_dtype == STDtype.F32:
        return d_w^
    return cast_tensor_slab(d_w^, out_dtype, ctx, slab, False)


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
    var d_x = _new_storage(M, out_features, grad_y.dtype(), ctx)
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = grad_y.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var src = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_y.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        var dst = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            d_x.buf.unsafe_ptr().bitcast[Float32](), src_rl)
        ctx.enqueue_function[
            _copy_kernel[DType.float32], _copy_kernel[DType.float32]
        ](src, dst, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var src = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_y.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        var dst = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            d_x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl)
        ctx.enqueue_function[
            _copy_kernel[DType.bfloat16], _copy_kernel[DType.bfloat16]
        ](src, dst, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var src = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_y.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        var dst = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            d_x.buf.unsafe_ptr().bitcast[Float16](), src_rl)
        ctx.enqueue_function[
            _copy_kernel[DType.float16], _copy_kernel[DType.float16]
        ](src, dst, n, grid_dim=grid, block_dim=_BLOCK)
    return AddBiasGrads(d_x^, d_b^)
