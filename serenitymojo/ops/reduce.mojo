# ops/reduce.mojo — multi-axis reduction primitive (sum / mean / var / std).
#
# LTX2_PORT_PLAN_2026-05-28 §P-reduce: reduce_sum / reduce_mean over an
# arbitrary set of dims, F32-accumulated, with keepdim. Unblocks AdaIN
# (per-(B,C) over (F,H,W)) and general PixelNorm.
#
# Rust mirror:
#   sampling/ltx2_multiscale.rs:75-92  mean_dim(&[2,3,4], true) + unbiased var
#   vae/ltx2_vae.rs:209-215            pixel_norm: mean_along_dims(&[1], true)
#
# Semantics (match torch / candle):
#   reduce_sum(x, dims)   = Σ over the listed axes              (F32 accumulate)
#   reduce_mean(x, dims)  = reduce_sum / N      , N = Π size of reduced axes
#   reduce_var(x, dims, unbiased=True) = Σ(x-μ)² / (N-1)   (torch default N-1)
#   reduce_std(x, dims, unbiased=True) = sqrt(var)
# keepdim=True  -> reduced axes become size-1 (shape rank preserved)
# keepdim=False -> reduced axes are dropped.
#
# Kernel design (KISS, correctness-first): one GPU thread per OUTPUT element.
# Each thread decomposes its output index into the kept-dim multi-index, forms
# the input base offset, then loops over the reduced subspace accumulating in
# F32. Arbitrary (possibly non-contiguous) axis sets are handled by uploading
# the input row-major strides + a per-dim reduce mask + the reduced-subspace
# shape/strides as small Int32 device buffers. F32 accumulation regardless of
# storage dtype (bf16/f16 upcast on read; cast back on store for sum/mean;
# var/std always emit F32 since the consumers — AdaIN std, PixelNorm rms — want
# F32 stats).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _MAXDIM = 8  # max tensor rank we support


# ── metadata structs uploaded to the device ────────────────────────────────
# We pass everything the kernel needs as flat Int32 LayoutTensors:
#   in_strides[ndim]   — element strides of the input (row-major)
#   kept_shape[n_kept] — sizes of the kept dims (output decomposition order)
#   kept_strides[n_kept] — input strides of the kept dims
#   red_shape[n_red]   — sizes of the reduced dims
#   red_strides[n_red] — input strides of the reduced dims
# The kernel takes ndim-agnostic small arrays + counts.


def _reduce_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    kept_shape: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    kept_strides: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    red_shape: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    red_strides: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    n_out: Int,
    n_kept: Int,
    n_red: Int,
    n_reduce_elems: Int,
    mode: Int,  # 0=sum, 1=mean, 2=var(unbiased), 3=std(unbiased)
):
    var oid = Int(global_idx.x)
    if oid >= n_out:
        return
    # Decompose output linear index over kept dims (row-major) -> input base.
    var base = 0
    var rem = oid
    # iterate kept dims from last to first to peel off row-major coords.
    for k in range(n_kept - 1, -1, -1):
        var sz = Int(rebind[Scalar[DType.int32]](kept_shape[k]))
        var coord = rem % sz
        rem = rem // sz
        base += coord * Int(rebind[Scalar[DType.int32]](kept_strides[k]))

    # Accumulate over the reduced subspace in F32.
    var acc = Float32(0.0)
    var acc_sq = Float32(0.0)
    for r in range(n_reduce_elems):
        # decompose r over reduced dims (row-major) -> input offset delta.
        var off = base
        var rr = r
        for d in range(n_red - 1, -1, -1):
            var sz = Int(rebind[Scalar[DType.int32]](red_shape[d]))
            var coord = rr % sz
            rr = rr // sz
            off += coord * Int(rebind[Scalar[DType.int32]](red_strides[d]))
        var v = rebind[Scalar[DType.float32]](x[off])
        acc += v
        acc_sq += v * v

    var n = Float32(n_reduce_elems)
    var result = acc  # mode 0: sum
    if mode == 1:
        result = acc / n
    elif mode == 2 or mode == 3:
        # unbiased variance: (Σx² - (Σx)²/N) / (N-1)
        var mean = acc / n
        var sse = acc_sq - acc * mean  # = Σx² - (Σx)²/N
        var denom = n - Float32(1.0)
        if denom < Float32(1.0):
            denom = Float32(1.0)  # match Rust .max(1.0)
        var var_u = sse / denom
        if var_u < Float32(0.0):
            var_u = Float32(0.0)  # numeric guard
        if mode == 3:
            result = sqrt(var_u)
        else:
            result = var_u
    o[oid] = rebind[o.element_type](result)


def _reduce_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    kept_shape: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    kept_strides: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    red_shape: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    red_strides: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    n_out: Int,
    n_kept: Int,
    n_red: Int,
    n_reduce_elems: Int,
    mode: Int,
):
    var oid = Int(global_idx.x)
    if oid >= n_out:
        return
    var base = 0
    var rem = oid
    for k in range(n_kept - 1, -1, -1):
        var sz = Int(rebind[Scalar[DType.int32]](kept_shape[k]))
        var coord = rem % sz
        rem = rem // sz
        base += coord * Int(rebind[Scalar[DType.int32]](kept_strides[k]))
    var acc = Float32(0.0)
    var acc_sq = Float32(0.0)
    for r in range(n_reduce_elems):
        var off = base
        var rr = r
        for d in range(n_red - 1, -1, -1):
            var sz = Int(rebind[Scalar[DType.int32]](red_shape[d]))
            var coord = rr % sz
            rr = rr // sz
            off += coord * Int(rebind[Scalar[DType.int32]](red_strides[d]))
        var v = rebind[Scalar[DType.bfloat16]](x[off]).cast[DType.float32]()
        acc += v
        acc_sq += v * v
    var n = Float32(n_reduce_elems)
    var result = acc
    if mode == 1:
        result = acc / n
    elif mode == 2 or mode == 3:
        var mean = acc / n
        var sse = acc_sq - acc * mean
        var denom = n - Float32(1.0)
        if denom < Float32(1.0):
            denom = Float32(1.0)
        var var_u = sse / denom
        if var_u < Float32(0.0):
            var_u = Float32(0.0)
        if mode == 3:
            result = sqrt(var_u)
        else:
            result = var_u
    o[oid] = rebind[o.element_type](result)


def _reduce_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    kept_shape: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    kept_strides: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    red_shape: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    red_strides: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    n_out: Int,
    n_kept: Int,
    n_red: Int,
    n_reduce_elems: Int,
    mode: Int,
):
    var oid = Int(global_idx.x)
    if oid >= n_out:
        return
    var base = 0
    var rem = oid
    for k in range(n_kept - 1, -1, -1):
        var sz = Int(rebind[Scalar[DType.int32]](kept_shape[k]))
        var coord = rem % sz
        rem = rem // sz
        base += coord * Int(rebind[Scalar[DType.int32]](kept_strides[k]))
    var acc = Float32(0.0)
    var acc_sq = Float32(0.0)
    for r in range(n_reduce_elems):
        var off = base
        var rr = r
        for d in range(n_red - 1, -1, -1):
            var sz = Int(rebind[Scalar[DType.int32]](red_shape[d]))
            var coord = rr % sz
            rr = rr // sz
            off += coord * Int(rebind[Scalar[DType.int32]](red_strides[d]))
        var v = rebind[Scalar[DType.float16]](x[off]).cast[DType.float32]()
        acc += v
        acc_sq += v * v
    var n = Float32(n_reduce_elems)
    var result = acc
    if mode == 1:
        result = acc / n
    elif mode == 2 or mode == 3:
        var mean = acc / n
        var sse = acc_sq - acc * mean
        var denom = n - Float32(1.0)
        if denom < Float32(1.0):
            denom = Float32(1.0)
        var var_u = sse / denom
        if var_u < Float32(0.0):
            var_u = Float32(0.0)
        if mode == 3:
            result = sqrt(var_u)
        else:
            result = var_u
    o[oid] = rebind[o.element_type](result)


# ── host driver ─────────────────────────────────────────────────────────────
@always_inline
def _normalize_dims(dims: List[Int], ndim: Int) raises -> List[Int]:
    """Resolve negative axes, validate range, sort ascending, dedupe."""
    var out = List[Int]()
    for i in range(len(dims)):
        var d = dims[i]
        if d < 0:
            d += ndim
        if d < 0 or d >= ndim:
            raise Error(
                String("reduce: axis ") + String(dims[i]) + " out of range for rank "
                + String(ndim)
            )
        # insertion (sorted, dedupe)
        var dup = False
        for j in range(len(out)):
            if out[j] == d:
                dup = True
        if not dup:
            out.append(d)
    return out^


def _upload_i32(vals: List[Int], ctx: DeviceContext) raises -> DeviceBuffer[DType.uint8]:
    var n = len(vals)
    var nb = n * 4
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nb)
    var hp = host.unsafe_ptr().bitcast[Int32]()
    for i in range(n):
        hp[i] = Int32(vals[i])
    var dev = ctx.enqueue_create_buffer[DType.uint8](nb)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return dev^


def _reduce_impl(
    x: Tensor, dims: List[Int], keepdim: Bool, mode: Int, ctx: DeviceContext
) raises -> Tensor:
    var shape = x.shape()
    var ndim = len(shape)
    if ndim == 0:
        raise Error("reduce: scalar input not supported")
    if ndim > _MAXDIM:
        raise Error("reduce: rank > 8 not supported")

    var rdims = _normalize_dims(dims, ndim)
    if len(rdims) == 0:
        raise Error("reduce: empty dim list")

    # row-major element strides of the input.
    var in_strides = List[Int]()
    for _ in range(ndim):
        in_strides.append(0)
    var acc = 1
    for d in range(ndim - 1, -1, -1):
        in_strides[d] = acc
        acc *= shape[d]

    # Split dims into kept / reduced (preserving original order).
    var is_red = List[Bool]()
    for _ in range(ndim):
        is_red.append(False)
    for i in range(len(rdims)):
        is_red[rdims[i]] = True

    var kept_shape = List[Int]()
    var kept_strides = List[Int]()
    var red_shape = List[Int]()
    var red_strides = List[Int]()
    for d in range(ndim):
        if is_red[d]:
            red_shape.append(shape[d])
            red_strides.append(in_strides[d])
        else:
            kept_shape.append(shape[d])
            kept_strides.append(in_strides[d])

    var n_out = 1
    for i in range(len(kept_shape)):
        n_out *= kept_shape[i]
    var n_reduce_elems = 1
    for i in range(len(red_shape)):
        n_reduce_elems *= red_shape[i]

    # Output shape: keepdim -> reduced dims become 1; else dropped.
    var out_shape = List[Int]()
    for d in range(ndim):
        if is_red[d]:
            if keepdim:
                out_shape.append(1)
        else:
            out_shape.append(shape[d])
    if len(out_shape) == 0:
        out_shape.append(1)  # full reduction -> scalar held as [1]

    # Guard against zero-length metadata arrays (e.g. full reduction -> no kept
    # dims). Pad to length 1 so the LayoutTensor wrap is well-formed; the kernel
    # never reads them because n_kept==0.
    var n_kept = len(kept_shape)
    var n_red = len(red_shape)
    var kshape_u = kept_shape.copy()
    var kstride_u = kept_strides.copy()
    if n_kept == 0:
        kshape_u.append(1)
        kstride_u.append(0)
    var rshape_u = red_shape.copy()
    var rstride_u = red_strides.copy()
    if n_red == 0:
        rshape_u.append(1)
        rstride_u.append(0)

    var ks_buf = _upload_i32(kshape_u, ctx)
    var kst_buf = _upload_i32(kstride_u, ctx)
    var rs_buf = _upload_i32(rshape_u, ctx)
    var rst_buf = _upload_i32(rstride_u, ctx)

    var KS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        ks_buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](len(kshape_u))),
    )
    var KST = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        kst_buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](len(kstride_u))),
    )
    var RS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        rs_buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](len(rshape_u))),
    )
    var RST = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        rst_buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](len(rstride_u))),
    )

    # Output is always F32 (stats consumers want F32; sum/mean of bf16 also
    # benefit from F32 storage to avoid double-rounding).
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n_out * 4)
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[_DYN1].row_major(IndexList[1](n_out)),
    )

    var n_in = x.numel()
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](n_in))
    var grid = (n_out + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl_in
        )
        ctx.enqueue_function[_reduce_kernel_f32, _reduce_kernel_f32](
            X, O, KS, KST, RS, RST, n_out, n_kept, n_red, n_reduce_elems, mode,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl_in
        )
        ctx.enqueue_function[_reduce_kernel_bf16, _reduce_kernel_bf16](
            X, O, KS, KST, RS, RST, n_out, n_kept, n_red, n_reduce_elems, mode,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl_in
        )
        ctx.enqueue_function[_reduce_kernel_f16, _reduce_kernel_f16](
            X, O, KS, KST, RS, RST, n_out, n_kept, n_red, n_reduce_elems, mode,
            grid_dim=grid, block_dim=_BLOCK,
        )
    ctx.synchronize()
    # keep metadata buffers alive until the kernel has run.
    _ = ks_buf^
    _ = kst_buf^
    _ = rs_buf^
    _ = rst_buf^
    return Tensor(out_buf^, out_shape^, STDtype.F32)


# ── public API ──────────────────────────────────────────────────────────────
def reduce_sum(
    x: Tensor, dims: List[Int], keepdim: Bool, ctx: DeviceContext
) raises -> Tensor:
    """Σ over `dims`, F32-accumulated. Output dtype F32. (torch.sum)."""
    return _reduce_impl(x, dims, keepdim, 0, ctx)


def reduce_mean(
    x: Tensor, dims: List[Int], keepdim: Bool, ctx: DeviceContext
) raises -> Tensor:
    """mean over `dims` = sum / N, F32-accumulated. Output F32. (torch.mean)."""
    return _reduce_impl(x, dims, keepdim, 1, ctx)


def reduce_var(
    x: Tensor, dims: List[Int], keepdim: Bool, ctx: DeviceContext
) raises -> Tensor:
    """Unbiased variance Σ(x-μ)²/(N-1) over `dims` (torch default). Output F32."""
    return _reduce_impl(x, dims, keepdim, 2, ctx)


def reduce_std(
    x: Tensor, dims: List[Int], keepdim: Bool, ctx: DeviceContext
) raises -> Tensor:
    """Unbiased std = sqrt(var) over `dims` (torch default). Output F32.

    Mirrors ltx2_multiscale.rs AdaIN per-(B,C)-over-(F,H,W) std."""
    return _reduce_impl(x, dims, keepdim, 3, ctx)
