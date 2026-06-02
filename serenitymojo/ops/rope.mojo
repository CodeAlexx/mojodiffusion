# ops/rope.mojo — rotary position embedding, two layouts.
#
# RoPE rotates pairs of channels by a position-dependent angle. cos/sin carry
# the precomputed cos(theta) / sin(theta) per pair, shape [rows, D/2] (one angle
# per pair, per row/position). x is [rows, D]. Two pairing conventions:
#
#   INTERLEAVED (FLUX / Klein): pair = (x[2i], x[2i+1]), angle index i.
#     out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
#     out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
#
#   HALFSPLIT (Z-Image): pair = (x[i], x[i + D/2]), angle index i in [0, D/2).
#     out[i]       = x[i]*cos[i] - x[i+D/2]*sin[i]
#     out[i+D/2]   = x[i+D/2]*cos[i] + x[i]*sin[i]
#
# One thread per PAIR (rows * D/2 threads). cos/sin are [rows, D/2]; row index
# is shared between the data tensor and the freq tensor (per-position freqs).
# F32 math; store casts back to storage dtype. `apply_rope` SDK op is the
# TileTensor+closure variant (UNCALLABLE per OP-CALLABILITY MAP) — hand-rolled.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256


# ── interleaved kernels ────────────────────────────────────────────────────
def _rope_interleaved_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,  # D/2
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[DType.float32]](x[r, 2 * i])
        var x1 = rebind[Scalar[DType.float32]](x[r, 2 * i + 1])
        var cv = rebind[Scalar[DType.float32]](cos[r, i])
        var sv = rebind[Scalar[DType.float32]](sin[r, i])
        o[r, 2 * i] = rebind[o.element_type](x0 * cv - x1 * sv)
        o[r, 2 * i + 1] = rebind[o.element_type](x0 * sv + x1 * cv)


def _rope_interleaved_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[DType.bfloat16]](x[r, 2 * i]).cast[DType.float32]()
        var x1 = rebind[Scalar[DType.bfloat16]](x[r, 2 * i + 1]).cast[DType.float32]()
        var cv = rebind[Scalar[DType.bfloat16]](cos[r, i]).cast[DType.float32]()
        var sv = rebind[Scalar[DType.bfloat16]](sin[r, i]).cast[DType.float32]()
        o[r, 2 * i] = rebind[o.element_type]((x0 * cv - x1 * sv).cast[DType.bfloat16]())
        o[r, 2 * i + 1] = rebind[o.element_type]((x0 * sv + x1 * cv).cast[DType.bfloat16]())


def _rope_interleaved_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[DType.float16]](x[r, 2 * i]).cast[DType.float32]()
        var x1 = rebind[Scalar[DType.float16]](x[r, 2 * i + 1]).cast[DType.float32]()
        var cv = rebind[Scalar[DType.float16]](cos[r, i]).cast[DType.float32]()
        var sv = rebind[Scalar[DType.float16]](sin[r, i]).cast[DType.float32]()
        o[r, 2 * i] = rebind[o.element_type]((x0 * cv - x1 * sv).cast[DType.float16]())
        o[r, 2 * i + 1] = rebind[o.element_type]((x0 * sv + x1 * cv).cast[DType.float16]())


# ── halfsplit kernels ──────────────────────────────────────────────────────
def _rope_halfsplit_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[DType.float32]](x[r, i])
        var x1 = rebind[Scalar[DType.float32]](x[r, i + half])
        var cv = rebind[Scalar[DType.float32]](cos[r, i])
        var sv = rebind[Scalar[DType.float32]](sin[r, i])
        o[r, i] = rebind[o.element_type](x0 * cv - x1 * sv)
        o[r, i + half] = rebind[o.element_type](x1 * cv + x0 * sv)


def _rope_halfsplit_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[DType.bfloat16]](x[r, i]).cast[DType.float32]()
        var x1 = rebind[Scalar[DType.bfloat16]](x[r, i + half]).cast[DType.float32]()
        var cv = rebind[Scalar[DType.bfloat16]](cos[r, i]).cast[DType.float32]()
        var sv = rebind[Scalar[DType.bfloat16]](sin[r, i]).cast[DType.float32]()
        o[r, i] = rebind[o.element_type]((x0 * cv - x1 * sv).cast[DType.bfloat16]())
        o[r, i + half] = rebind[o.element_type]((x1 * cv + x0 * sv).cast[DType.bfloat16]())


def _rope_halfsplit_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[DType.float16]](x[r, i]).cast[DType.float32]()
        var x1 = rebind[Scalar[DType.float16]](x[r, i + half]).cast[DType.float32]()
        var cv = rebind[Scalar[DType.float16]](cos[r, i]).cast[DType.float32]()
        var sv = rebind[Scalar[DType.float16]](sin[r, i]).cast[DType.float32]()
        o[r, i] = rebind[o.element_type]((x0 * cv - x1 * sv).cast[DType.float16]())
        o[r, i + half] = rebind[o.element_type]((x1 * cv + x0 * sv).cast[DType.float16]())


def _rope_halfsplit_full_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[DType.float32]](x[r, i])
        var x1 = rebind[Scalar[DType.float32]](x[r, i + half])
        var cv0 = rebind[Scalar[DType.float32]](cos[r, i])
        var sv0 = rebind[Scalar[DType.float32]](sin[r, i])
        var cv1 = rebind[Scalar[DType.float32]](cos[r, i + half])
        var sv1 = rebind[Scalar[DType.float32]](sin[r, i + half])
        o[r, i] = rebind[o.element_type](x0 * cv0 - x1 * sv0)
        o[r, i + half] = rebind[o.element_type](x1 * cv1 + x0 * sv1)


def _rope_halfsplit_full_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[DType.bfloat16]](x[r, i]).cast[DType.float32]()
        var x1 = rebind[Scalar[DType.bfloat16]](x[r, i + half]).cast[DType.float32]()
        var cv0 = rebind[Scalar[DType.bfloat16]](cos[r, i]).cast[DType.float32]()
        var sv0 = rebind[Scalar[DType.bfloat16]](sin[r, i]).cast[DType.float32]()
        var cv1 = rebind[Scalar[DType.bfloat16]](cos[r, i + half]).cast[DType.float32]()
        var sv1 = rebind[Scalar[DType.bfloat16]](sin[r, i + half]).cast[DType.float32]()
        o[r, i] = rebind[o.element_type]((x0 * cv0 - x1 * sv0).cast[DType.bfloat16]())
        o[r, i + half] = rebind[o.element_type]((x1 * cv1 + x0 * sv1).cast[DType.bfloat16]())


def _rope_halfsplit_full_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[DType.float16]](x[r, i]).cast[DType.float32]()
        var x1 = rebind[Scalar[DType.float16]](x[r, i + half]).cast[DType.float32]()
        var cv0 = rebind[Scalar[DType.float16]](cos[r, i]).cast[DType.float32]()
        var sv0 = rebind[Scalar[DType.float16]](sin[r, i]).cast[DType.float32]()
        var cv1 = rebind[Scalar[DType.float16]](cos[r, i + half]).cast[DType.float32]()
        var sv1 = rebind[Scalar[DType.float16]](sin[r, i + half]).cast[DType.float32]()
        o[r, i] = rebind[o.element_type]((x0 * cv0 - x1 * sv0).cast[DType.float16]())
        o[r, i + half] = rebind[o.element_type]((x1 * cv1 + x0 * sv1).cast[DType.float16]())


def _rope_common_validate(
    x: Tensor, cos: Tensor, sin: Tensor
) raises -> List[Int]:
    """Shared shape checks. Returns [rows, half=D/2]."""
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("rope: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    if d % 2 != 0:
        raise Error("rope: last dim D must be even")
    var half = d // 2
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    # cos/sin must total rows*half elements (we treat them flat as [rows, half]).
    var cnum = cos.numel()
    var snum = sin.numel()
    if cnum != rows * half:
        raise Error("rope: cos numel must equal rows*(D/2)")
    if snum != rows * half:
        raise Error("rope: sin numel must equal rows*(D/2)")
    if x.dtype() != cos.dtype() or x.dtype() != sin.dtype():
        raise Error("rope: x/cos/sin dtype mismatch")
    var out = List[Int]()
    out.append(rows)
    out.append(half)
    return out^


def _rope_full_validate(x: Tensor, cos: Tensor, sin: Tensor) raises -> List[Int]:
    """Validate Qwen2.5-VL multimodal RoPE tables. Returns [rows, half].

    Unlike standard half-split RoPE, Qwen2.5-VL's multimodal helper builds a
    full-width cos/sin table after selecting temporal/height/width sections.
    The two halves can carry different axes, so the kernel must read both
    cos[i] and cos[i + D/2] instead of reusing one half-table.
    """
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("rope_halfsplit_full: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    if d % 2 != 0:
        raise Error("rope_halfsplit_full: last dim D must be even")
    var half = d // 2
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    if cos.numel() != rows * d:
        raise Error("rope_halfsplit_full: cos numel must equal rows*D")
    if sin.numel() != rows * d:
        raise Error("rope_halfsplit_full: sin numel must equal rows*D")
    if x.dtype() != cos.dtype() or x.dtype() != sin.dtype():
        raise Error("rope_halfsplit_full: x/cos/sin dtype mismatch")
    var out = List[Int]()
    out.append(rows)
    out.append(half)
    return out^


def rope_interleaved(
    x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """RoPE, interleaved pairing (FLUX/Klein).

    x:   [..., D]        (D even; leading dims flattened to rows)
    cos: [rows, D/2]     (one cos per pair per row; same dtype as x)
    sin: [rows, D/2]
    returns [..., D]     (x's dtype; F32 math).
    """
    var dims = _rope_common_validate(x, cos, sin)
    var rows = dims[0]
    var half = dims[1]
    var d = half * 2
    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            cos.buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            sin.buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[
            _rope_interleaved_kernel_f32, _rope_interleaved_kernel_f32
        ](X, C, S, O, rows, half, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            cos.buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            sin.buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[
            _rope_interleaved_kernel_bf16, _rope_interleaved_kernel_bf16
        ](X, C, S, O, rows, half, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            cos.buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            sin.buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[
            _rope_interleaved_kernel_f16, _rope_interleaved_kernel_f16
        ](X, C, S, O, rows, half, grid_dim=grid, block_dim=_BLOCK)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


def rope_halfsplit(
    x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """RoPE, half-split pairing (Z-Image).

    x:   [..., D]        (D even; leading dims flattened to rows)
    cos: [rows, D/2]     (one cos per pair per row; same dtype as x)
    sin: [rows, D/2]
    returns [..., D]     (x's dtype; F32 math).
    """
    var dims = _rope_common_validate(x, cos, sin)
    var rows = dims[0]
    var half = dims[1]
    var d = half * 2
    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            cos.buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            sin.buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[
            _rope_halfsplit_kernel_f32, _rope_halfsplit_kernel_f32
        ](X, C, S, O, rows, half, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            cos.buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            sin.buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[
            _rope_halfsplit_kernel_bf16, _rope_halfsplit_kernel_bf16
        ](X, C, S, O, rows, half, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            cos.buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            sin.buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[
            _rope_halfsplit_kernel_f16, _rope_halfsplit_kernel_f16
        ](X, C, S, O, rows, half, grid_dim=grid, block_dim=_BLOCK)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


def rope_halfsplit_full(
    x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """RoPE, half-split pairing with full-width Qwen2.5-VL cos/sin tables.

    x:   [..., D]        (D even; leading dims flattened to rows)
    cos: [rows, D]       (full table after mRoPE section selection)
    sin: [rows, D]
    returns [..., D]
    """
    var dims = _rope_full_validate(x, cos, sin)
    var rows = dims[0]
    var half = dims[1]
    var d = half * 2
    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            cos.buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            sin.buf.unsafe_ptr().bitcast[Float32](), f_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[
            _rope_halfsplit_full_kernel_f32, _rope_halfsplit_full_kernel_f32
        ](X, C, S, O, rows, half, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            cos.buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            sin.buf.unsafe_ptr().bitcast[BFloat16](), f_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[
            _rope_halfsplit_full_kernel_bf16, _rope_halfsplit_full_kernel_bf16
        ](X, C, S, O, rows, half, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            cos.buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            sin.buf.unsafe_ptr().bitcast[Float16](), f_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[
            _rope_halfsplit_full_kernel_f16, _rope_halfsplit_full_kernel_f16
        ](X, C, S, O, rows, half, grid_dim=grid, block_dim=_BLOCK)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())
