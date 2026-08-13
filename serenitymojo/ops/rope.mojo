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
# F32 math; store casts back to storage dtype. BF16/F16 activations may consume
# F32 cos/sin tables: diffusers Flux/Klein applies `x.float() * cos/sin` and then
# casts the rotated tensor back to the activation dtype. The F32 table is an
# intentional compute boundary, not an activation/storage boundary.
# `apply_rope` SDK op is the TileTensor+closure variant (UNCALLABLE per
# OP-CALLABILITY MAP) — hand-rolled.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd_v2.step_slab import StepSlab


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256


# ── interleaved kernels ────────────────────────────────────────────────────
def _rope_interleaved_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows_w: Int32,
    half_w: Int32,  # D/2
):
    var rows = Int(rows_w)
    var half = Int(half_w)
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
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
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
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
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


def _rope_interleaved_kernel_f32_tables[x_dtype: DType](
    x: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[x_dtype]](x[r, 2 * i]).cast[DType.float32]()
        var x1 = rebind[Scalar[x_dtype]](x[r, 2 * i + 1]).cast[DType.float32]()
        var cv = rebind[Scalar[DType.float32]](cos[r, i])
        var sv = rebind[Scalar[DType.float32]](sin[r, i])
        o[r, 2 * i] = rebind[o.element_type]((x0 * cv - x1 * sv).cast[x_dtype]())
        o[r, 2 * i + 1] = rebind[o.element_type]((x0 * sv + x1 * cv).cast[x_dtype]())


def _rope_interleaved_head_broadcast_kernel[
    x_dtype: DType, table_dtype: DType
](
    x: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[table_dtype, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[table_dtype, _DYN2, MutAnyOrigin],
    o: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    rows_w: Int32,
    half_w: Int32,
    heads_w: Int32,
):
    """Interleaved RoPE with one compact table row shared by all heads."""
    var rows = Int(rows_w)
    var half = Int(half_w)
    var heads = Int(heads_w)
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var token = r // heads
        var x0 = rebind[Scalar[x_dtype]](x[r, 2 * i]).cast[DType.float32]()
        var x1 = rebind[Scalar[x_dtype]](x[r, 2 * i + 1]).cast[DType.float32]()
        var cv = rebind[Scalar[table_dtype]](cos[token, i]).cast[DType.float32]()
        var sv = rebind[Scalar[table_dtype]](sin[token, i]).cast[DType.float32]()
        o[r, 2 * i] = rebind[o.element_type]((x0 * cv - x1 * sv).cast[x_dtype]())
        o[r, 2 * i + 1] = rebind[o.element_type]((x0 * sv + x1 * cv).cast[x_dtype]())


# ── halfsplit kernels ──────────────────────────────────────────────────────
def _rope_halfsplit_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
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
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
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
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
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


def _rope_halfsplit_kernel_f32_tables[x_dtype: DType](
    x: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[x_dtype]](x[r, i]).cast[DType.float32]()
        var x1 = rebind[Scalar[x_dtype]](x[r, i + half]).cast[DType.float32]()
        var cv = rebind[Scalar[DType.float32]](cos[r, i])
        var sv = rebind[Scalar[DType.float32]](sin[r, i])
        o[r, i] = rebind[o.element_type]((x0 * cv - x1 * sv).cast[x_dtype]())
        o[r, i + half] = rebind[o.element_type]((x1 * cv + x0 * sv).cast[x_dtype]())


def _rope_halfsplit_full_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
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
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
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
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
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


def _rope_halfsplit_full_head_broadcast_kernel[
    x_dtype: DType, table_dtype: DType
](
    x: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[table_dtype, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[table_dtype, _DYN2, MutAnyOrigin],
    o: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    rows_w: Int32,
    half_w: Int32,
    heads_w: Int32,
):
    """Full-width half-split RoPE with one compact table row per token."""
    var rows = Int(rows_w)
    var half = Int(half_w)
    var heads = Int(heads_w)
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var token = r // heads
        var x0 = rebind[Scalar[x_dtype]](x[r, i]).cast[DType.float32]()
        var x1 = rebind[Scalar[x_dtype]](x[r, i + half]).cast[DType.float32]()
        var cv0 = rebind[Scalar[table_dtype]](cos[token, i]).cast[DType.float32]()
        var sv0 = rebind[Scalar[table_dtype]](sin[token, i]).cast[DType.float32]()
        var cv1 = rebind[Scalar[table_dtype]](cos[token, i + half]).cast[DType.float32]()
        var sv1 = rebind[Scalar[table_dtype]](sin[token, i + half]).cast[DType.float32]()
        o[r, i] = rebind[o.element_type]((x0 * cv0 - x1 * sv0).cast[x_dtype]())
        o[r, i + half] = rebind[o.element_type]((x1 * cv1 + x0 * sv1).cast[x_dtype]())


def _rope_halfsplit_full_kernel_f32_tables[x_dtype: DType](
    x: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    cos: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[x_dtype, _DYN2, MutAnyOrigin],
    rows_w: Int32,
    half_w: Int32,
):
    var rows = Int(rows_w)
    var half = Int(half_w)
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var r = idx // half
        var i = idx % half
        var x0 = rebind[Scalar[x_dtype]](x[r, i]).cast[DType.float32]()
        var x1 = rebind[Scalar[x_dtype]](x[r, i + half]).cast[DType.float32]()
        var cv0 = rebind[Scalar[DType.float32]](cos[r, i])
        var sv0 = rebind[Scalar[DType.float32]](sin[r, i])
        var cv1 = rebind[Scalar[DType.float32]](cos[r, i + half])
        var sv1 = rebind[Scalar[DType.float32]](sin[r, i + half])
        o[r, i] = rebind[o.element_type]((x0 * cv0 - x1 * sv0).cast[x_dtype]())
        o[r, i + half] = rebind[o.element_type]((x1 * cv1 + x0 * sv1).cast[x_dtype]())


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
    var x_dt = x.dtype()
    var cos_dt = cos.dtype()
    var sin_dt = sin.dtype()
    if cos_dt != sin_dt:
        raise Error("rope: cos/sin dtype mismatch")
    if x_dt != cos_dt:
        if cos_dt != STDtype.F32 or x_dt == STDtype.F32:
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
    var x_dt = x.dtype()
    var cos_dt = cos.dtype()
    var sin_dt = sin.dtype()
    if cos_dt != sin_dt:
        raise Error("rope_halfsplit_full: cos/sin dtype mismatch")
    if x_dt != cos_dt:
        if cos_dt != STDtype.F32 or x_dt == STDtype.F32:
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
    cos: [rows, D/2]     (one cos per pair per row; same dtype as x or F32)
    sin: [rows, D/2]
    returns [..., D]     (x's dtype; F32 math).
    """
    var dims = _rope_common_validate(x, cos, sin)
    var rows = dims[0]
    var half = dims[1]
    var d = half * 2
    var dt = x.dtype().to_mojo_dtype()
    var table_dt = cos.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        ctx.enqueue_function[_rope_interleaved_kernel_f32](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_kernel_f32_tables[DType.bfloat16]](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_kernel_bf16](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_kernel_f32_tables[DType.float16]](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_kernel_f16](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


def rope_interleaved_head_broadcast(
    x: Tensor, cos: Tensor, sin: Tensor, heads: Int, ctx: DeviceContext
) raises -> Tensor:
    """Interleaved RoPE using compact `[tokens,D/2]` tables for `[tokens,heads,D]`.

    The flattened activation rows remain token-major then head-major. The GPU
    kernel maps activation row `r` to compact table row `r // heads`, avoiding
    a persistent `heads`-fold materialization of identical frequency values.
    """
    if heads <= 0:
        raise Error("rope_interleaved_head_broadcast: heads must be positive")
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("rope_interleaved_head_broadcast: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    if d % 2 != 0:
        raise Error("rope_interleaved_head_broadcast: last dim must be even")
    var half = d // 2
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    if rows % heads != 0:
        raise Error("rope_interleaved_head_broadcast: activation rows/head mismatch")
    var tokens = rows // heads
    if cos.numel() != tokens * half or sin.numel() != tokens * half:
        raise Error("rope_interleaved_head_broadcast: compact table shape mismatch")
    var x_dt = x.dtype()
    var cos_dt = cos.dtype()
    if cos_dt != sin.dtype():
        raise Error("rope_interleaved_head_broadcast: cos/sin dtype mismatch")
    if x_dt != cos_dt:
        if cos_dt != STDtype.F32 or x_dt == STDtype.F32:
            raise Error("rope_interleaved_head_broadcast: x/table dtype mismatch")

    var dt = x_dt.to_mojo_dtype()
    var table_dt = cos_dt.to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](tokens, half))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        ctx.enqueue_function[_rope_interleaved_head_broadcast_kernel[ DType.float32, DType.float32 ]](X, C, S, O, Int32(rows), Int32(half), Int32(heads), grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_head_broadcast_kernel[ DType.bfloat16, DType.float32 ]](X, C, S, O, Int32(rows), Int32(half), Int32(heads), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_head_broadcast_kernel[ DType.bfloat16, DType.bfloat16 ]](X, C, S, O, Int32(rows), Int32(half), Int32(heads), grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_head_broadcast_kernel[ DType.float16, DType.float32 ]](X, C, S, O, Int32(rows), Int32(half), Int32(heads), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_head_broadcast_kernel[ DType.float16, DType.float16 ]](X, C, S, O, Int32(rows), Int32(half), Int32(heads), grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, x.shape(), x_dt)


def rope_interleaved_slab(
    x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `rope_interleaved` (this file :372) —
    byte-identical math (same kernels, same launch params); ONLY the
    allocation source changes (autograd_v2 contract C8, Phase P4)."""
    var dims = _rope_common_validate(x, cos, sin)
    var rows = dims[0]
    var half = dims[1]
    var d = half * 2
    var dt = x.dtype().to_mojo_dtype()
    var table_dt = cos.dtype().to_mojo_dtype()
    var out_buf = slab.alloc(x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        ctx.enqueue_function[_rope_interleaved_kernel_f32](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_kernel_f32_tables[DType.bfloat16]](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_kernel_bf16](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_kernel_f32_tables[DType.float16]](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_interleaved_kernel_f16](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


def rope_halfsplit(
    x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """RoPE, half-split pairing (Z-Image).

    x:   [..., D]        (D even; leading dims flattened to rows)
    cos: [rows, D/2]     (one cos per pair per row; same dtype as x or F32)
    sin: [rows, D/2]
    returns [..., D]     (x's dtype; F32 math).
    """
    var dims = _rope_common_validate(x, cos, sin)
    var rows = dims[0]
    var half = dims[1]
    var d = half * 2
    var dt = x.dtype().to_mojo_dtype()
    var table_dt = cos.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        ctx.enqueue_function[_rope_halfsplit_kernel_f32](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_kernel_f32_tables[DType.bfloat16]](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_kernel_bf16](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_kernel_f32_tables[DType.float16]](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_kernel_f16](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


def rope_halfsplit_slab(
    x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `rope_halfsplit` (this file :566) — byte-identical
    math (same kernels, same launch params); ONLY the allocation source
    changes (autograd_v2 contract C8, Phase P4)."""
    var dims = _rope_common_validate(x, cos, sin)
    var rows = dims[0]
    var half = dims[1]
    var d = half * 2
    var dt = x.dtype().to_mojo_dtype()
    var table_dt = cos.dtype().to_mojo_dtype()
    var out_buf = slab.alloc(x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        ctx.enqueue_function[_rope_halfsplit_kernel_f32](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_kernel_f32_tables[DType.bfloat16]](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_kernel_bf16](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_kernel_f32_tables[DType.float16]](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_kernel_f16](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


def rope_halfsplit_full(
    x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """RoPE, half-split pairing with full-width Qwen2.5-VL cos/sin tables.

    x:   [..., D]        (D even; leading dims flattened to rows)
    cos: [rows, D]       (full table after mRoPE section selection; same dtype as x or F32)
    sin: [rows, D]
    returns [..., D]
    """
    var dims = _rope_full_validate(x, cos, sin)
    var rows = dims[0]
    var half = dims[1]
    var d = half * 2
    var dt = x.dtype().to_mojo_dtype()
    var table_dt = cos.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        ctx.enqueue_function[_rope_halfsplit_full_kernel_f32](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_full_kernel_f32_tables[DType.bfloat16]](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_full_kernel_bf16](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        if table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_full_kernel_f32_tables[DType.float16]](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_full_kernel_f16](X, C, S, O, Int32(rows), Int32(half), grid_dim=grid, block_dim=_BLOCK)
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, x.shape(), x.dtype())


def rope_halfsplit_full_head_broadcast(
    x: Tensor, cos: Tensor, sin: Tensor, heads: Int, ctx: DeviceContext
) raises -> Tensor:
    """Full-width half-split RoPE with compact per-token frequency tables.

    `x` is token-major `[...tokens, heads, D]`; `cos` and `sin` are compact
    `[tokens, D]`. The kernel maps flattened activation row `r` to table row
    `r // heads`, avoiding a heads-fold materialization. Arithmetic and BF16
    rounding are identical to `rope_halfsplit_full` on explicitly repeated
    tables.
    """
    if heads <= 0:
        raise Error("rope_halfsplit_full_head_broadcast: heads must be positive")
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("rope_halfsplit_full_head_broadcast: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    if d % 2 != 0:
        raise Error("rope_halfsplit_full_head_broadcast: last dim must be even")
    var rows = x.numel() // d
    if rows % heads != 0:
        raise Error("rope_halfsplit_full_head_broadcast: row/head mismatch")
    var tokens = rows // heads
    if cos.numel() != tokens * d or sin.numel() != tokens * d:
        raise Error("rope_halfsplit_full_head_broadcast: compact table mismatch")
    if cos.dtype() != sin.dtype():
        raise Error("rope_halfsplit_full_head_broadcast: table dtype mismatch")
    var x_dt = x.dtype()
    var table_dt = cos.dtype()
    if x_dt != table_dt:
        if table_dt != STDtype.F32 or x_dt == STDtype.F32:
            raise Error("rope_halfsplit_full_head_broadcast: x/table dtype mismatch")

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](tokens, d))
    var out_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var total = rows * (d // 2)
    var grid = (total + _BLOCK - 1) // _BLOCK
    var mojo_x_dt = x_dt.to_mojo_dtype()
    var mojo_table_dt = table_dt.to_mojo_dtype()

    if mojo_x_dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=out_rl,
    )
        ctx.enqueue_function[_rope_halfsplit_full_head_broadcast_kernel[ DType.float32, DType.float32 ]](X, C, S, O, Int32(rows), Int32(d // 2), Int32(heads), grid_dim=grid, block_dim=_BLOCK)
    elif mojo_x_dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=out_rl,
    )
        if mojo_table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_full_head_broadcast_kernel[ DType.bfloat16, DType.float32 ]](X, C, S, O, Int32(rows), Int32(d // 2), Int32(heads), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_full_head_broadcast_kernel[ DType.bfloat16, DType.bfloat16 ]](X, C, S, O, Int32(rows), Int32(d // 2), Int32(heads), grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=x_rl,
    )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=out_rl,
    )
        if mojo_table_dt == DType.float32:
            var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_full_head_broadcast_kernel[ DType.float16, DType.float32 ]](X, C, S, O, Int32(rows), Int32(d // 2), Int32(heads), grid_dim=grid, block_dim=_BLOCK)
        else:
            var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=f_rl,
    )
            ctx.enqueue_function[_rope_halfsplit_full_head_broadcast_kernel[ DType.float16, DType.float16 ]](X, C, S, O, Int32(rows), Int32(d // 2), Int32(heads), grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, x.shape(), x.dtype())
