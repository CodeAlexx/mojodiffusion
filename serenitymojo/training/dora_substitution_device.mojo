# training/dora_substitution_device.mojo -- GPU direct DoRA W_eff substitution.
#
# This is the device-resident counterpart to dora_substitution_forward/backward
# in dora_adapter.mojo. It computes y = x @ W_eff^T directly from
# W_orig + DoRA(A,B,m) without materializing a dense LoRA carrier or a full
# W_eff tensor. Denominators are detached exactly like the host reference.
#
# First production slice: A/B/magnitude storage is BF16, den/grads are F32, and
# x/d_y/d_x preserve their storage dtype. F32 is used only inside compute.

from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.training.dora_adapter import DoRAAdapter


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime TArc = ArcPointer[Tensor]


struct DoRAAdapterDevice(Copyable, Movable):
    var a: TArc
    var b: TArc
    var m: TArc
    var rank: Int
    var in_f: Int
    var out_f: Int
    var alpha: Float32
    var scale: Float32
    var eps: Float32
    var wd_on_out: Bool
    var delta_zero: Bool

    def __init__(
        out self,
        var a: TArc, var b: TArc, var m: TArc,
        rank: Int, in_f: Int, out_f: Int, alpha: Float32,
        scale: Float32, eps: Float32, wd_on_out: Bool,
        delta_zero: Bool,
    ):
        self.a = a^
        self.b = b^
        self.m = m^
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f
        self.alpha = alpha
        self.scale = scale
        self.eps = eps
        self.wd_on_out = wd_on_out
        self.delta_zero = delta_zero


struct DoRADeviceGrads(Movable):
    var d_a: Tensor
    var d_b: Tensor
    var d_m: Tensor
    var d_x: Tensor

    def __init__(
        out self, var d_a: Tensor, var d_b: Tensor,
        var d_m: Tensor, var d_x: Tensor,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_m = d_m^
        self.d_x = d_x^


def _bf16_to_f32_list(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


def _bf16_all_zero(v: List[BFloat16]) -> Bool:
    for i in range(len(v)):
        if v[i].cast[DType.float32]() != Float32(0.0):
            return False
    return True


def dora_device_from_host(d: DoRAAdapter, ctx: DeviceContext) raises -> DoRAAdapterDevice:
    var a = Tensor.from_host(_bf16_to_f32_list(d.a), [d.rank, d.in_f], STDtype.BF16, ctx)
    var b = Tensor.from_host(_bf16_to_f32_list(d.b), [d.out_f, d.rank], STDtype.BF16, ctx)
    var mlen = d.out_f if d.wd_on_out else d.in_f
    var m = Tensor.from_host(d.m.copy(), [mlen], STDtype.BF16, ctx)
    return DoRAAdapterDevice(
        TArc(a^), TArc(b^), TArc(m^), d.rank, d.in_f, d.out_f,
        d.alpha, d.scale, d.eps, d.wd_on_out, _bf16_all_zero(d.b),
    )


def _m_f32_for_compute(d: DoRAAdapterDevice, ctx: DeviceContext) raises -> Tensor:
    # Resident DoRA magnitude stays BF16; kernels consume a transient F32 view.
    return cast_tensor(d.m[], STDtype.F32, ctx, False)


@always_inline
def _delta_at_bf16(
    a: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    rank: Int, in_f: Int, o: Int, i: Int, scale: Float32,
) -> Float32:
    var acc = Float32(0.0)
    for r in range(rank):
        var bv = rebind[Scalar[DType.bfloat16]](b[o, r]).cast[DType.float32]()
        var av = rebind[Scalar[DType.bfloat16]](a[r, i]).cast[DType.float32]()
        acc += bv * av
    return acc * scale


def _den_kernel[wdt: DType](
    w: LayoutTensor[wdt, _DYN2, MutAnyOrigin],
    a: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rank: Int, in_f: Int, out_f: Int,
    scale: Float32, eps: Float32, wd_on_out: Int, mlen: Int,
    delta_zero: Int,
):
    var idx = Int(global_idx.x)
    if idx < mlen:
        var ss = Float32(0.0)
        if wd_on_out != 0:
            var o = idx
            for i in range(in_f):
                var wv = rebind[Scalar[wdt]](w[o, i]).cast[DType.float32]()
                var wp = wv
                if delta_zero == 0:
                    wp += _delta_at_bf16(a, b, rank, in_f, o, i, scale)
                ss += wp * wp
        else:
            var i = idx
            for o in range(out_f):
                var wv = rebind[Scalar[wdt]](w[o, i]).cast[DType.float32]()
                var wp = wv
                if delta_zero == 0:
                    wp += _delta_at_bf16(a, b, rank, in_f, o, i, scale)
                ss += wp * wp
        den[0, idx] = rebind[den.element_type](sqrt(ss) + eps)


def _den_from_m_kernel(
    mag: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    mlen: Int,
    eps: Float32,
):
    var idx = Int(global_idx.x)
    if idx < mlen:
        den[0, idx] = rebind[den.element_type](
            rebind[Scalar[DType.float32]](mag[0, idx]) + eps
        )


def _forward_kernel[xdt: DType, wdt: DType](
    x: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    w: LayoutTensor[wdt, _DYN2, MutAnyOrigin],
    a: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    mag: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    y: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    rows: Int, rank: Int, in_f: Int, out_f: Int,
    scale: Float32, wd_on_out: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * out_f
    if idx < total:
        var m = idx // out_f
        var o = idx - m * out_f
        var acc = Float32(0.0)
        for i in range(in_f):
            var k = o if wd_on_out != 0 else i
            var wv = rebind[Scalar[wdt]](w[o, i]).cast[DType.float32]()
            var wp = wv + _delta_at_bf16(a, b, rank, in_f, o, i, scale)
            var mv = rebind[Scalar[DType.float32]](mag[0, k])
            var dv = rebind[Scalar[DType.float32]](den[0, k])
            var xv = rebind[Scalar[xdt]](x[m, i]).cast[DType.float32]()
            acc += xv * mv * wp / dv
        y[m, o] = rebind[y.element_type](acc.cast[xdt]())


def _dx_kernel[xdt: DType, wdt: DType](
    d_y: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    w: LayoutTensor[wdt, _DYN2, MutAnyOrigin],
    a: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    mag: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    d_x: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    rows: Int, rank: Int, in_f: Int, out_f: Int,
    scale: Float32, wd_on_out: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * in_f
    if idx < total:
        var m = idx // in_f
        var i = idx - m * in_f
        var acc = Float32(0.0)
        for o in range(out_f):
            var k = o if wd_on_out != 0 else i
            var wv = rebind[Scalar[wdt]](w[o, i]).cast[DType.float32]()
            var wp = wv + _delta_at_bf16(a, b, rank, in_f, o, i, scale)
            var mv = rebind[Scalar[DType.float32]](mag[0, k])
            var dv = rebind[Scalar[DType.float32]](den[0, k])
            var we = mv * wp / dv
            var gy = rebind[Scalar[xdt]](d_y[m, o]).cast[DType.float32]()
            acc += gy * we
        d_x[m, i] = rebind[d_x.element_type](acc.cast[xdt]())


def _scale_input_cols_bf16_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    mag: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    out_t: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    rows: Int, in_f: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * in_f
    if idx < total:
        var m = idx // in_f
        var i = idx - m * in_f
        var xv = rebind[Scalar[DType.bfloat16]](x[m, i]).cast[DType.float32]()
        var mv = rebind[Scalar[DType.float32]](mag[0, i])
        var dv = rebind[Scalar[DType.float32]](den[0, i])
        out_t[m, i] = rebind[out_t.element_type]((xv * mv / dv).cast[DType.bfloat16]())


def _finish_forward_bf16_kernel(
    base: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    delta: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    y: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    rows: Int, out_f: Int, scale: Float32, has_delta: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * out_f
    if idx < total:
        var m = idx // out_f
        var o = idx - m * out_f
        var v = rebind[Scalar[DType.float32]](base[m, o])
        if has_delta != 0:
            v += rebind[Scalar[DType.float32]](delta[m, o]) * scale
        y[m, o] = rebind[y.element_type](v.cast[DType.bfloat16]())


def _g_from_dwp_per_input_kernel(
    dwp: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    mag: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    out_f: Int, in_f: Int, scale: Float32,
):
    var idx = Int(global_idx.x)
    var total = out_f * in_f
    if idx < total:
        var o = idx // in_f
        var i = idx - o * in_f
        var gv = rebind[Scalar[DType.float32]](dwp[o, i])
        var mv = rebind[Scalar[DType.float32]](mag[0, i])
        var dv = rebind[Scalar[DType.float32]](den[0, i])
        g[o, i] = rebind[g.element_type](gv * mv / dv * scale)


def _dm_from_dwp_per_input_kernel[wdt: DType](
    dwp: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    w: LayoutTensor[wdt, _DYN2, MutAnyOrigin],
    a: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    d_m: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rank: Int, in_f: Int, out_f: Int, scale: Float32, delta_zero: Int,
):
    var i = Int(global_idx.x)
    if i < in_f:
        var acc = Float32(0.0)
        var dv = rebind[Scalar[DType.float32]](den[0, i])
        for o in range(out_f):
            var wp = rebind[Scalar[wdt]](w[o, i]).cast[DType.float32]()
            if delta_zero == 0:
                wp += _delta_at_bf16(a, b, rank, in_f, o, i, scale)
            acc += rebind[Scalar[DType.float32]](dwp[o, i]) * wp / dv
        d_m[0, i] = rebind[d_m.element_type](acc)


def _finish_dx_bf16_kernel(
    base_dx: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    low_dx: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    mag: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    d_x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    rows: Int, in_f: Int, scale: Float32, has_low: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * in_f
    if idx < total:
        var m = idx // in_f
        var i = idx - m * in_f
        var v = rebind[Scalar[DType.float32]](base_dx[m, i])
        if has_low != 0:
            v += rebind[Scalar[DType.float32]](low_dx[m, i]) * scale
        var mv = rebind[Scalar[DType.float32]](mag[0, i])
        var dv = rebind[Scalar[DType.float32]](den[0, i])
        d_x[m, i] = rebind[d_x.element_type]((v * mv / dv).cast[DType.bfloat16]())


def _dm_from_x_base_dx_per_input_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    base_dx: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    d_m: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int, in_f: Int,
):
    var i = Int(global_idx.x)
    if i < in_f:
        var acc = Float32(0.0)
        var dv = rebind[Scalar[DType.float32]](den[0, i])
        for m in range(rows):
            var xv = rebind[Scalar[DType.bfloat16]](x[m, i]).cast[DType.float32]()
            var dxv = rebind[Scalar[DType.float32]](base_dx[m, i])
            acc += xv * dxv / dv
        d_m[0, i] = rebind[d_m.element_type](acc)


@always_inline
def _d_wpdora[xdt: DType](
    d_y: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    x: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    rows: Int, in_f: Int, out_f: Int, o: Int, i: Int,
) -> Float32:
    var acc = Float32(0.0)
    for m in range(rows):
        var gy = rebind[Scalar[xdt]](d_y[m, o]).cast[DType.float32]()
        var xv = rebind[Scalar[xdt]](x[m, i]).cast[DType.float32]()
        acc += gy * xv
    return acc


def _dm_kernel[xdt: DType, wdt: DType](
    d_y: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    x: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    w: LayoutTensor[wdt, _DYN2, MutAnyOrigin],
    a: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    d_m: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int, rank: Int, in_f: Int, out_f: Int,
    scale: Float32, wd_on_out: Int, mlen: Int,
):
    var idx = Int(global_idx.x)
    if idx < mlen:
        var acc = Float32(0.0)
        var dv = rebind[Scalar[DType.float32]](den[0, idx])
        if wd_on_out != 0:
            var o = idx
            for i in range(in_f):
                var wv = rebind[Scalar[wdt]](w[o, i]).cast[DType.float32]()
                var wp = wv + _delta_at_bf16(a, b, rank, in_f, o, i, scale)
                acc += _d_wpdora(d_y, x, rows, in_f, out_f, o, i) * wp / dv
        else:
            var i = idx
            for o in range(out_f):
                var wv = rebind[Scalar[wdt]](w[o, i]).cast[DType.float32]()
                var wp = wv + _delta_at_bf16(a, b, rank, in_f, o, i, scale)
                acc += _d_wpdora(d_y, x, rows, in_f, out_f, o, i) * wp / dv
        d_m[0, idx] = rebind[d_m.element_type](acc)


def _da_kernel[xdt: DType](
    d_y: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    x: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    mag: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    d_a: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int, rank: Int, in_f: Int, out_f: Int,
    scale: Float32, wd_on_out: Int,
):
    var idx = Int(global_idx.x)
    var total = rank * in_f
    if idx < total:
        var r = idx // in_f
        var i = idx - r * in_f
        var acc = Float32(0.0)
        for o in range(out_f):
            var k = o if wd_on_out != 0 else i
            var dwp = _d_wpdora(d_y, x, rows, in_f, out_f, o, i)
            var mv = rebind[Scalar[DType.float32]](mag[0, k])
            var dv = rebind[Scalar[DType.float32]](den[0, k])
            var bv = rebind[Scalar[DType.bfloat16]](b[o, r]).cast[DType.float32]()
            acc += bv * dwp * mv / dv * scale
        d_a[r, i] = rebind[d_a.element_type](acc)


def _db_kernel[xdt: DType](
    d_y: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    x: LayoutTensor[xdt, _DYN2, MutAnyOrigin],
    a: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    mag: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    den: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    d_b: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int, rank: Int, in_f: Int, out_f: Int,
    scale: Float32, wd_on_out: Int,
):
    var idx = Int(global_idx.x)
    var total = out_f * rank
    if idx < total:
        var o = idx // rank
        var r = idx - o * rank
        var acc = Float32(0.0)
        for i in range(in_f):
            var k = o if wd_on_out != 0 else i
            var dwp = _d_wpdora(d_y, x, rows, in_f, out_f, o, i)
            var mv = rebind[Scalar[DType.float32]](mag[0, k])
            var dv = rebind[Scalar[DType.float32]](den[0, k])
            var av = rebind[Scalar[DType.bfloat16]](a[r, i]).cast[DType.float32]()
            acc += dwp * mv / dv * scale * av
        d_b[o, r] = rebind[d_b.element_type](acc)


def _rows_and_in(x: Tensor, expected_in: Int, name: String) raises -> List[Int]:
    var shape = x.shape()
    if len(shape) < 1:
        raise Error(name + String(": x rank must be >= 1"))
    if shape[len(shape) - 1] != expected_in:
        raise Error(name + String(": x trailing dim mismatch"))
    var rows = 1
    for i in range(len(shape) - 1):
        rows *= shape[i]
    var out = List[Int]()
    out.append(rows)
    out.append(expected_in)
    return out^


def _validate_dora_device(d: DoRAAdapterDevice, name: String) raises:
    if d.a[].dtype() != STDtype.BF16 or d.b[].dtype() != STDtype.BF16:
        raise Error(name + String(": A/B storage must be BF16"))
    if d.m[].dtype() != STDtype.BF16:
        raise Error(name + String(": magnitude storage must be BF16"))
    if d.a[].shape() != [d.rank, d.in_f]:
        raise Error(name + String(": A shape mismatch"))
    if d.b[].shape() != [d.out_f, d.rank]:
        raise Error(name + String(": B shape mismatch"))
    var mlen = d.out_f if d.wd_on_out else d.in_f
    if d.m[].shape() != [mlen]:
        raise Error(name + String(": magnitude shape mismatch"))


def _wrap_common(
    x: Tensor, w: Tensor, d: DoRAAdapterDevice, name: String,
) raises -> List[Int]:
    _validate_dora_device(d, name)
    var dims = _rows_and_in(x, d.in_f, name)
    var wshape = w.shape()
    if len(wshape) != 2 or wshape[0] != d.out_f or wshape[1] != d.in_f:
        raise Error(name + String(": w_orig shape mismatch"))
    var xdt = x.dtype()
    var wdt = w.dtype()
    if not (
        wdt == xdt
        or (xdt == STDtype.F32 and (wdt == STDtype.BF16 or wdt == STDtype.F16))
    ):
        raise Error(name + String(": unsupported x/w dtype pair"))
    return dims^


def _layouts(
    rows: Int, rank: Int, in_f: Int, out_f: Int, mlen: Int,
) -> List[RuntimeLayout[_DYN2]]:
    var out = List[RuntimeLayout[_DYN2]]()
    out.append(RuntimeLayout[_DYN2].row_major(IndexList[2](rows, in_f)))
    out.append(RuntimeLayout[_DYN2].row_major(IndexList[2](out_f, in_f)))
    out.append(RuntimeLayout[_DYN2].row_major(IndexList[2](rank, in_f)))
    out.append(RuntimeLayout[_DYN2].row_major(IndexList[2](out_f, rank)))
    out.append(RuntimeLayout[_DYN2].row_major(IndexList[2](1, mlen)))
    out.append(RuntimeLayout[_DYN2].row_major(IndexList[2](rows, out_f)))
    return out^


def _shape2(rows: Int, cols: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return sh^


def _new_f32_tensor(rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var buf = ctx.enqueue_create_buffer[DType.uint8](rows * cols * 4)
    return Tensor(buf^, _shape2(rows, cols), STDtype.F32)


def _matmul_bf16_to_f32(
    a: Tensor, b: Tensor, rows: Int, cols: Int, k: Int,
    ctx: DeviceContext, transpose_a: Bool = False, transpose_b: Bool = False,
) raises -> Tensor:
    var out = _new_f32_tensor(rows, cols, ctx)
    if a.dtype() != STDtype.BF16 or b.dtype() != STDtype.BF16:
        raise Error("_matmul_bf16_to_f32: inputs must be BF16")
    var a_rows = k if transpose_a else rows
    var a_cols = rows if transpose_a else k
    var b_rows = cols if transpose_b else k
    var b_cols = k if transpose_b else cols
    if a.numel() != a_rows * a_cols or b.numel() != b_rows * b_cols:
        raise Error("_matmul_bf16_to_f32: flattened shape mismatch")
    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](a_rows, a_cols))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](b_rows, b_cols))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[BFloat16](), a_rl
    )
    var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
    )
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out.buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    matmul(
        ctx, C, A, B, transpose_a=transpose_a, transpose_b=transpose_b,
        c_row_major=True,
    )
    return out^


def _matmul_f32_to_f32(
    a: Tensor, b: Tensor, rows: Int, cols: Int, k: Int,
    ctx: DeviceContext, transpose_a: Bool = False, transpose_b: Bool = False,
) raises -> Tensor:
    var out = _new_f32_tensor(rows, cols, ctx)
    if a.dtype() != STDtype.F32 or b.dtype() != STDtype.F32:
        raise Error("_matmul_f32_to_f32: inputs must be F32")
    var a_rows = k if transpose_a else rows
    var a_cols = rows if transpose_a else k
    var b_rows = cols if transpose_b else k
    var b_cols = k if transpose_b else cols
    if a.numel() != a_rows * a_cols or b.numel() != b_rows * b_cols:
        raise Error("_matmul_f32_to_f32: flattened shape mismatch")
    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](a_rows, a_cols))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](b_rows, b_cols))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[Float32](), a_rl
    )
    var B = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[Float32](), b_rl
    )
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out.buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    matmul(
        ctx, C, A, B, transpose_a=transpose_a, transpose_b=transpose_b,
        c_row_major=True,
    )
    return out^


def _launch_scale_input_cols_bf16(
    x: Tensor, d: DoRAAdapterDevice, m_f32: Tensor, den: Tensor,
    out_buf: DeviceBuffer[DType.uint8], rows: Int, ctx: DeviceContext,
) raises:
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d.in_f))
    var m_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](1, d.in_f))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var M = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        m_f32.buf.unsafe_ptr().bitcast[Float32](), m_rl
    )
    var DEN = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        den.buf.unsafe_ptr().bitcast[Float32](), m_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var total = rows * d.in_f
    ctx.enqueue_function[_scale_input_cols_bf16_kernel, _scale_input_cols_bf16_kernel](
        X, M, DEN, O, rows, d.in_f,
        grid_dim=(total + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
    )


def _launch_finish_forward_bf16(
    base: Tensor, delta: Tensor, out_buf: DeviceBuffer[DType.uint8],
    rows: Int, out_f: Int, scale: Float32, has_delta: Int,
    ctx: DeviceContext,
) raises:
    var y_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, out_f))
    var BASE = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        base.buf.unsafe_ptr().bitcast[Float32](), y_rl
    )
    var DELTA = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        delta.buf.unsafe_ptr().bitcast[Float32](), y_rl
    )
    var Y = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), y_rl
    )
    var total = rows * out_f
    ctx.enqueue_function[_finish_forward_bf16_kernel, _finish_forward_bf16_kernel](
        BASE, DELTA, Y, rows, out_f, scale, has_delta,
        grid_dim=(total + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
    )


def _launch_g_from_dwp_per_input(
    dwp: Tensor, d: DoRAAdapterDevice, m_f32: Tensor, den: Tensor,
    g_buf: DeviceBuffer[DType.uint8], ctx: DeviceContext,
) raises:
    var oi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](d.out_f, d.in_f))
    var m_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](1, d.in_f))
    var DWP = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dwp.buf.unsafe_ptr().bitcast[Float32](), oi_rl
    )
    var M = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        m_f32.buf.unsafe_ptr().bitcast[Float32](), m_rl
    )
    var DEN = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        den.buf.unsafe_ptr().bitcast[Float32](), m_rl
    )
    var G = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        g_buf.unsafe_ptr().bitcast[Float32](), oi_rl
    )
    var total = d.out_f * d.in_f
    ctx.enqueue_function[_g_from_dwp_per_input_kernel, _g_from_dwp_per_input_kernel](
        DWP, M, DEN, G, d.out_f, d.in_f, d.scale,
        grid_dim=(total + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
    )


def _launch_dm_from_dwp_per_input[wdt: DType](
    dwp: Tensor, w: Tensor, d: DoRAAdapterDevice, den: Tensor,
    dm_buf: DeviceBuffer[DType.uint8], ctx: DeviceContext,
) raises:
    var oi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](d.out_f, d.in_f))
    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](d.rank, d.in_f))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](d.out_f, d.rank))
    var m_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](1, d.in_f))
    var DWP = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dwp.buf.unsafe_ptr().bitcast[Float32](), oi_rl
    )
    var W = LayoutTensor[wdt, _DYN2, MutAnyOrigin](
        w.buf.unsafe_ptr().bitcast[Scalar[wdt]](), oi_rl
    )
    var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        d.a[].buf.unsafe_ptr().bitcast[BFloat16](), a_rl
    )
    var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        d.b[].buf.unsafe_ptr().bitcast[BFloat16](), b_rl
    )
    var DEN = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        den.buf.unsafe_ptr().bitcast[Float32](), m_rl
    )
    var DM = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dm_buf.unsafe_ptr().bitcast[Float32](), m_rl
    )
    var dz = 1 if d.delta_zero else 0
    ctx.enqueue_function[
        _dm_from_dwp_per_input_kernel[wdt],
        _dm_from_dwp_per_input_kernel[wdt],
    ](
        DWP, W, A, B, DEN, DM, d.rank, d.in_f, d.out_f, d.scale, dz,
        grid_dim=(d.in_f + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
    )


def _launch_finish_dx_bf16(
    base_dx: Tensor, low_dx: Tensor, d: DoRAAdapterDevice, m_f32: Tensor,
    den: Tensor, dx_buf: DeviceBuffer[DType.uint8], rows: Int, has_low: Int,
    ctx: DeviceContext,
) raises:
    var mi_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d.in_f))
    var m_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](1, d.in_f))
    var BASE = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        base_dx.buf.unsafe_ptr().bitcast[Float32](), mi_rl
    )
    var LOW = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        low_dx.buf.unsafe_ptr().bitcast[Float32](), mi_rl
    )
    var M = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        m_f32.buf.unsafe_ptr().bitcast[Float32](), m_rl
    )
    var DEN = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        den.buf.unsafe_ptr().bitcast[Float32](), m_rl
    )
    var DX = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        dx_buf.unsafe_ptr().bitcast[BFloat16](), mi_rl
    )
    var total = rows * d.in_f
    ctx.enqueue_function[_finish_dx_bf16_kernel, _finish_dx_bf16_kernel](
        BASE, LOW, M, DEN, DX, rows, d.in_f, d.scale, has_low,
        grid_dim=(total + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
    )


def _launch_dm_from_x_base_dx_per_input(
    x: Tensor, base_dx: Tensor, den: Tensor, dm_buf: DeviceBuffer[DType.uint8],
    rows: Int, in_f: Int, ctx: DeviceContext,
) raises:
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, in_f))
    var m_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](1, in_f))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var BDX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        base_dx.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var DEN = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        den.buf.unsafe_ptr().bitcast[Float32](), m_rl
    )
    var DM = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dm_buf.unsafe_ptr().bitcast[Float32](), m_rl
    )
    ctx.enqueue_function[
        _dm_from_x_base_dx_per_input_kernel,
        _dm_from_x_base_dx_per_input_kernel,
    ](
        X, BDX, DEN, DM, rows, in_f,
        grid_dim=(in_f + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
    )


def _launch_den[wdt: DType](
    ctx: DeviceContext,
    w: Tensor, d: DoRAAdapterDevice,
    DEN: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    w_rl: RuntimeLayout[_DYN2],
    a_rl: RuntimeLayout[_DYN2],
    b_rl: RuntimeLayout[_DYN2],
    mlen: Int,
) raises:
    var W = LayoutTensor[wdt, _DYN2, MutAnyOrigin](w.buf.unsafe_ptr().bitcast[Scalar[wdt]](), w_rl)
    var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](d.a[].buf.unsafe_ptr().bitcast[BFloat16](), a_rl)
    var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](d.b[].buf.unsafe_ptr().bitcast[BFloat16](), b_rl)
    var grid = (mlen + _BLOCK - 1) // _BLOCK
    var axis = 1 if d.wd_on_out else 0
    var dz = 1 if d.delta_zero else 0
    ctx.enqueue_function[_den_kernel[wdt], _den_kernel[wdt]](
        W, A, B, DEN, d.rank, d.in_f, d.out_f, d.scale, d.eps, axis, mlen,
        dz,
        grid_dim=grid, block_dim=_BLOCK,
    )


def dora_substitution_denominators_device(
    w_orig: Tensor, d: DoRAAdapterDevice, ctx: DeviceContext,
) raises -> Tensor:
    _validate_dora_device(d, String("dora_substitution_denominators_device"))
    var wshape = w_orig.shape()
    if len(wshape) != 2 or wshape[0] != d.out_f or wshape[1] != d.in_f:
        raise Error("dora_substitution_denominators_device: w_orig shape mismatch")
    var mlen = d.out_f if d.wd_on_out else d.in_f
    var den_buf = ctx.enqueue_create_buffer[DType.uint8](mlen * 4)
    if d.delta_zero:
        var m_f32 = _m_f32_for_compute(d, ctx)
        var m_rl0 = RuntimeLayout[_DYN2].row_major(IndexList[2](1, mlen))
        var M0 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            m_f32.buf.unsafe_ptr().bitcast[Float32](), m_rl0
        )
        var DEN0 = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            den_buf.unsafe_ptr().bitcast[Float32](), m_rl0
        )
        ctx.enqueue_function[_den_from_m_kernel, _den_from_m_kernel](
            M0, DEN0, mlen, d.eps,
            grid_dim=(mlen + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
        )
        return Tensor(den_buf^, [mlen], STDtype.F32)
    var w_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](d.out_f, d.in_f))
    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](d.rank, d.in_f))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](d.out_f, d.rank))
    var den_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](1, mlen))
    var DEN = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        den_buf.unsafe_ptr().bitcast[Float32](), den_rl
    )
    var wdt = w_orig.dtype().to_mojo_dtype()
    if wdt == DType.float32:
        _launch_den[DType.float32](ctx, w_orig, d, DEN, w_rl, a_rl, b_rl, mlen)
    elif wdt == DType.bfloat16:
        _launch_den[DType.bfloat16](ctx, w_orig, d, DEN, w_rl, a_rl, b_rl, mlen)
    elif wdt == DType.float16:
        _launch_den[DType.float16](ctx, w_orig, d, DEN, w_rl, a_rl, b_rl, mlen)
    else:
        raise Error("dora_substitution_denominators_device: unsupported weight dtype")
    return Tensor(den_buf^, [mlen], STDtype.F32)


def _dora_forward_per_input_bf16_fast(
    x: Tensor, w_orig: Tensor, d: DoRAAdapterDevice, den: Tensor,
    rows: Int, ctx: DeviceContext,
) raises -> Tensor:
    var m_f32 = _m_f32_for_compute(d, ctx)
    var xs_buf = ctx.enqueue_create_buffer[DType.uint8](rows * d.in_f * STDtype.BF16.byte_size())
    _launch_scale_input_cols_bf16(x, d, m_f32, den, xs_buf, rows, ctx)
    var xs = Tensor(xs_buf^, _shape2(rows, d.in_f), STDtype.BF16)
    var base = _matmul_bf16_to_f32(
        xs, w_orig, rows, d.out_f, d.in_f, ctx, False, True,
    )
    var out_shape = x.shape()
    out_shape[len(out_shape) - 1] = d.out_f
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * d.out_f * STDtype.BF16.byte_size()
    )
    if d.delta_zero:
        _launch_finish_forward_bf16(
            base, base, out_buf, rows, d.out_f, d.scale, 0, ctx,
        )
    else:
        var xs32 = cast_tensor(xs, STDtype.F32, ctx, False)
        var a32 = cast_tensor(d.a[], STDtype.F32, ctx, False)
        var t = _matmul_f32_to_f32(
            xs32, a32, rows, d.rank, d.in_f, ctx, False, True,
        )
        var b32 = cast_tensor(d.b[], STDtype.F32, ctx, False)
        var delta = _matmul_f32_to_f32(
            t, b32, rows, d.out_f, d.rank, ctx, False, True,
        )
        _launch_finish_forward_bf16(
            base, delta, out_buf, rows, d.out_f, d.scale, 1, ctx,
        )
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


def _dora_backward_per_input_bf16_fast(
    d_y: Tensor, x: Tensor, w_orig: Tensor, d: DoRAAdapterDevice,
    den: Tensor, rows: Int, ctx: DeviceContext,
) raises -> DoRADeviceGrads:
    var m_f32 = _m_f32_for_compute(d, ctx)
    var base_dx = _matmul_bf16_to_f32(
        d_y, w_orig, rows, d.in_f, d.out_f, ctx, False, False,
    )
    var dwp = _matmul_bf16_to_f32(
        d_y, x, d.out_f, d.in_f, rows, ctx, True, False,
    )
    var g_buf = ctx.enqueue_create_buffer[DType.uint8](d.out_f * d.in_f * 4)
    _launch_g_from_dwp_per_input(dwp, d, m_f32, den, g_buf, ctx)
    var g = Tensor(g_buf^, _shape2(d.out_f, d.in_f), STDtype.F32)

    var dm_buf = ctx.enqueue_create_buffer[DType.uint8](d.in_f * 4)
    _launch_dm_from_dwp_per_input[DType.bfloat16](dwp, w_orig, d, den, dm_buf, ctx)

    var a32 = cast_tensor(d.a[], STDtype.F32, ctx, False)
    var d_b = _matmul_f32_to_f32(
        g, a32, d.out_f, d.rank, d.in_f, ctx, False, True,
    )

    var d_a: Tensor
    if d.delta_zero:
        var da_buf = ctx.enqueue_create_buffer[DType.uint8](d.rank * d.in_f * 4)
        da_buf.enqueue_fill(UInt8(0))
        d_a = Tensor(da_buf^, _shape2(d.rank, d.in_f), STDtype.F32)
        var dx_buf = ctx.enqueue_create_buffer[DType.uint8](
            rows * d.in_f * STDtype.BF16.byte_size()
        )
        _launch_finish_dx_bf16(base_dx, base_dx, d, m_f32, den, dx_buf, rows, 0, ctx)
        return DoRADeviceGrads(
            d_a^,
            d_b^,
            Tensor(dm_buf^, [d.in_f], STDtype.F32),
            Tensor(dx_buf^, x.shape(), STDtype.BF16),
        )
    else:
        var b32 = cast_tensor(d.b[], STDtype.F32, ctx, False)
        d_a = _matmul_f32_to_f32(
            b32, g, d.rank, d.in_f, d.out_f, ctx, True, False,
        )
        var dy32 = cast_tensor(d_y, STDtype.F32, ctx, False)
        var t = _matmul_f32_to_f32(
            dy32, b32, rows, d.rank, d.out_f, ctx, False, False,
        )
        var low_dx = _matmul_f32_to_f32(
            t, a32, rows, d.in_f, d.rank, ctx, False, False,
        )
        var dx_buf = ctx.enqueue_create_buffer[DType.uint8](
            rows * d.in_f * STDtype.BF16.byte_size()
        )
        _launch_finish_dx_bf16(base_dx, low_dx, d, m_f32, den, dx_buf, rows, 1, ctx)
        return DoRADeviceGrads(
            d_a^,
            d_b^,
            Tensor(dm_buf^, [d.in_f], STDtype.F32),
            Tensor(dx_buf^, x.shape(), STDtype.BF16),
        )


def _dora_backward_per_input_bf16_zero_fast(
    d_y: Tensor, x: Tensor, w_orig: Tensor, d: DoRAAdapterDevice,
    den: Tensor, rows: Int, ctx: DeviceContext,
) raises -> DoRADeviceGrads:
    var m_f32 = _m_f32_for_compute(d, ctx)
    var base_dx = _matmul_bf16_to_f32(
        d_y, w_orig, rows, d.in_f, d.out_f, ctx, False, False,
    )

    var xs_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * d.in_f * STDtype.BF16.byte_size()
    )
    _launch_scale_input_cols_bf16(x, d, m_f32, den, xs_buf, rows, ctx)
    var xs = Tensor(xs_buf^, _shape2(rows, d.in_f), STDtype.BF16)
    var t = _matmul_bf16_to_f32(
        xs, d.a[], rows, d.rank, d.in_f, ctx, False, True,
    )

    var dy32 = cast_tensor(d_y, STDtype.F32, ctx, False)
    var d_b_raw = _matmul_f32_to_f32(
        dy32^, t, d.out_f, d.rank, rows, ctx, True, False,
    )
    var d_b = mul_scalar(d_b_raw^, d.scale, ctx)

    var da_buf = ctx.enqueue_create_buffer[DType.uint8](d.rank * d.in_f * 4)
    da_buf.enqueue_fill(UInt8(0))
    var dm_buf = ctx.enqueue_create_buffer[DType.uint8](d.in_f * 4)
    _launch_dm_from_x_base_dx_per_input(x, base_dx, den, dm_buf, rows, d.in_f, ctx)

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * d.in_f * STDtype.BF16.byte_size()
    )
    _launch_finish_dx_bf16(base_dx, base_dx, d, m_f32, den, dx_buf, rows, 0, ctx)
    return DoRADeviceGrads(
        Tensor(da_buf^, _shape2(d.rank, d.in_f), STDtype.F32),
        d_b^,
        Tensor(dm_buf^, [d.in_f], STDtype.F32),
        Tensor(dx_buf^, x.shape(), STDtype.BF16),
    )


def _launch_forward[xdt: DType, wdt: DType](
    ctx: DeviceContext,
    x: Tensor, w: Tensor, d: DoRAAdapterDevice, m_f32: Tensor, den: Tensor,
    out_buf: DeviceBuffer[DType.uint8],
    rows: Int, mlen: Int,
) raises:
    var all_rl = _layouts(rows, d.rank, d.in_f, d.out_f, mlen)
    var X = LayoutTensor[xdt, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Scalar[xdt]](), all_rl[0])
    var W = LayoutTensor[wdt, _DYN2, MutAnyOrigin](w.buf.unsafe_ptr().bitcast[Scalar[wdt]](), all_rl[1])
    var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](d.a[].buf.unsafe_ptr().bitcast[BFloat16](), all_rl[2])
    var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](d.b[].buf.unsafe_ptr().bitcast[BFloat16](), all_rl[3])
    var M = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](m_f32.buf.unsafe_ptr().bitcast[Float32](), all_rl[4])
    var DEN = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](den.buf.unsafe_ptr().bitcast[Float32](), all_rl[4])
    var Y = LayoutTensor[xdt, _DYN2, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Scalar[xdt]](), all_rl[5])
    var total = rows * d.out_f
    var grid = (total + _BLOCK - 1) // _BLOCK
    var axis = 1 if d.wd_on_out else 0
    ctx.enqueue_function[_forward_kernel[xdt, wdt], _forward_kernel[xdt, wdt]](
        X, W, A, B, M, DEN, Y, rows, d.rank, d.in_f, d.out_f, d.scale, axis,
        grid_dim=grid, block_dim=_BLOCK,
    )


def dora_substitution_forward_device(
    x: Tensor, w_orig: Tensor, d: DoRAAdapterDevice, ctx: DeviceContext,
) raises -> Tensor:
    if (
        (not d.wd_on_out)
        and d.delta_zero
        and (x.dtype() == STDtype.BF16 or x.dtype() == STDtype.F32)
        and d.eps <= Float32(1.0e-6)
    ):
        _validate_dora_device(d, String("dora_substitution_forward_device"))
        var init_dims = _rows_and_in(x, d.in_f, String("dora_substitution_forward_device"))
        var wshape = w_orig.shape()
        if len(wshape) != 2 or wshape[0] != d.out_f or wshape[1] != d.in_f:
            raise Error("dora_substitution_forward_device: w_orig shape mismatch")
        # OneTrainer DoRA initializes B=0 and m from W, so BF16 production
        # forward is the frozen linear path at init. This avoids a per-column
        # scale pass over every large projection before the first optimizer step.
        if x.dtype() == STDtype.F32:
            return linear(x, w_orig, Optional[Tensor](None), ctx)
        if w_orig.dtype() == STDtype.BF16:
            return linear(x, w_orig, Optional[Tensor](None), ctx)
        if w_orig.dtype() == STDtype.F32:
            var x32 = cast_tensor(x, STDtype.F32, ctx, False)
            var y32 = linear(x32^, w_orig, Optional[Tensor](None), ctx)
            return cast_tensor(y32^, STDtype.BF16, ctx, False)
        _ = init_dims[0]
    var dims = _wrap_common(x, w_orig, d, String("dora_substitution_forward_device"))
    var rows = dims[0]
    var mlen = d.out_f if d.wd_on_out else d.in_f
    var den = dora_substitution_denominators_device(w_orig, d, ctx)
    if (not d.wd_on_out) and x.dtype() == STDtype.BF16 and w_orig.dtype() == STDtype.BF16:
        return _dora_forward_per_input_bf16_fast(x, w_orig, d, den, rows, ctx)
    var out_shape = x.shape()
    out_shape[len(out_shape) - 1] = d.out_f
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](rows * d.out_f * x.dtype().byte_size())
    var m_f32 = _m_f32_for_compute(d, ctx)
    var xdt = x.dtype().to_mojo_dtype()
    var wdt = w_orig.dtype().to_mojo_dtype()
    if xdt == DType.float32 and wdt == DType.float32:
        _launch_forward[DType.float32, DType.float32](ctx, x, w_orig, d, m_f32, den, out_buf, rows, mlen)
    elif xdt == DType.float32 and wdt == DType.bfloat16:
        _launch_forward[DType.float32, DType.bfloat16](ctx, x, w_orig, d, m_f32, den, out_buf, rows, mlen)
    elif xdt == DType.float32 and wdt == DType.float16:
        _launch_forward[DType.float32, DType.float16](ctx, x, w_orig, d, m_f32, den, out_buf, rows, mlen)
    elif xdt == DType.bfloat16 and wdt == DType.bfloat16:
        _launch_forward[DType.bfloat16, DType.bfloat16](ctx, x, w_orig, d, m_f32, den, out_buf, rows, mlen)
    elif xdt == DType.float16 and wdt == DType.float16:
        _launch_forward[DType.float16, DType.float16](ctx, x, w_orig, d, m_f32, den, out_buf, rows, mlen)
    else:
        raise Error("dora_substitution_forward_device: unsupported x/w dtype pair")
    return Tensor(out_buf^, out_shape^, x.dtype())


def _launch_backward[xdt: DType, wdt: DType](
    ctx: DeviceContext,
    d_y: Tensor, x: Tensor, w: Tensor, d: DoRAAdapterDevice, m_f32: Tensor, den: Tensor,
    da_buf: DeviceBuffer[DType.uint8],
    db_buf: DeviceBuffer[DType.uint8],
    dm_buf: DeviceBuffer[DType.uint8],
    dx_buf: DeviceBuffer[DType.uint8],
    rows: Int, mlen: Int,
) raises:
    var all_rl = _layouts(rows, d.rank, d.in_f, d.out_f, mlen)
    var X = LayoutTensor[xdt, _DYN2, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Scalar[xdt]](), all_rl[0])
    var W = LayoutTensor[wdt, _DYN2, MutAnyOrigin](w.buf.unsafe_ptr().bitcast[Scalar[wdt]](), all_rl[1])
    var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](d.a[].buf.unsafe_ptr().bitcast[BFloat16](), all_rl[2])
    var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](d.b[].buf.unsafe_ptr().bitcast[BFloat16](), all_rl[3])
    var M = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](m_f32.buf.unsafe_ptr().bitcast[Float32](), all_rl[4])
    var DEN = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](den.buf.unsafe_ptr().bitcast[Float32](), all_rl[4])
    var GY = LayoutTensor[xdt, _DYN2, MutAnyOrigin](d_y.buf.unsafe_ptr().bitcast[Scalar[xdt]](), all_rl[5])
    var DA = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](da_buf.unsafe_ptr().bitcast[Float32](), all_rl[2])
    var DB = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](db_buf.unsafe_ptr().bitcast[Float32](), all_rl[3])
    var DM = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](dm_buf.unsafe_ptr().bitcast[Float32](), all_rl[4])
    var DX = LayoutTensor[xdt, _DYN2, MutAnyOrigin](dx_buf.unsafe_ptr().bitcast[Scalar[xdt]](), all_rl[0])
    var axis = 1 if d.wd_on_out else 0
    var dx_total = rows * d.in_f
    var da_total = d.rank * d.in_f
    var db_total = d.out_f * d.rank
    ctx.enqueue_function[_dx_kernel[xdt, wdt], _dx_kernel[xdt, wdt]](
        GY, W, A, B, M, DEN, DX, rows, d.rank, d.in_f, d.out_f, d.scale, axis,
        grid_dim=(dx_total + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
    )
    ctx.enqueue_function[_dm_kernel[xdt, wdt], _dm_kernel[xdt, wdt]](
        GY, X, W, A, B, DEN, DM, rows, d.rank, d.in_f, d.out_f, d.scale, axis, mlen,
        grid_dim=(mlen + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
    )
    ctx.enqueue_function[_da_kernel[xdt], _da_kernel[xdt]](
        GY, X, B, M, DEN, DA, rows, d.rank, d.in_f, d.out_f, d.scale, axis,
        grid_dim=(da_total + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
    )
    ctx.enqueue_function[_db_kernel[xdt], _db_kernel[xdt]](
        GY, X, A, M, DEN, DB, rows, d.rank, d.in_f, d.out_f, d.scale, axis,
        grid_dim=(db_total + _BLOCK - 1) // _BLOCK, block_dim=_BLOCK,
    )


def dora_substitution_backward_device(
    d_y: Tensor, x: Tensor, w_orig: Tensor, d: DoRAAdapterDevice,
    ctx: DeviceContext,
) raises -> DoRADeviceGrads:
    var dims = _wrap_common(x, w_orig, d, String("dora_substitution_backward_device"))
    var rows = dims[0]
    var yshape = d_y.shape()
    var expect = x.shape()
    expect[len(expect) - 1] = d.out_f
    if yshape != expect:
        raise Error("dora_substitution_backward_device: d_y shape mismatch")
    if d_y.dtype() != x.dtype():
        raise Error("dora_substitution_backward_device: d_y/x dtype mismatch")
    var mlen = d.out_f if d.wd_on_out else d.in_f
    var den = dora_substitution_denominators_device(w_orig, d, ctx)
    if (
        (not d.wd_on_out)
        and x.dtype() == STDtype.BF16
        and d_y.dtype() == STDtype.BF16
        and w_orig.dtype() == STDtype.BF16
        and d.delta_zero
    ):
        return _dora_backward_per_input_bf16_zero_fast(
            d_y, x, w_orig, d, den, rows, ctx
        )
    if (not d.wd_on_out) and x.dtype() == STDtype.BF16 and w_orig.dtype() == STDtype.BF16:
        return _dora_backward_per_input_bf16_fast(d_y, x, w_orig, d, den, rows, ctx)
    var da_buf = ctx.enqueue_create_buffer[DType.uint8](d.rank * d.in_f * 4)
    var db_buf = ctx.enqueue_create_buffer[DType.uint8](d.out_f * d.rank * 4)
    var dm_buf = ctx.enqueue_create_buffer[DType.uint8](mlen * 4)
    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](rows * d.in_f * x.dtype().byte_size())
    var m_f32 = _m_f32_for_compute(d, ctx)
    var xdt = x.dtype().to_mojo_dtype()
    var wdt = w_orig.dtype().to_mojo_dtype()
    if xdt == DType.float32 and wdt == DType.float32:
        _launch_backward[DType.float32, DType.float32](ctx, d_y, x, w_orig, d, m_f32, den, da_buf, db_buf, dm_buf, dx_buf, rows, mlen)
    elif xdt == DType.float32 and wdt == DType.bfloat16:
        _launch_backward[DType.float32, DType.bfloat16](ctx, d_y, x, w_orig, d, m_f32, den, da_buf, db_buf, dm_buf, dx_buf, rows, mlen)
    elif xdt == DType.float32 and wdt == DType.float16:
        _launch_backward[DType.float32, DType.float16](ctx, d_y, x, w_orig, d, m_f32, den, da_buf, db_buf, dm_buf, dx_buf, rows, mlen)
    elif xdt == DType.bfloat16 and wdt == DType.bfloat16:
        _launch_backward[DType.bfloat16, DType.bfloat16](ctx, d_y, x, w_orig, d, m_f32, den, da_buf, db_buf, dm_buf, dx_buf, rows, mlen)
    elif xdt == DType.float16 and wdt == DType.float16:
        _launch_backward[DType.float16, DType.float16](ctx, d_y, x, w_orig, d, m_f32, den, da_buf, db_buf, dm_buf, dx_buf, rows, mlen)
    else:
        raise Error("dora_substitution_backward_device: unsupported x/w dtype pair")
    return DoRADeviceGrads(
        Tensor(da_buf^, [d.rank, d.in_f], STDtype.F32),
        Tensor(db_buf^, [d.out_f, d.rank], STDtype.F32),
        Tensor(dm_buf^, [mlen], STDtype.F32),
        Tensor(dx_buf^, x.shape(), x.dtype()),
    )
