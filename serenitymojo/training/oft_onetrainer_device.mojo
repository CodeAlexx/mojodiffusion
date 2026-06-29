# training/oft_onetrainer_device.mojo -- GPU OneTrainer-OFT block-size-4 helpers.
#
# This is the device-resident counterpart to training/oft_onetrainer.mojo for the
# direct LyCORIS path. It implements the OneTrainer input-side OFT rotation:
#   x_rot[..., g, c] = sum_k x[..., g, k] * R_g[k, c]
# where block size is fixed to 4 and
#   R = I + 2Q + 2Q^2 + 2Q^3 + Q^4.
#
# Boundary contract: x and d_x keep their original storage dtype. All rotation
# math and OFT trainable gradients use F32 internally.

from std.collections import List
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.atomic import Atomic
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _B4_NE = 6


struct OFTOTDeviceGrads(Movable):
    var d_vec: Tensor
    var d_x: Tensor

    def __init__(out self, var d_vec: Tensor, var d_x: Tensor):
        self.d_vec = d_vec^
        self.d_x = d_x^


@always_inline
def _q_b4(
    v0: Float32, v1: Float32, v2: Float32,
    v3: Float32, v4: Float32, v5: Float32,
    i: Int, j: Int,
) -> Float32:
    if i == j:
        return Float32(0.0)
    if i == 0:
        if j == 1:
            return v0
        if j == 2:
            return v1
        if j == 3:
            return v2
    elif i == 1:
        if j == 0:
            return -v0
        if j == 2:
            return v3
        if j == 3:
            return v4
    elif i == 2:
        if j == 0:
            return -v1
        if j == 1:
            return -v3
        if j == 3:
            return v5
    else:
        if j == 0:
            return -v2
        if j == 1:
            return -v4
        if j == 2:
            return -v5
    return Float32(0.0)


@always_inline
def _qt_b4(
    v0: Float32, v1: Float32, v2: Float32,
    v3: Float32, v4: Float32, v5: Float32,
    i: Int, j: Int,
) -> Float32:
    return _q_b4(v0, v1, v2, v3, v4, v5, j, i)


@always_inline
def _q2_b4(
    v0: Float32, v1: Float32, v2: Float32,
    v3: Float32, v4: Float32, v5: Float32,
    i: Int, j: Int,
) -> Float32:
    var acc = Float32(0.0)
    for a in range(4):
        acc += (
            _q_b4(v0, v1, v2, v3, v4, v5, i, a)
            * _q_b4(v0, v1, v2, v3, v4, v5, a, j)
        )
    return acc


@always_inline
def _q3_b4(
    v0: Float32, v1: Float32, v2: Float32,
    v3: Float32, v4: Float32, v5: Float32,
    i: Int, j: Int,
) -> Float32:
    var acc = Float32(0.0)
    for a in range(4):
        acc += _q2_b4(v0, v1, v2, v3, v4, v5, i, a) * _q_b4(v0, v1, v2, v3, v4, v5, a, j)
    return acc


@always_inline
def _q4_b4(
    v0: Float32, v1: Float32, v2: Float32,
    v3: Float32, v4: Float32, v5: Float32,
    i: Int, j: Int,
) -> Float32:
    var acc = Float32(0.0)
    for a in range(4):
        acc += _q3_b4(v0, v1, v2, v3, v4, v5, i, a) * _q_b4(v0, v1, v2, v3, v4, v5, a, j)
    return acc


@always_inline
def _qt2_b4(
    v0: Float32, v1: Float32, v2: Float32,
    v3: Float32, v4: Float32, v5: Float32,
    i: Int, j: Int,
) -> Float32:
    var acc = Float32(0.0)
    for a in range(4):
        acc += _qt_b4(v0, v1, v2, v3, v4, v5, i, a) * _qt_b4(v0, v1, v2, v3, v4, v5, a, j)
    return acc


@always_inline
def _qt3_b4(
    v0: Float32, v1: Float32, v2: Float32,
    v3: Float32, v4: Float32, v5: Float32,
    i: Int, j: Int,
) -> Float32:
    var acc = Float32(0.0)
    for a in range(4):
        acc += _qt2_b4(v0, v1, v2, v3, v4, v5, i, a) * _qt_b4(v0, v1, v2, v3, v4, v5, a, j)
    return acc


@always_inline
def _r_b4(
    v0: Float32, v1: Float32, v2: Float32,
    v3: Float32, v4: Float32, v5: Float32,
    i: Int, j: Int,
) -> Float32:
    var out = Float32(0.0)
    if i == j:
        out = Float32(1.0)
    out += Float32(2.0) * _q_b4(v0, v1, v2, v3, v4, v5, i, j)
    out += Float32(2.0) * _q2_b4(v0, v1, v2, v3, v4, v5, i, j)
    out += Float32(2.0) * _q3_b4(v0, v1, v2, v3, v4, v5, i, j)
    out += _q4_b4(v0, v1, v2, v3, v4, v5, i, j)
    return out


@always_inline
def _gbar_b4(
    d_rg: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    g: Int, i: Int, j: Int,
) -> Float32:
    return rebind[Scalar[DType.float32]](d_rg[g, i * 4 + j])


@always_inline
def _dq_entry_b4(
    d_rg: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    g: Int,
    v0: Float32, v1: Float32, v2: Float32,
    v3: Float32, v4: Float32, v5: Float32,
    i: Int, j: Int,
) -> Float32:
    var n2a = Float32(0.0)
    var n2b = Float32(0.0)
    var n3a = Float32(0.0)
    var n3b = Float32(0.0)
    var n3c = Float32(0.0)
    var n4a = Float32(0.0)
    var n4b = Float32(0.0)
    var n4c = Float32(0.0)
    var n4d = Float32(0.0)
    for a in range(4):
        n2a += _gbar_b4(d_rg, g, i, a) * _qt_b4(v0, v1, v2, v3, v4, v5, a, j)
        n2b += _qt_b4(v0, v1, v2, v3, v4, v5, i, a) * _gbar_b4(d_rg, g, a, j)
        n3a += _gbar_b4(d_rg, g, i, a) * _qt2_b4(v0, v1, v2, v3, v4, v5, a, j)
        n3c += _qt2_b4(v0, v1, v2, v3, v4, v5, i, a) * _gbar_b4(d_rg, g, a, j)
        n4a += _gbar_b4(d_rg, g, i, a) * _qt3_b4(v0, v1, v2, v3, v4, v5, a, j)
        n4d += _qt3_b4(v0, v1, v2, v3, v4, v5, i, a) * _gbar_b4(d_rg, g, a, j)
        for b in range(4):
            n3b += (
                _qt_b4(v0, v1, v2, v3, v4, v5, i, a)
                * _gbar_b4(d_rg, g, a, b)
                * _qt_b4(v0, v1, v2, v3, v4, v5, b, j)
            )
            n4b += (
                _qt_b4(v0, v1, v2, v3, v4, v5, i, a)
                * _gbar_b4(d_rg, g, a, b)
                * _qt2_b4(v0, v1, v2, v3, v4, v5, b, j)
            )
            n4c += (
                _qt2_b4(v0, v1, v2, v3, v4, v5, i, a)
                * _gbar_b4(d_rg, g, a, b)
                * _qt_b4(v0, v1, v2, v3, v4, v5, b, j)
            )
    return (
        Float32(2.0) * _gbar_b4(d_rg, g, i, j)
        + Float32(2.0) * (n2a + n2b)
        + Float32(2.0) * (n3a + n3b + n3c)
        + (n4a + n4b + n4c + n4d)
    )


def _oft_b4_forward_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    vec: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    rows: Int,
    in_f: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * in_f
    if idx < total:
        var m = idx // in_f
        var col = idx % in_f
        var g = col // 4
        var c = col - g * 4
        var base = g * 4
        var v0 = rebind[Scalar[DType.float32]](vec[g, 0])
        var v1 = rebind[Scalar[DType.float32]](vec[g, 1])
        var v2 = rebind[Scalar[DType.float32]](vec[g, 2])
        var v3 = rebind[Scalar[DType.float32]](vec[g, 3])
        var v4 = rebind[Scalar[DType.float32]](vec[g, 4])
        var v5 = rebind[Scalar[DType.float32]](vec[g, 5])
        var acc = Float32(0.0)
        for k in range(4):
            var xv = rebind[Scalar[dtype]](x[m, base + k]).cast[DType.float32]()
            acc += xv * _r_b4(v0, v1, v2, v3, v4, v5, k, c)
        o[m, col] = rebind[o.element_type](acc.cast[dtype]())


def _oft_b4_dx_kernel[dtype: DType](
    d_x_rot: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    vec: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    rows: Int,
    in_f: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * in_f
    if idx < total:
        var m = idx // in_f
        var col = idx % in_f
        var g = col // 4
        var k = col - g * 4
        var base = g * 4
        var v0 = rebind[Scalar[DType.float32]](vec[g, 0])
        var v1 = rebind[Scalar[DType.float32]](vec[g, 1])
        var v2 = rebind[Scalar[DType.float32]](vec[g, 2])
        var v3 = rebind[Scalar[DType.float32]](vec[g, 3])
        var v4 = rebind[Scalar[DType.float32]](vec[g, 4])
        var v5 = rebind[Scalar[DType.float32]](vec[g, 5])
        var acc = Float32(0.0)
        for c in range(4):
            var gv = rebind[Scalar[dtype]](d_x_rot[m, base + c]).cast[DType.float32]()
            acc += gv * _r_b4(v0, v1, v2, v3, v4, v5, k, c)
        o[m, col] = rebind[o.element_type](acc.cast[dtype]())


def _oft_b4_drg_kernel[dtype: DType](
    d_x_rot: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    x: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    d_rg: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    in_f: Int,
    r: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * r * 16
    if idx < total:
        var rc = idx % 16
        var tmp = idx // 16
        var g = tmp % r
        var m = tmp // r
        var k = rc // 4
        var c = rc - k * 4
        var base = g * 4
        var gv = rebind[Scalar[dtype]](d_x_rot[m, base + c]).cast[DType.float32]()
        var xv = rebind[Scalar[dtype]](x[m, base + k]).cast[DType.float32]()
        var dst_ptr = d_rg.ptr + (g * 16 + rc)
        _ = Atomic[DType.float32].fetch_add(dst_ptr, gv * xv)


def _oft_b4_dvec_kernel(
    d_rg: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    vec: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    d_vec: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    r: Int,
):
    var idx = Int(global_idx.x)
    var total = r * 6
    if idx < total:
        var g = idx // 6
        var t = idx - g * 6
        var i = 0
        var j = 1
        if t == 1:
            j = 2
        elif t == 2:
            j = 3
        elif t == 3:
            i = 1
            j = 2
        elif t == 4:
            i = 1
            j = 3
        elif t == 5:
            i = 2
            j = 3
        var v0 = rebind[Scalar[DType.float32]](vec[g, 0])
        var v1 = rebind[Scalar[DType.float32]](vec[g, 1])
        var v2 = rebind[Scalar[DType.float32]](vec[g, 2])
        var v3 = rebind[Scalar[DType.float32]](vec[g, 3])
        var v4 = rebind[Scalar[DType.float32]](vec[g, 4])
        var v5 = rebind[Scalar[DType.float32]](vec[g, 5])
        var dq_ij = _dq_entry_b4(d_rg, g, v0, v1, v2, v3, v4, v5, i, j)
        var dq_ji = _dq_entry_b4(d_rg, g, v0, v1, v2, v3, v4, v5, j, i)
        d_vec[g, t] = rebind[d_vec.element_type](dq_ij - dq_ji)


def _shape_rows_in(x: Tensor, name: String) raises -> List[Int]:
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error(name + String(": x rank must be >= 1"))
    var in_f = xshape[len(xshape) - 1]
    if in_f % 4 != 0:
        raise Error(name + String(": trailing dim must be divisible by 4"))
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    var out = List[Int]()
    out.append(rows)
    out.append(in_f)
    return out^


def _validate_vec_b4(vec: Tensor, r: Int, name: String) raises -> List[Int]:
    if vec.dtype() != STDtype.F32:
        raise Error(name + String(": OFT vec must be F32"))
    var vshape = vec.shape()
    if len(vshape) == 1:
        if vshape[0] != r * _B4_NE:
            raise Error(name + String(": flat vec must have r*6 elements"))
    elif len(vshape) == 2:
        if vshape[0] != r or vshape[1] != _B4_NE:
            raise Error(name + String(": vec must be [r,6]"))
    else:
        raise Error(name + String(": vec must be flat [r*6] or [r,6]"))
    var out = List[Int]()
    out.append(r)
    out.append(_B4_NE)
    return out^


def oft_ot_rotate_b4(x: Tensor, vec: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Apply OneTrainer-OFT input rotation with block_size=4 on device."""
    var dims = _shape_rows_in(x, String("oft_ot_rotate_b4"))
    var rows = dims[0]
    var in_f = dims[1]
    var r = in_f // 4
    var v2 = _validate_vec_b4(vec, r, String("oft_ot_rotate_b4"))
    var dt = x.dtype().to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("oft_ot_rotate_b4: unsupported x dtype")

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, in_f))
    var v_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](v2[0], v2[1]))
    var total = rows * in_f
    var grid = (total + _BLOCK - 1) // _BLOCK

    var V = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        vec.buf.unsafe_ptr().bitcast[Float32](), v_rl
    )
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[
            _oft_b4_forward_kernel[DType.float32], _oft_b4_forward_kernel[DType.float32]
        ](X, V, O, rows, in_f, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[
            _oft_b4_forward_kernel[DType.bfloat16], _oft_b4_forward_kernel[DType.bfloat16]
        ](X, V, O, rows, in_f, grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[
            _oft_b4_forward_kernel[DType.float16], _oft_b4_forward_kernel[DType.float16]
        ](X, V, O, rows, in_f, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, x.shape(), x.dtype())


def oft_ot_rotate_backward_b4(
    d_x_rot: Tensor, x: Tensor, vec: Tensor, ctx: DeviceContext,
) raises -> OFTOTDeviceGrads:
    """Backward for x_rot = oft_ot_rotate_b4(x, vec)."""
    var dims = _shape_rows_in(x, String("oft_ot_rotate_backward_b4"))
    var rows = dims[0]
    var in_f = dims[1]
    if d_x_rot.shape() != x.shape():
        raise Error("oft_ot_rotate_backward_b4: d_x_rot/x shape mismatch")
    if d_x_rot.dtype() != x.dtype():
        raise Error("oft_ot_rotate_backward_b4: d_x_rot/x dtype mismatch")
    var r = in_f // 4
    var v2 = _validate_vec_b4(vec, r, String("oft_ot_rotate_backward_b4"))
    var dt = x.dtype().to_mojo_dtype()
    if dt != DType.float32 and dt != DType.bfloat16 and dt != DType.float16:
        raise Error("oft_ot_rotate_backward_b4: unsupported x dtype")

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var drg_buf = ctx.enqueue_create_buffer[DType.uint8](r * 16 * 4)
    drg_buf.enqueue_fill(UInt8(0))
    var dvec_buf = ctx.enqueue_create_buffer[DType.uint8](r * _B4_NE * 4)

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, in_f))
    var v_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](v2[0], v2[1]))
    var drg_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](r, 16))
    var V = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        vec.buf.unsafe_ptr().bitcast[Float32](), v_rl
    )
    var DRG = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        drg_buf.unsafe_ptr().bitcast[Float32](), drg_rl
    )

    var rot_total = rows * in_f
    var drg_total = rows * r * 16
    var rot_grid = (rot_total + _BLOCK - 1) // _BLOCK
    var drg_grid = (drg_total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var G = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            d_x_rot.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[
            _oft_b4_drg_kernel[DType.float32], _oft_b4_drg_kernel[DType.float32]
        ](G, X, DRG, rows, in_f, r, grid_dim=drg_grid, block_dim=_BLOCK)
        ctx.enqueue_function[
            _oft_b4_dx_kernel[DType.float32], _oft_b4_dx_kernel[DType.float32]
        ](G, V, DX, rows, in_f, grid_dim=rot_grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var G = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            d_x_rot.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var DX = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[
            _oft_b4_drg_kernel[DType.bfloat16], _oft_b4_drg_kernel[DType.bfloat16]
        ](G, X, DRG, rows, in_f, r, grid_dim=drg_grid, block_dim=_BLOCK)
        ctx.enqueue_function[
            _oft_b4_dx_kernel[DType.bfloat16], _oft_b4_dx_kernel[DType.bfloat16]
        ](G, V, DX, rows, in_f, grid_dim=rot_grid, block_dim=_BLOCK)
    else:
        var G = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            d_x_rot.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var DX = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[
            _oft_b4_drg_kernel[DType.float16], _oft_b4_drg_kernel[DType.float16]
        ](G, X, DRG, rows, in_f, r, grid_dim=drg_grid, block_dim=_BLOCK)
        ctx.enqueue_function[
            _oft_b4_dx_kernel[DType.float16], _oft_b4_dx_kernel[DType.float16]
        ](G, V, DX, rows, in_f, grid_dim=rot_grid, block_dim=_BLOCK)

    var DV = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dvec_buf.unsafe_ptr().bitcast[Float32](), v_rl
    )
    var dvec_grid = (r * _B4_NE + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_oft_b4_dvec_kernel, _oft_b4_dvec_kernel](
        DRG, V, DV, r, grid_dim=dvec_grid, block_dim=_BLOCK
    )

    return OFTOTDeviceGrads(
        Tensor(dvec_buf^, vec.shape(), STDtype.F32),
        Tensor(dx_buf^, x.shape(), x.dtype()),
    )
