# ops/elementwise_backward.mojo — BACKWARD for modulate (DiT AdaLN). F32.
#
# Backward partner of ops/elementwise.mojo's `modulate`. The other elementwise
# AdaLN primitive `residual_gate` already has a backward (`gate_residual_backward`
# in ops/rope_struct_backward.mojo), so only `modulate` needed one.
#
# Forward:  o = (1 + scale) * x + shift      (scale, shift per-channel [D])
# Backward (given go = dL/do, [..,D]):
#   d_x[r,c]    = go[r,c] * (1 + scale[c])
#   d_scale[c]  = sum_rows( go[r,c] * x[r,c] )      (cross-row column reduction)
#   d_shift[c]  = sum_rows( go[r,c] )               (cross-row column reduction)
# Same scaffolding/discipline as ops/norm_backward.mojo's layer_norm param grads
# (one thread per column accumulating over rows). All interior math F32; BF16/F16
# storage path casts up, runs F32, casts grads down.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# d_x[r,c] = go[r,c] * (1 + scale[c])  (elementwise; scale broadcast per channel)
def _modulate_bwd_dx_kernel(
    go: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dx: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var gov = rebind[Scalar[DType.float32]](go[r, c])
        var sv = rebind[Scalar[DType.float32]](s[c])
        dx[r, c] = rebind[dx.element_type](gov * (1.0 + sv))


# d_scale[c] = sum_r go[r,c]*x[r,c] ; d_shift[c] = sum_r go[r,c]  (one thread/col)
def _modulate_bwd_param_kernel(
    go: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dscale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dshift: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var col = Int(global_idx.x)
    if col >= cols:
        return
    var acc_s: Float32 = 0.0
    var acc_sh: Float32 = 0.0
    for r in range(rows):
        var gov = rebind[Scalar[DType.float32]](go[r, col])
        var xv = rebind[Scalar[DType.float32]](x[r, col])
        acc_s += gov * xv
        acc_sh += gov
    dscale[col] = rebind[dscale.element_type](acc_s)
    dshift[col] = rebind[dshift.element_type](acc_sh)


struct ModulateBackward(Movable):
    """Backward outputs of modulate: d_x [..,D], d_scale [D], d_shift [D]."""

    var d_x: Tensor
    var d_scale: Tensor
    var d_shift: Tensor

    def __init__(out self, var d_x: Tensor, var d_scale: Tensor, var d_shift: Tensor):
        self.d_x = d_x^
        self.d_scale = d_scale^
        self.d_shift = d_shift^


def modulate_backward(
    go: Tensor, x: Tensor, scale: Tensor, ctx: DeviceContext,
    compute_param_grads: Bool = True,
) raises -> ModulateBackward:
    """Backward of modulate (forward: o=(1+scale)*x+shift, scale/shift [D]).
    go/x [..,D]; scale [D]. Returns d_x (x's shape), d_scale [D], d_shift [D].
    `shift` is not needed for any grad (o is linear in shift)."""
    # BF16/F16 storage path: cast up, run F32 interior, cast grads down.
    if x.dtype() != STDtype.F32 or go.dtype() != STDtype.F32 or scale.dtype() != STDtype.F32:
        var out_dt = x.dtype()
        var go32 = cast_tensor(go, STDtype.F32, ctx)
        var x32 = cast_tensor(x, STDtype.F32, ctx)
        var s32 = cast_tensor(scale, STDtype.F32, ctx)
        var g32 = modulate_backward(go32, x32, s32, ctx, compute_param_grads)
        var dx_dn = cast_tensor(g32.d_x^, out_dt, ctx)
        var ds_dn = cast_tensor(g32.d_scale^, out_dt, ctx)
        var dsh_dn = cast_tensor(g32.d_shift^, out_dt, ctx)
        return ModulateBackward(dx_dn^, ds_dn^, dsh_dn^)
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var param_nbytes = scale.nbytes()
    if not compute_param_grads:
        param_nbytes = 0
    var ds_buf = ctx.enqueue_create_buffer[DType.uint8](param_nbytes)
    var dsh_buf = ctx.enqueue_create_buffer[DType.uint8](param_nbytes)

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))

    var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        go.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), x_rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), v_rl
    )
    var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dx_buf.unsafe_ptr().bitcast[Float32](), x_rl
    )

    var total = rows * d
    var dx_grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_modulate_bwd_dx_kernel, _modulate_bwd_dx_kernel](
        GO, S, DX, rows, d, grid_dim=dx_grid, block_dim=_BLOCK
    )
    var param_shape = List[Int]()
    if compute_param_grads:
        var DS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            ds_buf.unsafe_ptr().bitcast[Float32](), v_rl
        )
        var DSH = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            dsh_buf.unsafe_ptr().bitcast[Float32](), v_rl
        )
        var p_grid = (d + _BLOCK - 1) // _BLOCK
        ctx.enqueue_function[_modulate_bwd_param_kernel, _modulate_bwd_param_kernel](
            GO, X, DS, DSH, rows, d, grid_dim=p_grid, block_dim=_BLOCK
        )
        param_shape = scale.shape()
    else:
        param_shape.append(0)
    return ModulateBackward(
        Tensor(dx_buf^, xshape.copy(), STDtype.F32),
        Tensor(ds_buf^, param_shape.copy(), STDtype.F32),
        Tensor(dsh_buf^, param_shape^, STDtype.F32),
    )
