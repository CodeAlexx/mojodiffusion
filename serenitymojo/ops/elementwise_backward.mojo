# ops/elementwise_backward.mojo — BACKWARD for modulate (DiT AdaLN).
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
# (one thread per column accumulating over rows). Storage dtype is preserved;
# each kernel casts scalar elements to F32 for math and writes gradients back to
# the input dtype.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd_v2.step_slab import StepSlab


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# d_x[r,c] = go[r,c] * (1 + scale[c])  (elementwise; scale broadcast per channel)
def _modulate_bwd_dx_kernel[dtype: DType](
    go: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    s: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    dx: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    rows: Int,
    cols: Int,
    rows_per_vec: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var vo = (r // rows_per_vec) * cols
        var gov = rebind[Scalar[dtype]](go[r, c]).cast[DType.float32]()
        var sv = rebind[Scalar[dtype]](s[vo + c]).cast[DType.float32]()
        dx[r, c] = rebind[dx.element_type]((gov * (1.0 + sv)).cast[dtype]())


# d_scale[c] = sum_r go[r,c]*x[r,c] ; d_shift[c] = sum_r go[r,c]  (one thread/col)
def _modulate_bwd_param_kernel[dtype: DType](
    go: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    x: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    dscale: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    dshift: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var col = Int(global_idx.x)
    if col >= cols:
        return
    var acc_s: Float32 = 0.0
    var acc_sh: Float32 = 0.0
    for r in range(rows):
        var gov = rebind[Scalar[dtype]](go[r, col]).cast[DType.float32]()
        var xv = rebind[Scalar[dtype]](x[r, col]).cast[DType.float32]()
        acc_s += gov * xv
        acc_sh += gov
    dscale[col] = rebind[dscale.element_type](acc_s.cast[dtype]())
    dshift[col] = rebind[dshift.element_type](acc_sh.cast[dtype]())


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
    if x.dtype() != go.dtype() or x.dtype() != scale.dtype():
        raise Error("modulate_backward: go/x/scale dtype mismatch")
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    # scale: [D] or [B, D] (per-sample adaLN; rows split evenly). Param grads
    # are per-vec reductions — not implemented for B>1 (LoRA training discards
    # them); callers must pass compute_param_grads=False.
    var sshape = scale.shape()
    var nvec = 1
    if len(sshape) == 2 and sshape[1] == d:
        nvec = sshape[0]
        if compute_param_grads:
            raise Error(
                "modulate_backward: param grads unsupported for [B, D] scale"
            )
        if rows % nvec != 0:
            raise Error("modulate_backward: rows not divisible by vec count")
    elif len(sshape) != 1 or sshape[0] != d:
        raise Error("modulate_backward: scale must be [D] or [B, D]")
    var rows_per_vec = rows // nvec

    var dx_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var param_nbytes = scale.nbytes()
    if not compute_param_grads:
        param_nbytes = 0
    var ds_buf = ctx.enqueue_create_buffer[DType.uint8](param_nbytes)
    var dsh_buf = ctx.enqueue_create_buffer[DType.uint8](param_nbytes)

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nvec * d))

    var total = rows * d
    var dx_grid = (total + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    var param_shape = List[Int]()
    var p_grid = (d + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            go.buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            scale.buf.unsafe_ptr().bitcast[Float32](), v_rl)
        var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float32](), x_rl)
        ctx.enqueue_function[
            _modulate_bwd_dx_kernel[DType.float32],
            _modulate_bwd_dx_kernel[DType.float32],
        ](GO, S, DX, rows, d, rows_per_vec, grid_dim=dx_grid, block_dim=_BLOCK)
        if compute_param_grads:
            var DS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                ds_buf.unsafe_ptr().bitcast[Float32](), v_rl)
            var DSH = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                dsh_buf.unsafe_ptr().bitcast[Float32](), v_rl)
            ctx.enqueue_function[
                _modulate_bwd_param_kernel[DType.float32],
                _modulate_bwd_param_kernel[DType.float32],
            ](GO, X, DS, DSH, rows, d, grid_dim=p_grid, block_dim=_BLOCK)
            param_shape = scale.shape()
        else:
            param_shape.append(0)
    elif dt == DType.bfloat16:
        var GO = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            go.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            scale.buf.unsafe_ptr().bitcast[BFloat16](), v_rl)
        var DX = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        ctx.enqueue_function[
            _modulate_bwd_dx_kernel[DType.bfloat16],
            _modulate_bwd_dx_kernel[DType.bfloat16],
        ](GO, S, DX, rows, d, rows_per_vec, grid_dim=dx_grid, block_dim=_BLOCK)
        if compute_param_grads:
            var DS = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                ds_buf.unsafe_ptr().bitcast[BFloat16](), v_rl)
            var DSH = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                dsh_buf.unsafe_ptr().bitcast[BFloat16](), v_rl)
            ctx.enqueue_function[
                _modulate_bwd_param_kernel[DType.bfloat16],
                _modulate_bwd_param_kernel[DType.bfloat16],
            ](GO, X, DS, DSH, rows, d, grid_dim=p_grid, block_dim=_BLOCK)
            param_shape = scale.shape()
        else:
            param_shape.append(0)
    else:
        var GO = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            go.buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var S = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            scale.buf.unsafe_ptr().bitcast[Float16](), v_rl)
        var DX = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float16](), x_rl)
        ctx.enqueue_function[
            _modulate_bwd_dx_kernel[DType.float16],
            _modulate_bwd_dx_kernel[DType.float16],
        ](GO, S, DX, rows, d, rows_per_vec, grid_dim=dx_grid, block_dim=_BLOCK)
        if compute_param_grads:
            var DS = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                ds_buf.unsafe_ptr().bitcast[Float16](), v_rl)
            var DSH = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                dsh_buf.unsafe_ptr().bitcast[Float16](), v_rl)
            ctx.enqueue_function[
                _modulate_bwd_param_kernel[DType.float16],
                _modulate_bwd_param_kernel[DType.float16],
            ](GO, X, DS, DSH, rows, d, grid_dim=p_grid, block_dim=_BLOCK)
            param_shape = scale.shape()
        else:
            param_shape.append(0)
    ctx.synchronize()
    return ModulateBackward(
        Tensor(dx_buf^, xshape.copy(), x.dtype()),
        Tensor(ds_buf^, param_shape.copy(), scale.dtype()),
        Tensor(dsh_buf^, param_shape^, scale.dtype()),
    )


def modulate_backward_slab(
    go: Tensor, x: Tensor, scale: Tensor, ctx: DeviceContext,
    mut slab: StepSlab,
    compute_param_grads: Bool = True,
) raises -> ModulateBackward:
    """StepSlab variant of `modulate_backward` (this file :89) —
    byte-identical math (same kernels, same launch params, same sync); ONLY
    the allocation source changes (autograd_v2 contract C8, Phase P4)."""
    if x.dtype() != go.dtype() or x.dtype() != scale.dtype():
        raise Error("modulate_backward: go/x/scale dtype mismatch")
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    var sshape = scale.shape()
    var nvec = 1
    if len(sshape) == 2 and sshape[1] == d:
        nvec = sshape[0]
        if compute_param_grads:
            raise Error(
                "modulate_backward: param grads unsupported for [B, D] scale"
            )
        if rows % nvec != 0:
            raise Error("modulate_backward: rows not divisible by vec count")
    elif len(sshape) != 1 or sshape[0] != d:
        raise Error("modulate_backward: scale must be [D] or [B, D]")
    var rows_per_vec = rows // nvec

    var dx_buf = slab.alloc(x.nbytes())
    var param_nbytes = scale.nbytes()
    if not compute_param_grads:
        param_nbytes = 0
    var ds_buf = slab.alloc(param_nbytes)
    var dsh_buf = slab.alloc(param_nbytes)

    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nvec * d))

    var total = rows * d
    var dx_grid = (total + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    var param_shape = List[Int]()
    var p_grid = (d + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var GO = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            go.buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl)
        var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            scale.buf.unsafe_ptr().bitcast[Float32](), v_rl)
        var DX = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float32](), x_rl)
        ctx.enqueue_function[
            _modulate_bwd_dx_kernel[DType.float32],
            _modulate_bwd_dx_kernel[DType.float32],
        ](GO, S, DX, rows, d, rows_per_vec, grid_dim=dx_grid, block_dim=_BLOCK)
        if compute_param_grads:
            var DS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                ds_buf.unsafe_ptr().bitcast[Float32](), v_rl)
            var DSH = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                dsh_buf.unsafe_ptr().bitcast[Float32](), v_rl)
            ctx.enqueue_function[
                _modulate_bwd_param_kernel[DType.float32],
                _modulate_bwd_param_kernel[DType.float32],
            ](GO, X, DS, DSH, rows, d, grid_dim=p_grid, block_dim=_BLOCK)
            param_shape = scale.shape()
        else:
            param_shape.append(0)
    elif dt == DType.bfloat16:
        var GO = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            go.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            scale.buf.unsafe_ptr().bitcast[BFloat16](), v_rl)
        var DX = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
        ctx.enqueue_function[
            _modulate_bwd_dx_kernel[DType.bfloat16],
            _modulate_bwd_dx_kernel[DType.bfloat16],
        ](GO, S, DX, rows, d, rows_per_vec, grid_dim=dx_grid, block_dim=_BLOCK)
        if compute_param_grads:
            var DS = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                ds_buf.unsafe_ptr().bitcast[BFloat16](), v_rl)
            var DSH = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                dsh_buf.unsafe_ptr().bitcast[BFloat16](), v_rl)
            ctx.enqueue_function[
                _modulate_bwd_param_kernel[DType.bfloat16],
                _modulate_bwd_param_kernel[DType.bfloat16],
            ](GO, X, DS, DSH, rows, d, grid_dim=p_grid, block_dim=_BLOCK)
            param_shape = scale.shape()
        else:
            param_shape.append(0)
    else:
        var GO = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            go.buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl)
        var S = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            scale.buf.unsafe_ptr().bitcast[Float16](), v_rl)
        var DX = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            dx_buf.unsafe_ptr().bitcast[Float16](), x_rl)
        ctx.enqueue_function[
            _modulate_bwd_dx_kernel[DType.float16],
            _modulate_bwd_dx_kernel[DType.float16],
        ](GO, S, DX, rows, d, rows_per_vec, grid_dim=dx_grid, block_dim=_BLOCK)
        if compute_param_grads:
            var DS = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                ds_buf.unsafe_ptr().bitcast[Float16](), v_rl)
            var DSH = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                dsh_buf.unsafe_ptr().bitcast[Float16](), v_rl)
            ctx.enqueue_function[
                _modulate_bwd_param_kernel[DType.float16],
                _modulate_bwd_param_kernel[DType.float16],
            ](GO, X, DS, DSH, rows, d, grid_dim=p_grid, block_dim=_BLOCK)
            param_shape = scale.shape()
        else:
            param_shape.append(0)
    # P5-CAPTURE-SYNC-REMOVED (C9): single-stream ordering (TIER2 precedent,
    # ops/attention.mojo); no sync inside a captured region.
    return ModulateBackward(
        Tensor(dx_buf^, xshape.copy(), x.dtype()),
        Tensor(ds_buf^, param_shape.copy(), scale.dtype()),
        Tensor(dsh_buf^, param_shape^, scale.dtype()),
    )
