# ops/elementwise.mojo — modulate, residual_gate (DiT AdaLN building blocks).
#
#   modulate(x, scale, shift)   = (1 + scale) * x + shift
#   residual_gate(x, gate, y)   = x + gate * y
#
# These are the AdaLN-Zero modulation primitives used by DiT-style blocks
# (FLUX / Klein / Z-Image / Qwen-Image). `x` is [..., D] (leading dims are
# tokens, last dim D is channels). scale/shift/gate are per-channel vectors [D]
# broadcast across all leading rows (the standard adaLN layout where the
# modulation comes from a timestep/condition MLP, one value per channel).
#
# Computed as flat (row, col) over the last dim: out[r, c] uses param[c].
# F32 math; store casts back to storage dtype.
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
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── modulate ───────────────────────────────────────────────────────────────
def _modulate_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    sh: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var xv = rebind[Scalar[DType.float32]](x[r, c])
        var sv = rebind[Scalar[DType.float32]](s[c])
        var shv = rebind[Scalar[DType.float32]](sh[c])
        o[r, c] = rebind[o.element_type]((1.0 + sv) * xv + shv)


def _modulate_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    s: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    sh: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var xv = rebind[Scalar[DType.bfloat16]](x[r, c]).cast[DType.float32]()
        var sv = rebind[Scalar[DType.bfloat16]](s[c]).cast[DType.float32]()
        var shv = rebind[Scalar[DType.bfloat16]](sh[c]).cast[DType.float32]()
        o[r, c] = rebind[o.element_type](
            ((1.0 + sv) * xv + shv).cast[DType.bfloat16]()
        )


def _modulate_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    s: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    sh: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var xv = rebind[Scalar[DType.float16]](x[r, c]).cast[DType.float32]()
        var sv = rebind[Scalar[DType.float16]](s[c]).cast[DType.float32]()
        var shv = rebind[Scalar[DType.float16]](sh[c]).cast[DType.float32]()
        o[r, c] = rebind[o.element_type](
            ((1.0 + sv) * xv + shv).cast[DType.float16]()
        )


def modulate(
    x: Tensor, scale: Tensor, shift: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """modulate(x, scale, shift) = (1 + scale) * x + shift.

    x:     [..., D]   (any compute dtype; leading dims flattened to rows)
    scale: [D]        (per-channel; same dtype as x)
    shift: [D]        (per-channel; same dtype as x)
    returns [..., D]  (x's dtype; F32 math).
    """
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("modulate: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    var sshape = scale.shape()
    var shshape = shift.shape()
    if len(sshape) != 1 or sshape[0] != d:
        raise Error("modulate: scale must be [D]")
    if len(shshape) != 1 or shshape[0] != d:
        raise Error("modulate: shift must be [D]")
    if x.dtype() != scale.dtype() or x.dtype() != shift.dtype():
        raise Error("modulate: x/scale/shift dtype mismatch")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))
    var total = rows * d
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            scale.buf.unsafe_ptr().bitcast[Float32](), v_rl
        )
        var SH = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            shift.buf.unsafe_ptr().bitcast[Float32](), v_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[_modulate_kernel_f32, _modulate_kernel_f32](
            X, S, SH, O, rows, d, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            scale.buf.unsafe_ptr().bitcast[BFloat16](), v_rl
        )
        var SH = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            shift.buf.unsafe_ptr().bitcast[BFloat16](), v_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[_modulate_kernel_bf16, _modulate_kernel_bf16](
            X, S, SH, O, rows, d, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var S = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            scale.buf.unsafe_ptr().bitcast[Float16](), v_rl
        )
        var SH = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            shift.buf.unsafe_ptr().bitcast[Float16](), v_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[_modulate_kernel_f16, _modulate_kernel_f16](
            X, S, SH, O, rows, d, grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, xshape.copy(), x.dtype())


# ── residual_gate ──────────────────────────────────────────────────────────
def _resgate_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    y: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var xv = rebind[Scalar[DType.float32]](x[r, c])
        var gv = rebind[Scalar[DType.float32]](g[c])
        var yv = rebind[Scalar[DType.float32]](y[r, c])
        o[r, c] = rebind[o.element_type](xv + gv * yv)


def _resgate_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    y: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var xv = rebind[Scalar[DType.bfloat16]](x[r, c]).cast[DType.float32]()
        var gv = rebind[Scalar[DType.bfloat16]](g[c]).cast[DType.float32]()
        var yv = rebind[Scalar[DType.bfloat16]](y[r, c]).cast[DType.float32]()
        o[r, c] = rebind[o.element_type]((xv + gv * yv).cast[DType.bfloat16]())


def _resgate_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    y: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var xv = rebind[Scalar[DType.float16]](x[r, c]).cast[DType.float32]()
        var gv = rebind[Scalar[DType.float16]](g[c]).cast[DType.float32]()
        var yv = rebind[Scalar[DType.float16]](y[r, c]).cast[DType.float32]()
        o[r, c] = rebind[o.element_type]((xv + gv * yv).cast[DType.float16]())


def residual_gate(
    x: Tensor, gate: Tensor, y: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """residual_gate(x, gate, y) = x + gate * y.

    x:    [..., D]   (any compute dtype)
    gate: [D]        (per-channel; same dtype as x)
    y:    [..., D]   (same shape/dtype as x)
    returns [..., D] (x's dtype; F32 math).
    """
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("residual_gate: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    var gshape = gate.shape()
    if len(gshape) != 1 or gshape[0] != d:
        raise Error("residual_gate: gate must be [D]")
    if x.numel() != y.numel():
        raise Error("residual_gate: x/y numel mismatch")
    if x.dtype() != gate.dtype() or x.dtype() != y.dtype():
        raise Error("residual_gate: x/gate/y dtype mismatch")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))
    var total = rows * d
    var grid = (total + _BLOCK - 1) // _BLOCK

    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            gate.buf.unsafe_ptr().bitcast[Float32](), v_rl
        )
        var Y = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            y.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[_resgate_kernel_f32, _resgate_kernel_f32](
            X, G, Y, O, rows, d, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            gate.buf.unsafe_ptr().bitcast[BFloat16](), v_rl
        )
        var Y = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            y.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[_resgate_kernel_bf16, _resgate_kernel_bf16](
            X, G, Y, O, rows, d, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            gate.buf.unsafe_ptr().bitcast[Float16](), v_rl
        )
        var Y = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            y.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[_resgate_kernel_f16, _resgate_kernel_f16](
            X, G, Y, O, rows, d, grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, xshape.copy(), x.dtype())
