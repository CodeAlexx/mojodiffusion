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
from serenitymojo.ops.cast import cast_tensor


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── modulate ───────────────────────────────────────────────────────────────
# rows_per_vec: rows sharing one [D] vector. Single-vec callers pass `rows`
# (r // rows == 0 ⇒ identical math to the original kernel). BATCH-2+ callers
# pass rows//B with s/sh holding B stacked [D] vectors ([B·D] flat) — each
# sample's row range gets its own modulation (per-sample timestep adaLN).
def _modulate_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    sh: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
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
        var xv = rebind[Scalar[DType.float32]](x[r, c])
        var sv = rebind[Scalar[DType.float32]](s[vo + c])
        var shv = rebind[Scalar[DType.float32]](sh[vo + c])
        o[r, c] = rebind[o.element_type]((1.0 + sv) * xv + shv)


def _modulate_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    s: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    sh: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
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
        var xv = rebind[Scalar[DType.bfloat16]](x[r, c]).cast[DType.float32]()
        var sv = rebind[Scalar[DType.bfloat16]](s[vo + c]).cast[DType.float32]()
        var shv = rebind[Scalar[DType.bfloat16]](sh[vo + c]).cast[DType.float32]()
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
    rows_per_vec: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var vo = (r // rows_per_vec) * cols
        var xv = rebind[Scalar[DType.float16]](x[r, c]).cast[DType.float32]()
        var sv = rebind[Scalar[DType.float16]](s[vo + c]).cast[DType.float32]()
        var shv = rebind[Scalar[DType.float16]](sh[vo + c]).cast[DType.float32]()
        o[r, c] = rebind[o.element_type](
            ((1.0 + sv) * xv + shv).cast[DType.float16]()
        )


def modulate(
    x: Tensor, scale: Tensor, shift: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """modulate(x, scale, shift) = (1 + scale) * x + shift.

    x:     [..., D]   (any compute dtype; leading dims flattened to rows)
    scale: [D]        (per-channel; same dtype as x, or modulation storage dtype)
    shift: [D]        (per-channel; same dtype as x, or modulation storage dtype)
    returns [..., D]  (x's dtype; F32 math).
    """
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("modulate: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    var sshape = scale.shape()
    var shshape = shift.shape()
    # scale/shift: [D] (one vec, all rows) or [B, D] (B stacked vecs; rows
    # split evenly into B contiguous ranges — per-sample adaLN for batch>1).
    var nvec = 1
    if len(sshape) == 2 and sshape[1] == d:
        nvec = sshape[0]
    elif len(sshape) != 1 or sshape[0] != d:
        raise Error("modulate: scale must be [D] or [B, D]")
    if len(shshape) == 2 and shshape[1] == d:
        if shshape[0] != nvec:
            raise Error("modulate: shift vec count != scale vec count")
    elif len(shshape) != 1 or shshape[0] != d:
        raise Error("modulate: shift must be [D] or [B, D]")
    else:
        if nvec != 1:
            raise Error("modulate: shift must be [B, D] when scale is [B, D]")
    if x.dtype() != scale.dtype():
        var compute_scale = cast_tensor(scale, x.dtype(), ctx)
        if x.dtype() != shift.dtype():
            var compute_shift = cast_tensor(shift, x.dtype(), ctx)
            return modulate(x, compute_scale^, compute_shift^, ctx)
        return modulate(x, compute_scale^, shift, ctx)
    if x.dtype() != shift.dtype():
        var compute_shift = cast_tensor(shift, x.dtype(), ctx)
        return modulate(x, scale, compute_shift^, ctx)
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    if rows % nvec != 0:
        raise Error("modulate: rows not divisible by scale vec count")
    var rows_per_vec = rows // nvec

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nvec * d))
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
            X, S, SH, O, rows, d, rows_per_vec, grid_dim=grid, block_dim=_BLOCK
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
            X, S, SH, O, rows, d, rows_per_vec, grid_dim=grid, block_dim=_BLOCK
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
            X, S, SH, O, rows, d, rows_per_vec, grid_dim=grid, block_dim=_BLOCK
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
    rows_per_vec: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var vo = (r // rows_per_vec) * cols
        var xv = rebind[Scalar[DType.float32]](x[r, c])
        var gv = rebind[Scalar[DType.float32]](g[vo + c])
        var yv = rebind[Scalar[DType.float32]](y[r, c])
        o[r, c] = rebind[o.element_type](xv + gv * yv)


def _resgate_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    y: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
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
        var xv = rebind[Scalar[DType.bfloat16]](x[r, c]).cast[DType.float32]()
        var gv = rebind[Scalar[DType.bfloat16]](g[vo + c]).cast[DType.float32]()
        var yv = rebind[Scalar[DType.bfloat16]](y[r, c]).cast[DType.float32]()
        o[r, c] = rebind[o.element_type]((xv + gv * yv).cast[DType.bfloat16]())


def _resgate_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    g: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    y: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN2, MutAnyOrigin],
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
        var xv = rebind[Scalar[DType.float16]](x[r, c]).cast[DType.float32]()
        var gv = rebind[Scalar[DType.float16]](g[vo + c]).cast[DType.float32]()
        var yv = rebind[Scalar[DType.float16]](y[r, c]).cast[DType.float32]()
        o[r, c] = rebind[o.element_type]((xv + gv * yv).cast[DType.float16]())


def residual_gate(
    x: Tensor, gate: Tensor, y: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """residual_gate(x, gate, y) = x + gate * y.

    x:    [..., D]   (any compute dtype)
    gate: [D]        (per-channel; same dtype as x, or modulation storage dtype)
    y:    [..., D]   (same shape as x; cast to x dtype if needed)
    returns [..., D] (x's dtype; F32 math).
    """
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("residual_gate: x must have rank >= 1")
    var d = xshape[len(xshape) - 1]
    var gshape = gate.shape()
    # gate: [D] (one vec) or [B, D] (B stacked vecs, rows split evenly).
    var nvec = 1
    if len(gshape) == 2 and gshape[1] == d:
        nvec = gshape[0]
    elif len(gshape) != 1 or gshape[0] != d:
        raise Error("residual_gate: gate must be [D] or [B, D]")
    if x.numel() != y.numel():
        raise Error("residual_gate: x/y numel mismatch")
    if x.dtype() != gate.dtype():
        var compute_gate = cast_tensor(gate, x.dtype(), ctx)
        if x.dtype() != y.dtype():
            var compute_y = cast_tensor(y, x.dtype(), ctx)
            return residual_gate(x, compute_gate^, compute_y^, ctx)
        return residual_gate(x, compute_gate^, y, ctx)
    if x.dtype() != y.dtype():
        var compute_y = cast_tensor(y, x.dtype(), ctx)
        return residual_gate(x, gate, compute_y^, ctx)
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    if rows % nvec != 0:
        raise Error("residual_gate: rows not divisible by gate vec count")
    var rows_per_vec = rows // nvec

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var v_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nvec * d))
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
            X, G, Y, O, rows, d, rows_per_vec, grid_dim=grid, block_dim=_BLOCK
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
            X, G, Y, O, rows, d, rows_per_vec, grid_dim=grid, block_dim=_BLOCK
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
            X, G, Y, O, rows, d, rows_per_vec, grid_dim=grid, block_dim=_BLOCK
        )
    # TIER2-SYNC-REMOVED: single-stream ordering; downstream .to_host() syncs.
    return Tensor(out_buf^, xshape.copy(), x.dtype())
