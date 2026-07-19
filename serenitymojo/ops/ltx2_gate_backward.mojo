# ops/ltx2_gate_backward.mojo — device kernel for the LTX-2.3 per-head attention
# gate gradient, replacing the HOST round-trip in _av_attention_bwd
# (ltx2_av_backward.mojo:357-366 — d_att_g.to_host + acts.att_flat.to_host ->
# host F32 dot -> from_host). That round-trip is the L5 CUDA-graph-capture blocker
# (has_gate confirmed True on real video blocks). This is the ONLY host math in
# the attention backward; everything else (d_att_flat, sigmoid_backward,
# linear_backward_dx) is already device.
#
# ORACLE (the host math this reproduces EXACTLY, ltx2_av_backward.mojo:359-366):
#   d_gates[s,h] = sum_{d in DH} d_att_g[s, h*DH+d] * att_flat[s, h*DH+d]
# F32 accumulation, sequential d=0..DH-1 (one thread per (s,h) output keeps that
# exact order for bit-match); inputs upcast to F32 (bf16->F32 exact); result cast
# to the storage dtype. Gate: ops/tests/ltx2_gate_backward_parity.mojo (bit-level,
# bf16 AND F32 storage, real shapes).
#
# Mojo 1.0.0b1, NVIDIA GPU. `def` not `fn`.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.autograd_v2.step_slab import StepSlab

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# One thread per (s,h) output; sequential F32 dot over DH (matches the host order).
def _gate_dgates_kernel[dtype: DType](
    dag: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [SQ*inner]  d_att_g
    af: LayoutTensor[dtype, _DYN1, MutAnyOrigin],    # [SQ*inner]  att_flat
    dg: LayoutTensor[dtype, _DYN1, MutAnyOrigin],    # [SQ*H]      d_gates out
    sq: Int, h: Int, dh: Int,
):
    var idx = Int(global_idx.x)
    var total = sq * h
    if idx < total:
        var s = idx // h
        var head = idx % h
        var inner = h * dh
        var base = s * inner + head * dh
        var acc = Float32(0.0)
        for d in range(dh):
            var v1 = rebind[Scalar[dtype]](dag[base + d]).cast[DType.float32]()
            var v2 = rebind[Scalar[dtype]](af[base + d]).cast[DType.float32]()
            acc += v1 * v2
        dg[idx] = rebind[dg.element_type](acc.cast[dtype]())


def ltx2_gate_dgates(
    d_att_g: Tensor, att_flat: Tensor, sq: Int, h: Int, dh: Int, ctx: DeviceContext
) raises -> Tensor:
    """Device d_gates[s,h] = sum_d d_att_g[s,h*DH+d]*att_flat[s,h*DH+d]. d_att_g /
    att_flat are [SQ, H*DH] (any of F32/BF16/F16, same dtype); returns [SQ, H] in
    that dtype. NO host round-trip (capture-safe)."""
    if d_att_g.dtype() != att_flat.dtype():
        raise Error("ltx2_gate_dgates: d_att_g/att_flat dtype mismatch")
    var inner = h * dh
    var in_n = sq * inner
    var out_n = sq * h
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * d_att_g.dtype().byte_size())
    var in_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](in_n))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var dt = d_att_g.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var DAG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            d_att_g.buf.unsafe_ptr().bitcast[Float32](), in_rl)
        var AF = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            att_flat.buf.unsafe_ptr().bitcast[Float32](), in_rl)
        var DG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl)
        ctx.enqueue_function[
            _gate_dgates_kernel[DType.float32], _gate_dgates_kernel[DType.float32],
        ](DAG, AF, DG, sq, h, dh, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var DAG = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            d_att_g.buf.unsafe_ptr().bitcast[BFloat16](), in_rl)
        var AF = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            att_flat.buf.unsafe_ptr().bitcast[BFloat16](), in_rl)
        var DG = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl)
        ctx.enqueue_function[
            _gate_dgates_kernel[DType.bfloat16], _gate_dgates_kernel[DType.bfloat16],
        ](DAG, AF, DG, sq, h, dh, grid_dim=grid, block_dim=_BLOCK)
    else:
        var DAG = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            d_att_g.buf.unsafe_ptr().bitcast[Float16](), in_rl)
        var AF = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            att_flat.buf.unsafe_ptr().bitcast[Float16](), in_rl)
        var DG = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl)
        ctx.enqueue_function[
            _gate_dgates_kernel[DType.float16], _gate_dgates_kernel[DType.float16],
        ](DAG, AF, DG, sq, h, dh, grid_dim=grid, block_dim=_BLOCK)
    var sh = [sq, h]
    return Tensor(out_buf^, sh^, d_att_g.dtype())


def ltx2_gate_dgates_slab(
    d_att_g: Tensor, att_flat: Tensor, sq: Int, h: Int, dh: Int,
    ctx: DeviceContext, mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of ltx2_gate_dgates (C8, L5 capture path): BYTE-IDENTICAL
    recording — same kernel, same F32-sequential accumulation order — only the
    output buffer comes from slab.alloc (a ring sub-view; caller copies out before
    rewind) instead of enqueue_create_buffer. Gated bit-exact vs the non-slab twin."""
    if d_att_g.dtype() != att_flat.dtype():
        raise Error("ltx2_gate_dgates_slab: d_att_g/att_flat dtype mismatch")
    var inner = h * dh
    var in_n = sq * inner
    var out_n = sq * h
    var out_buf = slab.alloc(out_n * d_att_g.dtype().byte_size())
    var in_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](in_n))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var dt = d_att_g.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var DAG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            d_att_g.buf.unsafe_ptr().bitcast[Float32](), in_rl)
        var AF = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            att_flat.buf.unsafe_ptr().bitcast[Float32](), in_rl)
        var DG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl)
        ctx.enqueue_function[
            _gate_dgates_kernel[DType.float32], _gate_dgates_kernel[DType.float32],
        ](DAG, AF, DG, sq, h, dh, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var DAG = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            d_att_g.buf.unsafe_ptr().bitcast[BFloat16](), in_rl)
        var AF = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            att_flat.buf.unsafe_ptr().bitcast[BFloat16](), in_rl)
        var DG = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl)
        ctx.enqueue_function[
            _gate_dgates_kernel[DType.bfloat16], _gate_dgates_kernel[DType.bfloat16],
        ](DAG, AF, DG, sq, h, dh, grid_dim=grid, block_dim=_BLOCK)
    else:
        var DAG = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            d_att_g.buf.unsafe_ptr().bitcast[Float16](), in_rl)
        var AF = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            att_flat.buf.unsafe_ptr().bitcast[Float16](), in_rl)
        var DG = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl)
        ctx.enqueue_function[
            _gate_dgates_kernel[DType.float16], _gate_dgates_kernel[DType.float16],
        ](DAG, AF, DG, sq, h, dh, grid_dim=grid, block_dim=_BLOCK)
    var sh = [sq, h]
    return Tensor(out_buf^, sh^, d_att_g.dtype())
