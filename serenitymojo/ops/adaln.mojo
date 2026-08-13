# ops/adaln.mojo — GPU AdaLN kernels for LingBot-Video blocks (shared).
#
# Extracted VERBATIM from models/lingbotvideo/dense.mojo (the parity-gated dense
# arm, cos 0.9998) so the MoE block (backbone.mojo) can reuse them without a
# circular import (dense imports backbone). Same F32 math, same F32->bf16 store
# boundaries as the old host AdaLN — the port is numerically equivalent; only the
# host round-trip (S*12288 downloads + ~250M host element-ops/block at T2V S) is
# removed. dense.mojo keeps its own identical copies (untouched); dedupe later.
#
# Reference block math:
#   mod = temb6 + scale_shift_table (FP32); chunk 6; gate=tanh, scale=1+scale
#   attn_in = (norm(x) * scale + shift).to(bf16)
#   x = x + (gate * norm(sub_out)).to(x.dtype)
# temb is a SINGLE modulation row (timestep per-batch -> every token row identical),
# read with the right chunk column offsets (shift/scale/gate at 0..5 * hdim).

from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import tanh
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.norm import rms_norm


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _EW_TPB = 256


# attn/ffn input modulate: out = bf16( f32(normed) * (1 + (temb[scale_off+j] +
# sst[scale_off+j])) + (temb[shift_off+j] + sst[shift_off+j]) ).
def _adaln_modulate_kernel(
    normed: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],  # [S, H] bf16
    temb: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],     # [K] f32 row
    sst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],      # [K] f32 table
    outp: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],    # [S, H] bf16
    s_w: Int32,
    h_w: Int32,
    shift_off_w: Int64,
    scale_off_w: Int64,
):
    var s = Int(s_w)
    var h = Int(h_w)
    var shift_off = Int(shift_off_w)
    var scale_off = Int(scale_off_w)
    var idx = Int(global_idx.x)
    if idx < s * h:
        var t = idx // h
        var j = idx % h
        var nv = rebind[Scalar[DType.bfloat16]](normed[t, j]).cast[DType.float32]()
        var sc = rebind[Scalar[DType.float32]](temb[scale_off + j]) + rebind[
            Scalar[DType.float32]
        ](sst[scale_off + j])
        var shv = rebind[Scalar[DType.float32]](temb[shift_off + j]) + rebind[
            Scalar[DType.float32]
        ](sst[shift_off + j])
        var v = nv * (Float32(1.0) + sc) + shv
        outp[t, j] = rebind[outp.element_type](v.cast[DType.bfloat16]())


def adaln_modulate(
    normed: Tensor,     # [.., S, H] bf16
    temb_row: Tensor,   # [1, K] f32
    sst: Tensor,        # [1, K] or [K] f32
    s: Int,
    h: Int,
    shift_off: Int,
    scale_off: Int,
    var out_shape: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](s * h * 2)
    var n_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](s, h))
    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](temb_row.numel()))
    var N = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(normed.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=n_rl,
    )
    var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(temb_row.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=t_rl,
    )
    var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sst.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=t_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=n_rl,
    )
    var grid = (s * h + _EW_TPB - 1) // _EW_TPB
    ctx.enqueue_function[_adaln_modulate_kernel](
        N, T, C, O, Int32(s), Int32(h), Int64(shift_off), Int64(scale_off),
        grid_dim=grid, block_dim=_EW_TPB,
    )
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


# gated residual: out = bf16( f32(x) + f32(bf16( tanh(temb[gate_off+j] +
# sst[gate_off+j]) * f32(normed) )) ).
def _adaln_gate_residual_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],       # [S, H] bf16
    normed: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],  # [S, H] bf16
    temb: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],     # [K] f32 row
    sst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],      # [K] f32 table
    outp: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],    # [S, H] bf16
    s_w: Int32,
    h_w: Int32,
    gate_off_w: Int64,
):
    var s = Int(s_w)
    var h = Int(h_w)
    var gate_off = Int(gate_off_w)
    var idx = Int(global_idx.x)
    if idx < s * h:
        var t = idx // h
        var j = idx % h
        var g = tanh(
            rebind[Scalar[DType.float32]](temb[gate_off + j])
            + rebind[Scalar[DType.float32]](sst[gate_off + j])
        )
        var nv = rebind[Scalar[DType.bfloat16]](normed[t, j]).cast[DType.float32]()
        var d = (g * nv).cast[DType.bfloat16]()
        var xv = rebind[Scalar[DType.bfloat16]](x[t, j]).cast[DType.float32]()
        var res = xv + d.cast[DType.float32]()
        outp[t, j] = rebind[outp.element_type](res.cast[DType.bfloat16]())


def adaln_gate_residual(
    x: Tensor,          # [.., S, H] bf16
    normed: Tensor,     # [.., S, H] bf16
    temb_row: Tensor,   # [1, K] f32
    sst: Tensor,        # [1, K] f32
    s: Int,
    h: Int,
    gate_off: Int,
    var out_shape: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](s * h * 2)
    var n_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](s, h))
    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](temb_row.numel()))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=n_rl,
    )
    var N = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(normed.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=n_rl,
    )
    var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(temb_row.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=t_rl,
    )
    var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sst.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=t_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=n_rl,
    )
    var grid = (s * h + _EW_TPB - 1) // _EW_TPB
    ctx.enqueue_function[_adaln_gate_residual_kernel](
        X, N, T, C, O, Int32(s), Int32(h), Int64(gate_off), grid_dim=grid, block_dim=_EW_TPB
    )
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


# LingBotVideoRMSNorm on device, mirroring `_rms_norm_bf16_host` EXACTLY minus the
# host round trip: upcast bf16 x -> F32, rms_norm with the F32 gamma (F32 accum),
# then .to(bf16).
def rms_norm_bf16_dev(
    x: Tensor, gamma_f32: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    var xf = cast_tensor(x, STDtype.F32, ctx)
    var nf = rms_norm(xf, gamma_f32, eps, ctx)
    return cast_tensor(nf, STDtype.BF16, ctx)
