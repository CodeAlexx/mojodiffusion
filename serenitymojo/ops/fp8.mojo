# ops/fp8.mojo — FP8 E4M3 → BF16 dequantization (pure Mojo + MAX GPU port).
#
# Port of flame-core/src/cuda/fp8_dequant.cu (lines 1–61) to a pure-Mojo GPU
# kernel. Bit-exact with the CUDA reference and with the production FP8-resident
# streaming path used by LTX-2.3 22B distilled-fp8 (fp8_resident.rs:49,
# ltx2_model.rs:3357 — scale from the per-tensor `weight_scale` scalar).
#
# Format (matches PyTorch torch.float8_e4m3fn / OCP E4M3, no infinities):
#   - 1 byte per element: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
#   - decode(byte):
#       sign = (byte >> 7) & 1
#       exp  = (byte >> 3) & 0xF
#       mant =  byte       & 0x7
#       if exp==0 && mant==0:  val = 0
#       elif exp==0 (subnormal): val = (mant/8) * 2^-6   (i.e. ldexp(mant/8, -6))
#       else (normal):           val = (1 + mant/8) * 2^(exp-7)
#       if sign: val = -val
#       val *= scale            # per-tensor F32 weight_scale (1.0 if absent)
#       out  = bf16(val)
#   - There is NO block scale and NO companion scale tensor in the bytes; the
#     scale is a single F32 scalar passed in (the checkpoint stores it as a
#     0-D `*.weight_scale` tensor; the loader reads it host-side). E4M3 max ≈
#     ±448; the kernel does no saturation on decode (the on-disk bytes are
#     already valid E4M3; saturation happens only on the encode side, which we
#     never do here).
#
# Calling convention (mirrors fused_inference::dequant_fp8_to_bf16):
#   x:   U8 device Tensor, shape S — one E4M3 byte per element.
#   scale: Float32 — per-tensor weight_scale.
#   Returns: BF16 Tensor with the SAME shape S (numel preserved).
#
# Kernel: grid-stride loop, one thread decodes one byte (matches the CUDA
# `for (i = idx; i < n; i += stride)` form so behavior is identical for any n).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.gpu import global_idx, grid_dim, block_dim
from std.math import ldexp
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts


comptime _DYN1 = Layout.row_major(-1)
# 256 threads/block matches the CUDA kernel's `const int block = 256` (line 48).
comptime _BLOCK = 256


# ─────────────────────────────────────────────────────────────────────────────
# E4M3 byte → Float32. Faithful to fp8_dequant.cu's fp8_to_bf16_kernel decode.
# ldexp(m, e) == m * 2^e, matching CUDA `ldexpf`.
# ─────────────────────────────────────────────────────────────────────────────
@always_inline
def _fp8_e4m3_decode(byte: UInt32) -> Float32:
    var sign = (byte >> 7) & 1
    var exp = Int((byte >> 3) & 0xF)
    var mant = Int(byte & 0x7)

    var val: Float32 = 0.0
    if exp == 0 and mant == 0:
        val = 0.0
    elif exp == 0:
        # Subnormal: (mant/8) * 2^-6.
        val = ldexp(Float32(mant) / 8.0, Int32(-6))
    else:
        # Normal: (1 + mant/8) * 2^(exp-7).
        val = ldexp(1.0 + Float32(mant) / 8.0, Int32(exp - 7))

    if sign != 0:
        return -val
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: grid-stride loop, one thread per byte → one BF16 output.
#   out[i] = bf16(fp8_e4m3_decode(in[i]) * scale)
# Single-precision math matches the CUDA path (`ldexpf` + `__float2bfloat16`).
# ─────────────────────────────────────────────────────────────────────────────
def _fp8_dequant_kernel(
    x: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    scale: Float32,
    n: Int,
):
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    while i < n:
        var byte_u8 = rebind[Scalar[DType.uint8]](x[i])
        var byte_u32 = UInt32(Int(byte_u8))
        var v = _fp8_e4m3_decode(byte_u32) * scale
        o[i] = rebind[o.element_type](v.cast[DType.bfloat16]())
        i += stride


# ─────────────────────────────────────────────────────────────────────────────
# Host wrapper. Mirrors flame-core fused_inference::dequant_fp8_to_bf16:
#   - x: U8 tensor, any shape S (one E4M3 byte/element)
#   - scale: F32 per-tensor weight_scale
#   - returns: BF16 tensor, same shape S
# ─────────────────────────────────────────────────────────────────────────────
def fp8_e4m3_dequant_to_bf16(
    x: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Dequantize FP8 E4M3 bytes → BF16, applying a per-tensor F32 scale. GPU-only.

    out[i] = bf16(e4m3_decode(x[i]) * scale). Bit-exact with the CUDA reference
    kernel (flame-core/src/cuda/fp8_dequant.cu) used by the LTX-2.3 FP8 stream.

    Args:
        x: U8 tensor of any shape; each byte is one float8_e4m3fn value.
        scale: Per-tensor weight_scale (use 1.0 if the checkpoint has none).
        ctx: DeviceContext.

    Returns:
        BF16 tensor with the SAME shape as `x`.

    Raises:
        On non-U8 dtype or empty input.
    """
    if x.dtype() != STDtype.U8 and x.dtype() != STDtype.F8_E4M3:
        raise Error(
            String("fp8_e4m3_dequant_to_bf16: x must be U8/F8_E4M3, got ")
            + x.dtype().name()
        )
    var n = x.numel()
    if n == 0:
        raise Error("fp8_e4m3_dequant_to_bf16: empty input")

    var out_shape = x.shape()
    var out_bytes = n * STDtype.BF16.byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_bytes)

    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))

    var X = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr(), x_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )

    # Grid-stride: cap grid like the CUDA kernel (grid clamps to 65535).
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_fp8_dequant_kernel, _fp8_dequant_kernel](
        X, O, scale, n,
        grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


# ─────────────────────────────────────────────────────────────────────────────
# PER-ROW (per-output-channel) FP8 E4M3 → BF16 dequant — the Ideogram-4 path.
# Ideogram-4 weight-only FP8 Linear: weight [out,in] float8_e4m3fn + sibling F32
# per-output-row scale [out]. Reference (1:1):
#   /home/alex/ideogram4-ref/src/ideogram4/quantized_loading.py
#   Fp8Linear.forward:197-200  → w[o,i] = float(weight[o,i]) * scale[o]
#   (scale[:,None] broadcast over `in`). Decode F32 (exact), store BF16.
# ─────────────────────────────────────────────────────────────────────────────
def _fp8_dequant_perrow_kernel(
    x: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    scale: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    cols: Int,
    n: Int,
):
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    while i < n:
        var row = i // cols
        var byte_u8 = rebind[Scalar[DType.uint8]](x[i])
        var byte_u32 = UInt32(Int(byte_u8))
        var s = rebind[Scalar[DType.float32]](scale[row])
        var v = _fp8_e4m3_decode(byte_u32) * s
        o[i] = rebind[o.element_type](v.cast[DType.bfloat16]())
        i += stride


def fp8_e4m3_dequant_perrow_to_bf16(
    w: Tensor,
    scale: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Dequantize a weight-only FP8 E4M3 Linear weight with PER-ROW F32 scales.

    out[o, i] = bf16(e4m3_decode(w[o, i]) * scale[o]). Mirrors Fp8Linear.forward
    (quantized_loading.py:197-200): per-output-channel scale broadcast over `in`.
    """
    if w.dtype() != STDtype.U8 and w.dtype() != STDtype.F8_E4M3:
        raise Error(
            String("fp8_e4m3_dequant_perrow_to_bf16: w must be U8/F8_E4M3, got ")
            + w.dtype().name()
        )
    if scale.dtype() != STDtype.F32:
        raise Error(
            String("fp8_e4m3_dequant_perrow_to_bf16: scale must be F32, got ")
            + scale.dtype().name()
        )
    var wshape = w.shape()
    if len(wshape) != 2:
        raise Error(
            String("fp8_e4m3_dequant_perrow_to_bf16: w must be 2-D [out,in], rank=")
            + String(len(wshape))
        )
    var out_rows = wshape[0]
    var cols = wshape[1]
    var n = w.numel()
    if n == 0:
        raise Error("fp8_e4m3_dequant_perrow_to_bf16: empty input")
    if scale.numel() != out_rows:
        raise Error(
            String("fp8_e4m3_dequant_perrow_to_bf16: scale len ")
            + String(scale.numel()) + " != out rows " + String(out_rows)
        )

    var out_shape = w.shape()
    var out_bytes = n * STDtype.BF16.byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_bytes)

    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_rows))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))

    var X = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](w.buf.unsafe_ptr(), x_rl)
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )

    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_fp8_dequant_perrow_kernel, _fp8_dequant_perrow_kernel](
        X, S, O, cols, n, grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


def load_fp8_dequant(
    st: ShardedSafeTensors,
    weight_name: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """Read a weight-only-FP8 Linear weight `<weight_name>` (F8_E4M3 [out,in]) +
    its sibling F32 per-row scale `<weight_name>_scale` [out] from a (sharded)
    safetensors, and return the dequantized BF16 weight [out,in].

    Mirrors the diffusers/Ideogram convention (swap_linears_to_fp8 /
    FP8_SCALE_SUFFIX, quantized_loading.py:203-232): module-prefix + '.weight'
    and module-prefix + '.weight_scale'.
    """
    var scale_name = weight_name + "_scale"
    var w_info = st.tensor_info(weight_name)
    var w_bytes = st.tensor_bytes(weight_name)
    var w_view = from_parts(w_info.dtype, w_info.shape.copy(), w_bytes)
    var w = Tensor.from_view_raw(w_view, ctx)
    var s_info = st.tensor_info(scale_name)
    var s_bytes = st.tensor_bytes(scale_name)
    var s_view = from_parts(s_info.dtype, s_info.shape.copy(), s_bytes)
    var scale = Tensor.from_view_as_f32(s_view, ctx)
    return fp8_e4m3_dequant_perrow_to_bf16(w^, scale^, ctx)
