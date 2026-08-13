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

from max.gpu.host import DeviceContext
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
    n_w: Int64,
):
    var n = Int(n_w)
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
def _fp8_e4m3_dequant_to_bf16_impl(
    x: Tensor,
    scale: Float32,
    ctx: DeviceContext,
    sync_after_launch: Bool,
) raises -> Tensor:
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
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr())
        ),
        runtime_layout=x_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=out_rl,
    )

    # Grid-stride: cap grid like the CUDA kernel (grid clamps to 65535).
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_fp8_dequant_kernel](
        X, O, scale, Int64(n),
        grid_dim=grid, block_dim=_BLOCK,
    )
    if sync_after_launch:
        ctx.synchronize()
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


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
    return _fp8_e4m3_dequant_to_bf16_impl(x, scale, ctx, True)


def fp8_e4m3_dequant_to_bf16_no_sync(
    x: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Enqueue FP8 E4M3 → BF16 dequant without a host/device fence.

    Use only when the input stays resident, or when a streamed caller preserves
    same-DeviceContext ordering and fences after the complete batch before it
    returns. The synced wrapper above remains the default single-tensor API.
    """
    return _fp8_e4m3_dequant_to_bf16_impl(x, scale, ctx, False)


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
    cols_w: Int32,
    n_w: Int64,
):
    var cols = Int(cols_w)
    var n = Int(n_w)
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

    var X = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(w.buf.unsafe_ptr())
        ),
        runtime_layout=x_rl,
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=s_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=out_rl,
    )

    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_fp8_dequant_perrow_kernel](
        X, S, O, Int32(cols), Int64(n), grid_dim=grid, block_dim=_BLOCK,
    )
    # sync removed (single-stream ordering; was kernel-trailing host stall)
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


def fp8_e4m3_dequant_perrow_to_bf16_into(
    w: Tensor,
    scale: Tensor,
    dst: Tensor,
    ctx: DeviceContext,
) raises:
    """`fp8_e4m3_dequant_perrow_to_bf16` writing into a CALLER-OWNED BF16
    tensor instead of allocating — same kernel, same bytes. Exists for
    resident-fp8 hot loops that dequant hundreds of times per run: the
    allocating variant makes the per-block weight a FRESH ~0.19-0.31 GiB
    device allocation each call, and with no sync in the loop the host
    enqueues allocations far ahead of the stream-ordered frees — measured
    OOM on the MiniMax-H3 50-block loop next to its 17.96 GiB store
    (2026-08-03, minimax_h3_fp8_resident_gate first fp8-mode run). Reusing
    one preallocated scratch per weight slot makes the loop allocation-free.
    The CALLER owns the hazard ordering: `dst` must not still be read by
    previously enqueued kernels (drain with ctx.synchronize() before
    overwrite)."""
    if w.dtype() != STDtype.U8 and w.dtype() != STDtype.F8_E4M3:
        raise Error(
            String("fp8_e4m3_dequant_perrow_to_bf16_into: w must be U8/F8_E4M3, got ")
            + w.dtype().name()
        )
    if scale.dtype() != STDtype.F32:
        raise Error(
            String("fp8_e4m3_dequant_perrow_to_bf16_into: scale must be F32, got ")
            + scale.dtype().name()
        )
    if dst.dtype() != STDtype.BF16:
        raise Error(
            String("fp8_e4m3_dequant_perrow_to_bf16_into: dst must be BF16, got ")
            + dst.dtype().name()
        )
    var wshape = w.shape()
    if len(wshape) != 2:
        raise Error(
            String("fp8_e4m3_dequant_perrow_to_bf16_into: w must be 2-D [out,in], rank=")
            + String(len(wshape))
        )
    var out_rows = wshape[0]
    var cols = wshape[1]
    var n = w.numel()
    if n == 0:
        raise Error("fp8_e4m3_dequant_perrow_to_bf16_into: empty input")
    if scale.numel() != out_rows:
        raise Error(
            String("fp8_e4m3_dequant_perrow_to_bf16_into: scale len ")
            + String(scale.numel()) + " != out rows " + String(out_rows)
        )
    var osh = dst.shape()
    if len(osh) != 2 or osh[0] != out_rows or osh[1] != cols:
        raise Error(
            "fp8_e4m3_dequant_perrow_to_bf16_into: dst shape does not match w"
        )

    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_rows))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(w.buf.unsafe_ptr())
        ),
        runtime_layout=x_rl,
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=s_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(dst.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=out_rl,
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_fp8_dequant_perrow_kernel](
        X, S, O, Int32(cols), Int64(n), grid_dim=grid, block_dim=_BLOCK,
    )


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
    # dtype-contract: allow-f32-boundary - reference FP8 sidecar scales are F32.
    var scale = Tensor.from_view_as_f32(s_view, ctx)
    return fp8_e4m3_dequant_perrow_to_bf16(w^, scale^, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# PER-EXPERT FP8 E4M3 → BF16 dequant — the grouped-MoE weight path.
# A 3-D expert stack `w[E,M,N]` (float8_e4m3fn) + one F32 scale per expert `[E]`,
# scale broadcast over the whole [M,N] slice. This is exactly the per-row kernel
# with cols = M*N (row == expert index), so it REUSES `_fp8_dequant_perrow_kernel`
# verbatim; only the rank check + output shape differ. Used by LingBot-Video's
# MoE experts (blocks.*.ffn.experts.{w1,w2,w3}, per-expert amax/448 export —
# scripts/lingbot_quantize_fp8_experts.py).
# ─────────────────────────────────────────────────────────────────────────────
def fp8_e4m3_dequant_perexpert_to_bf16(
    w: Tensor,
    scale: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Dequantize a 3-D FP8 E4M3 expert stack with PER-EXPERT F32 scales.

    out[e, m, n] = bf16(e4m3_decode(w[e, m, n]) * scale[e]). `w` is [E,M,N]
    F8_E4M3/U8; `scale` is F32 [E] (one scalar per expert, broadcast over M*N).
    Returns BF16 [E,M,N]. Reuses `_fp8_dequant_perrow_kernel` (cols = M*N).
    """
    if w.dtype() != STDtype.U8 and w.dtype() != STDtype.F8_E4M3:
        raise Error(
            String("fp8_e4m3_dequant_perexpert_to_bf16: w must be U8/F8_E4M3, got ")
            + w.dtype().name()
        )
    if scale.dtype() != STDtype.F32:
        raise Error(
            String("fp8_e4m3_dequant_perexpert_to_bf16: scale must be F32, got ")
            + scale.dtype().name()
        )
    var wshape = w.shape()
    if len(wshape) != 3:
        raise Error(
            String("fp8_e4m3_dequant_perexpert_to_bf16: w must be 3-D [E,M,N], rank=")
            + String(len(wshape))
        )
    var experts = wshape[0]
    var n = w.numel()
    if n == 0:
        raise Error("fp8_e4m3_dequant_perexpert_to_bf16: empty input")
    if scale.numel() != experts:
        raise Error(
            String("fp8_e4m3_dequant_perexpert_to_bf16: scale len ")
            + String(scale.numel()) + " != experts " + String(experts)
        )
    var cols = n // experts   # M*N — the per-expert broadcast span (row == expert)

    var out_shape = w.shape()
    var out_bytes = n * STDtype.BF16.byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_bytes)

    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](experts))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))

    var X = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(w.buf.unsafe_ptr())
        ),
        runtime_layout=x_rl,
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=s_rl,
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=out_rl,
    )

    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_fp8_dequant_perrow_kernel](
        X, S, O, Int32(cols), Int64(n), grid_dim=grid, block_dim=_BLOCK,
    )
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


def load_fp8_expert_dequant(
    st: ShardedSafeTensors,
    weight_name: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """Read a 3-D FP8 expert stack `<weight_name>` (F8_E4M3 [E,M,N]) + its sibling
    F32 per-expert scale `<weight_name>_scale` [E] from a (sharded) safetensors,
    and return the dequantized BF16 weight [E,M,N]. Sidecar suffix `_scale` matches
    `load_fp8_dequant` / the LTX2 fp8 export convention."""
    var scale_name = weight_name + "_scale"
    var w_info = st.tensor_info(weight_name)
    var w_bytes = st.tensor_bytes(weight_name)
    var w_view = from_parts(w_info.dtype, w_info.shape.copy(), w_bytes)
    var w = Tensor.from_view_raw(w_view, ctx)
    var s_info = st.tensor_info(scale_name)
    var s_bytes = st.tensor_bytes(scale_name)
    var s_view = from_parts(s_info.dtype, s_info.shape.copy(), s_bytes)
    # dtype-contract: allow-f32-boundary - fp8 sidecar scales are F32.
    var scale = Tensor.from_view_as_f32(s_view, ctx)
    return fp8_e4m3_dequant_perexpert_to_bf16(w^, scale^, ctx)
