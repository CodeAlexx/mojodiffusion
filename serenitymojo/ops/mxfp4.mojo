# ops/mxfp4.mojo — MXFP4 → BF16 dequantization (pure Mojo + MAX GPU port).
#
# Port of flame-core/src/cuda/mxfp4_dequant.cu (lines 1–113) to a pure-Mojo
# GPU kernel. Bit-exact with the CUDA reference and with
# transformers/integrations/mxfp4.py::convert_moe_packed_tensors.
#
# Format (matches HuggingFace transformers MXFP4):
#   - 32 FP4 (E2M1) elements share one 8-bit E8M0 (exponent-only) scale.
#   - On-disk:
#       blocks: uint8[..., G, 16] — 16 bytes per block, 2 packed FP4 nibbles/byte
#       scales: uint8[..., G]     — one E8M0 exponent byte per 32-element block
#   - Nibble packing:
#       low nibble  (byte & 0x0F) → EVEN output index (0, 2, 4, …)
#       high nibble (byte >> 4)   → ODD  output index (1, 3, 5, …)
#   - FP4 LUT (16 values), indexed by the 4-bit nibble (0..15):
#       +0.0,+0.5,+1.0,+1.5,+2.0,+3.0,+4.0,+6.0,
#       -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0
#     Equivalently: magnitude from LUT[i & 7], sign from i & 8.
#   - E8M0 scale: out *= 2^(scale_byte - 127). 127 is the IEEE E8M0 bias.
#   - Output dtype: BF16.
#
# Calling convention (mirrors fused_inference::dequant_mxfp4_to_bf16):
#   blocks: U8 device Tensor, shape [..., G, 16]
#   scales: U8 device Tensor, shape [..., G]    (matching leading rank)
#   Returns: BF16 Tensor with shape = blocks.shape[:-2] + [G*32].
#
# Kernel: one thread per MXFP4 block (32 output BF16 values).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.math import ldexp
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
# 256 threads/block matches the CUDA kernel's `const int block = 256` (line 99).
comptime _BLOCK = 256


# ─────────────────────────────────────────────────────────────────────────────
# FP4 nibble → Float32 magnitude. The four-bit nibble selects one of 16
# values; magnitude is determined by the low 3 bits, sign by bit 3 (0x8).
# Inline switch (kept in-kernel; the Mojo compiler can lower this to a small
# constant table — equivalent to the CUDA __device__ __constant__ FP4_LUT).
# ─────────────────────────────────────────────────────────────────────────────
@always_inline
def _fp4_decode(nibble: UInt32) -> Float32:
    # Magnitude from low 3 bits.
    var mag3 = nibble & 0x7
    var m: Float32 = 0.0
    if mag3 == 0:
        m = 0.0
    elif mag3 == 1:
        m = 0.5
    elif mag3 == 2:
        m = 1.0
    elif mag3 == 3:
        m = 1.5
    elif mag3 == 4:
        m = 2.0
    elif mag3 == 5:
        m = 3.0
    elif mag3 == 6:
        m = 4.0
    else:  # 7
        m = 6.0
    # Sign from bit 3 (0x8). +0.0 with sign-bit becomes -0.0 (matches LUT).
    if (nibble & 0x8) != 0:
        return -m
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: one thread per MXFP4 block (32 output BF16 values).
# Each thread:
#   - reads 16 bytes from blocks[bid*16 .. bid*16 + 16]
#   - reads 1 byte from scales[bid] → scale_exp = Int(scales[bid]) - 127
#   - emits 32 BF16 outputs to out[bid*32 .. bid*32 + 32]
#
# Single-precision math matches the CUDA path (`ldexpf` + `__float2bfloat16`).
# ─────────────────────────────────────────────────────────────────────────────
def _mxfp4_dequant_kernel(
    blocks: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    scales: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    rows_total: Int,
):
    var bid = Int(global_idx.x)
    if bid < rows_total:
        # E8M0: scale = 2^(scale_byte - 127).
        # Use `ldexp(1.0, scale_exp)` (mirrors CUDA `ldexpf`) instead of
        # `exp2(Float32(scale_exp))`. On SM_9x, `exp2[f32]` lowers to
        # `ex2.approx.ftz.f32` (flush-to-zero) and the polynomial fallback
        # clamps inputs to [-126, 126]; `ldexp` builds the multiplier from
        # the f32 exponent bits directly and is bit-faithful to `ldexpf` for
        # subnormals and the exponent extrema (e.g. scale_byte=0 → exp=-127).
        var scale_byte = Int(rebind[Scalar[DType.uint8]](scales[bid]))
        var scale_exp = scale_byte - 127
        var scale_mul = ldexp(Float32(1.0), Int32(scale_exp))

        var blk_base = bid * 16
        var out_base = bid * 32

        # 16 bytes → 32 FP4 nibbles → 32 BF16 outputs.
        for i in range(16):
            var byte_u8 = rebind[Scalar[DType.uint8]](blocks[blk_base + i])
            var byte_u32 = UInt32(Int(byte_u8))
            var lo = byte_u32 & 0x0F
            var hi = (byte_u32 >> 4) & 0x0F

            var v_lo = _fp4_decode(lo) * scale_mul
            var v_hi = _fp4_decode(hi) * scale_mul

            o[out_base + 2 * i] = rebind[o.element_type](
                v_lo.cast[DType.bfloat16]()
            )
            o[out_base + 2 * i + 1] = rebind[o.element_type](
                v_hi.cast[DType.bfloat16]()
            )


# ─────────────────────────────────────────────────────────────────────────────
# Host wrapper. Mirrors flame-core/src/ops/fused_inference.rs:220
# (dequant_mxfp4_to_bf16) with the same shape contract:
#   - blocks: U8, shape [..., G, 16]
#   - scales: U8, shape [..., G]
#   - returns: BF16, shape blocks.shape[:-2] + [G*32]
# ─────────────────────────────────────────────────────────────────────────────
def mxfp4_dequant_to_bf16(
    blocks: Tensor,
    scales: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Dequantize MXFP4 packed blocks → BF16. GPU-only.

    Each MXFP4 block is 16 input bytes (32 packed FP4 nibbles) plus one E8M0
    scale byte; output is 32 BF16 values per block.

    Args:
        blocks: U8 tensor, shape [..., G, 16].
        scales: U8 tensor, shape [..., G]. The leading dims must match the
            blocks tensor's leading dims (everything except the final 16).
        ctx: DeviceContext.

    Returns:
        BF16 tensor whose shape is `blocks.shape[:-2] + [G*32]`.

    Raises:
        On dtype mismatch (must be U8), trailing-dim mismatch (must be 16),
        leading-dim shape mismatch between blocks and scales, or empty input.
    """
    # ── Dtype validation ────────────────────────────────────────────────────
    if blocks.dtype() != STDtype.U8:
        raise Error(
            String("mxfp4_dequant_to_bf16: blocks must be U8, got ")
            + blocks.dtype().name()
        )
    if scales.dtype() != STDtype.U8:
        raise Error(
            String("mxfp4_dequant_to_bf16: scales must be U8, got ")
            + scales.dtype().name()
        )

    # ── Shape validation ────────────────────────────────────────────────────
    var b_shape = blocks.shape()
    var s_shape = scales.shape()
    if len(b_shape) < 2:
        raise Error(
            String("mxfp4_dequant_to_bf16: blocks rank must be >= 2, got ")
            + String(len(b_shape))
        )
    if b_shape[len(b_shape) - 1] != 16:
        raise Error(
            String("mxfp4_dequant_to_bf16: blocks last dim must be 16, got ")
            + String(b_shape[len(b_shape) - 1])
        )
    # Leading dims of `blocks` (everything except the final 16) must equal the
    # full shape of `scales`. Specifically blocks.shape[:-1] == scales.shape.
    if len(s_shape) != len(b_shape) - 1:
        raise Error(
            String("mxfp4_dequant_to_bf16: scales rank=")
            + String(len(s_shape))
            + " must equal blocks rank - 1 = "
            + String(len(b_shape) - 1)
        )
    for i in range(len(s_shape)):
        if s_shape[i] != b_shape[i]:
            raise Error(
                String("mxfp4_dequant_to_bf16: shape mismatch at axis ")
                + String(i)
                + ": blocks="
                + String(b_shape[i])
                + " scales="
                + String(s_shape[i])
            )

    # ── Counts ──────────────────────────────────────────────────────────────
    var rows_total = scales.numel()  # number of 32-element MXFP4 blocks
    if rows_total == 0:
        raise Error("mxfp4_dequant_to_bf16: empty input (rows_total=0)")
    # Sanity: blocks should hold exactly rows_total * 16 bytes.
    var blocks_numel = blocks.numel()
    if blocks_numel != rows_total * 16:
        raise Error(
            String("mxfp4_dequant_to_bf16: blocks numel=")
            + String(blocks_numel)
            + " != rows_total*16="
            + String(rows_total * 16)
        )

    var out_numel = rows_total * 32  # BF16 elements

    # ── Output shape: blocks.shape[:-2] + [G*32] ────────────────────────────
    # blocks shape [..., G, 16]; G = b_shape[-2]. We replace the trailing
    # [G, 16] with [G*32].
    var leading_rank = len(b_shape) - 2  # rank of the [...] prefix
    var G = b_shape[len(b_shape) - 2]
    var out_shape = List[Int]()
    for i in range(leading_rank):
        out_shape.append(b_shape[i])
    out_shape.append(G * 32)

    # ── Allocate the output device buffer (raw bytes, BF16 = 2 bytes each) ─
    var out_bytes = out_numel * STDtype.BF16.byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_bytes)

    # ── Wrap inputs and output as flat 1-D LayoutTensors ────────────────────
    var blocks_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](blocks_numel))
    var scales_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows_total))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_numel))

    var B = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        blocks.buf.unsafe_ptr(), blocks_rl
    )
    var S = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        scales.buf.unsafe_ptr(), scales_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )

    # ── Launch: one thread per block, _BLOCK threads per CTA ────────────────
    var grid = (rows_total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_mxfp4_dequant_kernel, _mxfp4_dequant_kernel](
        B, S, O, rows_total,
        grid_dim=grid, block_dim=_BLOCK,
    )
    # sync removed (single-stream ordering; was kernel-trailing host stall)
    return Tensor(out_buf^, out_shape^, STDtype.BF16)
