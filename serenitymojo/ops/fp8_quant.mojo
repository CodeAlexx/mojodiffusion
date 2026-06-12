# ops/fp8_quant.mojo — BF16 → FP8 E4M3 per-row QUANTIZATION (encode side).
#
# T2.B fp8-quantized-resident base weights (TIER2_PARITY_CAMPAIGN_2026-06-11).
# The decode side (ops/fp8.mojo) is the parity-gated Ideogram-4 inference
# machinery: weight-only FP8 Linear = E4M3 bytes [out,in] + sibling F32
# PER-OUTPUT-ROW scale [out] (quantized_loading.py convention). This file adds
# the matching ENCODE so a bf16 checkpoint can be quantized ONCE at load and
# kept resident at 1 byte/param; per-block dequant goes back through the
# existing fp8_e4m3_dequant_perrow_to_bf16 (same scheme, same decode).
#
# Scheme (per-output-row symmetric absmax, the proven ideogram4 layout):
#   scale[o] = max_i |w[o,i]| / 448        (448 = E4M3-fn max; 1.0 if row is 0)
#   byte[o,i] = e4m3_rne(w[o,i] / scale[o])  — round-to-nearest-even, saturate
#               to ±448, subnormals handled (min subnormal 2^-9), no inf/NaN
#               encodings produced (fn format).
# Round-trip property: decode(encode(w)) error <= half-ULP of the scaled E4M3
# grid — <= 2^-4 relative for normals (3 mantissa bits), <= scale*2^-9/2
# absolute in the subnormal range. Gated by ops/tests/fp8_quant_smoke.mojo.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.gpu import global_idx, grid_dim, block_dim
from std.math import ldexp, floor
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _E4M3_MAX = Float32(448.0)


# ─────────────────────────────────────────────────────────────────────────────
# Round-to-nearest-even for a non-negative Float32 (exact: callers divide by a
# power of two first, so the half-way comparison is representable).
# ─────────────────────────────────────────────────────────────────────────────
@always_inline
def _rne_int(x: Float32) -> Int:
    var fl = floor(x)
    var frac = x - fl
    var n = Int(fl)
    if frac > Float32(0.5):
        n += 1
    elif frac == Float32(0.5) and (n & 1) == 1:
        n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Float32 → E4M3 byte (fn format: bias 7, max ±448, no infinities).
# Inverse of ops/fp8.mojo _fp8_e4m3_decode:
#   normal:    (1 + mant/8) * 2^(exp-7), exp 1..15 (15 with mant 7 = NaN —
#              never produced; saturation stops at exp=15 mant=6 = 448)
#   subnormal: (mant/8) * 2^-6, i.e. steps of 2^-9
# ─────────────────────────────────────────────────────────────────────────────
@always_inline
def _fp8_e4m3_encode(v: Float32) -> UInt32:
    var sign: UInt32 = 0
    var a = v
    if a < 0:
        sign = UInt32(0x80)
        a = -a
    if a != a:
        return sign  # NaN → ±0 (weights are finite; fail-soft, never emit NaN)
    if a > _E4M3_MAX:
        a = _E4M3_MAX
    var byte: UInt32 = 0
    if a < ldexp(Float32(1.0), Int32(-6)):
        # Subnormal range: quantum 2^-9; m in [0..8] (m==8 promotes to 2^-6).
        var m = _rne_int(a * Float32(512.0))
        if m >= 8:
            byte = UInt32(0x08)  # exp=1, mant=0 == 2^-6
        else:
            byte = UInt32(m)
    else:
        # Normal: find e with 2^e <= a < 2^(e+1)  (e in [-6, 8]).
        var e = -6
        while e < 8 and a >= ldexp(Float32(1.0), Int32(e + 1)):
            e += 1
        # Mantissa code in [8..16]: a / 2^(e-3) (power-of-two divide = exact).
        var m = _rne_int(ldexp(a, Int32(3 - e)))
        if m == 16:
            m = 8
            e += 1
        if e > 8:
            byte = UInt32((15 << 3) | 6)  # saturate to 448
        else:
            byte = UInt32(((e + 7) << 3) | (m - 8))
    return byte | sign


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: per-row absmax → scale[r] = amax/448 (1.0 for an all-zero row).
# Thread-per-row grid-stride (rows are 1024..12288 for HiDream — ample blocks).
# ─────────────────────────────────────────────────────────────────────────────
def _rowscale_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [rows*cols]
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [rows]
    cols: Int,
    rows: Int,
):
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var r = idx
    while r < rows:
        var m: Float32 = 0.0
        var base = r * cols
        for c in range(cols):
            var v = Float32(rebind[Scalar[DType.bfloat16]](x[base + c]))
            var a = v if v >= 0 else -v
            if a > m:
                m = a
        var sc = m / _E4M3_MAX if m > Float32(0.0) else Float32(1.0)
        s[r] = rebind[s.element_type](sc)
        r += stride


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: elementwise encode byte[i] = e4m3(x[i] / scale[i // cols]).
# ─────────────────────────────────────────────────────────────────────────────
def _encode_perrow_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [n]
    s: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],    # [rows]
    o: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],      # [n]
    cols: Int,
    n: Int,
):
    var idx = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = idx
    while i < n:
        var v = Float32(rebind[Scalar[DType.bfloat16]](x[i]))
        var sc = rebind[Scalar[DType.float32]](s[i // cols])
        var byte = _fp8_e4m3_encode(v / sc)
        o[i] = rebind[o.element_type](Scalar[DType.uint8](byte & 0xFF))
        i += stride


def fp8_e4m3_rowscale(w: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Per-output-row absmax scale for E4M3 quantization: F32 [out].

    scale[o] = max_i |w[o,i]| / 448 (1.0 for an all-zero row). w must be a
    BF16 2-D [out,in] device tensor."""
    if w.dtype() != STDtype.BF16:
        raise Error(
            String("fp8_e4m3_rowscale: w must be BF16, got ") + w.dtype().name()
        )
    var wsh = w.shape()
    if len(wsh) != 2:
        raise Error(
            String("fp8_e4m3_rowscale: w must be 2-D [out,in], rank=")
            + String(len(wsh))
        )
    var rows = wsh[0]
    var cols = wsh[1]
    if rows * cols == 0:
        raise Error("fp8_e4m3_rowscale: empty input")

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](rows * 4)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * cols))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        w.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    var grid = (rows + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_rowscale_kernel, _rowscale_kernel](
        X, S, cols, rows, grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()
    var s_shape: List[Int] = [rows]
    return Tensor(out_buf^, s_shape^, STDtype.F32)


def fp8_e4m3_encode_perrow(
    w: Tensor,
    scale: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Quantize a BF16 [out,in] weight to E4M3 bytes with per-row F32 scales.

    byte[o,i] = e4m3_rne(w[o,i] / scale[o]), RNE + saturation to ±448. The
    result decodes through the parity-gated fp8_e4m3_dequant_perrow_to_bf16
    (ops/fp8.mojo) — same per-row layout the Ideogram-4 checkpoints use."""
    if w.dtype() != STDtype.BF16:
        raise Error(
            String("fp8_e4m3_encode_perrow: w must be BF16, got ")
            + w.dtype().name()
        )
    if scale.dtype() != STDtype.F32:
        raise Error(
            String("fp8_e4m3_encode_perrow: scale must be F32, got ")
            + scale.dtype().name()
        )
    var wsh = w.shape()
    if len(wsh) != 2:
        raise Error(
            String("fp8_e4m3_encode_perrow: w must be 2-D [out,in], rank=")
            + String(len(wsh))
        )
    var rows = wsh[0]
    var cols = wsh[1]
    var n = w.numel()
    if n == 0:
        raise Error("fp8_e4m3_encode_perrow: empty input")
    if scale.numel() != rows:
        raise Error(
            String("fp8_e4m3_encode_perrow: scale len ")
            + String(scale.numel()) + " != out rows " + String(rows)
        )

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        w.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        scale.buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    var O = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr(), o_rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    if grid > 65535:
        grid = 65535
    ctx.enqueue_function[_encode_perrow_kernel, _encode_perrow_kernel](
        X, S, O, cols, n, grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()
    return Tensor(out_buf^, w.shape(), STDtype.F8_E4M3)
