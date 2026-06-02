# pipeline/mxfp4_dequant_smoke.mojo — parity smoke for ops/mxfp4.mojo.
#
# Mirrors the test patterns in flame-core/tests/mxfp4_dequant.rs:
#   Test A: identity LUT — 8 known bytes cover all 16 FP4 LUT entries,
#           scale = 127 (multiplier 1.0). Remaining bytes are zero.
#   Test B: scale = 128 → multiplier 2^1 = 2.0 (doubles Test-A first 8 outputs).
#   Test C: scale = 126 → multiplier 2^-1 = 0.5 (halves).
#   Test D: scale = 130 → multiplier 2^3  = 8.0.
#   Test E: scale = 120 → multiplier 2^-7 ≈ 0.0078125.
#   Test F: multi-block (16 blocks) with mixed scales — verifies grid coverage
#           and CPU-vs-GPU bit-exact agreement (BF16 round-to-nearest is
#           deterministic).
#   Test G: GPT-OSS-shape — blocks [E=2, R=8, G=4, 16], scales [E=2, R=8, G=4].
#           Confirms shape passthrough: output is [E=2, R=8, G*32=128] BF16.
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/mxfp4_dequant_smoke.mojo -o /tmp/mxfp4_dequant_smoke

from std.gpu.host import DeviceContext
from std.math import ldexp
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.mxfp4 import mxfp4_dequant_to_bf16


# ── FP4 LUT (host-side mirror of the kernel constant) ────────────────────────
def _fp4_lut(i: Int) -> Float32:
    # i in 0..16. Magnitude from i & 7, sign from i & 8.
    var mag3 = i & 7
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
    else:
        m = 6.0
    if (i & 8) != 0:
        return -m
    return m


# ── CPU reference dequant (mirrors the kernel) ───────────────────────────────
# Returns a List[Float32] of length rows_total * 32. Each entry is the BF16
# round-to-nearest result, returned as F32 (to compare against Tensor.to_host()).
def _cpu_dequant(blocks: List[UInt8], scales: List[UInt8]) -> List[Float32]:
    var rows_total = len(scales)
    var out = List[Float32]()
    for r in range(rows_total):
        var scale_exp = Int(scales[r]) - 127
        # Mirror the kernel: ldexp(1.0, exp) — faithful to CUDA ldexpf.
        var scale_mul = ldexp(Float32(1.0), Int32(scale_exp))
        var blk_base = r * 16
        for i in range(16):
            var byte = Int(blocks[blk_base + i])
            var lo = byte & 0x0F
            var hi = (byte >> 4) & 0x0F
            var v_lo = _fp4_lut(lo) * scale_mul
            var v_hi = _fp4_lut(hi) * scale_mul
            # Round-trip through BF16 → F32 to match the on-GPU truncation.
            out.append(v_lo.cast[DType.bfloat16]().cast[DType.float32]())
            out.append(v_hi.cast[DType.bfloat16]().cast[DType.float32]())
    return out^


# ── U8 host → device upload helper. Mirrors what Tensor.from_host does for
# floating dtypes (we cannot use Tensor.from_host since it takes List[Float32]
# and casts to a compute dtype; for raw U8 we copy bytes verbatim).
def _u8_to_tensor(
    bytes: List[UInt8], var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    if n != len(bytes):
        raise Error(
            String("_u8_to_tensor: numel(shape)=")
            + String(n)
            + " != len(bytes)="
            + String(len(bytes))
        )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var hp = host.unsafe_ptr()
    for i in range(n):
        hp[i] = bytes[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape^, STDtype.U8)


# ── Standard "identity LUT" block: 8 known bytes covering all 16 LUT entries
# in order, low-nibble first, then 8 zero bytes.
def _identity_block() -> List[UInt8]:
    return [
        UInt8(0x10),  # lo=0, hi=1 → (0.0, 0.5)
        UInt8(0x32),  # lo=2, hi=3 → (1.0, 1.5)
        UInt8(0x54),  # lo=4, hi=5 → (2.0, 3.0)
        UInt8(0x76),  # lo=6, hi=7 → (4.0, 6.0)
        UInt8(0x98),  # lo=8, hi=9 → (-0.0, -0.5)
        UInt8(0xBA),  # lo=a, hi=b → (-1.0, -1.5)
        UInt8(0xDC),  # lo=c, hi=d → (-2.0, -3.0)
        UInt8(0xFE),  # lo=e, hi=f → (-4.0, -6.0)
        UInt8(0x00), UInt8(0x00), UInt8(0x00), UInt8(0x00),
        UInt8(0x00), UInt8(0x00), UInt8(0x00), UInt8(0x00),
    ]


def _run_single_block(
    name: String,
    blocks: List[UInt8],
    scale_byte: UInt8,
    ctx: DeviceContext,
) raises -> Bool:
    var scales = [scale_byte]
    var bt = _u8_to_tensor(blocks.copy(), [1, 16], ctx)
    var st = _u8_to_tensor(scales.copy(), [1], ctx)
    var out = mxfp4_dequant_to_bf16(bt, st, ctx)
    # Shape check: should be [32].
    var oshape = out.shape()
    if len(oshape) != 1 or oshape[0] != 32:
        print(name, "FAIL — bad output shape", len(oshape), oshape[0] if len(oshape) > 0 else -1)
        return False
    var got = out.to_host(ctx)
    var expected = _cpu_dequant(blocks, [scale_byte])
    if len(got) != len(expected) or len(got) != 32:
        print(name, "FAIL — length mismatch", len(got), len(expected))
        return False
    # Bit-exact compare (BF16 round-to-nearest is deterministic on both sides).
    var mism = 0
    for i in range(32):
        if got[i] != expected[i]:
            mism += 1
            if mism <= 4:
                print("  idx", i, "got", got[i], "exp", expected[i])
    if mism == 0:
        print(name, "PASS (32/32 bit-exact)")
        return True
    print(name, "FAIL —", mism, "mismatches")
    return False


def _run_multi_block(
    name: String,
    blocks: List[UInt8],
    scales: List[UInt8],
    rows: Int,
    ctx: DeviceContext,
) raises -> Bool:
    var bt = _u8_to_tensor(blocks.copy(), [rows, 16], ctx)
    var st = _u8_to_tensor(scales.copy(), [rows], ctx)
    var out = mxfp4_dequant_to_bf16(bt, st, ctx)
    var oshape = out.shape()
    if len(oshape) != 1 or oshape[0] != rows * 32:
        print(name, "FAIL — bad output shape")
        return False
    var got = out.to_host(ctx)
    var expected = _cpu_dequant(blocks, scales)
    if len(got) != len(expected):
        print(name, "FAIL — length mismatch", len(got), len(expected))
        return False
    var mism = 0
    for i in range(len(got)):
        if got[i] != expected[i]:
            mism += 1
            if mism <= 4:
                print("  idx", i, "got", got[i], "exp", expected[i])
    if mism == 0:
        print(name, "PASS (", len(got), "/", len(expected), "bit-exact)")
        return True
    print(name, "FAIL —", mism, "mismatches out of", len(got))
    return False


def _run_shape_test(ctx: DeviceContext) raises -> Bool:
    # GPT-OSS-like: blocks [E=2, R=8, G=4, 16], scales [E=2, R=8, G=4].
    # Output expected: [E=2, R=8, G*32=128] BF16.
    var E = 2
    var R = 8
    var G = 4
    var rows_total = E * R * G  # = 64
    var n_bytes = rows_total * 16

    var blocks = List[UInt8]()
    for i in range(n_bytes):
        # Pseudo-random but reproducible byte pattern.
        var v = ((i * 13) ^ 0xA5) & 0xFF
        blocks.append(UInt8(v))
    var scales = List[UInt8]()
    for i in range(rows_total):
        scales.append(UInt8((110 + (i % 30)) & 0xFF))

    var bt = _u8_to_tensor(blocks.copy(), [E, R, G, 16], ctx)
    var st = _u8_to_tensor(scales.copy(), [E, R, G], ctx)
    var out = mxfp4_dequant_to_bf16(bt, st, ctx)
    var os = out.shape()
    if len(os) != 3 or os[0] != E or os[1] != R or os[2] != G * 32:
        print("Test G shape FAIL — expected [", E, ",", R, ",", G * 32, "] got len=", len(os))
        return False

    var got = out.to_host(ctx)
    var expected = _cpu_dequant(blocks, scales)
    if len(got) != len(expected) or len(got) != rows_total * 32:
        print("Test G FAIL — length mismatch")
        return False
    var mism = 0
    for i in range(len(got)):
        if got[i] != expected[i]:
            mism += 1
            if mism <= 4:
                print("  idx", i, "got", got[i], "exp", expected[i])
    if mism == 0:
        print("Test G (GPT-OSS shape) PASS (", len(got), "elements bit-exact)")
        return True
    print("Test G FAIL —", mism, "mismatches")
    return False


def main() raises:
    var ctx = DeviceContext()
    var pass_count = 0
    var total = 0

    # ── Test A — all 16 LUT entries, scale=127 (=1.0) ────────────────────────
    total += 1
    if _run_single_block(
        String("Test A (identity LUT, scale=127)"),
        _identity_block(),
        UInt8(127),
        ctx,
    ):
        pass_count += 1

    # ── Test B — scale=128 (×2.0) ────────────────────────────────────────────
    total += 1
    if _run_single_block(
        String("Test B (scale=128, mul=2.0)"),
        _identity_block(),
        UInt8(128),
        ctx,
    ):
        pass_count += 1

    # ── Test C — scale=126 (×0.5) ────────────────────────────────────────────
    total += 1
    if _run_single_block(
        String("Test C (scale=126, mul=0.5)"),
        _identity_block(),
        UInt8(126),
        ctx,
    ):
        pass_count += 1

    # ── Test D — scale=130 (×8.0) ────────────────────────────────────────────
    total += 1
    if _run_single_block(
        String("Test D (scale=130, mul=8.0)"),
        _identity_block(),
        UInt8(130),
        ctx,
    ):
        pass_count += 1

    # ── Test E — scale=120 (×2^-7) ───────────────────────────────────────────
    total += 1
    if _run_single_block(
        String("Test E (scale=120, mul=2^-7)"),
        _identity_block(),
        UInt8(120),
        ctx,
    ):
        pass_count += 1

    # ── Test F — multi-block, 1024 blocks, mixed scales (multi-CTA grid) ─────
    # 1024 blocks at block_dim=256 → 4 CTAs, exercising the grid-coverage path.
    # Output is 1024 * 32 = 32768 BF16 values. Scales sampled across 100..200
    # to cover a wider exponent range; byte pattern is deterministic.
    total += 1
    var nrows = 1024
    var mb_blocks = List[UInt8]()
    for i in range(nrows * 16):
        # Deterministic pseudo-random byte pattern.
        mb_blocks.append(UInt8(((i * 73) + 5) & 0xFF))
    var mb_scales = List[UInt8]()
    for i in range(nrows):
        # 100..200 covers 2^-27 .. 2^73 (clamped by BF16 range; valid for parity).
        mb_scales.append(UInt8((100 + (i % 101)) & 0xFF))
    if _run_multi_block(
        String("Test F (1024 blocks, mixed scales, multi-CTA)"),
        mb_blocks,
        mb_scales,
        nrows,
        ctx,
    ):
        pass_count += 1

    # ── Test G — GPT-OSS shape passthrough ───────────────────────────────────
    total += 1
    if _run_shape_test(ctx):
        pass_count += 1

    print("──────────────────────────────")
    print("mxfp4 smoke summary:", pass_count, "/", total)
