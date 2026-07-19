# pipeline/fp8_dequant_smoke.mojo — unit parity gate for ops/fp8.mojo.
#
# Gates the pure-Mojo FP8 E4M3 → BF16 dequant against:
#   Test A: a host E4M3 reference over a byte set covering every exponent /
#           mantissa / sign combination, at scale = 1.0 (bit-exact).
#   Test B: same bytes at scale = 0.5, 2.0, and a real checkpoint scale
#           (0.0020294189453125) — bit-exact vs host.
#   Test C: a REAL slice (64 elements) of
#           model.diffusion_model.transformer_blocks.4.attn1.to_q.weight from
#           ltx-2.3-22b-distilled-fp8.safetensors, dequantized with its real
#           weight_scale, bit-exact vs a torch.float8_e4m3fn reference
#           (computed by torch and embedded below — see /tmp/fp8_ref.json).
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/fp8_dequant_smoke.mojo -o /tmp/fp8_dequant_smoke
#
# GATE: all three tests bit-exact (BF16 round-to-nearest is deterministic on
# both the GPU kernel and the host cast). Mirrors mxfp4_dequant_smoke.

from std.gpu.host import DeviceContext
from std.math import ldexp
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_to_bf16


# ── Host E4M3 decode (mirrors the kernel + CUDA fp8_dequant.cu) ───────────────
def _e4m3_decode(byte: Int) -> Float32:
    var sign = (byte >> 7) & 1
    var exp = (byte >> 3) & 0xF
    var mant = byte & 0x7
    var val: Float32 = 0.0
    if exp == 0 and mant == 0:
        val = 0.0
    elif exp == 0:
        val = ldexp(Float32(mant) / 8.0, Int32(-6))
    else:
        val = ldexp(1.0 + Float32(mant) / 8.0, Int32(exp - 7))
    if sign != 0:
        return -val
    return val


def _cpu_dequant(bytes: List[UInt8], scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(bytes)):
        var v = _e4m3_decode(Int(bytes[i])) * scale
        out.append(v.cast[DType.bfloat16]().cast[DType.float32]())
    return out^


def _u8_to_tensor(
    bytes: List[UInt8], var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var hp = host.unsafe_ptr()
    for i in range(n):
        hp[i] = bytes[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape^, STDtype.U8)


# All 256 E4M3 byte values — full coverage of sign/exp/mant.
def _all_bytes() -> List[UInt8]:
    var b = List[UInt8]()
    for i in range(256):
        b.append(UInt8(i))
    return b^


def _run_bytes(
    name: String, bytes: List[UInt8], scale: Float32, ctx: DeviceContext
) raises -> Bool:
    var n = len(bytes)
    var xt = _u8_to_tensor(bytes.copy(), [n], ctx)
    var out = fp8_e4m3_dequant_to_bf16(xt, scale, ctx)
    var os = out.shape()
    if len(os) != 1 or os[0] != n:
        print(name, "FAIL — bad output shape")
        return False
    var got = out.to_host(ctx)
    var exp = _cpu_dequant(bytes, scale)
    var mism = 0
    for i in range(n):
        if got[i] != exp[i]:
            mism += 1
            if mism <= 4:
                print("  idx", i, "byte", Int(bytes[i]), "got", got[i], "exp", exp[i])
    if mism == 0:
        print(name, "PASS (", n, "/", n, "bit-exact)")
        return True
    print(name, "FAIL —", mism, "mismatches")
    return False


# ── Test C: real checkpoint slice (first 64 elements of block.4.attn1.to_q.weight)
# Raw FP8 bytes + torch float8_e4m3fn dequant reference (scale embedded).
comptime _REAL_SCALE: Float32 = 0.0020294189453125


def _real_raw() -> List[UInt8]:
    return [
        UInt8(64), UInt8(142), UInt8(205), UInt8(187), UInt8(198), UInt8(79),
        UInt8(72), UInt8(19), UInt8(195), UInt8(65), UInt8(212), UInt8(70),
        UInt8(207), UInt8(200), UInt8(157), UInt8(59), UInt8(172), UInt8(207),
        UInt8(174), UInt8(64), UInt8(173), UInt8(56), UInt8(55), UInt8(198),
        UInt8(73), UInt8(72), UInt8(63), UInt8(195), UInt8(63), UInt8(196),
        UInt8(71), UInt8(207), UInt8(200), UInt8(200), UInt8(196), UInt8(82),
        UInt8(164), UInt8(63), UInt8(198), UInt8(194), UInt8(66), UInt8(64),
        UInt8(216), UInt8(35), UInt8(163), UInt8(40), UInt8(205), UInt8(42),
        UInt8(80), UInt8(74), UInt8(74), UInt8(75), UInt8(76), UInt8(179),
        UInt8(159), UInt8(75), UInt8(56), UInt8(188), UInt8(169), UInt8(69),
        UInt8(48), UInt8(75), UInt8(69), UInt8(86),
    ]


def _real_deq() -> List[Float32]:
    return [
        0.004058837890625, -5.555152893066406e-05, -0.01318359375,
        -0.0027923583984375, -0.007110595703125, 0.01519775390625,
        0.00811767578125, 8.726119995117188e-05, -0.005584716796875,
        0.00457763671875, -0.0244140625, 0.007110595703125, -0.01519775390625,
        -0.00811767578125, -0.00020599365234375, 0.0027923583984375,
        -0.000762939453125, -0.01519775390625, -0.000888824462890625,
        0.004058837890625, -0.000823974609375, 0.0020294189453125,
        0.00189971923828125, -0.007110595703125, 0.0091552734375,
        0.00811767578125, 0.0037994384765625, -0.005584716796875,
        0.0037994384765625, -0.006103515625, 0.007598876953125,
        -0.01519775390625, -0.00811767578125, -0.00811767578125,
        -0.006103515625, 0.020263671875, -0.0003814697265625,
        0.0037994384765625, -0.007110595703125, -0.00506591796875,
        0.00506591796875, 0.004058837890625, -0.032470703125,
        0.0003490447998046875, -0.0003490447998046875, 0.000507354736328125,
        -0.01318359375, 0.00063323974609375, 0.0162353515625, 0.0101318359375,
        0.0101318359375, 0.01116943359375, 0.01220703125, -0.00139617919921875,
        -0.00023746490478515625, 0.01116943359375, 0.0020294189453125,
        -0.0030517578125, -0.00057220458984375, 0.006591796875,
        0.00101470947265625, 0.01116943359375, 0.006591796875, 0.0284423828125,
    ]


def _run_real(ctx: DeviceContext) raises -> Bool:
    var raw = _real_raw()
    var torch_ref = _real_deq()
    var n = len(raw)
    var xt = _u8_to_tensor(raw.copy(), [n], ctx)
    var out = fp8_e4m3_dequant_to_bf16(xt, _REAL_SCALE, ctx)
    var got = out.to_host(ctx)
    if len(got) != n:
        print("Test C FAIL — length", len(got))
        return False
    var mism = 0
    for i in range(n):
        if got[i] != torch_ref[i]:
            mism += 1
            if mism <= 6:
                print("  idx", i, "byte", Int(raw[i]), "got", got[i], "torch", torch_ref[i])
    if mism == 0:
        print("Test C (real ckpt block.4.attn1.to_q, torch ref) PASS (", n, "/", n, ")")
        return True
    print("Test C FAIL —", mism, "/", n, "mismatches vs torch")
    return False


def main() raises:
    var ctx = DeviceContext()
    var p = 0
    var t = 0

    var allb = _all_bytes()

    t += 1
    if _run_bytes(String("Test A (all 256 E4M3 bytes, scale=1.0)"), allb, 1.0, ctx):
        p += 1
    t += 1
    if _run_bytes(String("Test B1 (all 256 bytes, scale=0.5)"), allb, 0.5, ctx):
        p += 1
    t += 1
    if _run_bytes(String("Test B2 (all 256 bytes, scale=2.0)"), allb, 2.0, ctx):
        p += 1
    t += 1
    if _run_bytes(
        String("Test B3 (all 256 bytes, real scale)"), allb, _REAL_SCALE, ctx
    ):
        p += 1
    t += 1
    if _run_real(ctx):
        p += 1

    print("──────────────────────────────")
    print("fp8 dequant smoke summary:", p, "/", t)
    if p != t:
        raise Error("fp8 dequant smoke FAILED")
