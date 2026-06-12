# ops/tests/fp8_quant_smoke.mojo — round-trip gate for ops/fp8_quant.mojo.
#
# encode (fp8_quant.mojo, NEW) → decode (ops/fp8.mojo, the parity-gated
# Ideogram-4 dequant) must reproduce a bf16 weight within E4M3 precision:
#   * exactly-representable values (after per-row scaling) come back BIT-EXACT;
#   * random values: per-element error <= half-ULP of the scaled E4M3 grid —
#     <= 2^-4 relative for normals (3 mantissa bits), <= scale[row]*2^-9/2
#     absolute in the subnormal range — plus bf16 storage rounding;
#   * all-zero row → scale 1.0, bytes 0, decode 0.
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/ops/tests/fp8_quant_smoke.mojo -o /tmp/fp8_quant_smoke

from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, pi

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.ops.fp8_quant import fp8_e4m3_rowscale, fp8_e4m3_encode_perrow


def _gauss(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1 = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) / Float64(1 << 52)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2 = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) / Float64(1 << 52)
        if u1 < 1.0e-300:
            u1 = 1.0e-300
        var r = sqrt(Float64(-2.0) * flog(u1))
        out.append(Float32(r * fcos(Float64(2.0) * pi * u2)))
        i += 1
        if i < n:
            out.append(Float32(r * fsin(Float64(2.0) * pi * u2)))
            i += 1
    return out^


def main() raises:
    var ctx = DeviceContext()
    var fails = 0

    # ── Case 1: random gaussian weight [64, 256], typical LoRA-base stats ────
    var rows = 64
    var cols = 256
    var h = _gauss(rows * cols, UInt64(20260611))
    # scale row magnitudes differently to exercise per-row scales
    for r in range(rows):
        var amp = Float32(0.001) * Float32(r + 1)
        for c in range(cols):
            h[r * cols + c] *= amp
    # row 7 = all zeros (zero-row scale path)
    for c in range(cols):
        h[7 * cols + c] = 0.0
    var w = Tensor.from_host(h.copy(), [rows, cols], STDtype.BF16, ctx)
    var w_h = w.to_host(ctx)  # bf16-rounded reference

    var scale = fp8_e4m3_rowscale(w, ctx)
    var bytes_t = fp8_e4m3_encode_perrow(w, scale, ctx)
    if bytes_t.dtype() != STDtype.F8_E4M3:
        print("FAIL: bytes dtype", bytes_t.dtype().name())
        fails += 1
    var deq = fp8_e4m3_dequant_perrow_to_bf16(bytes_t, scale, ctx)
    var deq_h = deq.to_host(ctx)
    var scale_h = scale.to_host(ctx)

    if scale_h[7] != Float32(1.0):
        print("FAIL: zero-row scale", scale_h[7], "!= 1.0")
        fails += 1
    var max_rel = Float32(0.0)
    var max_abs_zero = Float32(0.0)
    for i in range(rows * cols):
        var r = i // cols
        var want = w_h[i]
        var got = deq_h[i]
        var d = got - want
        if d < 0:
            d = -d
        var a = want if want >= 0 else -want
        # tolerance: e4m3 has 3 mantissa bits -> ULP/value <= 2^-3, so RNE
        # half-ULP <= 6.25% relative for normals; subnormals are bounded
        # ABSOLUTELY by half the quantum (scale * 2^-9 / 2, + slack for the
        # non-exact F32 divide); + bf16 rounding of input/output storage.
        var tol = (
            a * Float32(0.0625)
            + scale_h[r] * Float32(0.001)
            + a * Float32(0.008)
        )
        if d > tol:
            if fails < 5:
                print("FAIL elem", i, " ref", want, " got", got, " tol", tol)
            fails += 1
        if a > 0 and d / a > max_rel:
            max_rel = d / a
        if want == 0 and d > max_abs_zero:
            max_abs_zero = d
    print("case1 random [64x256]: max_rel(nonzero)", max_rel,
          " max_abs(zero elems)", max_abs_zero)

    # ── Case 2: exactly representable values come back bit-exact ────────────
    # Row absmax 448 → scale 1.0 → E4M3 grid values encode/decode exactly.
    var ex: List[Float32] = [
        448.0, -448.0, 1.0, -1.0, 1.5, 0.5, 0.25, 240.0,
        0.015625, -0.015625, 0.001953125, 0.0, 2.0, -3.5, 96.0, 20.0,
    ]
    var w2 = Tensor.from_host(ex.copy(), [1, 16], STDtype.BF16, ctx)
    var s2 = fp8_e4m3_rowscale(w2, ctx)
    var s2_h = s2.to_host(ctx)
    if s2_h[0] != Float32(1.0):
        print("FAIL: case2 scale", s2_h[0], "!= 1.0 (absmax 448)")
        fails += 1
    var b2 = fp8_e4m3_encode_perrow(w2, s2, ctx)
    var d2 = fp8_e4m3_dequant_perrow_to_bf16(b2, s2, ctx)
    var d2_h = d2.to_host(ctx)
    for i in range(16):
        if d2_h[i] != ex[i]:
            print("FAIL case2 elem", i, " ref", ex[i], " got", d2_h[i])
            fails += 1
    print("case2 exact-grid [16]: done")

    # ── Case 3: saturation — values above 448 (after scale) clamp to 448 ────
    # Force scale=1.0 by building it on a max-448 row, then encode a row that
    # exceeds the grid; decode must give exactly ±448.
    var ex3: List[Float32] = [448.0, 460.0, -470.0, 1.0]
    var w3 = Tensor.from_host(ex3.copy(), [1, 4], STDtype.BF16, ctx)
    var ones: List[Float32] = [1.0]
    var s3 = Tensor.from_host(ones.copy(), [1], STDtype.F32, ctx)
    var b3 = fp8_e4m3_encode_perrow(w3, s3, ctx)
    var d3 = fp8_e4m3_dequant_perrow_to_bf16(b3, s3, ctx)
    var d3_h = d3.to_host(ctx)
    if (
        d3_h[0] != Float32(448.0) or d3_h[1] != Float32(448.0)
        or d3_h[2] != Float32(-448.0) or d3_h[3] != Float32(1.0)
    ):
        print("FAIL case3 saturation:", d3_h[0], d3_h[1], d3_h[2], d3_h[3])
        fails += 1
    print("case3 saturation [4]: done")
    if fails == 0:
        print("PASS: fp8_quant round-trip smoke")
    else:
        print("FAIL count:", fails)
        raise Error("fp8_quant_smoke failed")
