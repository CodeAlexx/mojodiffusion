# Ideogram-4 seven-shape geometry/MRoPE/tile gate against the torchref oracle.

from std.collections import List
from max.gpu.host import DeviceContext
from std.io.file import open

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor


comptime REF = "/tmp/serenity_ideogram4_product_geometry_ref.bin"
comptime RECORD_FLOATS = 537


def _abs(value: Float32) -> Float32:
    return value if value >= 0.0 else -value


def _close(label: String, actual: Float32, expected: Float32, tol: Float32) raises:
    if _abs(actual - expected) > tol:
        raise Error(
            label + String(" mismatch: actual=") + String(actual)
            + String(" expected=") + String(expected)
        )


def _check_shape[GH_: Int, GW_: Int](
    oracle: List[Float32], index: Int, ctx: DeviceContext
) raises:
    comptime NIMG_ = GH_ * GW_
    comptime VAE_H_ = 2 * GH_
    comptime VAE_W_ = 2 * GW_
    var base = index * RECORD_FLOATS
    if (
        Int(oracle[base + 2]) != GH_ or Int(oracle[base + 3]) != GW_
        or Int(oracle[base + 4]) != NIMG_
        or Int(oracle[base + 5]) != 1024 + NIMG_
        or Int(oracle[base + 6]) != VAE_H_
        or Int(oracle[base + 7]) != VAE_W_
    ):
        raise Error("Ideogram-4 geometry mismatch at record " + String(index))

    var positions = List[Float32]()
    positions.append(oracle[base + 8])
    positions.append(oracle[base + 9])
    positions.append(oracle[base + 10])
    var pos = Tensor.from_host(positions^, [1, 1, 3], STDtype.F32, ctx)
    var sections = [24, 20, 20]
    var rope = build_ideogram4_mrope(
        pos, 256, sections, Float32(5000000.0), ctx, STDtype.F32
    )
    var ref_cos = List[Float32](capacity=256)
    var ref_sin = List[Float32](capacity=256)
    for dim in range(256):
        ref_cos.append(oracle[base + 17 + dim])
        ref_sin.append(oracle[base + 273 + dim])
    var parity = ParityHarness(0.999)
    var cos_result = parity.compare(rope[0], ref_cos, ctx)
    var sin_result = parity.compare(rope[1], ref_sin, ctx)
    if not cos_result.passed or not sin_result.passed:
        raise Error("Ideogram-4 full MRoPE vector parity failed")
    # Large IMAGE_POSITION_OFFSET angles accumulate a measured ~3e-3 absolute
    # F32 trig delta while retaining >=0.999 vector cosine. Keep explicit H/W
    # interleave probes so a rectangular-axis swap cannot hide in the aggregate.
    var cos_h = rope[0].to_host(ctx)
    var sin_h = rope[1].to_host(ctx)
    var dims = [1, 2, 58, 59, 61, 127]
    for probe in range(6):
        var dim = dims[probe]
        _close("MRoPE cos", cos_h[dim], ref_cos[dim], 5.0e-3)
        _close("MRoPE sin", sin_h[dim], ref_sin[dim], 5.0e-3)

    var tile_h = VAE_H_ // 2
    var tile_w = VAE_W_ // 2
    if (
        Int(oracle[base + 529]) != tile_h
        or Int(oracle[base + 530]) != 0
        or Int(oracle[base + 531]) != tile_h // 2
        or Int(oracle[base + 532]) != tile_h
        or Int(oracle[base + 533]) != tile_w
        or Int(oracle[base + 534]) != 0
        or Int(oracle[base + 535]) != tile_w // 2
        or Int(oracle[base + 536]) != tile_w
    ):
        raise Error("Ideogram-4 rectangular 3x3 VAE tile mismatch")
    print("  PASS", Int(oracle[base]), "x", Int(oracle[base + 1]),
          "grid", GH_, "x", GW_, "N_IMG", NIMG_)


def main() raises:
    var file = open(REF, "r")
    var raw = file.read_bytes()
    file.close()
    if len(raw) != 7 * RECORD_FLOATS * 4:
        raise Error("Ideogram-4 oracle size mismatch; run the oracle first")
    var oracle = List[Float32](capacity=7 * RECORD_FLOATS)
    var ptr = raw.unsafe_ptr().bitcast[Float32]()
    for i in range(7 * RECORD_FLOATS):
        oracle.append(ptr[i])
    var ctx = DeviceContext()
    _check_shape[64, 64](oracle, 0, ctx)
    _check_shape[56, 72](oracle, 1, ctx)
    _check_shape[72, 56](oracle, 2, ctx)
    _check_shape[48, 84](oracle, 3, ctx)
    _check_shape[84, 48](oracle, 4, ctx)
    _check_shape[52, 80](oracle, 5, ctx)
    _check_shape[80, 52](oracle, 6, ctx)
    print("VERDICT: PASS — Ideogram-4 creator geometry/MRoPE/tile parity")
