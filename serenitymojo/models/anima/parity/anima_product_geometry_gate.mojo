# Anima seven-shape geometry/3-axis-RoPE/tile gate against SerenityTrainer Diffusers.

from std.collections import List
from std.gpu.host import DeviceContext
from std.io.file import open

from serenitymojo.serve.anima_backend import _rope_tables


comptime REF = "/tmp/serenity_anima_product_geometry_ref.bin"
comptime RECORD_FLOATS = 31
comptime HEADS = 16
comptime HALF_HEAD_DIM = 64


def _abs(value: Float32) -> Float32:
    return value if value >= 0.0 else -value


def _close(label: String, actual: Float32, expected: Float32, tol: Float32) raises:
    if _abs(actual - expected) > tol:
        raise Error(
            label + String(" mismatch: actual=") + String(actual)
            + String(" expected=") + String(expected)
        )


def _check_shape[LH_: Int, LW_: Int](
    oracle: List[Float32], index: Int, ctx: DeviceContext
) raises:
    comptime NH_ = LH_ // 2
    comptime NW_ = LW_ // 2
    comptime NIMG_ = NH_ * NW_
    var base = index * RECORD_FLOATS
    if (
        Int(oracle[base + 2]) != LH_ or Int(oracle[base + 3]) != LW_
        or Int(oracle[base + 4]) != NH_ or Int(oracle[base + 5]) != NW_
        or Int(oracle[base + 6]) != NIMG_
        or Int(oracle[base + 27]) != NH_ - 1
        or Int(oracle[base + 28]) != NW_ - 1
        or Int(oracle[base + 29]) != 3
        or Int(oracle[base + 30]) != 512
    ):
        raise Error("Anima geometry mismatch at record " + String(index))

    var tile_h = LH_ // 2
    var tile_w = LW_ // 2
    if (
        Int(oracle[base + 7]) != tile_h
        or Int(oracle[base + 8]) != 0
        or Int(oracle[base + 9]) != tile_h // 2
        or Int(oracle[base + 10]) != tile_h
        or Int(oracle[base + 11]) != tile_w
        or Int(oracle[base + 12]) != 0
        or Int(oracle[base + 13]) != tile_w // 2
        or Int(oracle[base + 14]) != tile_w
    ):
        raise Error("Anima rectangular 3x3 VAE tile mismatch")

    var rope = _rope_tables(NH_, NW_, ctx)
    var cos_h = rope.cos.to_host(ctx)
    var sin_h = rope.sin.to_host(ctx)
    var last = ((NIMG_ - 1) * HEADS) * HALF_HEAD_DIM
    var dims = [0, 22, 23, 43, 44, 63]
    for probe in range(6):
        var dim = dims[probe]
        _close("Anima RoPE cos", cos_h[last + dim], oracle[base + 15 + 2 * probe], 3.0e-5)
        _close("Anima RoPE sin", sin_h[last + dim], oracle[base + 16 + 2 * probe], 3.0e-5)
    print("  PASS", Int(oracle[base]), "x", Int(oracle[base + 1]),
          "latent", LH_, "x", LW_, "grid", NH_, "x", NW_, "N_IMG", NIMG_)


def main() raises:
    var file = open(REF, "r")
    var raw = file.read_bytes()
    file.close()
    if len(raw) != 7 * RECORD_FLOATS * 4:
        raise Error("Anima oracle size mismatch; run the oracle first")
    var oracle = List[Float32](capacity=7 * RECORD_FLOATS)
    var ptr = raw.unsafe_ptr().bitcast[Float32]()
    for i in range(7 * RECORD_FLOATS):
        oracle.append(ptr[i])
    var ctx = DeviceContext()
    _check_shape[128, 128](oracle, 0, ctx)
    _check_shape[112, 144](oracle, 1, ctx)
    _check_shape[144, 112](oracle, 2, ctx)
    _check_shape[96, 168](oracle, 3, ctx)
    _check_shape[168, 96](oracle, 4, ctx)
    _check_shape[104, 160](oracle, 5, ctx)
    _check_shape[160, 104](oracle, 6, ctx)
    print("VERDICT: PASS — Anima SerenityTrainer geometry/RoPE/tile parity")
