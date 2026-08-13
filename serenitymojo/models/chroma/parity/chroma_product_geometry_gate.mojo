# Chroma seven-shape core gate against the local creator-stack oracle.
# Exercises the actual generic pack/unpack and FLUX 3-axis RoPE used by Chroma.

from std.collections import List
from max.gpu.host import DeviceContext
from std.io.file import open

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.pipeline.chroma_pipeline_1024_multistep import (
    _pack_latent, _unpack_latent, _pack_latent_shape, _unpack_latent_shape,
)
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule


comptime REF = "serenitymojo/models/chroma/parity/chroma_product_geometry_ref.bin"
comptime N_TXT = 512
comptime HEADS = 24
comptime HALF_HEAD_DIM = 64
comptime CHANNELS = 16
comptime RECORD_FLOATS = 34


def _abs(value: Float32) -> Float32:
    return value if value >= 0.0 else -value


def _check_close(label: String, actual: Float32, expected: Float32, tol: Float32) raises:
    if _abs(actual - expected) > tol:
        raise Error(
            label + String(" mismatch: actual=") + String(actual)
            + String(" expected=") + String(expected)
        )


def _check_shape[LH_: Int, LW_: Int, N_IMG_: Int](
    oracle: List[Float32], record_index: Int, ctx: DeviceContext
) raises:
    comptime GRID_H_ = LH_ // 2
    comptime GRID_W_ = LW_ // 2
    comptime ELEMENTS_ = CHANNELS * LH_ * LW_
    var base = record_index * RECORD_FLOATS
    if (
        Int(oracle[base + 2]) != LH_ or Int(oracle[base + 3]) != LW_
        or Int(oracle[base + 4]) != GRID_H_
        or Int(oracle[base + 5]) != GRID_W_
        or Int(oracle[base + 6]) != N_IMG_
        or Int(oracle[base + 7]) != N_TXT + N_IMG_
    ):
        raise Error("Chroma oracle geometry mismatch at record " + String(record_index))

    var sigmas = build_flux1_sigma_schedule(4, N_IMG_)
    for i in range(5):
        _check_close("sigma", sigmas[i], oracle[base + 8 + i], 1.0e-6)

    var rope = build_flux1_rope_tables[N_IMG_, N_TXT, HEADS, 128](
        GRID_H_, GRID_W_, ctx, STDtype.F32
    )
    var cos_h = rope[0].to_host(ctx)
    var sin_h = rope[1].to_host(ctx)
    var last_row = ((N_TXT + N_IMG_ - 1) * HEADS) * HALF_HEAD_DIM
    var slots = List[Int]()
    slots.append(8)
    slots.append(9)
    slots.append(36)
    slots.append(37)
    for i in range(4):
        _check_close("rope cos", cos_h[last_row + slots[i]], oracle[base + 13 + 2 * i], 2.0e-5)
        _check_close("rope sin", sin_h[last_row + slots[i]], oracle[base + 14 + 2 * i], 2.0e-5)

    var tile_h = LH_ // 2
    var tile_w = LW_ // 2
    if (
        Int(oracle[base + 21]) != tile_h
        or Int(oracle[base + 22]) != 0
        or Int(oracle[base + 23]) != tile_h // 2
        or Int(oracle[base + 24]) != tile_h
        or Int(oracle[base + 25]) != tile_w
        or Int(oracle[base + 26]) != 0
        or Int(oracle[base + 27]) != tile_w // 2
        or Int(oracle[base + 28]) != tile_w
    ):
        raise Error("Chroma 3x3 VAE tile geometry mismatch")

    var source_h = List[Float32](capacity=ELEMENTS_)
    for i in range(ELEMENTS_):
        source_h.append(Float32(i))
    var source = Tensor.from_host(
        source_h, [1, CHANNELS, LH_, LW_], STDtype.F32, ctx
    )
    var packed = _pack_latent_shape[LH_, LW_](source, ctx)
    var pshape = packed.shape()
    if pshape[0] != 1 or pshape[1] != N_IMG_ or pshape[2] != 64:
        raise Error("Chroma packed tensor shape mismatch")
    var packed_h = packed.to_host(ctx)
    for i in range(4):
        if packed_h[i] != oracle[base + 29 + i]:
            raise Error("Chroma pack ordering mismatch")
    if packed_h[len(packed_h) - 1] != oracle[base + 33]:
        raise Error("Chroma final packed scalar mismatch")
    var unpacked = _unpack_latent_shape[LH_, LW_](packed, ctx)
    var unpacked_h = unpacked.to_host(ctx)
    if len(unpacked_h) != ELEMENTS_:
        raise Error("Chroma unpacked tensor length mismatch")
    for i in range(ELEMENTS_):
        if unpacked_h[i] != source_h[i]:
            raise Error("Chroma pack/unpack roundtrip mismatch at " + String(i))
    print("  PASS", Int(oracle[base]), "x", Int(oracle[base + 1]),
          "latent", LH_, "x", LW_, "grid", GRID_H_, "x", GRID_W_,
          "N_IMG", N_IMG_)


def _check_square_compatibility(ctx: DeviceContext) raises:
    var source_h = List[Float32](capacity=16 * 128 * 128)
    for i in range(16 * 128 * 128):
        source_h.append(Float32(i))
    var source = Tensor.from_host(source_h, [1, 16, 128, 128], STDtype.F32, ctx)
    var fixed_packed = _pack_latent(source, ctx).to_host(ctx)
    var generic_packed = _pack_latent_shape[128, 128](source, ctx).to_host(ctx)
    if len(fixed_packed) != len(generic_packed):
        raise Error("Chroma fixed/generic square packed length mismatch")
    for i in range(len(fixed_packed)):
        if fixed_packed[i] != generic_packed[i]:
            raise Error("Chroma fixed/generic square pack mismatch")
    var fixed_unpacked = _unpack_latent(
        _pack_latent(source, ctx), ctx
    ).to_host(ctx)
    if len(fixed_unpacked) != len(source_h):
        raise Error("Chroma fixed square unpack length mismatch")
    for i in range(len(source_h)):
        if fixed_unpacked[i] != source_h[i]:
            raise Error("Chroma fixed square pack/unpack regression")
    print("  PASS fixed 1024-square compatibility wrappers")


def main() raises:
    var file = open(REF, "r")
    var raw = file.read_bytes()
    file.close()
    if len(raw) != 7 * RECORD_FLOATS * 4:
        raise Error("Chroma product oracle size mismatch; run the oracle first")
    var oracle = List[Float32](capacity=7 * RECORD_FLOATS)
    var ptr = raw.unsafe_ptr().bitcast[Float32]()
    for i in range(7 * RECORD_FLOATS):
        oracle.append(ptr[i])

    var ctx = DeviceContext()
    _check_shape[128, 128, 4096](oracle, 0, ctx)
    _check_shape[112, 144, 4032](oracle, 1, ctx)
    _check_shape[144, 112, 4032](oracle, 2, ctx)
    _check_shape[96, 168, 4032](oracle, 3, ctx)
    _check_shape[168, 96, 4032](oracle, 4, ctx)
    _check_shape[104, 160, 4160](oracle, 5, ctx)
    _check_shape[160, 104, 4160](oracle, 6, ctx)
    _check_square_compatibility(ctx)
    print("VERDICT: PASS — Chroma product geometry/pack/RoPE/tile oracle parity")
