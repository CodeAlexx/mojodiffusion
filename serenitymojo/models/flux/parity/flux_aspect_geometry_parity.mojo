# FLUX.1 aspect core gate.  This validates the finite seven-shape product
# ladder's packed geometry + dynamic schedule against the local SerenityTrainer/BFL
# oracle and probes the actual Mojo 3-axis RoPE tables at each rectangle.
#
# This gate exercises the same generic pack/unpack and RoPE specializations now
# dispatched by flux_backend. A real generated artifact is still required to
# raise the product manifest's experimental readiness/parity labels.

from std.collections import List
from std.gpu.host import DeviceContext
from std.io.file import open

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.pipeline.flux_sample_cli import (
    _pack_latent, _unpack_latent, _pack_latent_shape, _unpack_latent_shape,
)
from serenitymojo.sampling.flux1_dev import (
    build_flux1_packed_latent_plan,
    build_flux1_sigma_schedule,
)


comptime REF = "serenitymojo/models/flux/parity/flux_aspect_geometry_ref.bin"
comptime N_TXT = 512
comptime HEADS = 24
comptime HALF_HEAD_DIM = 64
comptime CHANNELS = 16
comptime RECORD_FLOATS = 34


def _abs(value: Float32) -> Float32:
    return value if value >= 0.0 else -value


def _check_close(label: String, actual: Float32, expected: Float32, tol: Float32) raises:
    var diff = _abs(actual - expected)
    if diff > tol:
        raise Error(
            label + String(" mismatch: actual=") + String(actual)
            + String(" expected=") + String(expected)
            + String(" diff=") + String(diff)
        )


def _check_record_geometry(oracle: List[Float32], record_index: Int) raises:
    var base = record_index * RECORD_FLOATS
    var width = Int(oracle[base + 0])
    var height = Int(oracle[base + 1])
    var plan = build_flux1_packed_latent_plan(width, height, N_TXT)
    if plan.latent_h != Int(oracle[base + 2]):
        raise Error("latent_h mismatch for " + String(width) + "x" + String(height))
    if plan.latent_w != Int(oracle[base + 3]):
        raise Error("latent_w mismatch for " + String(width) + "x" + String(height))
    if plan.packed_h != Int(oracle[base + 4]):
        raise Error("packed_h mismatch for " + String(width) + "x" + String(height))
    if plan.packed_w != Int(oracle[base + 5]):
        raise Error("packed_w mismatch for " + String(width) + "x" + String(height))
    if plan.image_tokens != Int(oracle[base + 6]):
        raise Error("image_tokens mismatch for " + String(width) + "x" + String(height))
    if plan.total_sequence != Int(oracle[base + 7]):
        raise Error("total_sequence mismatch for " + String(width) + "x" + String(height))
    var sigmas = build_flux1_sigma_schedule(4, plan.image_tokens)
    for i in range(5):
        _check_close(
            String("sigma[") + String(i) + String("] ")
            + String(width) + String("x") + String(height),
            sigmas[i], oracle[base + 8 + i], 1.0e-6,
        )


def _check_rope[N_IMG: Int](
    oracle: List[Float32], record_index: Int, img_h2: Int, img_w2: Int,
    ctx: DeviceContext,
) raises:
    var base = record_index * RECORD_FLOATS
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, HEADS, 128](
        img_h2, img_w2, ctx, STDtype.F32
    )
    var cos_h = rope[0].to_host(ctx)
    var sin_h = rope[1].to_host(ctx)
    var last_token = N_TXT + N_IMG - 1
    var row = (last_token * HEADS) * HALF_HEAD_DIM
    # FLUX axes_dims_rope=[16,56,56]: row frequency slots start at 8 and
    # column frequency slots start at 8 + 28 = 36.
    var slots = List[Int]()
    slots.append(8)
    slots.append(9)
    slots.append(36)
    slots.append(37)
    for i in range(4):
        _check_close(
            String("rope cos record ") + String(record_index) + String(" slot ") + String(slots[i]),
            cos_h[row + slots[i]], oracle[base + 13 + 2 * i], 2.0e-5,
        )
        _check_close(
            String("rope sin record ") + String(record_index) + String(" slot ") + String(slots[i]),
            sin_h[row + slots[i]], oracle[base + 14 + 2 * i], 2.0e-5,
        )
    print(
        "  PASS", Int(oracle[base]), "x", Int(oracle[base + 1]),
        "packed", img_h2, "x", img_w2, "N_IMG", N_IMG,
    )


def _check_pack[LH_: Int, LW_: Int, N_IMG_: Int](
    oracle: List[Float32], record_index: Int, ctx: DeviceContext,
) raises:
    comptime ELEMENTS_ = CHANNELS * LH_ * LW_
    var base = record_index * RECORD_FLOATS
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
        raise Error("FLUX 3x3 VAE tile geometry mismatch")

    var source_h = List[Float32](capacity=ELEMENTS_)
    for i in range(ELEMENTS_):
        source_h.append(Float32(i))
    var source = Tensor.from_host(
        source_h, [1, CHANNELS, LH_, LW_], STDtype.F32, ctx
    )
    var packed = _pack_latent_shape[LH_, LW_](source, ctx)
    var pshape = packed.shape()
    if pshape[0] != 1 or pshape[1] != N_IMG_ or pshape[2] != 64:
        raise Error("FLUX packed tensor shape mismatch")
    var packed_h = packed.to_host(ctx)
    for i in range(4):
        if packed_h[i] != oracle[base + 29 + i]:
            raise Error("FLUX pack ordering mismatch")
    if packed_h[len(packed_h) - 1] != oracle[base + 33]:
        raise Error("FLUX final packed scalar mismatch")
    var unpacked_h = _unpack_latent_shape[LH_, LW_](packed, ctx).to_host(ctx)
    if len(unpacked_h) != ELEMENTS_:
        raise Error("FLUX unpacked tensor length mismatch")
    for i in range(ELEMENTS_):
        if unpacked_h[i] != source_h[i]:
            raise Error("FLUX pack/unpack roundtrip mismatch")


def _check_square_compatibility(ctx: DeviceContext) raises:
    var source_h = List[Float32](capacity=16 * 128 * 128)
    for i in range(16 * 128 * 128):
        source_h.append(Float32(i))
    var source = Tensor.from_host(source_h, [1, 16, 128, 128], STDtype.F32, ctx)
    var fixed_packed = _pack_latent(source, ctx).to_host(ctx)
    var generic_packed = _pack_latent_shape[128, 128](source, ctx).to_host(ctx)
    if len(fixed_packed) != len(generic_packed):
        raise Error("FLUX fixed/generic square packed length mismatch")
    for i in range(len(fixed_packed)):
        if fixed_packed[i] != generic_packed[i]:
            raise Error("FLUX fixed/generic square pack mismatch")
    var fixed_unpacked = _unpack_latent(_pack_latent(source, ctx), ctx).to_host(ctx)
    for i in range(len(source_h)):
        if fixed_unpacked[i] != source_h[i]:
            raise Error("FLUX fixed square pack/unpack regression")
    print("  PASS fixed 1024-square compatibility wrappers")


def main() raises:
    var file = open(REF, "r")
    var raw = file.read_bytes()
    file.close()
    if len(raw) != 7 * RECORD_FLOATS * 4:
        raise Error("FLUX aspect oracle size mismatch; run flux_aspect_geometry_oracle.py")
    var oracle = List[Float32](capacity=7 * RECORD_FLOATS)
    var ptr = raw.unsafe_ptr().bitcast[Float32]()
    for i in range(7 * RECORD_FLOATS):
        oracle.append(ptr[i])

    for i in range(7):
        _check_record_geometry(oracle, i)

    var ctx = DeviceContext()
    _check_rope[4096](oracle, 0, 64, 64, ctx)
    _check_rope[4032](oracle, 1, 56, 72, ctx)
    _check_rope[4032](oracle, 2, 72, 56, ctx)
    _check_rope[4032](oracle, 3, 48, 84, ctx)
    _check_rope[4032](oracle, 4, 84, 48, ctx)
    _check_rope[4160](oracle, 5, 52, 80, ctx)
    _check_rope[4160](oracle, 6, 80, 52, ctx)
    _check_pack[128, 128, 4096](oracle, 0, ctx)
    _check_pack[112, 144, 4032](oracle, 1, ctx)
    _check_pack[144, 112, 4032](oracle, 2, ctx)
    _check_pack[96, 168, 4032](oracle, 3, ctx)
    _check_pack[168, 96, 4032](oracle, 4, ctx)
    _check_pack[104, 160, 4160](oracle, 5, ctx)
    _check_pack[160, 104, 4160](oracle, 6, ctx)
    _check_square_compatibility(ctx)
    print("VERDICT: PASS — seven-shape FLUX geometry/schedule/RoPE/pack oracle parity")
