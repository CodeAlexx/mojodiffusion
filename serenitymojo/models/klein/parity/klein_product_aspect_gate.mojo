# klein_product_aspect_gate.mojo — finite Klein/Flux2 product-shape contract.

from std.collections import List

from serenitymojo.training.aspect_buckets import (
    default_aspect_ladder, generate_aspect_buckets,
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)

comptime MEGAPIXELS = Float64(1.048576)
comptime ALIGN = 64
comptime EDGE_UNITS = 16
comptime TEXT = 512


def main() raises:
    var buckets = generate_aspect_buckets(
        MEGAPIXELS, ALIGN, default_aspect_ladder()
    )
    if len(buckets) != DEFAULT_ASPECT_LADDER_LEN:
        raise Error("Klein product bucket count differs from compiled ladder")

    var grid_h = List[Int]()
    var grid_w = List[Int]()
    comptime for i in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_I = DEFAULT_ASPECT_LADDER_X100[i]
        # The shared ladder is expressed in stride-8 VAE units. Klein packs a
        # 2x2 latent patch per token, so one grid cell maps to 16 output pixels.
        comptime LH_I = aspect_lat_h_units(X100_I, EDGE_UNITS) // 2
        comptime LW_I = aspect_lat_w_units(X100_I, EDGE_UNITS) // 2
        comptime N_IMG_I = LH_I * LW_I
        comptime S_I = TEXT + N_IMG_I
        grid_h.append(LH_I)
        grid_w.append(LW_I)
        print(
            "[", i, "] aspect_x100=", X100_I,
            " pixels=", LW_I * 16, "x", LH_I * 16,
            " packed_grid=", LH_I, "x", LW_I,
            " image_tokens=", N_IMG_I, " sequence=", S_I,
            " heads_4b=24 heads_9b=32",
        )
        if buckets[i].height != LH_I * 16 or buckets[i].width != LW_I * 16:
            raise Error("Klein runtime/comptime bucket mismatch")
        if LH_I % 4 != 0 or LW_I % 4 != 0:
            raise Error("Klein grid cannot use exact half-tile strides")

    for i in range(DEFAULT_ASPECT_LADDER_LEN):
        for j in range(i + 1, DEFAULT_ASPECT_LADDER_LEN):
            if grid_h[i] == grid_h[j] and grid_w[i] == grid_w[j]:
                raise Error("Klein compiled ladder contains duplicate shapes")
    if (
        grid_h[1] != grid_w[2] or grid_w[1] != grid_h[2]
        or grid_h[3] != grid_w[4] or grid_w[3] != grid_h[4]
        or grid_h[5] != grid_w[6] or grid_w[5] != grid_h[6]
    ):
        raise Error("Klein portrait/landscape transpose pairs drifted")
    print("PASS: Klein seven-shape 4B/9B dispatch/token/tile contract")
