# qwenimage_product_aspect_gate.mojo — seven-shape product dispatch contract.

from std.collections import List

from serenitymojo.training.aspect_buckets import (
    default_aspect_ladder, generate_aspect_buckets,
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)

comptime MEGAPIXELS = Float64(1.048576)
comptime ALIGN = 64
comptime EDGE_UNITS = 16
comptime PATCH = 2
comptime TEXT = 512


def main() raises:
    var buckets = generate_aspect_buckets(
        MEGAPIXELS, ALIGN, default_aspect_ladder()
    )
    if len(buckets) != DEFAULT_ASPECT_LADDER_LEN:
        raise Error("Qwen-Image product bucket count differs from compiled ladder")

    var latent_h = List[Int]()
    var latent_w = List[Int]()
    comptime for i in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_I = DEFAULT_ASPECT_LADDER_X100[i]
        comptime LH_I = aspect_lat_h_units(X100_I, EDGE_UNITS)
        comptime LW_I = aspect_lat_w_units(X100_I, EDGE_UNITS)
        comptime GRID_H_I = LH_I // PATCH
        comptime GRID_W_I = LW_I // PATCH
        comptime N_IMG_I = GRID_H_I * GRID_W_I
        comptime S_I = TEXT + N_IMG_I
        latent_h.append(LH_I)
        latent_w.append(LW_I)
        print(
            "[", i, "] aspect_x100=", X100_I,
            " pixels=", LW_I * 8, "x", LH_I * 8,
            " latent=", LH_I, "x", LW_I,
            " grid=", GRID_H_I, "x", GRID_W_I,
            " image_tokens=", N_IMG_I, " sequence=", S_I,
            " flash_aligned=", S_I % 128 == 0,
        )
        if buckets[i].height != LH_I * 8 or buckets[i].width != LW_I * 8:
            raise Error("Qwen-Image runtime/comptime bucket mismatch")
        if LH_I % PATCH != 0 or LW_I % PATCH != 0:
            raise Error("Qwen-Image latent is not patch-2 divisible")
        if LH_I % 4 != 0 or LW_I % 4 != 0:
            raise Error("Qwen-Image latent cannot use exact half-tile strides")

    for i in range(DEFAULT_ASPECT_LADDER_LEN):
        for j in range(i + 1, DEFAULT_ASPECT_LADDER_LEN):
            if latent_h[i] == latent_h[j] and latent_w[i] == latent_w[j]:
                raise Error("Qwen-Image compiled ladder contains duplicate shapes")
    if (
        latent_h[1] != latent_w[2] or latent_w[1] != latent_h[2]
        or latent_h[3] != latent_w[4] or latent_w[3] != latent_h[4]
        or latent_h[5] != latent_w[6] or latent_w[5] != latent_h[6]
    ):
        raise Error("Qwen-Image portrait/landscape transpose pairs drifted")
    print("PASS: Qwen-Image seven-shape dispatch/token/tile contract")
