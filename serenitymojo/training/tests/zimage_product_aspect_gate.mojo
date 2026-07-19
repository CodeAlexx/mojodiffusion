# zimage_product_aspect_gate.mojo — host-only product contract for the finite
# Z-Image 1MP aspect ladder. Proves the comptime dispatch shapes exactly match
# the SimpleTuner-parity runtime generator and remain valid for Z-Image's
# patch-2 tokenization and 3x3 half-latent tiled VAE fallback.

from std.collections import List

from serenitymojo.training.aspect_buckets import (
    default_aspect_ladder, generate_aspect_buckets,
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)

comptime PRODUCT_MEGAPIXELS = Float64(1.048576)
comptime PRODUCT_ALIGN = 64
comptime PRODUCT_EDGE_UNITS = 16
comptime PATCH = 2


def main() raises:
    var buckets = generate_aspect_buckets(
        PRODUCT_MEGAPIXELS, PRODUCT_ALIGN, default_aspect_ladder()
    )
    if len(buckets) != DEFAULT_ASPECT_LADDER_LEN:
        raise Error("GATE FAIL: runtime 1MP bucket count differs from comptime ladder")

    comptime for i in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_I = DEFAULT_ASPECT_LADDER_X100[i]
        comptime LH_I = aspect_lat_h_units(X100_I, PRODUCT_EDGE_UNITS)
        comptime LW_I = aspect_lat_w_units(X100_I, PRODUCT_EDGE_UNITS)
        comptime N_IMG_I = (LH_I // PATCH) * (LW_I // PATCH)
        var rt_h = buckets[i].height // 8
        var rt_w = buckets[i].width // 8
        print(
            "[", i, "] latent=", LH_I, "x", LW_I,
            " pixels=", LW_I * 8, "x", LH_I * 8,
            " image_tokens=", N_IMG_I,
        )
        if rt_h != LH_I or rt_w != LW_I:
            raise Error(String("GATE FAIL: runtime/comptime shape mismatch at index ") + String(i))
        if LH_I % 8 != 0 or LW_I % 8 != 0:
            raise Error("GATE FAIL: latent shape cannot be split into even half/quarter tiles")
        var tile_h = LH_I // 2
        var tile_w = LW_I // 2
        var stride_h = tile_h // 2
        var stride_w = tile_w // 2
        if tile_h + tile_h != LH_I or tile_w + tile_w != LW_I:
            raise Error("GATE FAIL: half-latent tile does not exactly divide canvas")
        if stride_h + tile_h > LH_I or stride_w + tile_w > LW_I:
            raise Error("GATE FAIL: middle tile exceeds latent canvas")
        var final_h_start = tile_h
        var final_w_start = tile_w
        if final_h_start + tile_h != LH_I or final_w_start + tile_w != LW_I:
            raise Error("GATE FAIL: final tile does not end on latent boundary")
        if N_IMG_I % 32 != 0:
            raise Error("GATE FAIL: Z-Image image-token sequence needs unexpected padding")

    print("PASS: Z-Image seven-shape 1MP dispatch/token/tile contract")
