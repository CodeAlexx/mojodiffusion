# Host-only SD3.5 Large product aspect/token/position/tile contract.
# Output is diffed byte-for-byte against sd35_product_geometry_oracle.py.

from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_POS_EMBED_GRID, SD3_LARGE_TEXT_TOKENS,
    build_sd3_large_token_plan, sd3_lowmem_tile_start,
    sd3_pos_crop_origin, sd3_pos_crop_source_token,
)
from serenitymojo.training.aspect_buckets import (
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
    default_aspect_ladder, generate_aspect_buckets,
)

comptime EDGE_UNITS = 16
comptime PRODUCT_MEGAPIXELS = Float64(1.048576)
comptime PRODUCT_ALIGN = 64


def main() raises:
    var buckets = generate_aspect_buckets(
        PRODUCT_MEGAPIXELS, PRODUCT_ALIGN, default_aspect_ladder()
    )
    if len(buckets) != DEFAULT_ASPECT_LADDER_LEN:
        raise Error("SD3 product gate: runtime ladder length mismatch")

    comptime for i in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_I = DEFAULT_ASPECT_LADDER_X100[i]
        comptime LH_I = aspect_lat_h_units(X100_I, EDGE_UNITS)
        comptime LW_I = aspect_lat_w_units(X100_I, EDGE_UNITS)
        comptime PH_I = LH_I // 2
        comptime PW_I = LW_I // 2
        var width = LW_I * 8
        var height = LH_I * 8
        if buckets[i].width != width or buckets[i].height != height:
            raise Error("SD3 product gate: runtime/comptime aspect mismatch")
        var plan = build_sd3_large_token_plan(
            width, height, SD3_LARGE_TEXT_TOKENS
        )
        if plan.patch_grid_h != PH_I or plan.patch_grid_w != PW_I:
            raise Error("SD3 product gate: patch-grid mismatch")
        if plan.image_tokens != PH_I * PW_I:
            raise Error("SD3 product gate: image-token mismatch")

        var top = sd3_pos_crop_origin(SD3_LARGE_POS_EMBED_GRID, PH_I)
        var left = sd3_pos_crop_origin(SD3_LARGE_POS_EMBED_GRID, PW_I)
        var pos0 = sd3_pos_crop_source_token(
            SD3_LARGE_POS_EMBED_GRID, PH_I, PW_I, 0, 0
        )
        var poslast = sd3_pos_crop_source_token(
            SD3_LARGE_POS_EMBED_GRID, PH_I, PW_I, PH_I - 1, PW_I - 1
        )
        var tile_h = LH_I // 4
        var tile_w = LW_I // 4
        var h0 = sd3_lowmem_tile_start(LH_I, tile_h, 0)
        var h1 = sd3_lowmem_tile_start(LH_I, tile_h, 1)
        var h2 = sd3_lowmem_tile_start(LH_I, tile_h, 2)
        var h3 = sd3_lowmem_tile_start(LH_I, tile_h, 3)
        var h4 = sd3_lowmem_tile_start(LH_I, tile_h, 4)
        var w0 = sd3_lowmem_tile_start(LW_I, tile_w, 0)
        var w1 = sd3_lowmem_tile_start(LW_I, tile_w, 1)
        var w2 = sd3_lowmem_tile_start(LW_I, tile_w, 2)
        var w3 = sd3_lowmem_tile_start(LW_I, tile_w, 3)
        var w4 = sd3_lowmem_tile_start(LW_I, tile_w, 4)
        if h0 != 0 or w0 != 0 or h4 + tile_h != LH_I or w4 + tile_w != LW_I:
            raise Error("SD3 product gate: VAE tile endpoints do not cover canvas")
        if not (h0 < h1 and h1 < h2 and h2 < h3 and h3 < h4):
            raise Error("SD3 product gate: VAE height offsets not monotonic")
        if not (w0 < w1 and w1 < w2 and w2 < w3 and w3 < w4):
            raise Error("SD3 product gate: VAE width offsets not monotonic")

        var row = String(i) + "," + String(width) + "," + String(height)
        row += "," + String(LH_I) + "," + String(LW_I)
        row += "," + String(PH_I) + "," + String(PW_I)
        row += "," + String(plan.image_tokens) + "," + String(plan.total_sequence)
        row += "," + String(top) + "," + String(left)
        row += "," + String(pos0) + "," + String(poslast)
        row += "," + String(tile_h) + "," + String(h0) + "," + String(h1)
        row += "," + String(h2) + "," + String(h3) + "," + String(h4)
        row += "," + String(tile_w) + "," + String(w0) + "," + String(w1)
        row += "," + String(w2) + "," + String(w3) + "," + String(w4)
        print(row)
