# serenitymojo/models/minimax_h3/image_grid.mojo
#
# MiniMax-H3 conditioner image grid — unit 8 of the H3 port.
#
# Decides the resolution a keyframe or reference image is fed to Qwen3-VL at,
# and from that the number of vision tokens it contributes:
#
#   grid = (1, h_bar // patch_size, w_bar // patch_size)
#   num_vision_tokens = prod(grid) // merge_size**2
#
# That count is the input unit 7's presentation splices in as `<|image_pad|>`
# rows. Get it wrong and every row index after that block shifts, which moves
# every rotary coordinate and every AdaLN row assignment downstream of it.
#
# Source (op-for-op port):
#   transformers/models/qwen2_vl/image_processing_qwen2_vl.py  smart_resize
#   transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py
#     Qwen2VLImageProcessorFast — maps `size.shortest_edge` -> min_pixels and
#     `size.longest_edge` -> max_pixels, and imports the same `smart_resize`.
#
# CONFIG COMES FROM THE CHECKPOINT, NOT FROM THE QWEN2-VL DEFAULTS. The released
# `preprocessor_config.json` carries patch_size 16, merge_size 2,
# temporal_patch_size 2, shortest_edge 65536 (256x256) and longest_edge
# 16777216 (4096x4096) — measured from the conditioner we fetched. The ComfyUI
# implementation hardcodes min_pixels 3136 and max_pixels 12845056, the Qwen2-VL
# defaults, which are wrong for this checkpoint and would produce a different
# token count for most images.
#
# The pixel resampling itself (bilinear resize, mean/std normalize, patch
# flattening) is a separate unit; this one is the integer geometry that decides
# the sequence length.

from std.collections import List
from std.math import sqrt, floor, ceil

from serenitymojo.models.minimax_h3.packing import _round_half_even

# Measured from the released preprocessor_config.json of the H3 conditioner.
comptime MINIMAX_H3_VISION_PATCH_SIZE = 16
comptime MINIMAX_H3_VISION_MERGE_SIZE = 2
comptime MINIMAX_H3_VISION_TEMPORAL_PATCH = 2
comptime MINIMAX_H3_VISION_MIN_PIXELS = 65536
comptime MINIMAX_H3_VISION_MAX_PIXELS = 16777216
# The processor refuses anything more lopsided than this.
comptime MINIMAX_H3_VISION_MAX_ASPECT = Float64(200.0)


@fieldwise_init
struct MiniMaxH3ImageGrid(Copyable, Movable):
    """The resolution an image is resized to and the grid it becomes."""

    var height: Int
    var width: Int
    var grid_h: Int
    var grid_w: Int
    var num_vision_tokens: Int


def minimax_h3_smart_resize(
    height: Int,
    width: Int,
    factor: Int,
    min_pixels: Int,
    max_pixels: Int,
) raises -> List[Int]:
    """`smart_resize` (image_processing_qwen2_vl.py).

    Both axes end up divisible by `factor`, the pixel count lands inside
    [min_pixels, max_pixels], and the aspect ratio is preserved as closely as
    that allows. Returns [h_bar, w_bar].

    Three details that a rewrite tends to smooth over and must not:
      * `round` is Python's, i.e. half to EVEN;
      * the over-max branch divides TWICE — `h / beta / factor` — rather than
        dividing once by `beta * factor`, and the two differ in float;
      * only the over-max branch clamps to `factor`; the under-min branch does
        not, because it is already growing."""
    var tall = height if height > width else width
    var short = width if height > width else height
    if short <= 0:
        raise Error("MiniMax-H3: image dimensions must be positive")
    if Float64(tall) / Float64(short) > MINIMAX_H3_VISION_MAX_ASPECT:
        raise Error(
            "MiniMax-H3: absolute aspect ratio must be smaller than 200"
        )

    var factor_f = Float64(factor)
    var h_bar = Int(_round_half_even(Float64(height) / factor_f)) * factor
    var w_bar = Int(_round_half_even(Float64(width) / factor_f)) * factor

    if h_bar * w_bar > max_pixels:
        var beta = sqrt(Float64(height) * Float64(width) / Float64(max_pixels))
        h_bar = Int(floor(Float64(height) / beta / factor_f)) * factor
        w_bar = Int(floor(Float64(width) / beta / factor_f)) * factor
        if h_bar < factor:
            h_bar = factor
        if w_bar < factor:
            w_bar = factor
    elif h_bar * w_bar < min_pixels:
        var beta = sqrt(Float64(min_pixels) / (Float64(height) * Float64(width)))
        h_bar = Int(ceil(Float64(height) * beta / factor_f)) * factor
        w_bar = Int(ceil(Float64(width) * beta / factor_f)) * factor

    return [h_bar, w_bar]


def minimax_h3_image_grid(
    height: Int,
    width: Int,
    patch_size: Int = MINIMAX_H3_VISION_PATCH_SIZE,
    merge_size: Int = MINIMAX_H3_VISION_MERGE_SIZE,
    min_pixels: Int = MINIMAX_H3_VISION_MIN_PIXELS,
    max_pixels: Int = MINIMAX_H3_VISION_MAX_PIXELS,
) raises -> MiniMaxH3ImageGrid:
    """Resolve a source image size to its conditioner grid and token count.

    `factor` is `patch_size * merge_size`, so both grid axes are divisible by
    `merge_size` and the token count divides evenly."""
    var factor = patch_size * merge_size
    var resized = minimax_h3_smart_resize(height, width, factor, min_pixels, max_pixels)
    var grid_h = resized[0] // patch_size
    var grid_w = resized[1] // patch_size
    var tokens = grid_h * grid_w // (merge_size * merge_size)
    return MiniMaxH3ImageGrid(resized[0], resized[1], grid_h, grid_w, tokens)
