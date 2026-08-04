# serenitymojo/pipeline/minimax_h3_vision_preprocess.mojo
#
# The Qwen3-VL IMAGE PREPROCESSOR for MiniMax-H3 keyframes: canvas pixels ->
# the vision tower's `[num_patches, 1536]` patch rows. Host only: no Tensor, no
# DeviceContext, no GPU.
#
# This closes seam (a) of `pipeline/minimax_h3_i2va.mojo`. Its output is exactly
# what `models/text_encoder/minimax_h3_qwen3vl_vision.mojo::minimax_h3_vision_forward`
# takes as `pixel_patches`.
#
# ── THE NORMALIZATION IS *NOT* THE VIDEO VAE'S, AND THEY LIVE IN ONE FILE ──
# The conditioner normalizes with `image_mean = image_std = 0.5`
# (processor/preprocessor_config.json, verified against the loaded processor).
# The VIDEO VAE normalizes with ImageNet constants (packing.py:70-71). The
# keyframe pipeline applies BOTH, to the SAME pixels, a few dozen lines apart —
# the VAE constants for the condition rows, these for the conditioner. Using one
# where the other belongs produces a plausible tensor of the right shape and a
# wrong conditioning, which is why both call sites name which is which.
#
# Measured exact form, not inferred: transformers fuses rescale into normalize
# (`_fuse_mean_std_and_rescale_factor`) as `mean * (1/rescale_factor)` with
# `rescale_factor = 1/255`, and `1.0/(1.0/255)` is exactly 255.0, so the fused
# constants are 127.5 / 127.5 and the operation is
#     (uint8 -> float32 - 127.5) / 127.5
# NOT `(x/255 - 0.5)/0.5`, which differs in the last ulp. Verified bit-exact
# end-to-end against the real processor before this file was written.
#
# ── SMART_RESIZE IS THE IDENTITY FOR EVERY REAL KEYFRAME, AND THAT IS WHY
#    THERE IS NO RESAMPLER HERE ──────────────────────────────────────────────
# `resolve_canvas_size` always emits a 768-pixel short edge (packing.py:155-158)
# with an aspect ratio capped at 4, so the smallest possible canvas is
# 768 x 768 = 589,824 pixels — far above the processor's 65,536 `min_pixels` —
# and both axes are already multiples of 32 = patch_size * merge_size. Under
# those conditions `smart_resize` returns the input dimensions unchanged.
# MEASURED at 1184x768, 1344x768 and 832x480: identity in every case, and the
# no-resize pipeline reproduces the processor BIT-EXACTLY.
#
# So this module does NOT implement a resampler, and does not pretend to: when
# `smart_resize` would change the size it RAISES, naming what would be needed.
# That case is reachable only by a sub-65,536-pixel canvas, i.e. a tiny dev
# geometry, never a real request. Implementing it means reproducing
# torchvision's BICUBIC-with-antialias on uint8 — a different and harder filter
# than the PIL LANCZOS already ported in `minimax_h3_keyframe_image.mojo`, and
# emphatically not the same thing as PIL's bicubic. Guessing it would be worse
# than refusing it.
#
# Gate: pipeline/parity/minimax_h3_vision_preprocess_probe.mojo — bit-exact
# against the real processor's own `pixel_values`. Host, no GPU.

from std.collections import List

from serenitymojo.models.minimax_h3.image_grid import (
    MINIMAX_H3_VISION_MAX_PIXELS,
    MINIMAX_H3_VISION_MERGE_SIZE,
    MINIMAX_H3_VISION_MIN_PIXELS,
    MINIMAX_H3_VISION_PATCH_SIZE,
    MINIMAX_H3_VISION_TEMPORAL_PATCH,
    minimax_h3_smart_resize,
)
from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage

# The FUSED rescale+normalize constants. See this file's header: these are
# `0.5 * (1 / rescale_factor)` with `rescale_factor = 1/255`, i.e. exactly 127.5,
# and NOT a `/255` followed by a `-0.5`.
comptime MINIMAX_H3_VISION_FUSED_MEAN = Float32(127.5)
comptime MINIMAX_H3_VISION_FUSED_STD = Float32(127.5)

# `channels * temporal_patch * patch * patch` = 3 * 2 * 16 * 16.
comptime MINIMAX_H3_VISION_ROW_WIDTH = (
    3
    * MINIMAX_H3_VISION_TEMPORAL_PATCH
    * MINIMAX_H3_VISION_PATCH_SIZE
    * MINIMAX_H3_VISION_PATCH_SIZE
)


def minimax_h3_vision_resize_is_identity(height: Int, width: Int) raises -> Bool:
    """Whether `smart_resize` leaves this size alone.

    Uses the GATED `minimax_h3_smart_resize` rather than re-deriving the
    condition, so "would the processor resize this" is answered by the same code
    that computes the grid the presentation counts tokens from."""
    var resized = minimax_h3_smart_resize(
        height,
        width,
        MINIMAX_H3_VISION_PATCH_SIZE * MINIMAX_H3_VISION_MERGE_SIZE,
        MINIMAX_H3_VISION_MIN_PIXELS,
        MINIMAX_H3_VISION_MAX_PIXELS,
    )
    return resized[0] == height and resized[1] == width


def minimax_h3_vision_patch_rows(
    image: MiniMaxH3RgbImage,
) raises -> List[Float32]:
    """A canvas-sized keyframe -> the tower's `[grid_h*grid_w, 1536]` rows.

    Op order, from `Qwen2VLImageProcessorFast._preprocess`:
      1. (resize — IDENTITY here, see the header; refused otherwise)
      2. fused rescale+normalize: `(x - 127.5) / 127.5`
      3. temporal-patch duplication: an image is ONE frame, and
         `1 % temporal_patch_size != 0`, so the LAST frame is repeated to make
         2. For a still that means the frame is simply duplicated — the tower
         always sees a 2-deep temporal patch even for an image.
      4. `view(1, grid_t, T, C, gh/M, M, P, gw/M, M, P)` then
         `permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)` then reshape.

    THE PERMUTE IS THE WHOLE THING. It puts the rows in MERGE-BLOCK order
    (`gh/M, gw/M, M, M`) — the same order `minimax_h3_vision_rotary_positions`
    assigns coordinates in — and lays each row out as
    `(channel, temporal, patch_h, patch_w)`. Raster order is shape-identical and
    wrong in every row, and a within-row order of (t, c, h, w) is shape-identical
    and wrong in every value."""
    image.validate()
    var h = image.height
    var w = image.width

    if not minimax_h3_vision_resize_is_identity(h, w):
        var want = minimax_h3_smart_resize(
            h, w,
            MINIMAX_H3_VISION_PATCH_SIZE * MINIMAX_H3_VISION_MERGE_SIZE,
            MINIMAX_H3_VISION_MIN_PIXELS,
            MINIMAX_H3_VISION_MAX_PIXELS,
        )
        raise Error(
            String("minimax_h3_vision_preprocess: this ") + String(w) + "x"
            + String(h) + " image needs smart_resize to " + String(want[1]) + "x"
            + String(want[0]) + ", and the RESAMPLER IS NOT IMPLEMENTED."
            " Every real keyframe canvas is identity here (a 768 short edge is"
            " >= 589,824 pixels, well above the 65,536 min_pixels), so this is"
            " reachable only from a sub-65k dev geometry. Implementing it means"
            " reproducing torchvision's BICUBIC-with-antialias on uint8 — NOT"
            " PIL bicubic, and not the PIL LANCZOS already ported in"
            " pipeline/minimax_h3_keyframe_image.mojo. Feed a canvas-sized"
            " keyframe (which is what before_encoder.py:190-193 produces) or"
            " implement that filter."
        )

    var patch = MINIMAX_H3_VISION_PATCH_SIZE
    var merge = MINIMAX_H3_VISION_MERGE_SIZE
    var tpatch = MINIMAX_H3_VISION_TEMPORAL_PATCH
    if h % patch != 0 or w % patch != 0:
        raise Error("minimax_h3_vision_preprocess: size must divide the patch")
    var gh = h // patch
    var gw = w // patch
    if gh % merge != 0 or gw % merge != 0:
        raise Error("minimax_h3_vision_preprocess: grid must divide the merge")

    var rows = List[Float32]()
    rows.resize(gh * gw * MINIMAX_H3_VISION_ROW_WIDTH, Float32(0.0))

    # Rows sweep merge-blocks (bh, bw) then within-block (ih, iw); each row is
    # (c, t, ph, pw). The temporal axis is the duplicated still, so both t
    # slices read the SAME pixel — written explicitly rather than by copying the
    # buffer, because the duplication is a property of the layout, not an
    # optimization to elide.
    var row = 0
    for bh in range(gh // merge):
        for bw in range(gw // merge):
            for ih in range(merge):
                for iw in range(merge):
                    var patch_y = (bh * merge + ih) * patch
                    var patch_x = (bw * merge + iw) * patch
                    var base = row * MINIMAX_H3_VISION_ROW_WIDTH
                    for c in range(3):
                        for t in range(tpatch):
                            for py in range(patch):
                                for px in range(patch):
                                    var src = (
                                        ((patch_y + py) * w + (patch_x + px)) * 3 + c
                                    )
                                    var v = Float32(Int(image.pixels[src]))
                                    var at = (
                                        base
                                        + ((c * tpatch + t) * patch + py) * patch
                                        + px
                                    )
                                    rows[at] = (
                                        v - MINIMAX_H3_VISION_FUSED_MEAN
                                    ) / MINIMAX_H3_VISION_FUSED_STD
                    row += 1
    return rows^


def minimax_h3_vision_patch_grid(image: MiniMaxH3RgbImage) raises -> List[Int]:
    """`[grid_t, grid_h, grid_w]` for a still keyframe — `grid_t` is always 1.

    Returned alongside the rows because the tower takes both, and deriving the
    grid separately at the call site is how a row buffer and a grid drift into
    disagreeing about the same image."""
    image.validate()
    if not minimax_h3_vision_resize_is_identity(image.height, image.width):
        raise Error(
            "minimax_h3_vision_preprocess: grid requested for a size that would"
            " be resized — see minimax_h3_vision_patch_rows"
        )
    return [
        1,
        image.height // MINIMAX_H3_VISION_PATCH_SIZE,
        image.width // MINIMAX_H3_VISION_PATCH_SIZE,
    ]


# ═════════════════════════════════════════════════════════════════════════════
# The VIDEO half — what a ref2va VIDEO reference's vision blocks are built from.
#
# The video processor (`Qwen3VLVideoProcessor`, processor/
# video_preprocessor_config.json) shares the image path's patch/merge/temporal
# geometry and its image_mean = image_std = 0.5, but carries its OWN pixel
# bounds: `size.shortest_edge = 4096`, `size.longest_edge = 25165824` — NOT the
# image processor's 65,536 / 16,777,216. A 768-short-edge reference canvas
# (>= 589,824 px/frame) sits comfortably inside them, so smart_resize is the
# identity for every real reference video, same argument as the image path.
#
# MEASURED BIT-EXACT against the real processor (2026-08-04, the trimmed
# ref2va clip, 2 sampled frames at 1344x768 -> pixel_values_videos
# [4032, 1536]): the fused `(uint8 - 127.5) / 127.5` normalization and the
#     view(gt, tpatch, C, gh/M, M, P, gw/M, M, P)
#     permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
# row layout — i.e. rows sweep (temporal block, merge-block bh, bw, ih, iw)
# and each row is laid out (channel, temporal, patch_h, patch_w), exactly the
# image path's law except the two temporal slots read two DIFFERENT frames.
# The two-step `(x/255 - 0.5)/0.5` differs in the last ulp (max_abs 5.9e-8)
# and is NOT what the processor computes.
#
# An odd sampled-frame count is repeat-padded with the LAST frame to fill the
# temporal patch — the same padding `sample_reference_video_frames` already
# assumes when it repeats the last timestamp (packing_ref2va.py:707).
#
# Gate: pipeline/parity/minimax_h3_ref2va_presentation_probe.mojo — bit-exact
# against the real processor's own `pixel_values_videos`. Host, no GPU.
# ═════════════════════════════════════════════════════════════════════════════

comptime MINIMAX_H3_VIDEO_VISION_MIN_PIXELS = 4096
comptime MINIMAX_H3_VIDEO_VISION_MAX_PIXELS = 25165824


def minimax_h3_video_vision_resize_is_identity(
    height: Int, width: Int
) raises -> Bool:
    """Whether the VIDEO processor's smart_resize leaves this size alone.

    Same gated `minimax_h3_smart_resize`, the video processor's own pixel
    bounds (see the header above — they differ from the image processor's)."""
    var resized = minimax_h3_smart_resize(
        height,
        width,
        MINIMAX_H3_VISION_PATCH_SIZE * MINIMAX_H3_VISION_MERGE_SIZE,
        MINIMAX_H3_VIDEO_VISION_MIN_PIXELS,
        MINIMAX_H3_VIDEO_VISION_MAX_PIXELS,
    )
    return resized[0] == height and resized[1] == width


def minimax_h3_video_patch_grid(
    height: Int, width: Int, num_sampled_frames: Int
) raises -> List[Int]:
    """`[grid_t, grid_h, grid_w]` for a reference video's sampled frames.

    `grid_t` counts temporal blocks: sampled frames merged in pairs, the last
    one repeated for an odd count — `ceil(num_sampled / temporal_patch)`.
    Verified against the real processor's `video_grid_thw` for the trimmed
    ref2va clip ([1, 48, 84]) by this module's ref2va presentation probe."""
    if num_sampled_frames < 1:
        raise Error(
            "minimax_h3_vision_preprocess: a video needs at least one sampled"
            " frame"
        )
    if not minimax_h3_video_vision_resize_is_identity(height, width):
        var want = minimax_h3_smart_resize(
            height, width,
            MINIMAX_H3_VISION_PATCH_SIZE * MINIMAX_H3_VISION_MERGE_SIZE,
            MINIMAX_H3_VIDEO_VISION_MIN_PIXELS,
            MINIMAX_H3_VIDEO_VISION_MAX_PIXELS,
        )
        raise Error(
            String("minimax_h3_vision_preprocess: this ") + String(width) + "x"
            + String(height) + " video needs smart_resize to " + String(want[1])
            + "x" + String(want[0]) + ", and the RESAMPLER IS NOT IMPLEMENTED —"
            " unreachable from a real 768-short-edge reference canvas (see"
            " minimax_h3_vision_patch_rows for the same refusal on images)."
        )
    var tpatch = MINIMAX_H3_VISION_TEMPORAL_PATCH
    return [
        (num_sampled_frames + tpatch - 1) // tpatch,
        height // MINIMAX_H3_VISION_PATCH_SIZE,
        width // MINIMAX_H3_VISION_PATCH_SIZE,
    ]


def minimax_h3_vision_video_patch_rows(
    frames: List[UInt8],
    num_frames: Int,
    height: Int,
    width: Int,
) raises -> List[Float32]:
    """A reference video's sampled frames -> the tower's
    `[grid_t * grid_h * grid_w, 1536]` rows.

    `frames` is flat `[num_frames, height, width, 3]` uint8 RGB channels-last —
    the `MiniMaxH3RgbFrames.pixels` layout — already on the reference's own
    canvas (the prepared frames ARE; anything else is refused via the identity
    check). Layout and normalization measured bit-exact against the real video
    processor — see the VIDEO header above."""
    if num_frames < 1:
        raise Error("minimax_h3_vision_preprocess: no frames")
    if len(frames) != num_frames * height * width * 3:
        raise Error(
            String("minimax_h3_vision_preprocess: frame buffer holds ")
            + String(len(frames)) + " bytes, geometry implies "
            + String(num_frames * height * width * 3)
        )
    var grid = minimax_h3_video_patch_grid(height, width, num_frames)
    var grid_t = grid[0]
    var gh = grid[1]
    var gw = grid[2]

    var patch = MINIMAX_H3_VISION_PATCH_SIZE
    var merge = MINIMAX_H3_VISION_MERGE_SIZE
    var tpatch = MINIMAX_H3_VISION_TEMPORAL_PATCH
    if gh % merge != 0 or gw % merge != 0:
        raise Error("minimax_h3_vision_preprocess: grid must divide the merge")

    var frame_bytes = height * width * 3
    var rows = List[Float32]()
    rows.resize(grid_t * gh * gw * MINIMAX_H3_VISION_ROW_WIDTH, Float32(0.0))

    var row = 0
    for bt in range(grid_t):
        for bh in range(gh // merge):
            for bw in range(gw // merge):
                for ih in range(merge):
                    for iw in range(merge):
                        var patch_y = (bh * merge + ih) * patch
                        var patch_x = (bw * merge + iw) * patch
                        var base = row * MINIMAX_H3_VISION_ROW_WIDTH
                        for c in range(3):
                            for t in range(tpatch):
                                # Repeat-pad the LAST frame past the end.
                                var f = bt * tpatch + t
                                if f >= num_frames:
                                    f = num_frames - 1
                                for py in range(patch):
                                    for px in range(patch):
                                        var src = (
                                            f * frame_bytes
                                            + ((patch_y + py) * width
                                               + (patch_x + px)) * 3 + c
                                        )
                                        var v = Float32(Int(frames[src]))
                                        var at = (
                                            base
                                            + ((c * tpatch + t) * patch + py)
                                            * patch + px
                                        )
                                        rows[at] = (
                                            v - MINIMAX_H3_VISION_FUSED_MEAN
                                        ) / MINIMAX_H3_VISION_FUSED_STD
                        row += 1
    return rows^
