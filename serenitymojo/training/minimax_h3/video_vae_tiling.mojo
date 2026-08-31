# MiniMax-H3 one-frame VideoVAE encode tiling seam, pure Mojo/MAX.
#
# Oracle: kohya-ss/musubi-tuner b8717864713c9e4e7ef3d56eba1fc695a9b626a5
# src/musubi_tuner/minimax_h3/video_vae.py
# sha256 96e6698e5072adc258b6610881749d3748173d78c01c9b833e4cc42253165671
# MiniMaxH3VideoVAE._split_tiles:450-463, _blend:465-482,
# _stitch_tiles:484-504, _encode_clip:506-528, encode_moments:607-616.
#
# This seam is intentionally narrow: F32, batch-one, one-frame encoder input
# and F32 moments only.  It does not sample the posterior or write a cache.

from std.collections import List
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    minimax_h3_video_encode_device,
)
from serenitymojo.ops.tensor_algebra import concat, slice
from serenitymojo.pipeline.minimax_h3_video_vae_blend import (
    minimax_h3_video_blend,
)
from serenitymojo.tensor import Tensor


comptime MINIMAX_H3_VIDEO_VAE_TILE_SIZE = 256
comptime MINIMAX_H3_VIDEO_VAE_TILE_OVERLAP_MIN = 64
comptime MINIMAX_H3_VIDEO_VAE_SPACE_RATIO = 16
comptime MINIMAX_H3_VIDEO_VAE_MOMENT_CHANNELS = 48
comptime TArc = ArcPointer[Tensor]


@fieldwise_init
struct MiniMaxH3OneFrameTileAxis(Movable):
    """Exact pinned pixel-axis plan plus its latent-space overlaps."""

    var input_length: Int
    var starts: List[Int]
    var lengths: List[Int]
    var pixel_overlaps: List[Int]
    var latent_overlaps: List[Int]

    def num_tiles(self) -> Int:
        return len(self.starts)


def _require_axis_plan(plan: MiniMaxH3OneFrameTileAxis) raises:
    var count = plan.num_tiles()
    if count < 1 or len(plan.lengths) != count:
        raise Error("MiniMax H3 VAE tiling: malformed axis tile inventory")
    if (
        len(plan.pixel_overlaps) != count - 1
        or len(plan.latent_overlaps) != count - 1
    ):
        raise Error("MiniMax H3 VAE tiling: malformed axis overlap inventory")
    if plan.starts[0] != 0:
        raise Error("MiniMax H3 VAE tiling: first tile must start at zero")
    for index in range(count):
        if (
            plan.starts[index] < 0
            or plan.starts[index] % MINIMAX_H3_VIDEO_VAE_SPACE_RATIO != 0
            or plan.lengths[index] < MINIMAX_H3_VIDEO_VAE_SPACE_RATIO
            or plan.lengths[index] % MINIMAX_H3_VIDEO_VAE_SPACE_RATIO != 0
        ):
            raise Error("MiniMax H3 VAE tiling: unaligned tile geometry")
        if index > 0 and plan.starts[index] <= plan.starts[index - 1]:
            raise Error("MiniMax H3 VAE tiling: tile starts must increase")
    if plan.starts[count - 1] + plan.lengths[count - 1] != plan.input_length:
        raise Error("MiniMax H3 VAE tiling: tiles do not end at input boundary")
    for index in range(count - 1):
        var overlap = plan.pixel_overlaps[index]
        if (
            overlap < MINIMAX_H3_VIDEO_VAE_TILE_OVERLAP_MIN
            or overlap >= plan.lengths[index]
            or overlap % MINIMAX_H3_VIDEO_VAE_SPACE_RATIO != 0
            or plan.latent_overlaps[index]
                != overlap // MINIMAX_H3_VIDEO_VAE_SPACE_RATIO
            or plan.starts[index + 1]
                != plan.starts[index] + plan.lengths[index] - overlap
        ):
            raise Error("MiniMax H3 VAE tiling: invalid overlap conversion")


def minimax_h3_one_frame_tile_axis(
    input_length: Int,
) raises -> MiniMaxH3OneFrameTileAxis:
    """Pinned Musubi `_split_tiles` with strict encoder-alignment guards.

    Musubi distributes unused coverage in 16-pixel units round-robin across
    overlap slots.  This is not a fixed 192-pixel stride: for example, 272
    pixels becomes starts [0,16] with a 240-pixel overlap.
    """
    if (
        input_length < MINIMAX_H3_VIDEO_VAE_SPACE_RATIO
        or input_length % MINIMAX_H3_VIDEO_VAE_SPACE_RATIO != 0
    ):
        raise Error(
            "MiniMax H3 VAE tiling: axis length must be a positive multiple of 16"
        )

    var starts = List[Int]()
    var lengths = List[Int]()
    var overlaps = List[Int]()
    var latent_overlaps = List[Int]()
    if MINIMAX_H3_VIDEO_VAE_TILE_SIZE >= input_length:
        starts.append(0)
        lengths.append(input_length)
        var single = MiniMaxH3OneFrameTileAxis(
            input_length, starts^, lengths^, overlaps^, latent_overlaps^,
        )
        _require_axis_plan(single)
        return single^

    var count = (
        input_length + MINIMAX_H3_VIDEO_VAE_TILE_SIZE - 1
    ) // MINIMAX_H3_VIDEO_VAE_TILE_SIZE
    while (
        MINIMAX_H3_VIDEO_VAE_TILE_SIZE * count
        - MINIMAX_H3_VIDEO_VAE_TILE_OVERLAP_MIN * (count - 1)
        < input_length
    ):
        count += 1
    for _ in range(count - 1):
        overlaps.append(MINIMAX_H3_VIDEO_VAE_TILE_OVERLAP_MIN)
    var overlap_sum = 0
    for index in range(len(overlaps)):
        overlap_sum += overlaps[index]
    var remaining = (
        MINIMAX_H3_VIDEO_VAE_TILE_SIZE * count
        - overlap_sum
        - input_length
    )
    if remaining < 0 or remaining % MINIMAX_H3_VIDEO_VAE_SPACE_RATIO != 0:
        raise Error("MiniMax H3 VAE tiling: non-integral overlap redistribution")
    for index in range(remaining // MINIMAX_H3_VIDEO_VAE_SPACE_RATIO):
        overlaps[index % (count - 1)] += MINIMAX_H3_VIDEO_VAE_SPACE_RATIO

    starts.append(0)
    for index in range(count - 1):
        starts.append(
            starts[len(starts) - 1]
            + MINIMAX_H3_VIDEO_VAE_TILE_SIZE
            - overlaps[index]
        )
        latent_overlaps.append(
            overlaps[index] // MINIMAX_H3_VIDEO_VAE_SPACE_RATIO
        )
    for _ in range(count):
        lengths.append(MINIMAX_H3_VIDEO_VAE_TILE_SIZE)

    var plan = MiniMaxH3OneFrameTileAxis(
        input_length, starts^, lengths^, overlaps^, latent_overlaps^,
    )
    _require_axis_plan(plan)
    return plan^


def minimax_h3_stitch_one_frame_moment_tiles_f32(
    raw_tiles: List[TArc],
    height_plan: MiniMaxH3OneFrameTileAxis,
    width_plan: MiniMaxH3OneFrameTileAxis,
    channels: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Pinned row-major `_stitch_tiles` for raw NDHWC encoded tiles.

    Vertical blending uses the raw tile above.  Horizontal blending then uses
    the raw tile to the left, exactly as upstream; neither neighbor is a
    previously cropped stitched result.  Trailing overlaps are trimmed only
    after both blends, rows concatenate along W, then along H.
    """
    _require_axis_plan(height_plan)
    _require_axis_plan(width_plan)
    if channels < 1:
        raise Error("MiniMax H3 VAE tiling: moment channels must be positive")
    var rows = height_plan.num_tiles()
    var columns = width_plan.num_tiles()
    if len(raw_tiles) != rows * columns:
        raise Error("MiniMax H3 VAE tiling: raw tile count mismatch")

    for row in range(rows):
        for column in range(columns):
            var tile_index = row * columns + column
            var shape = raw_tiles[tile_index][].shape()
            if raw_tiles[tile_index][].dtype() != STDtype.F32:
                raise Error("MiniMax H3 VAE tiling: raw moments must be F32")
            if (
                len(shape) != 5 or shape[0] != 1 or shape[1] != 1
                or shape[2]
                    != height_plan.lengths[row] // MINIMAX_H3_VIDEO_VAE_SPACE_RATIO
                or shape[3]
                    != width_plan.lengths[column] // MINIMAX_H3_VIDEO_VAE_SPACE_RATIO
                or shape[4] != channels
            ):
                raise Error(
                    "MiniMax H3 VAE tiling: expected raw tile [1,1,H/16,W/16,C]"
                )

    var stitched_rows = List[TArc]()
    for row in range(rows):
        var stitched_row = List[TArc]()
        for column in range(columns):
            var tile = raw_tiles[row * columns + column][].clone(ctx)
            if row > 0:
                tile = minimax_h3_video_blend(
                    raw_tiles[(row - 1) * columns + column][],
                    tile,
                    height_plan.latent_overlaps[row - 1],
                    2,
                    ctx,
                )
            if column > 0:
                tile = minimax_h3_video_blend(
                    raw_tiles[row * columns + column - 1][],
                    tile,
                    width_plan.latent_overlaps[column - 1],
                    3,
                    ctx,
                )
            if row < rows - 1:
                tile = slice(
                    tile,
                    2,
                    0,
                    tile.shape()[2] - height_plan.latent_overlaps[row],
                    ctx,
                )
            if column < columns - 1:
                tile = slice(
                    tile,
                    3,
                    0,
                    tile.shape()[3] - width_plan.latent_overlaps[column],
                    ctx,
                )
            stitched_row.append(TArc(tile^))
        var row_tensor = stitched_row[0][].clone(ctx)
        for column in range(1, columns):
            row_tensor = concat(3, ctx, row_tensor, stitched_row[column][])
        stitched_rows.append(TArc(row_tensor^))

    var output = stitched_rows[0][].clone(ctx)
    for row in range(1, rows):
        output = concat(2, ctx, output, stitched_rows[row][])
    var output_shape = output.shape()
    var expected_shape: List[Int] = [
        1,
        1,
        height_plan.input_length // MINIMAX_H3_VIDEO_VAE_SPACE_RATIO,
        width_plan.input_length // MINIMAX_H3_VIDEO_VAE_SPACE_RATIO,
        channels,
    ]
    if output.dtype() != STDtype.F32 or output_shape != expected_shape:
        raise Error("MiniMax H3 VAE tiling: stitched moment shape mismatch")
    return output^


def minimax_h3_encode_one_frame_moments_tiled_f32(
    encoder: MiniMaxH3VideoEncoderDevice,
    pixels: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Thin F32 VideoVAE integration seam for `[1,1,H,W,3]` NDHWC.

    The caller must use `MiniMaxH3VideoEncoderDevice.load_f32_compute`; this
    function verifies that boundary before launching any tile.  It returns
    raw F32 moments `[1,1,H/16,W/16,48]`; posterior sampling remains separate.
    """
    var shape = pixels.shape()
    if pixels.dtype() != STDtype.F32:
        raise Error("MiniMax H3 VAE tiling: encoder pixels must be F32")
    if (
        len(shape) != 5 or shape[0] != 1 or shape[1] != 1
        or shape[2] < MINIMAX_H3_VIDEO_VAE_SPACE_RATIO
        or shape[3] < MINIMAX_H3_VIDEO_VAE_SPACE_RATIO
        or shape[4] != 3
    ):
        raise Error("MiniMax H3 VAE tiling: expected pixels [1,1,H,W,3]")
    for index in range(len(encoder.weights)):
        if encoder.weights[index][].dtype() != STDtype.F32:
            raise Error(
                "MiniMax H3 VAE tiling: encoder must use the F32-compute loader"
            )

    var height_plan = minimax_h3_one_frame_tile_axis(shape[2])
    var width_plan = minimax_h3_one_frame_tile_axis(shape[3])
    var raw_tiles = List[TArc]()
    for row in range(height_plan.num_tiles()):
        var pixel_row = slice(
            pixels,
            2,
            height_plan.starts[row],
            height_plan.lengths[row],
            ctx,
        )
        for column in range(width_plan.num_tiles()):
            var pixel_tile = slice(
                pixel_row,
                3,
                width_plan.starts[column],
                width_plan.lengths[column],
                ctx,
            )
            var moments = minimax_h3_video_encode_device(
                encoder, pixel_tile, ctx,
            )
            raw_tiles.append(TArc(moments^))
    return minimax_h3_stitch_one_frame_moment_tiles_f32(
        raw_tiles,
        height_plan,
        width_plan,
        MINIMAX_H3_VIDEO_VAE_MOMENT_CHANNELS,
        ctx,
    )
