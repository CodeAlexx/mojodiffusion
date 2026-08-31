# Pinned Musubi MiniMax-H3 image bucket geometry (host-only).

from std.collections import List
from std.math import abs, sqrt


comptime MINIMAX_H3_BUCKET_STEP = 32


@fieldwise_init
struct MiniMaxH3BucketResolution(Copyable, Movable, ImplicitlyCopyable):
    var width: Int
    var height: Int


def _aligned_down(value: Int) -> Int:
    return value - value % MINIMAX_H3_BUCKET_STEP


def _append_unique(
    mut values: List[MiniMaxH3BucketResolution], width: Int, height: Int,
):
    for value in values:
        if value.width == width and value.height == height:
            return
    values.append(MiniMaxH3BucketResolution(width, height))


def _sort_resolutions(mut values: List[MiniMaxH3BucketResolution]):
    for index in range(1, len(values)):
        var key = values[index]
        var cursor = index - 1
        while cursor >= 0 and (
            values[cursor].width > key.width
            or (
                values[cursor].width == key.width
                and values[cursor].height > key.height
            )
        ):
            values[cursor + 1] = values[cursor]
            cursor -= 1
        values[cursor + 1] = key


def minimax_h3_bucket_resolutions(
    resolution_width: Int, resolution_height: Int,
) raises -> List[MiniMaxH3BucketResolution]:
    if resolution_width <= 0 or resolution_height <= 0:
        raise Error("MiniMax H3 bucket resolution must be positive")
    var area = resolution_width * resolution_height
    var square = Int(sqrt(Float64(area)))
    var minimum = _aligned_down(square // 2)
    var values = List[MiniMaxH3BucketResolution]()
    for width in range(minimum, square + MINIMAX_H3_BUCKET_STEP, MINIMAX_H3_BUCKET_STEP):
        var height = _aligned_down(area // width)
        _append_unique(values, width, height)
        _append_unique(values, height, width)
    _sort_resolutions(values)
    return values^


def minimax_h3_select_bucket(
    source_width: Int,
    source_height: Int,
    resolution_width: Int = 1024,
    resolution_height: Int = 1024,
    enable_bucket: Bool = True,
    no_upscale: Bool = False,
) raises -> MiniMaxH3BucketResolution:
    """Match pinned Musubi's sorted-list/NumPy-first-argmin selection."""
    if source_width <= 0 or source_height <= 0:
        raise Error("MiniMax H3 source dimensions must be positive")
    if not enable_bucket:
        return MiniMaxH3BucketResolution(resolution_width, resolution_height)
    if no_upscale and source_width * source_height <= resolution_width * resolution_height:
        return MiniMaxH3BucketResolution(
            _aligned_down(source_width), _aligned_down(source_height)
        )
    var values = minimax_h3_bucket_resolutions(
        resolution_width, resolution_height
    )
    var source_ratio = Float64(source_width) / Float64(source_height)
    var best_index = 0
    var best_error = abs(
        Float64(values[0].width) / Float64(values[0].height) - source_ratio
    )
    for index in range(1, len(values)):
        var error = abs(
            Float64(values[index].width) / Float64(values[index].height)
            - source_ratio
        )
        if error < best_error:
            best_index = index
            best_error = error
    return values[best_index]
