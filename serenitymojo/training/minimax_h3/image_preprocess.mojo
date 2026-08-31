# Pinned Musubi MiniMax-H3 image resize/crop preprocessing, pure Mojo.
#
# Oracle: kohya-ss/musubi-tuner b8717864713c9e4e7ef3d56eba1fc695a9b626a5
# `src/musubi_tuner/dataset/media_utils.py::resize_image_to_bucket`.
# OpenCV storage/numeric oracle: opencv-python 4.10.0.84, INTER_AREA.
#
# This is host image preprocessing over RGB8 HWC bytes. Python and OpenCV are
# fixture-generation dependencies only; neither is imported by this module.

from std.collections import List
from std.benchmark import black_box
from std.math import ceil, floor, round

from serenitymojo.pipeline.minimax_h3_keyframe_image import (
    MiniMaxH3RgbImage,
    minimax_h3_crop,
    minimax_h3_lanczos_resize,
)


@fieldwise_init
struct MiniMaxH3ImageResizePlan(Copyable, Movable):
    var scale: Float64
    var resized_width: Int
    var resized_height: Int
    var crop_left: Int
    var crop_top: Int
    var use_lanczos: Bool
    var identity: Bool


@fieldwise_init
struct _AreaAlpha(Copyable, Movable):
    var source: Int
    var destination: Int
    var alpha: Float32


@always_inline
def _area_f32(value: Float32) -> Float32:
    """Materialize OpenCV's scalar F32 multiply/add boundaries (no FMA)."""
    return black_box(value)


def minimax_h3_image_resize_plan(
    source_width: Int,
    source_height: Int,
    bucket_width: Int,
    bucket_height: Int,
) raises -> MiniMaxH3ImageResizePlan:
    """Musubi's cover geometry and positive-value `int(x * scale + 0.5)`.

    The `scale > 1` branch selection is made from the unrounded cover scale,
    exactly where Musubi makes it. A scale of one belongs to INTER_AREA.
    """
    if source_width < 1 or source_height < 1:
        raise Error("MiniMax H3 image preprocess: source dimensions must be positive")
    if bucket_width < 1 or bucket_height < 1:
        raise Error("MiniMax H3 image preprocess: bucket dimensions must be positive")
    if source_width == bucket_width and source_height == bucket_height:
        return MiniMaxH3ImageResizePlan(
            1.0, source_width, source_height, 0, 0, False, True,
        )

    var scale_width = Float64(bucket_width) / Float64(source_width)
    var scale_height = Float64(bucket_height) / Float64(source_height)
    var scale = scale_width if scale_width > scale_height else scale_height
    var resized_width = Int(Float64(source_width) * scale + 0.5)
    var resized_height = Int(Float64(source_height) * scale + 0.5)
    var crop_left = (resized_width - bucket_width) // 2
    var crop_top = (resized_height - bucket_height) // 2
    if resized_width < bucket_width or resized_height < bucket_height:
        raise Error("MiniMax H3 image preprocess: rounded cover geometry underfilled bucket")
    return MiniMaxH3ImageResizePlan(
        scale,
        resized_width,
        resized_height,
        crop_left,
        crop_top,
        scale > 1.0,
        False,
    )


def _area_coefficients(
    source_size: Int, destination_size: Int, scale: Float64,
) raises -> List[_AreaAlpha]:
    """OpenCV 4.10 `computeResizeAreaTab`, preserving F64->F32 boundaries."""
    var table = List[_AreaAlpha]()
    for destination in range(destination_size):
        var start = Float64(destination) * scale
        var end = start + scale
        var remaining = Float64(source_size) - start
        var cell_width = scale if scale < remaining else remaining
        var source_first = Int(ceil(start))
        var source_last = Int(floor(end))
        if source_last > source_size - 1:
            source_last = source_size - 1
        if source_first > source_last:
            source_first = source_last

        if Float64(source_first) - start > 1.0e-3:
            table.append(
                _AreaAlpha(
                    source_first - 1,
                    destination,
                    Float32((Float64(source_first) - start) / cell_width),
                )
            )
        for source in range(source_first, source_last):
            table.append(
                _AreaAlpha(source, destination, Float32(1.0 / cell_width))
            )
        if end - Float64(source_last) > 1.0e-3:
            var fraction = end - Float64(source_last)
            if fraction > 1.0:
                fraction = 1.0
            if fraction > cell_width:
                fraction = cell_width
            table.append(
                _AreaAlpha(
                    source_last,
                    destination,
                    Float32(fraction / cell_width),
                )
            )
    return table^


def _area_u8(value: Float32) -> UInt8:
    """OpenCV `saturate_cast<uchar>`: F32 round-nearest-even then clamp."""
    var rounded = Int(round(value))
    if rounded < 0:
        return UInt8(0)
    if rounded > 255:
        return UInt8(255)
    return UInt8(rounded)


def _inter_area_integer(
    image: MiniMaxH3RgbImage,
    out_height: Int,
    out_width: Int,
    scale_y: Int,
    scale_x: Int,
) -> MiniMaxH3RgbImage:
    """OpenCV `resizeAreaFast_` for exact integer decimation."""
    var out = List[UInt8]()
    out.resize(out_height * out_width * 3, UInt8(0))
    var area = scale_x * scale_y
    var factor = Float32(1.0) / Float32(area)
    for destination_y in range(out_height):
        for destination_x in range(out_width):
            for channel in range(3):
                var total = 0
                for offset_y in range(scale_y):
                    var source_y = destination_y * scale_y + offset_y
                    for offset_x in range(scale_x):
                        var source_x = destination_x * scale_x + offset_x
                        total += Int(
                            image.pixels[
                                (source_y * image.width + source_x) * 3 + channel
                            ]
                        )
                var destination = (
                    destination_y * out_width + destination_x
                ) * 3 + channel
                # OpenCV has a specialized exact RGB8 2x2 path.
                if scale_x == 2 and scale_y == 2:
                    out[destination] = UInt8((total + 2) >> 2)
                else:
                    out[destination] = _area_u8(Float32(total) * factor)
    return MiniMaxH3RgbImage(out^, out_height, out_width)


def minimax_h3_inter_area_resize(
    image: MiniMaxH3RgbImage, out_height: Int, out_width: Int,
) raises -> MiniMaxH3RgbImage:
    """RGB8 `cv2.resize(..., interpolation=cv2.INTER_AREA)` for downscale.

    Integer factors take OpenCV's fast area path. Non-integer factors reproduce
    its F64 geometry, F32 coefficient tables, horizontal F32 accumulation,
    vertical F32 accumulation, and final round-nearest-even U8 saturation.
    """
    image.validate()
    if out_height < 1 or out_width < 1:
        raise Error("MiniMax H3 INTER_AREA: output dimensions must be positive")
    if out_height > image.height or out_width > image.width:
        raise Error("MiniMax H3 INTER_AREA: upscale is unsupported")
    if out_height == image.height and out_width == image.width:
        return MiniMaxH3RgbImage(image.pixels.copy(), image.height, image.width)

    if image.width % out_width == 0 and image.height % out_height == 0:
        return _inter_area_integer(
            image,
            out_height,
            out_width,
            image.height // out_height,
            image.width // out_width,
        )

    var scale_x = Float64(image.width) / Float64(out_width)
    var scale_y = Float64(image.height) / Float64(out_height)
    var x_table = _area_coefficients(image.width, out_width, scale_x)
    var y_table = _area_coefficients(image.height, out_height, scale_y)
    var out = List[UInt8]()
    out.resize(out_height * out_width * 3, UInt8(0))
    var horizontal = List[Float32]()
    horizontal.resize(out_width * 3, Float32(0.0))
    var accumulated = List[Float32]()
    accumulated.resize(out_width * 3, Float32(0.0))
    var y_index = 0

    for destination_y in range(out_height):
        for value in range(out_width * 3):
            accumulated[value] = 0.0
        while (
            y_index < len(y_table)
            and y_table[y_index].destination == destination_y
        ):
            for value in range(out_width * 3):
                horizontal[value] = 0.0
            var source_y = y_table[y_index].source
            for x_item in x_table:
                var source = (source_y * image.width + x_item.source) * 3
                var destination = x_item.destination * 3
                for channel in range(3):
                    horizontal[destination + channel] = _area_f32(
                        horizontal[destination + channel]
                        + _area_f32(
                            Float32(image.pixels[source + channel]) * x_item.alpha
                        )
                    )
            var beta = y_table[y_index].alpha
            for value in range(out_width * 3):
                accumulated[value] = _area_f32(
                    accumulated[value]
                    + _area_f32(beta * horizontal[value])
                )
            y_index += 1

        for destination_x in range(out_width):
            for channel in range(3):
                var offset = destination_x * 3 + channel
                out[(destination_y * out_width + destination_x) * 3 + channel] = (
                    _area_u8(accumulated[offset])
                )
    if y_index != len(y_table):
        raise Error("MiniMax H3 INTER_AREA: vertical coefficient table was not consumed")
    return MiniMaxH3RgbImage(out^, out_height, out_width)


def minimax_h3_resize_image_to_bucket(
    image: MiniMaxH3RgbImage, bucket_width: Int, bucket_height: Int,
) raises -> MiniMaxH3RgbImage:
    """Pinned Musubi `resize_image_to_bucket` for an RGB8 image."""
    image.validate()
    var plan = minimax_h3_image_resize_plan(
        image.width, image.height, bucket_width, bucket_height,
    )
    if plan.identity:
        return MiniMaxH3RgbImage(image.pixels.copy(), image.height, image.width)

    var resized: MiniMaxH3RgbImage
    if plan.use_lanczos:
        resized = minimax_h3_lanczos_resize(
            image, plan.resized_height, plan.resized_width,
        )
    else:
        resized = minimax_h3_inter_area_resize(
            image, plan.resized_height, plan.resized_width,
        )
    return minimax_h3_crop(
        resized,
        plan.crop_left,
        plan.crop_top,
        bucket_height,
        bucket_width,
    )
