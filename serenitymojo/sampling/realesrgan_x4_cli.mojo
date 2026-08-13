"""Pure-Mojo Real-ESRGAN x4 media CLI.

Modes:
  realesrgan_x4_cli image <input> <output.png> <weights.safetensors>
  realesrgan_x4_cli frames <input-prefix> <output-prefix> <count> <weights>
  realesrgan_x4_cli image-fast <input> <output.png> <weights.safetensors>
  realesrgan_x4_cli frames-fast <input-prefix> <output-prefix> <count> <weights>

Frame mode uses six-digit PNG names and keeps the RRDBNet weights resident
across the whole sequence. It is the post-generation media primitive used by
the Serenity Rust control plane; model execution remains Mojo + MAX.
"""

from max.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.image.png import ValueRange, save_png
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.realesrgan.rrdbnet import (
    RRDBNetWeights,
    load_rrdbnet,
    rrdbnet_forward,
)
from serenitymojo.models.realesrgan.srvggnet import (
    SRVGGNetWeights,
    load_srvggnet,
    srvggnet_forward,
)
from serenitymojo.serve.image_io import (
    decode_image_any,
    image_to_signed_nchw,
)
from serenitymojo.tensor import Tensor

comptime TILE = 128
comptime PAD = 16
comptime CORE = TILE - 2 * PAD
comptime SCALE = 4
comptime OUTPUT_TILE = TILE * SCALE
comptime OUTPUT_PAD = PAD * SCALE
comptime OUTPUT_CORE = CORE * SCALE


def _parse_positive_int(value: String, label: String) raises -> Int:
    if value.byte_length() == 0:
        raise Error(String("empty ") + label)
    var out = 0
    for byte in value.as_bytes():
        if byte < UInt8(ord("0")) or byte > UInt8(ord("9")):
            raise Error(String("invalid ") + label + String(": ") + value)
        out = out * 10 + Int(byte - UInt8(ord("0")))
    if out <= 0:
        raise Error(label + String(" must be positive"))
    return out


def _six_digits(index: Int) -> String:
    var value = String(index)
    while value.byte_length() < 6:
        value = String("0") + value
    return value^


def _frame_path(prefix: String, index: Int) -> String:
    return prefix + _six_digits(index) + String(".png")


def _clamp_index(value: Int, low: Int, high: Int) -> Int:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _clamp_unit(value: Float32) -> Float32:
    if value < Float32(0.0):
        return Float32(0.0)
    if value > Float32(1.0):
        return Float32(1.0)
    return value


def _upscale_image_rrdb(
    input_path: String,
    output_path: String,
    weights: RRDBNetWeights,
    ctx: DeviceContext,
) raises:
    var image = decode_image_any(input_path)
    var height = image.height
    var width = image.width
    if height <= 0 or width <= 0:
        raise Error(String("empty input image: ") + input_path)
    var plane = height * width
    var signed = image_to_signed_nchw(image)
    var source = List[Float32](capacity=3 * plane)
    for i in range(3 * plane):
        source.append(
            (signed[i] + Float32(1.0)) * Float32(0.5)
        )

    var output_width = width * SCALE
    var output_height = height * SCALE
    var output_plane = output_height * output_width
    var destination = List[Float32](capacity=3 * output_plane)
    destination.resize(3 * output_plane, Float32(0.0))

    var rows = (height + CORE - 1) // CORE
    var columns = (width + CORE - 1) // CORE
    var tile = List[Float32](capacity=TILE * TILE * 3)
    tile.resize(TILE * TILE * 3, Float32(0.0))
    for row in range(rows):
        var source_y0 = row * CORE
        for column in range(columns):
            var source_x0 = column * CORE
            for y in range(TILE):
                var source_y = _clamp_index(
                    source_y0 - PAD + y, 0, height - 1
                )
                for x in range(TILE):
                    var source_x = _clamp_index(
                        source_x0 - PAD + x, 0, width - 1
                    )
                    var tile_offset = (y * TILE + x) * 3
                    tile[tile_offset] = source[
                        source_y * width + source_x
                    ]
                    tile[tile_offset + 1] = source[
                        plane + source_y * width + source_x
                    ]
                    tile[tile_offset + 2] = source[
                        2 * plane + source_y * width + source_x
                    ]
            var tile_tensor = Tensor.from_host(
                tile, [1, TILE, TILE, 3], STDtype.F32, ctx
            )
            var output_tile = rrdbnet_forward(
                weights, tile_tensor, ctx
            ).to_host(ctx)
            for y in range(OUTPUT_CORE):
                var destination_y = source_y0 * SCALE + y
                if destination_y >= output_height:
                    break
                for x in range(OUTPUT_CORE):
                    var destination_x = source_x0 * SCALE + x
                    if destination_x >= output_width:
                        break
                    var tile_offset = (
                        ((y + OUTPUT_PAD) * OUTPUT_TILE + x + OUTPUT_PAD)
                        * 3
                    )
                    var destination_offset = (
                        destination_y * output_width + destination_x
                    )
                    destination[destination_offset] = _clamp_unit(
                        output_tile[tile_offset]
                    )
                    destination[output_plane + destination_offset] = (
                        _clamp_unit(output_tile[tile_offset + 1])
                    )
                    destination[2 * output_plane + destination_offset] = (
                        _clamp_unit(output_tile[tile_offset + 2])
                    )
        print(
            "[realesrgan] tile row", row + 1, "/", rows,
            "for", input_path,
        )

    var output = Tensor.from_host(
        destination,
        [1, 3, output_height, output_width],
        STDtype.F32,
        ctx,
    )
    save_png(output, output_path, ctx, ValueRange.UNIT)
    print(
        "[realesrgan] wrote", output_path,
        "(", output_width, "x", output_height, ")",
    )


def _upscale_image_srvgg(
    input_path: String,
    output_path: String,
    weights: SRVGGNetWeights,
    ctx: DeviceContext,
) raises:
    var image = decode_image_any(input_path)
    var height = image.height
    var width = image.width
    if height <= 0 or width <= 0:
        raise Error(String("empty input image: ") + input_path)
    var plane = height * width
    var signed = image_to_signed_nchw(image)
    var source = List[Float32](capacity=3 * plane)
    for i in range(3 * plane):
        source.append(
            (signed[i] + Float32(1.0)) * Float32(0.5)
        )

    var output_width = width * SCALE
    var output_height = height * SCALE
    var output_plane = output_height * output_width
    var destination = List[Float32](capacity=3 * output_plane)
    destination.resize(3 * output_plane, Float32(0.0))

    var rows = (height + CORE - 1) // CORE
    var columns = (width + CORE - 1) // CORE
    var tile = List[Float32](capacity=TILE * TILE * 3)
    tile.resize(TILE * TILE * 3, Float32(0.0))
    for row in range(rows):
        var source_y0 = row * CORE
        for column in range(columns):
            var source_x0 = column * CORE
            for y in range(TILE):
                var source_y = _clamp_index(
                    source_y0 - PAD + y, 0, height - 1
                )
                for x in range(TILE):
                    var source_x = _clamp_index(
                        source_x0 - PAD + x, 0, width - 1
                    )
                    var tile_offset = (y * TILE + x) * 3
                    tile[tile_offset] = source[
                        source_y * width + source_x
                    ]
                    tile[tile_offset + 1] = source[
                        plane + source_y * width + source_x
                    ]
                    tile[tile_offset + 2] = source[
                        2 * plane + source_y * width + source_x
                    ]
            var tile_tensor = Tensor.from_host(
                tile, [1, TILE, TILE, 3], STDtype.F32, ctx
            )
            var output_tile = srvggnet_forward(
                weights, tile_tensor, ctx
            ).to_host(ctx)
            for y in range(OUTPUT_CORE):
                var destination_y = source_y0 * SCALE + y
                if destination_y >= output_height:
                    break
                for x in range(OUTPUT_CORE):
                    var destination_x = source_x0 * SCALE + x
                    if destination_x >= output_width:
                        break
                    var tile_offset = (
                        ((y + OUTPUT_PAD) * OUTPUT_TILE + x + OUTPUT_PAD)
                        * 3
                    )
                    var destination_offset = (
                        destination_y * output_width + destination_x
                    )
                    destination[destination_offset] = _clamp_unit(
                        output_tile[tile_offset]
                    )
                    destination[output_plane + destination_offset] = (
                        _clamp_unit(output_tile[tile_offset + 1])
                    )
                    destination[2 * output_plane + destination_offset] = (
                        _clamp_unit(output_tile[tile_offset + 2])
                    )
        print(
            "[realesrgan-fast] tile row", row + 1, "/", rows,
            "for", input_path,
        )

    var output = Tensor.from_host(
        destination,
        [1, 3, output_height, output_width],
        STDtype.F32,
        ctx,
    )
    save_png(output, output_path, ctx, ValueRange.UNIT)
    print(
        "[realesrgan-fast] wrote", output_path,
        "(", output_width, "x", output_height, ")",
    )


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error(
            "usage: realesrgan_x4_cli image <input> <output> <weights> | "
            "frames <input-prefix> <output-prefix> <count> <weights>"
        )
    var mode = String(args[1])
    var ctx = DeviceContext()
    if mode == "image":
        if len(args) != 5:
            raise Error(
                "usage: realesrgan_x4_cli image <input> <output> <weights>"
            )
        var weights = load_rrdbnet(String(args[4]), ctx)
        _upscale_image_rrdb(
            String(args[2]), String(args[3]), weights, ctx
        )
        return
    if mode == "frames":
        if len(args) != 6:
            raise Error(
                "usage: realesrgan_x4_cli frames <input-prefix> "
                "<output-prefix> <count> <weights>"
            )
        var count = _parse_positive_int(String(args[4]), String("count"))
        var weights = load_rrdbnet(String(args[5]), ctx)
        for index in range(count):
            print("[realesrgan] frame", index + 1, "/", count)
            _upscale_image_rrdb(
                _frame_path(String(args[2]), index),
                _frame_path(String(args[3]), index),
                weights,
                ctx,
            )
        return
    if mode == "image-fast":
        if len(args) != 5:
            raise Error(
                "usage: realesrgan_x4_cli image-fast <input> <output> <weights>"
            )
        var weights = load_srvggnet(String(args[4]), ctx)
        _upscale_image_srvgg(
            String(args[2]), String(args[3]), weights, ctx
        )
        return
    if mode == "frames-fast":
        if len(args) != 6:
            raise Error(
                "usage: realesrgan_x4_cli frames-fast <input-prefix> "
                "<output-prefix> <count> <weights>"
            )
        var count = _parse_positive_int(String(args[4]), String("count"))
        var weights = load_srvggnet(String(args[5]), ctx)
        for index in range(count):
            print("[realesrgan-fast] frame", index + 1, "/", count)
            _upscale_image_srvgg(
                _frame_path(String(args[2]), index),
                _frame_path(String(args[3]), index),
                weights,
                ctx,
            )
        return
    raise Error(String("unknown Real-ESRGAN mode: ") + mode)
