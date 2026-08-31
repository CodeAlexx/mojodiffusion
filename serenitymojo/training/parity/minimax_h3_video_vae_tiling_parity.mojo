# Synthetic device parity for pinned MiniMax-H3 one-frame VideoVAE tiling.
# Python executes the exact upstream method bodies to generate/check fixture.

from json.parser import loads
from json.value import JSONValue
from std.collections import List
from std.math import abs, isfinite, sqrt
from std.memory import ArcPointer
from std.pathlib import Path
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.minimax_h3.video_vae_tiling import (
    MiniMaxH3OneFrameTileAxis,
    minimax_h3_one_frame_tile_axis,
    minimax_h3_stitch_one_frame_moment_tiles_f32,
)


comptime FIXTURE = (
    "serenitymojo/training/parity/fixtures/"
    "minimax_h3_video_vae_tiling_v1.json"
)
comptime SCHEMA = "serenity.minimax_h3.video_vae_one_frame_tiling.v1"
comptime ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime SOURCE = "src/musubi_tuner/minimax_h3/video_vae.py"
comptime SOURCE_SHA256 = "96e6698e5072adc258b6610881749d3748173d78c01c9b833e4cc42253165671"
comptime TArc = ArcPointer[Tensor]


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax H3 VideoVAE tiling parity failed: ") + message)


def _int_list(value: JSONValue) raises -> List[Int]:
    var output = List[Int](capacity=value.length())
    for index in range(value.length()):
        output.append(value[index].as_int())
    return output^


def _f32_list(value: JSONValue) raises -> List[Float32]:
    var output = List[Float32](capacity=value.length())
    for index in range(value.length()):
        output.append(Float32(value[index].as_float()))
    return output^


def _same_ints(left: List[Int], right: List[Int]) -> Bool:
    if len(left) != len(right):
        return False
    for index in range(len(left)):
        if left[index] != right[index]:
            return False
    return True


def _check_axis(plan: MiniMaxH3OneFrameTileAxis, expected: JSONValue) raises:
    _check(
        plan.input_length == expected[String("length")].as_int(),
        "axis input length",
    )
    _check(
        _same_ints(plan.starts, _int_list(expected[String("starts")])),
        "axis starts",
    )
    _check(
        _same_ints(plan.lengths, _int_list(expected[String("lengths")])),
        "axis lengths",
    )
    _check(
        _same_ints(
            plan.pixel_overlaps,
            _int_list(expected[String("pixel_overlaps")]),
        ),
        "axis pixel overlaps",
    )
    _check(
        _same_ints(
            plan.latent_overlaps,
            _int_list(expected[String("latent_overlaps")]),
        ),
        "axis latent overlaps",
    )


def _tile_values(
    case_index: Int,
    row: Int,
    column: Int,
    height: Int,
    width: Int,
    channels: Int,
) -> List[Float32]:
    # Fixture oracle uses NCTHW, then converts its output to NDHWC.  Populate
    # the same logical values directly in Serenity's NDHWC tile layout.
    var output = List[Float32](capacity=height * width * channels)
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                var value = (
                    Float64(case_index) * 17.0
                    + Float64(row) * 5.0
                    - Float64(column) * 3.0
                    + Float64(channel) * 0.375
                    + Float64(y) * 0.03125
                    - Float64(x) * 0.0078125
                    + Float64(
                        (y * 11 + x * 7 + row * 3 + column) % 9
                    ) * 0.001953125
                )
                output.append(Float32(value))
    return output^


def _run_stitch_case(
    item: JSONValue, ctx: DeviceContext,
) raises -> Float32:
    var case_index = item[String("case_index")].as_int()
    var channels = item[String("channels")].as_int()
    var height_plan = minimax_h3_one_frame_tile_axis(
        item[String("height")].as_int()
    )
    var width_plan = minimax_h3_one_frame_tile_axis(
        item[String("width")].as_int()
    )
    _check_axis(height_plan, item[String("height_axis")])
    _check_axis(width_plan, item[String("width_axis")])

    var raw_tiles = List[TArc]()
    for row in range(height_plan.num_tiles()):
        var tile_height = height_plan.lengths[row] // 16
        for column in range(width_plan.num_tiles()):
            var tile_width = width_plan.lengths[column] // 16
            var values = _tile_values(
                case_index, row, column, tile_height, tile_width, channels,
            )
            var shape: List[Int] = [1, 1, tile_height, tile_width, channels]
            raw_tiles.append(
                TArc(Tensor.from_host(values, shape^, STDtype.F32, ctx))
            )
    var got = minimax_h3_stitch_one_frame_moment_tiles_f32(
        raw_tiles, height_plan, width_plan, channels, ctx,
    )
    var expected_shape = _int_list(item[String("expected_shape_ndhwc")])
    _check(_same_ints(got.shape(), expected_shape), "stitched output shape")
    var expected = _f32_list(item[String("expected_ndhwc_f32")])
    var actual = got.to_host(ctx)
    _check(len(actual) == len(expected), "stitched output length")

    var max_abs = Float32(0.0)
    var squared_error = Float64(0.0)
    var squared_reference = Float64(0.0)
    for index in range(len(actual)):
        _check(isfinite(actual[index]), "stitched output finite")
        var difference = abs(actual[index] - expected[index])
        if difference > max_abs:
            max_abs = difference
        squared_error += Float64(difference) * Float64(difference)
        squared_reference += Float64(expected[index]) * Float64(expected[index])
    var relative_l2 = Float64(0.0)
    if squared_reference > 0.0:
        relative_l2 = sqrt(squared_error / squared_reference)
    _check(max_abs <= 0.000008, "stitched max-abs tolerance")
    _check(relative_l2 <= 0.00000025, "stitched relative-L2 tolerance")
    print(
        "  ", item[String("name")].as_string(),
        "max_abs=", max_abs, "rel_l2=", relative_l2,
    )
    return max_abs


def _check_fixture_contract(document: JSONValue) raises:
    _check(document[String("schema")].as_string() == String(SCHEMA), "schema")
    _check(
        document[String("oracle_commit")].as_string() == String(ORACLE_COMMIT),
        "oracle commit",
    )
    _check(document[String("source")].as_string() == String(SOURCE), "source")
    _check(
        document[String("source_sha256")].as_string() == String(SOURCE_SHA256),
        "source SHA-256",
    )
    var spans = document[String("source_spans")]
    _check(
        spans[String("_split_tiles")][0].as_int() == 450
        and spans[String("_split_tiles")][1].as_int() == 463,
        "split source span",
    )
    _check(
        spans[String("_blend")][0].as_int() == 466
        and spans[String("_blend")][1].as_int() == 482,
        "blend source span",
    )
    _check(
        spans[String("_stitch_tiles")][0].as_int() == 484
        and spans[String("_stitch_tiles")][1].as_int() == 504,
        "stitch source span",
    )
    var receipt = document[String("execution_receipt")]
    _check(
        receipt[String("method")].as_string()
        == String("AST-extracted unmodified upstream method bodies"),
        "upstream execution receipt",
    )
    var config = document[String("config")]
    _check(
        config[String("tile_size")].as_int() == 256
        and config[String("tile_overlap_min")].as_int() == 64
        and config[String("vae_ratio")].as_int() == 16,
        "released tiling config",
    )


def _check_rejections(ctx: DeviceContext) raises:
    var unaligned_rejected = False
    try:
        _ = minimax_h3_one_frame_tile_axis(271)
    except:
        unaligned_rejected = True
    _check(unaligned_rejected, "unaligned input rejection")

    var height_plan = minimax_h3_one_frame_tile_axis(272)
    var width_plan = minimax_h3_one_frame_tile_axis(320)
    var missing = List[TArc]()
    var missing_rejected = False
    try:
        _ = minimax_h3_stitch_one_frame_moment_tiles_f32(
            missing, height_plan, width_plan, 2, ctx,
        )
    except:
        missing_rejected = True
    _check(missing_rejected, "missing raw tile rejection")

    var single_height = minimax_h3_one_frame_tile_axis(256)
    var single_width = minimax_h3_one_frame_tile_axis(256)
    var wrong_dtype = List[TArc]()
    var values = List[Float32]()
    values.resize(1 * 1 * 16 * 16 * 2, Float32(0.25))
    wrong_dtype.append(
        TArc(Tensor.from_host(values, [1, 1, 16, 16, 2], STDtype.F16, ctx))
    )
    var dtype_rejected = False
    try:
        _ = minimax_h3_stitch_one_frame_moment_tiles_f32(
            wrong_dtype, single_height, single_width, 2, ctx,
        )
    except:
        dtype_rejected = True
    _check(dtype_rejected, "non-F32 moment tile rejection")


def main() raises:
    _check(has_accelerator(), "GPU accelerator required")
    var document = loads(Path(String(FIXTURE)).read_text())
    _check_fixture_contract(document)
    var axis_cases = document[String("axis_cases")]
    _check(axis_cases.length() == 8, "axis case inventory")
    for index in range(axis_cases.length()):
        var item = axis_cases[index]
        var plan = minimax_h3_one_frame_tile_axis(
            item[String("length")].as_int()
        )
        _check_axis(plan, item)

    var ctx = DeviceContext()
    var stitch_cases = document[String("stitch_cases")]
    _check(stitch_cases.length() == 3, "stitch case inventory")
    var observed_max = Float32(0.0)
    for index in range(stitch_cases.length()):
        var case_max = _run_stitch_case(stitch_cases[index], ctx)
        if case_max > observed_max:
            observed_max = case_max
    _check_rejections(ctx)
    print("MiniMax H3 VideoVAE tiling observed max_abs=", observed_max)
    print("MiniMax H3 VideoVAE tiling parity PASS")
