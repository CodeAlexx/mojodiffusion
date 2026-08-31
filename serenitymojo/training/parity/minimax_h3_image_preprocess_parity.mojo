# Exact RGB8 parity for pinned Musubi MiniMax-H3 resize/crop preprocessing.
# Fixture values come from the upstream function itself under
# opencv-python==4.10.0.84 and pillow==11.3.0.

from json.parser import loads
from json.value import JSONValue
from std.collections import List
from std.math import abs
from std.pathlib import Path

from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage
from serenitymojo.training.minimax_h3.image_preprocess import (
    minimax_h3_image_resize_plan,
    minimax_h3_inter_area_resize,
    minimax_h3_resize_image_to_bucket,
)


comptime FIXTURE = "serenitymojo/training/parity/fixtures/minimax_h3_image_preprocess_v1.json"
comptime SCHEMA = "serenity.minimax_h3.image_preprocess_oracle.v1"
comptime ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime SOURCE_FILE = "src/musubi_tuner/dataset/media_utils.py"
comptime SOURCE_SHA256 = "b3f99b9ef362c97788b365c4dc5ac3c2f75f29949e7fef91697df5a1950ed5f6"
comptime OPENCV_DIST_VERSION = "4.10.0.84"
comptime CV2_VERSION = "4.10.0"
comptime PILLOW_VERSION = "11.3.0"
comptime NUMPY_VERSION = "2.5.2"


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax H3 image preprocessing parity failed: ") + message)


def _u8_list(value: JSONValue) raises -> List[UInt8]:
    var out = List[UInt8](capacity=value.length())
    for index in range(value.length()):
        var item = value[index].as_int()
        _check(item >= 0 and item <= 255, "fixture RGB8 range")
        out.append(UInt8(item))
    return out^


def _compare_bytes(
    name: String, got: MiniMaxH3RgbImage, expected: JSONValue, target_width: Int,
    target_height: Int,
) raises:
    _check(
        got.width == target_width and got.height == target_height,
        name + String(" output geometry"),
    )
    _check(len(got.pixels) == expected.length(), name + String(" byte count"))
    var mismatches = 0
    var max_abs = 0
    for index in range(expected.length()):
        var wanted = expected[index].as_int()
        var difference = Int(got.pixels[index]) - wanted
        if difference < 0:
            difference = -difference
        if difference != 0:
            mismatches += 1
        if difference > max_abs:
            max_abs = difference
    _check(
        mismatches == 0,
        name + String(" byte parity: mismatches=") + String(mismatches)
        + String(" max_abs=") + String(max_abs),
    )


def _check_rejections() raises:
    var pixels = List[UInt8]()
    pixels.resize(4 * 4 * 3, UInt8(1))
    var image = MiniMaxH3RgbImage(pixels^, 4, 4)
    var rejected_geometry = False
    try:
        _ = minimax_h3_image_resize_plan(4, 4, 0, 4)
    except:
        rejected_geometry = True
    _check(rejected_geometry, "zero target geometry rejection")

    var rejected_area_upscale = False
    try:
        _ = minimax_h3_inter_area_resize(image, 5, 4)
    except:
        rejected_area_upscale = True
    _check(rejected_area_upscale, "INTER_AREA upscale rejection")


def main() raises:
    var document = loads(Path(String(FIXTURE)).read_text())
    _check(document[String("schema")].as_string() == String(SCHEMA), "schema")
    _check(
        document[String("oracle_commit")].as_string() == String(ORACLE_COMMIT),
        "oracle commit",
    )
    _check(
        document[String("source_file")].as_string() == String(SOURCE_FILE),
        "source file",
    )
    _check(
        document[String("source_sha256")].as_string() == String(SOURCE_SHA256),
        "source SHA-256",
    )
    _check(
        document[String("opencv_distribution_version")].as_string()
        == String(OPENCV_DIST_VERSION),
        "opencv-python version",
    )
    _check(
        document[String("cv2_runtime_version")].as_string() == String(CV2_VERSION),
        "cv2 runtime version",
    )
    _check(
        document[String("pillow_version")].as_string() == String(PILLOW_VERSION),
        "Pillow version",
    )
    _check(
        document[String("numpy_version")].as_string() == String(NUMPY_VERSION),
        "NumPy version",
    )

    var cases = document[String("cases")]
    _check(cases.length() == 7, "case count")
    var identity_count = 0
    var lanczos_count = 0
    var area_count = 0
    var integer_area_count = 0
    var noninteger_area_count = 0
    var crop_x_count = 0
    var crop_y_count = 0
    for case_index in range(cases.length()):
        var item = cases[case_index]
        var name = item[String("name")].as_string()
        var source_size = item[String("source_size")]
        var target_size = item[String("target_size")]
        var resized = item[String("resized")]
        var crop = item[String("crop")]
        var branch = item[String("branch")].as_string()
        var source_width = source_size[0].as_int()
        var source_height = source_size[1].as_int()
        var target_width = target_size[0].as_int()
        var target_height = target_size[1].as_int()
        var source_pixels = _u8_list(item[String("source_rgb8_hwc")])
        var image = MiniMaxH3RgbImage(
            source_pixels^, source_height, source_width,
        )
        var plan = minimax_h3_image_resize_plan(
            source_width, source_height, target_width, target_height,
        )
        _check(
            abs(plan.scale - item[String("scale")].as_float()) <= 1.0e-15,
            name + String(" scale"),
        )
        _check(
            plan.resized_width == resized[0].as_int()
            and plan.resized_height == resized[1].as_int(),
            name + String(" intermediate dimensions"),
        )
        _check(
            plan.crop_left == crop[0].as_int()
            and plan.crop_top == crop[1].as_int(),
            name + String(" center crop"),
        )
        if plan.crop_left > 0:
            crop_x_count += 1
        if plan.crop_top > 0:
            crop_y_count += 1

        if branch == String("identity"):
            identity_count += 1
            _check(plan.identity, name + String(" identity branch"))
        elif branch == String("pillow_lanczos"):
            lanczos_count += 1
            _check(plan.use_lanczos and not plan.identity, name + String(" Lanczos branch"))
        elif branch == String("opencv_inter_area"):
            area_count += 1
            _check(not plan.use_lanczos and not plan.identity, name + String(" area branch"))
            if plan.scale < 1.0:
                if (
                    source_width % plan.resized_width == 0
                    and source_height % plan.resized_height == 0
                ):
                    integer_area_count += 1
                else:
                    noninteger_area_count += 1
        else:
            raise Error(String("unknown fixture branch: ") + branch)

        var got = minimax_h3_resize_image_to_bucket(
            image, target_width, target_height,
        )
        _compare_bytes(
            name,
            got,
            item[String("expected_rgb8_hwc")],
            target_width,
            target_height,
        )

    _check(identity_count == 1, "identity inventory")
    _check(lanczos_count == 1, "Lanczos inventory")
    _check(area_count == 5, "INTER_AREA inventory")
    _check(integer_area_count == 2, "integer INTER_AREA inventory")
    _check(noninteger_area_count == 2, "non-integer INTER_AREA inventory")
    _check(crop_x_count >= 3, "horizontal crop coverage")
    _check(crop_y_count >= 1, "vertical crop coverage")
    _check_rejections()
    print("PASS MiniMax H3 Musubi image resize/crop byte parity; cases:", cases.length())
    print("  Pillow LANCZOS: reused existing bit-exact H3 resizer")
    print("  OpenCV INTER_AREA: integer + non-integer pure-Mojo paths")
    print("  Python/OpenCV product dependency: NONE")
