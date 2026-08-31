# Real released MiniMax-H3 VideoVAE tiled-moments receipt gate.
#
# This parity-only executable loads the released encoder through the explicit
# F32-compute loader, then exercises the pure-Mojo cache-side path:
#   JPEG decode -> 1024 bucket -> exact resize/crop -> 25 tiled VAE forwards
#   -> raw stitched moments [1,1,64,64,48].
# It compares a compact pinned Torch/Musubi receipt.  It does not draw the
# posterior, write a cache artifact, or enter the trainer.

from json.parser import loads
from json.value import JSONValue
from std.collections import List
from std.math import abs, isfinite, max, sqrt
from std.pathlib import Path
from std.sys import argv, has_accelerator
from max.gpu.host import DeviceContext

from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    minimax_h3_video_released_encoder_config,
)
from serenitymojo.training.minimax_h3.real_video_vae_cache_encode import (
    MINIMAX_H3_SOURCE_ADMISSION_MULTIPLE,
    MINIMAX_H3_VIDEO_VAE_SPATIAL_COMPRESSION,
    minimax_h3_real_one_frame_moments_f32,
)


comptime SCHEMA = "serenity.minimax_h3.real_video_vae_moments_f32.v1"
comptime ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime VIDEO_SHA256 = "96e6698e5072adc258b6610881749d3748173d78c01c9b833e4cc42253165671"
comptime CACHE_SHA256 = "a27d4541add4b256719de530a2daa5a3746d99a32ba168f5579a7e6cb69cb69b"
comptime MEDIA_SHA256 = "b3f99b9ef362c97788b365c4dc5ac3c2f75f29949e7fef91697df5a1950ed5f6"
comptime IMAGE_SHA256 = "fc41782cac93cafc92e83ddb57e93243c9f4f97c70f25f4b4fec5d64f875a996"
comptime RGB_SHA256 = "d9d996ff5f085ccc0ad7c4080dad532003a67881f6a7784799e6633ddbee042c"
comptime VAE_SHA256 = "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
comptime VAE_SIZE = 5_207_808_496

# Initial value-class bars.  The mandatory runner must only be accepted after
# these are replaced/tightened from repeated Torch/Mojo observations.
comptime SAMPLE_MAX_ABS_TOL = Float64(2.0e-2)
comptime SAMPLE_REL_L2_TOL = Float64(5.0e-4)
comptime SAMPLE_COSINE_MIN = Float64(0.999999)
comptime SAMPLE_NORM_RATIO_DELTA = Float64(5.0e-4)
comptime GLOBAL_REL_TOL = Float64(8.0e-4)
comptime CHANNEL_REL_TOL = Float64(1.0e-3)
comptime STAT_ABS_FLOOR = Float64(2.0e-4)


@fieldwise_init
struct Stats(Copyable, Movable):
    var minimum: Float64
    var maximum: Float64
    var mean: Float64
    var abs_mean: Float64
    var std_population: Float64
    var l2: Float64
    var sum: Float64


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax H3 real VideoVAE moments gate failed: ") + message)


def _stats(values: List[Float32]) raises -> Stats:
    _check(len(values) > 0, "empty moments")
    var minimum = Float64(values[0])
    var maximum = minimum
    var sum = Float64(0.0)
    var abs_sum = Float64(0.0)
    var square_sum = Float64(0.0)
    for value in values:
        _check(isfinite(value), "nonfinite raw moment")
        var wide = Float64(value)
        if wide < minimum:
            minimum = wide
        if wide > maximum:
            maximum = wide
        sum += wide
        abs_sum += abs(wide)
        square_sum += wide * wide
    var count = Float64(len(values))
    var mean = sum / count
    var variance = square_sum / count - mean * mean
    if variance < 0.0:
        variance = 0.0
    return Stats(
        minimum,
        maximum,
        mean,
        abs_sum / count,
        sqrt(variance),
        sqrt(square_sum),
        sum,
    )


def _channel_values(values: List[Float32], channel: Int) -> List[Float32]:
    var output = List[Float32](capacity=len(values) // 48)
    for spatial in range(len(values) // 48):
        output.append(values[spatial * 48 + channel])
    return output^


def _relative_error(actual: Float64, expected: Float64) -> Float64:
    return abs(actual - expected) / max(abs(expected), STAT_ABS_FLOOR)


def _check_stat(
    actual: Float64,
    expected: Float64,
    relative_tolerance: Float64,
    label: String,
) raises:
    var relative = _relative_error(actual, expected)
    if relative > relative_tolerance:
        print("  stat failure", label, "actual=", actual, "expected=", expected, "rel=", relative)
    _check(relative <= relative_tolerance, label)


def _check_stats(
    actual: Stats,
    expected: JSONValue,
    relative_tolerance: Float64,
    label: String,
) raises:
    _check_stat(actual.minimum, expected["min"].as_float(), relative_tolerance, label + String(" min"))
    _check_stat(actual.maximum, expected["max"].as_float(), relative_tolerance, label + String(" max"))
    _check_stat(actual.mean, expected["mean"].as_float(), relative_tolerance, label + String(" mean"))
    _check_stat(actual.abs_mean, expected["abs_mean"].as_float(), relative_tolerance, label + String(" abs_mean"))
    _check_stat(actual.std_population, expected["std_population"].as_float(), relative_tolerance, label + String(" std"))
    _check_stat(actual.l2, expected["l2"].as_float(), relative_tolerance, label + String(" l2"))
    _check_stat(actual.sum, expected["sum"].as_float(), relative_tolerance, label + String(" sum"))


def _check_fixture(document: JSONValue) raises:
    _check(document["schema"].as_string() == String(SCHEMA), "fixture schema")
    _check(document["oracle_commit"].as_string() == String(ORACLE_COMMIT), "oracle commit")
    var sources = document["source_contracts"]
    _check(sources["src/musubi_tuner/minimax_h3/video_vae.py"]["sha256"].as_string() == String(VIDEO_SHA256), "video source SHA")
    _check(sources["src/musubi_tuner/minimax_h3_cache_latents.py"]["sha256"].as_string() == String(CACHE_SHA256), "cache source SHA")
    _check(sources["src/musubi_tuner/dataset/media_utils.py"]["sha256"].as_string() == String(MEDIA_SHA256), "media source SHA")
    var image = document["image"]
    _check(image["relative_path"].as_string() == String("1.jpg"), "image path")
    _check(image["file_size"].as_int() == 499_926, "image size")
    _check(image["file_sha256"].as_string() == String(IMAGE_SHA256), "image SHA")
    _check(image["source_width"].as_int() == 1024 and image["source_height"].as_int() == 1024, "source geometry")
    _check(image["bucket_width"].as_int() == 1024 and image["bucket_height"].as_int() == 1024, "bucket geometry")
    var geometry = document["geometry_contract"]
    _check(geometry["source_admission_multiple"].as_int() == MINIMAX_H3_SOURCE_ADMISSION_MULTIPLE, "/32 source admission")
    _check(geometry["raw_moments_spatial_compression"].as_int() == MINIMAX_H3_VIDEO_VAE_SPATIAL_COMPRESSION, "/16 raw moments")
    _check(geometry["raw_moments_height"].as_int() == 64 and geometry["raw_moments_width"].as_int() == 64, "raw moment geometry")
    _check(geometry["downstream_packed_grid_spatial_compression"].as_int() == 32, "downstream /32 grid")
    _check(not geometry["downstream_packing_executed"].as_bool(), "packing must remain outside gate")
    var released = document["released_vae"]
    _check(released["file_size"].as_int() == VAE_SIZE, "VAE file size")
    _check(released["file_sha256"].as_string() == String(VAE_SHA256), "VAE SHA")
    _check(released["storage_dtype"].as_string() == String("F16") and released["compute_dtype"].as_string() == String("F32"), "VAE dtype boundary")
    var tiling = document["tiling"]
    _check(tiling["tile_size_pixels"].as_int() == 256 and tiling["minimum_overlap_pixels"].as_int() == 64, "tiling config")
    _check(tiling["grid_rows"].as_int() == 5 and tiling["grid_columns"].as_int() == 5, "full 5x5 tiling")
    var moments = document["moments"]
    _check(moments["shape"].length() == 5, "moment rank")
    _check(moments["shape"][0].as_int() == 1 and moments["shape"][1].as_int() == 1 and moments["shape"][2].as_int() == 64 and moments["shape"][3].as_int() == 64 and moments["shape"][4].as_int() == 48, "moment shape")


def _compare_samples(values: List[Float32], moments: JSONValue) raises:
    var indices = moments["sample_indices_flat_ndhwc"]
    var expected = moments["sample_values_f32"]
    _check(indices.length() == expected.length() and indices.length() >= 512, "sample receipt inventory")
    var max_abs = Float64(0.0)
    var error_square = Float64(0.0)
    var expected_square = Float64(0.0)
    var actual_square = Float64(0.0)
    var dot = Float64(0.0)
    for sample in range(indices.length()):
        var index = indices[sample].as_int()
        _check(index >= 0 and index < len(values), "sample index range")
        var actual = Float64(values[index])
        var reference = expected[sample].as_float()
        var difference = actual - reference
        var magnitude = abs(difference)
        if magnitude > max_abs:
            max_abs = magnitude
        error_square += difference * difference
        expected_square += reference * reference
        actual_square += actual * actual
        dot += actual * reference
    _check(expected_square > 0.0 and actual_square > 0.0, "nondegenerate samples")
    var relative_l2 = sqrt(error_square / expected_square)
    var cosine = dot / sqrt(actual_square * expected_square)
    var norm_ratio = sqrt(actual_square / expected_square)
    print("  sample max_abs=", max_abs)
    print("  sample rel_l2=", relative_l2)
    print("  sample cosine=", cosine)
    print("  sample norm_ratio=", norm_ratio)
    _check(max_abs <= SAMPLE_MAX_ABS_TOL, "sample max_abs")
    _check(relative_l2 <= SAMPLE_REL_L2_TOL, "sample relative L2")
    _check(cosine >= SAMPLE_COSINE_MIN, "sample cosine")
    _check(abs(norm_ratio - 1.0) <= SAMPLE_NORM_RATIO_DELTA, "sample norm ratio")


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error("usage: minimax_h3_real_video_vae_moments_parity <dataset> <vae-dir> <fixture.json>")
    _check(has_accelerator(), "GPU accelerator required")
    var document = loads(Path(args[3]).read_text())
    _check_fixture(document)

    var ctx = DeviceContext()
    var encoder = MiniMaxH3VideoEncoderDevice.load_f32_compute(
        args[2], minimax_h3_video_released_encoder_config(), ctx,
    )
    var encoded = minimax_h3_real_one_frame_moments_f32(
        encoder, args[1] + String("/1.jpg"), ctx,
    )
    _check(encoded.source_width == 1024 and encoded.source_height == 1024, "decoded source geometry")
    _check(encoded.bucket_width == 1024 and encoded.bucket_height == 1024, "prepared bucket geometry")
    _check(encoded.moment_width == 64 and encoded.moment_height == 64, "returned /16 moment geometry")
    _check(encoded.decoded_rgb_sha256 == String(RGB_SHA256), "decoded RGB SHA")
    _check(encoded.prepared_rgb_sha256 == String(RGB_SHA256), "prepared RGB SHA")

    var values = encoded.moments.to_host(ctx)
    _check(len(values) == 1 * 1 * 64 * 64 * 48, "moment value count")
    var moments = document["moments"]
    _compare_samples(values, moments)
    _check_stats(_stats(values), moments["global_stats_f64_from_f32"], GLOBAL_REL_TOL, String("global"))
    var channel_stats = moments["channel_stats_f64_from_f32"]
    _check(channel_stats.length() == 48, "channel-stat inventory")
    for channel in range(48):
        _check_stats(
            _stats(_channel_values(values, channel)),
            channel_stats[channel],
            CHANNEL_REL_TOL,
            String("channel ") + String(channel),
        )
    print("PASS real released MiniMax-H3 VideoVAE F32 tiled moments receipt")
    print("  source admission /32; raw moments/posterior /16; DiT packed grid /32 only downstream")
    print("  NOT TESTED: posterior RNG, cache artifact, trainer, or visual quality")
