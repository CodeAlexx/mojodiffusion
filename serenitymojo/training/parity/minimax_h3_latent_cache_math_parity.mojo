# Synthetic device parity for the pinned MiniMax-H3 one-frame latent-cache math.
# Python executes pinned Musubi source only to generate/check the fixture.

from json.parser import loads
from json.value import JSONValue
from std.collections import List
from std.math import abs, isfinite
from std.pathlib import Path
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage
from serenitymojo.tensor import Tensor
from serenitymojo.training.minimax_h3.latent_cache_math import (
    minimax_h3_prepare_pixels_single_frame_ndhwc,
    minimax_h3_split_single_frame_video_moments_f32,
    minimax_h3_target_noise_seed,
    minimax_h3_video_target_latent_cache_from_moments,
    minimax_h3_video_vae_input_single_frame_device,
)


comptime FIXTURE = "serenitymojo/training/parity/fixtures/minimax_h3_latent_cache_math_v1.json"
comptime SCHEMA = "serenity.minimax_h3.latent_cache_math_oracle.v1"
comptime ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime VIDEO_SOURCE = "src/musubi_tuner/minimax_h3/video_vae.py"
comptime VIDEO_SHA256 = "96e6698e5072adc258b6610881749d3748173d78c01c9b833e4cc42253165671"
comptime CACHE_SOURCE = "src/musubi_tuner/minimax_h3_cache_latents.py"
comptime CACHE_SHA256 = "a27d4541add4b256719de530a2daa5a3746d99a32ba168f5579a7e6cb69cb69b"


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax H3 latent-cache math parity failed: ") + message)


def _f32_list(value: JSONValue) raises -> List[Float32]:
    var out = List[Float32](capacity=value.length())
    for index in range(value.length()):
        out.append(Float32(value[index].as_float()))
    return out^


def _u8_list(value: JSONValue) raises -> List[UInt8]:
    var out = List[UInt8](capacity=value.length())
    for index in range(value.length()):
        var item = value[index].as_int()
        _check(item >= 0 and item <= 255, "RGB8 fixture range")
        out.append(UInt8(item))
    return out^


def _int_list(value: JSONValue) raises -> List[Int]:
    var out = List[Int](capacity=value.length())
    for index in range(value.length()):
        out.append(value[index].as_int())
    return out^


def _compare_tight(
    label: String, got: List[Float32], expected: List[Float32], tolerance: Float32,
) raises -> Float32:
    _check(len(got) == len(expected), label + String(" length"))
    var max_abs = Float32(0.0)
    for index in range(len(got)):
        _check(isfinite(got[index]), label + String(" finite"))
        var difference = abs(got[index] - expected[index])
        if difference > max_abs:
            max_abs = difference
        if difference > tolerance:
            raise Error(
                String("MiniMax H3 latent-cache math parity failed: ") + label
                + String(" index=") + String(index)
                + String(" got=") + String(got[index])
                + String(" expected=") + String(expected[index])
                + String(" abs=") + String(difference)
                + String(" tolerance=") + String(tolerance)
            )
    return max_abs


def _compare_posterior(
    got: List[Float32], expected: List[Float32],
) raises -> Float32:
    _check(len(got) == len(expected), "posterior length")
    var max_abs = Float32(0.0)
    for index in range(len(got)):
        _check(isfinite(got[index]), "posterior finite")
        var difference = abs(got[index] - expected[index])
        if difference > max_abs:
            max_abs = difference
        # The CUDA kernels and pinned Torch oracle both execute F32, but the
        # exp/multiply/subtract chain may round at different instruction
        # boundaries.  Bound that expected few-ULP drift without admitting a
        # percent-scale error at the fixture's deliberately large values.
        var tolerance = (
            Float32(0.000001)
            + abs(expected[index]) * Float32(0.0000004)
        )
        if difference > tolerance:
            raise Error(
                String("MiniMax H3 latent-cache math parity failed: posterior index=")
                + String(index) + String(" got=") + String(got[index])
                + String(" expected=") + String(expected[index])
                + String(" abs=") + String(difference)
                + String(" tolerance=") + String(tolerance)
            )
    return max_abs


def _check_source_contracts(document: JSONValue) raises:
    _check(document[String("schema")].as_string() == String(SCHEMA), "schema")
    _check(
        document[String("oracle_commit")].as_string() == String(ORACLE_COMMIT),
        "oracle commit",
    )
    var contracts = document[String("source_contracts")]
    _check(
        contracts[String(VIDEO_SOURCE)][String("sha256")].as_string()
        == String(VIDEO_SHA256),
        "video source SHA-256",
    )
    _check(
        contracts[String(CACHE_SOURCE)][String("sha256")].as_string()
        == String(CACHE_SHA256),
        "cache source SHA-256",
    )
    var video_qualnames = contracts[String(VIDEO_SOURCE)][String("qualnames")]
    var cache_qualnames = contracts[String(CACHE_SOURCE)][String("qualnames")]
    _check(
        video_qualnames[String("_video_posterior_sample")][0].as_int() == 648
        and video_qualnames[String("_video_posterior_sample")][1].as_int() == 659,
        "posterior source span",
    )
    _check(
        cache_qualnames[String("_prepare_pixels")][0].as_int() == 209
        and cache_qualnames[String("_prepare_pixels")][1].as_int() == 222,
        "pixel source span",
    )
    var receipt = document[String("execution_receipt")]
    _check(
        receipt[String("torch_cpu_rng")].as_string()
        == String("not generated or claimed"),
        "RNG evidence boundary",
    )


def _check_seed_cases(document: JSONValue) raises:
    var cases = document[String("seed_cases")]
    _check(cases.length() == 3, "seed case inventory")
    for index in range(cases.length()):
        var item = cases[index]
        var got = minimax_h3_target_noise_seed(
            item[String("cache_seed")].as_int(),
            item[String("canonical_item_key")].as_string(),
        )
        _check(got == item[String("derived_seed")].as_int(), "seed case " + String(index))


def _check_rejections(
    moments: Tensor,
    moment_shape: List[Int],
    noise_values: List[Float32],
    ctx: DeviceContext,
) raises:
    var missing_rejected = False
    try:
        var no_noise = Optional[Tensor](None)
        _ = minimax_h3_video_target_latent_cache_from_moments(
            moments, no_noise, ctx,
        )
    except:
        missing_rejected = True
    _check(missing_rejected, "missing injected noise rejection")

    var wrong_dtype_rejected = False
    try:
        var bad_noise = Tensor.from_host(
            noise_values, moment_shape.copy(), STDtype.F16, ctx,
        )
        var optional_bad = Optional[Tensor](bad_noise^)
        _ = minimax_h3_video_target_latent_cache_from_moments(
            moments, optional_bad, ctx,
        )
    except:
        wrong_dtype_rejected = True
    _check(wrong_dtype_rejected, "non-F32 injected noise rejection")

    var short_shape = moment_shape.copy()
    short_shape[3] = short_shape[3] - 1
    var short_values = List[Float32]()
    short_values.resize(
        short_shape[0] * short_shape[1] * short_shape[2]
        * short_shape[3] * short_shape[4],
        Float32(0.25),
    )
    var wrong_shape_rejected = False
    try:
        var bad_shape_noise = Tensor.from_host(
            short_values, short_shape^, STDtype.F32, ctx,
        )
        var optional_bad_shape = Optional[Tensor](bad_shape_noise^)
        _ = minimax_h3_video_target_latent_cache_from_moments(
            moments, optional_bad_shape, ctx,
        )
    except:
        wrong_shape_rejected = True
    _check(wrong_shape_rejected, "wrong injected noise shape rejection")


def main() raises:
    comptime assert has_accelerator(), "MiniMax H3 latent-cache parity requires GPU"
    var document = loads(Path(String(FIXTURE)).read_text())
    _check_source_contracts(document)
    _check_seed_cases(document)

    var pixel_case = document[String("pixel_case")]
    var height = pixel_case[String("height")].as_int()
    var width = pixel_case[String("width")].as_int()
    var image = MiniMaxH3RgbImage(
        _u8_list(pixel_case[String("rgb8_hwc")]), height, width,
    )
    var prepared = minimax_h3_prepare_pixels_single_frame_ndhwc(image)
    var prepared_max = _compare_tight(
        "_prepare_pixels",
        prepared,
        _f32_list(pixel_case[String("prepared_ndhwc_f32")]),
        Float32(0.00000025),
    )

    var ctx = DeviceContext()
    var encoder_pixels = minimax_h3_video_vae_input_single_frame_device(image, ctx)
    _check(
        encoder_pixels.shape() == [1, 1, height, width, 3],
        "encoder pixel shape [1,1,H,W,3]",
    )
    _check(encoder_pixels.dtype() == STDtype.F32, "encoder pixel dtype F32")
    var encoder_max = _compare_tight(
        "encode_moments ImageNet input",
        encoder_pixels.to_host(ctx),
        _f32_list(pixel_case[String("encoder_input_ndhwc_f32")]),
        Float32(0.000002),
    )

    var posterior = document[String("posterior_case")]
    var moment_shape = _int_list(posterior[String("moments_shape_ndhwc")])
    var noise_shape = _int_list(posterior[String("noise_shape_ndhwc")])
    var cache_shape = _int_list(posterior[String("cache_shape_cthw")])
    _check(moment_shape == [1, 1, 2, 3, 48], "fixture moment geometry")
    _check(noise_shape == [1, 1, 2, 3, 24], "fixture noise geometry")
    _check(cache_shape == [24, 1, 2, 3], "fixture cache geometry")
    var moments = Tensor.from_host(
        _f32_list(posterior[String("moments_ndhwc_f32")]),
        moment_shape^,
        STDtype.F32,
        ctx,
    )
    var split = minimax_h3_split_single_frame_video_moments_f32(moments, ctx)
    var mean_max = _compare_tight(
        "moment mean split",
        split.mean.to_host(ctx),
        _f32_list(posterior[String("mean_ndhwc_f32")]),
        Float32(0.0),
    )
    var logvar_expected = _f32_list(posterior[String("logvar_ndhwc_f32")])
    var logvar_max = _compare_tight(
        "moment logvar split", split.logvar.to_host(ctx), logvar_expected, Float32(0.0),
    )
    var saw_low = False
    var saw_high = False
    for value in logvar_expected:
        if value < Float32(-30.0):
            saw_low = True
        if value > Float32(20.0):
            saw_high = True
    _check(saw_low and saw_high, "nondegenerate logvar clamp coverage")

    var noise_values = _f32_list(posterior[String("noise_ndhwc_f32")])
    var max_noise = Float32(0.0)
    for value in noise_values:
        if abs(value) > max_noise:
            max_noise = abs(value)
    _check(max_noise > Float32(1.0), "nondegenerate injected noise")
    _check_rejections(moments, noise_shape, noise_values, ctx)

    var noise = Tensor.from_host(
        noise_values, noise_shape^, STDtype.F32, ctx,
    )
    var optional_noise = Optional[Tensor](noise^)
    var cache = minimax_h3_video_target_latent_cache_from_moments(
        moments, optional_noise, ctx,
    )
    _check(cache.shape() == cache_shape, "cache [24,1,H,W] layout")
    _check(cache.dtype() == STDtype.F32, "cache dtype F32")
    var posterior_expected = _f32_list(
        posterior[String("expected_cache_cthw_f32")]
    )
    var output_max = _compare_posterior(cache.to_host(ctx), posterior_expected)
    var expected_max = Float32(0.0)
    for value in posterior_expected:
        if abs(value) > expected_max:
            expected_max = abs(value)
    _check(expected_max > Float32(1000.0), "posterior fixture is nondegenerate")

    print("PASS MiniMax H3 one-frame latent-cache math synthetic device parity")
    print("  pinned Musubi:", String(ORACLE_COMMIT))
    print("  _prepare_pixels max_abs:", prepared_max)
    print("  ImageNet encoder-input max_abs:", encoder_max)
    print("  mean/logvar split max_abs:", mean_max, logvar_max)
    print("  posterior+normalize+CTHW max_abs:", output_max)
    print("  injected noise required; Torch CPU RNG: NOT IMPLEMENTED/NOT CLAIMED")
    print("  real image/VAE/cache artifact: NOT TESTED")
