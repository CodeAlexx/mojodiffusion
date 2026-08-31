# Exact host parity + synthetic device integration smoke for H3 one-frame
# target-cache noise preparation.  Python only generates/checks the fixture.

from json.parser import loads
from json.value import JSONValue
from std.collections import List
from std.math import abs, isfinite
from std.pathlib import Path
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.minimax_h3.sha256 import minimax_h3_sha256_bytes
from serenitymojo.training.minimax_h3.torch_cpu_randn_avx2_v1 import (
    TORCH_CPU_RANDN_F32_AVX2_V1,
    torch_cpu_randn_f32_avx2_v1_raw_le_bytes,
)
from serenitymojo.training.minimax_h3.one_frame_target_cache_prepare import (
    MINIMAX_H3_ONE_FRAME_TARGET_NOISE_LAYOUT,
    MINIMAX_H3_ONE_FRAME_TARGET_NOISE_RECEIPT_V1,
    MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION,
    MiniMaxH3OneFrameTargetNoisePlan,
    minimax_h3_one_frame_target_cache_rng_metadata,
    minimax_h3_one_frame_target_noise_ncthw_values,
    minimax_h3_one_frame_target_noise_ndhwc_values,
    minimax_h3_one_frame_target_noise_receipt_material,
    minimax_h3_plan_one_frame_target_noise,
    minimax_h3_prepare_one_frame_target_cache_from_moments,
)


comptime FIXTURE = (
    "serenitymojo/training/parity/fixtures/"
    "minimax_h3_one_frame_target_cache_prepare_v1.json"
)
comptime SCHEMA = (
    "serenity.minimax_h3.one_frame_target_cache_prepare.fixture.v1"
)
comptime ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime CACHE_SOURCE = "src/musubi_tuner/minimax_h3_cache_latents.py"
comptime VIDEO_SOURCE = "src/musubi_tuner/minimax_h3/video_vae.py"
comptime CACHE_SHA256 = "a27d4541add4b256719de530a2daa5a3746d99a32ba168f5579a7e6cb69cb69b"
comptime VIDEO_SHA256 = "96e6698e5072adc258b6610881749d3748173d78c01c9b833e4cc42253165671"
comptime TORCH_VERSION = "2.12.0+cu130"
comptime TORCH_GIT = "7661cd9c6b841b62b7f411aa52ec51f05457263b"


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(
            String("MiniMax H3 one-frame target-cache prepare gate failed: ")
            + message
        )


def _shape(value: JSONValue) raises -> List[Int]:
    var output = List[Int](capacity=value.length())
    for index in range(value.length()):
        output.append(value[index].as_int())
    return output^


@always_inline
def _hex_nibble(value: UInt8) raises -> UInt8:
    if value >= UInt8(ord("0")) and value <= UInt8(ord("9")):
        return value - UInt8(ord("0"))
    if value >= UInt8(ord("a")) and value <= UInt8(ord("f")):
        return value - UInt8(ord("a")) + UInt8(10)
    raise Error("MiniMax H3 one-frame target-cache fixture has malformed hex")


def _compare_raw_hex(
    label: String,
    actual: List[UInt8],
    expected_hex: String,
) raises:
    _check(
        expected_hex.byte_length() == len(actual) * 2,
        label + String(" byte length"),
    )
    var encoded = expected_hex.as_bytes()
    for index in range(len(actual)):
        var expected = UInt8(
            (UInt32(_hex_nibble(encoded[index * 2])) << UInt32(4))
            | UInt32(_hex_nibble(encoded[index * 2 + 1]))
        )
        if actual[index] != expected:
            raise Error(
                String("MiniMax H3 one-frame target-cache prepare gate failed: ")
                + label + String(" byte ") + String(index)
            )


def _compare_bytes(
    label: String,
    actual: List[UInt8],
    expected_hex: String,
    expected_sha: String,
) raises:
    _compare_raw_hex(label, actual, expected_hex)
    _check(
        minimax_h3_sha256_bytes(actual.copy())
        == String("sha256:") + expected_sha,
        label + String(" SHA-256"),
    )


def _check_source_contract(document: JSONValue) raises:
    _check(document[String("schema")].as_string() == String(SCHEMA), "schema")
    _check(
        document[String("oracle_commit")].as_string() == String(ORACLE_COMMIT),
        "oracle commit",
    )
    var sources = document[String("source_contracts")]
    _check(
        sources[String(CACHE_SOURCE)][String("sha256")].as_string()
        == String(CACHE_SHA256),
        "cache source SHA",
    )
    _check(
        sources[String(VIDEO_SOURCE)][String("sha256")].as_string()
        == String(VIDEO_SHA256),
        "video source SHA",
    )
    var cache_names = sources[String(CACHE_SOURCE)][String("qualnames")]
    var video_names = sources[String(VIDEO_SOURCE)][String("qualnames")]
    _check(
        cache_names[String("build_one_frame_latent_tensors")][0].as_int() == 425
        and cache_names[String("build_one_frame_latent_tensors")][1].as_int() == 507,
        "build_one_frame_latent_tensors source span",
    )
    _check(
        video_names[String("_video_posterior_sample")][0].as_int() == 648
        and video_names[String("_video_posterior_sample")][1].as_int() == 659,
        "_video_posterior_sample source span",
    )
    _check(
        video_names[String("MiniMaxH3VideoVAE.__init__")][0].as_int() == 397
        and video_names[String("MiniMaxH3VideoVAE.__init__")][1].as_int() == 448,
        "VideoVAE spatial-compression source span",
    )
    var rng = document[String("rng_oracle")]
    _check(
        rng[String("profile")].as_string()
        == String(TORCH_CPU_RANDN_F32_AVX2_V1),
        "RNG profile",
    )
    _check(
        rng[String("torch_version")].as_string() == String(TORCH_VERSION),
        "torch version",
    )
    _check(
        rng[String("torch_git_version")].as_string() == String(TORCH_GIT),
        "torch git",
    )
    _check(
        document[String("receipt_schema")].as_string()
        == String(MINIMAX_H3_ONE_FRAME_TARGET_NOISE_RECEIPT_V1),
        "receipt schema",
    )
    _check(
        document[String("noise_layout")].as_string()
        == String(MINIMAX_H3_ONE_FRAME_TARGET_NOISE_LAYOUT),
        "noise layout",
    )
    _check(
        document[String("spatial_compression")].as_int()
        == MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION,
        "VideoVAE spatial compression",
    )
    var boundary = document[String("evidence_boundary")]
    _check(
        not boundary[String("real_image")].as_bool()
        and not boundary[String("real_vae")].as_bool()
        and not boundary[String("cache_artifact_written")].as_bool()
        and not boundary[String("dataset_mutated")].as_bool(),
        "fixture evidence boundary",
    )


def _check_metadata(
    plan: MiniMaxH3OneFrameTargetNoisePlan,
    expected: JSONValue,
) raises:
    var metadata = minimax_h3_one_frame_target_cache_rng_metadata(plan)
    _check(len(metadata) == 7, "metadata exact key inventory")
    var keys = [
        String("h3_target_rng_profile"),
        String("h3_target_rng_receipt_schema"),
        String("h3_target_rng_receipt"),
        String("h3_target_rng_canonical_item_key"),
        String("h3_target_rng_seed"),
        String("h3_target_rng_noise_layout"),
        String("h3_target_cache_shape"),
    ]
    for key in keys:
        _check(metadata.__contains__(key), String("metadata key ") + key)
        _check(
            metadata[key] == expected[key].as_string(),
            String("metadata value ") + key,
        )


def _expect_plan_rejected(
    plan: MiniMaxH3OneFrameTargetNoisePlan,
    label: String,
) raises:
    var rejected = False
    try:
        _ = minimax_h3_one_frame_target_noise_ncthw_values(plan)
    except:
        rejected = True
    _check(rejected, label)


def _check_rejections(plan: MiniMaxH3OneFrameTargetNoisePlan) raises:
    var unknown_profile = False
    try:
        _ = minimax_h3_plan_one_frame_target_noise(
            String("item.png"), 0, 1024, 1024, String("torch_cpu_default")
        )
    except:
        unknown_profile = True
    _check(unknown_profile, "unknown RNG profile rejection")

    var empty_key = False
    try:
        _ = minimax_h3_plan_one_frame_target_noise(String(""), 0, 1024, 1024)
    except:
        empty_key = True
    _check(empty_key, "empty item key rejection")

    var bad_height = False
    try:
        _ = minimax_h3_plan_one_frame_target_noise(
            String("item.png"), 0, 1000, 1024,
        )
    except:
        bad_height = True
    _check(bad_height, "non-32-divisible height rejection")

    var bad_width = False
    try:
        _ = minimax_h3_plan_one_frame_target_noise(
            String("item.png"), 0, 1024, 1000,
        )
    except:
        bad_width = True
    _check(bad_width, "non-32-divisible width rejection")

    var mutated = plan.copy()
    mutated.canonical_item_key += String("#1f")
    _expect_plan_rejected(mutated, "double-suffixed canonical key rejection")
    mutated = plan.copy()
    mutated.derived_seed += 1
    _expect_plan_rejected(mutated, "changed derived seed rejection")
    mutated = plan.copy()
    mutated.latent_width += 1
    _expect_plan_rejected(mutated, "changed latent geometry rejection")
    mutated = plan.copy()
    mutated.receipt = String("sha256:broken")
    _expect_plan_rejected(mutated, "changed receipt rejection")


def _synthetic_device_smoke(
    plan: MiniMaxH3OneFrameTargetNoisePlan,
    ctx: DeviceContext,
) raises -> Float32:
    var shape = plan.moments_shape_ndhwc()
    var count = 1
    for dim in shape:
        count *= dim
    var moments_values = List[Float32](capacity=count)
    for index in range(count):
        var channel = index % 48
        if channel < 24:
            moments_values.append(
                Float32(((index * 17 + channel * 3) % 101) - 50)
                / Float32(37.0)
            )
        else:
            moments_values.append(
                Float32(((index * 13 + channel) % 17) - 8)
                / Float32(4.0)
            )
    var moments = Tensor.from_host(
        moments_values, shape.copy(), STDtype.F32, ctx,
    )
    var prepared = minimax_h3_prepare_one_frame_target_cache_from_moments(
        plan, moments, ctx,
    )
    _check(
        prepared.target_cache.shape() == plan.cache_shape_cthw(),
        "device target cache exact [24,1,H',W'] shape",
    )
    _check(
        prepared.target_cache.dtype() == STDtype.F32,
        "device target cache F32",
    )
    _check(
        prepared.noise_plan.receipt == plan.receipt,
        "device target cache receipt handoff",
    )
    var values = prepared.target_cache.to_host(ctx)
    var max_abs = Float32(0.0)
    for value in values:
        _check(isfinite(value), "device target cache finite")
        if abs(value) > max_abs:
            max_abs = abs(value)
    _check(max_abs > Float32(0.1), "device target cache nondegenerate")

    var wrong_shape = shape.copy()
    wrong_shape[3] -= 1
    var wrong_count = 1
    for dim in wrong_shape:
        wrong_count *= dim
    var wrong_values = List[Float32]()
    wrong_values.resize(wrong_count, Float32(0.0))
    var wrong_shape_rejected = False
    try:
        var wrong = Tensor.from_host(
            wrong_values, wrong_shape^, STDtype.F32, ctx,
        )
        _ = minimax_h3_prepare_one_frame_target_cache_from_moments(
            plan, wrong, ctx,
        )
    except:
        wrong_shape_rejected = True
    _check(wrong_shape_rejected, "wrong moment geometry rejection")
    return max_abs


def main() raises:
    comptime assert has_accelerator(), "H3 target-cache prepare gate requires GPU"
    var document = loads(Path(String(FIXTURE)).read_text())
    _check_source_contract(document)
    var cases = document[String("cases")]
    _check(cases.length() == 2, "case inventory")
    var expected_labels = [
        String("one_frame_30x52"), String("one_frame_32x32"),
    ]
    var plans = List[MiniMaxH3OneFrameTargetNoisePlan]()
    for index in range(cases.length()):
        var item = cases[index]
        var label = item[String("label")].as_string()
        _check(label == expected_labels[index], "case order")
        var plan = minimax_h3_plan_one_frame_target_noise(
            item[String("item_key")].as_string(),
            item[String("cache_seed")].as_int(),
            item[String("source_height")].as_int(),
            item[String("source_width")].as_int(),
            item[String("rng_profile")].as_string(),
        )
        _check(
            plan.canonical_item_key
            == item[String("canonical_item_key")].as_string(),
            label + String(" canonical #1f key"),
        )
        _check(
            plan.derived_seed == item[String("derived_seed")].as_int(),
            label + String(" SHA-derived seed"),
        )
        _check(
            plan.noise_shape_ncthw()
            == _shape(item[String("noise_shape_ncthw")]),
            label + String(" NCTHW shape"),
        )
        _check(
            plan.posterior_shape_ndhwc()
            == _shape(item[String("posterior_shape_ndhwc")]),
            label + String(" NDHWC shape"),
        )
        _check(
            plan.cache_shape_cthw()
            == _shape(item[String("cache_shape_cthw")]),
            label + String(" cache shape"),
        )
        _check(
            minimax_h3_one_frame_target_noise_receipt_material(plan)
            == item[String("receipt_material")].as_string(),
            label + String(" receipt material"),
        )
        _check(
            plan.receipt == item[String("receipt")].as_string(),
            label + String(" receipt"),
        )
        _check_metadata(plan, item[String("metadata")])

        var ncthw = minimax_h3_one_frame_target_noise_ncthw_values(plan)
        var ncthw_raw = torch_cpu_randn_f32_avx2_v1_raw_le_bytes(ncthw)
        _compare_bytes(
            label + String(" NCTHW"),
            ncthw_raw,
            item[String("noise_ncthw_raw_le_hex")].as_string(),
            item[String("noise_ncthw_sha256")].as_string(),
        )
        var ndhwc = minimax_h3_one_frame_target_noise_ndhwc_values(plan)
        var ndhwc_raw = torch_cpu_randn_f32_avx2_v1_raw_le_bytes(ndhwc)
        _compare_bytes(
            label + String(" NDHWC"),
            ndhwc_raw,
            item[String("noise_ndhwc_raw_le_hex")].as_string(),
            item[String("noise_ndhwc_sha256")].as_string(),
        )
        plans.append(plan.copy())

    _check(plans[0].derived_seed > 0xFFFFFFFF, "high-32-bit derived seed")
    _check_rejections(plans[0])
    var ctx = DeviceContext()
    var max_abs = _synthetic_device_smoke(plans[0], ctx)
    print("PASS MiniMax H3 one-frame target-cache preparation")
    print("  exact host parity: 2 real H3 geometries, NCTHW draw + NDHWC map")
    print("  canonical item key suffix: #1f; RNG metadata/receipt: exact")
    print("  synthetic device integration smoke max_abs:", max_abs)
    print("  evidence: component parity + device smoke; real VAE/cache: NOT RUN")
    print("  dataset/cache artifacts: NOT WRITTEN/NOT MUTATED")
