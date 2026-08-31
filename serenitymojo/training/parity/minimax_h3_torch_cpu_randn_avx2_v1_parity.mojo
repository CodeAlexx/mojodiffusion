# Exact-byte host parity for the narrow H3 torch CPU AVX2 F32 randn profile.

from json.parser import loads
from json.value import JSONValue
from std.collections import List
from std.pathlib import Path

from serenitymojo.training.minimax_h3.sha256 import (
    minimax_h3_sha256_bytes,
)
from serenitymojo.training.minimax_h3.torch_cpu_randn_avx2_v1 import (
    TORCH_CPU_RANDN_F32_AVX2_V1,
    TorchCpuRandnF32Avx2V1Generator,
    torch_cpu_randn_f32_avx2_v1_from,
    torch_cpu_randn_f32_avx2_v1_raw_le_bytes,
    torch_cpu_randn_f32_avx2_v1_state_bytes,
)


comptime FIXTURE = "serenitymojo/training/parity/fixtures/minimax_h3_torch_cpu_randn_avx2_v1.json"
comptime SCHEMA = "serenity.minimax_h3.torch_cpu_randn_f32_avx2_v1.fixture.v1"
comptime TORCH_VERSION = "2.12.0+cu130"
comptime TORCH_GIT = "7661cd9c6b841b62b7f411aa52ec51f05457263b"
comptime BUILD_SHA = "3e4fc6c11e746aa3905f3b7f7ba4a4f8b32019c6bad21301361a6df63b94b6fb"
comptime DIST_SHA = "dce75f8036a0dbeed823f7b282245673b74782e7af182755fd6babee7708331b"
comptime AVX_MATH_SHA = "e713d6fc64a0e034b13f2cf4b3b9d481fcb4ab0ea2a925154ee1286fe526d07a"
comptime MT_SHA = "8df329422d29c965f1356b511caa26f0f72f74db97b1987509e7bafa94f57467"
comptime TRANSFORM_SHA = "8eff8a994ada7b28c0f0ed8c6564d232160de78b211ba49e49145bc586259260"


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("H3 torch CPU AVX2 randn parity failed: ") + message)


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
    raise Error("H3 torch CPU AVX2 randn parity: malformed lowercase hex")


def _compare_raw_hex(
    label: String, actual: List[UInt8], expected_hex: String,
) raises:
    _check(
        expected_hex.byte_length() == len(actual) * 2,
        label + String(" raw-byte length"),
    )
    var encoded = expected_hex.as_bytes()
    for index in range(len(actual)):
        var expected = (
            _hex_nibble(encoded[index * 2]) << UInt8(4)
        ) | _hex_nibble(encoded[index * 2 + 1])
        if actual[index] != expected:
            raise Error(
                String("H3 torch CPU AVX2 randn parity failed: ") + label
                + String(" raw byte ") + String(index)
                + String(" got=") + String(actual[index])
                + String(" expected=") + String(expected)
            )


def _compare_sha(
    label: String, actual: List[UInt8], expected: String,
) raises:
    var digest = minimax_h3_sha256_bytes(actual.copy())
    _check(
        digest == String("sha256:") + expected,
        label + String(" SHA-256"),
    )


def _check_provenance(document: JSONValue) raises:
    _check(document[String("schema")].as_string() == String(SCHEMA), "schema")
    _check(
        document[String("profile")].as_string()
        == String(TORCH_CPU_RANDN_F32_AVX2_V1),
        "profile",
    )
    var oracle = document[String("oracle")]
    _check(
        oracle[String("torch_version")].as_string() == String(TORCH_VERSION),
        "torch version",
    )
    _check(
        oracle[String("torch_git_version")].as_string() == String(TORCH_GIT),
        "torch git version",
    )
    _check(
        oracle[String("torch_build_config_sha256")].as_string()
        == String(BUILD_SHA),
        "torch build config",
    )
    var headers = oracle[String("installed_header_sha256")]
    _check(
        headers[String("ATen/native/cpu/DistributionTemplates.h")].as_string()
        == String(DIST_SHA),
        "DistributionTemplates header",
    )
    _check(
        headers[String("ATen/native/cpu/avx_mathfun.h")].as_string()
        == String(AVX_MATH_SHA),
        "avx_mathfun header",
    )
    _check(
        headers[String("ATen/core/MT19937RNGEngine.h")].as_string()
        == String(MT_SHA),
        "MT19937 header",
    )
    _check(
        headers[String("ATen/core/TransformationHelper.h")].as_string()
        == String(TRANSFORM_SHA),
        "uniform transform header",
    )
    var platform = document[String("platform_contract")]
    _check(platform[String("arch")].as_string() == String("x86_64"), "arch")
    _check(
        platform[String("byteorder")].as_string() == String("little"),
        "byte order",
    )
    _check(
        platform[String("cpu_dispatch")].as_string() == String("AVX2"),
        "CPU dispatch",
    )
    _check(
        platform[String("dtype")].as_string() == String("float32"),
        "dtype",
    )
    _check(
        platform[String("layout")].as_string()
        == String("contiguous_c_order"),
        "layout",
    )


def _check_rejections() raises:
    var generator = TorchCpuRandnF32Avx2V1Generator(UInt64(42))
    var initial = torch_cpu_randn_f32_avx2_v1_state_bytes(generator)
    var wrong_profile = False
    try:
        _ = torch_cpu_randn_f32_avx2_v1_from(
            generator, [16], String("torch_cpu_randn_f32_scalar")
        )
    except:
        wrong_profile = True
    _check(wrong_profile, "unknown profile rejection")
    _check(
        torch_cpu_randn_f32_avx2_v1_state_bytes(generator) == initial,
        "unknown profile must not consume RNG",
    )

    var too_small = False
    try:
        _ = torch_cpu_randn_f32_avx2_v1_from(generator, [15])
    except:
        too_small = True
    _check(too_small, "N < 16 rejection")
    _check(
        torch_cpu_randn_f32_avx2_v1_state_bytes(generator) == initial,
        "small shape must not consume RNG",
    )

    var zero_dim = False
    try:
        _ = torch_cpu_randn_f32_avx2_v1_from(generator, [24, 1, 0, 32])
    except:
        zero_dim = True
    _check(zero_dim, "zero dimension rejection")

    var empty_shape = False
    try:
        _ = torch_cpu_randn_f32_avx2_v1_from(generator, List[Int]())
    except:
        empty_shape = True
    _check(empty_shape, "empty shape rejection")


def main() raises:
    var document = loads(Path(String(FIXTURE)).read_text())
    _check_provenance(document)
    _check_rejections()
    var cases = document[String("cases")]
    _check(cases.length() == 8, "case inventory")
    var expected_labels = [
        String("threshold_n16"), String("tail_n17"), String("tail_n31"),
        String("aligned_n32"), String("sha_derived_seed_high64"),
        String("sha_derived_seed_low32_alias"),
        String("one_frame_30x52"), String("one_frame_32x32"),
    ]
    var high_output = List[UInt8]()
    var alias_output = List[UInt8]()
    var high_state = List[UInt8]()
    var alias_state = List[UInt8]()
    for index in range(cases.length()):
        var item = cases[index]
        var label = item[String("label")].as_string()
        _check(label == expected_labels[index], "case order " + String(index))
        var shape = _shape(item[String("shape")])
        var numel = item[String("numel")].as_int()
        var calculated = 1
        for dim in shape:
            calculated *= dim
        _check(calculated == numel, label + String(" numel"))
        var consumed = numel if numel % 16 == 0 else numel + 16
        _check(
            item[String("uniform_words_consumed")].as_int() == consumed,
            label + String(" N versus N+16 consumption"),
        )
        var seed = UInt64(item[String("seed_u64")].as_int())
        _check(
            UInt64(item[String("seed_low32")].as_int())
            == (seed & UInt64(0xFFFFFFFF)),
            label + String(" low32 seed"),
        )
        var generator = TorchCpuRandnF32Avx2V1Generator(seed)
        var state_before = torch_cpu_randn_f32_avx2_v1_state_bytes(generator)
        _check(len(state_before) == 5056, label + String(" state-before size"))
        _compare_raw_hex(
            label + String(" state before"), state_before,
            item[String("state_before_raw_hex")].as_string(),
        )
        _compare_sha(
            label + String(" state before"), state_before,
            item[String("state_before_sha256")].as_string(),
        )
        var values = torch_cpu_randn_f32_avx2_v1_from(generator, shape^)
        _check(len(values) == numel, label + String(" output size"))
        var raw = torch_cpu_randn_f32_avx2_v1_raw_le_bytes(values)
        _compare_raw_hex(
            label + String(" output"), raw,
            item[String("output_raw_le_hex")].as_string(),
        )
        _compare_sha(
            label + String(" output"), raw,
            item[String("output_sha256")].as_string(),
        )
        var state_after = torch_cpu_randn_f32_avx2_v1_state_bytes(generator)
        _check(len(state_after) == 5056, label + String(" state-after size"))
        _compare_raw_hex(
            label + String(" state after"), state_after,
            item[String("state_after_raw_hex")].as_string(),
        )
        _compare_sha(
            label + String(" state after"), state_after,
            item[String("state_after_sha256")].as_string(),
        )
        if label == String("sha_derived_seed_high64"):
            high_output = raw.copy()
            high_state = state_after.copy()
        elif label == String("sha_derived_seed_low32_alias"):
            alias_output = raw.copy()
            alias_state = state_after.copy()

    _check(high_output == alias_output, "low32 alias output bytes")
    _check(len(high_state) == 5056 and len(alias_state) == 5056, "alias states")
    var seed_differs = False
    for index in range(8):
        if high_state[index] != alias_state[index]:
            seed_differs = True
    _check(seed_differs, "full seed retained in generator state")
    for index in range(8, 5056):
        _check(
            high_state[index] == alias_state[index],
            "low32 engine-state alias byte " + String(index),
        )
    _check(
        _shape(cases[6][String("shape")]) == [24, 1, 30, 52],
        "30x52 one-frame bucket",
    )
    _check(
        _shape(cases[7][String("shape")]) == [24, 1, 32, 32],
        "32x32 one-frame bucket",
    )
    print("PASS H3 torch_cpu_randn_f32_avx2_v1 exact-byte host parity")
    print("  cases:", cases.length(), "including N/N+16 tails and two H3 buckets")
    print("  output/state bytes: exact; low32 alias: exact")
    print("  portability: pinned x86-64 LE torch 2.12 AVX2 build only")
    print("  real H3 cache/dataset: NOT GENERATED/NOT MUTATED")
