# serenitymojo/models/minimax_h3/parity/minimax_h3_ref2va_parity.mojo
#
# MiniMax-H3 ref2va geometry parity gate.
#
# Proves serenitymojo/models/minimax_h3/packing_ref2va.mojo reproduces the
# reference BIT-EXACTLY. Reference: diffusers PR huggingface/diffusers#14355 at
# head e1b518df, run by scripts/minimax_h3_ref2va_oracle.py.
#
# The case list deliberately includes `mixed_order` and `mixed_reordered`: the
# same three references in a different request order. They have identical
# sequence lengths and identical row counts, and DIFFERENT coordinates —
# reference order advances the shared rotary clock, so a port that ignored order
# would pass every length check and fail here.
#
# `span_sequential` is the other reason this gate exists: this module's temporal
# span is a sequential float64 sum, not the numpy pairwise sum the fl2va
# keyframe anchor uses. Both are swept over 1..129, in two different gates,
# against two different reference functions.
#
# Oracle: python3 scripts/minimax_h3_ref2va_oracle.py
# Run:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_ref2va_parity.mojo \
#     -o output/checks/minimax_h3_ref2va_parity \
#   && output/checks/minimax_h3_ref2va_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.packing_ref2va import (
    MINIMAX_H3_REF_AUDIO,
    MINIMAX_H3_REF_IMAGE,
    MINIMAX_H3_REF_VIDEO,
    MiniMaxH3PreparedReference,
    minimax_h3_build_ref2va_packed_sequence,
    minimax_h3_resample_reference_repeats,
    minimax_h3_resolve_reference_image_size,
    minimax_h3_sample_reference_video_frames,
    minimax_h3_temporal_position_span_sequential,
    minimax_h3_trim_reference_num_frames,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_ref2va/ref2va_ref.safetensors"


def _load_f64(ref st: SafeTensors, name: String) raises -> List[Float64]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F64:
        raise Error(String("_load_f64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float64]()
    var out = List[Float64]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I64:
        raise Error(String("_load_i64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


struct Report(Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def exact_f64(mut self, label: String, got: List[Float64], want: List[Float64]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var bad = 0
        var first = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            print("  ok  ", label, "bit-exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, bad, "of", len(got), "differ; first at", first,
                "got", got[first], "want", want[first],
            )

    def exact_int(mut self, label: String, got: List[Int], want: List[Int]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var bad = 0
        var first = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            print("  ok  ", label, "exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, bad, "of", len(got), "differ; first at", first,
                "got", got[first], "want", want[first],
            )


def _image(h: Int, w: Int) -> MiniMaxH3PreparedReference:
    return MiniMaxH3PreparedReference(MINIMAX_H3_REF_IMAGE, 1, h, w, 0)


def _audio(n: Int) -> MiniMaxH3PreparedReference:
    return MiniMaxH3PreparedReference(MINIMAX_H3_REF_AUDIO, 1, 0, 0, n)


def _video(t: Int, h: Int, w: Int, a: Int) -> MiniMaxH3PreparedReference:
    return MiniMaxH3PreparedReference(MINIMAX_H3_REF_VIDEO, t, h, w, a)


def _tags(count: Int, value: Int) -> List[Int]:
    var out = List[Int]()
    for _ in range(count):
        out.append(value)
    return out^


def _check_case(
    mut report: Report,
    ref st: SafeTensors,
    name: String,
    references: List[MiniMaxH3PreparedReference],
    text_tags: List[Int],
) raises:
    var layout = minimax_h3_build_ref2va_packed_sequence(
        text_tags, references, 7, 8, 10, 37, 2, 2
    )
    report.exact_f64(
        name + ".position_ids", layout.position_ids, _load_f64(st, name + ".position_ids")
    )
    report.exact_int(name + ".token_tags", layout.token_tags, _load_i64(st, name + ".token_tags"))
    report.exact_int(
        name + ".video_indices", layout.video_indices, _load_i64(st, name + ".video_indices")
    )
    report.exact_int(
        name + ".audio_indices", layout.audio_indices, _load_i64(st, name + ".audio_indices")
    )
    report.exact_int(
        name + ".text_indices", layout.text_indices, _load_i64(st, name + ".text_indices")
    )


def main() raises:
    print("MiniMax-H3 ref2va parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var report = Report()

    # 1. the SEQUENTIAL temporal span (not the pairwise one packing.mojo uses)
    print("[1] sequential temporal span")
    var span_n = _load_i64(st, "span_sequential.num_latent_frames")
    var span_got = List[Float64]()
    for i in range(len(span_n)):
        span_got.append(minimax_h3_temporal_position_span_sequential(span_n[i]))
    report.exact_f64("span_sequential.value", span_got, _load_f64(st, "span_sequential.value"))

    # 2. reference image sizing
    print("[2] reference image size")
    var src = _load_i64(st, "ref_image_size.src")
    var size_got = List[Int]()
    for i in range(len(src) // 2):
        var hw = minimax_h3_resolve_reference_image_size(src[2 * i], src[2 * i + 1])
        size_got.append(hw[0])
        size_got.append(hw[1])
    report.exact_int("ref_image_size.hw", size_got, _load_i64(st, "ref_image_size.hw"))

    # 3. reference frame trimming
    print("[3] frame trimming")
    var trims = _load_i64(st, "trim.num_frames")
    var trim_got = List[Int]()
    for i in range(len(trims)):
        trim_got.append(minimax_h3_trim_reference_num_frames(trims[i]))
    report.exact_int("trim.value", trim_got, _load_i64(st, "trim.value"))

    # 4. 24 fps resampling — the oracle dumps the resampled frame ids, which the
    # repeat counts reproduce by expansion
    print("[4] 24 fps resampling")
    var fps_names = [
        String("resample.30_0_40"), String("resample.24_0_40"), String("resample.12_0_20"),
        String("resample.25_0_50"), String("resample.60_0_90"), String("resample.23_976_48"),
    ]
    var fps_values = [Float64(30.0), Float64(24.0), Float64(12.0), Float64(25.0), Float64(60.0), Float64(23.976)]
    var fps_counts = [40, 40, 20, 50, 90, 48]
    for i in range(len(fps_names)):
        var repeats = minimax_h3_resample_reference_repeats(fps_counts[i], fps_values[i])
        var expanded = List[Int]()
        for index in range(len(repeats)):
            for _ in range(repeats[index]):
                expanded.append(index % 251)
        report.exact_int(fps_names[i], expanded, _load_i64(st, fps_names[i]))

    # 5. 2 fps conditioner sampling
    print("[5] conditioner frame sampling")
    var counts = [1, 5, 12, 13, 24, 25, 124, 362]
    for i in range(len(counts)):
        var sampled = minimax_h3_sample_reference_video_frames(counts[i])
        var key = String("sample2fps.") + String(counts[i])
        var frame_ids = List[Int]()
        for j in range(len(sampled.indices)):
            frame_ids.append(sampled.indices[j] % 251)
        report.exact_int(key + ".frames", frame_ids, _load_i64(st, key + ".frames"))
        report.exact_f64(
            key + ".block_timestamps",
            sampled.block_timestamps,
            _load_f64(st, key + ".block_timestamps"),
        )

    # 6. packed layouts
    print("[6] ref2va packed sequences")
    _check_case(report, st, String("one_image"), [_image(8, 10)], _tags(6, 1))
    _check_case(report, st, String("one_audio"), [_audio(40)], _tags(6, 1))
    _check_case(report, st, String("one_video_silent"), [_video(2, 6, 8, 0)], _tags(6, 1))
    _check_case(report, st, String("one_video_sound"), [_video(2, 6, 8, 40)], _tags(6, 1))

    var mixed_tags = List[Int]()
    for _ in range(4):
        mixed_tags.append(0)
    for _ in range(5):
        mixed_tags.append(1)
    _check_case(
        report, st, String("mixed_order"),
        [_image(8, 10), _video(2, 6, 8, 40), _audio(12)], mixed_tags,
    )
    _check_case(
        report, st, String("mixed_reordered"),
        [_audio(12), _video(2, 6, 8, 40), _image(8, 10)], mixed_tags,
    )
    _check_case(report, st, String("long_video_span"), [_video(22, 4, 4, 3)], _tags(5, 1))
    _check_case(
        report, st, String("three_images"),
        [_image(4, 4), _image(6, 4), _image(8, 12)], _tags(3, 1),
    )

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks, all bit-exact")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks differ")
        raise Error("MiniMax-H3 ref2va parity gate failed")
