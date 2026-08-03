# serenitymojo/models/minimax_h3/parity/minimax_h3_packing_parity.mojo
#
# MiniMax-H3 packed-sequence geometry parity gate.
#
# Proves serenitymojo/models/minimax_h3/packing.mojo reproduces the reference
# geometry BIT-EXACTLY. The reference is the diffusers PR
# huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_packing_oracle.py — the oracle calls the reference's own
# functions and dumps their outputs; it reimplements nothing.
#
# The bar is bit-exactness, not a cosine. Every value here is integer or
# power-of-two arithmetic plus one division, and the reference's docstrings call
# out two places where float summation order is observable (numpy
# `linspace(endpoint=False)` and numpy pairwise summation of the temporal span).
# A last-ulp miss is a real finding: it would mean our rotary coordinates drift
# from the checkpoint's, and rotary coordinates are what aligns audio to video.
#
# Oracle: python3 scripts/minimax_h3_packing_oracle.py
# Run:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_packing_parity.mojo \
#     -o output/checks/minimax_h3_packing_parity \
#   && output/checks/minimax_h3_packing_parity

from std.collections import List
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.packing import (
    MINIMAX_H3_ANCHOR_FIRST,
    MINIMAX_H3_ANCHOR_LAST,
    MiniMaxH3PackedSequence,
    minimax_h3_align_num_frames,
    minimax_h3_audio_latent_num_frames,
    minimax_h3_build_packed_sequence,
    minimax_h3_build_row_timesteps,
    minimax_h3_resolve_canvas_size,
    minimax_h3_spatial_position_grid,
    minimax_h3_temporal_position_grid,
    minimax_h3_temporal_position_span,
    minimax_h3_video_latent_num_frames,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_packing/packing_ref.safetensors"


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


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
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
    """Running pass/fail tally. Every check reports its own worst deviation."""

    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def exact_f64(mut self, label: String, got: List[Float64], want: List[Float64]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print(
                "  FAIL", label, "length", len(got), "!=", len(want)
            )
            return
        var mismatches = 0
        var worst = Float64(0.0)
        var first_index = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                mismatches += 1
                if first_index < 0:
                    first_index = i
                var diff = got[i] - want[i]
                var mag = -diff if diff < 0.0 else diff
                if mag > worst:
                    worst = mag
        if mismatches == 0:
            print("  ok  ", label, "bit-exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL",
                label,
                mismatches,
                "of",
                len(got),
                "differ; first at",
                first_index,
                "got",
                got[first_index],
                "want",
                want[first_index],
                "max_abs",
                worst,
            )

    def exact_f32(mut self, label: String, got: List[Float32], want: List[Float32]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var mismatches = 0
        for i in range(len(got)):
            if got[i] != want[i]:
                mismatches += 1
        if mismatches == 0:
            print("  ok  ", label, "bit-exact over", len(got), "values")
        else:
            self.failures += 1
            print("  FAIL", label, mismatches, "of", len(got), "differ")

    def exact_int(mut self, label: String, got: List[Int], want: List[Int]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var mismatches = 0
        var first_index = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                mismatches += 1
                if first_index < 0:
                    first_index = i
        if mismatches == 0:
            print("  ok  ", label, "exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL",
                label,
                mismatches,
                "of",
                len(got),
                "differ; first at",
                first_index,
                "got",
                got[first_index],
                "want",
                want[first_index],
            )


def _case_layout(
    height: Int,
    width: Int,
    requested_frames: Int,
    anchors: List[Int],
    text_tags: List[Int],
) raises -> MiniMaxH3PackedSequence:
    var aligned = minimax_h3_align_num_frames(requested_frames)
    return minimax_h3_build_packed_sequence(
        text_tags,
        minimax_h3_video_latent_num_frames(aligned),
        height // 16,
        width // 16,
        minimax_h3_audio_latent_num_frames(aligned),
        2,
        2,
        anchors,
    )


def _check_case(
    mut report: Report,
    ref st: SafeTensors,
    name: String,
    height: Int,
    width: Int,
    requested_frames: Int,
    anchors: List[Int],
    text_tags: List[Int],
) raises:
    var layout = _case_layout(height, width, requested_frames, anchors, text_tags)
    report.exact_f64(
        name + ".position_ids", layout.position_ids, _load_f64(st, name + ".position_ids")
    )
    report.exact_int(
        name + ".token_tags", layout.token_tags, _load_i64(st, name + ".token_tags")
    )
    report.exact_int(
        name + ".video_indices", layout.video_indices, _load_i64(st, name + ".video_indices")
    )
    report.exact_int(
        name + ".audio_indices", layout.audio_indices, _load_i64(st, name + ".audio_indices")
    )
    report.exact_int(
        name + ".text_indices", layout.text_indices, _load_i64(st, name + ".text_indices")
    )

    # Row timesteps are dumped for the three tiny layouts only.
    if name.startswith("tiny"):
        var t_names = [String("t_mid"), String("t_collapse"), String("t_start")]
        var t_video = [Float32(0.5), Float32(0.999), Float32(1.0) / Float32(1000.0)]
        var t_audio = [Float32(0.8), Float32(0.999), Float32(0.0039)]
        var t_cond_v = [Float32(0.999), Float32(0.999), Float32(0.999)]
        var t_cond_a = [Float32(1.0), Float32(1.0), Float32(1.0)]
        for i in range(len(t_names)):
            var rows = minimax_h3_build_row_timesteps(
                layout, t_video[i], t_audio[i], t_cond_v[i], t_cond_a[i]
            )
            var key = name + "." + t_names[i]
            report.exact_f32(key + ".values", rows.values, _load_f32(st, key + ".values"))
            report.exact_int(key + ".indices", rows.indices, _load_i64(st, key + ".indices"))


def main() raises:
    print("MiniMax-H3 packing parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var report = Report()

    # 1. canvas resolution
    print("[1] canvas")
    var ratio_w = [16, 9, 1, 4, 1, 3, 2, 21, 1344]
    var ratio_h = [9, 16, 1, 1, 4, 2, 3, 9, 768]
    var canvas_ref = _load_i64(st, "canvas.hw")
    var canvas_got = List[Int]()
    for i in range(len(ratio_w)):
        var size = minimax_h3_resolve_canvas_size(Float64(ratio_w[i]), Float64(ratio_h[i]))
        canvas_got.append(size.height)
        canvas_got.append(size.width)
    report.exact_int("canvas.hw", canvas_got, canvas_ref)

    # 2. frame-count algebra
    print("[2] frame counts")
    var requested = _load_i64(st, "frames.requested")
    var aligned_got = List[Int]()
    var video_got = List[Int]()
    var audio_got = List[Int]()
    for i in range(len(requested)):
        var aligned = minimax_h3_align_num_frames(requested[i])
        aligned_got.append(aligned)
        video_got.append(minimax_h3_video_latent_num_frames(aligned))
        audio_got.append(minimax_h3_audio_latent_num_frames(aligned))
    report.exact_int("frames.aligned", aligned_got, _load_i64(st, "frames.aligned"))
    report.exact_int("frames.video_latents", video_got, _load_i64(st, "frames.video_latents"))
    report.exact_int("frames.audio_latents", audio_got, _load_i64(st, "frames.audio_latents"))

    # 3. spatial grids
    print("[3] spatial grids")
    var grid_h = [128, 96, 768, 64]
    var grid_w = [160, 96, 1344, 64]
    for i in range(len(grid_h)):
        var latent_h = grid_h[i] // 16
        var latent_w = grid_w[i] // 16
        # `sqrt`, not `** 0.5`: pow is not correctly rounded, and the reference
        # takes an IEEE square root. The two disagree in the 11th digit, which
        # this gate caught on its first run.
        var sqrt_area = sqrt(Float64(latent_h * latent_w))
        var key = String("grid.") + String(grid_h[i]) + "x" + String(grid_w[i])
        report.exact_f64(
            key + ".height",
            minimax_h3_spatial_position_grid(latent_h, 2, sqrt_area),
            _load_f64(st, key + ".height"),
        )
        report.exact_f64(
            key + ".width",
            minimax_h3_spatial_position_grid(latent_w, 2, sqrt_area),
            _load_f64(st, key + ".width"),
        )

    # 4. temporal span (numpy pairwise summation) and grid
    print("[4] temporal span + grid")
    var span_n = _load_i64(st, "span.num_latent_frames")
    var span_got = List[Float64]()
    for i in range(len(span_n)):
        span_got.append(minimax_h3_temporal_position_span(span_n[i]))
    report.exact_f64("span.value", span_got, _load_f64(st, "span.value"))
    report.exact_f64(
        "temporal_grid.107",
        minimax_h3_temporal_position_grid(107, Float64(17.0)),
        _load_f64(st, "temporal_grid.107"),
    )
    report.exact_f64(
        "temporal_grid.2",
        minimax_h3_temporal_position_grid(2, Float64(0.0)),
        _load_f64(st, "temporal_grid.2"),
    )

    # 5. packed sequences
    print("[5] packed sequences")
    var no_anchor = List[Int]()
    var first_anchor = [MINIMAX_H3_ANCHOR_FIRST]
    var both_anchors = [MINIMAX_H3_ANCHOR_FIRST, MINIMAX_H3_ANCHOR_LAST]
    var last_anchor = [MINIMAX_H3_ANCHOR_LAST]

    var tags7 = List[Int]()
    for _ in range(7):
        tags7.append(1)
    var tags_first = List[Int]()
    for _ in range(4):
        tags_first.append(0)
    for _ in range(5):
        tags_first.append(1)
    var tags_both = List[Int]()
    for _ in range(8):
        tags_both.append(0)
    for _ in range(5):
        tags_both.append(1)
    var tags3 = List[Int]()
    for _ in range(3):
        tags3.append(1)
    var tags11 = List[Int]()
    for _ in range(11):
        tags11.append(1)
    var tags17 = List[Int]()
    for _ in range(17):
        tags17.append(1)

    _check_case(report, st, String("tiny_t2va"), 128, 160, 22, no_anchor, tags7)
    _check_case(report, st, String("tiny_fl2va_first"), 128, 160, 22, first_anchor, tags_first)
    _check_case(report, st, String("tiny_fl2va_both"), 128, 160, 22, both_anchors, tags_both)
    _check_case(report, st, String("min_frames"), 96, 96, 5, no_anchor, tags3)
    _check_case(report, st, String("span_pairwise"), 64, 64, 260, last_anchor, tags11)
    _check_case(report, st, String("product_1344x768"), 768, 1344, 124, no_anchor, tags17)

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks, all bit-exact")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks differ")
        raise Error("MiniMax-H3 packing parity gate failed")
