# serenitymojo/pipeline/parity/minimax_h3_keyframe_layout_probe.mojo
#
# Gates the KEYFRAME (I2VA / FL2VA / L2VA) packed layout — the `[text |
# keyframe conditions | target audio | target video]` sequence with
# `num_condition_video_rows > 0`. Host only: no DeviceContext, no GPU, no
# weights, no oracle file on disk.
#
# ── WHAT IS AND IS NOT GATED HERE ────────────────────────────────────────────
# `models/minimax_h3/packing.mojo` is already gated bit-exact against the
# vendor's own run for the ANCHOR-FREE (pure t2va) layout. This probe covers the
# part that gate never exercises: what changes when `keyframe_anchors` is
# non-empty. It runs `models/dit/minimax_h3_sampling.mojo::minimax_h3_build_
# sampling_geometry` — the RUNTIME twin the pipeline actually calls, which
# deliberately reproduces rather than imports the oracle's math, so it needs its
# own evidence.
#
# ── WHERE THE EXPECTED NUMBERS COME FROM ─────────────────────────────────────
# INDEPENDENTLY derived, not read back from this port:
# scripts/minimax_h3_keyframe_layout_derive.py transcribes packing.py's formulas
# into numpy by hand
# (no diffusers import, no torch) and prints the literals embedded below.
# Comparison is EXACT — integers by equality, float64 coordinates by BIT
# equality, since both sides parse the same shortest-round-trip decimal.
#
# ── THE TINY LAYOUT (4 text rows, 2 latent frames of 4x4 at patch 2x2, 3 audio
# latents) ───────────────────────────────────────────────────────────────────
#   I2VA / L2VA, S = 22:  0..3 text | 4..7 condition | 8..13 audio | 14..21 video
#   FL2VA,       S = 26:  0..3 text | 4..11 condition | 12..17 audio | 18..25 video
# The text tags are `[1, 0, 0, 1]`, not all-text: a keyframe's presentation
# splices a VISION BLOCK into the text run and those rows are tagged VIDEO
# (encoders.py:166), which is the one way a keyframe request's TEXT region
# differs from t2va's. A layout builder that assumed text rows are all tag 1
# would pass every t2va gate and be wrong here.
#
# ── THE "last" ANCHOR IS PAST THE LAST LATENT FRAME, AND THAT IS CORRECT ─────
# `anchor_time = num_text_tokens + temporal_span(n) - 5/3` (packing.py:424),
# while the last latent frame sits at `num_text_tokens + temporal_span(n) -
# spans[n-1]`. Those differ whenever `spans[n-1] != 5/3`, i.e. whenever
# `(n-1) % 5 != 0` — which is EVERY real geometry, since `n = 5k+2`. At the
# product shape (124 frames, 37 latent frames) the derivation prints:
#     span              = 206.66666666666663
#     last latent frame = 199.99999999999991
#     last anchor       = 204.99999999999997   ==  5/3 * 123
# The video clock advances 5/3 rotary units per PIXEL frame, so the anchor lands
# on pixel frame 123 — the last one, 0-indexed. The last keyframe is aligned to
# the end of the PIXEL timeline, not to the last latent frame. Anything that
# "fixes" this to match the last latent frame moves the anchor 5 rotary units.
#
# Run (no GPU, no weights, no reference file):
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_keyframe_layout_probe.mojo \
#     -o /tmp/h3_keyframe_layout_probe -Xlinker -lm \
#   && /tmp/h3_keyframe_layout_probe
#
# The derivation itself (re-run it to re-check the embedded literals):
#   /home/alex/OneTrainer/venv/bin/python scripts/minimax_h3_keyframe_layout_derive.py

from std.collections import List

from serenitymojo.models.dit.minimax_h3_sampling import (
    minimax_h3_build_sampling_geometry,
    minimax_h3_sampling_temporal_grid,
    minimax_h3_sampling_temporal_span,
    MiniMaxH3SamplingGeometry,
)
from serenitymojo.models.minimax_h3.packing import (
    MINIMAX_H3_ANCHOR_FIRST,
    MINIMAX_H3_ANCHOR_LAST,
    minimax_h3_build_row_timesteps,
    minimax_h3_build_packed_sequence,
)
from serenitymojo.pipeline.minimax_h3_keyframe_image import (
    minimax_h3_keyframe_anchors,
)


struct Report(Copyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def ok(mut self, label: String, detail: String):
        self.checks += 1
        print("  ok  ", label, "—", detail)

    def fail(mut self, label: String, detail: String):
        self.checks += 1
        self.failures += 1
        print("  FAIL", label, "—", detail)

    def eq_int(mut self, label: String, got: Int, want: Int):
        if got == want:
            self.ok(label, String(got))
        else:
            self.fail(label, String("got ") + String(got) + ", want " + String(want))


def _text_tags() -> List[Int]:
    """A keyframe presentation's text run: a `"<Picture 1>: "` label (text) then
    a vision block (VIDEO), then prompt text."""
    return [1, 0, 0, 1]


def _expected_i2va_positions() -> List[Float64]:
    return [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        3.0, 0.0, 0.0,
        4.0, 0.0, 0.0,
        4.0, 0.0, 16.0,
        4.0, 16.0, 0.0,
        4.0, 16.0, 16.0,
        4.0, 0.0, 0.0,
        5.0, 0.0, 0.0,
        6.0, 0.0, 0.0,
        4.0, 0.0, 16.0,
        5.0, 0.0, 16.0,
        6.0, 0.0, 16.0,
        4.0, 0.0, 0.0,
        4.0, 0.0, 16.0,
        4.0, 16.0, 0.0,
        4.0, 16.0, 16.0,
        5.666666666666667, 0.0, 0.0,
        5.666666666666667, 0.0, 16.0,
        5.666666666666667, 16.0, 0.0,
        5.666666666666667, 16.0, 16.0,
    ]


def _expected_l2va_positions() -> List[Float64]:
    return [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        3.0, 0.0, 0.0,
        10.666666666666668, 0.0, 0.0,
        10.666666666666668, 0.0, 16.0,
        10.666666666666668, 16.0, 0.0,
        10.666666666666668, 16.0, 16.0,
        4.0, 0.0, 0.0,
        5.0, 0.0, 0.0,
        6.0, 0.0, 0.0,
        4.0, 0.0, 16.0,
        5.0, 0.0, 16.0,
        6.0, 0.0, 16.0,
        4.0, 0.0, 0.0,
        4.0, 0.0, 16.0,
        4.0, 16.0, 0.0,
        4.0, 16.0, 16.0,
        5.666666666666667, 0.0, 0.0,
        5.666666666666667, 0.0, 16.0,
        5.666666666666667, 16.0, 0.0,
        5.666666666666667, 16.0, 16.0,
    ]


def _expected_fl2va_positions() -> List[Float64]:
    return [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        3.0, 0.0, 0.0,
        4.0, 0.0, 0.0,
        4.0, 0.0, 16.0,
        4.0, 16.0, 0.0,
        4.0, 16.0, 16.0,
        10.666666666666668, 0.0, 0.0,
        10.666666666666668, 0.0, 16.0,
        10.666666666666668, 16.0, 0.0,
        10.666666666666668, 16.0, 16.0,
        4.0, 0.0, 0.0,
        5.0, 0.0, 0.0,
        6.0, 0.0, 0.0,
        4.0, 0.0, 16.0,
        5.0, 0.0, 16.0,
        6.0, 0.0, 16.0,
        4.0, 0.0, 0.0,
        4.0, 0.0, 16.0,
        4.0, 16.0, 0.0,
        4.0, 16.0, 16.0,
        5.666666666666667, 0.0, 0.0,
        5.666666666666667, 0.0, 16.0,
        5.666666666666667, 16.0, 0.0,
        5.666666666666667, 16.0, 16.0,
    ]


def _expected_i2va_tags() -> List[Int]:
    return [
        1, 0, 0, 1,
        0, 0, 0, 0,
        2, 2, 2, 2, 2, 2,
        0, 0, 0, 0, 0, 0, 0, 0,
    ]


def _expected_fl2va_tags() -> List[Int]:
    return [
        1, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2,
        0, 0, 0, 0, 0, 0, 0, 0,
    ]


def _expected_i2va_ts_indices() -> List[Int]:
    return [
        1, 1, 1, 1,
        2, 2, 2, 2,
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
    ]


def _expected_fl2va_ts_indices() -> List[Int]:
    return [
        1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2,
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
    ]


def _check_positions(
    mut report: Report, label: String, got: List[Float64], want: List[Float64]
):
    if len(got) != len(want):
        report.fail(
            label,
            String("length ") + String(len(got)) + ", want " + String(len(want)),
        )
        return
    var bad = 0
    var first = -1
    for i in range(len(want)):
        if got[i] != want[i]:
            bad += 1
            if first < 0:
                first = i
    if bad == 0:
        report.ok(
            label,
            String("bit-exact over ") + String(len(want)) + " float64 coordinates",
        )
    else:
        report.fail(
            label,
            String(bad) + " differ; first at flat index " + String(first)
            + " (row " + String(first // 3) + ", axis " + String(first % 3)
            + "): got " + String(got[first]) + ", want " + String(want[first]),
        )


def _check_ints(
    mut report: Report, label: String, got: List[Int], want: List[Int]
):
    if len(got) != len(want):
        report.fail(
            label,
            String("length ") + String(len(got)) + ", want " + String(len(want)),
        )
        return
    var bad = 0
    var first = -1
    for i in range(len(want)):
        if got[i] != want[i]:
            bad += 1
            if first < 0:
                first = i
    if bad == 0:
        report.ok(label, String("exact over ") + String(len(want)) + " entries")
    else:
        report.fail(
            label,
            String(bad) + " differ, first at " + String(first) + ": got "
            + String(got[first]) + ", want " + String(want[first]),
        )


def main() raises:
    print("MiniMax-H3 keyframe unit — I2VA / FL2VA / L2VA packed-layout probe")
    print("")
    var report = Report()

    var text_tags = _text_tags()
    var n_lat = 2
    var lat_h = 4
    var lat_w = 4
    var n_audio = 3

    # ── [1] I2VA: one condition block, anchored at the FIRST latent frame ────
    print("[1] I2VA — anchors [first]")
    var i2va_anchors = minimax_h3_keyframe_anchors(True, False)
    var i2va = minimax_h3_build_sampling_geometry(
        text_tags, n_lat, lat_h, lat_w, n_audio, 2, 2, i2va_anchors
    )
    report.eq_int("sequence_length", i2va.sequence_length, 22)
    report.eq_int("num_condition_video_rows", i2va.num_condition_video_rows, 4)
    report.eq_int("num_condition_audio_rows", i2va.num_condition_audio_rows, 0)
    report.eq_int("video row count (condition + target)", len(i2va.video_indices), 12)
    report.eq_int("audio row count", len(i2va.audio_indices), 6)
    report.eq_int("first condition row", i2va.video_indices[0], 4)
    report.eq_int("first target video row", i2va.video_indices[4], 14)
    _check_positions(report, "I2VA position_ids", i2va.position_ids, _expected_i2va_positions())
    _check_ints(report, "I2VA token_tags", i2va.token_tags, _expected_i2va_tags())

    # ── [2] L2VA: one condition block, anchored at the END ───────────────────
    print("")
    print("[2] L2VA — anchors [last]")
    var l2va_anchors = minimax_h3_keyframe_anchors(False, True)
    var l2va = minimax_h3_build_sampling_geometry(
        text_tags, n_lat, lat_h, lat_w, n_audio, 2, 2, l2va_anchors
    )
    report.eq_int("sequence_length", l2va.sequence_length, 22)
    report.eq_int("num_condition_video_rows", l2va.num_condition_video_rows, 4)
    _check_positions(report, "L2VA position_ids", l2va.position_ids, _expected_l2va_positions())
    # Same row COUNT as I2VA, different anchor TIME — the thing a count-only
    # check would miss entirely.
    if l2va.position_ids[12] != i2va.position_ids[12]:
        report.ok(
            "L2VA anchor time != I2VA anchor time",
            String(l2va.position_ids[12]) + " vs " + String(i2va.position_ids[12]),
        )
    else:
        report.fail(
            "L2VA anchor time != I2VA anchor time",
            "identical — the anchor code was ignored",
        )

    # ── [3] FL2VA: two condition blocks, first then last, in packed order ────
    print("")
    print("[3] FL2VA — anchors [first, last]")
    var fl2va_anchors = minimax_h3_keyframe_anchors(True, True)
    var fl2va = minimax_h3_build_sampling_geometry(
        text_tags, n_lat, lat_h, lat_w, n_audio, 2, 2, fl2va_anchors
    )
    report.eq_int("sequence_length", fl2va.sequence_length, 26)
    report.eq_int("num_condition_video_rows", fl2va.num_condition_video_rows, 8)
    report.eq_int("video row count", len(fl2va.video_indices), 16)
    report.eq_int("first target video row", fl2va.video_indices[8], 18)
    _check_positions(report, "FL2VA position_ids", fl2va.position_ids, _expected_fl2va_positions())
    _check_ints(report, "FL2VA token_tags", fl2va.token_tags, _expected_fl2va_tags())

    # ── [4] the runtime twin agrees with the ALREADY-GATED oracle builder ────
    # `models/minimax_h3/packing.mojo` is gated bit-exact against the vendor for
    # the anchor-free layout; if the two builders agree WITH anchors too, the
    # anchored path inherits that gate rather than resting on this file's
    # literals alone.
    print("")
    print("[4] runtime builder vs the gated oracle builder, with anchors")
    var oracle = minimax_h3_build_packed_sequence(
        text_tags, n_lat, lat_h, lat_w, n_audio, 2, 2, fl2va_anchors
    )
    report.eq_int("oracle sequence_length", oracle.sequence_length, fl2va.sequence_length)
    _check_positions(
        report, "oracle vs runtime position_ids", oracle.position_ids, fl2va.position_ids
    )
    _check_ints(report, "oracle vs runtime token_tags", oracle.token_tags, fl2va.token_tags)
    _check_ints(
        report, "oracle vs runtime video_indices", oracle.video_indices, fl2va.video_indices
    )

    # ── [5] row timesteps: condition rows pinned at max(video_t, 0.999) ──────
    print("")
    print("[5] row timesteps — condition_video = max(video_t, 0.999)")
    var ts = minimax_h3_build_row_timesteps(
        oracle, Float32(0.75), Float32(0.60), Float32(0.999), Float32(1.0)
    )
    report.eq_int("distinct timesteps @ (v=0.75, a=0.60)", len(ts.values), 3)
    if len(ts.values) == 3:
        var want = [Float32(0.60), Float32(0.75), Float32(0.999)]
        var bad = 0
        for i in range(3):
            if ts.values[i] != want[i]:
                bad += 1
        if bad == 0:
            report.ok(
                "timestep values (sorted)",
                String(ts.values[0]) + ", " + String(ts.values[1]) + ", "
                + String(ts.values[2]),
            )
        else:
            report.fail("timestep values (sorted)", String(bad) + " differ")
    _check_ints(report, "per-row timestep index", ts.indices, _expected_fl2va_ts_indices())

    var i2va_ts = minimax_h3_build_row_timesteps(
        minimax_h3_build_packed_sequence(
            text_tags, n_lat, lat_h, lat_w, n_audio, 2, 2, i2va_anchors
        ),
        Float32(0.75), Float32(0.60), Float32(0.999), Float32(1.0),
    )
    _check_ints(report, "I2VA per-row timestep index", i2va_ts.indices, _expected_i2va_ts_indices())

    # At video_t = 1.0 the condition timestep is max(1.0, 0.999) = 1.0 and the
    # table COLLAPSES to two values. Anything sizing an AdaLN modulation cache
    # at a fixed three rows per step for a keyframe run is wrong.
    var collapsed = minimax_h3_build_row_timesteps(
        oracle, Float32(1.0), Float32(0.5), Float32(1.0), Float32(1.0)
    )
    report.eq_int("collapse at video_t = 1.0", len(collapsed.values), 2)

    # ── [6] the "last" anchor at the real product geometry ───────────────────
    print("")
    print("[6] the 'last' anchor lands on the last PIXEL frame (124 frames)")
    var real_n = 37  # (124 - 5) / 17 * 5 + 2
    var span = minimax_h3_sampling_temporal_span(real_n)
    var grid = minimax_h3_sampling_temporal_grid(real_n, Float64(0.0))
    var last_frame = grid[real_n - 1]
    var last_anchor = span - (Float64(5.0) / Float64(3.0))
    var pixel_123 = Float64(5.0) / Float64(3.0) * Float64(123.0)
    if last_anchor != last_frame:
        report.ok(
            "anchor is past the last latent frame",
            String("anchor ") + String(last_anchor) + " vs last frame "
            + String(last_frame),
        )
    else:
        report.fail(
            "anchor is past the last latent frame",
            "they are equal — the span/grid distinction was lost",
        )
    var delta = last_anchor - pixel_123
    if delta < 0.0:
        delta = -delta
    if delta < 1.0e-9:
        report.ok(
            "anchor == 5/3 * 123 (the last pixel frame, 0-indexed)",
            String(last_anchor) + " vs " + String(pixel_123),
        )
    else:
        report.fail(
            "anchor == 5/3 * 123",
            String(last_anchor) + " vs " + String(pixel_123),
        )

    # ── [7] a bad anchor code must fail loud ─────────────────────────────────
    var rejected = False
    try:
        var bad_anchors = List[Int]()
        bad_anchors.append(7)
        var bad = minimax_h3_build_sampling_geometry(
            text_tags, n_lat, lat_h, lat_w, n_audio, 2, 2, bad_anchors
        )
        _ = bad.sequence_length
    except:
        rejected = True
    if rejected:
        report.ok("unknown anchor code", "rejected")
    else:
        report.fail("unknown anchor code", "silently built a layout")
    _ = MINIMAX_H3_ANCHOR_FIRST
    _ = MINIMAX_H3_ANCHOR_LAST

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_keyframe_layout probe FAILED")
