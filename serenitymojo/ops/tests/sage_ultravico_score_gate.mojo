# serenitymojo/ops/tests/sage_ultravico_score_gate.mojo
#
# Analytical gate for the opt-in packed-H3 UltraViCo score rule.  The cases
# deliberately cover every branch and use independently written expectations.

from std.math import abs

from serenitymojo.ops.sage_attention_int8 import (
    _sage_ultravico_decay_score,
    _sage_ultravico_decay_score_fast,
    _sage_ultravico_decay_score_tiled,
    _sage_ultravico_uniform_tile_factor,
)


def check(name: String, got: Float32, expected: Float32) raises:
    var error = abs(got - expected)
    print(name, " got=", got, " expected=", expected, " abs=", error)
    if error > 1.0e-6:
        raise Error(String("UltraViCo score gate failed: ") + name)


def adjusted(
    score: Float32,
    query_row: Int,
    key_row: Int,
    video_start: Int = 100,
    rows_per_frame: Int = 4,
    window_frames: Int = 3,
    period_frames: Int = 10,
    radius_frames: Int = 1,
) -> Float32:
    return _sage_ultravico_decay_score(
        score, query_row, key_row, video_start, rows_per_frame,
        window_frames, period_frames, radius_frames, 0.9, 0.6,
    )


def main() raises:
    check(
        "disabled",
        _sage_ultravico_decay_score(
            2.0, 100, 180, -1, 4, 3, 10, 1, 0.9, 0.6
        ),
        2.0,
    )
    check("negative score", adjusted(-2.0, 100, 180), -2.0)
    check("text query", adjusted(2.0, 99, 180), 2.0)
    check("text key", adjusted(2.0, 180, 99), 2.0)
    check("inside window", adjusted(2.0, 100, 112), 2.0)
    check("outside general", adjusted(2.0, 100, 124), 1.8)
    check("harmonic exact", adjusted(2.0, 100, 140), 1.2)
    check("harmonic lower edge", adjusted(2.0, 100, 136), 1.2)
    check("harmonic upper edge", adjusted(2.0, 100, 144), 1.2)
    check("next harmonic", adjusted(2.0, 100, 180), 1.2)
    check("symmetric distance", adjusted(2.0, 180, 100), 1.2)
    check("outside harmonic radius", adjusted(2.0, 100, 148), 1.8)
    # Exhaust every valid score in a geometry containing prefix-only,
    # boundary, local-window, alpha, and repeated-harmonic tiles.  A uniform
    # tile result must be bit-identical to the scalar source rule; mixed tiles
    # must fall back to that same rule.
    var sequence = 640
    var video_start = 37
    var uniform_tiles = 0
    var fallback_tiles = 0
    var mismatches = 0
    for q_start in range(0, sequence, 16):
        for key_start in range(0, sequence, 64):
            var tile_factor = _sage_ultravico_uniform_tile_factor(
                q_start, key_start, sequence, video_start,
                12, 40, 4, 0.9, 0.6,
            )
            if tile_factor < 0.0:
                fallback_tiles += 1
            else:
                uniform_tiles += 1
            var q_stop = q_start + 16
            if q_stop > sequence:
                q_stop = sequence
            var key_stop = key_start + 64
            if key_stop > sequence:
                key_stop = sequence
            for q in range(q_start, q_stop):
                for k in range(key_start, key_stop):
                    var scalar = _sage_ultravico_decay_score(
                        2.0, q, k, video_start, 4, 3, 10, 1,
                        0.9, 0.6,
                    )
                    var tiled = _sage_ultravico_decay_score_tiled(
                        2.0, q, k, tile_factor, video_start, 4, 3, 10, 1,
                        0.9, 0.6,
                    )
                    if scalar != tiled:
                        mismatches += 1
    print(
        "tile classifier uniform=", uniform_tiles,
        " fallback=", fallback_tiles, " mismatches=", mismatches,
    )
    if uniform_tiles == 0 or fallback_tiles == 0 or mismatches != 0:
        raise Error("UltraViCo exact tile-classifier gate failed")
    # The production kernel uses a reciprocal/round form to avoid dynamic
    # integer remainder in its register-heavy inner loop. Exhaust the full
    # admitted long-video distance envelope against the source modulo rule.
    var fast_mismatches = 0
    var production_video_start = 7829
    var production_rows_per_frame = 160
    var production_window_frames = 53
    var production_period_frames = 194
    var production_radius_frames = 16
    var production_window_rows = (
        production_rows_per_frame * production_window_frames
    )
    var production_period_rows = (
        production_rows_per_frame * production_period_frames
    )
    var production_radius_rows = (
        production_rows_per_frame * production_radius_frames
    )
    var production_period_inv = (
        Float32(1.0) / Float32(production_period_rows)
    )
    for distance in range(160001):
        var scalar = _sage_ultravico_decay_score(
            2.0, production_video_start,
            production_video_start + distance, production_video_start,
            production_rows_per_frame, production_window_frames,
            production_period_frames, production_radius_frames, 0.9, 0.6,
        )
        var fast = _sage_ultravico_decay_score_fast(
            2.0, production_video_start,
            production_video_start + distance, production_video_start,
            production_window_rows, production_period_rows,
            production_radius_rows, production_period_inv, 0.9, 0.6,
        )
        if scalar != fast:
            fast_mismatches += 1
    print("fast-rule envelope mismatches=", fast_mismatches)
    if fast_mismatches != 0:
        raise Error("UltraViCo reciprocal score rule differs from source")
    print("PASS: packed-H3 UltraViCo score-decay branches")
