# serenitymojo/models/dit/parity/minimax_h3_ref_geometry_probe.mojo
#
# UNIT B probe — MiniMax-H3 ref2va ref-pack geometry.
#
# Gates `serenitymojo/models/dit/minimax_h3_ref_geometry.mojo` host-side: no
# DeviceContext, no GPU, no weights, no oracle file on disk.
#
# ── WHAT IS AND IS NOT GATED HERE ────────────────────────────────────────────
# The ref2va rotary math itself is ALREADY gated bit-exact, elsewhere, against
# the vendor's own run: `models/minimax_h3/parity/minimax_h3_ref2va_parity.mojo`
# vs `scripts/minimax_h3_ref2va_oracle.py`. This probe does not duplicate that.
# It gates the layer UNDER test here — the media->latent resolution, the audio
# latent count, the row ranges, and the four-timestep assignment — plus ONE
# end-to-end tiny layout that proves the composition wires those pieces into the
# gated builder correctly.
#
# ── WHERE THE EXPECTED NUMBERS COME FROM ─────────────────────────────────────
# INDEPENDENTLY derived, not read back from this port. The tiny layout's 96
# position coordinates, 32 token tags and timestep table were computed by a
# standalone numpy transcription of the vendor's formulas
# (`_spatial_position_grid` packing.py:331, `_temporal_position_grid` :344,
# `_temporal_position_span` packing_ref2va.py:387 — the SEQUENTIAL sum,
# `_fill_audio_positions` :412, `build_ref2va_packed_sequence` :435,
# `build_row_timesteps` packing.py:469, and the four timestep VALUES from
# before_denoise.py:413-417), with no diffusers and no torch import, and are
# embedded below as literals. Comparison is EXACT: integers by equality,
# float64 coordinates by bit equality, since both sides parse the same
# shortest-round-trip decimal.
#
# THE TINY LAYOUT (4 text rows, S = 32):
#   rows  0.. 3  text
#   rows  4.. 9  reference soundtrack   (3 audio latents x 2 channels)
#   rows 10..17  reference video        (2 latent frames x 4 rows/frame)
#   rows 18..23  target audio           (3 audio latents x 2 channels)
#   rows 24..31  target video           (2 latent frames x 4 rows/frame)
# One VIDEO reference carrying a soundtrack is the case worth hand-tracing: it
# is the only reference kind that packs BOTH modalities, its audio rows go
# BEFORE its video rows, the two share a rotary origin, and it advances the
# shared clock by max(audio_latents, sequential_span(latent_frames)) — four
# distinct ways to get the layout wrong that an image-only case would not catch.
#
# Run (no GPU, no weights):
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/dit/parity/minimax_h3_ref_geometry_probe.mojo \
#     -o <scratch>/minimax_h3_ref_geometry_probe \
#   && <scratch>/minimax_h3_ref_geometry_probe

from std.collections import List

from serenitymojo.models.minimax_h3.audio_encoder import (
    MiniMaxH3AudioEncoderConfig,
)
from serenitymojo.models.minimax_h3.packing_ref2va import (
    MINIMAX_H3_REF_AUDIO,
    MINIMAX_H3_REF_IMAGE,
    MINIMAX_H3_REF_VIDEO,
    MiniMaxH3PreparedReference,
)
from serenitymojo.models.dit.minimax_h3_ref_geometry import (
    MINIMAX_H3_AUDIO_HOP,
    MINIMAX_H3_CONDITION_AUDIO_TIMESTEP,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MiniMaxH3ReferenceMedia,
    minimax_h3_build_ref2va_plan,
    minimax_h3_ref2va_row_timesteps,
    minimax_h3_reference_audio_latents,
    minimax_h3_resolve_reference,
    minimax_h3_resolve_references,
    minimax_h3_target_audio_latents,
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


def _expected_positions() -> List[Float64]:
    """Flat [32 rows x (t, h, w)] — independently derived, see file header."""
    return [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        3.0, 0.0, 0.0,
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
        12.333333333333334, 0.0, 0.0,
        13.333333333333334, 0.0, 0.0,
        14.333333333333334, 0.0, 0.0,
        12.333333333333334, 0.0, 16.0,
        13.333333333333334, 0.0, 16.0,
        14.333333333333334, 0.0, 16.0,
        12.333333333333334, 0.0, 0.0,
        12.333333333333334, 0.0, 16.0,
        12.333333333333334, 16.0, 0.0,
        12.333333333333334, 16.0, 16.0,
        14.0, 0.0, 0.0,
        14.0, 0.0, 16.0,
        14.0, 16.0, 0.0,
        14.0, 16.0, 16.0,
    ]


def _expected_tags() -> List[Int]:
    return [
        1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2,
        0, 0, 0, 0, 0, 0, 0, 0,
    ]


def _expected_ts_indices() -> List[Int]:
    return [
        1, 1, 1, 1,
        3, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2, 2, 2,
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
    ]


def main() raises:
    print("MiniMax-H3 ref2va UNIT B probe — ref-pack geometry")
    print("")
    var report = Report()

    # ── [1] audio latent count: CEILING, not rounding ────────────────────────
    print("[1] reference audio latents = ceil(samples / 800)")
    report.eq_int("0 samples", minimax_h3_reference_audio_latents(0), 0)
    report.eq_int("1 sample", minimax_h3_reference_audio_latents(1), 1)
    report.eq_int("799 samples", minimax_h3_reference_audio_latents(799), 1)
    report.eq_int("800 samples", minimax_h3_reference_audio_latents(800), 1)
    report.eq_int("801 samples", minimax_h3_reference_audio_latents(801), 2)
    report.eq_int("32000 (1 s @32k)", minimax_h3_reference_audio_latents(32000), 40)
    report.eq_int("32001 samples", minimax_h3_reference_audio_latents(32001), 41)

    # The hop is not a magic number: cross-check it against the audio encoder's
    # own config, whose rates are the ones the real-weights gate runs with.
    var enc_rates = [2, 4, 4, 5, 5]
    var enc_cfg = MiniMaxH3AudioEncoderConfig(
        1536, enc_rates^, 2048, 32, 8, Float32(1.0e-5)
    )
    report.eq_int(
        "hop vs audio_encoder.hop_length()",
        MINIMAX_H3_AUDIO_HOP,
        enc_cfg.hop_length(),
    )

    # ── [2] media geometry -> latent geometry ────────────────────────────────
    print("")
    print("[2] reference resolution (media -> latent)")
    var image_ref = minimax_h3_resolve_reference(
        MiniMaxH3ReferenceMedia(MINIMAX_H3_REF_IMAGE, 500, 1000, 0, 0), 39
    )
    report.eq_int("image 1000x500 latent_frames", image_ref.num_latent_frames, 1)
    report.eq_int("image 1000x500 latent_h", image_ref.latent_height, 128)
    report.eq_int("image 1000x500 latent_w", image_ref.latent_width, 256)
    report.eq_int("image 1000x500 audio latents", image_ref.num_audio_latents, 0)
    report.eq_int("image 1000x500 video rows", image_ref.num_video_rows(), 128 // 2 * (256 // 2))

    # 1920x1080 exceeds the 768*1344 area cap, so the canvas is scaled down and
    # rounded to 768x1344 — the reference's OWN canvas, not the target's.
    var video_ref = minimax_h3_resolve_reference(
        MiniMaxH3ReferenceMedia(MINIMAX_H3_REF_VIDEO, 1080, 1920, 100, 32000), 39
    )
    report.eq_int("video 1920x1080 latent_frames", video_ref.num_latent_frames, 12)
    report.eq_int("video 1920x1080 latent_h", video_ref.latent_height, 48)
    report.eq_int("video 1920x1080 latent_w", video_ref.latent_width, 84)
    report.eq_int("video 1920x1080 audio latents", video_ref.num_audio_latents, 40)

    # A portrait reference keeps its own orientation.
    var portrait = minimax_h3_resolve_reference(
        MiniMaxH3ReferenceMedia(MINIMAX_H3_REF_VIDEO, 1920, 1080, 200, 0), 56
    )
    report.eq_int("video 1080x1920 latent_frames", portrait.num_latent_frames, 17)
    report.eq_int("video 1080x1920 latent_h", portrait.latent_height, 84)
    report.eq_int("video 1080x1920 latent_w", portrait.latent_width, 48)

    var audio_ref = minimax_h3_resolve_reference(
        MiniMaxH3ReferenceMedia(MINIMAX_H3_REF_AUDIO, 0, 0, 0, 48000), 39
    )
    report.eq_int("audio ref latents", audio_ref.num_audio_latents, 60)
    report.eq_int("audio ref video rows", audio_ref.num_video_rows(), 0)

    # A reference under 22 frames is unrepresentable — the vendor's own trim
    # floor asks for more frames than exist. Must fail loudly, not mispredict.
    var short_rejected = False
    try:
        var bad = minimax_h3_resolve_reference(
            MiniMaxH3ReferenceMedia(MINIMAX_H3_REF_VIDEO, 640, 640, 20, 0), 39
        )
        _ = bad.num_latent_frames
    except:
        short_rejected = True
    if short_rejected:
        report.ok("20-frame video reference", "rejected (needs 22)")
    else:
        report.fail("20-frame video reference", "silently predicted a geometry")

    # ── [3] target audio latents ─────────────────────────────────────────────
    print("")
    print("[3] target audio latents = round_half_even(frames / 24 * 40)")
    report.eq_int("5 frames", minimax_h3_target_audio_latents(5), 8)
    report.eq_int("22 frames", minimax_h3_target_audio_latents(22), 37)
    report.eq_int("39 frames", minimax_h3_target_audio_latents(39), 65)
    report.eq_int("56 frames", minimax_h3_target_audio_latents(56), 93)

    # ── [4] the hand-traced tiny layout ──────────────────────────────────────
    print("")
    print("[4] tiny layout: 4 text | ref(3 alat + 2 lframes @4x4) | target(3 alat + 2 lframes @4x4)")
    var text_tags = [1, 1, 1, 1]
    var references = List[MiniMaxH3PreparedReference]()
    references.append(MiniMaxH3PreparedReference(MINIMAX_H3_REF_VIDEO, 2, 4, 4, 3))
    var plan = minimax_h3_build_ref2va_plan(text_tags, references, 2, 4, 4, 3, 2, 2)

    report.eq_int("sequence_length", plan.sequence_length(), 32)
    report.eq_int("num_text_tokens", plan.num_text_tokens, 4)
    report.eq_int("condition video rows", plan.num_condition_video_rows, 8)
    report.eq_int("condition audio rows", plan.num_condition_audio_rows, 6)
    report.eq_int("target audio rows", plan.num_target_audio_rows, 6)
    report.eq_int("target video rows", plan.num_target_video_rows, 8)
    report.eq_int("target_audio_start", plan.target_audio_start, 18)
    report.eq_int("target_video_start", plan.target_video_start, 24)
    report.eq_int("reference_start", plan.reference_start(), 4)
    report.eq_int("reference_end", plan.reference_end(), 18)

    var want_pos = _expected_positions()
    ref got_pos = plan.layout.position_ids
    if len(got_pos) != len(want_pos):
        report.fail(
            "position_ids length",
            String("got ") + String(len(got_pos)) + ", want " + String(len(want_pos)),
        )
    else:
        var bad = 0
        var first = -1
        for i in range(len(want_pos)):
            if got_pos[i] != want_pos[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            report.ok(
                "position_ids (t,h,w)",
                String("bit-exact over ") + String(len(want_pos)) + " float64 coordinates",
            )
        else:
            report.fail(
                "position_ids (t,h,w)",
                String(bad) + " differ; first at flat index " + String(first)
                + " (row " + String(first // 3) + ", axis " + String(first % 3)
                + "): got " + String(got_pos[first]) + ", want " + String(want_pos[first]),
            )

    var want_tags = _expected_tags()
    ref got_tags = plan.layout.token_tags
    var tag_bad = 0
    var tag_first = -1
    if len(got_tags) != len(want_tags):
        report.fail("token_tags length", String(len(got_tags)))
    else:
        for i in range(len(want_tags)):
            if got_tags[i] != want_tags[i]:
                tag_bad += 1
                if tag_first < 0:
                    tag_first = i
        if tag_bad == 0:
            report.ok("token_tags", String("exact over ") + String(len(want_tags)) + " rows")
        else:
            report.fail(
                "token_tags",
                String(tag_bad) + " differ, first at row " + String(tag_first),
            )

    # The reference's audio rows must come BEFORE its own video rows.
    var ref_audio_first = plan.layout.audio_indices[0]
    var ref_video_first = plan.layout.video_indices[0]
    if ref_audio_first == 4 and ref_video_first == 10:
        report.ok(
            "soundtrack packs before its video",
            String("audio_indices[0]=") + String(ref_audio_first)
            + ", video_indices[0]=" + String(ref_video_first),
        )
    else:
        report.fail(
            "soundtrack packs before its video",
            String("audio_indices[0]=") + String(ref_audio_first)
            + ", video_indices[0]=" + String(ref_video_first) + ", want 4 and 10",
        )

    # ── [5] the four row timesteps ───────────────────────────────────────────
    print("")
    print("[5] row timesteps: video=0.75, audio=0.60 -> cond_video=0.999, cond_audio=1.0")
    var ts = minimax_h3_ref2va_row_timesteps(
        plan.layout, Float32(0.75), Float32(0.60)
    )
    report.eq_int("distinct timestep count", len(ts.values), 4)
    var want_values = [Float32(0.60), Float32(0.75), Float32(0.999), Float32(1.0)]
    var value_bad = 0
    if len(ts.values) == 4:
        for i in range(4):
            if ts.values[i] != want_values[i]:
                value_bad += 1
        if value_bad == 0:
            report.ok(
                "timestep values (sorted)",
                String(ts.values[0]) + ", " + String(ts.values[1]) + ", "
                + String(ts.values[2]) + ", " + String(ts.values[3]),
            )
        else:
            report.fail("timestep values (sorted)", String(value_bad) + " differ")

    var want_ts_idx = _expected_ts_indices()
    var idx_bad = 0
    var idx_first = -1
    if len(ts.indices) != len(want_ts_idx):
        report.fail("timestep index length", String(len(ts.indices)))
    else:
        for i in range(len(want_ts_idx)):
            if ts.indices[i] != want_ts_idx[i]:
                idx_bad += 1
                if idx_first < 0:
                    idx_first = i
        if idx_bad == 0:
            report.ok(
                "per-row timestep index",
                String("exact over ") + String(len(want_ts_idx)) + " rows",
            )
        else:
            report.fail(
                "per-row timestep index",
                String(idx_bad) + " differ, first at row " + String(idx_first),
            )

    # The condition-video value TRACKS the video schedule and only clamps at
    # the bottom, so the table legitimately collapses at high video timesteps.
    # Anything sizing a modulation cache at a fixed 4 would be wrong.
    var collapsed = minimax_h3_ref2va_row_timesteps(
        plan.layout, Float32(1.0), Float32(0.5)
    )
    report.eq_int("collapse at video t=1.0", len(collapsed.values), 2)

    if MINIMAX_H3_KEYFRAME_NOISE_AUG == Float32(0.999):
        report.ok("keyframe noise aug constant", "0.999")
    else:
        report.fail("keyframe noise aug constant", String(MINIMAX_H3_KEYFRAME_NOISE_AUG))
    if MINIMAX_H3_CONDITION_AUDIO_TIMESTEP == Float32(1.0):
        report.ok("condition audio timestep constant", "1.0")
    else:
        report.fail(
            "condition audio timestep constant",
            String(MINIMAX_H3_CONDITION_AUDIO_TIMESTEP),
        )

    # ── [6] reference ORDER advances the shared clock ────────────────────────
    print("")
    print("[6] request order is semantic (same references, different order)")
    var medias_a = List[MiniMaxH3ReferenceMedia]()
    medias_a.append(MiniMaxH3ReferenceMedia(MINIMAX_H3_REF_IMAGE, 800, 800, 0, 0))
    medias_a.append(MiniMaxH3ReferenceMedia(MINIMAX_H3_REF_AUDIO, 0, 0, 0, 16000))
    var medias_b = List[MiniMaxH3ReferenceMedia]()
    medias_b.append(MiniMaxH3ReferenceMedia(MINIMAX_H3_REF_AUDIO, 0, 0, 0, 16000))
    medias_b.append(MiniMaxH3ReferenceMedia(MINIMAX_H3_REF_IMAGE, 800, 800, 0, 0))

    var refs_a = minimax_h3_resolve_references(medias_a, 39)
    var refs_b = minimax_h3_resolve_references(medias_b, 39)
    var plan_a = minimax_h3_build_ref2va_plan(text_tags, refs_a, 2, 4, 4, 3, 2, 2)
    var plan_b = minimax_h3_build_ref2va_plan(text_tags, refs_b, 2, 4, 4, 3, 2, 2)

    report.eq_int(
        "reordered sequence_length matches",
        plan_a.sequence_length(),
        plan_b.sequence_length(),
    )
    var coords_differ = False
    for i in range(len(plan_a.layout.position_ids)):
        if plan_a.layout.position_ids[i] != plan_b.layout.position_ids[i]:
            coords_differ = True
            break
    if coords_differ:
        report.ok("reordered coordinates differ", "order advances the rotary clock")
    else:
        report.fail(
            "reordered coordinates differ",
            "identical coordinates — reference order was ignored",
        )

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_ref_geometry probe FAILED")
