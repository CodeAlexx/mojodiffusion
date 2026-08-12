from std.collections import List

from serenitymojo.models.dit.minimax_h3_sampling import (
    MINIMAX_H3_TEXT_TAG,
    ROPE_FRAME_RESCALE,
)
from serenitymojo.models.minimax_h3.motion_context import (
    minimax_h3_build_motion_context_geometry,
    minimax_h3_motion_context_audio_latents,
    minimax_h3_motion_context_endpoint_steps,
    minimax_h3_motion_context_pixel_frames,
    minimax_h3_motion_context_steps,
)
from serenitymojo.models.minimax_h3.packing_ref2va import (
    MINIMAX_H3_REF_AUDIO,
    MINIMAX_H3_REF_IMAGE,
    MINIMAX_H3_REF_VIDEO,
    MiniMaxH3PreparedReference,
)
from serenitymojo.models.dit.minimax_h3_ref_geometry import (
    minimax_h3_build_ref2va_motion_context_plan,
    minimax_h3_build_ref2va_plan,
)


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _check(context_frames: Int, source_audio_residual: Float64) raises:
    var tags = List[Int]()
    for _ in range(7):
        tags.append(MINIMAX_H3_TEXT_TAG)
    var latent_h = 8
    var latent_w = 12
    var patch_h = 2
    var patch_w = 2
    var rows_per_frame = 24
    var target_video_steps = 17
    var target_audio_steps = 57
    var context_steps = minimax_h3_motion_context_steps(context_frames)
    var context_audio = minimax_h3_motion_context_audio_latents(context_frames)
    var geometry = minimax_h3_build_motion_context_geometry(
        tags,
        target_video_steps,
        latent_h,
        latent_w,
        target_audio_steps,
        patch_h,
        patch_w,
        context_frames,
        source_audio_residual,
    )
    _require(
        geometry.num_condition_video_rows == context_steps * rows_per_frame,
        "condition video row count",
    )
    _require(
        geometry.num_condition_audio_rows == context_audio * 2,
        "condition audio row count",
    )
    _require(
        len(geometry.video_indices)
            == (context_steps + target_video_steps) * rows_per_frame,
        "all video row count",
    )
    _require(
        len(geometry.audio_indices) == 2 * (context_audio + target_audio_steps),
        "all audio row count",
    )
    _require(
        minimax_h3_motion_context_pixel_frames(context_steps) == context_frames,
        "context frame coverage",
    )

    var text_time = Float64(len(tags))
    var target_origin = text_time + Float64(context_audio)
    var first_video = geometry.video_indices[0]
    _require(
        geometry.position_ids[3 * first_video] == target_origin,
        "first condition video time",
    )
    var second_video = geometry.video_indices[rows_per_frame]
    _require(
        geometry.position_ids[3 * second_video]
            == target_origin + ROPE_FRAME_RESCALE,
        "second condition video time",
    )
    var first_target_video = geometry.video_indices[
        geometry.num_condition_video_rows
    ]
    _require(
        geometry.position_ids[3 * first_target_video] == target_origin,
        "target video origin",
    )

    var first_audio = geometry.audio_indices[0]
    var second_channel_audio = geometry.audio_indices[context_audio]
    _require(
        geometry.position_ids[3 * first_audio]
            == geometry.position_ids[3 * second_channel_audio],
        "condition audio channel-major time",
    )
    var first_target_audio = geometry.audio_indices[
        geometry.num_condition_audio_rows
    ]
    _require(
        geometry.position_ids[3 * first_target_audio] == target_origin,
        "target audio origin",
    )
    var condition_audio_end = (
        geometry.position_ids[3 * first_audio] + Float64(context_audio)
    )
    var expected_end = target_origin + Float64(round(
        ROPE_FRAME_RESCALE * Float64(context_frames)
        + source_audio_residual
    ))
    _require(condition_audio_end == expected_end, "condition audio endpoint")


def _check_ref2va_coexistence() raises:
    var tags = List[Int]()
    for _ in range(7):
        tags.append(MINIMAX_H3_TEXT_TAG)
    var refs = List[MiniMaxH3PreparedReference]()
    refs.append(MiniMaxH3PreparedReference(
        MINIMAX_H3_REF_IMAGE, 1, 8, 12, 0
    ))
    refs.append(MiniMaxH3PreparedReference(
        MINIMAX_H3_REF_VIDEO, 2, 6, 10, 3
    ))
    refs.append(MiniMaxH3PreparedReference(
        MINIMAX_H3_REF_AUDIO, 0, 0, 0, 2
    ))
    var base = minimax_h3_build_ref2va_plan(
        tags, refs, 17, 8, 12, 57, 2, 2
    )
    var combined = minimax_h3_build_ref2va_motion_context_plan(
        tags, refs, 17, 8, 12, 57, 22,
        -Float64(1.0) / Float64(3.0), 2, 2,
    )
    var rows_per_frame = 24
    var motion_video_rows = 7 * rows_per_frame
    var motion_audio_latents = minimax_h3_motion_context_audio_latents(22)
    var motion_audio_rows = 2 * motion_audio_latents
    _require(
        combined.num_condition_video_rows
            == base.num_condition_video_rows + motion_video_rows,
        "combined condition video count",
    )
    _require(
        combined.num_condition_audio_rows
            == base.num_condition_audio_rows + motion_audio_rows,
        "combined condition audio count",
    )
    _require(
        combined.sequence_length()
            == base.sequence_length() + motion_video_rows + motion_audio_rows,
        "combined sequence count",
    )

    # Motion video is first in the modality payload; the first stock image
    # reference follows it without changing its own rotary coordinate.
    _require(
        combined.layout.video_indices[0] == len(tags),
        "motion video payload order",
    )
    var base_first_ref_video = base.layout.video_indices[0]
    var combined_first_ref_video = combined.layout.video_indices[
        motion_video_rows
    ]
    _require(
        combined_first_ref_video == base_first_ref_video + motion_video_rows,
        "reference video physical shift",
    )
    _require(
        combined.layout.position_ids[3 * combined_first_ref_video]
            == base.layout.position_ids[3 * base_first_ref_video],
        "ordinary reference clock preserved",
    )

    var base_target_video = base.layout.video_indices[
        base.num_condition_video_rows
    ]
    var combined_target_video = combined.layout.video_indices[
        combined.num_condition_video_rows
    ]
    var combined_target_origin = combined.layout.position_ids[
        3 * combined_target_video
    ]
    _require(
        combined_target_origin
            == base.layout.position_ids[3 * base_target_video]
                + Float64(motion_audio_latents),
        "motion audio advances target cursor",
    )
    _require(
        combined.layout.position_ids[3 * combined.layout.video_indices[0]]
            == combined_target_origin,
        "motion video lands on target head",
    )

    # Ordinary reference audio remains first; marked motion audio is appended
    # after it and ends on the same overlap endpoint as the fixed video run.
    var first_motion_audio_slot = base.num_condition_audio_rows
    var first_motion_audio = combined.layout.audio_indices[
        first_motion_audio_slot
    ]
    var motion_audio_end = (
        combined.layout.position_ids[3 * first_motion_audio]
        + Float64(motion_audio_latents)
    )
    var expected_end = combined_target_origin + Float64(round(
        ROPE_FRAME_RESCALE * Float64(22) - Float64(1.0) / Float64(3.0)
    ))
    _require(motion_audio_end == expected_end, "combined audio endpoint")
    var first_target_audio = combined.layout.audio_indices[
        combined.num_condition_audio_rows
    ]
    _require(
        combined.layout.position_ids[3 * first_target_audio]
            == combined_target_origin,
        "combined target audio origin",
    )


def main() raises:
    _require(
        minimax_h3_motion_context_endpoint_steps(360, 107) == 107,
        "360-frame cut should use the 362-frame endpoint",
    )
    _require(
        minimax_h3_motion_context_endpoint_steps(382, 117) == 112,
        "382-frame cut should use the 379-frame endpoint",
    )
    _check(5, Float64(1.0) / Float64(3.0))
    _check(22, -Float64(1.0) / Float64(3.0))
    _check(39, 0.0)
    _check_ref2va_coexistence()
    print("PASS minimax_h3 motion-context layout 5/22/39 + Ref2VA coexistence")
