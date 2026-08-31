from serenitymojo.models.minimax_h3.endless import (
    minimax_h3_endless_audio_latents,
    minimax_h3_endless_snap_frames_up,
    minimax_h3_endless_video_latents,
    minimax_h3_plan_endless_chunks,
    minimax_h3_plan_endless_seconds,
)


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def main() raises:
    _require(minimax_h3_endless_snap_frames_up(480) == 481, "20s grid snap")
    _require(minimax_h3_endless_video_latents(124) == 37, "124 video latents")
    _require(minimax_h3_endless_audio_latents(124) == 207, "124 audio latents")

    var plan = minimax_h3_plan_endless_seconds(20)
    _require(len(plan) == 4, "20s chunk count")
    _require(plan[0].sample_frames == 124, "opening sample frames")
    _require(plan[0].new_frames == 124, "opening new frames")
    _require(plan[0].output_trim_frames == 0, "opening trim")
    for index in range(1, 4):
        _require(plan[index].sample_frames == 124, "continuation sample frames")
        _require(plan[index].new_frames == 119, "continuation new frames")
        _require(plan[index].output_trim_frames == 5, "protected boundary trim")
        _require(plan[index].boundary_video_t == 2, "five-frame boundary latent")
        _require(plan[index].reference_video_t == 7, "22-frame reference latent")
    _require(plan[3].frame_end == 481, "complete internal timeline")
    _require(
        plan[0].new_audio_t + plan[1].new_audio_t
            + plan[2].new_audio_t + plan[3].new_audio_t
            == minimax_h3_endless_audio_latents(481),
        "phase-exact audio append",
    )

    var short_plan = minimax_h3_plan_endless_chunks(124)
    _require(len(short_plan) == 1, "single bounded chunk")
    print("PASS minimax_h3 endless 17k+5 planner + phase-exact audio")
