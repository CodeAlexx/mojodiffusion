# minimax_h3_av_layout_cpu_probe.mojo — CPU-only evidence that the ALREADY-GATED
# packed-sequence builder handles a full audio-video item with keyframe conditions.
#
# WHY THIS EXISTS. `training/train_minimax_h3.mojo` is the IMAGE-mode maiden arm:
# at :1170 it calls the same builder this probe calls, but pins three arguments —
# num_latent_frames = 1, num_audio_latents = 0, keyframe_anchors = [] — and at
# :1209 concats only [text | video]. That made it look like the packed AV training
# layout was missing. It is not missing. This probe is the receipt.
#
# NOT A GATE. It asserts nothing against an oracle; it prints structure. The real
# bit-exact gate against the diffusers PR #14355 dump already exists next door in
# `minimax_h3_packing_parity.mojo`. This probe only shows that the AV+condition
# arms of that same builder are live, and it runs with NO GPU — useful when the
# device is busy or absent.
#
# Run:  mojo run -I . serenitymojo/models/minimax_h3/parity/minimax_h3_av_layout_cpu_probe.mojo
#
# Measured 2026-08-26 (Mojo 1.0.0, no GPU):
#   S 82  = 4 text + 12 cond-video + 24 audio + 42 target video
#   video_indices 54 (12 conditioning + 42 target), audio_indices 24, cond_audio 0
#   distinct row timesteps 3 -> 0.4 video / 0.7 audio / 0.999 condition
from serenitymojo.models.minimax_h3.packing import (
    minimax_h3_build_packed_sequence, minimax_h3_build_row_timesteps,
)


def main() raises:
    # 4 text rows; a real AV item: 7 video latent frames (5n+2), 4x6 latent grid,
    # 12 audio latents, two keyframe anchors (first+last) — every argument the
    # trainer currently pins off.
    var tags = List[Int]()
    for _ in range(4):
        tags.append(1)
    var anchors = List[Int]()
    anchors.append(0)
    anchors.append(1)

    var lay = minimax_h3_build_packed_sequence(
        tags, 7, 4, 6, 12, 2, 2, anchors, Float64(1.0)
    )
    print("S", lay.sequence_length)
    print(
        "video_rows", len(lay.video_indices),
        "cond_video", lay.num_condition_video_rows,
    )
    print(
        "audio_rows", len(lay.audio_indices),
        "cond_audio", lay.num_condition_audio_rows,
    )
    print("text_rows", len(lay.text_indices))

    # the condition-clean coefficients the trainer already passes at :1174 —
    # inert today only because there are no condition rows to pin.
    var ts = minimax_h3_build_row_timesteps(
        lay, Float32(0.4), Float32(0.7), Float32(0.999), Float32(1.0)
    )
    print("distinct_timesteps", len(ts.values))
    for i in range(len(ts.values)):
        print("  ts", i, ts.values[i])
