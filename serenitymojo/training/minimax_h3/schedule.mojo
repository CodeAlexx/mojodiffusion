# MiniMax H3 joint video/audio flow schedule.
#
# Oracle (development evidence only):
#   kohya-ss/musubi-tuner
#   commit b8717864713c9e4e7ef3d56eba1fc695a9b626a5
#   src/musubi_tuner/minimax_h3_train_network.py
#
# One unshifted base sigma drives both modalities.  Each modality applies its
# own rational shift, then passes model_t = 1 - sigma to H3.  The native H3
# training target has the OPPOSITE sign from Serenity's shared flow helper:
# target = latent - noise.

from std.collections import List
from std.math import isfinite


comptime MINIMAX_H3_VIDEO_SHIFT = Float32(12.0)
comptime MINIMAX_H3_AUDIO_SHIFT = Float32(3.0)


@fieldwise_init
struct MiniMaxH3SchedulePoint(Copyable, Movable):
    var base_sigma: Float32
    var sigma_video: Float32
    var sigma_audio: Float32
    var model_t_video: Float32
    var model_t_audio: Float32


def minimax_h3_shift_sigma(base_sigma: Float32, shift: Float32) raises -> Float32:
    """Apply Musubi's H3 shift: shift*s/(1+(shift-1)*s)."""
    if not isfinite(base_sigma) or base_sigma < Float32(0.0) or base_sigma > Float32(1.0):
        raise Error("MiniMax H3 base sigma must be finite and in [0,1]")
    if (
        not isfinite(shift)
        or shift < Float32(0.01)
        or shift > Float32(100.0)
    ):
        raise Error("MiniMax H3 sigma shift must be finite and in [0.01,100]")
    return shift * base_sigma / (
        Float32(1.0) + (shift - Float32(1.0)) * base_sigma
    )


def minimax_h3_schedule_point(
    base_sigma: Float32,
    video_shift: Float32 = MINIMAX_H3_VIDEO_SHIFT,
    audio_shift: Float32 = MINIMAX_H3_AUDIO_SHIFT,
) raises -> MiniMaxH3SchedulePoint:
    """Derive both H3 modality clocks from exactly one base sigma."""
    var sigma_video = minimax_h3_shift_sigma(base_sigma, video_shift)
    var sigma_audio = minimax_h3_shift_sigma(base_sigma, audio_shift)
    return MiniMaxH3SchedulePoint(
        base_sigma,
        sigma_video,
        sigma_audio,
        Float32(1.0) - sigma_video,
        Float32(1.0) - sigma_audio,
    )


def minimax_h3_noisy_values(
    latent: List[Float32], noise: List[Float32], sigma: Float32
) raises -> List[Float32]:
    """Host-policy oracle for noisy=(1-sigma)*latent+sigma*noise."""
    if len(latent) != len(noise):
        raise Error("MiniMax H3 latent/noise length mismatch")
    if not isfinite(sigma) or sigma < Float32(0.0) or sigma > Float32(1.0):
        raise Error("MiniMax H3 shifted sigma must be finite and in [0,1]")
    var out = List[Float32]()
    for i in range(len(latent)):
        out.append(
            (Float32(1.0) - sigma) * latent[i] + sigma * noise[i]
        )
    return out^


def minimax_h3_native_targets(
    latent: List[Float32], noise: List[Float32]
) raises -> List[Float32]:
    """Return H3's native velocity target, latent-noise (sign is intentional)."""
    if len(latent) != len(noise):
        raise Error("MiniMax H3 latent/noise length mismatch")
    var out = List[Float32]()
    for i in range(len(latent)):
        out.append(latent[i] - noise[i])
    return out^
