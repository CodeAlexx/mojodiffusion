# MiniMax H3 audio-presence-gated joint-modality loss policy.
#
# Oracle (development evidence only):
#   kohya-ss/musubi-tuner
#   commit b8717864713c9e4e7ef3d56eba1fc695a9b626a5
#   src/musubi_tuner/minimax_h3_train_network.py::compute_loss
#   src/musubi_tuner/training/audio_loss.py::effective_audio_loss_weights
#
# R1 has batch_size=1. Video is always supervised.  Audio is supervised only
# when the cache's audio_present scalar is exactly 1 and video_only is false.
# Reductions are independent modality means, preventing tensor-size imbalance:
# total = mean(video MSE) + effective_audio_weight * mean(audio MSE).

from std.collections import List
from std.math import isfinite


@fieldwise_init
struct MiniMaxH3AVLoss(Copyable, Movable):
    var video: Float32
    var audio: Float32
    var effective_audio_weight: Float32
    var total: Float32


def minimax_h3_mse_mean(
    prediction: List[Float32], target: List[Float32]
) raises -> Float32:
    if len(prediction) == 0:
        raise Error("MiniMax H3 MSE requires at least one element")
    if len(prediction) != len(target):
        raise Error("MiniMax H3 prediction/target length mismatch")
    var total = Float32(0.0)
    for i in range(len(prediction)):
        var residual = prediction[i] - target[i]
        total += residual * residual
    return total / Float32(len(prediction))


def minimax_h3_effective_audio_weight(
    configured_weight: Float32,
    audio_present: Float32,
    video_only: Bool = False,
) raises -> Float32:
    if not isfinite(configured_weight) or configured_weight < Float32(0.0):
        raise Error("MiniMax H3 audio loss weight must be finite and nonnegative")
    if audio_present != Float32(0.0) and audio_present != Float32(1.0):
        raise Error("MiniMax H3 audio_present must be exactly 0.0 or 1.0")
    if video_only:
        return Float32(0.0)
    return configured_weight * audio_present


def minimax_h3_presence_gated_av_mse(
    video_prediction: List[Float32],
    video_target: List[Float32],
    audio_prediction: List[Float32],
    audio_target: List[Float32],
    configured_audio_weight: Float32,
    audio_present: Float32,
    video_only: Bool = False,
) raises -> MiniMaxH3AVLoss:
    """Independent means plus the cache-presence audio modality mask."""
    var video_loss = minimax_h3_mse_mean(video_prediction, video_target)
    var effective_weight = minimax_h3_effective_audio_weight(
        configured_audio_weight, audio_present, video_only
    )
    var audio_loss = Float32(0.0)
    if effective_weight != Float32(0.0):
        audio_loss = minimax_h3_mse_mean(audio_prediction, audio_target)
    return MiniMaxH3AVLoss(
        video_loss,
        audio_loss,
        effective_weight,
        video_loss + effective_weight * audio_loss,
    )
