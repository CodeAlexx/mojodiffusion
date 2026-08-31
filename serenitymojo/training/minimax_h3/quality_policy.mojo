# MiniMax H3 optional quality-policy math.
#
# This is a bounded host-math slice, not a trainer or launch surface.  Guidance
# and teacher matching remain unsupported by the cache launch contract.
#
# Pinned development oracle:
#   kohya-ss/musubi-tuner
#   commit b8717864713c9e4e7ef3d56eba1fc695a9b626a5
#   src/musubi_tuner/minimax_h3_train_network.py
#     _apply_timestep_focus                         lines 549-561
#     _dc_attenuated_prediction                    lines 563-571
#     _decomposed_flow_loss                        lines 573-591
#     _preservation_density_compensation           lines 593-610
#     _apply_guidance_loss_targets target rebuild  lines 1428-1440

from std.collections import List
from std.math import isfinite, max, sqrt


@fieldwise_init
struct MiniMaxH3GuidanceTargets(Copyable, Movable):
    var video: List[Float32]
    var audio: List[Float32]
    var applied: Bool


@fieldwise_init
struct MiniMaxH3DecomposedLoss(Copyable, Movable):
    var value: Float32
    var magnitude_term: Float32
    var direction_term: Float32
    var prediction_gradient: List[Float32]


@fieldwise_init
struct MiniMaxH3DCShapedMSE(Copyable, Movable):
    var shaped_prediction: List[Float32]
    var value: Float32
    var prediction_gradient: List[Float32]


def _validate_finite_vector(values: List[Float32], label: String) raises:
    if len(values) == 0:
        raise Error(label + String(" must be nonempty"))
    for value in values:
        if not isfinite(value):
            raise Error(label + String(" must be finite"))


def _validate_pair(
    lhs: List[Float32], rhs: List[Float32], label: String
) raises:
    _validate_finite_vector(lhs, label + String(" lhs"))
    _validate_finite_vector(rhs, label + String(" rhs"))
    if len(lhs) != len(rhs):
        raise Error(label + String(" vectors must have equal length"))


def validate_minimax_h3_quality_mode(
    guidance_enabled: Bool, teacher_matching: Bool
) raises:
    """Pinned Musubi makes guidance loss and teacher matching exclusive."""
    if guidance_enabled and teacher_matching:
        raise Error(
            "MiniMax H3 guidance loss and teacher matching are mutually exclusive"
        )


def minimax_h3_guidance_targets(
    base_sigma: Float32,
    sigma_min: Float32,
    video_velocity: List[Float32],
    video_uncond: List[Float32],
    video_scale: Float32,
    audio_velocity: List[Float32],
    audio_uncond: List[Float32],
    audio_scale: Float32,
) raises -> MiniMaxH3GuidanceTargets:
    """Build u+scale*(velocity-u), gated by the pre-shift base sigma."""
    if (
        not isfinite(base_sigma)
        or base_sigma < Float32(0.0)
        or base_sigma > Float32(1.0)
        or not isfinite(sigma_min)
        or sigma_min < Float32(0.0)
        or sigma_min > Float32(1.0)
    ):
        raise Error("MiniMax H3 guidance base sigma and gate must be in [0,1]")
    if (
        not isfinite(video_scale)
        or video_scale < Float32(0.0)
        or not isfinite(audio_scale)
        or audio_scale < Float32(0.0)
    ):
        raise Error("MiniMax H3 guidance scales must be finite and nonnegative")
    _validate_pair(video_velocity, video_uncond, String("guidance video"))
    _validate_pair(audio_velocity, audio_uncond, String("guidance audio"))

    var applied = base_sigma >= sigma_min
    var video = List[Float32]()
    var audio = List[Float32]()
    for i in range(len(video_velocity)):
        if applied:
            video.append(
                video_uncond[i]
                + video_scale * (video_velocity[i] - video_uncond[i])
            )
        else:
            video.append(video_velocity[i])
    for i in range(len(audio_velocity)):
        if applied:
            audio.append(
                audio_uncond[i]
                + audio_scale * (audio_velocity[i] - audio_uncond[i])
            )
        else:
            audio.append(audio_velocity[i])
    return MiniMaxH3GuidanceTargets(video^, audio^, applied)


def minimax_h3_timestep_focus(
    base_sigma: Float32,
    focus_min: Float32,
    focus_max: Float32,
    focus_probability: Float32,
) raises -> Float32:
    """Exact deterministic remap of one uniform [0,1] draw."""
    if (
        not isfinite(base_sigma)
        or base_sigma < Float32(0.0)
        or base_sigma > Float32(1.0)
        or not isfinite(focus_probability)
        or focus_probability < Float32(0.0)
        or focus_probability > Float32(1.0)
    ):
        raise Error("MiniMax H3 focus draw/probability must be finite in [0,1]")
    if focus_probability <= Float32(0.0):
        return base_sigma
    if (
        not isfinite(focus_min)
        or not isfinite(focus_max)
        or focus_min < Float32(0.0)
        or focus_min >= focus_max
        or focus_max > Float32(1.0)
    ):
        raise Error("MiniMax H3 focus band must satisfy 0 <= min < max <= 1")
    if base_sigma < focus_probability:
        return focus_min + (focus_max - focus_min) * (
            base_sigma / focus_probability
        )
    # Musubi uses max(1-p, 1e-8); p=1 only reaches this branch at base=1.
    return (base_sigma - focus_probability) / max(
        Float32(1.0) - focus_probability, Float32(1.0e-8)
    )


def minimax_h3_teacher_preservation_compensation(
    teacher_condition_sigma_max: Float32,
    focus_min: Float32,
    focus_max: Float32,
    focus_probability: Float32,
) raises -> Float32:
    """Keep the preservation anchor's expected share invariant under focus."""
    if (
        not isfinite(teacher_condition_sigma_max)
        or teacher_condition_sigma_max < Float32(0.0)
        or teacher_condition_sigma_max > Float32(1.0)
        or not isfinite(focus_probability)
        or focus_probability < Float32(0.0)
        or focus_probability > Float32(1.0)
    ):
        raise Error("MiniMax H3 teacher sigma/probability must be in [0,1]")
    if focus_probability > Float32(0.0) and (
        not isfinite(focus_min)
        or not isfinite(focus_max)
        or focus_min < Float32(0.0)
        or focus_min >= focus_max
        or focus_max > Float32(1.0)
    ):
        raise Error("MiniMax H3 focus band must satisfy 0 <= min < max <= 1")
    var anchor_width = Float32(1.0) - teacher_condition_sigma_max
    if anchor_width <= Float32(0.0) or focus_probability <= Float32(0.0):
        return Float32(1.0)
    var overlap = max(
        Float32(0.0),
        focus_max - max(focus_min, teacher_condition_sigma_max),
    )
    var focused_share = (
        (Float32(1.0) - focus_probability) * anchor_width
        + focus_probability * overlap / (focus_max - focus_min)
    )
    if focused_share <= Float32(0.0):
        return Float32(1.0)
    return anchor_width / focused_share


def minimax_h3_decomposed_flow_loss(
    prediction: List[Float32],
    target: List[Float32],
    magnitude_weight: Float32,
    direction_weight: Float32,
) raises -> MiniMaxH3DecomposedLoss:
    """Musubi magnitude/direction value and analytic prediction gradient."""
    _validate_pair(prediction, target, String("teacher decomposed loss"))
    if (
        not isfinite(magnitude_weight)
        or magnitude_weight < Float32(0.0)
        or not isfinite(direction_weight)
        or direction_weight < Float32(0.0)
    ):
        raise Error("MiniMax H3 decomposed loss weights must be nonnegative")
    var prediction_sq = Float32(0.0)
    var target_sq = Float32(0.0)
    var dot = Float32(0.0)
    for i in range(len(prediction)):
        prediction_sq += prediction[i] * prediction[i]
        target_sq += target[i] * target[i]
        dot += prediction[i] * target[i]
    var prediction_norm = sqrt(prediction_sq)
    var target_norm = sqrt(target_sq)
    if prediction_norm <= Float32(0.0) or target_norm <= Float32(0.0):
        raise Error("MiniMax H3 decomposed-loss fixtures require nonzero norms")
    var eps = Float32(1.0e-12)
    var denominator = prediction_norm * target_norm + eps
    var cosine = dot / denominator
    var magnitude_term = (prediction_norm - target_norm) * (
        prediction_norm - target_norm
    )
    var direction_term = (
        Float32(2.0) * prediction_norm * target_norm
        * (Float32(1.0) - cosine)
    )
    var count = Float32(len(prediction))
    var gradient = List[Float32]()
    for i in range(len(prediction)):
        var magnitude_gradient = (
            Float32(2.0) * magnitude_weight
            * (prediction_norm - target_norm)
            * prediction[i] / prediction_norm
        )
        # The leading prediction_norm in Musubi's direction term is detached;
        # the norm inside cosine remains differentiable.
        var cosine_gradient = (
            target[i] / denominator
            - dot * target_norm * prediction[i]
            / (prediction_norm * denominator * denominator)
        )
        var direction_gradient = (
            -Float32(2.0) * direction_weight
            * prediction_norm * target_norm * cosine_gradient
        )
        gradient.append((magnitude_gradient + direction_gradient) / count)
    var value = (
        magnitude_weight * magnitude_term
        + direction_weight * direction_term
    ) / count
    return MiniMaxH3DecomposedLoss(
        value, magnitude_term / count, direction_term / count, gradient^
    )


def minimax_h3_conditioned_video_dc_mse(
    prediction: List[Float32],
    target: List[Float32],
    batch_channel_groups: Int,
    dc_weight: Float32,
) raises -> MiniMaxH3DCShapedMSE:
    """Apply Musubi's per-[B,C] DC map, then return MSE and dL/dprediction."""
    _validate_pair(prediction, target, String("teacher video DC loss"))
    if (
        batch_channel_groups <= 0
        or len(prediction) % batch_channel_groups != 0
    ):
        raise Error("MiniMax H3 DC groups must evenly divide the tensor")
    if not isfinite(dc_weight) or dc_weight < Float32(0.0):
        raise Error("MiniMax H3 teacher DC weight must be nonnegative")
    var values_per_group = len(prediction) // batch_channel_groups
    var root_weight = sqrt(dc_weight)
    var shaped = List[Float32]()
    var gradient = List[Float32]()
    for _ in range(len(prediction)):
        shaped.append(Float32(0.0))
        gradient.append(Float32(0.0))
    var loss_sum = Float32(0.0)
    for group in range(batch_channel_groups):
        var start = group * values_per_group
        var dc = Float32(0.0)
        for offset in range(values_per_group):
            var index = start + offset
            dc += prediction[index] - target[index]
        dc /= Float32(values_per_group)
        for offset in range(values_per_group):
            var index = start + offset
            var residual = prediction[index] - target[index]
            var shaped_residual = residual - (Float32(1.0) - root_weight) * dc
            shaped[index] = target[index] + shaped_residual
            loss_sum += shaped_residual * shaped_residual
            # A^T A residual = AC(residual) + dc_weight*DC(residual).
            gradient[index] = (
                Float32(2.0) / Float32(len(prediction))
                * (residual - dc + dc_weight * dc)
            )
    return MiniMaxH3DCShapedMSE(
        shaped^, loss_sum / Float32(len(prediction)), gradient^
    )
