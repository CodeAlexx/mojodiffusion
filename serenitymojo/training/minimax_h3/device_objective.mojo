# MiniMax-H3 device-native dual-schedule flow objective.
#
# One scalar base sigma is shifted independently for video/audio. Cache
# latents and sampled noise remain F32 on device; H3's native target is
# latent-noise (the sign is intentionally opposite Serenity's generic helper).
# Video/audio losses are independent means, with audio gated by the exact
# cache scalar and configured weight.

from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.patchify3d import patchify3d
from serenitymojo.ops.tensor_algebra import (
    add,
    full_device,
    mul_scalar,
    permute,
    reshape,
    sub,
)
from serenitymojo.training.device_loss import device_mse_loss_grad
from serenitymojo.training.minimax_h3.loss import (
    minimax_h3_effective_audio_weight,
)
from serenitymojo.training.minimax_h3.schedule import (
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_VIDEO_SHIFT,
    MiniMaxH3SchedulePoint,
    minimax_h3_schedule_point,
)


comptime TArc = ArcPointer[Tensor]
comptime MINIMAX_H3_DEVICE_OBJECTIVE_BACKEND = (
    "h3-dual-shift-device-flow+independent-device-mse"
)


@fieldwise_init
struct MiniMaxH3DeviceNoising(Movable):
    # Packed rows consumed/emitted by the H3 frontend/final heads.
    var video_x_t_rows: TArc
    var audio_x_t_rows: TArc
    var video_target_rows: TArc
    var audio_target_rows: TArc
    var schedule: MiniMaxH3SchedulePoint
    var full_tensor_readback_count: Int


@fieldwise_init
struct MiniMaxH3DeviceAVLoss(Movable):
    var video_loss: Float32
    var audio_loss: Float32
    var effective_audio_weight: Float32
    var total_loss: Float32
    var d_video: TArc
    var d_audio: TArc
    var scalar_readback_count: Int
    var full_tensor_readback_count: Int
    var sync_count: Int
    var backend: String


def _same_shape(a: Tensor, b: Tensor) -> Bool:
    var ashape = a.shape()
    var bshape = b.shape()
    if len(ashape) != len(bshape):
        return False
    for axis in range(len(ashape)):
        if ashape[axis] != bshape[axis]:
            return False
    return True


def _validate_latent_noise(label: String, latent: Tensor, noise: Tensor) raises:
    if latent.dtype() != STDtype.F32 or noise.dtype() != STDtype.F32:
        raise Error(label + String(": latent/noise must be device F32"))
    if not _same_shape(latent, noise) or latent.numel() <= 0:
        raise Error(label + String(": latent/noise shape mismatch or empty tensor"))


def _pack_video_rows(latent: Tensor, ctx: DeviceContext) raises -> Tensor:
    var shape = latent.shape()
    if (
        len(shape) != 4 or shape[0] != 24 or shape[1] <= 0
        or shape[2] <= 0 or shape[3] <= 0
        or shape[2] % 2 != 0 or shape[3] % 2 != 0
    ):
        raise Error("MiniMax-H3 video latent must be F32 [24,F,H,W], H/W even")
    # H3 patch (1,2,2), with c-slowest within-patch order. This is the same
    # shared primitive used by the inference frontend.
    return patchify3d(latent, 1, 2, 2, ctx)


def _pack_audio_rows(latent: Tensor, ctx: DeviceContext) raises -> Tensor:
    var shape = latent.shape()
    if len(shape) != 3 or shape[0] != 32 or shape[1] != 2 or shape[2] <= 0:
        raise Error("MiniMax-H3 audio latent must be F32 [32,2,A]")
    # Inverse of Musubi unpack: [C,2,A] -> [2,A,C] -> [2A,C].
    var channel_last = permute(latent, [1, 2, 0], ctx)
    return reshape(channel_last, [2 * shape[2], 32], ctx)


def minimax_h3_device_dual_shift_noising(
    video_latent: Tensor,
    video_noise: Tensor,
    audio_latent: Tensor,
    audio_noise: Tensor,
    base_sigma: Float32,
    ctx: DeviceContext,
    video_shift: Float32 = MINIMAX_H3_VIDEO_SHIFT,
    audio_shift: Float32 = MINIMAX_H3_AUDIO_SHIFT,
) raises -> MiniMaxH3DeviceNoising:
    """Construct both noisy inputs and native velocity targets on device."""
    _validate_latent_noise("MiniMax-H3 video", video_latent, video_noise)
    _validate_latent_noise("MiniMax-H3 audio", audio_latent, audio_noise)
    var point = minimax_h3_schedule_point(
        base_sigma, video_shift, audio_shift
    )
    var video_clean = mul_scalar(
        video_latent, Float32(1.0) - point.sigma_video, ctx
    )
    var video_eps = mul_scalar(video_noise, point.sigma_video, ctx)
    var audio_clean = mul_scalar(
        audio_latent, Float32(1.0) - point.sigma_audio, ctx
    )
    var audio_eps = mul_scalar(audio_noise, point.sigma_audio, ctx)
    var video_x_t = add(video_clean, video_eps, ctx)
    var audio_x_t = add(audio_clean, audio_eps, ctx)
    var video_target = sub(video_latent, video_noise, ctx)
    var audio_target = sub(audio_latent, audio_noise, ctx)
    # Mean MSE is invariant under these bijective permutations, so loss may
    # remain in head-row layout and its gradient feeds the head directly.
    var video_x_t_rows = _pack_video_rows(video_x_t, ctx)
    var audio_x_t_rows = _pack_audio_rows(audio_x_t, ctx)
    var video_target_rows = _pack_video_rows(video_target, ctx)
    var audio_target_rows = _pack_audio_rows(audio_target, ctx)
    return MiniMaxH3DeviceNoising(
        TArc(video_x_t_rows^), TArc(audio_x_t_rows^),
        TArc(video_target_rows^), TArc(audio_target_rows^), point^, 0,
    )


def minimax_h3_device_presence_gated_av_mse(
    video_prediction: Tensor,
    video_target: Tensor,
    audio_prediction: Tensor,
    audio_target: Tensor,
    configured_audio_weight: Float32,
    audio_present: Float32,
    ctx: DeviceContext,
    video_only: Bool = False,
) raises -> MiniMaxH3DeviceAVLoss:
    """Independent device means and exact scalar audio-presence gating.

    Only the one F32 reduced scalar per active modality crosses to host. No
    prediction, target, or gradient tensor is read back. An inactive audio arm
    does not evaluate its MSE and returns an all-zero F32 device root.
    """
    var effective = minimax_h3_effective_audio_weight(
        configured_audio_weight, audio_present, video_only
    )
    if (
        video_prediction.dtype() != STDtype.F32
        or video_target.dtype() != STDtype.F32
        or not _same_shape(video_prediction, video_target)
        or video_prediction.numel() <= 0
    ):
        raise Error("MiniMax-H3 video prediction/target must be matching nonempty F32")
    # Validate the inactive arm too: absence must never hide corrupt geometry.
    if (
        audio_prediction.dtype() != STDtype.F32
        or audio_target.dtype() != STDtype.F32
        or not _same_shape(audio_prediction, audio_target)
        or audio_prediction.numel() <= 0
    ):
        raise Error("MiniMax-H3 audio prediction/target must be matching nonempty F32")
    var video = device_mse_loss_grad(
        video_prediction, video_target, STDtype.F32, ctx
    )
    var audio_loss = Float32(0.0)
    var d_audio: Tensor
    var scalar_reads = video.scalar_readback_count
    var full_reads = video.full_tensor_readback_count
    var syncs = video.sync_count
    if effective != Float32(0.0):
        var audio = device_mse_loss_grad(
            audio_prediction, audio_target, STDtype.F32, ctx
        )
        audio_loss = audio.loss
        d_audio = mul_scalar(audio.d_pred, effective, ctx)
        scalar_reads += audio.scalar_readback_count
        full_reads += audio.full_tensor_readback_count
        syncs += audio.sync_count
    else:
        d_audio = full_device(
            audio_prediction.shape(), Float32(0.0), STDtype.F32, ctx
        )
    if full_reads != 0:
        raise Error("MiniMax-H3 device AV loss performed a full tensor readback")
    var video_loss = video.loss
    var total_loss = video_loss + effective * audio_loss
    var d_video_tensor = video^.take_d_pred()
    var d_video = TArc(d_video_tensor^)
    return MiniMaxH3DeviceAVLoss(
        video_loss,
        audio_loss,
        effective,
        total_loss,
        d_video^,
        TArc(d_audio^),
        scalar_reads,
        full_reads,
        syncs,
        String(MINIMAX_H3_DEVICE_OBJECTIVE_BACKEND),
    )
