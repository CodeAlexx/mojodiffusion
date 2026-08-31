# MiniMax-H3 frozen final-head backward for LoRA block-stack training.
#
# This is the training-autocast counterpart of the ordinary (non-chunked)
# final layer in models/dit/minimax_h3_frontend.mojo. Frozen output parameters
# remain F32, but pinned Musubi CUDA autocast executes their Linear in BF16 and
# casts predictions back to the latent F32 dtype for loss. Norm, modulation,
# head compute, and returned stack root are BF16. Only d_hidden is produced.

from max.gpu.host import DeviceContext
from std.collections import Dict, List
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.minimax_h3_dit import MiniMaxH3DiTConfig
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.shape_backward import index_select_backward
from serenitymojo.ops.tensor_algebra import add, full_device, gather_rows, slice
from serenitymojo.ops.vec_rms_norm import vec_rms_norm


comptime TArc = ArcPointer[Tensor]
comptime MINIMAX_H3_FINAL_BACKWARD_ORDINARY_MAX_ROWS = 47999


@fieldwise_init
struct MiniMaxH3TrainingHeadBackward(Movable):
    var d_hidden: TArc
    var frozen_weight_grad_count: Int
    var full_tensor_readback_count: Int


@fieldwise_init
struct MiniMaxH3TrainingHeadForward(Movable):
    var video_prediction: TArc
    var audio_prediction: TArc
    var full_tensor_readback_count: Int


def _require_shape(label: String, tensor: Tensor, var expected: List[Int]) raises:
    if tensor.shape() != expected:
        raise Error(label + String(": MiniMax-H3 final-head shape mismatch"))


def _validate_final_head_backward(
    hidden: Tensor,
    final_modulation: Tensor,
    timestep_indices: List[Int],
    video_indices: List[Int],
    audio_indices: List[Int],
    d_video: Tensor,
    d_audio: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    config: MiniMaxH3DiTConfig,
) raises:
    config.validate()
    var hshape = hidden.shape()
    if len(hshape) != 2 or hshape[1] != config.hidden_size:
        raise Error("MiniMax-H3 final-head backward hidden shape mismatch")
    var rows = hshape[0]
    if rows <= 0 or rows > MINIMAX_H3_FINAL_BACKWARD_ORDINARY_MAX_ROWS:
        raise Error(
            "MiniMax-H3 final-head backward currently gates only the ordinary "
            "(<48000-row) frontend path; chunked backward is not implemented"
        )
    if len(timestep_indices) != rows:
        raise Error("MiniMax-H3 final-head backward timestep index mismatch")
    if hidden.dtype() != STDtype.BF16 or final_modulation.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 final-head hidden/modulation must be BF16")
    if len(final_modulation.shape()) != 2 or final_modulation.shape()[0] <= 0:
        raise Error("MiniMax-H3 final modulation must be a nonempty rank-2 table")
    _require_shape(
        "final modulation", final_modulation,
        [final_modulation.shape()[0], 2 * config.hidden_size],
    )
    _require_shape(
        "video root", d_video, [len(video_indices), config.video_patch_dim()]
    )
    _require_shape(
        "audio root", d_audio, [len(audio_indices), config.audio_latents_dim]
    )
    var selected = List[Bool](capacity=rows)
    for _ in range(rows):
        selected.append(False)
    for index in video_indices:
        if index < 0 or index >= rows or selected[index]:
            raise Error("MiniMax-H3 video indices must be unique and in range")
        selected[index] = True
    for index in audio_indices:
        if index < 0 or index >= rows or selected[index]:
            raise Error(
                "MiniMax-H3 audio indices must be unique, in range, and "
                "disjoint from video"
            )
        selected[index] = True
    var modulation_rows = final_modulation.shape()[0]
    for index in timestep_indices:
        if index < 0 or index >= modulation_rows:
            raise Error("MiniMax-H3 final-head timestep index out of range")
    if d_video.dtype() != STDtype.F32 or d_audio.dtype() != STDtype.F32:
        raise Error("MiniMax-H3 final-head output roots must be F32")
    _require_shape(
        "final norm", weights["final_layer.norm.weight"][],
        [config.hidden_size],
    )
    _require_shape(
        "video head", weights["final_layer.video_out.weight"][],
        [config.video_patch_dim(), config.hidden_size],
    )
    _require_shape(
        "audio head", weights["final_layer.audio_out.weight"][],
        [config.audio_latents_dim, config.hidden_size],
    )
    _require_shape(
        "video head bias", weights["final_layer.video_out.bias"][],
        [config.video_patch_dim()],
    )
    _require_shape(
        "audio head bias", weights["final_layer.audio_out.bias"][],
        [config.audio_latents_dim],
    )
    if weights["final_layer.norm.weight"][].dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 final norm must be BF16")
    if (
        weights["final_layer.video_out.weight"][].dtype() != STDtype.F32
        or weights["final_layer.audio_out.weight"][].dtype() != STDtype.F32
        or weights["final_layer.video_out.bias"][].dtype() != STDtype.F32
        or weights["final_layer.audio_out.bias"][].dtype() != STDtype.F32
    ):
        raise Error("MiniMax-H3 frozen output heads must be F32")
    if config.final_norm_eps != Float32(1.0e-5):
        raise Error("MiniMax-H3 final-head backward requires eps=1e-5")


def minimax_h3_training_final_head_forward(
    hidden: Tensor,
    final_modulation: Tensor,
    timestep_indices: List[Int],
    video_indices: List[Int],
    audio_indices: List[Int],
    weights: Dict[String, ArcPointer[Tensor]],
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingHeadForward:
    """Pinned Musubi training-autocast final head, returning F32 loss inputs."""
    # F32 roots with correct shapes let the shared validator enforce every
    # geometry/index/dtype contract without reading any device value.
    var video_root = full_device(
        [len(video_indices), config.video_patch_dim()],
        Float32(0.0), STDtype.F32, ctx,
    )
    var audio_root = full_device(
        [len(audio_indices), config.audio_latents_dim],
        Float32(0.0), STDtype.F32, ctx,
    )
    _validate_final_head_backward(
        hidden, final_modulation, timestep_indices, video_indices,
        audio_indices, video_root, audio_root, weights, config,
    )
    var hidden_size = config.hidden_size
    var normed = vec_rms_norm(
        hidden, weights["final_layer.norm.weight"][],
        config.final_norm_eps, ctx,
    )
    var row_mod = gather_rows(final_modulation, timestep_indices, ctx)
    var shift = slice(row_mod, 1, 0, hidden_size, ctx)
    var scale = slice(row_mod, 1, hidden_size, hidden_size, ctx)
    var modulated = modulate(normed, scale, shift, ctx)
    var video_weight = cast_tensor(
        weights["final_layer.video_out.weight"][], STDtype.BF16, ctx
    )
    var video_bias = cast_tensor(
        weights["final_layer.video_out.bias"][], STDtype.BF16, ctx
    )
    var audio_weight = cast_tensor(
        weights["final_layer.audio_out.weight"][], STDtype.BF16, ctx
    )
    var audio_bias = cast_tensor(
        weights["final_layer.audio_out.bias"][], STDtype.BF16, ctx
    )
    var video_all = linear(
        modulated, video_weight, Optional(video_bias^), ctx
    )
    var audio_all = linear(
        modulated, audio_weight, Optional(audio_bias^), ctx
    )
    var video_selected = gather_rows(video_all, video_indices, ctx)
    var audio_selected = gather_rows(audio_all, audio_indices, ctx)
    var video_f32 = cast_tensor(video_selected, STDtype.F32, ctx)
    var audio_f32 = cast_tensor(audio_selected, STDtype.F32, ctx)
    return MiniMaxH3TrainingHeadForward(
        TArc(video_f32^), TArc(audio_f32^), 0
    )


def minimax_h3_training_final_head_backward(
    hidden: Tensor,
    final_modulation: Tensor,
    timestep_indices: List[Int],
    video_indices: List[Int],
    audio_indices: List[Int],
    d_video: Tensor,
    d_audio: Tensor,
    weights: Dict[String, ArcPointer[Tensor]],
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingHeadBackward:
    """Return the BF16 root gradient consumed by 50-block reverse.

    Final norm/modulation/output-head parameters are frozen. The final AdaLN
    projection backward is outside this LoRA-only block-stack scope.
    """
    _validate_final_head_backward(
        hidden, final_modulation, timestep_indices, video_indices,
        audio_indices, d_video, d_audio, weights, config,
    )
    var rows = hidden.shape()[0]
    var hidden_size = config.hidden_size

    # Recompute the two BF16 activations needed by frozen-head backward.
    var normed = vec_rms_norm(
        hidden, weights["final_layer.norm.weight"][],
        config.final_norm_eps, ctx,
    )
    var row_mod = gather_rows(final_modulation, timestep_indices, ctx)
    var shift = slice(row_mod, 1, 0, hidden_size, ctx)
    var scale = slice(row_mod, 1, hidden_size, hidden_size, ctx)
    var modulated = modulate(normed, scale, shift, ctx)

    # Reverse each selected output independently, then sum at shared modulated.
    var d_video_all = index_select_backward(
        d_video, video_indices, 0,
        [rows, config.video_patch_dim()], ctx,
    )
    var d_audio_all = index_select_backward(
        d_audio, audio_indices, 0,
        [rows, config.audio_latents_dim], ctx,
    )
    # Frozen F32 parameters are cast to BF16 for exact Musubi autocast compute.
    var video_weight = cast_tensor(
        weights["final_layer.video_out.weight"][], STDtype.BF16, ctx
    )
    var audio_weight = cast_tensor(
        weights["final_layer.audio_out.weight"][], STDtype.BF16, ctx
    )
    var d_mod_video = linear_backward_dx(
        d_video_all, video_weight,
        rows, hidden_size, config.video_patch_dim(), ctx,
    )
    var d_mod_audio = linear_backward_dx(
        d_audio_all, audio_weight,
        rows, hidden_size, config.audio_latents_dim, ctx,
    )
    var d_mod_f32 = add(d_mod_video, d_mod_audio, ctx)
    var d_mod_bf16 = cast_tensor(d_mod_f32, STDtype.BF16, ctx)
    var d_normed = modulate_backward(
        d_mod_bf16, normed, scale, ctx, compute_param_grads=False
    )
    var d_hidden = rms_norm_backward_dx(
        d_normed.d_x, hidden, weights["final_layer.norm.weight"][],
        config.final_norm_eps, ctx,
    )
    if d_hidden.dtype() != STDtype.BF16 or d_hidden.shape() != hidden.shape():
        raise Error("MiniMax-H3 final-head stack root contract mismatch")
    return MiniMaxH3TrainingHeadBackward(TArc(d_hidden^), 0, 0)
