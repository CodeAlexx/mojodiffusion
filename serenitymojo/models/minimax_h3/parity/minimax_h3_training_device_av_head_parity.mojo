# Torch-pinned gate for the product H3 device AV objective and training head.
#
# Reduced hidden geometry with exact H3 video/audio channel widths. This gates
# device dual-shift noising, native target row packing, independent weighted
# loss roots, inactive-audio policy, and the frozen-head BF16 stack root.
# Required invocation:
#   bash serenitymojo/models/minimax_h3/parity/run_minimax_h3_training_device_av_head_parity.sh

from max.gpu.host import DeviceContext
from std.collections import Dict, List
from std.math import isfinite
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.dit.minimax_h3_dit import MiniMaxH3DiTConfig
from serenitymojo.models.minimax_h3.training_head import (
    minimax_h3_training_final_head_backward,
    minimax_h3_training_final_head_forward,
)
from serenitymojo.training.minimax_h3.device_objective import (
    minimax_h3_device_dual_shift_noising,
    minimax_h3_device_presence_gated_av_mse,
)
from serenitymojo.models.minimax_h3.parity.minimax_h3_training_stack50_bf16_flash_parity import (
    _bf16_as_f32,
    _check,
    _load_f32,
    _t,
    _t_f32,
    _ta,
    _ta_f32,
)


comptime TArc = ArcPointer[Tensor]
comptime FIXTURE = "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_training_device_av_head.safetensors"
comptime FIXTURE_SHA256 = "35aa2d4d40e36a51948b53d287a8184f037413525126063622670ac2123c2f95"
comptime S = 8
comptime D = 8
comptime VIDEO_ROWS = 2
comptime VIDEO_WIDTH = 96
comptime AUDIO_ROWS = 4
comptime AUDIO_WIDTH = 32


def _config() -> MiniMaxH3DiTConfig:
    return MiniMaxH3DiTConfig(
        D, 1, 1, 1, 8, 12, 24, 32, 8, 8, 8,
        6 * D * 3, 2 * D, 1,
        Float32(1.0e-5), Float32(1.0e-5), Float32(1.0e-5),
    )


def _weights(ref st: SafeTensors, ctx: DeviceContext) raises -> Dict[String, TArc]:
    var weights = Dict[String, TArc]()
    weights["final_layer.norm.weight"] = _ta(st, "head.norm_weight", [D], ctx)
    weights["final_layer.video_out.weight"] = _ta_f32(
        st, "head.video_weight", [VIDEO_WIDTH, D], ctx
    )
    weights["final_layer.video_out.bias"] = _ta_f32(
        st, "head.video_bias", [VIDEO_WIDTH], ctx
    )
    weights["final_layer.audio_out.weight"] = _ta_f32(
        st, "head.audio_weight", [AUDIO_WIDTH, D], ctx
    )
    weights["final_layer.audio_out.bias"] = _ta_f32(
        st, "head.audio_bias", [AUDIO_WIDTH], ctx
    )
    return weights^


def _near(label: String, got: Float32, want: Float32, tol: Float32) -> Bool:
    var delta = got - want
    var error = -delta if delta < Float32(0.0) else delta
    var ok = isfinite(got) and error <= tol
    print("  ", "ok" if ok else "FAIL", label, got, "want", want, "abs", error)
    return ok


def _all_zero(label: String, tensor: Tensor, ctx: DeviceContext) raises -> Bool:
    var values = tensor.to_host(ctx)
    for value in values:
        if value != Float32(0.0):
            print("  FAIL", label, "nonzero")
            return False
    print("  ok", label, "exact zero")
    return True


def main() raises:
    print("MiniMax-H3 device dual-shift AV objective + frozen-head parity")
    print("  reduced hidden synthetic gate; no checkpoint/dataset/trainer claim")
    print("  validated fixture sha256", FIXTURE_SHA256)
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(FIXTURE))
    var scalars = _load_f32(st, "meta.scalars", [12])
    var noised = minimax_h3_device_dual_shift_noising(
        _t_f32(st, "in.video_latent", [24, 1, 2, 4], ctx),
        _t_f32(st, "in.video_noise", [24, 1, 2, 4], ctx),
        _t_f32(st, "in.audio_latent", [32, 2, 2], ctx),
        _t_f32(st, "in.audio_noise", [32, 2, 2], ctx),
        scalars[0], ctx,
    )
    var ok = noised.full_tensor_readback_count == 0
    ok = _check(
        "video.x_t.rows", noised.video_x_t_rows[],
        _load_f32(st, "out.video_x_t_rows", [VIDEO_ROWS, VIDEO_WIDTH]),
        [VIDEO_ROWS, VIDEO_WIDTH], STDtype.F32, ctx,
        0.9999999, 1.4e-7, 2.0e-8, 2.0e-8,
    ) and ok
    ok = _check(
        "video.target.rows", noised.video_target_rows[],
        _load_f32(st, "out.video_target_rows", [VIDEO_ROWS, VIDEO_WIDTH]),
        [VIDEO_ROWS, VIDEO_WIDTH], STDtype.F32, ctx,
        1.0, 0.0, 0.0, 0.0,
    ) and ok
    ok = _check(
        "audio.x_t.rows", noised.audio_x_t_rows[],
        _load_f32(st, "out.audio_x_t_rows", [AUDIO_ROWS, AUDIO_WIDTH]),
        [AUDIO_ROWS, AUDIO_WIDTH], STDtype.F32, ctx,
        1.0, 0.0, 0.0, 0.0,
    ) and ok
    ok = _check(
        "audio.target.rows", noised.audio_target_rows[],
        _load_f32(st, "out.audio_target_rows", [AUDIO_ROWS, AUDIO_WIDTH]),
        [AUDIO_ROWS, AUDIO_WIDTH], STDtype.F32, ctx,
        1.0, 0.0, 0.0, 0.0,
    ) and ok
    ok = _near("sigma.video", noised.schedule.sigma_video, scalars[1], 1.0e-7) and ok
    ok = _near("sigma.audio", noised.schedule.sigma_audio, scalars[2], 1.0e-7) and ok
    ok = _near("model_t.video", noised.schedule.model_t_video, scalars[3], 1.0e-7) and ok
    ok = _near("model_t.audio", noised.schedule.model_t_audio, scalars[4], 1.0e-7) and ok

    var timestep_indices: List[Int] = [0, 1, 0, 1, 0, 1, 0, 1]
    var video_indices: List[Int] = [4, 7]
    var audio_indices: List[Int] = [0, 2, 3, 6]
    var hidden = _t(st, "head.hidden", [S, D], ctx)
    var final_modulation = _t(st, "head.final_modulation", [2, 2 * D], ctx)
    var weights = _weights(st, ctx)
    var head = minimax_h3_training_final_head_forward(
        hidden, final_modulation, timestep_indices,
        video_indices, audio_indices, weights, _config(), ctx,
    )
    ok = _check(
        "head.video.prediction", head.video_prediction[],
        _load_f32(st, "head.video_prediction", [VIDEO_ROWS, VIDEO_WIDTH]),
        [VIDEO_ROWS, VIDEO_WIDTH], STDtype.F32, ctx,
        0.999996, 4.5e-3, 3.1e-3, 5.0e-4,
    ) and ok
    ok = _check(
        "head.audio.prediction", head.audio_prediction[],
        _load_f32(st, "head.audio_prediction", [AUDIO_ROWS, AUDIO_WIDTH]),
        [AUDIO_ROWS, AUDIO_WIDTH], STDtype.F32, ctx,
        0.9999933, 4.5e-3, 3.95e-3, 2.0e-4,
    ) and ok

    var present = minimax_h3_device_presence_gated_av_mse(
        head.video_prediction[], noised.video_target_rows[],
        head.audio_prediction[], noised.audio_target_rows[],
        scalars[5], Float32(1.0), ctx,
    )
    ok = _near("loss.video", present.video_loss, scalars[6], 6.5e-5) and ok
    ok = _near("loss.audio", present.audio_loss, scalars[7], 1.35e-4) and ok
    ok = _near("loss.total", present.total_loss, scalars[8], 1.53e-4) and ok
    ok = present.scalar_readback_count == Int(scalars[10]) and ok
    ok = present.full_tensor_readback_count == 0 and ok
    ok = _check(
        "loss.d_video.present", present.d_video[],
        _load_f32(st, "loss.d_video_present", [VIDEO_ROWS, VIDEO_WIDTH]),
        [VIDEO_ROWS, VIDEO_WIDTH], STDtype.F32, ctx,
        0.9999997, 4.7e-5, 8.2e-4, 2.4e-5,
    ) and ok
    ok = _check(
        "loss.d_audio.present", present.d_audio[],
        _load_f32(st, "loss.d_audio_present", [AUDIO_ROWS, AUDIO_WIDTH]),
        [AUDIO_ROWS, AUDIO_WIDTH], STDtype.F32, ctx,
        0.9999992, 4.6e-5, 1.32e-3, 7.2e-5,
    ) and ok
    var root = minimax_h3_training_final_head_backward(
        hidden, final_modulation, timestep_indices,
        video_indices, audio_indices, present.d_video[], present.d_audio[],
        weights, _config(), ctx,
    )
    ok = _check(
        "head.d_hidden.stack_root", root.d_hidden[],
        _bf16_as_f32(st, "loss.d_hidden_present", [S, D]),
        [S, D], STDtype.BF16, ctx,
        0.999982, 5.7e-4, 6.55e-3, 1.3e-3,
    ) and ok
    ok = root.frozen_weight_grad_count == 0 and root.full_tensor_readback_count == 0 and ok

    var absent = minimax_h3_device_presence_gated_av_mse(
        head.video_prediction[], noised.video_target_rows[],
        head.audio_prediction[], noised.audio_target_rows[],
        scalars[5], Float32(0.0), ctx,
    )
    ok = _near("loss.absent.total", absent.total_loss, scalars[9], 6.5e-5) and ok
    ok = absent.scalar_readback_count == Int(scalars[11]) and ok
    ok = _all_zero("loss.d_audio.absent", absent.d_audio[], ctx) and ok

    var malformed_rejected = False
    try:
        _ = minimax_h3_device_presence_gated_av_mse(
            head.video_prediction[], noised.video_target_rows[],
            head.audio_prediction[], _t_f32(st, "head.audio_bias", [AUDIO_WIDTH], ctx),
            scalars[5], Float32(0.0), ctx,
        )
    except:
        malformed_rejected = True
    var duplicate_rejected = False
    var duplicate_video: List[Int] = [4, 4]
    try:
        _ = minimax_h3_training_final_head_backward(
            hidden, final_modulation, timestep_indices,
            duplicate_video, audio_indices, present.d_video[], present.d_audio[],
            weights, _config(), ctx,
        )
    except:
        duplicate_rejected = True
    var nonbinary_rejected = False
    try:
        _ = minimax_h3_device_presence_gated_av_mse(
            head.video_prediction[], noised.video_target_rows[],
            head.audio_prediction[], noised.audio_target_rows[],
            scalars[5], Float32(0.5), ctx,
        )
    except:
        nonbinary_rejected = True
    ok = malformed_rejected and duplicate_rejected and nonbinary_rejected and ok
    if not ok:
        raise Error("MiniMax-H3 device AV/head parity FAILED")
    print("PASS: dual shifts + native targets + independent AV roots")
    print("PASS: exact audio gating + malformed/duplicate/nonbinary rejection")
    print("PASS: BF16 training-autocast frozen head -> BF16 50-stack root")
