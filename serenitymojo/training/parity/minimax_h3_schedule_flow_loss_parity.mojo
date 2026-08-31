# MiniMax H3 schedule and native flow-loss parity against a pinned Musubi
# fixture. The fixture generator executes upstream `_shift_noise_amount` from
# the immutable commit and uses Torch F32 for noising, targets, and mean MSE.
#
# Generate:
#   python3 scripts/minimax_h3_training_schedule_oracle.py
# Run:
#   pixi run mojo run -I . -I vendor/mojo-libs serenitymojo/training/parity/minimax_h3_schedule_flow_loss_parity.mojo

from json.parser import loads
from json.value import JSONValue
from std.collections import List
from std.math import abs
from std.pathlib import Path

from serenitymojo.training.minimax_h3.loss import minimax_h3_presence_gated_av_mse
from serenitymojo.training.minimax_h3.schedule import (
    minimax_h3_native_targets,
    minimax_h3_noisy_values,
    minimax_h3_schedule_point,
)


comptime FIXTURE = "serenitymojo/training/parity/fixtures/minimax_h3_schedule_flow_loss_v1.json"
comptime SCHEMA = "serenity.minimax_h3.schedule_flow_loss_oracle.v1"
comptime ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax H3 schedule/flow-loss parity failed: ") + message)


def _close(a: Float32, b: Float32, tolerance: Float32 = Float32(1.0e-6)) -> Bool:
    return abs(a - b) <= tolerance


def _json_f32_list(value: JSONValue) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(value.length()):
        out.append(Float32(value[i].as_float()))
    return out^


def _check_list(got: List[Float32], want: List[Float32], label: String) raises:
    _check(len(got) == len(want), label + String(" length"))
    for i in range(len(got)):
        _check(_close(got[i], want[i]), label + String(" element ") + String(i))


def main() raises:
    var doc = loads(Path(String(FIXTURE)).read_text())
    _check(doc[String("schema")].as_string() == String(SCHEMA), "fixture schema")
    _check(
        doc[String("oracle_commit")].as_string() == String(ORACLE_COMMIT),
        "oracle commit",
    )
    _check(doc[String("torch_dtype")].as_string() == String("torch.float32"), "oracle dtype")
    _check(doc[String("source_sha256")].as_string().byte_length() == 64, "source hash")

    var inputs = doc[String("inputs")]
    var outputs = doc[String("outputs")]
    var point = minimax_h3_schedule_point(
        Float32(inputs[String("base_sigma")].as_float()),
        Float32(inputs[String("video_shift")].as_float()),
        Float32(inputs[String("audio_shift")].as_float()),
    )
    _check(_close(point.sigma_video, Float32(outputs[String("sigma_video")].as_float())), "video sigma")
    _check(_close(point.sigma_audio, Float32(outputs[String("sigma_audio")].as_float())), "audio sigma")
    _check(_close(point.model_t_video, Float32(outputs[String("model_t_video")].as_float())), "video model_t")
    _check(_close(point.model_t_audio, Float32(outputs[String("model_t_audio")].as_float())), "audio model_t")

    var video_latent = _json_f32_list(inputs[String("video_latent")])
    var video_noise = _json_f32_list(inputs[String("video_noise")])
    var audio_latent = _json_f32_list(inputs[String("audio_latent")])
    var audio_noise = _json_f32_list(inputs[String("audio_noise")])
    _check_list(
        minimax_h3_noisy_values(video_latent, video_noise, point.sigma_video),
        _json_f32_list(outputs[String("noisy_video")]),
        "noisy video",
    )
    _check_list(
        minimax_h3_noisy_values(audio_latent, audio_noise, point.sigma_audio),
        _json_f32_list(outputs[String("noisy_audio")]),
        "noisy audio",
    )
    _check_list(
        minimax_h3_native_targets(video_latent, video_noise),
        _json_f32_list(outputs[String("video_target")]),
        "video latent-noise target",
    )
    _check_list(
        minimax_h3_native_targets(audio_latent, audio_noise),
        _json_f32_list(outputs[String("audio_target")]),
        "audio latent-noise target",
    )

    var loss = minimax_h3_presence_gated_av_mse(
        _json_f32_list(inputs[String("video_prediction")]),
        _json_f32_list(inputs[String("video_loss_target")]),
        _json_f32_list(inputs[String("audio_prediction")]),
        _json_f32_list(inputs[String("audio_loss_target")]),
        Float32(inputs[String("audio_weight")].as_float()),
        Float32(1.0),
    )
    _check(_close(loss.video, Float32(outputs[String("video_loss")].as_float())), "video mean MSE")
    _check(_close(loss.audio, Float32(outputs[String("audio_loss")].as_float())), "audio mean MSE")
    _check(_close(loss.total, Float32(outputs[String("total_loss")].as_float())), "weighted AV total")

    print("MiniMax H3 schedule/flow-loss Musubi parity PASS")
