# MiniMax H3 optional quality-policy parity against a source-executed pinned
# Musubi fixture. This gate is host-only and proves no trainer/device/cache path.
#
# Generate oracle fixture:
#   /home/alex/LTX-2/.venv/bin/python scripts/minimax_h3_quality_policy_oracle.py
# Run gate:
#   pixi run mojo run -I . -I vendor/mojo-libs serenitymojo/training/parity/minimax_h3_quality_policy_parity.mojo

from json.parser import loads
from json.value import JSONValue
from std.collections import List
from std.math import abs
from std.pathlib import Path

from serenitymojo.training.minimax_h3.quality_policy import (
    minimax_h3_conditioned_video_dc_mse,
    minimax_h3_decomposed_flow_loss,
    minimax_h3_guidance_targets,
    minimax_h3_teacher_preservation_compensation,
    minimax_h3_timestep_focus,
    validate_minimax_h3_quality_mode,
)


comptime FIXTURE = "serenitymojo/training/parity/fixtures/minimax_h3_quality_policy_v1.json"
comptime SCHEMA = "serenity.minimax_h3.quality_policy_oracle.v1"
comptime ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime SOURCE_SHA256 = "4554196e24d5d85e9703b79602e8e4e64efc2c9ec2812011e9416192f2b1b99a"


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax H3 quality-policy parity failed: ") + message)


def _close(
    got: Float32, expected: Float32, tolerance: Float32 = Float32(2.0e-5)
) -> Bool:
    return abs(got - expected) <= tolerance


def _json_f32_list(value: JSONValue) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(value.length()):
        out.append(Float32(value[i].as_float()))
    return out^


def _check_list(
    got: List[Float32], expected: List[Float32], label: String
) raises:
    _check(len(got) == len(expected), label + String(" length"))
    for i in range(len(got)):
        _check(
            _close(got[i], expected[i]),
            label + String(" element ") + String(i),
        )


def _guidance_case(case_value: JSONValue, expected_applied: Bool) raises:
    var inputs = case_value[String("inputs")]
    var outputs = case_value[String("outputs")]
    var result = minimax_h3_guidance_targets(
        Float32(inputs[String("base_sigma")].as_float()),
        Float32(inputs[String("sigma_min")].as_float()),
        _json_f32_list(inputs[String("video_velocity")]),
        _json_f32_list(inputs[String("video_uncond")]),
        Float32(inputs[String("video_scale")].as_float()),
        _json_f32_list(inputs[String("audio_velocity")]),
        _json_f32_list(inputs[String("audio_uncond")]),
        Float32(inputs[String("audio_scale")].as_float()),
    )
    _check(result.applied == expected_applied, "guidance gate")
    _check(outputs[String("applied")].as_bool() == expected_applied, "fixture guidance gate")
    _check_list(result.video, _json_f32_list(outputs[String("video")]), "guidance video")
    _check_list(result.audio, _json_f32_list(outputs[String("audio")]), "guidance audio")


def main() raises:
    var doc = loads(Path(String(FIXTURE)).read_text())
    _check(doc[String("schema")].as_string() == String(SCHEMA), "fixture schema")
    _check(doc[String("oracle_commit")].as_string() == String(ORACLE_COMMIT), "oracle commit")
    _check(doc[String("source_sha256")].as_string() == String(SOURCE_SHA256), "pinned source hash")
    _check(doc[String("torch_dtype")].as_string() == String("torch.float32"), "oracle dtype")
    _check(doc[String("guidance_source_contract_verified")].as_bool(), "guidance source contract")
    _check(doc[String("executed_upstream_functions")].length() == 4, "executed source functions")

    var guidance = doc[String("guidance")]
    _guidance_case(guidance[String("applied")], True)
    _guidance_case(guidance[String("skipped")], False)

    var focus = doc[String("focus")]
    var focus_inputs = focus[String("inputs")]
    var focus_draws = _json_f32_list(focus_inputs[String("draws")])
    var focus_outputs = _json_f32_list(focus[String("outputs")])
    for i in range(len(focus_draws)):
        var got = minimax_h3_timestep_focus(
            focus_draws[i],
            Float32(focus_inputs[String("focus_min")].as_float()),
            Float32(focus_inputs[String("focus_max")].as_float()),
            Float32(focus_inputs[String("focus_probability")].as_float()),
        )
        _check(_close(got, focus_outputs[i]), String("focus draw ") + String(i))
    _check(
        minimax_h3_timestep_focus(
            Float32(0.438), Float32(0.3), Float32(0.77), Float32(0.0)
        ) == Float32(0.438),
        "focus disabled identity",
    )

    var preservation = doc[String("preservation")]
    var preservation_inputs = preservation[String("inputs")]
    var compensation = minimax_h3_teacher_preservation_compensation(
        Float32(preservation_inputs[String("sigma_max")].as_float()),
        Float32(preservation_inputs[String("focus_min")].as_float()),
        Float32(preservation_inputs[String("focus_max")].as_float()),
        Float32(preservation_inputs[String("focus_probability")].as_float()),
    )
    _check(
        _close(compensation, Float32(preservation[String("output")].as_float())),
        "preservation compensation",
    )

    var decomposed = doc[String("decomposed")]
    var decomposed_inputs = decomposed[String("inputs")]
    var decomposed_outputs = decomposed[String("outputs")]
    var loss = minimax_h3_decomposed_flow_loss(
        _json_f32_list(decomposed_inputs[String("prediction")]),
        _json_f32_list(decomposed_inputs[String("target")]),
        Float32(decomposed_inputs[String("magnitude_weight")].as_float()),
        Float32(decomposed_inputs[String("direction_weight")].as_float()),
    )
    _check(_close(loss.value, Float32(decomposed_outputs[String("value")].as_float())), "decomposed value")
    _check(_close(loss.magnitude_term, Float32(decomposed_outputs[String("magnitude_term")].as_float())), "magnitude term")
    _check(_close(loss.direction_term, Float32(decomposed_outputs[String("direction_term")].as_float())), "direction term")
    _check_list(
        loss.prediction_gradient,
        _json_f32_list(decomposed_outputs[String("prediction_gradient")]),
        "decomposed analytic gradient",
    )
    var gradient_l1 = Float32(0.0)
    for value in loss.prediction_gradient:
        gradient_l1 += abs(value)
    _check(gradient_l1 > Float32(0.5), "decomposed fixture is nondegenerate")

    var dc = doc[String("dc")]
    var dc_inputs = dc[String("inputs")]
    var dc_outputs = dc[String("outputs")]
    var dc_loss = minimax_h3_conditioned_video_dc_mse(
        _json_f32_list(dc_inputs[String("prediction")]),
        _json_f32_list(dc_inputs[String("target")]),
        dc_inputs[String("batch_channel_groups")].as_int(),
        Float32(dc_inputs[String("dc_weight")].as_float()),
    )
    _check(_close(dc_loss.value, Float32(dc_outputs[String("value")].as_float())), "DC-shaped value")
    _check_list(
        dc_loss.shaped_prediction,
        _json_f32_list(dc_outputs[String("shaped_prediction")]),
        "DC-shaped prediction",
    )
    _check_list(
        dc_loss.prediction_gradient,
        _json_f32_list(dc_outputs[String("prediction_gradient")]),
        "DC analytic gradient",
    )

    var rejected = False
    try:
        validate_minimax_h3_quality_mode(True, True)
    except:
        rejected = True
    _check(rejected, "guidance/teacher mutual exclusion")

    print("MiniMax H3 quality-policy Musubi value/gradient parity PASS")
    print(
        "Evidence: four pinned functions executed; guidance assignments source-verified; no trainer/device/cache launch"
    )
