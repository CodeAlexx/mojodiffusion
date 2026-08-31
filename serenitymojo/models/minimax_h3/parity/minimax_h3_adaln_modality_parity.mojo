# MiniMax-H3 AdaLN/modality-segment parity against source-executed Musubi.
#
# Generate:
#   /home/alex/LTX-2/.venv/bin/python serenitymojo/models/minimax_h3/parity/minimax_h3_adaln_modality_oracle.py
# Run (serialized with every other Mojo compile):
#   pixi run mojo run -I . -I vendor/mojo-libs serenitymojo/models/minimax_h3/parity/minimax_h3_adaln_modality_parity.mojo
#
# Bounded evidence only: host F32 projection/selection/modulation for synthetic
# packed rows. This does not prove a full block, device kernel, cache, or trainer.

from json.parser import loads
from json.value import JSONValue
from std.collections import List
from std.math import abs, exp
from std.pathlib import Path

from serenitymojo.models.minimax_h3.block_forward import linear_bias


comptime FIXTURE = "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_adaln_modality_v1.json"
comptime FIXTURE_SHA = "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_adaln_modality_v1.sha256"
comptime FIXTURE_DIGEST = "e40759d3b6bd04bd300116e4b8927721142a7112ef27f4c7c61dcfad1a702a2b"
comptime SCHEMA = "serenity.minimax_h3.adaln_modality_oracle.v1"
comptime ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime MODEL_SHA256 = "500fcacf93b40fac49b1ccbb21d8b382cb1f1b9fbd7954d1ac08155b2d0d243a"
comptime PACKING_SHA256 = "464371faca4f156de883ce37022533c0fd3e0965723648c904d2f1cc09be2cc3"


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax-H3 AdaLN/modality parity failed: ") + message)


def _json_f32_list(value: JSONValue) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(value.length()):
        out.append(Float32(value[i].as_float()))
    return out^


def _json_int_list(value: JSONValue) raises -> List[Int]:
    var out = List[Int]()
    for i in range(value.length()):
        out.append(value[i].as_int())
    return out^


def _check_int_list(got: List[Int], expected: List[Int], label: String) raises:
    _check(len(got) == len(expected), label + String(" length"))
    for i in range(len(got)):
        _check(got[i] == expected[i], label + String(" element ") + String(i))


def _check_f32_list(
    got: List[Float32], expected: List[Float32], label: String
) raises:
    _check(len(got) == len(expected), label + String(" length"))
    for i in range(len(got)):
        _check(
            abs(got[i] - expected[i]) <= Float32(3.0e-5),
            label + String(" element ") + String(i),
        )


def _silu(value: Float32) -> Float32:
    return value / (Float32(1.0) + exp(-value))


def _expected_indices(task: String) raises -> List[Int]:
    var out = List[Int]()
    if task == String("t2va"):
        for value in [0, 1, 0]:
            out.append(value)
        for _ in range(16):
            out.append(5)
        for _ in range(2):
            out.append(0)
    elif task == String("fl2va"):
        for value in [1, 0, 6, 6]:
            out.append(value)
        for _ in range(16):
            out.append(5)
        for _ in range(2):
            out.append(0)
    elif task == String("ref2va"):
        for value in [1, 0, 1, 6]:
            out.append(value)
        for _ in range(4):
            out.append(11)
        for _ in range(16):
            out.append(5)
        for _ in range(2):
            out.append(0)
    else:
        raise Error(String("unexpected task ") + task)
    return out^


def _verify_task_shape(case_value: JSONValue, task: String) raises:
    var kinds = case_value[String("segment_kinds")]
    var lengths = _json_int_list(case_value[String("segment_lengths")])
    if task == String("t2va"):
        _check(kinds.length() == 3, "T2VA segment count")
        _check(kinds[0].as_string() == String("text"), "T2VA text branch")
        _check(kinds[1].as_string() == String("target_audio"), "T2VA audio branch")
        _check(kinds[2].as_string() == String("target_video"), "T2VA video branch")
        _check_int_list(lengths, [3, 16, 2], "T2VA segment lengths")
    elif task == String("fl2va"):
        _check(kinds.length() == 5, "FL2VA segment count")
        _check(kinds[1].as_string() == String("visual_condition"), "FL2VA first branch")
        _check(kinds[2].as_string() == String("visual_condition"), "FL2VA last branch")
        _check(kinds[3].as_string() == String("target_audio"), "FL2VA audio branch")
        _check(kinds[4].as_string() == String("target_video"), "FL2VA video branch")
        _check_int_list(lengths, [2, 1, 1, 16, 2], "FL2VA segment lengths")
        _check(lengths[1] == lengths[2], "FL2VA first/last row counts match")
    else:
        _check(kinds.length() == 5, "Ref2VA segment count")
        _check(kinds[1].as_string() == String("visual_condition"), "Ref2VA visual branch")
        _check(kinds[2].as_string() == String("audio_condition"), "Ref2VA audio branch")
        _check(kinds[3].as_string() == String("target_audio"), "Ref2VA target audio branch")
        _check(kinds[4].as_string() == String("target_video"), "Ref2VA target video branch")
        _check_int_list(lengths, [3, 1, 4, 16, 2], "Ref2VA segment lengths")
        _check(lengths[2] % 2 == 0, "Ref2VA audio-condition rows are stereo-even")


def _run_case(
    case_value: JSONValue,
    weight: List[Float32],
    bias: List[Float32],
    hidden_size: Int,
    timestep_size: Int,
    expand_count: Int,
    modality_count: Int,
) raises:
    var task = case_value[String("task")].as_string()
    _verify_task_shape(case_value, task)
    var indices = _json_int_list(case_value[String("block_adaln_indices")])
    _check_int_list(indices, _expected_indices(task), task + String(" AdaLN indices"))

    var temb = _json_f32_list(case_value[String("timestep_embeddings")])
    var timestep_count = len(temb) // timestep_size
    var activated = List[Float32]()
    for value in temb:
        activated.append(_silu(value))
    var projection_width = expand_count * modality_count * hidden_size
    var projected = linear_bias(
        activated,
        timestep_count,
        timestep_size,
        weight,
        bias,
        projection_width,
    )

    var selected = List[Float32]()
    for r in range(len(indices)):
        var adaln_row = indices[r]
        var timestep_row = adaln_row // modality_count
        var modality_row = adaln_row % modality_count
        for parameter in range(expand_count):
            for d in range(hidden_size):
                selected.append(
                    projected[
                        timestep_row * projection_width
                        + modality_row * expand_count * hidden_size
                        + parameter * hidden_size
                        + d
                    ]
                )
    _check_f32_list(
        selected,
        _json_f32_list(case_value[String("selected_parameters")]),
        task + String(" projection reshape/chunk/segment expansion"),
    )

    var hidden = _json_f32_list(case_value[String("hidden")])
    var update = _json_f32_list(case_value[String("update")])
    var mod_msa = List[Float32]()
    var gate_msa = List[Float32]()
    var mod_mlp = List[Float32]()
    var gate_mlp = List[Float32]()
    for r in range(len(indices)):
        for d in range(hidden_size):
            var value_index = r * hidden_size + d
            var parameter_base = (r * expand_count) * hidden_size + d
            var value = hidden[value_index]
            var delta = update[value_index]
            mod_msa.append(
                value * (Float32(1.0) + selected[parameter_base + hidden_size])
                + selected[parameter_base]
            )
            gate_msa.append(
                value + delta * selected[parameter_base + 2 * hidden_size]
            )
            mod_mlp.append(
                value * (Float32(1.0) + selected[parameter_base + 4 * hidden_size])
                + selected[parameter_base + 3 * hidden_size]
            )
            gate_mlp.append(
                value + delta * selected[parameter_base + 5 * hidden_size]
            )
    _check_f32_list(mod_msa, _json_f32_list(case_value[String("mod_scale_shift")]), task + String(" _mod_scale_shift MSA"))
    _check_f32_list(gate_msa, _json_f32_list(case_value[String("mod_gate")]), task + String(" _mod_gate MSA"))
    _check_f32_list(mod_mlp, _json_f32_list(case_value[String("mod_scale_shift_mlp")]), task + String(" _mod_scale_shift MLP"))
    _check_f32_list(gate_mlp, _json_f32_list(case_value[String("mod_gate_mlp")]), task + String(" _mod_gate MLP"))


def main() raises:
    _check(
        Path(String(FIXTURE_SHA)).read_text()
        == String(FIXTURE_DIGEST) + String("  minimax_h3_adaln_modality_v1.json\n"),
        "fixture digest sidecar",
    )
    var doc = loads(Path(String(FIXTURE)).read_text())
    _check(doc[String("schema")].as_string() == String(SCHEMA), "fixture schema")
    _check(doc[String("oracle_commit")].as_string() == String(ORACLE_COMMIT), "oracle commit")
    var hashes = doc[String("source_sha256")]
    _check(hashes[String("model.py")].as_string() == String(MODEL_SHA256), "model.py hash")
    _check(hashes[String("packing.py")].as_string() == String(PACKING_SHA256), "packing.py hash")
    _check(doc[String("dtype")].as_string() == String("torch.float32"), "oracle dtype")
    _check(doc[String("executed_upstream_definitions")].length() == 11, "executed definitions")
    var dimensions = doc[String("dimensions")]
    var hidden_size = dimensions[String("hidden")].as_int()
    var timestep_size = dimensions[String("timestep")].as_int()
    var expand_count = dimensions[String("expand")].as_int()
    var modality_count = dimensions[String("modalities")].as_int()
    _check(hidden_size == 4, "hidden dimension")
    _check(timestep_size == 3, "timestep dimension")
    _check(expand_count == 6, "six DiTBlock parameters")
    _check(modality_count == 3, "three AdaLN modality rows")
    var weight = _json_f32_list(doc[String("adaln_weight")])
    var bias = _json_f32_list(doc[String("adaln_bias")])
    var cases = doc[String("cases")]
    _check(cases.length() == 3, "task case count")
    _check(cases[0][String("task")].as_string() == String("t2va"), "T2VA case order")
    _check(cases[1][String("task")].as_string() == String("fl2va"), "FL2VA case order")
    _check(cases[2][String("task")].as_string() == String("ref2va"), "Ref2VA case order")
    for i in range(cases.length()):
        _run_case(cases[i], weight, bias, hidden_size, timestep_size, expand_count, modality_count)
    print("MiniMax-H3 AdaLN/modality-segment Musubi parity PASS")
    print("Evidence: pinned source-executed host F32 projection + T2VA/FL2VA/Ref2VA slice modulation only")
