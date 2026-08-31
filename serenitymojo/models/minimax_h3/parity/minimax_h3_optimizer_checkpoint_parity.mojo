# MiniMax-H3 200-target inventory + four-representative optimizer/private-state gate.
#
# Component-only: synthetic reduced F32 LoRA tensors, shared host AdamW, shared
# F32 trainer-private-state save/load, and the existing H3 FC1 in-memory layout
# mapping. Musubi LoRA weight-file export/import is explicitly NOT exercised.
# No base model, dataset, cache, training loop, UI, or product path is exercised.

from json.parser import loads
from json.value import JSONValue
from max.gpu.host import DeviceContext
from std.collections import List
from std.math import abs
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.lora_save import (
    F32LoraState,
    load_lora_train_state_f32,
    load_lora_train_state_meta,
    lora_train_state_has_f32_masters,
    save_lora_train_state_f32,
)
from serenitymojo.training.minimax_h3.lora_layout import (
    minimax_h3_fc1_lora_up_musubi_to_runtime,
    minimax_h3_fc1_lora_up_runtime_to_musubi,
)
from serenitymojo.training.minimax_h3.lora_surface import minimax_h3_lora_surface
from serenitymojo.training.train_step import _adamw_host_list_f32


comptime FIXTURE = "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_optimizer_checkpoint_v1.json"
comptime FIXTURE_SHA = "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_optimizer_checkpoint_v1.sha256"
comptime FIXTURE_DIGEST = "92cd0e8f8ebc455e6400973705ea7fe37c390440b83a9039dd888fb54040e9d3"
comptime STATE_PATH = "/tmp/serenity_minimax_h3_optimizer_checkpoint_gate.safetensors"
comptime SCHEMA = "serenity.minimax_h3.target_inventory_reduced_optimizer_private_state.v1"
comptime COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime ARCH_SHA = "4c2c4c850f2ad6ad901e9d49088b54128f4e17ee09c88564db88602ede71fe17"
comptime LORA_SHA = "694bcf27bebd8911a7868628ac1bc075d07cc8e87fdb289993b17ecab71475d5"
comptime MODEL_SHA = "500fcacf93b40fac49b1ccbb21d8b382cb1f1b9fbd7954d1ac08155b2d0d243a"
comptime RANK = 2
comptime LR = Float32(3.0e-4)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime EPS = Float32(1.0e-8)
comptime WEIGHT_DECAY = Float32(0.01)


def _require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax-H3 inventory/reduced-optimizer/private-state gate failed: ") + message)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _json_f32_list(value: JSONValue) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(value.length()):
        out.append(Float32(value[i].as_float()))
    return out^


def _gradient(n: Int, adapter: Int, arm: Int, step: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var centered = Float32((i * 13 + adapter * 19 + arm * 5 + step * 23) % 37 - 18)
        out.append(centered * Float32(7.1e-5) + Float32(step) * Float32(2.9e-6))
    return out^


def _state(case_value: JSONValue) raises -> F32LoraState:
    var prefix = case_value[String("musubi_prefix")].as_string()
    var in_f = case_value[String("in_features")].as_int()
    var out_f = case_value[String("out_features")].as_int()
    var na = RANK * in_f
    var nb = out_f * RANK
    var a = _json_f32_list(case_value[String("initial_a")])
    var b = _json_f32_list(case_value[String("initial_b")])
    _require(len(a) == na and len(b) == nb, prefix + String(" oracle initial geometry"))
    return F32LoraState(
        prefix,
        a^,
        b^,
        _zeros(na),
        _zeros(na),
        _zeros(nb),
        _zeros(nb),
        RANK,
        in_f,
        out_f,
    )


def _optimizer_step(mut states: List[F32LoraState], step: Int) raises:
    for i in range(len(states)):
        var ga = _gradient(len(states[i].a), i, 0, step)
        var gb = _gradient(len(states[i].b), i, 1, step)
        _adamw_host_list_f32(
            states[i].a, ga, states[i].ma, states[i].va,
            step, LR, BETA1, BETA2, EPS, WEIGHT_DECAY,
        )
        _adamw_host_list_f32(
            states[i].b, gb, states[i].mb, states[i].vb,
            step, LR, BETA1, BETA2, EPS, WEIGHT_DECAY,
        )


def _bit_equal(left: List[Float32], right: List[Float32], label: String) raises:
    _require(len(left) == len(right), label + String(" length"))
    for i in range(len(left)):
        _require(
            left[i].to_bits[DType.uint32]() == right[i].to_bits[DType.uint32](),
            label + String(" element ") + String(i),
        )


def _torch_close(
    left: List[Float32], right: List[Float32], tolerance: Float32,
    minimum_l1: Float32, label: String,
) raises:
    _require(len(left) == len(right), label + String(" length"))
    _require(_abs_sum(right) > minimum_l1, label + String(" oracle nonzero"))
    _require(_abs_sum(left) > minimum_l1, label + String(" Mojo nonzero"))
    for i in range(len(left)):
        _require(
            abs(left[i] - right[i]) <= tolerance,
            label + String(" element ") + String(i),
        )


def _state_equal(left: F32LoraState, right: F32LoraState, label: String) raises:
    _require(left.prefix == right.prefix, label + String(" prefix"))
    _require(left.rank == right.rank and left.in_f == right.in_f and left.out_f == right.out_f, label + String(" geometry"))
    _bit_equal(left.a, right.a, label + String(" A master"))
    _bit_equal(left.b, right.b, label + String(" B master"))
    _bit_equal(left.ma, right.ma, label + String(" A m"))
    _bit_equal(left.va, right.va, label + String(" A v"))
    _bit_equal(left.mb, right.mb, label + String(" B m"))
    _bit_equal(left.vb, right.vb, label + String(" B v"))


def _abs_sum(values: List[Float32]) -> Float32:
    var total = Float32(0.0)
    for value in values:
        total += abs(value)
    return total


def _check_torch_step(
    states: List[F32LoraState], cases: JSONValue, step: Int,
) raises:
    _require(cases.length() == len(states), "Torch trajectory case count")
    for i in range(len(states)):
        var expected = cases[i][String("steps")][step - 1]
        _require(expected[String("step")].as_int() == step, String("Torch step tag ") + String(step))
        var suffix = String(" step ") + String(step) + String(" adapter ") + String(i)
        _torch_close(states[i].a, _json_f32_list(expected[String("a")]), Float32(3.0e-8), Float32(1.0e-3), String("Torch A") + suffix)
        _torch_close(states[i].b, _json_f32_list(expected[String("b")]), Float32(3.0e-8), Float32(1.0e-3), String("Torch B") + suffix)
        _torch_close(states[i].ma, _json_f32_list(expected[String("ma")]), Float32(3.0e-10), Float32(1.0e-7), String("Torch ma") + suffix)
        _torch_close(states[i].va, _json_f32_list(expected[String("va")]), Float32(3.0e-12), Float32(1.0e-11), String("Torch va") + suffix)
        _torch_close(states[i].mb, _json_f32_list(expected[String("mb")]), Float32(3.0e-10), Float32(1.0e-7), String("Torch mb") + suffix)
        _torch_close(states[i].vb, _json_f32_list(expected[String("vb")]), Float32(3.0e-12), Float32(1.0e-11), String("Torch vb") + suffix)


def _resume_next_step(meta: List[Float32], expected_saved_step: Int) raises -> Int:
    if len(meta) < 1:
        raise Error("MiniMax-H3 private state is missing optimizer step metadata")
    var saved_step = Int(meta[0])
    if Float32(saved_step) != meta[0]:
        raise Error("MiniMax-H3 private-state optimizer step is not integral")
    if saved_step != expected_saved_step:
        raise Error(
            String("MiniMax-H3 private-state optimizer step mismatch: expected ")
            + String(expected_saved_step) + String(", got ") + String(saved_step)
        )
    return saved_step + 1


def _check_inventory(doc: JSONValue) raises -> List[String]:
    var oracle = doc[String("inventory")]
    var surface = minimax_h3_lora_surface()
    _require(oracle.length() == 200, "pinned inventory count")
    _require(len(surface) == 200, "product inventory count")
    var prefixes = List[String]()
    var fc1_count = 0
    for i in range(200):
        var expected = oracle[i]
        var target = surface[i].copy()
        _require(target.layer == expected[String("layer")].as_int(), String("layer at target ") + String(i))
        _require(target.family == expected[String("family")].as_string(), String("family at target ") + String(i))
        _require(target.module_path == expected[String("module_path")].as_string(), String("module path at target ") + String(i))
        _require(target.musubi_prefix == expected[String("musubi_prefix")].as_string(), String("Musubi prefix at target ") + String(i))
        _require(target.in_features == expected[String("in_features")].as_int(), String("input width at target ") + String(i))
        _require(target.out_features == expected[String("out_features")].as_int(), String("output width at target ") + String(i))
        _require(target.fc1_up_requires_runtime_swap == expected[String("fc1_runtime_swap")].as_bool(), String("FC1 swap at target ") + String(i))
        _require(target.down_key() == target.musubi_prefix + String(".lora_down.weight"), String("down key at target ") + String(i))
        _require(target.up_key() == target.musubi_prefix + String(".lora_up.weight"), String("up key at target ") + String(i))
        _require(target.alpha_key() == target.musubi_prefix + String(".alpha"), String("alpha key at target ") + String(i))
        prefixes.append(target.musubi_prefix)
        if target.fc1_up_requires_runtime_swap:
            fc1_count += 1
        for j in range(i):
            _require(prefixes[j] != prefixes[i], String("duplicate target prefix ") + String(i))
    _require(fc1_count == 50, "exact FC1 swap inventory")
    return prefixes^


def _check_fc1_raw_mapping(runtime: List[Float32]) raises:
    comptime FF = 3
    var raw = minimax_h3_fc1_lora_up_runtime_to_musubi(runtime, FF, RANK)
    for row in range(FF):
        for col in range(RANK):
            _require(raw[row * RANK + col] == runtime[(FF + row) * RANK + col], "FC1 raw gate row")
            _require(raw[(FF + row) * RANK + col] == runtime[row * RANK + col], "FC1 raw value row")
    var roundtrip = minimax_h3_fc1_lora_up_musubi_to_runtime(raw, FF, RANK)
    _bit_equal(roundtrip, runtime, "FC1 in-memory runtime/raw roundtrip")


def _check_private_state_dtypes(prefixes: List[String]) raises:
    var st = SafeTensors.open(String(STATE_PATH))
    for ref prefix in prefixes:
        for suffix in [String(".lora_A.weight"), String(".lora_B.weight")]:
            _require(st.tensor_info(prefix + suffix).dtype == STDtype.BF16, prefix + suffix + String(" dtype"))
        for suffix in [
            String(".lora_A.master"), String(".lora_B.master"),
            String(".lora_A.adam_m"), String(".lora_A.adam_v"),
            String(".lora_B.adam_m"), String(".lora_B.adam_v"),
        ]:
            _require(st.tensor_info(prefix + suffix).dtype == STDtype.F32, prefix + suffix + String(" dtype"))
    _require(st.tensor_info(String("__meta__")).dtype == STDtype.F32, "step metadata dtype")


def main() raises:
    _require(
        Path(String(FIXTURE_SHA)).read_text()
        == String(FIXTURE_DIGEST) + String("  minimax_h3_optimizer_checkpoint_v1.json\n"),
        "fixture digest sidecar",
    )
    var doc = loads(Path(String(FIXTURE)).read_text())
    _require(doc.length() == 10, "fixture top-level key count")
    _require(doc[String("schema")].as_string() == String(SCHEMA), "fixture schema")
    _require(doc[String("oracle_commit")].as_string() == String(COMMIT), "oracle commit")
    var hashes = doc[String("source_sha256")]
    _require(hashes[String("networks/lora_minimax_h3.py")].as_string() == String(ARCH_SHA), "architecture source hash")
    _require(hashes[String("networks/lora.py")].as_string() == String(LORA_SHA), "LoRA source hash")
    _require(hashes[String("minimax_h3/model.py")].as_string() == String(MODEL_SHA), "model source hash")
    var dtype_probe = doc[String("executed_dtype_probe")]
    _require(dtype_probe[String("base")].as_string() == String("torch.bfloat16"), "oracle base dtype")
    _require(dtype_probe[String("lora_down")].as_string() == String("torch.float32"), "oracle LoRA-down dtype")
    _require(dtype_probe[String("lora_up")].as_string() == String("torch.float32"), "oracle LoRA-up dtype")
    _require(dtype_probe[String("lora_up_zero_initialized")].as_bool(), "oracle zero-up initialization")
    var excluded = doc[String("excluded_evidence")]
    _require(excluded.length() == 3, "explicit exclusion count")
    _require(excluded[0].as_string() == String("Musubi LoRA weight-file export/import"), "weight-file export/import exclusion")

    var trajectory = doc[String("optimizer_trajectory")]
    _require(trajectory[String("oracle")].as_string() == String("torch.optim.AdamW foreach=False fused=False"), "Torch optimizer oracle")
    _require(trajectory[String("rank")].as_int() == RANK, "Torch reduced rank")
    _require(abs(Float32(trajectory[String("learning_rate")].as_float()) - LR) <= Float32(1.0e-10), "Torch learning rate")
    _require(abs(Float32(trajectory[String("beta1")].as_float()) - BETA1) <= Float32(1.0e-7), "Torch beta1")
    _require(abs(Float32(trajectory[String("beta2")].as_float()) - BETA2) <= Float32(1.0e-7), "Torch beta2")
    _require(abs(Float32(trajectory[String("eps")].as_float()) - EPS) <= Float32(1.0e-12), "Torch epsilon")
    _require(abs(Float32(trajectory[String("weight_decay")].as_float()) - WEIGHT_DECAY) <= Float32(1.0e-9), "Torch weight decay")
    var trajectory_cases = trajectory[String("cases")]
    _require(trajectory_cases.length() == 4, "four representative Torch trajectories")

    var all_prefixes = _check_inventory(doc)
    var representative_prefixes = List[String]()
    for i in range(4):
        representative_prefixes.append(all_prefixes[i])
    var states = List[F32LoraState]()
    var replay = List[F32LoraState]()
    for i in range(4):
        _require(trajectory_cases[i][String("musubi_prefix")].as_string() == representative_prefixes[i], String("trajectory prefix ") + String(i))
        states.append(_state(trajectory_cases[i]))
        replay.append(_state(trajectory_cases[i]))
    var initial_norm = Float32(0.0)
    for i in range(len(states)):
        initial_norm += _abs_sum(states[i].a) + _abs_sum(states[i].b)
    _optimizer_step(states, 1)
    _optimizer_step(replay, 1)
    for i in range(4):
        _state_equal(states[i], replay[i], String("same-op replay step 1 adapter ") + String(i))
    _check_torch_step(states, trajectory_cases, 1)
    _optimizer_step(states, 2)
    _optimizer_step(replay, 2)
    for i in range(4):
        _state_equal(states[i], replay[i], String("same-op replay step 2 adapter ") + String(i))
    _check_torch_step(states, trajectory_cases, 2)
    var updated_norm = Float32(0.0)
    var moment_norm = Float32(0.0)
    for i in range(len(states)):
        updated_norm += _abs_sum(states[i].a) + _abs_sum(states[i].b)
        moment_norm += _abs_sum(states[i].ma) + _abs_sum(states[i].va) + _abs_sum(states[i].mb) + _abs_sum(states[i].vb)
    _require(abs(updated_norm - initial_norm) > Float32(1.0e-4), "nondegenerate two-step parameter update")
    _require(moment_norm > Float32(1.0e-5), "nondegenerate AdamW moments")
    _check_fc1_raw_mapping(states[2].b)

    var meta = List[Float32]()
    meta.append(Float32(2.0))
    meta.append(Float32(1029.0))
    var ctx = DeviceContext()
    _require(save_lora_train_state_f32(states, String(STATE_PATH), ctx, meta^) == 4, "saved adapter count")
    _require(lora_train_state_has_f32_masters(String(STATE_PATH), representative_prefixes[0]), "F32 master probe")
    _check_private_state_dtypes(representative_prefixes)
    var loaded_meta = load_lora_train_state_meta(String(STATE_PATH), ctx)
    _require(len(loaded_meta) == 2, "private-state metadata length")
    _require(loaded_meta[1] == Float32(1029.0), "saved deterministic seed")
    var missing_rejected = False
    try:
        _ = _resume_next_step(List[Float32](), 2)
    except:
        missing_rejected = True
    _require(missing_rejected, "missing optimizer step metadata rejection")
    var wrong_meta = List[Float32]()
    wrong_meta.append(Float32(1.0))
    var wrong_rejected = False
    try:
        _ = _resume_next_step(wrong_meta, 2)
    except:
        wrong_rejected = True
    _require(wrong_rejected, "wrong optimizer step metadata rejection")
    var next_step = _resume_next_step(loaded_meta, 2)
    _require(next_step == 3, "restored next optimizer step")
    var resumed = load_lora_train_state_f32(representative_prefixes, String(STATE_PATH), ctx)
    _require(len(resumed) == 4, "loaded adapter count")
    for i in range(4):
        _state_equal(states[i], resumed[i], String("save/load state ") + String(i))
    _optimizer_step(states, next_step)
    _optimizer_step(resumed, next_step)
    _check_torch_step(states, trajectory_cases, next_step)
    _check_torch_step(resumed, trajectory_cases, next_step)
    for i in range(4):
        _state_equal(states[i], resumed[i], String("resumed step equality ") + String(i))

    print("MiniMax-H3 target-inventory/reduced-optimizer/private-state component gate PASS")
    print("Evidence: 200-name inventory; four reduced adapters vs pinned Torch AdamW; private F32 state resume")
    print("Excluded: Musubi LoRA weight-file export/import, full 200-target update, model/dataset/cache/trainer/product")
