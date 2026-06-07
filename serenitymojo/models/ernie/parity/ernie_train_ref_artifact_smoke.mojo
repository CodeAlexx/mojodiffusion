# ernie_train_ref_artifact_smoke.mojo -- Ernie OneTrainer one-step artifact gate.
#
# Header and scalar-payload, no-CUDA gate. It opens the real local OneTrainer
# Ernie one-step dump, adapter dump, and metadata JSON and validates tensor
# names, shapes, dtypes, byte sizes, scalar payloads, and zero-lr optimizer
# state-init metadata. It deliberately does not claim transformer forward,
# backward, or AdamW numeric parity.

from std.collections import List
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors


comptime STEP_DUMP = "/home/alex/onetrainer-mojo/parity/ernie_train_ref_step000.safetensors"
comptime ADAPTER_DUMP = "/home/alex/onetrainer-mojo/parity/ernie_train_ref_step000_adapters.safetensors"
comptime META_JSON = "/home/alex/onetrainer-mojo/parity/ernie_train_ref_meta.json"

comptime RANK = 16
comptime HIDDEN = 4096
comptime FFN = 12288
comptime LAYERS = 36
comptime ADAPTER_MODULES = 7
comptime ADAPTER_PHASES = 4
comptime ADAPTER_TRAINABLE_TENSORS = LAYERS * ADAPTER_MODULES * 2
comptime ADAPTER_DUMP_KEYS = ADAPTER_PHASES * ADAPTER_TRAINABLE_TENSORS
comptime TRAINABLE_PARAMETER_ENTRIES = ADAPTER_TRAINABLE_TENSORS
comptime adapter_after_minus_post_tolerance = Float32(0.0)


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _require_contains(text: String, needle: String, label: String) raises:
    _require(text.find(needle) >= 0, String("missing Ernie OT meta field: ") + label)


def _shape0() -> List[Int]:
    return List[Int]()


def _shape1(a: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    return out^


def _shape2(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    return out^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    return out^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    return out^


def _shape_eq(got: List[Int], expected: List[Int]) -> Bool:
    if len(got) != len(expected):
        return False
    for i in range(len(expected)):
        if got[i] != expected[i]:
            return False
    return True


def _numel(shape: List[Int]) -> Int:
    if len(shape) == 0:
        return 1
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _shape_text(shape: List[Int]) -> String:
    var s = String("[")
    for i in range(len(shape)):
        if i > 0:
            s += String(",")
        s += String(shape[i])
    s += String("]")
    return s^


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _read_f32_value(st: SafeTensors, key: String, index: Int) raises -> Float32:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.F32, String("expected F32 payload for ") + key)
    _require(index >= 0 and index < info.size // 4, String("F32 index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    var fp = bytes.unsafe_ptr().bitcast[Float32]()
    return fp[index]


def _read_i32_value(st: SafeTensors, key: String, index: Int) raises -> Int32:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.I32, String("expected I32 payload for ") + key)
    _require(index >= 0 and index < info.size // 4, String("I32 index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    var ip = bytes.unsafe_ptr().bitcast[Int32]()
    return ip[index]


def _require_close(name: String, got: Float32, expected: Float32, tolerance: Float32) raises:
    _require(
        _abs(got - expected) <= tolerance,
        name + String(" value mismatch got=") + String(got) + String(" expected=") + String(expected),
    )


def _require_i32(name: String, got: Int32, expected: Int32) raises:
    _require(
        got == expected,
        name + String(" value mismatch got=") + String(got) + String(" expected=") + String(expected),
    )


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    _require(key in st.tensors, String("missing Ernie OT tensor ") + key)
    var info = st.tensor_info(key)
    _require(
        info.dtype == dtype,
        String("dtype mismatch for ")
        + key
        + String(" got=")
        + info.dtype.name()
        + String(" expected=")
        + dtype.name(),
    )
    _require(
        _shape_eq(info.shape, expected_shape),
        String("shape mismatch for ")
        + key
        + String(" got=")
        + _shape_text(info.shape)
        + String(" expected=")
        + _shape_text(expected_shape),
    )
    _require(
        info.size == _numel(expected_shape) * dtype.byte_size(),
        String("byte-size mismatch for ") + key,
    )


def _adapter_key(phase: String, layer: Int, module: String, suffix: String) -> String:
    return (
        phase
        + String(".transformer.layers.")
        + String(layer)
        + String(".")
        + module
        + String(".")
        + suffix
    )


def _max_abs(a: Float32, b: Float32) -> Float32:
    var aa = _abs(a)
    var bb = _abs(b)
    if bb > aa:
        return bb
    return aa


def _require_after_minus_post_key(
    st: SafeTensors, post_key: String, after_key: String, label: String
) raises -> Float32:
    var post_info = st.tensor_info(post_key)
    var after_info = st.tensor_info(after_key)
    _require(
        _shape_eq(post_info.shape, after_info.shape),
        String("adapter_after - adapter_post shape mismatch for ") + label,
    )
    _require(
        post_info.size == after_info.size,
        String("adapter_after - adapter_post byte-size mismatch for ") + label,
    )
    var last = post_info.size // 4 - 1
    var after_minus_post = _read_f32_value(st, after_key, 0) - _read_f32_value(st, post_key, 0)
    var last_after_minus_post = _read_f32_value(st, after_key, last) - _read_f32_value(st, post_key, last)
    _require_close(
        label + String(" after_minus_post[0]"),
        after_minus_post,
        Float32(0.0),
        adapter_after_minus_post_tolerance,
    )
    _require_close(
        label + String(" adapter_after - adapter_post[last]"),
        last_after_minus_post,
        Float32(0.0),
        adapter_after_minus_post_tolerance,
    )
    return _max_abs(after_minus_post, last_after_minus_post)


def _require_adapter_pair_after_minus_post(st: SafeTensors, layer: Int, module: String) raises -> Float32:
    var post_down = _adapter_key(String("adapter_post"), layer, module, String("lora_down.weight"))
    var after_down = _adapter_key(String("adapter_after"), layer, module, String("lora_down.weight"))
    var post_up = _adapter_key(String("adapter_post"), layer, module, String("lora_up.weight"))
    var after_up = _adapter_key(String("adapter_after"), layer, module, String("lora_up.weight"))
    var down_max = _require_after_minus_post_key(
        st,
        post_down,
        after_down,
        String("layer") + String(layer) + String(".") + module + String(".down"),
    )
    var up_max = _require_after_minus_post_key(
        st,
        post_up,
        after_up,
        String("layer") + String(layer) + String(".") + module + String(".up"),
    )
    if up_max > down_max:
        return up_max
    return down_max


def _require_adapter_layer_after_minus_post(st: SafeTensors, layer: Int) raises -> Float32:
    var max_abs = Float32(0.0)
    var diff = _require_adapter_pair_after_minus_post(st, layer, String("self_attention.to_q"))
    if diff > max_abs:
        max_abs = diff
    diff = _require_adapter_pair_after_minus_post(st, layer, String("self_attention.to_k"))
    if diff > max_abs:
        max_abs = diff
    diff = _require_adapter_pair_after_minus_post(st, layer, String("self_attention.to_v"))
    if diff > max_abs:
        max_abs = diff
    diff = _require_adapter_pair_after_minus_post(st, layer, String("self_attention.to_out.0"))
    if diff > max_abs:
        max_abs = diff
    diff = _require_adapter_pair_after_minus_post(st, layer, String("mlp.gate_proj"))
    if diff > max_abs:
        max_abs = diff
    diff = _require_adapter_pair_after_minus_post(st, layer, String("mlp.up_proj"))
    if diff > max_abs:
        max_abs = diff
    diff = _require_adapter_pair_after_minus_post(st, layer, String("mlp.linear_fc2"))
    if diff > max_abs:
        max_abs = diff
    return max_abs


def _require_adapter_pair(
    st: SafeTensors, phase: String, layer: Int, module: String, in_f: Int, out_f: Int
) raises:
    _require_tensor(
        st,
        _adapter_key(phase, layer, module, String("lora_down.weight")),
        STDtype.F32,
        _shape2(RANK, in_f),
    )
    _require_tensor(
        st,
        _adapter_key(phase, layer, module, String("lora_up.weight")),
        STDtype.F32,
        _shape2(out_f, RANK),
    )


def _require_adapter_layer_phase(st: SafeTensors, phase: String, layer: Int) raises:
    _require_adapter_pair(st, phase, layer, String("self_attention.to_q"), HIDDEN, HIDDEN)
    _require_adapter_pair(st, phase, layer, String("self_attention.to_k"), HIDDEN, HIDDEN)
    _require_adapter_pair(st, phase, layer, String("self_attention.to_v"), HIDDEN, HIDDEN)
    _require_adapter_pair(st, phase, layer, String("self_attention.to_out.0"), HIDDEN, HIDDEN)
    _require_adapter_pair(st, phase, layer, String("mlp.gate_proj"), HIDDEN, FFN)
    _require_adapter_pair(st, phase, layer, String("mlp.up_proj"), HIDDEN, FFN)
    _require_adapter_pair(st, phase, layer, String("mlp.linear_fc2"), FFN, HIDDEN)


def _check_meta_json() raises:
    var meta = Path(String(META_JSON)).read_text()
    _require(meta.byte_length() > 0, String("empty Ernie OneTrainer metadata JSON"))
    _require_contains(meta, String("\"prefix\": \"ernie_train_ref\""), String("prefix"))
    _require_contains(meta, String("\"max_steps\": 1"), String("max_steps"))
    _require_contains(meta, String("\"adapter_dump\": \"step\""), String("adapter_dump"))
    _require_contains(meta, String("\"optimizer\": \"ADAMW\""), String("optimizer ADAMW"))
    _require_contains(meta, String("\"count\": 504"), String("trainable_parameters.count"))
    _require_contains(meta, String("\"step_index\": 0"), String("steps[0].step_index"))
    _require_contains(meta, String("\"lr_before\": [\n        0.0"), String("lr_before 0.0"))
    _require_contains(
        meta,
        String("\"lr_after\": [\n        1.4999999999999998e-06"),
        String("lr_after scheduler post-step"),
    )
    _require_contains(meta, String("\"optimizer_before\": {"), String("optimizer_before"))
    _require_contains(meta, String("\"optimizer_after\": {"), String("optimizer_after"))
    _require_contains(meta, String("\"parameter_entries\": 0"), String("optimizer_before parameter_entries"))
    _require_contains(
        meta,
        String("\"parameter_entries\": ") + String(TRAINABLE_PARAMETER_ENTRIES),
        String("optimizer_after parameter_entries"),
    )


def _check_step_dump() raises:
    var st = SafeTensors.open(String(STEP_DUMP))
    _require(len(st.tensors) == 25, String("expected 25 Ernie step tensors"))

    _require_tensor(st, String("batch.latent_image"), STDtype.BF16, _shape4(2, 32, 80, 56))
    _require_tensor(st, String("batch.loss_weight"), STDtype.F32, _shape1(2))
    _require_tensor(st, String("batch.text_encoder_hidden_state"), STDtype.BF16, _shape3(2, 512, 3072))
    _require_tensor(st, String("batch.tokens"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("batch.tokens_mask"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("output.loss_for_backward"), STDtype.F32, _shape0())
    _require_tensor(st, String("output.loss_pre_scale"), STDtype.F32, _shape0())
    _require_tensor(st, String("output.predicted"), STDtype.BF16, _shape4(2, 32, 80, 56))
    _require_tensor(st, String("output.target"), STDtype.F32, _shape4(2, 32, 80, 56))
    _require_tensor(st, String("output.timestep"), STDtype.I32, _shape1(2))
    _require_tensor(st, String("trace.encode_text.cached_hidden_state"), STDtype.BF16, _shape3(2, 512, 3072))
    _require_tensor(st, String("trace.encode_text.tokens"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("trace.encode_text.tokens_mask"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("trace.encoder_hidden_states"), STDtype.BF16, _shape3(2, 201, 3072))
    _require_tensor(st, String("trace.flow"), STDtype.F32, _shape4(2, 128, 40, 28))
    _require_tensor(st, String("trace.latent_image_before_scale"), STDtype.F32, _shape4(2, 128, 40, 28))
    _require_tensor(st, String("trace.latent_noise"), STDtype.F32, _shape4(2, 128, 40, 28))
    _require_tensor(st, String("trace.noise_source_tensor"), STDtype.F32, _shape4(2, 128, 40, 28))
    _require_tensor(st, String("trace.packed_predicted_flow"), STDtype.BF16, _shape4(2, 128, 40, 28))
    _require_tensor(st, String("trace.scaled_latent_image"), STDtype.F32, _shape4(2, 128, 40, 28))
    _require_tensor(st, String("trace.scaled_noisy_latent_image"), STDtype.F32, _shape4(2, 128, 40, 28))
    _require_tensor(st, String("trace.sigma"), STDtype.F32, _shape4(2, 1, 1, 1))
    _require_tensor(st, String("trace.text_encoder_output"), STDtype.BF16, _shape3(2, 201, 3072))
    _require_tensor(st, String("trace.transformer_hidden_states"), STDtype.BF16, _shape4(2, 128, 40, 28))
    _require_tensor(st, String("trace.transformer_timestep"), STDtype.I32, _shape1(2))

    _require_close(
        String("output.loss_for_backward"),
        _read_f32_value(st, String("output.loss_for_backward"), 0),
        Float32(0.643847644329071),
        Float32(1.0e-8),
    )
    _require_close(
        String("output.loss_pre_scale"),
        _read_f32_value(st, String("output.loss_pre_scale"), 0),
        Float32(0.643847644329071),
        Float32(1.0e-8),
    )
    _require_close(
        String("batch.loss_weight[0]"),
        _read_f32_value(st, String("batch.loss_weight"), 0),
        Float32(1.0),
        Float32(0.0),
    )
    _require_close(
        String("batch.loss_weight[1]"),
        _read_f32_value(st, String("batch.loss_weight"), 1),
        Float32(1.0),
        Float32(0.0),
    )
    _require_close(
        String("trace.sigma[0]"),
        _read_f32_value(st, String("trace.sigma"), 0),
        Float32(0.5460000038146973),
        Float32(1.0e-8),
    )
    _require_close(
        String("trace.sigma[1]"),
        _read_f32_value(st, String("trace.sigma"), 1),
        Float32(0.3660000264644623),
        Float32(1.0e-8),
    )
    _require_i32(String("output.timestep[0]"), _read_i32_value(st, String("output.timestep"), 0), Int32(545))
    _require_i32(String("output.timestep[1]"), _read_i32_value(st, String("output.timestep"), 1), Int32(365))
    _require_i32(
        String("trace.transformer_timestep[0]"),
        _read_i32_value(st, String("trace.transformer_timestep"), 0),
        Int32(545),
    )
    _require_i32(
        String("trace.transformer_timestep[1]"),
        _read_i32_value(st, String("trace.transformer_timestep"), 1),
        Int32(365),
    )


def _check_adapter_dump() raises:
    var st = SafeTensors.open(String(ADAPTER_DUMP))
    _require(len(st.tensors) == ADAPTER_DUMP_KEYS, String("expected 2016 Ernie adapter tensors"))

    var max_abs_after_minus_post = Float32(0.0)
    for layer in range(LAYERS):
        _require_adapter_layer_phase(st, String("adapter_before"), layer)
        _require_adapter_layer_phase(st, String("adapter_pre"), layer)
        _require_adapter_layer_phase(st, String("adapter_post"), layer)
        _require_adapter_layer_phase(st, String("adapter_after"), layer)
        var layer_diff = _require_adapter_layer_after_minus_post(st, layer)
        if layer_diff > max_abs_after_minus_post:
            max_abs_after_minus_post = layer_diff

    var down_key = String("adapter_before.transformer.layers.0.self_attention.to_q.lora_down.weight")
    var up_key = String("adapter_before.transformer.layers.0.self_attention.to_q.lora_up.weight")
    var post_down_key = String("adapter_post.transformer.layers.0.self_attention.to_q.lora_down.weight")
    var after_down_key = String("adapter_after.transformer.layers.0.self_attention.to_q.lora_down.weight")
    _require_close(
        String("adapter_before.to_q.down[0]"),
        _read_f32_value(st, down_key, 0),
        Float32(-0.01278527919203043),
        Float32(1.0e-8),
    )
    _require_close(
        String("adapter_before.to_q.up[0]"),
        _read_f32_value(st, up_key, 0),
        Float32(0.0),
        Float32(0.0),
    )
    _require_close(
        String("adapter_after - adapter_post.to_q.down[0]"),
        _read_f32_value(st, after_down_key, 0) - _read_f32_value(st, post_down_key, 0),
        Float32(0.0),
        adapter_after_minus_post_tolerance,
    )
    _require_close(
        String("after_minus_post sampled max_abs"),
        max_abs_after_minus_post,
        Float32(0.0),
        adapter_after_minus_post_tolerance,
    )


def main() raises:
    _check_meta_json()
    _check_step_dump()
    _check_adapter_dump()
    # State-init bridge anchor only: the production update path is
    # ernie_lora_adamw_step(...), which delegates to _lora_adamw(...). This
    # no-CUDA artifact smoke consumes the zero-lr OneTrainer dumps and metadata;
    # it does not execute or claim full AdamW replay parity.
    print("[ernie-train-ref-artifact] PASS:", STEP_DUMP, ADAPTER_DUMP, META_JSON)
    print(
        "[ernie-train-ref-artifact] zero-lr state-init: lr_before=0.0, "
        "lr_after=1.4999999999999998e-06, "
        "optimizer_before/optimizer_after parameter_entries=0->504, "
        "after_minus_post max_abs=0.0"
    )
