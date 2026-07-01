# ZImage OneTrainer train-ref loss bridge.
#
# Opens the real OneTrainer ZImage step-0 dump and recomputes the flow-MSE loss
# from dumped BF16 predicted/target flow tensors. This is a no-CUDA artifact
# consumer/loss bridge. It does not execute the transformer, backward, AdamW,
# save/resume, or product loop.

from std.builtin.dtype import DType
from std.collections import List
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors


comptime META_JSON = "/home/alex/serenity-trainer/parity/zimage_train_ref_meta.json"
comptime STEP_DUMP = "/home/alex/serenity-trainer/parity/zimage_train_ref_step000.safetensors"
comptime BATCH = 2
comptime CHANNELS = 16
comptime HEIGHT = 64
comptime WIDTH = 64
comptime SAMPLE_NUMEL = CHANNELS * HEIGHT * WIDTH
comptime TOTAL_NUMEL = BATCH * SAMPLE_NUMEL
comptime EXPECTED_LOSS = Float32(0.40854018926620483)


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _shape0() -> List[Int]:
    return List[Int]()


def _shape1(a: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    return out^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    return out^


def _shape5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    out.append(e)
    return out^


def _shape_eq(got: List[Int], expected: List[Int]) -> Bool:
    if len(got) != len(expected):
        return False
    for i in range(len(expected)):
        if got[i] != expected[i]:
            return False
    return True


def _shape_text(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i > 0:
            out += String(",")
        out += String(shape[i])
    out += String("]")
    return out^


def _numel(shape: List[Int]) -> Int:
    if len(shape) == 0:
        return 1
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    _require(key in st.tensors, String("missing ZImage OT tensor ") + key)
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


def _read_f32_value(st: SafeTensors, key: String, index: Int) raises -> Float32:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.F32, String("expected F32 payload for ") + key)
    _require(index >= 0 and index < info.size // 4, String("F32 index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    var fp = bytes.unsafe_ptr().bitcast[Float32]()
    return fp[index]


def _read_bf16_value(st: SafeTensors, key: String, index: Int) raises -> Float32:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BF16, String("expected BF16 payload for ") + key)
    _require(index >= 0 and index < info.size // 2, String("BF16 index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    var bp = bytes.unsafe_ptr().bitcast[BFloat16]()
    return bp[index].cast[DType.float32]()


def _require_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    _require(
        _abs(got - expected) <= tol,
        name + String(" mismatch got=") + String(got) + String(" expected=") + String(expected),
    )


def _flow_mse_loss(st: SafeTensors) raises -> Float32:
    var total = Float32(0.0)
    for b in range(BATCH):
        var sample_sum = Float32(0.0)
        var offset = b * SAMPLE_NUMEL
        for i in range(SAMPLE_NUMEL):
            var pred = _read_bf16_value(st, String("predicted_flow"), offset + i)
            var target = _read_bf16_value(st, String("flow_target"), offset + i)
            var d = pred - target
            sample_sum += d * d
        var weight = _read_f32_value(st, String("batch.loss_weight"), b)
        total += (sample_sum / Float32(SAMPLE_NUMEL)) * weight
    return total / Float32(BATCH)


def _check_meta_json() raises:
    var meta = Path(String(META_JSON)).read_text()
    _require(meta.byte_length() > 0, String("empty ZImage OneTrainer metadata JSON"))
    _require(meta.find(String("\"producer\": \"scripts/zimage_dump_train_ref.py\"")) >= 0, String("metadata producer mismatch"))
    _require(meta.find(String("\"prefix\": \"zimage_train_ref\"")) >= 0, String("metadata prefix mismatch"))
    _require(meta.find(String("\"model_type\": \"Z_IMAGE\"")) >= 0, String("metadata model_type mismatch"))
    _require(meta.find(String("\"training_method\": \"LORA\"")) >= 0, String("metadata training_method mismatch"))
    _require(meta.find(String("\"train_dtype\": \"BFLOAT_16\"")) >= 0, String("metadata train_dtype mismatch"))
    _require(meta.find(String("\"adapter_dump\": \"step-with-grads\"")) >= 0, String("metadata adapter_dump mismatch"))
    _require(meta.find(String("\"count\": 420")) >= 0, String("metadata trainable count mismatch"))
    _require(meta.find(String("\"numel\": 35020800")) >= 0, String("metadata trainable numel mismatch"))
    _require(meta.find(String("\"loss_for_backward\": 0.40854018926620483")) >= 0, String("metadata loss mismatch"))
    _require(meta.find(String("\"lr_before\": [\n        0.0")) >= 0, String("metadata lr_before mismatch"))
    _require(meta.find(String("\"l2\": 0.0")) >= 0, String("metadata state-init adapter delta mismatch"))
    _require(meta.find(String(STEP_DUMP)) >= 0, String("metadata step path mismatch"))


def _check_step_dump() raises:
    var st = SafeTensors.open(String(STEP_DUMP))
    _require(st.count() == 42, String("expected 42 ZImage step tensors"))
    _require_tensor(st, String("predicted_flow"), STDtype.BF16, _shape4(BATCH, CHANNELS, HEIGHT, WIDTH))
    _require_tensor(st, String("flow_target"), STDtype.BF16, _shape4(BATCH, CHANNELS, HEIGHT, WIDTH))
    _require_tensor(st, String("output.predicted"), STDtype.BF16, _shape4(BATCH, CHANNELS, HEIGHT, WIDTH))
    _require_tensor(st, String("output.target"), STDtype.BF16, _shape4(BATCH, CHANNELS, HEIGHT, WIDTH))
    _require_tensor(st, String("batch.loss_weight"), STDtype.F32, _shape1(BATCH))
    _require_tensor(st, String("output.loss_pre_scale"), STDtype.F32, _shape0())
    _require_tensor(st, String("output.loss_for_backward"), STDtype.F32, _shape0())
    _require_tensor(st, String("timestep"), STDtype.F32, _shape1(BATCH))
    _require_tensor(st, String("sigma"), STDtype.F32, _shape4(BATCH, 1, 1, 1))
    _require_tensor(st, String("latent_input"), STDtype.BF16, _shape5(BATCH, CHANNELS, 1, HEIGHT, WIDTH))

    var stored_loss = _read_f32_value(st, String("output.loss_for_backward"), 0)
    _require_close(String("stored loss"), stored_loss, EXPECTED_LOSS, Float32(1.0e-8))
    var replayed_loss = _flow_mse_loss(st)
    _require_close(String("replayed flow-MSE loss"), replayed_loss, stored_loss, Float32(1.0e-5))

    print("[zimage-train-ref-loss] loss_bridge PASS stored=", stored_loss, " replayed=", replayed_loss)


def main() raises:
    _check_meta_json()
    _check_step_dump()
