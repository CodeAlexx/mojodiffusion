# ZImage OneTrainer train-ref adapter oracle metadata smoke.
#
# Legacy filename note: this file used to validate a parity-only F32 device
# adapter carrier. That path is intentionally removed. ZImage Mojo runtime
# boundaries stay BF16 in/out. The adapter dump dtype is comparison data from
# OneTrainer's live LoRA params, not authority for Mojo device storage.

from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors


comptime META_JSON = "/home/alex/serenity-trainer/parity/zimage_train_ref_meta.json"
comptime STEP_DUMP = "/home/alex/serenity-trainer/parity/zimage_train_ref_step000.safetensors"
comptime STEP0_ADAPTERS = "/home/alex/serenity-trainer/parity/zimage_train_ref_step000_adapters.safetensors"
comptime FORWARD_PHASE_PREFIX = "adapter_before."
comptime GRAD_PHASE_PREFIX = "adapter_post_clip_grad."
comptime LAYER = 0
comptime RANK = 16
comptime D_MODEL = 3840
comptime F_MODEL = 10240
comptime EXPECTED_TENSORS = 14
comptime EXPECTED_NUMEL = 1167360
comptime REPLAY_SCALE = Float32(0.0625)
comptime BATCH = 2
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime CAP_DIM = 2560
comptime VALID_CAP0 = 145
comptime VALID_CAP1 = 127


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _slot_suffix(slot: Int) raises -> String:
    if slot == 0:
        return String("attention.to_q")
    if slot == 1:
        return String("attention.to_k")
    if slot == 2:
        return String("attention.to_v")
    if slot == 3:
        return String("attention.to_out.0")
    if slot == 4:
        return String("feed_forward.w1")
    if slot == 5:
        return String("feed_forward.w3")
    if slot == 6:
        return String("feed_forward.w2")
    raise Error(String("bad ZImage LoRA slot ") + String(slot))


def _slot_in_f(slot: Int) raises -> Int:
    if slot == 6:
        return F_MODEL
    if slot >= 0 and slot <= 5:
        return D_MODEL
    raise Error(String("bad ZImage LoRA slot ") + String(slot))


def _slot_out_f(slot: Int) raises -> Int:
    if slot == 4 or slot == 5:
        return F_MODEL
    if slot >= 0 and slot <= 3:
        return D_MODEL
    if slot == 6:
        return D_MODEL
    raise Error(String("bad ZImage LoRA slot ") + String(slot))


def _adapter_key(phase_prefix: String, slot: Int, down: Bool) raises -> String:
    var key = phase_prefix
    key += String("transformer.layers.")
    key += String(LAYER)
    key += String(".")
    key += _slot_suffix(slot)
    if down:
        key += String(".lora_down.weight")
    else:
        key += String(".lora_up.weight")
    return key^


def _shape_text(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i > 0:
            out += String(",")
        out += String(shape[i])
    out += String("]")
    return out^


def _shape2(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
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


def _require_shape2(shape: List[Int], rows: Int, cols: Int, key: String) raises:
    _require(
        len(shape) == 2 and shape[0] == rows and shape[1] == cols,
        String("shape mismatch for ")
        + key
        + String(" got=")
        + _shape_text(shape)
        + String(" expected=[")
        + String(rows)
        + String(",")
        + String(cols)
        + String("]"),
    )


def _same_shape(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _numel(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    _require(key in st.tensors, String("missing ZImage train-ref tensor ") + key)
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
        _same_shape(info.shape, expected_shape),
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


def _require_f32_matrix(
    st: SafeTensors, key: String, rows: Int, cols: Int
) raises -> Int:
    _require(key in st.tensors, String("missing OT adapter tensor ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.F32, String("expected F32 adapter dump tensor ") + key)
    _require_shape2(info.shape, rows, cols, key)
    _require(info.size == rows * cols * 4, String("byte-size mismatch for ") + key)
    return rows * cols


def _check_slot(st: SafeTensors, slot: Int) raises -> Int:
    var in_f = _slot_in_f(slot)
    var out_f = _slot_out_f(slot)
    var total = 0
    total += _require_f32_matrix(
        st, _adapter_key(String(FORWARD_PHASE_PREFIX), slot, True), RANK, in_f
    )
    total += _require_f32_matrix(
        st, _adapter_key(String(FORWARD_PHASE_PREFIX), slot, False), out_f, RANK
    )
    total += _require_f32_matrix(
        st, _adapter_key(String(GRAD_PHASE_PREFIX), slot, True), RANK, in_f
    )
    total += _require_f32_matrix(
        st, _adapter_key(String(GRAD_PHASE_PREFIX), slot, False), out_f, RANK
    )
    return total


def _check_meta_json() raises:
    var meta = Path(String(META_JSON)).read_text()
    _require(meta.byte_length() > 0, String("empty ZImage train-ref metadata JSON"))
    _require(meta.find(String("\"producer\": \"scripts/zimage_dump_train_ref.py\"")) >= 0, String("metadata producer mismatch"))
    _require(meta.find(String("\"train_dtype\": \"BFLOAT_16\"")) >= 0, String("metadata train dtype mismatch"))
    _require(meta.find(String("\"model_type\": \"Z_IMAGE\"")) >= 0, String("metadata model type mismatch"))
    _require(meta.find(String("\"training_method\": \"LORA\"")) >= 0, String("metadata training method mismatch"))
    _require(meta.find(String("\"parameter_entries\": 420")) >= 0, String("metadata trainable count mismatch"))
    _require(meta.find(String(STEP_DUMP)) >= 0, String("metadata step0 path mismatch"))
    _require(meta.find(String(STEP0_ADAPTERS)) >= 0, String("metadata step0 adapter path mismatch"))


def main() raises:
    _check_meta_json()
    var step = SafeTensors.open(String(STEP_DUMP))
    _require_tensor(step, String("scaled_noisy_latent_image"), STDtype.BF16, _shape4(BATCH, LAT_C, LAT_H, LAT_W))
    _require_tensor(step, String("latent_input"), STDtype.BF16, _shape5(BATCH, LAT_C, 1, LAT_H, LAT_W))
    _require_tensor(step, String("predicted_flow"), STDtype.BF16, _shape4(BATCH, LAT_C, LAT_H, LAT_W))
    _require_tensor(step, String("flow_target"), STDtype.BF16, _shape4(BATCH, LAT_C, LAT_H, LAT_W))
    _require_tensor(step, String("text_encoder_output_0"), STDtype.BF16, _shape2(VALID_CAP0, CAP_DIM))
    _require_tensor(step, String("text_encoder_output_1"), STDtype.BF16, _shape2(VALID_CAP1, CAP_DIM))

    var st = SafeTensors.open(String(STEP0_ADAPTERS))
    var total_with_grad = 0
    for slot in range(7):
        total_with_grad += _check_slot(st, slot)
    var selected_numel = total_with_grad // 2
    _require(selected_numel == EXPECTED_NUMEL, String("layer-0 adapter oracle numel mismatch"))

    print(
        "[zimage-adapter-oracle-metadata] PASS layer=",
        LAYER,
        " forward_phase=adapter_before grad_target=adapter_post_clip_grad",
        " tensors=",
        EXPECTED_TENSORS,
        " selected_numel=",
        selected_numel,
        " runtime_boundary=BF16 adapter_dump_dtype=F32 mojo_storage_boundary=BF16 rank=",
        RANK,
        " replay_scale=",
        REPLAY_SCALE,
    )
    print(
        "[zimage-adapter-oracle-metadata] scope=OneTrainer BF16 runtime/step boundary plus live LoRA dump metadata; no device upload; adapter dump dtype is not Mojo storage authority"
    )
