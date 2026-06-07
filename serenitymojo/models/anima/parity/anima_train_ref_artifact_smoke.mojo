# anima_train_ref_artifact_smoke.mojo -- no-CUDA Anima OneTrainer artifact gate.
#
# Opens the real /home/alex/OneTrainer-anima-ref 100-step Anima LoRA artifact
# and validates the raw OneTrainer LoRA safetensors inventory. This is an
# artifact/source gate only; it does not prove transformer, backward, or AdamW
# parity.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import O_RDONLY, sys_close, sys_open
from serenitymojo.io.safetensors import SafeTensors


comptime OT_BASELINE = "/home/alex/OneTrainer-anima-ref/output/anima_100step_baseline/lora.safetensors"
comptime NUM_BLOCKS = 28
comptime ANIMA_MODULES = 10
comptime SAVE_SUFFIXES = 3
comptime RANK = 16
comptime D_MODEL = 2048
comptime JOINT = 1024
comptime F_MLP = 8192
comptime EXPECTED_TENSORS = NUM_BLOCKS * ANIMA_MODULES * SAVE_SUFFIXES


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _file_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _module(slot: Int) raises -> String:
    if slot == 0:
        return String("attn1.to_q")
    if slot == 1:
        return String("attn1.to_k")
    if slot == 2:
        return String("attn1.to_v")
    if slot == 3:
        return String("attn1.to_out.0")
    if slot == 4:
        return String("attn2.to_q")
    if slot == 5:
        return String("attn2.to_k")
    if slot == 6:
        return String("attn2.to_v")
    if slot == 7:
        return String("attn2.to_out.0")
    if slot == 8:
        return String("ff.net.0.proj")
    if slot == 9:
        return String("ff.net.2")
    raise Error(String("unsupported Anima LoRA slot ") + String(slot))


def _input_dim(slot: Int) raises -> Int:
    if slot == 5 or slot == 6:
        return JOINT
    if slot == 9:
        return F_MLP
    if slot >= 0 and slot < ANIMA_MODULES:
        return D_MODEL
    raise Error(String("unsupported Anima LoRA slot input dim ") + String(slot))


def _output_dim(slot: Int) raises -> Int:
    if slot == 8:
        return F_MLP
    if slot >= 0 and slot < ANIMA_MODULES:
        return D_MODEL
    raise Error(String("unsupported Anima LoRA slot output dim ") + String(slot))


def _prefix(block: Int, slot: Int) raises -> String:
    return (
        String("transformer.transformer_blocks.")
        + String(block)
        + String(".")
        + _module(slot)
    )


def _append_expected(mut expected: Dict[String, Bool], prefix: String):
    expected[prefix + String(".alpha")] = True
    expected[prefix + String(".lora_down.weight")] = True
    expected[prefix + String(".lora_up.weight")] = True


def _expected_nbytes(shape: List[Int], dtype: STDtype) -> Int:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n * dtype.byte_size()


def _require_alpha_meta(st: SafeTensors, key: String) raises -> Int:
    _require(key in st.tensors, String("missing key ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BF16, String("expected BF16 alpha for ") + key)
    _require(len(info.shape) == 0, String("expected scalar alpha for ") + key)
    _require(info.size == 2, String("alpha byte size mismatch for ") + key)
    return info.size


def _require_rank2_meta(
    st: SafeTensors, key: String, dim0: Int, dim1: Int
) raises -> Int:
    _require(key in st.tensors, String("missing key ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BF16, String("expected BF16 for ") + key)
    _require(len(info.shape) == 2, String("expected rank-2 tensor for ") + key)
    _require(
        info.shape[0] == dim0 and info.shape[1] == dim1,
        String("shape mismatch for ") + key,
    )
    var expected_bytes = _expected_nbytes(info.shape.copy(), info.dtype)
    _require(
        info.size == expected_bytes,
        String("byte size mismatch for ") + key,
    )
    return info.size


def _require_bf16_one_scalar(st: SafeTensors, key: String) raises:
    _ = _require_alpha_meta(st, key)
    var bytes = st.tensor_bytes(key)
    _require(len(bytes) == 2, String("alpha payload length mismatch for ") + key)
    _require(
        bytes[0] == UInt8(0x80) and bytes[1] == UInt8(0x3F),
        String("alpha payload is not BF16 1.0 for ") + key,
    )


def _validate_exact_names(st: SafeTensors) raises:
    var expected = Dict[String, Bool]()
    for block in range(NUM_BLOCKS):
        for slot in range(ANIMA_MODULES):
            _append_expected(expected, _prefix(block, slot))

    _require(
        len(st.tensors) == EXPECTED_TENSORS,
        String("expected 840 Anima LoRA tensors"),
    )

    var names = st.names()
    _require(
        len(names) == EXPECTED_TENSORS,
        String("SafeTensors names count mismatch"),
    )
    for i in range(len(names)):
        _require(
            names[i] in expected,
            String("unsupported Anima reference key ") + names[i],
        )


def _validate_all_tensor_meta(st: SafeTensors) raises -> Int:
    var total_bytes = 0
    for block in range(NUM_BLOCKS):
        for slot in range(ANIMA_MODULES):
            var prefix = _prefix(block, slot)
            var in_f = _input_dim(slot)
            var out_f = _output_dim(slot)
            total_bytes += _require_alpha_meta(st, prefix + String(".alpha"))
            total_bytes += _require_rank2_meta(
                st,
                prefix + String(".lora_down.weight"),
                RANK,
                in_f,
            )
            total_bytes += _require_rank2_meta(
                st,
                prefix + String(".lora_up.weight"),
                out_f,
                RANK,
            )
    _require(
        st.data_size() == total_bytes,
        String("safetensors data byte size mismatch"),
    )
    return total_bytes


def _validate_scalar_payloads(st: SafeTensors) raises:
    _require_bf16_one_scalar(
        st, String("transformer.transformer_blocks.0.attn1.to_q.alpha")
    )
    _require_bf16_one_scalar(
        st, String("transformer.transformer_blocks.13.attn2.to_k.alpha")
    )
    _require_bf16_one_scalar(
        st, String("transformer.transformer_blocks.27.ff.net.2.alpha")
    )


def main() raises:
    var path = String(OT_BASELINE)
    _require(_file_exists(path), String("missing OneTrainer Anima artifact: ") + path)

    var st = SafeTensors.open(path)
    _validate_exact_names(st)
    var total_bytes = _validate_all_tensor_meta(st)
    _validate_scalar_payloads(st)

    print(
        "[anima-train-ref-artifact] PASS tensors=",
        st.count(),
        "data_bytes=",
        total_bytes,
        "source=",
        path,
    )
