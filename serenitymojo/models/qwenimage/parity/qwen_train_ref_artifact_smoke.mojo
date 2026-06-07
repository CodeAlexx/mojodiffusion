# qwen_train_ref_artifact_smoke.mojo -- Qwen OneTrainer one-step artifact gate.
#
# Header-only, no-CUDA gate. It opens the real local OneTrainer Qwen one-step
# dump, adapter dump, and metadata JSON, then validates the tensor names, shapes,
# dtypes, byte sizes, zero-lr optimizer state-init metadata, and representative
# unchanged adapter payloads that future Mojo replay gates must consume. It
# deliberately does not claim transformer/loss/backward/AdamW numeric parity.
#
# AdamW anchor: production Qwen optimizer state-init/update is carried by
# qwen_lora_adamw_step(...). This no-CUDA smoke consumes the OneTrainer artifacts
# and checks the zero-lr state-init contract; it does not execute that model path.

from std.collections import List
from std.memory import alloc

from serenitymojo.io.ffi import (
    BytePtr, O_RDONLY, sys_close, sys_open, sys_pread,
)
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors


comptime STEP_DUMP = "/home/alex/onetrainer-mojo/parity/qwen_train_ref_step000.safetensors"
comptime ADAPTER_DUMP = "/home/alex/onetrainer-mojo/parity/qwen_train_ref_step000_adapters.safetensors"
comptime META_JSON = "/home/alex/onetrainer-mojo/parity/qwen_train_ref_meta.json"


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _read_file(path: String) raises -> List[UInt8]:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("failed to open Qwen OT artifact: ") + path)
    var out = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var n = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if n < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("failed to read Qwen OT artifact: ") + path)
        if n == 0:
            break
        for i in range(n):
            out.append(buf[i])
        offset += n
        if n < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    _require(len(out) > 0, String("empty Qwen OT artifact: ") + path)
    return out^


def _contains_bytes(haystack: List[UInt8], needle: String) -> Bool:
    var nb = needle.as_bytes()
    var n = needle.byte_length()
    if n == 0:
        return True
    if len(haystack) < n:
        return False
    for start in range(len(haystack) - n + 1):
        var ok = True
        for j in range(n):
            if haystack[start + j] != nb[j]:
                ok = False
                break
        if ok:
            return True
    return False


def _require_contains(haystack: List[UInt8], needle: String) raises:
    _require(
        _contains_bytes(haystack, needle),
        String("Qwen OT meta missing substring: ") + needle,
    )


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


def _require_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    _require(
        _abs(got - expected) <= tol,
        name + String(" value mismatch got=") + String(got) + String(" expected=") + String(expected),
    )


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    _require(key in st.tensors, String("missing Qwen OT tensor ") + key)
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


def _check_meta_json() raises:
    var meta = _read_file(String(META_JSON))

    _require_contains(meta, String('"safetensors": "/home/alex/onetrainer-mojo/parity/qwen_train_ref_step000.safetensors"'))
    _require_contains(meta, String('"adapter_safetensors": "/home/alex/onetrainer-mojo/parity/qwen_train_ref_step000_adapters.safetensors"'))

    # OneTrainer zero-lr state-init evidence, not nonzero AdamW update parity:
    # optimizer_before.state.parameter_entries is 0; optimizer_after.state
    # parameter_entries is 1440; lr_before is 0.0 and lr_after is the scheduler's
    # post-step value captured by the same metadata artifact.
    _require_contains(meta, String('"optimizer": "ADAMW"'))
    _require_contains(meta, String('"optimizer_before"'))
    _require_contains(meta, String('"optimizer_after"'))
    _require_contains(meta, String('"parameter_entries": 0'))
    _require_contains(meta, String('"parameter_entries": 1440'))
    _require_contains(meta, String('"lr_before": ['))
    _require_contains(meta, String('0.0'))
    _require_contains(meta, String('"lr_after": ['))
    _require_contains(meta, String('1.4999999999999998e-06'))
    _require_contains(meta, String('"keys": ['))
    _require_contains(meta, String('"exp_avg"'))
    _require_contains(meta, String('"exp_avg_sq"'))
    _require_contains(meta, String('"step"'))


def _check_step_dump() raises:
    var st = SafeTensors.open(String(STEP_DUMP))
    _require(len(st.tensors) == 29, String("expected 29 Qwen step tensors"))

    _require_tensor(st, String("batch.latent_image"), STDtype.BF16, _shape5(2, 16, 1, 64, 64))
    _require_tensor(st, String("batch.tokens"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("batch.tokens_mask"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("batch.text_encoder_hidden_state"), STDtype.BF16, _shape3(2, 512, 3584))
    _require_tensor(st, String("batch.loss_weight"), STDtype.F32, _shape1(2))
    _require_tensor(st, String("trace.encode_text.tokens"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("trace.encode_text.tokens_mask"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("trace.encode_text.cached_hidden_state"), STDtype.BF16, _shape3(2, 512, 3584))
    _require_tensor(st, String("trace.text_encoder_output"), STDtype.BF16, _shape3(2, 144, 3584))
    _require_tensor(st, String("trace.text_attention_mask"), STDtype.BOOL, _shape2(2, 144))
    _require_tensor(st, String("trace.latent_image_before_scale"), STDtype.BF16, _shape5(2, 16, 1, 64, 64))
    _require_tensor(st, String("trace.scaled_latent_image"), STDtype.F32, _shape5(2, 16, 1, 64, 64))
    _require_tensor(st, String("trace.noise_source_tensor"), STDtype.F32, _shape5(2, 16, 1, 64, 64))
    _require_tensor(st, String("trace.latent_noise"), STDtype.F32, _shape5(2, 16, 1, 64, 64))
    _require_tensor(st, String("trace.scaled_noisy_latent_image"), STDtype.F32, _shape5(2, 16, 1, 64, 64))
    _require_tensor(st, String("trace.sigma"), STDtype.F32, _shape5(2, 1, 1, 1, 1))
    _require_tensor(st, String("trace.latent_input"), STDtype.F32, _shape5(2, 16, 1, 64, 64))
    _require_tensor(st, String("trace.packed_latent_input"), STDtype.F32, _shape3(2, 1024, 64))
    _require_tensor(st, String("trace.transformer_hidden_states"), STDtype.BF16, _shape3(2, 1024, 64))
    _require_tensor(st, String("trace.transformer_timestep"), STDtype.F32, _shape1(2))
    _require_tensor(st, String("trace.encoder_hidden_states"), STDtype.BF16, _shape3(2, 144, 3584))
    _require_tensor(st, String("trace.encoder_hidden_states_mask"), STDtype.BOOL, _shape2(2, 144))
    _require_tensor(st, String("trace.packed_predicted_flow"), STDtype.BF16, _shape3(2, 1024, 64))
    _require_tensor(st, String("trace.flow"), STDtype.F32, _shape5(2, 16, 1, 64, 64))
    _require_tensor(st, String("output.timestep"), STDtype.I32, _shape1(2))
    _require_tensor(st, String("output.predicted"), STDtype.BF16, _shape5(2, 16, 1, 64, 64))
    _require_tensor(st, String("output.target"), STDtype.F32, _shape5(2, 16, 1, 64, 64))
    _require_tensor(st, String("output.loss_pre_scale"), STDtype.F32, _shape0())
    _require_tensor(st, String("output.loss_for_backward"), STDtype.F32, _shape0())
    _require_close(
        String("output.loss_for_backward"),
        _read_f32_value(st, String("output.loss_for_backward"), 0),
        Float32(0.08948630839586258),
        Float32(1.0e-8),
    )


def _check_adapter_dump() raises:
    var st = SafeTensors.open(String(ADAPTER_DUMP))
    _require(len(st.tensors) == 5760, String("expected 5760 Qwen adapter tensors"))
    _require_tensor(
        st,
        String("adapter_before.transformer.transformer_blocks.0.attn.to_q.lora_down.weight"),
        STDtype.F32,
        _shape2(16, 3072),
    )
    _require_tensor(
        st,
        String("adapter_pre.transformer.transformer_blocks.0.attn.to_q.lora_down.weight"),
        STDtype.F32,
        _shape2(16, 3072),
    )
    _require_tensor(
        st,
        String("adapter_post.transformer.transformer_blocks.0.attn.to_q.lora_down.weight"),
        STDtype.F32,
        _shape2(16, 3072),
    )
    _require_tensor(
        st,
        String("adapter_after.transformer.transformer_blocks.0.attn.to_q.lora_down.weight"),
        STDtype.F32,
        _shape2(16, 3072),
    )
    _require_tensor(
        st,
        String("adapter_after.transformer.transformer_blocks.59.txt_mlp.net.2.lora_up.weight"),
        STDtype.F32,
        _shape2(3072, 16),
    )
    var down_key = String("adapter_before.transformer.transformer_blocks.0.attn.to_q.lora_down.weight")
    var up_key = String("adapter_before.transformer.transformer_blocks.0.attn.to_q.lora_up.weight")
    var post_down_key = String("adapter_post.transformer.transformer_blocks.0.attn.to_q.lora_down.weight")
    var after_down_key = String("adapter_after.transformer.transformer_blocks.0.attn.to_q.lora_down.weight")
    var post_up_key = String("adapter_post.transformer.transformer_blocks.0.attn.to_q.lora_up.weight")
    var after_up_key = String("adapter_after.transformer.transformer_blocks.0.attn.to_q.lora_up.weight")
    _require_close(
        String("adapter_before.to_q.down[0]"),
        _read_f32_value(st, down_key, 0),
        Float32(0.013218194246292114),
        Float32(1.0e-8),
    )
    _require_close(
        String("adapter_before.to_q.up[0]"),
        _read_f32_value(st, up_key, 0),
        Float32(0.0),
        Float32(0.0),
    )
    _require_close(
        String("adapter_before_pre_after.to_q.down[0]"),
        _read_f32_value(st, down_key, 0),
        _read_f32_value(
            st,
            String("adapter_after.transformer.transformer_blocks.0.attn.to_q.lora_down.weight"),
            0,
        ),
        Float32(0.0),
    )
    var tolerance = Float32(0.0)
    var after_minus_post_down = _read_f32_value(st, after_down_key, 0) - _read_f32_value(st, post_down_key, 0)
    var after_minus_post_up = _read_f32_value(st, after_up_key, 0) - _read_f32_value(st, post_up_key, 0)
    # unchanged adapter comparison: after_minus_post == adapter_after - adapter_post.
    _require_close(
        String("after_minus_post max_abs tolerance to_q.down[0]"),
        after_minus_post_down,
        Float32(0.0),
        tolerance,
    )
    _require_close(
        String("after_minus_post max_abs tolerance to_q.up[0]"),
        after_minus_post_up,
        Float32(0.0),
        tolerance,
    )


def main() raises:
    _check_meta_json()
    _check_step_dump()
    _check_adapter_dump()
    print("[qwen-train-ref-artifact] PASS:", STEP_DUMP, ADAPTER_DUMP, META_JSON)
