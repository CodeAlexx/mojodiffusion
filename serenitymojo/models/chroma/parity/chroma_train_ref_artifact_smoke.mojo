# chroma_train_ref_artifact_smoke.mojo -- Chroma OneTrainer one-step artifact gate.
#
# Header-plus-payload, no-CUDA gate. It opens the real local OneTrainer Chroma
# one-step dump, adapter dump, and metadata JSON, then validates tensor names,
# shapes, dtypes, byte sizes, optimizer metadata, and representative update
# delta payloads. It deliberately does not claim transformer, backward, or
# AdamW numeric parity.
#
# Production AdamW anchor: train_chroma_real.mojo calls flux_lora_adamw_step(
# for the real update path. This smoke consumes OneTrainer update-delta
# artifacts only; it does not execute backward or that AdamW path.

from std.collections import List
from std.memory import alloc

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    BytePtr, O_RDONLY, file_size, sys_close, sys_open, sys_pread,
)
from serenitymojo.io.safetensors import SafeTensors


comptime STEP_DUMP = "/home/alex/onetrainer-mojo/parity/chroma_train_ref_step000.safetensors"
comptime ADAPTER_DUMP = "/home/alex/onetrainer-mojo/parity/chroma_train_ref_step000_adapters.safetensors"
comptime META_JSON = "/home/alex/onetrainer-mojo/parity/chroma_train_ref_meta.json"

comptime RANK = 16
comptime D = 3072
comptime FMLP = 12288
comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime TRAINABLE_PARAMETER_ENTRIES = 608
comptime OPTIMIZER_BEFORE_PARAMETER_ENTRIES = 0
comptime OPTIMIZER_AFTER_PARAMETER_ENTRIES = TRAINABLE_PARAMETER_ENTRIES
comptime OPTIMIZER_AFTER_TENSOR_COUNT = 1824
comptime OPTIMIZER_AFTER_TENSOR_NUMEL = 70976096
comptime UPDATE_DELTA_ATOL = Float32(1.0e-10)


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _read_utf8_file(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("Chroma OT smoke cannot open ") + path)
    var n = file_size(fd)
    var buf = alloc[UInt8](n if n > 0 else 1)
    var done = 0
    while done < n:
        var got = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf + done)), n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    if done != n:
        buf.free()
        raise Error(String("Chroma OT smoke short read from ") + path)
    var bytes = List[UInt8](capacity=n)
    for i in range(n):
        bytes.append(buf[i])
    buf.free()
    return String(unsafe_from_utf8=bytes)


def _contains(haystack: String, needle: String) -> Bool:
    var hb = haystack.as_bytes()
    var nb = needle.as_bytes()
    var nlen = len(nb)
    if nlen == 0:
        return True
    if nlen > len(hb):
        return False
    var i = 0
    while i + nlen <= len(hb):
        var matched = True
        for j in range(nlen):
            if hb[i + j] != nb[j]:
                matched = False
                break
        if matched:
            return True
        i += 1
    return False


def _require_contains(text: String, needle: String, label: String) raises:
    _require(
        _contains(text, needle),
        String("Chroma OT meta missing ") + label + String(": ") + needle,
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


def _read_i32_value(st: SafeTensors, key: String, index: Int) raises -> Int:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.I32, String("expected I32 payload for ") + key)
    _require(index >= 0 and index < info.size // 4, String("I32 index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    var byte_index = index * 4
    return (
        Int(bytes[byte_index])
        | (Int(bytes[byte_index + 1]) << 8)
        | (Int(bytes[byte_index + 2]) << 16)
        | (Int(bytes[byte_index + 3]) << 24)
    )


def _read_u16_payload(st: SafeTensors, key: String, index: Int) raises -> Int:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BF16, String("expected BF16 payload for ") + key)
    _require(index >= 0 and index < info.size // 2, String("BF16 index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    var byte_index = index * 2
    return Int(bytes[byte_index]) | (Int(bytes[byte_index + 1]) << 8)


def _read_u8_payload(st: SafeTensors, key: String, index: Int) raises -> Int:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BOOL, String("expected BOOL payload for ") + key)
    _require(index >= 0 and index < info.size, String("BOOL index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    return Int(bytes[index])


def _require_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    _require(
        _abs(got - expected) <= tol,
        name + String(" value mismatch got=") + String(got) + String(" expected=") + String(expected),
    )


def _require_after_minus_post_delta(
    st: SafeTensors,
    label: String,
    post_key: String,
    after_key: String,
    index: Int,
    expected: Float32,
    tolerance: Float32,
) raises:
    var adapter_post = _read_f32_value(st, post_key, index)
    var adapter_after = _read_f32_value(st, after_key, index)
    var after_minus_post = adapter_after - adapter_post
    var max_abs = _abs(after_minus_post - expected)
    _require(
        _abs(after_minus_post) > Float32(0.0),
        label + String(" expected nonzero adapter_post -> adapter_after update delta"),
    )
    _require(
        max_abs <= tolerance,
        label
        + String(" after_minus_post adapter_after - adapter_post max_abs=")
        + String(max_abs)
        + String(" tolerance=")
        + String(tolerance)
        + String(" got=")
        + String(after_minus_post)
        + String(" expected=")
        + String(expected),
    )


def _require_i32(name: String, got: Int, expected: Int) raises:
    _require(got == expected, name + String(" value mismatch"))


def _require_u16(name: String, got: Int, expected: Int) raises:
    _require(got == expected, name + String(" raw BF16 payload mismatch"))


def _require_u8(name: String, got: Int, expected: Int) raises:
    _require(got == expected, name + String(" raw U8 payload mismatch"))


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    _require(key in st.tensors, String("missing Chroma OT tensor ") + key)
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
    var meta = _read_utf8_file(String(META_JSON))

    _require_contains(meta, String('"safetensors": "') + String(STEP_DUMP) + String('"'), String("exact step dump path"))
    _require_contains(meta, String('"adapter_safetensors": "') + String(ADAPTER_DUMP) + String('"'), String("exact adapter dump path"))
    _require_contains(meta, String('"optimizer": "ADAMW"'), String("ADAMW optimizer"))
    _require_contains(meta, String('"class": "AdamW"'), String("AdamW optimizer class"))
    _require_contains(meta, String('"learning_rate": 0.0003'), String("learning_rate metadata"))
    _require_contains(meta, String('"lr_before": ['), String("lr_before metadata"))
    _require_contains(meta, String('"lr_after": ['), String("lr_after metadata"))
    _require_contains(meta, String('"optimizer_before"'), String("optimizer_before metadata"))
    _require_contains(meta, String('"optimizer_after"'), String("optimizer_after metadata"))
    _require_contains(meta, String('"parameter_entries": ') + String(OPTIMIZER_BEFORE_PARAMETER_ENTRIES), String("optimizer_before parameter_entries"))
    _require_contains(meta, String('"parameter_entries": ') + String(OPTIMIZER_AFTER_PARAMETER_ENTRIES), String("optimizer_after parameter_entries"))
    _require_contains(meta, String('"tensor_count": ') + String(OPTIMIZER_AFTER_TENSOR_COUNT), String("optimizer_after tensor_count"))
    _require_contains(meta, String('"tensor_numel": ') + String(OPTIMIZER_AFTER_TENSOR_NUMEL), String("optimizer_after tensor_numel"))
    _require_contains(meta, String('"exp_avg"'), String("optimizer exp_avg state key"))
    _require_contains(meta, String('"exp_avg_sq"'), String("optimizer exp_avg_sq state key"))
    _require_contains(meta, String('"step"'), String("optimizer step state key"))


def _check_step_dump() raises:
    var st = SafeTensors.open(String(STEP_DUMP))
    _require(len(st.tensors) == 33, String("expected 33 Chroma step tensors"))

    _require_tensor(st, String("batch.latent_image"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("batch.tokens"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("batch.tokens_mask"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("batch.text_encoder_hidden_state"), STDtype.BF16, _shape3(2, 512, 4096))
    _require_tensor(st, String("batch.loss_weight"), STDtype.F32, _shape1(2))
    _require_tensor(st, String("trace.encode_text.tokens"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("trace.encode_text.tokens_mask"), STDtype.I64, _shape2(2, 512))
    _require_tensor(st, String("trace.encode_text.cached_hidden_state"), STDtype.BF16, _shape3(2, 512, 4096))
    _require_tensor(st, String("trace.text_encoder_output"), STDtype.F32, _shape3(2, 224, 4096))
    _require_tensor(st, String("trace.text_attention_mask"), STDtype.BOOL, _shape2(2, 224))
    _require_tensor(st, String("trace.latent_image_before_scale"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("trace.scaled_latent_image"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("trace.noise_source_tensor"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("trace.latent_noise"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("trace.scaled_noisy_latent_image"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("trace.sigma"), STDtype.F32, _shape4(2, 1, 1, 1))
    _require_tensor(st, String("trace.latent_input"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("trace.packed_latent_input"), STDtype.BF16, _shape3(2, 1024, 64))
    _require_tensor(st, String("trace.image_ids"), STDtype.BF16, _shape2(1024, 3))
    _require_tensor(st, String("trace.image_ids_forward"), STDtype.BF16, _shape2(1024, 3))
    _require_tensor(st, String("trace.text_ids"), STDtype.BF16, _shape2(224, 3))
    _require_tensor(st, String("trace.transformer_hidden_states"), STDtype.BF16, _shape3(2, 1024, 64))
    _require_tensor(st, String("trace.transformer_timestep"), STDtype.F32, _shape1(2))
    _require_tensor(st, String("trace.encoder_hidden_states"), STDtype.BF16, _shape3(2, 224, 4096))
    _require_tensor(st, String("trace.attention_mask"), STDtype.BOOL, _shape2(2, 1248))
    _require_tensor(st, String("trace.packed_predicted_flow"), STDtype.BF16, _shape3(2, 1024, 64))
    _require_tensor(st, String("trace.predicted_flow"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("trace.flow"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("output.timestep"), STDtype.I32, _shape1(2))
    _require_tensor(st, String("output.predicted"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("output.target"), STDtype.BF16, _shape4(2, 16, 64, 64))
    _require_tensor(st, String("output.loss_pre_scale"), STDtype.F32, _shape0())
    _require_tensor(st, String("output.loss_for_backward"), STDtype.F32, _shape0())

    _require_close(
        String("output.loss_for_backward"),
        _read_f32_value(st, String("output.loss_for_backward"), 0),
        Float32(0.2957186698913574),
        Float32(1.0e-8),
    )
    _require_close(
        String("output.loss_pre_scale"),
        _read_f32_value(st, String("output.loss_pre_scale"), 0),
        Float32(0.2957186698913574),
        Float32(1.0e-8),
    )
    _require_close(String("batch.loss_weight[0]"), _read_f32_value(st, String("batch.loss_weight"), 0), Float32(1.0), Float32(0.0))
    _require_close(String("batch.loss_weight[1]"), _read_f32_value(st, String("batch.loss_weight"), 1), Float32(1.0), Float32(0.0))
    _require_close(String("trace.sigma[0]"), _read_f32_value(st, String("trace.sigma"), 0), Float32(0.9080000519752502), Float32(1.0e-8))
    _require_close(String("trace.sigma[1]"), _read_f32_value(st, String("trace.sigma"), 1), Float32(0.7100000381469727), Float32(1.0e-8))
    _require_close(String("trace.transformer_timestep[0]"), _read_f32_value(st, String("trace.transformer_timestep"), 0), Float32(0.9070000648498535), Float32(1.0e-8))
    _require_close(String("trace.transformer_timestep[1]"), _read_f32_value(st, String("trace.transformer_timestep"), 1), Float32(0.7090000510215759), Float32(1.0e-8))
    _require_i32(String("output.timestep[0]"), _read_i32_value(st, String("output.timestep"), 0), 907)
    _require_i32(String("output.timestep[1]"), _read_i32_value(st, String("output.timestep"), 1), 709)
    _require_u16(String("batch.latent_image[0]"), _read_u16_payload(st, String("batch.latent_image"), 0), 16416)
    _require_u16(String("output.predicted[0]"), _read_u16_payload(st, String("output.predicted"), 0), 49128)
    _require_u16(String("trace.transformer_hidden_states[0]"), _read_u16_payload(st, String("trace.transformer_hidden_states"), 0), 48963)
    _require_u16(String("trace.encoder_hidden_states[0]"), _read_u16_payload(st, String("trace.encoder_hidden_states"), 0), 15448)
    _require_u8(String("trace.text_attention_mask[0]"), _read_u8_payload(st, String("trace.text_attention_mask"), 0), 1)
    _require_u8(String("trace.attention_mask[0]"), _read_u8_payload(st, String("trace.attention_mask"), 0), 1)


def _check_lora_pair(
    st: SafeTensors,
    snapshot: String,
    block_root: String,
    target: String,
    in_f: Int,
    out_f: Int,
) raises:
    var root = (
        snapshot
        + String(".lora_transformer.")
        + block_root
        + String(".")
        + target
    )
    _require_tensor(
        st,
        root + String(".lora_down.weight"),
        STDtype.F32,
        _shape2(RANK, in_f),
    )
    _require_tensor(
        st,
        root + String(".lora_up.weight"),
        STDtype.F32,
        _shape2(out_f, RANK),
    )


def _check_double_block(st: SafeTensors, snapshot: String, block: Int) raises:
    var root = String("transformer_blocks.") + String(block)
    _check_lora_pair(st, snapshot, root, String("attn.to_q"), D, D)
    _check_lora_pair(st, snapshot, root, String("attn.to_k"), D, D)
    _check_lora_pair(st, snapshot, root, String("attn.to_v"), D, D)
    _check_lora_pair(st, snapshot, root, String("attn.to_out.0"), D, D)
    _check_lora_pair(st, snapshot, root, String("attn.add_q_proj"), D, D)
    _check_lora_pair(st, snapshot, root, String("attn.add_k_proj"), D, D)
    _check_lora_pair(st, snapshot, root, String("attn.add_v_proj"), D, D)
    _check_lora_pair(st, snapshot, root, String("attn.to_add_out"), D, D)
    _check_lora_pair(st, snapshot, root, String("ff.net.0.proj"), D, FMLP)
    _check_lora_pair(st, snapshot, root, String("ff.net.2"), FMLP, D)


def _check_single_block(st: SafeTensors, snapshot: String, block: Int) raises:
    var root = String("single_transformer_blocks.") + String(block)
    _check_lora_pair(st, snapshot, root, String("attn.to_q"), D, D)
    _check_lora_pair(st, snapshot, root, String("attn.to_k"), D, D)
    _check_lora_pair(st, snapshot, root, String("attn.to_v"), D, D)


def _check_snapshot(st: SafeTensors, snapshot: String) raises:
    for block in range(NUM_DOUBLE):
        _check_double_block(st, snapshot, block)
    for block in range(NUM_SINGLE):
        _check_single_block(st, snapshot, block)


def _check_adapter_dump() raises:
    var st = SafeTensors.open(String(ADAPTER_DUMP))
    _require(len(st.tensors) == 2432, String("expected 2432 Chroma adapter tensors"))

    _check_snapshot(st, String("adapter_before"))
    _check_snapshot(st, String("adapter_pre"))
    _check_snapshot(st, String("adapter_post"))
    _check_snapshot(st, String("adapter_after"))

    var before_down = String("adapter_before.lora_transformer.transformer_blocks.0.attn.to_q.lora_down.weight")
    var before_up = String("adapter_before.lora_transformer.transformer_blocks.0.attn.to_q.lora_up.weight")
    var pre_down = String("adapter_pre.lora_transformer.transformer_blocks.0.attn.to_q.lora_down.weight")
    var post_down = String("adapter_post.lora_transformer.transformer_blocks.0.attn.to_q.lora_down.weight")
    var after_down = String("adapter_after.lora_transformer.transformer_blocks.0.attn.to_q.lora_down.weight")
    var post_up = String("adapter_post.lora_transformer.transformer_blocks.0.attn.to_q.lora_up.weight")
    var after_up = String("adapter_after.lora_transformer.transformer_blocks.0.attn.to_q.lora_up.weight")
    var post_single_up = String("adapter_post.lora_transformer.single_transformer_blocks.37.attn.to_v.lora_up.weight")
    var after_single_up = String("adapter_after.lora_transformer.single_transformer_blocks.37.attn.to_v.lora_up.weight")

    _require_close(String("adapter_before.to_q.down[0]"), _read_f32_value(st, before_down, 0), Float32(-0.01258141454309225), Float32(1.0e-8))
    _require_close(String("adapter_before.to_q.down[1]"), _read_f32_value(st, before_down, 1), Float32(0.0033729986753314734), Float32(1.0e-8))
    _require_close(String("adapter_before.to_q.up[0]"), _read_f32_value(st, before_up, 0), Float32(0.0), Float32(0.0))
    _require_close(String("adapter_pre.to_q.down[0]"), _read_f32_value(st, pre_down, 0), Float32(-0.01258141454309225), Float32(1.0e-8))
    _require_close(String("adapter_post.to_q.down[0]"), _read_f32_value(st, post_down, 0), Float32(-0.01258141454309225), Float32(1.0e-8))
    _require_close(String("adapter_after.to_q.down[0]"), _read_f32_value(st, after_down, 0), Float32(-0.012581377290189266), Float32(1.0e-8))
    _require_close(String("adapter_after.to_q.up[0]"), _read_f32_value(st, after_up, 0), Float32(2.9697333957301453e-05), Float32(1.0e-10))
    _require_close(String("adapter_after.single37.to_v.up[0]"), _read_f32_value(st, after_single_up, 0), Float32(0.00011002632527379319), Float32(1.0e-10))

    _require_after_minus_post_delta(
        st,
        String("after_minus_post to_q.down[0]"),
        post_down,
        after_down,
        0,
        Float32(3.725290298461914e-08),
        UPDATE_DELTA_ATOL,
    )
    _require_after_minus_post_delta(
        st,
        String("after_minus_post to_q.up[0]"),
        post_up,
        after_up,
        0,
        Float32(2.9697333957301453e-05),
        UPDATE_DELTA_ATOL,
    )
    _require_after_minus_post_delta(
        st,
        String("after_minus_post single37.to_v.up[0]"),
        post_single_up,
        after_single_up,
        0,
        Float32(0.00011002632527379319),
        UPDATE_DELTA_ATOL,
    )


def main() raises:
    _check_meta_json()
    _check_step_dump()
    _check_adapter_dump()
    print("[chroma-train-ref-artifact] PASS:", STEP_DUMP, ADAPTER_DUMP, META_JSON)
    print("[chroma-train-ref-artifact] scope=update-delta artifact consumption only; no transformer/backward/AdamW parity")
