# anima_train_step_ref_artifact_smoke.mojo -- Anima OneTrainer one-step artifact gate.
#
# Header/payload-only, no-CUDA gate. It opens the real local Anima OneTrainer
# one-step dump, adapter dump, and metadata JSON, then validates tensor names,
# shapes, dtypes, byte sizes, loss payload, optimizer state-init metadata, and
# representative unchanged adapter payloads. This proves artifact/state-init
# consumption only; it does not claim transformer, backward, or AdamW numeric
# parity.
#
# Production AdamW anchor: train_anima_real.mojo calls anima_lora_adamw_step(
# for the real update path. This bounded smoke does not invoke that path.

from std.collections import List
from std.memory import alloc

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    BytePtr, O_RDONLY, file_size, sys_close, sys_open, sys_pread,
)
from serenitymojo.io.safetensors import SafeTensors


comptime STEP_DUMP = "/home/alex/onetrainer-mojo/parity/anima_train_ref_step000.safetensors"
comptime ADAPTER_DUMP = "/home/alex/onetrainer-mojo/parity/anima_train_ref_step000_adapters.safetensors"
comptime META_JSON = "/home/alex/onetrainer-mojo/parity/anima_train_ref_meta.json"
comptime NUM_BLOCKS = 28
comptime ANIMA_MODULES = 10
comptime STAGES = 4
comptime RANK = 16
comptime D_MODEL = 2048
comptime JOINT = 1024
comptime F_MLP = 8192
comptime TRAINABLE_PARAMETER_ENTRIES = 560
comptime OPTIMIZER_BEFORE_PARAMETER_ENTRIES = 0
comptime OPTIMIZER_AFTER_PARAMETER_ENTRIES = TRAINABLE_PARAMETER_ENTRIES
comptime OPTIMIZER_AFTER_TENSOR_COUNT = 1680
comptime OPTIMIZER_AFTER_TENSOR_NUMEL = 45875760
comptime STATE_INIT_ATOL = Float32(0.0)


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _read_utf8_file(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("Anima OT smoke cannot open ") + path)
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
        raise Error(String("Anima OT smoke short read from ") + path)
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
        String("Anima OT meta missing ") + label + String(": ") + needle,
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
    _require(
        index >= 0 and index < info.size // 4,
        String("F32 index out of bounds for ") + key,
    )
    var bytes = st.tensor_bytes(key)
    var fp = bytes.unsafe_ptr().bitcast[Float32]()
    return fp[index]


def _read_i32_value(st: SafeTensors, key: String, index: Int) raises -> Int32:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.I32, String("expected I32 payload for ") + key)
    _require(
        index >= 0 and index < info.size // 4,
        String("I32 index out of bounds for ") + key,
    )
    var bytes = st.tensor_bytes(key)
    var ip = bytes.unsafe_ptr().bitcast[Int32]()
    return ip[index]


def _require_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    _require(
        _abs(got - expected) <= tol,
        name
        + String(" value mismatch got=")
        + String(got)
        + String(" expected=")
        + String(expected),
    )


def _require_after_minus_post_zero(
    st: SafeTensors, post_key: String, after_key: String, index: Int
) raises:
    var after_minus_post = _read_f32_value(st, after_key, index) - _read_f32_value(st, post_key, index)
    var max_abs = _abs(after_minus_post)
    var tolerance = STATE_INIT_ATOL
    _require(
        max_abs <= tolerance,
        String("adapter_after - adapter_post changed for ")
        + after_key
        + String(" index=")
        + String(index)
        + String(" after_minus_post=")
        + String(after_minus_post)
        + String(" max_abs=")
        + String(max_abs)
        + String(" tolerance=")
        + String(tolerance),
    )


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, expected_shape: List[Int]
) raises -> Int:
    _require(key in st.tensors, String("missing Anima OT tensor ") + key)
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
    var expected_bytes = _numel(expected_shape) * dtype.byte_size()
    _require(
        info.size == expected_bytes,
        String("byte-size mismatch for ") + key,
    )
    return info.size


def _stage(slot: Int) raises -> String:
    if slot == 0:
        return String("adapter_before")
    if slot == 1:
        return String("adapter_pre")
    if slot == 2:
        return String("adapter_post")
    if slot == 3:
        return String("adapter_after")
    raise Error(String("unsupported Anima adapter dump stage ") + String(slot))


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


def _adapter_prefix(stage: Int, block: Int, slot: Int) raises -> String:
    return (
        _stage(stage)
        + String(".transformer.transformer_blocks.")
        + String(block)
        + String(".")
        + _module(slot)
    )


def _check_meta_json() raises:
    var meta = _read_utf8_file(String(META_JSON))

    _require_contains(meta, String('"producer": "scripts/anima_dump_train_ref.py"'), String("producer"))
    _require_contains(meta, String('"prefix": "anima_train_ref"'), String("train-ref prefix"))
    _require_contains(meta, String('"optimizer": "ADAMW"'), String("ADAMW optimizer"))
    _require_contains(
        meta,
        String('"safetensors": "') + String(STEP_DUMP) + String('"'),
        String("exact step dump path"),
    )
    _require_contains(
        meta,
        String('"adapter_safetensors": "') + String(ADAPTER_DUMP) + String('"'),
        String("exact adapter dump path"),
    )
    _require_contains(meta, String('"trainable_parameters"'), String("trainable parameter block"))
    _require_contains(
        meta,
        String('"count": ') + String(TRAINABLE_PARAMETER_ENTRIES),
        String("trainable parameter count"),
    )

    _require_contains(meta, String('"lr_before"'), String("lr_before"))
    _require_contains(meta, String('"lr_after"'), String("lr_after"))
    _require_contains(meta, String('0.0'), String("zero lr_before update value"))
    _require_contains(
        meta,
        String('1.5000000000000002e-07'),
        String("recorded post-step lr_after"),
    )
    _require_contains(meta, String('"optimizer_before"'), String("optimizer_before"))
    _require_contains(meta, String('"optimizer_after"'), String("optimizer_after"))
    _require_contains(meta, String('"parameter_entries"'), String("parameter_entries"))
    _require_contains(
        meta,
        String('"parameter_entries": ') + String(OPTIMIZER_BEFORE_PARAMETER_ENTRIES),
        String("optimizer_before parameter_entries"),
    )
    _require_contains(
        meta,
        String('"parameter_entries": ') + String(OPTIMIZER_AFTER_PARAMETER_ENTRIES),
        String("optimizer_after parameter_entries"),
    )
    _require_contains(
        meta,
        String('"tensor_count": ') + String(OPTIMIZER_AFTER_TENSOR_COUNT),
        String("optimizer_after tensor_count"),
    )
    _require_contains(
        meta,
        String('"tensor_numel": ') + String(OPTIMIZER_AFTER_TENSOR_NUMEL),
        String("optimizer_after tensor_numel"),
    )


def _check_step_dump() raises:
    var st = SafeTensors.open(String(STEP_DUMP))
    _require(len(st.tensors) == 28, String("expected 28 Anima step tensors"))

    var latent_shape = _shape5(2, 16, 1, 64, 64)
    var context_shape = _shape3(2, 512, 1024)
    var token_shape = _shape2(2, 512)
    var total = 0
    total += _require_tensor(st, String("batch.t5_tokens"), STDtype.I64, token_shape)
    total += _require_tensor(st, String("batch.t5_tokens_mask"), STDtype.I64, token_shape)
    total += _require_tensor(st, String("batch.tokens"), STDtype.I64, token_shape)
    total += _require_tensor(st, String("batch.tokens_mask"), STDtype.I64, token_shape)
    total += _require_tensor(st, String("trace.encode_text.tokens"), STDtype.I64, token_shape)
    total += _require_tensor(st, String("trace.encode_text.tokens_mask"), STDtype.I64, token_shape)
    total += _require_tensor(st, String("batch.loss_weight"), STDtype.F32, _shape1(2))
    total += _require_tensor(st, String("output.loss_for_backward"), STDtype.F32, _shape0())
    total += _require_tensor(st, String("output.loss_pre_scale"), STDtype.F32, _shape0())
    total += _require_tensor(st, String("output.target"), STDtype.F32, latent_shape)
    total += _require_tensor(st, String("trace.flow"), STDtype.F32, latent_shape)
    total += _require_tensor(st, String("trace.latent_noise"), STDtype.F32, latent_shape)
    total += _require_tensor(st, String("trace.noise_source_tensor"), STDtype.F32, latent_shape)
    total += _require_tensor(st, String("trace.scaled_latent_image"), STDtype.F32, latent_shape)
    total += _require_tensor(st, String("trace.scaled_noisy_latent_image"), STDtype.F32, latent_shape)
    total += _require_tensor(st, String("trace.sigma"), STDtype.F32, _shape5(2, 1, 1, 1, 1))
    total += _require_tensor(st, String("trace.transformer_timestep"), STDtype.F32, _shape1(2))
    total += _require_tensor(st, String("output.timestep"), STDtype.I32, _shape1(2))
    total += _require_tensor(st, String("batch.latent_image"), STDtype.BF16, latent_shape)
    total += _require_tensor(st, String("batch.text_encoder_hidden_state"), STDtype.BF16, context_shape)
    total += _require_tensor(st, String("output.predicted"), STDtype.BF16, latent_shape)
    total += _require_tensor(st, String("trace.encode_text.cached_hidden_state"), STDtype.BF16, context_shape)
    total += _require_tensor(st, String("trace.encoder_hidden_states"), STDtype.BF16, context_shape)
    total += _require_tensor(st, String("trace.latent_image_before_scale"), STDtype.BF16, latent_shape)
    total += _require_tensor(st, String("trace.padding_mask"), STDtype.BF16, _shape4(1, 1, 512, 512))
    total += _require_tensor(st, String("trace.predicted_flow"), STDtype.BF16, latent_shape)
    total += _require_tensor(st, String("trace.text_encoder_output"), STDtype.BF16, context_shape)
    total += _require_tensor(st, String("trace.transformer_hidden_states"), STDtype.BF16, latent_shape)
    _require(st.data_size() == total, String("Anima step dump data byte size mismatch"))

    _require_close(
        String("output.loss_for_backward"),
        _read_f32_value(st, String("output.loss_for_backward"), 0),
        Float32(0.0667838305234909),
        Float32(1.0e-8),
    )
    _require_close(
        String("output.loss_pre_scale"),
        _read_f32_value(st, String("output.loss_pre_scale"), 0),
        Float32(0.0667838305234909),
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
    _require(
        _read_i32_value(st, String("output.timestep"), 0) == Int32(545),
        String("output.timestep[0] payload mismatch"),
    )
    _require(
        _read_i32_value(st, String("output.timestep"), 1) == Int32(365),
        String("output.timestep[1] payload mismatch"),
    )


def _check_adapter_dump() raises:
    var st = SafeTensors.open(String(ADAPTER_DUMP))
    _require(len(st.tensors) == 2240, String("expected 2240 Anima adapter tensors"))

    var total = 0
    for stage in range(STAGES):
        for block in range(NUM_BLOCKS):
            for slot in range(ANIMA_MODULES):
                var prefix = _adapter_prefix(stage, block, slot)
                total += _require_tensor(
                    st,
                    prefix + String(".lora_down.weight"),
                    STDtype.F32,
                    _shape2(RANK, _input_dim(slot)),
                )
                total += _require_tensor(
                    st,
                    prefix + String(".lora_up.weight"),
                    STDtype.F32,
                    _shape2(_output_dim(slot), RANK),
                )
    _require(st.data_size() == total, String("Anima adapter dump data byte size mismatch"))

    var before_q_down = String("adapter_before.transformer.transformer_blocks.0.attn1.to_q.lora_down.weight")
    var before_q_up = String("adapter_before.transformer.transformer_blocks.0.attn1.to_q.lora_up.weight")
    var pre_q_down = String("adapter_pre.transformer.transformer_blocks.0.attn1.to_q.lora_down.weight")
    var post_q_down = String("adapter_post.transformer.transformer_blocks.0.attn1.to_q.lora_down.weight")
    var after_q_down = String("adapter_after.transformer.transformer_blocks.0.attn1.to_q.lora_down.weight")
    var post_ca_k_down = String("adapter_post.transformer.transformer_blocks.0.attn2.to_k.lora_down.weight")
    var after_ca_k_down = String("adapter_after.transformer.transformer_blocks.0.attn2.to_k.lora_down.weight")
    var post_ff2_down = String("adapter_post.transformer.transformer_blocks.27.ff.net.2.lora_down.weight")
    var after_ff2_down = String("adapter_after.transformer.transformer_blocks.27.ff.net.2.lora_down.weight")
    var post_ff2_up = String("adapter_post.transformer.transformer_blocks.27.ff.net.2.lora_up.weight")
    var after_ff2_up = String("adapter_after.transformer.transformer_blocks.27.ff.net.2.lora_up.weight")

    _require_close(
        String("adapter_before.block0.q.down[0]"),
        _read_f32_value(st, before_q_down, 0),
        Float32(0.003058146219700575),
        Float32(1.0e-8),
    )
    _require_close(
        String("adapter_before.block0.q.down[1]"),
        _read_f32_value(st, before_q_down, 1),
        Float32(0.007880986668169498),
        Float32(1.0e-8),
    )
    _require_close(
        String("adapter_before.block0.q.up[0]"),
        _read_f32_value(st, before_q_up, 0),
        Float32(0.0),
        Float32(0.0),
    )
    _require_close(
        String("adapter_pre.block0.q.down[0]"),
        _read_f32_value(st, pre_q_down, 0),
        Float32(0.003058146219700575),
        Float32(1.0e-8),
    )
    _require_close(
        String("adapter_post.block0.q.down[0]"),
        _read_f32_value(st, post_q_down, 0),
        Float32(0.003058146219700575),
        Float32(1.0e-8),
    )
    _require_close(
        String("adapter_after.block0.q.down[0]"),
        _read_f32_value(st, after_q_down, 0),
        Float32(0.003058146219700575),
        Float32(1.0e-8),
    )
    _require_after_minus_post_zero(st, post_q_down, after_q_down, 0)
    _require_after_minus_post_zero(st, post_q_down, after_q_down, 1)
    _require_close(
        String("adapter_after.block0.cross_k.down[0]"),
        _read_f32_value(st, after_ca_k_down, 0),
        Float32(-0.0011359453201293945),
        Float32(1.0e-8),
    )
    _require_after_minus_post_zero(st, post_ca_k_down, after_ca_k_down, 0)
    _require_close(
        String("adapter_after.block27.ff2.down[0]"),
        _read_f32_value(st, after_ff2_down, 0),
        Float32(-0.0018678330816328526),
        Float32(1.0e-8),
    )
    _require_after_minus_post_zero(st, post_ff2_down, after_ff2_down, 0)
    _require_close(
        String("adapter_after.block27.ff2.up[0]"),
        _read_f32_value(st, after_ff2_up, 0),
        Float32(0.0),
        Float32(0.0),
    )
    _require_after_minus_post_zero(st, post_ff2_up, after_ff2_up, 0)


def main() raises:
    _check_meta_json()
    _check_step_dump()
    _check_adapter_dump()
    print("[anima-train-step-ref-artifact] PASS:", STEP_DUMP, ADAPTER_DUMP, META_JSON)
    print(
        "[anima-train-step-ref-artifact] scope=zero-lr state-init artifact consumption only; no transformer/backward/AdamW parity"
    )
