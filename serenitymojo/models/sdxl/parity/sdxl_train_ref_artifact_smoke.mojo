# sdxl_train_ref_artifact_smoke.mojo -- SDXL OneTrainer one-step artifact gate.
#
# Header/payload-only, no-CUDA gate. It opens the real local OneTrainer SDXL
# one-step dump, adapter update dump, and metadata JSON and validates tensor
# names, shapes, dtypes, byte sizes, optimizer/LR metadata, and representative
# update-delta payloads. This is OneTrainer update-delta artifact consumption;
# it does not claim UNet, backward, AdamW, sampler, or image parity.

from std.collections import List
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors


comptime STEP_DUMP = "/home/alex/onetrainer-mojo/parity/sdxl_train_ref_step000.safetensors"
comptime ADAPTER_DUMP = "/home/alex/onetrainer-mojo/parity/sdxl_train_ref_step000_adapters.safetensors"
comptime META_JSON = "/home/alex/onetrainer-mojo/parity/sdxl_train_ref_meta.json"
comptime TRAINABLE_PARAMETER_ENTRIES = 1588
comptime ADAPTER_AFTER_MINUS_POST_TOLERANCE = Float32(1.0e-10)


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _require_contains(text: String, needle: String, label: String) raises:
    _require(text.find(needle) >= 0, String("missing SDXL OT meta field: ") + label)


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


def _max_abs(a: Float32, b: Float32) -> Float32:
    var aa = _abs(a)
    var bb = _abs(b)
    if bb > aa:
        return bb
    return aa


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


def _require_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    _require(
        _abs(got - expected) <= tol,
        name + String(" value mismatch got=") + String(got) + String(" expected=") + String(expected),
    )


def _require_after_minus_post_key(
    st: SafeTensors,
    post_key: String,
    after_key: String,
    expected_first: Float32,
    expected_last: Float32,
    label: String,
) raises -> Float32:
    var post_info = st.tensor_info(post_key)
    var after_info = st.tensor_info(after_key)
    _require(
        post_info.dtype == STDtype.F32 and after_info.dtype == STDtype.F32,
        String("adapter_after - adapter_post dtype mismatch for ") + label,
    )
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
        label + String(" after_minus_post[0] == adapter_after - adapter_post"),
        after_minus_post,
        expected_first,
        ADAPTER_AFTER_MINUS_POST_TOLERANCE,
    )
    _require_close(
        label + String(" adapter_after - adapter_post[last]"),
        last_after_minus_post,
        expected_last,
        ADAPTER_AFTER_MINUS_POST_TOLERANCE,
    )
    return _max_abs(after_minus_post, last_after_minus_post)


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    _require(key in st.tensors, String("missing SDXL OT tensor ") + key)
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
    var meta = Path(String(META_JSON)).read_text()
    _require(meta.byte_length() > 0, String("empty SDXL OneTrainer metadata JSON"))
    _require_contains(meta, String("\"prefix\": \"sdxl_train_ref\""), String("prefix"))
    _require_contains(meta, String("\"max_steps\": 1"), String("max_steps"))
    _require_contains(meta, String("\"adapter_dump\": \"step\""), String("adapter_dump"))
    _require_contains(meta, String("\"optimizer\": \"ADAMW\""), String("optimizer ADAMW"))
    _require_contains(meta, String("\"learning_rate\": 0.0001"), String("learning_rate"))
    _require_contains(meta, String("\"count\": 1588"), String("trainable_parameters.count"))
    _require_contains(meta, String("\"numel\": 49412736"), String("trainable_parameters.numel"))
    _require_contains(meta, String("\"step_index\": 0"), String("steps[0].step_index"))
    _require_contains(meta, String("\"lr_before\": [\n        0.0001"), String("lr_before"))
    _require_contains(meta, String("\"lr_after\": [\n        0.0001"), String("lr_after"))
    _require_contains(meta, String("\"optimizer_before\": {"), String("optimizer_before"))
    _require_contains(meta, String("\"optimizer_after\": {"), String("optimizer_after"))
    _require_contains(meta, String("\"class\": \"AdamW\""), String("optimizer class AdamW"))
    _require_contains(meta, String("\"parameter_entries\": 0"), String("optimizer_before parameter_entries"))
    _require_contains(
        meta,
        String("\"parameter_entries\": ") + String(TRAINABLE_PARAMETER_ENTRIES),
        String("optimizer_after parameter_entries"),
    )
    _require_contains(meta, String("\"keys\": ["), String("optimizer_after state keys"))
    _require_contains(meta, String("\"exp_avg\""), String("optimizer_after exp_avg"))
    _require_contains(meta, String("\"exp_avg_sq\""), String("optimizer_after exp_avg_sq"))
    _require_contains(meta, String("\"step\""), String("optimizer_after step"))
    _require_contains(
        meta,
        String("\"safetensors\": \"/home/alex/onetrainer-mojo/parity/sdxl_train_ref_step000.safetensors\""),
        String("step safetensors path"),
    )
    _require_contains(
        meta,
        String("\"adapter_safetensors\": \"/home/alex/onetrainer-mojo/parity/sdxl_train_ref_step000_adapters.safetensors\""),
        String("adapter safetensors path"),
    )


def _check_step_dump() raises:
    var st = SafeTensors.open(String(STEP_DUMP))
    _require(len(st.tensors) == 31, String("expected 31 SDXL step tensors"))

    _require_tensor(st, String("batch.latent_image"), STDtype.BF16, _shape4(1, 4, 168, 96))
    _require_tensor(st, String("batch.loss_weight"), STDtype.F32, _shape1(1))
    _require_tensor(st, String("batch.text_encoder_1_hidden_state"), STDtype.F32, _shape3(1, 77, 768))
    _require_tensor(st, String("batch.text_encoder_2_hidden_state"), STDtype.F32, _shape3(1, 77, 1280))
    _require_tensor(st, String("batch.text_encoder_2_pooled_state"), STDtype.BF16, _shape2(1, 1280))
    _require_tensor(st, String("batch.tokens_1"), STDtype.I64, _shape2(1, 77))
    _require_tensor(st, String("batch.tokens_2"), STDtype.I64, _shape2(1, 77))
    _require_tensor(st, String("output.loss_for_backward"), STDtype.F32, _shape0())
    _require_tensor(st, String("output.loss_pre_scale"), STDtype.F32, _shape0())
    _require_tensor(st, String("output.predicted"), STDtype.BF16, _shape4(1, 4, 168, 96))
    _require_tensor(st, String("output.target"), STDtype.BF16, _shape4(1, 4, 168, 96))
    _require_tensor(st, String("output.timestep"), STDtype.I32, _shape1(1))
    _require_tensor(st, String("trace.added_cond_text_embeds"), STDtype.F32, _shape2(1, 1280))
    _require_tensor(st, String("trace.added_cond_time_ids"), STDtype.BF16, _shape2(1, 6))
    _require_tensor(st, String("trace.combined_pooled_text_encoder_2_output"), STDtype.F32, _shape2(1, 1280))
    _require_tensor(st, String("trace.combined_text_encoder_output"), STDtype.F32, _shape3(1, 77, 2048))
    _require_tensor(st, String("trace.encode_text.pooled_text_encoder_2_output"), STDtype.BF16, _shape2(1, 1280))
    _require_tensor(st, String("trace.encode_text.text_encoder_1_output"), STDtype.F32, _shape3(1, 77, 768))
    _require_tensor(st, String("trace.encode_text.text_encoder_2_output"), STDtype.F32, _shape3(1, 77, 1280))
    _require_tensor(st, String("trace.encode_text.tokens_1"), STDtype.I64, _shape2(1, 77))
    _require_tensor(st, String("trace.encode_text.tokens_2"), STDtype.I64, _shape2(1, 77))
    _require_tensor(st, String("trace.encoder_hidden_states"), STDtype.BF16, _shape3(1, 77, 2048))
    _require_tensor(st, String("trace.latent_input"), STDtype.BF16, _shape4(1, 4, 168, 96))
    _require_tensor(st, String("trace.latent_noise"), STDtype.BF16, _shape4(1, 4, 168, 96))
    _require_tensor(st, String("trace.pooled_text_encoder_2_output"), STDtype.F32, _shape2(1, 1280))
    _require_tensor(st, String("trace.predicted_latent_noise"), STDtype.BF16, _shape4(1, 4, 168, 96))
    _require_tensor(st, String("trace.scaled_latent_image"), STDtype.BF16, _shape4(1, 4, 168, 96))
    _require_tensor(st, String("trace.scaled_noisy_latent_image"), STDtype.BF16, _shape4(1, 4, 168, 96))
    _require_tensor(st, String("trace.text_encoder_1_output"), STDtype.F32, _shape3(1, 77, 768))
    _require_tensor(st, String("trace.text_encoder_2_output"), STDtype.F32, _shape3(1, 77, 1280))
    _require_tensor(st, String("trace.unet_timestep"), STDtype.I32, _shape1(1))

    _require_close(
        String("output.loss_for_backward"),
        _read_f32_value(st, String("output.loss_for_backward"), 0),
        Float32(0.13533265888690948),
        Float32(1.0e-8),
    )
    _require_close(
        String("batch.loss_weight"),
        _read_f32_value(st, String("batch.loss_weight"), 0),
        Float32(1.0),
        Float32(0.0),
    )
    _require(
        _read_i32_value(st, String("output.timestep"), 0) == Int32(399),
        String("output.timestep payload mismatch"),
    )
    _require(
        _read_i32_value(st, String("trace.unet_timestep"), 0) == Int32(399),
        String("trace.unet_timestep payload mismatch"),
    )


def _check_adapter_dump() raises:
    var st = SafeTensors.open(String(ADAPTER_DUMP))
    _require(len(st.tensors) == 6352, String("expected 6352 SDXL adapter tensors"))

    var before_down = String("adapter_before.lora_unet.conv_in.lora_down.weight")
    var before_up = String("adapter_before.lora_unet.conv_in.lora_up.weight")
    var pre_down = String("adapter_pre.lora_unet.conv_in.lora_down.weight")
    var pre_up = String("adapter_pre.lora_unet.conv_in.lora_up.weight")
    var post_down = String("adapter_post.lora_unet.conv_in.lora_down.weight")
    var post_up = String("adapter_post.lora_unet.conv_in.lora_up.weight")
    var after_down = String("adapter_after.lora_unet.conv_in.lora_down.weight")
    var after_up = String("adapter_after.lora_unet.conv_in.lora_up.weight")

    _require_tensor(st, before_down, STDtype.F32, _shape4(16, 4, 3, 3))
    _require_tensor(st, before_up, STDtype.F32, _shape4(320, 16, 1, 1))
    _require_tensor(st, pre_down, STDtype.F32, _shape4(16, 4, 3, 3))
    _require_tensor(st, pre_up, STDtype.F32, _shape4(320, 16, 1, 1))
    _require_tensor(st, post_down, STDtype.F32, _shape4(16, 4, 3, 3))
    _require_tensor(st, post_up, STDtype.F32, _shape4(320, 16, 1, 1))
    _require_tensor(st, after_down, STDtype.F32, _shape4(16, 4, 3, 3))
    _require_tensor(st, after_up, STDtype.F32, _shape4(320, 16, 1, 1))
    _require_tensor(
        st,
        String("adapter_after.lora_unet.add_embedding.linear_1.lora_down.weight"),
        STDtype.F32,
        _shape2(16, 2816),
    )
    _require_tensor(
        st,
        String("adapter_after.lora_unet.add_embedding.linear_1.lora_up.weight"),
        STDtype.F32,
        _shape2(1280, 16),
    )
    _require_tensor(
        st,
        String("adapter_after.lora_unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_down.weight"),
        STDtype.F32,
        _shape2(16, 2048),
    )
    _require_tensor(
        st,
        String("adapter_after.lora_unet.up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2.lora_up.weight"),
        STDtype.F32,
        _shape2(640, 16),
    )

    var before0 = _read_f32_value(st, before_down, 0)
    var after0 = _read_f32_value(st, after_down, 0)
    _require_close(
        String("adapter_before.conv_in.down[0]"),
        before0,
        Float32(0.03245890140533447),
        Float32(1.0e-7),
    )
    _require_close(
        String("adapter_after.conv_in.down[0]"),
        after0,
        Float32(0.032458867877721786),
        Float32(1.0e-7),
    )
    _require_close(
        String("adapter_before.conv_in.up[0]"),
        _read_f32_value(st, before_up, 0),
        Float32(0.0),
        Float32(0.0),
    )
    _require(
        _abs(before0 - after0) > Float32(1.0e-8),
        String("adapter before/after value did not change"),
    )

    # Representative OneTrainer update delta: adapter_post -> adapter_after.
    # Production SDXL training anchors this class of update at
    # sdxl_lora_adamw_step(...); this artifact smoke consumes the oracle delta
    # but does not execute backward or that AdamW path.
    var down_after_minus_post_max_abs = _require_after_minus_post_key(
        st,
        post_down,
        after_down,
        Float32(-3.3527612686157227e-08),
        Float32(1.4901161193847656e-07),
        String("conv_in.down"),
    )
    var up_after_minus_post_max_abs = _require_after_minus_post_key(
        st,
        post_up,
        after_up,
        Float32(-9.998169116443023e-05),
        Float32(-9.99948097160086e-05),
        String("conv_in.up"),
    )
    _require_close(
        String("after_minus_post sampled down max_abs"),
        down_after_minus_post_max_abs,
        Float32(1.4901161193847656e-07),
        ADAPTER_AFTER_MINUS_POST_TOLERANCE,
    )
    _require_close(
        String("after_minus_post sampled up max_abs"),
        up_after_minus_post_max_abs,
        Float32(9.99948097160086e-05),
        ADAPTER_AFTER_MINUS_POST_TOLERANCE,
    )


def main() raises:
    _check_meta_json()
    _check_step_dump()
    _check_adapter_dump()
    print("[sdxl-train-ref-artifact] PASS:", STEP_DUMP, ADAPTER_DUMP, META_JSON)
    print(
        "[sdxl-train-ref-artifact] update-delta artifact: lr_before=0.0001, "
        "lr_after=0.0001, optimizer_before/optimizer_after parameter_entries=0->1588, "
        "after_minus_post sampled max_abs=9.99948097160086e-05"
    )
