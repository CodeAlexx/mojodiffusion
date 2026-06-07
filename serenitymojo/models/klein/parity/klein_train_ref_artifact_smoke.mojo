# klein_train_ref_artifact_smoke.mojo -- Klein/Flux2 OneTrainer one-step artifact gate.
#
# Header/payload-only, no-CUDA gate. It opens the real local OneTrainer Klein
# one-step dump, adapter/gradient dump, and metadata JSON and validates tensor
# names, shapes, dtypes, byte sizes, scalar loss/timestep payloads,
# representative LoRA gradient payloads, and bounded synthetic positive-lr
# AdamW math. This does not claim transformer, backward, real update-bearing
# OneTrainer step, sampler, speed, or image parity.

from std.collections import List
from std.math import sqrt
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors


comptime META_JSON = "/home/alex/onetrainer-mojo/parity/klein_train_ref_meta.json"
comptime STEP_DUMP = "/home/alex/onetrainer-mojo/parity/klein_train_ref_step000.safetensors"
comptime ADAPTER_DUMP = "/home/alex/onetrainer-mojo/parity/klein_train_ref_step000_adapters.safetensors"


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _require_contains(text: String, needle: String, label: String) raises:
    _require(text.find(needle) >= 0, String("Klein metadata missing ") + label)


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


def _require_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    _require(
        _abs(got - expected) <= tol,
        name + String(" value mismatch got=") + String(got) + String(" expected=") + String(expected),
    )


def _adamw_synthetic_positive_lr_delta(p: Float32, grad: Float32) -> Float32:
    # PyTorch AdamW-style decoupled weight decay before the Adam term, matching
    # scripts/check_klein_adapter_grad_update_replay.py. This is bounded
    # optimizer math from the captured gradient, not a real later model step.
    comptime lr = Float32(2.9999999999999997e-06)
    comptime beta1 = Float32(0.9)
    comptime beta2 = Float32(0.999)
    comptime eps = Float32(1.0e-8)
    comptime weight_decay = Float32(0.01)
    var exp_avg_step1 = (Float32(1.0) - beta1) * grad
    var exp_avg_sq_step1 = (Float32(1.0) - beta2) * grad * grad
    var exp_avg_step2 = beta1 * exp_avg_step1 + (Float32(1.0) - beta1) * grad
    var exp_avg_sq_step2 = beta2 * exp_avg_sq_step1 + (Float32(1.0) - beta2) * grad * grad
    var bias_correction1 = Float32(1.0) - beta1 * beta1
    var bias_correction2 = Float32(1.0) - beta2 * beta2
    var m_hat = exp_avg_step2 / bias_correction1
    var v_hat = exp_avg_sq_step2 / bias_correction2
    var after_weight_decay = p * (Float32(1.0) - lr * weight_decay)
    var after = after_weight_decay - lr * m_hat / (sqrt(v_hat) + eps)
    return after - p


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    _require(key in st.tensors, String("missing Klein OT tensor ") + key)
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
    _require(meta.byte_length() > 0, String("empty Klein OneTrainer metadata JSON"))
    _require_contains(meta, String("\"prefix\": \"klein_train_ref\""), String("prefix"))
    _require_contains(meta, String("\"max_steps\": 1"), String("max_steps"))
    _require_contains(meta, String("\"adapter_dump\": \"step-with-grads\""), String("adapter_dump"))
    _require_contains(meta, String("\"model_type\": \"FLUX_2\""), String("model_type FLUX_2"))
    _require_contains(meta, String("\"training_method\": \"LORA\""), String("training_method LORA"))
    _require_contains(meta, String("\"train_dtype\": \"BFLOAT_16\""), String("train_dtype BFLOAT_16"))
    _require_contains(meta, String("\"optimizer\": \"ADAMW\""), String("optimizer ADAMW"))
    _require_contains(meta, String("\"learning_rate\": 0.0003"), String("learning_rate"))
    _require_contains(meta, String("\"count\": 288"), String("trainable_parameters.count"))
    _require_contains(meta, String("\"numel\": 43515904"), String("trainable_parameters.numel"))
    _require_contains(meta, String("\"step_index\": 0"), String("steps[0].step_index"))
    _require_contains(meta, String("\"global_step\": 0"), String("steps[0].global_step"))
    _require_contains(meta, String("\"loss_for_backward\": 0.12243738770484924"), String("loss_for_backward"))
    _require_contains(meta, String("\"grad_norm_pre_clip\": 0.005975008010864258"), String("grad_norm_pre_clip"))
    _require_contains(meta, String("\"lr_before\": [\n        0.0"), String("lr_before 0.0"))
    _require_contains(
        meta,
        String("\"lr_after\": [\n        2.9999999999999997e-06"),
        String("lr_after scheduler post-step"),
    )
    _require_contains(meta, String("\"optimizer_before\": {"), String("optimizer_before"))
    _require_contains(meta, String("\"optimizer_after\": {"), String("optimizer_after"))
    _require_contains(meta, String("\"class\": \"AdamW\""), String("optimizer class AdamW"))
    _require_contains(meta, String("\"parameter_entries\": 0"), String("optimizer_before parameter_entries"))
    _require_contains(meta, String("\"parameter_entries\": 288"), String("optimizer_after parameter_entries"))
    _require_contains(meta, String("\"tensor_count\": 864"), String("optimizer_after tensor_count"))
    _require_contains(meta, String("\"tensor_numel\": 87032096"), String("optimizer_after tensor_numel"))
    _require_contains(meta, String("\"exp_avg\""), String("optimizer_after exp_avg"))
    _require_contains(meta, String("\"exp_avg_sq\""), String("optimizer_after exp_avg_sq"))
    _require_contains(meta, String("\"step\""), String("optimizer_after step"))
    _require_contains(
        meta,
        String("\"safetensors\": \"/home/alex/onetrainer-mojo/parity/klein_train_ref_step000.safetensors\""),
        String("step safetensors path"),
    )
    _require_contains(
        meta,
        String("\"adapter_safetensors\": \"/home/alex/onetrainer-mojo/parity/klein_train_ref_step000_adapters.safetensors\""),
        String("adapter safetensors path"),
    )


def _check_step_dump() raises:
    var st = SafeTensors.open(String(STEP_DUMP))
    _require(len(st.tensors) == 22, String("expected 22 Klein step tensors"))

    _require_tensor(st, String("batch.latent_image"), STDtype.BF16, _shape4(1, 32, 64, 64))
    _require_tensor(st, String("batch.loss_weight"), STDtype.F32, _shape1(1))
    _require_tensor(st, String("batch.text_encoder_hidden_state"), STDtype.BF16, _shape3(1, 512, 12288))
    _require_tensor(st, String("batch.tokens"), STDtype.I64, _shape2(1, 512))
    _require_tensor(st, String("batch.tokens_mask"), STDtype.I64, _shape2(1, 512))
    _require_tensor(st, String("output.loss_for_backward"), STDtype.F32, _shape0())
    _require_tensor(st, String("output.loss_pre_scale"), STDtype.F32, _shape0())
    _require_tensor(st, String("output.predicted"), STDtype.BF16, _shape4(1, 32, 64, 64))
    _require_tensor(st, String("output.target"), STDtype.F32, _shape4(1, 32, 64, 64))
    _require_tensor(st, String("output.timestep"), STDtype.I32, _shape1(1))
    _require_tensor(st, String("trace.encoder_hidden_states"), STDtype.BF16, _shape3(1, 512, 12288))
    _require_tensor(st, String("trace.flow"), STDtype.F32, _shape4(1, 128, 32, 32))
    _require_tensor(st, String("trace.image_ids"), STDtype.I64, _shape3(1, 1024, 4))
    _require_tensor(st, String("trace.latent_noise"), STDtype.F32, _shape4(1, 128, 32, 32))
    _require_tensor(st, String("trace.packed_latent_input"), STDtype.BF16, _shape3(1, 1024, 128))
    _require_tensor(st, String("trace.packed_predicted_flow"), STDtype.BF16, _shape3(1, 1024, 128))
    _require_tensor(st, String("trace.predicted_flow"), STDtype.BF16, _shape4(1, 128, 32, 32))
    _require_tensor(st, String("trace.scaled_latent_image"), STDtype.F32, _shape4(1, 128, 32, 32))
    _require_tensor(st, String("trace.scaled_noisy_latent_image"), STDtype.F32, _shape4(1, 128, 32, 32))
    _require_tensor(st, String("trace.sigma"), STDtype.F32, _shape4(1, 1, 1, 1))
    _require_tensor(st, String("trace.text_ids"), STDtype.I64, _shape3(1, 512, 4))
    _require_tensor(st, String("trace.transformer_timestep"), STDtype.F32, _shape1(1))

    _require_close(
        String("output.loss_for_backward"),
        _read_f32_value(st, String("output.loss_for_backward"), 0),
        Float32(0.12243738770484924),
        Float32(1.0e-8),
    )
    _require_close(
        String("output.loss_pre_scale"),
        _read_f32_value(st, String("output.loss_pre_scale"), 0),
        Float32(0.12243738770484924),
        Float32(1.0e-8),
    )
    _require_close(String("batch.loss_weight"), _read_f32_value(st, String("batch.loss_weight"), 0), Float32(1.0), Float32(0.0))
    _require_close(String("trace.sigma"), _read_f32_value(st, String("trace.sigma"), 0), Float32(0.5460000038146973), Float32(1.0e-8))
    _require_close(String("trace.transformer_timestep"), _read_f32_value(st, String("trace.transformer_timestep"), 0), Float32(0.5450000166893005), Float32(1.0e-8))
    _require(
        _read_i32_value(st, String("output.timestep"), 0) == Int32(545),
        String("output.timestep payload mismatch"),
    )


def _check_adapter_dump() raises:
    var st = SafeTensors.open(String(ADAPTER_DUMP))
    _require(len(st.tensors) == 1728, String("expected 1728 Klein adapter/grad tensors"))

    _require_tensor(
        st,
        String("adapter_before.transformer_blocks.0.attn.to_q.lora_down.weight"),
        STDtype.F32,
        _shape2(16, 4096),
    )
    _require_tensor(
        st,
        String("adapter_after.transformer_blocks.7.ff_context.linear_in.lora_up.weight"),
        STDtype.F32,
        _shape2(24576, 16),
    )
    _require_tensor(
        st,
        String("adapter_pre_clip_grad.transformer_blocks.0.attn.to_q.lora_up.weight"),
        STDtype.F32,
        _shape2(4096, 16),
    )
    _require_tensor(
        st,
        String("adapter_post_clip_grad.transformer_blocks.0.attn.to_q.lora_up.weight"),
        STDtype.F32,
        _shape2(4096, 16),
    )
    _require_tensor(
        st,
        String("adapter_after.single_transformer_blocks.23.attn.to_qkv_mlp_proj.lora_up.weight"),
        STDtype.F32,
        _shape2(36864, 16),
    )
    _require_tensor(
        st,
        String("adapter_after.single_transformer_blocks.23.attn.to_out.lora_down.weight"),
        STDtype.F32,
        _shape2(16, 16384),
    )

    _require_close(
        String("adapter_before.double0.to_q.down[0]"),
        _read_f32_value(
            st,
            String("adapter_before.transformer_blocks.0.attn.to_q.lora_down.weight"),
            0,
        ),
        Float32(-0.005761606618762016),
        Float32(1.0e-8),
    )
    _require_close(
        String("adapter_pre_clip_grad.double0.to_q.up[0]"),
        _read_f32_value(
            st,
            String("adapter_pre_clip_grad.transformer_blocks.0.attn.to_q.lora_up.weight"),
            0,
        ),
        Float32(0.000001259148120880127),
        Float32(1.0e-12),
    )
    _require_close(
        String("adapter_post_clip_grad.single0.to_out.up[0]"),
        _read_f32_value(
            st,
            String("adapter_post_clip_grad.single_transformer_blocks.0.attn.to_out.lora_up.weight"),
            0,
        ),
        Float32(-0.0000016838312149047852),
        Float32(1.0e-12),
    )

    var p = _read_f32_value(
        st,
        String("adapter_post_clip.transformer_blocks.0.attn.to_q.lora_up.weight"),
        0,
    )
    var grad = _read_f32_value(
        st,
        String("adapter_post_clip_grad.transformer_blocks.0.attn.to_q.lora_up.weight"),
        0,
    )
    _require_close(
        String("synthetic_adamw.double0.to_q.up[0]"),
        _adamw_synthetic_positive_lr_delta(p, grad),
        Float32(-2.9763620971370762e-06),
        Float32(5.0e-10),
    )

    p = _read_f32_value(
        st,
        String("adapter_post_clip.transformer_blocks.7.ff_context.linear_in.lora_up.weight"),
        1,
    )
    grad = _read_f32_value(
        st,
        String("adapter_post_clip_grad.transformer_blocks.7.ff_context.linear_in.lora_up.weight"),
        1,
    )
    _require_close(
        String("synthetic_adamw.double7.ff_context.linear_in.up[1]"),
        _adamw_synthetic_positive_lr_delta(p, grad),
        Float32(2.3773895989441487e-06),
        Float32(5.0e-10),
    )

    p = _read_f32_value(
        st,
        String("adapter_post_clip.single_transformer_blocks.23.attn.to_qkv_mlp_proj.lora_up.weight"),
        589823,
    )
    grad = _read_f32_value(
        st,
        String("adapter_post_clip_grad.single_transformer_blocks.23.attn.to_qkv_mlp_proj.lora_up.weight"),
        589823,
    )
    _require_close(
        String("synthetic_adamw.single23.qkv_mlp.up[589823]"),
        _adamw_synthetic_positive_lr_delta(p, grad),
        Float32(2.9575449240466656e-06),
        Float32(5.0e-10),
    )

    p = _read_f32_value(
        st,
        String("adapter_post_clip.transformer_blocks.0.attn.to_q.lora_down.weight"),
        0,
    )
    grad = _read_f32_value(
        st,
        String("adapter_post_clip_grad.transformer_blocks.0.attn.to_q.lora_down.weight"),
        0,
    )
    _require_close(
        String("synthetic_adamw.double0.to_q.down[0]"),
        _adamw_synthetic_positive_lr_delta(p, grad),
        Float32(1.7284819842783294e-10),
        Float32(5.0e-10),
    )


def main() raises:
    _check_meta_json()
    _check_step_dump()
    _check_adapter_dump()
    print("[klein-train-ref-artifact] PASS:", META_JSON, STEP_DUMP, ADAPTER_DUMP)
