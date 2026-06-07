# qwen_train_control_wiring_smoke.mojo -- config/control gate for Qwen training.
#
# Scope: TrainConfig + sample cadence + OneTrainer checkpoint/offload policy
# reachability. This deliberately does not load Qwen weights or run transformer
# math.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/qwen_train_control_wiring_smoke.mojo

from std.memory import alloc

from serenitymojo.io.ffi import (
    sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.sample_prompt_config import (
    SAMPLE_UNIT_NEVER,
    SAMPLE_UNIT_STEP,
    next_sample_completed_step,
    should_sample_completed_step,
)
from serenitymojo.training.train_qwenimage_real import (
    qwen_offload_config_from_train_config,
    qwen_patchified_out_channels,
    qwen_sample_cadence_from_train_config,
    qwen_state_path_for_lora,
    qwen_should_save_before_sample,
    validate_qwen_train_config,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("qwen control smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("qwen control smoke: short write to ") + path)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("qwen control smoke FAILED: ") + msg)


comptime _QWEN_ARCH = String(
    '"model_type":"qwenimage",'
    + '"checkpoint":"/tmp/qwen-transformer",'
    + '"validation_prompts_file":"/tmp/qwen_samples.json",'
    + '"inner_dim":3072,'
    + '"in_channels":64,'
    + '"joint_attention_dim":3584,'
    + '"out_channels":16,'
    + '"num_double":60,'
    + '"num_single":0,'
    + '"num_heads":24,'
    + '"head_dim":128,'
    + '"mlp_hidden":12288,'
    + '"timestep_dim":256,'
    + '"rope_theta":10000,'
    + '"learning_rate":0.0002,'
    + '"lora_rank":8,'
    + '"lora_alpha":12,'
    + '"timestep_shift":3.0,'
    + '"max_grad_norm":1.0,'
    + '"max_steps":400,'
    + '"save_every":200,'
    + '"sample_every":100'
)


def _write_json(path: String, tail: String) raises:
    _write_file(path, String("{") + _QWEN_ARCH + tail + String("}"))


def _expect_raises(label: String, path: String, tail: String, mode: Int) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        if mode == 0:
            validate_qwen_train_config(cfg)
        elif mode == 1:
            var cadence = qwen_sample_cadence_from_train_config(path, cfg)
            _ = cadence.sample_after
        else:
            var offload = qwen_offload_config_from_train_config(cfg)
            _ = offload.slot_count
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("qwen control smoke: expected raise for ") + label)


def _gate_good_config() raises:
    print("--- config drives Qwen control fields ---")
    var path = String("/tmp/qwen_train_control_good.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"enable_async_offloading":true,'
    tail += '"enable_activation_offloading":true,'
    tail += '"layer_offload_fraction":0.0,'
    tail += '"sample_after":100,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_skip_first":20,'
    tail += '"sample_at_start":false,'
    tail += '"save_before_sample":true,'
    tail += '"sample_definition_file_name":"/tmp/qwen_samples.json"'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    validate_qwen_train_config(cfg)
    _check(qwen_patchified_out_channels(cfg) == 64, "patchified out_channels")
    _check(cfg.lora_rank == 8, "lora_rank")
    _check(cfg.lora_alpha == Float32(12.0), "lora_alpha")
    _check(cfg.lr == Float32(0.0002), "learning_rate")

    var cadence = qwen_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after == 100, "cadence sample_after")
    _check(cadence.sample_after_unit == SAMPLE_UNIT_STEP, "cadence unit STEP")
    _check(cadence.sample_skip_first == 20, "cadence skip_first")
    _check(not should_sample_completed_step(cadence, 0), "no baseline sample")
    _check(not should_sample_completed_step(cadence, 20), "skip-first honored")
    _check(should_sample_completed_step(cadence, 100), "step cadence due")
    _check(next_sample_completed_step(cadence, 100, 400) == 200, "next sample")
    _check(qwen_should_save_before_sample(cadence, 100, False), "save before due sample")
    _check(not qwen_should_save_before_sample(cadence, 100, True), "already saved this step")
    _check(not qwen_should_save_before_sample(cadence, 99, False), "do not save before non-sample")
    _check(
        qwen_state_path_for_lora(String("/tmp/qwen/lora_step100.safetensors"))
        == String("/tmp/qwen/lora_step100.safetensors.state.safetensors"),
        "state sidecar path",
    )

    var offload = qwen_offload_config_from_train_config(cfg)
    _check(offload.slot_count == 1, "offload slot_count")
    _check(offload.lookahead == 1, "offload lookahead")
    print("  good config PASS")


def _gate_validation_prompt_fallback() raises:
    print("--- validation_prompts_file supplies sample definition fallback ---")
    var path = String("/tmp/qwen_train_control_prompt_fallback.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":100,'
    tail += '"sample_after_unit":"STEP"'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    var cadence = qwen_sample_cadence_from_train_config(path, cfg)
    _check(
        cadence.sample_definition_file_name == String("/tmp/qwen_samples.json"),
        "validation_prompts_file fallback",
    )
    print("  validation prompt fallback PASS")


def _gate_save_before_sample_disabled() raises:
    print("--- save_before_sample false suppresses pre-sample save ---")
    var path = String("/tmp/qwen_train_control_no_presave.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":100,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"save_before_sample":false'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    var cadence = qwen_sample_cadence_from_train_config(path, cfg)
    _check(should_sample_completed_step(cadence, 100), "sample still due")
    _check(
        not qwen_should_save_before_sample(cadence, 100, False),
        "save_before_sample false",
    )
    print("  save_before_sample false PASS")


def _gate_no_sample_when_nonpositive() raises:
    print("--- non-positive sample_after disables Qwen sampling ---")
    var path = String("/tmp/qwen_train_control_nosample.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":0,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"save_before_sample":true'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    var cadence = qwen_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after_unit == SAMPLE_UNIT_NEVER, "non-positive maps to NEVER")
    _check(not should_sample_completed_step(cadence, 0), "no baseline sample")
    _check(not should_sample_completed_step(cadence, 100), "no cadence sample")
    _check(next_sample_completed_step(cadence, 0, 400) == -1, "no next sample")
    _check(not qwen_should_save_before_sample(cadence, 100, False), "no presave when disabled")
    print("  non-positive sample_after PASS")


def _gate_fail_loud_controls() raises:
    print("--- unsupported Qwen controls fail before model math ---")
    _expect_raises(
        String("EPOCH cadence"),
        String("/tmp/qwen_train_control_epoch.json"),
        String(',"sample_after":1,"sample_after_unit":"EPOCH"'),
        1,
    )
    _expect_raises(
        String("MINUTE cadence"),
        String("/tmp/qwen_train_control_minute.json"),
        String(',"sample_after":1,"sample_after_unit":"MINUTE"'),
        1,
    )
    _expect_raises(
        String("HOUR cadence"),
        String("/tmp/qwen_train_control_hour.json"),
        String(',"sample_after":1,"sample_after_unit":"HOUR"'),
        1,
    )
    _expect_raises(
        String("CPU_OFFLOADED checkpointing"),
        String("/tmp/qwen_train_control_cpu_offload.json"),
        String(',"gradient_checkpointing":"CPU_OFFLOADED","layer_offload_fraction":0.5'),
        2,
    )
    _expect_raises(
        String("unpatchified out_channels mismatch"),
        String("/tmp/qwen_train_control_bad_out.json"),
        String(',"out_channels":15'),
        0,
    )
    print("  fail-loud controls PASS")


def main() raises:
    print("=== Qwen train control wiring smoke ===")
    _gate_good_config()
    _gate_validation_prompt_fallback()
    _gate_save_before_sample_disabled()
    _gate_no_sample_when_nonpositive()
    _gate_fail_loud_controls()
    print("qwen_train_control_wiring_smoke PASS")
