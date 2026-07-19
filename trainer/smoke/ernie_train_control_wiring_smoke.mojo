# ernie_train_control_wiring_smoke.mojo -- config/control gate for ERNIE training.
#
# Scope: TrainConfig + sample cadence + checkpoint/offload policy reachability.
# This deliberately does not load ERNIE weights, construct DeviceContext, or run
# transformer math.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/ernie_train_control_wiring_smoke.mojo

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
from serenity_trainer.trainer.train_ernie_real import (
    ernie_sample_cadence_from_train_config,
    ernie_sampling_enabled,
    ernie_should_save_before_sample,
    ernie_should_save_checkpoint,
    validate_ernie_train_config,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("ernie control smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("ernie control smoke: short write to ") + path)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("ernie control smoke FAILED: ") + msg)


def _near(a: Float32, b: Float32) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    return d <= Float32(1.0e-7)


comptime _ERNIE_ARCH = String(
    '"model_type":"ernie_image",'
    + '"checkpoint":"/tmp/ernie-transformer",'
    + '"vae":"/tmp/ernie-vae.safetensors",'
    + '"validation_prompts_file":"/tmp/ernie_samples.json",'
    + '"inner_dim":4096,'
    + '"in_channels":128,'
    + '"joint_attention_dim":3072,'
    + '"out_channels":128,'
    + '"num_double":0,'
    + '"num_single":36,'
    + '"num_heads":32,'
    + '"head_dim":128,'
    + '"mlp_hidden":12288,'
    + '"timestep_dim":4096,'
    + '"rope_theta":256,'
    + '"learning_rate":0.00031,'
    + '"lora_rank":6,'
    + '"lora_alpha":2.5,'
    + '"timestep_shift":1.0,'
    + '"max_grad_norm":0.75,'
    + '"max_steps":240,'
    + '"save_every":75,'
    + '"sample_every":80'
)


def _write_json(path: String, tail: String) raises:
    _write_file(path, String("{") + _ERNIE_ARCH + tail + String("}"))


def _expect_raises(label: String, path: String, tail: String, mode: Int) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        if mode == 0:
            validate_ernie_train_config(cfg)
        else:
            var cadence = ernie_sample_cadence_from_train_config(path, cfg)
            _ = cadence.sample_after
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("ernie control smoke: expected raise for ") + label)


def _gate_good_config() raises:
    print("--- config drives ERNIE control fields ---")
    var path = String("/tmp/ernie_train_control_good.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"enable_async_offloading":true,'
    tail += '"enable_activation_offloading":true,'
    tail += '"layer_offload_fraction":0.0,'
    tail += '"sample_after":80,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_skip_first":20,'
    tail += '"sample_at_start":false,'
    tail += '"save_before_sample":true,'
    tail += '"sample_definition_file_name":"/tmp/ernie_samples_control.json"'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    validate_ernie_train_config(cfg)
    _check(cfg.lora_rank == 6, "lora_rank")
    _check(_near(cfg.lora_alpha, Float32(2.5)), "lora_alpha")
    _check(_near(cfg.lr, Float32(0.00031)), "learning_rate")
    _check(cfg.save_every == 75, "save_every")
    _check(not ernie_should_save_checkpoint(cfg, 0), "no step-0 checkpoint")
    _check(not ernie_should_save_checkpoint(cfg, 74), "no checkpoint before cadence")
    _check(ernie_should_save_checkpoint(cfg, 75), "save cadence due")
    _check(ernie_should_save_checkpoint(cfg, 150), "save cadence repeats")

    var cadence = ernie_sample_cadence_from_train_config(path, cfg)
    _check(ernie_sampling_enabled(cadence), "sampling enabled")
    _check(cadence.sample_after == 80, "cadence sample_after")
    _check(cadence.sample_after_unit == SAMPLE_UNIT_STEP, "cadence unit STEP")
    _check(cadence.sample_skip_first == 20, "cadence skip_first")
    _check(
        cadence.sample_definition_file_name == String("/tmp/ernie_samples_control.json"),
        "explicit sample definition",
    )
    _check(not should_sample_completed_step(cadence, 0), "no baseline sample")
    _check(not should_sample_completed_step(cadence, 20), "skip-first honored")
    _check(should_sample_completed_step(cadence, 80), "step cadence due")
    _check(next_sample_completed_step(cadence, 80, 240) == 160, "next sample")
    _check(ernie_should_save_before_sample(cadence, 80, False), "save before due sample")
    _check(not ernie_should_save_before_sample(cadence, 80, True), "already saved this step")
    _check(not ernie_should_save_before_sample(cadence, 79, False), "do not save before non-sample")
    print("  good config PASS")


def _gate_validation_prompt_fallback() raises:
    print("--- validation_prompts_file supplies sample definition fallback ---")
    var path = String("/tmp/ernie_train_control_prompt_fallback.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":80,'
    tail += '"sample_after_unit":"STEP"'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    var cadence = ernie_sample_cadence_from_train_config(path, cfg)
    _check(
        cadence.sample_definition_file_name == String("/tmp/ernie_samples.json"),
        "validation_prompts_file fallback",
    )
    print("  validation prompt fallback PASS")


def _gate_save_before_sample_disabled() raises:
    print("--- save_before_sample false suppresses pre-sample save ---")
    var path = String("/tmp/ernie_train_control_no_presave.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":80,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"save_before_sample":false'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    var cadence = ernie_sample_cadence_from_train_config(path, cfg)
    _check(should_sample_completed_step(cadence, 80), "sample still due")
    _check(
        not ernie_should_save_before_sample(cadence, 80, False),
        "save_before_sample false",
    )
    print("  save_before_sample false PASS")


def _gate_no_sample_when_nonpositive() raises:
    print("--- non-positive sample_after disables ERNIE sampling ---")
    var path = String("/tmp/ernie_train_control_nosample.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":0,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"save_before_sample":true'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    var cadence = ernie_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after_unit == SAMPLE_UNIT_NEVER, "non-positive maps to NEVER")
    _check(not ernie_sampling_enabled(cadence), "sampling disabled")
    _check(not should_sample_completed_step(cadence, 0), "no baseline sample")
    _check(not should_sample_completed_step(cadence, 80), "no cadence sample")
    _check(next_sample_completed_step(cadence, 0, 240) == -1, "no next sample")
    _check(not ernie_should_save_before_sample(cadence, 80, False), "no presave when disabled")
    print("  non-positive sample_after PASS")


def _gate_fail_loud_controls() raises:
    print("--- unsupported ERNIE controls fail before model math ---")
    _expect_raises(
        String("EPOCH cadence"),
        String("/tmp/ernie_train_control_epoch.json"),
        String(',"sample_after":1,"sample_after_unit":"EPOCH"'),
        1,
    )
    _expect_raises(
        String("MINUTE cadence"),
        String("/tmp/ernie_train_control_minute.json"),
        String(',"sample_after":1,"sample_after_unit":"MINUTE"'),
        1,
    )
    _expect_raises(
        String("HOUR cadence"),
        String("/tmp/ernie_train_control_hour.json"),
        String(',"sample_after":1,"sample_after_unit":"HOUR"'),
        1,
    )
    _expect_raises(
        String("CPU_OFFLOADED checkpointing"),
        String("/tmp/ernie_train_control_cpu_offload.json"),
        String(',"gradient_checkpointing":"CPU_OFFLOADED","enable_activation_offloading":true,"layer_offload_fraction":0.5'),
        0,
    )
    print("  fail-loud controls PASS")


def main() raises:
    print("=== ERNIE train control wiring smoke ===")
    _gate_good_config()
    _gate_validation_prompt_fallback()
    _gate_save_before_sample_disabled()
    _gate_no_sample_when_nonpositive()
    _gate_fail_loud_controls()
    print("ernie_train_control_wiring_smoke PASS")
