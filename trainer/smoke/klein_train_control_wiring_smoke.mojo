# klein_train_control_wiring_smoke.mojo -- config/control gate for Klein training.
#
# Scope: TrainConfig + sample cadence + path/output reachability only. This does
# not construct DeviceContext, open the 9B checkpoint, or run train math.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/klein_train_control_wiring_smoke.mojo

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
from serenity_trainer.trainer.train_klein_real import (
    klein_cache_dir_from_train_config,
    klein_output_lora_path_from_train_config,
    klein_sample_cadence_from_train_config,
    klein_sampling_enabled,
    klein_should_save_before_sample,
    klein_should_save_checkpoint,
    validate_klein_train_config,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("klein control smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("klein control smoke: short write to ") + path)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("klein control smoke FAILED: ") + msg)


comptime _KLEIN9B_ARCH = String(
    '"model_type":"klein",'
    + '"checkpoint":"/tmp/klein-9b.safetensors",'
    + '"validation_prompts_file":"/tmp/klein_samples.json",'
    + '"inner_dim":4096,'
    + '"in_channels":128,'
    + '"joint_attention_dim":12288,'
    + '"out_channels":128,'
    + '"num_double":8,'
    + '"num_single":24,'
    + '"num_heads":32,'
    + '"head_dim":128,'
    + '"mlp_hidden":12288,'
    + '"timestep_dim":256,'
    + '"rope_theta":2000,'
    + '"learning_rate":0.0004,'
    + '"lora_rank":16,'
    + '"lora_alpha":16.0,'
    + '"timestep_shift":1.0,'
    + '"max_grad_norm":1.0,'
    + '"max_steps":2000,'
    + '"save_every":500,'
    + '"sample_every":500'
)


def _write_json(path: String, tail: String) raises:
    _write_file(path, String("{") + _KLEIN9B_ARCH + tail + String("}"))


def _expect_validate_raises(label: String, path: String, tail: String) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        validate_klein_train_config(cfg)
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("klein control smoke: expected raise for ") + label)


def _expect_cadence_raises(label: String, path: String, tail: String) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        var cadence = klein_sample_cadence_from_train_config(path, cfg)
        _ = cadence.sample_after
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("klein control smoke: expected cadence raise for ") + label)


def _gate_good_config() raises:
    print("--- config drives Klein control fields ---")
    var path = String("/tmp/klein_train_control_good.json")
    var tail = String(",")
    tail += '"training_method":"LORA",'
    tail += '"cache_dir":"/tmp/klein-cache",'
    tail += '"output_model_destination":"/tmp/klein/lora.safetensors",'
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":500,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_skip_first":100,'
    tail += '"sample_at_start":false,'
    tail += '"save_before_sample":true,'
    tail += '"sample_definition_file_name":"/tmp/klein_samples_control.json"'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    validate_klein_train_config(cfg)
    _check(klein_cache_dir_from_train_config(cfg) == String("/tmp/klein-cache"), "cache path")
    _check(
        klein_output_lora_path_from_train_config(cfg, 500)
        == String("/tmp/klein/lora_step500.safetensors"),
        "step output destination",
    )
    _check(
        klein_output_lora_path_from_train_config(cfg, 2000)
        == String("/tmp/klein/lora.safetensors"),
        "final output destination",
    )
    var cadence = klein_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after == 500, "cadence sample_after")
    _check(cadence.sample_after_unit == SAMPLE_UNIT_STEP, "cadence unit")
    _check(cadence.sample_skip_first == 100, "cadence skip")
    _check(cadence.sample_definition_file_name == String("/tmp/klein_samples_control.json"), "sample file")
    _check(klein_sampling_enabled(cadence), "sampling enabled")
    _check(not should_sample_completed_step(cadence, 100), "skip first suppresses step 100")
    _check(should_sample_completed_step(cadence, 500), "sample due at step 500")
    _check(next_sample_completed_step(cadence, 0, 2000) == 500, "next sample")
    _check(klein_should_save_checkpoint(cfg, 500), "save due at 500")
    _check(klein_should_save_before_sample(cadence, 500, False), "save before sample")
    _check(not klein_should_save_before_sample(cadence, 500, True), "no duplicate save before sample")
    print("  good Klein config PASS")


def _gate_disabled_sampling() raises:
    print("--- non-positive sample_after disables Klein sampling ---")
    var path = String("/tmp/klein_train_control_disabled_sample.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":0,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_definition_file_name":"/tmp/klein_samples_control.json"'
    _write_json(path, tail)
    var cfg = read_model_config(path)
    validate_klein_train_config(cfg)
    var cadence = klein_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after_unit == SAMPLE_UNIT_NEVER, "non-positive maps to NEVER")
    _check(not klein_sampling_enabled(cadence), "sampling disabled")
    print("  disabled sampling PASS")


def _gate_cpu_offloaded_config() raises:
    print("--- CPU_OFFLOADED policy reaches Klein activation-tape path ---")
    var path = String("/tmp/klein_train_control_cpu_offloaded.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"CPU_OFFLOADED",'
    tail += '"enable_activation_offloading":true,'
    tail += '"layer_offload_fraction":0.7,'
    tail += '"sample_after":0,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_definition_file_name":"/tmp/klein_samples_control.json"'
    _write_json(path, tail)
    var cfg = read_model_config(path)
    validate_klein_train_config(cfg)
    _check(cfg.gradient_checkpointing_offload(), "CPU_OFFLOADED parsed")
    _check(cfg.activation_offload_enabled(), "activation offload enabled")
    _check(cfg.layer_offload_enabled(), "layer offload enabled")
    print("  CPU_OFFLOADED Klein config PASS")


def _gate_fail_loud() raises:
    print("--- Klein unsupported controls fail loud ---")
    _expect_validate_raises(
        String("sharded checkpoint dir"),
        String("/tmp/klein_bad_checkpoint_dir.json"),
        String(',"checkpoint":"/tmp/klein/transformer","gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("wrong shape/4B-style heads"),
        String("/tmp/klein_bad_heads.json"),
        String(',"num_heads":24,"inner_dim":3072,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("full finetune not wired"),
        String("/tmp/klein_bad_full_ft.json"),
        String(',"training_method":"FINE_TUNE","gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("unsupported optimizer"),
        String("/tmp/klein_bad_adafactor.json"),
        String(',"optimizer":{"optimizer":"ADAFACTOR"},"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("activation checkpointing off"),
        String("/tmp/klein_bad_checkpointing_off.json"),
        String(',"gradient_checkpointing":"OFF"'),
    )
    _expect_cadence_raises(
        String("epoch sample cadence"),
        String("/tmp/klein_bad_epoch_cadence.json"),
        String(',"gradient_checkpointing":"ON","sample_after":1,"sample_after_unit":"EPOCH"'),
    )


def main() raises:
    print("==== Klein train-control wiring smoke ====")
    _gate_good_config()
    _gate_disabled_sampling()
    _gate_cpu_offloaded_config()
    _gate_fail_loud()
    print("klein_train_control_wiring_smoke PASS")
