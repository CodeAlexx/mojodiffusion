# zimage_train_control_wiring_smoke.mojo -- config/control gate for Z-Image training.
#
# Scope: TrainConfig + sample cadence + path/output reachability only. This does
# not construct DeviceContext, load the 24 GB-class transformer, or run train
# math.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/zimage_train_control_wiring_smoke.mojo

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
from serenitymojo.training.train_config import (
    GRADIENT_CHECKPOINTING_CPU_OFFLOADED,
    TRAINING_METHOD_FINE_TUNE,
    TRAIN_OPTIMIZER_ADAFACTOR,
)
from serenity_trainer.trainer.train_zimage_real import (
    validate_zimage_train_config,
    zimage_cache_dir_from_train_config,
    zimage_output_lora_path_from_train_config,
    zimage_patchified_out_channels,
    zimage_sample_cadence_from_train_config,
    zimage_sample_output_path,
    zimage_sample_request_dir,
    zimage_sample_request_path,
    zimage_sampling_enabled,
    zimage_should_save_before_sample,
    zimage_should_save_checkpoint,
    zimage_transformer_dir_from_train_config,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("zimage control smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("zimage control smoke: short write to ") + path)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("zimage control smoke FAILED: ") + msg)


comptime _ZIMAGE_ARCH = String(
    '"model_type":"zimage",'
    + '"checkpoint":"/tmp/zimage-transformer",'
    + '"validation_prompts_file":"/tmp/zimage_samples.json",'
    + '"inner_dim":3840,'
    + '"in_channels":16,'
    + '"joint_attention_dim":2560,'
    + '"out_channels":16,'
    + '"num_double":0,'
    + '"num_single":30,'
    + '"num_heads":30,'
    + '"head_dim":128,'
    + '"mlp_hidden":10240,'
    + '"rope_theta":256,'
    + '"learning_rate":0.0003,'
    + '"lora_rank":16,'
    + '"lora_alpha":1.0,'
    + '"timestep_shift":1.0,'
    + '"max_grad_norm":1.0,'
    + '"max_steps":2500,'
    + '"save_every":500,'
    + '"sample_every":500'
)


def _write_json(path: String, tail: String) raises:
    _write_file(path, String("{") + _ZIMAGE_ARCH + tail + String("}"))


def _expect_validate_raises(label: String, path: String, tail: String) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        validate_zimage_train_config(cfg)
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("zimage control smoke: expected raise for ") + label)


def _expect_cadence_raises(label: String, path: String, tail: String) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        var cadence = zimage_sample_cadence_from_train_config(path, cfg)
        _ = cadence.sample_after
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("zimage control smoke: expected cadence raise for ") + label)


def _gate_good_config() raises:
    print("--- config drives Z-Image control fields ---")
    var path = String("/tmp/zimage_train_control_good.json")
    var tail = String(",")
    tail += '"training_method":"LORA",'
    tail += '"cache_dir":"/tmp/zimage-cache",'
    tail += '"output_model_destination":"/tmp/zimage/lora.safetensors",'
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":500,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_skip_first":100,'
    tail += '"sample_at_start":false,'
    tail += '"save_before_sample":true,'
    tail += '"sample_definition_file_name":"/tmp/zimage_samples_control.json"'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    validate_zimage_train_config(cfg)
    _check(zimage_transformer_dir_from_train_config(cfg) == String("/tmp/zimage-transformer"), "checkpoint path")
    _check(zimage_cache_dir_from_train_config(cfg) == String("/tmp/zimage-cache"), "cache path")
    _check(
        zimage_output_lora_path_from_train_config(cfg, 500)
        == String("/tmp/zimage/lora.safetensors"),
        "output destination",
    )
    _check(zimage_patchified_out_channels(cfg) == 64, "patchified out channels")
    var cadence = zimage_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after == 500, "cadence sample_after")
    _check(cadence.sample_after_unit == SAMPLE_UNIT_STEP, "cadence unit")
    _check(cadence.sample_skip_first == 100, "cadence skip")
    _check(cadence.sample_definition_file_name == String("/tmp/zimage_samples_control.json"), "sample file")
    _check(zimage_sampling_enabled(cadence), "sampling enabled")
    _check(not should_sample_completed_step(cadence, 100), "skip first suppresses step 100")
    _check(should_sample_completed_step(cadence, 500), "sample due at step 500")
    _check(next_sample_completed_step(cadence, 0, 2500) == 500, "next sample")
    _check(zimage_should_save_checkpoint(cfg, 500), "save due at 500")
    _check(zimage_should_save_before_sample(cadence, 500, False), "save before sample")
    _check(not zimage_should_save_before_sample(cadence, 500, True), "no duplicate save before sample")
    _check(
        zimage_sample_request_dir()
        == String("/home/alex/mojodiffusion/output/zimage/sample_requests"),
        "sample request dir",
    )
    _check(
        zimage_sample_request_path(500).endswith(String("/step500_request.json")),
        "sample request path",
    )
    _check(
        zimage_sample_output_path(500).endswith(String("/step500_sample.png")),
        "sample output path",
    )
    print("  good Z-Image config PASS")


def _gate_disabled_sampling() raises:
    print("--- non-positive sample_after disables Z-Image sampling ---")
    var path = String("/tmp/zimage_train_control_disabled_sample.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":0,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_definition_file_name":"/tmp/zimage_samples_control.json"'
    _write_json(path, tail)
    var cfg = read_model_config(path)
    validate_zimage_train_config(cfg)
    var cadence = zimage_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after_unit == SAMPLE_UNIT_NEVER, "non-positive maps to NEVER")
    _check(not zimage_sampling_enabled(cadence), "sampling disabled")
    print("  disabled sampling PASS")


def _gate_fail_loud() raises:
    print("--- Z-Image unsupported controls fail loud ---")
    _expect_validate_raises(
        String("rank mismatch"),
        String("/tmp/zimage_bad_rank.json"),
        String(',"lora_rank":8,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("full finetune not wired"),
        String("/tmp/zimage_bad_full_ft.json"),
        String(',"training_method":"FINE_TUNE","gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("unsupported optimizer"),
        String("/tmp/zimage_bad_adafactor.json"),
        String(',"optimizer":{"optimizer":"ADAFACTOR"},"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("CPU offload not wired"),
        String("/tmp/zimage_bad_cpu_offload.json"),
        String(',"gradient_checkpointing":"CPU_OFFLOADED"'),
    )
    _expect_cadence_raises(
        String("epoch sample cadence"),
        String("/tmp/zimage_bad_epoch_cadence.json"),
        String(',"gradient_checkpointing":"ON","sample_after":1,"sample_after_unit":"EPOCH"'),
    )


def main() raises:
    print("==== Z-Image train-control wiring smoke ====")
    _gate_good_config()
    _gate_disabled_sampling()
    _gate_fail_loud()
    print("zimage_train_control_wiring_smoke PASS")
