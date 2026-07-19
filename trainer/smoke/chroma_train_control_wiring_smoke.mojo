# chroma_train_control_wiring_smoke.mojo -- config/control gate for Chroma training.
#
# Scope: TrainConfig + sample cadence + path/output reachability only. This does
# not construct DeviceContext, open the 8.9B checkpoint, or run train math.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/chroma_train_control_wiring_smoke.mojo

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
from serenity_trainer.trainer.train_chroma_real import (
    chroma_cache_dir_from_train_config,
    chroma_checkpoint_from_train_config,
    chroma_output_lora_path_from_train_config,
    chroma_sample_cadence_from_train_config,
    chroma_sampling_enabled,
    chroma_should_save_before_sample,
    chroma_should_save_checkpoint,
    validate_chroma_train_config,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("chroma control smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("chroma control smoke: short write to ") + path)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("chroma control smoke FAILED: ") + msg)


comptime _CHROMA_ARCH = String(
    '"model_type":"chroma",'
    + '"checkpoint":"/tmp/chroma.safetensors",'
    + '"validation_prompts_file":"/tmp/chroma_samples.json",'
    + '"inner_dim":3072,'
    + '"in_channels":64,'
    + '"joint_attention_dim":4096,'
    + '"out_channels":64,'
    + '"num_double":19,'
    + '"num_single":38,'
    + '"num_heads":24,'
    + '"head_dim":128,'
    + '"mlp_hidden":12288,'
    + '"learning_rate":0.0001,'
    + '"lora_rank":16,'
    + '"lora_alpha":16.0,'
    + '"timestep_shift":1.15,'
    + '"max_grad_norm":1.0,'
    + '"max_steps":2000,'
    + '"save_every":500,'
    + '"sample_every":500'
)


def _write_json(path: String, tail: String) raises:
    _write_file(path, String("{") + _CHROMA_ARCH + tail + String("}"))


def _expect_validate_raises(label: String, path: String, tail: String) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        validate_chroma_train_config(cfg)
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("chroma control smoke: expected raise for ") + label)


def _expect_cadence_raises(label: String, path: String, tail: String) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        var cadence = chroma_sample_cadence_from_train_config(path, cfg)
        _ = cadence.sample_after
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("chroma control smoke: expected cadence raise for ") + label)


def _gate_good_config() raises:
    print("--- config drives Chroma control fields ---")
    var path = String("/tmp/chroma_train_control_good.json")
    var tail = String(",")
    tail += '"training_method":"LORA",'
    tail += '"cache_dir":"/tmp/chroma-cache",'
    tail += '"output_model_destination":"/tmp/chroma/lora.safetensors",'
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":500,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_skip_first":100,'
    tail += '"sample_at_start":false,'
    tail += '"save_before_sample":true,'
    tail += '"sample_definition_file_name":"/tmp/chroma_samples_control.json"'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    validate_chroma_train_config(cfg)
    _check(chroma_checkpoint_from_train_config(cfg) == String("/tmp/chroma.safetensors"), "checkpoint path")
    _check(chroma_cache_dir_from_train_config(cfg) == String("/tmp/chroma-cache"), "cache path")
    _check(
        chroma_output_lora_path_from_train_config(cfg, 500)
        == String("/tmp/chroma/lora.safetensors"),
        "output destination",
    )
    var cadence = chroma_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after == 500, "cadence sample_after")
    _check(cadence.sample_after_unit == SAMPLE_UNIT_STEP, "cadence unit")
    _check(cadence.sample_skip_first == 100, "cadence skip")
    _check(cadence.sample_definition_file_name == String("/tmp/chroma_samples_control.json"), "sample file")
    _check(chroma_sampling_enabled(cadence), "sampling enabled")
    _check(not should_sample_completed_step(cadence, 100), "skip first suppresses step 100")
    _check(should_sample_completed_step(cadence, 500), "sample due at step 500")
    _check(next_sample_completed_step(cadence, 0, 2000) == 500, "next sample")
    _check(chroma_should_save_checkpoint(cfg, 500), "save due at 500")
    _check(chroma_should_save_before_sample(cadence, 500, False), "save before sample")
    _check(not chroma_should_save_before_sample(cadence, 500, True), "no duplicate save before sample")
    print("  good Chroma config PASS")


def _gate_disabled_sampling() raises:
    print("--- non-positive sample_after disables Chroma sampling ---")
    var path = String("/tmp/chroma_train_control_disabled_sample.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":0,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_definition_file_name":"/tmp/chroma_samples_control.json"'
    _write_json(path, tail)
    var cfg = read_model_config(path)
    validate_chroma_train_config(cfg)
    var cadence = chroma_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after_unit == SAMPLE_UNIT_NEVER, "non-positive maps to NEVER")
    _check(not chroma_sampling_enabled(cadence), "sampling disabled")
    print("  disabled sampling PASS")


def _gate_fail_loud() raises:
    print("--- Chroma unsupported controls fail loud ---")
    _expect_validate_raises(
        String("sharded checkpoint dir"),
        String("/tmp/chroma_bad_checkpoint_dir.json"),
        String(',"checkpoint":"/tmp/chroma/transformer","gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("rank mismatch"),
        String("/tmp/chroma_bad_rank.json"),
        String(',"lora_rank":8,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("full finetune not wired"),
        String("/tmp/chroma_bad_full_ft.json"),
        String(',"training_method":"FINE_TUNE","gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("unsupported optimizer"),
        String("/tmp/chroma_bad_adafactor.json"),
        String(',"optimizer":{"optimizer":"ADAFACTOR"},"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("CPU offload not wired"),
        String("/tmp/chroma_bad_cpu_offload.json"),
        String(',"gradient_checkpointing":"CPU_OFFLOADED"'),
    )
    _expect_cadence_raises(
        String("epoch sample cadence"),
        String("/tmp/chroma_bad_epoch_cadence.json"),
        String(',"gradient_checkpointing":"ON","sample_after":1,"sample_after_unit":"EPOCH"'),
    )


def main() raises:
    print("==== Chroma train-control wiring smoke ====")
    _gate_good_config()
    _gate_disabled_sampling()
    _gate_fail_loud()
    print("chroma_train_control_wiring_smoke PASS")
