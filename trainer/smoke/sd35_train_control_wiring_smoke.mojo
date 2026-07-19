# sd35_train_control_wiring_smoke.mojo -- config/control gate for SD3.5 training.
#
# Scope: TrainConfig + sample cadence + path/output reachability only. This does
# not construct DeviceContext, open SD3.5 weights, or run train math.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/sd35_train_control_wiring_smoke.mojo

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
from serenity_trainer.trainer.train_sd35_real import (
    sd35_cache_dir_from_train_config,
    sd35_checkpoint_from_train_config,
    sd35_output_lora_path_from_train_config,
    sd35_sample_cadence_from_train_config,
    sd35_sampling_enabled,
    sd35_should_save_before_sample,
    sd35_should_save_checkpoint,
    validate_sd35_train_config,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("sd35 control smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("sd35 control smoke: short write to ") + path)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("sd35 control smoke FAILED: ") + msg)


comptime _SD35_ARCH = String(
    '"model_type":"STABLE_DIFFUSION_35",'
    + '"base_model_name":"/tmp/sd35-large.safetensors",'
    + '"validation_prompts_file":"/tmp/sd35_samples.json",'
    + '"inner_dim":2432,'
    + '"in_channels":64,'
    + '"joint_attention_dim":4096,'
    + '"out_channels":64,'
    + '"num_double":38,'
    + '"num_single":0,'
    + '"num_heads":38,'
    + '"head_dim":64,'
    + '"mlp_hidden":9728,'
    + '"timestep_dim":256,'
    + '"learning_rate":0.0001,'
    + '"lora_rank":16,'
    + '"lora_alpha":16.0,'
    + '"timestep_shift":1.0,'
    + '"max_grad_norm":1.0,'
    + '"max_steps":2000,'
    + '"save_every":500,'
    + '"sample_every":500'
)


def _write_json(path: String, tail: String) raises:
    _write_file(path, String("{") + _SD35_ARCH + tail + String("}"))


def _expect_validate_raises(label: String, path: String, tail: String) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        validate_sd35_train_config(cfg)
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("sd35 control smoke: expected raise for ") + label)


def _expect_plain_sd3_raises() raises:
    var path = String("/tmp/sd35_bad_plain_sd3_model_type.json")
    var content = String("{")
    content += '"model_type":"STABLE_DIFFUSION_3",'
    content += '"base_model_name":"/tmp/sd3-large.safetensors",'
    content += '"inner_dim":2432,'
    content += '"in_channels":64,'
    content += '"joint_attention_dim":4096,'
    content += '"out_channels":64,'
    content += '"num_double":38,'
    content += '"num_single":0,'
    content += '"num_heads":38,'
    content += '"head_dim":64,'
    content += '"mlp_hidden":9728,'
    content += '"timestep_dim":256,'
    content += '"learning_rate":0.0001,'
    content += '"lora_rank":16,'
    content += '"lora_alpha":16.0,'
    content += '"timestep_shift":1.0,'
    content += '"max_grad_norm":1.0,'
    content += '"gradient_checkpointing":"ON"'
    content += String("}")
    _write_file(path, content)
    var raised = False
    try:
        var cfg = read_model_config(path)
        validate_sd35_train_config(cfg)
    except e:
        raised = True
        print("  raised as expected ( plain SD3 model type ): ", String(e))
    if not raised:
        raise Error("sd35 control smoke: expected raise for plain SD3 model type")


def _expect_cadence_raises(label: String, path: String, tail: String) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        var cadence = sd35_sample_cadence_from_train_config(path, cfg)
        _ = cadence.sample_after
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("sd35 control smoke: expected cadence raise for ") + label)


def _gate_good_config() raises:
    print("--- config drives SD3.5 control fields ---")
    var path = String("/tmp/sd35_train_control_good.json")
    var tail = String(",")
    tail += '"training_method":"LORA",'
    tail += '"cache_dir":"/tmp/sd35-cache",'
    tail += '"output_model_destination":"/tmp/sd35/lora.safetensors",'
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":500,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_skip_first":100,'
    tail += '"sample_at_start":false,'
    tail += '"save_before_sample":true,'
    tail += '"sample_definition_file_name":"/tmp/sd35_samples_control.json"'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    validate_sd35_train_config(cfg)
    _check(sd35_checkpoint_from_train_config(cfg) == String("/tmp/sd35-large.safetensors"), "base_model_name checkpoint fallback")
    _check(sd35_cache_dir_from_train_config(cfg) == String("/tmp/sd35-cache"), "cache path")
    _check(
        sd35_output_lora_path_from_train_config(cfg, 500)
        == String("/tmp/sd35/lora.safetensors"),
        "output destination",
    )
    var cadence = sd35_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after == 500, "cadence sample_after")
    _check(cadence.sample_after_unit == SAMPLE_UNIT_STEP, "cadence unit")
    _check(cadence.sample_skip_first == 100, "cadence skip")
    _check(cadence.sample_definition_file_name == String("/tmp/sd35_samples_control.json"), "sample file")
    _check(sd35_sampling_enabled(cadence), "sampling enabled")
    _check(not should_sample_completed_step(cadence, 100), "skip first suppresses step 100")
    _check(should_sample_completed_step(cadence, 500), "sample due at step 500")
    _check(next_sample_completed_step(cadence, 0, 2000) == 500, "next sample")
    _check(sd35_should_save_checkpoint(cfg, 500), "save due at 500")
    _check(sd35_should_save_before_sample(cadence, 500, False), "save before sample")
    _check(not sd35_should_save_before_sample(cadence, 500, True), "no duplicate save before sample")
    print("  good SD3.5 config PASS")


def _gate_disabled_sampling() raises:
    print("--- non-positive sample_after disables SD3.5 sampling ---")
    var path = String("/tmp/sd35_train_control_disabled_sample.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":0,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_definition_file_name":"/tmp/sd35_samples_control.json"'
    _write_json(path, tail)
    var cfg = read_model_config(path)
    validate_sd35_train_config(cfg)
    var cadence = sd35_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after_unit == SAMPLE_UNIT_NEVER, "non-positive maps to NEVER")
    _check(not sd35_sampling_enabled(cadence), "sampling disabled")
    print("  disabled sampling PASS")


def _gate_fail_loud() raises:
    print("--- SD3.5 unsupported controls fail loud ---")
    _expect_plain_sd3_raises()
    _expect_validate_raises(
        String("sharded checkpoint dir"),
        String("/tmp/sd35_bad_checkpoint_dir.json"),
        String(',"checkpoint":"/tmp/sd35/transformer","gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("rank mismatch"),
        String("/tmp/sd35_bad_rank.json"),
        String(',"lora_rank":8,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("alpha mismatch"),
        String("/tmp/sd35_bad_alpha.json"),
        String(',"lora_alpha":8.0,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("learning rate mismatch"),
        String("/tmp/sd35_bad_lr.json"),
        String(',"learning_rate":0.0002,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("timestep shift mismatch"),
        String("/tmp/sd35_bad_shift.json"),
        String(',"timestep_shift":3.0,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("depth mismatch"),
        String("/tmp/sd35_bad_depth.json"),
        String(',"num_double":24,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("channel mismatch"),
        String("/tmp/sd35_bad_channel.json"),
        String(',"in_channels":16,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("full finetune not wired"),
        String("/tmp/sd35_bad_full_ft.json"),
        String(',"training_method":"FINE_TUNE","gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("unsupported optimizer"),
        String("/tmp/sd35_bad_adafactor.json"),
        String(',"optimizer":{"optimizer":"ADAFACTOR"},"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("CPU offload not wired"),
        String("/tmp/sd35_bad_cpu_offload.json"),
        String(',"gradient_checkpointing":"CPU_OFFLOADED"'),
    )
    _expect_cadence_raises(
        String("epoch sample cadence"),
        String("/tmp/sd35_bad_epoch_cadence.json"),
        String(',"gradient_checkpointing":"ON","sample_after":1,"sample_after_unit":"EPOCH"'),
    )


def main() raises:
    print("==== SD3.5 train-control wiring smoke ====")
    _gate_good_config()
    _gate_disabled_sampling()
    _gate_fail_loud()
    print("sd35_train_control_wiring_smoke PASS")
