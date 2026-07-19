# flux_train_control_wiring_smoke.mojo -- config/control gate for Flux.1-dev training.
#
# Scope: TrainConfig + sample cadence + path/output reachability only. This does
# not construct DeviceContext, open Flux.1-dev weights, or run train math.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/flux_train_control_wiring_smoke.mojo

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
from serenity_trainer.trainer.train_flux_real import (
    flux_cache_dir_from_train_config,
    flux_checkpoint_from_train_config,
    flux_output_lora_path_from_train_config,
    flux_sample_cadence_from_train_config,
    flux_sampling_enabled,
    flux_should_save_before_sample,
    flux_should_save_checkpoint,
    validate_flux_train_config,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("flux control smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("flux control smoke: short write to ") + path)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("flux control smoke FAILED: ") + msg)


comptime _FLUX_ARCH = String(
    '"model_type":"flux",'
    + '"checkpoint":"/tmp/flux.safetensors",'
    + '"validation_prompts_file":"/tmp/flux_samples.json",'
    + '"inner_dim":3072,'
    + '"in_channels":64,'
    + '"joint_attention_dim":4096,'
    + '"out_channels":64,'
    + '"num_double":19,'
    + '"num_single":38,'
    + '"num_heads":24,'
    + '"head_dim":128,'
    + '"mlp_hidden":12288,'
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
    _write_file(path, String("{") + _FLUX_ARCH + tail + String("}"))


def _expect_validate_raises(label: String, path: String, tail: String) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        validate_flux_train_config(cfg)
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("flux control smoke: expected raise for ") + label)


def _expect_cadence_raises(label: String, path: String, tail: String) raises:
    _write_json(path, tail)
    var raised = False
    try:
        var cfg = read_model_config(path)
        var cadence = flux_sample_cadence_from_train_config(path, cfg)
        _ = cadence.sample_after
    except e:
        raised = True
        print("  raised as expected (", label, "): ", String(e))
    if not raised:
        raise Error(String("flux control smoke: expected cadence raise for ") + label)


def _gate_good_config() raises:
    print("--- config drives Flux control fields ---")
    var path = String("/tmp/flux_train_control_good.json")
    var tail = String(",")
    tail += '"training_method":"LORA",'
    tail += '"cache_dir":"/tmp/flux-cache",'
    tail += '"output_model_destination":"/tmp/flux/lora.safetensors",'
    tail += '"gradient_checkpointing":"CPU_OFFLOADED",'
    tail += '"layer_offload_fraction":0.7,'
    tail += '"optimizer":{"optimizer":"ADAMW","beta1":0.91,"beta2":0.98,"eps":0.000001,"weight_decay":0.0},'
    tail += '"sample_after":500,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_skip_first":100,'
    tail += '"sample_at_start":false,'
    tail += '"save_before_sample":true,'
    tail += '"sample_definition_file_name":"/tmp/flux_samples_control.json"'
    _write_json(path, tail)

    var cfg = read_model_config(path)
    validate_flux_train_config(cfg)
    _check(cfg.beta1 > Float32(0.90) and cfg.beta1 < Float32(0.92), "optimizer beta1")
    _check(cfg.beta2 > Float32(0.97) and cfg.beta2 < Float32(0.99), "optimizer beta2")
    _check(cfg.eps > Float32(0.0) and cfg.eps < Float32(0.00001), "optimizer eps")
    _check(cfg.weight_decay == Float32(0.0), "optimizer weight_decay")
    _check(flux_checkpoint_from_train_config(cfg) == String("/tmp/flux.safetensors"), "checkpoint path")
    _check(flux_cache_dir_from_train_config(cfg) == String("/tmp/flux-cache"), "cache path")
    _check(
        flux_output_lora_path_from_train_config(cfg, 500)
        == String("/tmp/flux/lora.safetensors"),
        "output destination",
    )
    var cadence = flux_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after == 500, "cadence sample_after")
    _check(cadence.sample_after_unit == SAMPLE_UNIT_STEP, "cadence unit")
    _check(cadence.sample_skip_first == 100, "cadence skip")
    _check(cadence.sample_definition_file_name == String("/tmp/flux_samples_control.json"), "sample file")
    _check(flux_sampling_enabled(cadence), "sampling enabled")
    _check(not should_sample_completed_step(cadence, 100), "skip first suppresses step 100")
    _check(should_sample_completed_step(cadence, 500), "sample due at step 500")
    _check(next_sample_completed_step(cadence, 0, 2000) == 500, "next sample")
    _check(flux_should_save_checkpoint(cfg, 500), "save due at 500")
    _check(flux_should_save_before_sample(cadence, 500, False), "save before sample")
    _check(not flux_should_save_before_sample(cadence, 500, True), "no duplicate save before sample")
    print("  good Flux config PASS")


def _gate_disabled_sampling() raises:
    print("--- non-positive sample_after disables Flux sampling ---")
    var path = String("/tmp/flux_train_control_disabled_sample.json")
    var tail = String(",")
    tail += '"gradient_checkpointing":"ON",'
    tail += '"sample_after":0,'
    tail += '"sample_after_unit":"STEP",'
    tail += '"sample_definition_file_name":"/tmp/flux_samples_control.json"'
    _write_json(path, tail)
    var cfg = read_model_config(path)
    validate_flux_train_config(cfg)
    var cadence = flux_sample_cadence_from_train_config(path, cfg)
    _check(cadence.sample_after_unit == SAMPLE_UNIT_NEVER, "non-positive maps to NEVER")
    _check(not flux_sampling_enabled(cadence), "sampling disabled")
    print("  disabled sampling PASS")


def _gate_fail_loud() raises:
    print("--- Flux unsupported controls fail loud ---")
    _expect_validate_raises(
        String("rank mismatch"),
        String("/tmp/flux_bad_rank.json"),
        String(',"lora_rank":8,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("alpha mismatch"),
        String("/tmp/flux_bad_alpha.json"),
        String(',"lora_alpha":1.0,"gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("full finetune not wired"),
        String("/tmp/flux_bad_full_ft.json"),
        String(',"training_method":"FINE_TUNE","gradient_checkpointing":"ON"'),
    )
    _expect_validate_raises(
        String("unsupported optimizer"),
        String("/tmp/flux_bad_adafactor.json"),
        String(',"optimizer":{"optimizer":"ADAFACTOR"},"gradient_checkpointing":"ON"'),
    )
    _write_json(
        String("/tmp/flux_good_cpu_offload.json"),
        String(',"gradient_checkpointing":"CPU_OFFLOADED","layer_offload_fraction":0.7'),
    )
    var cpu_cfg = read_model_config(String("/tmp/flux_good_cpu_offload.json"))
    validate_flux_train_config(cpu_cfg)
    print("  CPU_OFFLOADED Flux policy accepted")
    _expect_cadence_raises(
        String("epoch sample cadence"),
        String("/tmp/flux_bad_epoch_cadence.json"),
        String(',"gradient_checkpointing":"ON","sample_after":1,"sample_after_unit":"EPOCH"'),
    )


def main() raises:
    print("==== Flux train-control wiring smoke ====")
    _gate_good_config()
    _gate_disabled_sampling()
    _gate_fail_loud()
    print("flux_train_control_wiring_smoke PASS")
