# offload_checkpoint_config_smoke.mojo — OneTrainer offload/checkpoint config
# reachability contract for read_model_config.
#
# Scope: scalar config parsing/validation only. This does not allocate device
# tensors, run model math, or move activations through host F32 lists.
#
# Run:
#   pixi run mojo run -I . serenitymojo/io/offload_checkpoint_config_smoke.mojo

from std.memory import alloc

from serenitymojo.io.ffi import (
    sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import (
    TrainConfig,
    GRADIENT_CHECKPOINTING_OFF,
    GRADIENT_CHECKPOINTING_ON,
    GRADIENT_CHECKPOINTING_CPU_OFFLOADED,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("offload config smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("offload config smoke: short write to ") + path)


def _eq(name: String, a: Int, b: Int) raises:
    if a != b:
        raise Error(name + " expected " + String(b) + ", got " + String(a))


def _eqb(name: String, a: Bool, b: Bool) raises:
    if a != b:
        raise Error(name + " expected " + String(b) + ", got " + String(a))


def _close(name: String, a: Float64, b: Float64) raises:
    var d = a - b
    var ad = d if d >= Float64(0.0) else -d
    if ad > Float64(1e-9):
        raise Error(name + " mismatch: got " + String(a) + " expected " + String(b))


comptime _ARCH = String(
    '"model_type":"klein","inner_dim":64,"in_channels":4,'
    + '"joint_attention_dim":64,"out_channels":4,"num_double":1,'
    + '"num_single":1,"num_heads":2,"head_dim":32,"mlp_hidden":128,'
    + '"timestep_dim":256,"learning_rate":1e-4,"lora_rank":16,"lora_alpha":16.0'
)


def _read_json(path: String, body_tail: String) raises -> TrainConfig:
    var js = String("{") + _ARCH + body_tail + String("}")
    _write_file(path, js)
    return read_model_config(path)


def _gate_defaults() raises:
    print("--- defaults mirror OneTrainer scalar policy ---")
    var c = _read_json(String("/tmp/ot_offload_defaults.json"), String(""))
    _eq("gradient_checkpointing default", c.gradient_checkpointing, GRADIENT_CHECKPOINTING_ON)
    _eqb("enable_async_offloading default", c.enable_async_offloading, True)
    _eqb("enable_activation_offloading default", c.enable_activation_offloading, True)
    _close("layer_offload_fraction default", c.layer_offload_fraction, Float64(0.0))
    _eqb("checkpointing enabled", c.gradient_checkpointing_enabled(), True)
    _eqb("checkpointing offload", c.gradient_checkpointing_offload(), False)
    _eqb("activation offload derived", c.activation_offload_enabled(), False)
    _eqb("layer offload derived", c.layer_offload_enabled(), False)
    _eqb("async offload cuda derived", c.async_offload_enabled_for_cuda(True), True)


def _gate_cpu_offloaded() raises:
    print("--- CPU_OFFLOADED maps to active OneTrainer offload policy ---")
    var tail = String(",")
    tail += '"gradient_checkpointing":"CPU_OFFLOADED",'
    tail += '"enable_async_offloading":true,'
    tail += '"enable_activation_offloading":true,'
    tail += '"layer_offload_fraction":0.7'
    var c = _read_json(String("/tmp/ot_offload_cpu.json"), tail)
    _eq("gradient_checkpointing", c.gradient_checkpointing, GRADIENT_CHECKPOINTING_CPU_OFFLOADED)
    _eqb("checkpointing enabled", c.gradient_checkpointing_enabled(), True)
    _eqb("checkpointing offload", c.gradient_checkpointing_offload(), True)
    _eqb("activation offload derived", c.activation_offload_enabled(), True)
    _eqb("layer offload derived", c.layer_offload_enabled(), True)
    _eqb("async offload cuda derived", c.async_offload_enabled_for_cuda(True), True)
    _close("layer_offload_fraction", c.layer_offload_fraction, Float64(0.7))


def _gate_legacy_bool() raises:
    print("--- legacy bool migration matches OneTrainer True->ON / False->OFF ---")
    var on_cfg = _read_json(
        String("/tmp/ot_offload_bool_on.json"),
        String(',"gradient_checkpointing":true'),
    )
    var off_cfg = _read_json(
        String("/tmp/ot_offload_bool_off.json"),
        String(',"gradient_checkpointing":false'),
    )
    _eq("bool true -> ON", on_cfg.gradient_checkpointing, GRADIENT_CHECKPOINTING_ON)
    _eq("bool false -> OFF", off_cfg.gradient_checkpointing, GRADIENT_CHECKPOINTING_OFF)
    _eqb("OFF derived enabled", off_cfg.gradient_checkpointing_enabled(), False)


def _expect_reader_raises(path: String, tail: String) raises:
    var raised = False
    try:
        var c = _read_json(path, tail)
        _ = c.gradient_checkpointing
    except e:
        raised = True
        print("  raised as expected:", String(e))
    if not raised:
        raise Error("offload config smoke: invalid config did not raise")


def _gate_invalids() raises:
    print("--- invalid enum/range fail during config load ---")
    _expect_reader_raises(
        String("/tmp/ot_offload_bad_enum.json"),
        String(',"gradient_checkpointing":"GPU_MAGIC"'),
    )
    _expect_reader_raises(
        String("/tmp/ot_offload_bad_fraction.json"),
        String(',"layer_offload_fraction":1.25'),
    )


def main() raises:
    print("=== OneTrainer offload/checkpoint config smoke ===")
    _gate_defaults()
    _gate_cpu_offloaded()
    _gate_legacy_bool()
    _gate_invalids()
    print("offload_checkpoint_config_smoke PASS")
