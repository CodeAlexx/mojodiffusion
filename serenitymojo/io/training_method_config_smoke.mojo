# training_method_config_smoke.mojo - bounded config parser smoke for
# OneTrainer-style training_method strings.
#
# Scope: scalar config parsing only. This does not load model weights, allocate
# device tensors, or touch CUDA math.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/io/training_method_config_smoke.mojo

from std.memory import alloc

from serenitymojo.io.ffi import (
    sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import (
    TrainConfig,
    TRAINING_METHOD_LORA,
    TRAINING_METHOD_FINE_TUNE,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("training method smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("training method smoke: short write to ") + path)


def _eq(name: String, a: Int, b: Int) raises:
    if a != b:
        raise Error(name + " expected " + String(b) + ", got " + String(a))


def _eqb(name: String, a: Bool, b: Bool) raises:
    if a != b:
        raise Error(name + " expected " + String(b) + ", got " + String(a))


def _read_json(path: String, body: String) raises -> TrainConfig:
    var js = String("{") + body + String("}")
    _write_file(path, js)
    return read_model_config(path)


def _expect_method(path: String, body: String, expected: Int) raises:
    var c = _read_json(path, body)
    _eq("training_method tag", c.training_method, expected)
    _eqb("is_lora_training", c.is_lora_training(), expected == TRAINING_METHOD_LORA)
    _eqb(
        "is_full_finetune_training",
        c.is_full_finetune_training(),
        expected == TRAINING_METHOD_FINE_TUNE,
    )


def _gate_default_lora() raises:
    print("--- default absent training_method -> LORA ---")
    _expect_method(
        String("/tmp/training_method_default.json"),
        String(""),
        TRAINING_METHOD_LORA,
    )


def _gate_explicit_lora() raises:
    print("--- explicit LORA strings -> LORA ---")
    _expect_method(
        String("/tmp/training_method_lora.json"),
        String('"training_method":"LORA"'),
        TRAINING_METHOD_LORA,
    )
    _expect_method(
        String("/tmp/train_method_lora_lower.json"),
        String('"train_method":"lora"'),
        TRAINING_METHOD_LORA,
    )


def _gate_explicit_full_finetune() raises:
    print("--- explicit FINE_TUNE strings -> FINE_TUNE ---")
    _expect_method(
        String("/tmp/training_method_fine_tune.json"),
        String('"training_method":"FINE_TUNE"'),
        TRAINING_METHOD_FINE_TUNE,
    )
    _expect_method(
        String("/tmp/train_method_full.json"),
        String('"train_method":"full"'),
        TRAINING_METHOD_FINE_TUNE,
    )
    _expect_method(
        String("/tmp/method_finetune.json"),
        String('"method":"finetune"'),
        TRAINING_METHOD_FINE_TUNE,
    )


def _expect_reader_raises(path: String, body: String) raises:
    var raised = False
    try:
        var c = _read_json(path, body)
        _ = c.training_method
    except e:
        raised = True
        print("  raised as expected:", String(e))
    if not raised:
        raise Error("training method smoke: unknown method did not raise")


def _gate_unknown_fails_loud() raises:
    print("--- unknown training method fails loud ---")
    _expect_reader_raises(
        String("/tmp/training_method_unknown.json"),
        String('"method":"NOT_A_METHOD"'),
    )


def main() raises:
    print("=== OneTrainer training method config smoke ===")
    _gate_default_lora()
    _gate_explicit_lora()
    _gate_explicit_full_finetune()
    _gate_unknown_fails_loud()
    print("training_method_config_smoke PASS")
