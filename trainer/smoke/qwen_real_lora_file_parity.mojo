# Real Qwen LoRA file key/shape/dtype gate.
#
# Serenity reference file:
#   /home/alex/Serenity/output/qwen_100step_baseline/lora.safetensors
#
# Serenity source:
#   modules/modelSetup/QwenLoRASetup.py
#   modules/modelSaver/qwen/QwenLoRASaver.py
#   modules/modelLoader/qwen/QwenLoRALoader.py
#
# Gate:
#   load the real Serenity-saved LoRA file with raw keys, verify the full
#   transformer attn-mlp inventory: 60 blocks * 12 targets = 720 adapters,
#   2160 keys, BF16 tensors, rank 16, alpha 1.0. This is key/shape/dtype parity
#   only; it does not claim Qwen transformer forward/backward parity.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype

from serenity_trainer.modelSetup.qwenLoraTargets import (
    qwen_transformer_attn_mlp_target_specs,
)
from serenity_trainer.modelLoader.qwen.QwenLoRALoader import (
    load_qwen_lora_safetensors,
    load_qwen_lora_targets,
)


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _expect_float(name: String, got: Float32, expected: Float32) raises:
    var diff = got - expected
    if diff < Float32(0.0):
        diff = -diff
    if diff > Float32(0.001):
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _expect_shape(name: String, sh: List[Int], rows: Int, cols: Int) raises:
    _expect_int(name + String(".rank"), len(sh), 2)
    _expect_int(name + String(".rows"), sh[0], rows)
    _expect_int(name + String(".cols"), sh[1], cols)


def main() raises:
    var ctx = DeviceContext()
    var path = String("/home/alex/Serenity/output/qwen_100step_baseline/lora.safetensors")

    var state = load_qwen_lora_safetensors(path, ctx)
    _expect_int("real qwen lora key count", len(state.names), 2160)
    _expect_int("real qwen lora tensor count", len(state.tensors), 2160)

    var targets = qwen_transformer_attn_mlp_target_specs(60, 3072, 12288)
    _expect_int("full qwen target count", targets.len(), 720)

    var loaded = load_qwen_lora_targets(path, targets, ctx, STDtype.BF16)
    _expect_int("loaded adapter count", len(loaded.a), 720)
    _expect_int("loaded up count", len(loaded.b), 720)
    _expect_int("loaded alpha count", len(loaded.alpha), 720)
    _expect_int("rank", loaded.rank, 16)

    _expect_shape("block0.attn.to_q.down", loaded.a[0][].shape(), 16, 3072)
    _expect_shape("block0.attn.to_q.up", loaded.b[0][].shape(), 3072, 16)
    _expect_shape("block0.img_mlp.in.down", loaded.a[8][].shape(), 16, 3072)
    _expect_shape("block0.img_mlp.in.up", loaded.b[8][].shape(), 12288, 16)
    _expect_shape("block0.img_mlp.out.down", loaded.a[9][].shape(), 16, 12288)
    _expect_shape("block0.img_mlp.out.up", loaded.b[9][].shape(), 3072, 16)
    _expect_shape("block59.txt_mlp.out.down", loaded.a[719][].shape(), 16, 12288)
    _expect_shape("block59.txt_mlp.out.up", loaded.b[719][].shape(), 3072, 16)

    for i in range(len(loaded.alpha)):
        _expect_float(String("alpha ") + String(i), loaded.alpha[i], Float32(1.0))

    print("QWEN REAL LORA FILE PARITY OK: keys=2160 adapters=720 rank=16 dtype=BF16 alpha=1.0")
