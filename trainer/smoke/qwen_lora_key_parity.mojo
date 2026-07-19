# Tiny Qwen LoRA save/load key parity gate.
#
# Serenity reference:
#   modules/modelSetup/QwenLoRASetup.py wraps optional text_encoder LoRA with
#   prefix "text_encoder" and transformer LoRA with prefix "transformer";
#   modules/modelSaver/qwen/QwenLoRASaver.py and
#   modules/modelLoader/qwen/QwenLoRALoader.py both return no conversion key sets,
#   so raw LoRAModule state_dict keys are saved/loaded verbatim:
#     <prefix>.<module>.lora_down.weight
#     <prefix>.<module>.lora_up.weight
#     <prefix>.<module>.alpha
#
# This is bounded by design: it checks a tiny representative Qwen state dict,
# not numeric model parity or the full 60-layer target inventory.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSetup.qwenLoraTargets import (
    qwen_bounded_lora_target_specs,
)
from serenity_trainer.modelSaver.qwen.QwenLoRASaver import (
    QWEN_FMT_SAFETENSORS,
    build_qwen_lora_state_dict_from_targets,
    save_qwen_lora_state_dict,
)
from serenity_trainer.modelLoader.qwen.QwenLoRALoader import load_qwen_lora_targets


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


def _expect_name(index: Int, got: String, expected: String) raises:
    if got != expected:
        raise Error(
            String("name[") + String(index) + String("]: got ")
            + got + String(", expected ") + expected
        )


def _expect_shape(name: String, tensor: Tensor, rows: Int, cols: Int) raises:
    var sh = tensor.shape()
    _expect_int(name + String(".rank"), len(sh), 2)
    _expect_int(name + String(".rows"), sh[0], rows)
    _expect_int(name + String(".cols"), sh[1], cols)


def _expect_dtype(name: String, tensor: Tensor, expected: STDtype) raises:
    if tensor.dtype() != expected:
        raise Error(
            name + String(": dtype got ") + tensor.dtype().name()
            + String(", expected ") + expected.name()
        )


def main() raises:
    var ctx = DeviceContext()
    var targets = qwen_bounded_lora_target_specs(8, 32, 6)
    var rank = 2
    var alpha = Float32(4.0)
    var dtype = STDtype.BF16

    var sd = build_qwen_lora_state_dict_from_targets(targets, rank, alpha, ctx, dtype)

    _expect_int("target count", targets.len(), 5)
    _expect_int("state entry count", len(sd.names), targets.len() * 3)
    _expect_int("tensor entry count", len(sd.tensors), targets.len() * 3)

    _expect_name(
        0,
        sd.names[0],
        String("text_encoder.model.layers.0.self_attn.q_proj.lora_down.weight"),
    )
    _expect_name(
        3,
        sd.names[3],
        String("transformer.transformer_blocks.0.attn.to_q.lora_down.weight"),
    )
    _expect_name(
        6,
        sd.names[6],
        String("transformer.transformer_blocks.0.attn.add_q_proj.lora_down.weight"),
    )
    _expect_name(
        9,
        sd.names[9],
        String("transformer.transformer_blocks.0.img_mlp.net.0.proj.lora_down.weight"),
    )
    _expect_name(
        14,
        sd.names[14],
        String("transformer.transformer_blocks.0.img_mlp.net.2.alpha"),
    )

    _expect_shape("text_encoder.q.down", sd.tensors[0][], rank, 6)
    _expect_shape("text_encoder.q.up", sd.tensors[1][], 6, rank)
    _expect_shape("transformer.q.down", sd.tensors[3][], rank, 8)
    _expect_shape("transformer.q.up", sd.tensors[4][], 8, rank)
    _expect_shape("transformer.img_mlp_in.down", sd.tensors[9][], rank, 8)
    _expect_shape("transformer.img_mlp_in.up", sd.tensors[10][], 32, rank)
    _expect_shape("transformer.img_mlp_out.down", sd.tensors[12][], rank, 32)
    _expect_shape("transformer.img_mlp_out.up", sd.tensors[13][], 8, rank)
    _expect_dtype("alpha dtype", sd.tensors[14][], dtype)
    _expect_float("alpha value", sd.tensors[14][].to_host(ctx)[0], alpha)

    var entry_count = len(sd.names)
    var path = String("/tmp/qwen_lora_key_parity.safetensors")
    save_qwen_lora_state_dict(sd^, QWEN_FMT_SAFETENSORS, path, ctx)
    var loaded = load_qwen_lora_targets(path, targets, ctx, dtype)

    _expect_int("loaded rank", loaded.rank, rank)
    _expect_int("loaded adapters", len(loaded.a), targets.len())
    _expect_shape("loaded.text_encoder.q.down", loaded.a[0][], rank, 6)
    _expect_shape("loaded.text_encoder.q.up", loaded.b[0][], 6, rank)
    _expect_shape("loaded.transformer.q.down", loaded.a[1][], rank, 8)
    _expect_shape("loaded.transformer.q.up", loaded.b[1][], 8, rank)
    _expect_shape("loaded.transformer.img_mlp_in.down", loaded.a[3][], rank, 8)
    _expect_shape("loaded.transformer.img_mlp_in.up", loaded.b[3][], 32, rank)
    _expect_shape("loaded.transformer.img_mlp_out.down", loaded.a[4][], rank, 32)
    _expect_shape("loaded.transformer.img_mlp_out.up", loaded.b[4][], 8, rank)
    for i in range(len(loaded.alpha)):
        _expect_float(String("loaded alpha ") + String(i), loaded.alpha[i], alpha)

    print("QWEN LORA KEY+LOAD PARITY OK: targets =", targets.len(), " entries =", entry_count)
