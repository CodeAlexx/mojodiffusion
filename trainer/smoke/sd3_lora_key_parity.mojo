# Tiny SD3 LoRA save/load key parity gate.
#
# Serenity reference:
#   modules/modelSetup/StableDiffusion3LoRASetup.py wraps optional text encoders
#   with prefixes "lora_te1", "lora_te2", "lora_te3", and transformer with
#   prefix "lora_transformer".
#   modules/module/LoRAModule.py raw state_dict keys are:
#     <prefix>.<module>.lora_down.weight
#     <prefix>.<module>.lora_up.weight
#     <prefix>.<module>.alpha
#   modules/modelSaver/stableDiffusion3/StableDiffusion3LoRASaver.py and
#   modules/modelLoader/stableDiffusion3/StableDiffusion3LoRALoader.py use
#   convert_sd3_lora_key_sets() for external namespace conversion, but this gate
#   deliberately checks the bounded raw Serenity wrapper-key contract.
#
# This does not claim full adapter inventory, external legacy/OMI conversion, or
# numeric SD3 parity.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSetup.stableDiffusion3LoraTargets import (
    sd3_bounded_lora_target_specs,
)
from serenity_trainer.modelSaver.stableDiffusion3.StableDiffusion3LoRASaver import (
    SD3_FMT_SAFETENSORS,
    build_stable_diffusion3_lora_state_dict_from_targets,
    save_stable_diffusion3_lora_state_dict,
    stable_diffusion3_lora_save_plan,
)
from serenity_trainer.modelLoader.stableDiffusion3.StableDiffusion3ModelLoader import (
    load_stable_diffusion3_lora_targets,
    stable_diffusion3_lora_conversion_plan,
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


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
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
    var targets = sd3_bounded_lora_target_specs(8, 32, 6, 10, 40)
    var rank = 2
    var alpha = Float32(4.0)
    var dtype = STDtype.BF16

    var conversion = stable_diffusion3_lora_conversion_plan()
    _expect_bool("conversion key sets", conversion.has_convert_key_sets, True)
    _expect_int("conversion range", conversion.range_upper_bound, 100)
    _expect_name(0, conversion.transformer_diffusers_prefix, String("lora_transformer"))
    _expect_name(1, conversion.clip_l_omi_prefix, String("clip_l"))
    _expect_name(2, conversion.clip_g_omi_prefix, String("clip_g"))
    _expect_name(3, conversion.t5_omi_prefix, String("t5"))

    var save_plan = stable_diffusion3_lora_save_plan(SD3_FMT_SAFETENSORS, String("/tmp/sd3_lora_key_parity.safetensors"))
    _expect_bool("save plan convert sets", save_plan.has_convert_key_sets, True)
    _expect_name(4, save_plan.target_key_namespace, String("legacy_diffusers"))

    var sd = build_stable_diffusion3_lora_state_dict_from_targets(targets, rank, alpha, ctx, dtype)

    _expect_int("target count", targets.len(), 9)
    _expect_int("state entry count", len(sd.names), targets.len() * 3)
    _expect_int("tensor entry count", len(sd.tensors), targets.len() * 3)

    _expect_name(
        0,
        sd.names[0],
        String("lora_te1.text_model.encoder.layers.0.self_attn.q_proj.lora_down.weight"),
    )
    _expect_name(
        3,
        sd.names[3],
        String("lora_te2.text_model.encoder.layers.0.mlp.fc1.lora_down.weight"),
    )
    _expect_name(
        6,
        sd.names[6],
        String("lora_te3.encoder.block.0.layer.0.SelfAttention.q.lora_down.weight"),
    )
    _expect_name(
        9,
        sd.names[9],
        String("lora_transformer.pos_embed.proj.lora_down.weight"),
    )
    _expect_name(
        12,
        sd.names[12],
        String("lora_transformer.transformer_blocks.0.attn.to_q.lora_down.weight"),
    )
    _expect_name(
        15,
        sd.names[15],
        String("lora_transformer.transformer_blocks.0.attn.add_q_proj.lora_down.weight"),
    )
    _expect_name(
        18,
        sd.names[18],
        String("lora_transformer.transformer_blocks.0.ff.net.0.proj.lora_down.weight"),
    )
    _expect_name(
        23,
        sd.names[23],
        String("lora_transformer.transformer_blocks.0.ff_context.net.2.alpha"),
    )
    _expect_name(
        24,
        sd.names[24],
        String("lora_te3.encoder.block.0.layer.1.DenseReluDense.wi_0.lora_down.weight"),
    )

    _expect_shape("te1.q.down", sd.tensors[0][], rank, 6)
    _expect_shape("te1.q.up", sd.tensors[1][], 6, rank)
    _expect_shape("te2.fc1.down", sd.tensors[3][], rank, 6)
    _expect_shape("te2.fc1.up", sd.tensors[4][], 24, rank)
    _expect_shape("te3.q.down", sd.tensors[6][], rank, 10)
    _expect_shape("te3.q.up", sd.tensors[7][], 10, rank)
    _expect_shape("transformer.pos.down", sd.tensors[9][], rank, 8)
    _expect_shape("transformer.pos.up", sd.tensors[10][], 8, rank)
    _expect_shape("transformer.ff.down", sd.tensors[18][], rank, 8)
    _expect_shape("transformer.ff.up", sd.tensors[19][], 32, rank)
    _expect_shape("transformer.ff_ctx.down", sd.tensors[21][], rank, 32)
    _expect_shape("transformer.ff_ctx.up", sd.tensors[22][], 8, rank)
    _expect_shape("te3.wi_0.down", sd.tensors[24][], rank, 10)
    _expect_shape("te3.wi_0.up", sd.tensors[25][], 40, rank)
    _expect_dtype("alpha dtype", sd.tensors[26][], dtype)
    _expect_float("alpha value", sd.tensors[26][].to_host(ctx)[0], alpha)

    var entry_count = len(sd.names)
    var path = String("/tmp/sd3_lora_key_parity.safetensors")
    save_stable_diffusion3_lora_state_dict(sd^, SD3_FMT_SAFETENSORS, path, ctx)
    var loaded = load_stable_diffusion3_lora_targets(path, targets, ctx, dtype)

    _expect_int("loaded rank", loaded.rank, rank)
    _expect_int("loaded adapters", len(loaded.a), targets.len())
    _expect_shape("loaded.te1.q.down", loaded.a[0][], rank, 6)
    _expect_shape("loaded.te1.q.up", loaded.b[0][], 6, rank)
    _expect_shape("loaded.te2.fc1.down", loaded.a[1][], rank, 6)
    _expect_shape("loaded.te2.fc1.up", loaded.b[1][], 24, rank)
    _expect_shape("loaded.te3.q.down", loaded.a[2][], rank, 10)
    _expect_shape("loaded.te3.q.up", loaded.b[2][], 10, rank)
    _expect_shape("loaded.transformer.q.down", loaded.a[4][], rank, 8)
    _expect_shape("loaded.transformer.q.up", loaded.b[4][], 8, rank)
    _expect_shape("loaded.transformer.ff.down", loaded.a[6][], rank, 8)
    _expect_shape("loaded.transformer.ff.up", loaded.b[6][], 32, rank)
    _expect_shape("loaded.te3.wi_0.down", loaded.a[8][], rank, 10)
    _expect_shape("loaded.te3.wi_0.up", loaded.b[8][], 40, rank)
    for i in range(len(loaded.alpha)):
        _expect_float(String("loaded alpha ") + String(i), loaded.alpha[i], alpha)

    print("SD3 LORA RAW KEY+LOAD PARITY OK: targets =", targets.len(), " entries =", entry_count)
