# SD3 Linear LoRA inventory/contract smoke.
#
# No real Serenity SD3 LoRA file was established for this gate. This smoke is
# intentionally an inventory contract, not a full Serenity parity claim.
#
# Closest Serenity source paths:
#   /home/alex/Serenity/modules/modelSetup/StableDiffusion3LoRASetup.py
#   /home/alex/Serenity/modules/module/LoRAModule.py
#   /home/alex/Serenity/modules/util/convert/lora/convert_sd3_lora.py
#   /home/alex/Serenity/modules/util/convert/lora/convert_clip.py
#   /home/alex/Serenity/modules/util/convert/lora/convert_t5.py
#
# Scope checked here:
#   deterministic raw wrapper keys, expanded Linear target count, role/source
#   metadata, LoRA down/up/alpha shapes, BF16 storage dtype, scalar alpha, and
#   local save/load round trip against the generated contract targets.
#
# Gaps:
#   no in-session Serenity-generated SD3 LoRA reference file, no named_modules
#   trace from a real SD3 checkpoint, no Conv2d target coverage, no numeric
#   forward/backward parity.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSetup.stableDiffusion3LoraTargets import (
    sd3_linear_lora_inventory_target_specs,
)
from serenity_trainer.modelSaver.stableDiffusion3.StableDiffusion3LoRASaver import (
    SD3_FMT_SAFETENSORS,
    build_stable_diffusion3_lora_state_dict_from_targets,
    save_stable_diffusion3_lora_state_dict,
)
from serenity_trainer.modelLoader.stableDiffusion3.StableDiffusion3ModelLoader import (
    load_stable_diffusion3_lora_targets,
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
    var transformer_blocks = 2
    var clip_layers = 3
    var t5_blocks = 2
    var transformer_dim = 8
    var transformer_mlp_dim = 32
    var pooled_dim = 12
    var clip_dim = 6
    var clip_mlp_dim = 24
    var clip_projection_dim = 5
    var t5_dim = 10
    var t5_mlp_dim = 40
    var rank = 2
    var alpha = Float32(4.0)
    var dtype = STDtype.BF16

    var targets = sd3_linear_lora_inventory_target_specs(
        transformer_blocks,
        clip_layers,
        t5_blocks,
        transformer_dim,
        transformer_mlp_dim,
        pooled_dim,
        clip_dim,
        clip_mlp_dim,
        clip_projection_dim,
        t5_dim,
        t5_mlp_dim,
    )

    _expect_int("target count", targets.len(), 96)
    _expect_int("role count", len(targets.roles), targets.len())
    _expect_int("source path count", len(targets.source_paths), targets.len())
    _expect_int("in feature count", len(targets.in_features), targets.len())
    _expect_int("out feature count", len(targets.out_features), targets.len())

    _expect_name(0, targets.prefixes[0], String("lora_transformer.pos_embed.proj"))
    _expect_name(1, targets.prefixes[1], String("lora_transformer.context_embedder"))
    _expect_name(8, targets.prefixes[8], String("lora_transformer.transformer_blocks.0.attn.to_q"))
    _expect_name(16, targets.prefixes[16], String("lora_transformer.transformer_blocks.0.norm1.linear"))
    _expect_name(25, targets.prefixes[25], String("lora_transformer.transformer_blocks.0.ff_context.net.2"))
    _expect_name(43, targets.prefixes[43], String("lora_transformer.transformer_blocks.1.ff_context.net.2"))
    _expect_name(44, targets.prefixes[44], String("lora_te1.text_projection"))
    _expect_name(45, targets.prefixes[45], String("lora_te2.text_projection"))
    _expect_name(46, targets.prefixes[46], String("lora_te1.text_model.encoder.layers.0.mlp.fc1"))
    _expect_name(47, targets.prefixes[47], String("lora_te2.text_model.encoder.layers.0.mlp.fc1"))
    _expect_name(81, targets.prefixes[81], String("lora_te2.text_model.encoder.layers.2.self_attn.v_proj"))
    _expect_name(82, targets.prefixes[82], String("lora_te3.encoder.block.0.layer.0.SelfAttention.k"))
    _expect_name(95, targets.prefixes[95], String("lora_te3.encoder.block.1.layer.1.DenseReluDense.wo"))

    _expect_name(0, targets.roles[0], String("transformer.x_embedder.proj"))
    _expect_name(16, targets.roles[16], String("transformer.block.norm1.linear"))
    _expect_name(46, targets.roles[46], String("clip_l.mlp.fc1"))
    _expect_name(95, targets.roles[95], String("t5.layer.1.DenseReluDense.wo"))
    _expect_name(
        0,
        targets.source_paths[0],
        String("/home/alex/Serenity/modules/modelSetup/StableDiffusion3LoRASetup.py | /home/alex/Serenity/modules/module/LoRAModule.py | /home/alex/Serenity/modules/util/convert/lora/convert_sd3_lora.py"),
    )
    _expect_name(
        46,
        targets.source_paths[46],
        String("/home/alex/Serenity/modules/modelSetup/StableDiffusion3LoRASetup.py | /home/alex/Serenity/modules/module/LoRAModule.py | /home/alex/Serenity/modules/util/convert/lora/convert_clip.py"),
    )
    _expect_name(
        95,
        targets.source_paths[95],
        String("/home/alex/Serenity/modules/modelSetup/StableDiffusion3LoRASetup.py | /home/alex/Serenity/modules/module/LoRAModule.py | /home/alex/Serenity/modules/util/convert/lora/convert_t5.py"),
    )

    var sd = build_stable_diffusion3_lora_state_dict_from_targets(targets, rank, alpha, ctx, dtype)
    _expect_int("state entry count", len(sd.names), targets.len() * 3)
    _expect_int("tensor entry count", len(sd.tensors), targets.len() * 3)

    _expect_name(0, sd.names[0], String("lora_transformer.pos_embed.proj.lora_down.weight"))
    _expect_name(3, sd.names[3], String("lora_transformer.context_embedder.lora_down.weight"))
    _expect_name(48, sd.names[48], String("lora_transformer.transformer_blocks.0.norm1.linear.lora_down.weight"))
    _expect_name(132, sd.names[132], String("lora_te1.text_projection.lora_down.weight"))
    _expect_name(138, sd.names[138], String("lora_te1.text_model.encoder.layers.0.mlp.fc1.lora_down.weight"))
    _expect_name(285, sd.names[285], String("lora_te3.encoder.block.1.layer.1.DenseReluDense.wo.lora_down.weight"))

    _expect_shape("pos.down", sd.tensors[0][], rank, transformer_dim)
    _expect_shape("pos.up", sd.tensors[1][], transformer_dim, rank)
    _expect_dtype("pos.down dtype", sd.tensors[0][], dtype)
    _expect_dtype("pos.up dtype", sd.tensors[1][], dtype)
    _expect_shape("context.down", sd.tensors[3][], rank, t5_dim)
    _expect_shape("context.up", sd.tensors[4][], pooled_dim, rank)
    _expect_dtype("context.down dtype", sd.tensors[3][], dtype)
    _expect_dtype("context.up dtype", sd.tensors[4][], dtype)
    _expect_shape("norm1.down", sd.tensors[48][], rank, transformer_dim)
    _expect_shape("norm1.up", sd.tensors[49][], 6 * transformer_dim, rank)
    _expect_dtype("norm1.down dtype", sd.tensors[48][], dtype)
    _expect_dtype("norm1.up dtype", sd.tensors[49][], dtype)
    _expect_shape("clip.proj.down", sd.tensors[132][], rank, clip_dim)
    _expect_shape("clip.proj.up", sd.tensors[133][], clip_projection_dim, rank)
    _expect_shape("clip.fc1.down", sd.tensors[138][], rank, clip_dim)
    _expect_shape("clip.fc1.up", sd.tensors[139][], clip_mlp_dim, rank)
    _expect_dtype("clip.fc1.down dtype", sd.tensors[138][], dtype)
    _expect_dtype("clip.fc1.up dtype", sd.tensors[139][], dtype)
    _expect_shape("t5.wo.down", sd.tensors[285][], rank, t5_mlp_dim)
    _expect_shape("t5.wo.up", sd.tensors[286][], t5_dim, rank)
    _expect_dtype("t5.wo.down dtype", sd.tensors[285][], dtype)
    _expect_dtype("t5.wo.up dtype", sd.tensors[286][], dtype)
    _expect_dtype("alpha dtype", sd.tensors[287][], dtype)
    _expect_float("alpha value", sd.tensors[287][].to_host(ctx)[0], alpha)

    var path = String("/tmp/sd3_lora_inventory_contract.safetensors")
    save_stable_diffusion3_lora_state_dict(sd^, SD3_FMT_SAFETENSORS, path, ctx)
    var loaded = load_stable_diffusion3_lora_targets(path, targets, ctx, dtype)

    _expect_int("loaded rank", loaded.rank, rank)
    _expect_int("loaded adapters", len(loaded.a), targets.len())
    _expect_shape("loaded.pos.down", loaded.a[0][], rank, transformer_dim)
    _expect_shape("loaded.context.up", loaded.b[1][], pooled_dim, rank)
    _expect_shape("loaded.norm1.up", loaded.b[16][], 6 * transformer_dim, rank)
    _expect_shape("loaded.clip.fc1.up", loaded.b[46][], clip_mlp_dim, rank)
    _expect_shape("loaded.t5.wo.down", loaded.a[95][], rank, t5_mlp_dim)
    _expect_dtype("loaded.pos.down dtype", loaded.a[0][], dtype)
    _expect_dtype("loaded.context.up dtype", loaded.b[1][], dtype)
    _expect_dtype("loaded.norm1.up dtype", loaded.b[16][], dtype)
    _expect_dtype("loaded.clip.fc1.up dtype", loaded.b[46][], dtype)
    _expect_dtype("loaded.t5.wo.down dtype", loaded.a[95][], dtype)

    for i in range(len(loaded.alpha)):
        _expect_float(String("loaded alpha ") + String(i), loaded.alpha[i], alpha)

    print("SD3 LORA INVENTORY CONTRACT OK: linear_targets=96 entries=288 rank=2 dtype=BF16 alpha=4.0")
