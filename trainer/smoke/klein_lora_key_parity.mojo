# Tiny Flux2/Klein LoRA save-key parity gate.
#
# Serenity reference:
#   modules/modelSetup/Flux2LoRASetup.py:57-58 wraps Flux2 transformer Linears with
#   prefix "transformer"; modules/modelSaver/flux2/Flux2LoRASaver.py:17-30 returns
#   those raw LoRAModule state_dict keys with no conversion.
#
# This gate checks the Mojo saver emits the same block-key surface for a tiny
# synthetic set: 1 double block * 12 Linears + 1 single block * 2 Linears, each
# with lora_down/lora_up/alpha.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor

from serenity_trainer.model.klein.klein_stack_lora import build_klein_lora_set
from serenity_trainer.modelSetup.flux2LoraTargets import flux2_lora_count
from serenity_trainer.modelLoader.Flux2RuntimeLoader import load_flux2_lora_fused
from serenity_trainer.modelSaver.flux2.Flux2LoRASaver import (
    build_flux2_lora_state_dict, save_flux2_lora,
)


def _expect_int(name: String, got: Int, expected: Int) raises:
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


def _expect_adapter(
    name: String, rank: Int, in_f: Int, out_f: Int,
    expected_rank: Int, expected_in: Int, expected_out: Int,
) raises:
    _expect_int(name + String(".rank"), rank, expected_rank)
    _expect_int(name + String(".in"), in_f, expected_in)
    _expect_int(name + String(".out"), out_f, expected_out)


def main() raises:
    var ctx = DeviceContext()
    comptime D = 4
    comptime F = 8
    comptime RANK = 2

    var set = build_klein_lora_set(1, 1, D, F, RANK, Float32(2.0))
    var sd = build_flux2_lora_state_dict(set, D, ctx)

    var expected_modules = flux2_lora_count(1, 1)
    _expect_int("module count", expected_modules, 14)
    _expect_int("state entry count", len(sd.names), expected_modules * 3)
    _expect_int("tensor entry count", len(sd.tensors), expected_modules * 3)

    _expect_name(0, sd.names[0], String("transformer.transformer_blocks.0.attn.to_q.lora_down.weight"))
    _expect_name(3, sd.names[3], String("transformer.transformer_blocks.0.attn.to_k.lora_down.weight"))
    _expect_name(6, sd.names[6], String("transformer.transformer_blocks.0.attn.to_v.lora_down.weight"))
    _expect_name(12, sd.names[12], String("transformer.transformer_blocks.0.attn.add_q_proj.lora_down.weight"))
    _expect_name(21, sd.names[21], String("transformer.transformer_blocks.0.attn.to_add_out.lora_down.weight"))
    _expect_name(24, sd.names[24], String("transformer.transformer_blocks.0.ff.linear_in.lora_down.weight"))
    _expect_name(27, sd.names[27], String("transformer.transformer_blocks.0.ff.linear_out.lora_down.weight"))
    _expect_name(30, sd.names[30], String("transformer.transformer_blocks.0.ff_context.linear_in.lora_down.weight"))
    _expect_name(33, sd.names[33], String("transformer.transformer_blocks.0.ff_context.linear_out.lora_down.weight"))
    _expect_name(36, sd.names[36], String("transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_down.weight"))
    _expect_name(39, sd.names[39], String("transformer.single_transformer_blocks.0.attn.to_out.lora_down.weight"))
    _expect_name(41, sd.names[41], String("transformer.single_transformer_blocks.0.attn.to_out.alpha"))

    _expect_shape("to_q.down", sd.tensors[0][], RANK, D)
    _expect_shape("to_q.up", sd.tensors[1][], D, RANK)
    _expect_shape("txt_q.down", sd.tensors[12][], RANK, D)
    _expect_shape("txt_q.up", sd.tensors[13][], D, RANK)
    _expect_shape("ff_in.down", sd.tensors[24][], RANK, D)
    _expect_shape("ff_in.up", sd.tensors[25][], 2 * F, RANK)
    _expect_shape("ff_out.down", sd.tensors[27][], RANK, F)
    _expect_shape("ff_out.up", sd.tensors[28][], D, RANK)
    _expect_shape("single_qkv_mlp.down", sd.tensors[36][], RANK, D)
    _expect_shape("single_qkv_mlp.up", sd.tensors[37][], 3 * D + 2 * F, RANK)
    _expect_shape("single_out.down", sd.tensors[39][], RANK, D + F)
    _expect_shape("single_out.up", sd.tensors[40][], D, RANK)

    var path = String("/tmp/klein_flux2_lora_roundtrip.safetensors")
    save_flux2_lora(set, D, path, ctx)
    var loaded = load_flux2_lora_fused(path, 1, 1, ctx)

    _expect_int("loaded rank", loaded.rank, RANK)
    _expect_int("loaded double adapters", len(loaded.dbl), 12)
    _expect_int("loaded single adapters", len(loaded.sgl), 2)
    _expect_adapter("loaded.to_q", loaded.dbl[0].rank, loaded.dbl[0].in_f, loaded.dbl[0].out_f, RANK, D, D)
    _expect_adapter("loaded.to_k", loaded.dbl[1].rank, loaded.dbl[1].in_f, loaded.dbl[1].out_f, RANK, D, D)
    _expect_adapter("loaded.to_v", loaded.dbl[2].rank, loaded.dbl[2].in_f, loaded.dbl[2].out_f, RANK, D, D)
    _expect_adapter("loaded.to_out", loaded.dbl[3].rank, loaded.dbl[3].in_f, loaded.dbl[3].out_f, RANK, D, D)
    _expect_adapter("loaded.ff_in", loaded.dbl[4].rank, loaded.dbl[4].in_f, loaded.dbl[4].out_f, RANK, D, 2 * F)
    _expect_adapter("loaded.ff_out", loaded.dbl[5].rank, loaded.dbl[5].in_f, loaded.dbl[5].out_f, RANK, F, D)
    _expect_adapter("loaded.add_q", loaded.dbl[6].rank, loaded.dbl[6].in_f, loaded.dbl[6].out_f, RANK, D, D)
    _expect_adapter("loaded.add_k", loaded.dbl[7].rank, loaded.dbl[7].in_f, loaded.dbl[7].out_f, RANK, D, D)
    _expect_adapter("loaded.add_v", loaded.dbl[8].rank, loaded.dbl[8].in_f, loaded.dbl[8].out_f, RANK, D, D)
    _expect_adapter("loaded.add_out", loaded.dbl[9].rank, loaded.dbl[9].in_f, loaded.dbl[9].out_f, RANK, D, D)
    _expect_adapter("loaded.ff_ctx_in", loaded.dbl[10].rank, loaded.dbl[10].in_f, loaded.dbl[10].out_f, RANK, D, 2 * F)
    _expect_adapter("loaded.ff_ctx_out", loaded.dbl[11].rank, loaded.dbl[11].in_f, loaded.dbl[11].out_f, RANK, F, D)
    _expect_adapter("loaded.single_qkv_mlp", loaded.sgl[0].rank, loaded.sgl[0].in_f, loaded.sgl[0].out_f, RANK, D, 3 * D + 2 * F)
    _expect_adapter("loaded.single_out", loaded.sgl[1].rank, loaded.sgl[1].in_f, loaded.sgl[1].out_f, RANK, D + F, D)

    print("KLEIN FLUX2 LORA KEY+LOAD PARITY OK: modules =", expected_modules, " entries =", len(sd.names))
