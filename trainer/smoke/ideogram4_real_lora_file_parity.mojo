# Real ai-toolkit Ideogram4 LoRA file key/shape/dtype gate.
#
# Fixture:
#   /home/alex/Downloads/dever_arcane_style_ideogram4%20%28arcvfx%29.safetensors
#
# Verifies the file is the ai-toolkit block-stack inventory:
#   34 layers * 6 block targets = 204 adapters
#   408 tensors = lora_A + lora_B for every adapter
#   BF16 tensors, rank 32, alpha omitted -> loader defaults alpha to rank.
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors

from serenity_trainer.modelLoader.Ideogram4LoRALoader import (
    load_ideogram4_block_stack_lora,
)
from serenity_trainer.model.Ideogram4LoRABlock import Ideogram4LoraSet


comptime REAL_LORA = "/home/alex/Downloads/dever_arcane_style_ideogram4%20%28arcvfx%29.safetensors"


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


def _expect_adapter(
    name: String,
    idx: Int,
    loras: Ideogram4LoraSet,
    a_rows: Int,
    a_cols: Int,
    b_rows: Int,
    b_cols: Int,
) raises:
    if loras.ad[idx][].a.dtype() != STDtype.BF16:
        raise Error(name + String(".A dtype mismatch"))
    if loras.ad[idx][].b.dtype() != STDtype.BF16:
        raise Error(name + String(".B dtype mismatch"))
    _expect_shape(name + String(".A"), loras.ad[idx][].a.shape(), a_rows, a_cols)
    _expect_shape(name + String(".B"), loras.ad[idx][].b.shape(), b_rows, b_cols)
    _expect_float(name + String(".alpha"), loras.ad[idx][].alpha, Float32(32.0))


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(REAL_LORA))
    _expect_int("real ideogram4 lora tensor count", st.num_tensors(), 408)

    var loras = load_ideogram4_block_stack_lora(String(REAL_LORA), ctx)
    _expect_int("adapter count", len(loras.ad), 204)
    _expect_int("layer count", loras.n_layers, 34)
    _expect_int("rank", loras.rank, 32)

    # Slot order: qkv, o, w1, w2, w3, adaln_modulation.
    _expect_adapter(String("layer0.qkv"), 0, loras, 32, 4608, 13824, 32)
    _expect_adapter(String("layer0.o"), 1, loras, 32, 4608, 4608, 32)
    _expect_adapter(String("layer0.w1"), 2, loras, 32, 4608, 12288, 32)
    _expect_adapter(String("layer0.w2"), 3, loras, 32, 12288, 4608, 32)
    _expect_adapter(String("layer0.w3"), 4, loras, 32, 4608, 12288, 32)
    _expect_adapter(String("layer0.adaln"), 5, loras, 32, 512, 18432, 32)
    _expect_adapter(String("layer33.qkv"), 33 * 6, loras, 32, 4608, 13824, 32)
    _expect_adapter(String("layer33.adaln"), 33 * 6 + 5, loras, 32, 512, 18432, 32)

    print("IDEOGRAM4 REAL LORA FILE PARITY OK: tensors=408 adapters=204 rank=32 dtype=BF16 alpha=rank")
