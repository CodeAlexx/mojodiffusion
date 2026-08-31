# MiniMax-H3 released LoRA target/name/layout catalog.
#
# Oracle: kohya-ss/musubi-tuner b8717864713c9e4e7ef3d56eba1fc695a9b626a5
#   networks/lora_minimax_h3.py::MINIMAX_H3_DEFAULT_TARGET_PATTERN
#   networks/lora.py::LoRAModule/create_modules
# This owns metadata and checkpoint mapping only, never projection math.

from std.collections import List


comptime MINIMAX_H3_TRAINABLE_BLOCKS = 50
comptime MINIMAX_H3_HIDDEN = 5376
comptime MINIMAX_H3_INNER = 7168
comptime MINIMAX_H3_FFN = 14336
comptime MINIMAX_H3_LORA_TARGETS_PER_BLOCK = 4


@fieldwise_init
struct MiniMaxH3LoraTargetSpec(Copyable, Movable):
    var layer: Int
    var family: String
    var module_path: String
    var musubi_prefix: String
    var in_features: Int
    var out_features: Int
    var fc1_up_requires_runtime_swap: Bool

    def down_key(self) -> String:
        return self.musubi_prefix + String(".lora_down.weight")

    def up_key(self) -> String:
        return self.musubi_prefix + String(".lora_up.weight")

    def alpha_key(self) -> String:
        return self.musubi_prefix + String(".alpha")

    def down_shape(self, rank: Int) raises -> List[Int]:
        if rank <= 0:
            raise Error("MiniMax-H3 LoRA rank must be positive")
        return [rank, self.in_features]

    def up_shape(self, rank: Int) raises -> List[Int]:
        if rank <= 0:
            raise Error("MiniMax-H3 LoRA rank must be positive")
        return [self.out_features, rank]


def minimax_h3_lora_target(
    layer: Int, family: String,
) raises -> MiniMaxH3LoraTargetSpec:
    if layer < 0 or layer >= MINIMAX_H3_TRAINABLE_BLOCKS:
        raise Error("MiniMax-H3 LoRA layer must be in [0,50)")
    var module_suffix: String
    var musubi_suffix: String
    var inf: Int
    var outf: Int
    var swap_fc1 = False
    if family == String("qkv_proj"):
        module_suffix = String("attn.qkv_proj")
        musubi_suffix = String("attn_qkv_proj")
        inf = MINIMAX_H3_HIDDEN
        outf = 3 * MINIMAX_H3_INNER
    elif family == String("out_proj"):
        module_suffix = String("attn.out_proj")
        musubi_suffix = String("attn_out_proj")
        inf = MINIMAX_H3_INNER
        outf = MINIMAX_H3_HIDDEN
    elif family == String("fc1"):
        module_suffix = String("mlp.fc1")
        musubi_suffix = String("mlp_fc1")
        inf = MINIMAX_H3_HIDDEN
        outf = 2 * MINIMAX_H3_FFN
        swap_fc1 = True
    elif family == String("fc2"):
        module_suffix = String("mlp.fc2")
        musubi_suffix = String("mlp_fc2")
        inf = MINIMAX_H3_FFN
        outf = MINIMAX_H3_HIDDEN
    else:
        raise Error("MiniMax-H3 LoRA family must be qkv_proj, out_proj, fc1, or fc2")
    return MiniMaxH3LoraTargetSpec(
        layer,
        family,
        String("blocks.") + String(layer) + String(".") + module_suffix,
        String("lora_unet_blocks_") + String(layer) + String("_") + musubi_suffix,
        inf,
        outf,
        swap_fc1,
    )


def minimax_h3_lora_surface() raises -> List[MiniMaxH3LoraTargetSpec]:
    var out = List[MiniMaxH3LoraTargetSpec](
        capacity=MINIMAX_H3_TRAINABLE_BLOCKS * MINIMAX_H3_LORA_TARGETS_PER_BLOCK
    )
    for layer in range(MINIMAX_H3_TRAINABLE_BLOCKS):
        out.append(minimax_h3_lora_target(layer, String("qkv_proj")))
        out.append(minimax_h3_lora_target(layer, String("out_proj")))
        out.append(minimax_h3_lora_target(layer, String("fc1")))
        out.append(minimax_h3_lora_target(layer, String("fc2")))
    return out^
