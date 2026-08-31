# MiniMax-H3 LoRA checkpoint/runtime row-layout boundary.
#
# Musubi's module/checkpoint-facing FC1 LoRA-up tensor uses [gate; value] rows.
# Serenity's H3 runtime uses the inference loader convention [value; gate].
# The half swap is self-inverse, so one proven transform serves import/export.
# LoRA-down is indexed by input features and is therefore unchanged. QKV LoRA
# rows are already [all-q; all-k; all-v] on both module/runtime surfaces.

from std.collections import List

from serenitymojo.models.minimax_h3.loader import minimax_h3_swap_fc1


def minimax_h3_fc1_lora_up_musubi_to_runtime(
    values: List[Float32], ffn_dim: Int, rank: Int
) raises -> List[Float32]:
    """Map FC1 LoRA B/up [2*F,R] from Musubi to Serenity runtime order."""
    if rank <= 0:
        raise Error("MiniMax-H3 FC1 LoRA rank must be positive")
    return minimax_h3_swap_fc1(values, ffn_dim, rank)


def minimax_h3_fc1_lora_up_runtime_to_musubi(
    values: List[Float32], ffn_dim: Int, rank: Int
) raises -> List[Float32]:
    """Map FC1 LoRA B/up [2*F,R] from Serenity runtime to Musubi order."""
    if rank <= 0:
        raise Error("MiniMax-H3 FC1 LoRA rank must be positive")
    return minimax_h3_swap_fc1(values, ffn_dim, rank)
