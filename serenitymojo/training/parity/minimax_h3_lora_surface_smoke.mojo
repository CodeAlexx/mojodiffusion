# Deterministic MiniMax-H3 released LoRA target/name/shape catalog smoke.

from serenitymojo.training.minimax_h3.lora_surface import (
    MINIMAX_H3_LORA_TARGETS_PER_BLOCK,
    MINIMAX_H3_TRAINABLE_BLOCKS,
    minimax_h3_lora_surface,
)


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def main() raises:
    comptime RANK = 16
    var surface = minimax_h3_lora_surface()
    _require(
        len(surface) == MINIMAX_H3_TRAINABLE_BLOCKS * MINIMAX_H3_LORA_TARGETS_PER_BLOCK,
        "MiniMax-H3 LoRA surface must contain exactly 200 targets",
    )
    var fc1_count = 0
    for i in range(len(surface)):
        var target = surface[i].copy()
        _require(target.down_shape(RANK)[0] == RANK, "LoRA down rank axis")
        _require(target.up_shape(RANK)[1] == RANK, "LoRA up rank axis")
        _require(target.down_key().endswith(".lora_down.weight"), "down key")
        _require(target.up_key().endswith(".lora_up.weight"), "up key")
        _require(target.alpha_key().endswith(".alpha"), "alpha key")
        if target.fc1_up_requires_runtime_swap:
            fc1_count += 1
            _require(target.family == String("fc1"), "only FC1 up rows may swap")
        for j in range(i):
            _require(
                surface[j].musubi_prefix != target.musubi_prefix,
                "MiniMax-H3 LoRA checkpoint prefixes must be unique",
            )
    _require(fc1_count == 50, "exactly one FC1 row-swap target per block")
    _require(
        surface[0].musubi_prefix == String("lora_unet_blocks_0_attn_qkv_proj"),
        "first Musubi target name",
    )
    _require(
        surface[len(surface) - 1].musubi_prefix == String("lora_unet_blocks_49_mlp_fc2"),
        "last Musubi target name",
    )
    print("MiniMax-H3 200-target LoRA surface SMOKE PASS")
