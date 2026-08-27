# minimax_h3_prepare_w8a8_cache — one-shot tool: build the full 50-block W8A8
# runtime cache on a 16GB card via the streamed prepare (one block on device
# at a time). The T2VA/I2VA/Ref2VA runtimes hard-require this file for their
# reusable streamed tail whenever resident-blocks < 50.
from max.gpu.host import DeviceContext

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.minimax_h3_dit import minimax_h3_released_config
from serenitymojo.models.dit.minimax_h3_runtime_cache import (
    minimax_h3_prepare_resident_cache_w8a8_streamed,
)

comptime DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
comptime OUT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/serenity_runtime_cache_v1/resident_w8a8_row_blocks_50.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var shards = ShardedSafeTensors.open(String(DIR))
    var config = minimax_h3_released_config()
    config.validate()
    minimax_h3_prepare_resident_cache_w8a8_streamed(
        shards, config,
        String(DIR) + String("/model.safetensors.index.json"),
        String(OUT), ctx,
    )
