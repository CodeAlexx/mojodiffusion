# SKEPTIC PROBE (not part of the port) — measures REAL device memory for
# `minimax_h3_load_block_device` against REAL MiniMax-H3 checkpoint bytes
# (layers 0 and 1, both complete in the one real shard that has landed:
# model-00001-of-00013.safetensors). Replaces the earlier synthetic-fixture
# 1.3-1.5 GB figure with a measurement on actual weights. Same manual
# ShardedSafeTensors bypass as minimax_h3_real_block_device_probe.mojo (no
# index.json exists yet) -- see that file's header for why this is safe and
# does not modify the checkpoint directory.
#
# Memory is sampled externally via nvidia-smi around this process; this file
# just prints timestamped markers and sleeps briefly between them.

from std.collections import Dict, List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.minimax_h3_dit import minimax_h3_released_config
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_load_block_device,
)

comptime SHARD_PATH = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/model-00001-of-00013.safetensors"


def main() raises:
    print("MARK startup")
    var st = SafeTensors.open(String(SHARD_PATH))
    var names = st.names()
    var name_to_shard = Dict[String, Int]()
    for i in range(len(names)):
        name_to_shard[names[i]] = 0
    var shards = List[ArcPointer[SafeTensors]]()
    shards.append(ArcPointer(st^))
    var sharded = ShardedSafeTensors(shards^, name_to_shard^)

    var config = minimax_h3_released_config()
    config.validate()
    var ctx = DeviceContext()
    ctx.synchronize()

    sleep(1.5)
    print("MARK baseline_after_ctx")
    sleep(1.5)

    for layer in range(2):
        print("MARK before_load", layer)
        sleep(0.5)
        var weights = minimax_h3_load_block_device(sharded, layer, config, ctx)
        ctx.synchronize()
        sleep(0.5)
        print("MARK after_load", layer, "tensors=", len(weights))
        sleep(1.5)
        # weights drops here

    ctx.synchronize()
    sleep(1.0)
    print("MARK after_loop_all_dropped")
    sleep(1.5)
    print("DONE")
