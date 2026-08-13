# SKEPTIC PROBE (not part of the port) — streams several blocks of a
# REAL-SHAPE synthetic checkpoint (correct hidden/inner/ffn/head_dim, random
# bf16 bytes — the actual MiniMax-H3 weights are not downloaded yet) through
# `minimax_h3_load_block_device` on the real GPU, to empirically check the
# "~0.77 GiB bf16 per block, streams fine through 24 GiB" claim and whether
# device memory is actually released between iterations (no leak).
#
# Memory is observed externally via `nvidia-smi` polling around this process,
# not from inside Mojo. This file just prints markers before/after each block
# and sleeps briefly so the external poller has time to sample.

from max.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.minimax_h3_dit import minimax_h3_released_config
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_load_block_device,
)


def main() raises:
    comptime FIXTURE_DIR = "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/scratchpad/h3_real_shape_fixture"

    print("MARK startup")
    var config = minimax_h3_released_config()
    config.validate()
    var st = ShardedSafeTensors.open(String(FIXTURE_DIR))
    print("  shards:", st.num_shards(), " tensors:", st.num_tensors())
    var ctx = DeviceContext()

    sleep(1.0)
    print("MARK baseline_after_ctx")
    sleep(1.0)

    for layer in range(4):
        print("MARK before_load", layer)
        var weights = minimax_h3_load_block_device(st, layer, config, ctx)
        ctx.synchronize()
        print("MARK after_load", layer, "tensors=", len(weights))
        sleep(1.0)
        # weights drops here at end of loop body scope
    ctx.synchronize()
    print("MARK after_loop_all_dropped")
    sleep(1.5)
    print("DONE")
