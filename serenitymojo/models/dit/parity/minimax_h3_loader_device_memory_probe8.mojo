# SKEPTIC PROBE — same as minimax_h3_loader_device_memory_probe.mojo but
# streams 8 blocks with longer settle time between iterations, to distinguish
# "plateaus after warmup" (caching allocator / benign) from "grows every
# iteration" (genuine leak: the previous block's Dict is not actually being
# dropped).

from std.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.minimax_h3_dit import minimax_h3_released_config
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_load_block_device,
)


def main() raises:
    comptime FIXTURE_DIR = "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/scratchpad/h3_real_shape_fixture8"

    print("MARK startup")
    var config = minimax_h3_released_config()
    config.validate()
    var st = ShardedSafeTensors.open(String(FIXTURE_DIR))
    print("  shards:", st.num_shards(), " tensors:", st.num_tensors())
    var ctx = DeviceContext()
    ctx.synchronize()

    sleep(1.5)
    print("MARK baseline_after_ctx")
    sleep(1.5)

    for layer in range(8):
        print("MARK before_load", layer)
        sleep(0.5)
        var weights = minimax_h3_load_block_device(st, layer, config, ctx)
        ctx.synchronize()
        sleep(0.5)
        print("MARK after_load", layer, "tensors=", len(weights))
        sleep(1.5)
        # weights drops here at end of loop body scope
    ctx.synchronize()
    sleep(1.0)
    print("MARK after_loop_all_dropped")
    sleep(2.0)
    print("DONE")
