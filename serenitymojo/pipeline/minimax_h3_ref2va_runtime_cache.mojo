# Build Ref2VA's transformer-resident INT8 stores in a GPU-only process.
# This is intentionally separate from generation: the VAE, vision tower and
# long-sequence activations must never coexist with a full 18-19 GiB cache build.

from std.sys import argv
from max.gpu.host import DeviceContext

from serenitymojo.io.ffi import sys_system
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.minimax_h3_dit import minimax_h3_released_config
from serenitymojo.models.dit.minimax_h3_fp8_resident import (
    MINIMAX_H3_RESIDENT_INT8,
    MINIMAX_H3_RESIDENT_INT8_W8A8,
    minimax_h3_build_resident_fp8,
)
from serenitymojo.models.dit.minimax_h3_runtime_cache import (
    save_minimax_h3_resident_cache,
)
from serenitymojo.pipeline.gpu_free_vram_guard import require_free_vram


comptime H3_ROOT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/Ref2VA"
comptime TRANSFORMER_DIR = H3_ROOT + "/transformer"
comptime TRANSFORMER_INDEX = TRANSFORMER_DIR + "/model.safetensors.index.json"
comptime RUNTIME_CACHE_DIR = H3_ROOT + "/serenity_runtime_cache_v1"


def main() raises:
    var args = argv()
    if len(args) != 2:
        print("usage: minimax_h3_ref2va_runtime_cache groupwise|w8a8")
        return

    var mode = String(args[1])
    var scheme = MINIMAX_H3_RESIDENT_INT8
    var blocks = 48
    var path = String(RUNTIME_CACHE_DIR) \
        + String("/resident_groupwise_q16_o64_fc132_fc264_blocks_48.safetensors")
    if mode == String("w8a8"):
        scheme = MINIMAX_H3_RESIDENT_INT8_W8A8
        blocks = 50
        path = String(RUNTIME_CACHE_DIR) \
            + String("/resident_w8a8_row_blocks_50.safetensors")
    elif mode != String("groupwise"):
        raise Error("expected groupwise or w8a8")

    var config = minimax_h3_released_config()
    config.validate()
    var shards = ShardedSafeTensors.open(String(TRANSFORMER_DIR))
    _ = sys_system(String("mkdir -p '") + String(RUNTIME_CACHE_DIR) + "'")
    require_free_vram(
        20000, String(RUNTIME_CACHE_DIR) + String("/.gpu_guard_cache"),
        String("H3"), String("minimax_h3_ref2va_runtime_cache"),
    )
    var ctx = DeviceContext()
    print("building Ref2VA", mode, "resident cache:", blocks, "blocks")
    var store = minimax_h3_build_resident_fp8(
        shards, config, ctx, blocks, scheme=scheme
    )
    save_minimax_h3_resident_cache(
        store, String(TRANSFORMER_INDEX), path, ctx
    )
    print("saved", path)
