# SKEPTIC PROBE — real device-memory measurement for
# minimax_h3_modcache.minimax_h3_build_modulation_cache's streaming claim.
#
# The shipped probe (minimax_h3_modcache_probe.mojo) explicitly does NOT
# measure device memory at runtime — its header says cuMemGetInfo_v2 needs
# -Xlinker -lcuda, which plain `mojo run -I .` cannot supply, so it falls
# back to an ARITHMETIC argument ("w/b are loop-local, Mojo frees them at
# end of scope") instead of a measurement.
#
# This probe sidesteps the FFI/linking problem entirely: no cuMemGetInfo
# call from inside the Mojo binary. Instead it prints `MARK <label>`
# timestamped lines around each build phase, with `ctx.synchronize()` +
# `sleep()` settle windows, while an EXTERNAL `nvidia-smi --query-gpu=
# memory.used -lms 100` loop (driven by the wrapper shell command, not by
# this file) samples real device memory concurrently. Correlating the two
# logs answers the actual question: does peak device memory during
# `minimax_h3_build_modulation_cache` scale with num_layers (genuine
# streaming — never more than ~1 block's weight resident at a time) or with
# num_layers * block_weight_size (the claim is false and something is
# retaining every block)?
#
# Real production dimensions (hidden_size=5376, time_embed_dim=2688,
# adaln_out_features=96768 -> ~496 MiB bf16 weight per block), N=4 then N=8
# blocks in the SAME process (so the SAME caching allocator instance sees
# both), each build followed by an explicit drop + settle window.
#
#   pixi run mojo build -I . serenitymojo/models/dit/parity/minimax_h3_modcache_device_memory_skeptic_probe.mojo -o /tmp/.../modcache_mem_probe
#   (run under external nvidia-smi polling — see the wrapper command in the report)

from std.collections import List
from std.time import sleep
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.tensor_algebra import zeros_device

from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    minimax_h3_released_config,
)
from serenitymojo.models.dit.minimax_h3_modcache import (
    minimax_h3_block_adaln_weight_key,
    minimax_h3_block_adaln_bias_key,
    minimax_h3_build_modulation_cache,
    MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY,
    MINIMAX_H3_FINAL_ADALN_BIAS_KEY,
)

comptime TArc = ArcPointer[Tensor]
comptime CKPT8 = "/tmp/minimax_h3_devmem_skeptic_8block.safetensors"


def _real_prefix_config(blocks: Int) raises -> MiniMaxH3DiTConfig:
    var real = minimax_h3_released_config()
    return MiniMaxH3DiTConfig(
        real.hidden_size, blocks, real.token_refiner_num_layers,
        real.num_attention_heads, real.attention_head_dim, real.ffn_hidden_size,
        real.latents_dim, real.audio_latents_dim, real.text_dim,
        real.timestep_input_dim, real.time_embed_dim, real.adaln_out_features,
        real.final_adaln_out_features, real.rope_inv_freq_len,
        real.norm_eps, real.qk_norm_eps, real.final_norm_eps,
    )


def _write_checkpoint(cfg: MiniMaxH3DiTConfig, ctx: DeviceContext) raises:
    """Real-dimension, zero-valued (shape/dtype smoke only, matching the
    shipped probe's own phase-2 approach) checkpoint with `cfg.num_layers`
    blocks (~496 MiB bf16 weight each)."""
    var names = List[String]()
    var tensors = List[TArc]()
    for layer in range(cfg.num_layers):
        names.append(minimax_h3_block_adaln_weight_key(layer))
        tensors.append(TArc(zeros_device(
            [cfg.adaln_out_features, cfg.time_embed_dim], STDtype.BF16, ctx
        )))
        names.append(minimax_h3_block_adaln_bias_key(layer))
        tensors.append(TArc(zeros_device([cfg.adaln_out_features], STDtype.BF16, ctx)))
    names.append(MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY)
    tensors.append(TArc(zeros_device(
        [cfg.final_adaln_out_features, cfg.time_embed_dim], STDtype.BF16, ctx
    )))
    names.append(MINIMAX_H3_FINAL_ADALN_BIAS_KEY)
    tensors.append(TArc(zeros_device([cfg.final_adaln_out_features], STDtype.BF16, ctx)))
    save_safetensors(names, tensors, String(CKPT8), ctx)
    # `tensors` (the fixture-writer's own N zero blocks) falls out of scope
    # at return — this frees the FIXTURE side before the module under test
    # ever runs, so what nvidia-smi sees during the build phases below is
    # ONLY minimax_h3_build_modulation_cache's own footprint, not a
    # confound from how this probe manufactured its input file.


def main() raises:
    var ctx = DeviceContext()
    var real = minimax_h3_released_config()
    print("MARK startup one_block_weight_MiB=",
          Float64(real.adaln_out_features * real.time_embed_dim * 2) / Float64(1024 * 1024))
    ctx.synchronize()
    sleep(1.5)
    print("MARK baseline_after_ctx")
    sleep(1.5)

    var cfg8 = _real_prefix_config(8)
    print("MARK before_write_fixture_8blocks (expect a transient ~", 8 * 496, "MiB bump while writing, THEN drop)")
    _write_checkpoint(cfg8, ctx)
    ctx.synchronize()
    sleep(0.5)
    print("MARK after_write_fixture_8blocks_dropped")
    sleep(1.5)

    var shards = ShardedSafeTensors.open(String(CKPT8))

    # ── Build #1: N=4 (a PREFIX of the 8-block file) ──
    var cfg4 = _real_prefix_config(4)
    var temb4 = zeros_device([1, real.time_embed_dim], STDtype.F32, ctx)
    print("MARK before_build_N4")
    sleep(0.5)
    var cache4 = minimax_h3_build_modulation_cache(shards, temb4, cfg4, ctx)
    ctx.synchronize()
    sleep(0.5)
    print("MARK after_build_N4 cache_bytes=", cache4.total_bytes())
    sleep(1.5)
    _ = cache4^
    ctx.synchronize()
    sleep(0.5)
    print("MARK after_drop_cache4")
    sleep(1.5)

    # ── Build #2: N=8 (2x the block count — if streaming is real, PEAK
    # device memory during this build should look like the SAME shape as
    # build #1's peak, not double; if the caching allocator is masking
    # retention from build #1, or if `w`/`b` aren't actually being freed
    # per-iteration, this build's peak/plateau will sit measurably higher
    # than build #1's, not just "a bit more from the bigger final cache". ──
    var temb8 = zeros_device([1, real.time_embed_dim], STDtype.F32, ctx)
    print("MARK before_build_N8")
    sleep(0.5)
    var cache8 = minimax_h3_build_modulation_cache(shards, temb8, cfg8, ctx)
    ctx.synchronize()
    sleep(0.5)
    print("MARK after_build_N8 cache_bytes=", cache8.total_bytes())
    sleep(1.5)
    _ = cache8^
    ctx.synchronize()
    sleep(0.5)
    print("MARK after_drop_cache8")
    sleep(1.5)
    print("DONE")
