# SKEPTIC PROBE — in-process cuMemGetInfo_v2 device-memory measurement for
# minimax_h3_modcache.minimax_h3_build_modulation_cache's streaming claim.
#
# The shipped probe's header says runtime memory measurement "needs -Xlinker
# -lcuda, which plain `mojo run -I .` cannot supply, so it was dropped." That
# is true of BARE `mojo run`, but NOT a fundamental blocker: this exact
# link-flag combination is already a working, precedented pattern elsewhere
# in this repo (models/flux/parity/flux_offload_mem_smoke.mojo) via
# offload/vmm_cuda.cu_mem_get_info(). This probe reuses that precedent
# directly against minimax_h3_build_modulation_cache, giving EXACT
# before/after free-byte deltas at each phase instead of external nvidia-smi
# polling's coarse/noisy time-window sampling.
#
# Build (NOT plain `mojo run` — needs the link flags):
#   cd /home/alex/mojodiffusion
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/models/dit/parity/minimax_h3_modcache_inprocess_mem_skeptic_probe.mojo \
#       -o /tmp/.../modcache_inprocess_mem
#   /tmp/.../modcache_inprocess_mem

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.offload.vmm_cuda import cu_mem_get_info

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
comptime CKPT = "/tmp/minimax_h3_inprocess_mem_skeptic.safetensors"


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
    save_safetensors(names, tensors, String(CKPT), ctx)


def _mib(bytes: Int) -> Float64:
    return Float64(bytes) / Float64(1024 * 1024)


def _report(ctx: DeviceContext, label: String) raises:
    ctx.synchronize()
    var info = cu_mem_get_info()
    print("MARK", label, " free_MiB=", _mib(info.free_bytes),
          " used_MiB=", _mib(info.used_bytes()))


def main() raises:
    var ctx = DeviceContext()
    var real = minimax_h3_released_config()
    print("one_block_weight_MiB=",
          _mib(real.adaln_out_features * real.time_embed_dim * 2))
    _report(ctx, String("startup"))

    var cfg8 = _real_prefix_config(8)
    _write_checkpoint(cfg8, ctx)
    _report(ctx, String("after_write_fixture_8blocks"))

    var shards = ShardedSafeTensors.open(String(CKPT))

    var cfg4 = _real_prefix_config(4)
    var temb4 = zeros_device([1, real.time_embed_dim], STDtype.F32, ctx)
    _report(ctx, String("before_build_N4"))
    var cache4 = minimax_h3_build_modulation_cache(shards, temb4, cfg4, ctx)
    _report(ctx, String("after_build_N4"))
    _ = cache4^
    _report(ctx, String("after_drop_cache4"))

    var temb8 = zeros_device([1, real.time_embed_dim], STDtype.F32, ctx)
    _report(ctx, String("before_build_N8"))
    var cache8 = minimax_h3_build_modulation_cache(shards, temb8, cfg8, ctx)
    _report(ctx, String("after_build_N8"))
    _ = cache8^
    _report(ctx, String("after_drop_cache8"))
    print("DONE")
