# serenitymojo/models/dit/parity/minimax_h3_stack_real2layer_probe.mojo
#
# SKEPTIC PROBE for `models/dit/minimax_h3_stack.mojo::minimax_h3_run_stack`.
# NOT a numeric parity gate (no oracle here — see
# models/dit/parity/minimax_h3_block_device_gate.mojo for the per-block
# oracle contract this composes). What this proves, against the REAL
# MiniMax-H3 checkpoint (7/13 shards on disk, layers 0-1 fully present in
# shard 1 as of this writing):
#
#   [1] running the stack over 1 real streamed layer vs 2 real streamed
#       layers does NOT roughly double device memory — the flat-vs-
#       accumulating check team-lead asked for measured, not assumed. This
#       process prints MARK/sleep checkpoints for an EXTERNAL `nvidia-smi`
#       poller (same technique as minimax_h3_loader_device_memory_probe.mojo)
#       rather than reading memory from inside Mojo.
#   [2] both runs produce finite (NaN-free) output of the right shape.
#   [3] the transform-contract guard still fires on REAL checkpoint bytes
#       (not just the synthetic fixture minimax_h3_transform_guard_probe.mojo
#       used) when weights are pulled straight from ShardedSafeTensors,
#       bypassing minimax_h3_load_block_device.
#
# hidden/modulation/rope inputs are SYNTHETIC (random, right shape) — wiring
# the real frontend embed + real modulation cache end to end is a separate,
# UNTESTED integration surface (see this file's report). Only the block
# WEIGHTS are real bytes streamed from the actual checkpoint.
#
# Run (package-relative imports need -I .; sdpa_flash_infer_fwd needs the
# cuDNN shim explicitly linked — bare `mojo run -I .` fails with
# "JIT session error: Symbols not found: [ flame_cudnn_sdpa_bf16 ]"):
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/dit/parity/minimax_h3_stack_real2layer_probe.mojo

from std.collections import Dict, List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.random import randn
from serenitymojo.models.minimax_h3.dit_frontend import (
    minimax_h3_rope_inv_freq,
    minimax_h3_rope_table,
)
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    minimax_h3_released_config,
    minimax_h3_block_tensor_names,
    minimax_h3_block_forward,
)
from serenitymojo.models.dit.minimax_h3_modcache import MiniMaxH3ModCache
from serenitymojo.models.dit.minimax_h3_stack import minimax_h3_run_stack

# `ShardedSafeTensors.open` eagerly opens EVERY shard the index references
# (io/sharded.mojo:419-424), not just the tensors a caller ends up reading —
# so pointing this at the real checkpoint DIRECTORY (7/13 shards present)
# fails at open() with "failed to open: ...model-00002-of-00013.safetensors",
# before streaming a single real layer, even though layers 0-1 are 100%
# present in shard 1 alone (verified against the real index.json's
# weight_map, not guessed). Fix (h3-block-gate, cleaner than this file's
# earlier symlink+hand-trimmed-index workaround): point directly at the
# shard-1 FILE — `_looks_safetensors_file` detects a single-file path and
# takes ShardedSafeTensors' documented single-file fallback, which never
# touches the index or any other shard. No scratch state, works from any
# sandbox since it's just the real absolute path.
comptime CHECKPOINT_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/model-00001-of-00013.safetensors"
comptime S = 8


def _nan_free(t: Tensor, ctx: DeviceContext) raises -> Int:
    var host = t.to_host(ctx)
    var n_bad = 0
    for i in range(len(host)):
        if host[i] != host[i]:
            n_bad += 1
    return n_bad


def main() raises:
    var ctx = DeviceContext()
    var config = minimax_h3_released_config()
    config.validate()

    var st = ShardedSafeTensors.open(String(CHECKPOINT_DIR))
    print("MARK startup shards=", st.num_shards(), "tensors=", st.num_tensors())
    ctx.synchronize()
    sleep(1.0)

    # ── synthetic modcache: 2 layers' worth, 1 distinct timestep ──
    var distinct_ts = 1
    var block_mod = List[ArcPointer[Tensor]]()
    for i in range(2):
        var shape: List[Int] = [distinct_ts * 3, 6 * config.hidden_size]
        block_mod.append(ArcPointer(randn(shape^, UInt64(100 + i), STDtype.BF16, ctx)))
    var final_shape: List[Int] = [distinct_ts, 2 * config.hidden_size]
    var final_mod = ArcPointer(randn(final_shape^, 999, STDtype.BF16, ctx))
    var modcache = MiniMaxH3ModCache(block_mod^, final_mod^, distinct_ts)

    var adaln_indices = List[Int]()
    for _ in range(S):
        adaln_indices.append(0)

    var inv_freq = minimax_h3_rope_inv_freq(config.rope_inv_freq_len)
    var position_ids = List[Float64]()
    for r in range(S):
        position_ids.append(Float64(r))
        position_ids.append(Float64(0))
        position_ids.append(Float64(0))
    var rope = minimax_h3_rope_table(position_ids, S, inv_freq)
    var cos_shape: List[Int] = [S, rope.rotary_dim]
    var sin_shape: List[Int] = [S, rope.rotary_dim]
    var cos_t = Tensor.from_host(rope.cos, cos_shape^, STDtype.F32, ctx)
    var sin_t = Tensor.from_host(rope.sin, sin_shape^, STDtype.F32, ctx)

    # ── [1] 1 real streamed layer ──
    var x1_shape: List[Int] = [S, config.hidden_size]
    var hidden1 = randn(x1_shape^, 1, STDtype.BF16, ctx)
    ctx.synchronize()
    print("MARK before_1layer")
    sleep(1.5)
    var out1 = minimax_h3_run_stack[S](
        hidden1^, st, modcache, adaln_indices, cos_t, sin_t, rope.rotary_dim,
        config, ctx, 1,
    )
    ctx.synchronize()
    print("MARK after_1layer")
    sleep(1.5)
    var bad1 = _nan_free(out1, ctx)
    print("1-layer: shape", out1.shape()[0], out1.shape()[1], "nan_count", bad1)

    # ── [2] 2 real streamed layers (fresh call, same process) ──
    var x2_shape: List[Int] = [S, config.hidden_size]
    var hidden2 = randn(x2_shape^, 2, STDtype.BF16, ctx)
    ctx.synchronize()
    print("MARK before_2layer")
    sleep(1.5)
    var out2 = minimax_h3_run_stack[S](
        hidden2^, st, modcache, adaln_indices, cos_t, sin_t, rope.rotary_dim,
        config, ctx, 2,
    )
    ctx.synchronize()
    print("MARK after_2layer")
    sleep(1.5)
    var bad2 = _nan_free(out2, ctx)
    print("2-layer: shape", out2.shape()[0], out2.shape()[1], "nan_count", bad2)

    var n_fail = 0
    if bad1 != 0:
        print("FAIL: NaN in 1-layer output")
        n_fail += 1
    if bad2 != 0:
        print("FAIL: NaN in 2-layer output")
        n_fail += 1
    if out1.shape()[0] != S or out1.shape()[1] != config.hidden_size:
        print("FAIL: 1-layer output shape wrong")
        n_fail += 1
    if out2.shape()[0] != S or out2.shape()[1] != config.hidden_size:
        print("FAIL: 2-layer output shape wrong")
        n_fail += 1

    # ── [3] guard fires on REAL checkpoint bytes, no loader ──
    print("")
    print("[3] raw ShardedSafeTensors (real bytes, no loader) -> guard must raise")
    var raw_weights = Dict[String, ArcPointer[Tensor]]()
    var names = minimax_h3_block_tensor_names(0)
    for i in range(len(names)):
        var n = names[i]
        raw_weights[n] = ArcPointer(Tensor.from_view(st.tensor_view(n), ctx))
    var mod0_shape: List[Int] = [3, 6 * config.hidden_size]
    var mod0 = randn(mod0_shape^, 42, STDtype.BF16, ctx)
    var x3_shape: List[Int] = [1, S, config.hidden_size]
    var x3 = randn(x3_shape^, 3, STDtype.BF16, ctx)
    var raised = False
    try:
        var out3 = minimax_h3_block_forward[S](
            x3, raw_weights, 0, config, mod0, adaln_indices, cos_t, sin_t,
            rope.rotary_dim, ctx,
        )
        _ = out3.to_host(ctx)
    except e:
        raised = True
        print("  ok   raised:", e)
    if not raised:
        print("  FAIL guard did not fire on real raw checkpoint bytes")
        n_fail += 1

    print("")
    if n_fail != 0:
        raise Error(String("minimax_h3_stack_real2layer_probe: ") + String(n_fail) + " check(s) FAILED")
    print("ALL CHECKS PASS")
