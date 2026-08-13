# serenitymojo/models/dit/parity/minimax_h3_transform_guard_probe.mojo
#
# SKEPTIC PROBE — proves `minimax_h3_require_transformed_weights`
# (models/dit/minimax_h3_dit.mojo) actually closes the raw-vs-transformed
# hole: `minimax_h3_check_block_weights` validates SHAPE only, and a row
# permutation (the qkv de-interleave / fc1 swap `minimax_h3_load_block_device`
# applies) never changes shape, so shape-only preflight cannot tell a raw
# checkpoint tensor from a correctly pre-transformed one. This file builds
# BOTH kinds of weight Dict from the SAME real-geometry on-disk fixture and
# checks that `minimax_h3_block_forward` accepts one and rejects the other —
# "a guard that has never been observed to fire is not a guard."
#
# Four checks:
#   [1] unit:     minimax_h3_require_transformed_weights on a Dict WITH both
#                 markers -> must NOT raise.
#   [2] unit:     minimax_h3_require_transformed_weights on a Dict WITHOUT
#                 the markers -> must raise.
#   [3] positive: minimax_h3_load_block_device (real loader, stamps markers)
#                 -> minimax_h3_block_forward runs to completion.
#   [4] negative: weights built STRAIGHT FROM ShardedSafeTensors via
#                 Tensor.from_view, NO loader call, NO markers -> passed to
#                 minimax_h3_block_forward -> must raise the SAME guard, not
#                 merely "some" exception.
#
# Fixture: a real-released-geometry (hidden 5376, heads 56, head_dim 128,
# ffn 14336), 1-block, RANDOM-BYTES synthetic checkpoint — same shape/dtype
# contract minimax_h3_check_block_weights enforces, same generator script
# (make_h3_block_fixture.py) the loader's own memory-probe skeptics use.
# Regenerate with:
#   python3 <scratchpad>/make_h3_block_fixture.py <dir> 1
#
# Run (package-relative imports need -I .; sdpa_flash_infer_fwd needs the
# cuDNN shim explicitly linked — bare `mojo run -I .` fails with
# "JIT session error: Symbols not found: [ flame_cudnn_sdpa_bf16 ]"):
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/dit/parity/minimax_h3_transform_guard_probe.mojo

from std.collections import Dict, List
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

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
    minimax_h3_require_transformed_weights,
    minimax_h3_block_forward,
)
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_load_block_device,
)

comptime FIXTURE_DIR = "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/scratchpad/h3_guard_fixture"
comptime S = 4
comptime LAYER = 0


def main() raises:
    var ctx = DeviceContext()
    var config = minimax_h3_released_config()
    config.validate()

    var st = ShardedSafeTensors.open(String(FIXTURE_DIR))
    print("fixture opened:", st.num_shards(), "shard(s),", st.num_tensors(), "tensors")

    var n_fail = 0

    # ── [1]/[2] direct unit checks on minimax_h3_require_transformed_weights ──
    print("")
    print("[1] marked Dict -> minimax_h3_require_transformed_weights must NOT raise")
    var marked = Dict[String, ArcPointer[Tensor]]()
    marked["__h3_qkv_deinterleaved__"] = ArcPointer(randn([1], 1, STDtype.BF16, ctx))
    marked["__h3_fc1_swapped__"] = ArcPointer(randn([1], 2, STDtype.BF16, ctx))
    try:
        minimax_h3_require_transformed_weights(marked, LAYER)
        print("  ok   accepted a marked Dict")
    except e:
        print("  FAIL raised on a correctly marked Dict:", e)
        n_fail += 1

    print("")
    print("[2] unmarked Dict -> minimax_h3_require_transformed_weights MUST raise")
    var unmarked = Dict[String, ArcPointer[Tensor]]()
    unmarked["attn.qkv_proj.weight"] = ArcPointer(randn([1], 3, STDtype.BF16, ctx))
    var raised2 = False
    try:
        minimax_h3_require_transformed_weights(unmarked, LAYER)
    except e:
        raised2 = True
        print("  ok   raised:", e)
    if not raised2:
        print("  FAIL did not raise on an unmarked Dict — the guard is not guarding")
        n_fail += 1

    # ── [3] positive: real loader -> block forward runs ──────────────────────
    print("")
    print("[3] minimax_h3_load_block_device -> minimax_h3_block_forward (positive path)")
    var loader_weights = minimax_h3_load_block_device(st, LAYER, config, ctx)
    var inv_freq3 = minimax_h3_rope_inv_freq(config.rope_inv_freq_len)
    var position_ids3 = List[Float64]()
    for r in range(S):
        position_ids3.append(Float64(r))
        position_ids3.append(Float64(0))
        position_ids3.append(Float64(0))
    var rope3 = minimax_h3_rope_table(position_ids3, S, inv_freq3)
    var rotary_dim3 = rope3.rotary_dim
    var cos3_shape: List[Int] = [S, rotary_dim3]
    var sin3_shape: List[Int] = [S, rotary_dim3]
    var cos3 = Tensor.from_host(rope3.cos, cos3_shape^, STDtype.F32, ctx)
    var sin3 = Tensor.from_host(rope3.sin, sin3_shape^, STDtype.F32, ctx)
    var x3 = randn([1, S, config.hidden_size], 4, STDtype.BF16, ctx)
    var mod3_shape: List[Int] = [3, 6 * config.hidden_size]
    var mod3 = randn(mod3_shape^, 5, STDtype.F32, ctx)
    var adaln3 = List[Int]()
    for _ in range(S):
        adaln3.append(0)
    try:
        var out3 = minimax_h3_block_forward[S](
            x3, loader_weights, LAYER, config, mod3, adaln3, cos3, sin3, rotary_dim3, ctx,
        )
        var h3 = out3.to_host(ctx)
        print("  ok   ran to completion, out numel", len(h3))
    except e:
        print("  FAIL positive path raised:", e)
        n_fail += 1

    # ── [4] negative: raw ShardedSafeTensors tensors, NO loader, NO markers ──
    print("")
    print("[4] raw ShardedSafeTensors (no loader) -> minimax_h3_block_forward MUST raise")
    var raw_weights = Dict[String, ArcPointer[Tensor]]()
    var names = minimax_h3_block_tensor_names(LAYER)
    for i in range(len(names)):
        var n = names[i]
        raw_weights[n] = ArcPointer(Tensor.from_view(st.tensor_view(n), ctx))
    print(
        "  built", len(raw_weights), "raw tensors directly via Tensor.from_view",
        "(no minimax_h3_load_block_device call, so qkv is still per-head"
        " interleaved and fc1 is still [gate;value])",
    )
    var inv_freq4 = minimax_h3_rope_inv_freq(config.rope_inv_freq_len)
    var position_ids4 = List[Float64]()
    for r in range(S):
        position_ids4.append(Float64(r))
        position_ids4.append(Float64(0))
        position_ids4.append(Float64(0))
    var rope4 = minimax_h3_rope_table(position_ids4, S, inv_freq4)
    var rotary_dim4 = rope4.rotary_dim
    var cos4_shape: List[Int] = [S, rotary_dim4]
    var sin4_shape: List[Int] = [S, rotary_dim4]
    var cos4 = Tensor.from_host(rope4.cos, cos4_shape^, STDtype.F32, ctx)
    var sin4 = Tensor.from_host(rope4.sin, sin4_shape^, STDtype.F32, ctx)
    var x4 = randn([1, S, config.hidden_size], 6, STDtype.BF16, ctx)
    var mod4_shape: List[Int] = [3, 6 * config.hidden_size]
    var mod4 = randn(mod4_shape^, 7, STDtype.F32, ctx)
    var adaln4 = List[Int]()
    for _ in range(S):
        adaln4.append(0)
    var raised4 = False
    try:
        var out4 = minimax_h3_block_forward[S](
            x4, raw_weights, LAYER, config, mod4, adaln4, cos4, sin4, rotary_dim4, ctx,
        )
        _ = out4.to_host(ctx)
    except e:
        raised4 = True
        print("  ok   raised:", e)
    if not raised4:
        print("  FAIL raw ShardedSafeTensors weights were silently accepted")
        n_fail += 1

    print("")
    if n_fail != 0:
        raise Error(String("minimax_h3_transform_guard_probe: ") + String(n_fail) + " check(s) FAILED")
    print("ALL CHECKS PASS")
