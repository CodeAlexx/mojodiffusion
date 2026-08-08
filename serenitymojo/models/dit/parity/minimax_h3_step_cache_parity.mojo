# GPU gate for MiniMax-H3's opt-in High step cache.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.minimax_h3_step_cache import (
    MiniMaxH3StepCache,
    minimax_h3_cache_apply_residual_inplace,
    minimax_h3_cache_quantize_activation,
    minimax_h3_cache_should_reuse,
    minimax_h3_cache_store_middle_residual,
)


def main() raises:
    var ctx = DeviceContext()
    comptime rows = 8
    comptime hidden = 128

    # Cache final-first, then reconstruct final from a fresh copy of first.
    var first_for_snapshot = randn([rows, hidden], 20260807, STDtype.BF16, ctx)
    var first_for_reuse = randn([rows, hidden], 20260807, STDtype.BF16, ctx)
    var final_hidden = randn([rows, hidden], 20260808, STDtype.BF16, ctx)
    var expected = final_hidden.to_host(ctx)
    var first_snapshot = minimax_h3_cache_quantize_activation(
        first_for_snapshot, ctx
    )
    var cache = MiniMaxH3StepCache(True)
    minimax_h3_cache_store_middle_residual(
        final_hidden, first_snapshot, cache, ctx
    )
    minimax_h3_cache_apply_residual_inplace(first_for_reuse, cache, ctx)
    var actual = first_for_reuse.to_host(ctx)

    var dot = Float64(0.0)
    var ne = Float64(0.0)
    var na = Float64(0.0)
    var max_abs = Float32(0.0)
    for i in range(len(actual)):
        var d = actual[i] - expected[i]
        var ad = d if d >= 0.0 else -d
        if ad > max_abs:
            max_abs = ad
        dot += Float64(actual[i]) * Float64(expected[i])
        ne += Float64(expected[i]) * Float64(expected[i])
        na += Float64(actual[i]) * Float64(actual[i])
    var cosine = dot / (sqrt(ne) * sqrt(na) + 1.0e-30)
    print("step-cache reconstruction: cosine=", cosine, " max_abs=", max_abs)
    if cosine < 0.9999:
        raise Error("MiniMax-H3 step-cache reconstruction parity failed")

    # Decision policy: four warmup refreshes, then a stable residual reuses.
    var before = List[Float32](capacity=128)
    var after = List[Float32](capacity=128)
    for i in range(128):
        before.append(Float32(i) * Float32(0.01))
        after.append(before[i] + Float32(0.25))
    var decision_cache = MiniMaxH3StepCache(True)
    for step in range(4):
        if minimax_h3_cache_should_reuse(
            step, before, after, decision_cache
        ):
            raise Error("MiniMax-H3 step cache reused during warmup")
    # The real denoiser stores a middle residual after every full refresh.
    decision_cache.middle_residual = Optional(
        minimax_h3_cache_quantize_activation(first_for_snapshot, ctx)
    )
    if not minimax_h3_cache_should_reuse(
        4, before, after, decision_cache
    ):
        raise Error("MiniMax-H3 step cache did not reuse a stable residual")
    print("PASS: MiniMax-H3 High step cache GPU reconstruction + policy")
