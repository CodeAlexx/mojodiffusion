# GPU gate for MiniMax-H3's opt-in High step cache.

from std.collections import List, Optional
from max.gpu.host import DeviceContext
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

    # ── Adaptive policy: accumulated drift budget ──────────────────────────
    # A per-decision-acceptable diff (0.10 < 0.12) must be REFUSED once the
    # accumulated drift would exceed the 0.24 budget, and the refusal is a
    # full refresh that resets the accumulator.
    decision_cache.middle_residual = Optional(
        minimax_h3_cache_quantize_activation(first_for_snapshot, ctx)
    )
    decision_cache.accumulated_diff = Float32(0.20)
    var after_drift = List[Float32](capacity=128)
    for i in range(128):
        # Stored previous residual is 0.25/row; 0.275 gives rel-L1 diff 0.10.
        after_drift.append(before[i] + Float32(0.275))
    if minimax_h3_cache_should_reuse(5, before, after_drift, decision_cache):
        raise Error("MiniMax-H3 step cache reused past the drift budget")
    if decision_cache.accumulated_diff != Float32(0.0):
        raise Error("MiniMax-H3 step cache refresh did not reset the budget")
    # After the refresh the same residual is stable again and reuses.
    decision_cache.middle_residual = Optional(
        minimax_h3_cache_quantize_activation(first_for_snapshot, ctx)
    )
    if not minimax_h3_cache_should_reuse(
        6, before, after_drift, decision_cache
    ):
        raise Error("MiniMax-H3 step cache did not reuse after budget reset")

    # ── Audio veto: tighter band blocks reuse even when video passes ──────
    # Video diff 0 (stable) but audio diff 0.10 — above the 0.06 audio
    # threshold, below the 0.12 video one. Reuse must be REFUSED.
    decision_cache.middle_residual = Optional(
        minimax_h3_cache_quantize_activation(first_for_snapshot, ctx)
    )
    var audio_before = List[Float32](capacity=16)
    var audio_after_stable = List[Float32](capacity=16)
    for i in range(16):
        audio_before.append(Float32(i) * Float32(0.01))
        audio_after_stable.append(audio_before[i] + Float32(0.25))
    # Seed the previous audio probe via one refused refresh call.
    _ = minimax_h3_cache_should_reuse(
        7, before, after_drift, decision_cache,
        audio_before, audio_after_stable,
    )
    decision_cache.middle_residual = Optional(
        minimax_h3_cache_quantize_activation(first_for_snapshot, ctx)
    )
    var audio_after_moved = List[Float32](capacity=16)
    for i in range(16):
        # residual 0.275 vs previous 0.25 -> audio rel-L1 diff 0.10
        audio_after_moved.append(audio_before[i] + Float32(0.275))
    if minimax_h3_cache_should_reuse(
        8, before, after_drift, decision_cache,
        audio_before, audio_after_moved,
    ):
        raise Error("MiniMax-H3 step cache ignored the audio veto")
    if decision_cache.last_audio_residual_diff < Float32(0.09) \
            or decision_cache.last_audio_residual_diff > Float32(0.11):
        raise Error("MiniMax-H3 audio diff not measured as expected")
    # Stable audio (diff 0) with stable video must still reuse.
    decision_cache.middle_residual = Optional(
        minimax_h3_cache_quantize_activation(first_for_snapshot, ctx)
    )
    if not minimax_h3_cache_should_reuse(
        9, before, after_drift, decision_cache,
        audio_before, audio_after_moved,
    ):
        raise Error("MiniMax-H3 step cache refused with stable audio")

    # ── Adaptive policy: exact tail ────────────────────────────────────────
    # With a 10-evaluation schedule the final three evaluations never reuse,
    # and entering the tail disables the cache (no further snapshot cost).
    var tail_cache = MiniMaxH3StepCache(True, 10)
    tail_cache.middle_residual = Optional(
        minimax_h3_cache_quantize_activation(first_for_snapshot, ctx)
    )
    if minimax_h3_cache_should_reuse(7, before, after, tail_cache):
        raise Error("MiniMax-H3 step cache reused inside the exact tail")
    if tail_cache.enabled:
        raise Error("MiniMax-H3 step cache stayed enabled inside the tail")
    print(
        "PASS: MiniMax-H3 step cache GPU reconstruction + adaptive policy"
        " (budget + exact tail)"
    )
