# Isolated integration gate for the compile-time-dark MiniMax-H3 adaptive
# SM120 attention dispatch.  This is deliberately not a product entrypoint.
#
# The released H3 geometry is exercised at the real 56x128 head shape, but the
# Q/K/V fixture is synthetic: producing checkpoint-derived Q/K/V also requires
# the pipeline's packed text/audio/video frontend, AdaLN rows, rotary tables,
# and transformed block-weight loader.  This gate therefore proves dispatch,
# scratch ownership, dispatch-vs-direct-P6 identity, routing, and timing only;
# cU-DNN approximation metrics are informational and it must not be cited as
# a released-checkpoint quality result.
#
# Build (candidate branch enabled):
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#     -D H3_ADAPTIVE_SM120=1 -D H3_ADAPTIVE_SM120_FAST_ROUTE=1 \
#     -D H3_ADAPTIVE_SM120_TAU_X100=150 -I . -I vendor/mojo-libs \
#     serenitymojo/models/dit/parity/minimax_h3_adaptive_sm120_fixture_dispatch_gate.mojo \
#     -o /tmp/minimax_h3_adaptive_sm120_fixture_dispatch_gate \
#     -Xlinker -Loutput/lib -Xlinker -lserenity_minimax_h3_int8 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa -Xlinker -lcudnn \
#     -Xlinker -lcuda -Xlinker -lm

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import isfinite, sqrt
from std.time import perf_counter_ns

from serenitymojo.models.dit.minimax_h3_dit import (
    MINIMAX_H3_ATTN_ADAPTIVE_SM120,
    MINIMAX_H3_ATTN_COMFY_KITCHEN_INT8,
    MINIMAX_H3_ATTN_CUDNN,
    _minimax_h3_attention_dispatch,
    minimax_h3_adaptive_sm120_backend_name,
    minimax_h3_adaptive_sm120_tau,
)
from serenitymojo.ops.comfy_kitchen_attention import (
    ComfyKitchenAttentionScratch,
)
from serenitymojo.ops.adaptive_block_attention_sm120_bf16 import (
    adaptive_block_attention_sm120_bf16,
    adaptive_block_attention_sm120_compare_to_host,
    adaptive_block_attention_sm120_route_counts_to_host,
)
from serenitymojo.ops.adaptive_block_attention_tiled_bf16 import (
    AdaptiveBlockAttentionTiledScratch,
)
from serenitymojo.ops.random import randn
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor


comptime B = 1
comptime S = 4096
comptime S_REJECT = 1024
comptime H = 56
comptime D = 128
comptime SINK_TOKENS = 1
comptime COMPARE_SCRATCH_TOKENS = 46848
comptime S_REJECT_ERROR = (
    "MiniMax-H3 adaptive SM120 requires S>=4096 for H=56"
)


def _ms(start_ns: Int, end_ns: Int) -> Float64:
    return Float64(end_ns - start_ns) / 1_000_000.0


def _host_metrics(
    exact: List[Float32],
    candidate: List[Float32],
    repeated: List[Float32],
) -> List[Float64]:
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float64(0.0)
    var nonfinite = 0
    var repeat_mismatch = 0
    for i in range(len(exact)):
        var a = Float64(exact[i])
        var b = Float64(candidate[i])
        if not isfinite(a) or not isfinite(b):
            nonfinite += 1
        dot += a * b
        na += a * a
        nb += b * b
        max_abs = max(max_abs, abs(a - b))
        if candidate[i] != repeated[i]:
            repeat_mismatch += 1
    var result = List[Float64]()
    result.append(dot / sqrt(na * nb))
    result.append(sqrt(nb / na))
    result.append(max_abs)
    result.append(Float64(nonfinite))
    result.append(Float64(repeat_mismatch))
    return result^


def _fallback_identity(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    step: Int,
    layer: Int,
    ck_scratch: Optional[ComfyKitchenAttentionScratch],
    compare_scratch: AdaptiveBlockAttentionTiledScratch,
    ctx: DeviceContext,
) raises:
    var reference_backend = (
        MINIMAX_H3_ATTN_COMFY_KITCHEN_INT8
        if step < 10 else MINIMAX_H3_ATTN_CUDNN
    )
    var exact = _minimax_h3_attention_dispatch(
        q, k, v, scale, ctx, reference_backend,
        evg_step=step, evg_layer=layer,
        comfy_kitchen_scratch=ck_scratch,
    )
    var selected = _minimax_h3_attention_dispatch(
        q, k, v, scale, ctx, MINIMAX_H3_ATTN_ADAPTIVE_SM120,
        evg_step=step, evg_layer=layer,
        comfy_kitchen_scratch=ck_scratch,
    )
    var comparison = adaptive_block_attention_sm120_compare_to_host(
        exact, selected, compare_scratch, ctx
    )
    print(
        "H3_ADAPTIVE_FALLBACK step=", step, " layer=", layer,
        " max_abs=", comparison[3], " nonfinite=", Int(comparison[4]),
    )
    if comparison[3] != 0.0 or Int(comparison[4]) != 0:
        raise Error("H3 adaptive early-policy identity failed")


def _reject_measured_bad_h56(ctx: DeviceContext) raises:
    """The H56/S1024 rejection must happen before P6 kernel enqueue."""
    var q = randn([1, S_REJECT, H, D], UInt64(47101), STDtype.BF16, ctx)
    var k = randn([1, S_REJECT, H, D], UInt64(47102), STDtype.BF16, ctx)
    var v = randn([1, S_REJECT, H, D], UInt64(47103), STDtype.BF16, ctx)
    var scratch = AdaptiveBlockAttentionTiledScratch(1, S_REJECT, H, ctx)
    var owner = Optional[AdaptiveBlockAttentionTiledScratch](scratch.copy())
    var rejected = False
    try:
        _ = _minimax_h3_attention_dispatch(
            q, k, v, Float32(1.0) / sqrt(Float32(D)), ctx,
            MINIMAX_H3_ATTN_ADAPTIVE_SM120,
            evg_step=10, evg_layer=2,
            adaptive_sm120_scratch=owner.copy(),
            adaptive_sink_tokens=1,
        )
    except e:
        var message = String(e)
        if message != String(S_REJECT_ERROR):
            raise Error(
                String("unexpected H56/S1024 rejection: ") + message
            )
        rejected = True
        print("H3_ADAPTIVE_REJECT exact_error=", message)
    if not rejected:
        raise Error("H56/S1024 adaptive dispatch was not rejected")


def main() raises:
    var ctx = DeviceContext()
    _reject_measured_bad_h56(ctx)
    var q = randn([B, S, H, D], UInt64(47001), STDtype.BF16, ctx)
    var k = randn([B, S, H, D], UInt64(47002), STDtype.BF16, ctx)
    var v = randn([B, S, H, D], UInt64(47003), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var scratch = AdaptiveBlockAttentionTiledScratch(B, S, H, ctx)
    var scratch_owner = Optional[AdaptiveBlockAttentionTiledScratch](
        scratch.copy()
    )
    var direct_scratch = AdaptiveBlockAttentionTiledScratch(B, S, H, ctx)
    # The P6 device comparator reuses route-count storage for 4096 partials;
    # isolate that diagnostic capacity from both run-lifetime output owners.
    var compare_scratch = AdaptiveBlockAttentionTiledScratch(
        B, COMPARE_SCRATCH_TOKENS, H, ctx
    )
    var ck_scratch = ComfyKitchenAttentionScratch(S, H, ctx)
    var ck_scratch_owner = Optional[ComfyKitchenAttentionScratch](
        ck_scratch.copy()
    )
    _fallback_identity(
        q, k, v, scale, 9, 2, ck_scratch_owner.copy(), compare_scratch, ctx
    )
    _fallback_identity(
        q, k, v, scale, 10, 1, ck_scratch_owner.copy(), compare_scratch, ctx
    )

    var exact = _minimax_h3_attention_dispatch(
        q, k, v, scale, ctx, MINIMAX_H3_ATTN_CUDNN,
        evg_step=10, evg_layer=2,
    )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    exact = _minimax_h3_attention_dispatch(
        q, k, v, scale, ctx, MINIMAX_H3_ATTN_CUDNN,
        evg_step=10, evg_layer=2,
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()

    # Warm and time the exact same model dispatch seam.  Scratch is allocated
    # once above and never inside dispatch.
    var candidate = _minimax_h3_attention_dispatch(
        q, k, v, scale, ctx, MINIMAX_H3_ATTN_ADAPTIVE_SM120,
        evg_step=10, evg_layer=2,
        adaptive_sm120_scratch=scratch_owner.copy(),
        adaptive_sink_tokens=SINK_TOKENS,
    )
    ctx.synchronize()
    var t2 = perf_counter_ns()
    candidate = _minimax_h3_attention_dispatch(
        q, k, v, scale, ctx, MINIMAX_H3_ATTN_ADAPTIVE_SM120,
        evg_step=10, evg_layer=2,
        adaptive_sm120_scratch=scratch_owner.copy(),
        adaptive_sink_tokens=SINK_TOKENS,
    )
    ctx.synchronize()
    var t3 = perf_counter_ns()

    var direct = adaptive_block_attention_sm120_bf16(
        q, k, v, scale, minimax_h3_adaptive_sm120_tau(), 0, SINK_TOKENS,
        direct_scratch.copy(), ctx
    )
    ctx.synchronize()
    print("H3_ADAPTIVE_COPY exact")
    var exact_host = exact.to_host(ctx)
    print("H3_ADAPTIVE_COPY direct")
    var direct_host = direct.to_host(ctx)
    if direct_scratch.max_tokens < S:
        raise Error("direct P6 scratch lifetime guard failed")
    print("H3_ADAPTIVE_COPY dispatch")
    var candidate_host = candidate.to_host(ctx)
    var repeated = _minimax_h3_attention_dispatch(
        q, k, v, scale, ctx, MINIMAX_H3_ATTN_ADAPTIVE_SM120,
        evg_step=10, evg_layer=2,
        adaptive_sm120_scratch=scratch_owner.copy(),
        adaptive_sink_tokens=SINK_TOKENS,
    )
    var repeated_host = repeated.to_host(ctx)
    var metrics = _host_metrics(
        direct_host, candidate_host, repeated_host
    )
    var cudnn_info = _host_metrics(
        exact_host, candidate_host, repeated_host
    )
    var counts = adaptive_block_attention_sm120_route_counts_to_host(
        q, k, v, scale, minimax_h3_adaptive_sm120_tau(), 0, SINK_TOKENS,
        scratch, ctx
    )
    var blocks = (S + 63) // 64
    var exact_routes = Float64(0.0)
    var min_routes = blocks
    var max_routes = 0
    for value in counts:
        var n = Int(value)
        exact_routes += Float64(n)
        min_routes = min(min_routes, n)
        max_routes = max(max_routes, n)
    var density = exact_routes / Float64(len(counts) * blocks)
    var exact_ms = _ms(t0, t1)
    var candidate_ms = _ms(t2, t3)
    print(
        "H3_ADAPTIVE_FIXTURE backend=", minimax_h3_adaptive_sm120_backend_name(),
        " shape=", B, "x", S, "x", H, "x", D,
    )
    print(
        "H3_ADAPTIVE_METRICS cosine=", metrics[0],
        " magnitude=", metrics[1], " max_abs=", metrics[2],
        " nonfinite=", Int(metrics[3]),
        " repeat_mismatch=", Int(metrics[4]),
    )
    print(
        "H3_ADAPTIVE_CUDNN_INFORMATIONAL cosine=", cudnn_info[0],
        " magnitude=", cudnn_info[1], " max_abs=", cudnn_info[2],
        " (not an acceptance oracle)",
    )
    print(
        "H3_ADAPTIVE_ROUTES density=", density,
        " min_per_q=", min_routes, " max_per_q=", max_routes,
    )
    print(
        "H3_ADAPTIVE_TIMING cudnn_ms=", exact_ms,
        " candidate_ms=", candidate_ms,
        " speedup=", exact_ms / candidate_ms,
    )
    if metrics[0] < 0.9999 or metrics[1] < 0.995 \
            or metrics[1] > 1.005 or Int(metrics[3]) != 0 \
            or Int(metrics[4]) != 0 \
            or density <= 0.0 or density >= 1.0 \
            or candidate_ms > exact_ms * 0.8:
        raise Error("H3 adaptive SM120 fixture acceptance gate failed")
