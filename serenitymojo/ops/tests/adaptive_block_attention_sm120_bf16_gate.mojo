# Focused P6 correctness gate: odd tails, finite routing, dense equivalence,
# route identity, and determinism. Direct backend timing is added only after
# this small gate passes.

from max.gpu.host import DeviceContext
from std.math import isfinite, sqrt
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.env import env_or
from serenitymojo.tensor import Tensor
from serenitymojo.ops.adaptive_block_attention_sm120_bf16 import (
    adaptive_block_attention_sm120_bf16,
    adaptive_block_attention_sm120_compare_to_host,
    adaptive_block_attention_sm120_layout_microprobe,
    adaptive_block_attention_sm120_hot_route_signatures_to_host,
    adaptive_block_attention_sm120_output_probe_to_host,
    adaptive_block_attention_sm120_p3_hot_route_counts_to_host,
    adaptive_block_attention_sm120_phase_cycles_to_host,
    adaptive_block_attention_sm120_route_bitmap_to_host,
    adaptive_block_attention_sm120_route_counts_to_host,
)
from serenitymojo.ops.adaptive_block_attention_tiled_bf16 import (
    AdaptiveBlockAttentionTiledScratch,
    adaptive_block_attention_tiled_bf16,
    adaptive_block_attention_tiled_hot_route_signatures_to_host,
    adaptive_block_attention_tiled_route_bitmap_to_host,
    adaptive_block_attention_tiled_route_counts_to_host,
)
from serenitymojo.ops.attention import sdpa_nomask_tiled
from serenitymojo.ops.attention_flash import sdpa_flash_infer_fwd
from serenitymojo.ops.comfy_kitchen_attention import (
    ComfyKitchenAttentionScratch,
    comfy_kitchen_attention_available,
    comfy_kitchen_attention_current_sm,
    comfy_kitchen_attention_fwd_scratch,
    comfy_kitchen_attention_target_sm,
)
from serenitymojo.ops.random import randn


comptime B = 1
comptime D = 128
comptime S_FINITE = 193
comptime H_FINITE = 2
comptime S_DENSE = 129
comptime H_DENSE = 2
comptime B_MULTI = 2
comptime S_MULTI = 129
comptime H_MULTI = 1
comptime S_GROUPED = 4161
comptime H_GROUPED = 1
comptime FAST_ROUTE_REUSE = (
    get_defined_int["H3_ADAPTIVE_SM120_FAST_ROUTE", 0]() != 0
)
comptime PROFILE_TAU_X100 = get_defined_int[
    "H3_ADAPTIVE_SM120_TAU_X100", 100
]()
comptime PROFILE_TAU = Float32(PROFILE_TAU_X100) / Float32(100.0)
comptime PROFILE_SINK_TOKENS = get_defined_int[
    "H3_ADAPTIVE_SM120_PROFILE_SINK_TOKENS", 1
]()


def _layout_microgate(ctx: DeviceContext) raises:
    var source = randn([16, 128], UInt64(12701), STDtype.BF16, ctx)
    var probe = adaptive_block_attention_sm120_layout_microprobe(source, ctx)
    var src = source.to_host_bf16(ctx)
    var physical = probe[0].to_host_bf16(ctx)
    var qk = probe[1].to_host_bf16(ctx)
    var pv = probe[2].to_host_bf16(ctx)
    var physical_mismatch = 0
    var qk_mismatch = 0
    var pv_mismatch = 0
    for row in range(16):
        for d in range(128):
            var phys = row * 128 + (d ^ ((row & 7) * 8))
            var expected = src[row * 128 + d] \
                if row < 11 else BFloat16(0.0)
            if physical[phys] != expected:
                physical_mismatch += 1
    for lane in range(32):
        var group = lane >> 2
        var thread = lane & 3
        var key = 8 + group
        for i in range(8):
            var local = i & 3
            var d = (16 if i >= 4 else 0) + thread * 2 \
                + (local & 1) + (8 if local >= 2 else 0)
            var expected = src[key * 128 + d] \
                if key < 11 else BFloat16(0.0)
            if qk[lane * 8 + i] != expected:
                qk_mismatch += 1
        for i in range(8):
            var pair = i >> 2
            var local = i & 3
            var vrow = thread * 2 + (local & 1) \
                + (8 if local >= 2 else 0)
            var d = pair * 8 + group
            var expected = src[vrow * 128 + d] \
                if vrow < 11 else BFloat16(0.0)
            if pv[lane * 8 + i] != expected:
                pv_mismatch += 1
    print(
        "P6_LAYOUT physical_mismatch=", physical_mismatch,
        " qk_fragment_mismatch=", qk_mismatch,
        " pv_fragment_mismatch=", pv_mismatch,
    )
    if physical_mismatch != 0 or qk_mismatch != 0 or pv_mismatch != 0:
        raise Error("P6 swizzle/ldmatrix/odd-tail microgate failed")


def _metrics(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> List[Float64]:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float64(0.0)
    var bad = 0
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        if not isfinite(x) or not isfinite(y):
            bad += 1
        dot += x * y
        na += x * x
        nb += y * y
        var d = x - y
        d = d if d >= 0 else -d
        max_abs = max_abs if max_abs > d else d
    return [
        dot / (sqrt(na) * sqrt(nb) + Float64(1.0e-30)),
        max_abs,
        sqrt(na) / (sqrt(nb) + Float64(1.0e-30)),
        Float64(bad),
    ]


def _finite(ctx: DeviceContext) raises:
    var shape: List[Int] = [B, S_FINITE, H_FINITE, D]
    var q = randn(shape.copy(), UInt64(12101), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(12102), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(12103), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var tau = Float32(0.75)
    var oracle_scratch = AdaptiveBlockAttentionTiledScratch(
        B, S_FINITE, H_FINITE, ctx
    )
    var p6_scratch = AdaptiveBlockAttentionTiledScratch(
        B, S_FINITE, H_FINITE, ctx
    )
    var oracle = adaptive_block_attention_tiled_bf16(
        q, k, v, scale, tau, 1, 1, oracle_scratch, ctx
    )
    var p6 = adaptive_block_attention_sm120_bf16(
        q, k, v, scale, tau, 1, 1, p6_scratch, ctx
    )
    var metric = _metrics(p6, oracle, ctx)
    var first = p6.to_host_bf16(ctx)
    var p6_again = adaptive_block_attention_sm120_bf16(
        q, k, v, scale, tau, 1, 1, p6_scratch, ctx
    )
    var second = p6_again.to_host_bf16(ctx)
    var repeat_mismatch = 0
    for i in range(len(first)):
        if first[i] != second[i]:
            repeat_mismatch += 1
    var oracle_routes = adaptive_block_attention_tiled_route_bitmap_to_host(
        q, k, v, scale, tau, 1, 1, oracle_scratch, ctx
    )
    var p6_routes = adaptive_block_attention_sm120_route_bitmap_to_host(
        q, k, v, scale, tau, 1, 1, p6_scratch, ctx
    )
    var oracle_counts = adaptive_block_attention_sm120_p3_hot_route_counts_to_host(
        q, k, v, scale, tau, 1, 1, oracle_scratch, ctx
    )
    var p6_counts = adaptive_block_attention_sm120_route_counts_to_host(
        q, k, v, scale, tau, 1, 1, p6_scratch, ctx
    )
    var route_mismatch = 0
    var count_mismatch = 0
    var exact = 0
    for i in range(len(oracle_routes)):
        exact += Int(p6_routes[i])
        if p6_routes[i] != oracle_routes[i]:
            route_mismatch += 1
    for i in range(len(oracle_counts)):
        if p6_counts[i] != oracle_counts[i]:
            count_mismatch += 1
    print(
        "P6_FINITE cos=", metric[0], " max_abs=", metric[1],
        " magnitude=", metric[2], " nonfinite=", Int(metric[3]),
        " exact=", exact, "/", len(p6_routes),
        " route_mismatch=", route_mismatch,
        " count_mismatch=", count_mismatch,
        " repeat_mismatch=", repeat_mismatch,
    )
    if metric[0] < 0.999 or metric[1] > 0.0078125 \
            or metric[2] < 0.99 or metric[2] > 1.01 \
            or Int(metric[3]) != 0:
        raise Error("P6 finite numerical gate failed")
    if route_mismatch != 0 or count_mismatch != 0 or repeat_mismatch != 0:
        raise Error("P6 route identity/determinism gate failed")
    if exact <= 0 or exact >= len(p6_routes):
        raise Error("P6 finite gate did not exercise mixed routes")


def _dense(ctx: DeviceContext) raises:
    var shape: List[Int] = [B, S_DENSE, H_DENSE, D]
    var q = randn(shape.copy(), UInt64(12201), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(12202), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(12203), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var scratch = AdaptiveBlockAttentionTiledScratch(B, S_DENSE, H_DENSE, ctx)
    var exact = sdpa_nomask_tiled[B, S_DENSE, H_DENSE, D](
        q, k, v, scale, ctx
    )
    var p6 = adaptive_block_attention_sm120_bf16(
        q, k, v, scale, Float32(-1000.0), 0, 0, scratch, ctx
    )
    var metric = _metrics(p6, exact, ctx)
    print(
        "P6_DENSE cos=", metric[0], " max_abs=", metric[1],
        " magnitude=", metric[2], " nonfinite=", Int(metric[3]),
    )
    if metric[0] < 0.999 or metric[1] > 0.0078125 \
            or metric[2] < 0.99 or metric[2] > 1.01 \
            or Int(metric[3]) != 0:
        raise Error("P6 dense-equivalence gate failed")


def _multi_batch(ctx: DeviceContext) raises:
    var shape: List[Int] = [B_MULTI, S_MULTI, H_MULTI, D]
    var q = randn(shape.copy(), UInt64(12401), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(12402), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(12403), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var tau = Float32(0.75)
    var p3_scratch = AdaptiveBlockAttentionTiledScratch(
        B_MULTI, S_MULTI, H_MULTI, ctx
    )
    var p6_scratch = AdaptiveBlockAttentionTiledScratch(
        B_MULTI, S_MULTI, H_MULTI, ctx
    )
    var p3 = adaptive_block_attention_tiled_bf16(
        q, k, v, scale, tau, 1, 1, p3_scratch, ctx
    )
    var p6 = adaptive_block_attention_sm120_bf16(
        q, k, v, scale, tau, 1, 1, p6_scratch, ctx
    )
    var metric = _metrics(p6, p3, ctx)
    var p3_routes = adaptive_block_attention_tiled_route_bitmap_to_host(
        q, k, v, scale, tau, 1, 1, p3_scratch, ctx
    )
    var p6_routes = adaptive_block_attention_sm120_route_bitmap_to_host(
        q, k, v, scale, tau, 1, 1, p6_scratch, ctx
    )
    var p3_counts = adaptive_block_attention_sm120_p3_hot_route_counts_to_host(
        q, k, v, scale, tau, 1, 1, p3_scratch, ctx
    )
    var p6_counts = adaptive_block_attention_sm120_route_counts_to_host(
        q, k, v, scale, tau, 1, 1, p6_scratch, ctx
    )
    var mismatch = 0
    var count_mismatch = 0
    for i in range(len(p3_routes)):
        if p3_routes[i] != p6_routes[i]:
            mismatch += 1
    for i in range(len(p3_counts)):
        if p3_counts[i] != p6_counts[i]:
            count_mismatch += 1
    print(
        "P6_MULTI B=", B_MULTI, " S=", S_MULTI,
        " cos=", metric[0], " max_abs=", metric[1],
        " route_mismatch=", mismatch,
        " count_mismatch=", count_mismatch,
    )
    if metric[0] < 0.999 or metric[1] > 0.0078125 \
            or mismatch != 0 or count_mismatch != 0:
        raise Error("P6 multi-batch gate failed")


def _cross_route_group(ctx: DeviceContext) raises:
    var shape: List[Int] = [1, S_GROUPED, H_GROUPED, D]
    var q = randn(shape.copy(), UInt64(12501), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(12502), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(12503), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var tau = Float32(1.0)
    var p3_scratch = AdaptiveBlockAttentionTiledScratch(
        1, S_GROUPED, H_GROUPED, ctx
    )
    var p6_scratch = AdaptiveBlockAttentionTiledScratch(
        1, S_GROUPED, H_GROUPED, ctx
    )
    var p3 = adaptive_block_attention_tiled_bf16(
        q, k, v, scale, tau, 0, 1, p3_scratch, ctx
    )
    var p6 = adaptive_block_attention_sm120_bf16(
        q, k, v, scale, tau, 0, 1, p6_scratch, ctx
    )
    var metric = _metrics(p6, p3, ctx)
    var p3_counts = adaptive_block_attention_sm120_p3_hot_route_counts_to_host(
        q, k, v, scale, tau, 0, 1, p3_scratch, ctx
    )
    var p6_counts = adaptive_block_attention_sm120_route_counts_to_host(
        q, k, v, scale, tau, 0, 1, p6_scratch, ctx
    )
    var mismatch = 0
    var exact = 0
    for i in range(len(p3_counts)):
        exact += Int(p6_counts[i])
        if Int(p3_counts[i]) != Int(p6_counts[i]):
            mismatch += 1
    var blocks = (S_GROUPED + 63) // 64
    var all_routes = blocks * blocks * H_GROUPED
    print(
        "P6_GROUPED S=", S_GROUPED, " blocks=", blocks,
        " cos=", metric[0], " max_abs=", metric[1],
        " count_mismatch=", mismatch,
        " density=", Float64(exact) / Float64(all_routes),
    )
    if metric[0] < 0.999 or metric[1] > 0.0078125 or mismatch != 0:
        raise Error("P6 cross-route-group gate failed")


def _backend[S: Int, ITERS: Int](ctx: DeviceContext) raises -> Bool:
    comptime H = 56
    var shape: List[Int] = [B, S, H, D]
    var q = randn(shape.copy(), UInt64(12301 + S), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(12302 + S), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(12303 + S), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var tau = Float32(1.0)
    var p3_scratch = AdaptiveBlockAttentionTiledScratch(B, S, H, ctx)
    var p6_scratch = AdaptiveBlockAttentionTiledScratch(B, S, H, ctx)
    var ck_scratch = ComfyKitchenAttentionScratch(S, H, ctx)
    for _ in range(2):
        _ = adaptive_block_attention_tiled_bf16(
            q, k, v, scale, tau, 0, 1, p3_scratch, ctx
        )
        _ = adaptive_block_attention_sm120_bf16(
            q, k, v, scale, tau, 0, 1, p6_scratch, ctx
        )
        _ = comfy_kitchen_attention_fwd_scratch(
            q, k, v, scale, ck_scratch, ctx
        )
        _ = sdpa_flash_infer_fwd[B, S, H, D](q, k, v, scale, ctx)
    ctx.synchronize()

    # A/B/A-style order reversal reduces one-direction thermal/cache bias.
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = adaptive_block_attention_tiled_bf16(
            q, k, v, scale, tau, 0, 1, p3_scratch, ctx
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(ITERS):
        _ = adaptive_block_attention_sm120_bf16(
            q, k, v, scale, tau, 0, 1, p6_scratch, ctx
        )
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        _ = comfy_kitchen_attention_fwd_scratch(
            q, k, v, scale, ck_scratch, ctx
        )
    ctx.synchronize()
    var t3 = perf_counter_ns()
    for _ in range(ITERS):
        _ = sdpa_flash_infer_fwd[B, S, H, D](q, k, v, scale, ctx)
    ctx.synchronize()
    var t4 = perf_counter_ns()
    for _ in range(ITERS):
        _ = sdpa_flash_infer_fwd[B, S, H, D](q, k, v, scale, ctx)
    ctx.synchronize()
    var t5 = perf_counter_ns()
    for _ in range(ITERS):
        _ = comfy_kitchen_attention_fwd_scratch(
            q, k, v, scale, ck_scratch, ctx
        )
    ctx.synchronize()
    var t6 = perf_counter_ns()
    for _ in range(ITERS):
        _ = adaptive_block_attention_sm120_bf16(
            q, k, v, scale, tau, 0, 1, p6_scratch, ctx
        )
    ctx.synchronize()
    var t7 = perf_counter_ns()
    for _ in range(ITERS):
        _ = adaptive_block_attention_tiled_bf16(
            q, k, v, scale, tau, 0, 1, p3_scratch, ctx
        )
    ctx.synchronize()
    var t8 = perf_counter_ns()
    var p3_ms = Float64((t1 - t0) + (t8 - t7)) \
        / Float64(2 * ITERS) / 1.0e6
    var p6_ms = Float64((t2 - t1) + (t7 - t6)) \
        / Float64(2 * ITERS) / 1.0e6
    var ck_ms = Float64((t3 - t2) + (t6 - t5)) \
        / Float64(2 * ITERS) / 1.0e6
    var cudnn_ms = Float64((t4 - t3) + (t5 - t4)) \
        / Float64(2 * ITERS) / 1.0e6
    var counts = adaptive_block_attention_sm120_route_counts_to_host(
        q, k, v, scale, tau, 0, 1, p6_scratch, ctx
    )
    var exact_routes = 0
    for i in range(len(counts)):
        exact_routes += Int(counts[i])
    var blocks = (S + 63) // 64
    var all_routes = blocks * blocks * H
    print(
        "P6_BACKEND S=", S, " H=", H,
        " p3_ms=", p3_ms, " p6_ms=", p6_ms,
        " ck_ms=", ck_ms, " cudnn_ms=", cudnn_ms,
        " p6_vs_p3=", p3_ms / p6_ms,
        " p6_over_ck=", p6_ms / ck_ms,
        " exact_density=", Float64(exact_routes) / Float64(all_routes),
    )
    return p6_ms < p3_ms * 0.90 and p6_ms <= ck_ms * 1.25


def _profile_only[S: Int](ctx: DeviceContext) raises:
    comptime H = 56
    var shape: List[Int] = [1, S, H, D]
    var q = randn(shape.copy(), UInt64(12601 + S), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(12602 + S), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(12603 + S), STDtype.BF16, ctx)
    var scratch = AdaptiveBlockAttentionTiledScratch(1, S, H, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var cycles = adaptive_block_attention_sm120_phase_cycles_to_host(
        q, k, v, scale, Float32(1.0), 0, 1, scratch, ctx
    )
    var accounted = cycles[1] + cycles[2] + cycles[3] \
        + cycles[4] + cycles[5] + cycles[6]
    print(
        "P6_PROFILE_ONLY S=", S, " H=", H,
        " phase_scope=representative_warp0",
        " total_cycles=", cycles[0],
        " centroid_stage=", cycles[1],
        " route=", cycles[2],
        " approximate=", cycles[3],
        " exact_stage=", cycles[4],
        " exact_compute=", cycles[5],
        " output=", cycles[6],
        " exact_routes_cta=", cycles[7],
        " accounted=", accounted,
    )


def _median3(a_in: Float64, b_in: Float64, c_in: Float64) -> Float64:
    var a = a_in
    var b = b_in
    var c = c_in
    if a > b:
        var tmp = a
        a = b
        b = tmp
    if b > c:
        var tmp = b
        b = c
        c = tmp
    if a > b:
        var tmp = a
        a = b
        b = tmp
    return b


def _synthetic_only[S: Int](ctx: DeviceContext) raises:
    """Isolated exact-shape P6 milestone: no comparison backend allocations."""
    comptime H = 56
    var shape: List[Int] = [1, S, H, D]
    var q = randn(shape.copy(), UInt64(13601 + S), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(13602 + S), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(13603 + S), STDtype.BF16, ctx)
    var scratch = AdaptiveBlockAttentionTiledScratch(1, S, H, ctx)
    var qkv_bytes = q.nbytes() + k.nbytes() + v.nbytes()
    var scratch_bytes = scratch.resident_bytes()
    var output_bytes = scratch.output[].nbytes()
    print(
        "P6_SYNTH_MEMORY qkv_bytes=", qkv_bytes,
        " scratch_resident_bytes=", scratch_bytes,
        " scratch_aux_bytes=", scratch_bytes - output_bytes,
        " output_bytes_in_scratch=", output_bytes,
        " accounted_unique_bytes=", qkv_bytes + scratch_bytes,
    )
    var scale = Float32(1.0) / sqrt(Float32(D))
    # Exactly one warmup. The bounded probe runs after synchronization and is
    # excluded from all forward timings.
    var warm = adaptive_block_attention_sm120_bf16(
        q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS, scratch, ctx
    )
    ctx.synchronize()
    var reference_probe = adaptive_block_attention_sm120_output_probe_to_host(
        warm, scratch, ctx
    )
    var latency = List[Float64](capacity=3)
    var nonfinite_max = Int(reference_probe[0])
    var checksum_mismatch = 0
    var sample_mismatch = 0
    for rep in range(3):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var output = adaptive_block_attention_sm120_bf16(
            q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS,
            scratch, ctx
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var elapsed_ms = Float64(t1 - t0) / 1.0e6
        latency.append(elapsed_ms)
        var probe = adaptive_block_attention_sm120_output_probe_to_host(
            output, scratch, ctx
        )
        var bad = Int(probe[0])
        if bad > nonfinite_max:
            nonfinite_max = bad
        if probe[1] != reference_probe[1]:
            checksum_mismatch += 1
        for i in range(2, 18):
            if probe[i] != reference_probe[i]:
                sample_mismatch += 1
        print(
            "P6_SYNTH_TIMING rep=", rep,
            " latency_ms=", elapsed_ms,
            " nonfinite=", bad,
            " checksum=", probe[1],
        )
    var median_ms = _median3(latency[0], latency[1], latency[2])

    # Untimed diagnostic forwards only: two count passes establish route-count
    # identity, and the phase pass records one representative warp-0 trace.
    var counts0 = adaptive_block_attention_sm120_route_counts_to_host(
        q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS, scratch, ctx
    )
    var counts1 = adaptive_block_attention_sm120_route_counts_to_host(
        q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS, scratch, ctx
    )
    var count_mismatch = 0
    var exact_routes = 0
    var min_count = (S + 63) // 64
    var max_count = 0
    for i in range(len(counts0)):
        if counts0[i] != counts1[i]:
            count_mismatch += 1
        var count = Int(counts0[i])
        exact_routes += count
        if count < min_count:
            min_count = count
        if count > max_count:
            max_count = count
    var blocks = (S + 63) // 64
    var all_routes = blocks * blocks * H
    var density = Float64(exact_routes) / Float64(all_routes)
    var cycles = adaptive_block_attention_sm120_phase_cycles_to_host(
        q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS, scratch, ctx
    )
    var accounted = cycles[1] + cycles[2] + cycles[3] \
        + cycles[4] + cycles[5] + cycles[6]
    print(
        "P6_SYNTH_RESULT S=", S, " H=", H,
        " tau=", PROFILE_TAU,
        " sink_tokens=", PROFILE_SINK_TOKENS,
        " median_ms=", median_ms,
        " latency0_ms=", latency[0],
        " latency1_ms=", latency[1],
        " latency2_ms=", latency[2],
        " nonfinite=", nonfinite_max,
        " checksum=", reference_probe[1],
        " sample0=", reference_probe[2],
        " sample1=", reference_probe[3],
        " sample15=", reference_probe[17],
        " checksum_mismatch=", checksum_mismatch,
        " sample_mismatch=", sample_mismatch,
        " exact_routes=", exact_routes,
        " route_density=", density,
        " min_q_exact=", min_count,
        " max_q_exact=", max_count,
        " count_mismatch=", count_mismatch,
    )
    print(
        "P6_SYNTH_PHASE phase_scope=representative_warp0",
        " total_cycles=", cycles[0],
        " centroid_stage=", cycles[1],
        " route=", cycles[2],
        " approximate=", cycles[3],
        " exact_stage=", cycles[4],
        " exact_compute=", cycles[5],
        " output=", cycles[6],
        " exact_routes_warp0_qblock=", cycles[7],
        " accounted=", accounted,
    )
    if nonfinite_max != 0 or checksum_mismatch != 0 \
            or sample_mismatch != 0 or count_mismatch != 0:
        raise Error("P6 exact-shape finite/determinism gate failed")
    if density <= 0.0 or density >= 1.0 \
            or min_count < 0 or max_count > blocks:
        raise Error("P6 exact-shape route-density/count gate failed")
    if median_ms > 650.0:
        raise Error("P6 exact-shape hot-call median exceeds 650 ms")


def _synthetic_parity[S: Int](ctx: DeviceContext) raises:
    """Exact-shape P3/P6 parity without full-output host mirrors."""
    comptime H = 56
    var shape: List[Int] = [1, S, H, D]
    var q = randn(shape.copy(), UInt64(14601 + S), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(14602 + S), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(14603 + S), STDtype.BF16, ctx)
    var p3_scratch = AdaptiveBlockAttentionTiledScratch(1, S, H, ctx)
    var p6_scratch = AdaptiveBlockAttentionTiledScratch(1, S, H, ctx)
    var qkv_bytes = q.nbytes() + k.nbytes() + v.nbytes()
    var p3_bytes = p3_scratch.resident_bytes()
    var p6_bytes = p6_scratch.resident_bytes()
    print(
        "P6_PARITY_MEMORY qkv_bytes=", qkv_bytes,
        " p3_scratch_bytes=", p3_bytes,
        " p6_scratch_bytes=", p6_bytes,
        " accounted_unique_bytes=", qkv_bytes + p3_bytes + p6_bytes,
    )
    var scale = Float32(1.0) / sqrt(Float32(D))
    ctx.synchronize()
    var t0 = perf_counter_ns()
    var p3 = adaptive_block_attention_tiled_bf16(
        q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS,
        p3_scratch, ctx
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var p6 = adaptive_block_attention_sm120_bf16(
        q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS,
        p6_scratch, ctx
    )
    ctx.synchronize()
    var t2 = perf_counter_ns()
    var p3_ms = Float64(t1 - t0) / 1.0e6
    var p6_ms = Float64(t2 - t1) / 1.0e6
    var comparison = adaptive_block_attention_sm120_compare_to_host(
        p3, p6, p6_scratch, ctx
    )
    var cosine = comparison[0] / (
        sqrt(comparison[1]) * sqrt(comparison[2]) + Float64(1.0e-30)
    )
    var magnitude = sqrt(comparison[1]) / (
        sqrt(comparison[2]) + Float64(1.0e-30)
    )
    var p3_signatures = (
        adaptive_block_attention_tiled_hot_route_signatures_to_host(
            q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS,
            p3_scratch, ctx
        )
    )
    var p6_signatures = (
        adaptive_block_attention_sm120_hot_route_signatures_to_host(
            q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS,
            p6_scratch, ctx
        )
    )
    var p3_oracle_counts = (
        adaptive_block_attention_sm120_p3_hot_route_counts_to_host(
            q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS,
            p3_scratch, ctx
        )
    )
    var p3_serial_counts = adaptive_block_attention_tiled_route_counts_to_host(
        q, k, v, scale, PROFILE_TAU, 0, PROFILE_SINK_TOKENS,
        p3_scratch, ctx
    )
    var signature_row_mismatch = 0
    var signature_count_mismatch = 0
    var signature_sum_mismatch = 0
    var signature_xor_mismatch = 0
    var oracle_count_mismatch = 0
    var serial_sensitivity = 0
    var exact_routes = 0
    var blocks = (S + 63) // 64
    var min_count = blocks
    var max_count = 0
    var signature_rows = blocks * H
    if len(p3_signatures) != signature_rows * 3 \
            or len(p6_signatures) != signature_rows * 3:
        raise Error("P6 exact-shape route-signature size mismatch")
    for i in range(signature_rows):
        var base = i * 3
        var p3_count = p3_signatures[base]
        var p6_count = p6_signatures[base]
        var count_diff = p3_count != p6_count
        var sum_diff = p3_signatures[base + 1] \
            != p6_signatures[base + 1]
        var xor_diff = p3_signatures[base + 2] \
            != p6_signatures[base + 2]
        if count_diff:
            signature_count_mismatch += 1
        if sum_diff:
            signature_sum_mismatch += 1
        if xor_diff:
            signature_xor_mismatch += 1
        if count_diff or sum_diff or xor_diff:
            if signature_row_mismatch < 16:
                print(
                    "P6_PARITY_SIGNATURE_DIAG index=", i,
                    " qblock=", i // H,
                    " head=", i % H,
                    " p3_count=", p3_count,
                    " p6_count=", p6_count,
                    " p3_sum=", p3_signatures[base + 1],
                    " p6_sum=", p6_signatures[base + 1],
                    " p3_xor=", p3_signatures[base + 2],
                    " p6_xor=", p6_signatures[base + 2],
                )
            signature_row_mismatch += 1
        if UInt64(p3_oracle_counts[i]) != p3_count:
            oracle_count_mismatch += 1
        if UInt64(p3_serial_counts[i]) != p3_count:
            serial_sensitivity += 1
        var count = Int(p6_count)
        exact_routes += count
        if count < min_count:
            min_count = count
        if count > max_count:
            max_count = count
    var density = Float64(exact_routes) / Float64(blocks * blocks * H)
    print(
        "P6_PARITY_RESULT S=", S, " H=", H,
        " tau=", PROFILE_TAU,
        " sink_tokens=", PROFILE_SINK_TOKENS,
        " p3_ms=", p3_ms,
        " p6_ms=", p6_ms,
        " cosine=", cosine,
        " magnitude=", magnitude,
        " max_abs=", comparison[3],
        " nonfinite=", Int(comparison[4]),
        " sample_count=", Int(comparison[5]),
        " sample_max_abs=", comparison[6],
        " sample_checksum_p3=", comparison[7],
        " sample_checksum_p6=", comparison[8],
        " sample_nonfinite=", Int(comparison[9]),
        " boundary_samples=", Int(comparison[10]),
        " exact_routes=", exact_routes,
        " route_density=", density,
        " min_q_exact=", min_count,
        " max_q_exact=", max_count,
        " hot_signature_row_mismatch=", signature_row_mismatch,
        " hot_signature_count_mismatch=", signature_count_mismatch,
        " hot_signature_sum_mismatch=", signature_sum_mismatch,
        " hot_signature_xor_mismatch=", signature_xor_mismatch,
        " oracle_count_mismatch=", oracle_count_mismatch,
        " serial_diagnostic_sensitive_rows=", serial_sensitivity,
    )
    if cosine < 0.9999 or magnitude < 0.995 or magnitude > 1.005 \
            or comparison[3] > 0.0078125 \
            or comparison[4] != 0.0 or comparison[9] != 0.0 \
            or Int(comparison[5]) < 4096 or Int(comparison[10]) < 1:
        raise Error("P6 exact-shape bounded-output parity failed")
    # Tensor-core proxy reuse follows the upstream Sol SM120 accumulation
    # order. At threshold ties it may differ from the scalar P3 diagnostic by
    # one block in a tiny bounded number of rows; output parity remains the
    # correctness gate. The canonical control build still requires identity.
    var route_mismatch_limit = 8 if FAST_ROUTE_REUSE else 0
    if signature_row_mismatch > route_mismatch_limit \
            or signature_count_mismatch > route_mismatch_limit \
            or signature_sum_mismatch > route_mismatch_limit \
            or signature_xor_mismatch > route_mismatch_limit \
            or density <= 0.0 or density >= 1.0 \
            or min_count < 0 or max_count > blocks:
        raise Error("P6 exact-shape hot route-signature parity failed")


def main() raises:
    var profile_mode = env_or("SERENITY_P6_PROFILE", "")
    if profile_mode == "layout":
        var layout_ctx = DeviceContext()
        _layout_microgate(layout_ctx)
        return
    if profile_mode == "4096":
        var profile_ctx_4096 = DeviceContext()
        _profile_only[4096](profile_ctx_4096)
        return
    if profile_mode == "16384":
        var profile_ctx_16384 = DeviceContext()
        _profile_only[16384](profile_ctx_16384)
        return
    if profile_mode == "90808":
        var synthetic_ctx = DeviceContext()
        _synthetic_only[90808](synthetic_ctx)
        return
    if profile_mode == "90808parity":
        var parity_ctx = DeviceContext()
        _synthetic_parity[90808](parity_ctx)
        return
    if not comfy_kitchen_attention_available():
        raise Error("exact-SM CK attention DSO unavailable")
    var current_sm = comfy_kitchen_attention_current_sm()
    var target_sm = comfy_kitchen_attention_target_sm()
    print("P6_DEVICE current_sm=", current_sm, " target_sm=", target_sm)
    if current_sm <= 0 or current_sm != target_sm:
        raise Error("CK target does not match active GPU")
    var ctx = DeviceContext()
    _layout_microgate(ctx)
    _finite(ctx)
    _dense(ctx)
    _multi_batch(ctx)
    _cross_route_group(ctx)
    var passes = _backend[4096, 5](ctx)
    passes = _backend[16384, 2](ctx) and passes
    if not passes:
        raise Error("P6 missed material-P3/within-25%-CK hard gate")
