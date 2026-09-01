# serenitymojo/ops/tests/adaptive_block_attention_bf16_gate.mojo
#
# Isolated correctness gate for the native BF16 adaptive-attention prototype:
#   * repeated exact and approximate executions are bit deterministic;
#   * tau=-1000 is hard-bounded against repository dense attention;
#   * finite tau exercises both exact token rows and approximate KC/VC mass;
#   * a directly materialized host recurrence checks the mixed executor;
#   * odd tails check BF16 KC mean and VC sum explicitly;
#   * B=2/H=2 covers batch/head indexing without any product integration.

from max.gpu.host import DeviceContext
from std.math import ceildiv, exp, isfinite, sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.adaptive_block_attention_bf16 import (
    ADAPTIVE_ATTN_BLOCK,
    AdaptiveBlockAttentionScratch,
    adaptive_block_attention_bf16,
    adaptive_block_attention_prepare,
    adaptive_block_routes_to_host,
)
from serenitymojo.ops.attention import sdpa_nomask_tiled
from serenitymojo.ops.random import randn


comptime B = 2
comptime H = 2
comptime D = 128
comptime S_ODD_A = 79
comptime S_ODD_B = 145
comptime S_ROUTE = 193
comptime REPEATS = 6
comptime DENSE_MAX_ABS = Float32(0.001)
comptime APPROX_MAX_ABS = Float32(0.002)


def _assert_f32_bit_equal(
    got: List[Float32], expected: List[Float32], label: String
) raises:
    if len(got) != len(expected):
        raise Error(label + " length mismatch")
    for i in range(len(got)):
        if got[i] != expected[i]:
            raise Error(
                label + " changed at " + String(i)
                + ": got " + String(got[i])
                + " expected " + String(expected[i])
            )


def _assert_u8_equal(
    got: List[UInt8], expected: List[UInt8], label: String
) raises:
    if len(got) != len(expected):
        raise Error(label + " length mismatch")
    for i in range(len(got)):
        if got[i] != expected[i]:
            raise Error(label + " changed at " + String(i))


def _check_dense_case[tokens: Int](
    seed: UInt64,
    scratch: AdaptiveBlockAttentionScratch,
    ctx: DeviceContext,
) raises:
    var shape: List[Int] = [B, tokens, H, D]
    var q = randn(shape.copy(), seed, STDtype.BF16, ctx)
    var k = randn(shape.copy(), seed + UInt64(1), STDtype.BF16, ctx)
    var v = randn(shape.copy(), seed + UInt64(2), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var expected = sdpa_nomask_tiled[B, tokens, H, D](
        q, k, v, scale, ctx
    )
    var actual = adaptive_block_attention_bf16(
        q, k, v, scale, Float32(-1000.0), 0, 0, scratch, ctx
    )
    var expected_host = expected.to_host(ctx)
    var actual_host = actual.to_host(ctx)
    var route_host = adaptive_block_routes_to_host(B, tokens, scratch, ctx)
    var blocks = ceildiv(tokens, ADAPTIVE_ATTN_BLOCK)
    var exact_routes = 0
    for i in range(len(route_host)):
        if route_host[i] == UInt8(1):
            exact_routes += 1
        else:
            raise Error("tau=-1000 left a non-exact valid route")
    var want_routes = B * blocks * blocks * H
    if exact_routes != want_routes:
        raise Error("dense route count mismatch")

    # Reuse the same scratch and inputs enough times to expose cross-warp
    # shared-memory reuse races.  BF16 readback and routes must be bit stable.
    for repeat in range(REPEATS):
        var again = adaptive_block_attention_bf16(
            q, k, v, scale, Float32(-1000.0), 0, 0, scratch, ctx
        ).to_host(ctx)
        _assert_f32_bit_equal(
            again, actual_host, "dense repeat " + String(repeat)
        )
        _assert_u8_equal(
            adaptive_block_routes_to_host(B, tokens, scratch, ctx),
            route_host,
            "dense routes repeat " + String(repeat),
        )

    var dot = Float64(0.0)
    var expected_norm = Float64(0.0)
    var actual_norm = Float64(0.0)
    var max_abs = Float32(0.0)
    var nonfinite = 0
    for i in range(len(expected_host)):
        if not isfinite(expected_host[i]) or not isfinite(actual_host[i]):
            nonfinite += 1
        var delta = actual_host[i] - expected_host[i]
        var absolute = delta if delta >= Float32(0.0) else -delta
        if absolute > max_abs:
            max_abs = absolute
        dot += Float64(actual_host[i]) * Float64(expected_host[i])
        expected_norm += Float64(expected_host[i]) * Float64(expected_host[i])
        actual_norm += Float64(actual_host[i]) * Float64(actual_host[i])
    var cosine = dot / (
        sqrt(expected_norm) * sqrt(actual_norm) + Float64(1.0e-30)
    )
    var magnitude_ratio = sqrt(actual_norm) / (
        sqrt(expected_norm) + Float64(1.0e-30)
    )
    print(
        "dense-equivalence B=", B,
        " H=", H,
        " S=", tokens,
        " blocks=", blocks,
        " routes=", exact_routes,
        " repeats=", REPEATS,
        " cosine=", cosine,
        " max_abs=", max_abs,
        " magnitude_ratio=", magnitude_ratio,
        " nonfinite=", nonfinite,
    )
    if nonfinite != 0 or cosine < 0.999:
        raise Error("adaptive dense-equivalence numerical gate failed")
    if max_abs > DENSE_MAX_ABS:
        raise Error("adaptive dense-equivalence max_abs gate failed")
    if magnitude_ratio < 0.99 or magnitude_ratio > 1.01:
        raise Error("adaptive dense-equivalence magnitude gate failed")


def _check_tail_centroids(
    scratch: AdaptiveBlockAttentionScratch,
    ctx: DeviceContext,
) raises:
    comptime tokens = S_ODD_B
    var shape: List[Int] = [B, tokens, H, D]
    var q = randn(shape.copy(), 8301, STDtype.BF16, ctx)
    var k = randn(shape.copy(), 8302, STDtype.BF16, ctx)
    var v = randn(shape.copy(), 8303, STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    adaptive_block_attention_prepare(
        q, k, v, scale, Float32(4.0), 0, 0, scratch, ctx
    )
    var kh = k.to_host(ctx)
    var vh = v.to_host(ctx)
    var kch = scratch.kc[].to_host(ctx)
    var vch = scratch.vc[].to_host(ctx)
    var blocks = ceildiv(tokens, ADAPTIVE_ATTN_BLOCK)
    var tail_start = (blocks - 1) * ADAPTIVE_ATTN_BLOCK
    var tail_valid = tokens - tail_start
    var kc_max_abs = Float32(0.0)
    var vc_max_abs = Float32(0.0)
    for b in range(B):
        for head in range(H):
            for d in range(D):
                var ksum = Float32(0.0)
                var vsum = Float32(0.0)
                for slot in range(tail_valid):
                    var source = ((b * tokens + tail_start + slot) * H + head) * D + d
                    ksum += kh[source]
                    vsum += vh[source]
                var centroid = ((b * blocks + blocks - 1) * H + head) * D + d
                var expected_kc = Float32(BFloat16(ksum / Float32(tail_valid)))
                var expected_vc = Float32(BFloat16(vsum))
                var kdelta = kch[centroid] - expected_kc
                var vdelta = vch[centroid] - expected_vc
                if kdelta < Float32(0.0):
                    kdelta = -kdelta
                if vdelta < Float32(0.0):
                    vdelta = -vdelta
                if kdelta > kc_max_abs:
                    kc_max_abs = kdelta
                if vdelta > vc_max_abs:
                    vc_max_abs = vdelta
    print(
        "tail-centroids B=", B,
        " H=", H,
        " S=", tokens,
        " tail_valid=", tail_valid,
        " kc_max_abs=", kc_max_abs,
        " vc_max_abs=", vc_max_abs,
    )
    if tail_valid == ADAPTIVE_ATTN_BLOCK:
        raise Error("tail centroid gate did not exercise a partial block")
    if kc_max_abs != Float32(0.0) or vc_max_abs != Float32(0.0):
        raise Error("tail KC mean or VC sum mismatch")


def _check_local_sink_routes(
    scratch: AdaptiveBlockAttentionScratch,
    ctx: DeviceContext,
) raises:
    var shape: List[Int] = [B, S_ROUTE, H, D]
    var q = randn(shape.copy(), 8201, STDtype.BF16, ctx)
    var k = randn(shape.copy(), 8202, STDtype.BF16, ctx)
    var v = randn(shape.copy(), 8203, STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    adaptive_block_attention_prepare(
        q, k, v, scale, Float32(1.0e9), 1, 1, scratch, ctx
    )
    var host = adaptive_block_routes_to_host(B, S_ROUTE, scratch, ctx)
    var blocks = ceildiv(S_ROUTE, ADAPTIVE_ATTN_BLOCK)
    var exact_count = 0
    for b in range(B):
        for q_block in range(blocks):
            for kv_block in range(blocks):
                var distance = q_block - kv_block
                if distance < 0:
                    distance = -distance
                var expected = distance <= 1 or kv_block == 0
                for head in range(H):
                    var idx = (((b * blocks + q_block) * blocks + kv_block) * H + head)
                    var got = host[idx] == UInt8(1)
                    if got != expected:
                        raise Error("local/sink batch/head route identity mismatch")
                    if got:
                        exact_count += 1
    var expected_count = B * 12 * H
    print(
        "route-identity B=", B,
        " H=", H,
        " S=", S_ROUTE,
        " blocks=", blocks,
        " exact=", exact_count,
        " expected=", expected_count,
    )
    if exact_count != expected_count:
        raise Error("local/sink route count mismatch")


def _mixed_host_reference(
    qh: List[Float32],
    kh: List[Float32],
    vh: List[Float32],
    kch: List[Float32],
    vch: List[Float32],
    routes: List[UInt8],
    tokens: Int,
    scale: Float32,
) -> List[Float32]:
    var blocks = ceildiv(tokens, ADAPTIVE_ATTN_BLOCK)
    var out = List[Float32](capacity=B * tokens * H * D)
    for b in range(B):
        for query in range(tokens):
            var q_block = query // ADAPTIVE_ATTN_BLOCK
            for head in range(H):
                var accum = List[Float32](capacity=D)
                for _ in range(D):
                    accum.append(Float32(0.0))
                var denominator = Float32(0.0)
                var running_max = Float32(-1.0e30)
                for kv_block in range(blocks):
                    var route_index = (((b * blocks + q_block) * blocks + kv_block) * H + head)
                    var exact = routes[route_index] != UInt8(0)
                    var start = kv_block * ADAPTIVE_ATTN_BLOCK
                    var valid = tokens - start
                    if valid > ADAPTIVE_ATTN_BLOCK:
                        valid = ADAPTIVE_ATTN_BLOCK
                    if exact:
                        for slot in range(valid):
                            var key_row = start + slot
                            var score = Float32(0.0)
                            for d in range(D):
                                var q_index = ((b * tokens + query) * H + head) * D + d
                                var k_index = ((b * tokens + key_row) * H + head) * D + d
                                score += qh[q_index] * kh[k_index]
                            score *= scale
                            var probability: Float32
                            if denominator == Float32(0.0) or score > running_max:
                                var rescale = (
                                    Float32(0.0)
                                    if denominator == Float32(0.0)
                                    else exp(running_max - score)
                                )
                                for d in range(D):
                                    accum[d] *= rescale
                                denominator *= rescale
                                running_max = score
                                probability = Float32(1.0)
                            else:
                                probability = exp(score - running_max)
                            for d in range(D):
                                var value_index = ((b * tokens + key_row) * H + head) * D + d
                                accum[d] += probability * vh[value_index]
                            denominator += probability
                    else:
                        var score = Float32(0.0)
                        var centroid_index = ((b * blocks + kv_block) * H + head) * D
                        var q_index = ((b * tokens + query) * H + head) * D
                        for d in range(D):
                            score += qh[q_index + d] * kch[centroid_index + d]
                        score *= scale
                        var probability: Float32
                        if denominator == Float32(0.0) or score > running_max:
                            var rescale = (
                                Float32(0.0)
                                if denominator == Float32(0.0)
                                else exp(running_max - score)
                            )
                            for d in range(D):
                                accum[d] *= rescale
                            denominator *= rescale
                            running_max = score
                            probability = Float32(1.0)
                        else:
                            probability = exp(score - running_max)
                        for d in range(D):
                            accum[d] += probability * vch[centroid_index + d]
                        denominator += probability * Float32(valid)
                for d in range(D):
                    out.append(Float32(BFloat16(accum[d] / denominator)))
    return out^


def _check_finite_tau_reference(
    scratch: AdaptiveBlockAttentionScratch,
    ctx: DeviceContext,
) raises:
    var shape: List[Int] = [B, S_ROUTE, H, D]
    var q = randn(shape.copy(), 8401, STDtype.BF16, ctx)
    var k = randn(shape.copy(), 8402, STDtype.BF16, ctx)
    var v = randn(shape.copy(), 8403, STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var tau = Float32(4.0)
    var actual = adaptive_block_attention_bf16(
        q, k, v, scale, tau, 0, 0, scratch, ctx
    ).to_host(ctx)
    var routes = adaptive_block_routes_to_host(B, S_ROUTE, scratch, ctx)
    var baseline_routes = routes.copy()
    var exact_count = 0
    var approximate_count = 0
    for i in range(len(routes)):
        if routes[i] == UInt8(1):
            exact_count += 1
        else:
            approximate_count += 1
    if exact_count == 0 or approximate_count == 0:
        raise Error("finite tau did not exercise both exact and approximate routes")

    var qh = q.to_host(ctx)
    var kh = k.to_host(ctx)
    var vh = v.to_host(ctx)
    var kch = scratch.kc[].to_host(ctx)
    var vch = scratch.vc[].to_host(ctx)
    var expected = _mixed_host_reference(
        qh, kh, vh, kch, vch, routes, S_ROUTE, scale
    )
    var max_abs = Float32(0.0)
    var dot = Float64(0.0)
    var anorm = Float64(0.0)
    var enorm = Float64(0.0)
    var nonfinite = 0
    for i in range(len(expected)):
        if not isfinite(actual[i]) or not isfinite(expected[i]):
            nonfinite += 1
        var delta = actual[i] - expected[i]
        if delta < Float32(0.0):
            delta = -delta
        if delta > max_abs:
            max_abs = delta
        dot += Float64(actual[i]) * Float64(expected[i])
        anorm += Float64(actual[i]) * Float64(actual[i])
        enorm += Float64(expected[i]) * Float64(expected[i])
    var cosine = dot / (sqrt(anorm) * sqrt(enorm) + Float64(1.0e-30))
    print(
        "finite-tau-reference B=", B,
        " H=", H,
        " S=", S_ROUTE,
        " tau=", tau,
        " exact=", exact_count,
        " approximate=", approximate_count,
        " cosine=", cosine,
        " max_abs=", max_abs,
        " nonfinite=", nonfinite,
    )
    if nonfinite != 0 or cosine < 0.999:
        raise Error("finite-tau host reference cosine gate failed")
    if max_abs > APPROX_MAX_ABS:
        raise Error("finite-tau host reference max_abs gate failed")

    for repeat in range(REPEATS):
        var again = adaptive_block_attention_bf16(
            q, k, v, scale, tau, 0, 0, scratch, ctx
        ).to_host(ctx)
        _assert_f32_bit_equal(
            again, actual, "finite-tau repeat " + String(repeat)
        )
        _assert_u8_equal(
            adaptive_block_routes_to_host(B, S_ROUTE, scratch, ctx),
            baseline_routes,
            "finite-tau routes repeat " + String(repeat),
        )


def main() raises:
    var ctx = DeviceContext()
    var scratch = AdaptiveBlockAttentionScratch(B, S_ROUTE, H, ctx)
    print("scratch_bytes=", scratch.resident_bytes())
    _check_dense_case[S_ODD_A](8101, scratch, ctx)
    _check_dense_case[S_ODD_B](8111, scratch, ctx)
    _check_tail_centroids(scratch, ctx)
    _check_local_sink_routes(scratch, ctx)
    _check_finite_tau_reference(scratch, ctx)
    print(
        "PASS: adaptive BF16 race safety, dense bound, tails, B/H indexing, "
        "finite-tau KC/VC oracle, and deterministic scratch reuse"
    )
