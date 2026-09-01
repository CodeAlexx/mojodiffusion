# serenitymojo/ops/tests/adaptive_block_attention_tiled_bf16_gate.mojo
#
# P3 gate for the route-slab-free tensor-core adaptive BF16 operator:
#   * finite-tau output and exact route density vs the accepted scalar oracle;
#   * tau=-1000 dense equivalence vs repository exact attention;
#   * repeated-call bit determinism;
#   * exact per-route identity for the warp-parallel routing decision;
#   * alternating direct cuDNN/CK timing at S=4096/H56 and S=16384/H56;
#   * analytical S=90808/H56 scratch projection without launching H3.

from max.gpu.host import DeviceContext
from std.math import ceildiv, isfinite, sqrt
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.adaptive_block_attention_bf16 import (
    AdaptiveBlockAttentionScratch,
    adaptive_block_attention_bf16,
    adaptive_block_routes_to_host,
)
from serenitymojo.ops.adaptive_block_attention_tiled_bf16 import (
    AdaptiveBlockAttentionTiledScratch,
    adaptive_block_attention_tiled_bf16,
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
comptime S_SPEED = 4096
comptime H_SPEED = 1
comptime SPEED_ITERS = 20
comptime BACKEND_HEADS = 56


def _metrics(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> List[Float64]:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("adaptive tiled metric length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float64(0.0)
    var nonfinite = 0
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        if not isfinite(x) or not isfinite(y):
            nonfinite += 1
        dot += x * y
        na += x * x
        nb += y * y
        var delta = x - y
        delta = delta if delta >= 0 else -delta
        max_abs = max_abs if max_abs > delta else delta
    var cosine = dot / (sqrt(na) * sqrt(nb) + Float64(1.0e-30))
    var magnitude = sqrt(na) / (sqrt(nb) + Float64(1.0e-30))
    return [cosine, max_abs, magnitude, Float64(nonfinite)]


def _finite_tau_gate(ctx: DeviceContext) raises:
    var shape: List[Int] = [B, S_FINITE, H_FINITE, D]
    var q = randn(shape.copy(), UInt64(9101), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(9102), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(9103), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var tau = Float32(0.75)
    var scalar_scratch = AdaptiveBlockAttentionScratch(
        B, S_FINITE, H_FINITE, ctx
    )
    var tiled_scratch = AdaptiveBlockAttentionTiledScratch(
        B, S_FINITE, H_FINITE, ctx
    )
    var scalar = adaptive_block_attention_bf16(
        q, k, v, scale, tau, 1, 1, scalar_scratch, ctx
    )
    var tiled = adaptive_block_attention_tiled_bf16(
        q, k, v, scale, tau, 1, 1, tiled_scratch, ctx
    )
    var metric = _metrics(tiled, scalar, ctx)
    var first = tiled.to_host_bf16(ctx)
    var tiled_again = adaptive_block_attention_tiled_bf16(
        q, k, v, scale, tau, 1, 1, tiled_scratch, ctx
    )
    var second = tiled_again.to_host_bf16(ctx)
    var mismatches = 0
    for i in range(len(first)):
        if first[i] != second[i]:
            mismatches += 1
    var scalar_routes = adaptive_block_routes_to_host(
        B, S_FINITE, scalar_scratch, ctx
    )
    var scalar_exact = 0
    for i in range(len(scalar_routes)):
        if scalar_routes[i] == UInt8(1):
            scalar_exact += 1
    var tiled_routes = adaptive_block_attention_tiled_route_bitmap_to_host(
        q, k, v, scale, tau, 1, 1, tiled_scratch, ctx
    )
    var tiled_exact = 0
    var route_mismatches = 0
    if len(tiled_routes) != len(scalar_routes):
        raise Error("finite-tau route bitmap length mismatch")
    for i in range(len(tiled_routes)):
        tiled_exact += Int(tiled_routes[i])
        if tiled_routes[i] != scalar_routes[i]:
            route_mismatches += 1
    var blocks = ceildiv(S_FINITE, 64)
    var all_routes = B * blocks * blocks * H_FINITE
    print(
        "finite_tau S=", S_FINITE,
        " cos=", metric[0],
        " max_abs=", metric[1],
        " magnitude=", metric[2],
        " nonfinite=", Int(metric[3]),
        " route_exact=", tiled_exact, "/", all_routes,
        " scalar_route_exact=", scalar_exact,
        " density=", Float64(tiled_exact) / Float64(all_routes),
        " route_bitmap_mismatches=", route_mismatches,
        " repeat_bit_mismatches=", mismatches,
    )
    if metric[0] < 0.999 or metric[2] < 0.99 or metric[2] > 1.01 \
            or metric[1] > 0.0078125 or Int(metric[3]) != 0:
        raise Error("finite-tau tiled/scalar numerical gate failed")
    if tiled_exact != scalar_exact or route_mismatches != 0:
        raise Error("finite-tau tiled/scalar route bitmap mismatch")
    if mismatches != 0:
        raise Error("tiled adaptive attention is not bit deterministic")


def _dense_gate(ctx: DeviceContext) raises:
    var shape: List[Int] = [B, S_DENSE, H_DENSE, D]
    var q = randn(shape.copy(), UInt64(9201), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(9202), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(9203), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var scratch = AdaptiveBlockAttentionTiledScratch(B, S_DENSE, H_DENSE, ctx)
    var exact = sdpa_nomask_tiled[B, S_DENSE, H_DENSE, D](
        q, k, v, scale, ctx
    )
    var tiled = adaptive_block_attention_tiled_bf16(
        q, k, v, scale, Float32(-1000.0), 0, 0, scratch, ctx
    )
    var metric = _metrics(tiled, exact, ctx)
    var counts = adaptive_block_attention_tiled_route_counts_to_host(
        q, k, v, scale, Float32(-1000.0), 0, 0, scratch, ctx
    )
    var exact_routes = 0
    for i in range(len(counts)):
        exact_routes += Int(counts[i])
    var blocks = ceildiv(S_DENSE, 64)
    var want_routes = B * blocks * blocks * H_DENSE
    print(
        "dense_tau S=", S_DENSE,
        " cos=", metric[0],
        " max_abs=", metric[1],
        " magnitude=", metric[2],
        " nonfinite=", Int(metric[3]),
        " routes=", exact_routes, "/", want_routes,
    )
    if metric[0] < 0.999 or metric[2] < 0.99 or metric[2] > 1.01 \
            or metric[1] > 0.0078125 or Int(metric[3]) != 0:
        raise Error("tiled adaptive dense-equivalence gate failed")
    if exact_routes != want_routes:
        raise Error("tau=-1000 did not route every dense block exact")


def _speed_gate(ctx: DeviceContext) raises:
    var shape: List[Int] = [B, S_SPEED, H_SPEED, D]
    var q = randn(shape.copy(), UInt64(9301), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(9302), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(9303), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var tau = Float32(1.0)
    var scalar_scratch = AdaptiveBlockAttentionScratch(
        B, S_SPEED, H_SPEED, ctx
    )
    var tiled_scratch = AdaptiveBlockAttentionTiledScratch(
        B, S_SPEED, H_SPEED, ctx
    )
    _ = adaptive_block_attention_bf16(
        q, k, v, scale, tau, 0, 1, scalar_scratch, ctx
    )
    _ = adaptive_block_attention_tiled_bf16(
        q, k, v, scale, tau, 0, 1, tiled_scratch, ctx
    )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(SPEED_ITERS):
        _ = adaptive_block_attention_bf16(
            q, k, v, scale, tau, 0, 1, scalar_scratch, ctx
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(SPEED_ITERS):
        _ = adaptive_block_attention_tiled_bf16(
            q, k, v, scale, tau, 0, 1, tiled_scratch, ctx
        )
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(SPEED_ITERS):
        _ = adaptive_block_attention_tiled_bf16(
            q, k, v, scale, tau, 0, 1, tiled_scratch, ctx
        )
    ctx.synchronize()
    var t3 = perf_counter_ns()
    for _ in range(SPEED_ITERS):
        _ = adaptive_block_attention_bf16(
            q, k, v, scale, tau, 0, 1, scalar_scratch, ctx
        )
    ctx.synchronize()
    var t4 = perf_counter_ns()
    var scalar_ms = Float64((t1 - t0) + (t4 - t3)) \
        / Float64(2 * SPEED_ITERS) / 1.0e6
    var tiled_ms = Float64((t2 - t1) + (t3 - t2)) \
        / Float64(2 * SPEED_ITERS) / 1.0e6
    print(
        "alternating_speed S=", S_SPEED,
        " H=", H_SPEED,
        " scalar_ms=", scalar_ms,
        " tiled_ms=", tiled_ms,
        " speedup=", scalar_ms / tiled_ms,
        " scalar_scratch_mib=",
        Float64(scalar_scratch.resident_bytes()) / 1048576.0,
        " tiled_scratch_mib=",
        Float64(tiled_scratch.resident_bytes()) / 1048576.0,
    )
    if tiled_ms >= scalar_ms * 0.90:
        raise Error("P2 tiled path did not beat scalar prototype materially")


def _backend_gate[S: Int, ITERS: Int](ctx: DeviceContext) raises -> Bool:
    comptime H = BACKEND_HEADS
    var shape: List[Int] = [B, S, H, D]
    var q = randn(shape.copy(), UInt64(9401 + S), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(9402 + S), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(9403 + S), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))
    var tau = Float32(1.0)
    var tiled_scratch = AdaptiveBlockAttentionTiledScratch(B, S, H, ctx)
    var ck_scratch = ComfyKitchenAttentionScratch(S, H, ctx)

    # Every output aliases its owning scratch until that path's next call.
    # The benchmark consumes only completion time and never overlaps forwards.
    for _ in range(2):
        _ = adaptive_block_attention_tiled_bf16(
            q, k, v, scale, tau, 0, 1, tiled_scratch, ctx
        )
        _ = comfy_kitchen_attention_fwd_scratch(
            q, k, v, scale, ck_scratch, ctx
        )
        _ = sdpa_flash_infer_fwd[B, S, H, D](q, k, v, scale, ctx)
    ctx.synchronize()

    # Round A: P3, CK, cuDNN. Round B reverses that order.
    var a0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = adaptive_block_attention_tiled_bf16(
            q, k, v, scale, tau, 0, 1, tiled_scratch, ctx
        )
    ctx.synchronize()
    var a1 = perf_counter_ns()
    for _ in range(ITERS):
        _ = comfy_kitchen_attention_fwd_scratch(
            q, k, v, scale, ck_scratch, ctx
        )
    ctx.synchronize()
    var a2 = perf_counter_ns()
    for _ in range(ITERS):
        _ = sdpa_flash_infer_fwd[B, S, H, D](q, k, v, scale, ctx)
    ctx.synchronize()
    var a3 = perf_counter_ns()
    for _ in range(ITERS):
        _ = sdpa_flash_infer_fwd[B, S, H, D](q, k, v, scale, ctx)
    ctx.synchronize()
    var b1 = perf_counter_ns()
    for _ in range(ITERS):
        _ = comfy_kitchen_attention_fwd_scratch(
            q, k, v, scale, ck_scratch, ctx
        )
    ctx.synchronize()
    var b2 = perf_counter_ns()
    for _ in range(ITERS):
        _ = adaptive_block_attention_tiled_bf16(
            q, k, v, scale, tau, 0, 1, tiled_scratch, ctx
        )
    ctx.synchronize()
    var b3 = perf_counter_ns()
    var tiled_ms = Float64((a1 - a0) + (b3 - b2)) \
        / Float64(2 * ITERS) / 1.0e6
    var ck_ms = Float64((a2 - a1) + (b2 - b1)) \
        / Float64(2 * ITERS) / 1.0e6
    var cudnn_ms = Float64((a3 - a2) + (b1 - a3)) \
        / Float64(2 * ITERS) / 1.0e6
    var counts = adaptive_block_attention_tiled_route_counts_to_host(
        q, k, v, scale, tau, 0, 1, tiled_scratch, ctx
    )
    var exact_routes = 0
    for i in range(len(counts)):
        exact_routes += Int(counts[i])
    var blocks = ceildiv(S, 64)
    var all_routes = B * blocks * blocks * H
    print(
        "backend_alternating S=", S, " H=", H,
        " tiled_ms=", tiled_ms,
        " ck_ms=", ck_ms,
        " cudnn_ms=", cudnn_ms,
        " speedup_vs_ck=", ck_ms / tiled_ms,
        " speedup_vs_cudnn=", cudnn_ms / tiled_ms,
        " exact_route_density=", Float64(exact_routes) / Float64(all_routes),
        " tiled_scratch_mib=",
        Float64(tiled_scratch.resident_bytes()) / 1048576.0,
        " ck_scratch_mib=",
        Float64(ck_scratch.resident_bytes()) / 1048576.0,
    )
    return tiled_ms < ck_ms * 0.95 or tiled_ms < cudnn_ms * 0.95


def _projection() raises:
    var target_s = 90808
    var target_h = 56
    var blocks = ceildiv(target_s, 64)
    var centroid_elems = blocks * target_h * D
    var stat_elems = target_h * D
    var block_head_elems = blocks * target_h
    var output_elems = target_s * target_h * D
    var projected = (
        centroid_elems * 8
        + stat_elems * 8
        + block_head_elems * 8
        + output_elems * 2
        + 4096
    )
    var removed_route_slab = blocks * blocks * target_h
    var current_ck_scratch_mib = Float64(3.0368838012218475 * 1024.0)
    var tiled_mib = Float64(projected) / 1048576.0
    print(
        "projection S=", target_s,
        " H=", target_h,
        " blocks=", blocks,
        " tiled_scratch_mib=", tiled_mib,
        " eliminated_route_slab_mib=",
        Float64(removed_route_slab) / 1048576.0,
        " current_ck_scratch_mib=", current_ck_scratch_mib,
        " exclusive_replacement_saves_mib=",
        current_ck_scratch_mib - tiled_mib,
    )


def main() raises:
    if not comfy_kitchen_attention_available():
        raise Error("exact-SM Comfy Kitchen attention DSO unavailable")
    var current_sm = comfy_kitchen_attention_current_sm()
    var target_sm = comfy_kitchen_attention_target_sm()
    print("CK_DEVICE current_sm=", current_sm, " target_sm=", target_sm)
    if current_sm <= 0 or target_sm != current_sm:
        raise Error("CK attention DSO target does not match active GPU")
    var ctx = DeviceContext()
    _finite_tau_gate(ctx)
    _dense_gate(ctx)
    _speed_gate(ctx)
    var beats_backend = _backend_gate[4096, 6](ctx)
    beats_backend = _backend_gate[16384, 2](ctx) or beats_backend
    _projection()
    if not beats_backend:
        raise Error("P3 did not materially beat cuDNN or CK")
    print("PASS: route-slab-free tensor-core adaptive BF16 P3")
