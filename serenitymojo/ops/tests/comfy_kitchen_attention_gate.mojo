# Product-geometry acceptance gate for the direct Comfy Kitchen CUDA backend.
#
# All arms receive the same GPU-generated BF16 Q/K/V. Fidelity is measured
# against exact cuDNN SDPA. Speed compares zero-allocation resident-scratch
# paths and alternates measurement order to reduce thermal/order bias.

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.attention_flash import sdpa_flash_infer_fwd
from serenitymojo.ops.random import randn
from serenitymojo.ops.sage_attention_int8 import (
    SageInt8Scratch,
    sage_attention_int8_pv8_fwd_scratch,
)
from serenitymojo.ops.comfy_kitchen_attention import (
    ComfyKitchenAttentionScratch,
    comfy_kitchen_attention_available,
    comfy_kitchen_attention_fwd_scratch,
)


def _cos_max_mean(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> List[Float64]:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var mad = Float64(0.0)
    var mean = Float64(0.0)
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        dot += x * y
        na += x * x
        nb += y * y
        var d = x - y
        d = d if d >= 0 else -d
        mad = mad if mad > d else d
        mean += d
    return [
        dot / ((na**0.5) * (nb**0.5) + 1e-30),
        mad,
        mean / Float64(len(ah)),
    ]


def _run_geometry[S: Int](ctx: DeviceContext) raises -> Bool:
    comptime H = 56
    comptime D = 128
    comptime ITERS = 8
    var shape: List[Int] = [1, S, H, D]
    var q = randn(shape.copy(), UInt64(6301 + S), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(6302 + S), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(6303 + S), STDtype.BF16, ctx)
    var scale = Float32(0.088388348)
    var pv8_scratch = SageInt8Scratch(S, H, ctx, True)
    var ck_scratch = ComfyKitchenAttentionScratch(S, H, ctx)

    var exact = sdpa_flash_infer_fwd[1, S, H, D](q, k, v, scale, ctx)
    var ck = comfy_kitchen_attention_fwd_scratch(
        q, k, v, scale, ck_scratch, ctx
    )
    ctx.synchronize()
    var metric = _cos_max_mean(exact, ck, ctx)
    print(
        "S=", S, " CK vs exact: cos=", metric[0],
        " max_abs=", metric[1], " mean_abs=", metric[2],
        " scratch_mib=",
        Float64(ck_scratch.resident_bytes()) / (1024.0 * 1024.0),
    )

    # A repeated call through the exact same scratch must be bit-stable.
    var first = ck.to_host_bf16(ctx)
    var ck2 = comfy_kitchen_attention_fwd_scratch(
        q, k, v, scale, ck_scratch, ctx
    )
    var second = ck2.to_host_bf16(ctx)
    var mismatches = 0
    for i in range(len(first)):
        if first[i] != second[i]:
            mismatches += 1
    print("  repeat_bit_mismatches=", mismatches)

    for _ in range(3):
        _ = sage_attention_int8_pv8_fwd_scratch(
            q, k, v, scale, pv8_scratch, ctx
        )
        _ = comfy_kitchen_attention_fwd_scratch(
            q, k, v, scale, ck_scratch, ctx
        )
    ctx.synchronize()

    # Round A: existing PV8 first, CK second.
    var a0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = sage_attention_int8_pv8_fwd_scratch(
            q, k, v, scale, pv8_scratch, ctx
        )
    ctx.synchronize()
    var a1 = perf_counter_ns()
    for _ in range(ITERS):
        _ = comfy_kitchen_attention_fwd_scratch(
            q, k, v, scale, ck_scratch, ctx
        )
    ctx.synchronize()
    var a2 = perf_counter_ns()

    # Round B reverses order.
    for _ in range(ITERS):
        _ = comfy_kitchen_attention_fwd_scratch(
            q, k, v, scale, ck_scratch, ctx
        )
    ctx.synchronize()
    var b1 = perf_counter_ns()
    for _ in range(ITERS):
        _ = sage_attention_int8_pv8_fwd_scratch(
            q, k, v, scale, pv8_scratch, ctx
        )
    ctx.synchronize()
    var b2 = perf_counter_ns()

    var pv8_ms = Float64((a1 - a0) + (b2 - b1)) \
        / Float64(2 * ITERS) / 1.0e6
    var ck_ms = Float64((a2 - a1) + (b1 - a2)) \
        / Float64(2 * ITERS) / 1.0e6
    print(
        "  resident PV8=", pv8_ms, " ms CK=", ck_ms,
        " ms CK_speedup=", pv8_ms / ck_ms, "x",
    )
    return metric[0] >= 0.999 and mismatches == 0 and ck_ms < pv8_ms


def main() raises:
    if not comfy_kitchen_attention_available():
        raise Error("Comfy Kitchen CUDA launcher DSO is unavailable")
    var ctx = DeviceContext()
    var ok = True
    ok = _run_geometry[19029](ctx) and ok
    ok = _run_geometry[21291](ctx) and ok
    if not ok:
        raise Error("Comfy Kitchen attention product gate failed")
    print("PASS: CK fidelity, repeatability, and H3 speed gates")
