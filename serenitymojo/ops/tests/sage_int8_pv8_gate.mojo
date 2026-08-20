# INT8-PV Sage gate: parity + speed vs the shipped BF16-PV Sage kernel.
#
# The reference arm is `sage_attention_int8_fwd_dynamic` (INT8 QK + BF16 PV),
# the accepted product numerics for the experimental Sage backend. The
# candidate arm quantizes P per row (round(127*exp(s-m))) and V per
# (head, channel), running PV on the same m16n8k32 s8 MMA as QK
# (INT-FlashAttention, arXiv 2409.16997). PASS bar: cos >= 0.999 vs the
# BF16-PV arm at both probe lengths — the same bar the Sage-vs-INT8-exact
# acceptance used. Speed is reported, not gated (the win case is the PV
# half of the token kernel; GA102 s8 MMA is ~4x bf16-f32acc).
#
# GPU-generated inputs in matching dtype (no CPU parity — house rule).
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.attention_flash import sdpa_flash_infer_fwd
from serenitymojo.ops.random import randn
from serenitymojo.ops.sage_attention_int8 import (
    SageInt8Scratch,
    sage_attention_int8_fwd_dynamic,
    sage_attention_int8_fwd_scratch,
    sage_attention_int8_pv8_fwd_dynamic,
    sage_attention_int8_pv8_fwd_scratch,
)


def _cos_and_maxabs(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> List[Float64]:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var mad = Float64(0.0)
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        dot += x * y
        na += x * x
        nb += y * y
        var d = x - y
        d = d if d >= 0 else -d
        mad = mad if mad > d else d
    var cos = dot / ((na**0.5) * (nb**0.5) + 1e-30)
    var out: List[Float64] = [cos, mad]
    return out^


def _run_one(S: Int, H: Int, iters: Int, ctx: DeviceContext) raises -> Bool:
    var shape: List[Int] = [1, S, H, 128]
    var q = randn(shape.copy(), UInt64(1234), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(5678), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(9012), STDtype.BF16, ctx)
    var scale = Float32(0.088388348)  # 1/sqrt(128)

    var base = sage_attention_int8_fwd_dynamic(q, k, v, scale, ctx)
    var got = sage_attention_int8_pv8_fwd_dynamic(q, k, v, scale, ctx)
    ctx.synchronize()
    var m = _cos_and_maxabs(base, got, ctx)
    print("S=", S, ": cos=", m[0], " max_abs=", m[1])

    # The product path is preallocated. It must reproduce the dynamic PV8
    # entry bit-for-bit, including non-multiple-of-64 sequence tails.
    var pv8_scratch = SageInt8Scratch(S, H, ctx, True)
    var got_scratch = sage_attention_int8_pv8_fwd_scratch(
        q, k, v, scale, pv8_scratch, ctx
    )
    var dynamic_host = got.to_host_bf16(ctx)
    var scratch_host = got_scratch.to_host_bf16(ctx)
    var bit_mismatches = 0
    for i in range(len(dynamic_host)):
        if dynamic_host[i] != scratch_host[i]:
            bit_mismatches += 1
    print(
        "  scratch_bit_mismatches=", bit_mismatches,
        " scratch_mib=",
        Float64(pv8_scratch.resident_bytes()) / (1024.0 * 1024.0),
    )

    # Timing: whole forward (quant chain + token kernel), per call.
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var r = sage_attention_int8_fwd_dynamic(q, k, v, scale, ctx)
        _ = r^
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(iters):
        var r = sage_attention_int8_pv8_fwd_dynamic(q, k, v, scale, ctx)
        _ = r^
    ctx.synchronize()
    var t2 = perf_counter_ns()
    var ref_ms = Float64(t1 - t0) / 1.0e6 / Float64(iters)
    var pv8_ms = Float64(t2 - t1) / 1.0e6 / Float64(iters)
    print("  bf16-pv:", ref_ms, "ms  int8-pv:", pv8_ms, "ms  ratio:",
          ref_ms / pv8_ms, "x")
    return m[0] >= 0.999 and bit_mismatches == 0


def _run_product_geometry_speed[S: Int](ctx: DeviceContext) raises -> Bool:
    comptime H = 56
    comptime ITERS = 10
    var shape: List[Int] = [1, S, H, 128]
    var q = randn(shape.copy(), UInt64(5201), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(5202), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(5203), STDtype.BF16, ctx)
    var scale = Float32(0.088388348)
    var bf16_scratch = SageInt8Scratch(S, H, ctx)
    var pv8_scratch = SageInt8Scratch(S, H, ctx, True)

    var exact = sdpa_flash_infer_fwd[1, S, H, 128](
        q, k, v, scale, ctx
    )
    var candidate = sage_attention_int8_pv8_fwd_scratch(
        q, k, v, scale, pv8_scratch, ctx
    )
    ctx.synchronize()
    var direct = _cos_and_maxabs(exact, candidate, ctx)
    print(
        "H3 geometry PV8 vs cuDNN: cos=", direct[0],
        " max_abs=", direct[1],
    )

    for _ in range(3):
        _ = sage_attention_int8_fwd_scratch(
            q, k, v, scale, bf16_scratch, ctx
        )
        _ = sage_attention_int8_pv8_fwd_scratch(
            q, k, v, scale, pv8_scratch, ctx
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = sage_attention_int8_fwd_scratch(
            q, k, v, scale, bf16_scratch, ctx
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(ITERS):
        _ = sage_attention_int8_pv8_fwd_scratch(
            q, k, v, scale, pv8_scratch, ctx
        )
    ctx.synchronize()
    var t2 = perf_counter_ns()
    var bf16_ms = Float64(t1 - t0) / 1.0e6 / Float64(ITERS)
    var pv8_ms = Float64(t2 - t1) / 1.0e6 / Float64(ITERS)
    print(
        "H3 geometry S=", S, " H=", H,
        " bf16_pv_ms=", bf16_ms, " pv8_ms=", pv8_ms,
        " pv8_speedup=", bf16_ms / pv8_ms,
    )
    return direct[0] >= 0.999 and pv8_ms < bf16_ms


def main() raises:
    var ctx = DeviceContext()
    var ok = True
    ok = _run_one(1024, 32, 20, ctx) and ok
    ok = _run_one(9145, 32, 5, ctx) and ok
    # The fast-resident boundary — the long-S product regime where streamed
    # blocks dominate and the PV tile count per row is ~8x S=9145.
    ok = _run_one(37951, 32, 3, ctx) and ok
    # The current 1024x576x107 product shape and the longer H3 profile both
    # gate the adaptive backend. This prevents a win at one S from hiding a
    # crossover or regression at the other.
    ok = _run_product_geometry_speed[19029](ctx) and ok
    ok = _run_product_geometry_speed[21291](ctx) and ok
    if ok:
        print(
            "PASS: PV8 parity, scratch identity, and H3-geometry speed gates"
        )
    else:
        print("FAIL: int8-pv cosine below 0.999")
        raise Error("sage_int8_pv8_gate failed")
