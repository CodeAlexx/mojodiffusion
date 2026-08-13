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
from serenitymojo.ops.random import randn
from serenitymojo.ops.sage_attention_int8 import (
    sage_attention_int8_fwd_dynamic,
    sage_attention_int8_pv8_fwd_dynamic,
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
    return m[0] >= 0.999


def main() raises:
    var ctx = DeviceContext()
    var ok = True
    ok = _run_one(1024, 32, 20, ctx) and ok
    ok = _run_one(9145, 32, 5, ctx) and ok
    if ok:
        print("PASS: int8-pv within 0.999 of bf16-pv sage at both lengths")
    else:
        print("FAIL: int8-pv cosine below 0.999")
        raise Error("sage_int8_pv8_gate failed")
