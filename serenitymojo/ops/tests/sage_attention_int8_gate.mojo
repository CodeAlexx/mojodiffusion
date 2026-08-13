# serenitymojo/ops/tests/sage_attention_int8_gate.mojo
#
# Isolated correctness and timing gate for the opt-in Sage INT8-QK backend.

from max.gpu.host import DeviceContext
from std.math import isfinite, sqrt
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention_flash import sdpa_flash_infer_fwd
from serenitymojo.ops.random import randn
from serenitymojo.ops.sage_attention_int8 import sage_attention_int8_fwd


comptime B = 1
comptime S = get_defined_int["SAGE_GATE_S", 1024]()
comptime H = 56
comptime D = 128
comptime WARMUP = 5
comptime ITERS = 20


def main() raises:
    var ctx = DeviceContext()
    var shape: List[Int] = [B, S, H, D]
    var q = randn(shape.copy(), 101, STDtype.BF16, ctx)
    var k = randn(shape.copy(), 102, STDtype.BF16, ctx)
    var v = randn(shape.copy(), 103, STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))

    var ref_out = sdpa_flash_infer_fwd[B, S, H, D](q, k, v, scale, ctx)
    var got = sage_attention_int8_fwd[B, S, H, D](
        q, k, v, scale, ctx
    )
    var rh = ref_out.to_host(ctx)
    var gh = got.to_host(ctx)
    var dot = Float64(0.0)
    var nr = Float64(0.0)
    var ng = Float64(0.0)
    var max_abs = Float32(0.0)
    var ref_nonfinite = 0
    var got_nonfinite = 0
    for i in range(len(rh)):
        if not isfinite(rh[i]):
            ref_nonfinite += 1
        if not isfinite(gh[i]):
            got_nonfinite += 1
        var d = gh[i] - rh[i]
        var ad = d if d >= 0.0 else -d
        if ad > max_abs:
            max_abs = ad
        dot += Float64(gh[i]) * Float64(rh[i])
        nr += Float64(rh[i]) * Float64(rh[i])
        ng += Float64(gh[i]) * Float64(gh[i])
    var cos = dot / (sqrt(nr) * sqrt(ng) + 1.0e-30)
    print("sage-int8 vs cuDNN: cos=", cos, " max_abs=", max_abs,
          " n=", len(rh), " ref_nonfinite=", ref_nonfinite,
          " got_nonfinite=", got_nonfinite)
    if ref_nonfinite != 0 or got_nonfinite != 0:
        raise Error("sage attention finite gate failed")
    if cos < 0.999:
        raise Error("sage attention parity below 0.999")

    for _ in range(WARMUP):
        _ = sage_attention_int8_fwd[B, S, H, D](q, k, v, scale, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = sage_attention_int8_fwd[B, S, H, D](q, k, v, scale, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(WARMUP):
        _ = sdpa_flash_infer_fwd[B, S, H, D](q, k, v, scale, ctx)
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        _ = sdpa_flash_infer_fwd[B, S, H, D](q, k, v, scale, ctx)
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var sage_us = Float64(t1 - t0) / Float64(ITERS) / 1000.0
    var cudnn_us = Float64(t3 - t2) / Float64(ITERS) / 1000.0
    print("timing S=", S, " H=", H, ": sage_us=", sage_us,
          " cudnn_us=", cudnn_us, " speedup=", cudnn_us / sage_us)
    if sage_us >= cudnn_us:
        raise Error("sage attention speed gate failed: not faster than cuDNN")
    print("PASS: attention parity and speed gates")
