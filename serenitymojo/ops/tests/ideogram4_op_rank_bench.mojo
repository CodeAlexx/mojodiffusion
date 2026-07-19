# Reliable per-op timing at the block's real dims (nsys broken) to rank the GPU
# kill order. The trainer is GPU-bound; this finds which kernel dominates the
# ~122ms/block.
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.linear import linear
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import concat, slice


def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _s4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def _t(name: String, ms: Float64):
    print("  ", name, ":", ms, "ms/iter")


def main() raises:
    var ctx = DeviceContext()
    comptime S = 1280
    comptime Hidden = 4608
    comptime Heads = 18
    comptime Dh = 256
    comptime FF = 12288
    var N = 20
    var scale = Float32(1.0) / Float32(16.0)

    var x = randn(_s2(S, Hidden), UInt64(1), STDtype.BF16, ctx)
    var qkv_w = randn(_s2(3 * Hidden, Hidden), UInt64(2), STDtype.BF16, ctx)
    var w1 = randn(_s2(FF, Hidden), UInt64(3), STDtype.BF16, ctx)
    var q = randn(_s4(1, S, Heads, Dh), UInt64(4), STDtype.BF16, ctx)
    var k = randn(_s4(1, S, Heads, Dh), UInt64(5), STDtype.BF16, ctx)
    var v = randn(_s4(1, S, Heads, Dh), UInt64(6), STDtype.BF16, ctx)
    var dq = randn(_s2(S, Hidden), UInt64(7), STDtype.BF16, ctx)
    var dk = randn(_s2(S, Hidden), UInt64(8), STDtype.BF16, ctx)
    var dv = randn(_s2(S, Hidden), UInt64(9), STDtype.BF16, ctx)

    # warmup
    var a0 = sdpa_nomask[1, S, Heads, Dh](q, k, v, scale, ctx); ctx.synchronize()

    ctx.synchronize(); var t0 = perf_counter_ns()
    for _ in range(N):
        var a = sdpa_nomask[1, S, Heads, Dh](q, k, v, scale, ctx)
    ctx.synchronize(); var t1 = perf_counter_ns()
    _t("SDPA fwd (math, Dh=256)", Float64(t1 - t0) / 1.0e6 / Float64(N))

    ctx.synchronize(); var t2 = perf_counter_ns()
    for _ in range(N):
        var y = linear(x, qkv_w, None, ctx)
    ctx.synchronize(); var t3 = perf_counter_ns()
    _t("qkv GEMM [1280,4608]x[13824,4608]", Float64(t3 - t2) / 1.0e6 / Float64(N))

    ctx.synchronize(); var t4 = perf_counter_ns()
    for _ in range(N):
        var y2 = linear(x, w1, None, ctx)
    ctx.synchronize(); var t5 = perf_counter_ns()
    _t("w1 GEMM [1280,4608]x[12288,4608]", Float64(t5 - t4) / 1.0e6 / Float64(N))

    ctx.synchronize(); var t6 = perf_counter_ns()
    for _ in range(N):
        var c = concat(1, ctx, dq, dk, dv)
    ctx.synchronize(); var t7 = perf_counter_ns()
    _t("concat3 [1280,4608]x3 (byte scatter)", Float64(t7 - t6) / 1.0e6 / Float64(N))

    ctx.synchronize(); var t8 = perf_counter_ns()
    for _ in range(N):
        var sl = slice(qkv_w, 0, 0, Hidden, ctx)
    ctx.synchronize(); var t9 = perf_counter_ns()
    _t("slice [13824,4608]->[4608,4608] (byte gather)", Float64(t9 - t8) / 1.0e6 / Float64(N))
