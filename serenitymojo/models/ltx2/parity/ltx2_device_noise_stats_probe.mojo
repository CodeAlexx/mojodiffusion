# ltx2_device_noise_stats_probe.mojo — N(0,1) distribution gate for the ltx2
# driver's ACTUAL noise stream: ops.random.randn (device ChaCha12/Box-Muller)
# at the driver's exact seed pattern (seed + step*104729) and latent shape
# [1,128,NF,NH,NW]. Complements training/noise_stats_smoke.mojo (which gates
# the HOST _host_randn family, a different generator).
#
# Gates (n = 30 steps x 73,728 = 2,211,840 draws):
#   1. |mean| < 4/sqrt(n)                    2. |std - 1| < 0.01
#   3. P(|x|>1.96) in 0.05±20%, P(|x|>2.576) in 0.01±25%
#   4. adjacent-step same-index correlation |r| < 4/sqrt(73728)
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/ltx2/parity/ltx2_device_noise_stats_probe.mojo \
#     -o /tmp/ltx2_device_noise_probe
#   /tmp/ltx2_device_noise_probe          # GPU required (device randn)

from std.math import sqrt, abs
from max.gpu.host import DeviceContext
from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn

comptime C = 128
comptime NF = 4
comptime NH = 9
comptime NW = 16
comptime NEL = C * NF * NH * NW  # 73728
comptime STEPS = 30
comptime SEED = UInt64(42)


def _shape() -> List[Int]:
    var s = List[Int]()
    s.append(1); s.append(C); s.append(NF); s.append(NH); s.append(NW)
    return s^


def main() raises:
    var ctx = DeviceContext()
    var total_n = 0
    var sum1 = 0.0
    var sum2 = 0.0
    var sum3 = 0.0
    var sum4 = 0.0
    var n_196 = 0
    var n_2576 = 0
    var prev = List[Float32]()
    var worst_corr = 0.0

    for step in range(1, STEPS + 1):
        var t = randn(_shape(), SEED + UInt64(step) * UInt64(104729), STDtype.F32, ctx)
        var h = t.to_host(ctx)
        if len(h) != NEL:
            raise Error("unexpected numel")
        for i in range(NEL):
            var x = Float64(h[i])
            sum1 += x; sum2 += x * x; sum3 += x * x * x; sum4 += x * x * x * x
            if abs(x) > 1.96:
                n_196 += 1
            if abs(x) > 2.576:
                n_2576 += 1
        total_n += NEL
        if len(prev) == NEL:
            var sxy = 0.0
            var sx = 0.0
            var sy = 0.0
            var sxx = 0.0
            var syy = 0.0
            for i in range(NEL):
                var a = Float64(prev[i]); var b = Float64(h[i])
                sxy += a * b; sx += a; sy += b; sxx += a * a; syy += b * b
            var nn = Float64(NEL)
            var cov = sxy / nn - (sx / nn) * (sy / nn)
            var va = sxx / nn - (sx / nn) * (sx / nn)
            var vb = syy / nn - (sy / nn) * (sy / nn)
            var r = cov / sqrt(va * vb)
            if abs(r) > worst_corr:
                worst_corr = abs(r)
        prev = h^

    var n = Float64(total_n)
    var mean = sum1 / n
    var var_ = sum2 / n - mean * mean
    var std = sqrt(var_)
    var skew = (sum3 / n - 3.0 * mean * var_ - mean * mean * mean) / (std * std * std)
    var kurt = (sum4 / n) / (var_ * var_) - 3.0
    var f196 = Float64(n_196) / n
    var f2576 = Float64(n_2576) / n

    print("n =", total_n, " mean =", mean, " std =", std,
          " skew =", skew, " excess-kurt =", kurt)
    print("P(|x|>1.96) =", f196, "(exp 0.05)  P(|x|>2.576) =", f2576, "(exp 0.01)")
    print("worst adjacent-step corr =", worst_corr,
          " (bar", 4.0 / sqrt(Float64(NEL)), ")")

    var bar_mean = 4.0 / sqrt(n)
    var pass_ = True
    if abs(mean) > bar_mean:
        print("FAIL gate1 mean"); pass_ = False
    else:
        print("PASS gate1 mean")
    if abs(std - 1.0) > 0.01:
        print("FAIL gate2 std"); pass_ = False
    else:
        print("PASS gate2 std")
    if f196 < 0.04 or f196 > 0.06 or f2576 < 0.0075 or f2576 > 0.0125:
        print("FAIL gate3 tails"); pass_ = False
    else:
        print("PASS gate3 tails")
    if worst_corr > 4.0 / sqrt(Float64(NEL)):
        print("FAIL gate4 step-correlation"); pass_ = False
    else:
        print("PASS gate4 step-correlation")
    if not pass_:
        raise Error("ltx2 device noise probe FAILED")
    print("=== ALL 4 GATES PASS ===")
