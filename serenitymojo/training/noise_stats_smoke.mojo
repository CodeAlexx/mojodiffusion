# training/noise_stats_smoke.mojo — N(0,1) distribution gate for the shared
# host Box-Muller noise stream (G-T2 on BEAT_FLAME_SCOREBOARD).
#
# Tests the REAL generator (`_host_randn` in noise_modifiers.mojo — same
# constants/draw-order as every `_host_noise` copy in train_*_real.mojo).
#
# HISTORY: until 2026-06-10 the klein/flux/sdxl trainers + noise_modifiers
# divided a 52-bit mantissa by 2^53, so uniforms landed in [0, 0.5):
#   * noise mean ≈ +0.56, std ≈ 1.19 (not N(0,1)) → inflated training loss
#   * theta = 2π·u2 only spanned [0, π) → EVERY odd-index (sin) draw was ≥ 0
# Gate 4 below specifically catches that failure mode forever.
#
# CPU-ONLY (never constructs a DeviceContext — safe while the GPU is busy).
# Build + run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/training/noise_stats_smoke.mojo -o /tmp/noise_stats_smoke \
#     && /tmp/noise_stats_smoke
# PASS = all 5 gates print PASS and the binary exits 0. READ the numbers.

from std.math import sqrt
from serenitymojo.training.noise_modifiers import _host_randn

comptime N = 1_000_000


def main() raises:
    print("=== noise_stats_smoke — N(0,1) gate on _host_randn (N=", N, ") ===")
    var xs = _host_randn(N, UInt64(0xC0FFEE))

    var s = Float64(0)
    var s2 = Float64(0)
    var mn = Float64(1e30)
    var mx = Float64(-1e30)
    var neg = 0
    var odd_neg = 0
    var odd_n = 0
    for i in range(N):
        var v = Float64(xs[i])
        s += v
        s2 += v * v
        if v < mn:
            mn = v
        if v > mx:
            mx = v
        if v < 0.0:
            neg += 1
        if i % 2 == 1:
            odd_n += 1
            if v < 0.0:
                odd_neg += 1

    var mean = s / Float64(N)
    var std = sqrt(s2 / Float64(N) - mean * mean)
    var neg_frac = Float64(neg) / Float64(N)
    var odd_neg_frac = Float64(odd_neg) / Float64(odd_n)

    print("mean      =", mean, "   (gate |mean| < 0.01)")
    print("std       =", std, "    (gate |std-1| < 0.01)")
    print("min / max =", mn, "/", mx, "  (gate min<-3.5, max>3.5, |x|<6.5)")
    print("neg frac  =", neg_frac, " (gate in [0.49, 0.51])")
    print("odd-index neg frac =", odd_neg_frac, " (gate in [0.49, 0.51]; old bug -> 0.0)")

    var failures = 0
    if mean < -0.01 or mean > 0.01:
        print("FAIL gate1 mean")
        failures += 1
    else:
        print("PASS gate1 mean")
    if std < 0.99 or std > 1.01:
        print("FAIL gate2 std")
        failures += 1
    else:
        print("PASS gate2 std")
    if mn > -3.5 or mx < 3.5 or mn < -6.5 or mx > 6.5:
        print("FAIL gate3 tails")
        failures += 1
    else:
        print("PASS gate3 tails")
    if neg_frac < 0.49 or neg_frac > 0.51:
        print("FAIL gate4 sign balance (overall)")
        failures += 1
    else:
        print("PASS gate4 sign balance (overall)")
    if odd_neg_frac < 0.49 or odd_neg_frac > 0.51:
        print("FAIL gate5 sign balance (sin half — the [0,0.5) bug detector)")
        failures += 1
    else:
        print("PASS gate5 sign balance (sin half)")

    if failures != 0:
        raise Error("noise_stats_smoke: " + String(failures) + " gate(s) FAILED")
    print("=== ALL 5 GATES PASS ===")
