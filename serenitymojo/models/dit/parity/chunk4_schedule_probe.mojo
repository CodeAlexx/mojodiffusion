# chunk4 gate: logit-normal schedule values vs Wave-0 fixture (host scalar).
from std.math import abs as fabs
from serenitymojo.sampling.ideogram4_schedule import (
    ideogram4_logitnormal, ideogram4_schedule_mean, make_step_intervals,
)


def main() raises:
    var exp_vals = [
        0.999447226524353, 0.4893021881580353, 0.3731662333011627,
        0.29431718587875366, 0.232696533203125, 0.18067418038845062,
        0.13381539285182953, 0.08758409321308136, 0.00012339458044152707,
    ]
    var mean = ideogram4_schedule_mean(1024, 1024, 0.5)
    var si = make_step_intervals(8)
    var maxdiff: Float32 = 0.0
    for i in range(9):
        var v = ideogram4_logitnormal(Float64(si[i]), mean)
        var d = v - Float32(exp_vals[i])
        if d < 0:
            d = -d
        if d > maxdiff:
            maxdiff = d
        print("  step", i, "mine", v, "ref", Float32(exp_vals[i]))
    print("chunk4 schedule mean:", mean, "max_abs_diff:", maxdiff,
          "PASS" if maxdiff < 1.0e-4 else "FAIL")
