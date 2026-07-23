# serenitymojo/models/wan22/parity/bernini_timestep_parity.mojo
#
# G1: BERNINI-R timestep-sampling parity (Mojo side).
#
# Draws the SAME per-(weighting, shift) training-sigma distribution as the torch
# oracle (bernini_timestep_oracle.py) via training/schedule.bernini_sample_sigma,
# and prints:
#   * the per-shift rejection window (tmin,tmax)  [must MATCH the oracle exactly],
#   * N-sample sigma mean/std/min/max + histogram over [0.86,1.0]/14
#     [DISTRIBUTION parity — moments within tolerance; RNG streams differ], and
#   * deterministic anchors: mode_density(raw), logit_density(z), shifted_sigma(idx)
#     [BIT-exact vs the oracle closed forms].
#
# BUILD:
#   cd /home/alex/mojodiffusion; rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#     -I . -I /home/alex/MOJO-libs -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/wan22/parity/bernini_timestep_parity.mojo \
#     -o output/bin/bernini_timestep_parity
#
# Mojo 1.0.0b1.

from std.collections import List
from serenitymojo.training.schedule import (
    BerniniWindow, bernini_task_window, bernini_sample_sigma,
    bernini_mode_density_from_raw, bernini_logit_normal_density_from_z,
    bernini_shifted_sigma,
)

comptime NOISE_TMIN = Float64(0.875)
comptime NOISE_TMAX = Float64(1.0)
comptime LOGIT_MEAN = Float64(0.5)
comptime LOGIT_STD = Float64(1.0)
comptime MODE_SCALE = Float64(1.29)


def _case(is_mode: Bool, shift: Float64, tasks: String, n: Int) raises:
    var win = bernini_task_window(shift, NOISE_TMIN, NOISE_TMAX)
    var lo = Float64(2.0)
    var hi = Float64(-2.0)
    var s = Float64(0.0)
    var ss = Float64(0.0)
    var nb = 14
    var counts = List[Int]()
    for _ in range(nb):
        counts.append(0)
    var hlo = Float64(0.86)
    var hhi = Float64(1.0)
    for i in range(n):
        var seed = UInt64(0x9E3779B97F4A7C15) + UInt64(i) * UInt64(2654435761) + 1
        var sig = Float64(bernini_sample_sigma(
            is_mode, shift, seed, LOGIT_MEAN, LOGIT_STD, MODE_SCALE, win
        ))
        s += sig
        ss += sig * sig
        if sig < lo:
            lo = sig
        if sig > hi:
            hi = sig
        var b = Int((sig - hlo) / (hhi - hlo) * Float64(nb))
        if b < 0:
            b = 0
        if b >= nb:
            b = nb - 1
        counts[b] += 1
    var mean = s / Float64(n)
    var var_ = ss / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    var std = var_ ** 0.5
    var wname = String("mode") if is_mode else String("logit_normal")
    print("")
    print("[", wname, " shift=", shift, " tasks=", tasks, "]")
    print("  window u in [tmin=", win.tmin, ", tmax=", win.tmax, "]")
    print("  sigma mean=", mean, " std=", std, " min=", lo, " max=", hi)
    var line = String("  hist[0.86..1.0 /14]: [")
    for i in range(nb):
        line += String(counts[i])
        if i < nb - 1:
            line += String(", ")
    line += String("]")
    print(line)


def main() raises:
    var N = 200000
    print("=== BERNINI-R timestep parity (Mojo; N=", N, " per case) ===")
    _case(False, 3.0, String("t2i"), N)
    _case(False, 4.0, String("i2i"), N)
    _case(True, 3.0, String("t2v"), N)
    _case(True, 4.0, String("r2v"), N)
    _case(True, 5.0, String("i2v/v2v"), N)

    print("")
    print("=== deterministic anchors ===")
    for k in range(3):
        var raw = Float64(0.1) if k == 0 else (Float64(0.5) if k == 1 else Float64(0.9))
        print("  mode_density(raw=", raw, ") = ",
              bernini_mode_density_from_raw(raw, MODE_SCALE))
    for k in range(3):
        var z = Float64(-1.0) if k == 0 else (Float64(0.0) if k == 1 else Float64(1.0))
        print("  logit_density(z=", z, ") = ",
              bernini_logit_normal_density_from_z(z, LOGIT_MEAN, LOGIT_STD))
    var shifts = List[Float64]()
    shifts.append(3.0)
    shifts.append(4.0)
    shifts.append(5.0)
    var idxs = List[Int]()
    idxs.append(0)
    idxs.append(50)
    idxs.append(125)
    for si in range(len(shifts)):
        for ii in range(len(idxs)):
            print("  shifted_sigma(shift=", shifts[si], ", idx=", idxs[ii], ") = ",
                  bernini_shifted_sigma(shifts[si], idxs[ii]))
