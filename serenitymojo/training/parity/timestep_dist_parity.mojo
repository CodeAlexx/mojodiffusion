# training/parity/timestep_dist_parity.mojo — timestep-dist parity gate (item 2g).
#
# Gates the Wave 2A selectable distributions in training/schedule.mojo
# (sample_timestep_uniform, sample_timestep_sigmoid) against the ANALYTIC
# reference distribution the EDv2 Rust draw (timestep_dist.rs) converges to:
#   Uniform : flat density on [0,1]  -> flat histogram.
#   Sigmoid : t = sigmoid(w*z), z~N(0,1) -> logistic-of-normal density, computed
#             by numerically pushing a fine N(0,1) z-grid through sigmoid(w*z)
#             and binning (this IS the reference distribution).
# Match criterion: cosine of (Mojo sample histogram) vs (reference histogram)
# >= 0.999, per the task. Also asserts mean within band.
#
# BITROT GUARD: argv "--bitrot" poisons the reference histogram -> EXIT NONZERO.
#
# Run:
#   cd /home/alex/mojodiffusion && (rm -f serenitymojo.mojopkg) && \
#     pixi run mojo run -I . serenitymojo/training/parity/timestep_dist_parity.mojo
#
# Mojo 1.0.0b1. Loud-fail.

import sys
from std.math import sqrt, exp, pi
from serenitymojo.training.schedule import (
    sample_timestep_uniform, sample_timestep_sigmoid,
)


comptime _NBINS = 50
comptime _SQRT_2PI = Float64(2.5066282746310002)


def _cos(a: List[Float64], b: List[Float64]) -> Float64:
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (sqrt(na) * sqrt(nb))


def _normalize(h: List[Float64]) -> List[Float64]:
    var s = Float64(0.0)
    for i in range(len(h)):
        s += h[i]
    var out = List[Float64]()
    for i in range(len(h)):
        out.append(h[i] / s)
    return out^


def main() raises:
    var args = sys.argv()
    var bitrot = len(args) > 1 and args[1] == String("--bitrot")
    print("=== timestep_dist parity (item 2g) ===")
    if bitrot:
        print("  [BITROT MODE] poisoning the reference histogram")

    var N = 200000

    # ── Uniform: sample histogram vs flat reference ──────────────────────────
    var uhist = List[Float64]()
    for _ in range(_NBINS):
        uhist.append(Float64(0.0))
    var usum = Float64(0.0)
    for i in range(N):
        var t = Float64(sample_timestep_uniform(UInt64(i)))
        usum += t
        var b = Int(t * Float64(_NBINS))
        if b < 0:
            b = 0
        if b >= _NBINS:
            b = _NBINS - 1
        uhist[b] = uhist[b] + Float64(1.0)
    var umean = usum / Float64(N)
    var uref = List[Float64]()
    for _ in range(_NBINS):
        uref.append(Float64(1.0))  # flat
    if bitrot:
        uref[0] = uref[0] * Float64(100.0)  # spike one bin
    var c_uni = _cos(_normalize(uhist), _normalize(uref))
    print("uniform mean (want ~0.5) =", umean)
    print("uniform histogram cos    =", c_uni)
    if umean < Float64(0.49) or umean > Float64(0.51):
        print("  FAIL uniform mean out of [0.49,0.51]")
        raise Error("uniform mean fail")
    if c_uni < Float64(0.999):
        print("  FAIL uniform histogram cos < 0.999")
        raise Error("uniform dist parity fail")

    # ── Sigmoid(w=1.8, b=0): sample histogram vs analytic logistic-normal ────
    var weight = Float32(1.8)
    var bias = Float32(0.0)
    var shist = List[Float64]()
    for _ in range(_NBINS):
        shist.append(Float64(0.0))
    var ssum = Float64(0.0)
    for i in range(N):
        var t = Float64(sample_timestep_sigmoid(UInt64(i), weight, bias))
        ssum += t
        var b = Int(t * Float64(_NBINS))
        if b < 0:
            b = 0
        if b >= _NBINS:
            b = _NBINS - 1
        shist[b] = shist[b] + Float64(1.0)
    var smean = ssum / Float64(N)

    # Reference: push a fine N(0,1) z-grid through t=sigmoid(w*z), bin it. This
    # is the distribution the Rust draw (z~N(0,1) then sigmoid(w*z)) converges to.
    var sref = List[Float64]()
    for _ in range(_NBINS):
        sref.append(Float64(0.0))
    var w64 = Float64(weight)
    var GZ = 400000
    var zmin = Float64(-8.0)
    var zmax = Float64(8.0)
    var dz = (zmax - zmin) / Float64(GZ)
    for gi in range(GZ):
        var z = zmin + (Float64(gi) + Float64(0.5)) * dz
        var pdf = exp(-Float64(0.5) * z * z) / _SQRT_2PI  # N(0,1) density
        var t = Float64(1.0) / (Float64(1.0) + exp(-(w64 * z)))  # sigmoid(w*z)
        var b = Int(t * Float64(_NBINS))
        if b < 0:
            b = 0
        if b >= _NBINS:
            b = _NBINS - 1
        sref[b] = sref[b] + pdf * dz  # mass = density * dz
    if bitrot:
        sref[_NBINS // 2] = sref[_NBINS // 2] * Float64(50.0)  # poison center bin
    var c_sig = _cos(_normalize(shist), _normalize(sref))
    print("sigmoid mean (want ~0.5) =", smean)
    print("sigmoid histogram cos    =", c_sig)
    if smean < Float64(0.48) or smean > Float64(0.52):
        print("  FAIL sigmoid mean out of [0.48,0.52]")
        raise Error("sigmoid mean fail")
    if c_sig < Float64(0.999):
        print("  FAIL sigmoid histogram cos < 0.999")
        raise Error("sigmoid dist parity fail")

    # ── Sigmoid bias shift sanity: bias>0 pushes mean up ─────────────────────
    var ssum_pos = Float64(0.0)
    var Np = 50000
    for i in range(Np):
        ssum_pos += Float64(sample_timestep_sigmoid(UInt64(i), weight, Float32(1.0)))
    var mean_pos = ssum_pos / Float64(Np)
    print("sigmoid mean bias=+1 (want >0.5) =", mean_pos)
    if mean_pos <= Float64(0.5):
        print("  FAIL sigmoid bias=+1 did not push mean up")
        raise Error("sigmoid bias shift fail")

    print("PASS timestep_dist parity")
