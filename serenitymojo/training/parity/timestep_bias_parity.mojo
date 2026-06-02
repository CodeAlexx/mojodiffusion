# training/parity/timestep_bias_parity.mojo — timestep-bias parity gate (item 2f).
#
# Gates training/timestep_bias.mojo apply_bias against a host F64 re-derivation
# of the EDv2 Rust formula (timestep_bias.rs:90). Tolerance 1e-6 (rel).
# Asserts TSB_NONE is the exact identity.
#
# BITROT GUARD: argv "--bitrot" poisons expected value -> gate EXITS NONZERO.
#
# Run:
#   cd /home/alex/mojodiffusion && (rm -f serenitymojo.mojopkg) && \
#     pixi run mojo run -I . serenitymojo/training/parity/timestep_bias_parity.mojo
#
# Mojo 1.0.0b1. Loud-fail.

import sys
from serenitymojo.training.timestep_bias import (
    apply_bias, TSB_NONE, TSB_LATER, TSB_EARLIER, TSB_RANGE,
)


comptime _TOL = Float64(1.0e-6)


def _clamp01_64(x: Float64) -> Float64:
    if x < Float64(0.0):
        return Float64(0.0)
    if x > Float64(1.0):
        return Float64(1.0)
    return x


def _ref(
    t: Float64, total: Float64, strat: Int, m: Float64, rmin: Float64, rmax: Float64
) -> Float64:
    if strat == TSB_NONE:
        return t
    if strat == TSB_LATER:
        var mm = _clamp01_64(m)
        return t + mm * (total - t)
    if strat == TSB_EARLIER:
        var mm = _clamp01_64(m)
        return t * (Float64(1.0) - mm)
    if strat == TSB_RANGE:
        var lo = _clamp01_64(rmin)
        var hi = _clamp01_64(rmax)
        if hi < lo:
            hi = lo
        var frac = _clamp01_64(t / total)
        return (lo + frac * (hi - lo)) * total
    return t


def main() raises:
    var args = sys.argv()
    var bitrot = len(args) > 1 and args[1] == String("--bitrot")
    print("=== timestep_bias parity (item 2f) ===")
    if bitrot:
        print("  [BITROT MODE] feeding deliberately-wrong expected value")

    var total = Float32(1000.0)
    var ts = [Float32(0.0), Float32(1.0), Float32(200.0), Float32(500.0),
              Float32(800.0), Float32(999.0), Float32(1000.0)]
    var mults = [Float32(0.0), Float32(0.25), Float32(0.5), Float32(1.0), Float32(1.5)]
    var max_err = Float64(0.0)
    var n = 0

    # TSB_LATER and TSB_EARLIER over t x multiplier.
    var strats_m = [TSB_LATER, TSB_EARLIER]
    for sj in range(len(strats_m)):
        var strat = strats_m[sj]
        for mi in range(len(mults)):
            var m = mults[mi]
            for ti in range(len(ts)):
                var t = ts[ti]
                var got = Float64(apply_bias(t, total, strat, m, Float32(0.0), Float32(1.0)))
                var want = _ref(Float64(t), Float64(total), strat, Float64(m), Float64(0.0), Float64(1.0))
                if bitrot:
                    want = want + Float64(1.0)
                var dv = abs(want)
                if dv < Float64(1.0):
                    dv = Float64(1.0)
                var err = abs(got - want) / dv
                if err > max_err:
                    max_err = err
                n += 1
                if err > _TOL:
                    print("  FAIL strat=", strat, " m=", m, " t=", t, " got=", got, " want=", want)
                    raise Error("timestep_bias parity fail")

    # TSB_RANGE over t x (range_min, range_max) incl. collapsed + inverted bounds.
    var ranges = [(Float32(0.2), Float32(0.8)), (Float32(0.5), Float32(0.5)),
                  (Float32(0.0), Float32(1.0)), (Float32(0.8), Float32(0.2))]
    for ri in range(len(ranges)):
        var r = ranges[ri]
        for ti in range(len(ts)):
            var t = ts[ti]
            var got = Float64(apply_bias(t, total, TSB_RANGE, Float32(0.0), r[0], r[1]))
            var want = _ref(Float64(t), Float64(total), TSB_RANGE, Float64(0.0), Float64(r[0]), Float64(r[1]))
            if bitrot:
                want = want + Float64(1.0)
            var dv = abs(want)
            if dv < Float64(1.0):
                dv = Float64(1.0)
            var err = abs(got - want) / dv
            if err > max_err:
                max_err = err
            n += 1
            if err > _TOL:
                print("  FAIL RANGE r=(", r[0], ",", r[1], ") t=", t, " got=", got, " want=", want)
                raise Error("timestep_bias RANGE parity fail")

    print("checks =", n)
    print("max rel err =", max_err, " (tol", _TOL, ")")

    # ── Default-off: TSB_NONE is the EXACT identity (to_bits equality) ───────
    var any_bad = False
    for ti in range(len(ts)):
        var t = ts[ti]
        var got = apply_bias(t, total, TSB_NONE, Float32(0.5), Float32(0.2), Float32(0.8))
        if got != t:
            print("  FAIL TSB_NONE not identity: t=", t, " got=", got)
            any_bad = True
    if any_bad:
        raise Error("TSB_NONE identity fail")
    print("default-off TSB_NONE == identity: OK")

    print("PASS timestep_bias parity")
