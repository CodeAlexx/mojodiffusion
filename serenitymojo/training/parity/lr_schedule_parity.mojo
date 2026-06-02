# training/parity/lr_schedule_parity.mojo — LR scheduler parity gate (item 2a).
#
# Gates training/lr_schedule.mojo against a host F64 re-derivation of the EDv2
# Rust formula (lr_schedule.rs). The F64 host computation IS the oracle (same
# discipline as schedule_parity.mojo / optim_parity.mojo). Tolerance 1e-6.
#
# BITROT GUARD: pass argv "--bitrot" to feed a deliberately-wrong expected value
# and prove the gate EXITS NONZERO. Run without args for the real PASS.
#
# Run:
#   cd /home/alex/mojodiffusion && (rm -f serenitymojo.mojopkg) && \
#     pixi run mojo run -I . serenitymojo/training/parity/lr_schedule_parity.mojo
#   # bitrot demo:
#   ... lr_schedule_parity.mojo --bitrot
#
# Mojo 1.0.0b1. Loud-fail: prints FAIL + raises on any arm over tolerance.

import sys
from std.math import cos, pi
from serenitymojo.training.lr_schedule import (
    lr_for_step,
    constant_lr,
    LR_CONSTANT, LR_LINEAR, LR_COSINE, LR_COSINE_RESTARTS, LR_POLYNOMIAL, LR_REX,
)


comptime _PI64 = Float64(3.14159265358979323846)
comptime _TOL = Float64(1.0e-6)


def _progress64(step: Int, total: Int, warmup: Int) -> Float64:
    var denom = total - warmup
    if denom < 1:
        denom = 1
    var p = Float64(step - warmup) / Float64(denom)
    if p < Float64(0.0):
        return Float64(0.0)
    if p > Float64(1.0):
        return Float64(1.0)
    return p


def _constant64(base: Float64, step: Int, warmup: Int) -> Float64:
    if warmup == 0 or step >= warmup:
        return base
    return base * (Float64(step) + Float64(1.0)) / Float64(warmup)


def _ref(
    base: Float64, step: Int, warmup: Int, total: Int, kind: Int,
    min_factor: Float64, cycles: Float64, power: Float64,
) -> Float64:
    if kind == LR_CONSTANT:
        return _constant64(base, step, warmup)
    if step < warmup:
        return _constant64(base, step, warmup)
    var p = _progress64(step, total, warmup)
    if kind == LR_LINEAR:
        return base * (Float64(1.0) - (Float64(1.0) - min_factor) * p)
    if kind == LR_COSINE:
        var cf = Float64(0.5) * (Float64(1.0) + cos(_PI64 * p))
        return base * (min_factor + (Float64(1.0) - min_factor) * cf)
    if kind == LR_COSINE_RESTARTS:
        var c = cycles
        if c < Float64(1.0):
            c = Float64(1.0)
        var cp = p * c
        cp = cp - Float64(Int(cp))
        var cf = Float64(0.5) * (Float64(1.0) + cos(_PI64 * cp))
        return base * (min_factor + (Float64(1.0) - min_factor) * cf)
    if kind == LR_POLYNOMIAL:
        var f = (Float64(1.0) - p) ** power
        return base * (min_factor + (Float64(1.0) - min_factor) * f)
    if kind == LR_REX:
        var f = (Float64(1.0) - p) / (Float64(1.0) - Float64(0.5) * p)
        if f < Float64(0.0):
            f = Float64(0.0)
        return base * (min_factor + (Float64(1.0) - min_factor) * f)
    return _constant64(base, step, warmup)


def main() raises:
    var args = sys.argv()
    var bitrot = len(args) > 1 and args[1] == String("--bitrot")
    print("=== lr_schedule parity (item 2a) ===")
    if bitrot:
        print("  [BITROT MODE] feeding deliberately-wrong expected value")

    var base = Float32(3.0e-5)
    var base64 = Float64(base)
    var total = 3000
    var max_err = Float64(0.0)
    var n_checked = 0

    # Sweep kinds x warmups x steps. min_factor/cycles/power vary per kind.
    var kinds = [LR_CONSTANT, LR_LINEAR, LR_COSINE, LR_COSINE_RESTARTS, LR_POLYNOMIAL, LR_REX]
    var warmups = [0, 1, 50, 200]
    var steps = [0, 1, 5, 49, 99, 200, 750, 1499, 1500, 2999, 3000, 5000]
    var min_factor = Float32(0.1)
    var cycles = Float32(2.0)
    var power = Float32(2.0)

    for ki in range(len(kinds)):
        var kind = kinds[ki]
        for wi in range(len(warmups)):
            var warmup = warmups[wi]
            for si in range(len(steps)):
                var step = steps[si]
                var got = Float64(lr_for_step(
                    base, step, warmup, total, kind, min_factor, cycles, power
                ))
                var want = _ref(
                    base64, step, warmup, total, kind,
                    Float64(min_factor), Float64(cycles), Float64(power),
                )
                if bitrot:
                    want = want + Float64(1.0)  # poison
                var err = abs(got - want)
                if err > max_err:
                    max_err = err
                n_checked += 1
                if err > _TOL:
                    print("  FAIL kind=", kind, " warmup=", warmup, " step=", step,
                          " got=", got, " want=", want, " err=", err)
                    raise Error("lr_schedule parity fail")

    print("checks =", n_checked)
    print("max abs err =", max_err, " (tol", _TOL, ")")

    # ── Default-off byte-invariance: LR_CONSTANT warmup=0 == flat base_lr ─────
    var any_bad = False
    for si in range(len(steps)):
        var step = steps[si]
        var v = lr_for_step(base, step, 0, total, LR_CONSTANT,
                            Float32(0.0), Float32(1.0), Float32(2.0))
        # to_bits exact-equality against the flat base.
        if v.cast[DType.float32]() != base:
            print("  FAIL default-off LR_CONSTANT warmup=0 step=", step, " v=", v)
            any_bad = True
    if any_bad:
        raise Error("default-off invariance fail")
    print("default-off LR_CONSTANT warmup=0 == base_lr: OK")

    print("PASS lr_schedule parity")
