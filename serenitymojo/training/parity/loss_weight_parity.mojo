# training/parity/loss_weight_parity.mojo — loss-weight parity gate (item 2b).
#
# Gates training/loss_weight.mojo (min_snr_weight, debiased_weight,
# apply_loss_weight) against a host F64 re-derivation of the EDv2 Rust formula
# (loss_weight.rs). The F64 host computation IS the oracle. Tolerance 1e-6.
#
# BITROT GUARD: argv "--bitrot" poisons the expected value -> gate EXITS NONZERO.
#
# Run:
#   cd /home/alex/mojodiffusion && (rm -f serenitymojo.mojopkg) && \
#     pixi run mojo run -I . serenitymojo/training/parity/loss_weight_parity.mojo
#
# Mojo 1.0.0b1. Loud-fail.

import sys
from std.math import sqrt
from serenitymojo.training.loss_weight import (
    min_snr_weight, debiased_weight, apply_loss_weight,
)


comptime _TOL = Float64(1.0e-6)


def _snr64(sigma: Float64) -> Float64:
    var s = sigma
    if s < Float64(1.0e-8):
        s = Float64(1.0e-8)
    var r = (Float64(1.0) - s) / s
    return r * r


def _min_snr64(sigma: Float64, gamma: Float64, v: Bool) -> Float64:
    var snr = _snr64(sigma)
    var cap = snr
    if gamma < cap:
        cap = gamma
    if v:
        return cap / (snr + Float64(1.0))
    var d = snr
    if d < Float64(1.0e-8):
        d = Float64(1.0e-8)
    return cap / d


def _debiased64(sigma: Float64, v: Bool) -> Float64:
    var snr = _snr64(sigma)
    var c = snr
    if c > Float64(1.0e3):
        c = Float64(1.0e3)
    var adj = c
    if v:
        adj = c + Float64(1.0)
    var root = sqrt(adj)
    if root < Float64(1.0e-8):
        root = Float64(1.0e-8)
    return Float64(1.0) / root


def main() raises:
    var args = sys.argv()
    var bitrot = len(args) > 1 and args[1] == String("--bitrot")
    print("=== loss_weight parity (item 2b) ===")
    if bitrot:
        print("  [BITROT MODE] feeding deliberately-wrong expected value")

    var sigmas = [Float32(0.001), Float32(0.01), Float32(0.1), Float32(0.25),
                  Float32(0.5), Float32(0.7), Float32(0.9), Float32(0.99), Float32(0.999)]
    var gammas = [Float32(1.0), Float32(5.0), Float32(20.0)]
    var max_err = Float64(0.0)
    var n = 0

    # ── min_snr_weight: v-pred AND eps-pred, over sigma x gamma ──────────────
    for vi in range(2):
        var v = vi == 1
        for gi in range(len(gammas)):
            var gamma = gammas[gi]
            for si in range(len(sigmas)):
                var sigma = sigmas[si]
                var got = Float64(min_snr_weight(sigma, gamma, v))
                var want = _min_snr64(Float64(sigma), Float64(gamma), v)
                if bitrot:
                    want = want + Float64(1.0)
                # Relative error: the Rust reference is itself F32, so for large
                # weights (w~99) the F32 rounding floor exceeds a 1e-6 ABSOLUTE
                # tol while staying < 1e-6 RELATIVE. Compare rel err vs the F64
                # oracle (1e-6 = the F32 precision floor, not a math discrepancy).
                var denom = abs(want)
                if denom < Float64(1.0):
                    denom = Float64(1.0)
                var err = abs(got - want) / denom
                if err > max_err:
                    max_err = err
                n += 1
                if err > _TOL:
                    print("  FAIL min_snr v=", v, " gamma=", gamma, " sigma=", sigma,
                          " got=", got, " want=", want, " err=", err)
                    raise Error("min_snr parity fail")

    # ── debiased_weight: v-pred AND eps-pred over sigma ──────────────────────
    for vi in range(2):
        var v = vi == 1
        for si in range(len(sigmas)):
            var sigma = sigmas[si]
            var got = Float64(debiased_weight(sigma, v))
            var want = _debiased64(Float64(sigma), v)
            if bitrot:
                want = want + Float64(1.0)
            var denom = abs(want)
            if denom < Float64(1.0):
                denom = Float64(1.0)
            var err = abs(got - want) / denom
            if err > max_err:
                max_err = err
            n += 1
            if err > _TOL:
                print("  FAIL debiased v=", v, " sigma=", sigma,
                      " got=", got, " want=", want, " err=", err)
                raise Error("debiased parity fail")

    print("checks =", n)
    print("max rel err =", max_err, " (tol", _TOL, ")")

    # ── Default-off invariance: gamma<0 + debiased=False -> exactly 1.0 ──────
    var any_bad = False
    for si in range(len(sigmas)):
        var w = apply_loss_weight(sigmas[si], Float32(-1.0), False, True)
        if w != Float32(1.0):
            print("  FAIL default-off apply_loss_weight sigma=", sigmas[si], " w=", w)
            any_bad = True
    if any_bad:
        raise Error("default-off invariance fail")
    print("default-off apply_loss_weight(gamma<0, debiased=False) == 1.0: OK")

    # ── apply_loss_weight precedence: gamma_override wins over debiased ──────
    var w_ovr = apply_loss_weight(Float32(0.5), Float32(5.0), True, True)
    var w_ref = Float64(min_snr_weight(Float32(0.5), Float32(5.0), True))
    if abs(Float64(w_ovr) - w_ref) > _TOL:
        print("  FAIL apply precedence: override should win, got", w_ovr)
        raise Error("apply precedence fail")
    var w_deb = apply_loss_weight(Float32(0.5), Float32(-1.0), True, True)
    var w_deb_ref = Float64(debiased_weight(Float32(0.5), True))
    if abs(Float64(w_deb) - w_deb_ref) > _TOL:
        print("  FAIL apply debiased path, got", w_deb)
        raise Error("apply debiased fail")
    print("apply_loss_weight precedence (override>debiased>1.0): OK")

    print("PASS loss_weight parity")
