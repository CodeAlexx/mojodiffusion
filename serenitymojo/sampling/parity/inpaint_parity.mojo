# sampling/parity/inpaint_parity.mojo — mask-blend + overdamped LanPaint step
# parity gate against a hand-computed oracle, all on synthetic latents.
#
# Self-contained: no model weights. Three checks:
#   (1) mask_blend endpoints exact: mask=1 → base, mask=0 → denoised.
#   (2) mask_blend with a mixed mask: cos >= 0.999 vs hand oracle, AND exact
#       per-lane (within F32 tol).
#   (3) lanpaint_overdamped_step: one step vs an open-coded scalar oracle of
#       the same OU closed form, cos >= 0.999 and exact per-lane.
#
# PARITY-BITROT GUARD: pass `--bitrot` to flip the blend convention in the
# *oracle* (uses (1-mask) for base instead of mask). The module's (correct)
# output then disagrees → assertion fires → exit 1. Demonstrated below.
#
# Build / run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/sampling/parity/inpaint_parity.mojo
#   (append `--bitrot` for the deliberate-wrong exit-1 demo)

from std.collections import List
from sys import argv
from math import exp, sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.sampling.inpaint import (
    mask_blend,
    lanpaint_overdamped_step,
)


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def _cos(a: List[Float32], b: List[Float32]) -> Float32:
    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    var denom = sqrt(na) * sqrt(nb)
    if denom == 0.0:
        return 1.0
    return dot / denom


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var d = _abs(got - expected)
    if d > tol:
        raise Error(
            name + " got=" + String(got) + " expected=" + String(expected)
            + " |Δ|=" + String(d)
        )


def _shape(n: Int) -> List[Int]:
    var s = List[Int]()
    s.append(n)
    return s^


def main() raises:
    var args = argv()
    var bitrot = False
    for i in range(len(args)):
        if args[i] == "--bitrot":
            bitrot = True

    var ctx = DeviceContext()
    print("=== inpaint mask-blend + LanPaint overdamped parity ===" + (" [BITROT]" if bitrot else ""))

    var N = 8
    # Synthetic latents.
    var base_v = List[Float32]()
    var den_v = List[Float32]()
    for i in range(N):
        base_v.append(Float32(i) * 0.25 - 0.5)        # base latent
        den_v.append(Float32(N - i) * -0.3 + 0.7)     # denoised latent
    # Mixed mask: alternating 1/0, plus a fractional value to test lerp.
    var mask_v = List[Float32]()
    for i in range(N):
        if i == 0:
            mask_v.append(0.25)        # fractional → genuine lerp
        elif i % 2 == 0:
            mask_v.append(1.0)
        else:
            mask_v.append(0.0)

    var base = Tensor.from_host(base_v, _shape(N), STDtype.F32, ctx)
    var den = Tensor.from_host(den_v, _shape(N), STDtype.F32, ctx)
    var mask = Tensor.from_host(mask_v, _shape(N), STDtype.F32, ctx)

    # ---- (1) + (2) mask_blend ----
    var blended = mask_blend(mask, base, den, ctx)
    var blended_h = blended.to_host(ctx)

    var oracle_blend = List[Float32]()
    for i in range(N):
        var m = mask_v[i]
        # BITROT: swap which side the mask weights → wrong oracle.
        if bitrot:
            oracle_blend.append((1.0 - m) * base_v[i] + m * den_v[i])
        else:
            oracle_blend.append(m * base_v[i] + (1.0 - m) * den_v[i])

    var cos_blend = _cos(blended_h, oracle_blend)
    print("  blend cos vs oracle = " + String(cos_blend))
    if cos_blend < 0.999:
        raise Error("blend cos < 0.999 (got " + String(cos_blend) + ")")
    for i in range(N):
        _check_close("blend lane" + String(i), blended_h[i], oracle_blend[i], 1.0e-4)

    # Endpoint exactness: mask=1 → base (lane 2), mask=0 → denoised (lane 1).
    _check_close("endpoint mask=1 (lane2)", blended_h[2], base_v[2], 1.0e-6)
    _check_close("endpoint mask=0 (lane1)", blended_h[1], den_v[1], 1.0e-6)
    print("  endpoints exact: mask=1→base, mask=0→denoised  OK")
    print("  mask_blend: tensor == oracle  OK")

    # ---- (3) lanpaint_overdamped_step ----
    var x_v = List[Float32]()
    var score_v = List[Float32]()
    var noise_v = List[Float32]()
    for i in range(N):
        x_v.append(Float32(i) * 0.1 - 0.3)
        score_v.append(Float32(i) * -0.05 + 0.2)
        noise_v.append(Float32(i % 3) * 0.4 - 0.4)

    var x_t = Tensor.from_host(x_v, _shape(N), STDtype.F32, ctx)
    var score = Tensor.from_host(score_v, _shape(N), STDtype.F32, ctx)
    var noise = Tensor.from_host(noise_v, _shape(N), STDtype.F32, ctx)

    var a_drift: Float32 = 1.5
    var dt: Float32 = 0.3
    var abt: Float32 = 0.4

    var x_next = lanpaint_overdamped_step(x_t, score, noise, a_drift, dt, abt, ctx)
    var x_next_h = x_next.to_host(ctx)

    # Scalar oracle of the identical OU closed form.
    var sqrt_abt = sqrt(abt)
    var one_minus_abt = 1.0 - abt
    var a_dt = a_drift * dt
    var exp_neg = exp(-a_dt)
    var k = (1.0 - exp(-a_dt)) / a_drift
    var k2 = (1.0 - exp(-2.0 * a_dt)) / (2.0 * a_drift)
    var variance: Float32 = 2.0 * k2
    var sd: Float32 = 0.0
    if variance > 0.0:
        sd = sqrt(variance)

    var oracle_step = List[Float32]()
    for i in range(N):
        var x0 = x_v[i] + score_v[i]
        var c = (sqrt_abt * x0 - x_v[i]) / one_minus_abt + a_drift * x_v[i]
        var mean = exp_neg * x_v[i] + k * c
        oracle_step.append(mean + sd * noise_v[i])

    var cos_step = _cos(x_next_h, oracle_step)
    print("  lanpaint step cos vs oracle = " + String(cos_step))
    if cos_step < 0.999:
        raise Error("lanpaint step cos < 0.999 (got " + String(cos_step) + ")")
    for i in range(N):
        _check_close("lanpaint lane" + String(i), x_next_h[i], oracle_step[i], 1.0e-4)
    print("  lanpaint_overdamped_step: tensor == oracle  OK")

    if x_next.dtype() != STDtype.F32:
        raise Error("lanpaint step must preserve F32 latent dtype")

    print("PASS: inpaint mask-blend + LanPaint overdamped step parity")
