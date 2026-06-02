# sampling/parity/dpmpp_2m_parity.mojo — DPM++ 2M convergence-order gate.
#
# Self-contained scalar parity gate (NO model, NO device tensors). Ports the
# reference test `dpmpp_2m_convergence_order_honest` from
#   inference-flame/src/sampling/exponential_multistep.rs
#
# Method (verbatim from the reference):
#   * Toy ODE  dx/dσ = (x − D(σ))/σ  with the smooth oracle D(σ) = cos(σ),
#     integrated from σ = 0.9 → σ = 0.1 starting at x(0.9) = 0.1.
#   * Reference x(σ_end) produced by adaptive RK4 with 10_000 substeps on the
#     SAME ODE / SAME oracle.
#   * The DPM++ 2M scheme is then run at N = 10, 20, 40 steps; the global
#     errors e1, e2, e3 and their ratios r1 = e1/e2, r2 = e2/e3 are measured.
#   * 2nd-order ⇒ ratio ≈ 4. The gate requires best(r1, r2) ≥ 3.5 AND a total
#     error reduction e1/e3 > 10 (confirming we are in the convergence regime).
#
# The scheme's per-step coefficients come from `dpmpp_2m_coeffs` (the SAME f64
# scalar function the tensor `dpmpp_2m_step` uses), so this gate validates the
# exact production math, not a re-derivation.
#
# PARITY-BITROT GUARD: run with `--bitrot` to deliberately corrupt the
# 2nd-order correction (drop the `denoised_prev` term → 1st-order). That MUST
# break the convergence-order assertion and exit nonzero, proving the gate has
# teeth. The normal run exits 0.
#
# Build / run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/sampling/parity/dpmpp_2m_parity.mojo
#   (add a trailing `--bitrot` arg for the deliberate-wrong demo)

from collections import List
from std.math import cos, log
from sys import argv

from serenitymojo.sampling.dpmpp_2m import (
    dpmpp_2m_coeffs,
    lambda_from_sigma_f64,
)


def _abs(v: Float64) -> Float64:
    return v if v >= 0.0 else -v


def _denoised_oracle(sigma: Float64) -> Float64:
    return cos(sigma)


# RK4 reference on dx/dσ = (x − D(σ))/σ (σ decreases; dσ < 0).
def _rk4_reference(
    sigma_start: Float64, sigma_end: Float64, x_init: Float64, n_ref: Int
) -> Float64:
    var dsig = (sigma_end - sigma_start) / Float64(n_ref)  # negative
    var sigma = sigma_start
    var x = x_init
    for _ in range(n_ref):
        var k1 = (x - _denoised_oracle(sigma)) / sigma
        var k2 = (
            (x + 0.5 * dsig * k1) - _denoised_oracle(sigma + 0.5 * dsig)
        ) / (sigma + 0.5 * dsig)
        var k3 = (
            (x + 0.5 * dsig * k2) - _denoised_oracle(sigma + 0.5 * dsig)
        ) / (sigma + 0.5 * dsig)
        var k4 = (
            (x + dsig * k3) - _denoised_oracle(sigma + dsig)
        ) / (sigma + dsig)
        x += (dsig / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        sigma += dsig
    return x


# One DPM++ 2M scalar update using the production `dpmpp_2m_coeffs`.
# `bitrot` drops the 2nd-order `denoised_prev` term, collapsing the scheme to
# 1st order — used to prove the gate fails on wrong math.
def _dpmpp_2m_scalar(
    x: Float64,
    denoised: Float64,
    sigma: Float64,
    sigma_next: Float64,
    mut hist_denoised: List[Float64],
    mut hist_lambda: List[Float64],
    bitrot: Bool,
) -> Float64:
    var have_hist = len(hist_denoised) > 0
    var lam_prev: Float64 = 0.0
    if have_hist:
        lam_prev = hist_lambda[len(hist_lambda) - 1]
    var c = dpmpp_2m_coeffs(sigma, sigma_next, have_hist, lam_prev)

    var out = c.c_x * x + c.c_d * denoised
    if c.used_history and not bitrot:
        var d_prev = hist_denoised[len(hist_denoised) - 1]
        out += c.c_p * d_prev
    # When bitrot=True we also strip the c_d 2nd-order boost so the step is a
    # pure 1st-order data-pred step (c_x·x + α_next·(1-e^{-h})·denoised). Recompute
    # the 1st-order c_d by asking dpmpp_2m_coeffs with have_history=False.
    if bitrot and c.used_history:
        var c1 = dpmpp_2m_coeffs(sigma, sigma_next, False, 0.0)
        out = c1.c_x * x + c1.c_d * denoised

    # Push history (newest last), keep capacity 1.
    var lam_now = lambda_from_sigma_f64(sigma)
    hist_denoised.append(denoised)
    hist_lambda.append(lam_now)
    if len(hist_denoised) > 1:
        var nd = List[Float64]()
        var nl = List[Float64]()
        for i in range(1, len(hist_denoised)):
            nd.append(hist_denoised[i])
            nl.append(hist_lambda[i])
        hist_denoised = nd^
        hist_lambda = nl^
    return out


def _run(
    n: Int,
    sigma_start: Float64,
    sigma_end: Float64,
    x_init: Float64,
    x_ref: Float64,
    bitrot: Bool,
) -> Float64:
    var sigmas = List[Float64]()
    for i in range(n + 1):
        sigmas.append(
            sigma_start + (sigma_end - sigma_start) * (Float64(i) / Float64(n))
        )
    var x = x_init
    var hist_denoised = List[Float64]()
    var hist_lambda = List[Float64]()
    for i in range(n):
        var s = sigmas[i]
        var sn = sigmas[i + 1]
        var d = _denoised_oracle(s)
        x = _dpmpp_2m_scalar(
            x, d, s, sn, hist_denoised, hist_lambda, bitrot
        )
    return _abs(x - x_ref)


def main() raises:
    var args = argv()
    var bitrot = False
    for i in range(len(args)):
        if args[i] == "--bitrot":
            bitrot = True

    var sigma_start = 0.9
    var sigma_end = 0.1
    var x_init = 0.1

    var x_ref = _rk4_reference(sigma_start, sigma_end, x_init, 10000)

    var e1 = _run(10, sigma_start, sigma_end, x_init, x_ref, bitrot)
    var e2 = _run(20, sigma_start, sigma_end, x_init, x_ref, bitrot)
    var e3 = _run(40, sigma_start, sigma_end, x_init, x_ref, bitrot)
    var r1 = e1 / e2
    var r2 = e2 / e3
    var r_best = r1 if r1 > r2 else r2

    print("DPM++ 2M convergence-order gate" + (" [BITROT]" if bitrot else ""))
    print("  x_ref          =", x_ref)
    print("  e1 (N=10)      =", e1)
    print("  e2 (N=20)      =", e2)
    print("  e3 (N=40)      =", e3)
    print("  ratio r1=e1/e2 =", r1)
    print("  ratio r2=e2/e3 =", r2)
    print("  best ratio     =", r_best)
    print("  total e1/e3    =", e1 / e3)

    # 2nd-order target ≈ 4; accept best ratio ≥ 3.5 (startup step is 1st-order).
    if not (r_best >= 3.5):
        raise Error(
            String("DPM++ 2M: best convergence ratio ")
            + String(r_best)
            + " < 3.5 (not 2nd-order) — e1="
            + String(e1)
            + " e2="
            + String(e2)
            + " e3="
            + String(e3)
        )
    # Confirm we are in the convergence regime (not roundoff-limited).
    if not (e1 / e3 > 10.0):
        raise Error(
            String("DPM++ 2M: error reduction e1/e3 ")
            + String(e1 / e3)
            + " <= 10 (not converging)"
        )
    print("PASS: DPM++ 2M shows 2nd-order convergence (best ratio >= 3.5, e1/e3 > 10)")
