# sampling/parity/dpmpp_2m_tensor_smoke.mojo — DPM++ 2M TENSOR-path smoke.
#
# The scalar convergence gate (dpmpp_2m_parity.mojo) only imports the f64 free
# functions, so Mojo's lazy per-entry-point compilation never compiled the
# actual tensor inference entry points (`dpmpp_2m_step`, `MultistepHistory.push`,
# `denoised_from_velocity`). This smoke INSTANTIATES and RUNS them on real
# device tensors so the tensor paths are compiled, and asserts the GPU update
# matches the scalar `dpmpp_2m_coeffs` reference elementwise.
#
# Two steps are run so BOTH branches are exercised:
#   step 0 — empty history → 1st-order data-prediction update.
#   step 1 — history present → 2nd-order correction (c_p·denoised_prev term).
#
# Uses F32 tensors so the GPU result matches the f64 scalar coefficients to a
# tight tolerance (no BF16 rounding floor). PARITY-BITROT GUARD: `--bitrot`
# drops the 2nd-order history term in the *expected* scalar value, so the
# tensor (correct) output no longer matches → assertion fires → exit 1.
#
# Build / run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/sampling/parity/dpmpp_2m_tensor_smoke.mojo
#   (add a trailing `--bitrot` arg for the deliberate-wrong demo)

from collections import List
from sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.sampling.dpmpp_2m import (
    dpmpp_2m_step,
    dpmpp_2m_coeffs,
    lambda_from_sigma_f64,
    denoised_from_velocity,
    MultistepHistory,
)


def _shape4() -> List[Int]:
    var sh = List[Int]()
    sh.append(4)
    return sh^


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var d = _abs(got - expected)
    if d > tol:
        raise Error(
            name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
            + String(" |Δ|=")
            + String(d)
        )


def main() raises:
    var args = argv()
    var bitrot = False
    for i in range(len(args)):
        if args[i] == "--bitrot":
            bitrot = True

    var ctx = DeviceContext()
    print("=== DPM++ 2M tensor-path smoke ===" + (" [BITROT]" if bitrot else ""))

    # Two-step schedule.
    var sigma0: Float32 = 0.8
    var sigma1: Float32 = 0.5
    var sigma2: Float32 = 0.2

    # Host scalars (4 lanes, distinct values).
    var x_vals = List[Float32]()
    x_vals.append(0.30)
    x_vals.append(-0.70)
    x_vals.append(1.20)
    x_vals.append(-0.10)
    # Velocity at step 0 (model output); denoised0 = x - sigma0*v0.
    var v0_vals = List[Float32]()
    v0_vals.append(0.10)
    v0_vals.append(-0.20)
    v0_vals.append(0.50)
    v0_vals.append(0.05)
    # Velocity at step 1.
    var v1_vals = List[Float32]()
    v1_vals.append(-0.15)
    v1_vals.append(0.40)
    v1_vals.append(0.20)
    v1_vals.append(-0.30)

    var x = Tensor.from_host(x_vals, _shape4(), STDtype.F32, ctx)
    var v0 = Tensor.from_host(v0_vals, _shape4(), STDtype.F32, ctx)

    var history = MultistepHistory(1)

    # ---- STEP 0 (1st-order, empty history) ----
    var denoised0 = denoised_from_velocity(x, v0, sigma0, ctx)
    var d0_host = denoised0.to_host(ctx)
    var x1 = dpmpp_2m_step(x, denoised0, sigma0, sigma1, history, ctx)
    var x1_host = x1.to_host(ctx)
    var lam0 = lambda_from_sigma_f64(Float64(sigma0))
    history.push(denoised0^, lam0)

    # Scalar reference for step 0: 1st-order coeffs (no history).
    var c0 = dpmpp_2m_coeffs(Float64(sigma0), Float64(sigma1), False, 0.0)
    var x1_host_in = x.to_host(ctx)
    for i in range(4):
        var exp_x1 = Float32(c0.c_x) * x1_host_in[i] + Float32(c0.c_d) * d0_host[i]
        _check_close(String("step0 lane") + String(i), x1_host[i], exp_x1, 1.0e-4)
    print("  step 0 (1st-order, empty history): tensor == scalar  OK")

    # ---- STEP 1 (2nd-order, history present) ----
    var v1 = Tensor.from_host(v1_vals, _shape4(), STDtype.F32, ctx)
    var denoised1 = denoised_from_velocity(x1, v1, sigma1, ctx)
    var d1_host = denoised1.to_host(ctx)
    var x2 = dpmpp_2m_step(x1, denoised1, sigma1, sigma2, history, ctx)
    var x2_host = x2.to_host(ctx)

    # Scalar reference for step 1: 2nd-order coeffs (history = denoised0 @ lam0).
    var c1 = dpmpp_2m_coeffs(Float64(sigma1), Float64(sigma2), True, lam0)
    if not c1.used_history:
        raise Error("step1 expected 2nd-order (used_history) but got 1st-order")
    for i in range(4):
        var exp_x2 = (
            Float32(c1.c_x) * x1_host[i]
            + Float32(c1.c_d) * d1_host[i]
            + Float32(c1.c_p) * d0_host[i]
        )
        # BITROT: drop the 2nd-order history term from the expected value, so
        # the (correct) tensor output no longer matches → assertion must fire.
        if bitrot:
            exp_x2 = Float32(c1.c_x) * x1_host[i] + Float32(c1.c_d) * d1_host[i]
        _check_close(String("step1 lane") + String(i), x2_host[i], exp_x2, 1.0e-4)
    print("  step 1 (2nd-order, history): tensor == scalar  OK")

    if x2.dtype() != STDtype.F32:
        raise Error("DPM++ tensor step must preserve F32 latent dtype")

    print("PASS: DPM++ 2M tensor path compiles and matches scalar coefficients")
