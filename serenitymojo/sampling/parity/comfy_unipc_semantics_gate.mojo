# sampling/parity/comfy_unipc_semantics_gate.mojo
#
# Focused gate for generic Comfy `uni_pc` admission semantics. It does not
# claim tensor sampler parity. It proves the local Mojo surface keeps generic
# `uni_pc` distinct from the existing bh2/order-2 flow scheduler.
#
# Run:
#   pixi run mojo run -I . serenitymojo/sampling/parity/comfy_unipc_semantics_gate.mojo

from std.math import exp, log, sqrt

from serenitymojo.sampling.unipc import (
    alpha_from_sigma,
    build_comfy_unipc_timesteps,
    comfy_generic_unipc_b_h,
    comfy_generic_unipc_variant,
    comfy_sigma_convert_alpha,
    comfy_sigma_convert_lambda,
    comfy_sigma_convert_std,
    comfy_unipc_effective_order,
    comfy_unipc_initial_noise_scale,
)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _assert_close(label: String, got: Float64, expected: Float64, tol: Float64) raises:
    var err = _abs(got - expected)
    if err > tol:
        raise Error(
            label
            + ": got "
            + String(got)
            + " expected "
            + String(expected)
            + " err "
            + String(err)
        )


def main() raises:
    print("=== Comfy generic UniPC semantics gate ===")

    if comfy_generic_unipc_variant() != "bh1":
        raise Error("generic Comfy uni_pc variant must remain bh1")

    if comfy_unipc_effective_order(6) != 3:
        raise Error("generic Comfy uni_pc order for 6 timesteps must be 3")
    if comfy_unipc_effective_order(4) != 2:
        raise Error("generic Comfy uni_pc order for 4 timesteps must be 2")
    if comfy_unipc_effective_order(3) != 1:
        raise Error("generic Comfy uni_pc order for 3 timesteps must be 1")

    var sigmas = List[Float64]()
    sigmas.append(4.0)
    sigmas.append(2.0)
    sigmas.append(1.0)
    sigmas.append(0.0)
    var timesteps = build_comfy_unipc_timesteps(sigmas)
    _assert_close("final zero replacement", timesteps[3], 0.001, 0.0)

    var sigma = 2.0
    var denom = sqrt(1.0 + sigma * sigma)
    _assert_close("SigmaConvert alpha", comfy_sigma_convert_alpha(sigma), 1.0 / denom, 1.0e-12)
    _assert_close("SigmaConvert std", comfy_sigma_convert_std(sigma), sigma / denom, 1.0e-12)
    _assert_close("SigmaConvert lambda", comfy_sigma_convert_lambda(sigma), -log(sigma), 1.0e-12)
    _assert_close("initial noise scale", comfy_unipc_initial_noise_scale(sigma), 1.0 / denom, 1.0e-12)

    # The existing product scheduler is flow alpha=1-sigma and bh2/order-2.
    # Generic Comfy `uni_pc` uses SigmaConvert plus bh1, so these must differ.
    if _abs(alpha_from_sigma(sigma) - comfy_sigma_convert_alpha(sigma)) < 1.0e-6:
        raise Error("generic UniPC SigmaConvert alpha collapsed to flow alpha")
    var h = 0.25
    var bh1 = comfy_generic_unipc_b_h(h)
    var bh2 = exp(-h) - 1.0
    if _abs(bh1 - bh2) < 1.0e-6:
        raise Error("generic UniPC bh1 B_h collapsed to bh2 B_h")

    print("PASS: generic Comfy uni_pc semantics remain distinct from uni_pc_bh2")
