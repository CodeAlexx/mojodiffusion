# Compile/run smoke for host-side FLUX.2/Klein schedule helpers.
#
# This deliberately does not create a DeviceContext or run model inference.

from serenitymojo.sampling.flux2_klein import (
    Flux2KleinScheduler,
    build_flux2_fixed_shift_schedule,
    build_flux2_img2img_sigmas,
    build_flux2_sigma_schedule,
    compute_empirical_mu,
)


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _check_close(name: String, actual: Float64, expected: Float64) raises:
    if _abs(actual - expected) > 1.0e-5:
        raise Error(
            name
            + " mismatch: actual="
            + String(actual)
            + " expected="
            + String(expected)
        )


def main() raises:
    # 1024x1024 Klein uses output/16 => 64x64 packed tokens.
    var mu = compute_empirical_mu(4096, 50)
    _check_close("mu(4096,50)", mu, 2.0233511571292637)

    var sigmas = build_flux2_sigma_schedule(50, 4096)
    _check_close("sigma[0]", Float64(sigmas[0]), 1.0)
    _check_close("sigma[1]", Float64(sigmas[1]), 0.99730906)
    _check_close("sigma[49]", Float64(sigmas[49]), 0.13371896)
    _check_close("sigma[50]", Float64(sigmas[50]), 0.0)

    var sched = Flux2KleinScheduler(3, 4096)
    _check_close("sched.mu", sched.mu, 2.2970022579630993)
    _check_close("sched.timestep(1)", Float64(sched.timestep(1)), 0.95212712)
    _check_close("sched.dt(0)", Float64(sched.dt(0)), -0.04787288)

    var edit = build_flux2_fixed_shift_schedule(35, 2.02)
    _check_close("edit[0]", Float64(edit[0]), 1.0)
    _check_close("edit[1]", Float64(edit[1]), 0.99612349)
    _check_close("edit[34]", Float64(edit[34]), 0.18163168)
    _check_close("edit[35]", Float64(edit[35]), 0.0)

    var img = build_flux2_img2img_sigmas(35, 2.02, 0.7)
    _check_close("img2img[0]", Float64(img[0]), 0.94620597)
    _check_close("img2img[34]", Float64(img[34]), 0.13333124)
    _check_close("img2img[35]", Float64(img[35]), 0.0)
    print("FLUX2/Klein schedule smoke PASS")
