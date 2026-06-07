# Compile/run smoke for SDXL scheduler and tensor helpers.
#
# This does not run model inference.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.sampling.sdxl_euler import (
    SDXLEulerScheduler,
    build_sdxl_sigmas,
    build_sdxl_timesteps,
    sdxl_cfg,
    sdxl_euler_step,
    sdxl_initial_noise_sigma,
    sdxl_input_scale,
)
from serenitymojo.tensor import Tensor


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
    var sigmas = build_sdxl_sigmas(30)
    var timesteps = build_sdxl_timesteps(30)

    _check_close("timestep[0]", Float64(timesteps[0]), 958.0)
    _check_close("timestep[29]", Float64(timesteps[29]), 1.0)
    _check_close("sigma[0]", Float64(sigmas[0]), 11.47684646)
    _check_close("sigma[1]", Float64(sigmas[1]), 9.54358257)
    _check_close("sigma[29]", Float64(sigmas[29]), 0.04131441)
    _check_close("sigma[30]", Float64(sigmas[30]), 0.0)
    _check_close("init_sigma", Float64(sdxl_initial_noise_sigma(sigmas[0])), 11.52033006)
    _check_close("input_scale_0", Float64(sdxl_input_scale(sigmas[0])), 0.08680307)

    var sched = SDXLEulerScheduler(50)
    _check_close("sched.timestep(0)", Float64(sched.timestep(0)), 981.0)
    _check_close("sched.sigma(0)", Float64(sched.sigma(0)), 13.12041074)
    _check_close("sched.initial_noise_sigma", Float64(sched.initial_noise_sigma()), 13.15846412)
    _check_close("sched.input_scale(0)", Float64(sched.input_scale(0)), 0.07599671)

    var ctx = DeviceContext()
    var sh = List[Int]()
    sh.append(1)
    sh.append(2)
    var cond = Tensor.from_host([2.0, 4.0], sh.copy(), STDtype.BF16, ctx)
    var uncond = Tensor.from_host([1.0, 1.0], sh.copy(), STDtype.BF16, ctx)
    var cfg = sdxl_cfg(cond, uncond, 2.0, ctx)
    if cfg.dtype() != STDtype.BF16:
        raise Error("SDXL CFG changed tensor dtype")
    var cfg_host = cfg.to_host(ctx)
    _check_close("cfg[0]", Float64(cfg_host[0]), 3.0)
    _check_close("cfg[1]", Float64(cfg_host[1]), 7.0)

    var latent = Tensor.from_host([0.5, -0.5], sh.copy(), STDtype.BF16, ctx)
    var eps = Tensor.from_host([1.0, -2.0], sh^, STDtype.BF16, ctx)
    var stepped = sdxl_euler_step(latent, eps, 1.0, 0.75, ctx)
    if stepped.dtype() != STDtype.BF16:
        raise Error("SDXL Euler step changed tensor dtype")
    var stepped_host = stepped.to_host(ctx)
    _check_close("step[0]", Float64(stepped_host[0]), 0.25)
    _check_close("step[1]", Float64(stepped_host[1]), 0.0)
    print("SDXL Euler schedule smoke PASS")
