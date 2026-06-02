# ERNIE-Image FlowMatch scheduler smoke.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.sampling.ernie_sampling import (
    ErnieFlowMatchScheduler,
    build_ernie_sigma_schedule,
    ernie_cfg,
    ernie_model_timestep_from_sigma,
    ernie_shifted_sigma,
)
from serenitymojo.tensor import Tensor


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = got - expected
    if diff < 0.0:
        diff = -diff
    if diff > tol:
        raise Error(
            name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def main() raises:
    var sched = ErnieFlowMatchScheduler.default_50()
    var sigmas = sched.sigmas()
    if len(sigmas) != 51:
        raise Error("ERNIE sigma schedule length mismatch")
    _check_close(String("sigma[0]"), sigmas[0], 1.0, 0.000001)
    _check_close(String("sigma[25]"), sigmas[25], 0.75, 0.000001)
    _check_close(String("sigma[50]"), sigmas[50], 0.0, 0.000001)
    _check_close(String("timestep[0]"), sched.model_timestep(0), 1000.0, 0.000001)
    _check_close(
        String("timestep sigma 0.5"), ernie_model_timestep_from_sigma(0.5), 500.0, 0.000001
    )
    if sched.dt(0) >= 0.0:
        raise Error("ERNIE scheduler dt should be negative")
    var raw = build_ernie_sigma_schedule(2, 3.0)
    _check_close(String("sigma two-step mid"), raw[1], 0.75, 0.000001)
    _check_close(String("direct mid"), ernie_shifted_sigma(1, 2, 3.0), 0.75, 0.000001)

    var ctx = DeviceContext()
    var sh = List[Int]()
    sh.append(1)
    sh.append(2)
    var cond_vals = List[Float32]()
    cond_vals.append(2.0)
    cond_vals.append(4.0)
    var uncond_vals = List[Float32]()
    uncond_vals.append(1.0)
    uncond_vals.append(1.0)
    var cond = Tensor.from_host(cond_vals, sh.copy(), STDtype.BF16, ctx)
    var uncond = Tensor.from_host(uncond_vals, sh.copy(), STDtype.BF16, ctx)
    var cfg = ernie_cfg(cond, uncond, 2.0, ctx)
    var cfg_host = cfg.to_host(ctx)
    _check_close(String("cfg[0]"), cfg_host[0], 3.0, 0.0001)
    _check_close(String("cfg[1]"), cfg_host[1], 7.0, 0.0001)

    var lat_vals = List[Float32]()
    lat_vals.append(0.5)
    lat_vals.append(-0.5)
    var vel_vals = List[Float32]()
    vel_vals.append(1.0)
    vel_vals.append(-2.0)
    var latent = Tensor.from_host(lat_vals, sh.copy(), STDtype.BF16, ctx)
    var velocity = Tensor.from_host(vel_vals, sh^, STDtype.BF16, ctx)
    var tiny_sched = ErnieFlowMatchScheduler(2, 3.0)
    var stepped = tiny_sched.step(latent, velocity, 0, ctx)
    var stepped_host = stepped.to_host(ctx)
    _check_close(String("step[0]"), stepped_host[0], 0.25, 0.0001)
    _check_close(String("step[1]"), stepped_host[1], 0.0, 0.0001)

    print("ERNIE FlowMatch scheduler/tensor smoke PASS")
