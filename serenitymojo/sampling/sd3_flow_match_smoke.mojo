# SD3 shifted-flow tensor scheduler smoke.
#
# Tiny tensor-only check for the production SD3 scheduler surface: textbook CFG,
# model timestep scaling, negative Euler deltas, and `x + v * dt` update.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_NUM_STEPS,
    SD3_MEDIUM_NUM_STEPS,
)
from serenitymojo.sampling.sd3_flow_match import (
    SD3FlowMatchScheduler,
    sd3_cfg,
    sd3_euler_step,
)
from serenitymojo.tensor import Tensor


def _abs(v: Float32) -> Float32:
    if v < 0.0:
        return -v
    return v


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    if _abs(got - expected) > tol:
        raise Error(
            String("SD3 flow-match mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def main() raises:
    var ctx = DeviceContext()
    var sched = SD3FlowMatchScheduler.large_default()
    var sigmas = sched.sigmas()
    if len(sigmas) != SD3_LARGE_NUM_STEPS + 1:
        raise Error("SD3 flow-match schedule length mismatch")
    _check_close(String("sigma[0]"), sigmas[0], 1.0, 0.000001)
    _check_close(String("sigma[1]"), sigmas[1], 0.9878049, 0.000001)
    _check_close(String("sigma[14]"), sigmas[14], 0.75, 0.000001)
    _check_close(String("sigma[28]"), sigmas[28], 0.0, 0.000001)
    _check_close(String("model_timestep[1]"), sched.model_timestep(1), 987.8049, 0.001)

    for i in range(SD3_LARGE_NUM_STEPS):
        if sched.dt(i) >= 0.0:
            raise Error("SD3 flow-match Euler delta must be negative")

    var sh = List[Int]()
    sh.append(2)
    sh.append(2)
    var uncond = Tensor.from_host([1.0, 2.0, 3.0, 4.0], sh.copy(), STDtype.F32, ctx)
    var cond = Tensor.from_host([2.0, 4.0, 6.0, 8.0], sh.copy(), STDtype.F32, ctx)
    var guided = sd3_cfg(cond, uncond, 4.5, ctx)
    var gv = guided.to_host(ctx)
    _check_close(String("cfg[0]"), gv[0], 5.5, 0.000001)
    _check_close(String("cfg[3]"), gv[3], 22.0, 0.000001)

    var velocity = Tensor.from_host([0.5, 1.0, -0.5, -1.0], sh.copy(), STDtype.F32, ctx)
    var stepped = sched.step(uncond, velocity, 0, ctx)
    var sv = stepped.to_host(ctx)
    var dt = sched.dt(0)
    _check_close(String("step[0]"), sv[0], 1.0 + 0.5 * dt, 0.000001)
    _check_close(String("step[3]"), sv[3], 4.0 - 1.0 * dt, 0.000001)

    var stepped2 = sd3_euler_step(uncond, velocity, dt, ctx)
    var sv2 = stepped2.to_host(ctx)
    _check_close(String("standalone_step[1]"), sv2[1], 2.0 + dt, 0.000001)

    print(
        "[sd3-flow-match] steps/shift first/mid/last=",
        sched.num_steps,
        sched.shift,
        sigmas[0],
        sigmas[14],
        sigmas[28],
    )

    var medium = SD3FlowMatchScheduler.medium_default()
    var medium_sigmas = medium.sigmas()
    if len(medium_sigmas) != SD3_MEDIUM_NUM_STEPS + 1:
        raise Error("SD3 medium flow-match schedule length mismatch")
    _check_close(String("medium sigma[0]"), medium_sigmas[0], 1.0, 0.000001)
    _check_close(String("medium sigma[14]"), medium_sigmas[14], 0.75, 0.000001)
    _check_close(String("medium sigma[28]"), medium_sigmas[28], 0.0, 0.000001)
    _check_close(
        String("medium model_timestep[1]"),
        medium.medium_model_timestep(1),
        987.8049,
        0.001,
    )
    print(
        "[sd3-flow-match] medium steps/shift first/mid/last=",
        medium.num_steps,
        medium.shift,
        medium_sigmas[0],
        medium_sigmas[14],
        medium_sigmas[28],
    )
    print("SD3 FlowMatch tensor smoke PASS")
