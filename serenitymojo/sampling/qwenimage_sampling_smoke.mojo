# Qwen-Image OneTrainer FlowMatch scheduler smoke.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.sampling.qwenimage_sampling import (
    QwenImageFlowMatchScheduler,
    build_qwenimage_onetrainer_sigmas,
    qwenimage_cfg,
    qwenimage_dynamic_shift_value,
    qwenimage_model_timestep_from_sigma,
    qwenimage_mu,
    qwenimage_packed_seq_len,
    qwenimage_scheduler_timestep_from_sigma,
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
    var seq_len = qwenimage_packed_seq_len(128, 128)
    if seq_len != 4096:
        raise Error("Qwen-Image packed seq_len mismatch")
    _check_close(String("mu"), qwenimage_mu(Float32(seq_len)), 0.69354839, 0.000001)
    _check_close(
        String("dynamic shift"),
        qwenimage_dynamic_shift_value(Float32(seq_len)),
        2.0008018,
        0.00001,
    )

    var sched = QwenImageFlowMatchScheduler.default_1024_50()
    var sigmas = sched.sigmas()
    if len(sigmas) != 51:
        raise Error("Qwen-Image sigma schedule length mismatch")
    _check_close(String("sigma[0]"), sigmas[0], 1.0, 0.000001)
    _check_close(String("sigma[1]"), sigmas[1], 0.98989092, 0.00001)
    _check_close(String("sigma[2]"), sigmas[2], 0.97957135, 0.00001)
    _check_close(String("sigma[25]"), sigmas[25], 0.66425134, 0.00001)
    _check_close(String("sigma[49]"), sigmas[49], 0.02, 0.00001)
    _check_close(String("sigma[50]"), sigmas[50], 0.0, 0.000001)
    _check_close(String("scheduler timestep[1]"), sched.scheduler_timestep(1), 989.89092, 0.01)
    _check_close(String("model timestep[1]"), sched.model_timestep(1), 0.98989092, 0.00001)
    _check_close(
        String("scheduler timestep sigma 0.5"),
        qwenimage_scheduler_timestep_from_sigma(0.5),
        500.0,
        0.000001,
    )
    _check_close(
        String("model timestep sigma 0.5"),
        qwenimage_model_timestep_from_sigma(0.5),
        0.5,
        0.000001,
    )
    for i in range(sched.num_steps):
        if sched.dt(i) >= 0.0:
            raise Error("Qwen-Image scheduler dt should be negative")

    var sigmas_20 = build_qwenimage_onetrainer_sigmas(20, Float32(seq_len))
    _check_close(String("20-step sigma[1]"), sigmas_20[1], 0.97349807, 0.00001)
    _check_close(String("20-step sigma[10]"), sigmas_20[10], 0.64986519, 0.00001)
    _check_close(String("20-step sigma[19]"), sigmas_20[19], 0.02, 0.00001)
    _check_close(String("20-step sigma[20]"), sigmas_20[20], 0.0, 0.000001)

    var ctx = DeviceContext()
    var sh = List[Int]()
    sh.append(2)
    sh.append(2)
    var cond = Tensor.from_host([2.0, 4.0, 6.0, 8.0], sh.copy(), STDtype.BF16, ctx)
    var uncond = Tensor.from_host([1.0, 2.0, 3.0, 4.0], sh.copy(), STDtype.BF16, ctx)
    var cfg = qwenimage_cfg(cond, uncond, 4.0, ctx)
    if cfg.dtype() != STDtype.BF16:
        raise Error("Qwen-Image CFG changed tensor dtype")
    var cfg_host = cfg.to_host(ctx)
    _check_close(String("cfg[0]"), cfg_host[0], 5.0, 0.0001)
    _check_close(String("cfg[3]"), cfg_host[3], 20.0, 0.0001)

    var latent = Tensor.from_host([0.5, -0.5, 1.0, -1.0], sh.copy(), STDtype.BF16, ctx)
    var velocity = Tensor.from_host([1.0, -2.0, 0.5, -0.5], sh^, STDtype.BF16, ctx)
    var stepped = sched.step(latent, velocity, 0, ctx)
    if stepped.dtype() != STDtype.BF16:
        raise Error("Qwen-Image Euler step changed tensor dtype")
    var stepped_host = stepped.to_host(ctx)
    var dt = sched.dt(0)
    # The step output intentionally stores BF16, so compare with BF16-scale
    # tolerance while preserving the dtype assertion above.
    _check_close(String("step[0]"), stepped_host[0], 0.5 + dt, 0.001)
    _check_close(String("step[1]"), stepped_host[1], -0.5 - 2.0 * dt, 0.001)

    print("Qwen-Image FlowMatch scheduler/tensor smoke PASS")
