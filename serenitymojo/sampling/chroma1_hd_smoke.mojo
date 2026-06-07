# Compile/run smoke for host-side Chroma1-HD sampler contracts.
#
# This deliberately does not create a DeviceContext or run model inference.

from serenitymojo.sampling.chroma1_hd import (
    CHROMA1_HD_DEFAULT_CFG_SCALE,
    CHROMA1_HD_DEFAULT_SHIFT,
    CHROMA1_HD_DEFAULT_STEPS,
    Chroma1HDScheduler,
    build_chroma1_hd_sigma_schedule,
    chroma1_hd_cfg_batch_size,
    chroma1_hd_cfg_value,
    chroma1_hd_euler_update_value,
    chroma1_hd_model_timestep_from_scheduler_timestep,
    chroma1_hd_model_timestep_from_sigma,
    chroma1_hd_scheduler_timestep_from_sigma,
    chroma1_hd_shifted_sigma,
    chroma1_hd_uses_guidance_embedding,
)


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _check_equal(name: String, actual: Int, expected: Int) raises:
    if actual != expected:
        raise Error(
            name
            + " mismatch: actual="
            + String(actual)
            + " expected="
            + String(expected)
        )


def _check_false(name: String, actual: Bool) raises:
    if actual:
        raise Error(name + " expected false")


def _check_close(name: String, actual: Float64, expected: Float64) raises:
    if _abs(actual - expected) > 1.0e-5:
        raise Error(
            name
            + " mismatch: actual="
            + String(actual)
            + " expected="
            + String(expected)
        )


def _check_strictly_less(name: String, actual: Float64, upper_bound: Float64) raises:
    if actual >= upper_bound:
        raise Error(
            name
            + " expected strictly less than "
            + String(upper_bound)
            + ", got "
            + String(actual)
        )


def main() raises:
    _check_close("default shift", Float64(CHROMA1_HD_DEFAULT_SHIFT), 3.0)
    _check_equal("default steps", CHROMA1_HD_DEFAULT_STEPS, 30)
    _check_close("default cfg", Float64(CHROMA1_HD_DEFAULT_CFG_SCALE), 3.5)
    _check_equal("cfg batch size", chroma1_hd_cfg_batch_size(), 2)
    _check_false("guidance embedding", chroma1_hd_uses_guidance_embedding())
    _check_close("cfg scalar", Float64(chroma1_hd_cfg_value(2.0, -1.0, 3.5)), 9.5)
    _check_close(
        "scheduler timestep",
        Float64(chroma1_hd_scheduler_timestep_from_sigma(0.25)),
        250.0,
    )
    _check_close(
        "model timestep from scheduler",
        Float64(chroma1_hd_model_timestep_from_scheduler_timestep(250.0)),
        0.25,
    )
    _check_close(
        "model timestep from sigma",
        Float64(chroma1_hd_model_timestep_from_sigma(0.25)),
        0.25,
    )
    _check_close(
        "scalar Euler update",
        Float64(chroma1_hd_euler_update_value(1.0, 2.0, 1.0, 0.9)),
        0.8,
    )

    _check_close("shifted sigma index0", Float64(chroma1_hd_shifted_sigma(0, 4)), 1.0)
    _check_close("shifted sigma index1", Float64(chroma1_hd_shifted_sigma(1, 4)), 0.9)
    _check_close("shifted sigma index2", Float64(chroma1_hd_shifted_sigma(2, 4)), 0.75)
    _check_close("shifted sigma index3", Float64(chroma1_hd_shifted_sigma(3, 4)), 0.5)
    _check_close("shifted sigma index4", Float64(chroma1_hd_shifted_sigma(4, 4)), 0.0)

    var sigmas = build_chroma1_hd_sigma_schedule(4)
    _check_equal("schedule len", len(sigmas), 5)
    _check_close("sigma[0]", Float64(sigmas[0]), 1.0)
    _check_close("sigma[1]", Float64(sigmas[1]), 0.9)
    _check_close("sigma[2]", Float64(sigmas[2]), 0.75)
    _check_close("sigma[3]", Float64(sigmas[3]), 0.5)
    _check_close("sigma[4]", Float64(sigmas[4]), 0.0)
    for i in range(4):
        _check_strictly_less(
            String("sigma monotonic ") + String(i),
            Float64(sigmas[i + 1]),
            Float64(sigmas[i]),
        )

    var sched = Chroma1HDScheduler(4)
    _check_close("sched.shift", Float64(sched.shift), 3.0)
    _check_close("sched.timestep(1)", Float64(sched.timestep(1)), 0.9)
    _check_close("sched.scheduler_timestep(1)", Float64(sched.scheduler_timestep(1)), 900.0)
    _check_close("sched.model_timestep(1)", Float64(sched.model_timestep(1)), 0.9)
    _check_close("sched.dt(0)", Float64(sched.dt(0)), -0.1)
    print("Chroma1-HD schedule/CFG/update smoke PASS")
