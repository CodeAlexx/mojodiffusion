# SD3.5 shifted-flow scalar schedule smoke.
#
# No tensors or model loads. This verifies the Rust-facing SD3 schedule contract
# captured by models/dit/sd3_contract.mojo.

from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_NUM_STEPS,
    SD3_MEDIUM_NUM_STEPS,
    build_sd3_shifted_schedule,
    sd3_large_model_timestep,
    sd3_large_schedule_shift,
    sd3_medium_model_timestep,
    sd3_medium_schedule_shift,
    sd3_schedule_delta,
    sd3_shifted_sigma,
)


def _abs(v: Float32) -> Float32:
    if v < 0.0:
        return -v
    return v


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    if _abs(got - expected) > tol:
        raise Error(
            String("SD3 schedule mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def main() raises:
    var shift = sd3_large_schedule_shift()
    var sched = build_sd3_shifted_schedule(SD3_LARGE_NUM_STEPS, shift)
    if len(sched) != SD3_LARGE_NUM_STEPS + 1:
        raise Error("SD3 schedule length mismatch")

    _check_close(String("sigma[0]"), sched[0], 1.0, 0.000001)
    _check_close(String("sigma[1]"), sched[1], 0.9878049, 0.000001)
    _check_close(String("sigma[14]"), sched[14], 0.75, 0.000001)
    _check_close(String("sigma[28]"), sched[28], 0.0, 0.000001)
    _check_close(
        String("model_timestep[1]"),
        sd3_large_model_timestep(sched[1]),
        987.8049,
        0.001,
    )

    for i in range(SD3_LARGE_NUM_STEPS):
        if sched[i + 1] >= sched[i]:
            raise Error("SD3 schedule must strictly descend before terminal")
        var sigma = sd3_shifted_sigma(i, SD3_LARGE_NUM_STEPS, shift)
        _check_close(String("direct sigma"), sigma, sched[i], 0.000001)
        var delta = sd3_schedule_delta(i, SD3_LARGE_NUM_STEPS, shift)
        _check_close(String("delta"), delta, sched[i + 1] - sched[i], 0.000001)
        if delta >= 0.0:
            raise Error("SD3 Euler delta must be negative")

    print(
        "[sd3-schedule] steps/shift first/mid/last=",
        SD3_LARGE_NUM_STEPS,
        shift,
        sched[0],
        sched[14],
        sched[28],
    )

    var medium_shift = sd3_medium_schedule_shift()
    var medium_sched = build_sd3_shifted_schedule(SD3_MEDIUM_NUM_STEPS, medium_shift)
    if len(medium_sched) != SD3_MEDIUM_NUM_STEPS + 1:
        raise Error("SD3 medium schedule length mismatch")
    _check_close(String("medium sigma[0]"), medium_sched[0], 1.0, 0.000001)
    _check_close(String("medium sigma[1]"), medium_sched[1], 0.9878049, 0.000001)
    _check_close(String("medium sigma[14]"), medium_sched[14], 0.75, 0.000001)
    _check_close(String("medium sigma[28]"), medium_sched[28], 0.0, 0.000001)
    _check_close(
        String("medium model_timestep[1]"),
        sd3_medium_model_timestep(medium_sched[1]),
        987.8049,
        0.001,
    )
    for i in range(SD3_MEDIUM_NUM_STEPS):
        if medium_sched[i + 1] >= medium_sched[i]:
            raise Error("SD3 medium schedule must strictly descend before terminal")
        var sigma = sd3_shifted_sigma(i, SD3_MEDIUM_NUM_STEPS, medium_shift)
        _check_close(String("medium direct sigma"), sigma, medium_sched[i], 0.000001)
        var delta = sd3_schedule_delta(i, SD3_MEDIUM_NUM_STEPS, medium_shift)
        _check_close(
            String("medium delta"),
            delta,
            medium_sched[i + 1] - medium_sched[i],
            0.000001,
        )
        if delta >= 0.0:
            raise Error("SD3 medium Euler delta must be negative")

    print(
        "[sd3-schedule] medium steps/shift first/mid/last=",
        SD3_MEDIUM_NUM_STEPS,
        medium_shift,
        medium_sched[0],
        medium_sched[14],
        medium_sched[28],
    )
    print("SD3.5 Large+Medium schedule scalar smoke PASS")
