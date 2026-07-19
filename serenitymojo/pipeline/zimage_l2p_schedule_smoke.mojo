# Z-Image L2P shifted-flow scalar schedule smoke.
#
# Verifies the VAE-less pixel-space L2P schedule contract without creating a
# DeviceContext or loading tensors.

from serenitymojo.models.dit.zimage_l2p_contract import (
    ZIMAGE_L2P_DEFAULT_STEPS,
    build_zimage_l2p_sigma_schedule,
    zimage_l2p_default_shift,
    zimage_l2p_model_timestep,
    zimage_l2p_schedule_delta,
    zimage_l2p_sigma,
)


def _abs(v: Float32) -> Float32:
    if v < 0.0:
        return -v
    return v


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    if _abs(got - expected) > tol:
        raise Error(
            String("Z-Image L2P schedule mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def main() raises:
    var shift = zimage_l2p_default_shift()
    var sched = build_zimage_l2p_sigma_schedule(ZIMAGE_L2P_DEFAULT_STEPS, shift)
    if len(sched) != ZIMAGE_L2P_DEFAULT_STEPS + 1:
        raise Error("Z-Image L2P schedule length mismatch")

    _check_close(String("sigma[0]"), sched[0], 1.0, 0.000001)
    _check_close(String("sigma[1]"), sched[1], 0.98863636, 0.000001)
    _check_close(String("sigma[15]"), sched[15], 0.75, 0.000001)
    _check_close(String("sigma[29]"), sched[29], 0.09375, 0.000001)
    _check_close(String("sigma[30]"), sched[30], 0.0, 0.000001)
    _check_close(
        String("model_timestep[1]"),
        zimage_l2p_model_timestep(sched[1]),
        11.363636,
        0.001,
    )

    for i in range(ZIMAGE_L2P_DEFAULT_STEPS):
        if sched[i + 1] >= sched[i]:
            raise Error("Z-Image L2P schedule must strictly descend before terminal")
        var sigma = zimage_l2p_sigma(i, ZIMAGE_L2P_DEFAULT_STEPS, shift)
        _check_close(String("direct sigma"), sigma, sched[i], 0.000001)
        var delta = zimage_l2p_schedule_delta(i, ZIMAGE_L2P_DEFAULT_STEPS, shift)
        _check_close(String("delta"), delta, sched[i + 1] - sched[i], 0.000001)
        if delta >= 0.0:
            raise Error("Z-Image L2P Euler delta must be negative")

    print(
        "[zimage-l2p-schedule] steps/shift first/mid/last=",
        ZIMAGE_L2P_DEFAULT_STEPS,
        shift,
        sched[0],
        sched[15],
        sched[30],
    )
    print("Z-Image L2P schedule scalar smoke PASS")
