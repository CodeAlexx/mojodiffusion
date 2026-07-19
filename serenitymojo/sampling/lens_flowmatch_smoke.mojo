# Microsoft Lens FlowMatch scalar scheduler smoke.
#
# Host-only parity gate for inference-flame/src/sampling/lens_flowmatch.rs.
# This intentionally avoids DeviceContext and model tensors.

from serenitymojo.sampling.lens_flowmatch import (
    LensFlowMatchScheduler,
    build_lens_raw_sigmas,
    build_lens_shifted_sigmas,
    lens_compute_empirical_mu,
    lens_exponential_shift,
    lens_image_seq_len,
)


def _abs32(x: Float32) -> Float32:
    if x < 0.0:
        return -x
    return x


def _abs64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _check_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _check_close32(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    if _abs32(got - expected) > tol:
        raise Error(
            name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _check_close64(name: String, got: Float64, expected: Float64, tol: Float64) raises:
    if _abs64(got - expected) > tol:
        raise Error(
            name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def main() raises:
    var image_seq = lens_image_seq_len(1024, 1024)
    _check_int(String("image_seq_len"), image_seq, 4096)

    var mu = lens_compute_empirical_mu(image_seq, 20)
    _check_close64(String("mu(4096,20)"), mu, 2.1980220725551165, 1e-12)
    _check_close64(
        String("mu(4500,20)"),
        lens_compute_empirical_mu(4500, 20),
        1.21838166,
        1e-8,
    )

    var raw = build_lens_raw_sigmas(20)
    _check_int(String("raw_len"), len(raw), 20)
    _check_close32(String("raw[0]"), raw[0], 1.0, 1e-7)
    _check_close32(String("raw[10]"), raw[10], 0.5, 1e-7)
    _check_close32(String("raw[19]"), raw[19], 0.05000000074505806, 1e-7)
    for i in range(len(raw) - 1):
        if raw[i] <= raw[i + 1]:
            raise Error("raw Lens sigmas must be strictly decreasing")

    _check_close32(
        String("shift_identity"),
        lens_exponential_shift(0.5, 0.0),
        0.5,
        1e-7,
    )

    var shifted = build_lens_shifted_sigmas(20, image_seq)
    _check_int(String("shifted_len"), len(shifted), 20)
    _check_close32(String("shifted[0]"), shifted[0], 1.0, 1e-7)
    _check_close32(String("shifted[1]"), shifted[1], 0.9941906332969666, 1e-6)
    _check_close32(String("shifted[10]"), shifted[10], 0.9000717401504517, 1e-6)
    _check_close32(String("shifted[19]"), shifted[19], 0.32160255312919617, 1e-6)
    for i in range(len(shifted) - 1):
        if shifted[i] <= shifted[i + 1]:
            raise Error("shifted Lens sigmas must be strictly decreasing")

    var sched = LensFlowMatchScheduler.for_resolution(1024, 1024, 20)
    _check_int(String("sched_num_steps"), sched.num_steps, 20)
    _check_int(String("sched_image_seq_len"), sched.image_seq_len, 4096)
    _check_close64(String("sched_mu"), sched.mu, mu, 1e-12)
    _check_close32(String("sched_timestep0"), sched.timestep(0), 1.0, 1e-7)
    _check_close32(
        String("sched_dt0"),
        sched.dt(0),
        -0.005809366703033447,
        1e-6,
    )
    _check_close32(
        String("sched_dt18"),
        sched.dt(18),
        -0.17859682440757751,
        1e-6,
    )
    _check_close32(
        String("sched_dt19_terminal"),
        sched.dt(19),
        -0.32160255312919617,
        1e-6,
    )

    print("[lens-flowmatch] image_seq/mu:", image_seq, mu)
    print("[lens-flowmatch] raw first/mid/last:", raw[0], raw[10], raw[19])
    print(
        "[lens-flowmatch] shifted first/mid/last:",
        shifted[0],
        shifted[10],
        shifted[19],
    )
    print("[lens-flowmatch] terminal dt:", sched.dt(19))
    print("Microsoft Lens FlowMatch scalar scheduler PASS")
