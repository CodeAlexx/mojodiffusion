# perf_record_smoke.mojo — shared speed scorecard contract smoke.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/perf_record_smoke.mojo

from serenitymojo.training.benchmark_matrix import (
    training_benchmark_case,
    training_benchmark_matrix_size,
)
from serenitymojo.training.perf_record import (
    PERF_FAST_PATH_DEVICE,
    PERF_FAST_PATH_HOST_GRAD_COMPAT,
    PERF_LANE_MOJO_CURRENT,
    TrainingPerfRecord,
    TrainingPhaseTimings,
    emit_training_perf_record,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("perf_record_smoke FAILED: ") + msg)


def _gate_record() raises:
    var rec = TrainingPerfRecord(
        String("krea2"),
        PERF_LANE_MOJO_CURRENT,
        String("abc123"),
        String("BF16"),
        16,
        1,
        String("512"),
        String("AdamW"),
        String("strict"),
        2,
        5,
        0.35,
        TrainingPhaseTimings(
            0.10, 0.20, 0.01, 0.005, 0.0, 0.03, 0.0, 0.0,
        ),
        123456789,
        0,
        0,
        1,
        PERF_FAST_PATH_DEVICE,
        String("training-sdpa-strict-math"),
        String(""),
    )
    rec.validate()
    var line = rec.to_jsonl()
    _check(line.byte_length() > 80, "json line is too short")
    _check(rec.summary().byte_length() > 40, "summary is too short")
    emit_training_perf_record(rec)

    var bad_device = TrainingPerfRecord(
        String("zimage"),
        PERF_LANE_MOJO_CURRENT,
        String("abc124"),
        String("BF16"),
        16,
        1,
        String("512"),
        String("AdamW"),
        String("strict"),
        2,
        5,
        0.35,
        TrainingPhaseTimings(
            0.10, 0.20, 0.01, 0.005, 0.0, 0.03, 0.0, 0.0,
        ),
        123456789,
        1,
        1,
        1,
        PERF_FAST_PATH_DEVICE,
        String("training-sdpa-strict-math"),
        String(""),
    )
    var bad_device_raised = False
    try:
        bad_device.validate()
    except e:
        print("expected device fast path readback failure: ", e)
        bad_device_raised = True
    _check(
        bad_device_raised,
        "device fast path with full tensor readback must fail loud",
    )

    var compat = TrainingPerfRecord(
        String("krea2"),
        PERF_LANE_MOJO_CURRENT,
        String("abc125"),
        String("BF16"),
        16,
        1,
        String("512"),
        String("AdamW"),
        String("strict"),
        2,
        5,
        0.35,
        TrainingPhaseTimings(
            0.10, 0.20, 0.01, 0.005, 0.0, 0.03, 0.0, 0.0,
        ),
        123456789,
        3,
        5,
        2,
        PERF_FAST_PATH_HOST_GRAD_COMPAT,
        String("training-sdpa-strict-math"),
        String(""),
    )
    compat.validate()
    _check(not compat.is_device_fast_path(), "host compat must not report device fast path")


def _gate_matrix() raises:
    var n = training_benchmark_matrix_size()
    _check(n >= 4, "matrix must cover at least four model families")
    var saw_krea = False
    var saw_zimage = False
    var saw_klein = False
    var saw_other = False
    for i in range(n):
        var c = training_benchmark_case(i)
        c.validate()
        if c.model == String("krea2"):
            saw_krea = True
        elif c.model == String("zimage"):
            saw_zimage = True
        elif c.model == String("klein") or c.model == String("ideogram4"):
            saw_klein = True
        else:
            saw_other = True
    _check(saw_krea, "matrix missing Krea2")
    _check(saw_zimage, "matrix missing ZImage")
    _check(saw_klein, "matrix missing Klein or Ideogram")
    _check(saw_other, "matrix missing an additional architecture")


def main() raises:
    _gate_record()
    _gate_matrix()
    print("PASS: training perf record and benchmark matrix contracts")
