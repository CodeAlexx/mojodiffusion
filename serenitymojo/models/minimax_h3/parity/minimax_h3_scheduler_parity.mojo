# serenitymojo/models/minimax_h3/parity/minimax_h3_scheduler_parity.mojo
#
# MiniMax-H3 scheduler parity gate.
#
# Proves serenitymojo/models/minimax_h3/scheduler.mojo reproduces the reference
# schedule and Euler step BIT-EXACTLY. Reference: the diffusers PR
# huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_scheduler_oracle.py — the oracle calls the reference's own
# scheduler and dumps sigmas, timesteps, a full step trajectory (walked in
# order, so an error at step k contaminates everything after it), and
# `scale_noise` outputs. Sample and velocity inputs are dumped too, so this gate
# consumes byte-identical inputs instead of regenerating them.
#
# Bit-exact is the right bar: this is float32 scalar arithmetic on both sides
# with no reduction, so any difference is a wrong formula or a wrong evaluation
# order, never accumulation noise.
#
# Oracle: python3 scripts/minimax_h3_scheduler_oracle.py
# Run:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_scheduler_parity.mojo \
#     -o output/checks/minimax_h3_scheduler_parity \
#   && output/checks/minimax_h3_scheduler_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.scheduler import MiniMaxH3Scheduler

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_scheduler/scheduler_ref.safetensors"
comptime SAMPLE_LEN = 16


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


struct Report(Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def exact_f32(mut self, label: String, got: List[Float32], want: List[Float32]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var mismatches = 0
        var first_index = -1
        var worst = Float32(0.0)
        for i in range(len(got)):
            if got[i] != want[i]:
                mismatches += 1
                if first_index < 0:
                    first_index = i
                var diff = got[i] - want[i]
                var mag = -diff if diff < 0.0 else diff
                if mag > worst:
                    worst = mag
        if mismatches == 0:
            print("  ok  ", label, "bit-exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL",
                label,
                mismatches,
                "of",
                len(got),
                "differ; first at",
                first_index,
                "got",
                got[first_index],
                "want",
                want[first_index],
                "max_abs",
                worst,
            )


def _slice(values: List[Float32], start: Int, count: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(count):
        out.append(values[start + i])
    return out^


def _check_case(
    mut report: Report,
    ref st: SafeTensors,
    name: String,
    shift: Float32,
    steps: Int,
) raises:
    var scheduler = MiniMaxH3Scheduler(shift)
    scheduler.set_timesteps(steps)
    report.exact_f32(name + ".sigmas", scheduler.sigmas, _load_f32(st, name + ".sigmas"))
    report.exact_f32(
        name + ".timesteps", scheduler.timesteps, _load_f32(st, name + ".timesteps")
    )

    # Walk the whole trajectory in order, from the oracle's own inputs.
    var sample = _load_f32(st, name + ".sample_in")
    var velocities = _load_f32(st, name + ".velocities")
    var reference = _load_f32(st, name + ".trajectory")
    var num_steps = scheduler.num_inference_steps()
    var trajectory = List[Float32]()
    var current = sample.copy()
    for index in range(num_steps):
        var velocity = _slice(velocities, index * SAMPLE_LEN, SAMPLE_LEN)
        current = scheduler.step(velocity, scheduler.timesteps[index], current)
        for i in range(SAMPLE_LEN):
            trajectory.append(current[i])
    report.exact_f32(name + ".trajectory", trajectory, reference)


def main() raises:
    print("MiniMax-H3 scheduler parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var report = Report()

    print("[1] schedules + Euler trajectories")
    _check_case(report, st, String("video_30"), Float32(12.0), 30)
    _check_case(report, st, String("audio_30"), Float32(3.0), 30)
    _check_case(report, st, String("video_50"), Float32(12.0), 50)
    _check_case(report, st, String("audio_50"), Float32(3.0), 50)
    _check_case(report, st, String("video_8"), Float32(12.0), 8)
    _check_case(report, st, String("min_2"), Float32(12.0), 2)
    _check_case(report, st, String("collapse_1000_400"), Float32(1000.0), 400)

    print("[2] scale_noise (conditioning anchors)")
    var scheduler = MiniMaxH3Scheduler(Float32(12.0))
    var clean = _load_f32(st, "scale_noise.clean")
    var noise = _load_f32(st, "scale_noise.noise")
    var levels = _load_f32(st, "scale_noise.levels")
    var scaled = List[Float32]()
    for index in range(len(levels)):
        var row = scheduler.scale_noise(clean, levels[index], noise)
        for i in range(len(row)):
            scaled.append(row[i])
    report.exact_f32("scale_noise.out", scaled, _load_f32(st, "scale_noise.out"))

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks, all bit-exact")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks differ")
        raise Error("MiniMax-H3 scheduler parity gate failed")
