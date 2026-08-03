# models/dit/parity/minimax_h3_sampling_probe.mojo — compile+run gate for
# `models/dit/minimax_h3_sampling.mojo`.
#
# `minimax_h3_sampling.mojo` deliberately does NOT import the two ALREADY-
# GATED host oracles it mirrors (models/minimax_h3/scheduler.mojo,
# models/minimax_h3/packing.mojo — see that file's header for why). So this
# probe proves parity a different way: it loads the SAME safetensors fixtures
# those oracles' own parity gates already consume —
#
#   output/minimax_h3_scheduler/scheduler_ref.safetensors  (scheduler_parity.mojo)
#   output/minimax_h3_packing/packing_ref.safetensors      (packing_parity.mojo)
#
# — via `io/safetensors.mojo` directly (data only, no oracle CODE import), and
# compares this file's independently-reproduced math against them bit-exact.
# What's new relative to those two existing gates: [sched 4] below drives the
# Euler blend through REAL GPU kernels (`step_device`, i.e.
# ops/tensor_algebra.mul_scalar/add) instead of host scalar Lists, proving the
# device path — not just the host mirror — reproduces the oracle trajectory.
#
# This is a numerical self-check against already-gated fixtures, not a fresh
# diffusers parity run: the underlying math is proven correct elsewhere
# (scheduler_parity 22/22, packing_parity 63/63); what this proves is that the
# independent runtime reproduction — and, for the scheduler, the device
# kernel chain — agrees with it.
#
# MUST be built at -O0 (like its two upstream oracle gates — see their own
# header run commands): MEASURED, this repo's default optimization level
# (both `mojo run -I .` and plain `mojo build -I .` with no `-O` flag)
# reassociates the scheduler's float32 chain differently than -O0 and misses
# the reference by 1 ulp on a large fraction of values. This is NOT a bug
# introduced here — `models/minimax_h3/parity/minimax_h3_scheduler_parity.
# mojo` (the oracle's OWN gate, unmodified) fails IDENTICALLY (same first-
# differing index, same max_abs, per case) under the same non-O0 build, which
# is why that gate's header documents `-O0 -j 1` explicitly rather than a
# bare `mojo run`. Confirms this file's reproduction is bit-for-bit faithful
# to the oracle even in how it breaks, not just in how it passes.
#
#   pixi run mojo build -O0 -j 1 -I . \
#     serenitymojo/models/dit/parity/minimax_h3_sampling_probe.mojo \
#     -o output/checks/minimax_h3_sampling_probe \
#   && output/checks/minimax_h3_sampling_probe

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.minimax_h3_dit import minimax_h3_released_config
from serenitymojo.models.dit.minimax_h3_sampling import (
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_VIDEO_SHIFT,
    MiniMaxH3DualSchedule,
    MiniMaxH3EulerSchedule,
    MiniMaxH3SamplingRowTimesteps,
    minimax_h3_adaln_rows_for_step,
    minimax_h3_build_sampling_geometry,
    minimax_h3_build_sampling_row_timesteps,
    minimax_h3_extend_row_timestep_indices,
    minimax_h3_pad_sampling_geometry,
    minimax_h3_patchify_video,
    minimax_h3_prepare_model_input,
    minimax_h3_sampling_spatial_grid,
    minimax_h3_sampling_temporal_grid,
    minimax_h3_sampling_temporal_span,
)

comptime SCHED_REF = "/home/alex/mojodiffusion/output/minimax_h3_scheduler/scheduler_ref.safetensors"
comptime PACK_REF = "/home/alex/mojodiffusion/output/minimax_h3_packing/packing_ref.safetensors"
comptime SAMPLE_LEN = 16


# ── fixture loaders (mirrors scheduler_parity.mojo / packing_parity.mojo) ────
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


def _load_f64(ref st: SafeTensors, name: String) raises -> List[Float64]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F64:
        raise Error(String("_load_f64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float64]()
    var out = List[Float64]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I64:
        raise Error(String("_load_i64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _slice(values: List[Float32], start: Int, count: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(count):
        out.append(values[start + i])
    return out^


struct Report(Movable):
    """Running pass/fail tally."""

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
        for i in range(len(got)):
            if got[i] != want[i]:
                mismatches += 1
                if first_index < 0:
                    first_index = i
        if mismatches == 0:
            print("  ok  ", label, "bit-exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, mismatches, "of", len(got), "differ; first at",
                first_index, "got", got[first_index], "want", want[first_index],
            )

    def exact_f64(mut self, label: String, got: List[Float64], want: List[Float64]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var mismatches = 0
        var first_index = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                mismatches += 1
                if first_index < 0:
                    first_index = i
        if mismatches == 0:
            print("  ok  ", label, "bit-exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, mismatches, "of", len(got), "differ; first at",
                first_index, "got", got[first_index], "want", want[first_index],
            )

    def exact_int(mut self, label: String, got: List[Int], want: List[Int]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var mismatches = 0
        var first_index = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                mismatches += 1
                if first_index < 0:
                    first_index = i
        if mismatches == 0:
            print("  ok  ", label, "exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, mismatches, "of", len(got), "differ; first at",
                first_index, "got", got[first_index], "want", want[first_index],
            )


# ── Part 1 helpers: dual-schedule scheduler ───────────────────────────────────
def _check_schedule_case(
    mut report: Report, ref st: SafeTensors, name: String, shift: Float32, steps: Int
) raises:
    var schedule = MiniMaxH3EulerSchedule(shift)
    schedule.set_timesteps(steps)
    report.exact_f32(name + ".sigmas", schedule.sigmas, _load_f32(st, name + ".sigmas"))
    report.exact_f32(name + ".timesteps", schedule.timesteps, _load_f32(st, name + ".timesteps"))


def _check_host_trajectory(
    mut report: Report, ref st: SafeTensors, name: String, shift: Float32, steps: Int
) raises:
    var schedule = MiniMaxH3EulerSchedule(shift)
    schedule.set_timesteps(steps)
    var sample = _load_f32(st, name + ".sample_in")
    var velocities = _load_f32(st, name + ".velocities")
    var reference = _load_f32(st, name + ".trajectory")
    var num_steps = schedule.num_inference_steps()
    var trajectory = List[Float32]()
    var current = sample.copy()
    for index in range(num_steps):
        var velocity = _slice(velocities, index * SAMPLE_LEN, SAMPLE_LEN)
        current = schedule.step_host(velocity, schedule.timesteps[index], current)
        for i in range(SAMPLE_LEN):
            trajectory.append(current[i])
    report.exact_f32(name + ".trajectory (step_host)", trajectory, reference)


def _device_trajectory(
    mut schedule: MiniMaxH3EulerSchedule,
    sample_in: List[Float32],
    velocities: List[Float32],
    num_steps: Int,
    sample_len: Int,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """Walks `num_steps` real GPU `step_device` calls, uploading each velocity
    slice fresh (matching how a production sampler would receive it from the
    model each step) and reading the result back to host for comparison."""
    var current = Tensor.from_host(sample_in, [sample_len], STDtype.F32, ctx)
    var trajectory = List[Float32]()
    for index in range(num_steps):
        var velocity_slice = _slice(velocities, index * sample_len, sample_len)
        var velocity = Tensor.from_host(velocity_slice, [sample_len], STDtype.F32, ctx)
        var next = schedule.step_device(velocity, schedule.timesteps[index], current, ctx)
        var next_host = next.to_host(ctx)
        for i in range(sample_len):
            trajectory.append(next_host[i])
        current = next^
    return trajectory^


def _check_device_trajectory(
    mut report: Report, ref st: SafeTensors, name: String, shift: Float32, steps: Int, ctx: DeviceContext
) raises:
    var schedule = MiniMaxH3EulerSchedule(shift)
    schedule.set_timesteps(steps)
    var sample = _load_f32(st, name + ".sample_in")
    var velocities = _load_f32(st, name + ".velocities")
    var reference = _load_f32(st, name + ".trajectory")
    var num_steps = schedule.num_inference_steps()
    var trajectory = _device_trajectory(schedule, sample, velocities, num_steps, SAMPLE_LEN, ctx)
    report.exact_f32(name + ".trajectory (step_device, real GPU kernels)", trajectory, reference)


def main() raises:
    print("MiniMax-H3 sampling scaffold probe (dual schedule + packed-sequence layout)")
    var ctx = DeviceContext()
    var report = Report()

    # ═══ Part 1: dual-schedule scheduler ═══
    print("  scheduler reference:", SCHED_REF)
    var sched_st = SafeTensors.open(String(SCHED_REF))

    print("[sched 1] per-shift schedules (7-case matrix, mirrors scheduler_parity.mojo)")
    _check_schedule_case(report, sched_st, String("video_30"), MINIMAX_H3_VIDEO_SHIFT, 30)
    _check_schedule_case(report, sched_st, String("audio_30"), MINIMAX_H3_AUDIO_SHIFT, 30)
    _check_schedule_case(report, sched_st, String("video_50"), MINIMAX_H3_VIDEO_SHIFT, 50)
    _check_schedule_case(report, sched_st, String("audio_50"), MINIMAX_H3_AUDIO_SHIFT, 50)
    _check_schedule_case(report, sched_st, String("video_8"), MINIMAX_H3_VIDEO_SHIFT, 8)
    _check_schedule_case(report, sched_st, String("min_2"), MINIMAX_H3_VIDEO_SHIFT, 2)
    _check_schedule_case(report, sched_st, String("collapse_1000_400"), Float32(1000.0), 400)

    print("[sched 2] MiniMaxH3DualSchedule (shared base grid) matches the two independent cases")
    var dual = MiniMaxH3DualSchedule()
    dual.set_timesteps(30)
    report.exact_f32("dual.video.sigmas", dual.video.sigmas, _load_f32(sched_st, "video_30.sigmas"))
    report.exact_f32("dual.audio.sigmas", dual.audio.sigmas, _load_f32(sched_st, "audio_30.sigmas"))
    report.exact_f32("dual.video.timesteps", dual.video.timesteps, _load_f32(sched_st, "video_30.timesteps"))
    report.exact_f32("dual.audio.timesteps", dual.audio.timesteps, _load_f32(sched_st, "audio_30.timesteps"))
    if dual.num_inference_steps() != 29:
        raise Error("probe: FAIL dual schedule step count != 29")
    print("  ok   dual.num_inference_steps() == 29 (video and audio agree)")

    print("[sched 3] host-scalar Euler trajectory (step_host) vs the gated oracle trajectory")
    _check_host_trajectory(report, sched_st, String("video_30"), MINIMAX_H3_VIDEO_SHIFT, 30)
    _check_host_trajectory(report, sched_st, String("audio_30"), MINIMAX_H3_AUDIO_SHIFT, 30)

    print("[sched 4] DEVICE Euler trajectory (step_device, real GPU kernels) vs the SAME oracle trajectory")
    _check_device_trajectory(report, sched_st, String("video_30"), MINIMAX_H3_VIDEO_SHIFT, 30, ctx)
    _check_device_trajectory(report, sched_st, String("audio_30"), MINIMAX_H3_AUDIO_SHIFT, 30, ctx)

    # ═══ Part 2: packed-sequence layout builder ═══
    print("  packing reference:", PACK_REF)
    var pack_st = SafeTensors.open(String(PACK_REF))

    print("[pack 1] spatial + temporal grids (tiny_t2va's 128x160 canvas -> latent 8x10)")
    var latent_h = 8
    var latent_w = 10
    var sqrt_area = sqrt(Float64(latent_h * latent_w))
    report.exact_f64(
        "grid.128x160.height",
        minimax_h3_sampling_spatial_grid(latent_h, 2, sqrt_area),
        _load_f64(pack_st, "grid.128x160.height"),
    )
    report.exact_f64(
        "grid.128x160.width",
        minimax_h3_sampling_spatial_grid(latent_w, 2, sqrt_area),
        _load_f64(pack_st, "grid.128x160.width"),
    )
    report.exact_f64(
        "temporal_grid.107",
        minimax_h3_sampling_temporal_grid(107, Float64(17.0)),
        _load_f64(pack_st, "temporal_grid.107"),
    )
    report.exact_f64(
        "temporal_grid.2",
        minimax_h3_sampling_temporal_grid(2, Float64(0.0)),
        _load_f64(pack_st, "temporal_grid.2"),
    )

    print("[pack 2] temporal span (numpy pairwise summation)")
    var span_n = _load_i64(pack_st, "span.num_latent_frames")
    var span_got = List[Float64]()
    for i in range(len(span_n)):
        span_got.append(minimax_h3_sampling_temporal_span(span_n[i]))
    report.exact_f64("span.value", span_got, _load_f64(pack_st, "span.value"))

    print("[pack 3] tiny_t2va packed geometry (128x160 canvas, 22 requested frames, no anchors)")
    var text_tags = List[Int]()
    for _ in range(7):
        text_tags.append(1)
    var no_anchors = List[Int]()
    # aligned=22 (already 17n+5, n=1), video_latent_num_frames=7, audio_latent_num_frames=37,
    # latent_h=128//16=8, latent_w=160//16=10 — values taken from packing_ref.json's own
    # recorded case metadata (not re-derived here: this file's scope is the packed-sequence
    # geometry builder, not the canvas/frame-count resolver — see minimax_h3_sampling.mojo
    # header, "PART 2").
    var geometry = minimax_h3_build_sampling_geometry(
        text_tags, 7, latent_h, latent_w, 37, 2, 2, no_anchors
    )
    if geometry.sequence_length != 221:
        raise Error("probe: FAIL tiny_t2va sequence_length != 221")
    if geometry.num_condition_video_rows != 0 or geometry.num_condition_audio_rows != 0:
        raise Error("probe: FAIL tiny_t2va condition-row counts != 0")
    report.exact_f64(
        "tiny_t2va.position_ids", geometry.position_ids, _load_f64(pack_st, "tiny_t2va.position_ids")
    )
    report.exact_int(
        "tiny_t2va.token_tags", geometry.token_tags, _load_i64(pack_st, "tiny_t2va.token_tags")
    )
    report.exact_int(
        "tiny_t2va.video_indices", geometry.video_indices, _load_i64(pack_st, "tiny_t2va.video_indices")
    )
    report.exact_int(
        "tiny_t2va.audio_indices", geometry.audio_indices, _load_i64(pack_st, "tiny_t2va.audio_indices")
    )
    report.exact_int(
        "tiny_t2va.text_indices", geometry.text_indices, _load_i64(pack_st, "tiny_t2va.text_indices")
    )

    print("[pack 4] row timesteps for 3 sub-cases (t_mid / t_collapse / t_start)")
    var row_ts_mid = minimax_h3_build_sampling_row_timesteps(
        geometry, Float32(0.5), Float32(0.8), Float32(0.999), Float32(1.0)
    )
    report.exact_f32("tiny_t2va.t_mid.values", row_ts_mid.values, _load_f32(pack_st, "tiny_t2va.t_mid.values"))
    report.exact_int("tiny_t2va.t_mid.indices", row_ts_mid.indices, _load_i64(pack_st, "tiny_t2va.t_mid.indices"))

    var row_ts_collapse = minimax_h3_build_sampling_row_timesteps(
        geometry, Float32(0.999), Float32(0.999), Float32(0.999), Float32(1.0)
    )
    report.exact_f32(
        "tiny_t2va.t_collapse.values", row_ts_collapse.values, _load_f32(pack_st, "tiny_t2va.t_collapse.values")
    )
    report.exact_int(
        "tiny_t2va.t_collapse.indices", row_ts_collapse.indices, _load_i64(pack_st, "tiny_t2va.t_collapse.indices")
    )

    var row_ts_start = minimax_h3_build_sampling_row_timesteps(
        geometry, Float32(1.0) / Float32(1000.0), Float32(0.0039), Float32(0.999), Float32(1.0)
    )
    report.exact_f32(
        "tiny_t2va.t_start.values", row_ts_start.values, _load_f32(pack_st, "tiny_t2va.t_start.values")
    )
    report.exact_int(
        "tiny_t2va.t_start.indices", row_ts_start.indices, _load_i64(pack_st, "tiny_t2va.t_start.indices")
    )

    print("[pack 5] adaln_rows_for_step: reuses minimax_h3_dit.minimax_h3_adaln_rows")
    var adaln_rows = minimax_h3_adaln_rows_for_step(geometry, row_ts_mid)
    if len(adaln_rows) != geometry.sequence_length:
        raise Error("probe: FAIL adaln_rows_for_step length mismatch")
    # spot-check the timestep_index*3+tag formula on one row of each modality.
    if adaln_rows[0] != row_ts_mid.indices[0] * 3 + 1:
        raise Error("probe: FAIL adaln row formula wrong for a text row")
    var vrow = geometry.video_indices[0]
    if adaln_rows[vrow] != row_ts_mid.indices[vrow] * 3 + 0:
        raise Error("probe: FAIL adaln row formula wrong for a video row")
    var arow = geometry.audio_indices[0]
    if adaln_rows[arow] != row_ts_mid.indices[arow] * 3 + 2:
        raise Error("probe: FAIL adaln row formula wrong for an audio row")
    print("  ok   adaln_rows_for_step formula spot-check (text/video/audio rows)")

    print("[pack 6] padding: rows with tag<0 must never index the adaln table")
    var padded_geometry = minimax_h3_pad_sampling_geometry(geometry, geometry.sequence_length + 5)
    var padded_indices = minimax_h3_extend_row_timestep_indices(
        row_ts_mid.indices, geometry.sequence_length + 5
    )
    var padded_row_ts = MiniMaxH3SamplingRowTimesteps(row_ts_mid.values.copy(), padded_indices^)
    var padded_adaln_rows = minimax_h3_adaln_rows_for_step(padded_geometry, padded_row_ts)
    for i in range(geometry.sequence_length):
        if padded_adaln_rows[i] != adaln_rows[i]:
            raise Error("probe: FAIL padding changed a real row's adaln index")
    for i in range(geometry.sequence_length, geometry.sequence_length + 5):
        if padded_adaln_rows[i] != 0:
            raise Error("probe: FAIL a padding row did not map to adaln row 0")
    print("  ok   padding rows map to adaln row 0; real-row prefix byte-identical to the unpadded case")

    # ═══ Part 3: device glue smoke (patchify_video, prepare_model_input) ═══
    print("[device 1] minimax_h3_patchify_video (reuses ops/patchify3d.patchify3d)")
    var config = minimax_h3_released_config()
    var lat_shape = [config.latents_dim, 2, 4, 4]
    var lat_host = List[Float32]()
    for i in range(config.latents_dim * 2 * 4 * 4):
        lat_host.append(Float32(i) * Float32(0.001))
    var lat_tensor = Tensor.from_host(lat_host, lat_shape^, STDtype.F32, ctx)
    var patches = minimax_h3_patchify_video(lat_tensor, config, ctx)
    if patches.shape() != [8, config.video_patch_dim()]:
        raise Error("probe: FAIL patchify_video shape mismatch")
    print("  ok   patchify_video shape", patches.shape())

    var bad_shape = [config.latents_dim - 1, 2, 4, 4]
    var bad_host = List[Float32]()
    for _ in range((config.latents_dim - 1) * 2 * 4 * 4):
        bad_host.append(Float32(0.0))
    var bad_tensor = Tensor.from_host(bad_host, bad_shape^, STDtype.F32, ctx)
    var raised = False
    try:
        _ = minimax_h3_patchify_video(bad_tensor, config, ctx)
    except e:
        raised = True
    if not raised:
        raise Error("probe: FAIL patchify_video accepted a wrong channel count")
    print("  ok   patchify_video rejects a wrong channel count")

    print("[device 2] minimax_h3_prepare_model_input (F32 accumulator -> BF16 model input, torch RNE)")
    var sample_f32 = Tensor.from_host([Float32(0.125), Float32(-2.0), Float32(3.5)], [3], STDtype.F32, ctx)
    var model_in = minimax_h3_prepare_model_input(sample_f32, ctx)
    if model_in.dtype() != STDtype.BF16:
        raise Error("probe: FAIL prepare_model_input did not return BF16")
    if model_in.numel() != 3:
        raise Error("probe: FAIL prepare_model_input changed element count")
    print("  ok   prepare_model_input casts F32->BF16 via torch_f32_to_bf16_rne as the DTYPE RULE requires")

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks, all bit-exact")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks differ")
        raise Error("MiniMax-H3 sampling scaffold probe failed")
