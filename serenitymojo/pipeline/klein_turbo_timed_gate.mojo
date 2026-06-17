# klein_turbo_timed_gate.mojo - P0 timed gate for Klein Turbo offload.
#
# Runs the same tiny full-stack Klein forward twice:
#   1. TurboPlannedLoader forced through default-stream H2D copies.
#   2. TurboPlannedLoader production copy-stream H2D copies.
#
# Hard gate: outputs must match (cosine >= 0.999, byte-exact reported).
# Reported evidence: forward wall-clock seconds, observed VRAM at checkpoints,
# copy mode, and speedup ratio. Do not claim a 2.2x speedup unless this gate is
# run on the target GPU and the measured ratio proves it.
#
# Build/run only through the capped orchestrator path on this workstation.

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.offload.vmm_cuda import cu_mem_get_info, cu_mempool_trim_current
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.klein_dit import (
    Klein9BOffloadedTurbo,
    build_klein_rope_tables,
)


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"

comptime N_IMG = 4
comptime N_TXT = 8
comptime S = N_IMG + N_TXT


struct TimedRun(Movable):
    var mode: String
    var forward_seconds: Float64
    var observed_peak_vram_mib: Float64
    var output: List[Float32]

    def __init__(
        out self,
        mode: String,
        forward_seconds: Float64,
        observed_peak_vram_mib: Float64,
        var output: List[Float32],
    ):
        self.mode = mode
        self.forward_seconds = forward_seconds
        self.observed_peak_vram_mib = observed_peak_vram_mib
        self.output = output^


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("TIMED GATE FAIL: ") + msg)


def _linspace(
    start: Float32, end: Float32, var shape: List[Int], dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    var vals = List[Float32](capacity=n)
    for i in range(n):
        var t = Float32(i) / Float32(n - 1 if n > 1 else 1)
        vals.append(start + t * (end - start))
    return Tensor.from_host(vals, shape^, dtype, ctx)


def _cosine_sim(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cosine_sim: length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na < 1e-30 or nb < 1e-30:
        return Float64(1.0)
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("max_abs_diff: length mismatch")
    var mad = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i]) - Float64(b[i])
        var ad = d if d >= 0.0 else -d
        if ad > mad:
            mad = ad
    return mad


def _byte_exact(a: List[Float32], b: List[Float32]) raises -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _peak_mib(total_bytes: Int, min_free_bytes: Int) -> Float64:
    return Float64(total_bytes - min_free_bytes) / 1048576.0


def _update_min_free(current_min: Int) raises -> Int:
    var mem = cu_mem_get_info()
    if mem.free_bytes < current_min:
        return mem.free_bytes
    return current_min


def _make_img(ctx: DeviceContext) raises -> Tensor:
    var shape = List[Int]()
    shape.append(1)
    shape.append(N_IMG)
    shape.append(128)
    return _linspace(Float32(-0.1), Float32(0.1), shape^, STDtype.BF16, ctx)


def _make_txt(ctx: DeviceContext) raises -> Tensor:
    var shape = List[Int]()
    shape.append(1)
    shape.append(N_TXT)
    shape.append(12288)
    return _linspace(Float32(-0.05), Float32(0.05), shape^, STDtype.BF16, ctx)


def _make_timestep(ctx: DeviceContext) raises -> Tensor:
    var shape = List[Int]()
    shape.append(1)
    var vals = List[Float32]()
    vals.append(Float32(500.0))
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _run_mode(use_default_stream_copy: Bool, ctx: DeviceContext) raises -> TimedRun:
    var mode = String("default_stream") if use_default_stream_copy else String("copy_stream")
    print("[mode]", mode, "loading")
    var mem0 = cu_mem_get_info()
    var total = mem0.total_bytes
    var min_free = mem0.free_bytes

    var model = Klein9BOffloadedTurbo.load_with_copy_mode(
        KLEIN9B_PATH, ctx, use_default_stream_copy
    )
    ctx.synchronize()
    min_free = _update_min_free(min_free)
    print("[mode]", mode, "copy_mode", model.loader._turbo.copy_mode())
    print("[mode]", mode, "async_enabled", model.loader._turbo.async_enabled())

    var img = _make_img(ctx)
    var txt = _make_txt(ctx)
    var timestep = _make_timestep(ctx)
    var rope = build_klein_rope_tables[N_IMG, N_TXT, 32, 128](ctx, STDtype.BF16)
    ctx.synchronize()
    min_free = _update_min_free(min_free)

    print("[mode]", mode, "forward_start")
    var t0 = perf_counter_ns()
    var out = model.forward_full[N_IMG, N_TXT, S](
        img, txt, timestep, rope[0], rope[1], ctx
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    min_free = _update_min_free(min_free)

    var host = out.to_host(ctx)
    ctx.synchronize()
    min_free = _update_min_free(min_free)

    var seconds = Float64(Int(t1 - t0)) / 1.0e9
    var peak = _peak_mib(total, min_free)
    print("[mode]", mode, "forward_seconds", seconds)
    print("[mode]", mode, "observed_peak_vram_mib", peak)
    print("[mode]", mode, "elements", len(host))
    return TimedRun(mode, seconds, peak, host^)


def main() raises:
    print("=== Klein9B Turbo Timed Gate ===")
    print("[config] N_IMG=", N_IMG, " N_TXT=", N_TXT, " blocks=8+24=32")
    print("[checkpoint]", KLEIN9B_PATH)

    var ctx = DeviceContext()
    cu_mempool_trim_current()

    var default_run = _run_mode(True, ctx)
    cu_mempool_trim_current()
    var copy_run = _run_mode(False, ctx)

    var cos = _cosine_sim(default_run.output, copy_run.output)
    var mad = _max_abs_diff(default_run.output, copy_run.output)
    var exact = _byte_exact(default_run.output, copy_run.output)
    _check(len(default_run.output) == len(copy_run.output), "element count mismatch")
    _check(cos >= Float64(0.999), "cosine below threshold")

    var speedup = Float64(0.0)
    if copy_run.forward_seconds > 0.0:
        speedup = default_run.forward_seconds / copy_run.forward_seconds

    print("=== TIMED GATE RESULT ===")
    print("default_stream_seconds", default_run.forward_seconds)
    print("copy_stream_seconds", copy_run.forward_seconds)
    print("speedup_default_over_copy", speedup)
    print("default_stream_observed_peak_vram_mib", default_run.observed_peak_vram_mib)
    print("copy_stream_observed_peak_vram_mib", copy_run.observed_peak_vram_mib)
    print("cosine_similarity", cos)
    print("max_abs_diff", mad)
    print("byte_exact", exact)
    print("KLEIN9B TURBO TIMED GATE: PASS")
