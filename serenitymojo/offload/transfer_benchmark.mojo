# offload/transfer_benchmark.mojo — on-device PCIe bandwidth benchmark.
#
# Pure-Mojo port of flame-core/src/offload/transfer_benchmark.rs (which is itself
# a Rust port of flextensor/memory_transfer_benchmark.py). Measures the ACTUAL
# H2D and D2H bandwidth at the current hardware / CUDA-driver combo across a
# geometric sweep of transfer sizes, and reports the per-size GB/s plus the peak.
#
# ── Why this lives in offload/ (flame-core tenet §1) ──────────────────────────
# The bandwidth profile is hardware-specific, not model-specific. Every block
# offloader caller on a given box (klein 9B, qwen, ltx2, …) shares the same PCIe
# bus, so a single one-time benchmark at process init serves them all. This is
# the same "fix the primitive, ship every model" rationale flame-core's
# transfer_benchmark.rs cites.
#
# ── What it measures (and deliberately does NOT) ──────────────────────────────
# Measures: device-observed wall time for a host<->device copy of N bytes,
#   bracketed by ctx.synchronize() so the timer covers the completed transfer,
#   not just the enqueue. Reports bytes / seconds → bytes/s → GB/s (1e9 B/s).
# Does NOT measure: kernel-launch cost on the same stream, prefetch-overlap
#   quality, or per-step memory churn — those are telemetry concerns.
#
# ── Copy idiom (mirrors tensor.mojo from_host / to_host) ──────────────────────
# Uses the high-level DeviceContext buffer API the rest of the port already
# trusts: enqueue_create_host_buffer (pinned host), enqueue_create_buffer
# (device), enqueue_copy(dst_buf=, src_buf=), ctx.synchronize(). This is portable
# by construction (no raw cuMemcpy FFI). turbo_loader.mojo's cuMemcpyHtoDAsync_v2
# fast path is for the hot async block-swap; for a one-time init benchmark the
# synchronous enqueue_copy path is the right, simplest spelling.
#
# ── Sync contract ─────────────────────────────────────────────────────────────
# The bench is an INIT-TIME tool — never on the per-step path. The
# ctx.synchronize() bracketing each timed copy is the standard timing pattern and
# is exempt from the no-sync-on-hot-path rule because nothing here runs hot.
#
# Mojo 0.26.x: `def` not `fn`; move-only structs returned via ^; STDtype a value.

from std.collections import List
from std.time import perf_counter_ns
from std.math import log, exp
from std.gpu.host import DeviceContext


comptime DIR_H2D = 0
comptime DIR_D2H = 1


# ── one measured point on the bandwidth curve ─────────────────────────────────
@fieldwise_init
struct TransferMeasurement(Copyable, Movable):
    var nbytes: Int        # transfer size in bytes
    var direction: Int     # DIR_H2D | DIR_D2H
    var duration_ns: Int   # median wall time over the trials, nanoseconds
    var bandwidth_bps: Float64  # bytes / duration_seconds

    def bandwidth_gbps(self) -> Float64:
        """Bandwidth in GB/s (1e9 bytes/sec)."""
        return self.bandwidth_bps / Float64(1.0e9)


# ── sweep configuration ───────────────────────────────────────────────────────
@fieldwise_init
struct BenchmarkConfig(Copyable, Movable):
    var min_bytes: Int     # smallest transfer (>= 1)
    var max_bytes: Int     # largest transfer
    var samples: Int       # geometric sample points in [min,max], inclusive (>= 2)
    var trials: Int        # timed repeats per (size,dir); duration is the median
    var warmup_trials: Int # untimed warmups per (size,dir) to fault pages in
    var measure_d2h: Bool  # also measure D2H (else H2D only)


def default_benchmark_config() -> BenchmarkConfig:
    # 4 KiB → 64 MiB, 9 geometric points (mirrors the rs Default spirit; smaller
    # max so the smoke is fast yet still spans 4 orders of magnitude).
    return BenchmarkConfig(
        4 * 1024,           # min 4 KiB
        64 * 1024 * 1024,   # max 64 MiB
        9,                  # samples
        5,                  # trials
        1,                  # warmup
        True,               # measure_d2h
    )


# ── geometric size sweep: `samples` points from min_bytes..max_bytes inclusive ─
# Each point is rounded to a multiple of 8 bytes (the UInt64-chunk alignment the
# copy path prefers) and clamped >= 8. Monotone non-decreasing by construction;
# duplicates (after rounding at tiny sizes) are dropped so the table is strictly
# increasing — the smoke's monotone-ish check depends on distinct increasing
# sizes.
def geometric_sizes(cfg: BenchmarkConfig) raises -> List[Int]:
    if cfg.min_bytes < 1:
        raise Error("geometric_sizes: min_bytes must be >= 1")
    if cfg.max_bytes < cfg.min_bytes:
        raise Error("geometric_sizes: max_bytes must be >= min_bytes")
    if cfg.samples < 2:
        raise Error("geometric_sizes: samples must be >= 2")
    var out = List[Int]()
    var lo = Float64(cfg.min_bytes)
    var hi = Float64(cfg.max_bytes)
    var ln_lo = _ln(lo)
    var ln_hi = _ln(hi)
    var n = cfg.samples
    for i in range(n):
        var frac = Float64(i) / Float64(n - 1)
        var ln_v = ln_lo + (ln_hi - ln_lo) * frac
        var v = Int(_exp(ln_v) + Float64(0.5))
        v = (v // 8) * 8
        if v < 8:
            v = 8
        # keep strictly increasing (drop a rounding-collision duplicate)
        if len(out) == 0 or v > out[len(out) - 1]:
            out.append(v)
    return out^


def _ln(x: Float64) -> Float64:
    return log(x)


def _exp(x: Float64) -> Float64:
    return exp(x)


def _median_ns(mut times: List[Int]) -> Int:
    # insertion sort (tiny lists) then pick the middle.
    var n = len(times)
    for i in range(1, n):
        var key = times[i]
        var j = i - 1
        while j >= 0 and times[j] > key:
            times[j + 1] = times[j]
            j -= 1
        times[j + 1] = key
    return times[n // 2]


# ── time ONE direction at ONE size: returns a TransferMeasurement ─────────────
# Allocates a pinned host buffer + a device buffer of `nbytes`, runs warmups,
# then `trials` timed copies (each bracketed by ctx.synchronize()), and records
# the median wall time. Buffers are reused across the trials (allocation cost is
# never inside the timed window).
def _bench_one(
    nbytes: Int, direction: Int, cfg: BenchmarkConfig, ctx: DeviceContext
) raises -> TransferMeasurement:
    if nbytes < 1:
        raise Error("_bench_one: nbytes must be >= 1")
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    # touch the host buffer so its pages are resident before timing.
    var hp = host.unsafe_ptr()
    for i in range(0, nbytes, 4096):
        hp[i] = UInt8((i // 4096) & 0xFF)
    hp[nbytes - 1] = UInt8(0xAB)
    ctx.synchronize()

    # warmups (untimed) — fault pages + pull launch cost out of the first timed copy.
    for _ in range(cfg.warmup_trials):
        if direction == DIR_H2D:
            ctx.enqueue_copy(dst_buf=dev, src_buf=host)
        else:
            ctx.enqueue_copy(dst_buf=host, src_buf=dev)
    ctx.synchronize()

    var times = List[Int]()
    for _ in range(cfg.trials):
        var t0 = perf_counter_ns()
        if direction == DIR_H2D:
            ctx.enqueue_copy(dst_buf=dev, src_buf=host)
        else:
            ctx.enqueue_copy(dst_buf=host, src_buf=dev)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var dt = Int(t1 - t0)
        if dt < 1:
            dt = 1  # floor at 1ns so bandwidth stays finite
        times.append(dt)

    var med = _median_ns(times)
    var secs = Float64(med) * Float64(1.0e-9)
    var bps = Float64(nbytes) / secs
    return TransferMeasurement(nbytes, direction, med, bps)


# ── the full sweep result ─────────────────────────────────────────────────────
struct TransferBandwidthProfile(Copyable, Movable):
    var h2d: List[TransferMeasurement]
    var d2h: List[TransferMeasurement]     # empty if measure_d2h was False
    var peak_h2d_gbps: Float64             # max GB/s over all H2D sizes
    var peak_d2h_gbps: Float64             # max GB/s over all D2H sizes (0 if none)

    def __init__(
        out self, var h2d: List[TransferMeasurement], var d2h: List[TransferMeasurement],
        peak_h2d_gbps: Float64, peak_d2h_gbps: Float64,
    ):
        self.h2d = h2d^
        self.d2h = d2h^
        self.peak_h2d_gbps = peak_h2d_gbps
        self.peak_d2h_gbps = peak_d2h_gbps


# ── run_benchmark: sweep sizes through H2D (+ D2H), report peak GB/s ───────────
def run_benchmark(cfg: BenchmarkConfig, ctx: DeviceContext) raises -> TransferBandwidthProfile:
    var sizes = geometric_sizes(cfg)
    var h2d = List[TransferMeasurement]()
    var d2h = List[TransferMeasurement]()
    var peak_h2d = Float64(0.0)
    var peak_d2h = Float64(0.0)
    for i in range(len(sizes)):
        var m = _bench_one(sizes[i], DIR_H2D, cfg, ctx)
        if m.bandwidth_gbps() > peak_h2d:
            peak_h2d = m.bandwidth_gbps()
        h2d.append(m^)
    if cfg.measure_d2h:
        for i in range(len(sizes)):
            var m = _bench_one(sizes[i], DIR_D2H, cfg, ctx)
            if m.bandwidth_gbps() > peak_d2h:
                peak_d2h = m.bandwidth_gbps()
            d2h.append(m^)
    return TransferBandwidthProfile(h2d^, d2h^, peak_h2d, peak_d2h)


def _repeat(c: String, n: Int) -> String:
    var s = String("")
    for _ in range(n):
        s += c
    return s^


def _fmt_size(b: Int) -> String:
    var K = 1024
    var M = K * K
    if b >= M:
        return String(b // M) + String(" MiB")
    elif b >= K:
        return String(b // K) + String(" KiB")
    return String(b) + String(" B")


# ── format_table: stable diagnostic table for log output ──────────────────────
def format_table(prof: TransferBandwidthProfile) -> String:
    var bar = _repeat(String("="), 72)
    var s = bar + String("\n")
    s += String("Memory Transfer Bandwidth Profile\n")
    s += bar + String("\n")
    s += String("   Size (bytes)        Size    Dir    Duration(ms)   Bandwidth(GB/s)\n")
    s += _repeat(String("-"), 72) + String("\n")
    for i in range(len(prof.h2d)):
        var nb = prof.h2d[i].nbytes
        var dn = prof.h2d[i].duration_ns
        var gb = prof.h2d[i].bandwidth_gbps()
        s += String("   ") + String(nb) + String("   ") + _fmt_size(nb)
        s += String("   H2D   ") + String(Float64(dn) * 1.0e-6)
        s += String("   ") + String(gb) + String("\n")
    for i in range(len(prof.d2h)):
        var nb = prof.d2h[i].nbytes
        var dn = prof.d2h[i].duration_ns
        var gb = prof.d2h[i].bandwidth_gbps()
        s += String("   ") + String(nb) + String("   ") + _fmt_size(nb)
        s += String("   D2H   ") + String(Float64(dn) * 1.0e-6)
        s += String("   ") + String(gb) + String("\n")
    s += bar + String("\n")
    s += String("peak H2D = ") + String(prof.peak_h2d_gbps) + String(" GB/s   ")
    s += String("peak D2H = ") + String(prof.peak_d2h_gbps) + String(" GB/s\n")
    return s^
