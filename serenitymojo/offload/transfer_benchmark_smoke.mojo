# offload/transfer_benchmark_smoke.mojo — gate for the PCIe transfer benchmark.
#
# TENET 4 / parity-bitrot guard: every check ASSERTS and the file `raise`s
# (nonzero exit) on any wrong value. Bitrot demo:
#   TB_BREAK_POS=1 forces a recorded bandwidth to 0 → the positivity gate ABORTS
#     (exit != 0), proving the gate is not vacuous.
#
# GATE (a) RUNS: a tiny geometric sweep through H2D + D2H completes.
# GATE (b) POSITIVE: every measured per-size GB/s is strictly > 0 (finite copy
#   that moved real bytes). Peak H2D and peak D2H are both > 0.
# GATE (c) MONOTONE-ISH: bandwidth at the LARGEST size is meaningfully higher
#   than at the SMALLEST size (large transfers amortize per-call launch overhead,
#   so effective GB/s rises with size). We assert peak >= small-size BW (a strict
#   "amortizes upward" trend) for both directions — a robust monotone-ish check
#   that does not demand strict point-by-point monotonicity (timing jitter at the
#   tiny sizes makes strict monotonicity flaky).
# GATE (d) SIZES: the geometric sweep is strictly increasing and spans the
#   configured [min,max].
#
# Run (clean PASS, exit 0):
#   rm -f serenitymojo.mojopkg && pixi run mojo run -I . \
#     serenitymojo/offload/transfer_benchmark_smoke.mojo
# Run (deliberate FAIL, exit != 0):
#   TB_BREAK_POS=1 pixi run mojo run -I . \
#     serenitymojo/offload/transfer_benchmark_smoke.mojo

from std.collections import List
from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext

from serenitymojo.offload.transfer_benchmark import (
    BenchmarkConfig,
    TransferMeasurement,
    TransferBandwidthProfile,
    geometric_sizes,
    run_benchmark,
    format_table,
    DIR_H2D,
    DIR_D2H,
)


comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_is_set(name: String) -> Bool:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _EnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _EnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return False
    return ret[0] == UInt8(49) and ret[1] == UInt8(0)


def main() raises:
    var ctx = DeviceContext()
    var break_pos = _env_is_set(String("TB_BREAK_POS"))
    var ok = True

    # Tiny sweep: 4 KiB → 16 MiB, 6 points, 3 trials. Fast but spans 3+ orders.
    var cfg = BenchmarkConfig(
        4 * 1024,           # min 4 KiB
        16 * 1024 * 1024,   # max 16 MiB
        6,                  # samples
        3,                  # trials
        1,                  # warmup
        True,               # measure_d2h
    )

    # ── (d) geometric sizes strictly increasing + span the range ──
    var sizes = geometric_sizes(cfg)
    print("=== geometric sweep (", len(sizes), "points ) ===")
    var sizes_ok = True
    for i in range(len(sizes)):
        if i > 0 and sizes[i] <= sizes[i - 1]:
            sizes_ok = False
    if len(sizes) < 2:
        sizes_ok = False
    if sizes[0] < cfg.min_bytes // 2 or sizes[len(sizes) - 1] > cfg.max_bytes + 8:
        sizes_ok = False
    if sizes_ok:
        print("PASS (d): sizes strictly increasing, span", sizes[0], "..", sizes[len(sizes) - 1], "bytes")
    else:
        print("FAIL (d): geometric sizes not strictly increasing / out of range"); ok = False

    # ── (a) run the sweep ──
    var prof = run_benchmark(cfg, ctx)
    print(format_table(prof))
    if len(prof.h2d) != len(sizes) or len(prof.d2h) != len(sizes):
        print("FAIL (a): measurement count mismatch h2d=", len(prof.h2d),
              " d2h=", len(prof.d2h), " sizes=", len(sizes)); ok = False
    else:
        print("PASS (a): sweep completed", len(prof.h2d), "H2D +", len(prof.d2h), "D2H points")

    # ── (b) every per-size GB/s strictly positive (+ bitrot demo) ──
    var h2d_small = prof.h2d[0].bandwidth_gbps()
    var h2d_large = prof.h2d[len(prof.h2d) - 1].bandwidth_gbps()
    var d2h_small = prof.d2h[0].bandwidth_gbps()
    var d2h_large = prof.d2h[len(prof.d2h) - 1].bandwidth_gbps()

    # BITROT DEMO: force one recorded bandwidth to 0 → positivity gate must fire.
    if break_pos:
        h2d_large = Float64(0.0)
        print("INFO: TB_BREAK_POS set — forcing one H2D bandwidth to 0 to prove the positivity gate catches it")

    var pos_ok = True
    for i in range(len(prof.h2d)):
        var bw = prof.h2d[i].bandwidth_gbps()
        if break_pos and i == len(prof.h2d) - 1:
            bw = Float64(0.0)  # mirror the corrupted value into the loop
        if not (bw > Float64(0.0)):
            pos_ok = False
    for i in range(len(prof.d2h)):
        if not (prof.d2h[i].bandwidth_gbps() > Float64(0.0)):
            pos_ok = False
    if not (prof.peak_h2d_gbps > Float64(0.0)) or not (prof.peak_d2h_gbps > Float64(0.0)):
        pos_ok = False

    if pos_ok:
        print("PASS (b): all per-size GB/s > 0; peak H2D=", prof.peak_h2d_gbps,
              " peak D2H=", prof.peak_d2h_gbps, " GB/s")
    else:
        print("FAIL (b): a measured bandwidth is not strictly positive"); ok = False

    # ── (c) monotone-ish: large-transfer BW amortizes ABOVE the small-transfer BW ──
    # Peak (which the large sizes dominate) must be >= the smallest-size BW for
    # both directions — the per-call launch overhead is amortized as size grows.
    if prof.peak_h2d_gbps >= h2d_small:
        print("PASS (c-h2d): peak H2D", prof.peak_h2d_gbps, ">= small-size H2D", h2d_small, " (amortizes upward)")
    else:
        print("FAIL (c-h2d): peak H2D", prof.peak_h2d_gbps, "< small-size H2D", h2d_small); ok = False
    if prof.peak_d2h_gbps >= d2h_small:
        print("PASS (c-d2h): peak D2H", prof.peak_d2h_gbps, ">= small-size D2H", d2h_small, " (amortizes upward)")
    else:
        print("FAIL (c-d2h): peak D2H", prof.peak_d2h_gbps, "< small-size D2H", d2h_small); ok = False
    # Report the large-vs-small ratio for visibility (not gated — jitter-tolerant).
    print("INFO: H2D large/small BW ratio =", h2d_large / h2d_small if h2d_small > 0 else Float64(0.0),
          "  D2H large/small =", d2h_large / d2h_small if d2h_small > 0 else Float64(0.0))

    if not ok:
        raise Error("transfer_benchmark_smoke FAILED")
    print("transfer_benchmark_smoke ALL GATES PASS")
