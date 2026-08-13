# offload/parity/turbo_overlap_probe.mojo
#
# MEASURED probe: does an H2D weight copy on the explicit `copy_stream`
# (cuMemcpyHtoDAsync_v2 — the EXACT production path in turbo_loader.mojo)
# actually OVERLAP a compute kernel on the default stream, wall-clock, in the
# installed MAX / CUDA pin?  This is the ONE fact audit-item-12 turns on: no
# trainer currently overlaps H2D with compute, and before wiring double-buffered
# prefetch into TurboPlannedLoader we must PROVE the overlap is achievable at all.
#
# ── Why this probe exists (prior art gap) ─────────────────────────────────────
#   * turbo_probe_smoke.mojo (Phase-0) proved ctx.create_stream() streams are
#     genuinely INDEPENDENT and the record_event/enqueue_wait_for fence is
#     load-bearing — but it did so with a *compute poison kernel* cross-stream.
#     It NEVER measured an H2D COPY overlapping compute.
#   * SKEPTIC_FINDINGS_turbo_phase1: "The implementation ships correct bytes; it
#     does NOT ship a verified async overlap" — CHECK 2's overlap was collapsed
#     by a rogue ctx.synchronize(); overlap is UNPROVEN.
#   * transfer_benchmark.mojo measures raw H2D/D2H bandwidth but explicitly does
#     NOT measure "prefetch-overlap quality".
#   This probe fills that exact gap with a decisive, ratio-based A/B.
#
# ── The measurement (5 timed arms, each over N iterations after a warmup) ──────
#   copy_only     : N H2D copies on copy_stream, one final ctx.synchronize().
#   compute_only  : N spin kernels on the default stream, one final sync.
#   serial        : per iter { copy; sync; compute; sync } — copy and compute
#                   STRICTLY serialized. Reference for "no overlap" (~copy+compute).
#   concurrent    : per iter { copy into buf[i%2] on copy_stream ; spin over the
#                   OTHER buffer on default } with NO cross-stream fence, one final
#                   sync. Both engines saturated on independent data — the cleanest
#                   test of "can the copy engine and the SMs run at the same time".
#   double_buffer : the FAITHFUL loader shape — copy(i+1) staged on copy_stream
#                   while compute(i) reads block(i) on the default stream, with the
#                   full ev / compute_done handshake (exactly what wiring
#                   prefetch_next_with_ctx into the hot loop would produce).
#
# ── Verdict (printed; ratio-based so it is robust to compute-time miscalibration)
#   OVERLAP CAPABILITY : concurrent/iter must be meaningfully below (copy+compute)
#                        /iter and approach max(copy,compute)/iter. If instead it
#                        ~= copy+compute, the two streams are SERIALIZED in this
#                        pin → NO-GO, do not wire double-buffered prefetch.
#   LOADER SPEEDUP     : serial/iter ÷ double_buffer/iter — the per-step win the
#                        loader would actually capture.
#
# NOTE: build-only in this session (no GPU here). The verdict is produced when the
# main session RUNS the built binary on the RTX 3090 Ti (steps at the bottom).
#
# Knobs (argv, all optional):  copy_MiB  spin_iters  N_iters
#   defaults: 256  8000  20
# If the run prints "compute/iter" << "copy/iter" (or vice-versa), re-run with a
# larger/smaller spin_iters so both arms are non-trivial (each >~5 ms); the
# capability verdict is only meaningful when BOTH engines have real work.
#
# Build & run:
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm \
#       serenitymojo/offload/parity/turbo_overlap_probe.mojo \
#       -o /tmp/turbo_overlap_probe
#   /tmp/turbo_overlap_probe                 # defaults
#   /tmp/turbo_overlap_probe 512 12000 24    # heavier
#
# Mojo 1.0.0b1 / MAX: `def` not `fn`; kernels cannot raise; move-only buffers via ^.

from std.sys import argv, has_accelerator
from std.time import perf_counter_ns
from std.gpu import global_idx
from max.gpu.host import (
    DeviceContext, HostBuffer, DeviceBuffer, DeviceStream, DeviceEvent,
)

from serenitymojo.offload.turbo_loader import _h2d_dma_copy


comptime _BLOCK = 256


# ── compute proxy: every thread runs a dependent FMA chain of `iters` steps ────
# `iters` is a RUNTIME arg so the loop cannot be unrolled-to-nothing; the read +
# write-back over `buf` keeps the kernel from being elided. Multi-thread so the
# SMs are saturated — a realistic stand-in for transformer-block compute, and the
# strongest possible competitor to the copy engine for the overlap test.
def _spin_kernel(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    n_w: Int64,
    iters_w: Int32,
):
    var n = Int(n_w)
    var iters = Int(iters_w)
    var i = Int(global_idx.x)
    if i < n:
        var x = buf[i]
        var c = Float32(1.0000001)
        var d = Float32(0.0000001)
        for _ in range(iters):
            x = x * c + d
        buf[i] = x


def _fmt_ms(ns: Int) -> String:
    return String(Float64(ns) * 1.0e-6)


def main() raises:
    comptime if not has_accelerator():
        print("turbo_overlap_probe: no GPU — nothing to measure")
        return

    # ── knobs ────────────────────────────────────────────────────────────────
    var copy_mib = 256
    var spin_iters = 8000
    var n_iters = 20
    var args = argv()
    if len(args) >= 2:
        copy_mib = Int(args[1])
    if len(args) >= 3:
        spin_iters = Int(args[2])
    if len(args) >= 4:
        n_iters = Int(args[3])

    var copy_bytes = copy_mib * 1024 * 1024
    var n_elems = copy_bytes // 4          # spin over the copy dest as Float32
    var grid = (n_elems + _BLOCK - 1) // _BLOCK

    print("==== turbo_overlap_probe: H2D-copy / compute overlap ====")
    print("copy_bytes =", copy_bytes, "(", copy_mib, "MiB )")
    print("spin_iters =", spin_iters, "  compute_elems =", n_elems)
    print("N_iters    =", n_iters)
    print("copy path  = cuMemcpyHtoDAsync_v2 on ctx.create_stream() (production)")
    print("")

    var ctx = DeviceContext()
    var copy_stream = ctx.create_stream()

    # Pinned host source (device-DMA-able), touched so pages are resident.
    var host = ctx.enqueue_create_host_buffer[DType.uint8](copy_bytes)
    var hp = host.unsafe_ptr()
    for i in range(0, copy_bytes, 4096):
        hp[i] = UInt8((i // 4096) & 0xFF)
    hp[copy_bytes - 1] = UInt8(0xAB)

    # Two device slots (double buffer). Filled non-zero so spin reads are finite.
    var dev0 = ctx.enqueue_create_buffer[DType.uint8](copy_bytes)
    var dev1 = ctx.enqueue_create_buffer[DType.uint8](copy_bytes)
    dev0.enqueue_fill(UInt8(0x3F))
    dev1.enqueue_fill(UInt8(0x3F))
    ctx.synchronize()

    var p0 = dev0.unsafe_ptr().bitcast[Float32]()
    var p1 = dev1.unsafe_ptr().bitcast[Float32]()
    var u0 = UInt64(Int(dev0.unsafe_ptr()))
    var u1 = UInt64(Int(dev1.unsafe_ptr()))

    # Events for the faithful double-buffer arm.
    var ev0 = ctx.create_event[disable_timing=True]()
    var ev1 = ctx.create_event[disable_timing=True]()
    var cd0 = ctx.create_event[disable_timing=True]()
    var cd1 = ctx.create_event[disable_timing=True]()

    # ── warmup (JIT compile kernel, fault pages, spin up copy engine) ─────────
    _h2d_dma_copy(u0, host.unsafe_ptr(), copy_bytes, copy_stream)
    ctx.enqueue_function[_spin_kernel](
        p1, Int64(n_elems), Int32(spin_iters), grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()

    # ── ARM 1: copy_only ──────────────────────────────────────────────────────
    var t0 = perf_counter_ns()
    for _ in range(n_iters):
        _h2d_dma_copy(u0, host.unsafe_ptr(), copy_bytes, copy_stream)
    ctx.synchronize()
    var t_copy = Int(perf_counter_ns() - t0)

    # ── ARM 2: compute_only ───────────────────────────────────────────────────
    t0 = perf_counter_ns()
    for _ in range(n_iters):
        ctx.enqueue_function[_spin_kernel](
            p1, Int64(n_elems), Int32(spin_iters), grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    var t_compute = Int(perf_counter_ns() - t0)

    # ── ARM 3: serial (copy THEN compute, strictly serialized) ────────────────
    t0 = perf_counter_ns()
    for _ in range(n_iters):
        _h2d_dma_copy(u0, host.unsafe_ptr(), copy_bytes, copy_stream)
        ctx.synchronize()
        ctx.enqueue_function[_spin_kernel](
            p0, Int64(n_elems), Int32(spin_iters), grid_dim=grid, block_dim=_BLOCK
        )
        ctx.synchronize()
    var t_serial = Int(perf_counter_ns() - t0)

    # ── ARM 4a: concurrent (both engines, independent data, NO fence) ─────────
    # copy stages buf[i%2]; compute spins the OTHER buffer. No dependency, so if
    # the copy engine and SMs can co-run this collapses to ~max(copy,compute).
    t0 = perf_counter_ns()
    for i in range(n_iters):
        if i % 2 == 0:
            _h2d_dma_copy(u0, host.unsafe_ptr(), copy_bytes, copy_stream)
            ctx.enqueue_function[_spin_kernel](
                p1, Int64(n_elems), Int32(spin_iters), grid_dim=grid, block_dim=_BLOCK
            )
        else:
            _h2d_dma_copy(u1, host.unsafe_ptr(), copy_bytes, copy_stream)
            ctx.enqueue_function[_spin_kernel](
                p0, Int64(n_elems), Int32(spin_iters), grid_dim=grid, block_dim=_BLOCK
            )
    ctx.synchronize()
    var t_conc = Int(perf_counter_ns() - t0)

    # ── ARM 4b: double_buffer (faithful loader shape, full handshake) ─────────
    # Pre-stage block 0 into dev0; then each step stages block i+1 on copy_stream
    # while compute(i) reads block i on the default stream. ev fences compute on
    # its block's copy; compute_done fences the copy stream before it overwrites a
    # slot still being read. This is EXACTLY what prefetch_next_with_ctx +
    # mark_active_block_done would produce in a trainer's block loop.
    var cd_rec0 = False
    var cd_rec1 = False
    t0 = perf_counter_ns()
    _h2d_dma_copy(u0, host.unsafe_ptr(), copy_bytes, copy_stream)
    copy_stream.record_event(ev0)
    for i in range(n_iters):
        var cur_p = p0 if (i % 2 == 0) else p1
        var cur_ev = ev0 if (i % 2 == 0) else ev1
        # compute(i) waits for block i's H2D.
        ctx.stream().enqueue_wait_for(cur_ev)
        # stage block i+1 into the OTHER slot while compute(i) runs.
        if i + 1 < n_iters:
            if (i + 1) % 2 == 0:
                if cd_rec0:
                    copy_stream.enqueue_wait_for(cd0)
                    cd_rec0 = False
                _h2d_dma_copy(u0, host.unsafe_ptr(), copy_bytes, copy_stream)
                copy_stream.record_event(ev0)
            else:
                if cd_rec1:
                    copy_stream.enqueue_wait_for(cd1)
                    cd_rec1 = False
                _h2d_dma_copy(u1, host.unsafe_ptr(), copy_bytes, copy_stream)
                copy_stream.record_event(ev1)
        ctx.enqueue_function[_spin_kernel](
            cur_p, Int64(n_elems), Int32(spin_iters), grid_dim=grid, block_dim=_BLOCK
        )
        # mark this slot's compute done so the copy stream may later reuse it.
        if i % 2 == 0:
            ctx.stream().record_event(cd0)
            cd_rec0 = True
        else:
            ctx.stream().record_event(cd1)
            cd_rec1 = True
    ctx.synchronize()
    var t_db = Int(perf_counter_ns() - t0)

    # ── report ────────────────────────────────────────────────────────────────
    var N = Float64(n_iters)
    var copy_i = Int(Float64(t_copy) / N)
    var comp_i = Int(Float64(t_compute) / N)
    var serial_i = Int(Float64(t_serial) / N)
    var conc_i = Int(Float64(t_conc) / N)
    var db_i = Int(Float64(t_db) / N)
    var sum_i = copy_i + comp_i
    var max_i = copy_i if copy_i > comp_i else comp_i

    print("── per-iteration wall time (ms) ──")
    print("  copy_only      =", _fmt_ms(copy_i))
    print("  compute_only   =", _fmt_ms(comp_i))
    print("  ideal SUM      =", _fmt_ms(sum_i), " (copy+compute; no overlap)")
    print("  ideal MAX      =", _fmt_ms(max_i), " (perfect overlap floor)")
    print("  serial         =", _fmt_ms(serial_i))
    print("  concurrent     =", _fmt_ms(conc_i))
    print("  double_buffer  =", _fmt_ms(db_i))
    print("")

    # calibration sanity: both engines must have real work for a meaningful test.
    var min_i = copy_i if copy_i < comp_i else comp_i
    if min_i < 5_000_000:  # < 5 ms
        print("  WARNING: one arm is < 5 ms — retune spin_iters/copy_MiB so BOTH")
        print("           copy_only and compute_only are non-trivial before")
        print("           trusting the capability verdict.")
        print("")

    # ── verdicts ──────────────────────────────────────────────────────────────
    # CAPABILITY: concurrent must beat the no-overlap SUM by a clear margin.
    var cap_ratio = Float64(conc_i) / Float64(sum_i)         # <1 means faster than serial-sum
    var cap_speedup = Float64(sum_i) / Float64(conc_i)
    var overlap_frac = (Float64(sum_i) - Float64(conc_i)) / (Float64(sum_i) - Float64(max_i) + 1.0)
    print("── VERDICT: OVERLAP CAPABILITY ──")
    print("  concurrent / SUM   =", cap_ratio, " (want << 1.0)")
    print("  concurrent speedup =", cap_speedup, "x over no-overlap SUM")
    print("  overlap efficiency =", overlap_frac, " (1.0 = perfect, 0 = none)")
    if conc_i <= Int(Float64(sum_i) * 0.80):
        print("  RESULT: OVERLAP ACHIEVED — H2D copy on the copy stream co-runs")
        print("          with default-stream compute in this pin. Wiring")
        print("          double-buffered prefetch is JUSTIFIED.")
    else:
        print("  RESULT: NO OVERLAP — concurrent ~= copy+compute. The copy stream")
        print("          and default stream SERIALIZE in this pin. Do NOT wire")
        print("          double-buffered prefetch (dead machinery). NO-GO.")
    print("")

    # PRACTICAL: what the loader would actually capture, in loader shape.
    var db_speedup = Float64(serial_i) / Float64(db_i)
    print("── VERDICT: LOADER PRACTICAL (double-buffer vs serial) ──")
    print("  serial/iter        =", _fmt_ms(serial_i))
    print("  double_buffer/iter =", _fmt_ms(db_i))
    print("  loader speedup     =", db_speedup, "x")
    if db_i <= Int(Float64(serial_i) * 0.85):
        print("  RESULT: double-buffered prefetch would cut per-step time")
        print("          meaningfully in the streamed arm.")
    else:
        print("  RESULT: double-buffer gives little/no win here — either compute")
        print("          dominates (copy already hidden) or streams serialize.")

    # keep buffers alive to end of scope (ASAP-destroy guard).
    _ = host^
    _ = dev0^
    _ = dev1^
