# memory/bench.mojo — tiny benchmark timer for the allocator suite.
#
# perf_counter_ns() is a monotonic nanosecond clock. NOTE: the optimizer will
# delete a loop whose result is never observed (a dead loop times ~30ns). Every
# benchmark must accumulate a data-dependent value (e.g. Int(ptr) low bits) into
# a `sink` and pass it to report() so the work cannot be proven dead.

from std.time import perf_counter_ns


struct Timer(Movable):
    var _t0: Int

    @staticmethod
    def start() -> Timer:
        return Timer(Int(perf_counter_ns()))

    def __init__(out self, t0: Int):
        self._t0 = t0

    @always_inline
    def elapsed_ns(self) -> Int:
        return Int(perf_counter_ns()) - self._t0


def report(name: String, iters: Int, ns: Int, sink: Int):
    """Print ns/op and Mops/s. `sink` keeps the benched work observable."""
    var per = Float64(ns) / Float64(iters) if iters > 0 else 0.0
    var mops = (Float64(iters) * 1000.0) / Float64(ns) if ns > 0 else 0.0
    print(
        name,
        ":",
        iters,
        "ops /",
        ns,
        "ns =",
        per,
        "ns/op,",
        mops,
        "Mops/s  [sink=",
        sink & 0xFFFF,
        "]",
    )


def report_speedup(name: String, baseline_ns: Int, fast_ns: Int):
    var x = Float64(baseline_ns) / Float64(fast_ns) if fast_ns > 0 else 0.0
    print(name, ": speedup =", x, "x  (", baseline_ns, "ns ->", fast_ns, "ns )")
