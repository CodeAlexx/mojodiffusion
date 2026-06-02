# offload/telemetry.mojo - shared runtime counters for block offload.
#
# Mirrors the measurement surface of Flame Core's offload telemetry, but keeps
# the first Mojo version deliberately simple: counters live on the loader and
# print concise summaries. Trainers can later mirror these into SerenityBoard.

from std.time import perf_counter_ns


struct OffloadTelemetry(Movable):
    var name: String
    var enabled: Bool
    var trace: Bool
    var summary_every: Int
    var next_summary: Int
    var prefetch_calls: Int
    var prefetch_hits: Int
    var await_calls: Int
    var await_hits: Int
    var await_fallbacks: Int
    var h2d_bytes: Int
    var prefetch_ns: Int
    var await_ns: Int

    def __init__(
        out self,
        name: String,
        enabled: Bool = True,
        trace: Bool = False,
        summary_every: Int = 64,
    ):
        self.name = name
        self.enabled = enabled
        self.trace = trace
        self.summary_every = summary_every if summary_every > 0 else 64
        self.next_summary = self.summary_every
        self.prefetch_calls = 0
        self.prefetch_hits = 0
        self.await_calls = 0
        self.await_hits = 0
        self.await_fallbacks = 0
        self.h2d_bytes = 0
        self.prefetch_ns = 0
        self.await_ns = 0

    def now_ns(self) -> Int:
        if not self.enabled:
            return 0
        return Int(perf_counter_ns())

    def record_prefetch_hit(mut self, prefix: String):
        if not self.enabled:
            return
        self.prefetch_hits += 1
        if self.trace:
            print("[offload/turbo]", self.name, "prefetch_hit", prefix)

    def record_prefetch(
        mut self,
        prefix: String,
        nbytes: Int,
        elapsed_ns: Int,
    ):
        if not self.enabled:
            return
        self.prefetch_calls += 1
        self.h2d_bytes += nbytes
        self.prefetch_ns += elapsed_ns
        if self.trace:
            print(
                "[offload/turbo]", self.name,
                "prefetch", prefix,
                "bytes", nbytes,
                "ms", Float64(elapsed_ns) / 1.0e6,
            )
        self.maybe_summary()

    def record_await(
        mut self,
        prefix: String,
        slot_hit: Bool,
        elapsed_ns: Int,
    ):
        if not self.enabled:
            return
        self.await_calls += 1
        if slot_hit:
            self.await_hits += 1
        else:
            self.await_fallbacks += 1
        self.await_ns += elapsed_ns
        if self.trace:
            print(
                "[offload/turbo]", self.name,
                "await", prefix,
                "hit", slot_hit,
                "ms", Float64(elapsed_ns) / 1.0e6,
            )

    def maybe_summary(mut self):
        if not self.enabled:
            return
        if self.prefetch_calls < self.next_summary:
            return
        self.print_summary()
        self.next_summary += self.summary_every

    def print_summary(self):
        if not self.enabled:
            return
        var prefetch_avg_ms = Float64(0.0)
        if self.prefetch_calls > 0:
            prefetch_avg_ms = (
                Float64(self.prefetch_ns) / 1.0e6 / Float64(self.prefetch_calls)
            )
        var await_avg_ms = Float64(0.0)
        if self.await_calls > 0:
            await_avg_ms = (
                Float64(self.await_ns) / 1.0e6 / Float64(self.await_calls)
            )
        print(
            "[offload/turbo]", self.name,
            "prefetch", self.prefetch_calls,
            "prefetch_hits", self.prefetch_hits,
            "await", self.await_calls,
            "await_hits", self.await_hits,
            "fallbacks", self.await_fallbacks,
            "h2d_mib", Float64(self.h2d_bytes) / 1048576.0,
            "prefetch_avg_ms", prefetch_avg_ms,
            "await_avg_ms", await_avg_ms,
        )
