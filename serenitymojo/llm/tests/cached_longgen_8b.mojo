# cached_longgen_8b.mojo — how does cached ms/tok scale with generation length L?
# The captioner generates up to 1700 tokens; the cached path's per-token cost
# grows with L (K/V-cache host round-trip + O(L) concat). Measures cached-only
# (no-cache at maxseq=2048 is ~1749 ms/tok CONSTANT, already measured) at several
# N so we can see the L-growth curve and compare to that 1749 ms/tok ceiling.
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.text_encoder.qwen3_magic import generate_greedy_cached

comptime QWEN = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
comptime EOS = 151645


def _ms(ns0: UInt, ns1: UInt) -> Float64:
    return Float64(Int(ns1) - Int(ns0)) / 1.0e6


def _probe(enc: Qwen3Encoder, ids: List[Int], N: Int, ctx: DeviceContext) raises:
    var t0 = perf_counter_ns()
    var gen = generate_greedy_cached(enc, ids, N, EOS, ctx)
    var t1 = perf_counter_ns()
    var ms = _ms(t0, t1)
    print("N=", N, " genlen=", len(gen), " total_ms=", ms,
          " ms/tok=", ms / Float64(len(gen)) if len(gen) > 0 else 0.0,
          " (no-cache@2048 ceiling ~1749 ms/tok)")


def main() raises:
    var ctx = DeviceContext()
    var enc = Qwen3Encoder.load(QWEN, Qwen3Config.klein_9b(), ctx)
    var ids = List[Int]()
    for t in [785, 6722, 315, 9625, 374]:
        ids.append(t)
    _probe(enc, ids, 64, ctx)
    _probe(enc, ids, 256, ctx)
    _probe(enc, ids, 512, ctx)
    _probe(enc, ids, 1024, ctx)
