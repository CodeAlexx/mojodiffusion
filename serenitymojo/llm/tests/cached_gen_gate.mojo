# cached_gen_gate.mojo — Phase 1 gate for the KV-cache captioner speedup.
# Proves generate_greedy_cached == generate_greedy TOKEN-FOR-TOKEN (greedy is
# deterministic → exact equality, not tolerance) AND times both so the speedup
# is MEASURED, not assumed. Qwen3-0.6B (the checkpoint the decoder gate uses).
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.text_encoder.qwen3_magic import (
    generate_greedy, generate_greedy_cached,
)

comptime SNAP = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
comptime EOS = 151645
comptime PAD = 151643


def _ms(ns0: UInt, ns1: UInt) -> Float64:
    return Float64(Int(ns1) - Int(ns0)) / 1.0e6


def _run(enc: Qwen3Encoder, ids: List[Int], N: Int, MAXSEQ: Int, ctx: DeviceContext) raises -> Bool:
    var t0 = perf_counter_ns()
    var ref_gen = generate_greedy(enc, ids, N, EOS, PAD, MAXSEQ, ctx)
    var t1 = perf_counter_ns()
    var cac_gen = generate_greedy_cached(enc, ids, N, EOS, ctx)
    var t2 = perf_counter_ns()

    var same = len(ref_gen) == len(cac_gen)
    if same:
        for i in range(len(ref_gen)):
            if ref_gen[i] != cac_gen[i]:
                same = False
                break
    var nc = _ms(t0, t1)
    var ca = _ms(t1, t2)
    print("── N=", N, " MAXSEQ=", MAXSEQ, " genlen=", len(ref_gen))
    print("   no-cache ms:", nc, "  cached ms:", ca,
          "  speedup:", (nc / ca) if ca > 0.0 else 0.0, "x",
          "  gate:", "PASS" if same else "FAIL")
    return same


def main() raises:
    var ctx = DeviceContext()
    var enc = Qwen3Encoder.load(SNAP, Qwen3Config.qwen3_06b(), ctx)
    var ids = List[Int]()
    for t in [785, 6722, 315, 9625, 374]:   # "The capital of France is"
        ids.append(t)

    # 0.6B SDPA (h=16) is comptime-capped at seq=512, so sweep within that.
    # Production seq width (2048) is measured on the real 8B in cached_gen_gate_8b.
    var all_pass = True
    all_pass = _run(enc, ids, 48, 256, ctx) and all_pass
    all_pass = _run(enc, ids, 48, 512, ctx) and all_pass
    all_pass = _run(enc, ids, 128, 512, ctx) and all_pass     # long gen: stress cached L growth
    if all_pass:
        print("ALL GATES PASS (cached == no-cache token-for-token, every config)")
    else:
        print("GATE FAIL")
