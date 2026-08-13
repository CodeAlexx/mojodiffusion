# cached_gen_gate_8b.mojo — PRODUCTION-model gate for the captioner speedup.
# Qwen3-8B (klein_9b config) — the EXACT model + config ideogram4_magic loads.
# Proves generate_greedy_cached == generate_greedy token-for-token AND times both
# at the captioner's real seq width (maxseq=2048) vs 512, on the real model.
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.text_encoder.qwen3_magic import (
    generate_greedy, generate_greedy_cached,
)

comptime QWEN = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
comptime EOS = 151645      # <|im_end|>
comptime PAD = 151643      # <|endoftext|>


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
          "  ms/tok no-cache:", nc / Float64(len(ref_gen)) if len(ref_gen) > 0 else 0.0,
          "  ms/tok cached:", ca / Float64(len(cac_gen)) if len(cac_gen) > 0 else 0.0,
          "  gate:", "PASS" if same else "FAIL")
    return same


def main() raises:
    var ctx = DeviceContext()
    var enc = Qwen3Encoder.load(QWEN, Qwen3Config.klein_9b(), ctx)
    var ids = List[Int]()
    for t in [785, 6722, 315, 9625, 374]:   # "The capital of France is"
        ids.append(t)

    var all_pass = True
    all_pass = _run(enc, ids, 24, 512, ctx) and all_pass    # 8B @ 512 seq width
    all_pass = _run(enc, ids, 24, 2048, ctx) and all_pass   # 8B @ production 2048 seq width
    if all_pass:
        print("ALL GATES PASS (8B: cached == no-cache token-for-token, every config)")
    else:
        print("GATE FAIL")
