# Verify KV-cached decode == no-cache forward (token-for-token), and that the
# final argmax matches HF (" Paris", 12095).
from std.gpu.host import DeviceContext
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.llm.decoder import KVCache, decode_step

comptime SNAP = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

def _argmax(t: List[Float32]) -> Int:
    var b = 0
    var bv = t[0]
    for i in range(1, len(t)):
        if t[i] > bv: bv = t[i]; b = i
    return b

def main() raises:
    var ctx = DeviceContext()
    var enc = Qwen3Encoder.load(SNAP, Qwen3Config.qwen3_06b(), ctx)
    var ids = List[Int]()
    for t in [785, 6722, 315, 9625, 374]:
        ids.append(t)

    # cached: feed tokens one at a time, build cache; capture per-step argmax
    var cache = KVCache(enc.config.num_layers)
    var cached_argmax = List[Int]()
    for pos in range(len(ids)):
        var logits = decode_step(enc, cache, ids[pos], pos, ctx)
        var host = logits.to_host(ctx)
        var fl = List[Float32]()
        for i in range(len(host)): fl.append(Float32(host[i]))
        cached_argmax.append(_argmax(fl))

    # no-cache reference: lm_logits_last on each prefix (padded to 512)
    var passed = 0
    var failed = 0
    for p in range(len(ids)):
        var prefix = List[Int]()
        for j in range(p + 1): prefix.append(ids[j])
        var padded = prefix.copy()
        for _ in range(512 - len(prefix)): padded.append(0)
        var refl = enc.lm_logits_last(padded, p, ctx)
        var rh = refl.to_host(ctx)
        var rf = List[Float32]()
        for i in range(len(rh)): rf.append(Float32(rh[i]))
        var ra = _argmax(rf)
        var ca = cached_argmax[p]
        print("pos", p, " cached_argmax", ca, " nocache_argmax", ra, " match", ca == ra)
        if ca == ra: passed += 1
        else: failed += 1

    print("final pos-4 cached argmax:", cached_argmax[len(ids)-1], "(HF=12095 ' Paris')")
    print("passed:", passed, " failed:", failed)
    if failed == 0 and cached_argmax[len(ids)-1] == 12095:
        print("ALL DECODER CACHE TESTS PASSED")
