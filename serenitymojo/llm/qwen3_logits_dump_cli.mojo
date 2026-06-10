# Dump serenitymojo Qwen3-0.6B last-token logits (F32 binary) at a fixed prefix,
# for per-step parity vs HF. Prefix = prompt + first 5 (matched) greedy tokens.
from std.gpu.host import DeviceContext
from std.memory import alloc, UnsafePointer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config

comptime SNAP = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

def main() raises:
    var ctx = DeviceContext()
    var enc = Qwen3Encoder.load(SNAP, Qwen3Config.qwen3_06b(), ctx)
    var ids = List[Int]()
    for t in [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315]:
        ids.append(t)
    var pos = len(ids) - 1
    var padded = ids.copy()
    for _ in range(512 - len(ids)):
        padded.append(0)
    var logits = enc.lm_logits_last(padded, pos, ctx)
    var host = logits.to_host(ctx)
    var vocab = len(host)

    # serialize vocab F32 little-endian
    var buf = List[UInt8]()
    var tmp = alloc[Float32](1)
    for i in range(vocab):
        tmp[0] = Float32(host[i])
        var bp = tmp.bitcast[UInt8]()
        buf.append(bp[0]); buf.append(bp[1]); buf.append(bp[2]); buf.append(bp[3])
    tmp.free()
    with open("/tmp/mojo_logits.bin", "w") as f:
        f.write_bytes(Span(buf))

    var best = 0
    var bv = host[0]
    for i in range(1, vocab):
        if host[i] > bv:
            bv = host[i]; best = i
    print("mojo vocab:", vocab, " argmax:", best, " logit:", bv)
    print("mojo logit[9625 France]:", host[9625], " logit[15344]:", host[15344])
