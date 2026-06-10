# Logit-parity baseline: serenitymojo Qwen3-0.6B forward vs HuggingFace.
# Feeds the EXACT HF token ids for "The capital of France is" so the test
# isolates the transformer forward from any tokenizer difference.
# HF reference: argmax next-token = 12095 (" Paris"), logit ~17.375.
from std.gpu.host import DeviceContext
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config

comptime SNAP = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
comptime MAXSEQ = 512


def main() raises:
    var ctx = DeviceContext()
    var enc = Qwen3Encoder.load(SNAP, Qwen3Config.qwen3_06b(), ctx)

    # exact HF input_ids for "The capital of France is"
    var ids = List[Int]()
    for t in [785, 6722, 315, 9625, 374]:
        ids.append(t)
    var pos = len(ids) - 1   # last real token = position 4
    var padded = ids.copy()
    for _ in range(MAXSEQ - len(ids)):
        padded.append(0)

    var logits = enc.lm_logits_last(padded, pos, ctx)   # [1,1,vocab]
    var host = logits.to_host(ctx)
    var vocab = len(host)
    print("vocab:", vocab)

    # argmax + top5
    var best = 0
    var bv = host[0]
    for i in range(1, vocab):
        if host[i] > bv:
            bv = host[i]
            best = i
    print("argmax id:", best, " logit:", bv)
    print("HF reference argmax id: 12095 (' Paris'), logit ~17.375")
    if best == 12095:
        print("PARITY: argmax MATCHES HuggingFace")
    else:
        print("PARITY: MISMATCH (argmax", best, "!= 12095)")
