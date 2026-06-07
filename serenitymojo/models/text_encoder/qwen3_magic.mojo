# models/text_encoder/qwen3_magic.mojo — pure-Mojo autoregressive greedy decode
# for Qwen3-8B (the "magic prompt": plain text -> structured JSON caption).
# Reuses Qwen3Encoder's forward + the checkpoint's lm_head via lm_logits_last.
# No KV-cache yet: re-forwards the (padded) context each token, so it's correct
# but O(steps*seq) — fine for a one-shot caption, KV-cache is the speed follow-up.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder


def generate_greedy(
    qwen: Qwen3Encoder,
    prompt_ids: List[Int],
    max_new: Int,
    eos: Int,
    pad: Int,
    maxseq: Int,
    ctx: DeviceContext,
) raises -> List[Int]:
    """Greedy-decode up to max_new tokens after prompt_ids; returns the GENERATED
    ids (excluding the prompt). Stops on `eos`. `maxseq` must be a supported
    comptime sdpa seq (e.g. 1024)."""
    var ids = prompt_ids.copy()
    var gen = List[Int]()
    for _ in range(max_new):
        if len(ids) >= maxseq:
            break
        var pos = len(ids) - 1
        var padded = ids.copy()
        for _ in range(maxseq - len(ids)):
            padded.append(pad)
        var logits = qwen.lm_logits_last(padded, pos, ctx)   # [1,1,vocab]
        var host = logits.to_host(ctx)
        var best = 0
        var bv = host[0]
        for i in range(1, len(host)):
            if host[i] > bv:
                bv = host[i]
                best = i
        if best == eos:
            break
        ids.append(best)
        gen.append(best)
    return gen^
