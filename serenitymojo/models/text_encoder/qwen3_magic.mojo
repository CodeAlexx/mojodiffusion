# models/text_encoder/qwen3_magic.mojo — pure-Mojo autoregressive greedy decode
# for Qwen3-8B (the "magic prompt": plain text -> structured JSON caption).
# Reuses Qwen3Encoder's forward + the checkpoint's lm_head via lm_logits_last.
# No KV-cache yet: re-forwards the (padded) context each token, so it's correct
# but O(steps*seq) — fine for a one-shot caption, KV-cache is the speed follow-up.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder
from serenitymojo.llm.decoder import KVCache, decode_step


def _argmax_host(logits: Tensor, ctx: DeviceContext) raises -> Int:
    """argmax over a [1,1,vocab] logits tensor (host reduction)."""
    var host = logits.to_host(ctx)
    var best = 0
    var bv = host[0]
    for i in range(1, len(host)):
        if host[i] > bv:
            bv = host[i]
            best = i
    return best


def generate_greedy_cached_resume(
    qwen: Qwen3Encoder,
    mut cache: KVCache,
    start_pos: Int,
    tail_ids: List[Int],
    max_new: Int,
    eos: Int,
    ctx: DeviceContext,
) raises -> List[Int]:
    """Greedy decode RESUMING from a pre-primed KVCache covering positions
    [0, start_pos): primes `tail_ids` at absolute positions start_pos.., then
    generates. With cache = the saved fixed-prefix K/V (llm/prefix_cache) and
    tail_ids = the variable prompt tail, output is token-for-token identical to
    generate_greedy_cached over prefix+tail (same math, positions, and bytes) —
    gated by llm/tests/prefix_cache_gate.mojo. tail_ids must be non-empty (the
    last tail token's logits seed generation)."""
    if len(tail_ids) == 0:
        raise Error("generate_greedy_cached_resume: tail_ids must be non-empty")
    var n = len(tail_ids)
    for i in range(n - 1):
        _ = decode_step(qwen, cache, tail_ids[i], start_pos + i, ctx, want_logits=False)
    var last_logits = decode_step(
        qwen, cache, tail_ids[n - 1], start_pos + n - 1, ctx, want_logits=True
    )

    var gen = List[Int]()
    var pos = start_pos + n
    for _ in range(max_new):
        var best = _argmax_host(last_logits, ctx)
        if best == eos:
            break
        gen.append(best)
        last_logits = decode_step(qwen, cache, best, pos, ctx)
        pos += 1
    return gen^


def generate_greedy_cached(
    qwen: Qwen3Encoder,
    prompt_ids: List[Int],
    max_new: Int,
    eos: Int,
    ctx: DeviceContext,
) raises -> List[Int]:
    """KV-cached greedy decode — token-for-token IDENTICAL to `generate_greedy`
    but O(steps) not O(steps*seq): each new token forwards ONE position against a
    persistent per-layer K/V cache (llm.decoder.decode_step) instead of re-
    forwarding the whole padded context. Returns the GENERATED ids (excluding the
    prompt); stops on `eos`. No `maxseq`/`pad` — the cache grows to the true
    length, so there is no fixed-seq padding.

    Prime: feed the prompt one token at a time (pos 0..n-1); the LAST prime's
    logits are the distribution for the first generated token — exactly what
    `generate_greedy`'s first iteration (pos=len-1 over the full prefix) produces.
    """
    var cache = KVCache(qwen.config.num_layers)
    var gen = List[Int]()
    if len(prompt_ids) == 0:
        return gen^

    # Prime the prompt. Only the LAST prompt token's logits are needed (they give
    # the first generated token), so prime tokens 0..n-2 with want_logits=False —
    # fills the cache but skips the discarded lm_head GEMM per prompt token (big
    # win for the ~6.4k-token captioner prompt). The last prime keeps its logits.
    var n = len(prompt_ids)
    for pos in range(n - 1):
        _ = decode_step(qwen, cache, prompt_ids[pos], pos, ctx, want_logits=False)
    var last_logits = decode_step(qwen, cache, prompt_ids[n - 1], n - 1, ctx, want_logits=True)

    var pos = len(prompt_ids)
    for _ in range(max_new):
        var best = _argmax_host(last_logits, ctx)
        if best == eos:
            break
        gen.append(best)
        last_logits = decode_step(qwen, cache, best, pos, ctx)
        pos += 1
    return gen^


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
