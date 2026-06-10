# serenitymojo/llm/sampler.mojo — LLM token sampler (CPU, operates on host logits).
#
# After the decoder forward produces the last-token logits, they are copied to
# host and a next token is chosen here. Pure CPU float math — no GPU, no Tensor —
# so it is fully unit-testable without a device. Supports greedy + temperature +
# top-k + top-p (nucleus) with a correct uniform PRNG.
#
# RNG NOTE: uses xorshift64* and forms a uniform in [0,1) as (u64 >> 11) / 2^53,
# i.e. a full 53-bit mantissa — deliberately NOT the (mask52 / 2^53) form that
# only spans [0, 0.5) (a bug that has bitten this repo's noise code before).

from std.math import exp


struct Rng(Copyable, Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        # Avoid a zero state (xorshift fixed point); splitmix-ish warmup.
        var s = seed
        if s == 0:
            s = 0x9E3779B97F4A7C15
        self.state = s

    @always_inline
    def next_u64(mut self) -> UInt64:
        var x = self.state
        x = x ^ (x >> 12)
        x = x ^ (x << 25)
        x = x ^ (x >> 27)
        self.state = x
        return x * 0x2545F4914F6CDD1D

    @always_inline
    def next_f64(mut self) -> Float64:
        # 53-bit mantissa uniform in [0, 1).
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)  # 2^53


struct SamplerConfig(Copyable, Movable):
    var temperature: Float64   # <= 0 -> greedy
    var top_k: Int             # <= 0 -> disabled
    var top_p: Float64         # >= 1 or <= 0 -> disabled
    var seed: UInt64

    def __init__(
        out self,
        temperature: Float64 = 1.0,
        top_k: Int = 0,
        top_p: Float64 = 1.0,
        seed: UInt64 = 0,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed


def argmax(logits: List[Float32]) -> Int:
    """Greedy: index of the maximum logit."""
    var best = 0
    var bestv = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > bestv:
            bestv = logits[i]
            best = i
    return best


def _order_desc(logits: List[Float32]) -> List[Int]:
    """Indices of logits sorted descending (insertion sort; vocab fits fine for
    top-k/top-p which only need the head, but we sort fully for clarity)."""
    var idx = List[Int]()
    for i in range(len(logits)):
        idx.append(i)
    # insertion sort by logit desc
    for i in range(1, len(idx)):
        var j = i
        var cur = idx[i]
        var curv = logits[cur]
        while j > 0 and logits[idx[j - 1]] < curv:
            idx[j] = idx[j - 1]
            j -= 1
        idx[j] = cur
    return idx^


def sample(logits: List[Float32], cfg: SamplerConfig, mut rng: Rng) raises -> Int:
    """Choose a token id from logits under the sampler config."""
    var n = len(logits)
    if n == 0:
        raise Error("sample: empty logits")
    # Greedy fast path.
    if cfg.temperature <= 0.0:
        return argmax(logits)

    # Candidate set: full vocab, optionally restricted to top-k.
    var order = _order_desc(logits)
    var keep = n
    if cfg.top_k > 0 and cfg.top_k < n:
        keep = cfg.top_k

    # Temperature-scaled softmax over the kept candidates (numerically stable).
    var inv_t = 1.0 / cfg.temperature
    var maxl = Float64(logits[order[0]]) * inv_t
    var probs = List[Float64]()
    var total = 0.0
    for i in range(keep):
        var z = Float64(logits[order[i]]) * inv_t - maxl
        var p = exp(z)
        probs.append(p)
        total += p
    # normalize
    for i in range(keep):
        probs[i] = probs[i] / total

    # top-p (nucleus): keep the smallest prefix whose cumulative prob >= top_p.
    var cutoff = keep
    if cfg.top_p > 0.0 and cfg.top_p < 1.0:
        var cum = 0.0
        cutoff = 0
        for i in range(keep):
            cum += probs[i]
            cutoff = i + 1
            if cum >= cfg.top_p:
                break
        # renormalize over the nucleus
        var renorm = 0.0
        for i in range(cutoff):
            renorm += probs[i]
        for i in range(cutoff):
            probs[i] = probs[i] / renorm

    # Sample from the categorical over [0, cutoff).
    var u = rng.next_f64()
    var acc = 0.0
    for i in range(cutoff):
        acc += probs[i]
        if u < acc:
            return order[i]
    return order[cutoff - 1]  # numerical fallback
