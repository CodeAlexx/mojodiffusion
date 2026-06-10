# CPU verification of the LLM token sampler (no GPU).
from serenitymojo.llm.sampler import Rng, SamplerConfig, sample, argmax


def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond: p += 1
    else:
        f += 1
        print("  FAIL:", name)


def main() raises:
    var p = 0
    var f = 0

    # logits with a clear max at index 3
    var lg = List[Float32]()
    for v in [Float32(0.1), 0.5, 0.2, 5.0, 0.3, 1.0, 0.4, 2.0]:
        lg.append(v)

    # greedy / argmax
    check(p, f, argmax(lg) == 3, "argmax index 3")
    var greedy_cfg = SamplerConfig(temperature=0.0)
    var rng0 = Rng(123)
    check(p, f, sample(lg, greedy_cfg, rng0) == 3, "temperature<=0 -> greedy")

    # top_k = 1 always returns the argmax regardless of temperature/seed
    var k1 = SamplerConfig(temperature=1.0, top_k=1, top_p=1.0, seed=7)
    var allk1 = True
    var rngk = Rng(7)
    for _ in range(200):
        if sample(lg, k1, rngk) != 3:
            allk1 = False
    check(p, f, allk1, "top_k=1 always argmax")

    # top_k = 2 -> only the two largest indices (3 and 7) ever sampled
    var k2 = SamplerConfig(temperature=1.0, top_k=2, top_p=1.0, seed=42)
    var rngk2 = Rng(42)
    var only_top2 = True
    var saw3 = False
    var saw7 = False
    for _ in range(500):
        var t = sample(lg, k2, rngk2)
        if t == 3: saw3 = True
        elif t == 7: saw7 = True
        else: only_top2 = False
    check(p, f, only_top2, "top_k=2 only samples {3,7}")
    check(p, f, saw3 and saw7, "top_k=2 samples both 3 and 7 over 500 draws")

    # very low temperature ~ greedy (peaked)
    var lowt = SamplerConfig(temperature=0.01, top_k=0, top_p=1.0, seed=1)
    var rng_lt = Rng(1)
    var lt_all3 = True
    for _ in range(100):
        if sample(lg, lowt, rng_lt) != 3:
            lt_all3 = False
    check(p, f, lt_all3, "very low temperature -> argmax")

    # determinism: same seed -> same sequence
    var cfg = SamplerConfig(temperature=1.0, top_k=0, top_p=0.9, seed=999)
    var ra = Rng(999)
    var rb = Rng(999)
    var seq_match = True
    for _ in range(50):
        if sample(lg, cfg, ra) != sample(lg, cfg, rb):
            seq_match = False
    check(p, f, seq_match, "same seed -> identical token sequence")

    # RNG uniform spans the full [0,1) range (guards the [0,0.5) bug)
    var rng_u = Rng(2024)
    var umin = 1.0
    var umax = 0.0
    var usum = 0.0
    var N = 20000
    for _ in range(N):
        var u = rng_u.next_f64()
        if u < umin: umin = u
        if u > umax: umax = u
        usum += u
    var mean = usum / Float64(N)
    check(p, f, umax > 0.95, "RNG reaches > 0.95 (not capped at 0.5)")
    check(p, f, umin < 0.05, "RNG reaches < 0.05")
    check(p, f, mean > 0.45 and mean < 0.55, "RNG mean ~ 0.5 (got " + String(mean) + ")")

    # top_p tiny -> nucleus is just the top token (argmax)
    var tp = SamplerConfig(temperature=1.0, top_k=0, top_p=0.0001, seed=5)
    var rng_tp = Rng(5)
    var tp_all3 = True
    for _ in range(100):
        if sample(lg, tp, rng_tp) != 3:
            tp_all3 = False
    check(p, f, tp_all3, "top_p tiny -> argmax (nucleus = top token)")

    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL SAMPLER TESTS PASSED")
