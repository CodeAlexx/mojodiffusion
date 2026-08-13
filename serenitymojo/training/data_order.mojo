# Deterministic, resume-stable epoch shuffling for Mojo-native trainers.
#
# The output is indexed by absolute sample draw, not by a mutable RNG cursor.
# Rebuilding it after resume therefore yields the same sample at every global
# draw as an uninterrupted run. Fisher-Yates gives each epoch a permutation;
# the epoch number is mixed with the configured run seed.

from std.collections import List


def _data_order_mix64(state: UInt64) -> UInt64:
    var z = state + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def shuffled_epoch_order(
    base: List[Int], seed: UInt64, epoch: Int
) raises -> List[Int]:
    if len(base) < 1:
        raise Error("shuffled_epoch_order: empty base order")
    if epoch < 0:
        raise Error("shuffled_epoch_order: epoch must be non-negative")
    var out = base.copy()
    var state = _data_order_mix64(
        seed ^ (UInt64(epoch) * UInt64(0xD1B54A32D192ED03))
    )
    for i in range(len(out) - 1, 0, -1):
        state = _data_order_mix64(state + UInt64(i))
        var j = Int(state % UInt64(i + 1))
        if j != i:
            var tmp = out[i]
            out[i] = out[j]
            out[j] = tmp
    return out^


def build_epoch_sample_order(
    base: List[Int], seed: UInt64, draws: Int
) raises -> List[Int]:
    """Concatenate deterministic epoch permutations until `draws` entries exist.

    A trainer uses sample_order[global_draw]. For batch 1, global_draw=step; for
    batch 2, the pair is [2*step, 2*step+1]. No cursor state is required.
    """
    if draws < 0:
        raise Error("build_epoch_sample_order: draws must be non-negative")
    if len(base) < 1:
        raise Error("build_epoch_sample_order: empty base order")
    var out = List[Int](capacity=draws)
    var epoch = 0
    while len(out) < draws:
        var eo = shuffled_epoch_order(base, seed, epoch)
        for i in range(len(eo)):
            if len(out) == draws:
                break
            out.append(eo[i])
        epoch += 1
    return out^


def deterministic_draw_uniform(
    seed: UInt64, absolute_draw: Int, stream: UInt64
) raises -> Float32:
    """Resume-stable U[0,1) draw for per-sample policy decisions.

    Independent stream ids let text and condition dropout be independent while
    remaining functions of the absolute draw rather than mutable RNG state.
    """
    if absolute_draw < 0:
        raise Error("deterministic_draw_uniform: negative absolute draw")
    var word = _data_order_mix64(
        seed ^ (UInt64(absolute_draw) * UInt64(0xD1B54A32D192ED03))
        ^ (stream * UInt64(0x94D049BB133111EB))
    )
    var mantissa24 = (word >> 40) & UInt64(0xFFFFFF)
    return Float32(mantissa24) / Float32(16777216.0)


def _check(cond: Bool, message: String) raises:
    if not cond:
        raise Error("FAIL " + message)
    print("PASS", message)


def main() raises:
    var base = List[Int]()
    for i in range(17):
        base.append(i)
    var a = build_epoch_sample_order(base, UInt64(1234), 41)
    var b = build_epoch_sample_order(base, UInt64(1234), 41)
    _check(a == b, "same seed/global draws reproduce exactly")

    for epoch in range(2):
        var seen = List[Bool]()
        for _ in range(17):
            seen.append(False)
        for p in range(17):
            var idx = a[epoch * 17 + p]
            _check(idx >= 0 and idx < 17, "index is in range")
            _check(not seen[idx], "no duplicate within epoch")
            seen[idx] = True
        for i in range(17):
            _check(seen[i], "epoch contains every sample")

    var c = build_epoch_sample_order(base, UInt64(1235), 41)
    var differs = False
    for i in range(41):
        if a[i] != c[i]:
            differs = True
            break
    _check(differs, "different seeds change order")

    var prefix = build_epoch_sample_order(base, UInt64(1234), 23)
    var prefix_exact = True
    for i in range(23):
        if prefix[i] != a[i]:
            prefix_exact = False
            break
    _check(prefix_exact, "resume rebuild preserves absolute-draw prefix")
    print("data_order: ALL CHECKS PASSED")
