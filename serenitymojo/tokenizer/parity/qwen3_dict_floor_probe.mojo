# serenitymojo/tokenizer/parity/qwen3_dict_floor_probe.mojo
#
# What does a binary tokenizer cache actually buy?
#
# Construction of `Qwen3Tokenizer` costs 0.180 s, of which the raw file read is
# 0.0085 s — so ~0.172 s is "parse the JSON and fill two Dicts". A binary cache
# removes the PARSING but still has to fill the same two Dicts with the same
# 151643 + 151387 entries.
#
# This probe measures that irreducible floor before any cache is written. If
# filling the Dicts is most of the 0.172 s, a cache is not worth building and
# the answer is a different data structure (or not paying the cost per process
# at all).
#
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/tokenizer/parity/qwen3_dict_floor_probe.mojo \
#     -o output/checks/qwen3_dict_floor_probe \
#   && output/checks/qwen3_dict_floor_probe

from std.collections import List
from std.time import perf_counter_ns

comptime VOCAB_ENTRIES = 151643
comptime MERGE_ENTRIES = 151387


def _seconds(start: UInt, stop: UInt) -> Float64:
    return Float64(Int(stop) - Int(start)) / 1.0e9


def _make_keys(count: Int, prefix: String) -> List[String]:
    """Keys shaped like real vocabulary entries — short, mostly distinct."""
    var out = List[String]()
    for i in range(count):
        out.append(prefix + String(i))
    return out^


def main() raises:
    print("Qwen3 tokenizer Dict-fill floor probe")
    print("")

    # Key construction is not part of the floor; build them first.
    var vocab_keys = _make_keys(VOCAB_ENTRIES, String("v"))
    var merge_keys = _make_keys(MERGE_ENTRIES, String("m"))

    var t0 = perf_counter_ns()
    var vocab = Dict[String, Int]()
    for i in range(len(vocab_keys)):
        vocab[vocab_keys[i]] = i
    var t1 = perf_counter_ns()

    var merge_rank = Dict[String, Int]()
    for i in range(len(merge_keys)):
        merge_rank[merge_keys[i]] = i
    var t2 = perf_counter_ns()

    var vocab_seconds = _seconds(t0, t1)
    var merge_seconds = _seconds(t1, t2)
    print("  vocab fill  ", VOCAB_ENTRIES, "entries", vocab_seconds, "s")
    print("  merge fill  ", MERGE_ENTRIES, "entries", merge_seconds, "s")
    print("  total floor ", vocab_seconds + merge_seconds, "s")
    print("")
    print("  measured construct total: 0.180 s   raw read: 0.0085 s")
    print("  a binary cache can only remove what is ABOVE this floor.")
    # Keep both alive so the fills are not optimized away.
    print("  (kept:", len(vocab), len(merge_rank), ")")
