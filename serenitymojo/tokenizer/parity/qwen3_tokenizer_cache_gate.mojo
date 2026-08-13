# serenitymojo/tokenizer/parity/qwen3_tokenizer_cache_gate.mojo
#
# The tokenizer cache must be FASTER and IDENTICAL. This gate proves both, and
# the second is the one that matters: a cache that returns different token ids
# than the JSON path corrupts conditioning silently — no crash, no obviously
# wrong output, just a model conditioned on the wrong prompt.
#
# Checks:
#   1. every vocab, merge and special entry survives the round trip
#   2. encoding agrees on a spread of prompts, including unicode and the
#      MiniMax-H3 label strings with their vision specials
#   3. a corrupted source is REJECTED rather than silently accepted
#   4. the cache is faster than parsing the JSON
#
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/tokenizer/parity/qwen3_tokenizer_cache_gate.mojo \
#     -o output/checks/qwen3_tokenizer_cache_gate \
#   && output/checks/qwen3_tokenizer_cache_gate

from std.collections import List
from std.time import perf_counter_ns

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.tokenizer.tokenizer_cache import (
    qwen3_tokenizer_cache_write,
    qwen3_tokenizer_load_cached,
)

comptime SOURCE = "/home/alex/minimax_h3_ref/creator-MiniMax-H3/FL2VA/processor/tokenizer.json"
comptime CACHE = "/home/alex/mojodiffusion/output/checks/qwen3vl_32b_tokenizer.mjtok"

comptime PROMPTS = 12


def _seconds(start: Int, stop: Int) -> Float64:
    return Float64(Int(stop) - Int(start)) / 1.0e9


def _prompt(index: Int) -> String:
    if index == 0:
        return String("a cat")
    if index == 1:
        return String("A red fox trotting through a snowy pine forest, snow crunching underfoot")
    if index == 2:
        return String("夜の街を歩く女性、ネオンの光 — cinematic, 24fps")
    if index == 3:
        return String("Close-up: the subject's face, lit by neon. 35mm, f/1.4 — shallow depth!")
    if index == 4:
        return String("<Picture 1>: ")
    if index == 5:
        return String("<Audio 3>: ")
    if index == 6:
        return String("<Video 2>: ")
    if index == 7:
        return String("<0.2 seconds>")
    if index == 8:
        return String("<|vision_start|><|image_pad|><|vision_end|>")
    if index == 9:
        return String("   leading and trailing whitespace   ")
    if index == 10:
        return String("emoji 🎬 and punctuation!!! ...???")
    return String("1234567890 numbers 42 3.14159")


struct Report(Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def truth(mut self, label: String, condition: Bool):
        self.checks += 1
        if condition:
            print("  ok  ", label)
        else:
            self.failures += 1
            print("  FAIL", label)


def main() raises:
    print("Qwen3 tokenizer cache gate")
    print("  source:", SOURCE)
    print("  cache: ", CACHE)
    print("")

    print("[1] build from JSON, write the cache")
    var j0 = perf_counter_ns()
    var from_json = Qwen3Tokenizer(String(SOURCE))
    var j1 = perf_counter_ns()
    var json_seconds = _seconds(j0, j1)
    print("  construct from JSON ", json_seconds, "s")
    qwen3_tokenizer_cache_write(from_json, String(SOURCE), String(CACHE))

    print("")
    print("[2] load from the cache")
    var c0 = perf_counter_ns()
    var from_cache = qwen3_tokenizer_load_cached(String(SOURCE), String(CACHE))
    var c1 = perf_counter_ns()
    var cache_seconds = _seconds(c0, c1)
    print("  construct from cache", cache_seconds, "s")
    print("  speedup             ", json_seconds / cache_seconds, "x")

    var report = Report()
    print("")
    print("[3] tables round-trip")
    report.truth("vocab size matches", len(from_cache.vocab) == len(from_json.vocab))
    report.truth(
        "merge size matches", len(from_cache.merge_rank) == len(from_json.merge_rank)
    )
    report.truth(
        "special count matches",
        len(from_cache.special_tokens) == len(from_json.special_tokens),
    )
    report.truth("pre_o200k matches", from_cache.pre_o200k == from_json.pre_o200k)

    var vocab_bad = 0
    for entry in from_json.vocab.items():
        if entry.key not in from_cache.vocab:
            vocab_bad += 1
        elif from_cache.vocab[entry.key] != entry.value:
            vocab_bad += 1
    report.truth("every vocab entry identical", vocab_bad == 0)

    var merge_bad = 0
    for entry in from_json.merge_rank.items():
        if entry.key not in from_cache.merge_rank:
            merge_bad += 1
        elif from_cache.merge_rank[entry.key] != entry.value:
            merge_bad += 1
    report.truth("every merge entry identical", merge_bad == 0)

    var special_bad = 0
    for i in range(len(from_json.special_tokens)):
        if from_cache.special_tokens[i] != from_json.special_tokens[i]:
            special_bad += 1
        if from_cache.special_ids[i] != from_json.special_ids[i]:
            special_bad += 1
    report.truth("every special token identical", special_bad == 0)

    print("")
    print("[4] encodings agree")
    var encode_bad = 0
    for i in range(PROMPTS):
        var text = _prompt(i)
        var a = from_json.encode(text)
        var b = from_cache.encode(text)
        if len(a) != len(b):
            encode_bad += 1
            print("    length differs on prompt", i)
            continue
        for k in range(len(a)):
            if a[k] != b[k]:
                encode_bad += 1
                print("    id differs on prompt", i, "at", k)
                break
    report.truth("all prompts encode identically", encode_bad == 0)

    print("")
    print("[5] a mismatched source is rejected")
    var rejected = False
    try:
        # The cache was written for SOURCE; loading it against a different file
        # must fail the content check rather than return a usable tokenizer.
        var wrong = qwen3_tokenizer_load_cached(
            String("/home/alex/mojodiffusion/output/checks/qwen3vl_32b_tokenizer.mjtok"),
            String(CACHE),
        )
        _ = len(wrong.vocab)
    except:
        rejected = True
    report.truth("wrong source rejected", rejected)

    print("")
    print("[6] the cache is faster")
    report.truth("cache load beats JSON parse", cache_seconds < json_seconds)

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("Qwen3 tokenizer cache gate failed")
