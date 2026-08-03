# serenitymojo/tokenizer/parity/qwen3_tokenizer_bench.mojo
#
# Where the Qwen3 tokenizer spends its time, measured before anything is
# optimized.
#
# The MiniMax-H3 workload is short: a prompt of a few hundred tokens, encoded
# ONCE per request. So the number that matters for a cold click is construction
# — parsing a 7 MB tokenizer.json and building a 151k-entry vocabulary plus a
# 151k-entry merge table — not steady-state encode throughput. This benchmark
# separates the two so the optimization target is chosen from evidence rather
# than from the assumption that "tokenizer speed" means encode speed.
#
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/tokenizer/parity/qwen3_tokenizer_bench.mojo \
#     -o output/checks/qwen3_tokenizer_bench \
#   && output/checks/qwen3_tokenizer_bench

from std.collections import List
from std.time import perf_counter_ns

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer, _read_utf8_file

comptime TOKENIZER = "/home/alex/minimax_h3_ref/creator-MiniMax-H3/FL2VA/processor/tokenizer.json"

comptime PROMPT_SHORT = "a cat"
comptime PROMPT_TYPICAL = "A red fox trotting through a snowy pine forest, snow crunching underfoot, golden hour light through the branches, shallow depth of field, 35mm"
comptime PROMPT_LONG = (
    "A sweeping aerial shot over a rain-slicked neon city at night, camera "
    "descending past glass towers into a crowded street market where vendors "
    "sell steaming food under paper lanterns, reflections rippling in puddles, "
    "a lone figure in a red coat walking against the crowd, cinematic color "
    "grade, anamorphic lens flares, volumetric fog, 24fps, shallow depth of "
    "field, the camera slowly pushing in as she turns to look directly at us, "
    "her expression unreadable, the sound of rain and distant traffic"
)


def _seconds(start: UInt, stop: UInt) -> Float64:
    return Float64(Int(stop) - Int(start)) / 1.0e9


def main() raises:
    print("Qwen3 tokenizer benchmark")
    print("  tokenizer.json:", TOKENIZER)
    print("")

    # 0. The raw file read alone, to price a content hash against the parse.
    var r0 = perf_counter_ns()
    var raw = _read_utf8_file(String(TOKENIZER))
    var r1 = perf_counter_ns()
    print("[read]")
    print("  read 7 MB json     ", _seconds(r0, r1), "s  bytes", raw.byte_length())
    print("")

    # 1. Construction — the cold-start cost, paid once per process.
    var t0 = perf_counter_ns()
    var tokenizer = Qwen3Tokenizer(String(TOKENIZER))
    var t1 = perf_counter_ns()
    var load_seconds = _seconds(t0, t1)
    print("[construct]")
    print("  parse + build      ", load_seconds, "s")
    print("  special tokens     ", len(tokenizer.special_tokens))

    # 2. Encode — the per-request cost.
    var prompts = [String(PROMPT_SHORT), String(PROMPT_TYPICAL), String(PROMPT_LONG)]
    var names = [String("short"), String("typical"), String("long")]
    print("")
    print("[encode]")
    var total_encode = Float64(0.0)
    for i in range(len(prompts)):
        # One warm call so the first-touch page faults are not attributed to
        # the measured loop.
        var warm = tokenizer.encode(prompts[i])
        var iterations = 50
        var e0 = perf_counter_ns()
        for _ in range(iterations):
            var ids = tokenizer.encode(prompts[i])
            if len(ids) == 0:
                raise Error("empty encode")
        var e1 = perf_counter_ns()
        var per_call = _seconds(e0, e1) / Float64(iterations)
        total_encode += per_call
        print(
            "  ", names[i], "chars", prompts[i].byte_length(),
            "tokens", len(warm),
            "per call", per_call * 1000.0, "ms",
        )

    print("")
    print("[balance]")
    var typical_encode = total_encode / 3.0
    print("  construct / one typical encode ratio:", load_seconds / typical_encode)
    print("")
    print("  A cold request pays construct + one encode.")
    print("  Optimize whichever dominates that sum, not whichever is easier.")
