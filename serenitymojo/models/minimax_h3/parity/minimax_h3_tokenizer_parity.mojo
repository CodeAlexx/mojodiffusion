# serenitymojo/models/minimax_h3/parity/minimax_h3_tokenizer_parity.mojo
#
# MiniMax-H3 tokenizer parity gate — the seven config-only special tokens.
#
# THE DEFECT THIS EXISTS FOR. Unit 7 built H3's prompt presentation on
# `Qwen3Tokenizer(tokenizer.json)` and gated it to 49/49 exact. That gate was
# green and the tokenizer was still wrong for H3, because none of its 49 cases
# contained a token that only H3 declares:
#
#     tokenizer.json      151643 vocab + 26 added_tokens  -> 151669
#     the real tokenizer                                     151676
#
# Seven tokens — <d> </d> <|cutoff|> <|lyrics_start|> <|lyrics_end|>
# <|caption_start|> <|caption_end|> — appear in NEITHER the vocab NOR
# `added_tokens`. They are declared only in `tokenizer_config.json`, and
# `transformers` appends them at load. H3's README is explicit that they are
# used and that H3's own config files are required.
#
# Reading tokenizer.json alone turns `<d>` into `<`, `d`, `>`: three ids where
# there should be one, a different text embedding, a different video. Silent,
# and invisible to any test whose prompts are plain prose.
#
# So this gate is deliberately built from the cases that break it — the seven
# tokens alone, embedded in prompt text, adjacent with no whitespace — plus
# tokens that WERE already in tokenizer.json and plain prose, to prove the
# merge did not disturb anything that already worked.
#
# BAR: exact. Token ids have no tolerance.
#
# Oracle: python3 scripts/minimax_h3_tokenizer_oracle.py
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_tokenizer_parity.mojo \
#     -o output/checks/minimax_h3_tokenizer_parity \
#   && output/checks/minimax_h3_tokenizer_parity

from std.collections import List
from std.pathlib import Path

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

comptime PROC = "/home/alex/minimax_h3_ref/creator-MiniMax-H3/FL2VA/processor"
comptime CASES = "/home/alex/mojodiffusion/output/minimax_h3_tokenizer/tokenizer_ref.txt"


def main() raises:
    print("MiniMax-H3 tokenizer parity gate (config-only special tokens)")
    var checks = 0
    var failures = 0

    var tok = Qwen3Tokenizer(String(PROC) + "/tokenizer.json")
    var before = len(tok.special_tokens)
    var added = tok.merge_additional_special_tokens(
        String(PROC) + "/tokenizer_config.json"
    )
    print("  special tokens:", before, "from tokenizer.json +", added, "from config")

    print("")
    print("[1] the merge adds exactly the seven H3 tokens")
    checks += 1
    if added == 7 and len(tok.special_tokens) == before + 7:
        print("  ok   7 added, none of them already known")
    else:
        failures += 1
        print("  FAIL added", added, "total", len(tok.special_tokens))

    print("")
    print("[2] they land at 151669..151675, in config order")
    # Measured against Qwen2TokenizerFast on H3's own processor/ directory.
    var want_names = [
        String("<d>"), String("</d>"), String("<|cutoff|>"),
        String("<|lyrics_start|>"), String("<|lyrics_end|>"),
        String("<|caption_start|>"), String("<|caption_end|>"),
    ]
    checks += 1
    var bad = 0
    for i in range(7):
        var want_id = 151669 + i
        var found = -1
        for j in range(len(tok.special_tokens)):
            if tok.special_tokens[j] == want_names[i]:
                found = tok.special_ids[j]
                break
        if found != want_id:
            bad += 1
            print("  FAIL", want_names[i], "id", found, "want", want_id)
    if bad == 0:
        print("  ok   <d>=151669 ... <|caption_end|>=151675")
    else:
        failures += 1

    print("")
    print("[3] idempotent — merging twice adds nothing")
    checks += 1
    var again = tok.merge_additional_special_tokens(
        String(PROC) + "/tokenizer_config.json"
    )
    if again == 0:
        print("  ok   second merge added 0")
    else:
        failures += 1
        print("  FAIL second merge added", again)

    print("")
    print("[4] encode vs transformers, exact ids")
    var lines = Path(String(CASES)).read_text().split("\n")
    var cases = 0
    var mismatched = 0
    for li in range(len(lines)):
        ref line = lines[li]
        if line.byte_length() == 0:
            continue
        var parts = line.split("\t")
        if len(parts) < 1:
            continue
        var text = String(parts[0])
        var want = List[Int]()
        if len(parts) >= 2:
            var idstr = String(parts[1])
            if idstr.byte_length() > 0:
                var pieces = idstr.split(",")
                for k in range(len(pieces)):
                    want.append(Int(String(pieces[k])))
        var got = tok.encode(text)
        cases += 1
        var same = len(got) == len(want)
        if same:
            for k in range(len(got)):
                if got[k] != want[k]:
                    same = False
                    break
        if not same:
            mismatched += 1
            if mismatched <= 4:
                print("  FAIL", text)
                var gs = String("")
                for k in range(len(got)):
                    gs += String(got[k]) + " "
                var ws = String("")
                for k in range(len(want)):
                    ws += String(want[k]) + " "
                print("       got ", gs)
                print("       want", ws)
    checks += 1
    if mismatched == 0:
        print("  ok  ", cases, "cases, every id identical to transformers")
    else:
        failures += 1
        print("  FAIL", mismatched, "of", cases, "cases differ")

    print("")
    if failures == 0:
        print("PASS:", checks, "checks")
    else:
        print("FAIL:", failures, "of", checks, "checks")
        raise Error("MiniMax-H3 tokenizer parity gate failed")
