# SKEPTIC runner: encode a TSV of (escaped_text<TAB>oracle_ids) with the Mojo
# tokenizer and diff against the fresh HF oracle ids. Prints EVERY mismatch with
# both sequences. The TSV path may be passed as argv[1]; defaults to
# skeptic_cases.tsv. The other skeptic_*.tsv files (regex/ws/scripts) are run by
# passing their absolute path.
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#          serenitymojo/tokenizer/parity/skeptic_run.mojo \
#          [serenitymojo/tokenizer/parity/skeptic_<set>.tsv]

import sys
from std.pathlib import Path
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

comptime TOK_JSON: StaticString = "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json"
comptime DEFAULT_CASES: StaticString = "/home/alex/mojodiffusion/serenitymojo/tokenizer/parity/skeptic_cases.tsv"


def _unescape(s: String) -> String:
    var out = List[Byte]()
    var b = s.as_bytes()
    var n = len(b)
    var i = 0
    while i < n:
        var c = Int(b[i])
        if c == 0x5C and i + 1 < n:
            var e = Int(b[i + 1])
            if e == 0x5C:
                out.append(Byte(0x5C)); i += 2; continue
            elif e == ord("t"):
                out.append(Byte(0x09)); i += 2; continue
            elif e == ord("n"):
                out.append(Byte(0x0A)); i += 2; continue
            elif e == ord("r"):
                out.append(Byte(0x0D)); i += 2; continue
        out.append(b[i])
        i += 1
    return String(unsafe_from_utf8=out)


def _parse_ids(s: String) -> List[Int]:
    var out = List[Int]()
    var b = s.as_bytes()
    var cur = 0
    var have = False
    for i in range(len(b)):
        var c = Int(b[i])
        if c >= 48 and c <= 57:
            cur = cur * 10 + (c - 48)
            have = True
        elif c == 0x2C:
            if have:
                out.append(cur)
            cur = 0; have = False
    if have:
        out.append(cur)
    return out^


def _eq(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _show(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        if i != 0:
            s += String(",")
        s += String(ids[i])
    s += String("]")
    return s^


def main() raises:
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var args = sys.argv()
    var cases_path = String(DEFAULT_CASES)
    if len(args) > 1:
        cases_path = String(args[1])
    print("cases:", cases_path)
    var content = Path(cases_path).read_text()

    var passed = 0
    var failed = 0

    var lines = content.split("\n")
    for li in range(len(lines)):
        ref line = lines[li]
        if len(line) == 0:
            continue
        var parts = line.split("\t")
        if len(parts) < 2:
            # an empty-text case: text="" -> parts[0] is empty, split yields 1 part
            # handle empty text explicitly (line begins with tab)
            if len(line) > 0 and line.as_bytes()[0] == 0x09:
                var exp0 = _parse_ids(String(line))
                var got0 = tok.encode(String(""))
                if _eq(got0, exp0):
                    passed += 1
                else:
                    failed += 1
                    print("FAIL: text=<EMPTY>")
                    print("   got   =", _show(got0))
                    print("   oracle=", _show(exp0))
            continue
        var text = _unescape(String(parts[0]))
        var expected = _parse_ids(String(parts[1]))
        var got = tok.encode(text)
        if _eq(got, expected):
            passed += 1
        else:
            failed += 1
            print("FAIL: text=", parts[0])
            print("   got   =", _show(got))
            print("   oracle=", _show(expected))

    print("\n=== SKEPTIC RESULT:", passed, "passed,", failed, "failed (of", passed + failed, ") ===")
