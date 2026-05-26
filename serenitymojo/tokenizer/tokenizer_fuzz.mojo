# Fuzz/hardening gate: reads parity/fuzz_cases.tsv (text<TAB>oracle_ids),
# unescapes the text, encodes with the Mojo tokenizer, and diffs against the
# oracle ids. Surfaces ANY pre-tokenizer / BPE divergence beyond the smoke set.
#
# Run:  cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#           serenitymojo/tokenizer/tokenizer_fuzz.mojo

from std.pathlib import Path
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

comptime TOK_JSON: StaticString = "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json"
comptime CASES: StaticString = "/home/alex/mojodiffusion/serenitymojo/tokenizer/parity/fuzz_cases.tsv"


def _unescape(s: String) -> String:
    # reverse of fuzz_oracle.esc: \\ -> \, \t -> tab, \n -> nl, \r -> cr
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
    var content = Path(String(CASES)).read_text()

    var passed = 0
    var failed = 0
    var first_fails = 0

    # split lines on \n
    var lines = content.split("\n")
    for li in range(len(lines)):
        ref line = lines[li]
        if len(line) == 0:
            continue
        # split on first tab
        var parts = line.split("\t")
        if len(parts) < 2:
            continue
        var text = _unescape(String(parts[0]))
        var expected = _parse_ids(String(parts[1]))
        var got = tok.encode(text)
        if _eq(got, expected):
            passed += 1
        else:
            failed += 1
            if first_fails < 25:
                print("FAIL: text=", parts[0])
                print("   got   =", _show(got))
                print("   oracle=", _show(expected))
                first_fails += 1

    print("\n=== FUZZ RESULT:", passed, "passed,", failed, "failed (of", passed + failed, ") ===")
