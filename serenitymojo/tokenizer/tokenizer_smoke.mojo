# Smoke / parity gate for the pure-Mojo Qwen3 byte-level BPE tokenizer.
#
# Encodes each sample prompt with the Mojo tokenizer and compares the token-id
# sequence to the oracle (HF `tokenizers` crate, captured offline by
# parity/oracle.py). GATE: exact id-sequence match on ASCII/space/punct/digit.
#
# Run:  cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#           serenitymojo/tokenizer/tokenizer_smoke.mojo

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

comptime TOK_JSON: StaticString = "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json"


def _ids_eq(a: List[Int], b: List[Int]) -> Bool:
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
            s += String(", ")
        s += String(ids[i])
    s += String("]")
    return s^


def _check(
    tok: Qwen3Tokenizer,
    name: String,
    text: String,
    expected: List[Int],
    mut passed: Int,
    mut failed: Int,
) raises:
    var got = tok.encode(text)
    if _ids_eq(got, expected):
        print("  PASS  ", name)
        passed += 1
    else:
        print("  FAIL  ", name)
        print("        got     ", _show(got))
        print("        oracle  ", _show(expected))
        failed += 1


def main() raises:
    print("Loading tokenizer.json (pure-Mojo parse) ...")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    print("Loaded. Running parity gate vs oracle ids.\n")

    var passed = 0
    var failed = 0

    # Oracle ids captured by parity/oracle.py (HF tokenizers 0.22.2).
    # --- GATE cases: ASCII / space / punct / digit / mixed case ---
    _check(tok, "plain ASCII          ", String("hello world"), [14990, 1879], passed, failed)
    _check(tok, "mixed case + space   ", String("Hello World"), [9707, 4337], passed, failed)
    _check(tok, "leading spaces       ", String("  leading spaces"), [220, 6388, 12621], passed, failed)
    _check(tok, "trailing spaces      ", String("trailing spaces  "), [376, 14277, 12621, 256], passed, failed)
    _check(tok, "ascii+digits+punct   ", String("The quick brown fox jumps over 13 lazy dogs."), [785, 3974, 13876, 38835, 34208, 916, 220, 16, 18, 15678, 12590, 13], passed, failed)
    _check(tok, "letter/digit boundary", String("a1b2c3"), [64, 16, 65, 17, 66, 18], passed, failed)
    _check(tok, "contractions+punct   ", String("Don't stop, won't you?"), [8002, 944, 2936, 11, 2765, 944, 498, 30], passed, failed)
    _check(tok, "mixed case words     ", String("MixedCASE Words"), [86433, 40371, 27630], passed, failed)
    _check(tok, "digits + punct       ", String("123 456.78"), [16, 17, 18, 220, 19, 20, 21, 13, 22, 23], passed, failed)
    _check(tok, "embedded newline     ", String("newline\ntest"), [89202, 198, 1944], passed, failed)
    _check(tok, "special token        ", String("<|im_start|>user"), [151644, 872], passed, failed)

    # --- non-ASCII cases (may expose \p approximation; reported honestly) ---
    _check(tok, "cafe (precomposed)   ", String("café"), [924, 58858], passed, failed)
    _check(tok, "emoji (4-byte UTF-8) ", String("I love 🚀 rockets"), [40, 2948, 11162, 248, 222, 51998], passed, failed)

    print("\n==================================================")
    print("GATE RESULT: ", passed, " passed, ", failed, " failed (of ", passed + failed, ")")
    print("==================================================")

    # --- KNOWN-APPROXIMATION cases (NON-GATING, reported honestly) ---
    # These exercise the documented gaps: NFC normalization (no-op) and the
    # \p{N} ASCII-only digit approximation. Divergence here is EXPECTED.
    print("\nKnown-approximation cases (non-gating, expected to differ):")
    var decomposed = String("cafe") + String(chr(0x0301))  # e + combining acute
    var d_got = tok.encode(decomposed)
    print("  NFC decomposed 'cafe\\u0301': mojo=", _show(d_got), " oracle=[924, 58858]")
    print("    -> differs unless NFC composition is added (no-op NFC today).")
    var arabic_digits = String(chr(0x0664)) + String(chr(0x0665)) + String(chr(0x0666))
    var a_got = tok.encode(arabic_digits)
    print("  Arabic-Indic digits U+0664-0666: mojo=", _show(a_got), " oracle=[149,97,149,98,149,99]")
    print("    -> differs: \\p{N} approximated as ASCII 0-9 only.")

    # decode round-trip sanity
    print("\nDecode round-trip checks:")
    var d1 = tok.decode([14990, 1879])
    print("  [14990,1879] -> '", d1, "'")
    var d2 = tok.decode([924, 58858])
    print("  [924,58858]  -> '", d2, "'")
    var d3 = tok.decode([151644, 872])
    print("  [151644,872] -> '", d3, "'")
