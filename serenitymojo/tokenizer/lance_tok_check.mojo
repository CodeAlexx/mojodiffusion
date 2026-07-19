# Lance tokenizer parity smoke.
#
# Lance ships a Qwen2 tokenizer.json. This gate checks the prompt and multimodal
# specials used by the Lance T2V smoke against HF Qwen2TokenizerFast oracle IDs.

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime TOK_JSON: StaticString = "/home/alex/.serenity/models/lance/Lance_3B_Video/tokenizer.json"


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
        print("  PASS  ", name, " len=", len(got))
        passed += 1
    else:
        print("  FAIL  ", name, " len=", len(got))
        print("        got     ", _show(got))
        print("        oracle  ", _show(expected))
        failed += 1


def main() raises:
    print("Loading Lance Qwen2 tokenizer.json ...")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var passed = 0
    var failed = 0

    _check(tok, String("fairy"), String("fairy"), [69, 21939], passed, failed)
    _check(tok, String("a fairy"), String("a fairy"), [64, 44486], passed, failed)
    _check(
        tok,
        String("vision video specials"),
        String("<|vision_start|><|video_pad|><|vision_end|>"),
        [151652, 151656, 151653],
        passed,
        failed,
    )
    _check(
        tok,
        String("im chat prompt"),
        String("<|im_start|>user\nfairy<|im_end|>"),
        [151644, 872, 198, 69, 21939, 151645],
        passed,
        failed,
    )

    print("Lance tokenizer gate:", passed, "passed,", failed, "failed")
    if failed != 0:
        raise Error("Lance tokenizer parity failed")
