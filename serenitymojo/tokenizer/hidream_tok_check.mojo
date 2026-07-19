# HiDream-O1 tokenizer parity smoke.
#
# HiDream's tokenizer.json serializes BPE merges as strings ("a b"), unlike the
# array-pair format used by the Z-Image tokenizer.json. This checks the chat
# templates that the HiDream smoke relies on against HF tokenizers oracle ids.

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime TOK_JSON: StaticString = "/home/alex/HiDream-O1-Image-Dev-weights/tokenizer.json"


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


def _template(prompt: String) -> String:
    var s = String("<|im_start|>user\n")
    s += prompt
    s += "<|im_end|>\n<|im_start|>assistant\n"
    s += "<|boi_token|><|tms_token|>"
    return s^


def main() raises:
    print("Loading HiDream-O1 tokenizer.json ...")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var passed = 0
    var failed = 0

    _check(
        tok,
        String("hidream prompt"),
        _template(String("a serene mountain lake at dawn")),
        [
            151644, 872, 198, 64, 94763, 16301, 21800, 518, 38393,
            151645, 198, 151644, 77091, 198, 151669, 151673,
        ],
        passed,
        failed,
    )
    _check(
        tok,
        String("hidream uncond"),
        _template(String(" ")),
        [
            151644, 872, 198, 220, 151645, 198, 151644, 77091, 198,
            151669, 151673,
        ],
        passed,
        failed,
    )
    _check(
        tok,
        String("hidream specials"),
        String("<|boi_token|><|tms_token|>"),
        [151669, 151673],
        passed,
        failed,
    )

    print("HiDream tokenizer gate:", passed, "passed,", failed, "failed")
    if failed != 0:
        raise Error("HiDream tokenizer parity failed")
