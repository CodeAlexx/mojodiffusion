# SenseNova-U1 tokenizer parity smoke.
#
# SenseNova ships tokenizer assets split across vocab.json, merges.txt, and
# added_tokens.json rather than one tokenizer.json. This gate checks the Mojo
# split-file loader against HF AutoTokenizer oracle IDs for ordinary BPE text
# plus the chat and vision specials used by the T2I path.

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime VOCAB_JSON: StaticString = "/home/alex/.serenity/models/sensenova_u1/vocab.json"
comptime MERGES_TXT: StaticString = "/home/alex/.serenity/models/sensenova_u1/merges.txt"
comptime ADDED_TOKENS_JSON: StaticString = "/home/alex/.serenity/models/sensenova_u1/added_tokens.json"


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
    print("Loading SenseNova split tokenizer assets ...")
    var tok = Qwen3Tokenizer(
        String(VOCAB_JSON), String(MERGES_TXT), String(ADDED_TOKENS_JSON)
    )
    var passed = 0
    var failed = 0

    _check(tok, String("plain ASCII"), String("hello world"), [14990, 1879], passed, failed)
    _check(tok, String("t2i smoke prompt"), String("a photo of a cat"), [64, 6548, 315, 264, 8251], passed, failed)
    _check(
        tok,
        String("im chat special"),
        String("<|im_start|>user\nhello<|im_end|>"),
        [151644, 872, 198, 14990, 151645],
        passed,
        failed,
    )
    _check(
        tok,
        String("vision specials"),
        String("<|vision_start|><|image_pad|><|vision_end|>"),
        [151652, 151655, 151653],
        passed,
        failed,
    )

    print("SenseNova tokenizer gate:", passed, "passed,", failed, "failed")
    if failed != 0:
        raise Error("SenseNova tokenizer parity failed")
