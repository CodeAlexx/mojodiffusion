# Parity gate for the ERNIE/Mistral-3B tokenizer path.
#
# Oracle ids are from Python `tokenizers.Tokenizer.from_file(...).encode(...,
# add_special_tokens=True)` on /home/alex/models/ERNIE-Image/tokenizer/tokenizer.json.
# The shared Mojo BPE tokenizer returns model tokens only; `mistral3_tokenize`
# prepends the ERNIE TemplateProcessing BOS token id 1.
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run mojo run -I . serenitymojo/models/text_encoder/parity/mistral3_tokenizer_parity.mojo

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.mistral3b_encoder import mistral3_tokenize


comptime TOK_JSON: StaticString = "/home/alex/models/ERNIE-Image/tokenizer/tokenizer.json"


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
    var got = mistral3_tokenize(tok, text)
    if _ids_eq(got, expected):
        print("  PASS  ", name)
        passed += 1
    else:
        print("  FAIL  ", name)
        print("        got     ", _show(got))
        print("        oracle  ", _show(expected))
        failed += 1


def main() raises:
    print("Loading ERNIE tokenizer.json ...")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    print("Loaded. Running ERNIE/Mistral tokenizer parity gate.\n")

    var passed = 0
    var failed = 0
    _check(tok, "empty prompt", "", [1], passed, failed)
    _check(
        tok,
        "boxjana studio",
        "box1jana, a high-resolution studio portrait, ornate chair, cocktail, studio lighting",
        [1, 4873, 1049, 39404, 1044, 1261, 2738, 60549, 15541, 40648, 1044, 34634, 1419, 13971, 1044, 79014, 1044, 15541, 34890],
        passed,
        failed,
    )
    _check(
        tok,
        "boxjana beach",
        "box1jana standing on a beach at sunset, ocean waves, warm cinematic light",
        [1, 4873, 1049, 39404, 15866, 1408, 1261, 29397, 1513, 97558, 1044, 27208, 22140, 1044, 15701, 6576, 77663, 4391],
        passed,
        failed,
    )

    print("\n==================================================")
    print("MISTRAL TOKENIZER GATE: ", passed, " passed, ", failed, " failed")
    print("==================================================")
    if failed != 0:
        raise Error("mistral3 tokenizer parity failed")
