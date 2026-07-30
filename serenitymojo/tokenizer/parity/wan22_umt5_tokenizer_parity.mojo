# Wan2.2 UMT5 tokenizer parity against the pinned creator tokenizer.
#
# Oracle:
#   /home/alex/Wan2.2/wan/modules/tokenizers.py
#   creator commit 42bf4cfaa384bc21833865abc2f9e6c0e67233dc
#
# The 256k UMT5 vocabulary is not covered by the standard t5-base smoke test.
# This gate uses the exact prompt that exposed the Wan T2V quality regression.

from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer
from serenitymojo.tokenizer.wan_prompt_clean import wan_creator_clean


comptime TOK_JSON = (
    "/home/alex/.serenity/models/checkpoints/"
    "Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json"
)
comptime PROMPT = (
    "A crisp editorial photograph of a red ceramic teapot on a pale blue table, "
    "soft window light, realistic fine surface detail, balanced composition"
)
comptime NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _expected_prompt_ids() -> List[Int]:
    return [
        320, 154215, 62550, 273, 152496, 329, 289, 4062, 86477, 26845,
        9190, 369, 289, 46096, 15258, 8338, 275, 15147, 14172, 8403, 275,
        124690, 4338, 15925, 14777, 275, 191587, 55954, 1,
    ]


def _expected_negative_ids() -> List[Int]:
    return [
        273, 1838, 8483, 181911, 30993, 275, 4915, 85615, 275, 15665,
        13607, 275, 17121, 8024, 7921, 162805, 1013, 3815, 275, 5549,
        16096, 275, 6865, 3967, 275, 9103, 275, 7084, 2354, 275, 16927,
        275, 15665, 10671, 275, 44776, 2390, 46846, 275, 2740, 5731,
        20499, 275, 3846, 20499, 275, 958, 58395, 11376, 52222, 7019,
        7383, 275, 217390, 245822, 431, 275, 7019, 28644, 431, 275,
        1589, 8136, 431, 1320, 4050, 275, 7084, 2351, 1013, 23261,
        1320, 1542, 275, 7084, 2351, 1013, 23261, 67032, 1542, 275,
        245886, 2584, 431, 275, 160335, 10394, 431, 275, 2584, 13607,
        245886, 2584, 431, 69797, 1906, 275, 1320, 4050, 14116, 1539,
        275, 15665, 10671, 1013, 2312, 431, 16927, 275, 40671, 20559,
        431, 35216, 275, 1687, 4459, 136320, 275, 35216, 798, 19255,
        275, 11730, 1801, 4898, 1,
    ]


def _show(ids: List[Int]) -> String:
    var out = String("[")
    for i in range(len(ids)):
        if i != 0:
            out += String(", ")
        out += String(ids[i])
    out += String("]")
    return out^


def _check_equal(got: List[Int], expected: List[Int]) raises:
    if len(got) != len(expected):
        print("got     ", _show(got))
        print("expected", _show(expected))
        raise Error(
            String("Wan UMT5 token count mismatch: got ")
            + String(len(got)) + String(", expected ") + String(len(expected))
        )
    for i in range(len(expected)):
        if got[i] != expected[i]:
            print("got     ", _show(got))
            print("expected", _show(expected))
            raise Error(
                String("Wan UMT5 token mismatch at index ") + String(i)
                + String(": got ") + String(got[i])
                + String(", expected ") + String(expected[i])
            )


def main() raises:
    var tokenizer = T5Tokenizer.load(String(TOK_JSON))
    var got = tokenizer.encode(wan_creator_clean(String(PROMPT)))
    var expected = _expected_prompt_ids()
    _check_equal(got, expected)
    var negative_got = tokenizer.encode(wan_creator_clean(String(NEGATIVE)))
    var negative_expected = _expected_negative_ids()
    _check_equal(negative_got, negative_expected)
    print(
        "GATE PASS Wan2.2 UMT5 tokenizer matches creator ids: pos=",
        len(got), "neg=", len(negative_got),
    )
