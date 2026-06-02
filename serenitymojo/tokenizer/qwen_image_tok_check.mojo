# Parity gate: does the pure-Mojo Qwen BPE tokenizer work for **Qwen-Image**?
#
# Qwen-Image (Qwen--Qwen-Image-2512) uses a Qwen2.5-VL text encoder with
# tokenizer_class = Qwen2Tokenizer. Its shipped tokenizer/ dir contains ONLY
# tokenizer_config.json + special_tokens_map.json (no tokenizer.json / vocab /
# merges). We verified offline that the BPE vocab (151643), merges (151387) and
# Split pre-tokenizer regex are BYTE-IDENTICAL to Z-Image's Qwen3 tokenizer; the
# only delta is Z-Image carries 4 extra added specials (151665-151668:
# <tool_response>/<think> family) absent from Qwen2.5-VL/Qwen-Image. Those never
# appear in image prompts, so encoding is identical.
#
# This loads a Qwen-Image-FAMILY tokenizer.json (Qwen-Image-Edit-2511, the
# sibling that ships the full BPE files) into our Qwen3Tokenizer and compares
# the encoded ids to the HF `tokenizers` oracle (parity/qwen_image_oracle.py).
#
# Run:  cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#           serenitymojo/tokenizer/qwen_image_tok_check.mojo

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

# Qwen-Image-family tokenizer.json (same Qwen2.5-VL text path as Qwen-Image-2512).
comptime TOK_JSON: StaticString = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/processor/tokenizer.json"


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
    print("Loading Qwen-Image-family tokenizer.json (pure-Mojo parse) ...")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    print("Loaded. Running Qwen-Image parity gate vs HF oracle ids.\n")

    var passed = 0
    var failed = 0

    # Oracle ids captured by parity/qwen_image_oracle.py (HF tokenizers 0.22.2)
    # on the Qwen-Image-Edit-2511 tokenizer.json. ASCII/space/punct/digit/special
    # cases GATE; café/emoji exercise the documented \p approximation.

    # --- Qwen-Image-style T2I caption prompts ---
    _check(tok, "caption: cat on sofa  ", String("a photo of a cat sitting on a red sofa"), [64, 6548, 315, 264, 8251, 11699, 389, 264, 2518, 31069], passed, failed)
    _check(tok, "caption: mountain 8k  ", String("A majestic mountain landscape at sunset, highly detailed, 8k"), [32, 80289, 16301, 18414, 518, 42984, 11, 7548, 11682, 11, 220, 23, 74], passed, failed)
    _check(tok, "caption: portrait     ", String("portrait of a woman, soft lighting, bokeh background"), [95641, 315, 264, 5220, 11, 8413, 17716, 11, 708, 70617, 4004], passed, failed)
    _check(tok, "caption: cyberpunk    ", String("cyberpunk city street at night, neon signs, rain"), [11130, 652, 75509, 3283, 8592, 518, 3729, 11, 46652, 11929, 11, 11174], passed, failed)
    _check(tok, "caption: oil painting ", String("an oil painting of sunflowers in a blue vase"), [276, 5590, 18824, 315, 7015, 88670, 304, 264, 6303, 92384], passed, failed)
    _check(tok, "caption: macro dewdrop", String("close-up macro shot of a dewdrop on a green leaf"), [5552, 5239, 18072, 6552, 315, 264, 66432, 6719, 389, 264, 6176, 15933], passed, failed)
    _check(tok, "caption: 1girl tags   ", String("1girl, white dress, standing in a field of 100 flowers"), [16, 28552, 11, 4158, 8511, 11, 11259, 304, 264, 2070, 315, 220, 16, 15, 15, 19281], passed, failed)
    _check(tok, "caption: contractions ", String("The cat's whiskers, won't you look? 3 of them."), [785, 8251, 594, 40659, 388, 11, 2765, 944, 498, 1401, 30, 220, 18, 315, 1105, 13], passed, failed)
    _check(tok, "caption: quoted text  ", String("text reads: \"Hello, World!\" in bold red letters"), [1318, 15804, 25, 330, 9707, 11, 4337, 8958, 304, 13939, 2518, 11931], passed, failed)

    # --- shared stress cases (identical to Z-Image smoke) ---
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

    # --- special tokens shared by Qwen-Image & Z-Image (image path) ---
    _check(tok, "special: im_start    ", String("<|im_start|>user"), [151644, 872], passed, failed)
    _check(tok, "special: vision pad  ", String("<|vision_start|><|image_pad|><|vision_end|>"), [151652, 151655, 151653], passed, failed)

    # --- non-ASCII (documented \p approximation; reported honestly) ---
    _check(tok, "cafe (precomposed)   ", String("café"), [924, 58858], passed, failed)
    _check(tok, "emoji (4-byte UTF-8) ", String("I love 🚀 rockets"), [40, 2948, 11162, 248, 222, 51998], passed, failed)
    _check(tok, "caption: café + emoji", String("café terrace with a 🚀 rocket mural"), [924, 58858, 51478, 448, 264, 11162, 248, 222, 24306, 73373], passed, failed)

    print("\n==================================================")
    print("QWEN-IMAGE GATE: ", passed, " passed, ", failed, " failed (of ", passed + failed, ")")
    print("==================================================")
