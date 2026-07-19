# Parity probe for the Boogu T2I chat-template tokenizer (Chunk C7b).
#
# Oracle ids are from boogu_c7_oracle.py:
#   processor.apply_chat_template([{system: SYSTEM_PROMPT_4_T2I}, {user: instr}],
#                                 tokenize=True)
# on the Qwen3-VL tokenizer that ships in the Boogu mllm dir, for the
# instruction "A photorealistic portrait of an astronaut riding a horse on Mars."
#
# Run:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/text_encoder/parity/boogu_c7b_tok_probe.mojo

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.boogu_tokenizer import (
    boogu_tokenize,
    boogu_chat_template,
)


comptime TOK_JSON: StaticString = (
    "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/mllm/tokenizer.json"
)

comptime INSTRUCTION: StaticString = (
    "A photorealistic portrait of an astronaut riding a horse on Mars."
)


def _show(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        if i != 0:
            s += String(", ")
        s += String(ids[i])
    s += String("]")
    return s^


def main() raises:
    # Expected ids (45) decoded from the Boogu T2I chat template.
    var expected = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 429, 26885, 1550,
        22092, 5335, 3118, 389, 1196, 11221, 13, 576, 11221, 525, 438, 11017,
        13, 151645, 198, 151644, 872, 198, 32, 4503, 89768, 4532, 33033, 315,
        458, 46633, 19837, 264, 15223, 389, 21048, 13, 151645, 198,
    ]

    print("Loading Boogu Qwen3-VL tokenizer.json ...")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    print("Loaded.")

    print("Template string:")
    print(boogu_chat_template(String(INSTRUCTION)))

    var got = boogu_tokenize(tok, String(INSTRUCTION))
    print("produced ids (count=", len(got), "):")
    print(_show(got))
    print("expected ids (count=", len(expected), ")")

    if len(got) != len(expected):
        print(
            "LENGTH MISMATCH: produced ", len(got), " expected ",
            len(expected),
        )
        raise Error("boogu c7b tokenizer parity failed (length)")

    var first_mismatch = -1
    for i in range(len(expected)):
        if got[i] != expected[i]:
            first_mismatch = i
            break

    if first_mismatch < 0:
        print("MATCH")
    else:
        print(
            "MISMATCH at index ", first_mismatch, ": produced ",
            got[first_mismatch], " expected ", expected[first_mismatch],
        )
        raise Error("boogu c7b tokenizer parity failed (value)")
