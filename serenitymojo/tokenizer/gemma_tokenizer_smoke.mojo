# Pure-Mojo Gemma tokenizer parity probe for the LTX-2 conditioning path.
#
# Usage:
#   mojo run -I . serenitymojo/tokenizer/gemma_tokenizer_smoke.mojo \
#     /path/to/tokenizer.json "a lighthouse at sunset"

from std.sys import argv

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: gemma_tokenizer_smoke TOKENIZER_JSON PROMPT")
    var tokenizer = Qwen3Tokenizer(String(args[1]))
    var ids = tokenizer.encode_gemma(String(args[2]))
    for i in range(len(ids)):
        if i > 0:
            print(",", end="")
        print(ids[i], end="")
    print()
