# Verify the pure-Mojo Qwen3Tokenizer reproduces the Ideogram-4 chat-templated
# token ids (the EncodeQwen3VLText data-prep input) vs the chunk7 torch oracle.
# Template (Qwen3-VL, add_generation_prompt, no thinking block):
#   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

comptime TOK = "/home/alex/.serenity/models/ideogram-4-fp8/tokenizer/tokenizer.json"


def main() raises:
    var tok = Qwen3Tokenizer(String(TOK))
    var rendered = String(
        "<|im_start|>user\na red cube on a white table<|im_end|>\n<|im_start|>assistant\n"
    )
    var ids = tok.encode(rendered)
    var expected = [
        151644, 872, 198, 64, 2518, 23739, 389, 264, 4158, 1965, 151645, 198,
        151644, 77091, 198,
    ]
    print("got     n =", len(ids))
    print("oracle  n =", len(expected))
    var ok = len(ids) == len(expected)
    if ok:
        for i in range(len(ids)):
            if ids[i] != expected[i]:
                ok = False
                break
    var s = String("got: ")
    for i in range(len(ids)):
        s += String(ids[i]) + " "
    print(s)
    if ok:
        print("ideogram4 tokenizer parity: PASS")
    else:
        print("ideogram4 tokenizer parity: FAIL")
