# Smoke / parity harness for the pure-Mojo T5 Unigram tokenizer.
#
# Encodes the SAME fixed strings as parity/t5_ref.py and prints them in an
# identical, diffable `text -> [ids]` format. CPU-only, no DeviceContext.
#
# Build (compile only; the GPU may be busy -- this binary is CPU-only):
#   cd /home/alex/mojodiffusion && /home/alex/.pixi/bin/pixi run mojo build \
#       -I . serenitymojo/tokenizer/t5_tokenizer_smoke.mojo -o /tmp/t5_tok_smoke
#
# Verify ids match HF (later, CPU-only):
#   /tmp/t5_tok_smoke
#   python3 serenitymojo/tokenizer/parity/t5_ref.py
# and diff the two `-> [...]` id lists line by line.

from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer

comptime TOK_JSON: StaticString = "/home/alex/.cache/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1/tokenizer.json"


def _show(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        if i != 0:
            s += String(", ")
        s += String(ids[i])
    s += String("]")
    return s^


def _enc(tok: T5Tokenizer, text: String) raises:
    # Print ONLY the id list, one per line, matching parity/t5_ref.py exactly
    # (Python list str form `[a, b, c]`) so a raw `diff` is meaningful.
    print(_show(tok.encode(text)))


def main() raises:
    var tok = T5Tokenizer.load(String(TOK_JSON))
    # Same 8 strings as parity/t5_ref.py, in the same order.
    _enc(tok, String("a photo of a cat"))
    _enc(tok, String("don't"))
    _enc(tok, String("3 cats"))
    _enc(tok, String("HELLO World"))
    _enc(tok, String("café déjà vu — naïve"))
    _enc(tok, String(""))
    _enc(tok, String("a highly detailed photograph of a majestic lion standing on a rocky cliff at sunset, golden hour lighting, 8k"))
    _enc(tok, String("  leading and trailing  "))
