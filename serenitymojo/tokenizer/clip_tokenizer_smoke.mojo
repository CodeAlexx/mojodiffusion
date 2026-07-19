# CPU-only smoke / parity harness for the pure-Mojo CLIP tokenizer.
#
# Encodes a fixed set of strings with ClipTokenizer (CLIP-L vocab) and prints
# `text -> ids` in a diffable format. Pair with parity/clip_ref.py (HF ground
# truth). NO DeviceContext / GPU -- pure CPU string processing.
#
# Build (compile only, do NOT run while GPU is busy -- it is CPU-only anyway):
#   pixi run mojo build -I . serenitymojo/tokenizer/clip_tokenizer_smoke.mojo \
#       -o /tmp/clip_tok_smoke

from serenitymojo.tokenizer.clip_tokenizer import ClipTokenizer, load


def _print_row(tok: ClipTokenizer, text: String) raises:
    var ids = tok.encode(text)
    var line = String('"') + text + String('" -> [')
    for i in range(len(ids)):
        if i != 0:
            line += String(", ")
        line += String(ids[i])
    line += String("]")
    print(line)


def main() raises:
    # Resolve via the same snapshot the parity script uses.
    var json_path = String(
        "/home/alex/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/"
        "snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/tokenizer.json"
    )
    var tok = load(json_path)

    _print_row(tok, String("a photo of a cat"))
    _print_row(tok, String("Hello, world! It's a test."))
    _print_row(tok, String("don't stop"))
    _print_row(tok, String("3 cats and 42 dogs"))
    _print_row(tok, String("HELLO World"))
    _print_row(tok, String("a cat \U0001F408 sitting"))
    _print_row(tok, String(""))
    _print_row(
        tok,
        String(
            "a highly detailed photorealistic portrait of an astronaut riding a "
            "horse on the surface of mars during a golden sunset with dramatic "
            "cinematic lighting and intricate background scenery extending far "
            "beyond seventy seven tokens to test truncation behavior carefully"
        ),
    )
