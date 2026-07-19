#!/usr/bin/env python3
"""Ground-truth CLIP tokenizer ids for parity-checking the pure-Mojo port.

Uses transformers.CLIPTokenizer (the SLOW tokenizer that SDXL / SD3.5 / FLUX
pipelines use) and prints `text -> ids` in the SAME format as
clip_tokenizer_smoke.mojo, so the two can be diffed line-by-line.

The Mojo encode() returns [BOS, ..., EOS] with NO padding/truncation. To match,
this script encodes WITHOUT padding and WITHOUT truncation (full id list incl.
the BOS=49406 / EOS=49407 wrappers).

Run (CPU-only, no GPU needed):
    python3 serenitymojo/tokenizer/parity/clip_ref.py
"""

from transformers import CLIPTokenizer

TEXTS = [
    "a photo of a cat",
    "Hello, world! It's a test.",
    "don't stop",
    "3 cats and 42 dogs",
    "HELLO World",
    "a cat \U0001F408 sitting",
    "",
    (
        "a highly detailed photorealistic portrait of an astronaut riding a "
        "horse on the surface of mars during a golden sunset with dramatic "
        "cinematic lighting and intricate background scenery extending far "
        "beyond seventy seven tokens to test truncation behavior carefully"
    ),
]


def main():
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    for text in TEXTS:
        ids = tok.encode(text, add_special_tokens=True, padding=False, truncation=False)
        ids_str = ", ".join(str(i) for i in ids)
        print(f'"{text}" -> [{ids_str}]')


if __name__ == "__main__":
    main()
