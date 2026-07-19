#!/usr/bin/env python3
"""OFFLINE parity oracle for the Mojo Qwen3 byte-level BPE tokenizer.

Loads the SAME tokenizer.json as the Mojo tokenizer (HF `tokenizers` crate
binding) and emits ground-truth token-id sequences for the smoke-test prompts.

Usage:
    pixi run python3 serenitymojo/tokenizer/parity/oracle.py [tokenizer.json]

This is the gate reference. It is NOT used at runtime — pure-Mojo runtime only.
"""
import json
import sys

DEFAULT_JSON = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json"
)

# Sample prompts MUST cover: plain ASCII, leading/trailing spaces, punctuation,
# digits, mixed case, and non-ASCII (café, emoji) to exercise byte-level + the
# \p{L}/\p{N} approximation.
SAMPLES = [
    "hello world",                                  # plain ASCII
    "Hello World",                                  # mixed case + space
    "  leading spaces",                             # leading spaces
    "trailing spaces  ",                            # trailing spaces
    "The quick brown fox jumps over 13 lazy dogs.", # ascii + digits + punct
    "a1b2c3",                                        # letter/digit boundaries
    "Don't stop, won't you?",                       # contractions + punct
    "MixedCASE Words",                              # mixed case
    "café",                                   # cafe + combining acute (NFC test)
    "café",                                    # café precomposed
    "I love \U0001F680 rockets",                    # emoji (4-byte UTF-8)
    "123 456.78",                                    # digits + punct
    "newline\ntest",                                # embedded newline
    "<|im_start|>user",                             # special token
]


def main() -> None:
    from tokenizers import Tokenizer

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON
    tk = Tokenizer.from_file(path)
    out = []
    for s in SAMPLES:
        ids = tk.encode(s).ids
        out.append({"text": s, "ids": ids})
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
