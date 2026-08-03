#!/usr/bin/env python3
"""Oracle for the MiniMax-H3 tokenizer, including the seven config-only tokens.

MiniMax-H3's README: "We add several special tokens, such as `<d>`, to the
tokenizer configuration. When using H3, the tokenizer and associated
configuration files provided in the H3 repository are required."

They are not kidding, and the trap is sharper than it reads. MEASURED on H3's
own `processor/` directory:

    tokenizer.json      151643 vocab + 26 added_tokens  -> 151669
    the real tokenizer                                     151676

The seven extra tokens are in NEITHER the vocab NOR `added_tokens`. They exist
only as `additional_special_tokens` in `tokenizer_config.json`, and
`transformers` appends them at load. A port that reads `tokenizer.json` alone —
which ours did until this file existed — tokenizes `<d>` as `<`, `d`, `>`.
Three ids instead of one, a different text embedding, a different video.

This dumps the reference ids for prompts that USE those tokens, so the Mojo
side is gated on the case that actually breaks rather than on plain prose that
would pass either way.

Writes: output/minimax_h3_tokenizer/tokenizer_ref.txt   (tab-separated)
"""

import os
import sys

from transformers import AutoTokenizer

H3 = "/home/alex/minimax_h3_ref/creator-MiniMax-H3/FL2VA/processor"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_tokenizer"

CASES = [
    # the seven config-only tokens, alone — each MUST be a single id
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
    # embedded in real prompt text, which is how they will actually arrive
    "<d>A neon-lit alley in the rain.</d>",
    "<|caption_start|>a cat on a piano<|caption_end|>",
    "<|lyrics_start|>we are the champions<|lyrics_end|>",
    "A wide shot <|cutoff|> then a close-up.",
    # adjacency with no whitespace — the greedy-match case
    "<d></d>",
    "<d><|cutoff|></d>",
    "text<d>more",
    # tokens that WERE already in tokenizer.json, to prove nothing regressed
    "<|im_start|>hello<|im_end|>",
    "<|vision_start|><|image_pad|><|vision_end|>",
    # plain prose, the control
    "A cinematic shot of a fox in a clay-animation style, warm light.",
    "",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(H3)
    print(f"{type(tok).__name__}  vocab_size {tok.vocab_size}  len {len(tok)}")
    if len(tok) != 151676:
        print(f"  WARNING: expected 151676, got {len(tok)} — the trap may have moved")

    lines = []
    for text in CASES:
        ids = tok.encode(text, add_special_tokens=False)
        lines.append(text.replace("\t", " ") + "\t" + ",".join(str(i) for i in ids))
        shown = text if len(text) <= 42 else text[:39] + "..."
        print(f"  {shown!r:46s} -> {ids}")

    path = f"{OUT_DIR}/tokenizer_ref.txt"
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print()
    print(f"wrote {path} ({len(CASES)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
