#!/usr/bin/env python3
"""OFFLINE parity oracle for verifying the pure-Mojo Qwen byte-level BPE
tokenizer against **Qwen-Image** (Qwen2.5-VL text path).

Qwen-Image's `tokenizer/` snapshot ships only tokenizer_config.json +
special_tokens_map.json (tokenizer_class = Qwen2Tokenizer); it does NOT bundle
tokenizer.json / vocab.json / merges.txt. The actual BPE files come from a
Qwen-Image-family sibling that DOES ship them. We verified (md5 + parsed-content
compare) that the vocab + merges + pre_tokenizer of:

    Qwen-Image-Edit-2511 (processor/tokenizer.json)   [Qwen-Image family]
    Qwen2.5-VL-7B-Instruct (tokenizer.json)           [upstream text encoder]
    Z-Image (tokenizer/tokenizer.json)                [Qwen3, our build target]

are all BYTE-IDENTICAL for vocab (151643), merges (151387), and the Split
pre-tokenizer regex. The ONLY difference is Z-Image (Qwen3) carries 4 extra
added special tokens (151665-151668: <tool_response> </tool_response> <think>
</think>) that Qwen2.5-VL / Qwen-Image do not have. None of those appear in
image-generation prompts, so the BPE encoding of real prompts is identical.

This script emits ground-truth HF ids for the Qwen-Image tokenizer so the Mojo
side can prove exact-match. It is a GATE reference only; never used at runtime.

Usage:
    pixi run python3 serenitymojo/tokenizer/parity/qwen_image_oracle.py [tokenizer.json]
"""
import json
import sys

# Primary Qwen-Image-family tokenizer (sibling of Qwen-Image-2512; same
# Qwen2.5-VL text encoder, ships the full BPE files).
QWEN_IMAGE_JSON = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/"
    "snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/processor/tokenizer.json"
)

# Prompts chosen to look like real Qwen-Image text-to-image captions, plus the
# same stress cases the Z-Image smoke uses (ASCII/space/punct/digit/case/emoji/
# special-token) so divergence vs Z-Image would surface here.
SAMPLES = [
    "a photo of a cat sitting on a red sofa",
    "A majestic mountain landscape at sunset, highly detailed, 8k",
    "portrait of a woman, soft lighting, bokeh background",
    "cyberpunk city street at night, neon signs, rain",
    "an oil painting of sunflowers in a blue vase",
    "close-up macro shot of a dewdrop on a green leaf",
    "1girl, white dress, standing in a field of 100 flowers",
    "The cat's whiskers, won't you look? 3 of them.",
    "café terrace with a 🚀 rocket mural",
    "text reads: \"Hello, World!\" in bold red letters",
    "hello world",
    "Hello World",
    "  leading spaces",
    "trailing spaces  ",
    "The quick brown fox jumps over 13 lazy dogs.",
    "a1b2c3",
    "Don't stop, won't you?",
    "MixedCASE Words",
    "123 456.78",
    "newline\ntest",
    "café",
    "I love \U0001F680 rockets",
    "<|im_start|>user",                 # special token present in BOTH families
    "<|vision_start|><|image_pad|><|vision_end|>",  # image-path specials
]


def main() -> None:
    from tokenizers import Tokenizer

    path = sys.argv[1] if len(sys.argv) > 1 else QWEN_IMAGE_JSON
    tk = Tokenizer.from_file(path)
    out = []
    for s in SAMPLES:
        ids = tk.encode(s).ids
        out.append({"text": s, "ids": ids})
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
