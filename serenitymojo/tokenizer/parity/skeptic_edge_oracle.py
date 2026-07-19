#!/usr/bin/env python3
"""Fresh HF oracle for the exact edge cases in skeptic_edge.mojo."""
from tokenizers import Tokenizer
JSON = ("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
        "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json")
tk = Tokenizer.from_file(JSON)
cases = [
    ("empty",            ""),
    ("single space",     " "),
    ("single bang",      "!"),
    ("curly apostrophe", "it’s"),
    ("curly dquote",     "“hi”"),
    ("tis",              "'tis the season"),
    ("DON'T",            "DON'T"),
    ("yalldve",          "y'all'd've"),
    ("two specials adj", "two<|im_start|><|im_end|>adjacent"),
    ("partial special",  "partial <|im_ token here"),
    ("think tokens",     "use <think> reasoning </think> then answer"),
    ("backslash path",   "C:\\Users\\name"),
    ("crlf",             "windows\r\nline"),
    ("ZWJ family",       "\U0001F468‍\U0001F469‍\U0001F467"),
    ("regional flag US", "\U0001F1FA\U0001F1F8"),
    ("skin tone",        "\U0001F44D\U0001F3FD"),
    ("cafe precomposed", "café"),
    ("cafe decomposed",  "café"),
    ("u-umlaut precomp", "ü"),
    ("u-umlaut decomp",  "ü"),
    ("korean precomp",   "한국"),
    ("korean jamo",      "한국"),
]
for label, text in cases:
    print(f"{label:18} -> {tk.encode(text).ids}")
