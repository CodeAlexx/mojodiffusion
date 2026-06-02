#!/usr/bin/env python3
"""Generate a larger fuzz set of (text, oracle_ids) pairs for hardening the
Mojo pre-tokenizer beyond the hand-picked smoke cases.

Writes JSONL to stdout: one {"text":..., "ids":[...]} per line. The Mojo fuzz
runner (fuzz_cases.txt) consumes a flattened form. This script also writes a
tab-separated cases file the Mojo runner can parse without a JSON lib.

Usage:
    pixi run python3 serenitymojo/tokenizer/parity/fuzz_oracle.py > /dev/null
    # writes serenitymojo/tokenizer/parity/fuzz_cases.tsv
"""
import json
import os

DEFAULT_JSON = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json"
)

CASES = [
    # ascii words / spacing variety
    "the cat sat on the mat",
    "  the cat  sat   on    the     mat  ",
    "\tindented\twith\ttabs",
    "trailing tab\t",
    "MiXeD CaSe AnD camelCase words",
    "snake_case_identifier and kebab-case-thing",
    "a b c d e f g",
    "x  y   z",
    "word\n\n\nword",
    "line1\nline2\r\nline3",
    "   \n   \n   ",
    "...!!!???",
    "(parens) [brackets] {braces}",
    "email@example.com, http://url.test/path?q=1&b=2",
    "1234567890",
    "3.14159 and 2.71828",
    "version 1.2.3-rc4",
    "phone: +1 (555) 123-4567",
    "Don't can't won't shouldn't they're we've I'll he'd",
    "ABC's DEF'RE GHI'VE",
    "'leading apostrophe",
    "quote \"inside\" text",
    "back\\slash here",
    "tab\tand\nnewline mix",
    "100% sure & done!",
    "C++ and C# and F#",
    "<html><body>hi</body></html>",
    "a1 b2 c3 d4",
    "mixing123letters456and789digits",
    "UPPER lower 123 !@#",
    "café résumé naïve façade",
    "Zürich Köln Düsseldorf",
    "El Niño jalapeño piñata",
    "Ω α β γ δ Greek letters",
    "Привет мир Cyrillic",
    "日本語 のテキスト",
    "中文 文本 测试",
    "한국어 텍스트",
    "I love 🚀 and 🎉 emojis 😀",
    "mixed café 🚀 123 text!",
    "  ",
    " ",
    "",
    "a",
    "9",
    ".",
    "'",
    "'s",
    " 's",
    "a's",
    "\n",
    "\n\n",
    " \n ",
    "  hello",
    "hello  ",
    "  hello  world  ",
    "tab\tword space word",
    "punctuation,separated,by,commas",
    "semicolons;and:colons",
    "a very long sentence with many ordinary english words to exercise the bpe merges across a realistic prompt length similar to what a diffusion text encoder would actually receive at inference time",
    "The quick brown fox jumps over the lazy dog. 1234567890 !@#$%^&*()",
    "Numbers like 42 and 007 and 3000 and 1000000",
    "She said, \"It's a beautiful day, isn't it?\"",
    "function foo(x) { return x + 1; }",
    "import numpy as np\nimport torch",
    "git commit -m 'fix: resolve bug #123'",
    "https://example.com/path/to/page#section",
    "user@host:~/dir$ ls -la",
    "key=value; other=thing",
    "[2026-05-25] INFO: started",
    "emoji at end 🚀",
    "🚀 emoji at start",
    "🚀🎉😀 three emojis",
    "tab\ttab\ttabs",
    "newline\nthen tab\there",
    "multiple   internal    spaces",
    "ALLCAPS sentence here",
    "alllowercase sentence here",
    "Sentence. Another. Third.",
    "What? Why! How...",
    "1st 2nd 3rd 4th",
    "co-op re-do pre-set",
    "$100.50 and €200 and £300",
    "50% off! Buy 2 get 1 free.",
    "a_b_c d-e-f g.h.i",
    "CamelCaseWords AndMore",
    "trailing newline\n",
    "leading\nnewline",
]


def main() -> None:
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(DEFAULT_JSON)
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "fuzz_cases.tsv")
    # Encode each case; serialize text as a backslash-escaped single line so the
    # Mojo runner can recover the exact bytes (escape \n, \t, \r, \\).
    def esc(s: str) -> str:
        return (
            s.replace("\\", "\\\\")
            .replace("\t", "\\t")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for s in CASES:
            ids = tk.encode(s).ids
            f.write(esc(s) + "\t" + ",".join(str(i) for i in ids) + "\n")
    print(f"wrote {len(CASES)} cases to {out_path}")


if __name__ == "__main__":
    main()
