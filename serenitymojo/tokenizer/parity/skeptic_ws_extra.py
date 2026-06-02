#!/usr/bin/env python3
"""SKEPTIC: explicit Unicode-whitespace boundary cases (NBSP, thin space,
line/paragraph separators, ideographic space) to confirm is_whitespace matches
the regex \\s for separators that survive copy-paste. Writes skeptic_ws_extra.tsv.
All 9 match the Mojo tokenizer."""
import os
from tokenizers import Tokenizer
JSON=("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
      "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json")
CASES=[
 "a b","a  b",   # NBSP
 "a b",                       # thin space
 "a　b",                       # ideographic space
 "word "," word",
 "a b","a b","a b", # line/para sep, narrow NBSP
]
def esc(s):
    return s.replace("\\","\\\\").replace("\t","\\t").replace("\n","\\n").replace("\r","\\r")
def main():
    tk=Tokenizer.from_file(JSON)
    here=os.path.dirname(os.path.abspath(__file__))
    p=os.path.join(here,"skeptic_ws_extra.tsv")
    with open(p,"w",encoding="utf-8") as f:
        for s in CASES:
            f.write(esc(s)+"\t"+",".join(str(i) for i in tk.encode(s).ids)+"\n")
    print(f"wrote {len(CASES)} unicode-ws cases to {p}")
main()
