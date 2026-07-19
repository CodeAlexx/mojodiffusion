#!/usr/bin/env python3
"""SKEPTIC: pathological whitespace boundaries where the hand-rolled \\s* / \\s+
/ \\s+(?!\\S) / \\s*[\\r\\n]+ scanner is most likely to diverge from the regex."""
import os
from tokenizers import Tokenizer
JSON=("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
      "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json")
CASES=[
    # newline with trailing spaces after (non-newline ws after last newline)
    "a\n b","a\n  b","a\n\t b","a \n b","a  \n  b","a\n b\n c",
    # space before newline (\s* eats space then [\r\n]+)
    "a \nb","a  \nb","a\t\nb","a \t\nb","a\n \nb",
    # only-whitespace strings with newlines
    " \n"," \n ","  \n  ","\n ","\n  ","\t\n ","\n\t","  \n\n  ",
    # trailing ws runs of mixed type (\s+(?!\S))
    "word \t","word\t ","word \t\n","word\n\t ","word \n\t",
    # internal mixed ws between words (\s+ leaves one for next prefix)
    "a \tb","a\t b","a \t b","a\t\tb","a \t\t b",
    # multiple newlines separated by spaces
    "a\n\nb","a\n \nb","a \n \n b","a\n\n\nb",
    # CR alone, CRLF, LFCR
    "a\rb","a\r\rb","a\n\rb","a\r\nb","a\r\n\r\nb",
    # vertical tab / form feed (regex \s includes them; is_whitespace covers VT/FF)
    "a\x0bb","a\x0cb","a \x0b b",
    # nbsp (U+00A0) - regex \s in unicode mode includes it
    "a b","a  b",
    # leading newline then word
    "\nword","\n\nword"," \nword","\n word",
    # word then only newlines
    "word\n","word\n\n","word \n","word\n ",
]
def esc(s):
    return s.replace("\\","\\\\").replace("\t","\\t").replace("\n","\\n").replace("\r","\\r")
def main():
    tk=Tokenizer.from_file(JSON)
    here=os.path.dirname(os.path.abspath(__file__))
    p=os.path.join(here,"skeptic_ws_cases.tsv")
    with open(p,"w",encoding="utf-8") as f:
        for s in CASES:
            f.write(esc(s)+"\t"+",".join(str(i) for i in tk.encode(s).ids)+"\n")
    print(f"wrote {len(CASES)} ws cases to {p}")
main()
