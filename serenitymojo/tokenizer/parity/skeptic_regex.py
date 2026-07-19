#!/usr/bin/env python3
"""SKEPTIC item 4: regex pre-tokenizer tie-break stress -> fresh HF oracle TSV."""
import os
from tokenizers import Tokenizer
JSON=("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
      "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json")
CASES=[
    # multi-space runs of every length 1..8 between words
    "a b","a  b","a   b","a    b","a     b","a      b","a       b","a        b",
    # leading space runs
    " x","  x","   x","    x",
    # trailing space runs
    "x ","x  ","x   ","x    ",
    # punct then letter (the [^\r\n\p{L}\p{N}]?\p{L}+ branch)
    ".word","!word","?word",",word",";word",":word","-word","_word","#word","@word","$word","%word","&word","*word","(word",")word","[word","]word","{word","}word","/word","\\word","|word","~word","+word","=word","<word",">word",
    # space + punct + letter
    " .word"," !word"," (word",
    # multiple punct then letter (only the LAST punct attaches, rest is punct branch)
    "...word","!!!word","((word","-->arrow",
    # punct runs with trailing newline (the ' ?[^\s\p{L}\p{N}]+[\r\n]*' branch)
    "!!!\n","..\n\n","??\r\n",
    # digit runs (each \p{N} is single, then BPE merges)
    "1","12","123","1234","12345","123456","1234567","12345678","123456789","1234567890",
    # digit-letter-digit
    "1a2","a1a","1a1","99x99",
    # newline variants \s*[\r\n]+
    "\n","\n\n","\n\n\n","\r","\r\n","\r\n\r\n"," \n"," \n "," \n  \n ","\t\n","\t\n\t",
    # contractions with capitals (?i:)
    "DON'T","CAN'T","I'M","HE'S","WE'RE","YOU'VE","SHE'LL","THEY'D","It'S","cAn'T",
    "'tis","'TIS","'twas","y'know",
    # contraction not in the set (should NOT match contraction branch)
    "o'clock","rock'n'roll","'x","'z","'1",
    # apostrophe followed by space/punct
    "word' ","word'.","word',","'.","' ","'a'b",
    # CRLF mixed with content
    "line1\r\nline2","a\rb\nc","tab\tnewline\nend",
    # space runs that are entirely trailing (\s+(?!\S))
    "word   ","word\t\t","word \t ","word\n   ",
    # punct between spaces
    "a . b","a , b","a ! b","x - y","x_y_z",
    # everything-whitespace
    "   ","\t\t\t"," \t \t ",
]
def esc(s):
    return s.replace("\\","\\\\").replace("\t","\\t").replace("\n","\\n").replace("\r","\\r")
def main():
    tk=Tokenizer.from_file(JSON)
    here=os.path.dirname(os.path.abspath(__file__))
    p=os.path.join(here,"skeptic_regex_cases.tsv")
    with open(p,"w",encoding="utf-8") as f:
        for s in CASES:
            f.write(esc(s)+"\t"+",".join(str(i) for i in tk.encode(s).ids)+"\n")
    print(f"wrote {len(CASES)} regex cases to {p}")
main()
