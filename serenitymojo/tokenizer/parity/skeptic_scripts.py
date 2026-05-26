#!/usr/bin/env python3
"""SKEPTIC: non-Latin script / non-ASCII number prompts that stress the
\\p{L}/\\p{N} approximation. Writes fresh HF oracle TSV (skeptic_scripts.tsv).
7/15 of these diverge from the Mojo tokenizer (see SKEPTIC_FINDINGS F1)."""
import os
from tokenizers import Tokenizer
JSON=("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
      "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json")
CASES={
 'vietnamese':'Tôi yêu mèo đẹp',
 'vietnamese2':'cà phê sữa đá ngon',
 'polytonic greek':'ἀρχὴ ἥμισυ παντός',
 'georgian':'საქართველო ლამაზია',
 'bengali':'সুন্দর বিড়াল',
 'tamil':'அழகான பூனை',
 'thai':'แมวน่ารัก',
 'thai digits':'๑๒๓ แมว',
 'devanagari digits':'१२३ बिल्ली',
 'fullwidth latin':'Ｈｅｌｌｏ Ｗｏｒｌｄ',
 'fullwidth digits':'１２３４',
 'superscript':'E=mc² and x²+y²',
 'roman numerals':'Chapter Ⅳ and Ⅻ',
 'circled nums':'① ② ③ steps',
 'vietnamese precomposed dot':'ạụọ',
}
def esc(s):
    return s.replace("\\","\\\\").replace("\t","\\t").replace("\n","\\n").replace("\r","\\r")
def main():
    tk=Tokenizer.from_file(JSON)
    here=os.path.dirname(os.path.abspath(__file__))
    p=os.path.join(here,"skeptic_scripts.tsv")
    with open(p,"w",encoding="utf-8") as f:
        for _,s in CASES.items():
            f.write(esc(s)+"\t"+",".join(str(i) for i in tk.encode(s).ids)+"\n")
    print(f"wrote {len(CASES)} script cases to {p}")
main()
