#!/usr/bin/env python3
"""SKEPTIC: probe special/added-token matching behavior in HF, and the
special=False subset, so we can compare to the Mojo _split_on_specials which
treats ALL 26 added tokens as atomic.
"""
from tokenizers import Tokenizer
import json

JSON = ("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
        "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json")
tk = Tokenizer.from_file(JSON)

def show(s):
    enc = tk.encode(s)
    print(f"  {s!r:40} -> ids={enc.ids} toks={enc.tokens}")

print("=== special=True tokens inline ===")
show("<|im_start|>user")
show("hello<|im_end|>world")
show("<|im_start|><|im_end|>")          # two specials back-to-back
show("a<|im_start|>b")                   # special inside text
show("<|im_")                            # partial
show("<|im_start|")                      # partial (missing >)
show("<|im_start|>")                     # exact

print("\n=== special=False added tokens (tool_call, think, fim...) ===")
show("<tool_call>")
show("<think>think</think>")
show("<|fim_prefix|>")
show("x<tool_call>y")

print("\n=== overlap / longest-match ===")
show("<|vision_start|><|vision_end|>")
show("<|object_ref_start|>")
show("<|object_ref_")                    # partial of a longer one
