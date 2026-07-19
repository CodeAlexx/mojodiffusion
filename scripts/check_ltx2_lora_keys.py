#!/usr/bin/env python
"""Byte-compare two LTX-2 LoRA safetensors (musubi comfy vs Mojo re-save).

Usage: check_ltx2_lora_keys.py <original.safetensors> <resaved.safetensors>

PASS bar (chroma 912/912 pattern): identical key sets, shapes, dtypes, and
bit-exact payloads for every key. Also prints the module inventory so the
Phase-0 coverage claim (384 modules = 48 x {attn1,attn2} x {q,k,v,out.0})
gets re-proven on the real artifact.
"""
import sys
from collections import Counter
from safetensors import safe_open

a_path, b_path = sys.argv[1], sys.argv[2]

def load(path):
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out

A, B = load(a_path), load(b_path)
print(f"A: {len(A)} keys   B: {len(B)} keys")

only_a = sorted(set(A) - set(B))
only_b = sorted(set(B) - set(A))
if only_a: print("ONLY IN A:", only_a[:10], "..." if len(only_a) > 10 else "")
if only_b: print("ONLY IN B:", only_b[:10], "..." if len(only_b) > 10 else "")

mismatch = exact = 0
import torch
for k in sorted(set(A) & set(B)):
    ta, tb = A[k], B[k]
    if ta.shape != tb.shape or ta.dtype != tb.dtype:
        print(f"SHAPE/DTYPE MISMATCH {k}: {ta.shape}/{ta.dtype} vs {tb.shape}/{tb.dtype}")
        mismatch += 1
        continue
    if torch.equal(ta.view(torch.uint8) if ta.dtype == torch.uint8 else ta.view(torch.int16) if ta.element_size()==2 else ta,
                   tb.view(torch.uint8) if tb.dtype == torch.uint8 else tb.view(torch.int16) if tb.element_size()==2 else tb):
        exact += 1
    else:
        d = (ta.float() - tb.float()).abs().max().item()
        print(f"PAYLOAD DIFF {k}: max_abs={d}")
        mismatch += 1

# module inventory (comfy keys: diffusion_model.transformer_blocks.N.<fam>.<proj>.lora_A/B.weight)
fams = Counter()
for k in A:
    parts = k.split(".")
    if len(parts) >= 5 and parts[1] == "transformer_blocks":
        fams[parts[3]] += 1
print("module families (key counts):", dict(fams))
print(f"RESULT: {exact} bit-exact / {mismatch} mismatched / "
      f"{len(only_a)+len(only_b)} key-set diffs")
print("PASS" if mismatch == 0 and not only_a and not only_b and exact == len(A) else "FAIL")
