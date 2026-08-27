#!/usr/bin/env python3
# minimax_h3_inject_empty_te.py — append per-item empty-cond keys to mmh3 TE
# caches. The Mojo trainer's guidance objective (--guidance_scale > 0) reads
# `varlen_mmh3_empty_hidden_states_*` + `varlen_mmh3_empty_token_tags_int64`
# per item (akane --cache_guidance_empty contract); the pinned kohya branch
# only writes a run-level uncond probe (--uncond_output). The probe IS the
# distillation uncond (a single space, screened upstream), so sharing its
# rows across items is faithful.
#
# Usage: python minimax_h3_inject_empty_te.py <uncond.safetensors> <cache_dir>
import glob
import sys

import torch
from safetensors.torch import load_file, save_file

uncond_path, cache_dir = sys.argv[1], sys.argv[2]
u = load_file(uncond_path)
hid_keys = [k for k in u if "hidden" in k]
assert len(hid_keys) == 1, f"expected one hidden tensor in uncond file, got {list(u)}"
hidden = u[hid_keys[0]].to(torch.bfloat16)
tag_keys = [k for k in u if "tag" in k]
if tag_keys:
    tags = u[tag_keys[0]].to(torch.long)
else:
    tags = torch.ones(hidden.shape[0], dtype=torch.long)
print(f"uncond rows {hidden.shape} from {hid_keys[0]}; tags {'from file' if tag_keys else 'ones'}")

EK = "varlen_mmh3_empty_hidden_states_bfloat16"
ET = "varlen_mmh3_empty_token_tags_int64"
TK = "mmh3_conditioning_task"  # kohya TE writer omits it; reader requires it
n = 0
for p in sorted(glob.glob(f"{cache_dir}/*_mmh3_te.safetensors")):
    t = load_file(p)
    changed = False
    if EK not in t:
        t[EK] = hidden.clone()
        t[ET] = tags.clone()
        changed = True
    if TK not in t:
        t[TK] = torch.tensor(0, dtype=torch.long)  # t2va
        changed = True
    if changed:
        save_file(t, p)
        n += 1
print(f"injected empty-cond keys into {n} TE caches")
