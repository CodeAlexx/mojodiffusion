#!/usr/bin/env python
"""Generate the byte-matched fixture for the ltx2 trainer forward+loss parity
gate (torchref torch runtime vs Mojo stack). Writes:
  <out>/fixture.safetensors   noise_i (BF16, latent shape), sigma_i (F32 [1])
  <out>/pairs.txt             one line per pair: <arm> <lat_file> <te_file> <sigma>
Noise is stored BF16 because the torchref trainer's actual per-step noise is
randn_like(bf16 cache latents) — both sides upcast the SAME bf16 bytes to f32.
"""
import os
import sys
import glob

import torch
from safetensors.torch import save_file

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/alex/mojodiffusion/output/ltx2_parity_fwd"
VID = "/home/alex/datasets/ltx2_ref_v3/cache"
IMG = "/home/alex/datasets/ltx2_eri2_512/cache"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(123)

def lat_files(d):
    # exclude names with spaces/parens — they poison argv/text plumbing in
    # every downstream layer, and sample choice is arbitrary anyway
    fs = sorted(f for f in glob.glob(os.path.join(d, "*_ltx2.safetensors"))
                if not f.endswith("_te.safetensors")
                and " " not in os.path.basename(f)
                and "(" not in os.path.basename(f))
    return fs

def te_for(lat, d):
    tes = sorted(glob.glob(os.path.join(d, "*_ltx2_te.safetensors")))
    base = os.path.basename(lat)
    best = ""
    for te in tes:
        x = os.path.basename(te)[: -len("_ltx2_te.safetensors")]
        if base.startswith(x + "_") and len(x) > len(best):
            best = x
    assert best, f"no TE pair for {lat}"
    return os.path.join(d, best + "_ltx2_te.safetensors")

vid = lat_files(VID)
img = lat_files(IMG)
pairs = []
tensors = {}

VID_IDX = [0, 9, 18, 27, 36, 45, 54, 63]
VID_SIG = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.975]
IMG_IDX = [0, 29, 59, 88]
IMG_SIG = [0.10, 0.40, 0.675, 0.90]

k = 0
for arm, files, idxs, sigs, shape in (
    ("video", vid, VID_IDX, VID_SIG, (1, 128, 4, 9, 16)),
    ("image", img, IMG_IDX, IMG_SIG, (1, 128, 1, 16, 16)),
):
    for i, s in zip(idxs, sigs):
        lat = files[i]
        te = te_for(lat, os.path.dirname(lat))
        noise = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        tensors[f"noise_{k}"] = noise
        tensors[f"sigma_{k}"] = torch.tensor([s], dtype=torch.float32)
        pairs.append(f"{arm} {lat} {te} {s}")
        k += 1

save_file(tensors, os.path.join(OUT, "fixture.safetensors"))
with open(os.path.join(OUT, "pairs.txt"), "w") as f:
    f.write("\n".join(pairs) + "\n")
print(f"fixture: {k} pairs -> {OUT}")
for p in pairs:
    print(" ", p)
