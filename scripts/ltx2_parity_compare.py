#!/usr/bin/env python
"""Compare the ltx2 trainer fwd parity dumps: musubi runtime (ref_out) vs the
Mojo stack (mojo_out) on byte-matched inputs.

Bars (pre-stated, numeric-parity-testing skill):
  - pred cosine >= 0.999 per pair (forward pass; weights bit-identical —
    proven — so the only free variable is arithmetic: Mojo F32 stack vs
    musubi bf16 autocast).
  - loss: |mojo - ref| reported; the target-arithmetic class differs (musubi
    computes noise-lat in bf16, Mojo in F32), so losses are compared as
    values + relative diff, not bit-equal.
"""
import os
import sys

import torch
from safetensors import safe_open

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/alex/mojodiffusion/output/ltx2_parity_fwd"

def load(p):
    d = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            d[k] = f.get_tensor(k)
    return d

ref = load(os.path.join(OUT, "ref_out.safetensors"))
mojo = load(os.path.join(OUT, "mojo_out.safetensors"))
pairs = [ln.rstrip("\n") for ln in open(os.path.join(OUT, "pairs.txt")) if ln.strip()]

ok = True
print(f"{'pair':>4} {'arm':>6} {'sigma':>6} | {'cos(pred)':>10} {'relL2':>8} | "
      f"{'loss_ref':>9} {'loss_mojo':>9} {'rel_dloss':>9}")
for k, ln in enumerate(pairs):
    arm = ln.split(" ", 1)[0]
    sigma = float(ln.rsplit(" ", 1)[1])
    rp = ref[f"pred_{k}"].float()            # [1,C,F,H,W]
    b, c, F, H, W = rp.shape
    rp_tok = rp.reshape(1, c, F * H * W).permute(0, 2, 1).reshape(-1)
    mp = mojo[f"pred_{k}"].float().reshape(-1)
    assert rp_tok.numel() == mp.numel(), (rp_tok.shape, mp.shape)
    cos = torch.nn.functional.cosine_similarity(rp_tok, mp, dim=0).item()
    rel = ((rp_tok - mp).norm() / rp_tok.norm()).item()
    lr_ = float(ref[f"loss_{k}"][0])
    lm = float(mojo[f"loss_{k}"][0])
    rd = abs(lm - lr_) / max(abs(lr_), 1e-9)
    flag = ""
    if cos < 0.999:
        flag = "  << cos FAIL"
        ok = False
    print(f"{k:>4} {arm:>6} {sigma:>6.3f} | {cos:>10.6f} {rel:>8.4f} | "
          f"{lr_:>9.6f} {lm:>9.6f} {rd:>9.4f}{flag}")

print("PASS (all pred cos >= 0.999)" if ok else "FAIL")
