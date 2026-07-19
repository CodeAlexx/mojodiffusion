#!/usr/bin/env python
"""Compare pure-Mojo GPT-OSS captures vs the HF oracle.

Per captured layer: cosine similarity (flattened) AND magnitude ratio
|mine|/|ref|. cos is magnitude-blind so BOTH are reported. Gate: cos>=0.999 per
layer (0.99 allowed for the final/deep layer — raw numbers printed either way).

Run AFTER both dumps exist:
  mine_captures.safetensors  (Mojo, keys l5/l11/l17/l23)
  oracle_captures.safetensors (HF, keys l5/l11/l17/l23 + oracle_final_normed)
"""
import sys
from safetensors import safe_open
import torch

D = "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/"
MINE = D + "mine_captures.safetensors"
ORACLE = D + "oracle_captures.safetensors"
LAYERS = ["l5", "l11", "l17", "l18", "l19", "l20", "l21", "l22", "l23"]
GATE = 0.999
GATE_FINAL = 0.99  # allowed for the deepest captured layer (l23)


def load(path):
    out = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).float()
    return out


def cos(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def main():
    mine = load(MINE)
    orac = load(ORACLE)
    print("per-layer: cos | |mine|/|ref| | shapes")
    results = {}
    allpass = True
    for li in LAYERS:
        if li not in mine:
            print(f"{li}: MISSING in mine")
            allpass = False
            continue
        m, r = mine[li], orac[li]
        if m.shape != r.shape:
            print(f"{li}: SHAPE MISMATCH mine={tuple(m.shape)} ref={tuple(r.shape)}")
            allpass = False
            continue
        c = cos(m, r)
        mag = float(m.norm() / (r.norm() + 1e-30))
        thr = GATE_FINAL if li == "l23" else GATE
        ok = c >= thr
        allpass = allpass and ok
        results[li] = (c, mag)
        print(f"{li}: cos={c:.6f} mag={mag:.4f} thr={thr} "
              f"{'PASS' if ok else 'FAIL'} shapes mine={tuple(m.shape)}")
    print("GATE:", "PASS" if allpass else "FAIL")
    print("RESULTS_JSON", {k: {"cos": v[0], "mag": v[1]} for k, v in results.items()})


if __name__ == "__main__":
    main()
