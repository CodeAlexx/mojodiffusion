#!/usr/bin/env python3
"""SquareQ→INT4 self-quantizer (weight-only W4A16 SVDQuant, CLEAN row-major layout).

Our Mojo inference is dequant-first: weights → bf16, GEMM in bf16 (activations
STAY bf16). So we do NOT need activation-int4 or its smoothing — only a good
WEIGHT quantizer. That is exactly SVDQuant minus the hard A4 part:

    W [out,in] bf16
    → SVD: W ≈ L1 @ L2  (rank R),  L1[out,R]=U_R Σ_R,  L2[R,in]=V_R^T
    → residual = W − L1@L2
    → group-G symmetric int4 of residual: per (group, out) scale = max|.|/7
    → clean row-major pack: qweight[out,in/2] (2 nibbles/byte, lo=even),
       wscales[in/G,out], lora_down[in,R]=L2^T, lora_up[out,R]=L1, smooth[in]=1, bias

Mojo reconstruct (ops/svdquant.mojo, twos_complement/lo_even):
    W_rec[o,k] = int4(nibble)*wscales[k//G,o] + (lora_up@lora_down^T)[o,k]   (smooth=1)

This is the format ops/svdquant.mojo already reads at synthetic cos 1.0. Weight-
only + no calibration → no activation data needed; the low-rank branch recovers
the group-int4 error. Emits a fixture the Mojo gate loads to measure real-weight
fidelity vs the ORIGINAL bf16.

Usage:
  .venv/bin/python svdquant_selfquant.py <src.safetensors> <weight_key> <out_fixture.safetensors> \
      [--rank 32] [--group 64] [--M 8]
"""
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file

RANK = 32
G = 64
NBITS_MAX = 7  # symmetric int4 range [-8,7]; use /7 so max maps to 7


# quant math lives in SquareQ now (folded 2026-07-10); this preserves the exact
# original full-SVD behavior the per-layer Mojo gate was verified against.
import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # in-repo scripts/squareq package
from squareq.svdquant_int4 import quantize_svdquant_w4a16 as _sq_quant, NBITS_MAX as _SQ_NBITS  # noqa: E402

def quantize_svdquant_w4a16(W, rank, group):
    return _sq_quant(W, rank, group, full_svd=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("weight_key"); ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--group", type=int, default=G)
    ap.add_argument("--M", type=int, default=8)
    a = ap.parse_args()

    with safe_open(a.src, "pt") as f:
        W = f.get_tensor(a.weight_key)  # [out,in] bf16
    out, inh = W.shape
    print(f"layer {a.weight_key}  [{out},{inh}]  rank={a.rank} group={a.group}")

    qbyte, wscales, lora_down, lora_up, smooth = quantize_svdquant_w4a16(W, a.rank, a.group)

    # reference: reconstruct exactly as Mojo will, + a functional y on random x
    Wf = W.float()
    v = torch.zeros(out, inh, dtype=torch.int16)
    nib = (qbyte.view(torch.uint8).to(torch.int16) & 0xFF)
    lo = nib & 0xF; hi = (nib >> 4) & 0xF
    v[:, 0::2] = torch.where(lo >= 8, lo - 16, lo)
    v[:, 1::2] = torch.where(hi >= 8, hi - 16, hi)
    sc = wscales.float().t()[:, torch.arange(inh) // a.group]  # [out,in]
    W_rec = v.float() * sc + (lora_up.float() @ lora_down.float().t())
    cosW = torch.nn.functional.cosine_similarity(W_rec.flatten(), Wf.flatten(), dim=0).item()
    relW = (W_rec - Wf).norm() / Wf.norm()
    print(f"  [ref] W_recon cos={cosW:.5f}  relL2={relW:.4f}  (int4+r{a.rank} on the REAL weight)")

    x = torch.randn(a.M, inh)
    bias = torch.zeros(out)
    y_ref = (x * smooth.float()) @ W_rec.t() + bias  # what Mojo computes
    y_true = x @ Wf.t() + bias                        # the ideal bf16 output
    cosy = torch.nn.functional.cosine_similarity(y_ref.flatten(), y_true.flatten(), dim=0).item()
    print(f"  [ref] y(dequant) vs y(bf16-ideal) cos={cosy:.5f}")

    save_file({
        "qweight": qbyte.contiguous(),
        "wscales": wscales.contiguous(),
        "lora_down": lora_down.contiguous(),
        "lora_up": lora_up.contiguous(),
        "smooth": smooth.contiguous(),
        "bias": bias.to(torch.bfloat16).contiguous(),
        "x": x.to(torch.bfloat16).contiguous(),
        "W_orig": Wf.contiguous(),          # ground truth for the Mojo gate
        "y_true": y_true.contiguous(),      # ideal output
        "meta": torch.tensor([out, inh, a.rank, a.group, a.M], dtype=torch.int32),
    }, a.out)
    print(f"  wrote {a.out}  (qweight I8 {list(qbyte.shape)} = {qbyte.numel()/1e6:.1f}MB, "
          f"vs bf16 {W.numel()*2/1e6:.1f}MB → {qbyte.numel()/(W.numel()*2):.2f}x)")


if __name__ == "__main__":
    main()
