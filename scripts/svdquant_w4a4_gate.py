#!/usr/bin/env python3
"""W4A4 offline-quantizer GATE (B.3a) — validate quantize_svdquant_w4a4 on the
REAL big LTX-2.3 ff layers (not just the fixture), with outlier activations,
before regenerating the whole 22B slab.

Loads a handful of the dominant class-A linears from the source bf16 checkpoint,
runs the SquareQ W4A4 quantizer + w4a4_forward, and reports cos vs the bf16 ideal
under normal + channel-outlier activations. Bar: >= 0.985 (the sim showed ~0.99).

Usage: python scripts/svdquant_w4a4_gate.py
"""
import os
import sys
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # in-repo scripts/squareq package
from squareq.svdquant_int4 import quantize_svdquant_w4a4, w4a4_forward, hadamard, _unpack_int4_perout  # noqa: E402

SWEEP = "--sweep" in sys.argv

SRC = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
# The dominant + representative layers (block 0).
KEYS = [
    "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight",   # 16384x4096
    "model.diffusion_model.transformer_blocks.0.ff.net.2.weight",        # 4096x16384
    "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",      # 4096x4096
    "model.diffusion_model.transformer_blocks.0.audio_ff.net.0.proj.weight",  # 8192x2048
]
M = 1536


def cos(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _w4a4_cfg(W, rank, group):
    """W4A4 with configurable rank + weight-scale granularity (per-out if group=0,
    else group-`group`). Returns cos on normal + outlier acts."""
    out, inh = W.shape
    Wd = W.float()
    q = min(rank + 16, min(out, inh))
    Ul, Sl, Vl = torch.svd_lowrank(Wd, q=q, niter=6)
    L1 = (Ul[:, :rank] * Sl[:rank]).double(); L2 = Vl[:, :rank].t().double()
    R = Wd.double() - L1 @ L2
    H = hadamard(inh)
    Rrot = R @ H
    if group == 0:                                  # per-output
        ws = Rrot.abs().amax(1, keepdim=True) / 7.0
        ws = torch.where(ws == 0, torch.ones_like(ws), ws)
        Rq = torch.clamp(torch.round(Rrot / ws), -8, 7)
        Rdeq = Rq * ws
    else:                                           # group-`group` along in
        r = Rrot.view(out, inh // group, group)
        gs = r.abs().amax(2, keepdim=True) / 7.0
        gs = torch.where(gs == 0, torch.ones_like(gs), gs)
        Rdeq = (torch.clamp(torch.round(r / gs), -8, 7) * gs).view(out, inh)

    def fwd(X):
        Xr = X.double() @ H
        xs = Xr.abs().amax(1, keepdim=True) / 7.0
        xs = torch.where(xs == 0, torch.ones_like(xs), xs)
        Xq = torch.clamp(torch.round(Xr / xs), -8, 7) * xs
        return Xq @ Rdeq.t() + (X.double() @ L2.t()) @ L1.t()

    Xn = torch.randn(M, inh); Xo = torch.randn(M, inh)
    oc = torch.randperm(inh)[: max(1, inh // 64)]; Xo[:, oc] *= 20.0
    yn, yo = X_ideal(W, Xn), X_ideal(W, Xo)
    return cos(fwd(Xn), yn), cos(fwd(Xo), yo)


def X_ideal(W, X):
    return X.double() @ W.double().t()


def sweep():
    torch.manual_seed(0)
    layers = ["model.diffusion_model.transformer_blocks.0.ff.net.2.weight",
              "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight",
              "model.diffusion_model.transformer_blocks.0.audio_ff.net.0.proj.weight"]
    cfgs = [(32, 0), (64, 0), (128, 0), (32, 64), (64, 64)]
    with safe_open(SRC, "pt") as f:
        for k in layers:
            W = f.get_tensor(k).float()
            print(f"\n{k.split('blocks.0.')[-1]}  [{W.shape[0]}x{W.shape[1]}]")
            for rank, grp in cfgs:
                cn, co = _w4a4_cfg(W, rank, grp)
                tag = f"r{rank} " + ("per-out" if grp == 0 else f"grp{grp}")
                print(f"  {tag:14s} normal={cn:.5f} outlier={co:.5f}")


def main():
    if SWEEP:
        sweep(); return
    torch.manual_seed(0)
    print(f"{'layer':44s} {'in':>6s} {'out':>6s} {'normal':>8s} {'outlier':>8s}")
    worst = 1.0
    with safe_open(SRC, "pt") as f:
        avail = set(f.keys())
        for k in KEYS:
            if k not in avail:
                print(f"{k[-44:]:44s}  MISSING")
                continue
            W = f.get_tensor(k).float()             # [out,in]
            out, inh = W.shape
            if (inh & (inh - 1)) != 0:
                print(f"{k[-44:]:44s}  in={inh} not power-of-2 — skip")
                continue
            qb, ws, ld, lu = quantize_svdquant_w4a4(W)

            Xn = torch.randn(M, inh)
            Xo = torch.randn(M, inh)
            oc = torch.randperm(inh)[: max(1, inh // 64)]
            Xo[:, oc] *= 20.0

            cn = cos(w4a4_forward(Xn, qb, ws, ld, lu), Xn @ W.t())
            co = cos(w4a4_forward(Xo, qb, ws, ld, lu), Xo @ W.t())
            worst = min(worst, cn, co)
            print(f"{k.split('transformer_blocks.0.')[-1]:44s} {inh:6d} {out:6d} {cn:8.5f} {co:8.5f}")
    print(f"\nworst cos = {worst:.5f}   (bar >= 0.985)  {'PASS' if worst >= 0.985 else 'FAIL'}")


if __name__ == "__main__":
    main()
