#!/usr/bin/env python3
"""Test the SVDQuant-as-ARA hypothesis (2026-07-11, corrected).

Q: can our SVDQuant low-rank serve as the frozen recovery overlay (ARA) for
quant-training, i.e. does base+low-rank recover a layer's OUTPUT well, comparably
to a trained ARA (which fits outputs on real activations)?

On a REAL LTX-2 ff.net weight (in {2048,8192}), weight int4 (group-64, the
TRAINING/W4A16 case: acts bf16, no rotation), rank r in {16 (ai-toolkit), 128
(our slab)}, we measure OUTPUT relL2 vs the fp teacher for FIVE reconstructions:

  int4-only                : Wdq = dequant_g64(int4(W))                  (no overlay)
  A: int4 + SVD  (data-free): Wdq + SVD_r(W - Wdq)      <- ai-toolkit ARA STRUCTURE, our SVD
  A: int4 + OPT  (trained  ): Wdq + argmin_rank-r ||X(R-BA)^T||   <- trained-ARA UPPER BOUND
  B: SVDQuant SVD (data-free): L=SVD_r(W); dequant_g64(int4(W-L)) + L    <- OUR decomposition
  B: SVDQuant OPT (act-aware): L=act-optimal rank-r; int4(W-L) + L

Strategy A = quantize-then-correct (ARA). Strategy B = low-rank-then-quantize
(SVDQuant/Nunchaku). Both spend the same budget: int4 weight + rank-r bf16 overlay.

Acts: 'normal' (isotropic; G~=I -> SVD==OPT, sanity) and 'outlier' (~2% input
channels 20x scale, mimics diffusion activation outliers; non-isotropic G, the
DISCRIMINATING regime). Fit on train acts, eval on HELD-OUT test acts.
NOTE: synthetic outlier acts (not cached real LTX-2 acts) + no Hadamard smoothing
(our real svdquant adds a data-free rotation that would help B further). Single
layer per run. So: directional evidence, not a production number.
"""
import sys
import torch
from safetensors import safe_open

SRC = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
G = 64
torch.set_grad_enabled(False)

# real ff.net layer; arg picks in-dim (2048 fast, 8192 slower eigh)
IN = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
KEY = ("model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0."
       + ("ff.net.0.proj.weight" if IN == 2048 else "ff.net.2.weight"))


def dq_g64(W):
    out, inh = W.shape
    Wg = W.view(out, inh // G, G)
    s = Wg.abs().amax(-1, keepdim=True) / 7.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (torch.clamp(torch.round(Wg / s), -8, 7) * s).view(out, inh)


def svd_r(M, r):
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


def opt_r(M, X, r, eps=1e-8):
    # rank-r BA minimizing ||X(M-BA)^T|| = ||(M-BA) G^{1/2}||, G=X^TX/N
    Gm = (X.T @ X) / X.shape[0]
    ev, evec = torch.linalg.eigh(Gm)
    ev = ev.clamp_min(0)
    Gh = (evec * ev.sqrt()) @ evec.T
    Ghi = (evec * (1.0 / (ev.sqrt() + eps))) @ evec.T
    return svd_r(M @ Gh, r) @ Ghi


def acts(n, inh, regime, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, inh, generator=g, dtype=torch.float64)
    if regime == "outlier":
        sc = torch.ones(inh, dtype=torch.float64)
        sc[torch.randperm(inh, generator=g)[:max(1, inh // 50)]] = 20.0
        X = X * sc
    return X


def relL2(y, yf):
    return ((y - yf).norm() / yf.norm()).item()


with safe_open(SRC, "pt") as h:
    W = h.get_tensor(KEY).to(torch.float64)
print(f"layer: ...{KEY.split('.',3)[-1]}  W {tuple(W.shape)} (out,in)")
Wdq = dq_g64(W)
print(f"int4-only weight relL2 = {((W-Wdq).norm()/W.norm()).item():.4f}\n")

hdr = f"{'regime':<8}{'r':>4}{'int4-only':>11}{'A:int4+SVD':>12}{'A:int4+OPT':>12}{'B:SVDq-SVD':>12}{'B:SVDq-OPT':>12}"
print(hdr)
for regime in ["normal", "outlier"]:
    Xtr = acts(4096, W.shape[1], regime, 0)
    Xte = acts(4096, W.shape[1], regime, 1)
    yf = Xte @ W.T
    base = relL2(Xte @ Wdq.T, yf)
    for r in [16, 128]:
        # A: quantize-then-correct (ARA structure)
        R = W - Wdq
        a_svd = relL2(Xte @ (Wdq + svd_r(R, r)).T, yf)
        a_opt = relL2(Xte @ (Wdq + opt_r(R, Xtr, r)).T, yf)
        # B: low-rank-then-quantize (SVDQuant)
        Ls = svd_r(W, r)
        b_svd = relL2(Xte @ (dq_g64(W - Ls) + Ls).T, yf)
        Lo = opt_r(W, Xtr, r)
        b_opt = relL2(Xte @ (dq_g64(W - Lo) + Lo).T, yf)
        print(f"{regime:<8}{r:>4}{base:>11.4f}{a_svd:>12.4f}{a_opt:>12.4f}{b_svd:>12.4f}{b_opt:>12.4f}")
print("\nlower relL2 = better output recovery. A=ARA-style(int4 then low-rank "
      "correction), B=SVDQuant(low-rank then int4 residual). SVD=our data-free, "
      "OPT=activation-trained upper bound. Decisive regime=OUTLIER.")
