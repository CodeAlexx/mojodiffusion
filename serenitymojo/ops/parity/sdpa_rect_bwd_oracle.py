#!/usr/bin/env python3
# sdpa_rect_bwd_oracle.py — PyTorch F64 reference for the decomposed RECTANGULAR
# SDPA BACKWARD (S_q != S_kv) — the shared asymmetric cross-attention primitive.
#
# Mirrors sdpa_bwd_nondegen_oracle.py but with separate Sq (query seq) and Skv
# (key/value seq), as needed by Anima cross-attn (Sq=img, Skv=text, Dh=128) and
# SDXL cross-attn (Sq=H·W, Skv=77, Dh=64). Non-degenerate sinusoidal fills so the
# gradient is genuinely nonzero (see the nondegen post-mortem for why modular
# fills can alias the BSHD H*Dh seq stride and zero the softmax-bwd gradient).
#
# Cases (kept SMALL for the shared 24GB 3090):
#   Dh=64  : Sq=64, Skv=77  (SDXL cross-attn class)
#   Dh=128 : Sq=96, Skv=16  (Anima cross-attn class, asymmetric the other way)
#
# Emits per (Sq,Skv,H,Dh): rect_Sq<Sq>_Skv<Skv>_H<H>_Dh<Dh>_{dq,dk,dv}.bin
# (little-endian f32, BSHD), written into THIS directory.
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/ops/parity/sdpa_rect_bwd_oracle.py
import numpy as np
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
B = 1
# (Sq, Skv, H, Dh)
CASES = [
    (64, 77, 5, 64),    # SDXL cross-attn class (Dh=64)
    (96, 16, 4, 128),   # Anima cross-attn class (Dh=128)
]


# Non-degenerate sinusoidal fills. q/d_out length Sq*..., k/v length Skv*...
# Each tensor gets its own phase so q/k/v/do never coincide.
def fq(n):
    i = np.arange(n).astype(np.float64)
    return np.sin(0.07 * i + 1.1) * 0.2
def fk(n):
    i = np.arange(n).astype(np.float64)
    return np.cos(0.05 * i + 0.5) * 0.2
def fv(n):
    i = np.arange(n).astype(np.float64)
    return np.sin(0.10 * i + 0.3) * 0.2
def fdo(n):
    i = np.arange(n).astype(np.float64)
    return np.cos(0.09 * i + 0.2) * 0.2


def ref(B, Sq, Skv, H, Dh):
    scale = 1.0 / np.sqrt(Dh)
    nq = B * Sq * H * Dh
    nkv = B * Skv * H * Dh
    Q = torch.tensor(fq(nq).reshape(B, Sq, H, Dh), requires_grad=True)
    K = torch.tensor(fk(nkv).reshape(B, Skv, H, Dh), requires_grad=True)
    V = torch.tensor(fv(nkv).reshape(B, Skv, H, Dh), requires_grad=True)
    DO = torch.tensor(fdo(nq).reshape(B, Sq, H, Dh))
    # scores[b,h,i,j] = scale * sum_d Q[b,i,h,d] K[b,j,h,d] ; softmax over j (Skv)
    scores = scale * torch.einsum("bihd,bjhd->bhij", Q, K)
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", p, V)   # [B,Sq,H,Dh]
    out.backward(DO)
    return (Q.grad.detach().reshape(-1).numpy().astype("<f4"),
            K.grad.detach().reshape(-1).numpy().astype("<f4"),
            V.grad.detach().reshape(-1).numpy().astype("<f4"))


def main():
    for (Sq, Skv, H, Dh) in CASES:
        dq, dk, dv = ref(B, Sq, Skv, H, Dh)
        tag = f"rect_Sq{Sq}_Skv{Skv}_H{H}_Dh{Dh}"
        dq.tofile(os.path.join(HERE, tag + "_dq.bin"))
        dk.tofile(os.path.join(HERE, tag + "_dk.bin"))
        dv.tofile(os.path.join(HERE, tag + "_dv.bin"))
        print("wrote", tag, "|dq|=", float(np.linalg.norm(dq)),
              "|dk|=", float(np.linalg.norm(dk)), "|dv|=", float(np.linalg.norm(dv)))


if __name__ == "__main__":
    main()
