#!/usr/bin/env python3
# sdpa_bwd_nondegen_oracle.py — PyTorch F64 reference for the decomposed SDPA
# BACKWARD using NON-DEGENERATE sinusoidal inputs.
#
# WHY THIS EXISTS (BUG_sdpa_backward_H30_dq_dk_zero post-mortem):
#   The original sdpa_bwd_oracle / sdpa_bwd_realseq_oracle use modular index
#   fills, e.g. V[i] = ((i*3)%9 - 4)*0.05. In BSHD the seq stride per (head,dim)
#   is H*Dh. For H in {6,30} and Dh=128 the stride*3 ≡ 0 (mod 9), so V is
#   CONSTANT across the sequence → grad_attn rows constant → softmax-bwd
#   grad_scores is MATHEMATICALLY ZERO → d_q/d_k are genuinely ~0 (torch agrees:
#   |d_q| ≈ 2.5e-18 at H=30). Cosine of two ~zero vectors is meaningless noise,
#   which the old H=30 gate misread as a kernel FAILURE. The kernel is CORRECT.
#   These sinusoidal fills never alias with H*Dh stride, so d_q/d_k are nonzero
#   and the parity is a real test of the kernel at H=30/H=6 (non-32-aligned).
#
# Emits per (S,H): nd_S<S>_H<H>_{dq,dk,dv}.bin (little-endian f32, BSHD),
# written into THIS directory (read by sdpa_bwd_nondegen_parity.mojo and the
# fixed sdpa_bwd_realseq_parity.mojo).
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/ops/parity/sdpa_bwd_nondegen_oracle.py
import numpy as np
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
B = 1
DH = 128
# (S, H) cases. H=30 is Z-Image's real head count; H=6 a second non-32-aligned
# head count; H=32 a 32-aligned control. S in {256,384,1152,2304} cover the real
# Z-Image unified_len / image-token sequence lengths (256px..768px-class).
CASES = [
    (256, 30), (256, 6), (256, 32), (384, 30),  # head-count + small-S coverage
    (1152, 30), (2304, 30),                       # real 512px / 768px-class seqs
]


def fills(n):
    i = np.arange(n).astype(np.float64)
    q = np.sin(0.07 * i + 1.1) * 0.2
    k = np.cos(0.05 * i + 0.5) * 0.2
    v = np.sin(0.10 * i + 0.3) * 0.2
    do = np.cos(0.09 * i + 0.2) * 0.2
    return q, k, v, do


def ref(B, S, H, Dh):
    scale = 1.0 / np.sqrt(Dh)
    n = B * S * H * Dh
    q, k, v, do = fills(n)
    sh = (B, S, H, Dh)
    Q = torch.tensor(q.reshape(sh), requires_grad=True)
    K = torch.tensor(k.reshape(sh), requires_grad=True)
    V = torch.tensor(v.reshape(sh), requires_grad=True)
    DO = torch.tensor(do.reshape(sh))
    scores = scale * torch.einsum("bihd,bjhd->bhij", Q, K)
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", p, V)
    out.backward(DO)
    return (Q.grad.detach().reshape(-1).numpy().astype("<f4"),
            K.grad.detach().reshape(-1).numpy().astype("<f4"),
            V.grad.detach().reshape(-1).numpy().astype("<f4"))


def main():
    for (S, H) in CASES:
        dq, dk, dv = ref(B, S, H, DH)
        tag = f"nd_S{S}_H{H}"
        dq.tofile(os.path.join(HERE, tag + "_dq.bin"))
        dk.tofile(os.path.join(HERE, tag + "_dk.bin"))
        dv.tofile(os.path.join(HERE, tag + "_dv.bin"))
        print("wrote", tag, "|dq|=", float(np.linalg.norm(dq)),
              "|dk|=", float(np.linalg.norm(dk)))


if __name__ == "__main__":
    main()
