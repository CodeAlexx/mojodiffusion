#!/usr/bin/env python3
# sdpa_bwd_oracle.py — PyTorch reference for the DECOMPOSED SDPA BACKWARD
# (serenitymojo/ops/attention_backward.mojo, Phase T0 of FULL_PORT_TRAINING_PLAN).
#
# Oracle = PyTorch (NOT flame-core — flame-core's backward is mid-v2-refactor and
# has the live cuDNN-SDPA-bwd misalign bug; PyTorch is stable ground-truth math).
# Python is a DEV-ONLY oracle per the parity convention (sdpa_math_oracle.py).
#
# Inputs are the SAME deterministic closed-form fills as sdpa_math_oracle.py for
# q/k/v, plus a deterministic d_out fill (the upstream gradient). The Mojo driver
# reproduces all four on-device; only the reference GRADIENTS (d_q,d_k,d_v) are
# read back.
#
# Layout BSHD [B,S,H,Dh]; non-causal full attention, mask=0. Reference:
#   attn = softmax_j( scale * Q@Kᵀ )
#   out  = attn @ V
#   grads via torch.autograd against d_out.
# F64 throughout for a clean oracle (the Mojo path is F32 interior; cos>=0.999).
#
# Emits one line per (tag,grad): "<tag>_dq ...", "<tag>_dk ...", "<tag>_dv ..."
# Run: /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/sdpa_bwd_oracle.py

import numpy as np
import os
import torch

OUT = os.path.join(os.path.dirname(__file__), "sdpa_bwd_ref.txt")


def gen_qkv_dout(B, S, H, Dh):
    """Deterministic BSHD q/k/v/d_out. q/k/v MUST match sdpa_math_oracle.gen_qkv;
    d_out is a new deterministic fill the Mojo driver reproduces."""
    n = B * S * H * Dh
    q = np.empty(n, np.float64)
    k = np.empty(n, np.float64)
    v = np.empty(n, np.float64)
    do = np.empty(n, np.float64)
    for i in range(n):
        q[i] = (float((i * 7) % 13) - 6.0) * 0.05
        k[i] = (float((i * 5) % 11) - 5.0) * 0.05
        v[i] = (float((i * 3) % 9) - 4.0) * 0.05
        do[i] = (float((i * 2) % 7) - 3.0) * 0.05
    sh = (B, S, H, Dh)
    return (q.reshape(sh), k.reshape(sh), v.reshape(sh), do.reshape(sh))


def sdpa_bwd_ref(B, S, H, Dh, scale):
    q, k, v, do = gen_qkv_dout(B, S, H, Dh)
    # Torch, BSHD -> per-head math via einsum, F64, autograd.
    Q = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    K = torch.tensor(k, dtype=torch.float64, requires_grad=True)
    V = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    DO = torch.tensor(do, dtype=torch.float64)
    # scores[b,h,i,j] = scale * sum_d Q[b,i,h,d]*K[b,j,h,d]
    scores = scale * torch.einsum("bihd,bjhd->bhij", Q, K)
    p = torch.softmax(scores, dim=-1)               # [B,H,S,S]
    out = torch.einsum("bhij,bjhd->bihd", p, V)     # [B,S,H,Dh]
    out.backward(DO)
    return (Q.grad.detach().reshape(-1).numpy(),
            K.grad.detach().reshape(-1).numpy(),
            V.grad.detach().reshape(-1).numpy())


def emit(lines, tag, B, S, H, Dh):
    scale = 1.0 / np.sqrt(Dh)
    dq, dk, dv = sdpa_bwd_ref(B, S, H, Dh, scale)
    lines.append(tag + "_dq " + " ".join(f"{x:.8f}" for x in dq.tolist()))
    lines.append(tag + "_dk " + " ".join(f"{x:.8f}" for x in dk.tolist()))
    lines.append(tag + "_dv " + " ".join(f"{x:.8f}" for x in dv.tolist()))


def main():
    lines = []
    # Dh=128 encoder-ish shape (matches the forward parity gate 2 shape).
    emit(lines, "dh128", 1, 8, 32, 128)
    # Dh=64 small shape (a second, distinct case).
    emit(lines, "dh64", 1, 8, 8, 64)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("dh128 numel =", 1 * 8 * 32 * 128, " dh64 numel =", 1 * 8 * 8 * 64)


if __name__ == "__main__":
    main()
