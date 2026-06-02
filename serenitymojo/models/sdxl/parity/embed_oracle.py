#!/usr/bin/env python3
# embed_oracle.py — torch autograd reference for SDXL time+label embedding
# fwd+bwd (models/sdxl/embed.mojo).
#
# Forward: emb = time_mlp(sinusoidal(t)) + label_mlp(y).
#   sinusoidal: COS-first LDM (matches sdxl_unet.rs timestep_embedding and the
#   Mojo cos-first timestep_embedding kernel).
# Backward from d_emb gates both MLPs' Linear weights/biases + d_ts (grad wrt the
# sinusoidal vector) + d_y (grad wrt the ADM vector).
#
# Small dims for parity speed: B=2, Sdim=8, Tdim=16, Adm=12.
# Run: /home/alex/serenityflow-v2/.venv/bin/python .../embed_oracle.py

import os
import math
import numpy as np
import torch

OUT = os.path.join(os.path.dirname(__file__), "embed_ref.txt")

B, Sdim, Tdim, Adm = 2, 8, 16, 12
MAX_PERIOD = 10000.0


def fill(n, a, b, c, scale=0.05, off=0.0):
    out = np.empty(n, np.float64)
    for i in range(n):
        out[i] = (float((i * a) % b) - c) * scale + off
    return out


def sinusoidal_cos_first(t, dim):
    # MUST match ops/embeddings.mojo _timestep_embed_kernel_f32 (cos-first).
    half = dim // 2
    neg_ln = -math.log(MAX_PERIOD)
    bsz = t.shape[0]
    emb = torch.zeros(bsz, dim, dtype=torch.float64)
    for bi in range(bsz):
        for i in range(half):
            freq = math.exp(neg_ln * (i / half))
            angle = float(t[bi]) * freq
            emb[bi, i] = math.cos(angle)
            emb[bi, half + i] = math.sin(angle)
    return emb


def main():
    # timesteps: deterministic small values
    t = torch.tensor([3.0, 17.0], dtype=torch.float64)
    ts = sinusoidal_cos_first(t, Sdim).requires_grad_(True)  # [B,Sdim], treat as MLP input
    y = torch.tensor(fill(B * Adm, 5, 11, 5.0), dtype=torch.float64).reshape(B, Adm).requires_grad_(True)

    # weights [out, in]
    t0_w = torch.tensor(fill(Tdim * Sdim, 6, 17, 8.0), dtype=torch.float64).reshape(Tdim, Sdim).requires_grad_(True)
    t0_b = torch.tensor(fill(Tdim, 3, 9, 4.0), dtype=torch.float64).reshape(Tdim).requires_grad_(True)
    t2_w = torch.tensor(fill(Tdim * Tdim, 5, 11, 5.0), dtype=torch.float64).reshape(Tdim, Tdim).requires_grad_(True)
    t2_b = torch.tensor(fill(Tdim, 4, 10, 5.0), dtype=torch.float64).reshape(Tdim).requires_grad_(True)
    l0_w = torch.tensor(fill(Tdim * Adm, 7, 13, 6.0), dtype=torch.float64).reshape(Tdim, Adm).requires_grad_(True)
    l0_b = torch.tensor(fill(Tdim, 2, 7, 3.0), dtype=torch.float64).reshape(Tdim).requires_grad_(True)
    l2_w = torch.tensor(fill(Tdim * Tdim, 6, 17, 8.0), dtype=torch.float64).reshape(Tdim, Tdim).requires_grad_(True)
    l2_b = torch.tensor(fill(Tdim, 3, 9, 4.0), dtype=torch.float64).reshape(Tdim).requires_grad_(True)

    def silu(z):
        return z * torch.sigmoid(z)

    te = silu(ts @ t0_w.T + t0_b)
    te = te @ t2_w.T + t2_b
    le = silu(y @ l0_w.T + l0_b)
    le = le @ l2_w.T + l2_b
    emb = te + le  # [B,Tdim]

    go = torch.tensor(fill(B * Tdim, 2, 7, 3.0), dtype=torch.float64).reshape(B, Tdim)
    emb.backward(go)

    def flat(x):
        return x.detach().reshape(-1).numpy().tolist()

    lines = []
    lines.append("emb " + " ".join(f"{v:.8f}" for v in flat(emb)))
    lines.append("dt0_w " + " ".join(f"{v:.8f}" for v in flat(t0_w.grad)))
    lines.append("dt0_b " + " ".join(f"{v:.8f}" for v in flat(t0_b.grad)))
    lines.append("dt2_w " + " ".join(f"{v:.8f}" for v in flat(t2_w.grad)))
    lines.append("dt2_b " + " ".join(f"{v:.8f}" for v in flat(t2_b.grad)))
    lines.append("dl0_w " + " ".join(f"{v:.8f}" for v in flat(l0_w.grad)))
    lines.append("dl0_b " + " ".join(f"{v:.8f}" for v in flat(l0_b.grad)))
    lines.append("dl2_w " + " ".join(f"{v:.8f}" for v in flat(l2_w.grad)))
    lines.append("dl2_b " + " ".join(f"{v:.8f}" for v in flat(l2_b.grad)))
    lines.append("d_ts " + " ".join(f"{v:.8f}" for v in flat(ts.grad)))
    lines.append("d_y " + " ".join(f"{v:.8f}" for v in flat(y.grad)))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
