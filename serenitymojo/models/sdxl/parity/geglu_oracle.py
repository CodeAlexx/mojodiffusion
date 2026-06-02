#!/usr/bin/env python3
# geglu_oracle.py — torch autograd reference for SDXL SpatialTransformer FF GEGLU
# fwd+bwd (models/sdxl/geglu.mojo).
#
# GELU = TANH-APPROX (approximate='tanh'). This matches flame-core Tensor::gelu
# (gelu_bf16_contig_direct / fused_kernels gelu = 0.5*x*(1+tanh(sqrt(2/pi)*(x+
# 0.044715 x^3)))), which is what sdxl_unet.rs:649 gate_part.gelu() invokes, AND
# the Mojo ops/activations.gelu (tanh-approx). Using exact-erf here would create
# a spurious ~1e-3 mismatch with the real model path. Tenet: match the SHIPPED
# math, not a textbook variant.
#
# Forward: proj=Linear(x); x_part,gate = split(proj); out = x_part * gelu(gate).
# Small dims: M=6, Cin=8, Cff=5 (proj out = 10).
# Run: /home/alex/serenityflow-v2/.venv/bin/python .../geglu_oracle.py

import os
import numpy as np
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "geglu_ref.txt")

M, Cin, Cff = 6, 8, 5


def fill(n, a, b, c, scale=0.05):
    out = np.empty(n, np.float64)
    for i in range(n):
        out[i] = (float((i * a) % b) - c) * scale
    return out


def main():
    x = torch.tensor(fill(M * Cin, 7, 13, 6.0), dtype=torch.float64).reshape(M, Cin).requires_grad_(True)
    proj_w = torch.tensor(fill(2 * Cff * Cin, 5, 11, 5.0), dtype=torch.float64).reshape(2 * Cff, Cin).requires_grad_(True)
    proj_b = torch.tensor(fill(2 * Cff, 4, 10, 5.0), dtype=torch.float64).reshape(2 * Cff).requires_grad_(True)

    proj = x @ proj_w.T + proj_b            # [M, 2*Cff]
    x_part = proj[:, :Cff]
    gate = proj[:, Cff:]
    g = F.gelu(gate, approximate="tanh")    # TANH-approx (matches flame-core + Mojo)
    out = x_part * g                        # [M, Cff]

    go = torch.tensor(fill(M * Cff, 2, 7, 3.0), dtype=torch.float64).reshape(M, Cff)
    out.backward(go)

    def flat(t):
        return t.detach().reshape(-1).numpy().tolist()

    lines = []
    lines.append("out " + " ".join(f"{v:.8f}" for v in flat(out)))
    lines.append("d_x " + " ".join(f"{v:.8f}" for v in flat(x.grad)))
    lines.append("d_proj_w " + " ".join(f"{v:.8f}" for v in flat(proj_w.grad)))
    lines.append("d_proj_b " + " ".join(f"{v:.8f}" for v in flat(proj_b.grad)))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
