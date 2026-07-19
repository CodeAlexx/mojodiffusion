#!/usr/bin/env python3
# activation_bwd_oracle.py — PyTorch reference for the Tier-1 activation
# BACKWARD kernels (serenitymojo/ops/activation_backward.mojo).
#
# Oracle = PyTorch autograd (stable ground-truth). Python is a DEV-ONLY oracle
# per the parity convention (sdpa_bwd_oracle.py). Computes d_x = grad_out * f'(x)
# for relu / gelu(tanh-approx) / silu / sigmoid / tanh.
#
# GELU uses approximate="tanh" to match the serenitymojo forward
# (ops/activations.mojo) and flame-core/kernels/gelu_backward.cu.
#
# Inputs are the SAME deterministic closed-form fills the Mojo driver
# reproduces on-device; only the reference GRADIENTS (d_x) are read back.
#   x  = ((i*7)%13 - 6) * 0.5      (spreads ~[-3, 3) — exercises nonlinearity)
#   go = ((i*5)%11 - 5) * 0.3      (arbitrary upstream gradient)
#
# Emits one tagged space-separated line per arm: "<tag> v0 v1 ..."
# tags: relu_dx gelu_dx silu_dx sigmoid_dx tanh_dx
# Run: /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/activation_bwd_oracle.py

import os
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "activation_bwd_ref.txt")
N = 4096


def gen_inputs(n):
    x = torch.empty(n, dtype=torch.float64)
    go = torch.empty(n, dtype=torch.float64)
    for i in range(n):
        x[i] = (float((i * 7) % 13) - 6.0) * 0.5
        go[i] = (float((i * 5) % 11) - 5.0) * 0.3
    return x, go


def grad_of(act_fn, x_in, go):
    x = x_in.clone().requires_grad_(True)
    y = act_fn(x)
    (gx,) = torch.autograd.grad(y, x, grad_outputs=go)
    return gx.detach().reshape(-1)


def main():
    x, go = gen_inputs(N)
    arms = [
        ("relu_dx", lambda t: F.relu(t)),
        ("gelu_dx", lambda t: F.gelu(t, approximate="tanh")),
        ("silu_dx", lambda t: F.silu(t)),
        ("sigmoid_dx", lambda t: torch.sigmoid(t)),
        ("tanh_dx", lambda t: torch.tanh(t)),
    ]
    lines = []
    for tag, fn in arms:
        gx = grad_of(fn, x, go)
        lines.append(tag + " " + " ".join(f"{v:.8f}" for v in gx.tolist()))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT, "N =", N)


if __name__ == "__main__":
    main()
