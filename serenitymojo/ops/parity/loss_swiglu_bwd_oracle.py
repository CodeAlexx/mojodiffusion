#!/usr/bin/env python3
# loss_swiglu_bwd_oracle.py — PyTorch reference for MSE / Huber / SwiGLU BACKWARD
# (serenitymojo/ops/loss_swiglu_backward.mojo, Phase T of FULL_PORT_TRAINING_PLAN).
#
# Oracle = PyTorch autograd (stable ground-truth math), F64 throughout for a
# clean reference; the Mojo path is F32 interior, gated at cos >= 0.999.
# Python is a DEV-ONLY oracle (parity convention; see sdpa_bwd_oracle.py).
#
# Inputs are deterministic closed-form fills the Mojo driver reproduces on-device.
# Only the reference GRADIENTS are read back by the Mojo parity driver.
#
# Tags emitted (one space-separated float line each):
#   mse_dpred      d/dpred  mean((pred-target)^2)
#   huber_dpred    d/dpred  huber_loss(delta=DELTA, reduction='mean')
#   swiglu_dgate   d/dgate  of (silu(gate)*up), upstream grad_out
#   swiglu_dup     d/dup    of (silu(gate)*up), upstream grad_out
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/ops/parity/loss_swiglu_bwd_oracle.py

import numpy as np
import os
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "loss_swiglu_bwd_ref.txt")

# Problem size (single flat run; loss/activation backward are shape-agnostic).
N = 4096
DELTA = 1.0  # Huber delta; MUST match the Mojo driver.


def fill_pred(n):
    # deterministic; spans both Huber regions (|x| above and below DELTA).
    return np.array(
        [(float((i * 7) % 13) - 6.0) * 0.25 for i in range(n)], np.float64
    )


def fill_target(n):
    return np.array(
        [(float((i * 5) % 11) - 5.0) * 0.25 for i in range(n)], np.float64
    )


def fill_gate(n):
    return np.array(
        [(float((i * 3) % 9) - 4.0) * 0.20 for i in range(n)], np.float64
    )


def fill_up(n):
    return np.array(
        [(float((i * 2) % 7) - 3.0) * 0.20 for i in range(n)], np.float64
    )


def fill_grad_out(n):
    return np.array(
        [(float((i * 11) % 5) - 2.0) * 0.10 for i in range(n)], np.float64
    )


def mse_dpred(n):
    pred = torch.tensor(fill_pred(n), dtype=torch.float64, requires_grad=True)
    tgt = torch.tensor(fill_target(n), dtype=torch.float64)
    loss = F.mse_loss(pred, tgt, reduction="mean")
    (g,) = torch.autograd.grad(loss, pred)
    return g.detach().reshape(-1).numpy()


def huber_dpred(n):
    pred = torch.tensor(fill_pred(n), dtype=torch.float64, requires_grad=True)
    tgt = torch.tensor(fill_target(n), dtype=torch.float64)
    loss = F.huber_loss(pred, tgt, reduction="mean", delta=DELTA)
    (g,) = torch.autograd.grad(loss, pred)
    return g.detach().reshape(-1).numpy()


def swiglu_grads(n):
    gate = torch.tensor(fill_gate(n), dtype=torch.float64, requires_grad=True)
    up = torch.tensor(fill_up(n), dtype=torch.float64, requires_grad=True)
    go = torch.tensor(fill_grad_out(n), dtype=torch.float64)
    y = F.silu(gate) * up
    y.backward(go)
    return (
        gate.grad.detach().reshape(-1).numpy(),
        up.grad.detach().reshape(-1).numpy(),
    )


def main():
    lines = []
    dmse = mse_dpred(N)
    dhub = huber_dpred(N)
    dg, du = swiglu_grads(N)
    lines.append("mse_dpred " + " ".join(f"{x:.8f}" for x in dmse.tolist()))
    lines.append("huber_dpred " + " ".join(f"{x:.8f}" for x in dhub.tolist()))
    lines.append("swiglu_dgate " + " ".join(f"{x:.8f}" for x in dg.tolist()))
    lines.append("swiglu_dup " + " ".join(f"{x:.8f}" for x in du.tolist()))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("N =", N, " DELTA =", DELTA)


if __name__ == "__main__":
    main()
