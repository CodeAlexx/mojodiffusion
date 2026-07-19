#!/usr/bin/env python3
# optim_oracle.py — PyTorch reference for the Mojo optimizers + grad clip
# (serenitymojo/training/optim.mojo, Phase T4 of FULL_PORT_TRAINING_PLAN).
#
# Oracle = torch.optim (the canonical AdamW decoupled-WD and SGD-with-momentum
# implementations) + torch.nn.utils.clip_grad_norm_. Python is a DEV-ONLY
# oracle per the project parity convention.
#
# The Mojo driver (optim_parity.mojo) reproduces the SAME deterministic param /
# grad fills on-device, runs its own optimizer step(s), and compares the updated
# PARAMETER (and the clip total_norm scalar) against the tags emitted here.
#
# CRITICAL: we use torch.optim.AdamW (DECOUPLED weight decay), NOT
# torch.optim.Adam(weight_decay=...) (which is Adam+L2). PyTorch AdamW applies
# p *= (1 - lr * weight_decay) before the adaptive Adam subtraction.
# weight_decay > 0 is used so order/form bugs are load-bearing.
#
# Emits tagged space-separated float lines into optim_ref.txt:
#   adamw_p1   — param after 1 AdamW step
#   adamw_p3   — param after 3 AdamW steps (exercises bias correction over t)
#   sgd_p1     — param after 1 SGD step (momentum buffer init)
#   sgd_p3     — param after 3 SGD steps (exercises momentum accumulation)
#   clip_scaled — grads (concatenated) after global-norm clip
#   clip_norm  — the scalar total_norm returned by clip_grad_norm_
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/training/parity/optim_oracle.py

import numpy as np
import os
import torch

OUT = os.path.join(os.path.dirname(__file__), "optim_ref.txt")

# Optimizer hyperparameters — MUST match the Mojo driver exactly.
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WD = 0.01          # > 0 so decoupled-WD vs Adam+L2 differ (load-bearing)
SGD_LR = 0.1
SGD_MOMENTUM = 0.9
SGD_WD = 0.0       # gated SGD case uses wd=0 (torch SGD wd is coupled L2;
                   # the Mojo decoupled path matches only at wd=0 for SGD)

N = 64             # parameter element count

MAX_NORM = 1.0     # grad-clip threshold


def fill(n, a, b, c, scale=0.05):
    """Deterministic closed-form fill (matches the Mojo driver _fill)."""
    out = np.empty(n, np.float64)
    for i in range(n):
        out[i] = (float((i * a) % b) - c) * scale
    return out


def emit(lines, tag, arr):
    lines.append(tag + " " + " ".join(f"{x:.8f}" for x in np.asarray(arr).reshape(-1).tolist()))


def adamw_run(steps):
    p0 = fill(N, 7, 13, 6.0)            # initial param
    g = fill(N, 5, 11, 5.0)            # grad (same every step — deterministic)
    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.AdamW([p], lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WD)
    for _ in range(steps):
        opt.zero_grad()
        p.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
    return p.detach().numpy()


def sgd_run(steps):
    p0 = fill(N, 7, 13, 6.0)
    g = fill(N, 5, 11, 5.0)
    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([p], lr=SGD_LR, momentum=SGD_MOMENTUM, weight_decay=SGD_WD)
    for _ in range(steps):
        opt.zero_grad()
        p.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
    return p.detach().numpy()


def clip_run():
    # Two grad tensors so the global-norm spans multiple tensors (mirrors the
    # Mojo list-of-grads API). Fills chosen so the global norm exceeds MAX_NORM.
    g1 = fill(N, 5, 11, 5.0, scale=0.5)
    g2 = fill(N, 3, 9, 4.0, scale=0.5)
    t1 = torch.tensor(g1, dtype=torch.float64)
    t2 = torch.tensor(g2, dtype=torch.float64)
    # clip_grad_norm_ wants .grad on params; emulate via leaf tensors.
    p1 = torch.zeros(N, dtype=torch.float64, requires_grad=True)
    p2 = torch.zeros(N, dtype=torch.float64, requires_grad=True)
    p1.grad = t1.clone()
    p2.grad = t2.clone()
    total_norm = torch.nn.utils.clip_grad_norm_([p1, p2], max_norm=MAX_NORM, norm_type=2.0)
    scaled = np.concatenate([p1.grad.detach().numpy(), p2.grad.detach().numpy()])
    return scaled, float(total_norm)


def main():
    lines = []
    emit(lines, "adamw_p1", adamw_run(1))
    emit(lines, "adamw_p3", adamw_run(3))
    emit(lines, "sgd_p1", sgd_run(1))
    emit(lines, "sgd_p3", sgd_run(3))
    scaled, total_norm = clip_run()
    emit(lines, "clip_scaled", scaled)
    emit(lines, "clip_norm", np.array([total_norm]))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("N =", N, " clip total_norm =", total_norm)


if __name__ == "__main__":
    main()
