#!/usr/bin/env python3
# norm_bwd_oracle.py — PyTorch reference for the norm BACKWARD kernels
# (serenitymojo/ops/norm_backward.mojo). Mirrors sdpa_bwd_oracle.py: writes one
# tagged space-separated float line per gradient tensor to norm_bwd_ref.txt; the
# Mojo gate reproduces inputs deterministically and reads only the GRADIENTS.
#
# Oracle = PyTorch autograd (F64). Python is DEV-ONLY (parity convention).
#
# CRITICAL — layout / eps / axis MUST match serenitymojo/ops/norm.mojo forward:
#   RMSNorm   : x [rows, D], normalize over D, weight only (no bias).
#               y = x / sqrt(mean(x^2)+eps) * g          (F.rms_norm)
#   LayerNorm : x [rows, D], normalize over D, weight + bias, BIASED var.
#               y = (x-mean)/sqrt(var+eps) * g + b       (F.layer_norm)
#   GroupNorm : x is **NHWC** [N,H,W,C] in serenitymojo. PyTorch group_norm is
#               NCHW, so we permute NHWC->NCHW, run F.group_norm, permute the
#               result/grads back to NHWC. Per-channel weight/bias [C], BIASED
#               var over each (n, group) across (C/G, H*W).
#   eps = 1e-5 everywhere (the gate passes the same eps).
#
# Inputs use the SAME closed-form deterministic fills the Mojo gate reproduces
# (so only grads cross the boundary). See _fill_* below — keep in lockstep with
# norm_bwd_parity.mojo.
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/norm_bwd_oracle.py

import os
import numpy as np
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "norm_bwd_ref.txt")
EPS = 1e-5


# ── deterministic fills (MUST match norm_bwd_parity.mojo _fill_*) ────────────
def fill_x(n):
    return np.array([(float((i * 7) % 13) - 6.0) * 0.05 for i in range(n)], np.float64)


def fill_g(n):
    return np.array([(float((i * 5) % 11) - 5.0) * 0.05 + 1.0 for i in range(n)], np.float64)


def fill_b(n):
    return np.array([(float((i * 3) % 9) - 4.0) * 0.05 for i in range(n)], np.float64)


def fill_go(n):
    return np.array([(float((i * 2) % 7) - 3.0) * 0.05 for i in range(n)], np.float64)


def emit(lines, tag, arr):
    lines.append(tag + " " + " ".join(f"{x:.8f}" for x in arr.reshape(-1).tolist()))


def main():
    lines = []

    # ── RMSNorm: rows x D ────────────────────────────────────────────────────
    ROWS, D = 8, 64
    x = torch.tensor(fill_x(ROWS * D).reshape(ROWS, D), requires_grad=True)
    g = torch.tensor(fill_g(D), requires_grad=True)
    go = torch.tensor(fill_go(ROWS * D).reshape(ROWS, D))
    y = F.rms_norm(x, (D,), weight=g, eps=EPS)
    dx, dg = torch.autograd.grad(y, (x, g), grad_outputs=go)
    emit(lines, "rms_dx", dx.detach().numpy())
    emit(lines, "rms_dg", dg.detach().numpy())

    # ── LayerNorm: rows x D ──────────────────────────────────────────────────
    lx = torch.tensor(fill_x(ROWS * D).reshape(ROWS, D), requires_grad=True)
    lg = torch.tensor(fill_g(D), requires_grad=True)
    lb = torch.tensor(fill_b(D), requires_grad=True)
    lgo = torch.tensor(fill_go(ROWS * D).reshape(ROWS, D))
    ly = F.layer_norm(lx, (D,), weight=lg, bias=lb, eps=EPS)
    ldx, ldg, ldb = torch.autograd.grad(ly, (lx, lg, lb), grad_outputs=lgo)
    emit(lines, "ln_dx", ldx.detach().numpy())
    emit(lines, "ln_dg", ldg.detach().numpy())
    emit(lines, "ln_db", ldb.detach().numpy())

    # ── GroupNorm: NHWC in serenitymojo; torch wants NCHW ───────────────────
    N, H, W, C, G = 2, 4, 4, 8, 4
    nhwc = N * H * W * C
    x_nhwc = torch.tensor(fill_x(nhwc).reshape(N, H, W, C))
    go_nhwc = torch.tensor(fill_go(nhwc).reshape(N, H, W, C))
    # NHWC -> NCHW for torch
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    go_nchw = go_nhwc.permute(0, 3, 1, 2).contiguous()
    gg = torch.tensor(fill_g(C), requires_grad=True)
    gb = torch.tensor(fill_b(C), requires_grad=True)
    gy = F.group_norm(x_nchw, G, weight=gg, bias=gb, eps=EPS)
    gdx_nchw, gdg, gdb = torch.autograd.grad(gy, (x_nchw, gg, gb), grad_outputs=go_nchw)
    # dx back to NHWC to match the Mojo buffer layout
    gdx_nhwc = gdx_nchw.permute(0, 2, 3, 1).contiguous()
    emit(lines, "gn_dx", gdx_nhwc.detach().numpy())
    emit(lines, "gn_dg", gdg.detach().numpy())
    emit(lines, "gn_db", gdb.detach().numpy())

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print(f"rms rows={ROWS} D={D}; ln same; gn N={N} H={H} W={W} C={C} G={G}")


if __name__ == "__main__":
    main()
