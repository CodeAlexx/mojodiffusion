#!/usr/bin/env python3
# reduce_bwd_oracle.py — PyTorch reference for the Tier-0/Tier-1 BACKWARD arms
# (serenitymojo/ops/reduce_backward.mojo): sqrt, square, log, softmax,
# logsoftmax, sum, mean.
#
# Oracle = PyTorch autograd.grad (F64 for a clean ground-truth). Python is a
# DEV-ONLY oracle per the parity convention. The Mojo driver reproduces every
# deterministic input on-device; only the reference GRADIENTS are read back.
#
# For log/sqrt the inputs are POSITIVE (those grads divide by x / sqrt(x)).
#
# Emits one line per tag: "<tag> v0 v1 ...". Tags:
#   sqrt_dx, square_dx, log_dx, softmax_dx, logsoftmax_dx, sum_dx, mean_dx
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/ops/parity/reduce_bwd_oracle.py

import os
import numpy as np
import torch

OUT = os.path.join(os.path.dirname(__file__), "reduce_bwd_ref.txt")

# Shapes (must match reduce_bwd_parity.mojo exactly).
N_ELEM = 64          # elementwise arms (sqrt/square/log) + sum/mean input
ROWS = 8             # softmax / logsoftmax rows
COLS = 16            # softmax / logsoftmax cols (last dim)
COLS_WIDE = 1024     # WIDE softmax/logsoftmax cols (> _TPB=256; real attn width)


def fill_pos(n):
    """Deterministic POSITIVE input in (0, ~3.2]. Matches _fill_pos in Mojo."""
    return np.array([(float(i % 13) + 1.0) * 0.25 for i in range(n)], np.float64)


def fill_signed(n):
    """Deterministic signed input. Matches _fill_signed in Mojo."""
    return np.array([(float((i * 7) % 13) - 6.0) * 0.1 for i in range(n)], np.float64)


def fill_grad(n):
    """Deterministic upstream grad. Matches _fill_grad in Mojo."""
    return np.array([(float((i * 2) % 7) - 3.0) * 0.05 for i in range(n)], np.float64)


def emit(lines, tag, arr):
    lines.append(tag + " " + " ".join(f"{x:.8f}" for x in arr.reshape(-1).tolist()))


def main():
    lines = []

    # ── elementwise: sqrt, square, log (positive x) ─────────────────────────
    x_pos = torch.tensor(fill_pos(N_ELEM), dtype=torch.float64, requires_grad=True)
    g_elem = torch.tensor(fill_grad(N_ELEM), dtype=torch.float64)
    y = torch.sqrt(x_pos); y.backward(g_elem)
    emit(lines, "sqrt_dx", x_pos.grad.detach().numpy()); x_pos.grad = None
    y = x_pos * x_pos; y.backward(g_elem)
    emit(lines, "square_dx", x_pos.grad.detach().numpy()); x_pos.grad = None
    y = torch.log(x_pos); y.backward(g_elem)
    emit(lines, "log_dx", x_pos.grad.detach().numpy()); x_pos.grad = None

    # ── softmax / logsoftmax over last dim, 2D [ROWS, COLS] ─────────────────
    xs = torch.tensor(fill_signed(ROWS * COLS).reshape(ROWS, COLS),
                      dtype=torch.float64, requires_grad=True)
    gs = torch.tensor(fill_grad(ROWS * COLS).reshape(ROWS, COLS),
                      dtype=torch.float64)
    sm = torch.softmax(xs, dim=-1); sm.backward(gs)
    emit(lines, "softmax_dx", xs.grad.detach().numpy()); xs.grad = None
    lsm = torch.log_softmax(xs, dim=-1); lsm.backward(gs)
    emit(lines, "logsoftmax_dx", xs.grad.detach().numpy()); xs.grad = None

    # ── softmax / logsoftmax WIDE: cols=1024 (real attention-ish width) ───────
    # Exercises the cols>256 (>_TPB) reduction path in reduce_backward.mojo.
    xw = torch.tensor(fill_signed(ROWS * COLS_WIDE).reshape(ROWS, COLS_WIDE),
                      dtype=torch.float64, requires_grad=True)
    gw = torch.tensor(fill_grad(ROWS * COLS_WIDE).reshape(ROWS, COLS_WIDE),
                      dtype=torch.float64)
    smw = torch.softmax(xw, dim=-1); smw.backward(gw)
    emit(lines, "softmax_wide_dx", xw.grad.detach().numpy()); xw.grad = None
    lsmw = torch.log_softmax(xw, dim=-1); lsmw.backward(gw)
    emit(lines, "logsoftmax_wide_dx", xw.grad.detach().numpy()); xw.grad = None

    # ── sum / mean: scalar grad broadcast to input shape ────────────────────
    # Use a fixed scalar upstream grad of 1.0 (sum) — what the Mojo driver passes.
    xsum = torch.tensor(fill_signed(N_ELEM), dtype=torch.float64, requires_grad=True)
    s = xsum.sum(); s.backward(torch.tensor(1.0, dtype=torch.float64))
    emit(lines, "sum_dx", xsum.grad.detach().numpy()); xsum.grad = None
    m = xsum.mean(); m.backward(torch.tensor(1.0, dtype=torch.float64))
    emit(lines, "mean_dx", xsum.grad.detach().numpy()); xsum.grad = None

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
