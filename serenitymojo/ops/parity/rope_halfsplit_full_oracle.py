#!/usr/bin/env python3
# rope_halfsplit_full_oracle.py — PyTorch reference for the half-split RoPE
# FORWARD (rope_halfsplit_full) and BACKWARD on a REAL interleaved-doubled angle
# table where cos[i] != cos[i+half] (the table the real ERNIE model uses, NOT the
# degenerate cos[i]==cos[i+half] table that hides the backward bug).
#
# Convention (matches diffusers transformer_ernie_image.apply_rotary_emb):
#   freqs = interleaved-doubled angle table [theta0,theta0,theta1,theta1,...] (width D)
#   cos = cos(freqs), sin = sin(freqs)        (FULL-WIDTH [rows, D])
#   x1,x2 = x.chunk(2)  (half-split)  ; x_rot = cat(-x2, x1)
#   out = x*cos + x_rot*sin
#   => out[i]      = x[i]*cos[i]      - x[i+half]*sin[i]        (i<half)
#      out[i+half] = x[i+half]*cos[i+half] + x[i]*sin[i+half]
#
# This is exactly _rope_halfsplit_full_kernel in ops/rope.mojo and the backward we
# add (rope_halfsplit_full_backward). The Mojo driver rebuilds x/cos/sin/grad_out
# with the SAME fills; only the reference fwd-out + d_x are read back.
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/ops/parity/rope_halfsplit_full_oracle.py

import math
import os
import numpy as np
import torch

OUT = os.path.join(os.path.dirname(__file__), "rope_halfsplit_full_ref.txt")


# ── deterministic fills (MUST match the Mojo driver) ─────────────────────────
def fill(n, mul, sub, scale):
    a = np.empty(n, np.float64)
    for i in range(n):
        a[i] = (float((i * mul) % 13) - sub) * scale
    return a


def build_interleaved_doubled_angles(rows, D):
    # half distinct angles per row, each repeated CONSECUTIVELY [a0,a0,a1,a1,...].
    # angles are row/index dependent so cos[i] != cos[i+half] in general.
    half = D // 2
    ang = np.empty((rows, D), np.float64)
    for r in range(rows):
        for j in range(half):
            a = math.sin(0.11 * (r + 1) + 0.07 * j) * 1.3 + 0.2 * j
            ang[r, 2 * j] = a
            ang[r, 2 * j + 1] = a
    return ang


def main():
    rows = 16
    D = 64
    half = D // 2

    x_np = fill(rows * D, 7, 6.0, 0.05).reshape(rows, D)
    g_np = fill(rows * D, 2, 6.0, 0.05).reshape(rows, D)
    ang = build_interleaved_doubled_angles(rows, D)
    cos_np = np.cos(ang)
    sin_np = np.sin(ang)

    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    cos = torch.tensor(cos_np, dtype=torch.float64)
    sin = torch.tensor(sin_np, dtype=torch.float64)
    grad_out = torch.tensor(g_np, dtype=torch.float64)

    # diffusers apply_rotary_emb (half-split, full-width cos/sin)
    x1 = x[:, :half]
    x2 = x[:, half:]
    x_rot = torch.cat((-x2, x1), dim=1)
    out = x * cos + x_rot * sin
    out.backward(grad_out)

    # sanity: confirm cos[i] != cos[i+half] (real table, not degenerate)
    diff = float((cos[:, :half] - cos[:, half:]).abs().max())
    print("max |cos[:half]-cos[half:]| =", diff, "(must be >> 0)")

    def line(tag, arr):
        return tag + " " + " ".join(f"{v:.8f}" for v in arr.reshape(-1).tolist())

    with open(OUT, "w") as f:
        f.write(line("fwd_out", out.detach().numpy()) + "\n")
        f.write(line("dx", x.grad.detach().numpy()) + "\n")
    print("wrote", OUT, "rows", rows, "D", D)


if __name__ == "__main__":
    main()
