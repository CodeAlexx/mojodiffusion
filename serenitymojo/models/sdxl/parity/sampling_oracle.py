#!/usr/bin/env python3
# sampling_oracle.py — torch autograd reference for SDXL Down/Up sampling
# fwd+bwd (models/sdxl/sampling.mojo).
#
# Downsample: stride-2 pad-1 Conv3x3.
# Upsample:   F.interpolate(nearest, scale=2) then stride-1 pad-1 Conv3x3.
#
# Layout-bridge identical to conv2d_bwd_oracle.py: data built in MOJO layout
# (NHWC x, RSCF w), permuted to torch NCHW for fwd+autograd, grads permuted back.
# Emits forward outputs (mojo NHWC layout) AND grads for both samplers.
# Run: /home/alex/serenityflow-v2/.venv/bin/python .../sampling_oracle.py

import os
import numpy as np
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "sampling_ref.txt")

# Down: N=2, C=16, Hi=Wi=8 -> Ho=Wo=4 (stride2 pad1 k3)
# Up:   N=2, C=16, Hi=Wi=4 -> up 8x8 -> conv (stride1 pad1 k3) -> 8x8
N, C = 2, 16
DHi = DWi = 8     # downsample input
UHi = UWi = 4     # upsample input


def fill(n, a, b, c, scale=0.05):
    out = np.empty(n, np.float64)
    for i in range(n):
        out[i] = (float((i * a) % b) - c) * scale
    return out


def fx(n):   return fill(n, 7, 13, 6.0)
def fw(n):   return fill(n, 5, 11, 5.0)
def fb(n):   return fill(n, 4, 10, 5.0)
def fgo(n):  return fill(n, 2, 7, 3.0)


def conv_pack(x_nhwc, w_rscf, b_vec):
    x = x_nhwc.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    w = w_rscf.permute(3, 2, 0, 1).contiguous().requires_grad_(True)
    b = b_vec.clone().requires_grad_(True)
    return x, w, b


def main():
    lines = []

    # ── DOWNSAMPLE ────────────────────────────────────────────────────────────
    nx = N * DHi * DWi * C
    nw = 3 * 3 * C * C
    x_nhwc = torch.tensor(fx(nx), dtype=torch.float64).reshape(N, DHi, DWi, C)
    w_rscf = torch.tensor(fw(nw), dtype=torch.float64).reshape(3, 3, C, C)
    b_vec = torch.tensor(fb(C), dtype=torch.float64)
    x, w, b = conv_pack(x_nhwc, w_rscf, b_vec)
    y = F.conv2d(x, w, b, stride=2, padding=1)  # [N,C,4,4]
    Ho, Wo = y.shape[2], y.shape[3]
    go = torch.tensor(fgo(N * Ho * Wo * C), dtype=torch.float64).reshape(N, Ho, Wo, C)
    go_nchw = go.permute(0, 3, 1, 2).contiguous()
    y.backward(go_nchw)
    y_nhwc = y.detach().permute(0, 2, 3, 1).contiguous().reshape(-1).numpy()
    dx = x.grad.permute(0, 2, 3, 1).contiguous().reshape(-1).numpy()
    dw = w.grad.permute(2, 3, 1, 0).contiguous().reshape(-1).numpy()
    db = b.grad.detach().reshape(-1).numpy()
    lines.append("down_out " + " ".join(f"{v:.8f}" for v in y_nhwc.tolist()))
    lines.append("down_dx " + " ".join(f"{v:.8f}" for v in dx.tolist()))
    lines.append("down_dw " + " ".join(f"{v:.8f}" for v in dw.tolist()))
    lines.append("down_db " + " ".join(f"{v:.8f}" for v in db.tolist()))

    # ── UPSAMPLE ──────────────────────────────────────────────────────────────
    nx = N * UHi * UWi * C
    x_nhwc = torch.tensor(fx(nx), dtype=torch.float64).reshape(N, UHi, UWi, C)
    w_rscf = torch.tensor(fw(nw), dtype=torch.float64).reshape(3, 3, C, C)
    b_vec = torch.tensor(fb(C), dtype=torch.float64)
    x, w, b = conv_pack(x_nhwc, w_rscf, b_vec)
    up = F.interpolate(x, scale_factor=2, mode="nearest")  # [N,C,8,8]
    y = F.conv2d(up, w, b, stride=1, padding=1)            # [N,C,8,8]
    Ho, Wo = y.shape[2], y.shape[3]
    go = torch.tensor(fgo(N * Ho * Wo * C), dtype=torch.float64).reshape(N, Ho, Wo, C)
    go_nchw = go.permute(0, 3, 1, 2).contiguous()
    y.backward(go_nchw)
    y_nhwc = y.detach().permute(0, 2, 3, 1).contiguous().reshape(-1).numpy()
    dx = x.grad.permute(0, 2, 3, 1).contiguous().reshape(-1).numpy()
    dw = w.grad.permute(2, 3, 1, 0).contiguous().reshape(-1).numpy()
    db = b.grad.detach().reshape(-1).numpy()
    lines.append("up_out " + " ".join(f"{v:.8f}" for v in y_nhwc.tolist()))
    lines.append("up_dx " + " ".join(f"{v:.8f}" for v in dx.tolist()))
    lines.append("up_dw " + " ".join(f"{v:.8f}" for v in dw.tolist()))
    lines.append("up_db " + " ".join(f"{v:.8f}" for v in db.tolist()))

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
