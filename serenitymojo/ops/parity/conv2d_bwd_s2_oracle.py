#!/usr/bin/env python3
# conv2d_bwd_s2_oracle.py — PyTorch reference for the naive conv2d BACKWARD at
# STRIDE 2 (SDXL Downsample `.op` = stride-2 Conv3x3). Closes the [MED] skeptic
# gap: conv2d_backward's general-stride path was implemented but only stride-1
# was gated (SKEPTIC_FINDINGS_sdxl_P1, ATTACK 1b).
#
# Same layout-bridge discipline as conv2d_bwd_oracle.py: build data in MOJO
# layout, permute to torch NCHW / [Cout,Cin,Kh,Kw], run autograd, permute grads
# back to mojo layout, dump flat row-major.
#
# Shape: N=2, Cin=4, Cout=8, Hi=Wi=8, K=3, stride=2, pad=1 -> Ho=Wo=4.
# Run: /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/conv2d_bwd_s2_oracle.py

import numpy as np
import os
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "conv2d_bwd_s2_ref.txt")

# ── compile-time-matched shape (MUST match conv2d_bwd_s2_parity.mojo) ─────────
N, Hi, Wi, Cin = 2, 8, 8, 4
Kh, Kw, Cout = 3, 3, 8
SH, SW, PH, PW = 2, 2, 1, 1
Ho = (Hi + 2 * PH - Kh) // SH + 1
Wo = (Wi + 2 * PW - Kw) // SW + 1


def fill_x(n):
    a = np.empty(n, np.float64)
    for i in range(n):
        a[i] = (float((i * 7) % 13) - 6.0) * 0.05
    return a


def fill_w(n):
    a = np.empty(n, np.float64)
    for i in range(n):
        a[i] = (float((i * 5) % 11) - 5.0) * 0.05
    return a


def fill_gy(n):
    a = np.empty(n, np.float64)
    for i in range(n):
        a[i] = (float((i * 3) % 9) - 4.0) * 0.05
    return a


def main():
    nx = N * Hi * Wi * Cin
    nw = Kh * Kw * Cin * Cout
    ng = N * Ho * Wo * Cout

    # data in MOJO layout (flat row-major)
    x_nhwc = torch.tensor(fill_x(nx), dtype=torch.float64).reshape(N, Hi, Wi, Cin)
    w_rscf = torch.tensor(fill_w(nw), dtype=torch.float64).reshape(Kh, Kw, Cin, Cout)
    gy_nhwc = torch.tensor(fill_gy(ng), dtype=torch.float64).reshape(N, Ho, Wo, Cout)

    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    w_oikk = w_rscf.permute(3, 2, 0, 1).contiguous().requires_grad_(True)
    b = (torch.arange(Cout, dtype=torch.float64) * 0.1 - 0.15).requires_grad_(True)

    gy_nchw = gy_nhwc.permute(0, 3, 1, 2).contiguous()

    y = F.conv2d(x_nchw, w_oikk, b, stride=(SH, SW), padding=(PH, PW))
    assert y.shape == (N, Cout, Ho, Wo), y.shape
    y.backward(gy_nchw)

    dx_nhwc = x_nchw.grad.permute(0, 2, 3, 1).contiguous().reshape(-1).numpy()
    dw_rscf = w_oikk.grad.permute(2, 3, 1, 0).contiguous().reshape(-1).numpy()
    db = b.grad.detach().reshape(-1).numpy()

    lines = []
    lines.append("conv_dx " + " ".join(f"{v:.8f}" for v in dx_nhwc.tolist()))
    lines.append("conv_dw " + " ".join(f"{v:.8f}" for v in dw_rscf.tolist()))
    lines.append("conv_db " + " ".join(f"{v:.8f}" for v in db.tolist()))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print(f"shapes: dx numel={nx} dw numel={nw} db numel={Cout}  (Ho={Ho} Wo={Wo}, stride=2)")


if __name__ == "__main__":
    main()
