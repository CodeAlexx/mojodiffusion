#!/usr/bin/env python3
# conv2d_bwd_oracle.py — PyTorch reference for the naive conv2d BACKWARD
# (serenitymojo/ops/conv2d_backward.mojo). Tier-5 de-risk gate.
#
# Oracle = PyTorch F.conv2d + autograd (stable ground-truth math). Python is a
# DEV-ONLY oracle per the parity convention. F64 throughout for a clean oracle
# (the Mojo path is F32 interior; gate is cos >= 0.999).
#
# ── LAYOUT BRIDGE (the make-or-break) ─────────────────────────────────────────
# The Mojo forward (ops/conv.mojo) is NHWC input / RSCF filter [Kh,Kw,Cin,Cout]
# / NHWC output. Torch's F.conv2d is NCHW input / [Cout,Cin,Kh,Kw] filter / NCHW
# output. So we:
#   * build deterministic data DIRECTLY in mojo layout (flat row-major), the
#     SAME fills the Mojo driver reproduces;
#   * permute into torch NCHW / [Cout,Cin,Kh,Kw] for the forward;
#   * run autograd;
#   * permute the resulting grads BACK to mojo layout and dump flat row-major.
# That way the dumped conv_dx/conv_dw/conv_db are byte-position-comparable to the
# Mojo kernel outputs (which are NHWC d_x, RSCF d_w, [Cout] d_b).
#
# Permutations:
#   x_nhwc [N,Hi,Wi,Cin]      -> torch x_nchw   = permute(0,3,1,2)
#   w_rscf [Kh,Kw,Cin,Cout]   -> torch w_oikk   = permute(3,2,0,1)
#   gy_nhwc[N,Ho,Wo,Cout]     -> torch gy_nchw  = permute(0,3,1,2)
#   then grads back:
#   x.grad  [N,Cin,Hi,Wi]     -> nhwc d_x = permute(0,2,3,1)
#   w.grad  [Cout,Cin,Kh,Kw]  -> rscf d_w = permute(2,3,1,0)
#   b.grad  [Cout]            -> as is
#
# Shape: N=2, Cin=3, Cout=4, H=W=8, K=3, stride=1, pad=1.
#
# Emits 3 lines: "conv_dx ...", "conv_dw ...", "conv_db ..." (flat mojo layout).
# Run: /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/conv2d_bwd_oracle.py

import numpy as np
import os
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "conv2d_bwd_ref.txt")

# ── compile-time-matched shape (MUST match conv2d_bwd_parity.mojo) ────────────
N, Hi, Wi, Cin = 2, 8, 8, 3
Kh, Kw, Cout = 3, 3, 4
SH, SW, PH, PW = 1, 1, 1, 1
Ho = (Hi + 2 * PH - Kh) // SH + 1
Wo = (Wi + 2 * PW - Kw) // SW + 1


def fill_x(n):
    # MUST match _fill_x in conv2d_bwd_parity.mojo
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

    # permute into torch NCHW / [Cout,Cin,Kh,Kw]
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    w_oikk = w_rscf.permute(3, 2, 0, 1).contiguous().requires_grad_(True)
    b = torch.zeros(Cout, dtype=torch.float64, requires_grad=True)
    # NOTE: a deterministic nonzero bias does not affect d_x/d_w/d_b (db = sum of
    # grad_y regardless), but keep it learnable so b.grad is produced.
    b = (torch.arange(Cout, dtype=torch.float64) * 0.1 - 0.15).requires_grad_(True)

    gy_nchw = gy_nhwc.permute(0, 3, 1, 2).contiguous()

    y = F.conv2d(x_nchw, w_oikk, b, stride=(SH, SW), padding=(PH, PW))
    assert y.shape == (N, Cout, Ho, Wo), y.shape
    y.backward(gy_nchw)

    # grads back to mojo layout
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
    print(f"shapes: dx numel={nx} dw numel={nw} db numel={Cout}  (Ho={Ho} Wo={Wo})")


if __name__ == "__main__":
    main()
