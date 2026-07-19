#!/usr/bin/env python3
# gen_conv3d_oracle.py — numpy/torch oracle for the conv3d op probe.
#
# Produces a small fixed conv3d case (NDHWC input, QRSCF filter) and dumps the
# input, filter, bias, and the torch-computed output (NDHWC) as flat f32 .bin
# files. The Mojo probe reads these and compares (cos + max_abs).
#
# Run: /tmp/vae_oracle_venv/bin/python gen_conv3d_oracle.py
import os, numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

def dump(name, arr):
    a = np.ascontiguousarray(arr.astype(np.float32)).ravel()
    a.tofile(os.path.join(HERE, name + ".bin"))
    with open(os.path.join(HERE, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in arr.shape))
    print(f"  {name:20s} shape={arr.shape}")

def main():
    np.random.seed(7)
    N, D, H, W, Cin = 1, 3, 5, 6, 4
    Cout = 8
    Kd, Kh, Kw = 3, 3, 3
    pad_d, pad_h, pad_w = 1, 1, 1  # symmetric (probe checks the plain symmetric kernel)

    x_ndhwc = np.random.randn(N, D, H, W, Cin).astype(np.float32)
    # torch conv3d wants NCDHW input + OIDHW (Cout,Cin,Kd,Kh,Kw) weight.
    w_oidhw = np.random.randn(Cout, Cin, Kd, Kh, Kw).astype(np.float32)
    bias = np.random.randn(Cout).astype(np.float32)

    x_t = torch.from_numpy(x_ndhwc).permute(0, 4, 1, 2, 3).contiguous()  # NCDHW
    w_t = torch.from_numpy(w_oidhw)
    b_t = torch.from_numpy(bias)
    y = F.conv3d(x_t, w_t, b_t, stride=1, padding=(pad_d, pad_h, pad_w))
    y_ndhwc = y.permute(0, 2, 3, 4, 1).contiguous().numpy()  # NDHWC

    # QRSCF filter = [Kd,Kh,Kw,Cin,Cout] from OIDHW [Cout,Cin,Kd,Kh,Kw]
    w_qrscf = np.transpose(w_oidhw, (2, 3, 4, 1, 0))

    dump("c3d_x", x_ndhwc)
    dump("c3d_w", w_qrscf)
    dump("c3d_b", bias)
    dump("c3d_y", y_ndhwc)
    print("conv3d oracle done ->", HERE)

if __name__ == "__main__":
    main()
