#!/usr/bin/env python3
# pool_bwd_oracle.py — PyTorch reference for the naive pool/upsample BACKWARD
# (serenitymojo/ops/pool_backward.mojo). Tier-5 de-risk gate.
#
# Oracle = PyTorch F.max_pool2d / F.interpolate(mode='nearest') + autograd
# (stable ground-truth math). Python is a DEV-ONLY oracle per the parity
# convention. F64 throughout for a clean oracle (the Mojo path is F32 interior;
# gate is cos >= 0.999).
#
# ── LAYOUT BRIDGE (the make-or-break) ─────────────────────────────────────────
# The Mojo backward is NHWC (matches ops/conv.mojo + models/vae/upsample.mojo).
# Torch's F.max_pool2d / F.interpolate are NCHW. So we, mirroring
# conv2d_bwd_oracle.py:
#   * build deterministic data DIRECTLY in mojo layout (flat row-major NHWC),
#     the SAME fills the Mojo driver reproduces;
#   * permute into torch NCHW for the forward;
#   * run autograd;
#   * permute the resulting grads BACK to mojo NHWC and dump flat row-major.
# That way the dumped *_dx are byte-position-comparable to the Mojo kernel
# outputs (NHWC d_x).
#
# Permutations:
#   x_nhwc  [N,Hi,Wi,C]  -> torch x_nchw  = permute(0,3,1,2)
#   gy_nhwc [N,Ho,Wo,C]  -> torch gy_nchw = permute(0,3,1,2)
#   then:  x.grad [N,C,Hi,Wi] -> nhwc d_x = permute(0,2,3,1)
#
# ── MaxPool2D: N=2, C=3, Hi=Wi=8, K=2, stride=2, pad=0 (=> Ho=Wo=4). ──────────
#   x fill chosen so each 2x2 window has a UNIQUE max (no ties) — keeps the
#   argmax well-defined and the torch-vs-mojo first-max tie-break irrelevant for
#   THIS data (a tie would still match since both use first-max, but we avoid it
#   to make the gate unambiguous).
# ── UpsampleNearest2D: N=2, C=3, in_h=in_w=5, scale=2 (=> Ho=Wo=10). ──────────
#
# Emits 2 lines: "maxpool_dx ...", "upsample_dx ..." (flat mojo NHWC layout).
# Run: /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/pool_bwd_oracle.py

import numpy as np
import os
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "pool_bwd_ref.txt")

# ── MaxPool shape (MUST match pool_bwd_parity.mojo) ───────────────────────────
MP_N, MP_Hi, MP_Wi, MP_C = 2, 8, 8, 3
MP_K, MP_S = 2, 2
MP_Ho = (MP_Hi - MP_K) // MP_S + 1
MP_Wo = (MP_Wi - MP_K) // MP_S + 1

# ── Upsample shape (MUST match pool_bwd_parity.mojo) ──────────────────────────
US_N, US_h, US_w, US_C = 2, 5, 5, 3
US_SCALE = 2
US_Ho = US_h * US_SCALE
US_Wo = US_w * US_SCALE


def fill_maxpool_x(n):
    # MUST match _fill_mp_x in pool_bwd_parity.mojo.
    # A strictly-increasing-ish pattern via a large coprime stride mod a big
    # modulus => unique values within every 2x2 window (no max ties).
    a = np.empty(n, np.float64)
    for i in range(n):
        a[i] = (float((i * 37) % 257) - 128.0) * 0.01
    return a


def fill_maxpool_gy(n):
    # MUST match _fill_mp_gy.
    a = np.empty(n, np.float64)
    for i in range(n):
        a[i] = (float((i * 3) % 9) - 4.0) * 0.05
    return a


def fill_upsample_gy(n):
    # MUST match _fill_us_gy.
    a = np.empty(n, np.float64)
    for i in range(n):
        a[i] = (float((i * 7) % 13) - 6.0) * 0.05
    return a


def main():
    lines = []

    # ── MaxPool2D backward ────────────────────────────────────────────────────
    nx = MP_N * MP_Hi * MP_Wi * MP_C
    ngy = MP_N * MP_Ho * MP_Wo * MP_C
    x_nhwc = torch.tensor(fill_maxpool_x(nx), dtype=torch.float64).reshape(
        MP_N, MP_Hi, MP_Wi, MP_C
    )
    gy_nhwc = torch.tensor(fill_maxpool_gy(ngy), dtype=torch.float64).reshape(
        MP_N, MP_Ho, MP_Wo, MP_C
    )
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    gy_nchw = gy_nhwc.permute(0, 3, 1, 2).contiguous()

    y = F.max_pool2d(x_nchw, kernel_size=MP_K, stride=MP_S, padding=0)
    assert y.shape == (MP_N, MP_C, MP_Ho, MP_Wo), y.shape
    y.backward(gy_nchw)
    dx_nhwc = x_nchw.grad.permute(0, 2, 3, 1).contiguous().reshape(-1).numpy()
    lines.append("maxpool_dx " + " ".join(f"{v:.8f}" for v in dx_nhwc.tolist()))

    # ── UpsampleNearest2D backward ────────────────────────────────────────────
    in_nx = US_N * US_h * US_w * US_C
    out_ngy = US_N * US_Ho * US_Wo * US_C
    xin_nhwc = torch.zeros(US_N, US_h, US_w, US_C, dtype=torch.float64)
    xin_nchw = xin_nhwc.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    gy_us_nhwc = torch.tensor(
        fill_upsample_gy(out_ngy), dtype=torch.float64
    ).reshape(US_N, US_Ho, US_Wo, US_C)
    gy_us_nchw = gy_us_nhwc.permute(0, 3, 1, 2).contiguous()

    yout = F.interpolate(xin_nchw, scale_factor=US_SCALE, mode="nearest")
    assert yout.shape == (US_N, US_C, US_Ho, US_Wo), yout.shape
    yout.backward(gy_us_nchw)
    dx_us_nhwc = (
        xin_nchw.grad.permute(0, 2, 3, 1).contiguous().reshape(-1).numpy()
    )
    lines.append(
        "upsample_dx " + " ".join(f"{v:.8f}" for v in dx_us_nhwc.tolist())
    )

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print(
        f"shapes: maxpool dx numel={nx} (Ho={MP_Ho} Wo={MP_Wo}); "
        f"upsample dx numel={in_nx} (Ho={US_Ho} Wo={US_Wo})"
    )


if __name__ == "__main__":
    main()
