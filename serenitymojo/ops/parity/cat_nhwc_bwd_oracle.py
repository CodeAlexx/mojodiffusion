#!/usr/bin/env python3
# cat_nhwc_bwd_oracle.py — torch autograd reference for the NHWC channel-axis
# (axis=3) cat backward (SDXL decoder skip-concat). Gates the cat_backward
# channel-concat case (TRAINING_PLAN_sdxl.md:90, Phase 3).
#
# Forward: y = cat([h, skip], dim=3)  (NHWC last axis). Backward: split d_y back
# into d_h (first C0 channels) + d_skip (next C1 channels). Data is built in MOJO
# NHWC layout flat row-major; grads are already in that layout (axis-3 split is
# layout-preserving), so no permute needed.
#
# Shapes: N=2, H=W=4, C0=6, C1=10 (h has 6 channels, skip has 10).
# Run: /home/alex/serenityflow-v2/.venv/bin/python .../cat_nhwc_bwd_oracle.py

import os
import numpy as np
import torch

OUT = os.path.join(os.path.dirname(__file__), "cat_nhwc_bwd_ref.txt")

N, H, W, C0, C1 = 2, 4, 4, 6, 10


def fill(n, a, b, c):
    out = np.empty(n, np.float64)
    for i in range(n):
        out[i] = (float((i * a) % b) - c) * 0.05
    return out


def main():
    C = C0 + C1
    ng = N * H * W * C
    # upstream grad d_y in NHWC layout
    gy = torch.tensor(fill(ng, 3, 9, 4.0), dtype=torch.float64).reshape(N, H, W, C)

    h = torch.zeros(N, H, W, C0, dtype=torch.float64, requires_grad=True)
    skip = torch.zeros(N, H, W, C1, dtype=torch.float64, requires_grad=True)
    y = torch.cat([h, skip], dim=3)
    y.backward(gy)

    d_h = h.grad.reshape(-1).numpy()
    d_skip = skip.grad.reshape(-1).numpy()

    lines = [
        "d_h " + " ".join(f"{v:.8f}" for v in d_h.tolist()),
        "d_skip " + " ".join(f"{v:.8f}" for v in d_skip.tolist()),
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT, f"(d_h n={d_h.size}, d_skip n={d_skip.size})")


if __name__ == "__main__":
    main()
