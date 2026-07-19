#!/usr/bin/env python
"""Torch replay for ltx2_f32_adamw_torch_parity.mojo.

Two-tier comparison (2026-07-16, measured):
  (1) mojo vs a strict-f32 same-op-order numpy replay — abs-class bar.
  (2) numpy replay vs torch.optim.AdamW (single-tensor) — informational; op
      order differs (lerp_/addcdiv_), so bit-level equality is NOT expected.
Bit-ulp comparison is the WRONG instrument here: EMA cancellation points
(m ~ 1e-12 built from ±1e-6 terms) blow up ulp counts while abs diffs stay
sub-ulp of the intermediate scale. Measured classes @ N=4096, K=10:
mojo↔replay dp ≤ 7.5e-9, dm ≤ 9.1e-13, dv ≤ 1.7e-19; replay↔torch dp 3.7e-9.
Bar: max|dp| ≤ 2e-8 (= 2e-4 of one lr=1e-4 step), dm ≤ 1e-11, dv ≤ 1e-16.
"""
import sys
import numpy as np
import torch

N, K = 4096, 10
LR, WD = np.float32(1e-4), np.float32(0.01)
B1, B2, EPS = np.float32(0.9), np.float32(0.999), np.float32(1e-8)
MASK = (1 << 64) - 1
BAR = {"p": 2e-8, "m": 1e-11, "v": 1e-16}


def splitmix64(state):
    z = (state + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def u01(seed):
    return np.float32(np.float64(splitmix64(seed) >> 40) * (1.0 / 16777216.0))


mojo = {}
for line in open(sys.argv[1]):
    if line.startswith("R "):
        _, reg, i, pb, mb, vb = line.split()
        mojo[(int(reg), int(i))] = tuple(
            np.array([int(x)], dtype=np.uint32).view(np.float32)[0]
            for x in (pb, mb, vb))
assert len(mojo) == 2 * N, f"expected {2*N} rows, got {len(mojo)}"

half, one = np.float32(0.5), np.float32(1.0)
ok = True
for regime in range(2):
    pscale = np.float32(0.02 if regime == 0 else 0.002)
    p0 = np.array([(u01(regime * 7919 + i) - half) * pscale
                   for i in range(N)], dtype=np.float32)
    grads = [np.array([(u01(regime * 104729 + t * 1299709 + i) - half)
                       * np.float32(2.4e-5) for i in range(N)], dtype=np.float32)
             for t in range(1, K + 1)]

    # (1) strict-f32 replay with the mojo op order
    p, m, v = p0.copy(), np.zeros(N, np.float32), np.zeros(N, np.float32)
    for t in range(1, K + 1):
        g = grads[t - 1]
        b1p = one
        b2p = one
        for _ in range(t):
            b1p = np.float32(b1p * B1)
            b2p = np.float32(b2p * B2)
        bc1, bc2 = np.float32(one - b1p), np.float32(one - b2p)
        m = np.float32(B1 * m) + np.float32((one - B1) * g)
        v = np.float32(B2 * v) + np.float32((one - B2) * g * g)
        mh, vh = np.float32(m / bc1), np.float32(v / bc2)
        p = np.float32(p * (one - LR * WD))
        p = np.float32(p - np.float32(LR * mh) / np.float32(np.sqrt(vh) + EPS))

    mp = np.array([mojo[(regime, i)][0] for i in range(N)], dtype=np.float32)
    mm = np.array([mojo[(regime, i)][1] for i in range(N)], dtype=np.float32)
    mv = np.array([mojo[(regime, i)][2] for i in range(N)], dtype=np.float32)
    diffs = {"p": np.abs(mp - p).max(), "m": np.abs(mm - m).max(),
             "v": np.abs(mv - v).max()}
    print(f"regime {regime} mojo vs same-op-order replay: "
          + "  ".join(f"max|d{k}|={diffs[k]:.2e}" for k in "pmv"))
    for k in "pmv":
        if diffs[k] > BAR[k]:
            print(f"  FAIL: d{k} {diffs[k]:.2e} > bar {BAR[k]:.0e}")
            ok = False

    # (2) informational: replay vs torch.optim
    pt = torch.nn.Parameter(torch.from_numpy(p0.copy()))
    opt = torch.optim.AdamW([pt], lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.01, foreach=False, fused=False)
    for t in range(1, K + 1):
        pt.grad = torch.from_numpy(grads[t - 1])
        opt.step()
    print(f"  (info) replay vs torch.optim: max|dp|="
          f"{np.abs(pt.detach().numpy() - p).max():.2e}")

print("PASS" if ok else "FAIL")
