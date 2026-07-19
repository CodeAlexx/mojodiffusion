#!/usr/bin/env python3
# check_ltx2_noise_spatial.py — SPATIAL-structure gate for the ltx2 driver
# noise (the krea2-letterbox class: noise that passes marginal stats but
# carries spatial correlation that biases composition).
#
# Reads /tmp/ltx2_noise_spatial_dump.safetensors (from
# ltx2_noise_spatial_dump.mojo) and compares, against torch.randn references
# of identical shape:
#   1. lag-1 autocorrelation along each spatial/temporal axis (F, H, W) and
#      along the CHANNEL axis, per step, averaged;
#   2. flat-index lag-1 autocorrelation (stream order);
#   3. 2D spatial power-spectrum flatness (per-channel mean spectrum ratio
#      of low-frequency quadrant energy to total — letterbox-class bias shows
#      up as low-frequency excess).
# PASS bar: every |lag-1 corr| within the torch reference's own band
# (mean ± 6 sigma over 64 torch trials), spectrum low-freq ratio within the
# same band. Prints every number.
import sys

import numpy as np
import torch
from safetensors import safe_open


def lag1(x: np.ndarray, axis: int) -> float:
    a = np.moveaxis(x, axis, 0)
    a = a.reshape(a.shape[0], -1)
    u, v = a[:-1].ravel(), a[1:].ravel()
    u = u - u.mean()
    v = v - v.mean()
    return float((u * v).mean() / (u.std() * v.std() + 1e-12))


def lowfreq_ratio(x: np.ndarray) -> float:
    # x: [C, F, H, W] — mean over C,F of per-slice 2D spectra
    c, f, h, w = x.shape
    spec = np.zeros((h, w))
    for ci in range(c):
        for fi in range(f):
            s = np.abs(np.fft.fft2(x[ci, fi])) ** 2
            spec += np.fft.fftshift(s)
    ch, cw = h // 2, w // 2
    qh, qw = max(1, h // 4), max(1, w // 4)
    low = spec[ch - qh:ch + qh + 1, cw - qw:cw + qw + 1].sum()
    return float(low / spec.sum())


def stats(x: np.ndarray) -> dict:
    x4 = x[0]  # [C,F,H,W]
    return {
        "corr_C": lag1(x4, 0),
        "corr_F": lag1(x4, 1) if x4.shape[1] > 1 else 0.0,
        "corr_H": lag1(x4, 2),
        "corr_W": lag1(x4, 3),
        "corr_flat": lag1(x4.reshape(1, -1), 1),
        "lowfreq": lowfreq_ratio(x4),
    }


def main() -> int:
    f = safe_open("/tmp/ltx2_noise_spatial_dump.safetensors", framework="np")
    ok = True
    for geom, shape in (("video", (1, 128, 4, 9, 16)), ("image", (1, 128, 1, 16, 16))):
        ours = [stats(f.get_tensor(f"{geom}_step{s}")) for s in range(1, 9)]
        g = torch.Generator().manual_seed(1234)
        refs = [stats(torch.randn(shape, generator=g).numpy()) for _ in range(64)]
        print(f"=== {geom} {shape} ===")
        for key in ours[0]:
            ov = np.mean([o[key] for o in ours])
            rm = np.mean([r[key] for r in refs])
            rs = np.std([r[key] for r in refs]) + 1e-12
            z = (ov - rm) / rs
            verdict = "PASS" if abs(z) <= 6.0 else "FAIL"
            if verdict == "FAIL":
                ok = False
            print(f"  {key:10s} ours={ov:+.5f} torch={rm:+.5f}±{rs:.5f} z={z:+.2f} {verdict}")
    print("NOISE SPATIAL GATE:", "PASS" if ok else "FAIL — letterbox-class structure present")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
