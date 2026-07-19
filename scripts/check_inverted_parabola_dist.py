#!/usr/bin/env python3
"""Gate the Mojo INVERTED_PARABOLA timestep sampler against SerenityTrainer's math.

Reads the Mojo probe's stdout (WEIGHTS block + HIST block) and checks:
  1. weight table == SerenityTrainer's torch-computed weights (verbatim port of
     ModelSetupNoiseMixin._get_timestep_discrete, f64 tolerance)
  2. 200k-draw histogram matches the exact normalized PMF (cos >= 0.999)

Usage:
  pixi run mojo run -I . serenitymojo/training/timestep_inverted_parabola_smoke.mojo \
    | python3 scripts/check_inverted_parabola_dist.py
"""

import math
import sys

N = 1000
WEIGHT = 7.7
BIAS = 0.0
SHIFT = 1.0


def serenity_weights():
    # Verbatim SerenityTrainer 423c3b36 ModelSetupNoiseMixin lines (torch f32 —
    # torch.linspace defaults to float32, so the true reference trainer table is f32-computed;
    # the Mojo port computes in f64 from an f32 weight, hence the ~1e-7 bar).
    import torch

    linspace = torch.linspace(0, 1, N)
    linspace = linspace / (SHIFT - SHIFT * linspace + linspace)
    linspace_derivative = torch.linspace(0, 1, N)
    linspace_derivative = SHIFT / (SHIFT + linspace_derivative - (linspace_derivative * SHIFT)).pow(2)
    bias = BIAS + 0.5
    weights = torch.clamp(-WEIGHT * ((linspace - bias) ** 2) + 2, min=0.0)
    weights *= linspace_derivative
    return [float(w) for w in weights]


def main():
    lines = [ln.strip() for ln in sys.stdin if ln.strip()]
    wi = lines.index(next(l for l in lines if l.startswith("WEIGHTS")))
    hi = lines.index(next(l for l in lines if l.startswith("HIST")))
    mojo_w = [float(x) for x in lines[wi + 1 : wi + 1 + N]]
    draws = int(lines[hi].split()[1])
    hist = [int(x) for x in lines[hi + 1 : hi + 1 + N]]
    assert len(mojo_w) == N and len(hist) == N and sum(hist) == draws

    ref_w = serenity_weights()
    max_abs = max(abs(a - b) for a, b in zip(mojo_w, ref_w))
    print(f"[inv-parabola] weight table max_abs vs SerenityTrainer torch-f32: {max_abs:.3e}")
    if max_abs > 1e-6:
        print("[inv-parabola] FAIL: weight table diverges from SerenityTrainer")
        return 1

    total = sum(ref_w)
    pmf = [w / total for w in ref_w]
    emp = [h / draws for h in hist]
    dot = sum(a * b for a, b in zip(pmf, emp))
    na = math.sqrt(sum(a * a for a in pmf))
    nb = math.sqrt(sum(b * b for b in emp))
    cos = dot / (na * nb)
    zero_leak = sum(hist[i] for i in range(N) if pmf[i] == 0.0)
    print(f"[inv-parabola] histogram-vs-PMF cos: {cos:.6f}  (draws={draws})")
    print(f"[inv-parabola] draws in zero-weight buckets: {zero_leak}")
    if cos < 0.999 or zero_leak > 0:
        print("[inv-parabola] FAIL")
        return 1
    print("[inv-parabola] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
