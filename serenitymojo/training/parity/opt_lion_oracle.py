#!/usr/bin/env python3
# opt_lion_oracle.py — reference for serenitymojo/training/opt_lion.mojo.
#
# Replicates EriDiffusion-v2 optimizers.rs::Lion::step EXACTLY (~line 1083) in
# F64 numpy. The Lion algorithm (Chen et al., 2023):
#   c   = beta1*m + (1-beta1)*g ; dir = sign(c)
#   m   = beta2*m + (1-beta2)*g                       # from OLD m
#   if wd != 0:  p *= (1 - wd*lr)                      # decoupled, BEFORE step
#   p   = p - lr*dir
# This matches torch's Lion / pytorch_optimizer.Lion sign-update. Python is a
# DEV-ONLY oracle (project convention). Emits tagged float lines to opt_lion_ref.txt:
#   lion_p1  — param after 1 step (wd>0 so decoupled WD is exercised)
#   lion_p5  — param after 5 steps (momentum EMA accumulation)
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/training/parity/opt_lion_oracle.py

import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "opt_lion_ref.txt")

LR = 0.01
BETA1 = 0.9
BETA2 = 0.99
WD = 0.1          # > 0 so decoupled WD is load-bearing in the gate
N = 64


def fill(n, a, b, c, scale):
    out = np.empty(n, np.float64)
    for i in range(n):
        out[i] = (float((i * a) % b) - c) * scale
    return out


def emit(lines, tag, arr):
    lines.append(tag + " " + " ".join(f"{x:.8f}" for x in np.asarray(arr).reshape(-1).tolist()))


def lion_run(steps):
    p = fill(N, 7, 13, 6.0, 0.05).copy()
    g = fill(N, 5, 11, 5.0, 0.05)          # same grad each step (deterministic)
    m = np.zeros(N, np.float64)
    for _ in range(steps):
        c = BETA1 * m + (1.0 - BETA1) * g
        direction = np.sign(c)             # sign(0)=0, matches kernel
        m = BETA2 * m + (1.0 - BETA2) * g  # from OLD m
        if WD != 0.0:
            p = p * (1.0 - WD * LR)
        p = p - LR * direction
    return p


def main():
    lines = []
    emit(lines, "lion_p1", lion_run(1))
    emit(lines, "lion_p5", lion_run(5))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT, " N =", N)


if __name__ == "__main__":
    main()
