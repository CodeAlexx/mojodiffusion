#!/usr/bin/env python3
# opt_prodigy_oracle.py — reference for opt_prodigy.mojo.
#
# Replicates optimizers.rs::Prodigy::step EXACTLY (~line 853): single-param
# scope, decouple=True, safeguard_warmup=False, use_bias_correction=False,
# d0=1e-6, d_coef=1, growth_rate=INF. d_numerator in F64; the rest F32 to match
# the Rust F32 tensor math. Python = DEV-ONLY oracle.
#
# Drives the strongly-convex quadratic f(x)=0.5 x^T A x, A=diag(2,3,5), grad=A@x
# (the exact convergence test in optimizers.rs::prodigy_minimizes_quadratic).
# Emits the param trajectory at fixed step counts so the Mojo driver can match
# the FULL D-adaptation path, not just a static formula.
#
# Tags:
#   prodigy_x10  — x after 10 steps
#   prodigy_x50  — x after 50 steps
#   prodigy_x200 — x after 200 steps (||x|| should be < 0.1, convergence)
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/training/parity/opt_prodigy_oracle.py

import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "opt_prodigy_ref.txt")

LR = 1.0
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WD = 0.0
D0 = 1e-6
D_COEF = 1.0
F32_EPS = np.float32(1.1920929e-07)
A_DIAG = np.array([2.0, 3.0, 5.0], np.float32)


def f32(x):
    return np.float32(x)


def emit(lines, tag, arr):
    lines.append(tag + " " + " ".join(f"{x:.8f}" for x in np.asarray(arr).reshape(-1).tolist()))


class Prodigy:
    def __init__(self):
        self.d = f32(D0)
        self.d_max = f32(D0)
        self.d_numerator = float(0.0)   # F64
        self.t = 0
        self.m = None
        self.v = None
        self.s = None
        self.p0 = None

    def step(self, p, g):
        n = p.size
        if self.m is None:
            self.m = np.zeros(n, np.float32)
            self.v = np.zeros(n, np.float32)
            self.s = np.zeros(n, np.float32)
            self.p0 = p.copy().astype(np.float32)
        self.t += 1
        beta3 = f32(np.sqrt(f32(BETA2)))
        d = self.d
        lr = f32(LR)
        dlr = f32(d * lr)
        self.d_numerator *= float(beta3)
        inner = f32(0.0)
        for i in range(n):
            inner = f32(inner + g[i] * (self.p0[i] - p[i]))
        delta_numerator = (float(d) / float(D0)) * float(dlr) * float(inner)
        s_alpha = f32((d / f32(D0)) * dlr)
        d_denom = float(0.0)
        for i in range(n):
            self.m[i] = f32(BETA1) * self.m[i] + (f32(1.0) - f32(BETA1)) * d * g[i]
            self.v[i] = f32(BETA2) * self.v[i] + (f32(1.0) - f32(BETA2)) * d * d * g[i] * g[i]
            self.s[i] = beta3 * self.s[i] + s_alpha * g[i]
            d_denom += float(abs(self.s[i]))
        if d_denom > 0.0 and lr > 0.0:
            self.d_numerator += delta_numerator
            d_hat = (D_COEF * self.d_numerator) / d_denom
            d_hat_f32 = f32(d_hat)
            if abs(self.d - f32(D0)) < F32_EPS:
                self.d = max(self.d, d_hat_f32)
            self.d_max = max(self.d_max, d_hat_f32)
            self.d = self.d_max          # growth_rate=INF
        else:
            if beta3 > 0.0:
                self.d_numerator /= float(beta3)
        d = self.d
        dlr = f32(d * lr)
        for i in range(n):
            denom = f32(np.sqrt(self.v[i]) + d * f32(EPS))
            if WD != 0.0:
                p[i] = p[i] * (f32(1.0) - f32(WD) * dlr)
            p[i] = p[i] - dlr * self.m[i] / denom


def run(steps, wd=0.0):
    global WD
    saved = WD
    WD = wd
    try:
        x = np.array([1.0, -0.7, 0.4], np.float32)
        opt = Prodigy()
        for _ in range(steps):
            g = (A_DIAG * x).astype(np.float32)
            opt.step(x, g)
        return x
    finally:
        WD = saved


def main():
    lines = []
    emit(lines, "prodigy_x10", run(10))
    emit(lines, "prodigy_x50", run(50))
    x200 = run(200)
    emit(lines, "prodigy_x200", x200)
    # wd>0 decoupled-WD branch coverage (5 steps, wd=0.05)
    emit(lines, "prodigy_wd5", run(5, wd=0.05))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    norm = float(np.sqrt(np.sum(x200 * x200)))
    print("wrote", OUT, " ||x200|| =", norm)


if __name__ == "__main__":
    main()
