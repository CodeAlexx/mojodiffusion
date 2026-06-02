#!/usr/bin/env python3
# gen_ops_reference.py — DEV-ONLY numpy oracle for Phase A op parity.
#
# NOT in the runtime path. Run with `pixi run python` to emit the expected
# outputs for linear() and rms_norm() given the SAME fixed-seed inputs the Mojo
# smoke driver (ops_smoke.mojo) generates. The Mojo driver inlines the same
# inputs and the reference values printed here; this script is how those
# reference numbers were produced and how they can be regenerated/verified.
#
# Inputs use a fixed numpy seed so they are reproducible. We print the inputs
# and the expected outputs as flat lists so they can be pasted/checked against
# the Mojo side.

import numpy as np

SEED = 1234
np.random.seed(SEED)

# ── linear: x[M, IN] @ w[OUT, IN]ᵀ + b[OUT] ──────────────────────────────────
M, IN, OUT = 3, 5, 4
x = np.random.randn(M, IN).astype(np.float32)
w = np.random.randn(OUT, IN).astype(np.float32)
b = np.random.randn(OUT).astype(np.float32)
y = x @ w.T + b  # [M, OUT]

# ── rms_norm over last dim: y = x / sqrt(mean(x^2)+eps) * g ───────────────────
R, D = 3, 6
EPS = 1e-6
xr = np.random.randn(R, D).astype(np.float32)
g = np.random.randn(D).astype(np.float32)
ms = np.mean(xr * xr, axis=-1, keepdims=True)  # [R,1]
yr = (xr / np.sqrt(ms + EPS)) * g  # [R, D]


def emit(name, arr):
    flat = arr.astype(np.float32).reshape(-1).tolist()
    print(f"# {name} shape={list(arr.shape)}")
    print(f"{name} = " + ", ".join(f"{v:.8f}" for v in flat))


print("# ===== linear inputs =====")
emit("x_lin", x)
emit("w_lin", w)
emit("b_lin", b)
print("# ===== linear expected =====")
emit("y_lin", y)
print()
print("# ===== rms_norm inputs =====")
emit("x_rms", xr)
emit("g_rms", g)
print("# ===== rms_norm expected =====")
emit("y_rms", yr)
