#!/usr/bin/env python3
# opt_stableadamw_oracle.py — reference for opt_stableadamw.mojo.
#
# Replicates the inline F32 scalar reference in optimizers.rs's
# `stable_adamw_5_steps_matches_reference` test EXACTLY (which itself mirrors
# pytorch_optimizer.StableAdamW.step verbatim). debias_beta in F64; the rest
# in F32 to match the Rust reference's accumulation. Python = DEV-ONLY oracle.
#
# Two cases (each `steps` runs from the deterministic init, fresh grad/step):
#   sadamw_p1 — param after 1 step  (t=1: beta1_comp=1, beta2_hat=0 raw-grad)
#   sadamw_p5 — param after 5 steps (debias_beta trajectory + rms clipping)
# wd > 0 so the decoupled weight decay is exercised.
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/training/parity/opt_stableadamw_oracle.py

import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "opt_stableadamw_ref.txt")

LR = 1e-2
BETA1 = 0.9
BETA2 = 0.99
EPS = 1e-8
WD = 0.01
N = 64


def fill(n, a, b, c, scale):
    out = np.empty(n, np.float32)
    for i in range(n):
        out[i] = np.float32((float((i * a) % b) - c) * scale)
    return out


def emit(lines, tag, arr):
    lines.append(tag + " " + " ".join(f"{x:.8f}" for x in np.asarray(arr).reshape(-1).tolist()))


def db(beta, t):
    # F64 (matches Rust debias_beta), returns f32-rounded value like the Rust.
    b = float(beta)
    bn = b ** t
    return np.float32((bn - b) / (bn - 1.0))


def sadamw_run(steps):
    p = fill(N, 7, 13, 6.0, 0.05).astype(np.float32)
    g = fill(N, 5, 11, 5.0, 0.05).astype(np.float32)
    m = np.zeros(N, np.float32)
    v = np.zeros(N, np.float32)
    eps_p2 = np.float32(EPS * EPS)
    for t in range(1, steps + 1):
        beta1_hat = db(BETA1, t)
        beta2_hat = db(BETA2, t)
        beta1_comp = np.float32(1.0 - beta1_hat)
        # exp_avg.lerp_(grad, weight=beta1_comp)
        m = (m * (np.float32(1.0) - beta1_comp) + g * beta1_comp).astype(np.float32)
        # exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, 1-beta2_hat)
        v = (v * beta2_hat + g * g * (np.float32(1.0) - beta2_hat)).astype(np.float32)
        # rms = sqrt(mean(g^2 / max(v, eps^2))).max(1)
        vc = np.maximum(v, eps_p2)
        rms_inner = np.float32(np.sum((g * g / vc).astype(np.float32)) / np.float32(N))
        rms_inner = np.float32(max(float(rms_inner), 0.0))
        rms = np.float32(max(float(np.sqrt(rms_inner)), 1.0))
        lr_eff = np.float32(LR / rms)
        if WD != 0.0:
            p = (p * (np.float32(1.0) - np.float32(WD) * lr_eff)).astype(np.float32)
        denom = (np.sqrt(v) + np.float32(EPS)).astype(np.float32)
        p = (p - lr_eff * m / denom).astype(np.float32)
    return p


def main():
    lines = []
    emit(lines, "sadamw_p1", sadamw_run(1))
    emit(lines, "sadamw_p5", sadamw_run(5))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT, " N =", N)


if __name__ == "__main__":
    main()
