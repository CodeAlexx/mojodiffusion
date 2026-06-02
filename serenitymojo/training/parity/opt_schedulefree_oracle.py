#!/usr/bin/env python3
# opt_schedulefree_oracle.py — reference for opt_schedulefree.mojo.
#
# Replicates optimizers.rs::RAdamScheduleFree::step + the inline reference in
# radam_schedulefree_5_steps_matches_reference (~line 2549) EXACTLY. Scalar
# schedule (n_sma, rt, ckpt, lr_max, weight_sum) in F64; per-element y/z/v in
# F32. silent_sgd_phase=True, r=0, weight_lr_power=2. Python = DEV-ONLY oracle.
#
# Tags:
#   sf_y5   — y-sequence param after 5 train steps (the trajectory pinned by the
#             Rust test; n_sma crosses 4 so the denom branch is exercised)
#   sf_eval — x = y*(1/beta1) + z*(1-1/beta1) after 5 steps then enter_eval_mode
#
# init=[0.5,-0.2,0.7,-0.1], grad=[0.1,-0.05,0.2,-0.1], lr=2.5e-3,
# beta1=0.9, beta2=0.999, eps=1e-8, wd=0 (matches the Rust test exactly).
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/training/parity/opt_schedulefree_oracle.py

import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "opt_schedulefree_ref.txt")

LR = 2.5e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WD = 0.0
R_POW = 0.0
WEIGHT_LR_POWER = 2.0
SILENT_SGD_PHASE = True


def f32(x):
    return np.float32(x)


def emit(lines, tag, arr):
    lines.append(tag + " " + " ".join(f"{x:.8f}" for x in np.asarray(arr).reshape(-1).tolist()))


def run(steps, wd=None):
    global WD
    saved = WD
    if wd is not None:
        WD = wd
    init = np.array([0.5, -0.2, 0.7, -0.1], np.float32)
    grad = np.array([0.1, -0.05, 0.2, -0.1], np.float32)
    p = init.copy().astype(np.float32)
    z = init.copy().astype(np.float32)
    v = np.zeros(4, np.float32)
    lr_max = f32(-1.0)
    weight_sum = float(0.0)
    for t in range(1, steps + 1):
        beta2_pow = float(BETA2) ** t                     # F64
        bias_correction2 = f32(1.0 - beta2_pow)
        n_sma_max = 2.0 / (1.0 - float(BETA2)) - 1.0
        one_minus_b2t = 1.0 - beta2_pow
        n_sma = n_sma_max - 2.0 * t * beta2_pow / one_minus_b2t
        if n_sma >= 4.0:
            rt = np.sqrt(one_minus_b2t * (n_sma - 4.0) / (n_sma_max - 4.0)
                         * (n_sma - 2.0) / n_sma * n_sma_max / (n_sma_max - 2.0))
        else:
            rt = -1.0
        lr_t = f32(LR) * f32(rt)
        if lr_t < 0.0:
            lr_t = f32(0.0) if SILENT_SGD_PHASE else f32(1.0)
        lr_max = max(lr_max, lr_t)
        weight = (float(t) ** R_POW) * (float(lr_max) ** WEIGHT_LR_POWER)
        weight_sum += weight
        checkpoint = f32(weight / weight_sum) if weight_sum != 0.0 else f32(0.0)
        adaptive_y_lr = lr_t * (f32(BETA1) * (f32(1.0) - checkpoint) - f32(1.0))
        for i in range(4):
            v[i] = v[i] * f32(BETA2) + grad[i] * grad[i] * (f32(1.0) - f32(BETA2))
        if n_sma > 4.0:
            grad_eff = np.array([grad[i] / (np.sqrt(v[i]) / bias_correction2 + f32(EPS))
                                 for i in range(4)], np.float32)
        else:
            grad_eff = grad.copy().astype(np.float32)
        if WD > 0.0:
            grad_eff = (grad_eff + f32(WD) * p).astype(np.float32)
        for i in range(4):
            p[i] = p[i] * (f32(1.0) - checkpoint) + z[i] * checkpoint
            p[i] = p[i] + grad_eff[i] * adaptive_y_lr
            z[i] = z[i] - grad_eff[i] * lr_t
    WD = saved
    return p, z


def main():
    lines = []
    y5, z5 = run(5)
    emit(lines, "sf_y5", y5)
    # eval: x = y*(1/beta1) + z*(1 - 1/beta1)
    inv_beta1 = f32(1.0 / BETA1)
    one_minus_inv = f32(1.0) - inv_beta1
    x = (y5 * inv_beta1 + z5 * one_minus_inv).astype(np.float32)
    emit(lines, "sf_eval", x)
    # wd>0 coupled-L2 branch coverage (5 steps, wd=0.05)
    ywd5, _ = run(5, wd=0.05)
    emit(lines, "sf_wd5", ywd5)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
