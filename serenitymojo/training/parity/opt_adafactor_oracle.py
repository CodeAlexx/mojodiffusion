#!/usr/bin/env python3
# opt_adafactor_oracle.py — reference for opt_adafactor.mojo.
#
# Replicates the inline F32 reference in optimizers.rs's
# `adafactor_5_steps_matches_inline_reference` (~line 2197) EXACTLY, plus a 1D
# (per-element) case and a scale_parameter=True case. All F32 (matches the Rust
# inline reference + Adafactor::step F32 math). Python = DEV-ONLY oracle.
#
# Tags:
#   af_fac_p5    — 4x4 factored param after 5 steps, wd=0, scale_parameter=False
#   af_elem_p5   — 16-elem 1D param after 5 steps (per-element second moment)
#   af_scale_p1  — 4x4 param after 1 step with scale_parameter=True, wd=0
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/training/parity/opt_adafactor_oracle.py

import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "opt_adafactor_ref.txt")

LR = 1e-3
EPS = 1e-3
DECAY_RATE = -0.8
CLIP = 1.0
EPS_GRAD = 1e-30


def emit(lines, tag, arr):
    lines.append(tag + " " + " ".join(f"{x:.8f}" for x in np.asarray(arr).reshape(-1).tolist()))


def f32(x):
    return np.float32(x)


def rms(arr):
    a = np.asarray(arr, np.float32)
    return f32(np.sqrt(np.float32(np.sum((a * a).astype(np.float32)) / f32(a.size))))


def af_factored(steps, scale_parameter, wd):
    R, C = 4, 4
    n = R * C
    p = np.array([0.1 + i * 0.01 for i in range(n)], np.float32)
    g = np.array([0.05 - i * 0.003 for i in range(n)], np.float32)
    row = np.zeros(R, np.float32)
    col = np.zeros(C, np.float32)
    for t in range(1, steps + 1):
        beta2t = f32(1.0) - f32(np.float32(t) ** np.float32(DECAY_RATE))
        one_m = f32(1.0) - beta2t
        g_sq = (g * g + np.float32(EPS_GRAD)).astype(np.float32)
        mean_last = np.array([np.float32(np.sum(g_sq[r * C:(r + 1) * C]) / f32(C)) for r in range(R)], np.float32)
        mean_second = np.array([np.float32(np.sum(g_sq[c::C]) / f32(R)) for c in range(C)], np.float32)
        row = (beta2t * row + one_m * mean_last).astype(np.float32)
        col = (beta2t * col + one_m * mean_second).astype(np.float32)
        row_mean = f32(np.sum(row) / f32(R))
        r_factor = (f32(1.0) / np.sqrt(row / row_mean)).astype(np.float32)
        c_factor = (f32(1.0) / np.sqrt(col)).astype(np.float32)
        update = np.array([r_factor[i // C] * c_factor[i % C] * g[i] for i in range(n)], np.float32)
        r = rms(update)
        scale_div = max(float(r / np.float32(CLIP)), 1.0)
        update = (update / np.float32(scale_div)).astype(np.float32)
        lr_eff = np.float32(LR)
        if scale_parameter:
            p_rms = max(float(rms(p)), float(EPS))
            lr_eff = np.float32(LR * p_rms)
        update = (update * lr_eff).astype(np.float32)
        if wd != 0.0:
            p = (p * (f32(1.0) - np.float32(wd) * lr_eff)).astype(np.float32)
        p = (p - update).astype(np.float32)
    return p


def af_factored_nd(steps, L, R, C, scale_parameter, wd):
    # Replicates the Rust mean_dim semantics for rank>=3: leading dims [L] kept
    # separate, row_mean reduced only over R within each L-block.
    n = L * R * C
    p = np.array([0.1 + i * 0.011 for i in range(n)], np.float32)
    g = np.array([0.05 - i * 0.0021 for i in range(n)], np.float32)
    row = np.zeros(L * R, np.float32)
    col = np.zeros(L * C, np.float32)
    for t in range(1, steps + 1):
        beta2t = f32(1.0) - f32(np.float32(t) ** np.float32(DECAY_RATE))
        one_m = f32(1.0) - beta2t
        g_sq = (g * g + np.float32(EPS_GRAD)).astype(np.float32)
        update = np.zeros(n, np.float32)
        for l in range(L):
            base = l * R * C
            mean_last = np.array(
                [np.float32(np.sum(g_sq[base + r * C: base + (r + 1) * C]) / f32(C)) for r in range(R)],
                np.float32,
            )
            mean_second = np.array(
                [np.float32(np.sum(g_sq[base + c: base + R * C: C]) / f32(R)) for c in range(C)],
                np.float32,
            )
            rb = l * R
            cb = l * C
            row[rb:rb + R] = (beta2t * row[rb:rb + R] + one_m * mean_last).astype(np.float32)
            col[cb:cb + C] = (beta2t * col[cb:cb + C] + one_m * mean_second).astype(np.float32)
            row_mean = f32(np.sum(row[rb:rb + R]) / f32(R))
            r_factor = (f32(1.0) / np.sqrt(row[rb:rb + R] / row_mean)).astype(np.float32)
            c_factor = (f32(1.0) / np.sqrt(col[cb:cb + C])).astype(np.float32)
            for r in range(R):
                for c in range(C):
                    update[base + r * C + c] = np.float32(
                        r_factor[r] * c_factor[c] * g[base + r * C + c]
                    )
        rr = rms(update)
        scale_div = max(float(rr / np.float32(CLIP)), 1.0)
        update = (update / np.float32(scale_div)).astype(np.float32)
        lr_eff = np.float32(LR)
        if scale_parameter:
            p_rms = max(float(rms(p)), float(EPS))
            lr_eff = np.float32(LR * p_rms)
        update = (update * lr_eff).astype(np.float32)
        if wd != 0.0:
            p = (p * (f32(1.0) - np.float32(wd) * lr_eff)).astype(np.float32)
        p = (p - update).astype(np.float32)
    return p


def af_elementwise(steps, wd):
    n = 16
    p = np.array([0.1 + i * 0.01 for i in range(n)], np.float32)
    g = np.array([0.05 - i * 0.003 for i in range(n)], np.float32)
    v = np.zeros(n, np.float32)
    for t in range(1, steps + 1):
        beta2t = f32(1.0) - f32(np.float32(t) ** np.float32(DECAY_RATE))
        one_m = f32(1.0) - beta2t
        g_sq = (g * g + np.float32(EPS_GRAD)).astype(np.float32)
        v = (beta2t * v + one_m * g_sq).astype(np.float32)
        update = ((f32(1.0) / np.sqrt(v)) * g).astype(np.float32)
        r = rms(update)
        scale_div = max(float(r / np.float32(CLIP)), 1.0)
        update = (update / np.float32(scale_div)).astype(np.float32)
        lr_eff = np.float32(LR)
        update = (update * lr_eff).astype(np.float32)
        if wd != 0.0:
            p = (p * (f32(1.0) - np.float32(wd) * lr_eff)).astype(np.float32)
        p = (p - update).astype(np.float32)
    return p


def main():
    lines = []
    emit(lines, "af_fac_p5", af_factored(5, False, 0.0))
    emit(lines, "af_elem_p5", af_elementwise(5, 0.0))
    emit(lines, "af_scale_p1", af_factored(1, True, 0.0))
    # rank-3 [L=2,R=3,C=4] factored — exercises per-L-block row_mean (the fix)
    emit(lines, "af_nd_p5", af_factored_nd(5, 2, 3, 4, False, 0.0))
    # wd>0 coverage (decoupled WD branch): rank-2 4x4, 5 steps, wd=0.05
    emit(lines, "af_wd_p5", af_factored(5, False, 0.05))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
