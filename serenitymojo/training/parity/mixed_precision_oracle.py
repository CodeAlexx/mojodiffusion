#!/usr/bin/env python3
# mixed_precision_oracle.py — PyTorch reference for ONE mixed-precision training
# step (BF16 compute + F32 master weights), the real DiT training dtype regime.
#
# This is the F32/torch ground truth for serenitymojo/training/parity/
# mixed_precision_parity.mojo. Python is a DEV-ONLY oracle (project convention).
#
# ── The mixed-precision step we are proving ──────────────────────────────────
# Master weights W, b live in F32. Each step:
#   1. cast master  -> BF16            (W_bf, b_bf, x_bf)
#   2. forward in BF16:  y = x_bf @ W_bf^T + b_bf     (F32-accumulated GEMM,
#                                                       BF16 storage at the I/O
#                                                       boundary — matches the
#                                                       Mojo linear BF16 path)
#   3. loss = mse(y, target)           (mean over numel)
#   4. backward in BF16: grads come out BF16
#         d_y  = 2*(y - target)/N
#         d_W  = d_y^T @ x   ; d_b = colsum(d_y)
#   5. cast grads -> F32
#   6. AdamW updates the F32 master (W, b) with F32 moments.
#
# We emit the UPDATED F32 master W, b. Because BF16 carries ~3 decimal digits,
# the gate is cos >= 0.99 (NOT f32-exact). To make the oracle faithful to the
# Mojo path (which rounds to BF16 at exactly the same points), this oracle ALSO
# rounds at those points: master->bf16 cast, forward y stored bf16, grads bf16.
# The GEMMs accumulate in F32 (matching cuBLAS / the Mojo F32-accumulated GEMM).
#
# Emits tagged space-separated float lines into mixed_precision_ref.txt:
#   mp_W   — updated F32 master weight after one mixed-precision step
#   mp_b   — updated F32 master bias  after one mixed-precision step
#   f32_W  — updated F32 master after the SAME step done entirely in F32
#   f32_b    (a diagnostic the Mojo side prints: the BF16 noise floor, which is
#            exactly what the cos>=0.99 — not 0.999 — gate budgets for)
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/training/parity/mixed_precision_oracle.py

import os
import numpy as np
import torch

OUT_PATH = "/home/alex/mojodiffusion/serenitymojo/training/parity/mixed_precision_ref.txt"

# ── problem dims (tiny; MUST match the .mojo aliases) ────────────────────────
M = 4        # rows of x (batch * tokens)
IN = 6       # in_features
OUT = 5      # out_features

# ── AdamW hyperparameters (MUST match the .mojo aliases) ─────────────────────
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WD = 0.01    # > 0 so decoupled-WD path is exercised
T = 1        # one step (1-based)


def fill(n, a, b, c, scale):
    """Deterministic closed-form fill (matches the Mojo driver _fill)."""
    out = np.empty(n, np.float64)
    for i in range(n):
        out[i] = (float((i * a) % b) - c) * scale
    return out


def emit(lines, tag, arr):
    lines.append(
        tag + " " + " ".join(f"{x:.8f}" for x in np.asarray(arr).reshape(-1).tolist())
    )


def to_bf16_round(x):
    """Round an F32 numpy array to BF16 and back to F32 (the storage round-trip
    a BF16 cast incurs). torch.bfloat16 gives the exact RNE rounding."""
    return torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).numpy()


def adamw_step_f32(p, g, m, v, t, lr, b1, b2, eps, wd):
    """One AdamW (decoupled-WD) step, F32 numpy. Mirrors training/optim.mojo
    _adamw_kernel exactly: moments in F32, bias correction 1-beta^t, decoupled
    weight decay applied to p before the adaptive Adam subtraction. Returns
    (p, m, v) updated."""
    p = p.astype(np.float32).copy()
    g = g.astype(np.float32)
    m = m.astype(np.float32).copy()
    v = v.astype(np.float32).copy()
    if wd > 0.0:
        p = p * (np.float32(1.0) - np.float32(lr) * np.float32(wd))
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * g * g
    bc1 = np.float32(1.0 - b1 ** t)
    bc2 = np.float32(1.0 - b2 ** t)
    mhat = m / bc1
    vhat = v / bc2
    p = p - np.float32(lr) * mhat / (np.sqrt(vhat) + np.float32(eps))
    return p, m, v


def forward_backward(W_flat, b_flat, x_flat, target_flat, bf16):
    """One forward + MSE + backward. If bf16=True, round W/b/x and the forward
    output y to BF16 at the storage boundaries (GEMM accumulates in F32), and
    round the grads to BF16 — the mixed-precision compute path. If bf16=False,
    everything stays F32 (the pure-F32 reference path).

    Returns (loss, d_W_flat_f32, d_b_flat_f32). In the bf16 case the returned
    grads are BF16 values upcast to F32 (the 'cast grads -> F32' step)."""
    if bf16:
        W = to_bf16_round(W_flat)
        b = to_bf16_round(b_flat)
        x = to_bf16_round(x_flat)
    else:
        W = W_flat.astype(np.float32)
        b = b_flat.astype(np.float32)
        x = x_flat.astype(np.float32)

    W = W.reshape(OUT, IN)
    x = x.reshape(M, IN)
    b = b.reshape(OUT)
    target = target_flat.astype(np.float32).reshape(M, OUT)

    # Forward: y[M,OUT] = x[M,IN] @ W[OUT,IN]^T + b, F32-accumulated.
    y = (x.astype(np.float32) @ W.astype(np.float32).T) + b.astype(np.float32)
    if bf16:
        y = to_bf16_round(y)   # BF16 storage of the linear output

    N = y.size
    diff = y.astype(np.float32) - target
    loss = float(np.mean(diff * diff))

    # Backward of mse->linear against the SAME (rounded) x, W used in the forward.
    d_y = (2.0 / N) * diff                       # F32
    if bf16:
        d_y = to_bf16_round(d_y)                 # mse_backward BF16 output
    # d_W[OUT,IN] = d_y[M,OUT]^T @ x[M,IN] ; d_b = colsum(d_y) — F32-accum GEMM.
    d_W = d_y.astype(np.float32).T @ x.astype(np.float32)
    d_b = d_y.astype(np.float32).sum(axis=0)
    if bf16:
        d_W = to_bf16_round(d_W)                 # linear_backward BF16 grads
        d_b = to_bf16_round(d_b)
    # cast grads -> F32 (the master-update input dtype); return flat.
    return loss, d_W.astype(np.float32).reshape(-1), d_b.astype(np.float32).reshape(-1)


def main():
    # Deterministic master weights / inputs (F32). Same fills the Mojo side uses.
    W0 = fill(OUT * IN, 7, 13, 6.0, 0.05).astype(np.float32)
    b0 = fill(OUT, 5, 11, 5.0, 0.05).astype(np.float32)
    x0 = fill(M * IN, 3, 9, 4.0, 0.10).astype(np.float32)
    target0 = fill(M * OUT, 2, 7, 3.0, 0.10).astype(np.float32)

    # zero AdamW moments (F32)
    mW = np.zeros(OUT * IN, np.float32)
    vW = np.zeros(OUT * IN, np.float32)
    mb = np.zeros(OUT, np.float32)
    vb = np.zeros(OUT, np.float32)

    # ── MIXED-PRECISION step (BF16 compute, F32 master) ──────────────────────
    loss_mp, dW_mp, db_mp = forward_backward(W0, b0, x0, target0, bf16=True)
    Wp_mp, _, _ = adamw_step_f32(W0, dW_mp, mW, vW, T, LR, BETA1, BETA2, EPS, WD)
    bp_mp, _, _ = adamw_step_f32(b0, db_mp, mb, vb, T, LR, BETA1, BETA2, EPS, WD)

    # ── PURE-F32 reference step (no BF16 rounding anywhere) ───────────────────
    loss_f32, dW_f32, db_f32 = forward_backward(W0, b0, x0, target0, bf16=False)
    Wp_f32, _, _ = adamw_step_f32(W0, dW_f32, mW, vW, T, LR, BETA1, BETA2, EPS, WD)
    bp_f32, _, _ = adamw_step_f32(b0, db_f32, mb, vb, T, LR, BETA1, BETA2, EPS, WD)

    lines = []
    emit(lines, "mp_W", Wp_mp)
    emit(lines, "mp_b", bp_mp)
    emit(lines, "f32_W", Wp_f32)
    emit(lines, "f32_b", bp_f32)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    def cos(a, b):
        a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    print("wrote", OUT_PATH)
    print(f"M={M} IN={IN} OUT={OUT}  loss_mp={loss_mp:.6f}  loss_f32={loss_f32:.6f}")
    print(f"cos(mp_W, f32_W) = {cos(Wp_mp, Wp_f32):.8f}")
    print(f"cos(mp_b, f32_b) = {cos(bp_mp, bp_f32):.8f}")


if __name__ == "__main__":
    main()
