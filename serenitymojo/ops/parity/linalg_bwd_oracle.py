#!/usr/bin/env python3
# linalg_bwd_oracle.py — PyTorch reference for the GEMM-family backward kernels
# (serenitymojo/ops/linalg_backward.mojo: matmul, bmm, linear, addbias).
#
# Tier 2 of FULL_PORT_TRAINING_PLAN.md §3. Oracle = PyTorch autograd.grad (stable
# ground-truth math). Python is a DEV-ONLY oracle (parity convention).
#
# Emits tagged space-separated F32 lines to linalg_bwd_ref.txt, one per gated
# grad. The Mojo driver (linalg_bwd_parity.mojo) reproduces the SAME deterministic
# input fills on-device and reads back ONLY the reference grads.
#
# Tags: matmul_da, matmul_db, bmm_da, bmm_db, linear_dx, linear_dw, linear_db,
#       addbias_db.
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/linalg_bwd_oracle.py

import os
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linalg_bwd_ref.txt")


# ── Deterministic fills — MUST match linalg_bwd_parity.mojo _fill_* exactly. ──
def fill(n, a, m, sub):
    """v[i] = (((i*a) % m) - sub) * 0.05  as F64."""
    return torch.tensor(
        [(float((i * a) % m) - sub) * 0.05 for i in range(n)], dtype=torch.float64
    )


def emit(lines, tag, t):
    arr = t.detach().reshape(-1).double().tolist()
    lines.append(tag + " " + " ".join(f"{x:.8f}" for x in arr))


def main():
    lines = []

    # ── matmul: C[M,N] = A[M,K] @ B[K,N] ─────────────────────────────────────
    M, N, K = 6, 5, 7
    a = fill(M * K, 7, 13, 6.0).reshape(M, K).requires_grad_(True)
    b = fill(K * N, 5, 11, 5.0).reshape(K, N).requires_grad_(True)
    gc = fill(M * N, 3, 9, 4.0).reshape(M, N)
    c = a @ b
    da, db = torch.autograd.grad(c, [a, b], grad_outputs=gc)
    emit(lines, "matmul_da", da)
    emit(lines, "matmul_db", db)

    # ── bmm: C[Bt,M,N] = A[Bt,M,K] @ B[Bt,K,N] ───────────────────────────────
    Bt = 3
    ba = fill(Bt * M * K, 7, 13, 6.0).reshape(Bt, M, K).requires_grad_(True)
    bb = fill(Bt * K * N, 5, 11, 5.0).reshape(Bt, K, N).requires_grad_(True)
    bgc = fill(Bt * M * N, 3, 9, 4.0).reshape(Bt, M, N)
    bc = torch.bmm(ba, bb)
    bda, bdb = torch.autograd.grad(bc, [ba, bb], grad_outputs=bgc)
    emit(lines, "bmm_da", bda)
    emit(lines, "bmm_db", bdb)

    # ── linear: y = x @ Wᵀ + b ───────────────────────────────────────────────
    Mf, inf, outf = 4, 7, 5
    x = fill(Mf * inf, 7, 13, 6.0).reshape(Mf, inf).requires_grad_(True)
    w = fill(outf * inf, 5, 11, 5.0).reshape(outf, inf).requires_grad_(True)
    bias = fill(outf, 3, 9, 4.0).requires_grad_(True)
    gy = fill(Mf * outf, 2, 7, 3.0).reshape(Mf, outf)
    y = torch.nn.functional.linear(x, w, bias)
    dx, dw, dbias = torch.autograd.grad(y, [x, w, bias], grad_outputs=gy)
    emit(lines, "linear_dx", dx)
    emit(lines, "linear_dw", dw)
    emit(lines, "linear_db", dbias)

    # ── addbias: y = x + b (broadcast over the M rows) ───────────────────────
    Ma, outa = 4, 5
    xa = fill(Ma * outa, 7, 13, 6.0).reshape(Ma, outa).requires_grad_(True)
    ba2 = fill(outa, 3, 9, 4.0).requires_grad_(True)
    gya = fill(Ma * outa, 2, 7, 3.0).reshape(Ma, outa)
    ya = xa + ba2
    (dba,) = torch.autograd.grad(ya, [ba2], grad_outputs=gya)
    emit(lines, "addbias_db", dba)

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
