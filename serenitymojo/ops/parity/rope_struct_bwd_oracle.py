#!/usr/bin/env python3
# rope_struct_bwd_oracle.py — PyTorch reference for the BACKWARD of three DiT
# structural primitives (serenitymojo/ops/rope_struct_backward.mojo):
#   RoPePrecomputed   -> tag rope_dx  (interleaved variant, FLUX/Klein)
#   QkvSplitPermute   -> tag qkv_dqkv
#   GateResidual      -> tags gate_dx, gate_dg, gate_dy
#
# Oracle = PyTorch autograd, F64 throughout (Mojo path is F32 interior; gate
# cos >= 0.999). Python is a DEV-ONLY oracle per the parity convention.
#
# The RoPE rotation convention MUST match serenitymojo/ops/rope.mojo. We test the
# INTERLEAVED variant (pair = (x[2i], x[2i+1]), angle index i):
#     out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
#     out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
# cos/sin are precomputed (NOT learnable), so only d_x is produced. The Mojo
# driver reproduces grad_out/cos/sin/x/g/y on-device with the SAME fills below;
# only the reference GRADIENTS are read back.
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/rope_struct_bwd_oracle.py

import numpy as np
import os
import torch

OUT = os.path.join(os.path.dirname(__file__), "rope_struct_bwd_ref.txt")


# ── deterministic fills (MUST match the Mojo driver) ─────────────────────────
def fill(n, mul, sub, scale):
    a = np.empty(n, np.float64)
    for i in range(n):
        a[i] = (float((i * mul) % 13) - sub) * scale
    return a


# ── 1) RoPE interleaved backward ─────────────────────────────────────────────
def rope_interleaved_ref(rows, D):
    half = D // 2
    x = torch.tensor(fill(rows * D, 7, 6.0, 0.05).reshape(rows, D),
                     dtype=torch.float64, requires_grad=True)
    cos = torch.tensor(fill(rows * half, 5, 6.0, 0.10).reshape(rows, half),
                       dtype=torch.float64)
    sin = torch.tensor(fill(rows * half, 3, 6.0, 0.10).reshape(rows, half),
                       dtype=torch.float64)
    grad_out = torch.tensor(fill(rows * D, 2, 6.0, 0.05).reshape(rows, D),
                            dtype=torch.float64)
    x0 = x[:, 0::2]   # even channels  -> pair index i
    x1 = x[:, 1::2]   # odd channels
    o_even = x0 * cos - x1 * sin
    o_odd = x0 * sin + x1 * cos
    out = torch.empty_like(x)
    out[:, 0::2] = o_even
    out[:, 1::2] = o_odd
    out.backward(grad_out)
    return x.grad.detach().reshape(-1).numpy()


# ── 1b) RoPE HALFSPLIT backward (Z-Image) ────────────────────────────────────
# pairing = (x[i], x[i+half]), angle index i in [0, half). Forward MUST match
# serenitymojo/ops/rope.mojo _rope_halfsplit_kernel_f32:
#     out[i]      = x[i]*cos[i] - x[i+half]*sin[i]
#     out[i+half] = x[i+half]*cos[i] + x[i]*sin[i]
# We let torch autograd produce d_x so the inverse-rotation backward is checked
# against true autograd, not a hand-derived formula. Same deterministic fills as
# the interleaved case (identical grad_out/cos/sin tensors) — only the pairing
# differs, which is exactly what the gate isolates.
def rope_halfsplit_ref(rows, D):
    half = D // 2
    x = torch.tensor(fill(rows * D, 7, 6.0, 0.05).reshape(rows, D),
                     dtype=torch.float64, requires_grad=True)
    cos = torch.tensor(fill(rows * half, 5, 6.0, 0.10).reshape(rows, half),
                       dtype=torch.float64)
    sin = torch.tensor(fill(rows * half, 3, 6.0, 0.10).reshape(rows, half),
                       dtype=torch.float64)
    grad_out = torch.tensor(fill(rows * D, 2, 6.0, 0.05).reshape(rows, D),
                            dtype=torch.float64)
    x0 = x[:, :half]        # first half  -> pair lo
    x1 = x[:, half:]        # second half -> pair hi
    o_lo = x0 * cos - x1 * sin
    o_hi = x1 * cos + x0 * sin
    out = torch.cat([o_lo, o_hi], dim=1)
    out.backward(grad_out)
    return x.grad.detach().reshape(-1).numpy()


# ── 2) QkvSplitPermute backward ──────────────────────────────────────────────
# forward: fused [B,N,3*H*Dh] -> q/k/v slices (0|HD|2HD) reshaped [B,N,H,Dh].
# backward d_qkv = concat(grad_q, grad_k, grad_v) along last dim (the reshape is
# a no-op on row-major bytes). We feed distinct grad_q/k/v fills and check that
# d_qkv lays them out at the right column offsets.
def qkv_ref(B, N, H, Dh):
    HD = H * Dh
    fused = 3 * HD
    qkv = torch.tensor(fill(B * N * fused, 7, 6.0, 0.05).reshape(B, N, fused),
                       dtype=torch.float64, requires_grad=True)
    q = qkv[:, :, 0:HD].reshape(B, N, H, Dh)
    k = qkv[:, :, HD:2 * HD].reshape(B, N, H, Dh)
    v = qkv[:, :, 2 * HD:3 * HD].reshape(B, N, H, Dh)
    gq = torch.tensor(fill(B * N * HD, 2, 6.0, 0.05).reshape(B, N, H, Dh),
                      dtype=torch.float64)
    gk = torch.tensor(fill(B * N * HD, 3, 6.0, 0.05).reshape(B, N, H, Dh),
                      dtype=torch.float64)
    gv = torch.tensor(fill(B * N * HD, 5, 6.0, 0.05).reshape(B, N, H, Dh),
                      dtype=torch.float64)
    loss = (q * gq).sum() + (k * gk).sum() + (v * gv).sum()
    loss.backward()
    return qkv.grad.detach().reshape(-1).numpy()


# ── 3) GateResidual backward ─────────────────────────────────────────────────
# forward: o[r,c] = x[r,c] + g[c]*y[r,c], g per-channel [C].
def gate_ref(rows, C):
    x = torch.tensor(fill(rows * C, 7, 6.0, 0.05).reshape(rows, C),
                     dtype=torch.float64, requires_grad=True)
    y = torch.tensor(fill(rows * C, 3, 6.0, 0.05).reshape(rows, C),
                     dtype=torch.float64, requires_grad=True)
    g = torch.tensor(fill(C, 5, 6.0, 0.10), dtype=torch.float64,
                     requires_grad=True)
    grad_out = torch.tensor(fill(rows * C, 2, 6.0, 0.05).reshape(rows, C),
                            dtype=torch.float64)
    o = x + g.unsqueeze(0) * y
    o.backward(grad_out)
    return (x.grad.detach().reshape(-1).numpy(),
            g.grad.detach().reshape(-1).numpy(),
            y.grad.detach().reshape(-1).numpy())


def line(tag, arr):
    return tag + " " + " ".join(f"{v:.8f}" for v in arr.tolist())


def main():
    lines = []
    # RoPE interleaved: rows=16, D=64 (Klein-ish head dim).
    lines.append(line("rope_dx", rope_interleaved_ref(16, 64)))
    # RoPE halfsplit (Z-Image): rows=16, D=64 (same dims/fills as interleaved).
    lines.append(line("rope_halfsplit_dx", rope_halfsplit_ref(16, 64)))
    # QkvSplitPermute: B=1, N=8, H=4, Dh=16 -> fused 3*64=192.
    lines.append(line("qkv_dqkv", qkv_ref(1, 8, 4, 16)))
    # GateResidual: rows=12, C=32.
    dx, dg, dy = gate_ref(12, 32)
    lines.append(line("gate_dx", dx))
    lines.append(line("gate_dg", dg))
    lines.append(line("gate_dy", dy))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("rope_dx n =", 16 * 64, " qkv_dqkv n =", 1 * 8 * 3 * 4 * 16,
          " gate_dx n =", 12 * 32, " gate_dg n =", 32, " gate_dy n =", 12 * 32)


if __name__ == "__main__":
    main()
