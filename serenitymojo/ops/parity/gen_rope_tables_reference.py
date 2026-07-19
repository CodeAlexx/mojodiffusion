#!/usr/bin/env python3
# gen_rope_tables_reference.py — DEV-ONLY numpy/torch oracle for the multi-axis
# (3D) RoPE table builder parity probe (ops/rope_tables_probe.mojo).
#
# NOT in the runtime path. The Mojo probe inlines these inputs and recomputes the
# same reference on host; this script is how those numbers were produced and how
# they can be regenerated/verified against a GPU-bf16 torch path. No Python at
# runtime.
#
# Mirrors ops/rope_tables.mojo::build_multiaxis_rope_tables, whose layout matches
# the Phase 2-4 DiT refs (wan22_dit.rs:364 complex 3-axis; cosmos_predict25_dit.rs
# :26-42 half-split cat([t,h,w])). Per axis a (half_a = axes_dims[a]//2):
#     inv_freq_i = theta ** (-i / half_a),    i in [0, half_a)
#     angle[t, off_a+i] = pos[t,a] * inv_freq_i
#     cos = cos(angle); sin = sin(angle)        tables [rows, sum(half_a)]
#
# Run:
#   pixi run python serenitymojo/ops/parity/gen_rope_tables_reference.py

import numpy as np

try:
    import torch
    HAVE_TORCH = torch.cuda.is_available()
except Exception:
    HAVE_TORCH = False

AXES_DIMS = [4, 4, 2]          # half = 2 + 2 + 1 = 5
THETA = 100.0
# token-major positions [f, h, w] per token
POSITIONS = np.array(
    [[0, 0, 0],
     [1, 0, 1],
     [2, 3, 0]], dtype=np.float64
)


def build(positions, axes_dims, theta):
    rows = positions.shape[0]
    halves = [d // 2 for d in axes_dims]
    half = sum(halves)
    cos_t = np.zeros((rows, half), dtype=np.float64)
    sin_t = np.zeros((rows, half), dtype=np.float64)
    for t in range(rows):
        off = 0
        for a, ha in enumerate(halves):
            for i in range(ha):
                inv = theta ** (-i / ha)
                ang = positions[t, a] * inv
                cos_t[t, off + i] = np.cos(ang)
                sin_t[t, off + i] = np.sin(ang)
            off += ha
    return cos_t, sin_t


def main():
    cos_np, sin_np = build(POSITIONS, AXES_DIMS, THETA)
    print("# axes_dims =", AXES_DIMS, " theta =", THETA)
    print("# positions (token-major f,h,w):")
    print(POSITIONS.astype(int).tolist())
    print("# cos table [rows, half] (numpy F64 oracle):")
    print(np.round(cos_np, 8).tolist())
    print("# sin table [rows, half]:")
    print(np.round(sin_np, 8).tolist())

    if HAVE_TORCH:
        # Cross-check the same math on a GPU bf16 path (apply-site dtype). The
        # table itself stays F32/F64; bf16 enters only when cast at the rope
        # apply site, so we report the bf16 round-trip error of the table to
        # bound apply-time precision loss.
        c = torch.tensor(cos_np, device="cuda", dtype=torch.float32)
        c_bf = c.to(torch.bfloat16).to(torch.float32)
        max_bf = (c - c_bf).abs().max().item()
        print("# torch GPU bf16 round-trip max-abs on cos table:", max_bf)
    else:
        print("# (torch CUDA unavailable; numpy F64 oracle only)")


if __name__ == "__main__":
    main()
