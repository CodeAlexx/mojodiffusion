#!/usr/bin/env python3
# shape_bwd_oracle.py — PyTorch reference for the Tier-0 shape/structural BACKWARD
# arms (serenitymojo/ops/shape_backward.mojo):
#   Cat, Split, Slice, Reshape, Transpose, Permute, Broadcast, Repeat, Where,
#   Clamp, Maximum, Minimum, Cast, IndexSelect.
#
# Oracle = PyTorch autograd.grad (F64 ground-truth). Python is a DEV-ONLY oracle
# per the parity convention. The Mojo driver reproduces every deterministic input
# on-device; only the reference GRADIENTS are read back here.
#
# Emits one line per tag: "<tag> v0 v1 ...".
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/ops/parity/shape_bwd_oracle.py

import os
import numpy as np
import torch

OUT = os.path.join(os.path.dirname(__file__), "shape_bwd_ref.txt")


def fill(n, mul=7, mod=13, sub=6.0, scale=0.1):
    """Deterministic signed fill. Matches _fill in shape_bwd_parity.mojo."""
    return np.array([(float((i * mul) % mod) - sub) * scale for i in range(n)],
                    np.float64)


def fill_grad(n):
    """Deterministic upstream grad. Matches _fill_grad in the Mojo driver."""
    return np.array([(float((i * 2) % 7) - 3.0) * 0.05 for i in range(n)],
                    np.float64)


def t(arr, shape):
    return torch.tensor(arr.reshape(shape), dtype=torch.float64, requires_grad=True)


def g(arr, shape):
    return torch.tensor(arr.reshape(shape), dtype=torch.float64)


def emit(lines, tag, grad):
    a = grad.detach().reshape(-1).numpy()
    lines.append(tag + " " + " ".join(f"{x:.8f}" for x in a.tolist()))


def main():
    lines = []

    # ── CAT along dim 0: two pieces [2,4] + [3,4] -> [5,4]. d_y -> d_a, d_b ───
    a = t(fill(2 * 4), (2, 4))
    b = t(fill(3 * 4, mul=5, mod=11, sub=5.0), (3, 4))
    y = torch.cat([a, b], dim=0)
    gy = g(fill_grad(5 * 4), (5, 4))
    y.backward(gy)
    emit(lines, "cat_d0", a.grad)
    emit(lines, "cat_d1", b.grad)

    # ── SPLIT: x [5,4] split into [2,4] + [3,4] along dim 0 -> grads concat
    #     back to d_x. The 2-piece split is the diffusion-common case.
    x = t(fill(5 * 4), (5, 4))
    p0 = x[0:2, :]
    p1 = x[2:5, :]
    g0 = g(fill_grad(2 * 4), (2, 4))
    g1 = g(fill_grad(3 * 4), (3, 4)) * 2.0
    loss = (p0 * g0).sum() + (p1 * g1).sum()
    loss.backward()
    emit(lines, "split_dx", x.grad)

    # ── SLICE: x [6,4], slice dim0 [1:4) -> [3,4]; scatter d_y back to d_x ────
    xs = t(fill(6 * 4), (6, 4))
    ys = xs[1:4, :]
    gys = g(fill_grad(3 * 4), (3, 4))
    ys.backward(gys)
    emit(lines, "slice_dx", xs.grad)

    # ── RESHAPE: [4,6] -> [2,12]; grad reshaped back ─────────────────────────
    xr = t(fill(24), (4, 6))
    yr = xr.reshape(2, 12)
    gyr = g(fill_grad(24), (2, 12))
    yr.backward(gyr)
    emit(lines, "reshape_dx", xr.grad)

    # ── TRANSPOSE: [3,5] swap (0,1) -> [5,3] ─────────────────────────────────
    xt = t(fill(15), (3, 5))
    yt = xt.transpose(0, 1)
    gyt = g(fill_grad(15), (5, 3))
    yt.backward(gyt)
    emit(lines, "transpose_dx", xt.grad)

    # ── PERMUTE: [2,3,4] perm (2,0,1) -> [4,2,3] ─────────────────────────────
    xp = t(fill(24), (2, 3, 4))
    yp = xp.permute(2, 0, 1)
    gyp = g(fill_grad(24), (4, 2, 3))
    yp.backward(gyp)
    emit(lines, "permute_dx", xp.grad)

    # ── BROADCAST: [1,4] + [3,4] => out [3,4]; d_x for the [1,4] operand ─────
    xb = t(fill(4), (1, 4))
    other = torch.zeros((3, 4), dtype=torch.float64)
    yb = xb + other       # broadcast [1,4] -> [3,4]
    gyb = g(fill_grad(12), (3, 4))
    yb.backward(gyb)
    emit(lines, "broadcast_dx", xb.grad)

    # ── REPEAT: [2,3] repeat (2,2) -> [4,6]; sum tiled copies back to [2,3] ──
    xrep = t(fill(6), (2, 3))
    yrep = xrep.repeat(2, 2)
    gyrep = g(fill_grad(24), (4, 6))
    yrep.backward(gyrep)
    emit(lines, "repeat_dx", xrep.grad)

    # ── WHERE: cond ? a : b, elementwise [8] ─────────────────────────────────
    cond_np = np.array([1.0 if (i % 2 == 0) else 0.0 for i in range(8)], np.float64)
    aw = t(fill(8), (8,))
    bw = t(fill(8, mul=5, mod=11, sub=5.0), (8,))
    cw = torch.tensor(cond_np, dtype=torch.bool)
    yw = torch.where(cw, aw, bw)
    gyw = g(fill_grad(8), (8,))
    yw.backward(gyw)
    emit(lines, "where_da", aw.grad)
    emit(lines, "where_db", bw.grad)

    # ── CLAMP: clamp(x, -0.2, 0.2) on signed [16] ────────────────────────────
    xc = t(fill(16), (16,))
    yc = torch.clamp(xc, -0.2, 0.2)
    gyc = g(fill_grad(16), (16,))
    yc.backward(gyc)
    emit(lines, "clamp_dx", xc.grad)

    # ── MAXIMUM / MINIMUM elementwise [16] ───────────────────────────────────
    am = t(fill(16), (16,))
    bm = t(fill(16, mul=5, mod=11, sub=5.0), (16,))
    ymax = torch.maximum(am, bm)
    gym = g(fill_grad(16), (16,))
    ymax.backward(gym)
    emit(lines, "max_da", am.grad)
    emit(lines, "max_db", bm.grad)
    am.grad = None
    bm.grad = None
    ymin = torch.minimum(am, bm)
    ymin.backward(gym)
    emit(lines, "min_da", am.grad)
    emit(lines, "min_db", bm.grad)

    # ── CAST: identity grad in F32 gate ([12]) ───────────────────────────────
    xcast = t(fill(12), (12,))
    ycast = xcast.to(torch.float64)   # identity (F32->F32 in the Mojo gate)
    gyc2 = g(fill_grad(12), (12,))
    ycast.backward(gyc2)
    emit(lines, "cast_dx", xcast.grad)

    # ── INDEX_SELECT: table [5,4], select rows [0,2,2,4] (repeat 2) along dim0
    #     d_x accumulates the repeated selection.
    tbl = t(fill(20), (5, 4))
    idx = torch.tensor([0, 2, 2, 4], dtype=torch.long)
    ysel = torch.index_select(tbl, 0, idx)   # [4,4]
    gysel = g(fill_grad(16), (4, 4))
    ysel.backward(gysel)
    emit(lines, "index_select_dx", tbl.grad)

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
