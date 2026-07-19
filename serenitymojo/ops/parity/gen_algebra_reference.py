#!/usr/bin/env python3
# gen_algebra_reference.py — DEV-ONLY numpy oracle for the tensor-algebra +
# layout kit (ops/tensor_algebra.mojo + ops/layout.mojo).
#
# NOT in the runtime path. Run `pixi run python serenitymojo/ops/parity/
# gen_algebra_reference.py` to emit expected outputs for fixed-seed inputs; the
# Mojo smoke driver (ops/algebra_smoke.mojo) inlines the same inputs + these
# reference values. Python is a dev oracle only — nothing here runs at runtime.
#
# Seed = 7777 (distinct from ops2's 4321).

import numpy as np

SEED = 7777
np.random.seed(SEED)


def fl(name, arr):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
    print(f"# {name} shape={list(np.asarray(arr).shape)}")
    print(f"{name} = " + ", ".join(f"{v:.8f}" for v in flat))


print("# ===== elementwise tensor-tensor (same shape) [3,4] =====")
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(3, 4).astype(np.float32)
fl("ew_a", a)
fl("ew_b", b)
fl("ew_add", a + b)
fl("ew_sub", a - b)
fl("ew_mul", a * b)
# div: keep b away from zero for a clean ratio
bd = b + np.sign(b) * 1.0 + 0.5
fl("ew_bd", bd)
fl("ew_div", a / bd)
print()

print("# ===== broadcast [B,1,D] + [B,S,D] : B=2,S=3,D=4 =====")
B, S, D = 2, 3, 4
x_bc = np.random.randn(B, 1, D).astype(np.float32)   # [2,1,4]
y_bc = np.random.randn(B, S, D).astype(np.float32)   # [2,3,4]
fl("bc_x", x_bc)
fl("bc_y", y_bc)
fl("bc_add", (x_bc + y_bc).astype(np.float32))       # [2,3,4]
print()

print("# ===== broadcast scalar-like [1] over [2,5] (tensor-tensor) =====")
x_s1 = np.random.randn(2, 5).astype(np.float32)
y_s1 = np.random.randn(1).astype(np.float32)         # [1] broadcast everywhere
fl("s1_x", x_s1)
fl("s1_y", y_s1)
fl("s1_mul", (x_s1 * y_s1).astype(np.float32))
print()

print("# ===== tensor-scalar : x[2,5] op 1.75 =====")
x_sc = np.random.randn(2, 5).astype(np.float32)
SCAL = 1.75
fl("sc_x", x_sc)
print(f"# scalar = {SCAL}")
fl("sc_add", x_sc + SCAL)
fl("sc_sub", x_sc - SCAL)
fl("sc_mul", x_sc * SCAL)
fl("sc_div", x_sc / SCAL)
print()

print("# ===== reshape [2,3,4] -> [6,4] (data unchanged) =====")
x_rs = np.random.randn(2, 3, 4).astype(np.float32)
fl("rs_x", x_rs)
fl("rs_out", x_rs.reshape(6, 4))
print()

print("# ===== transpose [2,3,4] swap (1,2) -> [2,4,3] =====")
x_tr = np.random.randn(2, 3, 4).astype(np.float32)
fl("tr_x", x_tr)
fl("tr_out", np.ascontiguousarray(x_tr.transpose(0, 2, 1)))
print()

print("# ===== permute >2 dims [2,3,4,5] perm (2,0,3,1) -> [4,2,5,3] =====")
x_pm = np.random.randn(2, 3, 4, 5).astype(np.float32)
fl("pm_x", x_pm)
# perm[k] = source axis for output axis k; np.transpose uses same convention
fl("pm_out", np.ascontiguousarray(x_pm.transpose(2, 0, 3, 1)))
print()

print("# ===== concat along non-last dim: two [2,2,3] along dim=1 -> [2,4,3] =====")
c0 = np.random.randn(2, 2, 3).astype(np.float32)
c1 = np.random.randn(2, 2, 3).astype(np.float32)
fl("cc_a", c0)
fl("cc_b", c1)
fl("cc_dim1", np.concatenate([c0, c1], axis=1))
print()

print("# ===== concat along last dim: [2,3]+[2,2] along dim=1 -> [2,5] =====")
cl0 = np.random.randn(2, 3).astype(np.float32)
cl1 = np.random.randn(2, 2).astype(np.float32)
fl("ccl_a", cl0)
fl("ccl_b", cl1)
fl("ccl_dim1", np.concatenate([cl0, cl1], axis=1))
print()

print("# ===== slice [2,5,3] dim=1 start=1 length=3 -> [2,3,3] =====")
x_sl = np.random.randn(2, 5, 3).astype(np.float32)
fl("sl_x", x_sl)
fl("sl_out", np.ascontiguousarray(x_sl[:, 1:4, :]))
print()

print("# ===== gather_rows / embedding: table[6,4], ids=[3,0,5,1,5] -> [5,4] =====")
table = np.random.randn(6, 4).astype(np.float32)
ids = [3, 0, 5, 1, 5]
fl("gr_table", table)
print(f"# ids = {ids}")
fl("gr_out", table[ids])
print()

print("# ===== patchify [B=1,C=2,H=4,W=4] p=2 -> [1, 4, 8] =====")
Bp, Cp, Hp, Wp, pp = 1, 2, 4, 4, 2
x_pt = np.random.randn(Bp, Cp, Hp, Wp).astype(np.float32)
fl("pt_x", x_pt)
# 'b c (h p1) (w p2) -> b (h w) (c p1 p2)'
GH, GW = Hp // pp, Wp // pp
seq = np.zeros((Bp, GH * GW, Cp * pp * pp), dtype=np.float32)
for bb in range(Bp):
    for gh in range(GH):
        for gw in range(GW):
            l = gh * GW + gw
            for c in range(Cp):
                for ph in range(pp):
                    for pw in range(pp):
                        f = (c * pp + ph) * pp + pw
                        seq[bb, l, f] = x_pt[bb, c, gh * pp + ph, gw * pp + pw]
fl("pt_seq", seq)
# unpatchify(seq) must recover x exactly
fl("pt_unpatch", x_pt)
print()

print("# ===== deinterleave_pair [2, 6] -> evens[2,3], odds[2,3] =====")
x_di = np.random.randn(2, 6).astype(np.float32)
fl("di_x", x_di)
fl("di_even", np.ascontiguousarray(x_di[:, 0::2]))
fl("di_odd", np.ascontiguousarray(x_di[:, 1::2]))
print()
