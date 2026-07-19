#!/usr/bin/env python3
# gen_ops2_reference.py — DEV-ONLY numpy oracle for Phase A chunk A2 op parity.
#
# NOT in the runtime path. Run with `pixi run python` to emit expected outputs
# for the chunk-A2 ops (layer_norm, group_norm, rope interleaved/halfsplit,
# silu, gelu-tanh, swiglu, modulate, residual_gate, softmax, sdpa, conv2d) given
# fixed-seed inputs. The Mojo smoke driver (ops_smoke2.mojo) inlines the same
# inputs + these reference values. Python is a dev oracle only.

import numpy as np

SEED = 4321
np.random.seed(SEED)


def fl(name, arr):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
    print(f"# {name} shape={list(np.asarray(arr).shape)}")
    print(f"{name} = " + ", ".join(f"{v:.8f}" for v in flat))


# ── layer_norm: x[3,6], g[6], b[6], eps=1e-5 ────────────────────────────────
R, D = 3, 6
EPS = 1e-5
x_ln = np.random.randn(R, D).astype(np.float32)
g_ln = np.random.randn(D).astype(np.float32)
b_ln = np.random.randn(D).astype(np.float32)
mean = x_ln.mean(axis=-1, keepdims=True)
var = x_ln.var(axis=-1, keepdims=True)  # biased (population) variance
y_ln = (x_ln - mean) / np.sqrt(var + EPS) * g_ln + b_ln
print("# ===== layer_norm =====")
fl("x_ln", x_ln); fl("g_ln", g_ln); fl("b_ln", b_ln); fl("y_ln", y_ln)
print()

# ── group_norm NHWC: x[N=2,H=2,W=2,C=4], num_groups=2 ───────────────────────
N, H, W, C, NG = 2, 2, 2, 4, 2
EPS_GN = 1e-5
x_gn = np.random.randn(N, H, W, C).astype(np.float32)
g_gn = np.random.randn(C).astype(np.float32)
b_gn = np.random.randn(C).astype(np.float32)
cpg = C // NG
# reshape to (N, H, W, NG, cpg), normalize over (H,W,cpg) per (N,NG)
xr = x_gn.reshape(N, H, W, NG, cpg)
m = xr.mean(axis=(1, 2, 4), keepdims=True)
v = xr.var(axis=(1, 2, 4), keepdims=True)
yn = (xr - m) / np.sqrt(v + EPS_GN)
yn = yn.reshape(N, H, W, C)
y_gn = yn * g_gn + b_gn  # per-channel affine (broadcast over N,H,W)
print("# ===== group_norm NHWC =====")
fl("x_gn", x_gn); fl("g_gn", g_gn); fl("b_gn", b_gn); fl("y_gn", y_gn)
print()

# ── rope: x[rows=3, Dr=8], cos/sin [rows, Dr/2=4] ───────────────────────────
ROWS, DR = 3, 8
HALF = DR // 2
x_rope = np.random.randn(ROWS, DR).astype(np.float32)
# arbitrary angles per (row, pair)
ang = np.random.randn(ROWS, HALF).astype(np.float32)
cos = np.cos(ang).astype(np.float32)
sin = np.sin(ang).astype(np.float32)

# interleaved: pair=(x[2i], x[2i+1])
y_int = np.empty_like(x_rope)
for r in range(ROWS):
    for i in range(HALF):
        x0 = x_rope[r, 2 * i]; x1 = x_rope[r, 2 * i + 1]
        y_int[r, 2 * i] = x0 * cos[r, i] - x1 * sin[r, i]
        y_int[r, 2 * i + 1] = x0 * sin[r, i] + x1 * cos[r, i]

# halfsplit: pair=(x[i], x[i+HALF])
y_half = np.empty_like(x_rope)
for r in range(ROWS):
    for i in range(HALF):
        x0 = x_rope[r, i]; x1 = x_rope[r, i + HALF]
        y_half[r, i] = x0 * cos[r, i] - x1 * sin[r, i]
        y_half[r, i + HALF] = x1 * cos[r, i] + x0 * sin[r, i]

print("# ===== rope =====")
fl("x_rope", x_rope); fl("cos_rope", cos); fl("sin_rope", sin)
fl("y_rope_int", y_int); fl("y_rope_half", y_half)
print()

# ── activations: x[12] ──────────────────────────────────────────────────────
x_act = np.random.randn(12).astype(np.float32)
silu = x_act / (1.0 + np.exp(-x_act))
c = np.sqrt(2.0 / np.pi).astype(np.float32)
gelu = 0.5 * x_act * (1.0 + np.tanh(c * (x_act + 0.044715 * x_act**3)))
# swiglu: gate[10], up[10]
gate = np.random.randn(10).astype(np.float32)
up = np.random.randn(10).astype(np.float32)
swiglu = (gate / (1.0 + np.exp(-gate))) * up
print("# ===== activations =====")
fl("x_act", x_act); fl("silu", silu); fl("gelu", gelu)
fl("gate_sg", gate); fl("up_sg", up); fl("swiglu", swiglu)
print()

# ── elementwise: modulate, residual_gate. x[3,5], scale/shift/gate[5] ───────
RM, DM = 3, 5
x_md = np.random.randn(RM, DM).astype(np.float32)
scale = np.random.randn(DM).astype(np.float32)
shift = np.random.randn(DM).astype(np.float32)
modulate = (1.0 + scale) * x_md + shift
gate2 = np.random.randn(DM).astype(np.float32)
y_md = np.random.randn(RM, DM).astype(np.float32)
resgate = x_md + gate2 * y_md
print("# ===== elementwise =====")
fl("x_md", x_md); fl("scale_md", scale); fl("shift_md", shift); fl("modulate", modulate)
fl("gate_rg", gate2); fl("y_rg", y_md); fl("resgate", resgate)
print()

# ── softmax over last dim: x[3,7] ───────────────────────────────────────────
RS, DS = 3, 7
x_sm = np.random.randn(RS, DS).astype(np.float32)
m = x_sm.max(axis=-1, keepdims=True)
e = np.exp(x_sm - m)
softmax = e / e.sum(axis=-1, keepdims=True)
print("# ===== softmax =====")
fl("x_sm", x_sm); fl("softmax", softmax)
print()

# ── sdpa: non-causal full attention, BSHD layout B=1,S=4,H=2,Dh=8 ───────────
# We feed flash_attention q,k,v shaped [B, S, H, Dh]; mask [B,H,S,S]=0; scale.
B, S, Hn, Dh = 1, 4, 2, 8
SCALE = 1.0 / np.sqrt(Dh)
q = np.random.randn(B, S, Hn, Dh).astype(np.float32)
k = np.random.randn(B, S, Hn, Dh).astype(np.float32)
v = np.random.randn(B, S, Hn, Dh).astype(np.float32)
# reference: per (b,h): attn = softmax(q@k^T * scale) @ v   over BSHD
out_sdpa = np.empty_like(q)
for b in range(B):
    for h in range(Hn):
        qh = q[b, :, h, :]  # [S, Dh]
        kh = k[b, :, h, :]
        vh = v[b, :, h, :]
        scores = (qh @ kh.T) * SCALE  # [S, S]
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        p = e / e.sum(axis=-1, keepdims=True)
        out_sdpa[b, :, h, :] = p @ vh
print("# ===== sdpa BSHD =====")
fl("q_sdpa", q); fl("k_sdpa", k); fl("v_sdpa", v); fl("out_sdpa", out_sdpa)
print(f"# sdpa scale = {SCALE:.8f}")
print()

# ── conv2d NHWC: x[N=1,H=5,W=5,Cin=3], filter RSCF [Kh=3,Kw=3,Cin=3,Cout=4] ─
# stride=1, padding=1, dilation=1, num_groups=1
Nc, Hc, Wc, Cin = 1, 5, 5, 3
Kh, Kw, Cout = 3, 3, 4
STR, PAD, DIL = 1, 1, 1
x_cv = np.random.randn(Nc, Hc, Wc, Cin).astype(np.float32)
# filter layout RSCF = [Kh, Kw, Cin, Cout]
filt = np.random.randn(Kh, Kw, Cin, Cout).astype(np.float32)
bias_cv = np.random.randn(Cout).astype(np.float32)
Ho = (Hc + 2 * PAD - DIL * (Kh - 1) - 1) // STR + 1
Wo = (Wc + 2 * PAD - DIL * (Kw - 1) - 1) // STR + 1
out_cv = np.zeros((Nc, Ho, Wo, Cout), dtype=np.float32)
for n in range(Nc):
    for oh in range(Ho):
        for ow in range(Wo):
            for oc in range(Cout):
                acc = 0.0
                for kh in range(Kh):
                    for kw in range(Kw):
                        ih = oh * STR + kh * DIL - PAD
                        iw = ow * STR + kw * DIL - PAD
                        if 0 <= ih < Hc and 0 <= iw < Wc:
                            for ic in range(Cin):
                                acc += x_cv[n, ih, iw, ic] * filt[kh, kw, ic, oc]
                out_cv[n, oh, ow, oc] = acc + bias_cv[oc]
print("# ===== conv2d NHWC =====")
fl("x_cv", x_cv); fl("filt_cv", filt); fl("bias_cv", bias_cv); fl("out_cv", out_cv)
print(f"# conv out shape = [{Nc},{Ho},{Wo},{Cout}]")
