#!/usr/bin/env python3
# serenitymojo/models/zimage/parity/stack_oracle.py
#
# Torch oracle for the Z-Image (NextDiT) FULL STACK composition (post-embedder
# tokens -> noise refiners + context refiners -> concat -> main layers -> final
# layer), forward + autograd grads. Replicates the EXACT math of
# serenitymojo/models/zimage/zimage_stack.mojo, which composes the parity-verified
# Z-Image MODULATED block (block_oracle.py) and UNMODULATED refiner
# (refiner_oracle.py).
#
# This proves the COMPOSED backward = grad of the COMPOSED forward (the Klein
# composition-bug lesson: per-block-correct does NOT imply composition-correct).
#
# Topology (mirrors zimage_stack.mojo forward; reduced depth, REAL H=30/Dh=128):
#   for i in noise_refiner:   x_seq   = mod_block(x_seq, x_rope,  mod_nr[i])
#   for i in context_refiner: cap_seq = unmod_block(cap_seq, cap_rope)
#   unified = cat([x_seq, cap_seq], 0)              # [x, cap]
#   for i in main_layers:     unified = mod_block(unified, uni_rope, mod_main[i])
#   ln_u  = layer_norm(unified, eps=1e-6)           # no affine
#   x_out = (1 + f_scale) * ln_u                    # scale-only (no shift)
#   patches = x_out @ final_lin.T + final_lin_b
#   out   = patches[:N_IMG]
#
# Run (SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/stack_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── dims (REAL Z-Image head count H=30, head_dim Dh=128 -> hidden D=3840) ──
H = 30
Dh = 128
D = H * Dh           # 3840
IMG_H = 2
IMG_W = 3
N_IMG = IMG_H * IMG_W    # 6 image tokens
N_TXT = 4                # caption tokens
S = N_IMG + N_TXT        # 10 unified tokens
F = 96                   # reduced FFN hidden (real 10240)
OUT_CH = 16              # reduced final out_channels
EPS = 1e-5               # block / qk RMSNorm eps (matches zimage_dit.mojo)
FINAL_EPS = 1e-6         # final-layer LayerNorm eps
SCALE = 1.0 / math.sqrt(Dh)
ROPE_AXES = (32, 48, 48)
ROPE_THETA = 256
NUM_NR = 1               # noise refiners
NUM_CR = 1               # context refiners
NUM_MAIN = 2             # main layers

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def fillc(n, a, b, c):
    return torch.tensor([math.cos(a * i + b) * c for i in range(n)], dtype=DT)


def t2(n, m, a, b, c):
    return fill(n * m, a, b, c).reshape(n, m)


def rms_norm(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def silu(x):
    return x * torch.sigmoid(x)


def _axis_inv_freqs(axis_dim, theta):
    return [theta ** (-(2.0 * k) / axis_dim) for k in range(axis_dim // 2)]

F0 = _axis_inv_freqs(ROPE_AXES[0], ROPE_THETA)
F1 = _axis_inv_freqs(ROPE_AXES[1], ROPE_THETA)
F2 = _axis_inv_freqs(ROPE_AXES[2], ROPE_THETA)
HALF = Dh // 2
assert len(F0) + len(F1) + len(F2) == HALF


def rope_tables(positions):
    # positions: list of [p0,p1,p2]. Returns cos/sin [len*H, Dh/2] (interleaved).
    cos_rows, sin_rows = [], []
    for (p0, p1, p2) in positions:
        cos_tok, sin_tok = [], []
        for (p, freqs) in ((p0, F0), (p1, F1), (p2, F2)):
            for fi in freqs:
                ang = p * fi
                cos_tok.append(math.cos(ang))
                sin_tok.append(math.sin(ang))
        for _h in range(H):
            cos_rows.append(cos_tok)
            sin_rows.append(sin_tok)
    return torch.tensor(cos_rows, dtype=DT), torch.tensor(sin_rows, dtype=DT)


def rope_interleaved(x, cos, sin, n):
    cr = cos.reshape(n, H, HALF)
    sr = sin.reshape(n, H, HALF)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cr - x1 * sr
    out[..., 1::2] = x0 * sr + x1 * cr
    return out


def sdpa(q, k, v):
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)


def make_weights(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    w = {}
    w["wq"] = rnd(D, D) * 0.02
    w["wk"] = rnd(D, D) * 0.02
    w["wv"] = rnd(D, D) * 0.02
    w["wo"] = rnd(D, D) * 0.02
    w["q_norm"] = (rnd(Dh) * 0.1 + 1.0)
    w["k_norm"] = (rnd(Dh) * 0.1 + 1.0)
    w["n1"] = (rnd(D) * 0.1 + 1.0)
    w["n2"] = (rnd(D) * 0.1 + 1.0)
    w["fn1"] = (rnd(D) * 0.1 + 1.0)
    w["fn2"] = (rnd(D) * 0.1 + 1.0)
    w["w1"] = rnd(F, D) * 0.02
    w["w3"] = rnd(F, D) * 0.02
    w["w2"] = rnd(D, F) * 0.02
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_mod(seed):
    m = {}
    m["scale_msa"] = fillc(D, 0.017 + 0.001 * seed, 0.20, 0.20).requires_grad_(True)
    m["gate_msa"] = fill(D, 0.011 + 0.001 * seed, 0.30, 0.40).requires_grad_(True)
    m["scale_mlp"] = fillc(D, 0.019 + 0.001 * seed, 0.50, 0.15).requires_grad_(True)
    m["gate_mlp"] = fill(D, 0.012 + 0.001 * seed, 0.60, 0.35).requires_grad_(True)
    return m


def attention(xn, w, cos, sin, n):
    q = (xn @ w["wq"].T).reshape(1, n, H, Dh)
    k = (xn @ w["wk"].T).reshape(1, n, H, Dh)
    v = (xn @ w["wv"].T).reshape(1, n, H, Dh)
    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    qr = rope_interleaved(q, cos, sin, n)
    kr = rope_interleaved(k, cos, sin, n)
    att = sdpa(qr, kr, v).reshape(n, D)
    return att @ w["wo"].T


def ffn(xn, w):
    g_pre = xn @ w["w1"].T
    u = xn @ w["w3"].T
    act = silu(g_pre) * u
    return act @ w["w2"].T


def mod_block(x, w, m, cos, sin, n):
    xn1 = rms_norm(x, w["n1"])
    xn1s = (1.0 + m["scale_msa"]) * xn1
    att_o = attention(xn1s, w, cos, sin, n)
    attn_n2 = rms_norm(att_o, w["n2"])
    gate_msa = torch.tanh(m["gate_msa"])
    h = x + gate_msa * attn_n2
    xfn1 = rms_norm(h, w["fn1"])
    xfn1s = (1.0 + m["scale_mlp"]) * xfn1
    ff = ffn(xfn1s, w)
    ff_n2 = rms_norm(ff, w["fn2"])
    gate_mlp = torch.tanh(m["gate_mlp"])
    return h + gate_mlp * ff_n2


def unmod_block(x, w, cos, sin, n):
    xn1 = rms_norm(x, w["n1"])
    att_o = attention(xn1, w, cos, sin, n)
    attn_n2 = rms_norm(att_o, w["n2"])
    h = x + attn_n2
    xfn1 = rms_norm(h, w["fn1"])
    ff = ffn(xfn1, w)
    ff_n2 = rms_norm(ff, w["fn2"])
    return h + ff_n2


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, flat.shape)


def main():
    # ── precomputed embedder-output tokens (the stack inputs) ──
    x_seq = t2(N_IMG, D, 0.021, 0.05, 0.5).requires_grad_(True)
    cap_seq = t2(N_TXT, D, 0.018, 0.07, 0.4).requires_grad_(True)

    # per-stream weights / mod
    nr_w = [make_weights(200 + i) for i in range(NUM_NR)]
    nr_m = [make_mod(1 + i) for i in range(NUM_NR)]
    cr_w = [make_weights(300 + i) for i in range(NUM_CR)]
    main_w = [make_weights(400 + i) for i in range(NUM_MAIN)]
    main_m = [make_mod(10 + i) for i in range(NUM_MAIN)]

    g = torch.Generator().manual_seed(7)
    final_lin = (torch.randn(OUT_CH, D, generator=g, dtype=torch.float32).to(DT) * 0.02).requires_grad_(True)
    final_lin_b = fill(OUT_CH, 0.03, 0.01, 0.05).requires_grad_(True)
    f_scale = fillc(D, 0.015, 0.30, 0.10).requires_grad_(True)

    # ── RoPE positions (mirror zimage_dit.mojo) ──
    # caption tokens: (i+1, 0, 0)
    cap_pos = [[float(i + 1), 0.0, 0.0] for i in range(N_TXT)]
    # image tokens: (N_TXT+1, ih, iw)
    x0 = float(N_TXT + 1)
    img_pos = [[x0, float(ih), float(iw)] for ih in range(IMG_H) for iw in range(IMG_W)]
    # unified [x, cap]
    uni_pos = img_pos + cap_pos

    x_cos, x_sin = rope_tables(img_pos)
    cap_cos, cap_sin = rope_tables(cap_pos)
    uni_cos, uni_sin = rope_tables(uni_pos)
    assert float(uni_cos.std()) > 1e-3, "degenerate rope"

    # ── forward ──
    xs = x_seq
    for i in range(NUM_NR):
        xs = mod_block(xs, nr_w[i], nr_m[i], x_cos, x_sin, N_IMG)
    cs = cap_seq
    for i in range(NUM_CR):
        cs = unmod_block(cs, cr_w[i], cap_cos, cap_sin, N_TXT)
    unified = torch.cat([xs, cs], 0)             # [x, cap] -> [S, D]
    for i in range(NUM_MAIN):
        unified = mod_block(unified, main_w[i], main_m[i], uni_cos, uni_sin, S)
    x_final = unified
    ln_u = (x_final - x_final.mean(-1, keepdim=True)) / torch.sqrt(
        x_final.var(-1, unbiased=False, keepdim=True) + FINAL_EPS)
    x_out = (1.0 + f_scale) * ln_u                # scale-only
    patches = x_out @ final_lin.T + final_lin_b   # [S, out_ch]
    out = patches[:N_IMG]                         # [N_IMG, out_ch]

    d_out = t2(N_IMG, OUT_CH, 0.027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    # forward output
    W("sref_out", out)
    # token grads (full-chain proof back to the embedder outputs)
    W("sref_d_x_seq", x_seq.grad)
    W("sref_d_cap_seq", cap_seq.grad)
    # final-layer grads
    W("sref_d_f_scale", f_scale.grad)
    W("sref_d_final_lin", final_lin.grad)
    # representative weight grads: deepest main layer + the noise refiner + ctx refiner
    W("sref_d_main_deep_wq", main_w[NUM_MAIN - 1]["wq"].grad)
    W("sref_d_main_deep_w2", main_w[NUM_MAIN - 1]["w2"].grad)
    W("sref_d_main_shallow_wq", main_w[0]["wq"].grad)
    W("sref_d_nr0_wq", nr_w[0]["wq"].grad)
    W("sref_d_nr0_w2", nr_w[0]["w2"].grad)
    W("sref_d_cr0_wq", cr_w[0]["wq"].grad)
    W("sref_d_cr0_w2", cr_w[0]["w2"].grad)
    # per-block RAW mod-vec grads packed [4D]: scale_msa|gate_msa|scale_mlp|gate_mlp
    def pack4(m):
        return torch.cat([m["scale_msa"].grad, m["gate_msa"].grad,
                          m["scale_mlp"].grad, m["gate_mlp"].grad], 0)
    W("sref_d_nr0_mod", pack4(nr_m[0]))
    W("sref_d_main_deep_mod", pack4(main_m[NUM_MAIN - 1]))

    # ── inputs the Mojo gate reconstructs ──
    W("sin_x_seq", x_seq)
    W("sin_cap_seq", cap_seq)
    W("sin_f_scale", f_scale)
    W("sin_final_lin", final_lin)
    W("sin_final_lin_b", final_lin_b)
    W("sin_d_out", d_out)
    W("sin_x_cos", x_cos); W("sin_x_sin", x_sin)
    W("sin_cap_cos", cap_cos); W("sin_cap_sin", cap_sin)
    W("sin_uni_cos", uni_cos); W("sin_uni_sin", uni_sin)

    def dump_block(prefix, w):
        for kk in ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
                   "n2", "fn1", "w1", "w3", "w2", "fn2"]:
            W("%s_%s" % (prefix, kk), w[kk])

    def dump_mod(prefix, m):
        for kk in ["scale_msa", "gate_msa", "scale_mlp", "gate_mlp"]:
            W("%s_%s" % (prefix, kk), m[kk])

    for i in range(NUM_NR):
        dump_block("sin_nr%d" % i, nr_w[i]); dump_mod("sin_nr%d" % i, nr_m[i])
    for i in range(NUM_CR):
        dump_block("sin_cr%d" % i, cr_w[i])
    for i in range(NUM_MAIN):
        dump_block("sin_main%d" % i, main_w[i]); dump_mod("sin_main%d" % i, main_m[i])

    print("stack forward loss =", float(loss))
    print("NR=%d CR=%d MAIN=%d  S=%d D=%d F=%d" % (NUM_NR, NUM_CR, NUM_MAIN, S, D, F))
    print("DONE")


if __name__ == "__main__":
    main()
