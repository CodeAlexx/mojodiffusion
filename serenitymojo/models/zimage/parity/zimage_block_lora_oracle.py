#!/usr/bin/env python3
# serenitymojo/models/zimage/parity/zimage_block_lora_oracle.py
#
# Torch oracle for the Z-Image (NextDiT) MAIN-LAYER DiT block WITH LoRA adapters
# on all 7 trainable projections (to_q/k/v/out + SwiGLU w1/w3/w2). Replicates the
# EXACT math of serenitymojo/models/zimage/lora_block.mojo `zimage_block_lora_forward`
# (which == block.mojo forward + an additive LoRA delta on each linear), runs torch
# autograd, and dumps .bin references the Mojo LoRA gate (zimage_block_lora_parity.mojo)
# reads + compares at cos >= 0.999.
#
# LoRA: y' = x@W.T + scale*((x @ A.T) @ B.T), A=[rank,in], B=[out,rank],
#   scale = alpha/rank. B is NONZERO here (PEFT inits B=0 -> d_A==0, untestable);
#   parity needs nonzero A AND B to exercise both grads.
#   (Matches train_step._lora_fwd / zimage_lora_apply / zimage_lora_bwd.)
#
# All base-block conventions are identical to block_oracle.py (rms eps 1e-5,
# sandwich norm, (1+scale)*norm no shift, tanh gate, sdpa 1/sqrt(Dh), SwiGLU,
# INTERLEAVED half-width rope). See block_oracle.py for the detailed notes.
#
# Run (SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/zimage_block_lora_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64  # F64 reference interior

# ── dims (MUST match zimage_block_lora_parity.mojo + block_oracle.py) ──
H = 30
Dh = 128
D = H * Dh          # 3840
N_TXT = 2
IMG_H = 2
IMG_W = 3
N_IMG = IMG_H * IMG_W   # 6
S = N_TXT + N_IMG       # 8
F = 96
EPS = 1e-5
SCALE = 1.0 / math.sqrt(Dh)
ROPE_AXES = (32, 48, 48)
ROPE_THETA = 256

# ── LoRA hyperparams ──
RANK = 8
ALPHA = 16.0
SCALE_LORA = ALPHA / RANK   # 2.0  (non-trivial, exercises the scale field)

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def fillc(n, a, b, c):
    return torch.tensor([math.cos(a * i + b) * c for i in range(n)], dtype=DT)


def t2(n, m, a, b, c):
    return fill(n * m, a, b, c).reshape(n, m)


# ── ops (match the Mojo forward exactly) ──
def rms_norm(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def residual_gate(x, gate, y):
    return x + gate * y


def silu(x):
    return x * torch.sigmoid(x)


def rope_interleaved(x, cos, sin):
    Sx = x.shape[1]
    half = Dh // 2
    cr = cos.reshape(Sx, H, half)
    sr = sin.reshape(Sx, H, half)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cr - x1 * sr
    out[..., 1::2] = x0 * sr + x1 * cr
    return out


def _axis_inv_freqs(axis_dim, theta):
    return [theta ** (-(2.0 * k) / axis_dim) for k in range(axis_dim // 2)]


def build_real_rope_tables():
    f0 = _axis_inv_freqs(ROPE_AXES[0], ROPE_THETA)
    f1 = _axis_inv_freqs(ROPE_AXES[1], ROPE_THETA)
    f2 = _axis_inv_freqs(ROPE_AXES[2], ROPE_THETA)
    half = Dh // 2
    assert len(f0) + len(f1) + len(f2) == half
    cos_rows = []
    sin_rows = []
    for tok in range(S):
        if tok < N_TXT:
            p0 = float(tok + 1); p1 = 0.0; p2 = 0.0
        else:
            it = tok - N_TXT
            r = it // IMG_W
            c = it % IMG_W
            p0 = float(N_TXT + 1); p1 = float(r); p2 = float(c)
        cos_tok = []; sin_tok = []
        for (p, freqs) in ((p0, f0), (p1, f1), (p2, f2)):
            for fi in freqs:
                ang = p * fi
                cos_tok.append(math.cos(ang))
                sin_tok.append(math.sin(ang))
        for _h in range(H):
            cos_rows.append(cos_tok)
            sin_rows.append(sin_tok)
    cos = torch.tensor(cos_rows, dtype=DT)
    sin = torch.tensor(sin_rows, dtype=DT)
    return cos, sin


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


def make_mod():
    m = {}
    m["scale_msa"] = (fillc(D, 0.017, 0.20, 0.20)).requires_grad_(True)
    m["gate_msa"] = (fill(D, 0.011, 0.30, 0.40)).requires_grad_(True)
    m["scale_mlp"] = (fillc(D, 0.019, 0.50, 0.15)).requires_grad_(True)
    m["gate_mlp"] = (fill(D, 0.012, 0.60, 0.35)).requires_grad_(True)
    return m


# slot order MUST match lora_block.mojo: q,k,v,out,w1,w3,w2
LSLOTS = ["q", "k", "v", "out", "w1", "w3", "w2"]
LSPEC = {
    "q": (D, D), "k": (D, D), "v": (D, D), "out": (D, D),
    "w1": (D, F), "w3": (D, F), "w2": (F, D),
}


def make_lora(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape, s):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * s).to(DT)
    lo = {}
    for nm in LSLOTS:
        inf, outf = LSPEC[nm]
        A = rnd(RANK, inf, s=0.03).requires_grad_(True)      # [rank,in]
        B = rnd(outf, RANK, s=0.03).requires_grad_(True)     # [out,rank] (nonzero)
        lo[nm] = (A, B, inf, outf)
    return lo


def lora_delta(x, nm, lo):
    A, B, inf, outf = lo[nm]
    return SCALE_LORA * ((x @ A.T) @ B.T)


def main():
    x = t2(S, D, 0.021, 0.05, 0.5).requires_grad_(True)
    w = make_weights(1)
    m = make_mod()
    lo = make_lora(11)

    cos, sin = build_real_rope_tables()
    dvar = float(cos.std())
    assert dvar > 1e-3, "rope table is degenerate"

    # ── forward (base block + additive LoRA delta on each projection) ──
    xn1 = rms_norm(x, w["n1"])
    xn1s = (1.0 + m["scale_msa"]) * xn1                  # LoRA input for q/k/v
    q = ((xn1s @ w["wq"].T) + lora_delta(xn1s, "q", lo)).reshape(1, S, H, Dh)
    k = ((xn1s @ w["wk"].T) + lora_delta(xn1s, "k", lo)).reshape(1, S, H, Dh)
    v = ((xn1s @ w["wv"].T) + lora_delta(xn1s, "v", lo)).reshape(1, S, H, Dh)
    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v).reshape(S, D)                 # LoRA input for to_out
    att_o = (att @ w["wo"].T) + lora_delta(att, "out", lo)
    attn_n2 = rms_norm(att_o, w["n2"])
    gate_msa = torch.tanh(m["gate_msa"])
    h = residual_gate(x, gate_msa, attn_n2)

    xfn1 = rms_norm(h, w["fn1"])
    xfn1s = (1.0 + m["scale_mlp"]) * xfn1                # LoRA input for w1/w3
    g_pre = (xfn1s @ w["w1"].T) + lora_delta(xfn1s, "w1", lo)
    u = (xfn1s @ w["w3"].T) + lora_delta(xfn1s, "w3", lo)
    act = silu(g_pre) * u                               # LoRA input for w2
    ff = (act @ w["w2"].T) + lora_delta(act, "w2", lo)
    ff_n2 = rms_norm(ff, w["fn2"])
    gate_mlp = torch.tanh(m["gate_mlp"])
    out = residual_gate(h, gate_mlp, ff_n2)

    d_out = t2(S, D, 0.027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    def W(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
        path = os.path.join(REF_DIR, name + ".bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, flat.shape)

    # ── reference grads ──
    W("lref_out", out)
    W("lref_d_x", x.grad)
    for kk in ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "n2", "fn1", "w1", "w3", "w2", "fn2"]:
        W("lref_d_%s" % kk, w[kk].grad)
    for kk in ["scale_msa", "gate_msa", "scale_mlp", "gate_mlp"]:
        W("lref_d_%s" % kk, m[kk].grad)
    for nm in LSLOTS:
        W("lref_%s_dA" % nm, lo[nm][0].grad)
        W("lref_%s_dB" % nm, lo[nm][1].grad)

    # ── inputs the Mojo gate reconstructs ──
    W("lin_x", x)
    for kk in ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "n2", "fn1", "w1", "w3", "w2", "fn2"]:
        W("lin_w_%s" % kk, w[kk])
    for kk in ["scale_msa", "gate_msa", "scale_mlp", "gate_mlp"]:
        W("lin_m_%s" % kk, m[kk])
    for nm in LSLOTS:
        W("lin_lora_%s_A" % nm, lo[nm][0])
        W("lin_lora_%s_B" % nm, lo[nm][1])
    W("lin_cos", cos)
    W("lin_sin", sin)
    W("lin_d_out", d_out)

    print("forward loss =", float(loss))
    print("rope cos std =", dvar)
    print("RANK =", RANK, " SCALE_LORA =", SCALE_LORA)
    print("DONE")


if __name__ == "__main__":
    main()
