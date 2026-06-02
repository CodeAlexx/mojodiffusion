#!/usr/bin/env python3
# serenitymojo/models/zimage/parity/block_oracle.py
#
# Torch oracle for the Z-Image (NextDiT) MAIN-LAYER DiT block (forward + autograd
# grads). Replicates the EXACT math of serenitymojo/models/zimage/block.mojo's
# `zimage_block_forward`, which mirrors models/dit/zimage_dit.mojo `_block`
# (adaln branch) and inference-flame zimage_nextdit.rs `transformer_block` /
# the diffusers ZImageTransformer2DModel. Produces .bin references the Mojo gate
# (block_parity.mojo) reads + compares at cos >= 0.999.
#
# CONVENTIONS (must match the Mojo forward byte-for-byte):
#   rms_norm(x, w): x / sqrt(mean(x^2, -1) + eps) * w            (eps 1e-5)
#   sandwich norm: norm1 BEFORE sub-layer, norm2 AFTER, gated residual on norm2
#   scale modulation (NO shift): xn1s = (1 + scale_raw) * norm1(x)
#   gate: g = tanh(gate_raw); residual: out = x + g * norm2(sublayer)
#   sdpa: non-causal, scale = 1/sqrt(Dh)
#   MLP: SwiGLU -> w2( silu(w1(x)) * w3(x) )
#   RoPE: INTERLEAVED (pair (x[2i],x[2i+1])), cos/sin HALF-WIDTH [S*H, Dh/2].
#     This is diffusers transformer_z_image.py apply_rotary_emb:
#       x_c = view_as_complex(x.reshape(*shape[:-1], -1, 2))   # adjacent pairs
#       x_out = view_as_real(x_c * freqs_cis).flatten(-2)
#     which is exactly the interleaved form
#       out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
#       out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
#     NOT half-split. (The base Z-Image differs from Ernie here.)
#   The 4 modulation chunks are the RAW vectors (pre tanh / pre 1+); the block
#   applies tanh (gates) and +1 (scales) internally and owns those grads, so the
#   oracle treats scale_msa/gate_msa/scale_mlp/gate_mlp as the leaf parameters.
#
# Run (SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64  # F64 reference interior

# ── dims (REAL Z-Image head count H=30, head_dim Dh=128 -> hidden D=3840) ──
H = 30
Dh = 128
D = H * Dh          # 3840
# REAL 3-axis positional structure: caption-first (text) then image grid.
N_TXT = 2           # caption/text tokens
IMG_H = 2           # image grid rows
IMG_W = 3           # image grid cols
N_IMG = IMG_H * IMG_W   # 6 image tokens
S = N_TXT + N_IMG   # 8 (caption-first / image-second, matching diffusers concat)
F = 96              # FFN hidden for SwiGLU gate (small; real is 10240)
EPS = 1e-5
SCALE = 1.0 / math.sqrt(Dh)
# REAL Z-Image RoPE axis dims and theta.
ROPE_AXES = (32, 48, 48)   # sum = 64 = Dh//2
ROPE_THETA = 256

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
    # gate broadcast per-channel [D] over [..., D]
    return x + gate * y


def silu(x):
    return x * torch.sigmoid(x)


def rope_interleaved(x, cos, sin):
    # x [1,S,H,Dh]; cos/sin HALF-WIDTH [S*H, Dh/2] (rows = S*H, flattened (s,h)).
    Sx = x.shape[1]
    half = Dh // 2
    cr = cos.reshape(Sx, H, half)          # [S,H,half]
    sr = sin.reshape(Sx, H, half)
    x0 = x[..., 0::2]                       # even idx -> pair real part [..,half]
    x1 = x[..., 1::2]                       # odd idx  -> pair imag part
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cr - x1 * sr
    out[..., 1::2] = x0 * sr + x1 * cr
    return out


def _axis_inv_freqs(axis_dim, theta):
    # matches the Z-Image / Lumina rope: theta**(-(2k)/axis_dim), k in [0,axis_dim/2)
    return [theta ** (-(2.0 * k) / axis_dim) for k in range(axis_dim // 2)]


def build_real_rope_tables():
    # Replicates the 3-axis interleaved RoPE angle table. Per token, per axis,
    # angle = pos_axis * inv_freq[i]; axes concatenated 32|48|48 (sum=Dh/2=64).
    # ONE angle per pair (interleaved, NO consecutive doubling — unlike Ernie's
    # half-split-doubled table). Broadcast over H. Returns cos/sin HALF-WIDTH
    # [S*H, Dh/2] (rows flattened in (token, head) order).
    f0 = _axis_inv_freqs(ROPE_AXES[0], ROPE_THETA)
    f1 = _axis_inv_freqs(ROPE_AXES[1], ROPE_THETA)
    f2 = _axis_inv_freqs(ROPE_AXES[2], ROPE_THETA)
    half = Dh // 2
    assert len(f0) + len(f1) + len(f2) == half
    cos_rows = []
    sin_rows = []
    for tok in range(S):
        if tok < N_TXT:
            # caption positions: (1,0,0),(2,0,0),...
            p0 = float(tok + 1)
            p1 = 0.0
            p2 = 0.0
        else:
            it = tok - N_TXT
            r = it // IMG_W
            c = it % IMG_W
            p0 = float(N_TXT + 1)
            p1 = float(r)
            p2 = float(c)
        cos_tok = []
        sin_tok = []
        for (p, freqs) in ((p0, f0), (p1, f1), (p2, f2)):
            for fi in freqs:
                ang = p * fi
                cos_tok.append(math.cos(ang))   # one angle per pair
                sin_tok.append(math.sin(ang))
        for _h in range(H):
            cos_rows.append(cos_tok)
            sin_rows.append(sin_tok)
    cos = torch.tensor(cos_rows, dtype=DT)   # [S*H, Dh/2]
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
    # nn.Linear weight [out, in]; Mojo linear(x,W)=x@Wᵀ.
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
    # RAW modulation vectors (pre tanh / pre 1+). Kept small/centered.
    m = {}
    m["scale_msa"] = (fillc(D, 0.017, 0.20, 0.20)).requires_grad_(True)
    m["gate_msa"] = (fill(D, 0.011, 0.30, 0.40)).requires_grad_(True)
    m["scale_mlp"] = (fillc(D, 0.019, 0.50, 0.15)).requires_grad_(True)
    m["gate_mlp"] = (fill(D, 0.012, 0.60, 0.35)).requires_grad_(True)
    return m


def main():
    x = t2(S, D, 0.021, 0.05, 0.5).requires_grad_(True)
    w = make_weights(1)
    m = make_mod()

    cos, sin = build_real_rope_tables()    # HALF-WIDTH [S*H, Dh/2]
    # non-degeneracy: cos must vary across pairs (not a tautological table)
    dvar = float(cos.std())
    assert dvar > 1e-3, "rope table is degenerate (constant cos) — gate tautology"

    # ── forward ──
    # attention sub-block (sandwich norm)
    xn1 = rms_norm(x, w["n1"])
    xn1s = (1.0 + m["scale_msa"]) * xn1                  # scale modulation, no shift
    q = (xn1s @ w["wq"].T).reshape(1, S, H, Dh)
    k = (xn1s @ w["wk"].T).reshape(1, S, H, Dh)
    v = (xn1s @ w["wv"].T).reshape(1, S, H, Dh)
    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v).reshape(S, D)
    att_o = att @ w["wo"].T
    attn_n2 = rms_norm(att_o, w["n2"])
    gate_msa = torch.tanh(m["gate_msa"])
    h = residual_gate(x, gate_msa, attn_n2)

    # MLP sub-block (SwiGLU, sandwich norm)
    xfn1 = rms_norm(h, w["fn1"])
    xfn1s = (1.0 + m["scale_mlp"]) * xfn1
    g_pre = xfn1s @ w["w1"].T
    u = xfn1s @ w["w3"].T
    act = silu(g_pre) * u
    ff = act @ w["w2"].T
    ff_n2 = rms_norm(ff, w["fn2"])
    gate_mlp = torch.tanh(m["gate_mlp"])
    out = residual_gate(h, gate_mlp, ff_n2)

    # ── upstream grad (non-degenerate) ──
    d_out = t2(S, D, 0.027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    def W(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
        path = os.path.join(REF_DIR, name + ".bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, flat.shape)

    # forward output + grads
    W("ref_out", out)
    W("ref_d_x", x.grad)
    for kk in ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "n2", "fn1", "w1", "w3", "w2", "fn2"]:
        W("ref_d_%s" % kk, w[kk].grad)
    for kk in ["scale_msa", "gate_msa", "scale_mlp", "gate_mlp"]:
        W("ref_d_%s" % kk, m[kk].grad)

    # inputs the Mojo gate reconstructs
    W("in_x", x)
    for kk in ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "n2", "fn1", "w1", "w3", "w2", "fn2"]:
        W("in_w_%s" % kk, w[kk])
    for kk in ["scale_msa", "gate_msa", "scale_mlp", "gate_mlp"]:
        W("in_m_%s" % kk, m[kk])
    W("in_cos", cos)            # HALF-WIDTH [S*H, Dh/2] (used by fwd AND bwd)
    W("in_sin", sin)
    W("in_d_out", d_out)

    print("forward loss =", float(loss))
    print("rope cos std =", dvar)
    print("DONE")


if __name__ == "__main__":
    main()
