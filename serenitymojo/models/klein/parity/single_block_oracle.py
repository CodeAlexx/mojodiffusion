#!/usr/bin/env python3
# serenitymojo/models/klein/parity/single_block_oracle.py
#
# Torch oracle for the Klein SINGLE-STREAM DiT block (forward + autograd grads).
# Replicates the EXACT math of serenitymojo/models/klein/single_block.mojo's
# `single_block_forward`, which itself mirrors models/dit/klein_dit.mojo
# `_single_block` (lines 354-390). Produces .bin references that the Mojo gate
# (single_block_parity.mojo) reads and compares at cos >= 0.999.
#
# CONVENTIONS (must match the Mojo forward byte-for-byte):
#   modulate(x, scale, shift) = (1 + scale) * x + shift          (scale/shift [D])
#   residual_gate(x, gate, y) = x + gate * y                     (gate [D])
#   layer_norm: eps = 1e-6, weight = 1, bias = 0  (non-learnable)
#   rms_norm:   eps = 1e-6, over the last dim Dh (per head)
#   rope: INTERLEAVED (FLUX/Klein) pairing (2i, 2i+1)
#   sdpa: non-causal, scale = 1/sqrt(Dh)
#   fused = linear(x_norm, W1); W1 [3D+2F, D]
#     qkv = fused[:, :3D] ; gate_up = fused[:, 3D:3D+2F]         (CHANNEL slice)
#   out_in = concat([att_flat, mlp], axis=channel)  -> [S, D+F]
#   out = linear(out_in, W2); W2 [D, D+F]
#
# NON-DEGENERATE inputs: sinusoidal fills (NEVER modular (i*k)%9 — that aliases
# at real dims and zeros grads). Real Klein head count H = 32; small N/Dh to keep
# the oracle fast (D = H*Dh).
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/single_block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64  # F64 reference interior (gate compares cos in F64)

# ── dims (REAL Klein head count H=32; small S/Dh for a fast oracle) ──
H = 32
Dh = 16
D = H * Dh          # 512
S = 6
F = 24              # mlp hidden (swiglu); gate_up projects to 2F
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)

REF_DIR = os.path.dirname(os.path.abspath(__file__))


# ── non-degenerate sinusoidal fills ──
def fill(n, a, b, c):
    return torch.tensor(
        [math.sin(a * i + b) * c for i in range(n)], dtype=DT
    )


def fillc(n, a, b, c):
    return torch.tensor(
        [math.cos(a * i + b) * c for i in range(n)], dtype=DT
    )


def t2(n, m, a, b, c):
    return fill(n * m, a, b, c).reshape(n, m)


# ── ops (match the Mojo forward exactly) ──
def layer_norm(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS)


def modulate(x, scale, shift):
    return (1.0 + scale) * x + shift


def residual_gate(x, gate, y):
    return x + gate * y


def rms_norm_lastdim(x, weight):
    # x [..., d]; weight [d]
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def rope_interleaved(x, cos, sin):
    # x [1, S, H, Dh]; cos/sin [S*H, Dh/2] (flat per (s,h) row).
    Sx = x.shape[1]
    cr = cos.reshape(Sx, H, Dh // 2)
    sr = sin.reshape(Sx, H, Dh // 2)
    x0 = x[..., 0::2]   # even -> [1,S,H,Dh/2]
    x1 = x[..., 1::2]   # odd
    o0 = x0 * cr - x1 * sr
    o1 = x0 * sr + x1 * cr
    out = torch.empty_like(x)
    out[..., 0::2] = o0
    out[..., 1::2] = o1
    return out


def sdpa(q, k, v):
    # q,k,v [1,S,H,Dh] -> [1,H,S,Dh]
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh   # [1,H,S,Dh]
    return out.permute(0, 2, 1, 3)  # [1,S,H,Dh]


# ── weights ──
def make_weights(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    w = {}
    w["w1"] = rnd(3 * D + 2 * F, D)
    w["w2"] = rnd(D, D + F)
    w["q_norm"] = (rnd(Dh) * 0.1 + 1.0)
    w["k_norm"] = (rnd(Dh) * 0.1 + 1.0)
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_mod(off):
    m = {}
    m["shift"] = fill(D, 0.013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale"] = fillc(D, 0.017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate"] = fill(D, 0.011, 0.3 + off, 0.4).requires_grad_(True)
    return m


def main():
    # ── inputs (non-degenerate) ──
    x = t2(S, D, 0.021, 0.05, 0.5).requires_grad_(True)
    w = make_weights(1)
    m = make_mod(0.0)

    # rope tables for the sequence [S*H, Dh/2]
    cos = t2(S * H, Dh // 2, 0.03, 0.2, 1.0) * 0.6
    sin = t2(S * H, Dh // 2, 0.04, 0.5, 1.0) * 0.6

    # ── forward ──
    ln = layer_norm(x)
    norm = modulate(ln, m["scale"], m["shift"])
    fused = norm @ w["w1"].T              # [S, 3D+2F]
    qkv = fused[:, 0:3 * D]               # [S, 3D]
    gate_up = fused[:, 3 * D:3 * D + 2 * F]   # [S, 2F]

    q = qkv[:, 0:D].reshape(1, S, H, Dh)
    k = qkv[:, D:2 * D].reshape(1, S, H, Dh)
    v = qkv[:, 2 * D:3 * D].reshape(1, S, H, Dh)
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])

    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v)                 # [1,S,H,Dh]
    att_flat = att.reshape(S, D)

    mlp_gate = gate_up[:, 0:F]
    mlp_up = gate_up[:, F:2 * F]
    mlp = torch.nn.functional.silu(mlp_gate) * mlp_up   # [S,F]

    out_in = torch.cat([att_flat, mlp], dim=1)   # [S, D+F]  CHANNEL concat
    out_proj = out_in @ w["w2"].T                # [S, D]
    out = residual_gate(x, m["gate"], out_proj)

    # ── upstream grad (non-degenerate sinusoidal) ──
    d_out = t2(S, D, 0.027, 0.11, 0.05)

    loss = (out * d_out).sum()
    loss.backward()

    # ── collect references ──
    def W(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
        path = os.path.join(REF_DIR, name + ".bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, flat.shape)

    # forward output (sanity)
    W("ref_out", out)

    # input grad
    W("ref_d_x", x.grad)

    # weight grads
    W("ref_d_w1", w["w1"].grad)
    W("ref_d_w2", w["w2"].grad)
    W("ref_d_qnorm", w["q_norm"].grad)
    W("ref_d_knorm", w["k_norm"].grad)
    # mod-vec grads
    W("ref_d_shift", m["shift"].grad)
    W("ref_d_scale", m["scale"].grad)
    W("ref_d_gate", m["gate"].grad)

    # ── dump exact INPUTS the Mojo gate must reconstruct ──
    W("in_x", x)
    for kk in ["w1", "w2", "q_norm", "k_norm"]:
        W("in_w_%s" % kk, w[kk])
    for kk in ["shift", "scale", "gate"]:
        W("in_m_%s" % kk, m[kk])
    W("in_cos", cos)
    W("in_sin", sin)
    W("in_d_out", d_out)

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
