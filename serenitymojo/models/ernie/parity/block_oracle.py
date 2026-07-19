#!/usr/bin/env python3
# serenitymojo/models/ernie/parity/block_oracle.py
#
# Torch oracle for the ERNIE-Image SINGLE-STREAM DiT block (forward + autograd
# grads). Replicates the EXACT math of serenitymojo/models/ernie/block.mojo's
# `ernie_block_forward`, which itself mirrors models/dit/ernie_image.mojo
# `block0_smoke_forward` and inference-flame ernie_image.rs block_forward_from_map.
# Produces .bin references the Mojo gate (block_parity.mojo) reads + compares at
# cos >= 0.999.
#
# CONVENTIONS (must match the Mojo forward byte-for-byte):
#   rms_norm(x, w): x / sqrt(mean(x^2, -1) + eps) * w            (eps 1e-6)
#   modulate(x, scale, shift) = (1 + scale) * x + shift          (scale/shift [D])
#   residual_gate(x, gate, y) = x + gate * y                     (gate [D])
#   sdpa: non-causal, scale = 1/sqrt(Dh)
#   GELU: tanh-approximation (matches Mojo ops/activations gelu + flame-core)
#   MLP:  gelu(gate_proj(x)) * up_proj(x) -> linear_fc2          (GELU-gated)
#   half-split RoPE (diffusers ErnieImageSingleStreamAttnProcessor.apply_rotary_emb):
#     out[i]      = x[i]*cos[i]      - x[i+half]*sin[i]
#     out[i+half] = x[i+half]*cos[i+half] + x[i]*sin[i+half]
#     cos/sin are FULL-WIDTH [rows, D] built from the REAL interleaved-doubled
#     3-axis angle table (ErnieImageEmbedND3: per-axis angles repeated CONSECUTIVELY
#     [θ0,θ0,θ1,θ1,...] then axis-concatenated 32|48|48). On this real table
#     cos[i] != cos[i+half], so the FORWARD reads both halves (rope_halfsplit_full)
#     and the BACKWARD must read both halves too (rope_halfsplit_full_backward).
#     The OLD half-width rope_backward(interleaved=False) aliased one angle per pair
#     and was WRONG here (block bug, fixed 2026-06-01 — see
#     ops/parity/rope_halfsplit_full_parity.mojo). This gate now builds the REAL
#     3-axis table + REAL row/col/text positions, so it can no longer be green on a
#     compensating degenerate table.
#
# Run (SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/ernie/parity/block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64  # F64 reference interior

# ── dims (REAL ERNIE head count H=32, head_dim Dh=128 -> hidden D=4096) ──
H = 32
Dh = 128
D = H * Dh          # 4096
# REAL 3-axis positional structure: image-first (row/col grid) then text.
IMG_H = 2           # image grid rows
IMG_W = 3           # image grid cols
N_IMG = IMG_H * IMG_W   # 6 image tokens
N_TXT = 2           # text tokens
S = N_IMG + N_TXT   # 8 (image-first/text-second)
TEXT_LEN_REAL = N_TXT
F = 96              # FFN hidden for the gate (small; real is 12288)
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
# REAL ERNIE RoPE axis dims (ERNIE_DIT_ROPE_AXIS_{0,1,2}) and theta.
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


def modulate(x, scale, shift):
    return (1.0 + scale) * x + shift


def residual_gate(x, gate, y):
    return x + gate * y


def gelu_tanh(x):
    # tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))


def rope_halfsplit_full(x, cos, sin):
    # x [1,S,H,Dh]; cos/sin FULL-WIDTH [S*H, Dh] (rows = S*H, flattened (s,h)).
    Sx = x.shape[1]
    half = Dh // 2
    cr = cos.reshape(Sx, H, Dh)
    sr = sin.reshape(Sx, H, Dh)
    x0 = x[..., 0:half]          # [1,S,H,half]
    x1 = x[..., half:Dh]         # [1,S,H,half]
    c0 = cr[..., 0:half]
    s0 = sr[..., 0:half]
    c1 = cr[..., half:Dh]
    s1 = sr[..., half:Dh]
    out = torch.empty_like(x)
    out[..., 0:half] = x0 * c0 - x1 * s0
    out[..., half:Dh] = x1 * c1 + x0 * s1
    return out


def _axis_inv_freqs(axis_dim, theta):
    # matches ernie_image.mojo _rope_inv_freqs: theta**(-(2k)/axis_dim), k in [0,axis_dim/2)
    return [theta ** (-(2.0 * k) / axis_dim) for k in range(axis_dim // 2)]


def build_real_rope_tables():
    # Replicates diffusers ErnieImageEmbedND3 + the Mojo build_ernie_rope_tables:
    # per token, per axis, angle = pos_axis * inv_freq[i]; each angle repeated
    # CONSECUTIVELY (interleaved doubling) then axes concatenated; broadcast over H.
    # Returns cos/sin FULL-WIDTH [S*H, Dh] (rows flattened in (token, head) order).
    f0 = _axis_inv_freqs(ROPE_AXES[0], ROPE_THETA)
    f1 = _axis_inv_freqs(ROPE_AXES[1], ROPE_THETA)
    f2 = _axis_inv_freqs(ROPE_AXES[2], ROPE_THETA)
    assert len(f0) + len(f1) + len(f2) == Dh // 2
    cos_rows = []
    sin_rows = []
    for tok in range(S):
        if tok < N_IMG:
            r = tok // IMG_W
            c = tok % IMG_W
            p0 = float(TEXT_LEN_REAL)
            p1 = float(r)
            p2 = float(c)
        else:
            p0 = float(tok - N_IMG)
            p1 = 0.0
            p2 = 0.0
        cos_tok = []
        sin_tok = []
        for (p, freqs) in ((p0, f0), (p1, f1), (p2, f2)):
            for fi in freqs:
                ang = p * fi
                cos_tok += [math.cos(ang), math.cos(ang)]   # consecutive double
                sin_tok += [math.sin(ang), math.sin(ang)]
        for _h in range(H):
            cos_rows.append(cos_tok)
            sin_rows.append(sin_tok)
    cos = torch.tensor(cos_rows, dtype=DT)   # [S*H, Dh]
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
    # nn.Linear weight [out, in]; Mojo linear(x,W)=x@Wᵀ. Scale small to keep
    # activations BF16-believable while non-degenerate.
    w["wq"] = rnd(D, D) * 0.02
    w["wk"] = rnd(D, D) * 0.02
    w["wv"] = rnd(D, D) * 0.02
    w["wo"] = rnd(D, D) * 0.02
    w["q_norm"] = (rnd(Dh) * 0.1 + 1.0)
    w["k_norm"] = (rnd(Dh) * 0.1 + 1.0)
    w["sa_norm"] = (rnd(D) * 0.1 + 1.0)
    w["mlp_norm"] = (rnd(D) * 0.1 + 1.0)
    w["wgate"] = rnd(F, D) * 0.02
    w["wup"] = rnd(F, D) * 0.02
    w["wdown"] = rnd(D, F) * 0.02
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_mod():
    m = {}
    m["shift_msa"] = (fill(D, 0.013, 0.10, 0.30)).requires_grad_(True)
    m["scale_msa"] = (fillc(D, 0.017, 0.20, 0.20)).requires_grad_(True)
    m["gate_msa"] = (fill(D, 0.011, 0.30, 0.40)).requires_grad_(True)
    m["shift_mlp"] = (fill(D, 0.015, 0.40, 0.25)).requires_grad_(True)
    m["scale_mlp"] = (fillc(D, 0.019, 0.50, 0.15)).requires_grad_(True)
    m["gate_mlp"] = (fill(D, 0.012, 0.60, 0.35)).requires_grad_(True)
    return m


def main():
    x = t2(S, D, 0.021, 0.05, 0.5).requires_grad_(True)
    w = make_weights(1)
    m = make_mod()

    # REAL interleaved-doubled 3-axis rope tables: FULL-WIDTH [S*H, Dh] where
    # cos[i] != cos[i+half] (the table the real ERNIE model produces).
    cos, sin = build_real_rope_tables()
    half = Dh // 2
    dbl = float((cos[:, :half] - cos[:, half:]).abs().max())
    assert dbl > 1e-3, "table is degenerate (cos[i]==cos[i+half]) — gate would be a tautology"

    # ── forward ──
    # self-attention sub-block
    sa_norm = rms_norm(x, w["sa_norm"])
    sa_in = modulate(sa_norm, m["scale_msa"], m["shift_msa"])
    q = (sa_in @ w["wq"].T).reshape(1, S, H, Dh)
    k = (sa_in @ w["wk"].T).reshape(1, S, H, Dh)
    v = (sa_in @ w["wv"].T).reshape(1, S, H, Dh)
    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    qr = rope_halfsplit_full(q, cos, sin)
    kr = rope_halfsplit_full(k, cos, sin)
    att = sdpa(qr, kr, v).reshape(S, D)
    att_out = att @ w["wo"].T
    h = residual_gate(x, m["gate_msa"], att_out)

    # MLP sub-block (GELU-gated)
    mlp_norm = rms_norm(h, w["mlp_norm"])
    mlp_in = modulate(mlp_norm, m["scale_mlp"], m["shift_mlp"])
    gate_pre = mlp_in @ w["wgate"].T
    up = mlp_in @ w["wup"].T
    activated = gelu_tanh(gate_pre) * up
    mlp_out = activated @ w["wdown"].T
    out = residual_gate(h, m["gate_mlp"], mlp_out)

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
    for kk in ["wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "sa_norm", "mlp_norm", "wgate", "wup", "wdown"]:
        W("ref_d_%s" % kk, w[kk].grad)
    for kk in ["shift_msa", "scale_msa", "gate_msa",
               "shift_mlp", "scale_mlp", "gate_mlp"]:
        W("ref_d_%s" % kk, m[kk].grad)

    # inputs the Mojo gate reconstructs
    W("in_x", x)
    for kk in ["wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "sa_norm", "mlp_norm", "wgate", "wup", "wdown"]:
        W("in_w_%s" % kk, w[kk])
    for kk in ["shift_msa", "scale_msa", "gate_msa",
               "shift_mlp", "scale_mlp", "gate_mlp"]:
        W("in_m_%s" % kk, m[kk])
    W("in_cos", cos)            # FULL-WIDTH [S*H, Dh] (used by fwd AND bwd now)
    W("in_sin", sin)
    W("in_d_out", d_out)

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
