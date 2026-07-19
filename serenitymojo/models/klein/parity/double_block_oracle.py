#!/usr/bin/env python3
# serenitymojo/models/klein/parity/double_block_oracle.py
#
# Torch oracle for the Klein DOUBLE-STREAM DiT block (forward + autograd grads).
# Replicates the EXACT math of serenitymojo/models/klein/double_block.mojo's
# `double_block_forward`, which itself mirrors models/dit/klein_dit.mojo
# `_double_block` (lines 267-352). Produces .bin references that the Mojo gate
# (double_block_parity.mojo) reads and compares at cos >= 0.999.
#
# CONVENTIONS (must match the Mojo forward byte-for-byte):
#   modulate(x, scale, shift) = (1 + scale) * x + shift          (scale/shift [D])
#   residual_gate(x, gate, y) = x + gate * y                     (gate [D])
#   layer_norm: eps = 1e-6, weight = 1, bias = 0  (non-learnable)
#   rms_norm:   eps = 1e-6, over the last dim (Dh for q/k per head, D for n/a)
#   rope: INTERLEAVED (FLUX/Klein) pairing (2i, 2i+1)
#   sdpa: non-causal, scale = 1/sqrt(Dh)
#   joint concat order: txt FIRST, then img (axis = sequence)
#
# NON-DEGENERATE inputs: sinusoidal fills (NEVER modular (i*k)%9 — that aliases
# at real dims and zeros grads). Real Klein head count H = 32; small N/Dh to keep
# the oracle fast (D = H*Dh).
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/double_block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64  # F64 reference interior (gate compares cos in F64)

# ── dims (REAL Klein head count H=32; small N/Dh for a fast oracle) ──
H = 32
Dh = 16
D = H * Dh          # 512
N_IMG = 4
N_TXT = 2
S = N_TXT + N_IMG
F = 24              # mlp hidden (swiglu); gu projects to 2F
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


# ── per-stream weights ──
def make_stream(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    w = {}
    w["wqkv"] = rnd(3 * D, D)
    w["wproj"] = rnd(D, D)
    w["wgu"] = rnd(2 * F, D)
    w["wd"] = rnd(D, F)
    w["q_norm"] = (rnd(Dh) * 0.1 + 1.0)
    w["k_norm"] = (rnd(Dh) * 0.1 + 1.0)
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_mod(off):
    # 6 vectors each [D], non-degenerate sinusoidal
    m = {}
    m["shift1"] = fill(D, 0.013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale1"] = fillc(D, 0.017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate1"] = fill(D, 0.011, 0.3 + off, 0.4).requires_grad_(True)
    m["shift2"] = fillc(D, 0.019, 0.4 + off, 0.3).requires_grad_(True)
    m["scale2"] = fill(D, 0.015, 0.5 + off, 0.2).requires_grad_(True)
    m["gate2"] = fillc(D, 0.012, 0.6 + off, 0.4).requires_grad_(True)
    return m


def stream_forward(x, w, m, qkv_to_qkv, cos, sin):
    # returns q_rms, k_rms, v  (each [1,N,H,Dh]) for the joint attention
    N = x.shape[0]
    ln1 = layer_norm(x)
    norm = modulate(ln1, m["scale1"], m["shift1"])
    qkv = norm @ w["wqkv"].T   # [N, 3D]
    q = qkv[:, 0:D].reshape(1, N, H, Dh)
    k = qkv[:, D:2 * D].reshape(1, N, H, Dh)
    v = qkv[:, 2 * D:3 * D].reshape(1, N, H, Dh)
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])
    return q, k, v


def stream_post(x, att, w, m):
    # att [N,D]
    N = x.shape[0]
    out = att @ w["wproj"].T
    attn_res = residual_gate(x, m["gate1"], out)
    ln2 = layer_norm(attn_res)
    mlp_in = modulate(ln2, m["scale2"], m["shift2"])
    gu = mlp_in @ w["wgu"].T   # [N, 2F]
    gate = gu[:, 0:F]
    up = gu[:, F:2 * F]
    act = torch.nn.functional.silu(gate) * up
    mlp = act @ w["wd"].T
    final = residual_gate(attn_res, m["gate2"], mlp)
    return final


def main():
    # ── inputs (non-degenerate) ──
    img = t2(N_IMG, D, 0.021, 0.05, 0.5).requires_grad_(True)
    txt = t2(N_TXT, D, 0.023, 0.07, 0.5).requires_grad_(True)
    iw = make_stream(1)
    tw = make_stream(2)
    im = make_mod(0.0)
    tm = make_mod(1.0)

    # rope tables for the JOINT sequence [S*H, Dh/2]
    cos = t2(S * H, Dh // 2, 0.03, 0.2, 1.0)  # not unit but fine (table values)
    sin = t2(S * H, Dh // 2, 0.04, 0.5, 1.0)
    # make them a proper rotation magnitude (cos^2+sin^2 ~ varies; that's ok for
    # a parity test — the Mojo rope uses the SAME tables). Keep modest scale.
    cos = cos * 0.6
    sin = sin * 0.6

    # ── forward ──
    iq, ik, iv = stream_forward(img, iw, im, None, cos, sin)
    tq, tk, tv = stream_forward(txt, tw, tm, None, cos, sin)

    # joint concat: txt FIRST, then img (axis=1, the sequence)
    q = torch.cat([tq, iq], dim=1)   # [1,S,H,Dh]
    k = torch.cat([tk, ik], dim=1)
    v = torch.cat([tv, iv], dim=1)

    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v)            # [1,S,H,Dh]

    txt_att = att[:, 0:N_TXT].reshape(N_TXT, D)
    img_att = att[:, N_TXT:N_TXT + N_IMG].reshape(N_IMG, D)

    img_out = stream_post(img, img_att, iw, im)
    txt_out = stream_post(txt, txt_att, tw, tm)

    # ── upstream grads (non-degenerate sinusoidal) ──
    d_img = t2(N_IMG, D, 0.027, 0.11, 0.05)
    d_txt = t2(N_TXT, D, 0.029, 0.13, 0.05)

    loss = (img_out * d_img).sum() + (txt_out * d_txt).sum()
    loss.backward()

    # ── collect references ──
    def W(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
        path = os.path.join(REF_DIR, name + ".bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, flat.shape)

    # forward outputs (sanity)
    W("ref_img_out", img_out)
    W("ref_txt_out", txt_out)

    # input grads
    W("ref_d_img", img.grad)
    W("ref_d_txt", txt.grad)

    # img stream weight grads
    W("ref_img_d_wqkv", iw["wqkv"].grad)
    W("ref_img_d_wproj", iw["wproj"].grad)
    W("ref_img_d_wgu", iw["wgu"].grad)
    W("ref_img_d_wd", iw["wd"].grad)
    W("ref_img_d_qnorm", iw["q_norm"].grad)
    W("ref_img_d_knorm", iw["k_norm"].grad)
    # img mod-vec grads
    W("ref_img_d_shift1", im["shift1"].grad)
    W("ref_img_d_scale1", im["scale1"].grad)
    W("ref_img_d_gate1", im["gate1"].grad)
    W("ref_img_d_shift2", im["shift2"].grad)
    W("ref_img_d_scale2", im["scale2"].grad)
    W("ref_img_d_gate2", im["gate2"].grad)

    # txt stream weight grads
    W("ref_txt_d_wqkv", tw["wqkv"].grad)
    W("ref_txt_d_wproj", tw["wproj"].grad)
    W("ref_txt_d_wgu", tw["wgu"].grad)
    W("ref_txt_d_wd", tw["wd"].grad)
    W("ref_txt_d_qnorm", tw["q_norm"].grad)
    W("ref_txt_d_knorm", tw["k_norm"].grad)
    W("ref_txt_d_shift1", tm["shift1"].grad)
    W("ref_txt_d_scale1", tm["scale1"].grad)
    W("ref_txt_d_gate1", tm["gate1"].grad)
    W("ref_txt_d_shift2", tm["shift2"].grad)
    W("ref_txt_d_scale2", tm["scale2"].grad)
    W("ref_txt_d_gate2", tm["gate2"].grad)

    # also dump the exact INPUTS the Mojo gate must reconstruct (so both sides
    # use identical numbers). The gate regenerates these via the SAME fills, but
    # we dump the weights (randn) which the gate cannot regenerate identically.
    W("in_img", img)
    W("in_txt", txt)
    for nm, w in [("iw", iw), ("tw", tw)]:
        for kk in ["wqkv", "wproj", "wgu", "wd", "q_norm", "k_norm"]:
            W("in_%s_%s" % (nm, kk), w[kk])
    for nm, m in [("im", im), ("tm", tm)]:
        for kk in ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]:
            W("in_%s_%s" % (nm, kk), m[kk])
    W("in_cos", cos)
    W("in_sin", sin)
    W("in_d_img", d_img)
    W("in_d_txt", d_txt)

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
