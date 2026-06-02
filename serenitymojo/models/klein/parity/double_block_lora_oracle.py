#!/usr/bin/env python3
# serenitymojo/models/klein/parity/double_block_lora_oracle.py
#
# Torch oracle for the Klein DOUBLE-STREAM DiT block WITH LoRA on the attention
# projections (wqkv + wproj, per stream). Replicates the EXACT math of
# serenitymojo/models/klein/double_block.mojo's `double_block_lora_forward`
# (= the base block with a LoRA delta scale*((x@Aᵀ)@Bᵀ) on the two attention
# linears of each stream). Produces .bin references that the Mojo gate
# (double_block_lora_parity.mojo) reads and compares at cos >= 0.999, INCLUDING
# d_A and d_B for every adapter.
#
# LoRA math (matches train_step.mojo / lora_block.mojo):
#   y' = linear(x, W) + scale*((x @ Aᵀ) @ Bᵀ),  A=[rank,in], B=[out,rank].
# Targets:
#   wqkv (FULL [3D,D]): in=D, out=3D, input = norm.
#   wproj(FULL [D, D]): in=D, out=D,  input = att (per-stream attention slice).
#
# NON-DEGENERATE inputs (sinusoidal/randn; A AND B nonzero so both d_A and d_B
# are exercised). Real Klein head count H = 32; small N/Dh for a fast oracle.
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/double_block_lora_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

H = 32
Dh = 16
D = H * Dh          # 512
N_IMG = 4
N_TXT = 2
S = N_TXT + N_IMG
F = 24
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)

RANK = 8
ALPHA = 16.0
LSCALE = ALPHA / RANK   # 2.0

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def fillc(n, a, b, c):
    return torch.tensor([math.cos(a * i + b) * c for i in range(n)], dtype=DT)


def t2(n, m, a, b, c):
    return fill(n * m, a, b, c).reshape(n, m)


def layer_norm(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS)


def modulate(x, scale, shift):
    return (1.0 + scale) * x + shift


def residual_gate(x, gate, y):
    return x + gate * y


def rms_norm_lastdim(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def rope_interleaved(x, cos, sin):
    Sx = x.shape[1]
    cr = cos.reshape(Sx, H, Dh // 2)
    sr = sin.reshape(Sx, H, Dh // 2)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    o0 = x0 * cr - x1 * sr
    o1 = x0 * sr + x1 * cr
    out = torch.empty_like(x)
    out[..., 0::2] = o0
    out[..., 1::2] = o1
    return out


def sdpa(q, k, v):
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)


def lora_delta(x, A, B):
    # x [N,in]; A [rank,in]; B [out,rank] -> [N,out]
    return LSCALE * ((x @ A.T) @ B.T)


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


def make_lora(seed):
    # A small randn, B small randn (NON-degenerate; B != 0 so d_A is exercised).
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    lo = {}
    lo["qkv_A"] = (rnd(RANK, D) * 0.02).requires_grad_(True)
    lo["qkv_B"] = (rnd(3 * D, RANK) * 0.02).requires_grad_(True)
    lo["proj_A"] = (rnd(RANK, D) * 0.02).requires_grad_(True)
    lo["proj_B"] = (rnd(D, RANK) * 0.02).requires_grad_(True)
    return lo


def make_mod(off):
    m = {}
    m["shift1"] = fill(D, 0.013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale1"] = fillc(D, 0.017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate1"] = fill(D, 0.011, 0.3 + off, 0.4).requires_grad_(True)
    m["shift2"] = fillc(D, 0.019, 0.4 + off, 0.3).requires_grad_(True)
    m["scale2"] = fill(D, 0.015, 0.5 + off, 0.2).requires_grad_(True)
    m["gate2"] = fillc(D, 0.012, 0.6 + off, 0.4).requires_grad_(True)
    return m


def stream_forward(x, w, m, lo, cos, sin):
    N = x.shape[0]
    ln1 = layer_norm(x)
    norm = modulate(ln1, m["scale1"], m["shift1"])
    qkv = norm @ w["wqkv"].T + lora_delta(norm, lo["qkv_A"], lo["qkv_B"])  # [N,3D]
    q = qkv[:, 0:D].reshape(1, N, H, Dh)
    k = qkv[:, D:2 * D].reshape(1, N, H, Dh)
    v = qkv[:, 2 * D:3 * D].reshape(1, N, H, Dh)
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])
    return q, k, v


def stream_post(x, att, w, m, lo):
    N = x.shape[0]
    out = att @ w["wproj"].T + lora_delta(att, lo["proj_A"], lo["proj_B"])  # [N,D]
    attn_res = residual_gate(x, m["gate1"], out)
    ln2 = layer_norm(attn_res)
    mlp_in = modulate(ln2, m["scale2"], m["shift2"])
    gu = mlp_in @ w["wgu"].T
    gate = gu[:, 0:F]
    up = gu[:, F:2 * F]
    act = torch.nn.functional.silu(gate) * up
    mlp = act @ w["wd"].T
    final = residual_gate(attn_res, m["gate2"], mlp)
    return final


def main():
    img = t2(N_IMG, D, 0.021, 0.05, 0.5).requires_grad_(True)
    txt = t2(N_TXT, D, 0.023, 0.07, 0.5).requires_grad_(True)
    iw = make_stream(1)
    tw = make_stream(2)
    ilo = make_lora(11)
    tlo = make_lora(22)
    im = make_mod(0.0)
    tm = make_mod(1.0)

    cos = t2(S * H, Dh // 2, 0.03, 0.2, 1.0) * 0.6
    sin = t2(S * H, Dh // 2, 0.04, 0.5, 1.0) * 0.6

    iq, ik, iv = stream_forward(img, iw, im, ilo, cos, sin)
    tq, tk, tv = stream_forward(txt, tw, tm, tlo, cos, sin)

    q = torch.cat([tq, iq], dim=1)
    k = torch.cat([tk, ik], dim=1)
    v = torch.cat([tv, iv], dim=1)

    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v)

    txt_att = att[:, 0:N_TXT].reshape(N_TXT, D)
    img_att = att[:, N_TXT:N_TXT + N_IMG].reshape(N_IMG, D)

    img_out = stream_post(img, img_att, iw, im, ilo)
    txt_out = stream_post(txt, txt_att, tw, tm, tlo)

    d_img = t2(N_IMG, D, 0.027, 0.11, 0.05)
    d_txt = t2(N_TXT, D, 0.029, 0.13, 0.05)

    loss = (img_out * d_img).sum() + (txt_out * d_txt).sum()
    loss.backward()

    def W(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
        path = os.path.join(REF_DIR, name + ".bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, flat.shape)

    # forward outputs (sanity)
    W("lref_img_out", img_out)
    W("lref_txt_out", txt_out)

    # base input grads + a couple base weight grads (no-regression check)
    W("lref_d_img", img.grad)
    W("lref_d_txt", txt.grad)
    W("lref_img_d_wgu", iw["wgu"].grad)
    W("lref_img_d_wd", iw["wd"].grad)
    W("lref_txt_d_wgu", tw["wgu"].grad)
    W("lref_txt_d_wd", tw["wd"].grad)

    # LoRA grads (the deliverable)
    W("lref_img_qkv_dA", ilo["qkv_A"].grad)
    W("lref_img_qkv_dB", ilo["qkv_B"].grad)
    W("lref_img_proj_dA", ilo["proj_A"].grad)
    W("lref_img_proj_dB", ilo["proj_B"].grad)
    W("lref_txt_qkv_dA", tlo["qkv_A"].grad)
    W("lref_txt_qkv_dB", tlo["qkv_B"].grad)
    W("lref_txt_proj_dA", tlo["proj_A"].grad)
    W("lref_txt_proj_dB", tlo["proj_B"].grad)

    # exact INPUTS the Mojo gate reconstructs (weights/lora cannot be regenerated)
    W("lin_img", img)
    W("lin_txt", txt)
    for nm, w in [("iw", iw), ("tw", tw)]:
        for kk in ["wqkv", "wproj", "wgu", "wd", "q_norm", "k_norm"]:
            W("lin_%s_%s" % (nm, kk), w[kk])
    for nm, lo in [("ilo", ilo), ("tlo", tlo)]:
        for kk in ["qkv_A", "qkv_B", "proj_A", "proj_B"]:
            W("lin_%s_%s" % (nm, kk), lo[kk])
    for nm, m in [("im", im), ("tm", tm)]:
        for kk in ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]:
            W("lin_%s_%s" % (nm, kk), m[kk])
    W("lin_cos", cos)
    W("lin_sin", sin)
    W("lin_d_img", d_img)
    W("lin_d_txt", d_txt)

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
