#!/usr/bin/env python3
# serenitymojo/models/qwenimage/parity/qwenimage_block_lora_oracle.py
#
# Torch oracle for the Qwen-Image MMDiT double-stream block WITH LoRA on all 12
# projections (img/txt x q/k/v/out/ff_up/ff_down). Mirrors
# qwenimage_block.mojo::double_block_lora_forward/backward. Produces .bin refs
# for d_A/d_B of every adapter (+ d_img/d_txt input grads) the Mojo gate compares.
#
# LoRA: y' = linear(x,W,b) + scale*((x @ A.T) @ B.T), A=[rank,in], B=[out,rank],
#   scale = alpha/rank. (Matches train_step._lora_fwd / klein_lora_fwd.)
#
# Run (SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/qwenimage/parity/qwenimage_block_lora_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

H = 24
Dh = 16
D = H * Dh          # 384
N_IMG = 4
N_TXT = 3
S = N_TXT + N_IMG
F = 40
RANK = 8
ALPHA = 8.0
SCALE_LORA = ALPHA / RANK
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)

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


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))


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


def make_stream(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    w = {}
    w["wq"] = rnd(D, D); w["bq"] = rnd(D) * 0.1
    w["wk"] = rnd(D, D); w["bk"] = rnd(D) * 0.1
    w["wv"] = rnd(D, D); w["bv"] = rnd(D) * 0.1
    w["wout"] = rnd(D, D); w["bout"] = rnd(D) * 0.1
    w["wup"] = rnd(F, D); w["bup"] = rnd(F) * 0.1
    w["wdn"] = rnd(D, F); w["bdn"] = rnd(D) * 0.1
    w["q_norm"] = (rnd(Dh) * 0.1 + 1.0)
    w["k_norm"] = (rnd(Dh) * 0.1 + 1.0)
    # base weights frozen (no grad)
    return w


def make_lora(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape, s=0.02):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * s).to(DT)
    # in/out per projection: q/k/v/out: in=D,out=D ; ff_up: in=D,out=F ; ff_down: in=F,out=D
    spec = {
        "q": (D, D), "k": (D, D), "v": (D, D), "out": (D, D),
        "ff_up": (D, F), "ff_down": (F, D),
    }
    lo = {}
    for name, (inf, outf) in spec.items():
        A = rnd(RANK, inf, s=0.02).requires_grad_(True)
        # B nonzero so d_A is exercised (PEFT would init B=0; parity needs nonzero)
        B = rnd(outf, RANK, s=0.02).requires_grad_(True)
        lo[name] = (A, B, inf, outf)
    return lo


def lora_delta(x, A, B):
    # x [M,in] -> (x @ A.T) @ B.T * scale
    return SCALE_LORA * ((x @ A.T) @ B.T)


def make_mod(off):
    m = {}
    m["shift1"] = fill(D, 0.013, 0.1 + off, 0.3)
    m["scale1"] = fillc(D, 0.017, 0.2 + off, 0.2)
    m["gate1"] = fill(D, 0.011, 0.3 + off, 0.4)
    m["shift2"] = fillc(D, 0.019, 0.4 + off, 0.3)
    m["scale2"] = fill(D, 0.015, 0.5 + off, 0.2)
    m["gate2"] = fillc(D, 0.012, 0.6 + off, 0.4)
    return m


def stream_pre(x, w, m, lo):
    N = x.shape[0]
    ln1 = layer_norm(x)
    normed = modulate(ln1, m["scale1"], m["shift1"])
    q = (normed @ w["wq"].T + w["bq"]) + lora_delta(normed, lo["q"][0], lo["q"][1])
    k = (normed @ w["wk"].T + w["bk"]) + lora_delta(normed, lo["k"][0], lo["k"][1])
    v = (normed @ w["wv"].T + w["bv"]) + lora_delta(normed, lo["v"][0], lo["v"][1])
    q = q.reshape(1, N, H, Dh)
    k = k.reshape(1, N, H, Dh)
    v = v.reshape(1, N, H, Dh)
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])
    return q, k, v


def stream_post(x, att, w, m, lo):
    out = (att @ w["wout"].T + w["bout"]) + lora_delta(att, lo["out"][0], lo["out"][1])
    attn_res = residual_gate(x, m["gate1"], out)
    ln2 = layer_norm(attn_res)
    ff_in = modulate(ln2, m["scale2"], m["shift2"])
    ff_up = (ff_in @ w["wup"].T + w["bup"]) + lora_delta(ff_in, lo["ff_up"][0], lo["ff_up"][1])
    ff_act = gelu_tanh(ff_up)
    ff_down = (ff_act @ w["wdn"].T + w["bdn"]) + lora_delta(ff_act, lo["ff_down"][0], lo["ff_down"][1])
    final = residual_gate(attn_res, m["gate2"], ff_down)
    return final


def main():
    img = t2(N_IMG, D, 0.021, 0.05, 0.5).requires_grad_(True)
    txt = t2(N_TXT, D, 0.023, 0.07, 0.5).requires_grad_(True)
    iw = make_stream(1)
    tw = make_stream(2)
    im = make_mod(0.0)
    tm = make_mod(1.0)
    ilo = make_lora(11)
    tlo = make_lora(12)

    cos = t2(S * H, Dh // 2, 0.03, 0.2, 1.0) * 0.6
    sin = t2(S * H, Dh // 2, 0.04, 0.5, 1.0) * 0.6

    iq, ik, iv = stream_pre(img, iw, im, ilo)
    tq, tk, tv = stream_pre(txt, tw, tm, tlo)
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

    W("lref_d_img", img.grad)
    W("lref_d_txt", txt.grad)
    LKEYS = ["q", "k", "v", "out", "ff_up", "ff_down"]
    for nm, lo in [("img", ilo), ("txt", tlo)]:
        for kk in LKEYS:
            W("lref_%s_%s_dA" % (nm, kk), lo[kk][0].grad)
            W("lref_%s_%s_dB" % (nm, kk), lo[kk][1].grad)

    # dump inputs (base weights + lora A/B + mod + cos/sin + upstream grads)
    W("lin_img", img)
    W("lin_txt", txt)
    WKEYS = ["wq", "wk", "wv", "bq", "bk", "bv", "wout", "bout",
             "wup", "bup", "wdn", "bdn", "q_norm", "k_norm"]
    for nm, w in [("iw", iw), ("tw", tw)]:
        for kk in WKEYS:
            W("lin_%s_%s" % (nm, kk), w[kk])
    MKEYS = ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]
    for nm, m in [("im", im), ("tm", tm)]:
        for kk in MKEYS:
            W("lin_%s_%s" % (nm, kk), m[kk])
    for nm, lo in [("ilo", ilo), ("tlo", tlo)]:
        for kk in LKEYS:
            W("lin_%s_%s_A" % (nm, kk), lo[kk][0])
            W("lin_%s_%s_B" % (nm, kk), lo[kk][1])
    W("lin_cos", cos)
    W("lin_sin", sin)
    W("lin_d_img", d_img)
    W("lin_d_txt", d_txt)

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
