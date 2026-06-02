#!/usr/bin/env python3
# serenitymojo/models/klein/parity/klein_stack_oracle.py
#
# Torch oracle for the Klein FULL DiT STACK (small depth) forward + autograd.
# Replicates the EXACT math of serenitymojo/models/klein/klein_stack.mojo's
# klein_stack_forward/backward, which composes the verified double/single blocks
# (whose per-block math is already gated cos>=0.999) into the full graph:
#   img = linear(img_tokens, img_in) ; txt = linear(txt_tokens, txt_in)
#   for bi: x = double_block(...) ; (txt,img) = re-split
#   x = concat(txt, img)
#   for bi: x = single_block(...)
#   img_out = slice(x, txt:); normed = modulate(layer_norm(img_out), scale, shift)
#   out = linear(normed, final_lin)
#
# This proves the COMPOSITION SURFACE: input projection, the double->single
# transition (concat/slice), the final layer, and the d_x->d_y inter-block
# handoff across DEPTH (num_double=2, num_single=2). The individual blocks are
# already proven; this gate proves they STACK.
#
# CONVENTIONS (must match the Mojo forward byte-for-byte):
#   modulate(x, scale, shift) = (1 + scale) * x + shift
#   residual_gate(x, gate, y) = x + gate * y
#   layer_norm: eps = 1e-6, weight = 1, bias = 0  (non-learnable)
#   rms_norm:   eps = 1e-6, over the last dim (Dh per head)
#   rope: INTERLEAVED (FLUX/Klein) pairing (2i, 2i+1)
#   sdpa: non-causal, scale = 1/sqrt(Dh)
#   joint concat order: txt FIRST, then img (axis = sequence)
#   single block: channel-axis qkv|gate_up split of fused; channel concat att|mlp.
#
# NON-DEGENERATE inputs: sinusoidal/random fills (NEVER modular aliasing).
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/klein_stack_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── dims (small; H is the real-style head count, kept small for a fast oracle) ─
H = 4
Dh = 8
D = H * Dh           # 32
N_IMG = 4
N_TXT = 2
S = N_TXT + N_IMG
F = 24               # mlp hidden
IN_CH = 10           # img token channels
TXT_CH = 14          # txt token channels
OUT_CH = 6           # final output channels
NUM_DOUBLE = 2
NUM_SINGLE = 2
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def fillc(n, a, b, c):
    return torch.tensor([math.cos(a * i + b) * c for i in range(n)], dtype=DT)


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


# ── double-block weights / mod ──
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
    m = {}
    m["shift1"] = fill(D, 0.013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale1"] = fillc(D, 0.017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate1"] = fill(D, 0.011, 0.3 + off, 0.4).requires_grad_(True)
    m["shift2"] = fillc(D, 0.019, 0.4 + off, 0.3).requires_grad_(True)
    m["scale2"] = fill(D, 0.015, 0.5 + off, 0.2).requires_grad_(True)
    m["gate2"] = fillc(D, 0.012, 0.6 + off, 0.4).requires_grad_(True)
    return m


def make_single(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    w = {}
    w["w1"] = rnd(3 * D + 2 * F, D) * 0.05
    w["w2"] = rnd(D, D + F) * 0.05
    w["q_norm"] = (rnd(Dh) * 0.1 + 1.0)
    w["k_norm"] = (rnd(Dh) * 0.1 + 1.0)
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_single_mod(off):
    m = {}
    m["shift"] = fill(D, 0.013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale"] = fillc(D, 0.017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate"] = fill(D, 0.011, 0.3 + off, 0.4).requires_grad_(True)
    return m


def double_block(img, txt, iw, tw, im, tm, cos, sin):
    def stream_pre(x, w, m):
        N = x.shape[0]
        ln1 = layer_norm(x)
        norm = modulate(ln1, m["scale1"], m["shift1"])
        qkv = norm @ w["wqkv"].T
        q = qkv[:, 0:D].reshape(1, N, H, Dh)
        k = qkv[:, D:2 * D].reshape(1, N, H, Dh)
        v = qkv[:, 2 * D:3 * D].reshape(1, N, H, Dh)
        q = rms_norm_lastdim(q, w["q_norm"])
        k = rms_norm_lastdim(k, w["k_norm"])
        return q, k, v

    iq, ik, iv = stream_pre(img, iw, im)
    tq, tk, tv = stream_pre(txt, tw, tm)
    q = torch.cat([tq, iq], dim=1)
    k = torch.cat([tk, ik], dim=1)
    v = torch.cat([tv, iv], dim=1)
    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v)
    txt_att = att[:, 0:N_TXT].reshape(N_TXT, D)
    img_att = att[:, N_TXT:N_TXT + N_IMG].reshape(N_IMG, D)

    def stream_post(x, att, w, m):
        out = att @ w["wproj"].T
        attn_res = residual_gate(x, m["gate1"], out)
        ln2 = layer_norm(attn_res)
        mlp_in = modulate(ln2, m["scale2"], m["shift2"])
        gu = mlp_in @ w["wgu"].T
        gate = gu[:, 0:F]
        up = gu[:, F:2 * F]
        act = torch.nn.functional.silu(gate) * up
        mlp = act @ w["wd"].T
        return residual_gate(attn_res, m["gate2"], mlp)

    img_out = stream_post(img, img_att, iw, im)
    txt_out = stream_post(txt, txt_att, tw, tm)
    return img_out, txt_out


def single_block(x, w, m, cos, sin):
    Sx = x.shape[0]
    ln = layer_norm(x)
    norm = modulate(ln, m["scale"], m["shift"])
    fused = norm @ w["w1"].T                       # [S, 3D+2F]
    qkv = fused[:, 0:3 * D]
    gate_up = fused[:, 3 * D:3 * D + 2 * F]
    q = qkv[:, 0:D].reshape(1, Sx, H, Dh)
    k = qkv[:, D:2 * D].reshape(1, Sx, H, Dh)
    v = qkv[:, 2 * D:3 * D].reshape(1, Sx, H, Dh)
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])
    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v).reshape(Sx, D)
    mlp_gate = gate_up[:, 0:F]
    mlp_up = gate_up[:, F:2 * F]
    mlp = torch.nn.functional.silu(mlp_gate) * mlp_up
    out_in = torch.cat([att, mlp], dim=1)          # [S, D+F]
    out = out_in @ w["w2"].T                       # [S, D]
    return residual_gate(x, m["gate"], out)


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, flat.shape)


def main():
    # ── frozen base weights ──
    img_in = (t2(D, IN_CH, 0.021, 0.05, 0.3)).requires_grad_(True)     # [D, in_ch]
    txt_in = (t2(D, TXT_CH, 0.019, 0.07, 0.3)).requires_grad_(True)    # [D, txt_ch]
    final_lin = (t2(OUT_CH, D, 0.023, 0.09, 0.3)).requires_grad_(True) # [out_ch, D]
    final_shift = fill(D, 0.011, 0.13, 0.2).requires_grad_(True)
    final_scale = fillc(D, 0.012, 0.17, 0.2).requires_grad_(True)

    # ── tokens (non-degenerate) ──
    img_tokens = t2(N_IMG, IN_CH, 0.031, 0.05, 0.5).requires_grad_(True)
    txt_tokens = t2(N_TXT, TXT_CH, 0.033, 0.07, 0.5).requires_grad_(True)

    # ── per-block weights ──
    dbl = []
    for bi in range(NUM_DOUBLE):
        dbl.append((make_stream(100 + bi * 2), make_stream(101 + bi * 2)))
    sgl = []
    for bi in range(NUM_SINGLE):
        sgl.append(make_single(200 + bi))

    # ── SHARED modulation vectors (one img, one txt, one single) ──
    im = make_mod(0.0)
    tm = make_mod(1.0)
    sm = make_single_mod(2.0)

    # ── rope tables (joint sequence) ──
    cos = t2(S * H, Dh // 2, 0.03, 0.2, 1.0) * 0.6
    sin = t2(S * H, Dh // 2, 0.04, 0.5, 1.0) * 0.6

    # ── forward ──
    img = img_tokens @ img_in.T      # [N_IMG, D]
    txt = txt_tokens @ txt_in.T      # [N_TXT, D]
    for bi in range(NUM_DOUBLE):
        iw, tw = dbl[bi]
        img, txt = double_block(img, txt, iw, tw, im, tm, cos, sin)
    x = torch.cat([txt, img], dim=0)  # [S, D]   (txt FIRST)
    for bi in range(NUM_SINGLE):
        x = single_block(x, sgl[bi], sm, cos, sin)
    img_out = x[N_TXT:N_TXT + N_IMG]  # [N_IMG, D]
    normed = modulate(layer_norm(img_out), final_scale, final_shift)
    out = normed @ final_lin.T        # [N_IMG, out_ch]

    # ── upstream grad ──
    d_out = t2(N_IMG, OUT_CH, 0.027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    # forward output (sanity)
    W("ref_out", out)
    # load-bearing input-token grads
    W("ref_d_img_tokens", img_tokens.grad)
    W("ref_d_txt_tokens", txt_tokens.grad)
    # sample of base-weight grads (proves the proj + final layer backward)
    W("ref_d_img_in", img_in.grad)
    W("ref_d_txt_in", txt_in.grad)
    W("ref_d_final_lin", final_lin.grad)
    W("ref_d_final_shift", final_shift.grad)
    W("ref_d_final_scale", final_scale.grad)
    # sample of per-block weight grads: deepest double (bi=0) and last single.
    iw0, tw0 = dbl[0]
    W("ref_d0_img_wqkv", iw0["wqkv"].grad)
    W("ref_d0_img_wproj", iw0["wproj"].grad)
    W("ref_d0_txt_wqkv", tw0["wqkv"].grad)
    iwL, twL = dbl[NUM_DOUBLE - 1]
    W("ref_dL_img_wqkv", iwL["wqkv"].grad)
    W("ref_s0_w1", sgl[0]["w1"].grad)
    W("ref_s0_w2", sgl[0]["w2"].grad)
    W("ref_sL_w1", sgl[NUM_SINGLE - 1]["w1"].grad)
    # shared modvec grads (summed across the blocks that reuse them)
    W("ref_d_img_mod", torch.cat([
        im["shift1"].grad, im["scale1"].grad, im["gate1"].grad,
        im["shift2"].grad, im["scale2"].grad, im["gate2"].grad]))
    W("ref_d_txt_mod", torch.cat([
        tm["shift1"].grad, tm["scale1"].grad, tm["gate1"].grad,
        tm["shift2"].grad, tm["scale2"].grad, tm["gate2"].grad]))
    W("ref_d_single_mod", torch.cat([
        sm["shift"].grad, sm["scale"].grad, sm["gate"].grad]))

    # ── dump all INPUTS the Mojo gate cannot regenerate identically ──
    W("in_img_in", img_in); W("in_txt_in", txt_in)
    W("in_final_lin", final_lin); W("in_final_shift", final_shift)
    W("in_final_scale", final_scale)
    W("in_img_tokens", img_tokens); W("in_txt_tokens", txt_tokens)
    W("in_cos", cos); W("in_sin", sin)
    W("in_d_out", d_out)
    for bi in range(NUM_DOUBLE):
        iw, tw = dbl[bi]
        for nm, w in [("d%d_iw" % bi, iw), ("d%d_tw" % bi, tw)]:
            for kk in ["wqkv", "wproj", "wgu", "wd", "q_norm", "k_norm"]:
                W("in_%s_%s" % (nm, kk), w[kk])
    for bi in range(NUM_SINGLE):
        w = sgl[bi]
        for kk in ["w1", "w2", "q_norm", "k_norm"]:
            W("in_s%d_%s" % (bi, kk), w[kk])
    for nm, m in [("im", im), ("tm", tm)]:
        for kk in ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]:
            W("in_%s_%s" % (nm, kk), m[kk])
    for kk in ["shift", "scale", "gate"]:
        W("in_sm_%s" % kk, sm[kk])

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
