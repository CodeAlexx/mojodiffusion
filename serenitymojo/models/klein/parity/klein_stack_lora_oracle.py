#!/usr/bin/env python3
# serenitymojo/models/klein/parity/klein_stack_lora_oracle.py
#
# Torch oracle for the Klein FULL DiT STACK *WITH LoRA* (small depth: 1 double +
# 1 single) forward + autograd. Replicates the EXACT math of
# serenitymojo/models/klein/klein_stack_lora.mojo's klein_stack_lora_forward/
# backward — the base stack (klein_stack.mojo) with a LoRA delta
# scale*((x@Aᵀ)@Bᵀ) on EVERY trained attention projection:
#   * double block: img/txt × {qkv (in=D,out=3D), proj (in=D,out=D)}
#   * single block: qkv (w1 qkv-ROWS, in=D,out=3D) + out (w2 att-COLS, in=D,out=D)
#
# The per-block LoRA is already gated cos>=0.999 (double_block_lora_parity,
# single_block_lora_parity). THIS gate proves the STACK threads each block's
# adapters and COLLECTS each adapter's d_A/d_B correctly across the full graph.
#
# CONVENTIONS (match the Mojo forward byte-for-byte): see klein_stack_oracle.py.
#   LoRA: y' = linear(x, W) + (alpha/rank)*((x @ Aᵀ) @ Bᵀ), A=[rank,in], B=[out,rank].
#   single qkv LoRA lands in the first 3D cols of `fused` (the qkv rows of w1).
#   single out  LoRA input is att_flat [S,D] (the first D cols of out_in).
#
# NON-DEGENERATE inputs: sinusoidal/random fills (A AND B nonzero per adapter so
# both d_A and d_B are exercised). NEVER modular aliasing.
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/klein_stack_lora_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── dims (small; 1 double + 1 single) ──
H = 4
Dh = 8
D = H * Dh           # 32
N_IMG = 4
N_TXT = 2
S = N_TXT + N_IMG
F = 24
IN_CH = 10
TXT_CH = 14
OUT_CH = 6
NUM_DOUBLE = 1
NUM_SINGLE = 1
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)

RANK = 4
ALPHA = 8.0
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


# ── weights / mod ──
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


def make_lora(seed, in_f, out_f):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    lo = {}
    lo["A"] = (rnd(RANK, in_f) * 0.02).requires_grad_(True)
    lo["B"] = (rnd(out_f, RANK) * 0.02).requires_grad_(True)
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


def double_block_lora(img, txt, iw, tw, im, tm, ilo, tlo, cos, sin):
    # ilo/tlo: dict with qkv (A,B) and proj (A,B) adapters.
    def stream_pre(x, w, m, lo):
        N = x.shape[0]
        ln1 = layer_norm(x)
        norm = modulate(ln1, m["scale1"], m["shift1"])
        qkv = norm @ w["wqkv"].T + lora_delta(norm, lo["qkv"]["A"], lo["qkv"]["B"])
        q = qkv[:, 0:D].reshape(1, N, H, Dh)
        k = qkv[:, D:2 * D].reshape(1, N, H, Dh)
        v = qkv[:, 2 * D:3 * D].reshape(1, N, H, Dh)
        q = rms_norm_lastdim(q, w["q_norm"])
        k = rms_norm_lastdim(k, w["k_norm"])
        return q, k, v

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

    def stream_post(x, att, w, m, lo):
        out = att @ w["wproj"].T + lora_delta(att, lo["proj"]["A"], lo["proj"]["B"])
        attn_res = residual_gate(x, m["gate1"], out)
        ln2 = layer_norm(attn_res)
        mlp_in = modulate(ln2, m["scale2"], m["shift2"])
        gu = mlp_in @ w["wgu"].T
        gate = gu[:, 0:F]
        up = gu[:, F:2 * F]
        act = torch.nn.functional.silu(gate) * up
        mlp = act @ w["wd"].T
        return residual_gate(attn_res, m["gate2"], mlp)

    img_out = stream_post(img, img_att, iw, im, ilo)
    txt_out = stream_post(txt, txt_att, tw, tm, tlo)
    return img_out, txt_out


def single_block_lora(x, w, m, lo, cos, sin):
    # lo: dict with qkv (A,B) [in=D,out=3D] and out (A,B) [in=D,out=D].
    Sx = x.shape[0]
    ln = layer_norm(x)
    norm = modulate(ln, m["scale"], m["shift"])
    fused = norm @ w["w1"].T                       # [S, 3D+2F]
    # LoRA on w1 qkv-rows: delta [S,3D] folded into the first 3D cols of fused.
    qkv_delta = lora_delta(norm, lo["qkv"]["A"], lo["qkv"]["B"])   # [S,3D]
    fused = fused.clone()
    fused = torch.cat([fused[:, 0:3 * D] + qkv_delta, fused[:, 3 * D:]], dim=1)
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
    # LoRA on w2 cols: input = att_flat (the first D cols of out_in).
    out = out + lora_delta(att, lo["out"]["A"], lo["out"]["B"])
    return residual_gate(x, m["gate"], out)


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, flat.shape)


def main():
    # ── frozen base weights ──
    img_in = (t2(D, IN_CH, 0.021, 0.05, 0.3)).requires_grad_(True)
    txt_in = (t2(D, TXT_CH, 0.019, 0.07, 0.3)).requires_grad_(True)
    final_lin = (t2(OUT_CH, D, 0.023, 0.09, 0.3)).requires_grad_(True)
    final_shift = fill(D, 0.011, 0.13, 0.2).requires_grad_(True)
    final_scale = fillc(D, 0.012, 0.17, 0.2).requires_grad_(True)

    img_tokens = t2(N_IMG, IN_CH, 0.031, 0.05, 0.5).requires_grad_(True)
    txt_tokens = t2(N_TXT, TXT_CH, 0.033, 0.07, 0.5).requires_grad_(True)

    # ── per-block weights ──
    dbl = []
    dbl_lora = []
    for bi in range(NUM_DOUBLE):
        iw = make_stream(100 + bi * 2)
        tw = make_stream(101 + bi * 2)
        ilo = {
            "qkv": make_lora(300 + bi * 4 + 0, D, 3 * D),
            "proj": make_lora(300 + bi * 4 + 1, D, D),
        }
        tlo = {
            "qkv": make_lora(300 + bi * 4 + 2, D, 3 * D),
            "proj": make_lora(300 + bi * 4 + 3, D, D),
        }
        dbl.append((iw, tw))
        dbl_lora.append((ilo, tlo))
    sgl = []
    sgl_lora = []
    for bi in range(NUM_SINGLE):
        sgl.append(make_single(200 + bi))
        sgl_lora.append({
            "qkv": make_lora(400 + bi * 2 + 0, D, 3 * D),
            "out": make_lora(400 + bi * 2 + 1, D, D),
        })

    im = make_mod(0.0)
    tm = make_mod(1.0)
    sm = make_single_mod(2.0)

    cos = t2(S * H, Dh // 2, 0.03, 0.2, 1.0) * 0.6
    sin = t2(S * H, Dh // 2, 0.04, 0.5, 1.0) * 0.6

    # ── forward ──
    img = img_tokens @ img_in.T
    txt = txt_tokens @ txt_in.T
    for bi in range(NUM_DOUBLE):
        iw, tw = dbl[bi]
        ilo, tlo = dbl_lora[bi]
        img, txt = double_block_lora(img, txt, iw, tw, im, tm, ilo, tlo, cos, sin)
    x = torch.cat([txt, img], dim=0)
    for bi in range(NUM_SINGLE):
        x = single_block_lora(x, sgl[bi], sm, sgl_lora[bi], cos, sin)
    img_out = x[N_TXT:N_TXT + N_IMG]
    normed = modulate(layer_norm(img_out), final_scale, final_shift)
    out = normed @ final_lin.T

    d_out = t2(N_IMG, OUT_CH, 0.027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    # forward output (sanity)
    W("klref_out", out)
    # load-bearing input-token grads
    W("klref_d_img_tokens", img_tokens.grad)
    W("klref_d_txt_tokens", txt_tokens.grad)
    # shared modvec grads (summed across blocks)
    W("klref_d_img_mod", torch.cat([
        im["shift1"].grad, im["scale1"].grad, im["gate1"].grad,
        im["shift2"].grad, im["scale2"].grad, im["gate2"].grad]))
    W("klref_d_txt_mod", torch.cat([
        tm["shift1"].grad, tm["scale1"].grad, tm["gate1"].grad,
        tm["shift2"].grad, tm["scale2"].grad, tm["gate2"].grad]))
    W("klref_d_single_mod", torch.cat([
        sm["shift"].grad, sm["scale"].grad, sm["gate"].grad]))

    # ── the DELIVERABLE: a sample of LoRA d_A / d_B from each block kind ──
    # double block 0: all four adapters.
    ilo0, tlo0 = dbl_lora[0]
    W("klref_d0_img_qkv_dA", ilo0["qkv"]["A"].grad)
    W("klref_d0_img_qkv_dB", ilo0["qkv"]["B"].grad)
    W("klref_d0_img_proj_dA", ilo0["proj"]["A"].grad)
    W("klref_d0_img_proj_dB", ilo0["proj"]["B"].grad)
    W("klref_d0_txt_qkv_dA", tlo0["qkv"]["A"].grad)
    W("klref_d0_txt_qkv_dB", tlo0["qkv"]["B"].grad)
    W("klref_d0_txt_proj_dA", tlo0["proj"]["A"].grad)
    W("klref_d0_txt_proj_dB", tlo0["proj"]["B"].grad)
    # single block 0: both adapters.
    slo0 = sgl_lora[0]
    W("klref_s0_qkv_dA", slo0["qkv"]["A"].grad)
    W("klref_s0_qkv_dB", slo0["qkv"]["B"].grad)
    W("klref_s0_out_dA", slo0["out"]["A"].grad)
    W("klref_s0_out_dB", slo0["out"]["B"].grad)

    # ── INPUTS the Mojo gate reconstructs (weights/lora cannot be regenerated) ──
    W("klin_img_in", img_in); W("klin_txt_in", txt_in)
    W("klin_final_lin", final_lin); W("klin_final_shift", final_shift)
    W("klin_final_scale", final_scale)
    W("klin_img_tokens", img_tokens); W("klin_txt_tokens", txt_tokens)
    W("klin_cos", cos); W("klin_sin", sin)
    W("klin_d_out", d_out)
    for bi in range(NUM_DOUBLE):
        iw, tw = dbl[bi]
        for nm, w in [("d%d_iw" % bi, iw), ("d%d_tw" % bi, tw)]:
            for kk in ["wqkv", "wproj", "wgu", "wd", "q_norm", "k_norm"]:
                W("klin_%s_%s" % (nm, kk), w[kk])
        ilo, tlo = dbl_lora[bi]
        for nm, lo in [("d%d_ilo" % bi, ilo), ("d%d_tlo" % bi, tlo)]:
            for slot in ["qkv", "proj"]:
                W("klin_%s_%s_A" % (nm, slot), lo[slot]["A"])
                W("klin_%s_%s_B" % (nm, slot), lo[slot]["B"])
    for bi in range(NUM_SINGLE):
        w = sgl[bi]
        for kk in ["w1", "w2", "q_norm", "k_norm"]:
            W("klin_s%d_%s" % (bi, kk), w[kk])
        lo = sgl_lora[bi]
        for slot in ["qkv", "out"]:
            W("klin_s%d_%s_A" % (bi, slot), lo[slot]["A"])
            W("klin_s%d_%s_B" % (bi, slot), lo[slot]["B"])
    for nm, m in [("im", im), ("tm", tm)]:
        for kk in ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]:
            W("klin_%s_%s" % (nm, kk), m[kk])
    for kk in ["shift", "scale", "gate"]:
        W("klin_sm_%s" % kk, sm[kk])

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
