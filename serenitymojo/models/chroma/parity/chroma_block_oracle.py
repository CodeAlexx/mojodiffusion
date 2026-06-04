#!/usr/bin/env python3
# serenitymojo/models/chroma/parity/chroma_block_oracle.py
#
# Torch-autograd oracle for the Chroma1-HD DOUBLE + SINGLE DiT blocks
# (forward + grads, base + LoRA). Built INDEPENDENTLY from the Chroma reference
# (EriDiffusion-v2 chroma.rs + inference models/dit/chroma_dit.mojo
# double_block_smoke_forward / single_block_smoke_forward). Produces .bin
# references the Mojo gate (chroma_block_parity.mojo) reads at cos >= 0.999.
#
# CHROMA-SPECIFIC (vs Flux): SEPARATE attn projections + SEPARATE proj_mlp.
#   double img : to_q, to_k, to_v ([D,D] each, WITH bias) + to_out.0 + ff.net.0.proj
#                + ff.net.2.  txt mirrors with add_q/k/v_proj + to_add_out +
#                ff_context.net.0.proj + ff_context.net.2.
#   single     : to_q, to_k, to_v ([D,D] each) + proj_mlp ([Fmlp,D]) + proj_out
#                ([D, D+Fmlp]).
# This oracle keeps the projections SEPARATE (Chroma-faithful) and dumps separate
# weight grads (d_to_q, d_to_k, d_to_v, d_proj_mlp, ...). The Mojo gate row-stacks
# them via the Chroma loader and decomposes the fused grad back to compare — that
# is the load-bearing proof the separate<->fused mapping is correct.
#
# Block math otherwise identical to Flux (GELU MLP, q/k rms_norm over Dh, joint
# concat txt-FIRST, interleaved RoPE, sdpa scale 1/sqrt(Dh), modulate_pre =
# (1+scale)*LayerNorm(x,eps) + shift, residual_gate(x,gate,y)=x+gate*y).
#
# REAL Chroma dims: D=3072, H=24, Dh=128. Small N/Fmlp keep the torch oracle fast;
# H/Dh/D are REAL. NON-DEGENERATE sinusoidal/random fills (NEVER modular (i*k)%9,
# which aliases at real strides and zeros grads). 3-axis Chroma RoPE asserted
# non-degenerate.
#
# Run (SEPARATE command, never chained after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/chroma/parity/chroma_block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── REAL Chroma block dims ──
H = 24
Dh = 128
D = H * Dh                # 3072 (REAL Chroma inner_dim)
N_IMG = 4
N_TXT = 3
S = N_TXT + N_IMG
S_SINGLE = 6
FMLP = 32                 # GELU MLP hidden (real = 4*D; small here for speed)
RANK = 4                  # LoRA rank
ALPHA = 8.0               # LoRA alpha -> scale = ALPHA/RANK = 2.0
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
AXES = [16, 56, 56]
ROPE_THETA = 10000.0
LORA_SCALE = ALPHA / RANK

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
    return (1.0 + scale) * layer_norm(x) + shift


def residual_gate(x, gate, y):
    return x + gate * y


def rms_norm_lastdim(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def gelu_tanh(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))


def rope_interleaved(x, cos, sin, n):
    cr = cos.reshape(n, H, Dh // 2)
    sr = sin.reshape(n, H, Dh // 2)
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


def build_rope(n, ids):
    cos_parts, sin_parts = [], []
    for axis, axis_dim in enumerate(AXES):
        half = axis_dim // 2
        omega = torch.tensor(
            [1.0 / (ROPE_THETA ** ((2 * i) / axis_dim)) for i in range(half)],
            dtype=DT,
        ).reshape(1, half)
        pos = ids[:, axis].reshape(n, 1)
        angles = pos @ omega
        cos_parts.append(torch.cos(angles))
        sin_parts.append(torch.sin(angles))
    cos = torch.cat(cos_parts, dim=1)
    sin = torch.cat(sin_parts, dim=1)
    cos = cos.unsqueeze(1).expand(n, H, Dh // 2).reshape(n * H, Dh // 2).contiguous()
    sin = sin.unsqueeze(1).expand(n, H, Dh // 2).reshape(n * H, Dh // 2).contiguous()
    return cos, sin


def rnd(g, *shape):
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.05).to(DT)


# LoRA: A [rank,in], B [out,rank]; y' = base + scale*((x@A.T)@B.T)
def make_lora(seed, in_f, out_f):
    g = torch.Generator().manual_seed(seed)
    a = rnd(g, RANK, in_f).requires_grad_(True)
    # B = 0 at PEFT identity would give zero LoRA grad on A's path-through; use a
    # small NON-ZERO B so both d_A and d_B are non-degenerate (gate needs both).
    b = rnd(g, out_f, RANK).requires_grad_(True)
    return {"a": a, "b": b, "in_f": in_f, "out_f": out_f}


def lora_apply(base_y, x, lo):
    t = x @ lo["a"].T                # [M, rank]
    dy = t @ lo["b"].T               # [M, out]
    return base_y + LORA_SCALE * dy


# ── DOUBLE stream weights: SEPARATE to_q/to_k/to_v (+bias), to_out, ff.0, ff.2 ──
def make_double_stream(seed):
    g = torch.Generator().manual_seed(seed)
    w = {}
    for nm in ("to_q", "to_k", "to_v"):
        w[nm] = rnd(g, D, D)
        w[nm + "_b"] = rnd(g, D)
    w["out"] = rnd(g, D, D); w["out_b"] = rnd(g, D)
    w["mlp0"] = rnd(g, FMLP, D); w["mlp0_b"] = rnd(g, FMLP)
    w["mlp2"] = rnd(g, D, FMLP); w["mlp2_b"] = rnd(g, D)
    w["q_norm"] = rnd(g, Dh) * 2.0 + 1.0
    w["k_norm"] = rnd(g, Dh) * 2.0 + 1.0
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_mod(off):
    m = {}
    m["shift1"] = fill(D, 0.0013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale1"] = fillc(D, 0.0017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate1"] = fill(D, 0.0011, 0.3 + off, 0.4).requires_grad_(True)
    m["shift2"] = fillc(D, 0.0019, 0.4 + off, 0.3).requires_grad_(True)
    m["scale2"] = fill(D, 0.0015, 0.5 + off, 0.2).requires_grad_(True)
    m["gate2"] = fillc(D, 0.0012, 0.6 + off, 0.4).requires_grad_(True)
    return m


def make_double_lora(seed):
    # slots: to_q, to_k, to_v, proj(out), mlp0, mlp2
    return {
        "to_q": make_lora(seed + 0, D, D),
        "to_k": make_lora(seed + 1, D, D),
        "to_v": make_lora(seed + 2, D, D),
        "proj": make_lora(seed + 3, D, D),
        "mlp0": make_lora(seed + 4, D, FMLP),
        "mlp2": make_lora(seed + 5, FMLP, D),
    }


def stream_pre(x, w, m, lo, n):
    norm = modulate(x, m["scale1"], m["shift1"])
    q = lora_apply(norm @ w["to_q"].T + w["to_q_b"], norm, lo["to_q"]).reshape(1, n, H, Dh)
    k = lora_apply(norm @ w["to_k"].T + w["to_k_b"], norm, lo["to_k"]).reshape(1, n, H, Dh)
    v = lora_apply(norm @ w["to_v"].T + w["to_v_b"], norm, lo["to_v"]).reshape(1, n, H, Dh)
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])
    return q, k, v


def stream_post(x, att, w, m, lo, n):
    out = lora_apply(att @ w["out"].T + w["out_b"], att, lo["proj"])
    attn_res = residual_gate(x, m["gate1"], out)
    mlp_in = modulate(attn_res, m["scale2"], m["shift2"])
    mlp_pre = lora_apply(mlp_in @ w["mlp0"].T + w["mlp0_b"], mlp_in, lo["mlp0"])
    mlp_h = gelu_tanh(mlp_pre)
    mlp = lora_apply(mlp_h @ w["mlp2"].T + w["mlp2_b"], mlp_h, lo["mlp2"])
    return residual_gate(attn_res, m["gate2"], mlp)


def Wbin(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))


# fuse separate (q;k;v) [D,D] weights into [3D,D]; biases into [3D].
def fuse3(wq, wk, wv):
    return torch.cat([wq, wk, wv], dim=0)


def gen_double():
    img = t2(N_IMG, D, 0.0021, 0.05, 0.5).requires_grad_(True)
    txt = t2(N_TXT, D, 0.0023, 0.07, 0.5).requires_grad_(True)
    iw = make_double_stream(1)
    tw = make_double_stream(2)
    im = make_mod(0.0)
    tm = make_mod(1.0)
    ilo = make_double_lora(100)
    tlo = make_double_lora(200)

    ids = torch.tensor(
        [[float(s), float((s * 3) % 5), float((s * 2) % 7)] for s in range(S)],
        dtype=DT,
    )
    cos, sin = build_rope(S, ids)
    half = Dh // 2
    diff = (cos[:, : half // 2] - cos[:, half // 2 :]).abs().max()
    assert float(diff) > 1e-6, "DEGENERATE rope table"

    iq, ik, iv = stream_pre(img, iw, im, ilo, N_IMG)
    tq, tk, tv = stream_pre(txt, tw, tm, tlo, N_TXT)
    q = torch.cat([tq, iq], dim=1)
    k = torch.cat([tk, ik], dim=1)
    v = torch.cat([tv, iv], dim=1)
    qr = rope_interleaved(q, cos, sin, S)
    kr = rope_interleaved(k, cos, sin, S)
    att = sdpa(qr, kr, v)
    txt_att = att[:, 0:N_TXT].reshape(N_TXT, D)
    img_att = att[:, N_TXT:N_TXT + N_IMG].reshape(N_IMG, D)
    img_out = stream_post(img, img_att, iw, im, ilo, N_IMG)
    txt_out = stream_post(txt, txt_att, tw, tm, tlo, N_TXT)

    d_img = t2(N_IMG, D, 0.0027, 0.11, 0.05)
    d_txt = t2(N_TXT, D, 0.0029, 0.13, 0.05)
    loss = (img_out * d_img).sum() + (txt_out * d_txt).sum()
    loss.backward()

    Wbin("d_ref_img_out", img_out)
    Wbin("d_ref_txt_out", txt_out)
    Wbin("d_ref_d_img", img.grad)
    Wbin("d_ref_d_txt", txt.grad)
    # fused weight-grad references (gate compares the Mojo fused d_wqkv to this)
    for nm, w in [("im", iw), ("tm", tw)]:
        Wbin("d_ref_%s_d_wqkv" % nm, fuse3(w["to_q"].grad, w["to_k"].grad, w["to_v"].grad))
        Wbin("d_ref_%s_d_bqkv" % nm, torch.cat([w["to_q_b"].grad, w["to_k_b"].grad, w["to_v_b"].grad]))
        Wbin("d_ref_%s_d_wproj" % nm, w["out"].grad)
        Wbin("d_ref_%s_d_bproj" % nm, w["out_b"].grad)
        Wbin("d_ref_%s_d_wmlp0" % nm, w["mlp0"].grad)
        Wbin("d_ref_%s_d_bmlp0" % nm, w["mlp0_b"].grad)
        Wbin("d_ref_%s_d_wmlp2" % nm, w["mlp2"].grad)
        Wbin("d_ref_%s_d_bmlp2" % nm, w["mlp2_b"].grad)
        Wbin("d_ref_%s_d_q_norm" % nm, w["q_norm"].grad)
        Wbin("d_ref_%s_d_k_norm" % nm, w["k_norm"].grad)
    for nm, m in [("im", im), ("tm", tm)]:
        for kk in ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]:
            Wbin("d_ref_%s_d_%s" % (nm, kk), m[kk].grad)
    # LoRA grad references
    for nm, lo in [("ilo", ilo), ("tlo", tlo)]:
        for slot in ["to_q", "to_k", "to_v", "proj", "mlp0", "mlp2"]:
            Wbin("d_ref_%s_%s_d_a" % (nm, slot), lo[slot]["a"].grad)
            Wbin("d_ref_%s_%s_d_b" % (nm, slot), lo[slot]["b"].grad)

    # inputs the Mojo gate reconstructs identically (SEPARATE projections)
    Wbin("d_in_img", img)
    Wbin("d_in_txt", txt)
    for nm, w in [("iw", iw), ("tw", tw)]:
        for kk in ["to_q", "to_k", "to_v", "to_q_b", "to_k_b", "to_v_b",
                   "out", "out_b", "mlp0", "mlp0_b", "mlp2", "mlp2_b", "q_norm", "k_norm"]:
            Wbin("d_in_%s_%s" % (nm, kk), w[kk])
    for nm, m in [("im", im), ("tm", tm)]:
        for kk in ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]:
            Wbin("d_in_%s_%s" % (nm, kk), m[kk])
    for nm, lo in [("ilo", ilo), ("tlo", tlo)]:
        for slot in ["to_q", "to_k", "to_v", "proj", "mlp0", "mlp2"]:
            Wbin("d_in_%s_%s_a" % (nm, slot), lo[slot]["a"])
            Wbin("d_in_%s_%s_b" % (nm, slot), lo[slot]["b"])
    Wbin("d_in_cos", cos)
    Wbin("d_in_sin", sin)
    Wbin("d_in_d_img", d_img)
    Wbin("d_in_d_txt", d_txt)
    print("DOUBLE: forward loss =", float(loss))


def make_single_weights(seed):
    g = torch.Generator().manual_seed(seed)
    w = {}
    for nm in ("to_q", "to_k", "to_v"):
        w[nm] = rnd(g, D, D); w[nm + "_b"] = rnd(g, D)
    w["proj_mlp"] = rnd(g, FMLP, D); w["proj_mlp_b"] = rnd(g, FMLP)
    w["w2"] = rnd(g, D, D + FMLP); w["b2"] = rnd(g, D)
    w["q_norm"] = rnd(g, Dh) * 2.0 + 1.0
    w["k_norm"] = rnd(g, Dh) * 2.0 + 1.0
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_single_mod(off):
    m = {}
    m["shift"] = fill(D, 0.0013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale"] = fillc(D, 0.0017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate"] = fill(D, 0.0011, 0.3 + off, 0.4).requires_grad_(True)
    return m


def make_single_lora(seed):
    # slots: to_q, to_k, to_v, proj_mlp, linear2
    return {
        "to_q": make_lora(seed + 0, D, D),
        "to_k": make_lora(seed + 1, D, D),
        "to_v": make_lora(seed + 2, D, D),
        "proj_mlp": make_lora(seed + 3, D, FMLP),
        "linear2": make_lora(seed + 4, D + FMLP, D),
    }


def gen_single():
    n = S_SINGLE
    x = t2(n, D, 0.0021, 0.05, 0.5).requires_grad_(True)
    w = make_single_weights(3)
    m = make_single_mod(0.0)
    lo = make_single_lora(300)

    ids = torch.tensor(
        [[float(s), float((s * 3) % 5), float((s * 2) % 7)] for s in range(n)],
        dtype=DT,
    )
    cos, sin = build_rope(n, ids)
    half = Dh // 2
    diff = (cos[:, : half // 2] - cos[:, half // 2 :]).abs().max()
    assert float(diff) > 1e-6, "DEGENERATE rope table (single)"

    norm = modulate(x, m["scale"], m["shift"])
    q = lora_apply(norm @ w["to_q"].T + w["to_q_b"], norm, lo["to_q"]).reshape(1, n, H, Dh)
    k = lora_apply(norm @ w["to_k"].T + w["to_k_b"], norm, lo["to_k"]).reshape(1, n, H, Dh)
    v = lora_apply(norm @ w["to_v"].T + w["to_v_b"], norm, lo["to_v"]).reshape(1, n, H, Dh)
    mlp_in = lora_apply(norm @ w["proj_mlp"].T + w["proj_mlp_b"], norm, lo["proj_mlp"])
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])
    qr = rope_interleaved(q, cos, sin, n)
    kr = rope_interleaved(k, cos, sin, n)
    att = sdpa(qr, kr, v).reshape(n, D)
    mlp_h = gelu_tanh(mlp_in)
    out_in = torch.cat([att, mlp_h], dim=1)
    out_proj = lora_apply(out_in @ w["w2"].T + w["b2"], out_in, lo["linear2"])
    out = residual_gate(x, m["gate"], out_proj)

    d_out = t2(n, D, 0.0027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    Wbin("s_ref_out", out)
    Wbin("s_ref_d_x", x.grad)
    Wbin("s_ref_d_w1", torch.cat([w["to_q"].grad, w["to_k"].grad, w["to_v"].grad, w["proj_mlp"].grad], dim=0))
    Wbin("s_ref_d_b1", torch.cat([w["to_q_b"].grad, w["to_k_b"].grad, w["to_v_b"].grad, w["proj_mlp_b"].grad]))
    Wbin("s_ref_d_w2", w["w2"].grad)
    Wbin("s_ref_d_b2", w["b2"].grad)
    Wbin("s_ref_d_q_norm", w["q_norm"].grad)
    Wbin("s_ref_d_k_norm", w["k_norm"].grad)
    for kk in ["shift", "scale", "gate"]:
        Wbin("s_ref_d_%s" % kk, m[kk].grad)
    for slot in ["to_q", "to_k", "to_v", "proj_mlp", "linear2"]:
        Wbin("s_ref_%s_d_a" % slot, lo[slot]["a"].grad)
        Wbin("s_ref_%s_d_b" % slot, lo[slot]["b"].grad)

    Wbin("s_in_x", x)
    for kk in ["to_q", "to_k", "to_v", "to_q_b", "to_k_b", "to_v_b",
               "proj_mlp", "proj_mlp_b", "w2", "b2", "q_norm", "k_norm"]:
        Wbin("s_in_w_%s" % kk, w[kk])
    for kk in ["shift", "scale", "gate"]:
        Wbin("s_in_m_%s" % kk, m[kk])
    for slot in ["to_q", "to_k", "to_v", "proj_mlp", "linear2"]:
        Wbin("s_in_%s_a" % slot, lo[slot]["a"])
        Wbin("s_in_%s_b" % slot, lo[slot]["b"])
    Wbin("s_in_cos", cos)
    Wbin("s_in_sin", sin)
    Wbin("s_in_d_out", d_out)
    print("SINGLE: forward loss =", float(loss))


if __name__ == "__main__":
    gen_double()
    gen_single()
    print("DONE")
