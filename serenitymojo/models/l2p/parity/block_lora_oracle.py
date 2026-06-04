#!/usr/bin/env python3
# serenitymojo/models/l2p/parity/block_lora_oracle.py
#
# Torch-autograd oracle for the Z-Image L2P (pixel-space) MAIN-LAYER DiT block
# *WITH LoRA* on all 7 trained projections per block:
#   attention.{to_q, to_k, to_v, to_out.0}  and  feed_forward.{w1, w3, w2}
#
# L2P REUSES the Z-Image-Turbo DiT body VERBATIM (hidden 3840, 30 heads, head_dim
# 128, QK-norm, SwiGLU, adaLN scale+gate, INTERLEAVED 3-axis RoPE). The L2P deltas
# (pixel-space patchify16 input proj + MicroDiffusionModel local-decoder head)
# live OUTSIDE the transformer block, so the per-block backward surface == the
# Z-Image main block. This oracle therefore replicates the EXACT math of
# serenitymojo/models/zimage/lora_block.mojo::zimage_block_lora_{forward,backward}
# (the host-list variant), so the L2P block LoRA gate (block_lora_parity.mojo) can
# verify the input grad + every base weight grad + every LoRA d_A/d_B at cos>=0.999.
#
# LoRA math (matches lora_block.mojo + train_step._lora_fwd):
#   y' = x @ W.T + scale * ((x @ A.T) @ B.T)    A=[rank,in] B=[out,rank]
#   scale = alpha / rank.  B is perturbed NON-zero here so adapters are LIVE and
#   their grads are non-degenerate.
#
# Dims confirmed from the real Z-Image base safetensors header (transformer
# diffusion_pytorch_model-00001-of-00002.safetensors): D=3840, Dh=128, H=30,
# attention.to_{q,k,v,out.0}=[3840,3840], feed_forward.w1/w3=[10240,3840],
# w2=[3840,10240], adaLN_modulation.0=[15360,256] (4 chunks). The real L2P LoRA
# /home/alex/samples/l2p_lora_box1jana_1000steps_bf16.safetensors uses rank=16
# on FUSED qkv [16,11520] + out [16,3840] + w1 [16,10240]; the per-block backward
# MATH (dA/dB via linear_backward composition) is identical regardless of QKV
# fusion, so the gate uses the un-fused 7-slot form (small F for oracle speed).
#
# Run (SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/l2p/parity/block_lora_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64  # F64 reference interior

# ── dims (REAL Z-Image/L2P head count H=30, head_dim Dh=128 -> hidden D=3840) ──
H = 30
Dh = 128
D = H * Dh          # 3840
N_TXT = 2
IMG_H = 2
IMG_W = 3
N_IMG = IMG_H * IMG_W   # 6
S = N_TXT + N_IMG       # 8
F = 96                  # FFN hidden for SwiGLU gate (small; real is 10240)
EPS = 1e-5
SCALE = 1.0 / math.sqrt(Dh)
ROPE_AXES = (32, 48, 48)   # sum = 64 = Dh//2
ROPE_THETA = 256

# LoRA config (rank=8/alpha=16 keeps the live perturbation well-scaled, mirroring
# the zimage lora_stack gate; the real L2P export is rank=16/alpha=16 — same math).
RANK = 8
ALPHA = 16.0
LSCALE = ALPHA / RANK

REF_DIR = os.path.dirname(os.path.abspath(__file__))

# slot order MUST match lora_block.mojo SLOT_*: Q,K,V,O,w1,w3,w2
SLOTS = ["to_q", "to_k", "to_v", "to_out", "w1", "w3", "w2"]
SLOT_SHAPE = {
    "to_q": (D, D), "to_k": (D, D), "to_v": (D, D), "to_out": (D, D),
    "w1": (D, F), "w3": (D, F), "w2": (F, D),
}


def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def fillc(n, a, b, c):
    return torch.tensor([math.cos(a * i + b) * c for i in range(n)], dtype=DT)


def t2(n, m, a, b, c):
    return fill(n * m, a, b, c).reshape(n, m)


def rms_norm(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


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
    cos_rows, sin_rows = [], []
    for tok in range(S):
        if tok < N_TXT:
            p0, p1, p2 = float(tok + 1), 0.0, 0.0
        else:
            it = tok - N_TXT
            p0, p1, p2 = float(N_TXT + 1), float(it // IMG_W), float(it % IMG_W)
        cos_tok, sin_tok = [], []
        for (p, freqs) in ((p0, f0), (p1, f1), (p2, f2)):
            for fi in freqs:
                ang = p * fi
                cos_tok.append(math.cos(ang))
                sin_tok.append(math.sin(ang))
        for _h in range(H):
            cos_rows.append(cos_tok)
            sin_rows.append(sin_tok)
    return (torch.tensor(cos_rows, dtype=DT), torch.tensor(sin_rows, dtype=DT))


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


# A small randn (matches lora_block init scale 0.01); B perturbed NON-zero (LIVE).
def lcg_randn(n, seed, scale):
    out = []
    state = seed & ((1 << 64) - 1)
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = float(state >> 40) * (1.0 / 16777216.0)
        out.append((u - 0.5) * scale)
    return torch.tensor(out, dtype=DT)


def make_lora():
    lo = {}
    seed = 8000
    for s in SLOTS:
        in_f, out_f = SLOT_SHAPE[s]
        A = lcg_randn(RANK * in_f, seed, 0.01).reshape(RANK, in_f)
        B = lcg_randn(out_f * RANK, seed + 777, 0.05).reshape(out_f, RANK)
        lo[s] = (A.clone().requires_grad_(True), B.clone().requires_grad_(True))
        seed += 1
    return lo


def lora(x, AB):
    A, B = AB
    return LSCALE * ((x @ A.T) @ B.T)


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, flat.shape)


def main():
    x = t2(S, D, 0.021, 0.05, 0.5).requires_grad_(True)
    w = make_weights(1)
    m = make_mod()
    lo = make_lora()

    cos, sin = build_real_rope_tables()
    dvar = float(cos.std())
    assert dvar > 1e-3, "rope table degenerate — gate tautology"

    # ── forward (LoRA on all 7 projections; modulated main block) ──
    xn1 = rms_norm(x, w["n1"])
    xn1s = (1.0 + m["scale_msa"]) * xn1
    q = (xn1s @ w["wq"].T + lora(xn1s, lo["to_q"])).reshape(1, S, H, Dh)
    k = (xn1s @ w["wk"].T + lora(xn1s, lo["to_k"])).reshape(1, S, H, Dh)
    v = (xn1s @ w["wv"].T + lora(xn1s, lo["to_v"])).reshape(1, S, H, Dh)
    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v).reshape(S, D)
    att_o = att @ w["wo"].T + lora(att, lo["to_out"])
    attn_n2 = rms_norm(att_o, w["n2"])
    gate_msa = torch.tanh(m["gate_msa"])
    h = x + gate_msa * attn_n2

    xfn1 = rms_norm(h, w["fn1"])
    xfn1s = (1.0 + m["scale_mlp"]) * xfn1
    g_pre = xfn1s @ w["w1"].T + lora(xfn1s, lo["w1"])
    u = xfn1s @ w["w3"].T + lora(xfn1s, lo["w3"])
    act = silu(g_pre) * u
    ff = act @ w["w2"].T + lora(act, lo["w2"])
    ff_n2 = rms_norm(ff, w["fn2"])
    gate_mlp = torch.tanh(m["gate_mlp"])
    out = h + gate_mlp * ff_n2

    d_out = t2(S, D, 0.027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    # forward output + base grads
    W("ref_out", out)
    W("ref_d_x", x.grad)
    for kk in ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "n2", "fn1", "w1", "w3", "w2", "fn2"]:
        W("ref_d_%s" % kk, w[kk].grad)
    for kk in ["scale_msa", "gate_msa", "scale_mlp", "gate_mlp"]:
        W("ref_d_%s" % kk, m[kk].grad)
    # LoRA d_A / d_B per slot
    for s in SLOTS:
        A, B = lo[s]
        W("ref_d%s_A" % s, A.grad)
        W("ref_d%s_B" % s, B.grad)

    # inputs the Mojo gate reconstructs
    W("in_x", x)
    for kk in ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "n2", "fn1", "w1", "w3", "w2", "fn2"]:
        W("in_w_%s" % kk, w[kk])
    for kk in ["scale_msa", "gate_msa", "scale_mlp", "gate_mlp"]:
        W("in_m_%s" % kk, m[kk])
    for s in SLOTS:
        A, B = lo[s]
        W("in_l%s_A" % s, A)
        W("in_l%s_B" % s, B)
    W("in_cos", cos)
    W("in_sin", sin)
    W("in_d_out", d_out)

    print("forward loss =", float(loss))
    print("rope cos std =", dvar, " RANK =", RANK, " ALPHA =", ALPHA)
    print("DONE")


if __name__ == "__main__":
    main()
