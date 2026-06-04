#!/usr/bin/env python3
# serenitymojo/models/wan22/parity/wan22_block_lora_oracle.py
#
# Torch oracle for the Wan2.2 WanAttentionBlock WITH LoRA on the 8 attention
# projections (self/cross x q/k/v/o). Replicates wan22_block_lora_forward /
# wan22_block_lora_backward. Produces LoRA d_A/d_B + input-grad references the
# Mojo gate (wan22_block_lora_parity.mojo) reads at cos >= 0.999.
#
# LoRA math (matches train_step._lora_fwd): y' = linear(x,W) + scale*((x@Aᵀ)@Bᵀ),
# A=[rank,in], B=[out,rank], scale=alpha/rank.
#
# Run (SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/wan22/parity/wan22_block_lora_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
# BF16 oracle: matches the Mojo block's native bf16 compute (bf16·bf16 Linears,
# F32 GEMM-accumulate). Parity reflects bf16 rounding (cos ~0.99-0.999).
DT = torch.bfloat16

H = 24
Dh = 8
DIM = H * Dh         # 192
S = 5
TXT = 4
FFN = 40
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
RANK = 4
ALPHA = 4.0
LSCALE = ALPHA / RANK

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def t2(n, m, a, b, c):
    return fill(n * m, a, b, c).reshape(n, m)


def layer_norm_noaffine(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS)


def layer_norm_affine(x, w, b):
    return layer_norm_noaffine(x) * w + b


def rms_norm_lastdim(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))


def mod_pre(x, scale, shift):
    return layer_norm_noaffine(x) * (1.0 + scale) + shift


def gated_residual(x, y, gate):
    return x + gate * y


def rope_interleaved(x, cos, sin):
    Sx = x.shape[1]
    cr = cos.reshape(Sx, 1, Dh // 2)
    sr = sin.reshape(Sx, 1, Dh // 2)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    o0 = x0 * cr - x1 * sr
    o1 = x0 * sr + x1 * cr
    out = torch.empty_like(x)
    out[..., 0::2] = o0
    out[..., 1::2] = o1
    return out


def sdpa_square(q, k, v):
    qh = q.permute(0, 2, 1, 3); kh = k.permute(0, 2, 1, 3); vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    return (torch.softmax(scores, dim=-1) @ vh).permute(0, 2, 1, 3)


def sdpa_rect(q, k, v):
    qh = q.permute(0, 2, 1, 3); kh = k.permute(0, 2, 1, 3); vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    return (torch.softmax(scores, dim=-1) @ vh).permute(0, 2, 1, 3)


def make_weights(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    w = {}
    for pfx in ("sa", "ca"):
        w[pfx + "_wq"] = rnd(DIM, DIM); w[pfx + "_bq"] = rnd(DIM) * 0.1
        w[pfx + "_wk"] = rnd(DIM, DIM); w[pfx + "_bk"] = rnd(DIM) * 0.1
        w[pfx + "_wv"] = rnd(DIM, DIM); w[pfx + "_bv"] = rnd(DIM) * 0.1
        w[pfx + "_wo"] = rnd(DIM, DIM); w[pfx + "_bo"] = rnd(DIM) * 0.1
        w[pfx + "_qn"] = rnd(Dh) * 0.1 + 1.0
        w[pfx + "_kn"] = rnd(Dh) * 0.1 + 1.0
    w["n3_w"] = rnd(DIM) * 0.1 + 1.0
    w["n3_b"] = rnd(DIM) * 0.1
    w["ffn0_w"] = rnd(FFN, DIM); w["ffn0_b"] = rnd(FFN) * 0.1
    w["ffn2_w"] = rnd(DIM, FFN); w["ffn2_b"] = rnd(DIM) * 0.1
    for v in w.values():
        v.requires_grad_(False)   # FROZEN base (LoRA-only training)
    return w


def make_mod():
    m = {}
    m["shift_sa"] = t2(S, DIM, 0.013, 0.1, 0.3)
    m["scale_sa"] = t2(S, DIM, 0.017, 0.2, 0.2)
    m["gate_sa"] = t2(S, DIM, 0.011, 0.3, 0.4)
    m["shift_ffn"] = t2(S, DIM, 0.019, 0.4, 0.3)
    m["scale_ffn"] = t2(S, DIM, 0.015, 0.5, 0.2)
    m["gate_ffn"] = t2(S, DIM, 0.012, 0.6, 0.4)
    return m


def make_lora(seed):
    # 8 adapters: sa_q,sa_k,sa_v,sa_o,ca_q,ca_k,ca_v,ca_o (all in=out=DIM).
    g = torch.Generator().manual_seed(seed)
    lo = {}
    names = ["sa_q", "sa_k", "sa_v", "sa_o", "ca_q", "ca_k", "ca_v", "ca_o"]
    for nm in names:
        A = torch.randn(RANK, DIM, generator=g, dtype=torch.float32).to(DT) * 0.05
        B = torch.randn(DIM, RANK, generator=g, dtype=torch.float32).to(DT) * 0.05
        lo[nm + "_A"] = A.requires_grad_(True)
        lo[nm + "_B"] = B.requires_grad_(True)
    return lo


def lin_lora(x, W, b, A, B):
    base = x @ W.T + b
    delta = (x @ A.T) @ B.T
    return base + LSCALE * delta


def main():
    x = t2(S, DIM, 0.021, 0.05, 0.5).requires_grad_(True)
    context = t2(TXT, DIM, 0.023, 0.07, 0.5).requires_grad_(True)
    w = make_weights(1)
    m = make_mod()
    lo = make_lora(7)
    cos = t2(S, Dh // 2, 0.03, 0.2, 1.0) * 0.6
    sin = t2(S, Dh // 2, 0.04, 0.5, 1.0) * 0.6

    # self-attn (LoRA q/k/v/o)
    sa_in = mod_pre(x, m["scale_sa"], m["shift_sa"])
    q = lin_lora(sa_in, w["sa_wq"], w["sa_bq"], lo["sa_q_A"], lo["sa_q_B"]).reshape(1, S, H, Dh)
    k = lin_lora(sa_in, w["sa_wk"], w["sa_bk"], lo["sa_k_A"], lo["sa_k_B"]).reshape(1, S, H, Dh)
    v = lin_lora(sa_in, w["sa_wv"], w["sa_bv"], lo["sa_v_A"], lo["sa_v_B"]).reshape(1, S, H, Dh)
    q = rms_norm_lastdim(q, w["sa_qn"]); k = rms_norm_lastdim(k, w["sa_kn"])
    q = rope_interleaved(q, cos, sin); k = rope_interleaved(k, cos, sin)
    sa_att = sdpa_square(q, k, v).reshape(S, DIM)
    sa_out = lin_lora(sa_att, w["sa_wo"], w["sa_bo"], lo["sa_o_A"], lo["sa_o_B"])
    x_sa = gated_residual(x, sa_out, m["gate_sa"])

    # cross-attn (LoRA q/k/v/o)
    n3 = layer_norm_affine(x_sa, w["n3_w"], w["n3_b"])
    caq = lin_lora(n3, w["ca_wq"], w["ca_bq"], lo["ca_q_A"], lo["ca_q_B"]).reshape(1, S, H, Dh)
    cak = lin_lora(context, w["ca_wk"], w["ca_bk"], lo["ca_k_A"], lo["ca_k_B"]).reshape(1, TXT, H, Dh)
    cav = lin_lora(context, w["ca_wv"], w["ca_bv"], lo["ca_v_A"], lo["ca_v_B"]).reshape(1, TXT, H, Dh)
    caq = rms_norm_lastdim(caq, w["ca_qn"]); cak = rms_norm_lastdim(cak, w["ca_kn"])
    ca_att = sdpa_rect(caq, cak, cav).reshape(S, DIM)
    ca_out = lin_lora(ca_att, w["ca_wo"], w["ca_bo"], lo["ca_o_A"], lo["ca_o_B"])
    x_ca = x_sa + ca_out

    # ffn (no LoRA)
    ffn_in = mod_pre(x_ca, m["scale_ffn"], m["shift_ffn"])
    ffn_h = ffn_in @ w["ffn0_w"].T + w["ffn0_b"]
    ffn_act = gelu_tanh(ffn_h)
    ffn_out = ffn_act @ w["ffn2_w"].T + w["ffn2_b"]
    x_final = gated_residual(x_ca, ffn_out, m["gate_ffn"])

    d_out = t2(S, DIM, 0.027, 0.11, 0.05)
    loss = (x_final * d_out).sum()
    loss.backward()

    def W(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
        with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, flat.shape)

    W("lref_x_out", x_final)
    W("lref_d_x", x.grad)
    W("lref_d_context", context.grad)
    names = ["sa_q", "sa_k", "sa_v", "sa_o", "ca_q", "ca_k", "ca_v", "ca_o"]
    for nm in names:
        W("lref_" + nm + "_dA", lo[nm + "_A"].grad)
        W("lref_" + nm + "_dB", lo[nm + "_B"].grad)

    # inputs
    W("lin_x", x)
    W("lin_context", context)
    WKEYS = ["sa_wq", "sa_wk", "sa_wv", "sa_wo", "sa_bq", "sa_bk", "sa_bv", "sa_bo",
             "sa_qn", "sa_kn",
             "ca_wq", "ca_wk", "ca_wv", "ca_wo", "ca_bq", "ca_bk", "ca_bv", "ca_bo",
             "ca_qn", "ca_kn",
             "n3_w", "n3_b", "ffn0_w", "ffn0_b", "ffn2_w", "ffn2_b"]
    MKEYS = ["shift_sa", "scale_sa", "gate_sa", "shift_ffn", "scale_ffn", "gate_ffn"]
    for kk in WKEYS:
        W("lin_" + kk, w[kk])
    for kk in MKEYS:
        W("lin_" + kk, m[kk])
    for nm in names:
        W("lin_" + nm + "_A", lo[nm + "_A"])
        W("lin_" + nm + "_B", lo[nm + "_B"])
    W("lin_cos", cos)
    W("lin_sin", sin)
    W("lin_d_out", d_out)

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
