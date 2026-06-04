#!/usr/bin/env python3
# serenitymojo/models/wan22/parity/wan22_block_oracle.py
#
# Torch oracle for the Wan2.2 WanAttentionBlock (forward + autograd). Replicates
# the EXACT math of models/wan22/wan22_block.mojo::wan22_block_forward, which
# mirrors models/dit/wan22_dit.mojo::wan22_block_forward (WanModel WanAttentionBlock).
# Produces .bin references the Mojo gate (wan22_block_parity.mojo) reads at cos>=0.999.
#
# CONVENTIONS (must match the Mojo forward byte-for-byte):
#   PER-TOKEN AdaLN: scale/shift/gate are [S,dim] tensors (one per token).
#   mod_pre(x, scale, shift) = LN_no_affine(x) * (1+scale) + shift
#   gated_residual(x, y, gate) = x + gate*y         (per-token)
#   self-attn:  LN_no_affine -> mod_pre -> q/k/v biased linear -> qk-rms ->
#               3-axis INTERLEAVED rope -> SQUARE sdpa -> o linear -> gated resid
#   cross-attn: norm3 (AFFINE LN) -> q biased+rms ; k/v biased(context)+k-rms ->
#               RECT sdpa(Sq=S, Skv=TXT) -> o linear -> UNGATED residual (add)
#   ffn:        LN_no_affine -> mod_pre -> ffn.0 +gelu(tanh) -> ffn.2 -> gated resid
#   layer_norm eps=1e-6; rms_norm eps=1e-6 over head_dim; sdpa scale=1/sqrt(Dh).
#
# NON-DEGENERATE inputs: sinusoidal/random (NEVER modular (i*k)%9). REAL Wan2.2
# head count H=24; small Dh/S/TXT for a fast oracle.
#
# Run (SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/wan22/parity/wan22_block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
# BF16 oracle: the Mojo block now runs NATIVE bf16 compute (every Linear is
# bf16·bf16 with F32 GEMM-accumulate, matching flame-core). The reference is
# computed in bf16 too so parity reflects the bf16 rounding (cos ~0.99-0.999).
DT = torch.bfloat16

# ── dims (REAL Wan2.2 head count H=24; small Dh/S/TXT for a fast oracle) ──
H = 24
Dh = 8
DIM = H * Dh         # 192
S = 5                # image tokens
TXT = 4              # text tokens (cross-attn kv length, distinct from S)
FFN = 40             # ffn hidden (gelu)
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def fillc(n, a, b, c):
    return torch.tensor([math.cos(a * i + b) * c for i in range(n)], dtype=DT)


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
    # x [1,S,H,Dh]; cos/sin [S, Dh/2]; pairs (2i,2i+1). Each token row applies to
    # all H heads (the Mojo _expand_rope_per_head repeats the row across heads).
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
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)


def sdpa_rect(q, k, v):
    # q [1,S,H,Dh], k/v [1,TXT,H,Dh] -> out [1,S,H,Dh]
    qh = q.permute(0, 2, 1, 3)   # [1,H,S,Dh]
    kh = k.permute(0, 2, 1, 3)   # [1,H,TXT,Dh]
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE  # [1,H,S,TXT]
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh              # [1,H,S,Dh]
    return out.permute(0, 2, 1, 3)


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
        v.requires_grad_(True)
    return w


def make_mod():
    m = {}
    m["shift_sa"] = t2(S, DIM, 0.013, 0.1, 0.3).requires_grad_(True)
    m["scale_sa"] = t2(S, DIM, 0.017, 0.2, 0.2).requires_grad_(True)
    m["gate_sa"] = t2(S, DIM, 0.011, 0.3, 0.4).requires_grad_(True)
    m["shift_ffn"] = t2(S, DIM, 0.019, 0.4, 0.3).requires_grad_(True)
    m["scale_ffn"] = t2(S, DIM, 0.015, 0.5, 0.2).requires_grad_(True)
    m["gate_ffn"] = t2(S, DIM, 0.012, 0.6, 0.4).requires_grad_(True)
    return m


def main():
    x = t2(S, DIM, 0.021, 0.05, 0.5).requires_grad_(True)
    context = t2(TXT, DIM, 0.023, 0.07, 0.5).requires_grad_(True)
    w = make_weights(1)
    m = make_mod()
    cos = t2(S, Dh // 2, 0.03, 0.2, 1.0) * 0.6
    sin = t2(S, Dh // 2, 0.04, 0.5, 1.0) * 0.6

    # ── self-attention ──
    sa_in = mod_pre(x, m["scale_sa"], m["shift_sa"])
    q = (sa_in @ w["sa_wq"].T + w["sa_bq"]).reshape(1, S, H, Dh)
    k = (sa_in @ w["sa_wk"].T + w["sa_bk"]).reshape(1, S, H, Dh)
    v = (sa_in @ w["sa_wv"].T + w["sa_bv"]).reshape(1, S, H, Dh)
    q = rms_norm_lastdim(q, w["sa_qn"])
    k = rms_norm_lastdim(k, w["sa_kn"])
    q = rope_interleaved(q, cos, sin)
    k = rope_interleaved(k, cos, sin)
    sa_att = sdpa_square(q, k, v).reshape(S, DIM)
    sa_out = sa_att @ w["sa_wo"].T + w["sa_bo"]
    x_sa = gated_residual(x, sa_out, m["gate_sa"])

    # ── cross-attention ──
    n3 = layer_norm_affine(x_sa, w["n3_w"], w["n3_b"])
    caq = (n3 @ w["ca_wq"].T + w["ca_bq"]).reshape(1, S, H, Dh)
    cak = (context @ w["ca_wk"].T + w["ca_bk"]).reshape(1, TXT, H, Dh)
    cav = (context @ w["ca_wv"].T + w["ca_bv"]).reshape(1, TXT, H, Dh)
    caq = rms_norm_lastdim(caq, w["ca_qn"])
    cak = rms_norm_lastdim(cak, w["ca_kn"])
    ca_att = sdpa_rect(caq, cak, cav).reshape(S, DIM)
    ca_out = ca_att @ w["ca_wo"].T + w["ca_bo"]
    x_ca = x_sa + ca_out

    # ── ffn ──
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
        path = os.path.join(REF_DIR, name + ".bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, flat.shape)

    # forward output + input grads
    W("ref_x_out", x_final)
    W("ref_d_x", x.grad)
    W("ref_d_context", context.grad)

    WKEYS = ["sa_wq", "sa_wk", "sa_wv", "sa_wo", "sa_bq", "sa_bk", "sa_bv", "sa_bo",
             "sa_qn", "sa_kn",
             "ca_wq", "ca_wk", "ca_wv", "ca_wo", "ca_bq", "ca_bk", "ca_bv", "ca_bo",
             "ca_qn", "ca_kn",
             "n3_w", "n3_b", "ffn0_w", "ffn0_b", "ffn2_w", "ffn2_b"]
    MKEYS = ["shift_sa", "scale_sa", "gate_sa", "shift_ffn", "scale_ffn", "gate_ffn"]
    for kk in WKEYS:
        W("ref_d_" + kk, w[kk].grad)
    for kk in MKEYS:
        W("ref_d_" + kk, m[kk].grad)

    # inputs the gate cannot regenerate (random weights)
    W("in_x", x)
    W("in_context", context)
    for kk in WKEYS:
        W("in_" + kk, w[kk])
    for kk in MKEYS:
        W("in_" + kk, m[kk])
    W("in_cos", cos)
    W("in_sin", sin)
    W("in_d_out", d_out)

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
