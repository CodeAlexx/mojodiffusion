#!/usr/bin/env python3
# serenitymojo/models/sd35/parity/sd35_dual_block_oracle.py
#
# Torch oracle for the SD3.5 DUAL-attention joint block (forward + autograd).
# The math is the diffusers JointTransformerBlock(use_dual_attention=True) path,
# already verified byte-faithful to the REAL diffusers block in
# sd35_dual_block_vs_diffusers.py (cos=1.0 F64). This dumps .bin references the
# Mojo gate (sd35_dual_block_parity.mojo) reads at cos >= 0.999.
#
# Dual block = the non-dual block (sd35_block_oracle.py) PLUS:
#   - x-stream norm is SD35AdaLayerNormZeroX: one shared LayerNorm of x, then
#       norm_h  = (1+scale_msa )*ln_x + shift_msa   (joint attn input)
#       norm_h2 = (1+scale_msa2)*ln_x + shift_msa2  (self-attn2 input)
#   - a SECOND self-attention attn2 on the x stream only (own to_q/k/v/out +
#     qk-rms norms), added to hidden BEFORE the MLP, gated by gate_msa2:
#       hidden += gate_msa2 * attn2(norm_h2)
#   The context stream is identical to the non-dual block (AdaLayerNormZero).
# Also dumps LoRA refs on x-qkv (joint) AND attn2-qkv (the new dual slot).
#
# Run (SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/sd35/parity/sd35_dual_block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

H = 24
Dh = 8
D = H * Dh          # 192
N_CTX = 3
N_IMG = 5
S = N_CTX + N_IMG
MLP = 32
EPS = 1e-6
QK_EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
RANK = 4
ALPHA = 8.0
LSCALE = ALPHA / RANK

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


def gated_residual(s, gate, y):
    return s + gate * y


def rms_norm_lastdim(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + QK_EPS) * weight


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def sdpa(q, k, v):
    qh = q.permute(0, 2, 1, 3); kh = k.permute(0, 2, 1, 3); vh = v.permute(0, 2, 1, 3)
    attn = torch.softmax((qh @ kh.transpose(-1, -2)) * SCALE, dim=-1)
    return (attn @ vh).permute(0, 2, 1, 3)


def make_stream(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    w = {}
    w["wqkv"] = rnd(3 * D, D) * 0.05; w["bqkv"] = rnd(3 * D) * 0.05
    w["wproj"] = rnd(D, D) * 0.05; w["bproj"] = rnd(D) * 0.05
    w["wfc1"] = rnd(MLP, D) * 0.05; w["bfc1"] = rnd(MLP) * 0.05
    w["wfc2"] = rnd(D, MLP) * 0.05; w["bfc2"] = rnd(D) * 0.05
    w["q_norm"] = (rnd(Dh) * 0.1 + 1.0); w["k_norm"] = (rnd(Dh) * 0.1 + 1.0)
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_attn2(seed):
    # self-attention only: qkv/proj + qk norms (no mlp, no added-kv)
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    w = {}
    w["wqkv"] = rnd(3 * D, D) * 0.05; w["bqkv"] = rnd(3 * D) * 0.05
    w["wproj"] = rnd(D, D) * 0.05; w["bproj"] = rnd(D) * 0.05
    w["q_norm"] = (rnd(Dh) * 0.1 + 1.0); w["k_norm"] = (rnd(Dh) * 0.1 + 1.0)
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_mod(off):
    m = {}
    m["shift_msa"] = fill(D, 0.013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale_msa"] = fillc(D, 0.017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate_msa"] = fill(D, 0.011, 0.3 + off, 0.4).requires_grad_(True)
    m["shift_mlp"] = fillc(D, 0.019, 0.4 + off, 0.3).requires_grad_(True)
    m["scale_mlp"] = fill(D, 0.015, 0.5 + off, 0.2).requires_grad_(True)
    m["gate_mlp"] = fillc(D, 0.012, 0.6 + off, 0.4).requires_grad_(True)
    return m


def make_mod2(off):
    # the extra msa2 triple for the x stream (SD35AdaLayerNormZeroX)
    m = {}
    m["shift_msa2"] = fill(D, 0.014, 0.15 + off, 0.3).requires_grad_(True)
    m["scale_msa2"] = fillc(D, 0.016, 0.25 + off, 0.2).requires_grad_(True)
    m["gate_msa2"] = fill(D, 0.012, 0.35 + off, 0.4).requires_grad_(True)
    return m


def qkv_split(qkv, N):
    q = qkv[:, 0:D].reshape(1, N, H, Dh)
    k = qkv[:, D:2 * D].reshape(1, N, H, Dh)
    v = qkv[:, 2 * D:3 * D].reshape(1, N, H, Dh)
    return q, k, v


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


def run(with_lora):
    context = t2(N_CTX, D, 0.021, 0.05, 0.5).requires_grad_(True)
    x = t2(N_IMG, D, 0.023, 0.07, 0.5).requires_grad_(True)
    cw = make_stream(1)     # context stream (joint attn + mlp)
    xw = make_stream(2)     # x stream joint attn + mlp
    a2 = make_attn2(3)      # x stream SECOND self-attention
    cm = make_mod(0.0)
    xm = make_mod(1.0)
    xm2 = make_mod2(1.0)

    lora_x = lora_a2 = None
    if with_lora:
        g = torch.Generator().manual_seed(7)
        def mk():
            return {
                "A": (torch.randn(RANK, D, generator=g, dtype=torch.float32).to(DT) * 0.02).requires_grad_(True),
                "B": (torch.randn(3 * D, RANK, generator=g, dtype=torch.float32).to(DT) * 0.02).requires_grad_(True),
            }
        lora_x = mk()      # on x joint-attn qkv
        lora_a2 = mk()     # on attn2 qkv (the new dual slot)

    # ── context stream pre (AdaLayerNormZero) ──
    c_norm = modulate(layer_norm(context), cm["scale_msa"], cm["shift_msa"])
    cqkv = c_norm @ cw["wqkv"].T + cw["bqkv"]
    cq, ck, cv = qkv_split(cqkv, N_CTX)
    cq = rms_norm_lastdim(cq, cw["q_norm"]); ck = rms_norm_lastdim(ck, cw["k_norm"])

    # ── x stream pre (SD35AdaLayerNormZeroX): shared ln, two modulations ──
    ln_x = layer_norm(x)
    x_norm = modulate(ln_x, xm["scale_msa"], xm["shift_msa"])     # joint attn input
    x_norm2 = modulate(ln_x, xm2["scale_msa2"], xm2["shift_msa2"])  # attn2 input
    xqkv = x_norm @ xw["wqkv"].T + xw["bqkv"]
    if lora_x is not None:
        xqkv = xqkv + LSCALE * ((x_norm @ lora_x["A"].T) @ lora_x["B"].T)
    xq, xk, xv = qkv_split(xqkv, N_IMG)
    xq = rms_norm_lastdim(xq, xw["q_norm"]); xk = rms_norm_lastdim(xk, xw["k_norm"])

    # ── joint attention (context first, then x) ──
    q = torch.cat([cq, xq], dim=1); k = torch.cat([ck, xk], dim=1); v = torch.cat([cv, xv], dim=1)
    att = sdpa(q, k, v)
    ctx_att = att[:, 0:N_CTX].reshape(N_CTX, D)
    x_att = att[:, N_CTX:N_CTX + N_IMG].reshape(N_IMG, D)

    # ── x stream: attn1 residual ──
    x_proj = x_att @ xw["wproj"].T + xw["bproj"]
    x_hid = gated_residual(x, xm["gate_msa"], x_proj)

    # ── x stream: SECOND self-attention (attn2) on x_norm2 ──
    a2qkv = x_norm2 @ a2["wqkv"].T + a2["bqkv"]
    if lora_a2 is not None:
        a2qkv = a2qkv + LSCALE * ((x_norm2 @ lora_a2["A"].T) @ lora_a2["B"].T)
    a2q, a2k, a2v = qkv_split(a2qkv, N_IMG)
    a2q = rms_norm_lastdim(a2q, a2["q_norm"]); a2k = rms_norm_lastdim(a2k, a2["k_norm"])
    a2_att = sdpa(a2q, a2k, a2v).reshape(N_IMG, D)
    a2_proj = a2_att @ a2["wproj"].T + a2["bproj"]
    x_hid = gated_residual(x_hid, xm2["gate_msa2"], a2_proj)

    # ── x stream: MLP ──
    x_ln2 = layer_norm(x_hid)
    x_mlp_in = modulate(x_ln2, xm["scale_mlp"], xm["shift_mlp"])
    x_h1 = gelu_tanh(x_mlp_in @ xw["wfc1"].T + xw["bfc1"])
    x_mlp = x_h1 @ xw["wfc2"].T + xw["bfc2"]
    x_out = gated_residual(x_hid, xm["gate_mlp"], x_mlp)

    # ── context stream post (identical to non-dual) ──
    c_proj = ctx_att @ cw["wproj"].T + cw["bproj"]
    c_res = gated_residual(context, cm["gate_msa"], c_proj)
    c_ln2 = layer_norm(c_res)
    c_mlp_in = modulate(c_ln2, cm["scale_mlp"], cm["shift_mlp"])
    c_h1 = gelu_tanh(c_mlp_in @ cw["wfc1"].T + cw["bfc1"])
    c_mlp = c_h1 @ cw["wfc2"].T + cw["bfc2"]
    ctx_out = gated_residual(c_res, cm["gate_mlp"], c_mlp)

    d_ctx = t2(N_CTX, D, 0.027, 0.11, 0.05)
    d_x = t2(N_IMG, D, 0.029, 0.13, 0.05)
    loss = (ctx_out * d_ctx).sum() + (x_out * d_x).sum()
    loss.backward()

    if not with_lora:
        W("dref_ctx_out", ctx_out); W("dref_x_out", x_out)
        W("dref_d_ctx", context.grad); W("dref_d_x", x.grad)
        for nm, w in [("ctx", cw), ("x", xw)]:
            for kk in ["wqkv", "bqkv", "wproj", "bproj", "wfc1", "bfc1", "wfc2", "bfc2", "q_norm", "k_norm"]:
                W("dref_%s_d_%s" % (nm, kk), w[kk].grad)
        for kk in ["wqkv", "bqkv", "wproj", "bproj", "q_norm", "k_norm"]:
            W("dref_a2_d_%s" % kk, a2[kk].grad)
        for nm, m in [("ctx", cm), ("x", xm)]:
            for kk in ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]:
                W("dref_%s_d_%s" % (nm, kk), m[kk].grad)
        for kk in ["shift_msa2", "scale_msa2", "gate_msa2"]:
            W("dref_x_d_%s" % kk, xm2[kk].grad)
        # inputs
        W("din_context", context); W("din_x", x)
        for nm, w in [("cw", cw), ("xw", xw)]:
            for kk in ["wqkv", "bqkv", "wproj", "bproj", "wfc1", "bfc1", "wfc2", "bfc2", "q_norm", "k_norm"]:
                W("din_%s_%s" % (nm, kk), w[kk])
        for kk in ["wqkv", "bqkv", "wproj", "bproj", "q_norm", "k_norm"]:
            W("din_a2_%s" % kk, a2[kk])
        for nm, m in [("cm", cm), ("xm", xm)]:
            for kk in ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]:
                W("din_%s_%s" % (nm, kk), m[kk])
        for kk in ["shift_msa2", "scale_msa2", "gate_msa2"]:
            W("din_xm2_%s" % kk, xm2[kk])
        W("din_d_ctx", d_ctx); W("din_d_x", d_x)
        print("dual forward loss (no-lora) =", float(loss))
    else:
        W("dref_lora_x_d_A", lora_x["A"].grad); W("dref_lora_x_d_B", lora_x["B"].grad)
        W("dref_lora_a2_d_A", lora_a2["A"].grad); W("dref_lora_a2_d_B", lora_a2["B"].grad)
        W("din_lora_x_A", lora_x["A"]); W("din_lora_x_B", lora_x["B"])
        W("din_lora_a2_A", lora_a2["A"]); W("din_lora_a2_B", lora_a2["B"])
        print("dual forward loss (lora) =", float(loss))


def main():
    run(with_lora=False)
    run(with_lora=True)
    print("DONE")


if __name__ == "__main__":
    main()
