#!/usr/bin/env python3
# serenitymojo/models/sd35/parity/sd35_block_oracle.py
#
# Torch oracle for the SD3.5 MMDiT JointTransformerBlock (forward + autograd).
# Replicates the EXACT math of serenitymojo/models/sd35/sd35_block.mojo's
# sd35_joint_block_forward (the standard non-dual, non-pre_only path), which
# mirrors the inference forward models/dit/sd3_mmdit.mojo `_sd3_joint_block`.
# Produces .bin references the Mojo gate (sd35_block_parity.mojo) reads at
# cos >= 0.999. Also emits a LoRA-on-x_block-qkv reference (d_A / d_B).
#
# CONVENTIONS (must match the Mojo forward byte-for-byte):
#   modulate(x, scale, shift) = (1 + scale) * x + shift   (scale/shift [D] broadcast over N)
#   gated_residual(s, gate, y) = s + gate[None,:] * y      (gate [D] broadcast over N)
#   token norm : LayerNorm no-affine, eps 1e-6
#   qk norm    : RMSNorm over head_dim, eps 1e-6  (ln_q/ln_k, no bias)
#   MLP        : fc1 -> GELU(tanh approx) -> fc2  (linears carry bias)
#   NO RoPE
#   sdpa       : non-causal, scale = 1/sqrt(Dh)
#   joint concat order : context FIRST, then x (axis = sequence)
#   linears (qkv,proj,fc1,fc2) carry BIAS.
#
# NON-DEGENERATE inputs: sinusoidal / randn fills (NEVER modular (i*k)%9).
# REAL SD3.5 head count H = 24; small N/Dh for a fast oracle (D = H*Dh).
#
# Run (SEPARATE command, never chained after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/sd35/parity/sd35_block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── dims (REAL SD3.5 head count H=24; small N/Dh for a fast oracle) ──
H = 24
Dh = 8
D = H * Dh          # 192
N_CTX = 3
N_IMG = 5
S = N_CTX + N_IMG
MLP = 32            # mlp hidden (fc1 out); small for speed
EPS = 1e-6
QK_EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
RANK = 4            # LoRA rank for the qkv adapter
ALPHA = 8.0
LSCALE = ALPHA / RANK

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


def gated_residual(s, gate, y):
    return s + gate * y


def rms_norm_lastdim(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + QK_EPS) * weight


def gelu_tanh(x):
    # tanh-approx GELU (matches ops/activations.gelu + gelu_backward.cu)
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def sdpa(q, k, v):
    # q,k,v [1,S,H,Dh] -> [1,H,S,Dh]
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)  # [1,S,H,Dh]


def make_stream(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    w = {}
    w["wqkv"] = rnd(3 * D, D) * 0.05
    w["bqkv"] = rnd(3 * D) * 0.05
    w["wproj"] = rnd(D, D) * 0.05
    w["bproj"] = rnd(D) * 0.05
    w["wfc1"] = rnd(MLP, D) * 0.05
    w["bfc1"] = rnd(MLP) * 0.05
    w["wfc2"] = rnd(D, MLP) * 0.05
    w["bfc2"] = rnd(D) * 0.05
    w["q_norm"] = (rnd(Dh) * 0.1 + 1.0)
    w["k_norm"] = (rnd(Dh) * 0.1 + 1.0)
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


def stream_pre(s, w, m, N, lora=None):
    ln1 = layer_norm(s)
    norm = modulate(ln1, m["scale_msa"], m["shift_msa"])
    qkv = norm @ w["wqkv"].T + w["bqkv"]              # [N, 3D]
    if lora is not None:
        qkv = qkv + LSCALE * ((norm @ lora["A"].T) @ lora["B"].T)
    q = qkv[:, 0:D].reshape(1, N, H, Dh)
    k = qkv[:, D:2 * D].reshape(1, N, H, Dh)
    v = qkv[:, 2 * D:3 * D].reshape(1, N, H, Dh)
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])
    return q, k, v


def stream_post(s, att, w, m, N):
    proj = att @ w["wproj"].T + w["bproj"]
    attn_res = gated_residual(s, m["gate_msa"], proj)
    ln2 = layer_norm(attn_res)
    mlp_in = modulate(ln2, m["scale_mlp"], m["shift_mlp"])
    h1 = mlp_in @ w["wfc1"].T + w["bfc1"]
    hg = gelu_tanh(h1)
    mlp = hg @ w["wfc2"].T + w["bfc2"]
    out = gated_residual(attn_res, m["gate_mlp"], mlp)
    return out


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


def run(with_lora):
    context = t2(N_CTX, D, 0.021, 0.05, 0.5).requires_grad_(True)
    x = t2(N_IMG, D, 0.023, 0.07, 0.5).requires_grad_(True)
    cw = make_stream(1)
    xw = make_stream(2)
    cm = make_mod(0.0)
    xm = make_mod(1.0)

    lora = None
    if with_lora:
        g = torch.Generator().manual_seed(7)
        lora = {
            "A": (torch.randn(RANK, D, generator=g, dtype=torch.float32).to(DT) * 0.02).requires_grad_(True),
            "B": (torch.randn(3 * D, RANK, generator=g, dtype=torch.float32).to(DT) * 0.0).requires_grad_(True),
        }
        # B starts at 0 (PEFT identity); to get nonzero d_B we perturb B a touch.
        lora["B"] = (torch.randn(3 * D, RANK, generator=g, dtype=torch.float32).to(DT) * 0.02).requires_grad_(True)

    cq, ck, cv = stream_pre(context, cw, cm, N_CTX)
    xq, xk, xv = stream_pre(x, xw, xm, N_IMG, lora=lora)

    q = torch.cat([cq, xq], dim=1)
    k = torch.cat([ck, xk], dim=1)
    v = torch.cat([cv, xv], dim=1)
    att = sdpa(q, k, v)                  # [1,S,H,Dh]
    ctx_att = att[:, 0:N_CTX].reshape(N_CTX, D)
    x_att = att[:, N_CTX:N_CTX + N_IMG].reshape(N_IMG, D)

    ctx_out = stream_post(context, ctx_att, cw, cm, N_CTX)
    x_out = stream_post(x, x_att, xw, xm, N_IMG)

    d_ctx = t2(N_CTX, D, 0.027, 0.11, 0.05)
    d_x = t2(N_IMG, D, 0.029, 0.13, 0.05)
    loss = (ctx_out * d_ctx).sum() + (x_out * d_x).sum()
    loss.backward()

    if not with_lora:
        W("ref_ctx_out", ctx_out)
        W("ref_x_out", x_out)
        W("ref_d_ctx", context.grad)
        W("ref_d_x", x.grad)
        for nm, w in [("ctx", cw), ("x", xw)]:
            W("ref_%s_d_wqkv" % nm, w["wqkv"].grad)
            W("ref_%s_d_bqkv" % nm, w["bqkv"].grad)
            W("ref_%s_d_wproj" % nm, w["wproj"].grad)
            W("ref_%s_d_bproj" % nm, w["bproj"].grad)
            W("ref_%s_d_wfc1" % nm, w["wfc1"].grad)
            W("ref_%s_d_bfc1" % nm, w["bfc1"].grad)
            W("ref_%s_d_wfc2" % nm, w["wfc2"].grad)
            W("ref_%s_d_bfc2" % nm, w["bfc2"].grad)
            W("ref_%s_d_qnorm" % nm, w["q_norm"].grad)
            W("ref_%s_d_knorm" % nm, w["k_norm"].grad)
        for nm, m in [("ctx", cm), ("x", xm)]:
            for kk in ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]:
                W("ref_%s_d_%s" % (nm, kk), m[kk].grad)
        # dump the exact INPUTS the Mojo gate cannot regenerate (randn weights).
        W("in_context", context)
        W("in_x", x)
        for nm, w in [("cw", cw), ("xw", xw)]:
            for kk in ["wqkv", "bqkv", "wproj", "bproj", "wfc1", "bfc1", "wfc2", "bfc2", "q_norm", "k_norm"]:
                W("in_%s_%s" % (nm, kk), w[kk])
        for nm, m in [("cm", cm), ("xm", xm)]:
            for kk in ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]:
                W("in_%s_%s" % (nm, kk), m[kk])
        W("in_d_ctx", d_ctx)
        W("in_d_x", d_x)
        print("forward loss (no-lora) =", float(loss))
    else:
        W("ref_lora_d_A", lora["A"].grad)
        W("ref_lora_d_B", lora["B"].grad)
        W("in_lora_A", lora["A"])
        W("in_lora_B", lora["B"])
        print("forward loss (lora) =", float(loss))


def main():
    run(with_lora=False)
    run(with_lora=True)
    print("DONE")


if __name__ == "__main__":
    main()
