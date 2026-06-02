#!/usr/bin/env python3
# serenitymojo/models/klein/parity/single_block_lora_oracle.py
#
# Torch oracle for the Klein SINGLE-STREAM DiT block WITH LoRA on the attention
# projections (linear1 qkv-rows + linear2 cols). Replicates the EXACT math of
# serenitymojo/models/klein/single_block.mojo's `single_block_lora_forward`.
# Produces .bin references that the Mojo gate (single_block_lora_parity.mojo)
# reads and compares at cos >= 0.999, INCLUDING d_A and d_B for both adapters.
#
# LoRA math (matches train_step.mojo / lora_block.mojo):
#   y' = linear(x, W) + scale*((x @ Aᵀ) @ Bᵀ),  A=[rank,in], B=[out,rank].
# Targets (matches lora.mojo _map_klein_trainer single_blocks):
#   linear1 (w1) QKV-ROWS: delta added ONLY to the first 3D output rows (qkv) of
#     `fused`. input = norm [S,D], A=[r,D], B=[3D,r], delta [S,3D] -> fused[:, :3D].
#   linear2 (w2) COLS: delta on the first D INPUT cols of W2 = the att_flat slice.
#     input = att_flat [S,D], A=[r,D], B=[D,r], delta [S,D] -> out projection.
#
# NON-DEGENERATE inputs; A AND B nonzero. Real Klein head count H = 32.
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/single_block_lora_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

H = 32
Dh = 16
D = H * Dh          # 512
S = 6
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
    return LSCALE * ((x @ A.T) @ B.T)


def make_weights(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    w = {}
    w["w1"] = rnd(3 * D + 2 * F, D)
    w["w2"] = rnd(D, D + F)
    w["q_norm"] = (rnd(Dh) * 0.1 + 1.0)
    w["k_norm"] = (rnd(Dh) * 0.1 + 1.0)
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_lora(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    lo = {}
    lo["qkv_A"] = (rnd(RANK, D) * 0.02).requires_grad_(True)    # in=D
    lo["qkv_B"] = (rnd(3 * D, RANK) * 0.02).requires_grad_(True)  # out=3D (qkv rows)
    lo["out_A"] = (rnd(RANK, D) * 0.02).requires_grad_(True)    # in=D (att_flat)
    lo["out_B"] = (rnd(D, RANK) * 0.02).requires_grad_(True)    # out=D
    return lo


def make_mod(off):
    m = {}
    m["shift"] = fill(D, 0.013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale"] = fillc(D, 0.017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate"] = fill(D, 0.011, 0.3 + off, 0.4).requires_grad_(True)
    return m


def main():
    x = t2(S, D, 0.021, 0.05, 0.5).requires_grad_(True)
    w = make_weights(1)
    lo = make_lora(33)
    m = make_mod(0.0)

    cos = t2(S * H, Dh // 2, 0.03, 0.2, 1.0) * 0.6
    sin = t2(S * H, Dh // 2, 0.04, 0.5, 1.0) * 0.6

    ln = layer_norm(x)
    norm = modulate(ln, m["scale"], m["shift"])
    fused = norm @ w["w1"].T               # [S, 3D+2F]
    # LoRA qkv-rows: delta [S,3D] added to the first 3D cols (the qkv rows).
    qkv_delta = lora_delta(norm, lo["qkv_A"], lo["qkv_B"])   # [S,3D]
    pad = torch.zeros(S, 2 * F, dtype=DT)
    fused = fused + torch.cat([qkv_delta, pad], dim=1)

    qkv = fused[:, 0:3 * D]
    gate_up = fused[:, 3 * D:3 * D + 2 * F]

    q = qkv[:, 0:D].reshape(1, S, H, Dh)
    k = qkv[:, D:2 * D].reshape(1, S, H, Dh)
    v = qkv[:, 2 * D:3 * D].reshape(1, S, H, Dh)
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])

    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v)
    att_flat = att.reshape(S, D)

    mlp_gate = gate_up[:, 0:F]
    mlp_up = gate_up[:, F:2 * F]
    mlp = torch.nn.functional.silu(mlp_gate) * mlp_up

    out_in = torch.cat([att_flat, mlp], dim=1)   # [S, D+F]
    out_proj = out_in @ w["w2"].T                # [S, D]
    # LoRA on w2 cols: input is the att_flat slice [S,D]; delta [S,D] added to out.
    out_proj = out_proj + lora_delta(att_flat, lo["out_A"], lo["out_B"])
    out = residual_gate(x, m["gate"], out_proj)

    d_out = t2(S, D, 0.027, 0.11, 0.05)

    loss = (out * d_out).sum()
    loss.backward()

    def W(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
        path = os.path.join(REF_DIR, name + ".bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, flat.shape)

    # forward output (sanity)
    W("slref_out", out)

    # base grads (no-regression check)
    W("slref_d_x", x.grad)
    W("slref_d_w2_FULL", w["w2"].grad)   # full w2 base grad (frozen, sanity only)

    # LoRA grads (the deliverable)
    W("slref_qkv_dA", lo["qkv_A"].grad)
    W("slref_qkv_dB", lo["qkv_B"].grad)
    W("slref_out_dA", lo["out_A"].grad)
    W("slref_out_dB", lo["out_B"].grad)

    # exact INPUTS the Mojo gate reconstructs
    W("slin_x", x)
    for kk in ["w1", "w2", "q_norm", "k_norm"]:
        W("slin_w_%s" % kk, w[kk])
    for kk in ["qkv_A", "qkv_B", "out_A", "out_B"]:
        W("slin_lo_%s" % kk, lo[kk])
    for kk in ["shift", "scale", "gate"]:
        W("slin_m_%s" % kk, m[kk])
    W("slin_cos", cos)
    W("slin_sin", sin)
    W("slin_d_out", d_out)

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
