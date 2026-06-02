#!/usr/bin/env python3
# serenitymojo/models/anima/parity/block_oracle.py
#
# Torch oracle for ONE Anima MiniTrainDIT transformer block (forward + autograd
# grads). Built INDEPENDENTLY from the REFERENCE math (inference-flame
# anima.rs::transformer_block + diffusers Cosmos), NOT from the Mojo block — so
# it is a real adversarial check. Produces .bin references that the Mojo gate
# (block_parity.mojo) reads and compares at cos >= 0.999.
#
# REFERENCE (anima.rs:458-511 transformer_block; F32 residual stream):
#   t_silu = silu(t_cond)                      # [B,2048]
#   for each sub in (self_attn, cross_attn, mlp):
#     mod_h   = silu(t_cond) @ mod1.T          # [B,256]
#     mod_out = mod_h @ mod2.T + base_adaln    # [B,6144]
#     shift,scale,gate = chunk3(mod_out)       # each [B,D]
#     x_mod = (1+scale)*LayerNorm_noaffine(x) + shift   (eps=1e-6)
#     sub_out = SUB(x_mod)
#     x = x + gate * sub_out                   # gate [B,D] broadcast over seq
#   SELF (anima.rs:351-391): q,k,v=Linear_nobias(x_mod); reshape [B,S,H,Dh];
#     qk per-head RMSNorm(eps=1e-6); HALF-SPLIT RoPE on q,k (interleaved=False);
#     sdpa scale=1/sqrt(Dh); out=Linear_nobias.
#   CROSS (anima.rs:397-441): q=Linear(x_mod) [B,S_img,2048];
#     k,v=Linear(context) [B,S_txt,1024->2048]; reshape; qk per-head RMSNorm;
#     NO RoPE; sdpa NO MASK; out=Linear.
#   MLP (anima.rs:447-452): Linear(2048->F) -> GELU(tanh) -> Linear(F->2048).
#
# AdaLN-pre = LayerNorm with NO affine (flame-core modulate_pre_fused needs_grad
# path uses layer_norm(x,[dim],None,None,eps)) — VERIFIED 3-source by skeptic.
#
# F32 EVERYWHERE (the Mojo gate runs the F32 path too): cos compares clean,
# no BF16 floor. REAL head dims H=16, Dh=128. Small S_img/S_txt to keep it light
# on the shared GPU; the math is S-independent so cos at real H/Dh is the gate.
#
# Run (SEPARATE command, NEVER chained after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64   # F64 reference interior (gate compares cos in F64-vs-F32)

# ── dims: REAL Anima head count/dim; small seqs for a fast, light oracle ──
B = 1
H = 16
Dh = 128
D = H * Dh          # 2048 (== ANIMA_HIDDEN)
S_IMG = 6
S_TXT = 8
JOINT = 1024        # cross-attn context dim
F = 32              # mlp hidden (tiny stand-in for 8192; math identical)
ADALN = 256         # adaln-lora down dim
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
THETA = 10000.0

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


def rnd(g, *shape):
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)


# ── ops (match anima.rs math) ──
def layer_norm_noaffine(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS)


def rms_norm_lastdim(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def rope_halfsplit(x, cos, sin):
    # x [B,S,H,Dh]; cos/sin [S, Dh/2] (per position, broadcast over B,H).
    # anima.rs:379 rope_halfsplit_bf16 (interleaved=False):
    #   out[i]      = x[i]*cos - x[i+half]*sin
    #   out[i+half] = x[i+half]*cos + x[i]*sin
    half = Dh // 2
    c = cos.reshape(1, x.shape[1], 1, half)
    s = sin.reshape(1, x.shape[1], 1, half)
    x1 = x[..., :half]
    x2 = x[..., half:]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2], dim=-1)


def sdpa(q, k, v):
    # q [B,Sq,H,Dh], k/v [B,Skv,H,Dh] -> [B,H,Sq,Dh]
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)   # [B,Sq,H,Dh]


def gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def main():
    g = torch.Generator().manual_seed(1)

    # ── inputs ──
    x_init = (rnd(g, B, S_IMG, D) * 0.5)
    x_leaf = x_init.clone().requires_grad_(True)
    x = x_leaf
    t_cond = (rnd(g, B, D) * 0.5)
    # t_silu is the differentiated LEAF (the block consumes silu(t_cond); the
    # gate's d_t_silu compares grad-of-silu-output, the value the block returns).
    t_silu = torch.nn.functional.silu(t_cond).clone().requires_grad_(True)
    base_adaln = (rnd(g, B, 3 * D) * 0.3).requires_grad_(True)
    context = (rnd(g, B, S_TXT, JOINT) * 0.5).requires_grad_(True)

    # rope tables (per position [S_IMG, Dh/2]) — non-degenerate angles.
    half = Dh // 2
    pos = torch.arange(S_IMG, dtype=DT).reshape(S_IMG, 1)
    freq = torch.tensor(
        [1.0 / (THETA ** (2.0 * i / Dh)) for i in range(half)], dtype=DT
    ).reshape(1, half)
    ang = pos * freq            # [S_IMG, half]
    cos = torch.cos(ang)
    sin = torch.sin(ang)

    # ── weights (per-block, 20 base + handled below) ──
    def lin(out_f, in_f):
        return rnd(g, out_f, in_f).requires_grad_(True)

    w = {}
    for sub in ("sa", "ca", "mlp"):
        w[f"{sub}_mod1"] = (rnd(g, ADALN, D) * 0.1).requires_grad_(True)  # [256,2048]
        w[f"{sub}_mod2"] = (rnd(g, 3 * D, ADALN) * 0.1).requires_grad_(True)  # [6144,256]
    w["sa_q"] = lin(D, D); w["sa_k"] = lin(D, D); w["sa_v"] = lin(D, D); w["sa_out"] = lin(D, D)
    w["sa_qn"] = (rnd(g, Dh) * 0.1 + 1.0).requires_grad_(True)
    w["sa_kn"] = (rnd(g, Dh) * 0.1 + 1.0).requires_grad_(True)
    w["ca_q"] = lin(D, D); w["ca_k"] = lin(D, JOINT); w["ca_v"] = lin(D, JOINT); w["ca_out"] = lin(D, D)
    w["ca_qn"] = (rnd(g, Dh) * 0.1 + 1.0).requires_grad_(True)
    w["ca_kn"] = (rnd(g, Dh) * 0.1 + 1.0).requires_grad_(True)
    w["mlp1"] = lin(F, D); w["mlp2"] = lin(D, F)

    # ── forward (t_silu is the leaf) ──
    def adaln_mod(sub):
        h = t_silu @ w[f"{sub}_mod1"].T           # [B,256]
        mod_out = h @ w[f"{sub}_mod2"].T + base_adaln   # [B,6144]
        shift = mod_out[:, 0:D]
        scale = mod_out[:, D:2 * D]
        gate = mod_out[:, 2 * D:3 * D]
        return shift, scale, gate

    def adaln_pre(xx, shift, scale):
        ln = layer_norm_noaffine(xx)
        return (1.0 + scale).unsqueeze(1) * ln + shift.unsqueeze(1)

    # self-attn
    sh, sc, ga = adaln_mod("sa")
    xmod = adaln_pre(x, sh, sc)
    q = (xmod @ w["sa_q"].T).reshape(B, S_IMG, H, Dh)
    k = (xmod @ w["sa_k"].T).reshape(B, S_IMG, H, Dh)
    v = (xmod @ w["sa_v"].T).reshape(B, S_IMG, H, Dh)
    q = rms_norm_lastdim(q, w["sa_qn"])
    k = rms_norm_lastdim(k, w["sa_kn"])
    q = rope_halfsplit(q, cos, sin)
    k = rope_halfsplit(k, cos, sin)
    att = sdpa(q, k, v).reshape(B, S_IMG, D)
    sa_out = att @ w["sa_out"].T
    x = x + ga.unsqueeze(1) * sa_out

    # cross-attn
    sh, sc, ga = adaln_mod("ca")
    xmod = adaln_pre(x, sh, sc)
    q = (xmod @ w["ca_q"].T).reshape(B, S_IMG, H, Dh)
    k = (context @ w["ca_k"].T).reshape(B, S_TXT, H, Dh)
    v = (context @ w["ca_v"].T).reshape(B, S_TXT, H, Dh)
    q = rms_norm_lastdim(q, w["ca_qn"])
    k = rms_norm_lastdim(k, w["ca_kn"])
    att = sdpa(q, k, v).reshape(B, S_IMG, D)
    ca_out = att @ w["ca_out"].T
    x = x + ga.unsqueeze(1) * ca_out

    # mlp
    sh, sc, ga = adaln_mod("mlp")
    xmod = adaln_pre(x, sh, sc)
    h = gelu_tanh(xmod @ w["mlp1"].T)
    mlp_out = h @ w["mlp2"].T
    x = x + ga.unsqueeze(1) * mlp_out

    out = x

    # ── upstream grad (non-degenerate) ──
    d_out = (rnd(g, B, S_IMG, D) * 0.05)
    (out * d_out).sum().backward()

    # ── references ──
    W("ref_out", out)
    W("ref_d_x", x_leaf.grad)
    W("ref_d_t_silu", t_silu.grad)
    for kk in ("sa_q", "sa_k", "sa_v", "sa_out", "sa_qn", "sa_kn",
               "ca_q", "ca_k", "ca_v", "ca_out", "ca_qn", "ca_kn",
               "mlp1", "mlp2",
               "sa_mod1", "sa_mod2", "ca_mod1", "ca_mod2", "mlp_mod1", "mlp_mod2"):
        W("ref_d_" + kk, w[kk].grad)

    # ── inputs the Mojo gate reconstructs ──
    W("in_x", x_init)
    W("in_t_cond", t_cond.detach())
    W("in_base_adaln", base_adaln.detach())
    W("in_context", context.detach())
    W("in_cos", cos)
    W("in_sin", sin)
    W("in_d_out", d_out)
    for kk in w:
        W("in_w_" + kk, w[kk].detach())
    print("DONE")


if __name__ == "__main__":
    main()
