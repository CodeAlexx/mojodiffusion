#!/usr/bin/env python3
# serenitymojo/models/krea2/parity/krea2_block_oracle.py
#
# Torch oracle for ONE Krea-2-Raw SingleStreamBlock WITH LoRA on the 8 block
# nn.Linears (wq wk wv gate wo mlp_gate mlp_up mlp_down). Replicates the EXACT
# math of the ai-toolkit reference (extensions_built_in/diffusion_models/krea2/
# src/mmdit.py: SingleStreamBlock.forward 328-337, Attention.forward 212-228,
# QKNorm/RMSNorm 153-177, SwiGLU 180-194, DoubleSharedModulation 122-133) and of
# serenitymojo/models/krea2/krea2_block.mojo's forward. Dumps .bin references the
# Mojo gate (krea2_block_parity.mojo) reads and compares at cos >= 0.999,
# INCLUDING d_x and d_A/d_B for all 8 adapters.
#
# LoRA math (ai-toolkit lora_special.py / network_mixins.py):
#   y' = linear(x, W) + scale*((x @ Aᵀ) @ Bᵀ),  A=[rank,in] (lora_down),
#        B=[out,rank] (lora_up), scale = alpha/rank.  (multiplier=1.0 in training)
#
# REAL head counts: HEADS=48, KVHEADS=12, HEADDIM=128 (single_mmdit_large_wide).
# NON-DEGENERATE inputs (sinusoidal/seeded-random, NEVER modular fills — they
# alias at H·Dh strides → false zero-grad). NONZERO LoRA B (so dA is non-
# degenerate, unlike the production zero-init).  Small L for oracle speed.
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/krea2/parity/krea2_block_oracle.py

import math
import os
import struct

import torch

torch.manual_seed(0)
DT = torch.float64

# ── dims (MUST match krea2_block_parity.mojo) ────────────────────────────────
HEADS = 48
KVHEADS = 12
HEADDIM = 128
FEATURES = HEADS * HEADDIM          # 6144
N_REP = HEADS // KVHEADS            # 4
HALF = HEADDIM // 2                 # 64
L = 8                              # small seq for speed
MLPDIM = 256                       # SwiGLU hidden (kept small for the gate; the
#                                    real model's 16384 is irrelevant to the
#                                    chain-rule check — what matters is the down
#                                    weight shape [FEATURES, MLPDIM]).
EPS = 1e-5
SCALE = 1.0 / math.sqrt(HEADDIM)

RANK = 8
ALPHA = 16.0
LSCALE = ALPHA / RANK              # 2.0

REF_DIR = os.path.dirname(os.path.abspath(__file__))


# ── deterministic non-degenerate fills ──────────────────────────────────────
def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def t2(n, m, a, b, c):
    return fill(n * m, a, b, c).reshape(n, m)


def rnd(g, *shape, sc=1.0):
    return (torch.randn(*shape, generator=g, dtype=torch.float32).to(DT) * sc)


# ── leaf ops (mmdit.py-faithful) ─────────────────────────────────────────────
def rms_norm_scale1(x, raw_scale):
    """RMSNorm (mmdit.py:172-177): F32-internal, weight = raw_scale + 1.0."""
    w = raw_scale + 1.0
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * w


def modulate(x, scale, shift):
    return (1.0 + scale) * x + shift


def residual_gate(x, gate, y):
    return x + gate * y


def rope_interleaved(x, cos, sin, heads):
    """Interleaved RoPE matching ops/rope.rope_interleaved + the Mojo tiling.
    x: [1, L, heads, HEADDIM]; cos/sin: per-token [L, HALF] broadcast over heads.
      out[..,2i]   = x[..,2i]*cos[i] - x[..,2i+1]*sin[i]
      out[..,2i+1] = x[..,2i]*sin[i] + x[..,2i+1]*cos[i]
    """
    cr = cos.reshape(1, L, 1, HALF)
    sr = sin.reshape(1, L, 1, HALF)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    o0 = x0 * cr - x1 * sr
    o1 = x0 * sr + x1 * cr
    out = torch.empty_like(x)
    out[..., 0::2] = o0
    out[..., 1::2] = o1
    return out


def repeat_kv(x, n_rep):
    """[1,L,KVHEADS,Dh] -> [1,L,HEADS,Dh] via repeat_interleave on the head axis
    (torch enable_gqa convention: dst head h reads kv head h // n_rep)."""
    b, l, kv, d = x.shape
    return x.repeat_interleave(n_rep, dim=2)


def sdpa(q, k, v):
    """q,k,v: [1,L,H,Dh] BSHD -> per-head softmax attention -> [1,L,H,Dh]."""
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)


def lora_delta(x, A, B):
    """LSCALE * ((x @ Aᵀ) @ Bᵀ); x [.., in], A [rank,in], B [out,rank]."""
    return LSCALE * ((x @ A.T) @ B.T)


def lin_lora(x, W, A, B):
    return x @ W.T + lora_delta(x, A, B)


def main():
    g = torch.Generator().manual_seed(7)

    # ── inputs ───────────────────────────────────────────────────────────────
    x = t2(L, FEATURES, 0.0007, 0.05, 0.5).reshape(1, L, FEATURES).requires_grad_(True)
    vec = t2(L if False else 1, 6 * FEATURES, 0.0003, 0.11, 0.2)  # [1, 6F]
    vec = vec.requires_grad_(True)

    # frozen base weights (torch Linear layout [out, in]); norms/mod raw params.
    W = {}
    W["wq"] = rnd(g, HEADS * HEADDIM, FEATURES, sc=0.03).requires_grad_(True)
    W["wk"] = rnd(g, KVHEADS * HEADDIM, FEATURES, sc=0.03).requires_grad_(True)
    W["wv"] = rnd(g, KVHEADS * HEADDIM, FEATURES, sc=0.03).requires_grad_(True)
    W["gate"] = rnd(g, FEATURES, FEATURES, sc=0.03).requires_grad_(True)
    W["wo"] = rnd(g, FEATURES, FEATURES, sc=0.03).requires_grad_(True)
    W["mlp_gate"] = rnd(g, MLPDIM, FEATURES, sc=0.03).requires_grad_(True)
    W["mlp_up"] = rnd(g, MLPDIM, FEATURES, sc=0.03).requires_grad_(True)
    W["mlp_down"] = rnd(g, FEATURES, MLPDIM, sc=0.03).requires_grad_(True)
    qnorm = (rnd(g, HEADDIM, sc=0.1)).requires_grad_(True)
    knorm = (rnd(g, HEADDIM, sc=0.1)).requires_grad_(True)
    prenorm = (rnd(g, FEATURES, sc=0.1)).requires_grad_(True)
    postnorm = (rnd(g, FEATURES, sc=0.1)).requires_grad_(True)
    mod_lin = (rnd(g, 6 * FEATURES, sc=0.05)).requires_grad_(True)

    # LoRA adapters (A kaiming-ish small, B NONZERO so dA is non-degenerate).
    LO = {}
    names_io = {
        "wq": (FEATURES, HEADS * HEADDIM),
        "wk": (FEATURES, KVHEADS * HEADDIM),
        "wv": (FEATURES, KVHEADS * HEADDIM),
        "gate": (FEATURES, FEATURES),
        "wo": (FEATURES, FEATURES),
        "mlp_gate": (FEATURES, MLPDIM),
        "mlp_up": (FEATURES, MLPDIM),
        "mlp_down": (MLPDIM, FEATURES),
    }
    for nm, (in_f, out_f) in names_io.items():
        LO[nm + "_A"] = rnd(g, RANK, in_f, sc=0.02).requires_grad_(True)
        LO[nm + "_B"] = rnd(g, out_f, RANK, sc=0.02).requires_grad_(True)

    # per-token RoPE table [L, HALF] (non-degenerate); shared by q & k.
    cos = t2(L, HALF, 0.03, 0.2, 1.0) * 0.6
    sin = t2(L, HALF, 0.04, 0.5, 1.0) * 0.6

    # ── mod(vec) → 6 raw chunks [FEATURES] ───────────────────────────────────
    out_mod = vec + mod_lin                       # [1, 6F]
    chunks = out_mod.reshape(6, FEATURES)
    prescale = chunks[0]
    preshift = chunks[1]
    pregate = chunks[2]
    postscale = chunks[3]
    postshift = chunks[4]
    postgate = chunks[5]

    # ── ATTENTION branch ─────────────────────────────────────────────────────
    xn = rms_norm_scale1(x, prenorm)
    xm = modulate(xn, prescale, preshift)                       # [1,L,F]

    q = lin_lora(xm, W["wq"], LO["wq_A"], LO["wq_B"]).reshape(1, L, HEADS, HEADDIM)
    k = lin_lora(xm, W["wk"], LO["wk_A"], LO["wk_B"]).reshape(1, L, KVHEADS, HEADDIM)
    v = lin_lora(xm, W["wv"], LO["wv_A"], LO["wv_B"]).reshape(1, L, KVHEADS, HEADDIM)
    gate_pre = lin_lora(xm, W["gate"], LO["gate_A"], LO["gate_B"])  # [1,L,F]

    q = rms_norm_scale1(q, qnorm)
    k = rms_norm_scale1(k, knorm)
    qr = rope_interleaved(q, cos, sin, HEADS)
    kr = rope_interleaved(k, cos, sin, KVHEADS)

    k_full = repeat_kv(kr, N_REP)
    v_full = repeat_kv(v, N_REP)
    att = sdpa(qr, k_full, v_full).reshape(1, L, FEATURES)

    sg = torch.sigmoid(gate_pre)
    gated = att * sg
    a = lin_lora(gated, W["wo"], LO["wo_A"], LO["wo_B"])         # [1,L,F]
    x1 = residual_gate(x, pregate, a)

    # ── MLP branch ───────────────────────────────────────────────────────────
    xn2 = rms_norm_scale1(x1, postnorm)
    xm2 = modulate(xn2, postscale, postshift)
    mg = lin_lora(xm2, W["mlp_gate"], LO["mlp_gate_A"], LO["mlp_gate_B"])
    mu = lin_lora(xm2, W["mlp_up"], LO["mlp_up_A"], LO["mlp_up_B"])
    sw = torch.nn.functional.silu(mg) * mu
    m = lin_lora(sw, W["mlp_down"], LO["mlp_down_A"], LO["mlp_down_B"])
    x2 = residual_gate(x1, postgate, m)

    # ── backward via a fixed upstream grad ───────────────────────────────────
    d_out = t2(L, FEATURES, 0.0011, 0.07, 0.05).reshape(1, L, FEATURES)
    loss = (x2 * d_out).sum()
    loss.backward()

    # ── dump ─────────────────────────────────────────────────────────────────
    def Wb(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
        with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))

    # forward output (sanity)
    Wb("kref_out", x2)
    # base input grad (no-regression)
    Wb("kref_d_x", x.grad)
    # LoRA grads (the deliverable)
    for nm in names_io:
        Wb("kref_%s_dA" % nm, LO[nm + "_A"].grad)
        Wb("kref_%s_dB" % nm, LO[nm + "_B"].grad)

    # exact INPUTS the Mojo gate reconstructs
    Wb("kin_x", x)
    Wb("kin_vec", vec)
    for nm in ["wq", "wk", "wv", "gate", "wo", "mlp_gate", "mlp_up", "mlp_down"]:
        Wb("kin_W_%s" % nm, W[nm])
    Wb("kin_qnorm", qnorm)
    Wb("kin_knorm", knorm)
    Wb("kin_prenorm", prenorm)
    Wb("kin_postnorm", postnorm)
    Wb("kin_mod_lin", mod_lin)
    for nm in names_io:
        Wb("kin_lo_%s_A" % nm, LO[nm + "_A"])
        Wb("kin_lo_%s_B" % nm, LO[nm + "_B"])
    Wb("kin_cos", cos)
    Wb("kin_sin", sin)
    Wb("kin_d_out", d_out)

    print("forward loss =", float(loss))
    print("HEADS=%d KVHEADS=%d HEADDIM=%d L=%d MLPDIM=%d RANK=%d LSCALE=%g" %
          (HEADS, KVHEADS, HEADDIM, L, MLPDIM, RANK, LSCALE))
    print("DONE — wrote refs to", REF_DIR)


if __name__ == "__main__":
    main()
