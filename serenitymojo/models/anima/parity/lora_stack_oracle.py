#!/usr/bin/env python3
# serenitymojo/models/anima/parity/lora_stack_oracle.py
#
# Torch oracle for the ANIMA MiniTrainDIT FULL STACK *WITH LoRA* on all 10 target
# projections (self_attn.{q,k,v,output}, cross_attn.{q,k,v,output}, mlp.{layer1,
# layer2}). Replicates anima_stack_lora.mojo's forward + autograd grads. Proves the
# LoRA COMPOSITION backward (A/B grads on every block, scattered to the flat carrier)
# = grad of the composed LoRA forward, AND that the base path is unchanged when the
# adapters are present-but-additive (B perturbed NON-zero so the adapters are LIVE
# and their grads are non-degenerate — B=0 at real init makes d_A vanish at step 0).
#
# LoRA math (matches anima/lora_block.mojo + train_step._lora_fwd):
#   y' = x @ W.T + scale * ((x @ A.T) @ B.T)   A=[rank,in] B=[out,rank]
#   scale = alpha / rank
# Cross-attn k/v LoRA input is the FROZEN context (NOT x_mod) — matching the
# inference chokepoint linear_no_bias(context, "cross_attn.k_proj") in anima.rs:413.
#
# This oracle REUSES the base stack's input .bin files (in_*, written by
# stack_oracle.py — run that FIRST) for the shared base weights / patches / t_cond /
# context / rope / final-layer, and writes its OWN LoRA A/B inits (lin_*) and
# LoRA-aware reference grads (lref_*).
#
# Run (AFTER stack_oracle.py, SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/stack_oracle.py
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/lora_stack_oracle.py

import math
import struct
import os
import torch

DT = torch.float64

# ── dims (MUST match stack_oracle.py) ──
B = 1
H = 16
Dh = 128
D = H * Dh          # 2048
S_IMG = 6
S_TXT = 8
JOINT = 1024
F = 32
ADALN = 256
IN_PATCH = 68
OUT_PATCH = 64
L = 3
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
THETA = 10000.0

# LoRA config
RANK = 8
ALPHA = 16.0
LSCALE = ALPHA / RANK

REF_DIR = os.path.dirname(os.path.abspath(__file__))

# slot order (must match lora_block.mojo SLOT_*): sa q,k,v,out / ca q,k,v,out / mlp1,mlp2
SLOTS = ["sa_q", "sa_k", "sa_v", "sa_out",
         "ca_q", "ca_k", "ca_v", "ca_out", "mlp1", "mlp2"]
# (in, out) per slot
SLOT_SHAPE = {
    "sa_q": (D, D), "sa_k": (D, D), "sa_v": (D, D), "sa_out": (D, D),
    "ca_q": (D, D), "ca_k": (JOINT, D), "ca_v": (JOINT, D), "ca_out": (D, D),
    "mlp1": (D, F), "mlp2": (F, D),
}


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


def R(name):
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return torch.tensor(struct.unpack("<%df" % n, data), dtype=DT)


# ── ops (match anima.rs / stack_oracle.py math) ──
def layer_norm_noaffine(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS)


def rms_norm_lastdim(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def rope_halfsplit(x, cos, sin):
    half = Dh // 2
    c = cos.reshape(1, x.shape[1], 1, half)
    s = sin.reshape(1, x.shape[1], 1, half)
    x1 = x[..., :half]
    x2 = x[..., half:]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2], dim=-1)


def sdpa(q, k, v):
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)


def gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def lora(x, A, B_):
    # y_lora = scale * (x @ A.T) @ B.T
    return LSCALE * ((x @ A.T) @ B_.T)


def adaln_mod(w, sub, t_silu, base_adaln):
    h = t_silu @ w[f"{sub}_mod1"].T
    mod_out = h @ w[f"{sub}_mod2"].T + base_adaln
    shift = mod_out[:, 0:D]
    scale = mod_out[:, D:2 * D]
    gate = mod_out[:, 2 * D:3 * D]
    return shift, scale, gate


def adaln_pre(xx, shift, scale):
    ln = layer_norm_noaffine(xx)
    return (1.0 + scale).unsqueeze(1) * ln + shift.unsqueeze(1)


def block_forward_lora(x, w, t_silu, base_adaln, context, cos, sin, lo):
    # self-attn (LoRA on q/k/v/out, input = sa xmod / attn_flat)
    sh, sc, ga = adaln_mod(w, "sa", t_silu, base_adaln)
    xmod = adaln_pre(x, sh, sc)
    xmod2d = xmod.reshape(B * S_IMG, D)
    q = (xmod2d @ w["sa_q"].T + lora(xmod2d, lo["sa_q"][0], lo["sa_q"][1])).reshape(B, S_IMG, H, Dh)
    k = (xmod2d @ w["sa_k"].T + lora(xmod2d, lo["sa_k"][0], lo["sa_k"][1])).reshape(B, S_IMG, H, Dh)
    v = (xmod2d @ w["sa_v"].T + lora(xmod2d, lo["sa_v"][0], lo["sa_v"][1])).reshape(B, S_IMG, H, Dh)
    q = rms_norm_lastdim(q, w["sa_qn"])
    k = rms_norm_lastdim(k, w["sa_kn"])
    q = rope_halfsplit(q, cos, sin)
    k = rope_halfsplit(k, cos, sin)
    att = sdpa(q, k, v).reshape(B * S_IMG, D)
    sa_out = (att @ w["sa_out"].T + lora(att, lo["sa_out"][0], lo["sa_out"][1])).reshape(B, S_IMG, D)
    x = x + ga.unsqueeze(1) * sa_out
    # cross-attn (LoRA on q [input ca xmod], k/v [input frozen context], out)
    sh, sc, ga = adaln_mod(w, "ca", t_silu, base_adaln)
    xmod = adaln_pre(x, sh, sc)
    xmod2d = xmod.reshape(B * S_IMG, D)
    ctx2d = context.reshape(B * S_TXT, JOINT)
    q = (xmod2d @ w["ca_q"].T + lora(xmod2d, lo["ca_q"][0], lo["ca_q"][1])).reshape(B, S_IMG, H, Dh)
    k = (ctx2d @ w["ca_k"].T + lora(ctx2d, lo["ca_k"][0], lo["ca_k"][1])).reshape(B, S_TXT, H, Dh)
    v = (ctx2d @ w["ca_v"].T + lora(ctx2d, lo["ca_v"][0], lo["ca_v"][1])).reshape(B, S_TXT, H, Dh)
    q = rms_norm_lastdim(q, w["ca_qn"])
    k = rms_norm_lastdim(k, w["ca_kn"])
    att = sdpa(q, k, v).reshape(B * S_IMG, D)
    ca_out = (att @ w["ca_out"].T + lora(att, lo["ca_out"][0], lo["ca_out"][1])).reshape(B, S_IMG, D)
    x = x + ga.unsqueeze(1) * ca_out
    # mlp (LoRA on layer1 [input mlp xmod] + layer2 [input gelu(h)])
    sh, sc, ga = adaln_mod(w, "mlp", t_silu, base_adaln)
    xmod = adaln_pre(x, sh, sc)
    xmod2d = xmod.reshape(B * S_IMG, D)
    h_pre = xmod2d @ w["mlp1"].T + lora(xmod2d, lo["mlp1"][0], lo["mlp1"][1])
    h = gelu_tanh(h_pre)
    mlp_out = (h @ w["mlp2"].T + lora(h, lo["mlp2"][0], lo["mlp2"][1])).reshape(B, S_IMG, D)
    x = x + ga.unsqueeze(1) * mlp_out
    return x


# ── LoRA A/B init MUST match the Mojo loader in lora_stack_parity.mojo ─────────
# A ~ small randn (LCG, scale 0.01); B perturbed NON-zero (LIVE adapter so grads
# are non-degenerate). The Mojo gate loads these SAME A/B from lin_*.bin.
def lcg_randn(n, seed, scale):
    out = []
    state = seed & ((1 << 64) - 1)
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = float(state >> 40) * (1.0 / 16777216.0)
        out.append((u - 0.5) * scale)
    return torch.tensor(out, dtype=DT)


def make_lora_for_block(block_seed_base):
    lo = {}
    seed = block_seed_base
    for s in SLOTS:
        in_f, out_f = SLOT_SHAPE[s]
        A = lcg_randn(RANK * in_f, seed, 0.01).reshape(RANK, in_f)
        B_ = lcg_randn(out_f * RANK, seed + 777, 0.05).reshape(out_f, RANK)
        A = A.clone().requires_grad_(True)
        B_ = B_.clone().requires_grad_(True)
        lo[s] = (A, B_)
        seed += 1
    return lo


def make_block_weights(g):
    # match stack_oracle.make_block_weights (0.02-scaled projections so the residual
    # stream stays bounded across L blocks — see that file's NOTE).
    w = {}
    for sub in ("sa", "ca", "mlp"):
        w[f"{sub}_mod1"] = (rnd(g, ADALN, D) * 0.02).requires_grad_(True)
        w[f"{sub}_mod2"] = (rnd(g, 3 * D, ADALN) * 0.02).requires_grad_(True)
    w["sa_q"] = (rnd(g, D, D) * 0.02)
    w["sa_k"] = (rnd(g, D, D) * 0.02)
    w["sa_v"] = (rnd(g, D, D) * 0.02)
    w["sa_out"] = (rnd(g, D, D) * 0.02)
    w["sa_qn"] = (rnd(g, Dh) * 0.1 + 1.0)
    w["sa_kn"] = (rnd(g, Dh) * 0.1 + 1.0)
    w["ca_q"] = (rnd(g, D, D) * 0.02)
    w["ca_k"] = (rnd(g, D, JOINT) * 0.02)
    w["ca_v"] = (rnd(g, D, JOINT) * 0.02)
    w["ca_out"] = (rnd(g, D, D) * 0.02)
    w["ca_qn"] = (rnd(g, Dh) * 0.1 + 1.0)
    w["ca_kn"] = (rnd(g, Dh) * 0.1 + 1.0)
    w["mlp1"] = (rnd(g, F, D) * 0.02)
    w["mlp2"] = (rnd(g, D, F) * 0.02)
    return w  # base is FROZEN for LoRA (no requires_grad on projections)


def rnd(g, *shape):
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)


def main():
    # reuse the base-stack inputs (run stack_oracle.py first)
    patches = R("in_patches").reshape(B * S_IMG, IN_PATCH).requires_grad_(True)
    # RAW t_cond is the leaf; the block + final layer silu it internally. We retain
    # grad on t_silu so lref_d_t_silu == the grad-w.r.t.-silu-output the Mojo
    # anima_block_lora_backward returns (block returns d_t_silu, not d_t_cond).
    t_cond_leaf = R("in_t_cond").reshape(B, D).clone().requires_grad_(True)
    t_silu = torch.nn.functional.silu(t_cond_leaf)
    t_silu.retain_grad()
    base_adaln = R("in_base_adaln").reshape(B, 3 * D).requires_grad_(True)
    context = R("in_context").reshape(B, S_TXT, JOINT).requires_grad_(True)

    x_embed = R("in_x_embed").reshape(D, IN_PATCH)              # frozen base
    fl_mod1 = R("in_fl_mod1").reshape(ADALN, D)                 # frozen base
    fl_mod2 = R("in_fl_mod2").reshape(2 * D, ADALN)             # frozen base
    fl_lin = R("in_fl_lin").reshape(OUT_PATCH, D)              # frozen base

    blk = [make_block_weights(torch.Generator().manual_seed(100 + l)) for l in range(L)]
    lo = [make_lora_for_block(7000 + 100 * l) for l in range(L)]

    # rope tables (per position [S_IMG, Dh/2]) — rebuilt to match stack_oracle.py
    half = Dh // 2
    pos = torch.arange(S_IMG, dtype=DT).reshape(S_IMG, 1)
    freq = torch.tensor(
        [1.0 / (THETA ** (2.0 * i / Dh)) for i in range(half)], dtype=DT
    ).reshape(1, half)
    ang = pos * freq
    cos = torch.cos(ang)
    sin = torch.sin(ang)

    # ── forward (LoRA on every target projection) ──
    x = (patches @ x_embed.T).reshape(B, S_IMG, D)
    for l in range(L):
        x = block_forward_lora(x, blk[l], t_silu, base_adaln, context, cos, sin, lo[l])
    x_final = x
    fl_h = t_silu @ fl_mod1.T
    fl_modout = fl_h @ fl_mod2.T + base_adaln[:, :2 * D]
    fl_shift = fl_modout[:, 0:D]
    fl_scale = fl_modout[:, D:2 * D]
    x_mod = adaln_pre(x_final, fl_shift, fl_scale)
    out = (x_mod.reshape(B * S_IMG, D) @ fl_lin.T)

    d_out = R("in_d_out").reshape(B * S_IMG, OUT_PATCH)
    (out * d_out).sum().backward()

    # forward output (LoRA-modified)
    W("lref_out", out)
    # input grads (full-chain proof)
    W("lref_d_patches", patches.grad)
    W("lref_d_t_silu", t_silu.grad)
    W("lref_d_base_adaln", base_adaln.grad)

    # LoRA A/B inits + their grads, all L blocks × 10 slots.
    for l in range(L):
        for s in SLOTS:
            A, B_ = lo[l][s]
            W("lin_l%d_%s_A" % (l, s), A)
            W("lin_l%d_%s_B" % (l, s), B_)
            W("lref_l%d_%s_dA" % (l, s), A.grad)
            W("lref_l%d_%s_dB" % (l, s), B_.grad)

    print("forward loss =", float((out * d_out).sum()))
    print("RANK =", RANK, " ALPHA =", ALPHA, " LSCALE =", LSCALE)
    print("L =", L, " S_IMG =", S_IMG, " S_TXT =", S_TXT, " D =", D, " F =", F)
    print("DONE")


if __name__ == "__main__":
    main()
