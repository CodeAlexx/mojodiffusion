#!/usr/bin/env python3
# serenitymojo/models/ernie/parity/lora_stack_oracle.py
#
# Torch oracle for the ERNIE-Image FULL STACK *WITH LoRA* on all 7 projections
# (self_attention.{to_q,to_k,to_v,to_out.0}, mlp.{gate_proj,up_proj,linear_fc2}).
# Replicates ernie_stack_lora.mojo's forward + autograd grads. Proves the LoRA
# COMPOSITION backward (A/B grads on every block, scattered to the flat carrier)
# = grad of the composed LoRA forward, AND that the base path is unchanged when
# the adapters are present-but-additive (B=0 at init would be identity; here we
# perturb B so the adapters are LIVE and their grads are non-degenerate).
#
# LoRA math (matches ernie/lora_block.mojo + train_step._lora_fwd):
#   y' = x @ W.T + scale * ((x @ A.T) @ B.T)   A=[rank,in] B=[out,rank]
#   scale = alpha / rank
#
# This oracle REUSES the base stack's input .bin files (in_*, written by
# stack_oracle.py — run that FIRST) for the shared base weights / tokens / rope /
# mod, and writes its OWN LoRA A/B inits (lin_*) and LoRA-aware reference grads
# (lref_*). The base-weight grads change vs the no-LoRA oracle (the LoRA path adds
# to each projection output), so we re-dump out + token grads + shared mod under
# the lref_ prefix.
#
# Run (AFTER stack_oracle.py, SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/ernie/parity/stack_oracle.py
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/ernie/parity/lora_stack_oracle.py

import math
import struct
import os
import torch

DT = torch.float64

# ── dims (MUST match stack_oracle.py) ──
H = 32
Dh = 128
D = H * Dh           # 4096
IMG_H = 2
IMG_W = 3
N_IMG = IMG_H * IMG_W    # 6
N_TXT = 2
S = N_IMG + N_TXT        # 8
TEXT_LEN_REAL = N_TXT
F = 96
IN_CH = 16
TEXT_IN = 24
OUT_CH = 16
L = 3
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
ROPE_AXES = (32, 48, 48)
ROPE_THETA = 256

# LoRA config
RANK = 8
ALPHA = 16.0
LSCALE = ALPHA / RANK

REF_DIR = os.path.dirname(os.path.abspath(__file__))

# slot order (must match lora_block.mojo SLOT_*): Q,K,V,O,gate,up,down
SLOTS = ["to_q", "to_k", "to_v", "to_out", "gate_proj", "up_proj", "linear_fc2"]
# (in, out) per slot
SLOT_SHAPE = {
    "to_q": (D, D), "to_k": (D, D), "to_v": (D, D), "to_out": (D, D),
    "gate_proj": (D, F), "up_proj": (D, F), "linear_fc2": (F, D),
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


def modulate(x, scale, shift):
    return (1.0 + scale) * x + shift


def residual_gate(x, gate, y):
    return x + gate * y


def gelu_tanh(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))


def rope_halfsplit_full(x, cos, sin):
    Sx = x.shape[1]
    half = Dh // 2
    cr = cos.reshape(Sx, H, Dh)
    sr = sin.reshape(Sx, H, Dh)
    x0 = x[..., 0:half]
    x1 = x[..., half:Dh]
    c0 = cr[..., 0:half]
    s0 = sr[..., 0:half]
    c1 = cr[..., half:Dh]
    s1 = sr[..., half:Dh]
    out = torch.empty_like(x)
    out[..., 0:half] = x0 * c0 - x1 * s0
    out[..., half:Dh] = x1 * c1 + x0 * s1
    return out


def sdpa(q, k, v):
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)


def lora(x, A, B):
    # y_lora = scale * (x @ A.T) @ B.T
    return LSCALE * ((x @ A.T) @ B.T)


def block_forward_lora(x, w, m, cos, sin, lo):
    sa_norm = rms_norm(x, w["sa_norm"])
    sa_in = modulate(sa_norm, m["scale_msa"], m["shift_msa"])
    q = (sa_in @ w["wq"].T + lora(sa_in, lo["to_q"][0], lo["to_q"][1])).reshape(1, S, H, Dh)
    k = (sa_in @ w["wk"].T + lora(sa_in, lo["to_k"][0], lo["to_k"][1])).reshape(1, S, H, Dh)
    v = (sa_in @ w["wv"].T + lora(sa_in, lo["to_v"][0], lo["to_v"][1])).reshape(1, S, H, Dh)
    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    qr = rope_halfsplit_full(q, cos, sin)
    kr = rope_halfsplit_full(k, cos, sin)
    att = sdpa(qr, kr, v).reshape(S, D)
    att_out = att @ w["wo"].T + lora(att, lo["to_out"][0], lo["to_out"][1])
    h = residual_gate(x, m["gate_msa"], att_out)
    mlp_norm = rms_norm(h, w["mlp_norm"])
    mlp_in = modulate(mlp_norm, m["scale_mlp"], m["shift_mlp"])
    gate_pre = mlp_in @ w["wgate"].T + lora(mlp_in, lo["gate_proj"][0], lo["gate_proj"][1])
    up = mlp_in @ w["wup"].T + lora(mlp_in, lo["up_proj"][0], lo["up_proj"][1])
    activated = gelu_tanh(gate_pre) * up
    mlp_out = activated @ w["wdown"].T + lora(activated, lo["linear_fc2"][0], lo["linear_fc2"][1])
    return residual_gate(h, m["gate_mlp"], mlp_out)


def _axis_inv_freqs(axis_dim, theta):
    return [theta ** (-(2.0 * k) / axis_dim) for k in range(axis_dim // 2)]


def build_real_rope_tables():
    f0 = _axis_inv_freqs(ROPE_AXES[0], ROPE_THETA)
    f1 = _axis_inv_freqs(ROPE_AXES[1], ROPE_THETA)
    f2 = _axis_inv_freqs(ROPE_AXES[2], ROPE_THETA)
    cos_rows, sin_rows = [], []
    for tok in range(S):
        if tok < N_IMG:
            r, c = tok // IMG_W, tok % IMG_W
            p0, p1, p2 = float(TEXT_LEN_REAL), float(r), float(c)
        else:
            p0, p1, p2 = float(tok - N_IMG), 0.0, 0.0
        cos_tok, sin_tok = [], []
        for (p, freqs) in ((p0, f0), (p1, f1), (p2, f2)):
            for fi in freqs:
                ang = p * fi
                cos_tok += [math.cos(ang), math.cos(ang)]
                sin_tok += [math.sin(ang), math.sin(ang)]
        for _h in range(H):
            cos_rows.append(cos_tok)
            sin_rows.append(sin_tok)
    return torch.tensor(cos_rows, dtype=DT), torch.tensor(sin_rows, dtype=DT)


def make_block_weights(seed):
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
    w["sa_norm"] = (rnd(D) * 0.1 + 1.0)
    w["mlp_norm"] = (rnd(D) * 0.1 + 1.0)
    w["wgate"] = rnd(F, D) * 0.02
    w["wup"] = rnd(F, D) * 0.02
    w["wdown"] = rnd(D, F) * 0.02
    return w  # NOT requires_grad (base is frozen for LoRA)


def make_mod():
    m = {}
    m["shift_msa"] = (fill(D, 0.013, 0.10, 0.30)).requires_grad_(True)
    m["scale_msa"] = (fillc(D, 0.017, 0.20, 0.20)).requires_grad_(True)
    m["gate_msa"] = (fill(D, 0.011, 0.30, 0.40)).requires_grad_(True)
    m["shift_mlp"] = (fill(D, 0.015, 0.40, 0.25)).requires_grad_(True)
    m["scale_mlp"] = (fillc(D, 0.019, 0.50, 0.15)).requires_grad_(True)
    m["gate_mlp"] = (fill(D, 0.012, 0.60, 0.35)).requires_grad_(True)
    return m


# ── LoRA A/B init MUST match the Mojo build_ernie_lora_set / make_lora_adapter ─
# A ~ small randn (LCG, scale 0.01), B perturbed NON-zero here so the adapters are
# LIVE and their grads are non-degenerate (B=0 at real init makes the d_A path
# vanish at step 0; for a parity gate of the math we need a live B). The Mojo gate
# loads these SAME A/B from lin_*.bin, so the inits are identical on both sides.
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
        # B: distinct small perturbation per slot/block (LIVE adapter)
        B = lcg_randn(out_f * RANK, seed + 777, 0.05).reshape(out_f, RANK)
        A = A.clone().requires_grad_(True)
        B = B.clone().requires_grad_(True)
        lo[s] = (A, B)
        seed += 1
    return lo


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, flat.shape)


def R(name):
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return torch.tensor(struct.unpack("<%df" % n, data), dtype=DT)


def main():
    # reuse the base-stack inputs (run stack_oracle.py first)
    img_tokens = R("in_img_tokens").reshape(N_IMG, IN_CH).requires_grad_(True)
    txt_tokens = R("in_txt_tokens").reshape(N_TXT, TEXT_IN).requires_grad_(True)
    patch_w = R("in_patch_w").reshape(D, IN_CH)
    patch_b = R("in_patch_b").reshape(D)
    text_proj = R("in_text_proj").reshape(D, TEXT_IN)
    final_lin = R("in_final_lin").reshape(OUT_CH, D)
    final_lin_b = R("in_final_lin_b").reshape(OUT_CH)
    f_scale = R("in_f_scale").reshape(D).requires_grad_(True)
    f_shift = R("in_f_shift").reshape(D).requires_grad_(True)

    m = make_mod()
    blk = [make_block_weights(100 + l) for l in range(L)]
    lo = [make_lora_for_block(5000 + 100 * l) for l in range(L)]

    cos, sin = build_real_rope_tables()

    # ── forward (LoRA on every projection) ──
    img = img_tokens @ patch_w.T + patch_b
    txt = txt_tokens @ text_proj.T
    x = torch.cat([img, txt], 0)
    for l in range(L):
        x = block_forward_lora(x, blk[l], m, cos, sin, lo[l])
    x_final = x
    ln_x = (x_final - x_final.mean(-1, keepdim=True)) / torch.sqrt(
        x_final.var(-1, unbiased=False, keepdim=True) + EPS)
    x_out = (1.0 + f_scale) * ln_x + f_shift
    patches = x_out @ final_lin.T + final_lin_b
    out = patches[:N_IMG]

    d_out = R("in_d_out").reshape(N_IMG, OUT_CH)
    loss = (out * d_out).sum()
    loss.backward()

    # forward output (LoRA-modified)
    W("lref_out", out)
    # token grads (full-chain proof)
    W("lref_d_img_tokens", img_tokens.grad)
    W("lref_d_txt_tokens", txt_tokens.grad)
    # final-layer grads
    W("lref_d_f_scale", f_scale.grad)
    W("lref_d_f_shift", f_shift.grad)
    # summed shared mod grads [6D]
    shared = torch.cat([m["shift_msa"].grad, m["scale_msa"].grad, m["gate_msa"].grad,
                        m["shift_mlp"].grad, m["scale_mlp"].grad, m["gate_mlp"].grad], 0)
    W("lref_d_shared_mod", shared)

    # LoRA A/B inits + their grads, all L blocks × 7 slots.
    for l in range(L):
        for s in SLOTS:
            A, B = lo[l][s]
            W("lin_l%d_%s_A" % (l, s), A)
            W("lin_l%d_%s_B" % (l, s), B)
            W("lref_l%d_%s_dA" % (l, s), A.grad)
            W("lref_l%d_%s_dB" % (l, s), B.grad)

    print("forward loss =", float(loss))
    print("RANK =", RANK, " ALPHA =", ALPHA, " LSCALE =", LSCALE)
    print("L =", L, " S =", S, " D =", D, " F =", F)
    print("DONE")


if __name__ == "__main__":
    main()
