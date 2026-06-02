#!/usr/bin/env python3
# serenitymojo/models/zimage/parity/lora_stack_oracle.py
#
# Torch oracle for the Z-Image (NextDiT) FULL STACK *WITH LoRA* on all 7
# projections per block (attention.{to_q,to_k,to_v,to_out.0},
# feed_forward.{w1,w3,w2}) across the noise refiners, context refiners, and main
# layers. Replicates zimage_stack_lora.mojo's forward + autograd grads. Proves the
# LoRA COMPOSITION backward (A/B grads on every block, scattered to the flat
# carrier) = grad of the composed LoRA forward.
#
# LoRA math (matches zimage/lora_block.mojo + train_step._lora_fwd):
#   y' = x @ W.T + scale * ((x @ A.T) @ B.T)   A=[rank,in] B=[out,rank]
#   scale = alpha / rank
#
# This oracle is INDEPENDENT torch autograd derived from the zimage_nextdit.rs /
# transformer_z_image.py topology (the exact base math of stack_oracle.py). It
# REUSES the base stack's input .bin files (sin_*, written by stack_oracle.py — run
# that FIRST) for the shared base weights / tokens / rope / mod, and writes its OWN
# LoRA A/B inits (lin_*) and LoRA-aware reference grads (lref_*). B is perturbed
# NON-zero here so the adapters are LIVE and their grads non-degenerate.
#
# Run (AFTER stack_oracle.py, SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/stack_oracle.py
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/lora_stack_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── dims (MUST match stack_oracle.py) ──
H = 30
Dh = 128
D = H * Dh           # 3840
IMG_H = 2
IMG_W = 3
N_IMG = IMG_H * IMG_W    # 6
N_TXT = 4
S = N_IMG + N_TXT        # 10
F = 96
OUT_CH = 16
EPS = 1e-5
FINAL_EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
ROPE_AXES = (32, 48, 48)
ROPE_THETA = 256
NUM_NR = 1
NUM_CR = 1
NUM_MAIN = 2

# LoRA config (OneTrainer defaults: rank=16, alpha=1.0 — gate uses rank=8/alpha=16
# to keep the live perturbation well-scaled, mirroring the Ernie gate).
RANK = 8
ALPHA = 16.0
LSCALE = ALPHA / RANK

REF_DIR = os.path.dirname(os.path.abspath(__file__))

# slot order MUST match lora_block.mojo SLOT_*: Q,K,V,O,w1,w3,w2
SLOTS = ["to_q", "to_k", "to_v", "to_out", "w1", "w3", "w2"]
# (in, out) per slot
SLOT_SHAPE = {
    "to_q": (D, D), "to_k": (D, D), "to_v": (D, D), "to_out": (D, D),
    "w1": (D, F), "w3": (D, F), "w2": (F, D),
}


def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def fillc(n, a, b, c):
    return torch.tensor([math.cos(a * i + b) * c for i in range(n)], dtype=DT)


def rms_norm(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def silu(x):
    return x * torch.sigmoid(x)


def _axis_inv_freqs(axis_dim, theta):
    return [theta ** (-(2.0 * k) / axis_dim) for k in range(axis_dim // 2)]

F0 = _axis_inv_freqs(ROPE_AXES[0], ROPE_THETA)
F1 = _axis_inv_freqs(ROPE_AXES[1], ROPE_THETA)
F2 = _axis_inv_freqs(ROPE_AXES[2], ROPE_THETA)
HALF = Dh // 2


def rope_interleaved(x, cos, sin, n):
    cr = cos.reshape(n, H, HALF)
    sr = sin.reshape(n, H, HALF)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cr - x1 * sr
    out[..., 1::2] = x0 * sr + x1 * cr
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
    return LSCALE * ((x @ A.T) @ B.T)


def attention(xn, w, lo, cos, sin, n):
    q = (xn @ w["wq"].T + lora(xn, lo["to_q"][0], lo["to_q"][1])).reshape(1, n, H, Dh)
    k = (xn @ w["wk"].T + lora(xn, lo["to_k"][0], lo["to_k"][1])).reshape(1, n, H, Dh)
    v = (xn @ w["wv"].T + lora(xn, lo["to_v"][0], lo["to_v"][1])).reshape(1, n, H, Dh)
    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    qr = rope_interleaved(q, cos, sin, n)
    kr = rope_interleaved(k, cos, sin, n)
    att = sdpa(qr, kr, v).reshape(n, D)
    return att @ w["wo"].T + lora(att, lo["to_out"][0], lo["to_out"][1])


def ffn(xn, w, lo):
    g_pre = xn @ w["w1"].T + lora(xn, lo["w1"][0], lo["w1"][1])
    u = xn @ w["w3"].T + lora(xn, lo["w3"][0], lo["w3"][1])
    act = silu(g_pre) * u
    return act @ w["w2"].T + lora(act, lo["w2"][0], lo["w2"][1])


def mod_block(x, w, m, lo, cos, sin, n):
    xn1 = rms_norm(x, w["n1"])
    xn1s = (1.0 + m["scale_msa"]) * xn1
    att_o = attention(xn1s, w, lo, cos, sin, n)
    attn_n2 = rms_norm(att_o, w["n2"])
    gate_msa = torch.tanh(m["gate_msa"])
    h = x + gate_msa * attn_n2
    xfn1 = rms_norm(h, w["fn1"])
    xfn1s = (1.0 + m["scale_mlp"]) * xfn1
    ff = ffn(xfn1s, w, lo)
    ff_n2 = rms_norm(ff, w["fn2"])
    gate_mlp = torch.tanh(m["gate_mlp"])
    return h + gate_mlp * ff_n2


def unmod_block(x, w, lo, cos, sin, n):
    xn1 = rms_norm(x, w["n1"])
    att_o = attention(xn1, w, lo, cos, sin, n)
    attn_n2 = rms_norm(att_o, w["n2"])
    h = x + attn_n2
    xfn1 = rms_norm(h, w["fn1"])
    ff = ffn(xfn1, w, lo)
    ff_n2 = rms_norm(ff, w["fn2"])
    return h + ff_n2


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


def load_block(prefix):
    w = {}
    for kk in ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "n2", "fn1", "w1", "w3", "w2", "fn2"]:
        if kk in ("wq", "wk", "wv", "wo"):
            w[kk] = R("%s_%s" % (prefix, kk)).reshape(D, D)
        elif kk in ("w1", "w3"):
            w[kk] = R("%s_%s" % (prefix, kk)).reshape(F, D)
        elif kk == "w2":
            w[kk] = R("%s_%s" % (prefix, kk)).reshape(D, F)
        elif kk in ("q_norm", "k_norm"):
            w[kk] = R("%s_%s" % (prefix, kk)).reshape(Dh)
        else:
            w[kk] = R("%s_%s" % (prefix, kk)).reshape(D)
        w[kk] = w[kk].clone()  # base frozen (no requires_grad)
    return w


def load_mod(prefix):
    m = {}
    for kk in ["scale_msa", "gate_msa", "scale_mlp", "gate_mlp"]:
        m[kk] = R("%s_%s" % (prefix, kk)).reshape(D).clone().requires_grad_(True)
    return m


def rope_pair(prefix, n):
    cos = R("sin_%s_cos" % prefix).reshape(n * H, HALF)
    sin = R("sin_%s_sin" % prefix).reshape(n * H, HALF)
    return cos, sin


# ── LoRA A/B init: A small randn (LCG, scale 0.01) matching build_zimage_lora_set;
# B perturbed NON-zero here (LIVE adapter). The Mojo gate loads these SAME A/B from
# lin_*.bin, so inits are identical on both sides.
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
        B = lcg_randn(out_f * RANK, seed + 777, 0.05).reshape(out_f, RANK)
        A = A.clone().requires_grad_(True)
        B = B.clone().requires_grad_(True)
        lo[s] = (A, B)
        seed += 1
    return lo


def main():
    # reuse the base-stack inputs (run stack_oracle.py first)
    x_seq = R("sin_x_seq").reshape(N_IMG, D).requires_grad_(True)
    cap_seq = R("sin_cap_seq").reshape(N_TXT, D).requires_grad_(True)
    f_scale = R("sin_f_scale").reshape(D).requires_grad_(True)
    final_lin = R("sin_final_lin").reshape(OUT_CH, D)
    final_lin_b = R("sin_final_lin_b").reshape(OUT_CH)

    nr_w = [load_block("sin_nr%d" % i) for i in range(NUM_NR)]
    nr_m = [load_mod("sin_nr%d" % i) for i in range(NUM_NR)]
    cr_w = [load_block("sin_cr%d" % i) for i in range(NUM_CR)]
    main_w = [load_block("sin_main%d" % i) for i in range(NUM_MAIN)]
    main_m = [load_mod("sin_main%d" % i) for i in range(NUM_MAIN)]

    # LoRA adapters: 3 segments (nr | cr | main), block-flat seeds match
    # build_zimage_lora_set start (informational only — gate loads from lin_*.bin).
    nr_lo = [make_lora_for_block(6000 + 100 * i) for i in range(NUM_NR)]
    cr_lo = [make_lora_for_block(7000 + 100 * i) for i in range(NUM_CR)]
    main_lo = [make_lora_for_block(8000 + 100 * i) for i in range(NUM_MAIN)]

    x_cos, x_sin = rope_pair("x", N_IMG)
    cap_cos, cap_sin = rope_pair("cap", N_TXT)
    uni_cos, uni_sin = rope_pair("uni", S)

    # ── forward (LoRA on every projection) ──
    xs = x_seq
    for i in range(NUM_NR):
        xs = mod_block(xs, nr_w[i], nr_m[i], nr_lo[i], x_cos, x_sin, N_IMG)
    cs = cap_seq
    for i in range(NUM_CR):
        cs = unmod_block(cs, cr_w[i], cr_lo[i], cap_cos, cap_sin, N_TXT)
    unified = torch.cat([xs, cs], 0)
    for i in range(NUM_MAIN):
        unified = mod_block(unified, main_w[i], main_m[i], main_lo[i], uni_cos, uni_sin, S)
    x_final = unified
    ln_u = (x_final - x_final.mean(-1, keepdim=True)) / torch.sqrt(
        x_final.var(-1, unbiased=False, keepdim=True) + FINAL_EPS)
    x_out = (1.0 + f_scale) * ln_u
    patches = x_out @ final_lin.T + final_lin_b
    out = patches[:N_IMG]

    d_out = R("sin_d_out").reshape(N_IMG, OUT_CH)
    loss = (out * d_out).sum()
    loss.backward()

    # forward output (LoRA-modified)
    W("lref_out", out)
    # token grads (full-chain proof)
    W("lref_d_x_seq", x_seq.grad)
    W("lref_d_cap_seq", cap_seq.grad)
    # final-layer grad
    W("lref_d_f_scale", f_scale.grad)

    # per-block RAW mod-vec grads [4D]: scale_msa|gate_msa|scale_mlp|gate_mlp
    def pack4(m):
        return torch.cat([m["scale_msa"].grad, m["gate_msa"].grad,
                          m["scale_mlp"].grad, m["gate_mlp"].grad], 0)
    for i in range(NUM_NR):
        W("lref_nr%d_mod" % i, pack4(nr_m[i]))
    for i in range(NUM_MAIN):
        W("lref_main%d_mod" % i, pack4(main_m[i]))

    # LoRA A/B inits + their grads. Segment-tagged: nr%d / cr%d / main%d.
    def dump_lora(tag, lo):
        for s in SLOTS:
            A, B = lo[s]
            W("lin_%s_%s_A" % (tag, s), A)
            W("lin_%s_%s_B" % (tag, s), B)
            W("lref_%s_%s_dA" % (tag, s), A.grad)
            W("lref_%s_%s_dB" % (tag, s), B.grad)

    for i in range(NUM_NR):
        dump_lora("nr%d" % i, nr_lo[i])
    for i in range(NUM_CR):
        dump_lora("cr%d" % i, cr_lo[i])
    for i in range(NUM_MAIN):
        dump_lora("main%d" % i, main_lo[i])

    print("lora stack forward loss =", float(loss))
    print("RANK =", RANK, " ALPHA =", ALPHA, " LSCALE =", LSCALE)
    print("NR=%d CR=%d MAIN=%d  S=%d D=%d F=%d" % (NUM_NR, NUM_CR, NUM_MAIN, S, D, F))
    print("DONE")


if __name__ == "__main__":
    main()
