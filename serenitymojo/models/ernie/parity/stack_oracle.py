#!/usr/bin/env python3
# serenitymojo/models/ernie/parity/stack_oracle.py
#
# Torch oracle for the ERNIE-Image FULL STACK composition (input projections +
# L× single-stream block + final layer), forward + autograd grads. Replicates
# the EXACT math of serenitymojo/models/ernie/ernie_stack.mojo, which composes
# the parity-verified ernie_block (block_oracle.py). The shared-AdaLN modulation
# (6 chunks [D]) is broadcast to EVERY block, so its grad SUMS across all L
# blocks — this oracle dumps that summed grad so the Mojo gate can prove the
# composed backward = grad of the composed forward (the Klein composition lesson:
# project_klein_runaway_composition_backward — per-block-correct does NOT imply
# composition-correct; this gate is the proof).
#
# Small depth + small S + reduced F to keep the torch graph + GPU bounded. REAL
# H=32, Dh=128 (D=4096) so the head structure + RoPE table are the real thing.
#
# Graph (mirrors ernie_stack.mojo forward):
#   img = img_tokens @ patch_w.T + patch_b           # [N_IMG, D]
#   txt = txt_tokens @ text_proj.T                   # [N_TXT, D] (no bias)
#   x   = cat([img, txt], 0)                          # IMAGE FIRST -> [S, D]
#   for l in range(L): x = ernie_block(x, blk[l], shared_mod, cos, sin)
#   ln_x  = layer_norm(x, eps) (non-learnable, weight 1 bias 0)
#   x_out = (1 + f_scale) * ln_x + f_shift
#   patches = x_out @ final_lin.T + final_lin_b       # [S, out_ch]
#   out   = patches[:N_IMG]                            # narrow img rows
#
# Run (SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/ernie/parity/stack_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── dims ──
H = 32
Dh = 128
D = H * Dh           # 4096
IMG_H = 2
IMG_W = 3
N_IMG = IMG_H * IMG_W    # 6
N_TXT = 2
S = N_IMG + N_TXT        # 8
TEXT_LEN_REAL = N_TXT
F = 96                   # reduced FFN hidden (real 12288)
IN_CH = 16               # reduced latent channels (real 128)
TEXT_IN = 24             # reduced text_in_dim (real 3072)
OUT_CH = 16              # reduced out_channels (real 128)
L = 3                    # number of layers (deepest + shallowest probed)
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
ROPE_AXES = (32, 48, 48)
ROPE_THETA = 256

REF_DIR = os.path.dirname(os.path.abspath(__file__))


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


def _axis_inv_freqs(axis_dim, theta):
    return [theta ** (-(2.0 * k) / axis_dim) for k in range(axis_dim // 2)]


def build_real_rope_tables():
    f0 = _axis_inv_freqs(ROPE_AXES[0], ROPE_THETA)
    f1 = _axis_inv_freqs(ROPE_AXES[1], ROPE_THETA)
    f2 = _axis_inv_freqs(ROPE_AXES[2], ROPE_THETA)
    assert len(f0) + len(f1) + len(f2) == Dh // 2
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


def sdpa(q, k, v):
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)


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
    for v in w.values():
        v.requires_grad_(True)
    return w


def block_forward(x, w, m, cos, sin):
    sa_norm = rms_norm(x, w["sa_norm"])
    sa_in = modulate(sa_norm, m["scale_msa"], m["shift_msa"])
    q = (sa_in @ w["wq"].T).reshape(1, S, H, Dh)
    k = (sa_in @ w["wk"].T).reshape(1, S, H, Dh)
    v = (sa_in @ w["wv"].T).reshape(1, S, H, Dh)
    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    qr = rope_halfsplit_full(q, cos, sin)
    kr = rope_halfsplit_full(k, cos, sin)
    att = sdpa(qr, kr, v).reshape(S, D)
    att_out = att @ w["wo"].T
    h = residual_gate(x, m["gate_msa"], att_out)
    mlp_norm = rms_norm(h, w["mlp_norm"])
    mlp_in = modulate(mlp_norm, m["scale_mlp"], m["shift_mlp"])
    gate_pre = mlp_in @ w["wgate"].T
    up = mlp_in @ w["wup"].T
    activated = gelu_tanh(gate_pre) * up
    mlp_out = activated @ w["wdown"].T
    return residual_gate(h, m["gate_mlp"], mlp_out)


def make_mod():
    m = {}
    m["shift_msa"] = (fill(D, 0.013, 0.10, 0.30)).requires_grad_(True)
    m["scale_msa"] = (fillc(D, 0.017, 0.20, 0.20)).requires_grad_(True)
    m["gate_msa"] = (fill(D, 0.011, 0.30, 0.40)).requires_grad_(True)
    m["shift_mlp"] = (fill(D, 0.015, 0.40, 0.25)).requires_grad_(True)
    m["scale_mlp"] = (fillc(D, 0.019, 0.50, 0.15)).requires_grad_(True)
    m["gate_mlp"] = (fill(D, 0.012, 0.60, 0.35)).requires_grad_(True)
    return m


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, flat.shape)


def main():
    # input tokens + base weights
    img_tokens = t2(N_IMG, IN_CH, 0.021, 0.05, 0.5).requires_grad_(True)
    txt_tokens = t2(N_TXT, TEXT_IN, 0.018, 0.07, 0.4).requires_grad_(True)
    g = torch.Generator().manual_seed(7)
    patch_w = (torch.randn(D, IN_CH, generator=g, dtype=torch.float32).to(DT) * 0.02).requires_grad_(True)
    patch_b = (fill(D, 0.009, 0.02, 0.05)).requires_grad_(True)
    text_proj = (torch.randn(D, TEXT_IN, generator=g, dtype=torch.float32).to(DT) * 0.02).requires_grad_(True)
    final_lin = (torch.randn(OUT_CH, D, generator=g, dtype=torch.float32).to(DT) * 0.02).requires_grad_(True)
    final_lin_b = (fill(OUT_CH, 0.03, 0.01, 0.05)).requires_grad_(True)
    f_scale = (fillc(D, 0.015, 0.30, 0.10)).requires_grad_(True)
    f_shift = (fill(D, 0.014, 0.40, 0.10)).requires_grad_(True)

    # ONE shared modulation, broadcast to ALL blocks (grad sums across blocks)
    m = make_mod()
    # per-block weights (distinct per layer so the deepest/shallowest grads differ)
    blk = [make_block_weights(100 + l) for l in range(L)]

    cos, sin = build_real_rope_tables()
    half = Dh // 2
    assert float((cos[:, :half] - cos[:, half:]).abs().max()) > 1e-3, "degenerate table"

    # ── forward ──
    img = img_tokens @ patch_w.T + patch_b               # [N_IMG, D]
    txt = txt_tokens @ text_proj.T                       # [N_TXT, D]
    x = torch.cat([img, txt], 0)                         # IMAGE FIRST [S, D]
    for l in range(L):
        x = block_forward(x, blk[l], m, cos, sin)
    x_final = x
    ln_x = (x_final - x_final.mean(-1, keepdim=True)) / torch.sqrt(
        x_final.var(-1, unbiased=False, keepdim=True) + EPS)   # layer_norm (w=1,b=0)
    x_out = (1.0 + f_scale) * ln_x + f_shift
    patches = x_out @ final_lin.T + final_lin_b          # [S, out_ch]
    out = patches[:N_IMG]                                # [N_IMG, out_ch]

    # ── upstream grad ──
    d_out = t2(N_IMG, OUT_CH, 0.027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    # forward output
    W("ref_out", out)
    # token grads (load-bearing: prove the full chain back to the inputs)
    W("ref_d_img_tokens", img_tokens.grad)
    W("ref_d_txt_tokens", txt_tokens.grad)
    # final-layer modulation grads
    W("ref_d_f_scale", f_scale.grad)
    W("ref_d_f_shift", f_shift.grad)
    # base-weight grads
    W("ref_d_final_lin", final_lin.grad)
    # SUMMED shared-AdaLN mod grads packed [6D] = shift_msa|scale_msa|gate_msa|
    #   shift_mlp|scale_mlp|gate_mlp (the composition detail)
    shared = torch.cat([m["shift_msa"].grad, m["scale_msa"].grad, m["gate_msa"].grad,
                        m["shift_mlp"].grad, m["scale_mlp"].grad, m["gate_mlp"].grad], 0)
    W("ref_d_shared_mod", shared)
    # per-block weight grads: deepest (L-1) and shallowest (0). Probe wq + wdown.
    W("ref_d_wq_deep", blk[L - 1]["wq"].grad)
    W("ref_d_wdown_deep", blk[L - 1]["wdown"].grad)
    W("ref_d_wq_shallow", blk[0]["wq"].grad)
    W("ref_d_wdown_shallow", blk[0]["wdown"].grad)

    # ── inputs the Mojo gate reconstructs ──
    W("in_img_tokens", img_tokens)
    W("in_txt_tokens", txt_tokens)
    W("in_patch_w", patch_w)
    W("in_patch_b", patch_b)
    W("in_text_proj", text_proj)
    W("in_final_lin", final_lin)
    W("in_final_lin_b", final_lin_b)
    W("in_f_scale", f_scale)
    W("in_f_shift", f_shift)
    for kk in ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]:
        W("in_m_%s" % kk, m[kk])
    for l in range(L):
        for kk in ["wq", "wk", "wv", "wo", "q_norm", "k_norm",
                   "sa_norm", "mlp_norm", "wgate", "wup", "wdown"]:
            W("in_blk%d_%s" % (l, kk), blk[l][kk])
    W("in_cos", cos)
    W("in_sin", sin)
    W("in_d_out", d_out)

    print("forward loss =", float(loss))
    print("L =", L, " S =", S, " D =", D, " F =", F)
    print("DONE")


if __name__ == "__main__":
    main()
