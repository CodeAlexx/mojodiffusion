#!/usr/bin/env python3
# serenitymojo/models/zimage/parity/refiner_oracle.py
#
# Torch oracle for the Z-Image (NextDiT) UNMODULATED context-refiner block
# (forward + autograd grads). Replicates the EXACT math of
# serenitymojo/models/zimage/block.mojo `zimage_refiner_forward`, which mirrors
# the `has_adaln == false` branch of zimage_nextdit.rs `transformer_block`
# (lines 349-355, 365-370, 377-383, 393-398) and zimage_dit.mojo `_block`
# else-branch (lines 459-468):
#   xn1   = rms_norm(x, n1)                  # NO scale modulation
#   attn  = attention(xn1)
#   h     = x + rms_norm(attn, n2)           # PLAIN residual (no gate/tanh)
#   xfn1  = rms_norm(h, fn1)                 # NO scale modulation
#   ff    = swiglu_ffn(xfn1)
#   out   = h + rms_norm(ff, fn2)            # PLAIN residual (no gate/tanh)
#
# Same attention + SwiGLU + INTERLEAVED-RoPE math as the (verified) main block,
# with modulation removed. cos >= 0.999 gate proves the unmodulated path before
# it is composed into the stack.
#
# Run (SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/refiner_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

H = 30
Dh = 128
D = H * Dh          # 3840
N_TXT = 5           # context refiner runs on caption/text tokens only
S = N_TXT
F = 96
EPS = 1e-5
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


def silu(x):
    return x * torch.sigmoid(x)


def rope_interleaved(x, cos, sin):
    half = Dh // 2
    cr = cos.reshape(S, H, half)
    sr = sin.reshape(S, H, half)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cr - x1 * sr
    out[..., 1::2] = x0 * sr + x1 * cr
    return out


def _axis_inv_freqs(axis_dim, theta):
    return [theta ** (-(2.0 * k) / axis_dim) for k in range(axis_dim // 2)]


def build_real_rope_tables():
    f0 = _axis_inv_freqs(ROPE_AXES[0], ROPE_THETA)
    f1 = _axis_inv_freqs(ROPE_AXES[1], ROPE_THETA)
    f2 = _axis_inv_freqs(ROPE_AXES[2], ROPE_THETA)
    half = Dh // 2
    assert len(f0) + len(f1) + len(f2) == half
    cos_rows, sin_rows = [], []
    for tok in range(S):
        # caption positions: (1,0,0),(2,0,0),... (axis0 only, like context_refiner)
        p0, p1, p2 = float(tok + 1), 0.0, 0.0
        cos_tok, sin_tok = [], []
        for (p, freqs) in ((p0, f0), (p1, f1), (p2, f2)):
            for fi in freqs:
                ang = p * fi
                cos_tok.append(math.cos(ang))
                sin_tok.append(math.sin(ang))
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


def make_weights(seed):
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
    w["n1"] = (rnd(D) * 0.1 + 1.0)
    w["n2"] = (rnd(D) * 0.1 + 1.0)
    w["fn1"] = (rnd(D) * 0.1 + 1.0)
    w["fn2"] = (rnd(D) * 0.1 + 1.0)
    w["w1"] = rnd(F, D) * 0.02
    w["w3"] = rnd(F, D) * 0.02
    w["w2"] = rnd(D, F) * 0.02
    for v in w.values():
        v.requires_grad_(True)
    return w


def main():
    x = t2(S, D, 0.021, 0.05, 0.5).requires_grad_(True)
    w = make_weights(3)

    cos, sin = build_real_rope_tables()
    assert float(cos.std()) > 1e-3, "degenerate rope table"

    # ── forward (UNMODULATED) ──
    xn1 = rms_norm(x, w["n1"])                       # NO scale
    q = (xn1 @ w["wq"].T).reshape(1, S, H, Dh)
    k = (xn1 @ w["wk"].T).reshape(1, S, H, Dh)
    v = (xn1 @ w["wv"].T).reshape(1, S, H, Dh)
    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v).reshape(S, D)
    att_o = att @ w["wo"].T
    attn_n2 = rms_norm(att_o, w["n2"])
    h = x + attn_n2                                  # PLAIN residual

    xfn1 = rms_norm(h, w["fn1"])                     # NO scale
    g_pre = xfn1 @ w["w1"].T
    u = xfn1 @ w["w3"].T
    act = silu(g_pre) * u
    ff = act @ w["w2"].T
    ff_n2 = rms_norm(ff, w["fn2"])
    out = h + ff_n2                                  # PLAIN residual

    d_out = t2(S, D, 0.027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    def W(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
        path = os.path.join(REF_DIR, name + ".bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, flat.shape)

    W("rref_out", out)
    W("rref_d_x", x.grad)
    for kk in ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "n2", "fn1", "w1", "w3", "w2", "fn2"]:
        W("rref_d_%s" % kk, w[kk].grad)

    W("rin_x", x)
    for kk in ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
               "n2", "fn1", "w1", "w3", "w2", "fn2"]:
        W("rin_w_%s" % kk, w[kk])
    W("rin_cos", cos)
    W("rin_sin", sin)
    W("rin_d_out", d_out)

    print("refiner forward loss =", float(loss))
    print("rope cos std =", float(cos.std()))
    print("DONE")


if __name__ == "__main__":
    main()
