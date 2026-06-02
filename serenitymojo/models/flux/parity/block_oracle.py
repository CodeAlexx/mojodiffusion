#!/usr/bin/env python3
# serenitymojo/models/flux/parity/block_oracle.py
#
# Torch-autograd oracle for the Flux (flux1-dev) DOUBLE + SINGLE DiT blocks
# (forward + grads). Built INDEPENDENTLY from the Flux reference
# inference-flame/src/models/flux1_dit.rs (NOT transcribed from the Mojo block):
#   double_block_forward (lines 729-888), single_block_forward (890-1008),
#   modulate_pre (307-312), rms_norm (281-288), split_qkv, apply_rope_complex.
# Produces .bin references the Mojo gates (double_block_parity.mojo /
# single_block_parity.mojo) read and compare at cos >= 0.999.
#
# FLUX-vs-Klein block differences encoded here (flux1_dit.rs:5-12, 729-1008):
#   - BIASES on every linear (qkv, proj, mlp.0, mlp.2 / linear1, linear2).
#   - GELU MLP (tanh approximation — matches flame-core gelu_tanh_derivative.cu
#     and the Mojo ops/activations.mojo `gelu`), NOT SwiGLU.
#   - modulate_pre = (1+scale)*LayerNorm(x,eps=1e-6) + shift   (no affine LN).
#   - q/k rms_norm over Dh (eps 1e-6); v un-normed.
#   - joint concat txt FIRST then img (double); rope INTERLEAVED-pair (2i,2i+1).
#   - sdpa non-causal, scale = 1/sqrt(Dh).
#
# REAL Flux dims: hidden D = 3072, H = 24, Dh = 128. To keep the oracle fast we
# use SMALL sequence lengths (N_IMG/N_TXT/S) and a SMALL mlp factor — the dims
# that matter for catching bugs (H, Dh, D) are kept REAL. NON-DEGENERATE
# sinusoidal fills (NEVER modular (i*k)%9, which aliases at real dims and zeros
# grads — the Ernie degenerate-table lesson). The RoPE table is built with the
# real 3-axis Flux omega·pos construction and we ASSERT cos[i] != cos[i+half].
#
# Run (SEPARATE command, never chained with && after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/flux/parity/block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64  # F64 reference interior (gates compare cos in F64)

# ── REAL Flux block dims ──
H = 24
Dh = 128
D = H * Dh                # 3072 (REAL Flux inner_dim)
# small sequence + mlp to keep the oracle fast; H/Dh/D are REAL.
N_IMG = 4
N_TXT = 3
S = N_TXT + N_IMG         # double-block joint length
S_SINGLE = 6              # single-block sequence length
FMLP = 32                # GELU MLP hidden (real = 4*D; small here for speed)
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
# Flux 3-axis RoPE: axes_dims_rope = [16, 56, 56], halves = 8 + 28 + 28 = 64 = Dh/2.
AXES = [16, 56, 56]
ROPE_THETA = 10000.0

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def fillc(n, a, b, c):
    return torch.tensor([math.cos(a * i + b) * c for i in range(n)], dtype=DT)


def t2(n, m, a, b, c):
    return fill(n * m, a, b, c).reshape(n, m)


# ── ops (match flux1_dit.rs + Mojo block exactly) ──
def layer_norm(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS)


def modulate(x, scale, shift):
    # modulate_pre: (1+scale)*LayerNorm(x) + shift
    return (1.0 + scale) * layer_norm(x) + shift


def residual_gate(x, gate, y):
    return x + gate * y


def rms_norm_lastdim(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def gelu_tanh(x):
    # tanh approximation — matches flame-core gelu_backward.cu + Mojo gelu.
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))


def rope_interleaved(x, cos, sin, n):
    # x [1, n, H, Dh]; cos/sin [n*H, Dh/2] flat per (s,h) row.
    cr = cos.reshape(n, H, Dh // 2)
    sr = sin.reshape(n, H, Dh // 2)
    x0 = x[..., 0::2]   # even
    x1 = x[..., 1::2]   # odd
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


def build_rope(n, ids):
    # Flux 3-axis RoPE (flux1_dit.rs::build_rope_2d). ids: [n, 3] position ids.
    # For each axis a: omega = 1/theta**(2i/axis_dim) for i in 0..axis_dim/2;
    # angles = pos[:,a] outer omega -> [n, half]; cat halves over axes -> [n, Dh/2].
    cos_parts, sin_parts = [], []
    for axis, axis_dim in enumerate(AXES):
        half = axis_dim // 2
        omega = torch.tensor(
            [1.0 / (ROPE_THETA ** ((2 * i) / axis_dim)) for i in range(half)],
            dtype=DT,
        ).reshape(1, half)
        pos = ids[:, axis].reshape(n, 1)
        angles = pos @ omega   # [n, half]
        cos_parts.append(torch.cos(angles))
        sin_parts.append(torch.sin(angles))
    cos = torch.cat(cos_parts, dim=1)   # [n, Dh/2]
    sin = torch.cat(sin_parts, dim=1)
    # tile per-head: [n, Dh/2] -> [n*H, Dh/2]
    cos = cos.unsqueeze(1).expand(n, H, Dh // 2).reshape(n * H, Dh // 2).contiguous()
    sin = sin.unsqueeze(1).expand(n, H, Dh // 2).reshape(n * H, Dh // 2).contiguous()
    return cos, sin


# ── weights ──
def make_stream(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.05).to(DT)
    w = {}
    w["wqkv"] = rnd(3 * D, D)
    w["bqkv"] = rnd(3 * D)
    w["wproj"] = rnd(D, D)
    w["bproj"] = rnd(D)
    w["wmlp0"] = rnd(FMLP, D)
    w["bmlp0"] = rnd(FMLP)
    w["wmlp2"] = rnd(D, FMLP)
    w["bmlp2"] = rnd(D)
    w["q_norm"] = (rnd(Dh) * 2.0 + 1.0)
    w["k_norm"] = (rnd(Dh) * 2.0 + 1.0)
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_mod(off):
    m = {}
    m["shift1"] = fill(D, 0.0013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale1"] = fillc(D, 0.0017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate1"] = fill(D, 0.0011, 0.3 + off, 0.4).requires_grad_(True)
    m["shift2"] = fillc(D, 0.0019, 0.4 + off, 0.3).requires_grad_(True)
    m["scale2"] = fill(D, 0.0015, 0.5 + off, 0.2).requires_grad_(True)
    m["gate2"] = fillc(D, 0.0012, 0.6 + off, 0.4).requires_grad_(True)
    return m


def make_single_weights(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.05).to(DT)
    w = {}
    w["w1"] = rnd(3 * D + FMLP, D)
    w["b1"] = rnd(3 * D + FMLP)
    w["w2"] = rnd(D, D + FMLP)
    w["b2"] = rnd(D)
    w["q_norm"] = (rnd(Dh) * 2.0 + 1.0)
    w["k_norm"] = (rnd(Dh) * 2.0 + 1.0)
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_single_mod(off):
    m = {}
    m["shift"] = fill(D, 0.0013, 0.1 + off, 0.3).requires_grad_(True)
    m["scale"] = fillc(D, 0.0017, 0.2 + off, 0.2).requires_grad_(True)
    m["gate"] = fill(D, 0.0011, 0.3 + off, 0.4).requires_grad_(True)
    return m


def stream_pre(x, w, m, cos, sin, n):
    norm = modulate(x, m["scale1"], m["shift1"])
    qkv = norm @ w["wqkv"].T + w["bqkv"]
    q = qkv[:, 0:D].reshape(1, n, H, Dh)
    k = qkv[:, D:2 * D].reshape(1, n, H, Dh)
    v = qkv[:, 2 * D:3 * D].reshape(1, n, H, Dh)
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])
    return q, k, v


def stream_post(x, att, w, m):
    out = att @ w["wproj"].T + w["bproj"]
    attn_res = residual_gate(x, m["gate1"], out)
    mlp_in = modulate(attn_res, m["scale2"], m["shift2"])
    mlp_pre = mlp_in @ w["wmlp0"].T + w["bmlp0"]
    mlp_h = gelu_tanh(mlp_pre)
    mlp = mlp_h @ w["wmlp2"].T + w["bmlp2"]
    return residual_gate(attn_res, m["gate2"], mlp)


def Wbin(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))


def gen_double():
    img = t2(N_IMG, D, 0.0021, 0.05, 0.5).requires_grad_(True)
    txt = t2(N_TXT, D, 0.0023, 0.07, 0.5).requires_grad_(True)
    iw = make_stream(1)
    tw = make_stream(2)
    im = make_mod(0.0)
    tm = make_mod(1.0)

    # JOINT-sequence position ids [S, 3] (txt FIRST then img); non-degenerate.
    ids = torch.tensor(
        [[float(s), float((s * 3) % 5), float((s * 2) % 7)] for s in range(S)],
        dtype=DT,
    )
    cos, sin = build_rope(S, ids)
    # NON-DEGENERATE table assertion (Ernie lesson): cos[i] != cos[i+half].
    half = Dh // 2
    cflat = cos.reshape(-1)
    assert abs(float(cflat[0]) - float(cflat[0])) < 1e-12  # sanity
    diff = (cos[:, : half // 2] - cos[:, half // 2 :]).abs().max() if half >= 2 else torch.tensor(1.0)
    assert float(diff) > 1e-6, "DEGENERATE rope table (cos halves equal)"

    iq, ik, iv = stream_pre(img, iw, im, cos, sin, N_IMG)
    tq, tk, tv = stream_pre(txt, tw, tm, cos, sin, N_TXT)

    q = torch.cat([tq, iq], dim=1)
    k = torch.cat([tk, ik], dim=1)
    v = torch.cat([tv, iv], dim=1)
    qr = rope_interleaved(q, cos, sin, S)
    kr = rope_interleaved(k, cos, sin, S)
    att = sdpa(qr, kr, v)

    txt_att = att[:, 0:N_TXT].reshape(N_TXT, D)
    img_att = att[:, N_TXT:N_TXT + N_IMG].reshape(N_IMG, D)

    img_out = stream_post(img, img_att, iw, im)
    txt_out = stream_post(txt, txt_att, tw, tm)

    d_img = t2(N_IMG, D, 0.0027, 0.11, 0.05)
    d_txt = t2(N_TXT, D, 0.0029, 0.13, 0.05)
    loss = (img_out * d_img).sum() + (txt_out * d_txt).sum()
    loss.backward()

    Wbin("d_ref_img_out", img_out)
    Wbin("d_ref_txt_out", txt_out)
    Wbin("d_ref_d_img", img.grad)
    Wbin("d_ref_d_txt", txt.grad)
    for nm, w in [("im", iw), ("tm", tw)]:
        for kk in ["wqkv", "bqkv", "wproj", "bproj", "wmlp0", "bmlp0", "wmlp2", "bmlp2", "q_norm", "k_norm"]:
            Wbin("d_ref_%s_d_%s" % (nm, kk), w[kk].grad)
    for nm, m in [("im", im), ("tm", tm)]:
        for kk in ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]:
            Wbin("d_ref_%s_d_%s" % (nm, kk), m[kk].grad)

    # inputs the Mojo gate must reconstruct identically
    Wbin("d_in_img", img)
    Wbin("d_in_txt", txt)
    for nm, w in [("iw", iw), ("tw", tw)]:
        for kk in ["wqkv", "bqkv", "wproj", "bproj", "wmlp0", "bmlp0", "wmlp2", "bmlp2", "q_norm", "k_norm"]:
            Wbin("d_in_%s_%s" % (nm, kk), w[kk])
    for nm, m in [("im", im), ("tm", tm)]:
        for kk in ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]:
            Wbin("d_in_%s_%s" % (nm, kk), m[kk])
    Wbin("d_in_cos", cos)
    Wbin("d_in_sin", sin)
    Wbin("d_in_d_img", d_img)
    Wbin("d_in_d_txt", d_txt)
    print("DOUBLE: forward loss =", float(loss))


def gen_single():
    n = S_SINGLE
    x = t2(n, D, 0.0021, 0.05, 0.5).requires_grad_(True)
    w = make_single_weights(3)
    m = make_single_mod(0.0)

    ids = torch.tensor(
        [[float(s), float((s * 3) % 5), float((s * 2) % 7)] for s in range(n)],
        dtype=DT,
    )
    cos, sin = build_rope(n, ids)
    half = Dh // 2
    diff = (cos[:, : half // 2] - cos[:, half // 2 :]).abs().max()
    assert float(diff) > 1e-6, "DEGENERATE rope table (single)"

    norm = modulate(x, m["scale"], m["shift"])
    fused = norm @ w["w1"].T + w["b1"]          # [n, 3D+FMLP]
    qkv = fused[:, 0:3 * D]
    mlp_in = fused[:, 3 * D:3 * D + FMLP]

    q = qkv[:, 0:D].reshape(1, n, H, Dh)
    k = qkv[:, D:2 * D].reshape(1, n, H, Dh)
    v = qkv[:, 2 * D:3 * D].reshape(1, n, H, Dh)
    q = rms_norm_lastdim(q, w["q_norm"])
    k = rms_norm_lastdim(k, w["k_norm"])
    qr = rope_interleaved(q, cos, sin, n)
    kr = rope_interleaved(k, cos, sin, n)
    att = sdpa(qr, kr, v).reshape(n, D)

    mlp_h = gelu_tanh(mlp_in)
    out_in = torch.cat([att, mlp_h], dim=1)     # [n, D+FMLP]
    out_proj = out_in @ w["w2"].T + w["b2"]
    out = residual_gate(x, m["gate"], out_proj)

    d_out = t2(n, D, 0.0027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    Wbin("s_ref_out", out)
    Wbin("s_ref_d_x", x.grad)
    for kk in ["w1", "b1", "w2", "b2", "q_norm", "k_norm"]:
        Wbin("s_ref_d_%s" % kk, w[kk].grad)
    for kk in ["shift", "scale", "gate"]:
        Wbin("s_ref_d_%s" % kk, m[kk].grad)

    Wbin("s_in_x", x)
    for kk in ["w1", "b1", "w2", "b2", "q_norm", "k_norm"]:
        Wbin("s_in_w_%s" % kk, w[kk])
    for kk in ["shift", "scale", "gate"]:
        Wbin("s_in_m_%s" % kk, m[kk])
    Wbin("s_in_cos", cos)
    Wbin("s_in_sin", sin)
    Wbin("s_in_d_out", d_out)
    print("SINGLE: forward loss =", float(loss))


if __name__ == "__main__":
    gen_double()
    gen_single()
    print("DONE")
