#!/usr/bin/env python3
# serenitymojo/models/flux/parity/stack_oracle.py
#
# Torch-autograd oracle for the Flux (flux1-dev) FULL DiT STACK at REDUCED-but-
# structurally-complete depth (NUM_DOUBLE double + NUM_SINGLE single, REAL
# H/Dh/D). Built INDEPENDENTLY from inference-flame/src/models/flux1_dit.rs (the
# forward) + models/dit/flux1_dit.mojo (the composition oracle):
#   vec = time_in(t*1000) + guidance_in(g*1000) + vector_in(clip_pooled)
#   per double block: silu(vec)->img_mod.lin/txt_mod.lin->chunk(6) ; double_block
#   x = concat(txt,img) ; per single block: silu(vec)->modulation.lin->chunk(3)
#   img_out = x[N_TXT:] ; final: silu(vec)->adaLN.1->(shift,scale)->modulate_pre
#             -> final_layer.linear -> out
#
# This proves the COMPOSITION SURFACE the Phase-1 blocks do NOT: input proj +
# the embed/vec MLP chain + per-block modulation projections + the double->single
# seam + the final layer + the d_img/d_txt -> d_y inter-block handoff across
# DEPTH + the d_vec accumulation across EVERY block and the final layer.
#
# CONVENTIONS (byte-match the Mojo forward):
#   modulate_pre(x, shift, scale) = (1+scale)*LayerNorm(x,1e-6) + shift
#   residual_gate(x, gate, y) = x + gate*y ; rms_norm over Dh (eps 1e-6)
#   rope INTERLEAVED (2i,2i+1) ; sdpa non-causal scale 1/sqrt(Dh)
#   joint concat txt FIRST then img ; GELU tanh-approx MLP ; biases everywhere
#   timestep_embedding: COS first then SIN, t pre-scaled by 1000 by the CALLER
#     (the Mojo gate passes t*1000 as the timestep input).
#
# NON-DEGENERATE inputs: sinusoidal/random fills, never modular aliasing; 3-axis
# Flux RoPE with the real omega*pos construction (cos halves asserted unequal).
#
# Run (SEPARATE command, never chained after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/flux/parity/stack_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── REAL Flux block dims (reduced depth + small seq/mlp for a fast oracle) ──
H = 24
Dh = 128
D = H * Dh                # 3072 (REAL Flux inner_dim)
N_IMG = 4
N_TXT = 3
S = N_TXT + N_IMG
FMLP = 32                 # GELU MLP hidden (real = 4*D; small for speed)
IN_CH = 64                # REAL flux in_channels
TXT_CH = 40               # T5 joint dim (real 4096; small for speed)
OUT_CH = 64               # REAL flux out channels
T_DIM = 16                # timestep sinusoid dim (real 256; small for speed)
VEC_DIM = 20              # CLIP pooled dim (real 768; small for speed)
NUM_DOUBLE = 3
NUM_SINGLE = 3
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
AXES = [16, 56, 56]
ROPE_THETA = 10000.0
MAX_PERIOD = 10000.0
GUIDANCE = 3.5            # arbitrary non-zero guidance scalar (Dev)

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


def modulate_pre(x, shift, scale):
    return (1.0 + scale) * layer_norm(x) + shift


def residual_gate(x, gate, y):
    return x + gate * y


def rms_norm_lastdim(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def gelu_tanh(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))


def rope_interleaved(x, cos, sin, n):
    cr = cos.reshape(n, H, Dh // 2)
    sr = sin.reshape(n, H, Dh // 2)
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


def build_rope(n, ids):
    cos_parts, sin_parts = [], []
    for axis, axis_dim in enumerate(AXES):
        half = axis_dim // 2
        omega = torch.tensor(
            [1.0 / (ROPE_THETA ** ((2 * i) / axis_dim)) for i in range(half)],
            dtype=DT,
        ).reshape(1, half)
        pos = ids[:, axis].reshape(n, 1)
        angles = pos @ omega
        cos_parts.append(torch.cos(angles))
        sin_parts.append(torch.sin(angles))
    cos = torch.cat(cos_parts, dim=1)
    sin = torch.cat(sin_parts, dim=1)
    cos = cos.unsqueeze(1).expand(n, H, Dh // 2).reshape(n * H, Dh // 2).contiguous()
    sin = sin.unsqueeze(1).expand(n, H, Dh // 2).reshape(n * H, Dh // 2).contiguous()
    return cos, sin


def timestep_embedding(t_scalar):
    # COS first then SIN (matches ops/embeddings.timestep_embedding). t_scalar is
    # ALREADY scaled (caller passes t*1000). Returns [T_DIM].
    half = T_DIM // 2
    freqs = torch.tensor(
        [math.exp(-math.log(MAX_PERIOD) * i / half) for i in range(half)],
        dtype=DT,
    )
    angle = t_scalar * freqs
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=0)  # [T_DIM]


# ── weights ──
def make_stream(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.05).to(DT)
    w = {}
    w["wqkv"] = rnd(3 * D, D); w["bqkv"] = rnd(3 * D)
    w["wproj"] = rnd(D, D); w["bproj"] = rnd(D)
    w["wmlp0"] = rnd(FMLP, D); w["bmlp0"] = rnd(FMLP)
    w["wmlp2"] = rnd(D, FMLP); w["bmlp2"] = rnd(D)
    w["q_norm"] = (rnd(Dh) * 2.0 + 1.0)
    w["k_norm"] = (rnd(Dh) * 2.0 + 1.0)
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_single_weights(seed):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.05).to(DT)
    w = {}
    w["w1"] = rnd(3 * D + FMLP, D); w["b1"] = rnd(3 * D + FMLP)
    w["w2"] = rnd(D, D + FMLP); w["b2"] = rnd(D)
    w["q_norm"] = (rnd(Dh) * 2.0 + 1.0)
    w["k_norm"] = (rnd(Dh) * 2.0 + 1.0)
    for v in w.values():
        v.requires_grad_(True)
    return w


def make_modlin(seed, chunk):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(chunk, D, generator=g, dtype=torch.float32) * 0.02).to(DT)
    b = (torch.randn(chunk, generator=g, dtype=torch.float32) * 0.02).to(DT)
    w.requires_grad_(True); b.requires_grad_(True)
    return {"w": w, "b": b}


def make_embed(seed, in_dim):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.05).to(DT)
    e = {}
    e["in_w"] = rnd(D, in_dim); e["in_b"] = rnd(D)
    e["out_w"] = rnd(D, D); e["out_b"] = rnd(D)
    for v in e.values():
        v.requires_grad_(True)
    return e


def mlp_embed(emb_in, e):
    hid = emb_in @ e["in_w"].T + e["in_b"]
    act = torch.nn.functional.silu(hid)
    return act @ e["out_w"].T + e["out_b"]


def double_block(img, txt, iw, tw, im, tm, cos, sin):
    def stream_pre(x, w, m, n):
        norm = modulate_pre(x, m["shift1"], m["scale1"])
        qkv = norm @ w["wqkv"].T + w["bqkv"]
        q = qkv[:, 0:D].reshape(1, n, H, Dh)
        k = qkv[:, D:2 * D].reshape(1, n, H, Dh)
        v = qkv[:, 2 * D:3 * D].reshape(1, n, H, Dh)
        q = rms_norm_lastdim(q, w["q_norm"])
        k = rms_norm_lastdim(k, w["k_norm"])
        return q, k, v

    iq, ik, iv = stream_pre(img, iw, im, N_IMG)
    tq, tk, tv = stream_pre(txt, tw, tm, N_TXT)
    q = torch.cat([tq, iq], dim=1)
    k = torch.cat([tk, ik], dim=1)
    v = torch.cat([tv, iv], dim=1)
    qr = rope_interleaved(q, cos, sin, S)
    kr = rope_interleaved(k, cos, sin, S)
    att = sdpa(qr, kr, v)
    txt_att = att[:, 0:N_TXT].reshape(N_TXT, D)
    img_att = att[:, N_TXT:N_TXT + N_IMG].reshape(N_IMG, D)

    def stream_post(x, att, w, m):
        out = att @ w["wproj"].T + w["bproj"]
        attn_res = residual_gate(x, m["gate1"], out)
        mlp_in = modulate_pre(attn_res, m["shift2"], m["scale2"])
        mlp_pre = mlp_in @ w["wmlp0"].T + w["bmlp0"]
        mlp_h = gelu_tanh(mlp_pre)
        mlp = mlp_h @ w["wmlp2"].T + w["bmlp2"]
        return residual_gate(attn_res, m["gate2"], mlp)

    img_out = stream_post(img, img_att, iw, im)
    txt_out = stream_post(txt, txt_att, tw, tm)
    return img_out, txt_out


def single_block(x, w, m, cos, sin):
    n = x.shape[0]
    norm = modulate_pre(x, m["shift"], m["scale"])
    fused = norm @ w["w1"].T + w["b1"]           # [n, 3D+FMLP]
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
    out_in = torch.cat([att, mlp_h], dim=1)      # [n, D+FMLP]
    out = out_in @ w["w2"].T + w["b2"]
    return residual_gate(x, m["gate"], out)


def chunk6(mods):
    d = D
    return {
        "shift1": mods[:, 0 * d:1 * d].reshape(d), "scale1": mods[:, 1 * d:2 * d].reshape(d),
        "gate1": mods[:, 2 * d:3 * d].reshape(d), "shift2": mods[:, 3 * d:4 * d].reshape(d),
        "scale2": mods[:, 4 * d:5 * d].reshape(d), "gate2": mods[:, 5 * d:6 * d].reshape(d),
    }


def chunk3(mods):
    d = D
    return {
        "shift": mods[:, 0 * d:1 * d].reshape(d), "scale": mods[:, 1 * d:2 * d].reshape(d),
        "gate": mods[:, 2 * d:3 * d].reshape(d),
    }


def Wbin(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))


def main():
    # ── base: input proj + embeds + per-block mod.lin + final layer ──
    img_in = t2(D, IN_CH, 0.0021, 0.05, 0.3).requires_grad_(True)
    img_in_b = fill(D, 0.0011, 0.02, 0.1).requires_grad_(True)
    txt_in = t2(D, TXT_CH, 0.0019, 0.07, 0.3).requires_grad_(True)
    txt_in_b = fill(D, 0.0013, 0.03, 0.1).requires_grad_(True)
    time_in = make_embed(50, T_DIM)
    guidance_in = make_embed(51, T_DIM)
    vector_in = make_embed(52, VEC_DIM)
    dbl_imod = [make_modlin(300 + bi * 2, 6 * D) for bi in range(NUM_DOUBLE)]
    dbl_tmod = [make_modlin(301 + bi * 2, 6 * D) for bi in range(NUM_DOUBLE)]
    sgl_mod = [make_modlin(400 + bi, 3 * D) for bi in range(NUM_SINGLE)]
    final_adaln_w = (t2(2 * D, D, 0.0023, 0.09, 0.02)).requires_grad_(True)
    final_adaln_b = fill(2 * D, 0.0012, 0.04, 0.05).requires_grad_(True)
    final_lin = t2(OUT_CH, D, 0.0025, 0.11, 0.02).requires_grad_(True)
    final_lin_b = fill(OUT_CH, 0.0014, 0.05, 0.05).requires_grad_(True)

    # ── inputs ──
    img_tokens = t2(N_IMG, IN_CH, 0.0031, 0.05, 0.5).requires_grad_(True)
    txt_tokens = t2(N_TXT, TXT_CH, 0.0033, 0.07, 0.5).requires_grad_(True)
    # timestep / guidance are SCALED scalars (t*1000) — the Mojo gate feeds the
    # same scaled values. Keep them moderate so the sinusoid is non-saturating.
    t_scaled = torch.tensor([0.37 * 1000.0], dtype=DT).requires_grad_(True)
    g_scaled = torch.tensor([GUIDANCE * 1000.0], dtype=DT).requires_grad_(True)
    vector = t2(1, VEC_DIM, 0.041, 0.09, 0.4).requires_grad_(True)

    # ── per-block weights ──
    dbl = [(make_stream(100 + bi * 2), make_stream(101 + bi * 2)) for bi in range(NUM_DOUBLE)]
    sgl = [make_single_weights(200 + bi) for bi in range(NUM_SINGLE)]

    # ── rope tables (joint sequence, txt FIRST then img) ──
    ids = torch.tensor(
        [[float(s), float((s * 3) % 5), float((s * 2) % 7)] for s in range(S)],
        dtype=DT,
    )
    cos, sin = build_rope(S, ids)
    half = Dh // 2
    diff = (cos[:, : half // 2] - cos[:, half // 2 :]).abs().max()
    assert float(diff) > 1e-6, "DEGENERATE rope table"

    # ── embeds -> vec ──
    t_emb = timestep_embedding(t_scaled).reshape(1, T_DIM)
    g_emb = timestep_embedding(g_scaled).reshape(1, T_DIM)
    vec = mlp_embed(t_emb, time_in) + mlp_embed(g_emb, guidance_in) + mlp_embed(vector, vector_in)  # [1,D]
    vec.retain_grad()
    vec_silu = torch.nn.functional.silu(vec)

    # ── forward ──
    img = img_tokens @ img_in.T + img_in_b      # [N_IMG, D]
    txt = txt_tokens @ txt_in.T + txt_in_b      # [N_TXT, D]
    for bi in range(NUM_DOUBLE):
        iw, tw = dbl[bi]
        imods = chunk6(vec_silu @ dbl_imod[bi]["w"].T + dbl_imod[bi]["b"])
        tmods = chunk6(vec_silu @ dbl_tmod[bi]["w"].T + dbl_tmod[bi]["b"])
        img, txt = double_block(img, txt, iw, tw, imods, tmods, cos, sin)
    x = torch.cat([txt, img], dim=0)            # [S, D] (txt FIRST)
    for bi in range(NUM_SINGLE):
        smods = chunk3(vec_silu @ sgl_mod[bi]["w"].T + sgl_mod[bi]["b"])
        x = single_block(x, sgl[bi], smods, cos, sin)
    img_out = x[N_TXT:N_TXT + N_IMG]            # [N_IMG, D]
    fmods = vec_silu @ final_adaln_w.T + final_adaln_b  # [1, 2D]
    f_shift = fmods[:, 0:D].reshape(D)
    f_scale = fmods[:, D:2 * D].reshape(D)
    normed = modulate(layer_norm(img_out), f_scale, f_shift)
    out = normed @ final_lin.T + final_lin_b    # [N_IMG, OUT_CH]

    # ── upstream grad ──
    d_out = t2(N_IMG, OUT_CH, 0.0027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    # ── outputs the gate checks ──
    Wbin("ref_out", out)
    Wbin("ref_d_img_tokens", img_tokens.grad)
    Wbin("ref_d_txt_tokens", txt_tokens.grad)
    Wbin("ref_d_vec", vec.grad)
    Wbin("ref_d_timestep", t_scaled.grad)
    Wbin("ref_d_guidance", g_scaled.grad)
    Wbin("ref_d_vector", vector.grad)
    # deep double block (bi=0, rode whole chain) + last + deep single
    iw0, tw0 = dbl[0]
    Wbin("ref_d0_img_wqkv", iw0["wqkv"].grad)
    Wbin("ref_d0_img_wproj", iw0["wproj"].grad)
    Wbin("ref_d0_img_wmlp0", iw0["wmlp0"].grad)
    Wbin("ref_d0_txt_wqkv", tw0["wqkv"].grad)
    iwL, twL = dbl[NUM_DOUBLE - 1]
    Wbin("ref_dL_img_wqkv", iwL["wqkv"].grad)
    Wbin("ref_s0_w1", sgl[0]["w1"].grad)
    Wbin("ref_s0_w2", sgl[0]["w2"].grad)
    Wbin("ref_sL_w1", sgl[NUM_SINGLE - 1]["w1"].grad)

    # ── inputs the Mojo gate reconstructs ──
    Wbin("in_img_in", img_in); Wbin("in_img_in_b", img_in_b)
    Wbin("in_txt_in", txt_in); Wbin("in_txt_in_b", txt_in_b)
    for tag, e in [("time", time_in), ("guid", guidance_in), ("vec", vector_in)]:
        for kk in ["in_w", "in_b", "out_w", "out_b"]:
            Wbin("in_%s_%s" % (tag, kk), e[kk])
    for bi in range(NUM_DOUBLE):
        Wbin("in_d%d_imod_w" % bi, dbl_imod[bi]["w"]); Wbin("in_d%d_imod_b" % bi, dbl_imod[bi]["b"])
        Wbin("in_d%d_tmod_w" % bi, dbl_tmod[bi]["w"]); Wbin("in_d%d_tmod_b" % bi, dbl_tmod[bi]["b"])
    for bi in range(NUM_SINGLE):
        Wbin("in_s%d_mod_w" % bi, sgl_mod[bi]["w"]); Wbin("in_s%d_mod_b" % bi, sgl_mod[bi]["b"])
    Wbin("in_final_adaln_w", final_adaln_w); Wbin("in_final_adaln_b", final_adaln_b)
    Wbin("in_final_lin", final_lin); Wbin("in_final_lin_b", final_lin_b)
    Wbin("in_img_tokens", img_tokens); Wbin("in_txt_tokens", txt_tokens)
    Wbin("in_timestep", t_scaled.detach()); Wbin("in_guidance", g_scaled.detach())
    Wbin("in_vector", vector)
    Wbin("in_cos", cos); Wbin("in_sin", sin); Wbin("in_d_out", d_out)
    for bi in range(NUM_DOUBLE):
        iw, tw = dbl[bi]
        for nm, w in [("d%d_iw" % bi, iw), ("d%d_tw" % bi, tw)]:
            for kk in ["wqkv", "bqkv", "wproj", "bproj", "wmlp0", "bmlp0", "wmlp2", "bmlp2", "q_norm", "k_norm"]:
                Wbin("in_%s_%s" % (nm, kk), w[kk])
    for bi in range(NUM_SINGLE):
        w = sgl[bi]
        for kk in ["w1", "b1", "w2", "b2", "q_norm", "k_norm"]:
            Wbin("in_s%d_%s" % (bi, kk), w[kk])

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
