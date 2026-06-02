#!/usr/bin/env python3
# serenitymojo/models/flux/parity/lora_stack_oracle.py
#
# Torch-autograd oracle for the Flux (flux1-dev) FULL DiT STACK *WITH LoRA* at
# REDUCED depth (NUM_DOUBLE double + NUM_SINGLE single, REAL H/Dh/D). Built
# INDEPENDENTLY from inference-flame/src/models/flux1_dit.rs (the forward) + the
# OneTrainer Flux LoRA target set (convert_flux_lora.py:6-41): LoRA on the SPLIT
# q/k/v + proj + mlp0/mlp2 (double) and split q/k/v + proj_mlp + linear2 (single).
#
# LoRA math (matches train_step._lora_fwd): y' = base(x) + scale*((x@Aᵀ)@Bᵀ),
# A=[rank,in], B=[out,rank], scale=alpha/rank. q/k/v are SEPARATE adapters acting
# on the 3 D-slices of the fused qkv output (the OT-faithful per-projection model)
# — each shares the SAME projection input (the modulated layer-norm `norm`).
#
# This dumps, per adapter, the torch-autograd d_A and d_B (the gate asserts ALL of
# them cos>=0.999), plus the load-bearing input-token + embed grads and the base
# output (no-regression check uses a separate alpha-agnostic comparison in the
# step smoke). B is initialized NONZERO here so every grad arm is exercised.
#
# Run (SEPARATE command, never chained after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/flux/parity/lora_stack_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── dims (MUST match lora_stack_parity.mojo) ──
H = 24
Dh = 128
D = H * Dh                # 3072 (REAL Flux inner_dim)
N_IMG = 4
N_TXT = 3
S = N_TXT + N_IMG
FMLP = 32                 # GELU MLP hidden (small for speed)
IN_CH = 64
TXT_CH = 40
OUT_CH = 64
T_DIM = 16
VEC_DIM = 20
NUM_DOUBLE = 2
NUM_SINGLE = 2
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
AXES = [16, 56, 56]
ROPE_THETA = 10000.0
MAX_PERIOD = 10000.0
GUIDANCE = 3.5
RANK = 4
ALPHA = 1.0               # OT default
LSCALE = ALPHA / RANK

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def fill(n, a, b, c):
    return torch.tensor([math.sin(a * i + b) * c for i in range(n)], dtype=DT)


def t2(n, m, a, b, c):
    return fill(n * m, a, b, c).reshape(n, m)


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
    half = T_DIM // 2
    freqs = torch.tensor(
        [math.exp(-math.log(MAX_PERIOD) * i / half) for i in range(half)],
        dtype=DT,
    )
    angle = t_scalar * freqs
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=0)


# ── LoRA adapter (A,B trainable; B NONZERO so grads are exercised) ──
def make_lora(seed, in_f, out_f):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(RANK, in_f, generator=g, dtype=torch.float32) * 0.02).to(DT)
    b = (torch.randn(out_f, RANK, generator=g, dtype=torch.float32) * 0.02).to(DT)
    a.requires_grad_(True); b.requires_grad_(True)
    return {"a": a, "b": b}


def lora_apply(x, lo):
    # x [M,in] -> [M,out]
    return LSCALE * ((x @ lo["a"].T) @ lo["b"].T)


# ── base weights (mirror stack_oracle.py) ──
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


def double_block(img, txt, iw, tw, im, tm, il, tl, cos, sin):
    def stream_pre(x, w, lo, m, n):
        norm = modulate_pre(x, m["shift1"], m["scale1"])
        qkv = norm @ w["wqkv"].T + w["bqkv"]
        q = qkv[:, 0:D] + lora_apply(norm, lo["to_q"])
        k = qkv[:, D:2 * D] + lora_apply(norm, lo["to_k"])
        v = qkv[:, 2 * D:3 * D] + lora_apply(norm, lo["to_v"])
        q = rms_norm_lastdim(q.reshape(1, n, H, Dh), w["q_norm"])
        k = rms_norm_lastdim(k.reshape(1, n, H, Dh), w["k_norm"])
        v = v.reshape(1, n, H, Dh)
        return q, k, v

    iq, ik, iv = stream_pre(img, iw, il, im, N_IMG)
    tq, tk, tv = stream_pre(txt, tw, tl, tm, N_TXT)
    q = torch.cat([tq, iq], dim=1)
    k = torch.cat([tk, ik], dim=1)
    v = torch.cat([tv, iv], dim=1)
    qr = rope_interleaved(q, cos, sin, S)
    kr = rope_interleaved(k, cos, sin, S)
    att = sdpa(qr, kr, v)
    txt_att = att[:, 0:N_TXT].reshape(N_TXT, D)
    img_att = att[:, N_TXT:N_TXT + N_IMG].reshape(N_IMG, D)

    def stream_post(x, att, w, lo, m):
        out = att @ w["wproj"].T + w["bproj"] + lora_apply(att, lo["proj"])
        attn_res = residual_gate(x, m["gate1"], out)
        mlp_in = modulate_pre(attn_res, m["shift2"], m["scale2"])
        mlp_pre = mlp_in @ w["wmlp0"].T + w["bmlp0"] + lora_apply(mlp_in, lo["mlp0"])
        mlp_h = gelu_tanh(mlp_pre)
        mlp = mlp_h @ w["wmlp2"].T + w["bmlp2"] + lora_apply(mlp_h, lo["mlp2"])
        return residual_gate(attn_res, m["gate2"], mlp)

    img_out = stream_post(img, img_att, iw, il, im)
    txt_out = stream_post(txt, txt_att, tw, tl, tm)
    return img_out, txt_out


def single_block(x, w, lo, m, cos, sin):
    n = x.shape[0]
    norm = modulate_pre(x, m["shift"], m["scale"])
    fused = norm @ w["w1"].T + w["b1"]
    q = fused[:, 0:D] + lora_apply(norm, lo["to_q"])
    k = fused[:, D:2 * D] + lora_apply(norm, lo["to_k"])
    v = fused[:, 2 * D:3 * D] + lora_apply(norm, lo["to_v"])
    mlp_in = fused[:, 3 * D:3 * D + FMLP] + lora_apply(norm, lo["proj_mlp"])
    q = rms_norm_lastdim(q.reshape(1, n, H, Dh), w["q_norm"])
    k = rms_norm_lastdim(k.reshape(1, n, H, Dh), w["k_norm"])
    v = v.reshape(1, n, H, Dh)
    qr = rope_interleaved(q, cos, sin, n)
    kr = rope_interleaved(k, cos, sin, n)
    att = sdpa(qr, kr, v).reshape(n, D)
    mlp_h = gelu_tanh(mlp_in)
    out_in = torch.cat([att, mlp_h], dim=1)
    out = out_in @ w["w2"].T + w["b2"] + lora_apply(out_in, lo["linear2"])
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
    # base
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

    img_tokens = t2(N_IMG, IN_CH, 0.0031, 0.05, 0.5).requires_grad_(True)
    txt_tokens = t2(N_TXT, TXT_CH, 0.0033, 0.07, 0.5).requires_grad_(True)
    t_scaled = torch.tensor([0.37 * 1000.0], dtype=DT).requires_grad_(True)
    g_scaled = torch.tensor([GUIDANCE * 1000.0], dtype=DT).requires_grad_(True)
    vector = t2(1, VEC_DIM, 0.041, 0.09, 0.4).requires_grad_(True)

    dbl = [(make_stream(100 + bi * 2), make_stream(101 + bi * 2)) for bi in range(NUM_DOUBLE)]
    sgl = [make_single_weights(200 + bi) for bi in range(NUM_SINGLE)]

    # ── LoRA adapters: per double block, img + txt each {to_q,to_k,to_v,proj,mlp0,mlp2} ──
    def make_stream_lora(seed0):
        return {
            "to_q": make_lora(seed0 + 0, D, D), "to_k": make_lora(seed0 + 1, D, D),
            "to_v": make_lora(seed0 + 2, D, D), "proj": make_lora(seed0 + 3, D, D),
            "mlp0": make_lora(seed0 + 4, D, FMLP), "mlp2": make_lora(seed0 + 5, FMLP, D),
        }
    dbl_lora = []
    sd = 5000
    for bi in range(NUM_DOUBLE):
        il = make_stream_lora(sd); sd += 6
        tl = make_stream_lora(sd); sd += 6
        dbl_lora.append((il, tl))
    sgl_lora = []
    for bi in range(NUM_SINGLE):
        sl = {
            "to_q": make_lora(sd + 0, D, D), "to_k": make_lora(sd + 1, D, D),
            "to_v": make_lora(sd + 2, D, D), "proj_mlp": make_lora(sd + 3, D, FMLP),
            "linear2": make_lora(sd + 4, D + FMLP, D),
        }
        sd += 5
        sgl_lora.append(sl)

    ids = torch.tensor(
        [[float(s), float((s * 3) % 5), float((s * 2) % 7)] for s in range(S)],
        dtype=DT,
    )
    cos, sin = build_rope(S, ids)

    t_emb = timestep_embedding(t_scaled).reshape(1, T_DIM)
    g_emb = timestep_embedding(g_scaled).reshape(1, T_DIM)
    vec = mlp_embed(t_emb, time_in) + mlp_embed(g_emb, guidance_in) + mlp_embed(vector, vector_in)
    vec.retain_grad()
    vec_silu = torch.nn.functional.silu(vec)

    img = img_tokens @ img_in.T + img_in_b
    txt = txt_tokens @ txt_in.T + txt_in_b
    for bi in range(NUM_DOUBLE):
        iw, tw = dbl[bi]
        il, tl = dbl_lora[bi]
        imods = chunk6(vec_silu @ dbl_imod[bi]["w"].T + dbl_imod[bi]["b"])
        tmods = chunk6(vec_silu @ dbl_tmod[bi]["w"].T + dbl_tmod[bi]["b"])
        img, txt = double_block(img, txt, iw, tw, imods, tmods, il, tl, cos, sin)
    x = torch.cat([txt, img], dim=0)
    for bi in range(NUM_SINGLE):
        smods = chunk3(vec_silu @ sgl_mod[bi]["w"].T + sgl_mod[bi]["b"])
        x = single_block(x, sgl[bi], sgl_lora[bi], smods, cos, sin)
    img_out = x[N_TXT:N_TXT + N_IMG]
    fmods = vec_silu @ final_adaln_w.T + final_adaln_b
    f_shift = fmods[:, 0:D].reshape(D)
    f_scale = fmods[:, D:2 * D].reshape(D)
    normed = modulate(layer_norm(img_out), f_scale, f_shift)
    out = normed @ final_lin.T + final_lin_b

    d_out = t2(N_IMG, OUT_CH, 0.0027, 0.11, 0.05)
    loss = (out * d_out).sum()
    loss.backward()

    # ── gate outputs ──
    Wbin("lr_out", out)
    Wbin("lr_d_img_tokens", img_tokens.grad)
    Wbin("lr_d_txt_tokens", txt_tokens.grad)
    Wbin("lr_d_vec", vec.grad)
    Wbin("lr_d_timestep", t_scaled.grad)
    Wbin("lr_d_guidance", g_scaled.grad)
    Wbin("lr_d_vector", vector.grad)

    # per-adapter d_A / d_B (ALL of them — the composition gate asserts each)
    def dump_stream_lora(tag, lo):
        for slot in ["to_q", "to_k", "to_v", "proj", "mlp0", "mlp2"]:
            Wbin("lr_%s_%s_dA" % (tag, slot), lo[slot]["a"].grad)
            Wbin("lr_%s_%s_dB" % (tag, slot), lo[slot]["b"].grad)
    for bi in range(NUM_DOUBLE):
        il, tl = dbl_lora[bi]
        dump_stream_lora("d%d_img" % bi, il)
        dump_stream_lora("d%d_txt" % bi, tl)
    for bi in range(NUM_SINGLE):
        sl = sgl_lora[bi]
        for slot in ["to_q", "to_k", "to_v", "proj_mlp", "linear2"]:
            Wbin("lr_s%d_%s_dA" % (bi, slot), sl[slot]["a"].grad)
            Wbin("lr_s%d_%s_dB" % (bi, slot), sl[slot]["b"].grad)

    # ── base inputs the Mojo gate reconstructs (same names as stack_oracle.py) ──
    Wbin("lin_img_in", img_in); Wbin("lin_img_in_b", img_in_b)
    Wbin("lin_txt_in", txt_in); Wbin("lin_txt_in_b", txt_in_b)
    for tag, e in [("time", time_in), ("guid", guidance_in), ("vec", vector_in)]:
        for kk in ["in_w", "in_b", "out_w", "out_b"]:
            Wbin("lin_%s_%s" % (tag, kk), e[kk])
    for bi in range(NUM_DOUBLE):
        Wbin("lin_d%d_imod_w" % bi, dbl_imod[bi]["w"]); Wbin("lin_d%d_imod_b" % bi, dbl_imod[bi]["b"])
        Wbin("lin_d%d_tmod_w" % bi, dbl_tmod[bi]["w"]); Wbin("lin_d%d_tmod_b" % bi, dbl_tmod[bi]["b"])
    for bi in range(NUM_SINGLE):
        Wbin("lin_s%d_mod_w" % bi, sgl_mod[bi]["w"]); Wbin("lin_s%d_mod_b" % bi, sgl_mod[bi]["b"])
    Wbin("lin_final_adaln_w", final_adaln_w); Wbin("lin_final_adaln_b", final_adaln_b)
    Wbin("lin_final_lin", final_lin); Wbin("lin_final_lin_b", final_lin_b)
    Wbin("lin_img_tokens", img_tokens); Wbin("lin_txt_tokens", txt_tokens)
    Wbin("lin_timestep", t_scaled.detach()); Wbin("lin_guidance", g_scaled.detach())
    Wbin("lin_vector", vector)
    Wbin("lin_cos", cos); Wbin("lin_sin", sin); Wbin("lin_d_out", d_out)
    for bi in range(NUM_DOUBLE):
        iw, tw = dbl[bi]
        for nm, w in [("d%d_iw" % bi, iw), ("d%d_tw" % bi, tw)]:
            for kk in ["wqkv", "bqkv", "wproj", "bproj", "wmlp0", "bmlp0", "wmlp2", "bmlp2", "q_norm", "k_norm"]:
                Wbin("lin_%s_%s" % (nm, kk), w[kk])
    for bi in range(NUM_SINGLE):
        w = sgl[bi]
        for kk in ["w1", "b1", "w2", "b2", "q_norm", "k_norm"]:
            Wbin("lin_s%d_%s" % (bi, kk), w[kk])

    # LoRA A/B masters (the Mojo gate seeds its FluxLoraSet from these)
    def dump_stream_lora_ab(tag, lo):
        for slot in ["to_q", "to_k", "to_v", "proj", "mlp0", "mlp2"]:
            Wbin("lin_%s_%s_A" % (tag, slot), lo[slot]["a"])
            Wbin("lin_%s_%s_B" % (tag, slot), lo[slot]["b"])
    for bi in range(NUM_DOUBLE):
        il, tl = dbl_lora[bi]
        dump_stream_lora_ab("d%d_img" % bi, il)
        dump_stream_lora_ab("d%d_txt" % bi, tl)
    for bi in range(NUM_SINGLE):
        sl = sgl_lora[bi]
        for slot in ["to_q", "to_k", "to_v", "proj_mlp", "linear2"]:
            Wbin("lin_s%d_%s_A" % (bi, slot), sl[slot]["a"])
            Wbin("lin_s%d_%s_B" % (bi, slot), sl[slot]["b"])

    print("forward loss =", float(loss))
    print("RANK =", RANK, "ALPHA =", ALPHA, "NUM_DOUBLE =", NUM_DOUBLE, "NUM_SINGLE =", NUM_SINGLE)
    print("DONE")


if __name__ == "__main__":
    main()
