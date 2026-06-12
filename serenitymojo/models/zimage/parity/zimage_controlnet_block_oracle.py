#!/usr/bin/env python3
# serenitymojo/models/zimage/parity/zimage_controlnet_block_oracle.py
#
# Torch oracle for the Z-Image CONTROLNET control block + 2-block control stack
# (models/zimage/controlnet_block.mojo). T2.E.
#
# REFERENCE: diffusers 0.38.0.dev0 ZImageControlTransformerBlock
# (diffusers/models/controlnets/controlnet_z_image.py — the official Alibaba
# Z-Image ControlNet). Control block = ZImageTransformerBlock body (the math
# already gated by block_oracle.py / block_parity.mojo, 19/19) plus:
#   block 0:  c = before_proj(c) + x          (before_proj: Linear(D,D)+bias)
#   hint   :  c_skip = after_proj(c_body_out) (after_proj:  Linear(D,D)+bias)
# chained: block i>0 consumes block i-1's c_out; each block emits one hint.
#
# TWO LAYERS OF PROOF in this oracle:
#   1. F64 hand math (the .bin gate reference): block body copied VERBATIM from
#      block_oracle.py (proven vs the mojo block), wrapped with the control
#      projections; torch autograd supplies every reference grad.
#   2. GROUNDING CROSS-CHECK: the SAME weights are loaded into the REAL
#      diffusers ZImageControlTransformerBlock (run in F64) and the 2-block
#      chained forward must match the hand math. This pins the hand math to the
#      reference implementation, not to my reading of it. (diffusers'
#      apply_rotary_emb internally drops to F32 — `x_in.float()` — so the
#      cross-check bar is max|diff| < 1e-3 on O(1) values; the STRICT gate bar
#      lives in the .bin compare, mojo vs exact-F64 autograd.)
#
# NOTE on projection weights: the gate uses RANDOM (non-zero) projections so the
# full gradient path through before/after_proj is exercised (zero-init after_proj
# would make d(everything upstream of it) tautologically zero). The zero-init
# convention itself is covered by the trainer smoke (projections start 0, move
# off 0).
#
# Run (SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/models/zimage/parity/zimage_controlnet_block_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── dims (MUST match block_oracle.py / the mojo gate) ──
H = 30
Dh = 128
D = H * Dh          # 3840
N_TXT = 2
IMG_H = 2
IMG_W = 3
N_IMG = IMG_H * IMG_W
S = N_TXT + N_IMG   # 8
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


# ── ops (VERBATIM from block_oracle.py) ──
def rms_norm(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def residual_gate(x, gate, y):
    return x + gate * y


def silu(x):
    return x * torch.sigmoid(x)


def rope_interleaved(x, cos, sin):
    Sx = x.shape[1]
    half = Dh // 2
    cr = cos.reshape(Sx, H, half)
    sr = sin.reshape(Sx, H, half)
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
    cos_rows = []
    sin_rows = []
    for tok in range(S):
        if tok < N_TXT:
            p0, p1, p2 = float(tok + 1), 0.0, 0.0
        else:
            it = tok - N_TXT
            p0 = float(N_TXT + 1)
            p1 = float(it // IMG_W)
            p2 = float(it % IMG_W)
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
    return (attn @ vh).permute(0, 2, 1, 3)


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


def make_proj(seed, with_before):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)
    p = {}
    if with_before:
        p["before_w"] = rnd(D, D) * 0.02
        p["before_b"] = rnd(D) * 0.02
    p["after_w"] = rnd(D, D) * 0.02
    p["after_b"] = rnd(D) * 0.02
    for v in p.values():
        v.requires_grad_(True)
    return p


def make_mod(o):
    m = {}
    m["scale_msa"] = (fillc(D, 0.017 + o, 0.20, 0.20)).requires_grad_(True)
    m["gate_msa"] = (fill(D, 0.011 + o, 0.30, 0.40)).requires_grad_(True)
    m["scale_mlp"] = (fillc(D, 0.019 + o, 0.50, 0.15)).requires_grad_(True)
    m["gate_mlp"] = (fill(D, 0.012 + o, 0.60, 0.35)).requires_grad_(True)
    return m


def block_body(x, w, m, cos, sin):
    """VERBATIM Z-Image main-block math from block_oracle.py."""
    xn1 = rms_norm(x, w["n1"])
    xn1s = (1.0 + m["scale_msa"]) * xn1
    q = (xn1s @ w["wq"].T).reshape(1, S, H, Dh)
    k = (xn1s @ w["wk"].T).reshape(1, S, H, Dh)
    v = (xn1s @ w["wv"].T).reshape(1, S, H, Dh)
    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    qr = rope_interleaved(q, cos, sin)
    kr = rope_interleaved(k, cos, sin)
    att = sdpa(qr, kr, v).reshape(S, D)
    att_o = att @ w["wo"].T
    attn_n2 = rms_norm(att_o, w["n2"])
    h = residual_gate(x, torch.tanh(m["gate_msa"]), attn_n2)
    xfn1 = rms_norm(h, w["fn1"])
    xfn1s = (1.0 + m["scale_mlp"]) * xfn1
    g_pre = xfn1s @ w["w1"].T
    u = xfn1s @ w["w3"].T
    act = silu(g_pre) * u
    ff = act @ w["w2"].T
    ff_n2 = rms_norm(ff, w["fn2"])
    return residual_gate(h, torch.tanh(m["gate_mlp"]), ff_n2)


def control_block(c, x, w, p, m, cos, sin, is_first):
    """diffusers ZImageControlTransformerBlock (global-modulation branch)."""
    if is_first:
        c = c @ p["before_w"].T + p["before_b"] + x
    c = block_body(c, w, m, cos, sin)
    hint = c @ p["after_w"].T + p["after_b"]
    return hint, c


def diffusers_cross_check(c0, x, w0, p0, m0, w1, p1, m1, cos, sin,
                          ref_h0, ref_h1, ref_cf):
    """Load the SAME weights into the REAL diffusers blocks; chained forward
    must match the hand math (F64 modulo diffusers' internal F32 rope)."""
    from diffusers.models.controlnets.controlnet_z_image import (
        ZImageControlTransformerBlock, FeedForward,
    )

    def build(block_id, w, p, m):
        blk = ZImageControlTransformerBlock(
            layer_id=block_id, dim=D, n_heads=H, n_kv_heads=H,
            norm_eps=EPS, qk_norm=True, modulation=True, block_id=block_id,
        ).to(DT)
        # small-F gate scale: swap the hardcoded int(D/3*8) FFN for F=96
        blk.feed_forward = FeedForward(dim=D, hidden_dim=F).to(DT)
        sd = {
            "attention.to_q.weight": w["wq"], "attention.to_k.weight": w["wk"],
            "attention.to_v.weight": w["wv"],
            "attention.to_out.0.weight": w["wo"],
            "attention.norm_q.weight": w["q_norm"],
            "attention.norm_k.weight": w["k_norm"],
            "attention_norm1.weight": w["n1"], "attention_norm2.weight": w["n2"],
            "ffn_norm1.weight": w["fn1"], "ffn_norm2.weight": w["fn2"],
            "feed_forward.w1.weight": w["w1"], "feed_forward.w3.weight": w["w3"],
            "feed_forward.w2.weight": w["w2"],
            # adaLN: weight=0, bias=the RAW chunks -> mod == chunks exactly
            "adaLN_modulation.0.weight": torch.zeros(4 * D, 256, dtype=DT),
            "adaLN_modulation.0.bias": torch.cat(
                [m["scale_msa"], m["gate_msa"], m["scale_mlp"], m["gate_mlp"]]),
            "after_proj.weight": p["after_w"], "after_proj.bias": p["after_b"],
        }
        if block_id == 0:
            sd["before_proj.weight"] = p["before_w"]
            sd["before_proj.bias"] = p["before_b"]
        missing, unexpected = blk.load_state_dict(
            {k: v.detach() for k, v in sd.items()}, strict=True)
        assert not missing and not unexpected
        return blk

    blk0 = build(0, w0, p0, m0)
    blk1 = build(1, w1, p1, m1)

    # freqs_cis [1, S, Dh/2] complex from the head-0 rows of the tables
    cos_tok = cos.reshape(S, H, Dh // 2)[:, 0, :]
    sin_tok = sin.reshape(S, H, Dh // 2)[:, 0, :]
    freqs_cis = torch.complex(cos_tok, sin_tok).unsqueeze(0)
    adaln_input = torch.zeros(1, 256, dtype=DT)

    with torch.no_grad():
        cb = c0.detach().unsqueeze(0)
        xb = x.detach().unsqueeze(0)
        out0 = blk0(cb, xb, None, freqs_cis, adaln_input)   # [2,1,S,D]
        out1 = blk1(out0, xb, None, freqs_cis, adaln_input) # [3,1,S,D]
        parts = torch.unbind(out1)
        d_h0 = (parts[0][0] - ref_h0).abs().max().item()
        d_h1 = (parts[1][0] - ref_h1).abs().max().item()
        d_cf = (parts[2][0] - ref_cf).abs().max().item()
    print(f"diffusers cross-check max|diff|: hint0={d_h0:.3e} "
          f"hint1={d_h1:.3e} c_final={d_cf:.3e}")
    assert d_h0 < 1e-3 and d_h1 < 1e-3 and d_cf < 1e-3, \
        "hand math diverges from the REAL diffusers ZImageControlTransformerBlock"
    print("diffusers cross-check: PASS (hand math == reference implementation)")


def main():
    c0 = t2(S, D, 0.021, 0.05, 0.5).requires_grad_(True)
    x = t2(S, D, 0.023, 0.31, 0.5).requires_grad_(True)
    w0, w1 = make_weights(1), make_weights(2)
    p0, p1 = make_proj(11, True), make_proj(12, False)
    m0, m1 = make_mod(0.0), make_mod(0.003)

    cos, sin = build_real_rope_tables()
    assert float(cos.std()) > 1e-3, "degenerate rope table"

    # ── 2-block control stack forward (hand math == gate reference) ──
    h0, c1 = control_block(c0, x, w0, p0, m0, cos, sin, True)
    h1, cf = control_block(c1, x, w1, p1, m1, cos, sin, False)

    # ── upstream grads on the two hints (the injection-site grads); c_final
    #    receives NO grad (the trainer never consumes it) ──
    dh0 = t2(S, D, 0.027, 0.11, 0.05)
    dh1 = t2(S, D, 0.029, 0.41, 0.05)
    loss = (h0 * dh0).sum() + (h1 * dh1).sum()
    loss.backward()

    # ── ground the hand math against the REAL diffusers implementation ──
    diffusers_cross_check(c0, x, w0, p0, m0, w1, p1, m1, cos, sin,
                          h0.detach(), h1.detach(), cf.detach())

    def W(name, tensor):
        flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
        with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
            f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
        print("wrote", name, flat.shape)

    # forward refs
    W("cn_ref_hint0", h0)
    W("cn_ref_hint1", h1)
    W("cn_ref_c_final", cf)
    # stream grads
    W("cn_ref_d_c0", c0.grad)
    W("cn_ref_d_x", x.grad)
    # per-block trainable grads
    WKEYS = ["n1", "wq", "wk", "wv", "wo", "q_norm", "k_norm",
             "n2", "fn1", "w1", "w3", "w2", "fn2"]
    MKEYS = ["scale_msa", "gate_msa", "scale_mlp", "gate_mlp"]
    for tag, w, p, m in (("b0", w0, p0, m0), ("b1", w1, p1, m1)):
        for kk in WKEYS:
            W(f"cn_ref_{tag}_d_{kk}", w[kk].grad)
        for kk in MKEYS:
            W(f"cn_ref_{tag}_d_{kk}", m[kk].grad)
        if "before_w" in p:
            W(f"cn_ref_{tag}_d_before_w", p["before_w"].grad)
            W(f"cn_ref_{tag}_d_before_b", p["before_b"].grad)
        W(f"cn_ref_{tag}_d_after_w", p["after_w"].grad)
        W(f"cn_ref_{tag}_d_after_b", p["after_b"].grad)

    # inputs the mojo gate reconstructs
    W("cn_in_c0", c0)
    W("cn_in_x", x)
    for tag, w, p, m in (("b0", w0, p0, m0), ("b1", w1, p1, m1)):
        for kk in WKEYS:
            W(f"cn_in_{tag}_w_{kk}", w[kk])
        for kk in MKEYS:
            W(f"cn_in_{tag}_m_{kk}", m[kk])
        if "before_w" in p:
            W(f"cn_in_{tag}_before_w", p["before_w"])
            W(f"cn_in_{tag}_before_b", p["before_b"])
        W(f"cn_in_{tag}_after_w", p["after_w"])
        W(f"cn_in_{tag}_after_b", p["after_b"])
    W("cn_in_cos", cos)
    W("cn_in_sin", sin)
    W("cn_in_d_hint0", dh0)
    W("cn_in_d_hint1", dh1)

    print("forward loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()
