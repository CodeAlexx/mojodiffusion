#!/usr/bin/env python3
# serenitymojo/models/anima/parity/stack_oracle.py
#
# Torch oracle for the ANIMA MiniTrainDIT FULL STACK composition (patch_embed +
# L x transformer block + final layer), forward + autograd grads. Replicates the
# EXACT math of serenitymojo/models/anima/anima_stack.mojo, which composes the
# parity-verified anima_block (block_oracle.py). Built INDEPENDENTLY from the
# REFERENCE math (inference-flame anima.rs), NOT from the Mojo — so it is a real
# adversarial composition check.
#
# ANIMA composition detail (THE thing this gate proves): modulation is PER-BLOCK
# (each block has its own mod1/mod2 for self/cross/mlp), computed from the SHARED
# t_silu + base_adaln. So d_t_silu + d_base_adaln SUM across all L blocks + the
# final layer (the Klein composition lesson: per-block-correct does NOT imply
# composition-correct; this gate is the proof). Per-block mod-weight grads are
# distinct per layer (NOT summed) — probed deepest + shallowest.
#
# Small depth L + small S to keep the torch graph + GPU bounded. REAL H=16,
# Dh=128 (D=2048) so the head structure + RoPE table are the real thing.
#
# Graph (mirrors anima_stack.mojo forward):
#   x = patches @ x_embed.T                              # [BS, D]  (no bias)
#   t_silu = leaf (the block consumes silu(t_cond))
#   for l in range(L): x = anima_block(x, blk[l], t_silu, base_adaln, ctx, cos, sin)
#   final: fl_h = t_silu @ fl_mod1.T; fl_modout = fl_h @ fl_mod2.T + base_adaln[:4096]
#          shift,scale = chunk2(fl_modout)
#          x_mod = (1+scale)*LayerNorm_noaffine(x) + shift
#          out = x_mod @ fl_lin.T                        # [BS, 64]
#
# Run (SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/stack_oracle.py

import math
import struct
import os
import torch

torch.manual_seed(0)
DT = torch.float64

# ── dims: REAL Anima head count/dim; small seqs / tiny F for a light oracle ──
B = 1
H = 16
Dh = 128
D = H * Dh          # 2048
S_IMG = 6
S_TXT = 8
JOINT = 1024        # cross-attn context dim
F = 32              # mlp hidden (tiny stand-in for 8192; math identical)
ADALN = 256
IN_PATCH = 68       # x_embedder input (16+1)*2*2
OUT_PATCH = 64      # final linear output 16*2*2
L = 3               # number of stacked blocks
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
THETA = 10000.0

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


def rnd(g, *shape):
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(DT)


# ── ops (match anima.rs math) ──
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


def make_block_weights(g):
    # NOTE on scale: at D=2048 with L stacked blocks and F32-vs-F64 interiors, an
    # UNSCALED random projection weight amplifies the residual stream to ~1e6 after
    # one block (sqrt(D)*std growth), and the composition cosine then measures F32
    # rounding noise on huge magnitudes, NOT composition correctness (the inverse
    # of the Klein degenerate-table trap). Real Anima weights are small (BF16,
    # ~1/sqrt(D)); scaling the projections by 0.02 keeps the residual bounded so the
    # gate measures the genuine composed fwd/bwd. Per-block weights are still
    # DISTINCT per layer (seed differs), so deepest != shallowest grads.
    w = {}
    for sub in ("sa", "ca", "mlp"):
        w[f"{sub}_mod1"] = (rnd(g, ADALN, D) * 0.02).requires_grad_(True)   # [256,2048]
        w[f"{sub}_mod2"] = (rnd(g, 3 * D, ADALN) * 0.02).requires_grad_(True)  # [6144,256]
    w["sa_q"] = (rnd(g, D, D) * 0.02).requires_grad_(True)
    w["sa_k"] = (rnd(g, D, D) * 0.02).requires_grad_(True)
    w["sa_v"] = (rnd(g, D, D) * 0.02).requires_grad_(True)
    w["sa_out"] = (rnd(g, D, D) * 0.02).requires_grad_(True)
    w["sa_qn"] = (rnd(g, Dh) * 0.1 + 1.0).requires_grad_(True)
    w["sa_kn"] = (rnd(g, Dh) * 0.1 + 1.0).requires_grad_(True)
    w["ca_q"] = (rnd(g, D, D) * 0.02).requires_grad_(True)
    w["ca_k"] = (rnd(g, D, JOINT) * 0.02).requires_grad_(True)
    w["ca_v"] = (rnd(g, D, JOINT) * 0.02).requires_grad_(True)
    w["ca_out"] = (rnd(g, D, D) * 0.02).requires_grad_(True)
    w["ca_qn"] = (rnd(g, Dh) * 0.1 + 1.0).requires_grad_(True)
    w["ca_kn"] = (rnd(g, Dh) * 0.1 + 1.0).requires_grad_(True)
    w["mlp1"] = (rnd(g, F, D) * 0.02).requires_grad_(True)
    w["mlp2"] = (rnd(g, D, F) * 0.02).requires_grad_(True)
    return w


def adaln_mod(w, sub, t_silu, base_adaln):
    # NOTE: t_silu here is already silu(t_cond) (the shared intermediate the block
    # produces internally). The stack passes RAW t_cond; the block applies silu.
    h = t_silu @ w[f"{sub}_mod1"].T            # [B,256]
    mod_out = h @ w[f"{sub}_mod2"].T + base_adaln   # [B,6144]
    shift = mod_out[:, 0:D]
    scale = mod_out[:, D:2 * D]
    gate = mod_out[:, 2 * D:3 * D]
    return shift, scale, gate


def adaln_pre(xx, shift, scale):
    ln = layer_norm_noaffine(xx)
    return (1.0 + scale).unsqueeze(1) * ln + shift.unsqueeze(1)


def block_forward(x, w, t_silu, base_adaln, context, cos, sin):
    # self-attn
    sh, sc, ga = adaln_mod(w, "sa", t_silu, base_adaln)
    xmod = adaln_pre(x, sh, sc)
    q = (xmod @ w["sa_q"].T).reshape(B, S_IMG, H, Dh)
    k = (xmod @ w["sa_k"].T).reshape(B, S_IMG, H, Dh)
    v = (xmod @ w["sa_v"].T).reshape(B, S_IMG, H, Dh)
    q = rms_norm_lastdim(q, w["sa_qn"])
    k = rms_norm_lastdim(k, w["sa_kn"])
    q = rope_halfsplit(q, cos, sin)
    k = rope_halfsplit(k, cos, sin)
    att = sdpa(q, k, v).reshape(B, S_IMG, D)
    sa_out = att @ w["sa_out"].T
    x = x + ga.unsqueeze(1) * sa_out
    # cross-attn
    sh, sc, ga = adaln_mod(w, "ca", t_silu, base_adaln)
    xmod = adaln_pre(x, sh, sc)
    q = (xmod @ w["ca_q"].T).reshape(B, S_IMG, H, Dh)
    k = (context @ w["ca_k"].T).reshape(B, S_TXT, H, Dh)
    v = (context @ w["ca_v"].T).reshape(B, S_TXT, H, Dh)
    q = rms_norm_lastdim(q, w["ca_qn"])
    k = rms_norm_lastdim(k, w["ca_kn"])
    att = sdpa(q, k, v).reshape(B, S_IMG, D)
    ca_out = att @ w["ca_out"].T
    x = x + ga.unsqueeze(1) * ca_out
    # mlp
    sh, sc, ga = adaln_mod(w, "mlp", t_silu, base_adaln)
    xmod = adaln_pre(x, sh, sc)
    h = gelu_tanh(xmod @ w["mlp1"].T)
    mlp_out = h @ w["mlp2"].T
    x = x + ga.unsqueeze(1) * mlp_out
    return x


def main():
    g = torch.Generator().manual_seed(1)

    # ── inputs ──
    patches = (rnd(g, B * S_IMG, IN_PATCH) * 0.5).requires_grad_(True)
    # RAW t_cond is the leaf (the stack passes RAW t_cond; the block + final layer
    # apply silu internally — matching inference-flame anima.rs and the block
    # oracle). t_silu is the SHARED intermediate; we retain_grad on it so
    # ref_d_t_silu == the grad-w.r.t.-silu-output that anima_block_backward returns
    # (block returns d_t_silu, NOT d_t_cond — the t_embedder backprop is phase E).
    t_cond = (rnd(g, B, D) * 0.5).requires_grad_(True)
    t_silu = torch.nn.functional.silu(t_cond)
    t_silu.retain_grad()
    base_adaln = (rnd(g, B, 3 * D) * 0.3).requires_grad_(True)
    context = (rnd(g, B, S_TXT, JOINT) * 0.5).requires_grad_(True)

    # base weights
    x_embed = (rnd(g, D, IN_PATCH) * 0.05).requires_grad_(True)     # [D,68]
    fl_mod1 = (rnd(g, ADALN, D) * 0.1).requires_grad_(True)         # [256,2048]
    fl_mod2 = (rnd(g, 2 * D, ADALN) * 0.1).requires_grad_(True)     # [4096,256]
    fl_lin = (rnd(g, OUT_PATCH, D) * 0.05).requires_grad_(True)     # [64,2048]

    # rope tables (per position [S_IMG, Dh/2]) — non-degenerate angles.
    half = Dh // 2
    pos = torch.arange(S_IMG, dtype=DT).reshape(S_IMG, 1)
    freq = torch.tensor(
        [1.0 / (THETA ** (2.0 * i / Dh)) for i in range(half)], dtype=DT
    ).reshape(1, half)
    ang = pos * freq
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    assert float((cos[:, :] - 1.0).abs().max()) > 1e-3, "degenerate table"

    # per-block weights (distinct per layer so deepest != shallowest grads)
    blk = [make_block_weights(torch.Generator().manual_seed(100 + l)) for l in range(L)]

    # ── forward ──
    x = patches @ x_embed.T                              # [BS, D]
    W("ref_x_emb", x)                                    # debug: patch-embed output
    x = x.reshape(B, S_IMG, D)
    x_after = []
    for l in range(L):
        x = block_forward(x, blk[l], t_silu, base_adaln, context, cos, sin)
        x_after.append(x)
    for l in range(L):
        W("ref_x_after_%d" % l, x_after[l])              # debug: per-block output
    x_final = x
    # final layer
    fl_h = t_silu @ fl_mod1.T                            # [B,256]
    fl_modout = fl_h @ fl_mod2.T + base_adaln[:, :2 * D]  # [B,4096]
    fl_shift = fl_modout[:, 0:D]
    fl_scale = fl_modout[:, D:2 * D]
    x_mod = adaln_pre(x_final, fl_shift, fl_scale)        # [B,S,D]
    out = (x_mod.reshape(B * S_IMG, D) @ fl_lin.T)        # [BS, 64]

    # ── upstream grad ──
    d_out = (rnd(g, B * S_IMG, OUT_PATCH) * 0.05)
    (out * d_out).sum().backward()

    # ── references ──
    W("ref_x_final", x_final)   # last-block output (pre final layer) — debug probe
    W("ref_out", out)
    W("ref_d_patches", patches.grad)
    W("ref_d_t_silu", t_silu.grad)
    W("ref_d_base_adaln", base_adaln.grad)
    W("ref_d_x_embed", x_embed.grad)
    W("ref_d_fl_lin", fl_lin.grad)
    W("ref_d_fl_mod1", fl_mod1.grad)
    W("ref_d_fl_mod2", fl_mod2.grad)
    # per-block weight grads: deepest (L-1) + shallowest (0). Probe sa_q + mlp2 +
    # sa_mod1 (the per-block AdaLN-LoRA arm).
    for tag, l in (("deep", L - 1), ("shallow", 0)):
        W("ref_d_sa_q_%s" % tag, blk[l]["sa_q"].grad)
        W("ref_d_mlp2_%s" % tag, blk[l]["mlp2"].grad)
        W("ref_d_sa_mod1_%s" % tag, blk[l]["sa_mod1"].grad)
        W("ref_d_ca_v_%s" % tag, blk[l]["ca_v"].grad)

    # ── inputs the Mojo gate reconstructs ──
    W("in_patches", patches.detach())
    # Dump RAW t_cond (the stack input; block + final layer silu it internally).
    W("in_t_cond", t_cond.detach())
    W("in_base_adaln", base_adaln.detach())
    W("in_context", context.detach())
    W("in_x_embed", x_embed.detach())
    W("in_fl_mod1", fl_mod1.detach())
    W("in_fl_mod2", fl_mod2.detach())
    W("in_fl_lin", fl_lin.detach())
    W("in_cos", cos)
    W("in_sin", sin)
    W("in_d_out", d_out)
    for l in range(L):
        for kk in blk[l]:
            W("in_blk%d_%s" % (l, kk), blk[l][kk].detach())

    print("forward out norm =", float(out.norm()))
    print("L =", L, " S_IMG =", S_IMG, " S_TXT =", S_TXT, " D =", D, " F =", F)
    print("DONE")


if __name__ == "__main__":
    main()
