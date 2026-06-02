#!/usr/bin/env python3
# spatial_transformer_oracle.py — torch autograd reference for the SDXL
# SpatialTransformer cross-attn block fwd+bwd (models/sdxl/spatial_transformer.mojo).
#
# Built INDEPENDENTLY from inference-flame/src/models/sdxl_unet.rs
# (spatial_transformer 761-817 + basic_transformer_block 702-755 +
# cross_attention 661-697 + geglu 637-655). NOT derived from the Mojo port.
#
# Math (all F32 interior; the Mojo port works NHWC, the math is layout-agnostic):
#   ST: residual=x; xn=GroupNorm(x,32,eps=1e-6); tok=flatten(xn)->[B,N,C];
#       h=Linear(tok, proj_in); for j: h=BTB(h, ctx); po=Linear(h, proj_out);
#       out = residual + reshape(po)
#   BTB: x=x+to_out(selfattn(LN1(x))); x=x+to_out(crossattn(LN2(x),ctx));
#        x=x+Linear(GEGLU(LN3(x)))
#   attn: q=x@Wqᵀ; k=c@Wkᵀ; v=c@Wvᵀ; heads=C/Dh; sdpa scale=1/sqrt(Dh);
#         out=concat_heads @ ... ; to_out applied OUTSIDE (in BTB) = Linear(o,Wo,bo)
#   GELU = tanh-approx (matches Mojo ops gelu + flame-core).
#
# GroupNorm here: we feed NCHW to torch GN, but the Mojo port is NHWC. GroupNorm
# normalizes per (group of channels) over spatial — identical regardless of
# memory layout. The oracle emits per-element grads flattened in the SAME order
# the Mojo harness reads them: weights are 1-D/2-D parameter tensors (layout
# matches), and d_x/out are compared via cosine (order-invariant up to the
# consistent flatten both sides use — the Mojo side flattens NHWC, torch NCHW, so
# for the spatial tensors out/d_x/d_context we compare a CANONICAL [B,N,C]-ordered
# flatten on BOTH sides to avoid a layout-permutation false-negative).
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python .../spatial_transformer_oracle.py

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# DEPTH from argv (default 1); ref file is suffixed for depth!=1.
DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 1
_suffix = "" if DEPTH == 1 else f"_d{DEPTH}"
OUT = os.path.join(os.path.dirname(__file__), f"spatial_transformer_ref{_suffix}.txt")

# small parity dims
B, H, W = 1, 4, 4
C = 128          # 2 heads * 64
Dh = 64
Hh = C // Dh     # 2
N = H * W        # 16
Nkv = 77         # text tokens (real SDXL)
Cctx = 16        # context dim (small for parity)
Cff = 32         # FF inner dim
G = 32           # group norm groups
LN_EPS = 1e-5
GN_EPS = 1e-6


def fill(n, a, b, c, scale=0.05):
    out = np.empty(n, np.float64)
    for i in range(n):
        out[i] = (float((i * a) % b) - c) * scale
    return out


def T(n, a, b, c, shape, scale=0.05, grad=False):
    t = torch.tensor(fill(n, a, b, c, scale), dtype=torch.float64).reshape(shape)
    if grad:
        t = t.requires_grad_(True)
    return t


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


def sdpa(q, k, v):
    # q [B,Hh,Sq,Dh], k/v [B,Hh,Skv,Dh]
    scale = 1.0 / np.sqrt(Dh)
    scores = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return attn @ v  # [B,Hh,Sq,Dh]


def attn(x, c, Wq, Wk, Wv, Wo, bo):
    # x [B,Sq,C]; c [B,Skv,Cctx]; returns to_out output [B,Sq,C]
    Sq = x.shape[1]
    Skv = c.shape[1]
    q = x @ Wq.T                 # [B,Sq,C]
    k = c @ Wk.T                 # [B,Skv,C]
    v = c @ Wv.T                 # [B,Skv,C]
    q = q.reshape(B, Sq, Hh, Dh).permute(0, 2, 1, 3)
    k = k.reshape(B, Skv, Hh, Dh).permute(0, 2, 1, 3)
    v = v.reshape(B, Skv, Hh, Dh).permute(0, 2, 1, 3)
    o = sdpa(q, k, v)            # [B,Hh,Sq,Dh]
    o = o.permute(0, 2, 1, 3).reshape(B, Sq, C)
    return o @ Wo.T + bo


def layer_norm(x, w, b):
    return F.layer_norm(x, (C,), w, b, LN_EPS)


def geglu(x, pw, pb):
    proj = x @ pw.T + pb         # [.,2*Cff]
    xp = proj[..., :Cff]
    gate = proj[..., Cff:]
    return xp * gelu_tanh(gate)


def main():
    torch.manual_seed(0)

    # ── inputs ──
    # x is NHWC [B,H,W,C] — the SAME memory layout the Mojo port fills, so the
    # `fill()` linear sequence matches element-for-element. We permute to NCHW
    # only inside torch GroupNorm.
    x = T(B * H * W * C, 7, 13, 6.0, (B, H, W, C), grad=True)       # NHWC leaf
    context = T(B * Nkv * Cctx, 5, 11, 5.0, (B, Nkv, Cctx), grad=True)

    # ── ST-level weights ──
    gn_w = T(C, 3, 9, 4.0, (C,), grad=True)
    gn_b = T(C, 2, 7, 3.0, (C,), grad=True)
    proj_in_w = T(C * C, 5, 13, 6.0, (C, C), 0.02, grad=True)
    proj_in_b = T(C, 4, 11, 5.0, (C,), grad=True)
    proj_out_w = T(C * C, 6, 17, 8.0, (C, C), 0.02, grad=True)
    proj_out_b = T(C, 3, 8, 3.0, (C,), grad=True)

    # ── per-block weights (DEPTH blocks; here 1) ──
    blocks = []
    for j in range(DEPTH):
        s = j + 1
        bw = dict(
            n1w=T(C, 3, 9, 4.0, (C,), grad=True),
            n1b=T(C, 2, 7, 3.0, (C,), grad=True),
            q1=T(C * C, 5 + s, 13, 6.0, (C, C), 0.02, grad=True),
            k1=T(C * C, 6 + s, 13, 6.0, (C, C), 0.02, grad=True),
            v1=T(C * C, 7 + s, 13, 6.0, (C, C), 0.02, grad=True),
            o1=T(C * C, 8 + s, 13, 6.0, (C, C), 0.02, grad=True),
            o1b=T(C, 4, 11, 5.0, (C,), grad=True),
            n2w=T(C, 3, 9, 4.0, (C,), grad=True),
            n2b=T(C, 2, 7, 3.0, (C,), grad=True),
            q2=T(C * C, 5 + s, 17, 8.0, (C, C), 0.02, grad=True),
            k2=T(C * Cctx, 6 + s, 17, 8.0, (C, Cctx), 0.02, grad=True),
            v2=T(C * Cctx, 7 + s, 17, 8.0, (C, Cctx), 0.02, grad=True),
            o2=T(C * C, 8 + s, 17, 8.0, (C, C), 0.02, grad=True),
            o2b=T(C, 4, 11, 5.0, (C,), grad=True),
            n3w=T(C, 3, 9, 4.0, (C,), grad=True),
            n3b=T(C, 2, 7, 3.0, (C,), grad=True),
            fpw=T(2 * Cff * C, 5 + s, 13, 6.0, (2 * Cff, C), 0.02, grad=True),
            fpb=T(2 * Cff, 4, 10, 5.0, (2 * Cff,), grad=True),
            fow=T(C * Cff, 6 + s, 13, 6.0, (C, Cff), 0.02, grad=True),
            fob=T(C, 3, 8, 3.0, (C,), grad=True),
        )
        blocks.append(bw)

    # ── forward ──
    residual = x                                       # NHWC [B,H,W,C]
    x_nchw = x.permute(0, 3, 1, 2)                     # NHWC -> NCHW for torch GN
    xn_nchw = F.group_norm(x_nchw, G, gn_w, gn_b, GN_EPS)
    # NCHW -> [B,N,C] tokens (== Mojo NHWC reshape [B,H,W,C]->[B,H*W,C])
    tok = xn_nchw.permute(0, 2, 3, 1).reshape(B, N, C)
    h = tok @ proj_in_w.T + proj_in_b                  # [B,N,C]
    for bw in blocks:
        # self-attn
        x1n = layer_norm(h, bw["n1w"], bw["n1b"])
        a1 = attn(x1n, x1n, bw["q1"], bw["k1"], bw["v1"], bw["o1"], bw["o1b"])
        h = h + a1
        # cross-attn
        x2n = layer_norm(h, bw["n2w"], bw["n2b"])
        a2 = attn(x2n, context, bw["q2"], bw["k2"], bw["v2"], bw["o2"], bw["o2b"])
        h = h + a2
        # FF
        x3n = layer_norm(h, bw["n3w"], bw["n3b"])
        ff = geglu(x3n, bw["fpw"], bw["fpb"]) @ bw["fow"].T + bw["fob"]
        h = h + ff
    po = h @ proj_out_w.T + proj_out_b                 # [B,N,C]
    po_nhwc = po.reshape(B, H, W, C)                   # tokens -> NHWC
    out = residual + po_nhwc                           # NHWC [B,H,W,C]

    # ── backward: seed dL/dout (NHWC, matches Mojo go) ──
    go = T(B * H * W * C, 2, 7, 3.0, (B, H, W, C))     # NHWC grad seed
    out.backward(go)

    def flat(t):
        return t.detach().reshape(-1).numpy().tolist()

    lines = []
    lines.append("out " + " ".join(f"{v:.8f}" for v in flat(out)))       # NHWC
    lines.append("d_x " + " ".join(f"{v:.8f}" for v in flat(x.grad)))    # NHWC
    lines.append("d_context " + " ".join(f"{v:.8f}" for v in flat(context.grad)))
    lines.append("d_gn_w " + " ".join(f"{v:.8f}" for v in flat(gn_w.grad)))
    lines.append("d_gn_b " + " ".join(f"{v:.8f}" for v in flat(gn_b.grad)))
    lines.append("d_proj_in_w " + " ".join(f"{v:.8f}" for v in flat(proj_in_w.grad)))
    lines.append("d_proj_in_b " + " ".join(f"{v:.8f}" for v in flat(proj_in_b.grad)))
    lines.append("d_proj_out_w " + " ".join(f"{v:.8f}" for v in flat(proj_out_w.grad)))
    lines.append("d_proj_out_b " + " ".join(f"{v:.8f}" for v in flat(proj_out_b.grad)))
    for j, bw in enumerate(blocks):
        p = f"b{j}_"
        lines.append(p + "d_n1w " + " ".join(f"{v:.8f}" for v in flat(bw["n1w"].grad)))
        lines.append(p + "d_n1b " + " ".join(f"{v:.8f}" for v in flat(bw["n1b"].grad)))
        lines.append(p + "d_q1 " + " ".join(f"{v:.8f}" for v in flat(bw["q1"].grad)))
        lines.append(p + "d_k1 " + " ".join(f"{v:.8f}" for v in flat(bw["k1"].grad)))
        lines.append(p + "d_v1 " + " ".join(f"{v:.8f}" for v in flat(bw["v1"].grad)))
        lines.append(p + "d_o1 " + " ".join(f"{v:.8f}" for v in flat(bw["o1"].grad)))
        lines.append(p + "d_o1b " + " ".join(f"{v:.8f}" for v in flat(bw["o1b"].grad)))
        lines.append(p + "d_n2w " + " ".join(f"{v:.8f}" for v in flat(bw["n2w"].grad)))
        lines.append(p + "d_n2b " + " ".join(f"{v:.8f}" for v in flat(bw["n2b"].grad)))
        lines.append(p + "d_q2 " + " ".join(f"{v:.8f}" for v in flat(bw["q2"].grad)))
        lines.append(p + "d_k2 " + " ".join(f"{v:.8f}" for v in flat(bw["k2"].grad)))
        lines.append(p + "d_v2 " + " ".join(f"{v:.8f}" for v in flat(bw["v2"].grad)))
        lines.append(p + "d_o2 " + " ".join(f"{v:.8f}" for v in flat(bw["o2"].grad)))
        lines.append(p + "d_o2b " + " ".join(f"{v:.8f}" for v in flat(bw["o2b"].grad)))
        lines.append(p + "d_n3w " + " ".join(f"{v:.8f}" for v in flat(bw["n3w"].grad)))
        lines.append(p + "d_n3b " + " ".join(f"{v:.8f}" for v in flat(bw["n3b"].grad)))
        lines.append(p + "d_fpw " + " ".join(f"{v:.8f}" for v in flat(bw["fpw"].grad)))
        lines.append(p + "d_fpb " + " ".join(f"{v:.8f}" for v in flat(bw["fpb"].grad)))
        lines.append(p + "d_fow " + " ".join(f"{v:.8f}" for v in flat(bw["fow"].grad)))
        lines.append(p + "d_fob " + " ".join(f"{v:.8f}" for v in flat(bw["fob"].grad)))

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print(f"dims: B={B} H={H} W={W} C={C} N={N} Nkv={Nkv} Cctx={Cctx} Hh={Hh} Dh={Dh} Cff={Cff} depth={DEPTH}")


if __name__ == "__main__":
    main()
