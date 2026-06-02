#!/usr/bin/env python3
# resblock_oracle.py — PyTorch autograd reference for the SDXL ResBlock
# forward + backward (serenitymojo/models/sdxl/block.mojo).
#
# Oracle = PyTorch autograd (F64 interior for a stable reference). Python is
# DEV-ONLY (parity convention). Writes tagged space-separated float lines to
# resblock_ref.txt; the Mojo gate reproduces inputs deterministically with the
# SAME fills and reads only the forward OUTPUT + the GRADIENTS.
#
# ── ResBlock structure (verified vs inference-flame sdxl_unet.rs::resblock) ──
#   h1 = GroupNorm(x,  gn1_w, gn1_b, G, eps)      # in_layers.0
#   s1 = SiLU(h1)
#   c1 = Conv3x3(s1, conv1, pad=1) + conv1_b      # in_layers.2
#   e  = SiLU(emb)
#   el = Linear(e, emb_w, emb_b)                  # emb_layers.1  -> [N,Cout]
#   h2 = c1 + el[:, :, None, None]                # per-channel time-emb add
#   h3 = GroupNorm(h2, gn2_w, gn2_b, G, eps)      # out_layers.0
#   s2 = SiLU(h3)
#   c2 = Conv3x3(s2, conv2, pad=1) + conv2_b      # out_layers.3
#   r  = Conv1x1(x, skip, pad=0) + skip_b   (Cin != Cout)
#   out = r + c2
#
# IMPORTANT LAYOUT: torch is NCHW; the Mojo gate is NHWC. The oracle works in
# NCHW (native torch conv), then permutes the OUTPUT and the d_x / NHWC-shaped
# grads back to NHWC so the Mojo side compares in its own layout. Conv weight
# grads are emitted in RSCF [Kh,Kw,Cin,Cout] (the Mojo filter layout); torch
# stores OIHW [Cout,Cin,Kh,Kw], so we permute (2,3,1,0).
#
# eps = 1e-5 (ResBlock GroupNorm; matches GN_EPS_RES on the Mojo side).
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/models/sdxl/parity/resblock_oracle.py

import os
import numpy as np
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "resblock_ref.txt")

# ── shape (MUST match resblock_parity.mojo) ─────────────────────────────────
N, Hi, Wi = 2, 8, 8
CIN, COUT = 64, 128       # Cin != Cout -> exercises the 1x1 skip conv
EEMB = 256                # time-embed input dim
G = 32                    # GroupNorm groups (divides 64 and 128)
EPS = 1e-5

torch.set_default_dtype(torch.float64)


# ── deterministic fills (MUST match resblock_parity.mojo _fill_*) ───────────
def fill(n, a, b, c):
    return np.array([(float((i * a) % b) - c) * 0.05 for i in range(n)], np.float64)


def fill_x(n):    return fill(n, 7, 13, 6.0)
def fill_emb(n):  return fill(n, 2, 7, 3.0)
def fill_gnw(n):  return np.array([(float((i * 5) % 11) - 5.0) * 0.05 + 1.0 for i in range(n)], np.float64)
def fill_gnb(n):  return fill(n, 3, 9, 4.0)
def fill_conv(n): return fill(n, 5, 11, 5.0)
def fill_convb(n):return fill(n, 4, 10, 5.0)
def fill_embw(n): return fill(n, 6, 17, 8.0)
def fill_embb(n): return fill(n, 3, 9, 4.0)
def fill_go(n):   return fill(n, 2, 7, 3.0)


def t(np_arr, shape, req=False):
    return torch.tensor(np_arr.reshape(shape), requires_grad=req)


def main():
    # ── inputs (NCHW for torch) ──
    # Mojo builds x in NHWC [N,Hi,Wi,Cin]; the fill is a flat index sequence.
    # To make NCHW torch see the SAME values per (n,c,h,w), we build NHWC then
    # permute to NCHW.
    x_nhwc = t(fill_x(N * Hi * Wi * CIN), (N, Hi, Wi, CIN), req=False)
    x = x_nhwc.permute(0, 3, 1, 2).contiguous().requires_grad_(True)  # NCHW

    emb = t(fill_emb(N * EEMB), (N, EEMB), req=False).requires_grad_(True)

    gn1_w = t(fill_gnw(CIN), (CIN,), req=True)
    gn1_b = t(fill_gnb(CIN), (CIN,), req=True)
    # conv1 OIHW [Cout, Cin, 3, 3] — but Mojo fills RSCF [3,3,Cin,Cout] flat.
    # Build RSCF then permute to OIHW so torch & Mojo see identical values.
    conv1_rscf = t(fill_conv(3 * 3 * CIN * COUT), (3, 3, CIN, COUT))
    conv1 = conv1_rscf.permute(3, 2, 0, 1).contiguous().requires_grad_(True)  # OIHW
    conv1_b = t(fill_convb(COUT), (COUT,), req=True)

    emb_w = t(fill_embw(COUT * EEMB), (COUT, EEMB), req=True)
    emb_b = t(fill_embb(COUT), (COUT,), req=True)

    gn2_w = t(fill_gnw(COUT), (COUT,), req=True)
    gn2_b = t(fill_gnb(COUT), (COUT,), req=True)
    conv2_rscf = t(fill_conv(3 * 3 * COUT * COUT), (3, 3, COUT, COUT))
    conv2 = conv2_rscf.permute(3, 2, 0, 1).contiguous().requires_grad_(True)
    conv2_b = t(fill_convb(COUT), (COUT,), req=True)

    skip_rscf = t(fill_conv(1 * 1 * CIN * COUT), (1, 1, CIN, COUT))
    skip = skip_rscf.permute(3, 2, 0, 1).contiguous().requires_grad_(True)  # [Cout,Cin,1,1]
    skip_b = t(fill_convb(COUT), (COUT,), req=True)

    # ── forward (NCHW) ──
    h1 = F.group_norm(x, G, gn1_w, gn1_b, EPS)
    s1 = F.silu(h1)
    c1 = F.conv2d(s1, conv1, conv1_b, stride=1, padding=1)
    e = F.silu(emb)
    el = F.linear(e, emb_w, emb_b)               # [N, Cout]
    h2 = c1 + el[:, :, None, None]
    h3 = F.group_norm(h2, G, gn2_w, gn2_b, EPS)
    s2 = F.silu(h3)
    c2 = F.conv2d(s2, conv2, conv2_b, stride=1, padding=1)
    r = F.conv2d(x, skip, skip_b, stride=1, padding=0)
    out = r + c2                                  # [N, Cout, Hi, Wi]

    # ── upstream grad go (NHWC flat -> NCHW) ──
    go_nhwc = t(fill_go(N * Hi * Wi * COUT), (N, Hi, Wi, COUT))
    go = go_nhwc.permute(0, 3, 1, 2).contiguous()

    out.backward(go)

    # ── collect grads, permute conv grads OIHW->RSCF, x/out NCHW->NHWC ──
    def nhwc(tt):  # NCHW tensor -> NHWC flat list
        return tt.permute(0, 2, 3, 1).contiguous().reshape(-1).detach().numpy()

    def rscf(g):   # OIHW grad -> RSCF flat list
        return g.permute(2, 3, 1, 0).contiguous().reshape(-1).detach().numpy()

    rows = {
        "out":       nhwc(out),
        "d_x":       nhwc(x.grad),
        "d_emb_in":  emb.grad.reshape(-1).detach().numpy(),
        "d_gn1_w":   gn1_w.grad.reshape(-1).detach().numpy(),
        "d_gn1_b":   gn1_b.grad.reshape(-1).detach().numpy(),
        "d_conv1_w": rscf(conv1.grad),
        "d_conv1_b": conv1_b.grad.reshape(-1).detach().numpy(),
        "d_emb_w":   emb_w.grad.reshape(-1).detach().numpy(),
        "d_emb_b":   emb_b.grad.reshape(-1).detach().numpy(),
        "d_gn2_w":   gn2_w.grad.reshape(-1).detach().numpy(),
        "d_gn2_b":   gn2_b.grad.reshape(-1).detach().numpy(),
        "d_conv2_w": rscf(conv2.grad),
        "d_conv2_b": conv2_b.grad.reshape(-1).detach().numpy(),
        "d_skip_w":  rscf(skip.grad),
        "d_skip_b":  skip_b.grad.reshape(-1).detach().numpy(),
    }

    with open(OUT, "w") as f:
        for tag, arr in rows.items():
            f.write(tag + " " + " ".join(repr(float(v)) for v in arr) + "\n")

    print("wrote", OUT)
    print("shapes: out", out.shape, "d_x", x.grad.shape,
          "d_conv1_w(RSCF)", (3, 3, CIN, COUT), "d_emb_w", emb_w.grad.shape)


if __name__ == "__main__":
    main()
