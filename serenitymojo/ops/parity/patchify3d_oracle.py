#!/usr/bin/env python
# patchify3d_oracle.py — torch GPU bf16 oracle for the 3D (video) DiT patch embed.
#
# Proves & cross-checks the rank-5 audit claim: the video-DiT "Conv3d patch embed"
# (wan22_dit.rs:454, cosmos_predict25_dit.rs) with kernel == stride == patch_size
# is mathematically IDENTICAL to `patchify3d (unfold) + linear`. We:
#   1. build a real torch.nn.Conv3d(C, OUT, k=(pf,ph,pw), stride=(pf,ph,pw)),
#   2. run it on a [1,C,F,H,W] input in bf16 on CUDA  -> conv tokens,
#   3. ALSO compute the unfold+linear form (flatten weight [OUT,C,pf,ph,pw] ->
#      [OUT, C*pf*ph*pw] with NO transpose; unfold input with the (c,pf,ph,pw)
#      c-slowest within-patch order, F-major token order) and assert it equals
#      the conv output  -> EQUIVALENCE PROOF, printed.
#
# The Mojo parity probe (patchify3d_parity.mojo) reads the SAME bytes and runs
# `patchify3d` + `ops/linear.linear`, gating cos >= 0.999 (bf16) against the conv
# reference dumped here. The conv output is the "ground truth" both the torch
# unfold+linear AND the Mojo path must match.
#
# Dumps (all little-endian f32):
#   patchify3d_x.bin       input            [C*F*H*W]      (C-major, no batch)
#   patchify3d_w.bin       conv weight      [OUT*C*pf*ph*pw]  (== flat linear W)
#   patchify3d_b.bin       conv bias        [OUT]
#   patchify3d_ref.bin     conv tokens      [n_patches*OUT]  (bf16-rounded -> f32)
#   patchify3d_unfold.bin  unfold tensor    [n_patches*(C*pf*ph*pw)] (for unpatch test)
#
# Run with the serenityflow venv python:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/ops/parity/patchify3d_oracle.py

import os
import numpy as np
import torch

DIR = os.path.dirname(os.path.abspath(__file__))

# Geometry: a small but non-degenerate video latent. wan22 patch (1,2,2).
C, F, H, W = 16, 4, 8, 8
PF, PH, PW = 1, 2, 2
OUT = 64                      # patch-embed output dim (kept modest for the probe)
FO, HO, WO = F // PF, H // PH, W // PW
N_PATCHES = FO * HO * WO
PATCH_DIM = C * PF * PH * PW


def dump(name, arr):
    arr.astype("<f4").tofile(os.path.join(DIR, name))


def main():
    assert torch.cuda.is_available(), "CUDA required for the bf16 GPU oracle"
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(1234)

    # f32 master inputs (dumped so the Mojo side is fed byte-identical f32).
    x = torch.randn(1, C, F, H, W, generator=g, dtype=torch.float32)
    conv = torch.nn.Conv3d(C, OUT, kernel_size=(PF, PH, PW),
                           stride=(PF, PH, PW), bias=True)
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=g))
        conv.bias.copy_(torch.randn(conv.bias.shape, generator=g))
    w = conv.weight.detach().clone()        # [OUT, C, PF, PH, PW]
    b = conv.bias.detach().clone()          # [OUT]

    # ── (1) Conv3d reference, bf16 on CUDA ──
    x_bf = x.to(dev).to(torch.bfloat16)
    conv_bf = conv.to(dev).to(torch.bfloat16)
    with torch.no_grad():
        y_conv = conv_bf(x_bf)              # [1, OUT, FO, HO, WO] bf16
    # to token layout [n_patches, OUT], F-major token order (t,h,w) -> matches
    # patchify token order. permute [B,OUT,FO,HO,WO] -> [B,FO,HO,WO,OUT] then flatten.
    y_conv_tok = (
        y_conv.permute(0, 2, 3, 4, 1).reshape(N_PATCHES, OUT).to(torch.float32).cpu()
    )

    # ── (2) Equivalence: unfold + linear, c-slowest within-patch, F-major tokens ──
    # Build the unfold tensor in f32 EXACTLY as patchify3d does, then matmul with
    # the conv weight flattened [OUT, C*PF*PH*PW] (no transpose).
    xf = x[0]                               # [C, F, H, W]
    unfold = np.zeros((N_PATCHES, PATCH_DIM), dtype=np.float32)
    for fi in range(FO):
        for hi in range(HO):
            for wi in range(WO):
                patch = fi * HO * WO + hi * WO + wi
                for ci in range(C):
                    for pfi in range(PF):
                        for phi in range(PH):
                            for pwi in range(PW):
                                dst = ((ci * PF + pfi) * PH + phi) * PW + pwi
                                sf, sh, sw = fi*PF+pfi, hi*PH+phi, wi*PW+pwi
                                unfold[patch, dst] = float(xf[ci, sf, sh, sw])
    w_flat = w.reshape(OUT, PATCH_DIM).to(torch.float32)            # [OUT, PATCH_DIM]
    # f32 unfold+linear (the algebraic identity; exact up to fp rounding)
    y_unf_f32 = (torch.from_numpy(unfold) @ w_flat.t() + b.to(torch.float32))
    # bf16 unfold+linear on CUDA (what the Mojo bf16 path approximates)
    y_unf_bf = (
        torch.from_numpy(unfold).to(dev).to(torch.bfloat16)
        @ w.reshape(OUT, PATCH_DIM).to(dev).to(torch.bfloat16).t()
        + b.to(dev).to(torch.bfloat16)
    ).to(torch.float32).cpu()

    # Cosine conv-vs-unfold(f32): if ~1.0, conv3d(stride=kernel) == unfold+linear.
    def cos(a, bb):
        a = a.reshape(-1).to(torch.float64); bb = bb.reshape(-1).to(torch.float64)
        return float((a @ bb) / (a.norm() * bb.norm() + 1e-30))
    # Compare the f32 conv (pristine f32 conv from the saved weights — `conv` was
    # mutated to bf16/CUDA above, so rebuild from `w`/`b` to avoid contamination):
    conv_f32 = torch.nn.Conv3d(C, OUT, kernel_size=(PF, PH, PW),
                               stride=(PF, PH, PW), bias=True)
    with torch.no_grad():
        conv_f32.weight.copy_(w)
        conv_f32.bias.copy_(b)
        y_conv_f32 = (
            conv_f32(x).permute(0, 2, 3, 4, 1).reshape(N_PATCHES, OUT)
        )
    c_eq = cos(y_conv_f32, y_unf_f32)
    max_eq = float((y_conv_f32 - y_unf_f32).abs().max())
    print(f"# EQUIVALENCE conv3d(stride=k) vs unfold+linear (f32):"
          f" cos={c_eq:.10f}  max_abs={max_eq:.3e}")
    print(f"#   -> {'PROVEN' if c_eq >= 0.99999 and max_eq < 1e-3 else 'MISMATCH'}")

    # bf16 conv-vs-unfold for context (both bf16 GPU paths):
    print(f"# bf16 conv vs bf16 unfold+linear: cos={cos(y_conv_tok, y_unf_bf):.8f}")

    # ── dumps ──
    dump("patchify3d_x.bin", x.reshape(-1).numpy())
    dump("patchify3d_w.bin", w.reshape(-1).numpy())
    dump("patchify3d_b.bin", b.reshape(-1).numpy())
    # Reference for the Mojo gate = the bf16 CONV output in token layout.
    dump("patchify3d_ref.bin", y_conv_tok.reshape(-1).numpy())
    dump("patchify3d_unfold.bin", unfold.reshape(-1))
    print(f"# dumped: C={C} F={F} H={H} W={W} patch=({PF},{PH},{PW}) OUT={OUT}"
          f" n_patches={N_PATCHES} patch_dim={PATCH_DIM}")


if __name__ == "__main__":
    main()
