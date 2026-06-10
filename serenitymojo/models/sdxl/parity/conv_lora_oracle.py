#!/usr/bin/env python3
# serenitymojo/models/sdxl/parity/conv_lora_oracle.py
#
# Torch-autograd oracle for the SDXL conv-LoRA (LyCORIS LoCon) adapter
# (training/locon_conv_adapter.mojo). This gates the MATH of the conv-LoRA family
# that OneTrainer trains on the SDXL UNet:
#   lora_unet_conv_in / conv_out, *_resnets_*_conv1 / conv2 / conv_shortcut,
#   *_downsamplers_*_conv, *_upsamplers_*_conv
# — every one is a Conv2d wrapped by the SAME LoCon down(spatial)->up(1x1)*scale
# decomposition, differing only in (Kh,Kw,stride,pad,Cin,Cout). So gating a
# representative set of kernel/stride configs gates the whole conv family's
# d_down / d_up / d_x against torch's conv2d autograd (a DIFFERENT code path than
# the Mojo open-coded locon_backward, and than the existing finite-difference
# smoke).
#
# DELTA-ONLY graph: Δy = up(down(x))*scale (scale=alpha/rank). The frozen base
# conv contributes a separate, additive d_x that the integration layer (resblock)
# sums; here we isolate and gate the LoCon DELTA path exactly as locon_backward
# returns it (d_y given at the delta output).
#
# Layouts (match locon_conv_adapter.mojo EXACTLY; .bin = flat host f32):
#   x    NHWC  [N,Hi,Wi,Cin]
#   down RSCF  [Kh,Kw,Cin,rank]
#   up   RSCF  [1,1,rank,Cout]
#   go   NHWC  [N,Ho,Wo,Cout]     (d_y at the delta output)
# Torch uses NCHW / OIHW; we permute in and permute grads back to the Mojo layout.
#
# Run (SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/sdxl/parity/conv_lora_oracle.py

import os
import struct
import torch
import torch.nn.functional as F

DT = torch.float64
REF = os.path.dirname(os.path.abspath(__file__))

ALPHA = 4.0   # scale = alpha / rank


def fill(n, a, b, c, scale=0.05):
    return torch.tensor([(float((i * a) % b) - c) * scale for i in range(n)], dtype=DT)


def WB(name, flat_list):
    arr = [float(v) for v in flat_list]
    with open(os.path.join(REF, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % len(arr), *arr))


def out_hw(Hi, Wi, Kh, Kw, s, p):
    Ho = (Hi + 2 * p - Kh) // s + 1
    Wo = (Wi + 2 * p - Kw) // s + 1
    return Ho, Wo


def run(tag, N, Hi, Wi, Cin, Cout, Kh, Kw, rank, s, p):
    scale = ALPHA / rank
    Ho, Wo = out_hw(Hi, Wi, Kh, Kw, s, p)

    # ── flat host tensors in the Mojo layout ──
    x_nhwc = fill(N * Hi * Wi * Cin, 7, 13, 6.0, 0.05)          # [N,Hi,Wi,Cin]
    down_rscf = fill(Kh * Kw * Cin * rank, 5, 11, 5.0, 0.07)    # [Kh,Kw,Cin,rank]
    up_rscf = fill(rank * Cout, 3, 7, 3.0, 0.09)                # [1,1,rank,Cout]
    go_nhwc = fill(N * Ho * Wo * Cout, 2, 9, 4.0, 0.05)         # [N,Ho,Wo,Cout]

    # ── build torch NCHW/OIHW views ──
    x = x_nhwc.reshape(N, Hi, Wi, Cin).permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    down = down_rscf.reshape(Kh, Kw, Cin, rank).permute(3, 2, 0, 1).contiguous().requires_grad_(True)  # [rank,Cin,Kh,Kw]
    up = up_rscf.reshape(rank, Cout).t().reshape(Cout, rank, 1, 1).contiguous().requires_grad_(True)    # [Cout,rank,1,1]
    go = go_nhwc.reshape(N, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous()                              # [N,Cout,Ho,Wo]

    # ── delta-only forward + backward ──
    h = F.conv2d(x, down, stride=s, padding=p)        # [N,rank,Ho,Wo]
    delta = F.conv2d(h, up, stride=1, padding=0) * scale   # [N,Cout,Ho,Wo]
    loss = (delta * go).sum()
    loss.backward()

    # ── forward Δy back to NHWC ──
    dy_nhwc = delta.detach().permute(0, 2, 3, 1).contiguous().reshape(-1)
    # ── grads back to Mojo layout ──
    d_x_nhwc = x.grad.permute(0, 2, 3, 1).contiguous().reshape(-1)                   # [N,Hi,Wi,Cin]
    d_down_rscf = down.grad.permute(2, 3, 1, 0).contiguous().reshape(-1)             # [Kh,Kw,Cin,rank]
    d_up_rscf = up.grad.reshape(Cout, rank).t().contiguous().reshape(-1)             # [1,1,rank,Cout]

    WB("conv_%s_x" % tag, x_nhwc)
    WB("conv_%s_down" % tag, down_rscf)
    WB("conv_%s_up" % tag, up_rscf)
    WB("conv_%s_go" % tag, go_nhwc)
    WB("conv_%s_ydelta" % tag, dy_nhwc)
    WB("conv_%s_d_x" % tag, d_x_nhwc)
    WB("conv_%s_d_down" % tag, d_down_rscf)
    WB("conv_%s_d_up" % tag, d_up_rscf)
    print("[%s] N=%d HxW=%dx%d Cin=%d Cout=%d K=%dx%d rank=%d s=%d p=%d -> HoxWo=%dx%d scale=%.3f loss=%.6f"
          % (tag, N, Hi, Wi, Cin, Cout, Kh, Kw, rank, s, p, Ho, Wo, scale, float(loss)))


def main():
    # k3s1p1: resnet conv1/conv2, conv_in, conv_out, upsampler conv (same-size 3x3).
    run("k3s1p1", 2, 5, 4, 3, 4, 3, 3, 2, 1, 1)
    # k3s2p1: downsampler conv (strided 3x3).
    run("k3s2p1", 1, 6, 6, 2, 3, 3, 3, 2, 2, 1)
    # k1s1p0: resnet conv_shortcut (1x1).
    run("k1s1p0", 1, 4, 4, 3, 5, 1, 1, 2, 1, 0)
    print("wrote conv-LoRA oracle refs to", REF)


if __name__ == "__main__":
    main()
