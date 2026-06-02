#!/usr/bin/env python3
# unet_stack_oracle.py — torch-autograd reference for the SDXL FULL conv-UNet
# stack fwd+bwd (models/sdxl/sdxl_unet_stack.mojo), at a REDUCED but
# STRUCTURALLY-COMPLETE config so the full encoder->skip->decoder topology is
# exercised (conv_in, 3 levels w/ downsample, skip push/pop stack, middle
# Res+ST+Res, output concat+Res(+ST)(+Up), final GN->SiLU->conv_out).
#
# Built INDEPENDENTLY from inference-flame/src/models/sdxl_unet.rs (forward
# 851-995, build_block_descriptors 318-373) — torch primitives + a single
# out.backward(go), NOT transcribed from the Mojo hand-chained backward.
#
# REDUCED config (mc=16, channel_mult=(1,2,4), num_res_blocks=1, attn depth 1 at
# the two deeper levels, tiny H=W=8 latent, Nkv=7 context tokens). Topology
# (from build_block_descriptors with these params):
#   input blocks:  0 ConvIn(4->16)
#                  1 Res(16->16, td0)
#                  2 Down(16)
#                  3 Res(16->32, td1 +ST)
#                  4 Down(32)
#                  5 Res(32->64, td1 +ST)
#   skip stack (pushed in order): [16,16,16,32,32,64]
#   middle: Res(64->64) + ST(td1) + Res(64->64)
#   output blocks: 0 OutRes(64+64=128 -> 64, td1 +ST)
#                  1 OutRes(64+32=96  -> 64, td1 +ST +UP)
#                  2 OutRes(64+16=80  -> 32, td0)        [wait: see note]
#   NOTE: build_block_descriptors gives output in_ch = carry + popped skip.
#   For nrb=1 the popped skips are [64,32,16,32,16,16] (reverse of the push
#   stack with the level interleave) -> in_ch = [128,96,96,48,48,32], out_ch =
#   [64,64,32,32,16,16], UP on blocks 1 and 3. This script computes them the
#   SAME way the Rust oracle does (verified by the build_block_descriptors port).
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python .../unet_stack_oracle.py
#   (pass "real" as argv[1] to print the real-config topology for reference)

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "unet_stack_ref.txt")

# ── REDUCED config ──
MC = 16
CHANNEL_MULT = [1, 2, 4]
NRB = 1
IN_CH = 4
OUT_CH = 4
TEMB = 32           # time/label embedding dim (reduced from 1280)
SDIM = 16           # sinusoidal dim (= model_channels analogue; reduced)
ADM = 24            # ADM vector dim (reduced from 2816)
NKV = 7             # context tokens (reduced from 77)
CCTX = 16           # context dim (reduced from 2048)
HEAD_DIM = 8        # so num_heads = C/8
B = 1
H0 = 8              # latent H,W
W0 = 8
G = 16             # group-norm groups (must divide every channel width: 16,32,64)
GN_EPS_RES = 1e-5
GN_EPS_ST = 1e-6
LN_EPS = 1e-5
TD_INPUT = [0, 1, 1]    # per input res-block (3 of them): td for in1,in3,in5
TD_OUTPUT = [1, 1, 0]   # per output res-block group head; expanded below
TD_MIDDLE = 1


def build_descriptors(mc, channel_mult, num_res_blocks, td_input, td_output):
    """Port of sdxl_unet.rs::build_block_descriptors (318-373)."""
    input_descs = []
    input_channels = []
    input_descs.append(("ConvIn", None, mc, 0))
    input_channels.append(mc)
    ti = list(reversed(td_input))
    ch = mc
    for level, mult in enumerate(channel_mult):
        out_ch = mc * mult
        for _ in range(num_res_blocks):
            nt = ti.pop() if ti else 0
            input_descs.append(("Res", ch, out_ch, nt))
            ch = out_ch
            input_channels.append(ch)
        if level < len(channel_mult) - 1:
            input_descs.append(("Down", ch, ch, 0))
            input_channels.append(ch)
    output_descs = []
    to = list(reversed(td_output))
    ic = list(input_channels)
    num_levels = len(channel_mult)
    ch = mc * channel_mult[num_levels - 1]
    for level in reversed(range(num_levels)):
        out_ch = mc * channel_mult[level]
        for i in range(num_res_blocks + 1):
            skip = ic.pop() if ic else 0
            in_ch = ch + skip
            nt = to.pop() if to else 0
            has_up = level > 0 and i == num_res_blocks
            output_descs.append(("OutRes", in_ch, out_ch, nt, has_up, skip))
            ch = out_ch
    return input_descs, input_channels, output_descs


def fill(n, a, b, c, scale=0.05):
    out = np.empty(n, np.float64)
    for i in range(n):
        out[i] = (float((i * a) % b) - c) * scale
    return out


_PARAMS = {}


def P(name, n, a, b, c, shape, scale=0.05):
    t = torch.tensor(fill(n, a, b, c, scale), dtype=torch.float64).reshape(shape)
    t = t.requires_grad_(True)
    _PARAMS[name] = t
    return t


def numel(shape):
    r = 1
    for s in shape:
        r *= s
    return r


# ── primitive ops (NHWC math; torch GN/conv want NCHW so we permute inside) ──
def gn(x_nhwc, w, b, eps):
    # x_nhwc [B,H,W,C]
    xnchw = x_nhwc.permute(0, 3, 1, 2)
    y = F.group_norm(xnchw, G, w, b, eps)
    return y.permute(0, 2, 3, 1)


def conv3x3(x_nhwc, w_rscf, b, stride=1, pad=1):
    # w_rscf [Kh,Kw,Cin,Cout] -> torch OIHW [Cout,Cin,Kh,Kw]
    w_oihw = w_rscf.permute(3, 2, 0, 1)
    xnchw = x_nhwc.permute(0, 3, 1, 2)
    y = F.conv2d(xnchw, w_oihw, b, stride=stride, padding=pad)
    return y.permute(0, 2, 3, 1)


def conv1x1(x_nhwc, w_rscf, b):
    w_oihw = w_rscf.permute(3, 2, 0, 1)
    xnchw = x_nhwc.permute(0, 3, 1, 2)
    y = F.conv2d(xnchw, w_oihw, b, stride=1, padding=0)
    return y.permute(0, 2, 3, 1)


def silu(x):
    return F.silu(x)


def resblock(x, emb, pfx, cin, cout):
    # in_layers: GN -> SiLU -> conv3x3
    h = gn(x, _PARAMS[pfx + "gn1_w"], _PARAMS[pfx + "gn1_b"], GN_EPS_RES)
    h = silu(h)
    h = conv3x3(h, _PARAMS[pfx + "conv1_w"], _PARAMS[pfx + "conv1_b"])
    # emb_layers: SiLU -> Linear -> bcast add
    e = silu(emb)
    el = e @ _PARAMS[pfx + "emb_w"].T + _PARAMS[pfx + "emb_b"]   # [B,cout]
    h = h + el.reshape(B, 1, 1, cout)
    # out_layers: GN -> SiLU -> conv3x3
    h = gn(h, _PARAMS[pfx + "gn2_w"], _PARAMS[pfx + "gn2_b"], GN_EPS_RES)
    h = silu(h)
    h = conv3x3(h, _PARAMS[pfx + "conv2_w"], _PARAMS[pfx + "conv2_b"])
    # skip
    if cin != cout:
        r = conv1x1(x, _PARAMS[pfx + "skip_w"], _PARAMS[pfx + "skip_b"])
    else:
        r = x
    return r + h


def layer_norm(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b, LN_EPS)


def sdpa(q, k, v, dh):
    scale = 1.0 / np.sqrt(dh)
    scores = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def attn(x, c, Wq, Wk, Wv, Wo, bo, C, dh):
    Sq = x.shape[1]
    Skv = c.shape[1]
    hh = C // dh
    q = (x @ Wq.T).reshape(B, Sq, hh, dh).permute(0, 2, 1, 3)
    k = (c @ Wk.T).reshape(B, Skv, hh, dh).permute(0, 2, 1, 3)
    v = (c @ Wv.T).reshape(B, Skv, hh, dh).permute(0, 2, 1, 3)
    o = sdpa(q, k, v, dh).permute(0, 2, 1, 3).reshape(B, Sq, C)
    return o @ Wo.T + bo


def geglu(x, pw, pb):
    proj = x @ pw.T + pb
    cff = proj.shape[-1] // 2
    return proj[..., :cff] * F.gelu(proj[..., cff:], approximate="tanh")


def spatial_transformer(x, context, pfx, C, depth):
    # x NHWC [B,H,W,C]
    Bb, Hh, Ww, Cc = x.shape
    Nn = Hh * Ww
    residual = x
    xn = gn(x, _PARAMS[pfx + "gn_w"], _PARAMS[pfx + "gn_b"], GN_EPS_ST)
    tok = xn.reshape(B, Nn, C)
    h = tok @ _PARAMS[pfx + "proj_in_w"].T + _PARAMS[pfx + "proj_in_b"]
    dh = HEAD_DIM
    for j in range(depth):
        bp = pfx + f"b{j}_"
        x1n = layer_norm(h, _PARAMS[bp + "n1w"], _PARAMS[bp + "n1b"])
        a1 = attn(x1n, x1n, _PARAMS[bp + "q1"], _PARAMS[bp + "k1"], _PARAMS[bp + "v1"],
                  _PARAMS[bp + "o1"], _PARAMS[bp + "o1b"], C, dh)
        h = h + a1
        x2n = layer_norm(h, _PARAMS[bp + "n2w"], _PARAMS[bp + "n2b"])
        a2 = attn(x2n, context, _PARAMS[bp + "q2"], _PARAMS[bp + "k2"], _PARAMS[bp + "v2"],
                  _PARAMS[bp + "o2"], _PARAMS[bp + "o2b"], C, dh)
        h = h + a2
        x3n = layer_norm(h, _PARAMS[bp + "n3w"], _PARAMS[bp + "n3b"])
        ff = geglu(x3n, _PARAMS[bp + "fpw"], _PARAMS[bp + "fpb"]) @ _PARAMS[bp + "fow"].T + _PARAMS[bp + "fob"]
        h = h + ff
    po = h @ _PARAMS[pfx + "proj_out_w"].T + _PARAMS[pfx + "proj_out_b"]
    return residual + po.reshape(B, Hh, Ww, C)


def downsample(x, pfx, c):
    return conv3x3(x, _PARAMS[pfx + "op_w"], _PARAMS[pfx + "op_b"], stride=2, pad=1)


def upsample(x, pfx, c):
    # nearest 2x then conv3x3 stride1 pad1
    Bb, Hh, Ww, Cc = x.shape
    xnchw = x.permute(0, 3, 1, 2)
    up = F.interpolate(xnchw, scale_factor=2, mode="nearest").permute(0, 2, 3, 1)
    return conv3x3(up, _PARAMS[pfx + "conv_w"], _PARAMS[pfx + "conv_b"])


# ── allocate all weights ──
def alloc_resblock(pfx, cin, cout, ff):
    s = ff
    P(pfx + "gn1_w", cin, 5, 11, 5.0, (cin,), 0.05)
    P(pfx + "gn1_b", cin, 3, 9, 4.0, (cin,), 0.05)
    P(pfx + "conv1_w", 3 * 3 * cin * cout, 5 + s, 11, 5.0, (3, 3, cin, cout), 0.02)
    P(pfx + "conv1_b", cout, 4, 10, 5.0, (cout,), 0.05)
    P(pfx + "emb_w", cout * TEMB, 6 + s, 17, 8.0, (cout, TEMB), 0.02)
    P(pfx + "emb_b", cout, 3, 9, 4.0, (cout,), 0.05)
    P(pfx + "gn2_w", cout, 5, 11, 5.0, (cout,), 0.05)
    P(pfx + "gn2_b", cout, 3, 9, 4.0, (cout,), 0.05)
    P(pfx + "conv2_w", 3 * 3 * cout * cout, 6 + s, 11, 5.0, (3, 3, cout, cout), 0.02)
    P(pfx + "conv2_b", cout, 4, 10, 5.0, (cout,), 0.05)
    if cin != cout:
        P(pfx + "skip_w", 1 * 1 * cin * cout, 7 + s, 11, 5.0, (1, 1, cin, cout), 0.02)
        P(pfx + "skip_b", cout, 4, 10, 5.0, (cout,), 0.05)


def alloc_st(pfx, C, depth, ff):
    P(pfx + "gn_w", C, 3, 9, 4.0, (C,), 0.05)
    P(pfx + "gn_b", C, 2, 7, 3.0, (C,), 0.05)
    P(pfx + "proj_in_w", C * C, 5 + ff, 13, 6.0, (C, C), 0.02)
    P(pfx + "proj_in_b", C, 4, 11, 5.0, (C,), 0.05)
    P(pfx + "proj_out_w", C * C, 6 + ff, 17, 8.0, (C, C), 0.02)
    P(pfx + "proj_out_b", C, 3, 8, 3.0, (C,), 0.05)
    Cff = 2 * C
    for j in range(depth):
        bp = pfx + f"b{j}_"
        s = j + 1 + ff
        P(bp + "n1w", C, 3, 9, 4.0, (C,), 0.05); P(bp + "n1b", C, 2, 7, 3.0, (C,), 0.05)
        P(bp + "q1", C * C, 5 + s, 13, 6.0, (C, C), 0.02)
        P(bp + "k1", C * C, 6 + s, 13, 6.0, (C, C), 0.02)
        P(bp + "v1", C * C, 7 + s, 13, 6.0, (C, C), 0.02)
        P(bp + "o1", C * C, 8 + s, 13, 6.0, (C, C), 0.02)
        P(bp + "o1b", C, 4, 11, 5.0, (C,), 0.05)
        P(bp + "n2w", C, 3, 9, 4.0, (C,), 0.05); P(bp + "n2b", C, 2, 7, 3.0, (C,), 0.05)
        P(bp + "q2", C * C, 5 + s, 17, 8.0, (C, C), 0.02)
        P(bp + "k2", C * CCTX, 6 + s, 17, 8.0, (C, CCTX), 0.02)
        P(bp + "v2", C * CCTX, 7 + s, 17, 8.0, (C, CCTX), 0.02)
        P(bp + "o2", C * C, 8 + s, 17, 8.0, (C, C), 0.02)
        P(bp + "o2b", C, 4, 11, 5.0, (C,), 0.05)
        P(bp + "n3w", C, 3, 9, 4.0, (C,), 0.05); P(bp + "n3b", C, 2, 7, 3.0, (C,), 0.05)
        P(bp + "fpw", Cff * C, 5 + s, 13, 6.0, (Cff, C), 0.02)
        P(bp + "fpb", Cff, 4, 10, 5.0, (Cff,), 0.05)
        P(bp + "fow", C * (Cff // 2), 6 + s, 13, 6.0, (C, Cff // 2), 0.02)
        P(bp + "fob", C, 3, 8, 3.0, (C,), 0.05)


def main():
    torch.manual_seed(0)
    ind, inc, outd = build_descriptors(MC, CHANNEL_MULT, NRB, TD_INPUT, TD_OUTPUT)
    print("input blocks:", ind)
    print("skip stack push channels:", inc)
    print("output blocks:", outd)

    # inputs
    x = P("x", B * H0 * W0 * IN_CH, 7, 13, 6.0, (B, H0, W0, IN_CH), 0.05)
    context = P("context", B * NKV * CCTX, 5, 11, 5.0, (B, NKV, CCTX), 0.05)
    t = torch.tensor(fill(B, 3, 100, 0.0, 1.0), dtype=torch.float64).reshape(B)  # scalar timesteps (non-grad)
    y = P("y", B * ADM, 4, 13, 6.0, (B, ADM), 0.05)

    # embed weights
    P("t0_w", TEMB * SDIM, 5, 13, 6.0, (TEMB, SDIM), 0.02)
    P("t0_b", TEMB, 4, 11, 5.0, (TEMB,), 0.05)
    P("t2_w", TEMB * TEMB, 6, 17, 8.0, (TEMB, TEMB), 0.02)
    P("t2_b", TEMB, 3, 9, 4.0, (TEMB,), 0.05)
    P("l0_w", TEMB * ADM, 5, 13, 6.0, (TEMB, ADM), 0.02)
    P("l0_b", TEMB, 4, 11, 5.0, (TEMB,), 0.05)
    P("l2_w", TEMB * TEMB, 6, 17, 8.0, (TEMB, TEMB), 0.02)
    P("l2_b", TEMB, 3, 9, 4.0, (TEMB,), 0.05)

    # conv_in / conv_out / final GN
    P("conv_in_w", 3 * 3 * IN_CH * MC, 5, 11, 5.0, (3, 3, IN_CH, MC), 0.02)
    P("conv_in_b", MC, 4, 10, 5.0, (MC,), 0.05)
    P("out_gn_w", MC, 5, 11, 5.0, (MC,), 0.05)
    P("out_gn_b", MC, 3, 9, 4.0, (MC,), 0.05)
    P("conv_out_w", 3 * 3 * MC * OUT_CH, 6, 11, 5.0, (3, 3, MC, OUT_CH), 0.02)
    P("conv_out_b", OUT_CH, 4, 10, 5.0, (OUT_CH,), 0.05)

    # per-block weights (named by their stack position)
    # encoder res blocks: in1 Res(16->16,td0), in3 Res(16->32,td1), in5 Res(32->64,td1)
    alloc_resblock("in1_", 16, 16, 1)
    alloc_st("in3_st_", 32, 1, 1); alloc_resblock("in3_", 16, 32, 2)
    alloc_st("in5_st_", 64, 1, 1); alloc_resblock("in5_", 32, 64, 3)
    # downsamples in2 (16), in4 (32)
    P("in2_op_w", 3 * 3 * 16 * 16, 5, 11, 5.0, (3, 3, 16, 16), 0.02)
    P("in2_op_b", 16, 4, 10, 5.0, (16,), 0.05)
    P("in4_op_w", 3 * 3 * 32 * 32, 6, 11, 5.0, (3, 3, 32, 32), 0.02)
    P("in4_op_b", 32, 4, 10, 5.0, (32,), 0.05)
    # middle
    alloc_resblock("mid0_", 64, 64, 4)
    alloc_st("mid_st_", 64, TD_MIDDLE, 5);
    alloc_resblock("mid2_", 64, 64, 6)
    # decoder: out0 Res(128->64,td1+ST), out1 Res(96->64,td1+ST+Up), out2 Res(96->32,td0),
    #          out3 Res(48->32,td0+Up), out4 Res(48->16,td0), out5 Res(32->16,td0)
    alloc_resblock("out0_", 128, 64, 7); alloc_st("out0_st_", 64, 1, 7)
    alloc_resblock("out1_", 96, 64, 8); alloc_st("out1_st_", 64, 1, 8)
    P("out1_up_conv_w", 3 * 3 * 64 * 64, 5, 11, 5.0, (3, 3, 64, 64), 0.02)
    P("out1_up_conv_b", 64, 4, 10, 5.0, (64,), 0.05)
    alloc_resblock("out2_", 96, 32, 9)
    alloc_resblock("out3_", 48, 32, 10)
    P("out3_up_conv_w", 3 * 3 * 32 * 32, 6, 11, 5.0, (3, 3, 32, 32), 0.02)
    P("out3_up_conv_b", 32, 4, 10, 5.0, (32,), 0.05)
    alloc_resblock("out4_", 48, 16, 11)
    alloc_resblock("out5_", 32, 16, 12)

    # ── FORWARD ──
    # embed
    half = SDIM // 2
    ts = torch.zeros(B, SDIM, dtype=torch.float64)
    for bi in range(B):
        for i in range(half):
            freq = np.exp(-np.log(10000.0) * i / half)
            ang = float(t[bi]) * freq
            ts[bi, i] = np.cos(ang)
            ts[bi, half + i] = np.sin(ang)
    te = silu(ts @ _PARAMS["t0_w"].T + _PARAMS["t0_b"])
    te = te @ _PARAMS["t2_w"].T + _PARAMS["t2_b"]
    le = silu(y @ _PARAMS["l0_w"].T + _PARAMS["l0_b"])
    le = le @ _PARAMS["l2_w"].T + _PARAMS["l2_b"]
    emb = te + le                          # [B,TEMB]

    hs = []
    h = conv3x3(x, _PARAMS["conv_in_w"], _PARAMS["conv_in_b"])   # in0 ConvIn
    hs.append(h)
    h = resblock(h, emb, "in1_", 16, 16); hs.append(h)          # in1
    h = downsample(h, "in2_", 16); hs.append(h)                 # in2
    h = resblock(h, emb, "in3_", 16, 32)                        # in3 Res
    h = spatial_transformer(h, context, "in3_st_", 32, 1); hs.append(h)  # in3 ST
    h = downsample(h, "in4_", 32); hs.append(h)                 # in4
    h = resblock(h, emb, "in5_", 32, 64)                        # in5 Res
    h = spatial_transformer(h, context, "in5_st_", 64, 1); hs.append(h)  # in5 ST

    # middle
    h = resblock(h, emb, "mid0_", 64, 64)
    h = spatial_transformer(h, context, "mid_st_", 64, TD_MIDDLE)
    h = resblock(h, emb, "mid2_", 64, 64)

    # decoder
    def cat_skip(h):
        skip = hs.pop()
        return torch.cat([h, skip], dim=3)     # NHWC channel axis; h FIRST
    h = cat_skip(h); h = resblock(h, emb, "out0_", 128, 64); h = spatial_transformer(h, context, "out0_st_", 64, 1)
    h = cat_skip(h); h = resblock(h, emb, "out1_", 96, 64); h = spatial_transformer(h, context, "out1_st_", 64, 1)
    h = upsample(h, "out1_up_", 64)
    h = cat_skip(h); h = resblock(h, emb, "out2_", 96, 32)
    h = cat_skip(h); h = resblock(h, emb, "out3_", 48, 32)
    h = upsample(h, "out3_up_", 32)
    h = cat_skip(h); h = resblock(h, emb, "out4_", 48, 16)
    h = cat_skip(h); h = resblock(h, emb, "out5_", 32, 16)
    assert len(hs) == 0, f"skip stack not empty: {len(hs)}"

    # final
    h = gn(h, _PARAMS["out_gn_w"], _PARAMS["out_gn_b"], GN_EPS_RES)
    h = silu(h)
    out = conv3x3(h, _PARAMS["conv_out_w"], _PARAMS["conv_out_b"])   # [B,H0,W0,OUT_CH]

    # ── BACKWARD ──
    go = torch.tensor(fill(B * H0 * W0 * OUT_CH, 2, 7, 3.0, 0.05),
                      dtype=torch.float64).reshape(B, H0, W0, OUT_CH)
    out.backward(go)

    def flat(t):
        return t.detach().reshape(-1).numpy().tolist()

    lines = []
    lines.append("out " + " ".join(f"{v:.8f}" for v in flat(out)))
    lines.append("d_x " + " ".join(f"{v:.8f}" for v in flat(x.grad)))
    lines.append("d_context " + " ".join(f"{v:.8f}" for v in flat(context.grad)))
    lines.append("d_y " + " ".join(f"{v:.8f}" for v in flat(y.grad)))
    # representative weight grads: conv_in, a deep ResBlock (mid0 conv1), an ST
    # (in5 attn2 q2), conv_out.
    lines.append("d_conv_in_w " + " ".join(f"{v:.8f}" for v in flat(_PARAMS["conv_in_w"].grad)))
    lines.append("d_conv_out_w " + " ".join(f"{v:.8f}" for v in flat(_PARAMS["conv_out_w"].grad)))
    lines.append("d_mid0_conv1_w " + " ".join(f"{v:.8f}" for v in flat(_PARAMS["mid0_conv1_w"].grad)))
    lines.append("d_in5_st_b0_q2 " + " ".join(f"{v:.8f}" for v in flat(_PARAMS["in5_st_b0_q2"].grad)))
    lines.append("d_out0_conv2_w " + " ".join(f"{v:.8f}" for v in flat(_PARAMS["out0_conv2_w"].grad)))
    lines.append("d_t0_w " + " ".join(f"{v:.8f}" for v in flat(_PARAMS["t0_w"].grad)))
    lines.append("d_l0_w " + " ".join(f"{v:.8f}" for v in flat(_PARAMS["l0_w"].grad)))

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print(f"dims: B={B} H0={H0} W0={W0} IN_CH={IN_CH} MC={MC} OUT_CH={OUT_CH} "
          f"TEMB={TEMB} SDIM={SDIM} ADM={ADM} NKV={NKV} CCTX={CCTX} HEAD_DIM={HEAD_DIM} G={G}")
    print("out abs-mean:", float(out.detach().abs().mean()))


if __name__ == "__main__":
    main()
