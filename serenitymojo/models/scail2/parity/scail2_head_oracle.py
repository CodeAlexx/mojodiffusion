#!/usr/bin/env python3
# serenitymojo/models/scail2/parity/scail2_head_oracle.py
#
# G1 head fwd+bwd torch-autograd oracle for the SCAIL-2 video HEAD
# (models/scail2/scail2_streamed_dit.mojo::_head_video) + its NEW backward.
#
# Reproduces, in float32 at a reduced dim, the exact SCAIL-2 head:
#   video   = img[:, video_offset : video_offset+video_rows, :]      # [1,VR,dim]
#   head_e  = head.modulation[1,2,dim] + e_head[1,1,dim].unsqueeze    # [1,2,dim]
#   shift,scale = head_e.chunk(2, dim=1)                              # [1,1,dim] each
#   head_in = (1+scale) * LN_noaffine(video) + shift                 # [1,VR,dim]
#   proj    = head_in @ head.head.weight.T + head.head.bias          # [1,VR,out*4]
#   out     = unpatchify3d(proj.reshape[VR,out*4])                   # [out,FT,GH*2,GW*2]
# where unpatchify3d matches ops/patchify3d.mojo::unpatchify3d EXACTLY:
#   within-patch READ order (pf,ph,pw,c) channel-FASTEST, patch F-major.
#
# AR=0 so video_offset=0 and S == video_rows: the head input-grad IS the full
# block-40 output grad in the video region (all other rows are zero — trivially
# so here since there are no other rows).
#
# Dumps float32 .bin (little-endian) for scail2_head_parity.mojo. Frozen head
# weights: only the input-grad path (d_video) is checked, matching the Mojo
# backward which discards d_scale/d_shift/d_head_w (head modulation + linear
# FROZEN).
#
# Run (SEPARATE command, BEFORE the mojo gate):
#   /home/alex/torchref-image/venv/bin/python \
#       serenitymojo/models/scail2/parity/scail2_head_oracle.py

import os
import struct
import torch

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REF = os.path.dirname(os.path.abspath(__file__))

# ── reduced head dims ──
DIM = 512
OUT_DIM = 16
FT, GH, GW = 1, 2, 2
PF, PH, PW = 1, 2, 2
AR = 0
VIDEO_OFFSET = (AR + 1) * GH * GW      # 4
VIDEO_ROWS = FT * GH * GW              # 4
S = VIDEO_OFFSET + VIDEO_ROWS          # 8 (ref block rows + video rows)
PD = OUT_DIM * PF * PH * PW            # 64
EPS = 1e-6


def W(name, t):
    a = t.detach().reshape(-1).to(torch.float32).cpu().numpy()
    with open(os.path.join(REF, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % a.size, *a.tolist()))


def unpatchify3d(seq, C, F, Hh, Ww, pf, ph, pw):
    """Match ops/patchify3d.mojo::unpatchify3d: out[c,f,h,w] = seq[patch, src_ch]
    with src_ch = pfi*(ph*pw*C)+phi*(pw*C)+pwi*C+ci (c FASTEST),
    patch = fi*HO*WO+hi*WO+wi (F-major). seq: [L, C*pf*ph*pw]."""
    FO, HO, WO = F // pf, Hh // ph, Ww // pw
    out = seq.new_zeros((C, F, Hh, Ww))
    for c in range(C):
        for f in range(F):
            for h in range(Hh):
                for w in range(Ww):
                    fi, pfi = f // pf, f % pf
                    hi, phi = h // ph, h % ph
                    wi, pwi = w // pw, w % pw
                    patch = fi * HO * WO + hi * WO + wi
                    src = pfi * (ph * pw * C) + phi * (pw * C) + pwi * C + c
                    out[c, f, h, w] = seq[patch, src]
    return out


def main():
    g = torch.Generator().manual_seed(7)
    img = (torch.randn(1, S, DIM, generator=g) * 0.7).to(DEV).float().requires_grad_(True)
    e_head = (torch.randn(1, 1, DIM, generator=g) * 0.2).to(DEV).float()
    head_mod = (torch.randn(1, 2, DIM, generator=g) * 0.1).to(DEV).float()
    head_w = (torch.randn(PD, DIM, generator=g) * (1.0 / DIM ** 0.5)).to(DEV).float()
    head_b = (torch.randn(PD, generator=g) * 0.05).to(DEV).float()

    # ── forward (faithful to _head_video) ──
    video = img[:, VIDEO_OFFSET:VIDEO_OFFSET + VIDEO_ROWS, :]       # [1,VR,dim]
    head_e = head_mod + e_head                                     # [1,2,dim]
    shift, scale = head_e.chunk(2, dim=1)                          # [1,1,dim]
    mean = video.mean(-1, keepdim=True)
    var = video.var(-1, unbiased=False, keepdim=True)
    ln = (video - mean) / torch.sqrt(var + EPS)
    head_in = (1.0 + scale) * ln + shift                          # [1,VR,dim]
    proj = head_in.reshape(VIDEO_ROWS, DIM) @ head_w.t() + head_b # [VR, out*4]
    out = unpatchify3d(proj, OUT_DIM, FT, GH * PH, GW * PW, PF, PH, PW)

    d_out = (torch.randn(OUT_DIM, FT, GH * PH, GW * PW, generator=g) * 0.3).to(DEV).float()
    loss = (out * d_out).sum()
    loss.backward()
    d_video = img.grad[:, VIDEO_OFFSET:VIDEO_OFFSET + VIDEO_ROWS, :].reshape(VIDEO_ROWS, DIM)

    # ── dump inputs ──
    W("s2h_img", img.reshape(S, DIM))
    W("s2h_e_head", e_head.reshape(DIM))
    W("s2h_head_mod", head_mod.reshape(2 * DIM))
    W("s2h_head_w", head_w)
    W("s2h_head_b", head_b)
    W("s2h_d_out", d_out)
    # ── dump refs ──
    W("s2h_ref_out", out)
    W("s2h_ref_d_video", d_video)

    print("head oracle: DIM=%d OUT_DIM=%d FT=%d GH=%d GW=%d VR=%d S=%d PD=%d"
          % (DIM, OUT_DIM, FT, GH, GW, VIDEO_ROWS, S, PD))
    print("forward loss =", float(loss))
    print("out norm =", float(out.norm()), " d_video norm =", float(d_video.norm()))
    print("DONE")


if __name__ == "__main__":
    main()
