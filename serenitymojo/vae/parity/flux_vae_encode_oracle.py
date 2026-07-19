#!/usr/bin/env python3
# flux_vae_encode_oracle.py — torch reference for the Flux.1 AE ENCODER.
#
# Reproduces the BFL Flux autoencoder encoder forward (the same math the Mojo
# FluxVaeEncoder implements) loading the REAL ae.safetensors, on a deterministic
# input. Dumps:
#   flux_vae_in.bin   — input image [1,3,IH,IW] F32 NCHW (row-major)
#   flux_vae_mu.bin   — mean latent  [1,16,LH,LW] F32 NCHW (mu = moments[:, :16])
#   flux_vae_meta.txt — shapes
#
# The encoder block math is the standard LDM/BFL AE (block.0/1 resnets, stride-2
# downsample with asymmetric (0,1,0,1) pad, mid resnet+attn+resnet, norm_out+silu
# +conv_out -> 32ch = mu|logvar). GroupNorm groups=32 eps=1e-6, attn scale=1/sqrt(C).
#
# Usage: python3 flux_vae_encode_oracle.py [LH] [LW]   (default 8 8 -> 64x64 input)

import sys, struct, math
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

AE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/serenitymojo/vae/parity"

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def dump(path, t):
    # Raw F32 little-endian, NO length prefix (matches block_oracle.py + the
    # parity _read_bin_f32 reader which reads file_size/4 floats).
    v = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    with open(path, "wb") as f:
        f.write(v.astype("<f4").tobytes())

def gn(x, w, b, groups=32, eps=1e-6):
    return F.group_norm(x, groups, w, b, eps)

def resnet(x, W, p, eps=1e-6):
    h = gn(x, W[f"{p}.norm1.weight"], W[f"{p}.norm1.bias"])
    h = F.silu(h)
    h = F.conv2d(h, W[f"{p}.conv1.weight"], W[f"{p}.conv1.bias"], stride=1, padding=1)
    h = gn(h, W[f"{p}.norm2.weight"], W[f"{p}.norm2.bias"])
    h = F.silu(h)
    h = F.conv2d(h, W[f"{p}.conv2.weight"], W[f"{p}.conv2.bias"], stride=1, padding=1)
    if f"{p}.nin_shortcut.weight" in W:
        x = F.conv2d(x, W[f"{p}.nin_shortcut.weight"], W[f"{p}.nin_shortcut.bias"], stride=1, padding=0)
    return x + h

def attn(x, W, p):
    b, c, hh, ww = x.shape
    res = x
    h = gn(x, W[f"{p}.norm.weight"], W[f"{p}.norm.bias"])
    q = F.conv2d(h, W[f"{p}.q.weight"], W[f"{p}.q.bias"])
    k = F.conv2d(h, W[f"{p}.k.weight"], W[f"{p}.k.bias"])
    v = F.conv2d(h, W[f"{p}.v.weight"], W[f"{p}.v.bias"])
    q = q.reshape(b, c, hh*ww).permute(0, 2, 1)   # [b, n, c]
    k = k.reshape(b, c, hh*ww).permute(0, 2, 1)
    v = v.reshape(b, c, hh*ww).permute(0, 2, 1)
    scale = 1.0 / math.sqrt(c)
    att = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
    out = att @ v                                  # [b, n, c]
    out = out.permute(0, 2, 1).reshape(b, c, hh, ww)
    out = F.conv2d(out, W[f"{p}.proj_out.weight"], W[f"{p}.proj_out.bias"])
    return res + out

def downsample(x, W, p):
    x = F.pad(x, (0, 1, 0, 1))  # right, bottom
    return F.conv2d(x, W[f"{p}.downsample.conv.weight"], W[f"{p}.downsample.conv.bias"], stride=2, padding=0)

def main():
    LH = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    LW = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    IH, IW = 8*LH, 8*LW

    raw = load_file(AE_PATH)
    W = {k: v.to(DEV, torch.float32) for k, v in raw.items() if k.startswith("encoder.")}

    # Deterministic input matching the Mojo gate (same ramp formula).
    img = torch.empty(1, 3, IH, IW, dtype=torch.float32)
    for c in range(3):
        for y in range(IH):
            for xx in range(IW):
                img[0, c, y, xx] = (float(((c*IH + y)*IW + xx) % 17) / 8.0) - 1.0
    x = img.to(DEV)

    h = F.conv2d(x, W["encoder.conv_in.weight"], W["encoder.conv_in.bias"], stride=1, padding=1)
    # down.0
    h = resnet(h, W, "encoder.down.0.block.0")
    h = resnet(h, W, "encoder.down.0.block.1")
    h = downsample(h, W, "encoder.down.0")
    # down.1
    h = resnet(h, W, "encoder.down.1.block.0")
    h = resnet(h, W, "encoder.down.1.block.1")
    h = downsample(h, W, "encoder.down.1")
    # down.2
    h = resnet(h, W, "encoder.down.2.block.0")
    h = resnet(h, W, "encoder.down.2.block.1")
    h = downsample(h, W, "encoder.down.2")
    # down.3 (no downsample)
    h = resnet(h, W, "encoder.down.3.block.0")
    h = resnet(h, W, "encoder.down.3.block.1")
    # mid
    h = resnet(h, W, "encoder.mid.block_1")
    h = attn(h, W, "encoder.mid.attn_1")
    h = resnet(h, W, "encoder.mid.block_2")
    # head
    h = gn(h, W["encoder.norm_out.weight"], W["encoder.norm_out.bias"])
    h = F.silu(h)
    h = F.conv2d(h, W["encoder.conv_out.weight"], W["encoder.conv_out.bias"], stride=1, padding=1)
    moments = h                       # [1,32,LH,LW]
    mu = moments[:, :16]              # [1,16,LH,LW]

    dump(f"{OUT_DIR}/flux_vae_in.bin", img)
    dump(f"{OUT_DIR}/flux_vae_mu.bin", mu)
    with open(f"{OUT_DIR}/flux_vae_meta.txt", "w") as f:
        f.write(f"LH={LH} LW={LW} IH={IH} IW={IW}\n")
        f.write(f"moments_shape={list(moments.shape)} mu_shape={list(mu.shape)}\n")
        f.write(f"mu_mean={mu.mean().item():.6f} mu_std={mu.std().item():.6f}\n")
    print(f"[oracle] LH={LH} LW={LW} IH={IH} IW={IW}")
    print(f"[oracle] moments {list(moments.shape)} mu {list(mu.shape)}")
    print(f"[oracle] mu mean={mu.mean().item():.6f} std={mu.std().item():.6f} "
          f"min={mu.min().item():.6f} max={mu.max().item():.6f}")

if __name__ == "__main__":
    main()
