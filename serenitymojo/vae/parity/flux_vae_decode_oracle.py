#!/usr/bin/env python3
# flux_vae_decode_oracle.py — torch reference for the Flux.1 AE DECODER.
#
# Mirror of flux_vae_encode_oracle.py for the decode direction. Reproduces the
# BFL Flux autoencoder DECODER forward (the same math the Mojo LdmVaeDecoder
# implements via load_flux1_ldm_decoder) loading the REAL ae.safetensors on a
# deterministic latent. Critically: the Mojo decode() folds the rescale
#   z = z / scale + shift      (scale=0.3611, shift=0.1159)
# INSIDE decode(), so this oracle does the same and the dumped latent is the RAW
# (pre-rescale) latent that the Mojo probe reads byte-for-byte.
#
# Dumps:
#   flux_vae_dec_z.bin    — raw latent [1,16,LH,LW] F32 NCHW (row-major)
#   flux_vae_dec_img.bin  — decoded image [1,3,8*LH,8*LW] F32 NCHW
#   flux_vae_dec_meta.txt — shapes + stats
#
# Usage: python3 flux_vae_decode_oracle.py [LH] [LW]   (default 64 64 -> 512x512)

import sys, math
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

AE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/serenitymojo/vae/parity"
FLUX_SCALE = 0.3611
FLUX_SHIFT = 0.1159

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def dump(path, t):
    # Raw F32 little-endian, NO length prefix (matches the parity _read_bin_f32
    # reader which reads file_size/4 floats).
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


def upsample(x, W, p):
    # LDM/BFL: nearest x2 then 3x3 conv (stride1 pad1).
    x = F.interpolate(x, scale_factor=2.0, mode="nearest")
    return F.conv2d(x, W[f"{p}.upsample.conv.weight"], W[f"{p}.upsample.conv.bias"], stride=1, padding=1)


def main():
    LH = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    LW = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    raw = load_file(AE_PATH)
    W = {k: v.to(DEV, torch.float32) for k, v in raw.items() if k.startswith("decoder.")}

    # Deterministic raw latent [1,16,LH,LW] ~ N(0,1) (decode parity is input-
    # agnostic; both sides read these exact bytes). Realistic post-denoise
    # magnitude is a few; N(0,1) exercises the full decoder dynamic range.
    g = torch.Generator().manual_seed(1234)
    z_raw = torch.randn(1, 16, LH, LW, generator=g, dtype=torch.float32)
    dump(f"{OUT_DIR}/flux_vae_dec_z.bin", z_raw)

    z = z_raw.to(DEV)
    # Rescale exactly as the Mojo decode() does (folded inside).
    z = z / FLUX_SCALE + FLUX_SHIFT

    h = F.conv2d(z, W["decoder.conv_in.weight"], W["decoder.conv_in.bias"], stride=1, padding=1)
    # mid
    h = resnet(h, W, "decoder.mid.block_1")
    h = attn(h, W, "decoder.mid.attn_1")
    h = resnet(h, W, "decoder.mid.block_2")
    # up.3 (highest channels) -> up.0 (final). 3 upsamples on up.3/2/1, none on up.0.
    for blk in (3, 2, 1):
        h = resnet(h, W, f"decoder.up.{blk}.block.0")
        h = resnet(h, W, f"decoder.up.{blk}.block.1")
        h = resnet(h, W, f"decoder.up.{blk}.block.2")
        h = upsample(h, W, f"decoder.up.{blk}")
    # up.0 (no upsample)
    h = resnet(h, W, "decoder.up.0.block.0")
    h = resnet(h, W, "decoder.up.0.block.1")
    h = resnet(h, W, "decoder.up.0.block.2")
    # head
    h = gn(h, W["decoder.norm_out.weight"], W["decoder.norm_out.bias"])
    h = F.silu(h)
    img = F.conv2d(h, W["decoder.conv_out.weight"], W["decoder.conv_out.bias"], stride=1, padding=1)

    dump(f"{OUT_DIR}/flux_vae_dec_img.bin", img)
    with open(f"{OUT_DIR}/flux_vae_dec_meta.txt", "w") as f:
        f.write(f"LH={LH} LW={LW} IH={8*LH} IW={8*LW}\n")
        f.write(f"z_shape={list(z_raw.shape)} img_shape={list(img.shape)}\n")
        f.write(f"img_mean={img.mean().item():.6f} img_std={img.std().item():.6f} "
                f"min={img.min().item():.6f} max={img.max().item():.6f}\n")
    print(f"[oracle] LH={LH} LW={LW} -> img {list(img.shape)}")
    print(f"[oracle] img mean={img.mean().item():.6f} std={img.std().item():.6f} "
          f"min={img.min().item():.6f} max={img.max().item():.6f}")


if __name__ == "__main__":
    main()
