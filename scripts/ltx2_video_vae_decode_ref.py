#!/usr/bin/env python3
"""LTX-2.3 Video VAE decoder parity reference (P5 video-VAE gate).

Faithful PyTorch port of the Rust video-VAE decoder
`inference-flame/src/vae/ltx2_vae.rs` (LTX2VaeDecoder::decode_with_dump),
specialised to the production LTX-2.3 distilled checkpoint:

  un_normalize -> conv_in (CausalConv3d 128->1024)
  up_blocks.0  Mid          1024  n_res=2
  up_blocks.1  DepthToSpace (2,2,2) red=2  -> 512
  up_blocks.2  Mid          512   n_res=2
  up_blocks.3  DepthToSpace (2,2,2) red=1  -> 512
  up_blocks.4  Mid          512   n_res=4
  up_blocks.5  DepthToSpace (2,1,1) red=2  -> 256
  up_blocks.6  Mid          256   n_res=6
  up_blocks.7  DepthToSpace (1,2,2) red=2  -> 128
  up_blocks.8  Mid          128   n_res=4
  conv_norm_out: PixelNorm (no weights, RMS over channel) -> SiLU
  conv_out (CausalConv3d 128->48) -> unpatchify (patch=4)

CausalConv3d (decoder, causal=False): replicate first AND last frame
(kT-1)/2=1 each side of TIME, symmetric zero-pad on H/W; F32-accumulated.
PixelNorm eps=1e-6. DepthToSpace drops first frame when temporal stride==2.

A DETERMINISTIC normalized latent [1,128,F,H,W] is generated (seeded), saved as
`latent` (NCDHW, F32), decoded, and the decoded video saved as `decoded`
(NCDHW [1,3,F_out,H_out,W_out], F32). Activations are cast to BF16 between major
ops to mirror the Rust BF16 storage path; convs accumulate in F32.

Outputs -> output/ltx2_video_vae/video_vae_ref.safetensors  (latent, decoded).

The Mojo smoke (pipeline/ltx2_video_vae_decode_smoke.mojo) loads the SAME latent,
decodes with the pure-Mojo decoder, and gates cos(decoded_mojo, decoded) >= 0.999.
"""

import json
import os
import struct
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
OUT = "/home/alex/mojodiffusion/output/ltx2_video_vae/video_vae_ref.safetensors"

PREFIX = "vae.decoder."
STATS = "vae.per_channel_statistics."
PIXEL_NORM_EPS = 1e-6
PATCH = 4
SEED = 20260528

# Latent shape. F=2 -> F_out = 1 + (2-1)*8 = 9 frames; H=W=2 -> 64x64 pixels.
B, C, FL, HL, WL = 1, 128, 2, 2, 2

DEV = "cuda" if torch.cuda.is_available() else "cpu"

DECODER_BLOCKS = [
    ("mid", 1024, 2),
    ("d2s", 1024, (2, 2, 2), 2),
    ("mid", 512, 2),
    ("d2s", 512, (2, 2, 2), 1),
    ("mid", 512, 4),
    ("d2s", 512, (2, 1, 1), 2),
    ("mid", 256, 6),
    ("d2s", 256, (1, 2, 2), 2),
    ("mid", 128, 4),
]


# ---------------------------------------------------------------------------
# safetensors partial load (header + selected tensors), -> F32 on DEV
# ---------------------------------------------------------------------------
def load_weights(path, want_prefixes):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        data_start = 8 + hlen
        out = {}
        DT = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
        for k, info in hdr.items():
            if k == "__metadata__":
                continue
            if not any(k.startswith(p) for p in want_prefixes):
                continue
            dt = DT[info["dtype"]]
            shape = info["shape"]
            b0, b1 = info["data_offsets"]
            f.seek(data_start + b0)
            raw = f.read(b1 - b0)
            t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(shape)
            out[k] = t.to(torch.float32)
        return out


W = load_weights(CKPT, [PREFIX, STATS])
print(f"loaded {len(W)} VAE tensors")


def w(key):
    if key not in W:
        raise KeyError(f"missing weight: {key}")
    return W[key].to(DEV)


def bf16(x):
    """Mirror the Rust BF16 activation storage between major ops."""
    return x.to(torch.bfloat16).to(torch.float32)


# ---------------------------------------------------------------------------
# CausalConv3d (non-causal replicate pad on time, symmetric zero pad on H/W).
# F32 accumulate. Input/Output NCDHW [B,C,D,H,W].
# ---------------------------------------------------------------------------
def causal_conv3d(x, prefix, k=3):
    weight = w(prefix + ".weight")  # [Cout,Cin,kD,kH,kW]
    bias = w(prefix + ".bias")      # [Cout]
    d = x.shape[2]
    if d == 0:
        return x
    half = (k - 1) // 2
    if half > 0:
        first = x[:, :, 0:1].repeat(1, 1, half, 1, 1)
        last = x[:, :, d - 1:d].repeat(1, 1, half, 1, 1)
        x = torch.cat([first, x, last], dim=2)
    hp = k // 2
    wp = k // 2
    # F32 conv, symmetric pad on H/W only (D already replicate-padded, pad_d=0).
    out = F.conv3d(x, weight, bias=bias, stride=1, padding=(0, hp, wp))
    return out


def pixel_norm(x):
    xf = x.to(torch.float32)
    mean_sq = (xf * xf).mean(dim=1, keepdim=True)
    denom = torch.rsqrt(mean_sq + PIXEL_NORM_EPS)
    return xf * denom


def resnet_block(x, prefix):
    h = pixel_norm(x)
    h = F.silu(h)
    h = bf16(h)
    h = causal_conv3d(h, prefix + ".conv1.conv")
    h = bf16(h)
    h = pixel_norm(h)
    h = F.silu(h)
    h = bf16(h)
    h = causal_conv3d(h, prefix + ".conv2.conv")
    h = bf16(h)
    return x + h


def depth_to_space(x, stride):
    b, c_total, f, h, wd = x.shape
    p1, p2, p3 = stride
    c = c_total // (p1 * p2 * p3)
    y = x.reshape(b, c, p1, p2, p3, f, h, wd)
    y = y.permute(0, 1, 5, 2, 6, 3, 7, 4)  # [b,c,f,p1,h,p2,w,p3]
    return y.reshape(b, c, f * p1, h * p2, wd * p3)


def d2s_block(x, prefix, stride):
    y = causal_conv3d(x, prefix + ".conv.conv")
    y = bf16(y)
    y = depth_to_space(y, stride)
    if stride[0] == 2:
        d = y.shape[2]
        if d > 0:
            y = y[:, :, 1:]
    return y


def unpatchify(x):
    # [B,48,F,H,W] -> [B,3,F,H*4,W*4], einops b (c p r q) f h w with p=1,q=4,r=4
    b, c_total, f, h, wd = x.shape
    c = c_total // (PATCH * PATCH)
    y = x.reshape(b, c, PATCH, PATCH, f, h, wd)
    y = y.permute(0, 1, 4, 5, 3, 6, 2)  # [b,c,f,h,q,w,r]
    return y.reshape(b, c, f, h * PATCH, wd * PATCH)


def un_normalize(x):
    std = w(STATS + "std-of-means").reshape(1, C, 1, 1, 1)
    mean = w(STATS + "mean-of-means").reshape(1, C, 1, 1, 1)
    return x * std + mean


def decode(latent):
    x = un_normalize(latent)
    x = bf16(x)
    h = causal_conv3d(x, PREFIX + "conv_in.conv")
    h = bf16(h)
    for i, spec in enumerate(DECODER_BLOCKS):
        bp = PREFIX + f"up_blocks.{i}"
        if spec[0] == "mid":
            _, channels, n_res = spec
            for r in range(n_res):
                h = resnet_block(h, bp + f".res_blocks.{r}")
        else:
            _, in_ch, stride, red = spec
            h = d2s_block(h, bp, stride)
        h = bf16(h)
    h = pixel_norm(h)
    h = F.silu(h)
    h = bf16(h)
    h = causal_conv3d(h, PREFIX + "conv_out.conv")
    h = bf16(h)
    return unpatchify(h)


def main():
    torch.manual_seed(SEED)
    # Deterministic normalized latent [1,128,FL,HL,WL] (standard-normal-ish,
    # bounded). Stored F32, NCDHW.
    latent = torch.randn(B, C, FL, HL, WL, dtype=torch.float32) * 0.5
    latent = latent.to(DEV)

    with torch.no_grad():
        decoded = decode(latent)

    decoded = decoded.to(torch.float32).cpu().contiguous()
    print("latent  shape:", tuple(latent.shape))
    print("decoded shape:", tuple(decoded.shape))
    print(
        "decoded stats: min=%.4f max=%.4f mean=%.4f std=%.4f"
        % (decoded.min(), decoded.max(), decoded.mean(), decoded.std())
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(
        {
            "latent": latent.to(torch.float32).cpu().contiguous(),
            "decoded": decoded,
        },
        OUT,
    )
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
