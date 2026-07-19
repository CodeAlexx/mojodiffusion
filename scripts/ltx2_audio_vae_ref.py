#!/usr/bin/env python3
"""LTX-2.3 Audio VAE decoder parity reference (P4 audio-VAE gate).

Faithful PyTorch port of the Rust audio-VAE decoder
`inference-flame/src/vae/ltx2_audio_vae.rs` (LTX2AudioVaeDecoder::decode),
which itself mirrors `ltx_core.model.audio_vae.audio_vae.AudioDecoder` for the
production LTX-2.3 distilled checkpoint (prefix `audio_vae.decoder.*` +
`audio_vae.per_channel_statistics.*`):

  latent [B, 8, T, 16]  (normalized)
  1. un_normalize: rearrange "b c t f -> b t (c f)" (128-dim) -> *std + mean
                   -> rearrange back to [B, 8, T, F]
  2. conv_in:  CausalConv2d(8 -> 512, k=3, causality=HEIGHT)
  3. mid:      block_1 (Resnet 512->512), attn_1 = Identity, block_2 (512->512)
  4. up stages, forward iterates REVERSED -> up[2] -> up[1] -> up[0]:
        up[2]: 3 ResnetBlocks 512->512, upsample(512->512)
        up[1]: ResnetBlocks (512->256, 256->256, 256->256), upsample(256->256)
        up[0]: ResnetBlocks (256->128, 128->128, 128->128), NO upsample
  5. norm_out: PixelNorm (no weights, RMS over channel, eps=1e-6)
  6. SiLU
  7. conv_out: CausalConv2d(128 -> 2, k=3)  -> stereo mel-equivalent

CausalConv2d: causality on HEIGHT (dim 2, time), ZERO pad (not replicate);
F.pad order (W_left, W_right, H_top, H_bottom) = (pad_w//2, pad_w-pad_w//2,
pad_h, 0) with pad_h = pad_w = k-1 = 2 for k=3 -> (1, 1, 2, 0).
Upsample: nearest x2 (H and W) -> CausalConv2d -> drop FIRST height frame
  (x[:, :, 1:, :]).
PixelNorm eps = 1e-6 (build_normalization_layer overrides the 1e-8 default).
ResnetBlock: norm1(PixelNorm) -> SiLU -> conv1 -> norm2 -> SiLU -> conv2
  + (nin_shortcut(x) if in!=out else x).  temb is None (temb_ch == 0).

Activations are cast to BF16 between major ops to mirror the Mojo BF16 storage
path; convs accumulate in F32.

A DETERMINISTIC normalized latent [1,8,T,16] is generated (seeded), saved as
`latent` (NCHW, F32), decoded, and the decoded mel saved as `decoded`
(NCHW [1,2,T_out,F_out], F32). The Mojo smoke loads the SAME latent, decodes
with the pure-Mojo audio-VAE decoder, and gates cos(decoded_mojo, decoded) >= 0.999.

Outputs -> output/ltx2_audio_vae/audio_vae_ref.safetensors  (latent, decoded).
"""

import json
import os
import struct

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
OUT = "/home/alex/mojodiffusion/output/ltx2_audio_vae/audio_vae_ref.safetensors"

PREFIX = "audio_vae.decoder."
STATS = "audio_vae.per_channel_statistics."
PIXEL_NORM_EPS = 1e-6
PATCHED_CH = 128  # 8 latent x 16 mel bins
SEED = 20260528

# Latent shape [B, 8, T, 16]. T=8 -> time out = 4*8 - 3 = 29; F=16 -> 16*4 = 64.
B, C, T, FL = 1, 8, 8, 16

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# up stage spec: (n_blocks, has_upsample). Stored ascending up[0],up[1],up[2];
# forward iterates reversed.
UP_STAGES = [
    (3, False),  # up[0]: 256->128, 128->128, 128->128, NO upsample
    (3, True),   # up[1]: 512->256, 256->256, 256->256, upsample
    (3, True),   # up[2]: 512->512 x3, upsample
]


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
print(f"loaded {len(W)} audio-VAE tensors")


def w(key):
    if key not in W:
        raise KeyError(f"missing weight: {key}")
    return W[key].to(DEV)


def has(key):
    return key in W


def bf16(x):
    """Mirror the Mojo BF16 activation storage between major ops."""
    return x.to(torch.bfloat16).to(torch.float32)


# ---------------------------------------------------------------------------
# CausalConv2d — causality HEIGHT (dim 2 = time), ZERO pad.
# F.pad order (W_left, W_right, H_top, H_bottom). For k=3: (1, 1, 2, 0).
# Input/Output NCHW [B,C,H,W] = [B,C,T,F]. F32 accumulate.
# ---------------------------------------------------------------------------
def causal_conv2d(x, prefix):
    weight = w(prefix + ".weight")  # [Cout,Cin,kH,kW]
    bias = w(prefix + ".bias")      # [Cout]
    kh, kw = weight.shape[2], weight.shape[3]
    pad_h = kh - 1   # all on top (causal time)
    pad_w = kw - 1   # symmetric split on freq
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    # F.pad: (W_left, W_right, H_top, H_bottom)
    x = F.pad(x, (pad_w_left, pad_w_right, pad_h, 0))
    return F.conv2d(x, weight, bias=bias, stride=1, padding=0)


def pixel_norm(x):
    xf = x.to(torch.float32)
    mean_sq = (xf * xf).mean(dim=1, keepdim=True)
    denom = torch.rsqrt(mean_sq + PIXEL_NORM_EPS)
    return xf * denom


def resnet_block(x, prefix):
    h = pixel_norm(x)
    h = F.silu(h)
    h = bf16(h)
    h = causal_conv2d(h, prefix + ".conv1.conv")
    h = bf16(h)
    h = pixel_norm(h)
    h = F.silu(h)
    h = bf16(h)
    h = causal_conv2d(h, prefix + ".conv2.conv")
    h = bf16(h)
    if has(prefix + ".nin_shortcut.conv.weight"):
        x = causal_conv2d(x, prefix + ".nin_shortcut.conv")
        x = bf16(x)
    return x + h


def upsample(x, prefix):
    # nearest x2 on H (time) and W (freq).
    x = F.interpolate(x, scale_factor=2.0, mode="nearest")
    x = bf16(x)
    x = causal_conv2d(x, prefix + ".conv.conv")
    x = bf16(x)
    # Drop FIRST height frame to undo encoder padding (causality_axis=HEIGHT).
    return x[:, :, 1:, :]


def un_normalize(x):
    # x [B,8,T,16] -> [B,T,128] -> *std + mean -> back to [B,8,T,16].
    b, c, t, f = x.shape
    cf = c * f
    std = w(STATS + "std-of-means").reshape(1, 1, cf)
    mean = w(STATS + "mean-of-means").reshape(1, 1, cf)
    flat = x.permute(0, 2, 1, 3).reshape(b, t, cf)
    denorm = flat * std + mean
    return denorm.reshape(b, t, c, f).permute(0, 2, 1, 3).contiguous()


def decode(latent):
    h = un_normalize(latent)
    h = bf16(h)
    h = causal_conv2d(h, PREFIX + "conv_in.conv")
    h = bf16(h)

    # mid: block_1 -> (Identity) -> block_2
    h = resnet_block(h, PREFIX + "mid.block_1")
    h = resnet_block(h, PREFIX + "mid.block_2")
    h = bf16(h)

    # up stages in REVERSE order: up[2] -> up[1] -> up[0]
    for level in reversed(range(len(UP_STAGES))):
        n_blocks, has_up = UP_STAGES[level]
        bp = PREFIX + f"up.{level}"
        for bidx in range(n_blocks):
            h = resnet_block(h, bp + f".block.{bidx}")
        h = bf16(h)
        if has_up:
            h = upsample(h, bp + ".upsample")
            h = bf16(h)

    # norm_out -> SiLU -> conv_out
    h = pixel_norm(h)
    h = F.silu(h)
    h = bf16(h)
    h = causal_conv2d(h, PREFIX + "conv_out.conv")
    return bf16(h)


def main():
    torch.manual_seed(SEED)
    latent = torch.randn(B, C, T, FL, dtype=torch.float32) * 0.5
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
