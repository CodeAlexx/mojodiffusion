# gen_krea2_vae.py — torch reference for the krea2 VAE DECODE (latent -> RGB).
#
# krea2 (ai-toolkit extensions_built_in/diffusion_models/krea2/krea2.py) uses the
# Qwen-Image VAE (AutoencoderKLQwenImage, "Qwen/Qwen-Image" subfolder="vae", f8,
# 16 latent channels). Its decode_latents (krea2.py:445-471) is, EXACTLY:
#
#   latents = latents.unsqueeze(2)                       # (B,16,h,w) -> (B,16,1,h,w)
#   mean = cfg.latents_mean.view(1,z_dim,1,1,1)
#   std  = cfg.latents_std.view(1,z_dim,1,1,1)
#   latents = latents * std + mean                       # DENORM (note: * std, + mean)
#   images = vae.decode(latents).sample                  # (B,3,1,H,W)
#   images = images.squeeze(2)                           # (B,3,H,W) in [-1,1]
#
# This dumps a FIXED random 16-ch latent and the decoded RGB so the Mojo
# qwenimage decoder (models/vae/qwenimage_decoder.mojo, which already does the
# identical z*std+mean denorm + image-mode decode) can be gated against it.
#
# Resident/display-safe: latent h=w=32 -> 256x256 image, VAE only (no DiT).
#
# DEV-ONLY parity oracle. Never shipped. Run with the serenityflow venv python:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/vae/parity/gen_krea2_vae.py [LH] [LW]

import json
import os
import struct
import sys

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage

# The VAE krea2.py loads: AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image",
# subfolder="vae"). Resolve the local HF snapshot so this is offline + pinned.
VAE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image/"
    "snapshots/75e0b4be04f60ec59a75f475837eced720f823b6/vae"
)
OUTDIR = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"


def main():
    LH = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    LW = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    H, W = LH * 8, LW * 8
    tag = f"{LH}x{LW}"

    assert os.path.isdir(VAE_DIR), f"VAE dir not found: {VAE_DIR}"
    vae = AutoencoderKLQwenImage.from_pretrained(VAE_DIR, torch_dtype=torch.bfloat16)
    vae = vae.to("cuda", dtype=torch.bfloat16).eval()
    vae.requires_grad_(False)

    z_dim = vae.config.z_dim
    assert z_dim == 16, z_dim

    # Fixed random 16-ch latent [1,16,LH,LW]. Generated on CPU f32 for a stable,
    # reproducible reference, then cast to the bf16 the krea2 VAE runs in.
    g = torch.Generator(device="cpu").manual_seed(20260624)
    latent_f32 = torch.randn(1, z_dim, LH, LW, generator=g, dtype=torch.float32)

    # ---- krea2 decode_latents (krea2.py:445-471), bf16 path ----
    latents = latent_f32.to("cuda", dtype=torch.bfloat16)
    latents = latents.unsqueeze(2)  # (1,16,1,LH,LW)
    mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    std = (
        torch.tensor(vae.config.latents_std)
        .view(1, z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents * std + mean
    with torch.no_grad():
        images = vae.decode(latents).sample  # (1,3,1,H,W)
    images = images.squeeze(2)  # (1,3,H,W), already clamped [-1,1] by the VAE

    rgb = images.float().cpu().contiguous()  # [1,3,H,W] f32
    print(
        f"[oracle] latent {tuple(latent_f32.shape)} -> rgb {tuple(rgb.shape)} "
        f"min={rgb.min().item():.4f} max={rgb.max().item():.4f} "
        f"mean={rgb.mean().item():.4f} std={rgb.std().item():.4f}"
    )

    # Dump latent (the EXACT bytes the Mojo probe will feed) + decoded RGB.
    latent_f32.cpu().contiguous().numpy().astype("<f4").tofile(
        f"{OUTDIR}/krea2vae_latent_{tag}.bin"
    )
    rgb.numpy().astype("<f4").tofile(f"{OUTDIR}/krea2vae_rgb_{tag}.bin")
    with open(f"{OUTDIR}/krea2vae_meta_{tag}.json", "w") as f:
        json.dump(
            {
                "LH": LH,
                "LW": LW,
                "H": H,
                "W": W,
                "latent_shape": list(latent_f32.shape),
                "rgb_shape": list(rgb.shape),
                "vae_dir": VAE_DIR,
                "latents_mean": list(vae.config.latents_mean),
                "latents_std": list(vae.config.latents_std),
            },
            f,
        )
    print(
        f"[oracle] wrote krea2vae_latent_{tag}.bin, krea2vae_rgb_{tag}.bin, "
        f"krea2vae_meta_{tag}.json"
    )


if __name__ == "__main__":
    main()
