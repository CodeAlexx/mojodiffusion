#!/usr/bin/env python
# Oracle: AutoencoderKLWan.encode (Wan 2.1-style 3D causal VAE), image mode T=1.
# Loads the REAL LingBot VAE from the vae dir (diffusers), builds a deterministic
# [1,3,1,H,W] pixel in [0,1], runs the i2v encode_image_latent contract:
#   (pixel-0.5)/0.5 -> vae.encode(...).latent_dist.mode() -> (mu-mean)*(1/std)
# and saves BOTH the input pixel and the normalized latent so the Mojo probe
# consumes the identical input. fp32 throughout (production vae_dtype=fp32).
#
# Run: /home/alex/SerenityTrainer/venv/bin/python \
#        serenitymojo/models/lingbotvideo/parity/oracle_vae_encode.py
import os
import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLWan

VAE_DIR = "/mnt/disk1/models/lingbot-video-moe/vae"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
H = W = 256
DEV = "cuda"


def main():
    vae = AutoencoderKLWan.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    vae = vae.to(DEV).eval()

    torch.manual_seed(0)
    pixel = torch.rand(1, 3, 1, H, W, dtype=torch.float32)  # [0,1], deterministic

    x = ((pixel - 0.5) / 0.5).to(DEV)
    with torch.no_grad():
        mu = vae.encode(x).latent_dist.mode()   # [1,16,1,H/8,W/8] = first 16 ch

    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, 16, 1, 1, 1)
    z = (mu.float().cpu() - mean) / std          # normalized i2v latent

    caps = {
        "pixel": pixel.contiguous(),
        "latent": z.contiguous(),
    }
    save_file(caps, os.path.join(OUT, "oracle_vae_encode.safetensors"))
    print(f"SAVED oracle_vae_encode.safetensors  pixel{list(pixel.shape)} -> latent{list(z.shape)}")
    print(f"  mu   mean {mu.mean():.5f} std {mu.std():.5f} absmax {mu.abs().max():.4f}")
    print(f"  z    mean {z.mean():.5f} std {z.std():.5f} absmax {z.abs().max():.4f}")


if __name__ == "__main__":
    main()
