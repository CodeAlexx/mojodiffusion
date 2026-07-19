#!/usr/bin/env python
# Oracle: AutoencoderKLWan.encode TEMPORAL/VIDEO mode (Wan 2.1-style 3D causal VAE).
# Mirrors oracle_vae_encode.py but with a multi-frame clip so the causal feat-cache
# temporal path (chunked _encode: first frame + 4-frame chunks, cache persists) is
# exercised. T=13 -> 1 + (13-1)//4 = 4 latent frames (multi-chunk).
#   (pixel-0.5)/0.5 -> vae.encode(clip).latent_dist.mode() -> (mu-mean)*(1/std)
# Saves the input clip + normalized latent so the Mojo probe consumes identical input.
# fp32 throughout (production vae_dtype=fp32).
#
# Run: /home/alex/SerenityTrainer/venv/bin/python \
#        serenitymojo/models/lingbotvideo/parity/oracle_vae_encode_video.py
import os
import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLWan

VAE_DIR = "/mnt/disk1/models/lingbot-video-moe/vae"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
T, H, W = 13, 256, 256
DEV = "cuda"


def main():
    vae = AutoencoderKLWan.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    vae = vae.to(DEV).eval()

    torch.manual_seed(0)
    pixel = torch.rand(1, 3, T, H, W, dtype=torch.float32)  # [0,1], deterministic

    x = ((pixel - 0.5) / 0.5).to(DEV)
    with torch.no_grad():
        mu = vae.encode(x).latent_dist.mode()   # [1,16,Tlat,H/8,W/8]

    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, 16, 1, 1, 1)
    z = (mu.float().cpu() - mean) / std          # normalized latent

    caps = {
        "pixel": pixel.contiguous(),
        "latent": z.contiguous(),
    }
    save_file(caps, os.path.join(OUT, "oracle_vae_encode_video.safetensors"))
    print(f"SAVED oracle_vae_encode_video.safetensors  pixel{list(pixel.shape)} -> latent{list(z.shape)}")
    print(f"  mu   mean {mu.mean():.5f} std {mu.std():.5f} absmax {mu.abs().max():.4f}")
    print(f"  z    mean {z.mean():.5f} std {z.std():.5f} absmax {z.abs().max():.4f}")


if __name__ == "__main__":
    main()
