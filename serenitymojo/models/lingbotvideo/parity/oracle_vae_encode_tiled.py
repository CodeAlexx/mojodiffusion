#!/usr/bin/env python
# Oracle: AutoencoderKLWan.tiled_encode (spatial tiling) — for the full-2x super-res
# high-res VAE encode that OOMs a 16GB card unTiled. enable_tiling() with the
# diffusers defaults (tile_sample_min 256, stride 192 -> 64px overlap, blend 8 latent).
# Encodes a clip large enough to trigger a 2x2 tile grid (both blend_v + blend_h).
#   (pixel-0.5)/0.5 -> vae.encode(clip).latent_dist.mode() -> (mu-mean)*(1/std)
# Saves clip + normalized latent so the Mojo encode_video_tiled probe matches input.
#
# Run: /home/alex/SerenityTrainer/venv/bin/python \
#        serenitymojo/models/lingbotvideo/parity/oracle_vae_encode_tiled.py
import os
import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLWan

VAE_DIR = "/mnt/disk1/models/lingbot-video-moe/vae"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
T, H, W = 13, 384, 384   # >256 in both dims -> 2x2 overlapping tiles (stride 192)
DEV = "cuda"


def main():
    vae = AutoencoderKLWan.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    vae = vae.to(DEV).eval()
    vae.enable_tiling()   # diffusers defaults: min 256, stride 192

    torch.manual_seed(0)
    pixel = torch.rand(1, 3, T, H, W, dtype=torch.float32)  # [0,1], deterministic

    x = ((pixel - 0.5) / 0.5).to(DEV)
    with torch.no_grad():
        mu = vae.encode(x).latent_dist.mode()   # tiled path (width/height > 256)

    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, 16, 1, 1, 1)
    z = (mu.float().cpu() - mean) / std

    save_file({"pixel": pixel.contiguous(), "latent": z.contiguous()},
              os.path.join(OUT, "oracle_vae_encode_tiled.safetensors"))
    print(f"SAVED oracle_vae_encode_tiled.safetensors  pixel{list(pixel.shape)} -> latent{list(z.shape)}")
    print(f"  tile cfg: min_h={vae.tile_sample_min_height} min_w={vae.tile_sample_min_width} "
          f"stride_h={vae.tile_sample_stride_height} stride_w={vae.tile_sample_stride_width}")
    print(f"  z mean {z.mean():.5f} std {z.std():.5f} absmax {z.abs().max():.4f}")


if __name__ == "__main__":
    main()
