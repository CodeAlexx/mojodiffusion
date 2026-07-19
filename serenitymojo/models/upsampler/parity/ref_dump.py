#!/usr/bin/env python3
# Reference dump for the LTX-2 LatentUpsampler (spatial-x2 + temporal-x2).
# Loads the real safetensors, runs the bare upsampler forward on a small fixed
# latent, and dumps input/output as .npy for the Mojo unit gate.
#
# Layout convention dumped: NCDHW (PyTorch native, channel-first).
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-core/src")
from ltx_core.model.upsampler.model import LatentUpsampler

W = "/home/alex/.serenity/models/ltx2_upscalers"
OUT = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.bfloat16


def build_spatial():
    # initial_conv [1024,128,3,3,3] -> mid=1024, dims=3; upsampler has
    # blur_down.kernel + conv [4096,1024,3,3] -> SpatialRationalResampler scale=2
    m = LatentUpsampler(
        in_channels=128,
        mid_channels=1024,
        num_blocks_per_stage=4,
        dims=3,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=True,
    )
    sd = load_file(os.path.join(W, "ltx-2-spatial-upscaler-x2-1.0.safetensors"))
    sd = {k: v.to(dtype=DTYPE) for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=True)
    m.to(dtype=DTYPE)
    m.eval()
    return m


def build_temporal():
    # initial_conv [512,128,3,3,3] -> mid=512; upsampler.0 [1024,512,3,3,3] ->
    # Conv3d(512->2*512) + PixelShuffleND(1)  => temporal_upsample, no rational.
    m = LatentUpsampler(
        in_channels=128,
        mid_channels=512,
        num_blocks_per_stage=4,
        dims=3,
        spatial_upsample=False,
        temporal_upsample=True,
        spatial_scale=2.0,
        rational_resampler=False,
    )
    sd = load_file(os.path.join(W, "ltx-2-temporal-upscaler-x2-1.0.safetensors"))
    sd = {k: v.to(dtype=DTYPE) for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    m.to(dtype=DTYPE)
    m.eval()
    return m


@torch.no_grad()
def run(tag, model, latent):
    out = model(latent)
    lat_np = latent.cpu().numpy().astype(np.float32)
    out_np = out.cpu().numpy().astype(np.float32)
    np.save(os.path.join(OUT, f"{tag}_in.npy"), lat_np)
    np.save(os.path.join(OUT, f"{tag}_out.npy"), out_np)
    lat_np.tofile(os.path.join(OUT, f"{tag}_in.bin"))
    out_np.tofile(os.path.join(OUT, f"{tag}_out.bin"))
    print(f"{tag}: in {tuple(latent.shape)} -> out {tuple(out.shape)}")
    return out


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Small spatial latent: B=1, C=128, F=2, H=8, W=8 -> H,W double to 16
    sp = build_spatial().to(dev)
    lat_sp = (torch.randn(1, 128, 2, 8, 8) * 0.5).to(device=dev, dtype=DTYPE)
    run("spatial", sp, lat_sp)

    # Small temporal latent: B=1, C=128, F=3, H=6, W=6 -> F doubles then -1 frame
    tp = build_temporal().to(dev)
    lat_tp = (torch.randn(1, 128, 3, 6, 6) * 0.5).to(device=dev, dtype=DTYPE)
    run("temporal", tp, lat_tp)

    print("DONE")


if __name__ == "__main__":
    main()
