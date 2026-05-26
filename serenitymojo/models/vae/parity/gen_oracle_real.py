#!/usr/bin/env python3
# gen_oracle_real.py — SKEPTIC real-resolution oracle for the Z-Image VAE decoder.
#
# Parametrized version of gen_oracle.py. Dumps z_raw + final (and a histogram of
# the per-pixel abs-diff structure is computed on the Mojo side). Output files are
# suffixed by LHxLW so multiple sizes coexist.
#
#   /tmp/vae_oracle_venv/bin/python gen_oracle_real.py LH LW
#
# Default: 64 64  (-> 512x512 image).

import os
import sys
import numpy as np
import torch
from diffusers import AutoencoderKL

HERE = os.path.dirname(os.path.abspath(__file__))
VAE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae"
)
SEED = 1234


def dump(name, t):
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    arr.astype(np.float32).tofile(os.path.join(HERE, name + ".bin"))
    with open(os.path.join(HERE, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  dumped {name:28s} shape={tuple(t.shape)} "
          f"mean={float(t.mean()):+.4f} std={float(t.std()):+.4f} "
          f"min={float(t.min()):+.4f} max={float(t.max()):+.4f}")


def main():
    LH = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    LW = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    tag = f"_{LH}x{LW}"
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    vae.eval()
    cfg = vae.config
    scale = float(cfg.scaling_factor)
    shift = float(cfg.shift_factor)
    print(f"[{LH}x{LW}] scaling={scale} shift={shift}")

    torch.manual_seed(SEED)
    z_raw = torch.randn(1, cfg.latent_channels, LH, LW, dtype=torch.float32)
    dump("z_raw" + tag, z_raw)
    z = z_raw / scale + shift

    with torch.no_grad():
        img = vae.decode(z).sample
    dump("final" + tag, img)
    print(f"[{LH}x{LW}] done -> {HERE}")


if __name__ == "__main__":
    main()
