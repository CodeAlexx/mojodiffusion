#!/usr/bin/env python3
# gen_oracle.py — diffusers oracle for the Z-Image (ldm_decoder) VAE decoder.
#
# DEV-ONLY oracle (pure-Rust/Mojo runtime rule: Python never runs in the
# shipped path). Loads the Z-Image AutoencoderKL with diffusers, decodes a
# FIXED-seed latent, and dumps:
#   * the input latent z (already rescaled: z/scale + shift), so the Mojo
#     `decode` rescale step is validated end-to-end,
#   * the final decoded image,
#   * per-block intermediate activations (conv_in, mid, each up_block, final),
# each as a flat float32 .bin with a sidecar .shape text file.
#
# Run with the scratch venv:
#   /tmp/vae_oracle_venv/bin/python gen_oracle.py
#
# The Mojo parity harness reads these .bin/.shape files for cos+max_abs.

import os
import struct
import numpy as np
import torch
from diffusers import AutoencoderKL

HERE = os.path.dirname(os.path.abspath(__file__))
VAE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae"
)

# Small latent so the full decode + per-block dumps stay tiny and CPU-fast.
LH, LW = 8, 8
SEED = 1234


def dump(name, t):
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    arr.astype(np.float32).tofile(os.path.join(HERE, name + ".bin"))
    with open(os.path.join(HERE, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  dumped {name:24s} shape={tuple(t.shape)} "
          f"mean={float(t.mean()):+.4f} std={float(t.std()):+.4f}")


def main():
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    vae.eval()
    cfg = vae.config
    scale = float(cfg.scaling_factor)
    shift = float(cfg.shift_factor)
    print(f"scaling={scale} shift={shift} latent_ch={cfg.latent_channels} "
          f"post_quant={cfg.use_post_quant_conv}")

    torch.manual_seed(SEED)
    # Raw model-space latent (what the sampler would produce).
    z_raw = torch.randn(1, cfg.latent_channels, LH, LW, dtype=torch.float32)
    dump("z_raw", z_raw)

    # ldm_decoder.rs decode(): z = z/scale + shift, then decoder(z).
    z = z_raw / scale + shift
    dump("z_rescaled", z)

    dec = vae.decoder
    hooks_out = {}

    with torch.no_grad():
        # Manually walk decoder to capture per-block intermediates, matching
        # diffusers AutoencoderKL.Decoder.forward (no post_quant_conv here).
        h = dec.conv_in(z)
        dump("conv_in", h)
        h = dec.mid_block(h)
        dump("mid_block", h)
        for i, up in enumerate(dec.up_blocks):
            h = up(h)
            dump(f"up_block_{i}", h)
        h = dec.conv_norm_out(h)
        dump("after_norm_out", h)
        h = dec.conv_act(h)
        dump("after_silu", h)
        h = dec.conv_out(h)
        dump("final", h)

        # Cross-check: the full vae.decode path should match `final`.
        img = vae.decode(z).sample
        d = float((img - h).abs().max())
        print(f"  full-decode vs manual-walk max_abs_diff = {d:.3e}")

    print("oracle dump complete ->", HERE)


if __name__ == "__main__":
    main()
