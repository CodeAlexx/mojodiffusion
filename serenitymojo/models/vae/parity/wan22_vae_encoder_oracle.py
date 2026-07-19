# wan22_vae_encoder_oracle.py — torch reference for the Wan2.2 high-compression
# VAE ENCODER (z_dim=48, c_dim=160). DEV-ONLY parity oracle. Never shipped.
#
# Builds the real `WanVAE_` (Lance/modeling/vae/wan/vae2_2.py), loads the
# converted safetensors (same bytes the Mojo encoder reads), and encodes a FIXED
# non-degenerate RGB image (in [-1,1]) as a single video frame [1,3,1,H,W].
#
# DTYPE: matches the Phase-0 lesson — model + input in BF16 on GPU (the Wan VAE
# runs BF16 with F32-accumulate inside conv kernels; the public encode()
# normalizes mu in the model dtype then Wan2_2_VAE.encode returns .float()).
# We run WanVAE_.encode directly in BF16, returning the normalized mu (48-ch).
#
# Dumps (F32 little-endian):
#   wan22enc_img_<HxW>.bin        [1,3,H,W]          the fixed input image
#   wan22enc_mu_<HxW>.bin         [1,48,1,H/16,W/16] normalized latent mu (mode)
#   wan22enc_meta_<HxW>.json      shapes + stats
#
# Run with system python3 (torch cu128 + einops). NOT the pixi env.
#
#   python3 serenitymojo/models/vae/parity/wan22_vae_encoder_oracle.py [H] [W]

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/alex/Lance/modeling/vae/wan")
import vae2_2  # noqa: E402

SAFET = "/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors"
OUTDIR = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"

# Per-channel latents_mean / latents_std — hardcoded in Wan2_2_VAE.__init__.
MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


def load_state_dict(path):
    from safetensors.torch import load_file
    return load_file(path)


def main():
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    # Optional 3rd arg: number of input frames T (>1 -> dump T2V temporal case).
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    dev = "cuda"
    dt = torch.bfloat16

    # Build WanVAE_ with the production config (z_dim=48, c_dim=160).
    model = vae2_2.WanVAE_(
        dim=160,
        dec_dim=256,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    sd = load_state_dict(SAFET)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    enc_missing = [
        m for m in missing
        if m.startswith("encoder.") or m in ("conv1.weight", "conv1.bias")
    ]
    assert not enc_missing, f"encoder keys not filled: {enc_missing[:8]}"
    print(f"[oracle] loaded; encoder fully covered "
          f"(missing total {len(missing)}, unexpected {len(unexpected)})")

    model = model.eval().to(dev, dtype=dt)
    for p in model.parameters():
        p.requires_grad_(False)

    mean = torch.tensor(MEAN, dtype=dt, device=dev)
    std = torch.tensor(STD, dtype=dt, device=dev)
    scale = [mean, 1.0 / std]

    g = torch.Generator(device="cpu").manual_seed(1234)
    vid01 = torch.rand(1, 3, T, H, W, generator=g, dtype=torch.float32)
    vid = vid01 * 2.0 - 1.0  # [-1,1], [1,3,T,H,W]
    x5 = vid.to(dev, dtype=dt)  # [1,3,T,H,W]

    with torch.no_grad():
        mu, log_var = model.encode(x5, scale)  # normalized mu [1,48,T',H/16,W/16]
    mu_f = mu.float().cpu().contiguous()
    print(f"[oracle] vid {tuple(vid.shape)} -> mu {tuple(mu_f.shape)} "
          f"mean={mu_f.mean().item():.4f} std={mu_f.std().item():.4f} "
          f"|mean|={mu_f.abs().mean().item():.4f} "
          f"range=[{mu_f.min().item():.3f},{mu_f.max().item():.3f}]")

    os.makedirs(OUTDIR, exist_ok=True)
    # tag: image cases keep the legacy HxW name; temporal cases add _tT.
    tag = f"{H}x{W}" if T == 1 else f"{H}x{W}_t{T}"
    vid.cpu().contiguous().numpy().astype("<f4").tofile(
        f"{OUTDIR}/wan22enc_img_{tag}.bin")
    mu_f.numpy().astype("<f4").tofile(f"{OUTDIR}/wan22enc_mu_{tag}.bin")
    with open(f"{OUTDIR}/wan22enc_meta_{tag}.json", "w") as f:
        json.dump({
            "H": H, "W": W, "T": T,
            "img_shape": list(vid.shape),
            "mu_shape": list(mu_f.shape),
            "mu_mean": float(mu_f.mean().item()),
            "mu_std": float(mu_f.std().item()),
            "mu_absmean": float(mu_f.abs().mean().item()),
        }, f, indent=2)
    print(f"[oracle] wrote wan22enc_img_{tag}.bin, wan22enc_mu_{tag}.bin")


if __name__ == "__main__":
    main()
