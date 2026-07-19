#!/usr/bin/env python
"""Lens sampler oracle (Serenity-only): schedule + VAE decode-tail references.

Schedule: FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=linspace(1,1/N,N),
mu=compute_empirical_mu(seq,N)) — exactly LensSampler.__sample_base / pipeline.py:483-485.
Decode tail: the EXACT LensSampler.py:139-147 order — unpack_latents -> unscale_latents
-> unpatchify_latents -> vae.decode (plain AutoencoderKLFlux2; NO internal unscale). Dumping
a fixed packed "scaled" latent -> final image catches a double-unscale in the Mojo tail.

Run: /home/alex/Serenity/venv/bin/python parity/lens/lens_sampler_oracle.py
"""
import json
import os
import numpy as np
import torch
from safetensors.torch import save_file, load_file

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/home/alex/.serenity/models/microsoft_lens"

# small spatial so CPU decode is cheap: H=W=128 -> latent 16x16 (32ch) -> packed [1,64,128]
H = W = 128
VAE_SF = 8
NCH = 32
PATCH = 2
STEPS = 20
SEED = 4242


def compute_empirical_mu(image_seq_len, num_steps):
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def main():
    # ---- schedule ----
    from diffusers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(CKPT, subfolder="scheduler")
    h_lat, w_lat = H // VAE_SF // PATCH, W // VAE_SF // PATCH        # 8,8
    seq_len = h_lat * w_lat                                          # 64
    mu = compute_empirical_mu(seq_len, STEPS)
    sigmas = np.linspace(1.0, 1.0 / STEPS, STEPS)
    sched.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)
    sched_ref = {
        "H": H, "W": W, "steps": STEPS, "image_seq_len": seq_len, "mu": mu,
        "raw_sigmas": sigmas.tolist(),
        "shifted_sigmas": [float(s) for s in sched.sigmas.tolist()],
        "timesteps": [float(t) for t in sched.timesteps.tolist()],
    }
    json.dump(sched_ref, open(os.path.join(HERE, "sampler_schedule_ref.json"), "w"), indent=2)

    # ---- VAE decode tail (LensSampler.py:139-147 order) ----
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(CKPT, subfolder="vae", torch_dtype=torch.float32).eval()
    bn_mean = vae.bn.running_mean.float().view(1, -1, 1, 1)          # [1,128,1,1]
    bn_std = torch.sqrt(vae.bn.running_var.float() + float(vae.config.batch_norm_eps)).view(1, -1, 1, 1)

    g = torch.Generator().manual_seed(SEED)
    packed = torch.randn(1, seq_len, NCH * PATCH * PATCH, generator=g, dtype=torch.float32)   # [1,64,128]

    def unpack(latents, h, w):
        b, s, c = latents.shape
        return latents.reshape(b, h, w, c).permute(0, 3, 1, 2)        # [1,128,8,8]

    def unscale(z):
        return z * bn_std + bn_mean

    def unpatchify(z):
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
        return z.reshape(b, c // 4, h * 2, w * 2)                     # [1,32,16,16]

    with torch.no_grad():
        lat = unpack(packed, h_lat, w_lat)
        lat = unscale(lat)
        lat = unpatchify(lat)
        decoded = vae.decode(lat, return_dict=False)[0]              # [1,3,128,128]

    save_file({"x": packed.contiguous()}, os.path.join(HERE, "sampler_tail_in.safetensors"))
    save_file({"x": lat.contiguous()}, os.path.join(HERE, "sampler_vae_in.safetensors"))
    save_file({"x": decoded.float().contiguous()}, os.path.join(HERE, "sampler_tail_out.safetensors"))
    meta = {
        "seed": SEED, "h_lat": h_lat, "w_lat": w_lat, "n_latent_channels": NCH,
        "packed_shape": list(packed.shape), "vae_in_shape": list(lat.shape),
        "image_shape": list(decoded.shape),
        "image_mean": float(decoded.mean()), "image_std": float(decoded.std()),
        "image_min": float(decoded.min()), "image_max": float(decoded.max()),
        "tail_order": "unpack -> unscale -> unpatchify -> vae.decode (LensSampler.py:139-147)",
    }
    json.dump(meta, open(os.path.join(HERE, "sampler_tail_meta.json"), "w"), indent=2)
    print("SAMPLER ORACLE OK")
    print(f"  schedule: mu={mu:.6f} seq_len={seq_len} steps={STEPS}  "
          f"shifted_sigma[0..2]={sched_ref['shifted_sigmas'][:3]}")
    print(f"  tail: packed{list(packed.shape)} -> vae_in{list(lat.shape)} -> image{list(decoded.shape)} "
          f"mean {meta['image_mean']:.4f} std {meta['image_std']:.4f} range [{meta['image_min']:.3f},{meta['image_max']:.3f}]")


if __name__ == "__main__":
    main()
