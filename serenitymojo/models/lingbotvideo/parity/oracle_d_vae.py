#!/usr/bin/env python
# Oracle-D: AutoencoderKLWan.decode (Wan 2.1-style 3D causal VAE). Real config
# (base_dim 96, dim_mult [1,2,4,4], z_dim 16, temperal_downsample [F,T,T]),
# SEEDED synthetic weights, TINY latent so the decoded video is small. fp32
# throughout (production vae_dtype=fp32). Dumps full state_dict + z + decoded so
# the Mojo decoder gates the DECODE MATH now; real weights swap in at chunk E.
import os, sys, json
import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLWan

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
CONF = dict(base_dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
            attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
DEV = "cuda"


def main():
    torch.manual_seed(77)
    vae = AutoencoderKLWan(**CONF).to(DEV).float().eval()
    # re-seed all params small so decode is well-conditioned and every op is exercised
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_(0.05 * torch.randn_like(p))
    for p in vae.parameters():
        p.requires_grad_(False)

    z = torch.randn(1, 16, 1, 2, 2, device=DEV, dtype=torch.float32)   # (B,z,T,lh,lw), T2I single frame
    with torch.no_grad():
        dec = vae.decode(z)
        out = dec.sample if hasattr(dec, "sample") else (dec[0] if isinstance(dec, tuple) else dec)
        out = out.float()

    caps = {"z": z.float().cpu(), "out": out.float().cpu()}
    # dump the full state_dict (fp32) under real names — the Mojo loader maps these
    sd = vae.state_dict()
    for k, v in sd.items():
        if v.is_floating_point():
            caps[f"w::{k}"] = v.float().cpu().contiguous()
    save_file(caps, os.path.join(OUT, "oracle_d.safetensors"))

    names = sorted(k for k in sd.keys())
    json.dump({"config": CONF, "z_shape": list(z.shape), "out_shape": list(out.shape),
               "n_params_tensors": len(names), "weight_names_sample": names[:60]},
              open(os.path.join(OUT, "oracle_d_meta.json"), "w"), indent=2)
    print(f"SAVED oracle_d.safetensors  z{list(z.shape)} -> out{list(out.shape)}")
    print(f"  out mean {out.mean():.5f} std {out.std():.5f} absmax {out.abs().max():.4f}")
    print(f"  {len(names)} weight tensors. decoder top-level keys:")
    tops = sorted(set(k.split('.')[0] + ('.' + k.split('.')[1] if '.' in k[len(k.split('.')[0]):] else '') for k in names))
    print("   ", [k for k in sorted(set(n.split('.')[0] for n in names))])


if __name__ == "__main__":
    main()
