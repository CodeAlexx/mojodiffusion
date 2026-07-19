#!/usr/bin/env python
# Oracle-Dvid: AutoencoderKLWan.decode in TEMPORAL (multi-frame video) mode. Real
# config (base_dim 96, dim_mult [1,2,4,4], z_dim 16, temperal_downsample [F,T,T]),
# SEEDED synthetic weights (SAME seed/scale as oracle_d_vae.py so the image-mode
# gate stays valid), a MULTI-FRAME latent so the full causal temporal machinery
# runs: time_conv + DupUp temporal interleave + the causal feat-cache across
# latent frames. 2 latent frames -> (2-1)*4+1 = 5 pixel frames. fp32 throughout.
#
# Dumps full state_dict + z + decoded so the Mojo temporal decoder gates the
# DECODE MATH now (download-independent); real weights swap in for T2V.
#
# Run:
#   /home/alex/SerenityTrainer/venv/bin/python \
#     /home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/oracle_dvid_vae.py
import os, json
import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLWan

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
CONF = dict(base_dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
            attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
DEV = "cuda"
LF = 2   # latent frames -> (LF-1)*4+1 = 5 pixel frames
LH = 4
LW = 4


def main():
    torch.manual_seed(77)
    vae = AutoencoderKLWan(**CONF).to(DEV).float().eval()
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_(0.05 * torch.randn_like(p))
    for p in vae.parameters():
        p.requires_grad_(False)

    torch.manual_seed(1234)
    z = torch.randn(1, 16, LF, LH, LW, device=DEV, dtype=torch.float32)
    with torch.no_grad():
        dec = vae.decode(z)
        out = dec.sample if hasattr(dec, "sample") else (dec[0] if isinstance(dec, tuple) else dec)
        out = out.float()

    caps = {"z": z.float().cpu(), "out": out.float().cpu()}
    sd = vae.state_dict()
    for k, v in sd.items():
        if v.is_floating_point():
            caps[f"w::{k}"] = v.float().cpu().contiguous()
    save_file(caps, os.path.join(OUT, "oracle_dvid.safetensors"))

    json.dump({"config": CONF, "z_shape": list(z.shape), "out_shape": list(out.shape),
               "LF": LF, "LH": LH, "LW": LW},
              open(os.path.join(OUT, "oracle_dvid_meta.json"), "w"), indent=2)
    print(f"SAVED oracle_dvid.safetensors  z{list(z.shape)} -> out{list(out.shape)}")
    print(f"  out mean {out.mean():.5f} std {out.std():.5f} absmax {out.abs().max():.4f}")


if __name__ == "__main__":
    main()
