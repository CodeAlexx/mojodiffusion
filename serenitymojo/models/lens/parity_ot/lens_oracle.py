#!/usr/bin/env python
"""Lens OneTrainer oracle — dumps parity references from OneTrainer's OWN deps.

Reference policy: OneTrainer ONLY. The `lens` package (lens/transformer.py) is
OneTrainer's actual dependency (modules/model/LensModel.py imports it), so it is
the 1:1 forward-math source. NO Rust / inference-flame.

Run with OneTrainer's venv:
  /home/alex/OneTrainer/venv/bin/python serenitymojo/models/lens/parity_ot/lens_oracle.py

Dumps (into this dir):
  dit_fwd_in_hidden.safetensors   [1, S_img, 128]   f32
  dit_fwd_in_txt_{0..3}.safetensors [1, S_txt, 2880] f32   (per selected layer)
  dit_fwd_in_mask.safetensors     [1, S_txt]        f32 (1.0 = valid)
  dit_fwd_in_timestep.safetensors [1]               f32   (in [0,1])
  dit_fwd_out.safetensors         [1, S_img, 128]   f32   (proj_out output)
  vae_bn.safetensors              running_mean/var  f32   (scale_latents stats)
  meta.json                       shapes, seeds, config, timestep-shift constants
"""
import importlib.util
import json
import os

import torch
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/microsoft_lens"
OUT = os.path.dirname(os.path.abspath(__file__))
LENS_TRANSFORMER_PY = "/home/alex/vendor-refs/Lens/lens/transformer.py"

# small but full-coverage geometry: exercises all 48 blocks, RoPE, mask,
# multi-layer text concat, joint SDPA — without 4096-token cost on CPU.
S_IMG_H, S_IMG_W = 8, 8          # img_shapes = (1, 8, 8) -> 64 image tokens
S_TXT = 16                       # text tokens (post-crop)
SEED = 1234


def load_dit():
    spec = importlib.util.spec_from_file_location("lens_transformer", LENS_TRANSFORMER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # transformer.py only needs torch + diffusers (no einops)
    cls = mod.LensTransformer2DModel
    m = cls.from_pretrained(CKPT, subfolder="transformer", torch_dtype=torch.float32)
    return m.eval()


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)

    m = load_dit()
    cfg = dict(m.config)
    n_layers = len(cfg["selected_layer_index"])
    enc_dim = cfg["enc_hidden_dim"]          # 2880
    in_ch = cfg["in_channels"]               # 128

    hidden = torch.randn(1, S_IMG_H * S_IMG_W, in_ch, generator=g, dtype=torch.float32)
    txt = [torch.randn(1, S_TXT, enc_dim, generator=g, dtype=torch.float32) for _ in range(n_layers)]
    mask = torch.ones(1, S_TXT, dtype=torch.bool)
    timestep = torch.tensor([0.5], dtype=torch.float32)
    img_shapes = [(1, S_IMG_H, S_IMG_W)]

    with torch.no_grad():
        out = m(
            hidden_states=hidden,
            encoder_hidden_states=txt,
            encoder_hidden_states_mask=mask,
            timestep=timestep,
            img_shapes=img_shapes,
        )

    save_file({"x": hidden.contiguous()}, os.path.join(OUT, "dit_fwd_in_hidden.safetensors"))
    for i, t in enumerate(txt):
        save_file({"x": t.contiguous()}, os.path.join(OUT, f"dit_fwd_in_txt_{i}.safetensors"))
    save_file({"x": mask.float().contiguous()}, os.path.join(OUT, "dit_fwd_in_mask.safetensors"))
    save_file({"x": timestep.contiguous()}, os.path.join(OUT, "dit_fwd_in_timestep.safetensors"))
    save_file({"x": out.float().contiguous()}, os.path.join(OUT, "dit_fwd_out.safetensors"))

    # VAE batch-norm scaling stats (LensModel.scale_latents) — read from config-only would
    # need weights; load the vae and pull the bn buffers.
    bn = {}
    try:
        from diffusers import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_pretrained(CKPT, subfolder="vae", torch_dtype=torch.float32).eval()
        bn["running_mean"] = vae.bn.running_mean.float().contiguous()
        bn["running_var"] = vae.bn.running_var.float().contiguous()
        bn_eps = float(vae.config.batch_norm_eps)
        save_file(bn, os.path.join(OUT, "vae_bn.safetensors"))
    except Exception as e:
        bn_eps = None
        print("VAE bn dump skipped:", repr(e))

    meta = {
        "seed": SEED,
        "s_img_h": S_IMG_H, "s_img_w": S_IMG_W, "s_img_tokens": S_IMG_H * S_IMG_W,
        "s_txt": S_TXT,
        "img_shapes": img_shapes,
        "config": cfg,
        "out_shape": list(out.shape),
        "out_mean": float(out.mean()), "out_std": float(out.std()),
        "out_absmax": float(out.abs().max()),
        "batch_norm_eps": bn_eps,
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("ORACLE OK")
    print(f"  out shape {tuple(out.shape)} mean {meta['out_mean']:.6f} "
          f"std {meta['out_std']:.6f} absmax {meta['out_absmax']:.6f}")
    print(f"  dumped to {OUT}")


if __name__ == "__main__":
    main()
