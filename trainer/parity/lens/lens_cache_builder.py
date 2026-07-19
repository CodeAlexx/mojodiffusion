#!/usr/bin/env python
"""Build a real single-sample Lens safetensors cache for the Mojo dataLoader gate.

Serenity's on-disk cache is torch-pickle .pt (not Mojo-readable); per the
constraints memo we dump an equivalent safetensors cache the Mojo CacheReader
consumes. Keys (LensBaseDataLoader cache fields):
  latent_image              [32, H/8, W/8]   (VAE mean latent, pre-scale, pre-patchify)
  text_encoder_hidden_state [L, 4*2880=11520] (EncodeLensText: 4 layers concat dim=-1, cropped)
  tokens_mask               [L]

Run: /home/alex/ai-toolkit/venv/bin/python parity/lens/lens_cache_builder.py
"""
import importlib.util, json, os
import numpy as np, torch
from PIL import Image
from safetensors.torch import load_file, save_file

HERE=os.path.dirname(os.path.abspath(__file__)); CKPT="/home/alex/.serenity/models/microsoft_lens"
IMG=os.path.join(HERE,"gen","lens_mojo.png")
OUT=os.path.join(HERE,"cache"); os.makedirs(OUT,exist_ok=True)


def main():
    # text_encoder_hidden_state = concat(feat_0..3, dim=-1) from loss_ref (already cropped, [201,2880] each)
    r=load_file(HERE+"/loss_ref.safetensors")
    feats=[r[f"feat_{i}"][0].float() for i in range(4)]   # each [201,2880]
    tehs=torch.cat(feats,dim=-1)                            # [201, 11520]
    L=tehs.shape[0]
    tokens_mask=torch.ones(L,dtype=torch.int32)

    # latent_image: VAE mean of the real image
    from diffusers import AutoencoderKLFlux2
    vae=AutoencoderKLFlux2.from_pretrained(CKPT+"/vae",torch_dtype=torch.float32).to("cuda").eval()
    im=Image.open(IMG).convert("RGB")
    x=torch.from_numpy(np.asarray(im)).float().permute(2,0,1)[None]/127.5-1.0
    with torch.no_grad():
        lat=vae.encode(x.to("cuda")).latent_dist.mode()[0].float().cpu()   # [32, H/8, W/8]

    save_file({"latent_image":lat.contiguous(),
               "text_encoder_hidden_state":tehs.contiguous(),
               "tokens_mask":tokens_mask.contiguous()},
              os.path.join(OUT,"lens_sample.safetensors"))
    json.dump({"latent_image_shape":list(lat.shape),"tehs_shape":list(tehs.shape),
               "L":int(L),"hidden_concat":11520,"img":IMG,
               "note":"single-sample Lens cache; tehs = 4-layer GPT-OSS concat (cropped)"},
              os.path.join(OUT,"lens_sample_meta.json"),"w" if False else open(os.path.join(OUT,"lens_sample_meta.json"),"w"),indent=2)
    print("CACHE BUILT")
    print(f"  latent_image {list(lat.shape)}  text_encoder_hidden_state {list(tehs.shape)}  tokens_mask [{L}]")
    print(f"  → {OUT}/lens_sample.safetensors")


if __name__=="__main__":
    main()
