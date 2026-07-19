#!/usr/bin/env python
# Decode the Oracle-2 reference token grid -> 1024^2 PNG (visual ref for chunk F).
# Standalone VQ decoder (1.8GB fp32) so it fits 16GB after the main model is gone.
import os, sys, json, types, importlib.util, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
from PIL import Image

VQ = "/mnt/disk1/models/NL-Diffusion-Image/emu3_vqvae"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"

def load_vqvae(vqvae_path: Path):
    pkg = f"_emu3_vqvae_{vqvae_path.name}"
    m = types.ModuleType(pkg); m.__path__=[str(vqvae_path)]; m.__package__=pkg; sys.modules[pkg]=m
    def L(n, f):
        s = importlib.util.spec_from_file_location(f"{pkg}.{n}", vqvae_path/f, submodule_search_locations=[str(vqvae_path)])
        mm = importlib.util.module_from_spec(s); mm.__package__=pkg; sys.modules[f"{pkg}.{n}"]=mm; s.loader.exec_module(mm); return mm
    cfg = L("configuration_emu3p5visionvq","configuration_emu3p5visionvq.py")
    mdl = L("modeling_emu3p5visionvq","modeling_emu3p5visionvq.py")
    cfgd = json.load(open(vqvae_path/"config.json"))
    v = mdl.Emu3p5VisionVQModel(cfg.Emu3p5VisionVQConfig(**cfgd))
    v.load_state_dict(load_file(str(vqvae_path/"model.safetensors")))
    return v

def main():
    dev = "cuda"
    ids = load_file(os.path.join(OUT, "oracle2_tokens.safetensors"))["x0_img"].long().to(dev)  # [1,4096]
    v = load_vqvae(Path(VQ)).to(dev).float().eval()
    for p in v.parameters(): p.requires_grad_(False)
    from einops import rearrange
    with torch.no_grad():
        cb = v.quantize.get_codebook_entry(ids)                 # [1,4096,256]
        z = rearrange(cb, "b (h w) d -> b d h w", h=64, w=64)   # [1,256,64,64]
        img = v.decode(z).float().clamp(-1, 1)                  # [1,3,1024,1024]
    arr = ((img + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    Image.fromarray(arr).save(os.path.join(OUT, "oracle2_reference_1024.png"))
    print(f"SAVED {OUT}/oracle2_reference_1024.png  shape={arr.shape} range[{arr.min()},{arr.max()}]")

if __name__ == "__main__":
    main()
