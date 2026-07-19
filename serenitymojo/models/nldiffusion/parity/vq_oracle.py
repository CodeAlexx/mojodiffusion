#!/usr/bin/env python
# Oracle for chunk D: Emu3 VQ-VAE DECODE path (token ids -> pixels).
# Loads ONLY the emu3_vqvae (1.8GB fp32) standalone on GPU (the main model
# offloads it to meta). Reference authority = NVIDIA creator source.
import os, sys, json, warnings, types, importlib.util
warnings.filterwarnings("ignore")
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

VQ = "/mnt/disk1/models/NL-Diffusion-Image/emu3_vqvae"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"
SEED = 42

def load_vqvae(vqvae_path: Path):
    pkg = f"_emu3_vqvae_{vqvae_path.name}"
    pkg_mod = types.ModuleType(pkg); pkg_mod.__path__=[str(vqvae_path)]; pkg_mod.__package__=pkg
    sys.modules[pkg]=pkg_mod
    def _load_mod(mod_name, filename):
        spec = importlib.util.spec_from_file_location(f"{pkg}.{mod_name}", vqvae_path/filename,
                                                      submodule_search_locations=[str(vqvae_path)])
        mod = importlib.util.module_from_spec(spec); mod.__package__=pkg
        sys.modules[f"{pkg}.{mod_name}"]=mod; spec.loader.exec_module(mod); return mod
    cfg_mod=_load_mod("configuration_emu3p5visionvq","configuration_emu3p5visionvq.py")
    mdl_mod=_load_mod("modeling_emu3p5visionvq","modeling_emu3p5visionvq.py")
    cfg_data=json.load(open(vqvae_path/"config.json"))
    vqvae=mdl_mod.Emu3p5VisionVQModel(cfg_mod.Emu3p5VisionVQConfig(**cfg_data))
    vqvae.load_state_dict(load_file(str(vqvae_path/"model.safetensors")))
    return vqvae

def main():
    torch.manual_seed(SEED)
    dev="cuda"
    vqvae=load_vqvae(Path(VQ)).to(dev).float().eval()
    for p in vqvae.parameters(): p.requires_grad_(False)
    cbk=vqvae.config.codebook_size
    h16=w16=16
    ids=torch.randint(0,cbk,(1,h16*w16),dtype=torch.long,device=dev)
    caps={}; meta={}
    def sv(n,t,note=""):
        caps[n]=t.detach().cpu().float().contiguous()
        meta[n]={"shape":list(t.shape),"dtype":str(t.dtype),"note":note}
        print(f"  [tap] {n:26s} {list(t.shape)} {t.dtype} {note}")
    sv("vqdec.ids", ids.float(), "input token ids [1,256]")
    from einops import rearrange
    with torch.no_grad():
        cb=vqvae.quantize.get_codebook_entry(ids)
        sv("vqdec.codebook_entry", cb, "[1,256,256] embed gather")
        z=rearrange(cb,"b (h w) d -> b d h w",h=h16,w=w16)
        sv("vqdec.z", z, "[1,256,16,16] pre-decode")
        dec=vqvae.decode(z).float()
        sv("vqdec.decoded_raw", dec, "[1,3,256,256] decoder out")
        sv("vqdec.decoded_clamped", dec.clamp(-1,1), "clamped[-1,1]")
    save_file(caps, os.path.join(OUT,"vq_oracle.safetensors"))
    json.dump(meta, open(os.path.join(OUT,"vq_oracle_meta.json"),"w"), indent=2, default=str)
    print(f"SAVED {len(caps)} tensors -> {OUT}/vq_oracle.safetensors")

if __name__=="__main__":
    main()
