#!/usr/bin/env python3
# zimage_oracle_velt.py — FIXED latent (noise), VARY t: dumps diffusers
# transformer velocity for (noise, cond) at several t values, to isolate pure
# t-dependence of the DiT velocity from latent-dependence.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import numpy as np
from diffusers import ZImagePipeline

ZROOT = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
)
PD = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity"
TS = [0.0, 0.13462, 0.30435, 0.82353]


def load_bin(name):
    shape = tuple(int(x) for x in open(os.path.join(PD, name + ".shape")).read().split(","))
    return torch.from_numpy(np.fromfile(os.path.join(PD, name + ".bin"), dtype="<f4").reshape(shape))


def dump(name, t):
    t.detach().to(torch.float32).contiguous().cpu().numpy().ravel().tofile(os.path.join(PD, name + ".bin"))
    open(os.path.join(PD, name + ".shape"), "w").write(",".join(str(int(x)) for x in t.shape))


pipe = ZImagePipeline.from_pretrained(ZROOT, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16)
pipe.to("cuda")
noise = load_bin("noise").to("cuda", torch.bfloat16)   # [1,16,32,32]
cond = load_bin("cond").to("cuda", torch.bfloat16)     # [173,2560]

# transformer call convention (from pipeline): x -> unsqueeze frame dim -> unbind batch
# FRESH single-sample velocity for the EXACT (latent,t) pairs that parity_vel_vs_t
# flagged (it used the batched-hook velc_NN as ref). vf_k = velocity(latent_k, t_k).
PAIRS = [("noise", 0.0), ("lat_step_06", 0.05036), ("lat_step_13", 0.13462),
         ("lat_step_20", 0.30435), ("lat_step_27", 0.82353)]
uncond = load_bin("uncond").to("cuda", torch.bfloat16)  # [8,2560]
with torch.no_grad():
    for k, (name, t) in enumerate(PAIRS):
        lat = load_bin(name).to("cuda", torch.bfloat16)
        x_list = list(lat.unsqueeze(2).unbind(dim=0))
        tt = torch.tensor([t], dtype=torch.bfloat16, device="cuda")
        out = pipe.transformer(x_list, tt, [cond], return_dict=False)[0]
        dump(f"vf_{k}", out[0] if isinstance(out, (list, tuple)) else out)
        outu = pipe.transformer(x_list, tt, [uncond], return_dict=False)[0]
        dump(f"vfu_{k}", outu[0] if isinstance(outu, (list, tuple)) else outu)
        torch.cuda.empty_cache()
        print(f"[velt] vf_{k}/vfu_{k} = vel({name}, t={t}) cond+uncond")
print("[velt] done")
