#!/usr/bin/env python3
# minimax_h3_lora_to_kohya.py — convert a serenitymojo H3 LoRA (canonical PEFT:
# diffusion_model.blocks.N.{attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2}.lora_A/B)
# to the kohya/musubi convention ComfyUI loads (lora_unet_{path}_.lora_down/up
# + .alpha). ComfyUI's H3 module paths match ours exactly, so this is purely a
# key/naming transform; missing alpha keys mean alpha=rank (scale 1.0), which
# this makes explicit. Verified 2026-08-27: renders through the fp8 pruned
# pipeline at 864x480 (LoraLoaderModelOnly).
#
# Usage: minimax_h3_lora_to_kohya.py <peft.safetensors> <out.safetensors>
import sys

import torch
from safetensors.torch import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
sd = load_file(src)
out = {}
for k, v in sd.items():
    if not (k.startswith("diffusion_model.") and (".lora_A.weight" in k or ".lora_B.weight" in k)):
        raise SystemExit(f"unexpected key (not H3 PEFT): {k}")
    path = k[len("diffusion_model."):]
    ab = "lora_down" if ".lora_A." in k else "lora_up"
    mod = path.split(".lora_")[0]
    base = "lora_unet_" + mod.replace(".", "_")
    out[f"{base}.{ab}.weight"] = v
    ak = f"{base}.alpha"
    if ak not in out:
        rank = v.shape[0] if ab == "lora_down" else v.shape[1]
        out[ak] = torch.tensor(float(rank))
save_file(out, dst)
print(f"{len(out)} tensors -> {dst}")
