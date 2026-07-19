#!/usr/bin/env python
# Oracle for the YaRN RoPE tables (chunk A dependency). Instantiates ONLY the
# Ministral3RotaryEmbedding (CPU, no weights) and dumps cos/sin for positions
# 0..1140 (the seq len of the 1024^2 oracle forward: 117 prompt + 1024 gen),
# plus inv_freq, attention_scaling, and the Llama-4 post-rope query scale.
import os, json, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from safetensors.torch import save_file

DIR = "/mnt/disk1/models/NL-Diffusion-Image"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"
SEQ = 1141

def main():
    cfg = AutoConfig.from_pretrained(DIR, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    RE = get_class_from_dynamic_module("modeling_ministral.Ministral3RotaryEmbedding", DIR)
    rope = RE(config=cfg)   # builds inv_freq via ROPE_INIT_FUNCTIONS['yarn']
    pos = torch.arange(SEQ, dtype=torch.long)[None]     # [1, SEQ]
    x = torch.zeros(1, SEQ, 1, dtype=torch.bfloat16)    # only used for dtype/device
    with torch.no_grad():
        cos, sin = rope(x, pos)                          # [1,SEQ,head_dim] bf16
    # llama-4 post-rope query length scale (identity for pos<orig_max=16384)
    beta = cfg.rope_parameters.get("llama_4_scaling_beta")
    omax = cfg.rope_parameters.get("original_max_position_embeddings")
    if beta and omax:
        l4 = 1 + beta * torch.log(1 + torch.floor(pos.float() / omax))
    else:
        l4 = torch.ones_like(pos.float())
    caps = {
        "rope.cos": cos.detach().cpu().float().contiguous(),         # [1,SEQ,128]
        "rope.sin": sin.detach().cpu().float().contiguous(),
        "rope.inv_freq": rope.inv_freq.detach().cpu().float().contiguous(),  # [64]
        "rope.attention_scaling": torch.tensor([float(rope.attention_scaling)]),
        "rope.llama4_scale": l4.detach().cpu().float().contiguous(),  # [1,SEQ]
        "rope.positions": pos.detach().cpu().float().contiguous(),
    }
    meta = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in caps.items()}
    meta["attention_scaling_value"] = float(rope.attention_scaling)
    meta["rope_type"] = cfg.rope_parameters["rope_type"]
    save_file(caps, os.path.join(OUT, "rope_oracle.safetensors"))
    json.dump(meta, open(os.path.join(OUT, "rope_oracle_meta.json"), "w"), indent=2, default=str)
    print("attention_scaling =", float(rope.attention_scaling))
    print("inv_freq[:4] =", rope.inv_freq[:4].tolist())
    print("cos[0,0,:4] =", cos[0,0,:4].tolist(), " cos[0,1000,:4] =", cos[0,1000,:4].tolist())
    print("llama4_scale unique =", torch.unique(l4).tolist())
    print(f"SAVED -> {OUT}/rope_oracle.safetensors  cos{list(cos.shape)} sin{list(sin.shape)}")

if __name__ == "__main__":
    main()
