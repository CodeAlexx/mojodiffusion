#!/usr/bin/env python3
"""Emit the chunk-3 loader-parity expectations: oracle W_hat (bf16) for two
representative layers of the BUILT Klein-4B squareq slab, computed by the
Python byte-level oracle from the exact slab bytes the Mojo loader will pin.

Output: <slab_dir>/loader_expected.safetensors
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from squareq import core  # noqa: E402

SLAB = sys.argv[1] if len(sys.argv) > 1 else "models/klein4b/squareq_w4_r32"
KEYS = ["double_blocks.0.img_attn.qkv", "single_blocks.0.linear2"]


def main():
    with open(os.path.join(SLAB, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    out = {}
    for base in KEYS:
        shard = wm[base + ".qweight"]
        with safe_open(os.path.join(SLAB, shard), "pt") as f:
            w_hat = core.reconstruct_weight(
                f.get_tensor(base + ".qweight"),
                f.get_tensor(base + ".wscales"),
                f.get_tensor(base + ".lora_down"),
                f.get_tensor(base + ".lora_up"),
            )
        out[base + ".w_hat"] = w_hat.to(torch.bfloat16).contiguous()
        print(f"{base}: w_hat {list(w_hat.shape)}")
    dst = os.path.join(SLAB, "loader_expected.safetensors")
    save_file(out, dst)
    print("wrote", dst)


if __name__ == "__main__":
    main()
