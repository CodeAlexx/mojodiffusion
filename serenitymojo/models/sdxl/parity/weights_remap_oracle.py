#!/usr/bin/env python3
# weights_remap_oracle.py — VALUE-level reference for the OIHW->RSCF conv-weight
# remap, on a REAL SDXL checkpoint tensor. Closes the [LOW] skeptic gap
# (SKEPTIC_FINDINGS_sdxl_P1, ATTACK 3): the weights smoke checks SHAPES only, and
# an OIHW<->RSCF mismatch is shape-preserving. This gate reads the raw OIHW tensor
# from the checkpoint, applies the documented permute in numpy, and dumps the
# RSCF-flattened values. The Mojo side (_load_conv_rscf) must produce the SAME
# flat RSCF buffer element-for-element (cos==1, max_abs==0 expected; gate >=0.999).
#
# Remap (weights.mojo:66-72, matches sdxl_unet.rs weight_ocickhkw_to_khwkicoc):
#   OIHW [Cout,Cin,Kh,Kw] -> RSCF [Kh,Kw,Cin,Cout]  == permute(2,3,1,0)
#
# Tensor: input_blocks.4.0.in_layers.2.weight (320->640 3x3 conv), and the 1x1
# skip_connection.weight (the two layouts the smoke loads).
# Run: /home/alex/serenityflow-v2/.venv/bin/python .../weights_remap_oracle.py

import os
import numpy as np
import torch
from safetensors import safe_open

CKPT = "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors"
OUT = os.path.join(os.path.dirname(__file__), "weights_remap_ref.txt")

KEYS = [
    ("conv1", "input_blocks.4.0.in_layers.2.weight"),  # OIHW [640,320,3,3]
    ("skip", "input_blocks.4.0.skip_connection.weight"),  # OIHW [640,320,1,1]
]


def main():
    lines = []
    with safe_open(CKPT, framework="pt") as f:
        for tag, key in KEYS:
            wt = f.get_tensor(key)  # torch tensor, OIHW, bf16 stored
            # cast to f32 for the value comparison (Mojo loads as F32 too,
            # via cast_tensor BF16->F32 — same rounding: bf16->f32 is exact)
            w = wt.to(torch.float32).numpy()
            assert w.ndim == 4, f"{key} not rank-4: {w.shape}"
            cout, cin, kh, kw = w.shape
            # OIHW -> RSCF [Kh,Kw,Cin,Cout] = permute(2,3,1,0)
            rscf = np.transpose(w, (2, 3, 1, 0)).reshape(-1)
            lines.append(f"{tag} " + " ".join(f"{v:.8f}" for v in rscf.tolist()))
            print(f"{tag}: {key} OIHW{list(w.shape)} -> RSCF[{kh},{kw},{cin},{cout}] numel={rscf.size}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
