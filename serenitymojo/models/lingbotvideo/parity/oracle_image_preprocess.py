#!/usr/bin/env python
# oracle_image_preprocess.py — PIL/torch oracle for the pure-Mojo i2v condition
# image preprocessor. Runs the SAME preprocess used by the real i2v prep
# (prep_dance_i2v.preprocess) on a fixed test PNG and saves the expected
# [1,3,1,H,W] f32 tensor for the Mojo probe to compare against.
#
#   /home/alex/SerenityTrainer/venv/bin/python \
#     serenitymojo/models/lingbotvideo/parity/oracle_image_preprocess.py
import os
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
# Fixed test PNG (converted from an existing generated image); arbitrary input
# size is fine — preprocess cover-resizes + center-crops to (H, W).
IMG = "/home/alex/mojodiffusion/output/media/lingbotvideo/mojo_e_generated.png"
H, W = 576, 320   # i2v portrait geometry


def preprocess(path, h, w):
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    s = max(w / iw, h / ih)                       # cover
    im = im.resize((round(iw * s), round(ih * s)), Image.BICUBIC)
    rw, rh = im.size
    left, top = (rw - w) // 2, (rh - h) // 2
    im = im.crop((left, top, left + w, top + h))  # center-crop to WxH
    arr = np.asarray(im, np.float32) / 255.0      # HWC [0,1]
    chw = arr.transpose(2, 0, 1)[None, :, None]   # [1,3,1,H,W]
    return chw


def main():
    cond = preprocess(IMG, H, W)
    cond = np.ascontiguousarray(cond)
    save_file(
        {"image": torch.from_numpy(cond)},
        os.path.join(OUT, "oracle_image_preprocess.safetensors"),
    )
    print(f"[oracle] {IMG}")
    print(f"[oracle] condition {cond.shape} range [{cond.min():.4f},{cond.max():.4f}]")


if __name__ == "__main__":
    main()
