#!/usr/bin/env python
# Render the Mojo dense pipeline pixels (dense_mojo_pixels.safetensors, key
# "pixels", [1,3,1,480,832] in [0,1]) to dense_mojo_generated.png. Oracle-only.
import os
import numpy as np
from safetensors import safe_open
from PIL import Image

D = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
with safe_open(os.path.join(D, "dense_mojo_pixels.safetensors"), framework="np") as f:
    px = f.get_tensor("pixels")            # [1,3,1,480,832]
# The updated probe saves [0,1] pixels; if a stale run saved raw [-1,1], remap.
if float(px.min()) < -0.01:
    px = (np.clip(px, -1.0, 1.0) + 1.0) / 2.0
img = px[0, :, 0].transpose(1, 2, 0)       # H,W,3
img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
Image.fromarray(img).save(os.path.join(D, "dense_mojo_generated.png"))
print("SAVED dense_mojo_generated.png  shape", img.shape,
      "mean", float(px.mean()), "std", float(px.std()))
