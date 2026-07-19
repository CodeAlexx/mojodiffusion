#!/usr/bin/env python
# Extract N evenly-spaced PNG frames from a pixels safetensors [1,3,F,H,W] (values
# in [0,1]) for visual inspection. Usage:
#   python extract_frames.py <pixels.safetensors> <out_dir> <out_prefix> [n_frames] [tensor_key]
import sys, os
import numpy as np
from safetensors import safe_open
from PIL import Image

path = sys.argv[1]
out_dir = sys.argv[2]
prefix = sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 5
key = sys.argv[5] if len(sys.argv) > 5 else "pixels"

os.makedirs(out_dir, exist_ok=True)
with safe_open(path, framework="np") as f:
    px = np.asarray(f.get_tensor(key), dtype=np.float32)   # [1,3,F,H,W]
_, C, F, H, W = px.shape
idxs = [int(round(i * (F - 1) / max(1, n - 1))) for i in range(n)] if n > 1 else [0]
saved = []
for j, fi in enumerate(idxs):
    img = px[0, :, fi].transpose(1, 2, 0)                   # HWC
    img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    fn = os.path.join(out_dir, f"{prefix}_f{fi:03d}.png")
    Image.fromarray(img).save(fn)
    saved.append(fn)
print(f"shape [1,{C},{F},{H},{W}]  saved {len(saved)} frames: {saved}")
