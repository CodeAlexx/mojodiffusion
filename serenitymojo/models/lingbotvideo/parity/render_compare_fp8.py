#!/usr/bin/env python
"""E-gate fp8 vs bf16 A/B: render the fp8 Mojo pixels and compare to the bf16
Mojo run + the torch oracle. Phase-1 IMAGE GATE helper.

Reads (after e_t2i_probe.mojo runs with LINGBOT_CKPT=transformer_fp8):
  mojo_e_pixels.safetensors            <- fp8 run pixels [1,3,1,256,256]
  mojo_e_pixels_bf16.safetensors       <- backed-up bf16 run pixels
  oracle_e.safetensors  (key 'pixels') <- torch reference pixels
Writes:
  mojo_e_generated_fp8.png             (parity dir + output/media)
  e_fp8_vs_bf16_contactsheet.png       (oracle | bf16 | fp8)
Prints pixel cosine: fp8-vs-bf16, fp8-vs-oracle, bf16-vs-oracle (informational —
the model is knife-edge bf16-MoE-sensitive; the visual scene is the real gate).
"""
import os
import numpy as np
from safetensors import safe_open
from PIL import Image

D = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
MEDIA = "/home/alex/mojodiffusion/output/media/lingbotvideo"


def load_px(path, key="pixels"):
    with safe_open(path, framework="np") as f:
        return f.get_tensor(key)  # [1,3,1,256,256]


def to_img(px):
    img = px[0, :, 0].transpose(1, 2, 0)
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


fp8 = load_px(os.path.join(D, "mojo_e_pixels.safetensors"))
bf16 = load_px(os.path.join(D, "mojo_e_pixels_bf16.safetensors"))
orac = load_px(os.path.join(D, "oracle_e.safetensors"))

img_fp8 = to_img(fp8)
Image.fromarray(img_fp8).save(os.path.join(D, "mojo_e_generated_fp8.png"))
Image.fromarray(img_fp8).save(os.path.join(MEDIA, "mojo_e_generated_fp8.png"))

print("=== E-gate fp8 A/B (pixel cosine, informational) ===")
print(f"  fp8   vs bf16   : {cos(fp8, bf16):.4f}")
print(f"  fp8   vs oracle : {cos(fp8, orac):.4f}")
print(f"  bf16  vs oracle : {cos(bf16, orac):.4f}   (baseline, was ~0.21)")
print(f"  fp8   mean/std  : {float(fp8.mean()):.4f} / {float(fp8.std()):.4f}")
print(f"  bf16  mean/std  : {float(bf16.mean()):.4f} / {float(bf16.std()):.4f}")

# Contact sheet: oracle | bf16 | fp8
imgs = [to_img(orac), to_img(bf16), img_fp8]
labels = ["oracle (torch)", "mojo bf16", "mojo fp8"]
h, w, _ = imgs[0].shape
pad = 8
sheet = np.full((h + 24, w * 3 + pad * 2, 3), 255, np.uint8)
for i, im in enumerate(imgs):
    x = i * (w + pad)
    sheet[24:24 + h, x:x + w] = im
out = os.path.join(MEDIA, "e_fp8_vs_bf16_contactsheet.png")
Image.fromarray(sheet).save(out)
print("SAVED", out)
print("SAVED", os.path.join(MEDIA, "mojo_e_generated_fp8.png"))
